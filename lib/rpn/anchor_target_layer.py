import os
import caffe
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from ranchor import generate_ranchors
#from utils.cython_bbox import bbox_overlaps
from utils.rbox import prepare_overlaps
from rotation.rbbox_overlaps import rbbx_overlaps as rbox_overlaps
#from rotation.rbbox import rbbx_overlaps as rbox_overlaps
from fast_rcnn.bbox_transform import rbox_transform

DEBUG = False

class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        #anchor_scales = layer_params.get('scales', (8, 16, 32))
        #self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._anchors = generate_ranchors()
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * height, width)
        # rbox_targets
        top[1].reshape(1, A * 4, height, width)
        # rbox_inside_weights
        top[2].reshape(1, A * 4, height, width)
        # rbox_outside_weights
        top[3].reshape(1, A * 4, height, width)
        
        top[4].reshape(1, A, height, width)
        
        top[5].reshape(1, A, height, width)
        
        top[6].reshape(1, A, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x1, y1, x2, y2, label)
        # (x, y, w, h, theta, label)
        gt_rboxes = bottom[1].data
        # im_info
        im_info = bottom[2].data[0, :]

        if DEBUG:
            print ''
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_rboxes.shape', gt_rboxes.shape
            print 'rpn: gt_rboxes', gt_rboxes

        anchors_xy = self._anchors[:, :2]
        anchors_rem = self._anchors[:, 2:5]
        anchors_end = anchors_rem
        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        for i in range(K-1):
            anchors_end = np.vstack((anchors_end, anchors_rem))
        anchors_xy = anchors_xy.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2))
        anchors_xy = anchors_xy.reshape((K * A, 2))
        all_anchors = np.hstack((anchors_xy, anchors_end))
        total_anchors = int(K * A)

        # only keep anchors inside the image
#        inds_inside = np.where(
#            (all_anchors[:, 0] >= -self._allowed_border) &
#            (all_anchors[:, 1] >= -self._allowed_border) &
#            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
#            (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
#        )[0]

        inds_inside = np.arange(total_anchors)
        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors.shape', anchors.shape

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        pre_anchors = prepare_overlaps(anchors)
        pre_gt_rboxes = prepare_overlaps(gt_rboxes)    
        overlaps = rbox_overlaps(
            np.ascontiguousarray(pre_anchors, dtype=np.float32),
            np.ascontiguousarray(pre_gt_rboxes, dtype=np.float32))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))

        box_targets = np.zeros((len(inds_inside), 5), dtype=np.float32)
        box_targets = _compute_targets(anchors, gt_rboxes[argmax_overlaps, :])
        #rbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        #theta_targets = np.zeros((len(inds_inside), 1), dtype=np.float32)
        rbox_targets = box_targets[:, :4]
        theta_targets = box_targets[:, 4].reshape(-1, 1)


        box_inside_weights = np.zeros((len(inds_inside), 5), dtype=np.float32)
        box_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_rbox_inside_weights)
        
        rbox_inside_weights = box_inside_weights[:, :4]
        theta_inside_weights = box_inside_weights[:, 4]
        

        box_outside_weights = np.zeros((len(inds_inside), 5), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 5)) * 1.0 / num_examples
            negative_weights = np.ones((1, 5)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        box_outside_weights[labels == 1, :] = positive_weights
        box_outside_weights[labels == 0, :] = negative_weights
        
        rbox_outside_weights = box_outside_weights[:, :4]
        theta_outside_weights = box_outside_weights[:, 4]

        if DEBUG:
            self._sums += rbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (rbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        rbox_targets = _unmap(rbox_targets, total_anchors, inds_inside, fill=0)
        rbox_inside_weights = _unmap(rbox_inside_weights, total_anchors, inds_inside, fill=0)
        rbox_outside_weights = _unmap(rbox_outside_weights, total_anchors, inds_inside, fill=0)
        theta_inside_weights = _unmap(theta_inside_weights, total_anchors, inds_inside, fill=0)
        theta_outside_weights = _unmap(theta_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # rbox_targets
        rbox_targets = rbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        top[1].reshape(*rbox_targets.shape)
        top[1].data[...] = rbox_targets

        # rbox_inside_weights
        rbox_inside_weights = rbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert rbox_inside_weights.shape[2] == height
        assert rbox_inside_weights.shape[3] == width
        top[2].reshape(*rbox_inside_weights.shape)
        top[2].data[...] = rbox_inside_weights

        # rbox_outside_weights
        rbox_outside_weights = rbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert rbox_outside_weights.shape[2] == height
        assert rbox_outside_weights.shape[3] == width
        top[3].reshape(*rbox_outside_weights.shape)
        top[3].data[...] = rbox_outside_weights

        theta_targets = theta_targets.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        top[4].reshape(*theta_targets.shape)
        top[4].data[...] = theta_targets

        theta_inside_weights = theta_inside_weights.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        top[5].reshape(*theta_inside_weights.shape)
        top[5].data[...] = theta_inside_weights

        theta_outside_weights = theta_outside_weights.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        top[6].reshape(*theta_outside_weights.shape)
        top[6].data[...] = theta_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 5
    assert gt_rois.shape[1] == 6

    return rbox_transform(ex_rois, gt_rois[:, :5]).astype(np.float32, copy=False)
