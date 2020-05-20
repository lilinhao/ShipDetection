import caffe
import yaml
import numpy as np
import math
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform

from utils.rbox import prepare_overlaps

from rotation.rbbox_overlaps import rbbx_overlaps as rbox_overlaps

from fast_rcnn.bbox_transform import rbox_transform
from fast_rcnn.bbox_transform import wh_to_xy

from utils.cython_bbox import bbox_overlaps

DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 6)
        # labels
        top[1].reshape(1, 1)
        # rbox_targets
        top[2].reshape(1, self._num_classes * 5)
        # rbox_inside_weights
        top[3].reshape(1, self._num_classes * 5)
        # rbox_outside_weights
        top[4].reshape(1, self._num_classes * 5)
        # square_rois
        #top[5].reshape(1, 5)
        top[5].reshape(1, )
        
    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        
        #all_rois = to_hudu_rois(all_rois)
        
        #all_rois = to_hudu_rois(all_rois)
        
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_rboxes = bottom[1].data
        
        #print '++++++++++++++++++++++'
        #print gt_rboxes.shape
        #print '++++++++++++++++++++++'

        #rot_gt_rboxes = bottom[2].data

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_rboxes.shape[0], 1), dtype=gt_rboxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_rboxes[:, :-1])))
        )
        
        
        #print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        #judge = np.where(all_rois[:, 4]>0)
        #print len(judge[0]) 
        #print all_rois.shape
        #print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        
        
        
        #print '++++++++++++++++++++++'        
        #print all_rois.shape
        #print '++++++++++++++++++++++'        
#        boxes = wh_to_xy(rot_gt_rboxes[:, :-2])
#        all_rois = np.vstack(
#            (all_rois, np.hstack((zeros, boxes)))
#        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, rbox_targets, rbox_inside_weights, iou_targets = _sample_rois(
            all_rois, gt_rboxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        #square_rois = rois_to_square_rois(rois)
        #rois = to_jiaodu_rois(rois)
        
        
        #print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        #judge = np.where(rois[:, 3]>0)
        #print len(judge[0]) 
        #print rois.shape
        #print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        
        #rois = to_jiaodu_rois(rois)

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # sampled rois
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # rbox_targets
        top[2].reshape(*rbox_targets.shape)
        top[2].data[...] = rbox_targets

        # rbox_inside_weights
        top[3].reshape(*rbox_inside_weights.shape)
        top[3].data[...] = rbox_inside_weights

        # rbox_outside_weights
        top[4].reshape(*rbox_inside_weights.shape)
        top[4].data[...] = np.array(rbox_inside_weights > 0).astype(np.float32)

        top[5].reshape(*iou_targets.shape)
        top[5].data[...] = iou_targets
        
        # tsquare_rois
        #top[5].reshape(*square_rois.shape)
        #top[5].data[...] = square_rois
        
        

        # theta_targets
        #top[5].reshape(*theta_targets.shape)
        #top[5].data[...] = theta_targets

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_rbox_regression_labels(rbox_target_data, num_classes):
    """Bounding-box regression targets (rbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        rbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = rbox_target_data[:, 0]
    rbox_targets = np.zeros((clss.size, 5 * num_classes), dtype=np.float32)
    rbox_inside_weights = np.zeros(rbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 5 * cls
        end = start + 5
        rbox_targets[ind, start:end] = rbox_target_data[ind, 1:]
        rbox_inside_weights[ind, start:end] = cfg.TRAIN.rbox_inside_weights
    return rbox_targets, rbox_inside_weights


def _get_theta_regression_labels(theta_target_data, num_classes, labels):

    clss = labels
    theta_targets = np.zeros((clss.size, num_classes), dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = cls
        end = start + 1
        theta_targets[ind, start:end] = theta_target_data[ind]
        
        #print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        #print theta_targets[0:20,:]
        #print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        
    return theta_targets

def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 5
    assert gt_rois.shape[1] == 5

#    targets = bbox_transform(ex_rois, gt_rois)

    targets = rbox_transform(ex_rois, gt_rois)

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_rboxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    pre_rois = prepare_overlaps(all_rois[:, 1:6])
    pre_gt_rboxes = prepare_overlaps(gt_rboxes[:, :5])    
    
    
    # overlaps: (rois x gt_rboxes)
    overlaps = rbox_overlaps(
        np.ascontiguousarray(pre_rois, dtype=np.float32),
        np.ascontiguousarray(pre_gt_rboxes, dtype=np.float32))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    
    #print '++++++++++++++++++++++'        
    #print gt_assignment[-4:]
    #print '++++++++++++++++++++++' 
    
    labels = gt_rboxes[gt_assignment, 5]
    
    #print '++++++++++++++++++++++'
    #print labels.shape
    #print '++++++++++++++++++++++'

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]
    
    iou_targets = max_overlaps[keep_inds]

    #print '++++++++++++++++++++++++++++++++++++++++++++'
    #print labels
    #print '++++++++++++++++++++++++++++++++++++++++++++'

#    rbox_target_data = _compute_targets(
#        rois[:, 1:5], gt_rboxes[gt_assignment[keep_inds], :4], labels)

    #print '++++++++++++++++++++++++++++++++++++++++++++'
    #print rot_gt_rboxes[gt_assignment[keep_inds], :4]
    #print rot_gt_rboxes
    #print '++++++++++++++++++++++++++++++++++++++++++++'
    
    rbox_target_data = _compute_targets(
        rois[:, 1:6], gt_rboxes[gt_assignment[keep_inds], :5], labels)

    #print '++++++++++++++++++++++++++++++++++++++++++++'
    #print rbox_target_data[:,0]
    #print '++++++++++++++++++++++++++++++++++++++++++++'

#    rbox_targets, rbox_inside_weights = \
#        _get_bbox_regression_labels(rbox_target_data, num_classes)

    rbox_targets, rbox_inside_weights = \
        _get_rbox_regression_labels(rbox_target_data, num_classes)
        

    #print '++++++++++++++++++++++++++++++++++++++++++++'
    #print rbox_targets[:,4:8]
    #print rbox_targets[:,5:10]
    #test = np.hstack((labels.reshape(-1,1),rbox_targets[:,0:5]))
    #print test
    #print '++++++++++++++++++++++++++++++++++++++++++++'

    #theta_target_data = rot_gt_rboxes[gt_assignment[keep_inds], 4]
    #theta_targets = _get_theta_regression_labels(theta_target_data, num_classes, labels)

    #print '++++++++++++++++++++++++++++++++++++++++++++'
    #print theta_targets[:,1]
    #print '++++++++++++++++++++++++++++++++++++++++++++'

    return labels, rois, rbox_targets, rbox_inside_weights, iou_targets

def to_hudu_rois(rois_in):
    
    inds = rois_in[:,0].reshape(-1,1)
    x = rois_in[:,1].reshape(-1,1)
    y = rois_in[:,2].reshape(-1,1)
    h = rois_in[:,3].reshape(-1,1)
    w = rois_in[:,4].reshape(-1,1)
    theta = rois_in[:,5].reshape(-1,1)
    theta = theta * np.pi / 180
    rois_out = np.hstack((inds, x, y, w, h, -theta))
    
    return rois_out    

def to_jiaodu_rois(rois_in):
    
    inds = rois_in[:,0].reshape(-1,1)
    x = rois_in[:,1].reshape(-1,1)
    y = rois_in[:,2].reshape(-1,1)
    w = rois_in[:,3].reshape(-1,1)
    h = rois_in[:,4].reshape(-1,1)
    theta = rois_in[:,5].reshape(-1,1)
    theta = 180 * theta / np.pi
    rois_out = np.hstack((inds, x, y, h, w, -theta))
    
    return rois_out

def rois_to_square_rois(rois):
    '''
    n = rois[0].reshape(-1.1)
    x = rois[1] 
    y = rois[2]
    w = rois[3]
    h = rois[4]
    theta = rois[5]
    lt = np.array([-w/2,-h/2]).reshape(2,-1)
    rt = np.array([ w/2,-h/2]).reshape(2,-1)
    ld = np.array([-w/2, h/2]).reshape(2,-1)
    rd = np.array([ w/2, h/2]).reshape(2,-1)
    #theta = math.pi/180*35
    roat = np.array([[math.cos(theta),-math.sin(theta)]
                    ,[math.sin(theta), math.cos(theta)]])
    p1 = roat.dot(lt) + np.array([x,y]).reshape(2,-1)
    p2 = roat.dot(rt) + np.array([x,y]).reshape(2,-1)
    p3 = roat.dot(ld) + np.array([x,y]).reshape(2,-1)
    p4 = roat.dot(rd) + np.array([x,y]).reshape(2,-1)
    px = np.hstack((p1[:,0], p2[:,0], p3[:,0], p4[:,0]))
    py = np.hstack((p1[:,1], p2[:,1], p3[:,1], p4[:,1]))
    
    x1 = px.min(axis=1).reshape(-1, 1)
    y1 = py.min(axis=1).reshape(-1, 1)
    x2 = px.max(axis=1).reshape(-1, 1)
    y2 = px.max(axis=1).reshape(-1, 1)
    square_rois = np.hstack((n, x1, y1, x2, y2))
    '''
    '''
    square_rois = np.zeros((rois.shape[0], 5))
    for i in range(rois.shape[0]):
        n = np.zeros((1,1))
        x = rois[i, 0] 
        y = rois[i, 1]
        w = rois[i, 2]
        h = rois[i, 3]
        theta = rois[i, 4]
        lt = np.array([-w/2,-h/2]).reshape(2,1)
        rt = np.array([ w/2,-h/2]).reshape(2,1)
        ld = np.array([-w/2, h/2]).reshape(2,1)
        rd = np.array([ w/2, h/2]).reshape(2,1)
        #theta = math.pi/180*35
        roat = np.array([[math.cos(theta),-math.sin(theta)]
                        ,[math.sin(theta), math.cos(theta)]])
        p1 = roat.dot(lt) + np.array([x,y]).reshape(1,2)
        p2 = roat.dot(rt) + np.array([x,y]).reshape(1,2)
        p3 = roat.dot(ld) + np.array([x,y]).reshape(1,2)
        p4 = roat.dot(rd) + np.array([x,y]).reshape(1,2)
        px = np.hstack((p1[:,0], p2[:,0], p3[:,0], p4[:,0]))
        py = np.hstack((p1[:,1], p2[:,1], p3[:,1], p4[:,1]))

        x1 = px.min(axis=0).reshape(-1, 1)
        y1 = py.min(axis=0).reshape(-1, 1)
        x2 = px.max(axis=0).reshape(-1, 1)
        y2 = py.max(axis=0).reshape(-1, 1)
        square_rois[i, :] = np.hstack((n, x1, y1, x2, y2))
    '''
    square_rois = np.zeros((rois.shape[0], 5))
    #print '+++++++++++++++++++++++'
    #print rois[:10,:]
    #print '+++++++++++++++++++++++'
    for i in range(rois.shape[0]):
        ind = rois[i, 0] 
        x = rois[i, 1] 
        y = rois[i, 2]
        w = rois[i, 3]
        h = rois[i, 4]
        theta = rois[i, 5]
        lt = np.array([-w/2,-h/2]).reshape(2,1)
        rt = np.array([ w/2,-h/2]).reshape(2,1)
        ld = np.array([-w/2, h/2]).reshape(2,1)
        rd = np.array([ w/2, h/2]).reshape(2,1)
        #theta = math.pi/180*35
        roat = np.array([[math.cos(theta),-math.sin(theta)]
                        ,[math.sin(theta), math.cos(theta)]])
        p1 = roat.dot(lt) + np.array([x,y]).reshape(2,1)
        p2 = roat.dot(rt) + np.array([x,y]).reshape(2,1)
        p3 = roat.dot(ld) + np.array([x,y]).reshape(2,1)
        p4 = roat.dot(rd) + np.array([x,y]).reshape(2,1)
        px = np.hstack((p1[0], p2[0], p3[0], p4[0]))
        py = np.hstack((p1[1], p2[1], p3[1], p4[1]))

        x1 = px.min(axis=0).reshape(-1, 1)
        y1 = py.min(axis=0).reshape(-1, 1)
        x2 = px.max(axis=0).reshape(-1, 1)
        y2 = py.max(axis=0).reshape(-1, 1)
        square_rois[i, :] = np.hstack((ind.reshape(-1,1), x1, y1, x2, y2))
    
    return square_rois
