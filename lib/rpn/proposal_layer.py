import caffe
import numpy as np
import yaml
import math
from fast_rcnn.config import cfg
#from generate_anchors import generate_anchors
from ranchor import generate_ranchors
from fast_rcnn.bbox_transform import rbox_transform_inv
#from fast_rcnn.nms_wrapper import nms
from nms.rbox_nms import prepare_nms

from rotation.rotate_polygon_nms import rotate_gpu_nms as rotate_cpu_nms
#from rotation.rotate_cpu_nms import rotate_cpu_nms

from utils.timer import Timer

DEBUG = False

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._feat_stride = layer_params['feat_stride']
        #anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_ranchors()
        self._num_anchors = self._anchors.shape[0]

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # (n, x, y, w, h, theta)
        top[0].reshape(1, 6)
        #  square_rois
        top[1].reshape(1, 5)
        # (n, x, y, h, w, theta_jiaodu)
        top[2].reshape(1, 6)         

        # scores blob: holds scores for R regions of interest
        #if len(top) > 1:
        #    top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
        timer = Timer()
        timer.tic()
        
        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :]
        rbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]
        
        theta_deltas = bottom[3].data

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

        anchors_xy = self._anchors[:, :2]
        anchors_rem = self._anchors[:, 2:5]
        #anchors_end = anchors_rem
        
        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
        
        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors

        
        A = self._num_anchors
        K = shifts.shape[0]
        #for i in range(K-1):
        #    anchors_end = np.vstack((anchors_end, anchors_rem))
            
        anchors_end = anchors_rem.reshape(1, -1).repeat(K, axis=0).reshape(-1, 3)
        #print anchors_end.dtype
        anchors_xy = anchors_xy.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2))
        anchors_xy = anchors_xy.reshape((K * A, 2))
        anchors = np.hstack((anchors_xy, anchors_end))
        
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        
        rbox_deltas = rbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
        theta_deltas = theta_deltas.transpose((0, 2, 3, 1)).reshape((-1, 1))
        rbox_deltas = np.hstack((rbox_deltas, theta_deltas))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)    x = dets_in[:,0].reshape(-1,1)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
        #print scores.shape
        # Convert anchors into proposals via bbox transformations
        proposals = rbox_transform_inv(anchors, rbox_deltas)

        # 2. clip predicted boxes to image
        #proposals = clip_rboxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_rboxes(proposals, min_size * im_info[2])
        
        #keep = _filter_rboxes(proposals, min_size)
        
        #print '+++++++++++++++++++++++++++++++++++++++++'
        #print im_info[2]
        #print '+++++++++++++++++++++++++++++++++++++++++'        
        #print keep
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        dets = prepare_nms(proposals)
        
        keep = rotate_cpu_nms(np.hstack((dets, scores)), nms_thresh,cfg.GPU_ID)
        
        #keep = rotate_cpu_nms(np.hstack((dets, scores)), nms_thresh)
        #keep = rbox_nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        
        scores = scores[keep]
        square_rois = rois_to_square_rois(proposals)
        
        jiaiodu_proposals = prepare_rois(proposals)
        
        #print '++++++++++++++++++++++++++'
        #print proposals[:10,0:4]/0.71174377
        #print proposals[:10,:]
        #print scores
        #print '++++++++++++++++++++++++++'
        
        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob_0 = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob_0.shape))
        top[0].data[...] = blob_0
        blob_1 = np.hstack((batch_inds, square_rois.astype(np.float32, copy=False)))
        top[1].reshape(*(blob_1.shape))
        top[1].data[...] = blob_1
        
        blob_2 = np.hstack((batch_inds, jiaiodu_proposals.astype(np.float32, copy=False)))
        top[2].reshape(*(blob_2.shape))
        top[2].data[...] = blob_2
        # [Optional] output scores blob
        #if len(top) > 1:
        #    top[1].reshape(*(scores.shape))
        #    top[1].data[...] = scores

        timer.toc()
#        print ('Detection took {:.3f}s for ''proposal forward').format(timer.total_time)
        
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def _filter_rboxes(rboxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    keep = np.where((w >= min_size) & (h >= min_size))[0]
    return keep

def prepare_rois(rois_in):
    
    x = rois_in[:,0].reshape(-1,1)
    y = rois_in[:,1].reshape(-1,1)
    w = rois_in[:,2].reshape(-1,1)
    h = rois_in[:,3].reshape(-1,1)
    theta = rois_in[:,4].reshape(-1,1)
    theta = 180 * theta / np.pi
    rois_out = np.hstack((x, y, h, w, -theta))
    
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
    square_rois = np.zeros((rois.shape[0], 4))
    for i in range(rois.shape[0]):
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
        y2 = px.max(axis=0).reshape(-1, 1)
        square_rois[i, :] = np.hstack((x1, y1, x2, y2))
    '''
    square_rois = np.zeros((rois.shape[0], 4))
    #print '+++++++++++++++++++++++'
    #print rois.shape
    #print '+++++++++++++++++++++++'
    for i in range(rois.shape[0]):
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
        square_rois[i, :] = np.hstack((x1, y1, x2, y2))
        
    return square_rois
