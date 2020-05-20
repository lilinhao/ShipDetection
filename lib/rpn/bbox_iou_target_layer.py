# -*- coding:utf-8 -*-  
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from utils.cython_bbox import bbox_overlaps

DEBUG = False

class BboxIouTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 1)
		
    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        rois = bottom[0].data
        bbox_pred = bottom[1].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        im_info = bottom[2].data
        gt_boxes = bottom[3].data
        #print 'gt_boxes.shape = '
        #print bbox_pred.shape
        roi_boxes = rois[:, 1:5]# / im_info[0,2]
        pred_boxes = bbox_transform_inv(roi_boxes, bbox_pred)
        imshape = [im_info[0,1],im_info[0,1],3]
        pred_boxes = clip_boxes(pred_boxes, imshape)
        overlaps = bbox_overlaps(
        np.ascontiguousarray(pred_boxes, dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        # 同每个proposal交并比最大的gtbox索引值
        gt_assignment = overlaps.argmax(axis=1)
        # 同每个pred_boxes交并比最大的gtbox的IOU
        max_overlaps = overlaps.max(axis=1)
        best_iou = np.sort(max_overlaps, axis=None)

        print '++++++++++++++++++++++++++++++'
        print best_iou[-5:]
        print '++++++++++++++++++++++++++++++'
        top[0].reshape(*max_overlaps.shape)
        top[0].data[...] = max_overlaps		

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
