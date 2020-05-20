# -*- coding: utf-8 -*-

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg

DEBUG = False

class IOU2ProbLayer(caffe.Layer):

    def setup(self, bottom, top):

        top[0].reshape(1, 2)

    def forward(self, bottom, top):

        iou = bottom[0].data
        rows = iou.shape[0]
        prob = np.concatenate((np.zeros((rows,1)), iou), axis=1)
        top[0].reshape(*(prob.shape))
        top[0].data[...] = prob

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
