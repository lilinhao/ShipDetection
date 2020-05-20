# -*- coding: utf-8 -*-

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg

DEBUG = False

class RboxPredLayer(caffe.Layer):

    def setup(self, bottom, top):

        top[0].reshape(1, 10)

    def forward(self, bottom, top):

        box = bottom[0].data
        theta = bottom[1].data
        box_0 = box[:, 0:4]
        box_1 = box[:, 4:8]
        theta_0 = theta[:, 0].reshape(-1,1)
        theta_1 = theta[:, 1].reshape(-1,1)
        blob = np.hstack((box_0, theta_0, box_1, theta_1))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
