# -*- coding: utf-8 -*-

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg

DEBUG = False

class ClsScoreLayer(caffe.Layer):

    def setup(self, bottom, top):

        top[0].reshape(1, 2)

    def forward(self, bottom, top):

        box_score = bottom[0].data.reshape(-1,1)
        cls_score_pre = bottom[1].data

        #cls_score = cls_score_pre * box_score
        #cls_score[:, 0] = cls_score_pre[:, 0]
        
        cls_score = cls_score_pre

        top[0].reshape(*(cls_score.shape))
        top[0].data[...] = cls_score

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
