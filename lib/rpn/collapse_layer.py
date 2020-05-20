# -*- coding: utf-8 -*-

import caffe
import numpy as np

DEBUG = False

class CollapseLayer(caffe.Layer):

    def setup(self, bottom, top):

        data_shape = np.array(bottom[0].data.shape)

        data_shape[2] = data_shape[2]+1

        data_shape[3] = data_shape[3]+1
        
            
        top[0].reshape(1, data_shape[1]*4, data_shape[2]/2, data_shape[3]/2)           
        
    def forward(self, bottom, top):
        
        conv_in = bottom[0].data
        
        if conv_in.shape[2]%2 == 1:
            hzero = np.zeros((conv_in.shape[0], conv_in.shape[1], 1, conv_in.shape[3]), dtype=np.float32)
            conv_in = np.concatenate((conv_in, hzero), axis=2)
        if conv_in.shape[3]%2 == 1:
            wzero = np.zeros((conv_in.shape[0], conv_in.shape[1], conv_in.shape[2], 1), dtype=np.float32)
            conv_in = np.concatenate((conv_in, wzero), axis=3)
            
            
        lt =  conv_in[:, :, 0::2, 0::2]
        rt =  conv_in[:, :, 0::2, 1::2]
        ld =  conv_in[:, :, 1::2, 0::2]
        rd =  conv_in[:, :, 1::2, 1::2]
        conv_out = np.concatenate((lt, rt, ld, rd), axis=1)
             
        top[0].reshape(*(conv_out.shape))
        top[0].data[...] = conv_out
    
    def backward(self, top, propagate_down, bottom):
        

        grad_lt, grad_rt, grad_ld, grad_rd = np.split(top[0].diff[...], 4, axis=1)

        
        shape_in = np.array(bottom[0].data.shape)
        if shape_in[2]%2 == 1:
            shape_in[2] = shape_in[2]+1
        if shape_in[3]%2 == 1:
            shape_in[3] = shape_in[3]+1
        
        
        diff_temp = np.zeros(shape_in, dtype=np.float32)
        
        diff_temp[:, :, 0::2, 0::2] = grad_lt
        diff_temp[:, :, 0::2, 1::2] = grad_rt
        diff_temp[:, :, 1::2, 0::2] = grad_ld
        diff_temp[:, :, 1::2, 1::2] = grad_rd
        
        real_shape = bottom[0].data.shape
        
        bottom[0].diff[...] = diff_temp[:, :, 0:real_shape[2], 0:real_shape[3]]
    
        self.shape = 0
        
    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass