# -*- coding: utf-8 -*-

import caffe
import numpy as np
import math

DEBUG = False


class RboxPoolLayer(caffe.Layer):

    def setup(self, bottom, top):

        top[0].reshape(1, 256, 7, 7)      
        self.ratio = np.ones((512, 7 ,7), dtype=np.float32)
        self.grad_set = np.zeros(bottom[0].data.shape, dtype=np.float32)
        self.coord_bp = []
        
        self.mask_in = np.zeros((256, 7 ,7), dtype=np.float32)
        self.mask_out = np.zeros((256, 7 ,7), dtype=np.float32)
        
    def forward(self, bottom, top):
        
        conv_in = bottom[0].data
        rois = bottom[1].data
        
        list_out = []
        mask_in(self)
        mask_out(self)
        
        for row in rois:    
            coord = sample_ring_region(row)
            roi_map = sum_region(self, conv_in, coord)
            
            m_in = roi_map[0:256,:,:] * self.mask_in
            m_out = roi_map[256:512,:,:] * self.mask_out
            m_all = m_in + m_out
            
            list_out.append(m_all)
            
        map_out = np.array(list_out)
                        
        top[0].reshape(*(map_out.shape))
        top[0].data[...] = map_out
    
    def backward(self, top, propagate_down, bottom):
        
        self.grad_set = np.zeros(bottom[0].data.shape, dtype=np.float32)

        coord_bp = np.array(self.coord_bp)
        
        mask_in(self)
        mask_out(self)
                
        for i in range(coord_bp.shape[0]):
            
            grad_lt = np.zeros(bottom[0].data.shape, dtype=np.float32)
            grad_rt = np.zeros(bottom[0].data.shape, dtype=np.float32)
            grad_ld = np.zeros(bottom[0].data.shape, dtype=np.float32)
            grad_rd = np.zeros(bottom[0].data.shape, dtype=np.float32)
            grad = np.zeros(bottom[0].data.shape, dtype=np.float32)
            
            coord_map = coord_bp[i, :, : ,: ].reshape(2, -1).transpose()
            x_min = coord_map[:, 0].astype(np.int)
            x_max = (coord_map[:, 0] + 1).astype(np.int)
            y_min = coord_map[:, 1].astype(np.int)
            y_max = (coord_map[:, 1] + 1).astype(np.int)

            dw = (coord_map[:, 0] - x_min).reshape(49, -1)
            dh = (coord_map[:, 1] - y_min).reshape(49, -1)
            _ ,dw = np.meshgrid(np.zeros(512), dw)
            _ ,dh = np.meshgrid(np.zeros(512), dh)

            
            diff_in =  top[0].diff[i, :, :, :] * self.mask_in

            diff_out = top[0].diff[i, :, :, :] * self.mask_out
            diff = np.concatenate((diff_in, diff_out), axis=0)
            
            grad_lt[0, :, y_min, x_min] = (1-dw)*(1-dh) * diff.reshape(512, -1).transpose()
            grad_rt[0, :, y_min, x_max] = dw*(1-dh) * diff.reshape(512, -1).transpose()
            grad_ld[0, :, y_max, x_min] = (1-dw)*dh * diff.reshape(512, -1).transpose()
            grad_rd[0, :, y_max, x_max] = dw*dh * diff.reshape(512, -1).transpose()
            
            
            
            grad[0, :, :, :] = grad_lt + grad_rt + grad_ld + grad_rd        
            self.grad_set[...] = self.grad_set[...] + grad

        bottom[0].diff[...] = self.grad_set
        
        self.coord_bp = []
            

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def sample_region(roi):
    
    x = roi[1]/16
    y = roi[2]/16
    w = roi[3]/16
    h = roi[4]/16   
    theta = roi[5]
    
    start_x = -w/2
    end_x = w/2
    start_y = -h/2
    end_y = h/2
    bin_w = np.linspace(start_x, end_x, num=8).astype(np.float32)
    bin_h = np.linspace(start_y, end_y, num=8).astype(np.float32)
    
    sample_row_y = np.arange(7).astype(np.float32)
    sample_col_x = np.arange(7).astype(np.float32)
    
    for i in range(7):
        sample_row_y[i] = (bin_h[i] + bin_h[i+1])/2
    for i in range(7):
        sample_col_x[i] = (bin_w[i] + bin_w[i+1])/2
        
    coord = np.zeros((2, len(sample_row_y) * len(sample_col_x)), dtype=np.float32)
    for i in range(len(sample_row_y)):
        for j in range(len(sample_col_x)):
            coord[0, i*sample_col_x.shape[0]+j] = sample_col_x[j]
            coord[1, i*sample_col_x.shape[0]+j] = sample_row_y[i]
    
    roat = np.array([[math.cos(theta),-math.sin(theta)]
                    ,[math.sin(theta), math.cos(theta)]])
    
    coord = roat.dot(coord) + np.array([x,y]).reshape(2,1)
    
    return coord

def cal_ring(w, h, no, no_ring):
      
    w = w*no/3
    h = h*no/3
    
    len_c = (w+h)*2
    
    bin_c = np.linspace(0, len_c, num=no_ring+1).astype(np.float32)
    bin_c = np.delete(bin_c, no_ring)
    c_x = np.arange(no_ring).astype(np.float32)
    c_y = np.arange(no_ring).astype(np.float32)
    for i in range(no_ring):
        if 0 <= bin_c[i] < w:
            c_x[i] = bin_c[i]
            c_y[i] = 0
        elif w <= bin_c[i] < w+h:
            c_x[i] = w
            c_y[i] = -(bin_c[i]-w)
        elif w+h <= bin_c[i] < w*2 + h:
            c_x[i] = w-(bin_c[i]-w-h)
            c_y[i] = -h
        else:
            c_x[i] = 0
            c_y[i] = -h-(bin_c[i]-w*2-h)
    c_x = c_x - w/2
    c_y = c_y + h/2
    
    return c_x, c_y
    
def rev_order(a):
    b = np.arange(a.shape[0])
    for i in range(a.shape[0]):
        b[i] = a[a.shape[0]-1-i]
    return b

def embad(c3_x, c3_y, c2_x, c2_y, c1_x, c1_y):
    
    ring_x = np.zeros((7, 7), dtype=np.float32)
    ring_y = np.zeros((7, 7), dtype=np.float32)
    
    ring_x[0, 0:6] = c3_x[0:6]
    ring_y[0, 0:6] = c3_y[0:6]
    ring_x[0:6, 6] = c3_x[6:12]
    ring_y[0:6, 6] = c3_y[6:12]
    ring_x[6, 1:7] = rev_order(c3_x[12:18])
    ring_y[6, 1:7] = rev_order(c3_y[12:18])
    ring_x[1:7, 0] = rev_order(c3_x[18:24])
    ring_y[1:7, 0] = rev_order(c3_y[18:24])
    
    ring_x[1, 1:5] = c2_x[0:4]
    ring_y[1, 1:5] = c2_y[0:4]
    ring_x[1:5, 5] = c2_x[4:8]
    ring_y[1:5, 5] = c2_y[4:8]
    ring_x[5, 2:6] = rev_order(c2_x[8:12])
    ring_y[5, 2:6] = rev_order(c2_y[8:12])
    ring_x[2:6, 1] = rev_order(c2_x[12:16])
    ring_y[2:6, 1] = rev_order(c2_y[12:16])
    
    ring_x[2, 2:4] = c1_x[0:2]
    ring_y[2, 2:4] = c1_y[0:2]
    ring_x[2:4, 4] = c1_x[2:4]
    ring_y[2:4, 4] = c1_y[2:4]
    ring_x[4, 3:5] = rev_order(c1_x[4:6])
    ring_y[4, 3:5] = rev_order(c1_y[4:6])
    ring_x[3:5, 2] = rev_order(c1_x[6:8])
    ring_y[3:5, 2] = rev_order(c1_y[6:8])
    
    return ring_x, ring_y

def sample_ring_region(roi):
    
    x = roi[1]/16
    y = roi[2]/16
    w = roi[3]/16
    h = roi[4]/16    
    theta = roi[5]

    c3_x, c3_y = cal_ring(w, h, 3, 24)
    c2_x, c2_y = cal_ring(w, h, 2, 16)
    c1_x, c1_y = cal_ring(w, h, 1, 8)
    
    ring_x, ring_y = embad(c3_x, c3_y, c2_x, c2_y, c1_x, c1_y)
    
    coord = np.zeros((2, 49), dtype=np.float32)
    coord[0, :] = ring_x.reshape(1, 49)
    coord[1, :] = ring_y.reshape(1, 49)
    
    roat = np.array([[math.cos(theta),-math.sin(theta)]
                    ,[math.sin(theta), math.cos(theta)]])
    
    coord = roat.dot(coord) + np.array([x,y]).reshape(2,1)
    
    return coord

def interpolation(self, conv_in, coord):
        
    x_min = (coord[0, :]).reshape(7,7).astype(np.int)
    x_max = (coord[0, :] + 1).reshape(7,7).astype(np.int)
    y_min = (coord[1, :]).reshape(7,7).astype(np.int)
    y_max = (coord[1, :] + 1).reshape(7,7).astype(np.int)
    # array(0,0)
    lt = conv_in[0, :, y_min, x_min].transpose(2, 0, 1)
    # (0,1)
    rt = conv_in[0, :, y_min, x_max].transpose(2, 0, 1)
    # (1,0)
    ld = conv_in[0, :, y_max, x_min].transpose(2, 0, 1)
    # (1,1)
    rd = conv_in[0, :, y_max, x_max].transpose(2, 0, 1)
    
    dw = coord[0].reshape(7,7) - x_min
    dh = coord[1].reshape(7,7) - y_min
    
    inter_out = (1-dw)*(1-dh)*lt + dw*(1-dh)*rt + (1-dw)*dh*ld + dw*dh*rd
    
    
    return inter_out
    

def sum_region(self, conv_in, coord):   
    
    roi_map = np.zeros((conv_in.shape[1], 7, 7), dtype=np.float32)
    
    x_min_beyond = np.where(coord[0, :] < 0)[0]
    x_max_beyond = np.where(coord[0, :] > conv_in.shape[3] - 1.5)[0]
    y_min_beyond = np.where(coord[1, :] < 0)[0]
    y_max_beyond = np.where(coord[1, :] > conv_in.shape[2] - 1.5)[0]

    coord[0, x_min_beyond] = 0.5
    coord[0, x_max_beyond] = conv_in.shape[3] - 1.5    
    coord[1, y_min_beyond] = 0.5
    coord[1, y_max_beyond] = conv_in.shape[2] - 1.5
    
    roi_map = interpolation(self, conv_in, coord)
              
    self.coord_bp.append(coord.reshape(2,7,7))
    
    return roi_map

    
def mask_in(self):
    
    mask = np.zeros((256, 7, 7), dtype=np.float32)
    ring3 = np.zeros((256, 6), dtype=np.float32)
    mask[:, 0, 0:6] = ring3
    mask[:, 0:6, 6] = ring3
    mask[:, 6, 1:7] = ring3
    mask[:, 1:7, 0] = ring3

    ring2 = np.ones((256, 4), dtype=np.float32)
    mask[:, 1, 1:5] = ring2
    mask[:, 1:5, 5] = ring2
    mask[:, 5, 2:6] = ring2
    mask[:, 2:6, 1] = ring2
    
    ring1 = np.ones((256, 3, 3), dtype=np.float32)
    mask[:, 2:5, 2:5] = ring1
    
    self.mask_in = mask
    
def mask_out(self):
    
    mask = np.zeros((256, 7, 7), dtype=np.float32)
    ring3 = np.ones((256, 6), dtype=np.float32)
    mask[:, 0, 0:6] = ring3
    mask[:, 0:6, 6] = ring3
    mask[:, 6, 1:7] = ring3
    mask[:, 1:7, 0] = ring3

    ring2 = np.zeros((256, 4), dtype=np.float32)
    mask[:, 1, 1:5] = ring2
    mask[:, 1:5, 5] = ring2
    mask[:, 5, 2:6] = ring2
    mask[:, 2:6, 1] = ring2
    
    ring1 = np.zeros((256, 3, 3), dtype=np.float32)
    mask[:, 2:5, 2:5] = ring1
    
    self.mask_out = mask
    
