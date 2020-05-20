# -*- coding:utf-8 -*-
import numpy as np

def convert_nms(boxes,scores):
    x = boxes[:,0] 
    y = boxes[:,1]
    w = boxes[:,2]
    h = boxes[:,3]
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    dets = np.hstack((x1.reshape(-1,1), y1.reshape(-1,1) , x2.reshape(-1,1), y2.reshape(-1,1), scores))
    return dets
