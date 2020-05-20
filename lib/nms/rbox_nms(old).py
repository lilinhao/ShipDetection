# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def rbox_nms(rboxes, thresh):
    """Pure Python NMS baseline."""
    dx = rboxes[:, 0]
    dy = rboxes[:, 1]
    dw = rboxes[:, 2]
    dh = rboxes[:, 3]
    theta = rboxes[:, 4]
    scores = rboxes[:, 5]
    
    x1 = dx - 0.5 * dw
    y1 = dy - 0.5 * dh
    x2 = dx + 0.5 * dw
    y2 = dy + 0.5 * dh

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        theta_gap = theta[i] - theta[order[1:]]
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = np.abs(np.cos(theta_gap)) * inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
