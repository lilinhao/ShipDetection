import numpy as np
import math

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def rbox_transform(ex_rois, r_gt_rois):
    #ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    #ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    #ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    #ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
    ex_ctr_x = ex_rois[:, 0]
    ex_ctr_y = ex_rois[:, 1]
    ex_widths = ex_rois[:, 2]
    ex_heights = ex_rois[:, 3]
    ex_theta = ex_rois[:, 4]

    r_gt_ctr_x = r_gt_rois[:, 0]
    r_gt_ctr_y = r_gt_rois[:, 1]
    r_gt_widths = r_gt_rois[:, 2]
    r_gt_heights = r_gt_rois[:, 3]
    r_gt_theta = r_gt_rois[:, 4]
    
    #targets_dx = (r_gt_ctr_x - ex_ctr_x) / ex_widths
    #targets_dy = (r_gt_ctr_y - ex_ctr_y) / ex_heights
    
    #inds = np.where(ex_heights > ex_widths)[0]
    #m = np.zeros(ex_widths.shape, dtype=ex_widths.dtype)
    #for i in inds:
    #    m[i] = ex_widths[i]
    #    ex_widths[i] = ex_heights[i]
    #    ex_heights[i] = m[i]
        
    targets_dx = (r_gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (r_gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(r_gt_widths / ex_widths)
    targets_dh = np.log(r_gt_heights / ex_heights)
    targets_dtheta = np.tan(r_gt_theta - ex_theta)
    #targets_dtheta = r_gt_theta - ex_theta

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh, targets_dtheta)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def rbox_transform_inv(rboxes, deltas):
    if rboxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    rboxes = rboxes.astype(deltas.dtype, copy=False)

    #widths = rboxes[:, 2] - rboxes[:, 0] + 1.0
    #heights = rboxes[:, 3] - rboxes[:, 1] + 1.0
    #ctr_x = rboxes[:, 0] + 0.5 * widths
    #ctr_y = rboxes[:, 1] + 0.5 * heights
    ctr_x = rboxes[:, 0]
    ctr_y = rboxes[:, 1]
    widths = rboxes[:, 2]
    heights = rboxes[:, 3]
    theta = rboxes[:, 4]
    dx = deltas[:, 0::5]
    dy = deltas[:, 1::5]
    dw = deltas[:, 2::5]
    dh = deltas[:, 3::5]
    dtheta = deltas[:, 4::5]

    #pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    #pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]

    #inds = np.where(heights > widths)[0]
    #m = np.zeros(widths.shape, dtype=widths.dtype)
    #for i in inds:
    #    m[i] = widths[i]
    #    widths[i] = heights[i]
    #    heights[i] = m[i]
    #wh = np.hstack((widths[:, np.newaxis], heights[:, np.newaxis]))
    #print wh
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    pred_theta = np.arctan(dtheta) + theta[:, np.newaxis]
    
    fu_beyond = np.where(pred_theta < -1.5707963)[0]
    zheng_beyond = np.where(pred_theta > 1.5707963)[0]
    pred_theta[fu_beyond] = pred_theta[fu_beyond] + 3.1415927
    pred_theta[zheng_beyond] = pred_theta[zheng_beyond] - 3.1415927
    
    #pred_theta = dtheta + theta[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x
    pred_boxes[:, 0::5] = pred_ctr_x
    # y
    pred_boxes[:, 1::5] = pred_ctr_y
    # w
    pred_boxes[:, 2::5] = pred_w
    # h
    pred_boxes[:, 3::5] = pred_h
    
    pred_boxes[:, 4::5] = pred_theta

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def clip_rboxes(rboxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    x = rboxes[0] 
    y = rboxes[1]
    w = rboxes[2]
    h = rboxes[3]
    theta = rboxes[4]
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
    inds = np.ones((rboxes.shape[0], 1))
    
    for i in range(rboxes.shape[0]):
        max_x = np.maximum(px[i, :])
        min_x = np.minimum(px[i, :])
        max_y = np.maximum(py[i, :])
        min_y = np.minimum(py[i, :])
        if max_x > im_shape[1] - 1 or min_x < 0 or max_y > im_shape[0] - 1 or min_y < 0:
            inds[i] = 0
        
    remain_inds = np.where(inds > 0)[0]
    rboxes = rboxes[remain_inds, :]

    # x1 >= 0
    #rboxes[:, 0::4] = np.maximum(np.minimum(rboxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    #rboxes[:, 1::4] = np.maximum(np.minimum(rboxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    #rboxes[:, 2::4] = np.maximum(np.minimum(rboxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    #rboxes[:, 3::4] = np.maximum(np.minimum(rboxes[:, 3::4], im_shape[0] - 1), 0)
    return rboxes

def wh_to_xy(deltas):
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]
    boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    boxes[:, 0] = dx - 0.5 * dw
    # y1
    boxes[:, 1] = dy - 0.5 * dh
    # x2
    boxes[:, 2] = dx + 0.5 * dw
    # y2
    boxes[:, 3] = dy + 0.5 * dh

    return boxes
