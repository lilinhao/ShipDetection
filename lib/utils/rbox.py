import numpy as np

def rbox_overlaps(rboxes, query_rboxes):
    N = rboxes.shape[0]
    K = query_rboxes.shape[0]
    rboxes = wh_to_xy(rboxes)
    query_rboxes = wh_to_xy(query_rboxes)
    
    overlaps = np.zeros((N, K))
    for k in range(K):
        box_area = ((query_rboxes[k, 2] - query_rboxes[k, 0] + 1) * (query_rboxes[k, 3] - query_rboxes[k, 1] + 1))
        for n in range(N):
            iw = (min(rboxes[n, 2], query_rboxes[k, 2]) - max(rboxes[n, 0], query_rboxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(rboxes[n, 3], query_rboxes[k, 3]) - max(rboxes[n, 1], query_rboxes[k, 1]) + 1)
                if ih > 0:
                    theta_gap = query_rboxes[k, 4] - rboxes[n, 4]
                    ua = np.float((rboxes[n, 2] - rboxes[n, 0] + 1) * (rboxes[n, 3] - rboxes[n, 1] + 1) + box_area - iw * ih)
                    overlaps[n, k] = np.abs(np.cos(theta_gap)) * iw * ih / ua       
    return overlaps
               
def prepare_overlaps(boxes_in):
    x = boxes_in[:,0].reshape(-1,1)
    y = boxes_in[:,1].reshape(-1,1)
    w = boxes_in[:,2].reshape(-1,1)
    h = boxes_in[:,3].reshape(-1,1)
    theta = boxes_in[:,4].reshape(-1,1)
    theta = 180 * theta / np.pi
    boxes_out = np.hstack((x, y, h, w, -theta))
    return boxes_out
                    
def wh_to_xy(rboxes):
    x = rboxes[:, 0]
    y = rboxes[:, 1]
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    theta = rboxes[:, 4]
    
    xyboxes = np.zeros(rboxes.shape, dtype=rboxes.dtype)
    # x1
    xyboxes[:, 0] = x - 0.5 * w
    # y1
    xyboxes[:, 1] = y - 0.5 * h
    # x2
    xyboxes[:, 2] = x + 0.5 * w
    # y2
    xyboxes[:, 3] = y + 0.5 * h
    # theta
    xyboxes[:, 4] = theta

    return xyboxes