import numpy as np
import cv2


def prepare_nms(dets_in):
    
    x = dets_in[:,0].reshape(-1,1)
    y = dets_in[:,1].reshape(-1,1)
    w = dets_in[:,2].reshape(-1,1)
    h = dets_in[:,3].reshape(-1,1)
    theta = dets_in[:,4].reshape(-1,1)
    theta = 180 * theta / np.pi
    dets_out = np.hstack((x, y, h, w, -theta))
    
    return dets_out    
    
    
    
def rbox_nms(dets, threshold):
    
    keep = []
    scores = dets[:, -1]
    theta = 180 * dets[:,4] / np.pi
    
    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype = np.int)
    
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((dets[i,0],dets[i,1]),(dets[i,2],dets[i,3]),theta[i])
        area_r1 = dets[i,2]*dets[i,3]
        for _j in range(_i+1,ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            r2 = ((dets[j,0],dets[j,1]),(dets[j,2],dets[j,3]),theta[j])
            area_r2 = dets[j,2]*dets[j,3]
            ovr = 0.0
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if  None != int_pts:
                order_pts = cv2.convexHull(int_pts, returnPoints = True)
                int_area = cv2.contourArea(order_pts)
                ovr = int_area*1.0/(area_r1+area_r2-int_area)
                
            if ovr>=threshold:
                suppressed[j]=1
                
    return keep



if __name__ == "__main__":
    boxes = np.array([[50, 50, 100, 10, 0, 0.99],
                      #[60, 60, 100, 10, 0, 0.88],#keep 0.68
                      [50, 50, 100, 10, 0.0706858, 0.66],#discard 0.70
                      #[200, 200, 100, 10, 0, 0.77],#keep 0.0
                      ])
    a = rbox_nms(boxes, 0.7)
    print boxes[a]
