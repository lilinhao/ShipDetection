import numpy as np
np.set_printoptions(suppress=True)
import math

#  45=0.7853982 
#  90=1.5707963
# 135=-0.7853982
'''
small = 35*245
big = 60*420
'''
#def generate_ranchors(ratio=[3, 6], size=[3750, 15000, 60000]):
#def generate_ranchors(ratio=[7], size=[8575, 25200]):
#def generate_ranchors(ratio=[4, 8], size=[2048, 8192, 32768]):
#def generate_ranchors(ratio=[4, 7], size=[2800, 11200, 44800]):
#def generate_ranchors(ratio=[4, 7], size=[1024, 4096, 16384]):
def generate_ranchors(ratio=[4, 7], size=[1024, 4096, 16384, 65536]):
    
    anchor_dic = {}
    #anchors = _generate_ranchor(size, ratio)
    for i in ratio:
        anchor_dic[i] = np.vstack([_generate_ranchor(i, j) for j in size])
    anchors = np.vstack([anchor_dic[k] for k in ratio])
    return anchors
    

def _generate_base_anchor(ratio, size):

    x = np.array([0])
    y = np.array([0])
    size_ratios = size / ratio
    h = np.round(np.sqrt(size_ratios))
    w = np.round(h * ratio)
    w = np.array([w])
    h = np.array([h])
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    anchors = np.hstack((x, y, w, h))
    return anchors

def _generate_ranchor(ratio, size):

    anchor = _generate_base_anchor(ratio, size)
    '''
    ang0 = np.array([0]).reshape(1,1)
    ang30 = np.array([math.pi/180*30]).reshape(1,1)
    ang60 = np.array([math.pi/180*60]).reshape(1,1)
    ang90 = np.array([math.pi/180*90]).reshape(1,1)
    ang120 = np.array([-math.pi/180*30]).reshape(1,1)
    ang150 = np.array([-math.pi/180*60]).reshape(1,1)
    theta0 = np.hstack((anchor, ang0))
    theta30 = np.hstack((anchor, ang30))
    theta60 = np.hstack((anchor, ang60))
    theta90 = np.hstack((anchor, ang90))
    theta120 = np.hstack((anchor, ang120))
    theta150 = np.hstack((anchor, ang150))
    '''
    angfu90 = np.array([-math.pi/180*90]).reshape(1,1)
    angfu60 = np.array([-math.pi/180*60]).reshape(1,1)
    angfu30 = np.array([-math.pi/180*30]).reshape(1,1)
    ang0 = np.array([0]).reshape(1,1)
    ang30 = np.array([math.pi/180*30]).reshape(1,1)
    ang60 = np.array([math.pi/180*60]).reshape(1,1)
    thetafu90 = np.hstack((anchor, angfu90))
    thetafu60 = np.hstack((anchor, angfu60))
    thetafu30 = np.hstack((anchor, angfu30))
    theta0 = np.hstack((anchor, ang0))
    theta30 = np.hstack((anchor, ang30))
    theta60 = np.hstack((anchor, ang60))
    ranchors = np.vstack((thetafu90, thetafu60, thetafu30, theta0, theta30, theta60))
    return ranchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_ranchors()
    print time.time() - t
    print a
    print a.shape
    #from IPython import embed; embed()
