import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

from utils.rbox_tools import convert_nms
from utils.DrawBox_ver2 import DrawBox
from nms.rbox_nms import prepare_nms
from rotation.rotate_polygon_nms import rotate_gpu_nms as rotate_cpu_nms
import math
#from rotation.rotate_cpu_nms import rotate_cpu_nms

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__', 'vessel')

NETS = {'vgg16': ('VGG16', 'vgg16_faster_rcnn_iter_80000.caffemodel'),
        'zf': ('ZF', 'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.6):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def vis_rbox_detections(im, dets, image_name, thresh=0.6):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    rbox = dets[inds, :5]
    
    im = DrawBox(im, rbox)
    name = image_name[:-4] + "out.jpg"
    cv2.imwrite(name, im)
    plt.imshow(im)

def filter_boxes(all_boxes, imshape):

    keep = []
    for i in range(all_boxes.shape[0]):
        x = all_boxes[i, 0] 
        y = all_boxes[i, 1]
        w = all_boxes[i, 2]
        h = all_boxes[i, 3]
        theta = all_boxes[i, 4]

        lt = np.array([-w/2,-h/2]).reshape(2,-1)
        rt = np.array([ w/2,-h/2]).reshape(2,-1)
        ld = np.array([-w/2, h/2]).reshape(2,-1)
        rd = np.array([ w/2, h/2]).reshape(2,-1)
        #theta = math.pi/180*35
        sets = np.hstack((lt, rt, ld, rd))
        
        roat = np.array([[math.cos(theta),-math.sin(theta)]
                        ,[math.sin(theta), math.cos(theta)]])
        
        points_out = roat.dot(sets) + np.array([x,y]).reshape(2,-1)
        '''
        if coord[0] < 0 or coord[0] > conv_in.shape[3] - 2 or coord[1] < 0 or coord[1] > conv_in.shape[2] - 2:
            roi_map[:, i, j] = zero_pad
            self.coord[:, i, j] = np.zeros(2)
        else:
        '''
        x_min_beyond = np.where(points_out[0, :] < 0)[0]
        x_max_beyond = np.where(points_out[0, :] > imshape[1])[0]
        y_min_beyond = np.where(points_out[1, :] < 0)[0]
        y_max_beyond = np.where(points_out[1, :] > imshape[0])[0]
        
        if x_min_beyond.shape[0] == 0 and x_max_beyond.shape[0] == 0 and y_min_beyond.shape[0] == 0 and y_max_beyond.shape[0] == 0:
            keep.append(i)
    remains = np.array(keep)
            
    return remains


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    #CONF_THRESH = 0.8
    NMS_THRESH = 0.1
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 5*cls_ind:5*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]    
        pre_dets = prepare_nms(cls_boxes)        
        dets = np.hstack((pre_dets, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = rotate_cpu_nms(dets, NMS_THRESH, cfg.GPU_ID)
        candidates = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        candidates = candidates[keep, :]
        
        vis_rbox_detections(im, candidates, image_name)
        

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0], 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')

    # end-to-end train model
    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0], 'faster_rcnn_end2end', 'test.prototxt')

    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
   
    
    im_names = []
    rootdir = '/home/username/ShipDetection/data/demo'
    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:
            #name = filename[:-4]
            im_names.append(filename)
    
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
