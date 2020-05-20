# -*- coding:utf-8 -*-
import cv2
import numpy as np
import math
def DrawBox(pic,rbox):
    for row in rbox:
        #print pic.shape
        # 中点横坐标，中点纵坐标，框宽，框高，顺时针旋转角度
        x = row[0] 
        y = row[1]
        w = row[2]
        h = row[3]
        theta = row[4]
        # 坐标顺序为（列，行），即（x,y）
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
	'''
        cv2.line(pic,(p1[0],p1[1]),(p2[0],p2[1]),(0,255,0),2); #上边
        cv2.line(pic,(p1[0],p1[1]),(p3[0],p3[1]),(0,255,0),2); #左边
        cv2.line(pic,(p2[0],p2[1]),(p4[0],p4[1]),(0,255,0),2); #右边
        cv2.line(pic,(p3[0],p3[1]),(p4[0],p4[1]),(0,255,0),2); #下边
	'''
        cv2.line(pic,(p1[0],p1[1]),(p2[0],p2[1]),(0,0,255),2); #上边
        cv2.line(pic,(p1[0],p1[1]),(p3[0],p3[1]),(0,0,255),2); #左边
        cv2.line(pic,(p2[0],p2[1]),(p4[0],p4[1]),(0,0,255),2); #右边
        cv2.line(pic,(p3[0],p3[1]),(p4[0],p4[1]),(0,0,255),2); #下边
    return pic
