import os
import sys
sys.path.append('/workspace')
import time
import math
import numpy as np
import copy
import cv2
import argparse
import pycuda.driver as cuda ## I had problem with this, so must import. Plz check yourself 
import pycuda.autoinit ## I had problem with this, so must import. Plz check yourself

from src_code.detection.det_infer import Predictor
from src_code.detection.calibration import Calibration
from src_code.tl_detector.detector_yolov7 import Detecotr_YoloV7

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Pose

###
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
###

from sort import *

class Camemra_Node:
    def __init__(self):
        rospy.init_node('Camemra_node')

        camera_path = [
                    '/workspace/src_code/detection/calibration_data/epiton_cal/f60.txt',
                    '/workspace/src_code/detection/calibration_data/epiton_cal/f120.txt',
                    '/workspace/src_code/detection/calibration_data/epiton_cal/r120.txt'
                    ]
        self.calib = Calibration(camera_path)

        self.get_f60_new_image = False
        self.cur_f60_img = {'img':None, 'header':None}
        self.sub_f60_img = {'img':None, 'header':None}
        self.bbox_f60 = PoseArray()
        
        self.get_f120_new_image = False
        self.cur_f120_img = {'img':None, 'header':None}
        self.sub_f120_img = {'img':None, 'header':None}
        self.bbox_f120 = PoseArray()

        self.pub_od_f60 = rospy.Publisher('/mobinha/perception/camera/bounding_box', PoseArray, queue_size=1)
        self.pub_od_f120 = rospy.Publisher('/mobinha/perception/camera/bounding_box/f120', PoseArray, queue_size=1)

        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_f60_callback)
        rospy.Subscriber('/gmsl_camera/dev/video1/compressed', CompressedImage, self.IMG_f120_callback)

        ##########################
        self.pub_f60_det = rospy.Publisher('/det_result/f60', Image, queue_size=1)
        self.pub_f120_det = rospy.Publisher('/det_result/f120', Image, queue_size=1)
      
        self.bridge = CvBridge()
        self.sup = []
        ##########################

    def IMG_f60_callback(self,msg):
        if not self.get_f60_new_image:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.cur_f60_img['img'] = front_img
            self.cur_f60_img['header'] = msg.header
            self.get_f60_new_image = True

    def IMG_f120_callback(self,msg):
        if not self.get_f60_new_image:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.cur_f120_img['img'] = front_img
            self.cur_f120_img['header'] = msg.header
            self.get_f120_new_image = True

    def pose_set(self,bboxes,flag):
        bbox_pose = PoseArray()

        for bbox in bboxes:
            pose = Pose()
            pose.position.x = bbox[0]# box class
            pose.position.y = bbox[1]# box area
            pose.position.z = bbox[2]# box score
            pose.orientation.x = bbox[3][0]# box mid x
            pose.orientation.y = bbox[3][1]# box mid y
            pose.orientation.z = bbox[3][2]# box mid y
            pose.orientation.w = bbox[3][3]# box mid y
            bbox_pose.poses.append(pose)

        if flag == 'f60':
            self.pub_od_f60.publish(bbox_pose)
        if flag == 'f120':
            self.pub_od_f120.publish(bbox_pose)

    def det_pubulissher(self,det_img,det_box,flag):
        if flag =='f60':
            det_f60_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
            self.pose_set(det_box,flag)
            self.pub_f60_det.publish(det_f60_msg)
        if flag =='f120':
            det_f120_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
            self.pose_set(det_box,flag)
            self.pub_f120_det.publish(det_f120_msg)
    
            
def run(detector_yolov, camemra_node):
    while not rospy.is_shutdown():
        if camemra_node.get_f60_new_image:
            camemra_node.sub_f60_img['img'] = camemra_node.cur_f60_img['img']
            orig_im_f60 = copy.copy(camemra_node.sub_f60_img['img']) 
            filter_img, tl_boxes = detector_yolov.image_process(orig_im_f60,'f60')
            camemra_node.get_f60_new_image = False
            camemra_node.det_pubulissher(filter_img, tl_boxes,'f60')
          
if __name__ == "__main__":

    detector_yolov = Detecotr_YoloV7()

    camemra_node = Camemra_Node()

    run(detector_yolov, camemra_node)