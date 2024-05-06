import os
import time
import math
import numpy as np
import copy
import cv2
import argparse
import pycuda.driver as cuda ## I had problem with this, so must import. Plz check yourself 
import pycuda.autoinit ## I had problem with this, so must import. Plz check yourself

from code.detection.det_infer import Predictor
from code.detection.calibration import Calibration

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Pose

###
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
###

from sort import *

class Camemra_Node:
    def __init__(self,args,day_night):
        rospy.init_node('Camemra_node')
        self.fps_list=[]
        self.args = args
        
        self.class_names = ['person', 'bicycle','car','bus','motorcycle','truck', 'green', 'red', 'yellow',
                            'red_arrow', 'red_yellow', 'green_arrow','green_yellow','green_right',
                            'warn','black','tl_v', 'tl_p', 'traffic_sign', 'warning', 
                            'tl_bus']
        
        self.rgb_day_list = [(148, 108, 138), (207, 134, 240), (138, 88, 55), (26, 99, 86), (38, 125, 80), (72, 80, 57), # 6 
                        (118, 23, 200), (117, 189, 183), (0, 128, 128), (60, 185, 90), (20, 20, 150), (13, 131, 204), 
                        (30, 200, 200), (43, 38, 105), (104, 235, 178), (135, 68, 28), (140, 202, 15), (67, 115, 220),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30)]


        self.allowed_unrecognized_frames = 0
        self.last_observed_time = 0
        self.baseline_boxes = [960,270]

        sort_max_age = 15
        sort_min_hits = 2
        sort_iou_thresh = 0.1
        self.sort_tracker_f60 = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
        
        local = os.getcwd()
        # print('now is here', local)
        camera_path = [
                    '/workspace/code/detection/calibration_data/epiton_cal/f60.txt',
                    '/workspace/code/detection/calibration_data/epiton_cal/f120.txt',
                    '/workspace/code/detection/calibration_data/epiton_cal/r120.txt'
                    ]
        self.calib = Calibration(camera_path)

        self.get_f60_new_image = False
        self.cur_f60_img = {'img':None, 'header':None}
        self.sub_f60_img = {'img':None, 'header':None}
        self.bbox_f60 = PoseArray()
        
        self.get_f120_new_image = False
        self.cur_f120_img = {'img':None, 'header':None}
        self.sub_f120_img = {'img':None, 'header':None}

        ### Det Model initiolization
        self.day_night = day_night
        self.det_pred = Predictor(engine_path=args.det_weight , day_night=day_night)
        
        self.pub_od_f60 = rospy.Publisher('/mobinha/perception/camera/bounding_box', PoseArray, queue_size=1)

        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_f60_callback)
        rospy.Subscriber('/gmsl_camera/dev/video1/compressed', CompressedImage, self.IMG_f120_callback)

        ##########################
        self.pub_f60_det = rospy.Publisher('/det_result/f60', Image, queue_size=1)
        self.pub_f120_det = rospy.Publisher('/det_result/f120', Image, queue_size=1)
      
        self.bridge = CvBridge()
        self.sup = []
        ##########################
       
        ##########################
        ########filter part#######

        self.real_cls_hist = 0

        ########filter part#######
        ##########################

        ##########################
        ########filter part#######

    def filtered_obs(self,traffic_light_obs):
        new_obs = []
        for obs in traffic_light_obs:
            if obs[0] == 18 and self.real_cls_hist != 0:
                obs[0] = self.real_cls_hist
            new_obs.append(obs)
        return new_obs

        ########filter part#######
        ##########################

    def get_one_boxes(self,traffic_light_obs):
        ##########################
        ########filter part#######

        traffic_light_obs = self.filtered_obs(traffic_light_obs)

        ########filter part#######
        ##########################
        if len(traffic_light_obs) >0:
            # print(f'traffic_light_obs is {traffic_light_obs}')            
            boxes = np.array(traffic_light_obs)[:,3]
            distances = [math.sqrt(((box[0] + box[2]) / 2 - self.baseline_boxes[0]) ** 2 + ((box[1] + box[3]) / 2 - self.baseline_boxes[1]) ** 2) for box in boxes]
            areas = [((box[2] - box[0]) * (box[3] - box[1])) for box in boxes]
            weights = [0.6*(distances[x] / max(distances)) + 0.4*(1 - (areas[x] / max(areas)))  for x in range(len(boxes))]
            result_box = [traffic_light_obs[weights.index(min(weights))]]
            return result_box
        else:
            result_box = traffic_light_obs
            return result_box

    def get_traffic_light_objects(self,bbox_f60):
        traffic_light_obs = []

        if len(bbox_f60) > 0:
            for traffic_light in bbox_f60:
                if traffic_light[2] > 0.2:  # if probability exceed 20%
                    traffic_light_obs.append(traffic_light)
        # sorting by size
        traffic_light_obs = self.get_one_boxes(traffic_light_obs)
        return traffic_light_obs

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

    def det_pubulissher(self,det_img,det_box,flag):
        if flag =='f60':
            det_f60_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
            self.pose_set(det_box,flag)
            self.pub_f60_det.publish(det_f60_msg)
        if flag =='f120':
            det_f120_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
            self.pub_f120_det.publish(det_f120_msg)
    
    def update_tracking(self,box_result,flag):
        update_list = []
        if len(box_result)>0:
            cls_id = np.array(box_result)[:,0]
            areas = np.array(box_result)[:,1]
            scores = np.array(box_result)[:,2]
            boxes = np.array(box_result)[:,3]
            dets_to_sort = np.empty((0,6))

            for i,box in enumerate(boxes):
                x0, y0, x1, y1 = box
                cls_name = cls_id[i]
                dets_to_sort = np.vstack((dets_to_sort, 
                            np.array([x0, y0, x1, y1, scores[i], cls_name])))
            if flag == 'f60':
                tracked_dets = self.sort_tracker_f60.update(dets_to_sort)
                tracks = self.sort_tracker_f60.getTrackers()

            bbox_xyxy = tracked_dets[:,:4]
            categories = tracked_dets[:, 4]

            new_areas = (bbox_xyxy[:,2] - bbox_xyxy[:,0]) * (bbox_xyxy[:,3] - bbox_xyxy[:,1])
            update_list = [[int(categories[x]),new_areas[x],scores[x],bbox_xyxy[x]] for x in range(len(tracked_dets)) ]

        else:
            tracked_dets = self.sort_tracker_f60.update()

        return update_list

    def image_process(self,img,flag):
        ### using with vs
        box_result = self.det_pred.steam_inference(img,conf=0.1, end2end='end2end' ,day_night=self.day_night)
        ### using shell file named 'vision.sh'
        # box_result = self.det_pred.steam_inference(img,conf=0.1, end2end=args.end2end,day_night=self.day_night)
        if flag == 'f60' :
            box_result = self.update_tracking(box_result,flag)
        if flag == 'f120' :
            box_result = self.update_tracking(box_result,flag)

        det_img = self.det_pred.draw_img(img,box_result, [255, 255, 255],self.class_names)
        tl_boxes = self.get_traffic_light_objects(box_result)
        filter_img = self.det_pred.draw_img(det_img, tl_boxes, [0, 0, 0], self.class_names)

        self.det_pubulissher(filter_img, tl_boxes,flag)
            
    def main(self):
        rate = rospy.Rate(150)
        while not rospy.is_shutdown():
            if self.get_f60_new_image:
                self.sub_f60_img['img'] = self.cur_f60_img['img']
                orig_im_f60 = copy.copy(self.sub_f60_img['img']) 
                self.image_process(orig_im_f60,'f60')
                self.get_f60_new_image = False

            rate.sleep()
          
          

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--end2end", default=False, action="store_true",help="use end2end engine")
    
    day_night_list = ['day','night']
    day_night = day_night_list[0]
    if day_night == 'day':
        parser.add_argument('--det_weight', default="/workspace/weights/yolov7/trt/new_inchen_on_pc.trt")  ### no end2end xingyou  



    if day_night == 'night':
        print('*'*12)
        print('*** NIGHT TIME ***')
        print('*'*12)
        parser.add_argument('--det_weight', default="./detection/weights/230615_night_songdo_no_nms_2.trt")  ### end2end

    args = parser.parse_args()
    
    camemra_node = Camemra_Node(args,day_night)
    camemra_node.main()

