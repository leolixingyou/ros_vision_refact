import os
import time
import numpy as np
import copy
import cv2

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Pose

class Save_Image_with_TL_Dis:
    def __init__(self,save_dir):
        rospy.init_node('Camemra_node')
        self.fps_list=[]

        self.save_dir = save_dir

        self.cur_f60_img = {'img':None, 'header':None}

        self.cur_f120_img = {'img':None, 'header':None}

        self.tl_distance = -1000
        self.hist_dist = -1000
        
        self.distance_offset = 0.5 # meter previous was 0.8m

        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_f60_callback) 
        rospy.Subscriber('/gmsl_camera/dev/video1/compressed', CompressedImage, self.IMG_f120_callback) 
        rospy.Subscriber('/tl_distance/pose', PoseArray, self.TL_Dis_callback) 

    def TL_Dis_callback(self,msg):
        self.tl_distance = msg.poses[0].position.x

    def IMG_f60_callback(self,msg):
        t1 = time.time()
        np_arr = np.frombuffer(msg.data, np.uint8)
        front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.cur_f60_img['img'] = front_img
        self.cur_f60_img['header'] = msg.header

    def IMG_f120_callback(self,msg):
        t1 = time.time()
        np_arr = np.frombuffer(msg.data, np.uint8)
        front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.cur_f120_img['img'] = front_img
        self.cur_f120_img['header'] = msg.header

    def make_dir(self,dir):
        if(not os.path.exists(dir)):
            os.makedirs(dir)

    def image_process(self,img, tl_distance,flag):
        save_dir_img = f'{self.save_dir}{flag}/'
        self.make_dir(save_dir_img)
        cv2.imwrite(f"{save_dir_img}tl_{int(tl_distance)}.png", img)

    def run(self):
        while not rospy.is_shutdown():
            tl_distance = self.tl_distance
            if tl_distance <= 180 and tl_distance > 0:
                if abs(self.hist_dist - tl_distance) > self.distance_offset:
                    self.hist_dist = tl_distance

                    if self.cur_f60_img['img'] is not None:
                        orig_im_f60 = copy.copy(self.cur_f60_img['img']) 
                        self.image_process(orig_im_f60, tl_distance, 'f60')
                        self.get_f60_new_image = False

                    if self.cur_f120_img['img'] is not None:
                        orig_im_f120 = copy.copy(self.cur_f120_img['img']) 
                        self.image_process(orig_im_f120, tl_distance, 'f120')
                        self.get_f120_new_image = False


if __name__ == "__main__":
    save_dir = '/workspace/demo/runs/img_tl_dis_1/'
    img2tl_dis = Save_Image_with_TL_Dis(save_dir)
    img2tl_dis.run()

