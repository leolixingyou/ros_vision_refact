import os
import time
import math
import json
import numpy as np
import geopandas as gpd
import cv2

import rospy
from novatel_oem7_msgs.msg import INSPVA
from geometry_msgs.msg import PoseArray, Pose

from math import pi, sin, cos, atan2, sqrt

# translation with front only 1.78m by laser
CAM2GPS = 1.78 

def parse_c1_trafficlight(path):
    shape_data = gpd.read_file(path)
    c1_trafficlight = []
    for index, data in shape_data.iterrows():
        c1_trafficlight.append(data)
    return c1_trafficlight

class LaneletMap:
    def __init__(self, map_path):
        with open(map_path, 'r') as f:
            map_data = json.load(f)
            print(f'map_path us {map_path}')
        self.map_data = map_data
        self.lanelets = map_data['lanelets']
        self.groups = map_data['groups']
        self.precision = map_data['precision']
        self.for_viz = map_data['for_vis']
        self.basella = map_data['base_lla']
        self.surfacemarks = map_data['surfacemarks']
        self.stoplines = map_data['stoplines']
        self.traffic_light = map_data['trafficlights']

def gps_to_meter(point_1, point_2) -> float:
    long1, lat1 = point_1
    long2, lat2 = point_2
    R = 6378.137 # radius of the earth in KM
    lat_to_deg = lat2 * pi/180 - lat1 * pi/180
    long_to_deg = long2 * pi/180 - long1 * pi/180

    a = sin(lat_to_deg/2)**2 + cos(lat1 * pi/180) * cos(lat2 * pi/180) * sin(long_to_deg/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c
    
    return d * 1000 # meter


class GPS_RECIEVER:
    def __init__(self):
        rospy.init_node('GPS')
        self.get_new_info = False
        rospy.Subscriber('/novatel/oem7/inspva', INSPVA, self.GPS_callback)
        self.pub_distance_gps = rospy.Publisher('/tl_distance/pose', PoseArray, queue_size=1)

        self.latitude = None
        self.longitude = None

    def GPS_callback(self,msg):
        self.latitude = msg.latitude
        self.longitude = msg.longitude

    def pose_set(self,bboxes):
        bbox_pose = PoseArray()
        for _ in range(1):
            pose = Pose()
            pose.position.x = bboxes[0]# box class 'position.x is tl to my distance(m)'
            pose.position.y = 0.0
            pose.position.z = 0.0
            pose.orientation.x = bboxes[1][0] # 'my ENU coordinate lon'
            pose.orientation.y = bboxes[1][1] # 'my ENU coordinate lat'
            pose.orientation.z = bboxes[2][0] # 'tl ENU coordinate lon'
            pose.orientation.w = bboxes[2][1] # 'tl ENU coordinate lat'
            bbox_pose.poses.append(pose)

        self.pub_distance_gps.publish(bbox_pose)
    
    def main(self,tl_coor):
        while not rospy.is_shutdown():
            if self.longitude is not None and self.latitude is not None:
                my_coor = (self.longitude, self.latitude)
                distance = gps_to_meter(my_coor, tl_coor) - CAM2GPS
                self.pose_set([distance, my_coor, tl_coor])

if __name__ == "__main__":
    input = LaneletMap('/workspace/mobinha_xingyou/src/mobinha/selfdrive/planning/map/songdo.json')
    temP_list = parse_c1_trafficlight('/workspace/mobinha_xingyou/src/mobinha/selfdrive/planning/map/songdo/C1_TRAFFICLIGHT.shp')
    tar_tl_DI = 'C1225A120043'
    selected_tl = [list(x.geometry.coords)[0] for x in temP_list if x.ID==tar_tl_DI][0][:2]

    # bag_file = 
    gps_listener = GPS_RECIEVER()
    gps_listener.main(selected_tl)
