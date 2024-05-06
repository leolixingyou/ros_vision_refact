import os
import sys
import time
import math
import numpy as np
import copy


import rospy
from geometry_msgs.msg import Pose

class DISTANCE_REF_HD_MAP:
    def __init__(self):
        rospy.init_node('Distance from HD map')
        rospy.Subscriber('/mobinha/planning/lane_information',Pose, self.lane_information_cb)
        self.lane_information = None
        self.M_TO_IDX = 1/CP.mapParam.precision
        self.IDX_TO_M = CP.mapParam.precision

        
    def lane_information_cb(self, msg):
        # [0] id, [1] forward_direction, [2] stop line distance [3] forward_curvature
        self.lane_information = [msg.position.x,msg.position.y, msg.position.z, msg.orientation.x]

    def distance_ref_HD_map(self,):
        pass

    def main(self):
        while not rospy.is_shutdown():
            pass

if __name__ == "__main__":

    camemra_node = DISTANCE_REF_HD_MAP()
    camemra_node.main()

