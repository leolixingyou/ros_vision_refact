import sys
sys.path.append('/workspace')
import cv2
import copy
import argparse
import numpy as np
from src_code.detection.calibration import Calibration
from cv_bridge import CvBridge

from src_code.tl_detector.detector_yolov7 import Detecotr_YoloV7
from src_code.tool.read_bag import Read_Bag

### modifying this function
def args_init():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--end2end", default=True, action="store_true",help="use end2end engine")
    
    day_night_list = ['day','night']
    day_night = day_night_list[0]
    if day_night == 'day':
        parser.add_argument('--det_weight', default="/workspace/weights/yolov7/trt_3080ti/new_incheon.trt")  ### no end2end xingyou  
        parser.add_argument('--day_night', default="day")  ### no end2end xingyou  

    if day_night == 'night':
        parser.add_argument('--det_weight', default="./detection/weights/230615_night_songdo_no_nms_2.trt")  ### end2end
        parser.add_argument('--day_night', default="night")  ### end2end

    args = parser.parse_args()
    return args

class Run_Bag_YOLO:
    def __init__(self, record, detector_yolov) -> None:
        self.detector_yolov = detector_yolov
        self.record = record
        self.bridge = CvBridge()

    def image_read(self,msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if self.record.iscalibration_flag:
            img = cv2.resize(img, (self.width,self.height))
            img = self.calib.undistort(img)
            
        cur_img = copy.copy(img)
        return cur_img

def run(run_bag):
    bags, bags_name = run_bag.record.read_bag()
    save_topic = run_bag.record.param_dict['image_topic']
    for bag in bags:
        for i, (topic, msg, t) in enumerate(bag.read_messages(topics=save_topic)):
            ### modify the target topic 
            if topic == '/gmsl_camera/dev/video0/compressed':
                image = run_bag.image_read(msg)
                processed_img, _ = run_bag.detector_yolov.image_process(image,'f60')
                cv2.imshow('temp_f60',processed_img)
                cv2.waitKey(0)
        bag.close()

if __name__ == "__main__":
    args = args_init()
    detector_yolov = Detecotr_YoloV7(args)

    iscalibration_flag = False  # or False, depending on your requirement
    img_size = None
    camera_path = None
    if iscalibration_flag:
        img_size = (1920, 1080)
        camera_path = 'calibration/f_camera_video1.txt'

    bag_path = '/workspace/demo/bag/'
    ## try to use one topic otherwise use 'if' when reading bag file
    image_topic = ['/gmsl_camera/dev/video0/compressed', '/gmsl_camera/dev/video1/compressed']

    record = Read_Bag(bag_path, image_topic, iscalibration_flag, img_size, camera_path)

    run_bag = Run_Bag_YOLO(record, detector_yolov)
    run(run_bag)
