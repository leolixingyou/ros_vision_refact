import math
import numpy as np
import pycuda.driver as cuda ## I had problem with this, so must import. Plz check yourself 
import pycuda.autoinit ## I had problem with this, so must import. Plz check yourself

from sort import *

from src_code.detection.det_infer import Predictor

##### Modify the args_init

class Detecotr_YoloV7:
    def __init__(self):
        
        self.class_names = ['person', 'bicycle','car','bus','motorcycle','truck', 'green', 'red', 'yellow',
                            'red_arrow', 'red_yellow', 'green_arrow','green_yellow','green_right',
                            'warn','black','tl_v', 'tl_p', 'traffic_sign', 'warning', 
                            'tl_bus']
        
        self.baseline_boxes = [960,270]
        
        self.args = self.args_init()

        sort_max_age = 15
        sort_min_hits = 2
        sort_iou_thresh = 0.1
        self.sort_tracker_f60 = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
        
        ### Det Model initiolization
        self.det_pred = Predictor(engine_path=self.args.det_weight , day_night=self.args.day_night)

        self.real_cls_hist = 0

    ### modifying this function
    def args_init(self):
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
    
    def filtered_obs(self,traffic_light_obs):
        new_obs = []
        for obs in traffic_light_obs:
            if obs[0] == 18 and self.real_cls_hist != 0:
                obs[0] = self.real_cls_hist
            new_obs.append(obs)
        return new_obs

    def get_one_boxes(self,traffic_light_obs):
        ########filter part#######
        traffic_light_obs = self.filtered_obs(traffic_light_obs)

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
        box_result = self.det_pred.steam_inference(img,conf=0.1, end2end='end2end' ,day_night=self.args.day_night)
        ### using shell file named 'vision.sh'
        # box_result = self.det_pred.steam_inference(img,conf=0.1, end2end=args.end2end,day_night=self.day_night)
        # box_result = self.update_tracking(box_result,flag)

        det_img = self.det_pred.draw_img(img,box_result, [255, 255, 255],self.class_names)
        tl_boxes = self.get_traffic_light_objects(box_result)
        filter_img = self.det_pred.draw_img(det_img, tl_boxes, [0, 0, 0], self.class_names)

        return filter_img, tl_boxes

