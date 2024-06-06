import os
from utils.utils import BaseEngine
import numpy as np
import cv2
import pycuda.driver as cuda
import time
import copy

def get_file_list(path, ftype):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
      print(maindir)
      for filename in file_name_list:
          apath = os.path.join(maindir, filename)
          ext = os.path.splitext(apath)[1]
          if ext in ftype:
              image_names.append(apath)
    return image_names

def make_dir(dir):
    if(not os.path.exists(dir)):
        os.makedirs(dir)

RGB_DAY_LIST = [(148, 108, 138), (207, 134, 240), (138, 88, 55), (26, 99, 86), (38, 125, 80), (72, 80, 57), # 6 
                (118, 23, 200), (117, 189, 183), (0, 128, 128), (60, 185, 90), (20, 20, 150), (13, 131, 204), 
                (30, 200, 200), (43, 38, 105), (104, 235, 178), (135, 68, 28), (140, 202, 15), (67, 115, 220),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30)]


def draw_img_filter(orig_img, resized_img, boxes, class_names, off_set, target_size, remapping = False):
    # print(f'boxes is {boxes}')
    height, weight, _ = resized_img.shape
    tl = 3 or round(0.002 * (height + weight) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)  # font thickness
    cur_img = copy.copy(resized_img)
    if remapping:
        cur_img_2 = copy.copy(orig_img)

    if len(boxes) > 0 :
      for box_info in boxes:
        box = box_info[3]
        cls_id = box_info[0]
        score = box_info[2]

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        x0 = 0 if  x0 < 0 else x0
        y0 = 0 if  y0 < 0 else y0

        _COLORS = RGB_DAY_LIST

        c1, c2 = (x0,y0), (x1,y1)
        cv2.rectangle(cur_img, c1, c2, _COLORS[6], thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        text = '{}'.format('traffic light')
        t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(cur_img, c1, c2, _COLORS[6], -1, cv2.LINE_AA)  # filled
        cv2.putText(cur_img, text, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
        
        if remapping:
            ### remapping part calculation
            x0_new = int((box[0]/weight) * target_size[0]) + off_set[0]
            y0_new = int((box[1]/height) * target_size[1]) + off_set[1]
            x1_new = int((box[2]/weight) * target_size[0]) + off_set[0]
            y1_new = int((box[3]/height) * target_size[1]) + off_set[1]

            c1_new, c2_new = (x0_new,y0_new), (x1_new,y1_new)
            cv2.rectangle(cur_img_2, c1_new, c2_new, _COLORS[6], thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            text = '{}'.format('traffic light')
            t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
            c2_new = c1_new[0] + t_size[0], c1_new[1] - t_size[1] - 3
            cv2.rectangle(cur_img_2, c1_new, c2_new, _COLORS[6], -1, cv2.LINE_AA)  # filled
            cv2.putText(cur_img_2, text, (c1_new[0], c1_new[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    if remapping:
        img = cur_img
        img_2 = cur_img_2
        return img, img_2
    
    else:
        img = cur_img
        return img, None

class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 18  # your model classes
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'green3_h', 'bus',
                             'red3_h', 'truck', 'yellow3_h', 'green4_h', 'red4_h', 'yellow4_h',
                             'redgreen4_h', 'redyellow4_h', 'greenarrow4_h', 'red_v', 'yellow_v', 'green_v']
        
if __name__ == '__main__':
    ### integrate_each_class_fp16_0314 weight with result in ./test_infe_data/result 
    # pred = Predictor(engine_path='./weights/onnx/integrate_each_class_fp16_0314.trt')
    pred = Predictor(engine_path='/workspace/weights/yolov7/trt/integrate_each_class_fp16_0314.trt')
    class_name = pred.class_names
    print(class_name)

    save_dir_ori = f"/workspace/demo/runs/tl_distance/tl_{time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))}/"

    # pred.get_fps()

    ftype = ['.png']
    img_path_ori = '/workspace/demo/runs/img_tl_dis/'
    came_list = ['f60', 'f120']
    for cam_name in came_list:
        img_path = f'{img_path_ori}{cam_name}'

        if cam_name == 'f60':
            img_list = sorted(get_file_list(img_path,ftype))
            off_set = [783,440]
            target_size = [1137,640]
            save_dir = f'{save_dir_ori}{cam_name}/'


        if cam_name == 'f120':    
            img_list = sorted(get_file_list(img_path,ftype))
            off_set = [783,440]
            target_size = [1137,640]
            save_dir = f'{save_dir_ori}{cam_name}/'

        resized_img_dir = f"{save_dir}resized/"
        origi_img_dir = f"{save_dir}origi/"
        mapped_origi_img_dir = f"{save_dir}mapped/"
        make_dir(save_dir) 
        make_dir(resized_img_dir)
        make_dir(origi_img_dir)
        make_dir(mapped_origi_img_dir)

        for pfile in img_list:
            fname = pfile.split('/')[-1].split(ftype[0])[0]
            orig_img = cv2.imread(pfile)
            resize_small_img= orig_img[off_set[1]:off_set[1]+target_size[1], off_set[0]:off_set[0]+target_size[0]]
            resized_img = cv2.resize(resize_small_img,(1920,1080))

            box_result_ori = pred.steam_inference(orig_img, conf=0.1, end2end=True,day_night=1)
            box_result_resized = pred.steam_inference(resized_img, conf=0.1, end2end=True,day_night=1)
        
            draw_resized_img, draw_mapped_img = draw_img_filter(orig_img, resized_img, box_result_resized, class_name, off_set, target_size, remapping=True)
            draw_orig_img, _ = draw_img_filter(None, orig_img, box_result_ori, class_name, off_set, target_size, remapping=False)

            save_resized_img = f'{resized_img_dir}{fname}{ftype[0]}'
            save_origi_img = f'{origi_img_dir}{fname}{ftype[0]}'
            save_mapped_origi_img = f'{mapped_origi_img_dir}{fname}{ftype[0]}'
            cv2.imwrite(save_resized_img, draw_resized_img)
            cv2.imwrite(save_origi_img, draw_orig_img)
            cv2.imwrite(save_mapped_origi_img, draw_mapped_img)
