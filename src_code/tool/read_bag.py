### it can use multi topic
import os
import rosbag

def get_bag_list(path,file_type):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        print(maindir)
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in file_type:
                image_names.append(apath)
    return image_names

class Read_Bag:
    def __init__(self, bag_path, image_topic, iscalibration_flag, img_size, camera_path):
        self.params(bag_path, image_topic, iscalibration_flag, img_size, camera_path)

    def params(self, bag_path, image_topic, flag, img_size, camera_path):
        self.iscalibration_flag = flag
        bag_list = sorted(get_bag_list(bag_path, '.bag'))

        self.param_dict = {
            'image_topic': image_topic,
            'bag_list': bag_list
        }

        #### undistort image option
        if flag:
            self.param_dict.update({
                'camera_path': camera_path,
                'img_size': img_size
            })

    def read_bag(self):
        bags = [rosbag.Bag(bag_p) for bag_p in self.param_dict['bag_list']]
        bags_name = [bag_p.split('.bag')[0].split(os.sep)[-1] for bag_p in self.param_dict['bag_list']]
        return bags, bags_name
    
def run(record):
    bags, bags_name = record.read_bag()
    save_topic = record.param_dict['image_topic']
    for bag in bags:
        for i, (topic, msg, t) in enumerate(bag.read_messages(topics=save_topic)):
            print(f'{bags_name}_{topic}_{t}')
        bag.close()

if __name__ == "__main__":

    iscalibration_flag = False  # or False, depending on your requirement
    img_size = None
    camera_path = None
    if iscalibration_flag:
        img_size = (1920, 1080)
        camera_path = 'calibration/f_camera_video1.txt'

    bag_path = '/workspace/bag/'
    ## try to use one topic otherwise use 'if' when reading bag file
    image_topic = ['/gmsl_camera/dev/video0/compressed', '/gmsl_camera/dev/video1/compressed']

    record = Read_Bag(bag_path, image_topic, iscalibration_flag, img_size, camera_path)
    run(record)
