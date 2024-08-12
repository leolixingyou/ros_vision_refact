import numpy as np
import cv2

segmentation_colors = np.array([[0,    0,    0],
								[255,  191,  0],
								[192,  67,   251],
								[0,    255,  0],
								[0,  0,    255],
								[255,  0,    0]], dtype=np.uint8)
segmentation_gray   = np.array([[0,    0,    0],
								[1,    1,    1],
								[2,    2,    2],
								[3,    3,    3],
								[4,    4,    4],
								[5,    5,    5]], dtype=np.uint8)
# segmentation_colors = np.array([[0,    0,    0],
# 								[255,  191,  0],
# 								[192,  67,   251]], dtype=np.uint8)

labels_v = ['vehicle', 'truck', 'schoollBus', 'otherCar', 'trafficLight', 'trafficSign', 'bicycle', 'pedestrian', 'twoWheeler', 'bus', 'motorcycle', 'ambulance', 'policeCar']
labels_b = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motor', 'bike', 'traffic light', 'traffic sign']
detection_colors = [(np.random.randint(0, 255), np.random.randint(0, 255),np.random.randint(0, 255)) for _ in range(len(labels_v))]

ORIGINAL_HORIZON_POINTS = np.float32([[571, 337], [652, 337]])

num_horizon_points = 0
new_horizon_points = []
def util_draw_seg(seg_map, image, alpha = 0.5):

	# color_segmap = cv2.resize(image, (seg_map.shape[1], seg_map.shape[0]))
	# color_segmap[seg_map>0] = segmentation_colors[seg_map[seg_map>0]]	
	# color_segmap = cv2.resize(color_segmap, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_NEAREST)

	gray_segmap = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
	# gray_segmap[seg_map>0] = segmentation_gray[seg_map[seg_map>0]]	
	gray_segmap[seg_map>0] = segmentation_colors[seg_map[seg_map>0]]	
	###lane only
	# gray_segmap[seg_map==2] = segmentation_gray[seg_map[seg_map==2]]	
	gray_segmap = cv2.resize(gray_segmap, (image.shape[1],image.shape[0]))
	# gray_segmap = cv2.resize(gray_segmap, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_NEAREST)
	

	# color_segmap = cv2.resize(color_segmap, (image.shape[1],image.shape[0]))

	# if(alpha == 0):
	# 	combined_img = np.hstack((image, color_segmap))
	# else:
	# 	combined_img = cv2.addWeighted(image, alpha, color_segmap, (1-alpha),0)

	return  gray_segmap
	# return combined_img, gray_segmap

def util_draw_detections(boxes, scores, idxs, image, text=True, nc=10):
	labels = labels_v
	if nc == 10:
		labels = labels_b
	tl = int(round(0.0015 * max(image.shape[0:2])))  # line thickness
	tf = max(tl, 1)  # font thickness
	for box, score, idx in zip(boxes, scores, idxs):
		c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
		cv2.rectangle(image, c1, c2, detection_colors[idx], thickness=tl)
		if text:
			s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
			t_size = cv2.getTextSize(labels[idx], 0, fontScale=float(tl) / 3, thickness=tf)[0]
			c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
			cv2.rectangle(image, c1, c2, detection_colors[idx], -1)  # filled
			cv2.putText(image, '{}: {:.0%}'.format(labels[idx], score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
						thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

	return image

def util_draw_bird_eye_view(seg_map, hoizon_points=ORIGINAL_HORIZON_POINTS):

	img_h, img_w = seg_map.shape[:2]
	bird_eye_view_w, bird_eye_view_h = (img_h, img_h)
	offset = bird_eye_view_w/2.5
	bird_eye_view_points = np.float32([[offset, bird_eye_view_h], [bird_eye_view_w - offset, bird_eye_view_h], 
										[offset, 0], [bird_eye_view_w - offset, 0]])

	image_points = np.vstack((np.float32([[0, img_h], [img_w, img_h]]), hoizon_points))
	M = cv2.getPerspectiveTransform(image_points, bird_eye_view_points)
	bird_eye_seg_map = cv2.warpPerspective(seg_map, M, (bird_eye_view_w, bird_eye_view_h))
	return bird_eye_seg_map

def transform_boxes(boxes, anchors):

	y_centers_a = (anchors[:, 0] + anchors[:, 2]) / 2
	x_centers_a = (anchors[:, 1] + anchors[:, 3]) / 2
	ha = anchors[:, 2] - anchors[:, 0]
	wa = anchors[:, 3] - anchors[:, 1]

	w = np.exp(boxes[:, 3]) * wa
	h = np.exp(boxes[:, 2]) * ha

	y_centers = boxes[:, 0] * ha + y_centers_a
	x_centers = boxes[:, 1] * wa + x_centers_a

	ymin = y_centers - h / 2.
	xmin = x_centers - w / 2.
	ymax = y_centers + h / 2.
	xmax = x_centers + w / 2.

	return np.vstack((xmin, ymin, xmax, ymax)).T

def iou_np(box, boxes, area, areas):

	x_min = np.maximum(box[0], boxes[:,0])
	y_min = np.maximum(box[1], boxes[:,1])
	x_max = np.minimum(box[2], boxes[:,2])
	y_max = np.minimum(box[3], boxes[:,3])

	w = np.maximum(0, x_max - x_min + 1)
	h = np.maximum(0, y_max - y_min + 1)
	intersect = w*h
	
	iou_np = intersect / (area + areas - intersect)
	return iou_np

def nms_fast(bboxes, scores, idx, iou_threshold=0.5):
	 
	areas = (bboxes[:,2] - bboxes[:,0] + 1) \
			 * (bboxes[:,3] - bboxes[:,1] + 1)
	
	sort_index = np.argsort(scores)
	
	i = -1
	while(len(sort_index) >= 1 - i):

		max_scr_ind = sort_index[i]
		ind_list = sort_index[:i]

		iou = iou_np(bboxes[max_scr_ind], bboxes[ind_list], \
					 areas[max_scr_ind], areas[ind_list])
		
		del_index = np.where(iou >= iou_threshold)
		sort_index = np.delete(sort_index, del_index)
		i -= 1
	
	bboxes = bboxes[sort_index]
	scores = scores[sort_index]
	idx = idx[sort_index]
	
	return bboxes, scores, idx

def get_horizon_points(image):

	cv2.namedWindow("Get horizon points", cv2.WINDOW_NORMAL)
	cv2.setMouseCallback("Get horizon points", get_horizon_point)

	image = cv2.line(image, (0,image.shape[0]//2), 
							(image.shape[1],image.shape[0]//2), 
							(0,  0,   251), 1)

	cv2.imshow("Get horizon points", image)

	num_lines = 0
	while True:

		if (num_lines == 0) and (num_horizon_points == 1):

			image = cv2.line(image, (0,image.shape[0]), 
							(new_horizon_points[0][0], new_horizon_points[0][1]), 
							(192,  67,   251), 3)

			image = cv2.circle(image, (new_horizon_points[0][0], new_horizon_points[0][1]), 
							5, (251,  191,   67), -1)
			
			cv2.imshow("Get horizon points", image)
			num_lines += 1

		elif(num_lines == 1) and (num_horizon_points == 2):

			image = cv2.line(image, (image.shape[1],image.shape[0]), 
				(new_horizon_points[1][0], new_horizon_points[1][1]), 
				(192,  67,   251), 3)

			image = cv2.circle(image, (new_horizon_points[1][0], new_horizon_points[1][1]), 
								5, (251,  191,   67), -1)
			
			cv2.imshow("Get horizon points", image)
			num_lines += 1
			break

		cv2.waitKey(100)

	cv2.waitKey(1000)
	cv2.destroyWindow("Get horizon points")

	horizon_points = np.float32(new_horizon_points)
	print(f"horizon_points = np.{repr(horizon_points)}")

	return horizon_points

def get_horizon_point(event,x,y,flags,param):

	global num_horizon_points, new_horizon_points 

	if event == cv2.EVENT_LBUTTONDBLCLK:

		new_horizon_points.append([x,y])
		num_horizon_points += 1
