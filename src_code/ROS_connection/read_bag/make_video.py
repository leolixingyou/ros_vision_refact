import rosbag
import cv2
from cv_bridge import CvBridge
import os





# Path to the ROS bag file
bag_file_path = '/workspace/demo/bag/2024-03-17-18-33-20.bag'
# Topic name where compressed images are published
image_topic = '/gmsl_camera/dev/video0/compressed'
# Output video file path
output_video_path = 'output_video.avi'
# Frame per second for the output video
fps = 30

# Initialize CvBridge
bridge = CvBridge()

# Open bag file
bag = rosbag.Bag(bag_file_path, 'r')

# Get image size from the first image message
first_msg = next(bag.read_messages(topics=[image_topic]))[1]
cv_image = bridge.compressed_imgmsg_to_cv2(first_msg, desired_encoding='passthrough')
height, width, layers = cv_image.shape
size = (width, height)

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

# Iterate over all messages in the bag file
for topic, msg, t in bag.read_messages(topics=[image_topic]):
    try:
        # Convert the compressed image message to OpenCV image
        cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        
        # Write frame to video
        out.write(cv_image)
    except Exception as e:
        print(f"Error processing frame: {e}")

# Release resources
bag.close()
out.release()

print(f"Video saved to {output_video_path}")
