import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


import numpy as np  # just once at top


class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        
        # add cli arg to path to vid file
        self.declare_parameter('video_path', '')
        self.declare_parameter('width', 0)
        self.declare_parameter('height', 0)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('quartered', False)

        fps = self.get_parameter('fps').get_parameter_value().double_value
        video_path = self.get_parameter('video_path').get_parameter_value().string_value
        width = self.get_parameter('width').get_parameter_value().integer_value
        height = self.get_parameter('height').get_parameter_value().integer_value

        self.quartered = self.get_parameter('quartered').get_parameter_value().bool_value

        if not video_path:
            self.get_logger().error('Parameter "video_path" not set or empty! Exiting...')
            raise RuntimeError('video_path parameter not set')

        self.resize_enabled = width > 0 and height > 0
        self.resize_dims = (width, height)
        
        
        if not video_path:
            self.get_logger().error('Parameter "video_path" not set or empty! Exiting...')
            raise RuntimeError('video_path parameter not set')
        

        
        # build publisher and utils
        self.publisher_ = self.create_publisher(Image, '/image_raw', 10)
        self.timer = self.create_timer( 1.0 / fps, self.timer_callback)
        self.video_capture = cv2.VideoCapture(video_path)
        
        
        if not self.video_capture.isOpened():
            self.get_logger().error(f'Failed to open video file: {video_path}')
            raise RuntimeError(f'Failed to open video file: {video_path}')
        
        
        self.bridge = CvBridge()


    def timer_callback(self):
        ret, frame = self.video_capture.read()
        if ret:
            if self.quartered:
                orig_height, orig_width = frame.shape[:2]

                new_height = orig_height // 2
                new_width = int(orig_width * (new_height / orig_height))

                resized_frame = cv2.resize(frame, (new_width, new_height))

                # Create black image using numpy zeros
                output_frame = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)

                # Place resized frame top-left
                output_frame[0:new_height, 0:new_width] = resized_frame

                ros_image = self.bridge.cv2_to_imgmsg(output_frame, encoding='bgr8')
                self.publisher_.publish(ros_image)
                self.get_logger().info('Publishing video frame')
            else:

                if self.resize_enabled:
                    frame = cv2.resize(frame, self.resize_dims)
           
                ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            
                self.publisher_.publish(ros_image)
                self.get_logger().info('Publishing video frame')
                
        else:
            self.get_logger().info('End of video stream, restarting...')
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
