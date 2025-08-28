from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='deepstream_tracker',
            executable='deepstream_tracker_node',
            name='traffic_detect_node',
            output='screen',
            parameters=[{
                'source_topic': '/image_raw'

            }]
        ),
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam_node',
            parameters=[{
                    "video_device": "/dev/video0",
                    "image_width": 640,
                    "image_height": 480,
                    "framerate": 30.0,
                    #"pixel_format": "uyvy",
                    "pixel_format": "yuyv",
                    "output_encoding": "bgr8",
                    "io_method": "mmap"
                }],  
            output='screen'
        ), 
        Node(
            package='detections_img',
            executable='detections_img_node',
            name='detections_img_node',
            output='screen'
        )
    ])
