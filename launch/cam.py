from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='traffic_detect',
            executable='traffic_detect_node',
            name='traffic_detect_node',
            output='screen',
            parameters=[{
                'conf_thresh': 0.35,
                'intersection_max': 0.1
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
                    "pixel_format": "uyvy"
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
