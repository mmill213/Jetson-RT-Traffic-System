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
            output='screen',
            parameters=[{
                "video_device": "/dev/video0",
                "image_width": 1280,
                "image_height": 720,
                "framerate": 30.0,
                "pixel_format": "uyvy",
                "io_method": "mmap",

                "brightness": 2,
                "contrast": 3,
                "saturation": 4,
                "sharpness": 1,
                "gain": 2,
                "white_balance_automatic": True,
                "auto_exposure": 3
            }]
        ), 
        Node(
            package='detections_img',
            executable='detections_img_node',
            name='detections_img_node',
            output='screen'
        )
    ])
