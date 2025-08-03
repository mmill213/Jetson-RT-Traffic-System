from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():

    video_path_arg = DeclareLaunchArgument(
        'video_path',
        default_value= os.path.expanduser('~/Jetson-RT-Traffic-System/video/vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4'),
        description='Path to the video file to publish'
    )

    return LaunchDescription([
        video_path_arg,

        Node(
            package='traffic_detect',
            executable='traffic_detect_node',
            name='traffic_detect_node',
            output='screen',
            parameters=[{
                'conf_thresh': 0.55,
                'intersection_max': 0.2

            }]
        ),
        Node(
            package='videopub',
            executable='video_publisher',
            name='video_publisher_node',
            output='screen',
            parameters=[{
                'video_path': LaunchConfiguration('video_path'),
                'width': 960,
                'height': 544
                }]
        ),

        
        Node(
            package='detections_img',
            executable='detections_img_node',
            name='detections_img_node',
            output='screen'
        ),
    ])
