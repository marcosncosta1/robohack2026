"""Launch only the simple stereo image viewer/relay."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera = LaunchConfiguration("camera")
    topic_type = LaunchConfiguration("topic_type")
    image_topic = LaunchConfiguration("image_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera",
                default_value="stereo_head_front_left",
                description="stereo_head_front_left or stereo_head_front_right",
            ),
            DeclareLaunchArgument(
                "topic_type",
                default_value="rgb_image",
                description="rgb_image or rgb_image_compressed",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="",
                description="Optional direct image topic override",
            ),
            Node(
                package="yolo_person_detector",
                executable="stereo_image_view_node",
                name="stereo_image_view",
                output="screen",
                parameters=[
                    {
                        "camera": camera,
                        "topic_type": topic_type,
                        "image_topic": image_topic,
                    }
                ],
            ),
        ]
    )
