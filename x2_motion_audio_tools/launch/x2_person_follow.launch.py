"""Launch camera+lidar person detection, torso tracking, and optional following."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    camera_topic = LaunchConfiguration("camera_topic")
    lidar_topic = LaunchConfiguration("lidar_topic")
    device = LaunchConfiguration("device")
    follow_enabled = LaunchConfiguration("follow_enabled")
    stop_distance_m = LaunchConfiguration("stop_distance_m")
    lidar_angle_offset_deg = LaunchConfiguration("lidar_angle_offset_deg")
    waist_tracking_enabled = LaunchConfiguration("waist_tracking_enabled")
    waist_state_topic = LaunchConfiguration("waist_state_topic")
    waist_command_topic = LaunchConfiguration("waist_command_topic")
    waist_yaw_gain = LaunchConfiguration("waist_yaw_gain")
    waist_soft_limit_deg = LaunchConfiguration("waist_soft_limit_deg")
    waist_max_velocity = LaunchConfiguration("waist_max_velocity")
    waist_invert_direction = LaunchConfiguration("waist_invert_direction")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_topic",
                default_value="/aima/hal/sensor/rgbd_head_front/rgb_image",
                description="ROS image topic to run YOLO on.",
            ),
            DeclareLaunchArgument(
                "lidar_topic",
                default_value="/scan",
                description="sensor_msgs/LaserScan topic used for distance.",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cpu",
                description="YOLO inference device: cpu, cuda, or mps.",
            ),
            DeclareLaunchArgument(
                "follow_enabled",
                default_value="false",
                description="When true, publish locomotion commands.",
            ),
            DeclareLaunchArgument(
                "stop_distance_m",
                default_value="1.2",
                description="Distance where the robot stops approaching.",
            ),
            DeclareLaunchArgument(
                "lidar_angle_offset_deg",
                default_value="0.0",
                description="Offset between camera bearing and lidar frame.",
            ),
            DeclareLaunchArgument(
                "waist_tracking_enabled",
                default_value="true",
                description="When true, turn the torso toward visible people.",
            ),
            DeclareLaunchArgument(
                "waist_state_topic",
                default_value="/aima/hal/joint/waist/state",
                description="HAL waist joint state topic.",
            ),
            DeclareLaunchArgument(
                "waist_command_topic",
                default_value="/aima/hal/joint/waist/command",
                description="HAL waist joint command topic.",
            ),
            DeclareLaunchArgument(
                "waist_yaw_gain",
                default_value="1.0",
                description="Visual-servo gain from camera bearing to waist yaw.",
            ),
            DeclareLaunchArgument(
                "waist_soft_limit_deg",
                default_value="90.0",
                description="Symmetric software yaw limit for torso tracking.",
            ),
            DeclareLaunchArgument(
                "waist_max_velocity",
                default_value="0.7",
                description="Maximum commanded waist joint speed in rad/s.",
            ),
            DeclareLaunchArgument(
                "waist_invert_direction",
                default_value="false",
                description="Flip waist yaw direction if the torso turns away.",
            ),
            Node(
                package="x2_motion_audio_tools",
                executable="x2_person_follow",
                name="x2_person_follow",
                output="screen",
                parameters=[
                    {
                        "camera_topic": camera_topic,
                        "lidar_topic": lidar_topic,
                        "device": device,
                        "follow_enabled": ParameterValue(
                            follow_enabled, value_type=bool
                        ),
                        "stop_distance_m": ParameterValue(
                            stop_distance_m, value_type=float
                        ),
                        "lidar_angle_offset_deg": ParameterValue(
                            lidar_angle_offset_deg, value_type=float
                        ),
                        "waist_tracking_enabled": ParameterValue(
                            waist_tracking_enabled, value_type=bool
                        ),
                        "waist_state_topic": waist_state_topic,
                        "waist_command_topic": waist_command_topic,
                        "waist_yaw_gain": ParameterValue(
                            waist_yaw_gain, value_type=float
                        ),
                        "waist_soft_limit_deg": ParameterValue(
                            waist_soft_limit_deg, value_type=float
                        ),
                        "waist_max_velocity": ParameterValue(
                            waist_max_velocity, value_type=float
                        ),
                        "waist_invert_direction": ParameterValue(
                            waist_invert_direction, value_type=bool
                        ),
                    }
                ],
            ),
        ]
    )
