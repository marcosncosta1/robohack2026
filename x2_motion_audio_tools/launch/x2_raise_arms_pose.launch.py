"""Launch the one-shot chair-assist arm pose."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    auto_start = LaunchConfiguration("auto_start")
    trigger_topic = LaunchConfiguration("trigger_topic")
    shoulder_pitch_deg = LaunchConfiguration("shoulder_pitch_deg")
    shoulder_roll_deg = LaunchConfiguration("shoulder_roll_deg")
    shoulder_yaw_deg = LaunchConfiguration("shoulder_yaw_deg")
    elbow_bend_deg = LaunchConfiguration("elbow_bend_deg")
    move_seconds = LaunchConfiguration("move_seconds")
    hold_indefinitely = LaunchConfiguration("hold_indefinitely")
    hold_seconds = LaunchConfiguration("hold_seconds")
    release_seconds = LaunchConfiguration("release_seconds")
    move_stiffness = LaunchConfiguration("move_stiffness")
    move_damping = LaunchConfiguration("move_damping")
    hold_stiffness = LaunchConfiguration("hold_stiffness")
    hold_damping = LaunchConfiguration("hold_damping")
    require_no_other_arm_publishers = LaunchConfiguration(
        "require_no_other_arm_publishers"
    )
    require_balanced_mode = LaunchConfiguration("require_balanced_mode")
    require_balanced_status = LaunchConfiguration("require_balanced_status")
    control_hz = LaunchConfiguration("control_hz")
    arm_state_timeout_s = LaunchConfiguration("arm_state_timeout_s")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("x2_motion_audio_tools"),
                        "config",
                        "x2_raise_arms_pose.yaml",
                    ]
                ),
                description="YAML parameter file for the arm pose routine.",
            ),
            DeclareLaunchArgument(
                "auto_start",
                default_value="true",
                description="Run the arm pose immediately instead of waiting for trigger.",
            ),
            DeclareLaunchArgument(
                "trigger_topic",
                default_value="/x2/assist/raise_arms_trigger",
                description="Bool topic used to start or deactivate the arm pose.",
            ),
            DeclareLaunchArgument("shoulder_pitch_deg", default_value="10.0"),
            DeclareLaunchArgument("shoulder_roll_deg", default_value="0.0"),
            DeclareLaunchArgument("shoulder_yaw_deg", default_value="0.0"),
            DeclareLaunchArgument("elbow_bend_deg", default_value="90.0"),
            DeclareLaunchArgument("move_seconds", default_value="3.0"),
            DeclareLaunchArgument("hold_indefinitely", default_value="true"),
            DeclareLaunchArgument("hold_seconds", default_value="3.0"),
            DeclareLaunchArgument("release_seconds", default_value="0.5"),
            DeclareLaunchArgument("move_stiffness", default_value="8.0"),
            DeclareLaunchArgument("move_damping", default_value="0.8"),
            DeclareLaunchArgument("hold_stiffness", default_value="8.0"),
            DeclareLaunchArgument("hold_damping", default_value="0.8"),
            DeclareLaunchArgument(
                "require_no_other_arm_publishers",
                default_value="true",
                description="Abort if another node is already publishing arm HAL commands.",
            ),
            DeclareLaunchArgument("require_balanced_mode", default_value="true"),
            DeclareLaunchArgument(
                "require_balanced_status",
                default_value="false",
                description="When true, require GetMcAction status value 100.",
            ),
            DeclareLaunchArgument("control_hz", default_value="200.0"),
            DeclareLaunchArgument("arm_state_timeout_s", default_value="5.0"),
            Node(
                package="x2_motion_audio_tools",
                executable="x2_raise_arms_pose",
                name="x2_raise_arms_pose",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "auto_start": ParameterValue(auto_start, value_type=bool),
                        "trigger_topic": trigger_topic,
                        "shoulder_pitch_deg": ParameterValue(
                            shoulder_pitch_deg, value_type=float
                        ),
                        "shoulder_roll_deg": ParameterValue(
                            shoulder_roll_deg, value_type=float
                        ),
                        "shoulder_yaw_deg": ParameterValue(
                            shoulder_yaw_deg, value_type=float
                        ),
                        "elbow_bend_deg": ParameterValue(
                            elbow_bend_deg, value_type=float
                        ),
                        "move_seconds": ParameterValue(
                            move_seconds, value_type=float
                        ),
                        "hold_indefinitely": ParameterValue(
                            hold_indefinitely, value_type=bool
                        ),
                        "hold_seconds": ParameterValue(
                            hold_seconds, value_type=float
                        ),
                        "release_seconds": ParameterValue(
                            release_seconds, value_type=float
                        ),
                        "move_stiffness": ParameterValue(
                            move_stiffness, value_type=float
                        ),
                        "move_damping": ParameterValue(
                            move_damping, value_type=float
                        ),
                        "hold_stiffness": ParameterValue(
                            hold_stiffness, value_type=float
                        ),
                        "hold_damping": ParameterValue(
                            hold_damping, value_type=float
                        ),
                        "require_no_other_arm_publishers": ParameterValue(
                            require_no_other_arm_publishers, value_type=bool
                        ),
                        "require_balanced_mode": ParameterValue(
                            require_balanced_mode, value_type=bool
                        ),
                        "require_balanced_status": ParameterValue(
                            require_balanced_status, value_type=bool
                        ),
                        "control_hz": ParameterValue(control_hz, value_type=float),
                        "arm_state_timeout_s": ParameterValue(
                            arm_state_timeout_s, value_type=float
                        ),
                    },
                ],
            ),
        ]
    )
