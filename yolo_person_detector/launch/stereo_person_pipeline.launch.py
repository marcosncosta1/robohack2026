"""Launch the simple stereo image -> YOLO bbox pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera = LaunchConfiguration("camera")
    topic_type = LaunchConfiguration("topic_type")
    device = LaunchConfiguration("device")
    model = LaunchConfiguration("model")
    confidence = LaunchConfiguration("confidence")
    enable_depth = LaunchConfiguration("enable_depth")
    baseline_m = LaunchConfiguration("baseline_m")
    annotated_compressed_topic = LaunchConfiguration("annotated_compressed_topic")
    depth_debug_compressed_topic = LaunchConfiguration("depth_debug_compressed_topic")
    jpeg_quality = LaunchConfiguration("jpeg_quality")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera",
                default_value="stereo_head_front_left",
                description="stereo_head_front_left or stereo_head_front_right",
            ),
            DeclareLaunchArgument(
                "topic_type",
                default_value="rgb_image_compressed",
                description="rgb_image or rgb_image_compressed",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cpu",
                description="YOLO inference device: cpu, cuda, or mps",
            ),
            DeclareLaunchArgument(
                "model",
                default_value="yolov8n.pt",
                description="YOLO model file path",
            ),
            DeclareLaunchArgument(
                "confidence",
                default_value="0.5",
                description="YOLO confidence threshold",
            ),
            DeclareLaunchArgument(
                "enable_depth",
                default_value="false",
                description="Start stereo bbox depth estimation",
            ),
            DeclareLaunchArgument(
                "baseline_m",
                default_value="0.0",
                description="Manual stereo baseline in meters if CameraInfo lacks Tx",
            ),
            DeclareLaunchArgument(
                "annotated_compressed_topic",
                default_value="/stereo_person/annotated_image/compressed",
                description="Compressed JPEG topic for YOLO-annotated stereo images",
            ),
            DeclareLaunchArgument(
                "depth_debug_compressed_topic",
                default_value="/stereo_person/depth_debug_image/compressed",
                description="Compressed JPEG topic for depth-annotated stereo images",
            ),
            DeclareLaunchArgument(
                "jpeg_quality",
                default_value="85",
                description="JPEG quality for annotated compressed outputs",
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
                    }
                ],
            ),
            Node(
                package="yolo_person_detector",
                executable="stereo_yolo_detect_node",
                name="stereo_yolo_detect",
                output="screen",
                parameters=[
                    {
                        "device": device,
                        "model_path": model,
                        "confidence_threshold": confidence,
                        "annotated_compressed_topic": annotated_compressed_topic,
                        "jpeg_quality": jpeg_quality,
                    }
                ],
            ),
            Node(
                package="yolo_person_detector",
                executable="stereo_bbox_depth_node",
                name="stereo_bbox_depth",
                output="screen",
                condition=IfCondition(enable_depth),
                parameters=[
                    {
                        "baseline_m": baseline_m,
                        "debug_compressed_topic": depth_debug_compressed_topic,
                        "jpeg_quality": jpeg_quality,
                    }
                ],
            ),
        ]
    )
