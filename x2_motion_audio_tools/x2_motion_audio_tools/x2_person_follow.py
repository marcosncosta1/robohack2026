#!/usr/bin/env python3
"""Detect a person with the head camera, read lidar distance, and track them.

The node logs every detection in the console. By default it uses the X2 HAL
waist interface to turn the torso toward the selected person while they remain
visible. When follow_enabled is true and aimdk_msgs is available, it also
publishes X2 locomotion velocity commands that turn toward the person and walk
forward until stop_distance_m.
"""

from __future__ import annotations

import math
import signal
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

try:
    import ruckig
except ImportError:
    ruckig = None

try:
    from aimdk_msgs.msg import (
        JointCommand,
        JointCommandArray,
        JointStateArray,
        McLocomotionVelocity,
        MessageHeader,
    )
    from aimdk_msgs.srv import SetMcInputSource

    AIMDK_AVAILABLE = True
except ImportError:
    AIMDK_AVAILABLE = False

from yolo_person_detector.yolo_wrapper import Detection, InferenceResult, YOLOWrapper


SOURCE_NAME = "person_follower"
DEFAULT_MODEL_PATH = "yolov8n.pt"

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


@dataclass(frozen=True)
class JointInfo:
    name: str
    lower_limit: float
    upper_limit: float
    kp: float
    kd: float


WAIST_JOINTS = [
    JointInfo("waist_yaw_joint", -3.43, 2.382, 20.0, 4.0),
    JointInfo("waist_pitch_joint", -0.314, 0.314, 20.0, 4.0),
    JointInfo("waist_roll_joint", -0.488, 0.488, 20.0, 4.0),
]
WAIST_YAW_INDEX = 0


@dataclass
class PersonTarget:
    confidence: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    image_width: int
    image_height: int
    bearing_rad: float
    base_bearing_rad: float
    distance_m: Optional[float]
    distance_source: str
    inference_time_ms: float
    stamp_monotonic: float


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def move_toward(value: float, target: float, max_delta: float) -> float:
    if target > value + max_delta:
        return value + max_delta
    if target < value - max_delta:
        return value - max_delta
    return target


def angular_delta(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def bool_param(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class X2PersonFollow(Node):
    """Camera plus lidar person detector/follower for the Agibot X2."""

    def __init__(self) -> None:
        super().__init__("x2_person_follow")

        self.declare_parameter(
            "camera_topic", "/aima/hal/sensor/rgb_head_front_center/rgb_image"
        )
        self.declare_parameter("lidar_topic", "/scan")
        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("nms_threshold", 0.45)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("input_size", 640)
        self.declare_parameter("camera_horizontal_fov_deg", 69.0)
        self.declare_parameter("lidar_window_deg", 8.0)
        self.declare_parameter("lidar_angle_offset_deg", 0.0)
        self.declare_parameter("follow_enabled", False)
        self.declare_parameter("stop_distance_m", 1.2)
        self.declare_parameter("forward_gain", 0.25)
        self.declare_parameter("angular_gain", 1.2)
        self.declare_parameter("max_forward_speed", 0.30)
        self.declare_parameter("min_forward_speed", 0.20)
        self.declare_parameter("max_angular_speed", 0.50)
        self.declare_parameter("min_angular_speed", 0.06)
        self.declare_parameter("center_deadzone_deg", 3.0)
        self.declare_parameter("watchdog_timeout_sec", 0.7)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("log_every_sec", 1.0)
        self.declare_parameter("track_same_person", True)
        self.declare_parameter("target_max_center_jump_ratio", 0.45)
        self.declare_parameter("waist_tracking_enabled", True)
        self.declare_parameter("waist_state_topic", "/aima/hal/joint/waist/state")
        self.declare_parameter("waist_command_topic", "/aima/hal/joint/waist/command")
        self.declare_parameter("waist_yaw_gain", 1.0)
        self.declare_parameter("waist_center_deadzone_deg", 2.0)
        self.declare_parameter("waist_soft_limit_deg", 90.0)
        self.declare_parameter("waist_max_velocity", 0.7)
        self.declare_parameter("waist_max_acceleration", 1.0)
        self.declare_parameter("waist_max_jerk", 25.0)
        self.declare_parameter("waist_invert_direction", False)
        self.declare_parameter("waist_hold_on_lost", True)
        self.declare_parameter("waist_use_ruckig", True)

        self.camera_topic = self.get_parameter("camera_topic").value
        self.lidar_topic = self.get_parameter("lidar_topic").value
        model_path = self.get_parameter("model_path").value
        confidence = float(self.get_parameter("confidence_threshold").value)
        nms_threshold = float(self.get_parameter("nms_threshold").value)
        device = self.get_parameter("device").value
        input_size = int(self.get_parameter("input_size").value)
        self.camera_horizontal_fov_rad = math.radians(
            float(self.get_parameter("camera_horizontal_fov_deg").value)
        )
        self.lidar_window_rad = math.radians(
            float(self.get_parameter("lidar_window_deg").value)
        )
        self.lidar_angle_offset_rad = math.radians(
            float(self.get_parameter("lidar_angle_offset_deg").value)
        )
        self.follow_enabled = bool_param(self.get_parameter("follow_enabled").value)
        self.stop_distance_m = float(self.get_parameter("stop_distance_m").value)
        self.forward_gain = float(self.get_parameter("forward_gain").value)
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_forward_speed = float(self.get_parameter("max_forward_speed").value)
        self.min_forward_speed = float(self.get_parameter("min_forward_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.min_angular_speed = float(self.get_parameter("min_angular_speed").value)
        self.center_deadzone_rad = math.radians(
            float(self.get_parameter("center_deadzone_deg").value)
        )
        self.watchdog_timeout_sec = float(
            self.get_parameter("watchdog_timeout_sec").value
        )
        self.log_every_sec = float(self.get_parameter("log_every_sec").value)
        self.track_same_person = bool_param(
            self.get_parameter("track_same_person").value
        )
        self.target_max_center_jump_ratio = float(
            self.get_parameter("target_max_center_jump_ratio").value
        )
        self.waist_tracking_enabled = bool_param(
            self.get_parameter("waist_tracking_enabled").value
        )
        self.waist_state_topic = self.get_parameter("waist_state_topic").value
        self.waist_command_topic = self.get_parameter("waist_command_topic").value
        self.waist_yaw_gain = float(self.get_parameter("waist_yaw_gain").value)
        self.waist_center_deadzone_rad = math.radians(
            float(self.get_parameter("waist_center_deadzone_deg").value)
        )
        self.waist_max_velocity = float(
            self.get_parameter("waist_max_velocity").value
        )
        self.waist_max_acceleration = float(
            self.get_parameter("waist_max_acceleration").value
        )
        self.waist_max_jerk = float(self.get_parameter("waist_max_jerk").value)
        self.waist_invert_direction = bool_param(
            self.get_parameter("waist_invert_direction").value
        )
        self.waist_hold_on_lost = bool_param(
            self.get_parameter("waist_hold_on_lost").value
        )
        self.waist_use_ruckig = bool_param(
            self.get_parameter("waist_use_ruckig").value
        )
        waist_soft_limit_rad = math.radians(
            float(self.get_parameter("waist_soft_limit_deg").value)
        )
        yaw_joint = WAIST_JOINTS[WAIST_YAW_INDEX]
        if waist_soft_limit_rad > 0.0:
            self.waist_yaw_lower_limit = max(
                yaw_joint.lower_limit, -waist_soft_limit_rad
            )
            self.waist_yaw_upper_limit = min(
                yaw_joint.upper_limit, waist_soft_limit_rad
            )
        else:
            self.waist_yaw_lower_limit = yaw_joint.lower_limit
            self.waist_yaw_upper_limit = yaw_joint.upper_limit
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.control_period_sec = 1.0 / control_rate_hz

        if CvBridge is None:
            raise RuntimeError(
                "cv_bridge is not available. Install ros-humble-cv-bridge on the robot."
            )
        self.bridge = CvBridge()

        self.get_logger().info(f"Loading YOLO model: {model_path} on {device}")
        self.yolo = YOLOWrapper(
            model_path=model_path,
            confidence_threshold=confidence,
            nms_threshold=nms_threshold,
            device=device,
            input_size=input_size,
        )
        self.get_logger().info("YOLO model loaded")

        self.latest_scan: Optional[LaserScan] = None
        self.target: Optional[PersonTarget] = None
        self.last_log_time = 0.0
        self.last_no_lidar_warning = 0.0
        self.last_stop_publish = 0.0
        self.last_no_waist_state_warning = 0.0
        self.last_waist_ruckig_warning = 0.0
        self.input_source_registered = False
        self.waist_positions: Optional[list[float]] = None
        self.waist_velocities: Optional[list[float]] = None
        self.waist_command_positions: Optional[list[float]] = None
        self.waist_command_velocities: Optional[list[float]] = None
        self.waist_ruckig = None

        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            SENSOR_QOS,
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.lidar_topic,
            self.scan_callback,
            SENSOR_QOS,
        )

        self.vel_pub = None
        self.waist_pub = None
        self.input_source_client = None
        if AIMDK_AVAILABLE:
            self.vel_pub = self.create_publisher(
                McLocomotionVelocity,
                "/aima/mc/locomotion/velocity",
                RELIABLE_QOS,
            )
            self.input_source_client = self.create_client(
                SetMcInputSource,
                "/aimdk_5Fmsgs/srv/SetMcInputSource",
            )
            if self.waist_tracking_enabled:
                self.waist_pub = self.create_publisher(
                    JointCommandArray,
                    self.waist_command_topic,
                    RELIABLE_QOS,
                )
                self.waist_state_sub = self.create_subscription(
                    JointStateArray,
                    self.waist_state_topic,
                    self.waist_state_callback,
                    SENSOR_QOS,
                )
                if self.waist_use_ruckig and ruckig is not None:
                    self.waist_ruckig = ruckig.Ruckig(
                        len(WAIST_JOINTS), self.control_period_sec
                    )
                elif self.waist_use_ruckig:
                    self.get_logger().warn(
                        "ruckig is not installed; waist tracking will use "
                        "velocity-limited interpolation."
                    )
        elif self.follow_enabled:
            self.get_logger().error(
                "follow_enabled=true, but aimdk_msgs is not available. "
                "Detection logging will run, movement is disabled."
            )
        elif self.waist_tracking_enabled:
            self.get_logger().error(
                "waist_tracking_enabled=true, but aimdk_msgs is not available. "
                "Detection logging will run, torso tracking is disabled."
            )

        if self.follow_enabled and AIMDK_AVAILABLE:
            self.input_source_registered = self.register_input_source()

        self.control_timer = self.create_timer(
            1.0 / control_rate_hz, self.control_loop
        )

        mode = "follow" if self.follow_enabled else "log-only"
        waist_mode = "waist-track" if self.waist_tracking_enabled else "waist-off"
        self.get_logger().info(
            f"Started in {mode}/{waist_mode} mode. camera_topic={self.camera_topic}, "
            f"lidar_topic={self.lidar_topic}, stop_distance={self.stop_distance_m:.2f}m"
        )

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def waist_state_callback(self, msg: JointStateArray) -> None:
        by_name = {
            joint.name: joint
            for joint in msg.joints
            if getattr(joint, "name", "")
        }

        if all(joint.name in by_name for joint in WAIST_JOINTS):
            self.waist_positions = [
                float(by_name[joint.name].position) for joint in WAIST_JOINTS
            ]
            self.waist_velocities = [
                float(getattr(by_name[joint.name], "velocity", 0.0))
                for joint in WAIST_JOINTS
            ]
        elif len(msg.joints) >= len(WAIST_JOINTS):
            self.waist_positions = [
                float(msg.joints[i].position) for i in range(len(WAIST_JOINTS))
            ]
            self.waist_velocities = [
                float(getattr(msg.joints[i], "velocity", 0.0))
                for i in range(len(WAIST_JOINTS))
            ]
        else:
            return

        if self.waist_command_positions is None:
            self.waist_command_positions = list(self.waist_positions)
            self.waist_command_velocities = list(self.waist_velocities)

    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Image conversion failed: {exc}")
            return

        try:
            result = self.yolo.detect(image)
        except Exception as exc:
            self.get_logger().error(f"YOLO inference failed: {exc}")
            return

        self.target = self.select_target(result)
        self.log_detection(result, self.target)

    def select_target(self, result: InferenceResult) -> Optional[PersonTarget]:
        if not result.detections:
            return None

        detection = self.select_detection(result)
        bearing_rad = self.bearing_from_detection(detection, result.image_width)
        base_bearing_rad = self.base_bearing_from_camera_bearing(bearing_rad)
        distance_m, distance_source = self.distance_from_lidar(base_bearing_rad)

        return PersonTarget(
            confidence=detection.confidence,
            bbox_x=detection.bbox_x,
            bbox_y=detection.bbox_y,
            bbox_w=detection.bbox_w,
            bbox_h=detection.bbox_h,
            image_width=result.image_width,
            image_height=result.image_height,
            bearing_rad=bearing_rad,
            base_bearing_rad=base_bearing_rad,
            distance_m=distance_m,
            distance_source=distance_source,
            inference_time_ms=result.inference_time_ms,
            stamp_monotonic=time.monotonic(),
        )

    def select_detection(self, result: InferenceResult) -> Detection:
        previous = self.target
        now = time.monotonic()
        if (
            self.track_same_person
            and previous is not None
            and now - previous.stamp_monotonic <= self.watchdog_timeout_sec
        ):
            previous_center_x = previous.bbox_x + previous.bbox_w / 2.0
            previous_center_y = previous.bbox_y + previous.bbox_h / 2.0
            max_jump_px = (
                max(result.image_width, result.image_height)
                * self.target_max_center_jump_ratio
            )
            nearby = []
            for detection in result.detections:
                center_x = detection.bbox_x + detection.bbox_w / 2.0
                center_y = detection.bbox_y + detection.bbox_h / 2.0
                jump_px = math.hypot(
                    center_x - previous_center_x,
                    center_y - previous_center_y,
                )
                if jump_px <= max_jump_px:
                    nearby.append((jump_px, detection))
            if nearby:
                return min(nearby, key=lambda item: item[0])[1]

        return max(result.detections, key=lambda det: det.bbox_w * det.bbox_h)

    def bearing_from_detection(self, detection: Detection, image_width: int) -> float:
        bbox_center_x = detection.bbox_x + detection.bbox_w / 2.0
        normalized_left_positive = (image_width / 2.0 - bbox_center_x) / (
            image_width / 2.0
        )
        return normalized_left_positive * (self.camera_horizontal_fov_rad / 2.0)

    def base_bearing_from_camera_bearing(self, bearing_rad: float) -> float:
        if not self.waist_tracking_enabled or self.waist_positions is None:
            return bearing_rad

        waist_yaw = self.waist_positions[WAIST_YAW_INDEX]
        camera_yaw_rad = -waist_yaw if self.waist_invert_direction else waist_yaw
        return angular_delta(camera_yaw_rad + bearing_rad, 0.0)

    def distance_from_lidar(self, bearing_rad: float) -> tuple[Optional[float], str]:
        scan = self.latest_scan
        if scan is None:
            now = time.monotonic()
            if now - self.last_no_lidar_warning > 5.0:
                self.last_no_lidar_warning = now
                self.get_logger().warn(
                    f"No LaserScan messages yet on {self.lidar_topic}"
                )
            return None, "lidar-missing"

        target_angle = bearing_rad + self.lidar_angle_offset_rad
        valid_ranges: list[float] = []
        range_min = max(float(scan.range_min), 0.02)
        range_max = float(scan.range_max) if scan.range_max > 0.0 else float("inf")

        for index, range_m in enumerate(scan.ranges):
            if not math.isfinite(range_m):
                continue
            if range_m < range_min or range_m > range_max:
                continue

            sample_angle = scan.angle_min + index * scan.angle_increment
            if abs(angular_delta(sample_angle, target_angle)) <= self.lidar_window_rad:
                valid_ranges.append(float(range_m))

        if not valid_ranges:
            return None, "lidar-empty"

        valid_ranges.sort()
        closest_sample_count = min(5, len(valid_ranges))
        return statistics.median(valid_ranges[:closest_sample_count]), "lidar"

    def log_detection(
        self, result: InferenceResult, target: Optional[PersonTarget]
    ) -> None:
        now = time.monotonic()
        if now - self.last_log_time < self.log_every_sec:
            return
        self.last_log_time = now

        if target is None:
            self.get_logger().info(
                f"No person detected | inference={result.inference_time_ms:.1f}ms"
            )
            return

        if target.distance_m is None:
            distance_text = f"distance=unavailable ({target.distance_source})"
        else:
            distance_text = f"distance={target.distance_m:.2f}m ({target.distance_source})"

        self.get_logger().info(
            "Person detected: "
            f"{distance_text}, confidence={target.confidence:.2f}, "
            f"bearing={math.degrees(target.bearing_rad):+.1f}deg, "
            f"base_bearing={math.degrees(target.base_bearing_rad):+.1f}deg, "
            f"bbox={target.bbox_w}x{target.bbox_h}, "
            f"inference={target.inference_time_ms:.1f}ms"
        )

    def control_loop(self) -> None:
        target = self.fresh_target()
        self.control_waist(target)

        if not self.follow_enabled or not AIMDK_AVAILABLE:
            return

        if not self.input_source_registered:
            self.publish_stop_throttled()
            return

        if target is None:
            self.publish_stop_throttled()
            return

        angular_velocity = self.angular_velocity_for_target(target)
        forward_velocity = self.forward_velocity_for_target(target)
        self.publish_velocity(forward_velocity, angular_velocity)

    def fresh_target(self) -> Optional[PersonTarget]:
        target = self.target
        if target is None:
            return None

        if time.monotonic() - target.stamp_monotonic > self.watchdog_timeout_sec:
            return None

        return target

    def control_waist(self, target: Optional[PersonTarget]) -> None:
        if (
            not self.waist_tracking_enabled
            or not AIMDK_AVAILABLE
            or self.waist_pub is None
        ):
            return

        if self.waist_positions is None:
            now = time.monotonic()
            if now - self.last_no_waist_state_warning > 3.0:
                self.last_no_waist_state_warning = now
                self.get_logger().warn(
                    f"No waist joint state messages yet on {self.waist_state_topic}"
                )
            return

        if target is None:
            if self.waist_hold_on_lost:
                self.publish_waist_hold()
            return

        bearing_rad = target.bearing_rad
        if self.waist_invert_direction:
            bearing_rad = -bearing_rad

        desired_positions = list(self.waist_positions)
        current_yaw = self.waist_positions[WAIST_YAW_INDEX]
        if abs(bearing_rad) >= self.waist_center_deadzone_rad:
            desired_positions[WAIST_YAW_INDEX] = clamp(
                current_yaw + self.waist_yaw_gain * bearing_rad,
                self.waist_yaw_lower_limit,
                self.waist_yaw_upper_limit,
            )
        else:
            desired_positions[WAIST_YAW_INDEX] = current_yaw

        self.publish_waist_target(desired_positions)

    def publish_waist_hold(self) -> None:
        if self.waist_pub is None or self.waist_positions is None:
            return

        positions = self.waist_command_positions or self.waist_positions
        self.waist_pub.publish(
            self.make_waist_command(positions, [0.0] * len(WAIST_JOINTS))
        )

    def publish_waist_target(self, target_positions: list[float]) -> None:
        if self.waist_pub is None or self.waist_positions is None:
            return

        target_positions = self.clamp_waist_positions(target_positions)
        if self.waist_ruckig is not None:
            positions, velocities = self.next_waist_ruckig_step(target_positions)
        else:
            positions, velocities = self.next_waist_slew_step(target_positions)

        self.waist_command_positions = list(positions)
        self.waist_command_velocities = list(velocities)
        self.waist_pub.publish(self.make_waist_command(positions, velocities))

    def next_waist_ruckig_step(
        self, target_positions: list[float]
    ) -> tuple[list[float], list[float]]:
        if self.waist_positions is None or self.waist_ruckig is None:
            return self.next_waist_slew_step(target_positions)

        dofs = len(WAIST_JOINTS)
        inp = ruckig.InputParameter(dofs)
        out = ruckig.OutputParameter(dofs)
        current_positions = self.waist_command_positions or self.waist_positions
        current_velocities = (
            self.waist_command_velocities
            or self.waist_velocities
            or [0.0] * dofs
        )
        inp.current_position = list(current_positions)
        inp.current_velocity = list(current_velocities)
        inp.current_acceleration = [0.0] * dofs
        inp.target_position = list(target_positions)
        inp.target_velocity = [0.0] * dofs
        inp.target_acceleration = [0.0] * dofs
        inp.max_velocity = [self.waist_max_velocity] * dofs
        inp.max_acceleration = [self.waist_max_acceleration] * dofs
        inp.max_jerk = [self.waist_max_jerk] * dofs

        result = self.waist_ruckig.update(inp, out)
        if result not in [ruckig.Result.Working, ruckig.Result.Finished]:
            now = time.monotonic()
            if now - self.last_waist_ruckig_warning > 3.0:
                self.last_waist_ruckig_warning = now
                self.get_logger().warn(
                    "Ruckig waist step failed; using velocity-limited fallback."
                )
            return self.next_waist_slew_step(target_positions)

        return list(out.new_position), list(out.new_velocity)

    def next_waist_slew_step(
        self, target_positions: list[float]
    ) -> tuple[list[float], list[float]]:
        start_positions = self.waist_command_positions or self.waist_positions
        if start_positions is None:
            return target_positions, [0.0] * len(WAIST_JOINTS)

        max_delta = self.waist_max_velocity * self.control_period_sec
        positions = [
            move_toward(start_positions[i], target_positions[i], max_delta)
            for i in range(len(WAIST_JOINTS))
        ]
        velocities = [
            (positions[i] - start_positions[i]) / self.control_period_sec
            for i in range(len(WAIST_JOINTS))
        ]
        return positions, velocities

    def clamp_waist_positions(self, positions: list[float]) -> list[float]:
        clamped = []
        for i, joint in enumerate(WAIST_JOINTS):
            lower_limit = joint.lower_limit
            upper_limit = joint.upper_limit
            if i == WAIST_YAW_INDEX:
                lower_limit = self.waist_yaw_lower_limit
                upper_limit = self.waist_yaw_upper_limit
            clamped.append(clamp(positions[i], lower_limit, upper_limit))
        return clamped

    def make_waist_command(
        self, positions: list[float], velocities: list[float]
    ) -> JointCommandArray:
        cmd = JointCommandArray()
        try:
            cmd.header = MessageHeader()
            cmd.header.stamp = self.get_clock().now().to_msg()
        except Exception:
            pass

        positions = self.clamp_waist_positions(positions)
        for i, joint_info in enumerate(WAIST_JOINTS):
            joint = JointCommand()
            joint.name = joint_info.name
            joint.position = positions[i]
            joint.velocity = velocities[i]
            joint.effort = 0.0
            joint.stiffness = joint_info.kp
            joint.damping = joint_info.kd
            cmd.joints.append(joint)

        return cmd

    def angular_velocity_for_target(self, target: PersonTarget) -> float:
        if abs(target.base_bearing_rad) < self.center_deadzone_rad:
            return 0.0

        angular_velocity = self.angular_gain * target.base_bearing_rad
        return self.deadband_clamp(
            angular_velocity, self.min_angular_speed, self.max_angular_speed
        )

    def forward_velocity_for_target(self, target: PersonTarget) -> float:
        if target.distance_m is None:
            return 0.0

        distance_error = target.distance_m - self.stop_distance_m
        if distance_error <= 0.0:
            return 0.0

        forward_velocity = self.forward_gain * distance_error
        return self.deadband_clamp(
            forward_velocity, self.min_forward_speed, self.max_forward_speed
        )

    def deadband_clamp(self, value: float, min_abs: float, max_abs: float) -> float:
        if abs(value) < min_abs:
            return 0.0
        return clamp(value, -max_abs, max_abs)

    def register_input_source(self) -> bool:
        if self.input_source_client is None:
            return False

        self.get_logger().info("Registering locomotion input source...")
        timeout_sec = 8.0
        start = self.get_clock().now().nanoseconds / 1e9

        while not self.input_source_client.wait_for_service(timeout_sec=2.0):
            now = self.get_clock().now().nanoseconds / 1e9
            if now - start > timeout_sec:
                self.get_logger().error("Locomotion input source service timed out")
                return False
            self.get_logger().info("Waiting for locomotion input source service...")

        req = SetMcInputSource.Request()
        req.action.value = 1001
        req.input_source.name = SOURCE_NAME
        req.input_source.priority = 40
        req.input_source.timeout = 1000

        future = None
        for attempt in range(8):
            req.request.header.stamp = self.get_clock().now().to_msg()
            future = self.input_source_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.25)
            if future.done():
                break
            self.get_logger().info(
                f"Trying to register input source... [{attempt + 1}/8]"
            )

        if future is None or not future.done():
            self.get_logger().error("Input source registration failed or timed out")
            return False

        try:
            resp = future.result()
            state = resp.response.state.value
            self.get_logger().info(
                f"Input source registered: state={state}, task_id={resp.response.task_id}"
            )
            return True
        except Exception as exc:
            self.get_logger().error(f"Input source registration exception: {exc}")
            return False

    def publish_velocity(
        self, forward_velocity: float, angular_velocity: float
    ) -> None:
        if self.vel_pub is None:
            return

        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = SOURCE_NAME
        msg.forward_velocity = float(forward_velocity)
        msg.lateral_velocity = 0.0
        msg.angular_velocity = float(angular_velocity)
        self.vel_pub.publish(msg)

    def publish_stop_throttled(self) -> None:
        now = time.monotonic()
        if now - self.last_stop_publish < 0.05:
            return
        self.last_stop_publish = now
        self.publish_velocity(0.0, 0.0)

    def stop(self) -> None:
        self.publish_velocity(0.0, 0.0)
        if self.waist_tracking_enabled:
            self.publish_waist_hold()


_GLOBAL_NODE: Optional[X2PersonFollow] = None


def signal_handler(sig, _frame) -> None:
    if _GLOBAL_NODE is not None:
        _GLOBAL_NODE.get_logger().info(f"Received signal {sig}; stopping robot")
        _GLOBAL_NODE.stop()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(0)


def main(args=None) -> None:
    global _GLOBAL_NODE

    rclpy.init(args=args)
    node = X2PersonFollow()
    _GLOBAL_NODE = node

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
