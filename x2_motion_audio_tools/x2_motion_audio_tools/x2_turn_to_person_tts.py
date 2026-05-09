#!/usr/bin/env python3
"""Rotate the X2 torso toward a person and say "On my way" with TTS.

Input convention:
  - angle is degrees from straight ahead, positive to the robot's left
  - distance is accepted for future face-recognition integration, but torso
    rotation only needs the angle

This script uses the HAL waist joint interface, not locomotion velocity, so it
turns the torso/waist instead of stepping the legs.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray
from aimdk_msgs.msg import MessageHeader
from aimdk_msgs.srv import PlayTts


SOURCE_NAME = "turn_to_person_tts"

SUBSCRIBER_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

PUBLISHER_QOS = QoSProfile(
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


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def smoothstep(alpha: float) -> float:
    alpha = clamp(alpha, 0.0, 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * alpha)


def prompt_float(label: str, default: Optional[float] = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw_value = input(f"{label}{suffix}: ").strip()
        if not raw_value and default is not None:
            return default

        try:
            return float(raw_value)
        except ValueError:
            print("Please enter a number.")


class TurnToPersonTts(Node):
    def __init__(
        self,
        waist_state_topic: str,
        waist_command_topic: str,
        waist_soft_limit_deg: float,
        control_hz: float,
    ) -> None:
        super().__init__("x2_turn_to_person_tts")
        self.control_hz = control_hz
        self.period = 1.0 / control_hz
        self.waist_positions: Optional[list[float]] = None
        self.waist_velocities: Optional[list[float]] = None

        yaw_joint = WAIST_JOINTS[WAIST_YAW_INDEX]
        soft_limit_rad = math.radians(waist_soft_limit_deg)
        if soft_limit_rad > 0.0:
            self.waist_yaw_lower_limit = max(yaw_joint.lower_limit, -soft_limit_rad)
            self.waist_yaw_upper_limit = min(yaw_joint.upper_limit, soft_limit_rad)
        else:
            self.waist_yaw_lower_limit = yaw_joint.lower_limit
            self.waist_yaw_upper_limit = yaw_joint.upper_limit

        self.waist_pub = self.create_publisher(
            JointCommandArray,
            waist_command_topic,
            PUBLISHER_QOS,
        )
        self.create_subscription(
            JointStateArray,
            waist_state_topic,
            self.waist_state_callback,
            SUBSCRIBER_QOS,
        )
        self.tts_client = self.create_client(
            PlayTts,
            "/aimdk_5Fmsgs/srv/PlayTts",
        )

        self.get_logger().info(
            f"Using waist state={waist_state_topic}, command={waist_command_topic}"
        )

    def waist_state_callback(self, msg: JointStateArray) -> None:
        by_name = {joint.name: joint for joint in msg.joints}
        if all(joint.name in by_name for joint in WAIST_JOINTS):
            self.waist_positions = [
                by_name[joint.name].position for joint in WAIST_JOINTS
            ]
            self.waist_velocities = [
                by_name[joint.name].velocity for joint in WAIST_JOINTS
            ]
            return

        if len(msg.joints) >= len(WAIST_JOINTS):
            self.waist_positions = [
                msg.joints[i].position for i in range(len(WAIST_JOINTS))
            ]
            self.waist_velocities = [
                msg.joints[i].velocity for i in range(len(WAIST_JOINTS))
            ]

    def wait_for_waist_state(self, timeout_sec: float) -> bool:
        self.get_logger().info("Waiting for waist joint state...")
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.waist_positions is not None:
                self.get_logger().info("Waist joint state received.")
                return True
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().error(
            "Timed out waiting for waist joint state. Check "
            "/aima/hal/joint/waist/state or pass --waist-state-topic."
        )
        return False

    def say(self, text: str) -> bool:
        self.get_logger().info("Waiting for TTS service...")
        while not self.tts_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("TTS service unavailable, waiting...")

        request = PlayTts.Request()
        request.tts_req.text = text
        request.tts_req.domain = SOURCE_NAME
        request.tts_req.trace_id = "turn_to_person"
        request.tts_req.is_interrupted = True
        request.tts_req.priority_weight = 0
        request.tts_req.priority_level.value = 6

        self.get_logger().info(f"Sending TTS: {text}")
        future = None
        for attempt in range(8):
            try:
                request.header.header.stamp = self.get_clock().now().to_msg()
            except Exception:
                pass

            future = self.tts_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.25)
            if future.done():
                break
            self.get_logger().info(f"Retrying TTS request... [{attempt + 1}/8]")

        if future is None or not future.done():
            self.get_logger().error("TTS service call timed out.")
            return False

        response = future.result()
        if response is None:
            self.get_logger().error("TTS service returned no response.")
            return False

        if response.tts_resp.is_success:
            self.get_logger().info("TTS sent successfully.")
            return True

        self.get_logger().error("TTS request failed.")
        return False

    def make_waist_command(
        self,
        positions: list[float],
        velocities: Optional[list[float]] = None,
    ) -> JointCommandArray:
        if velocities is None:
            velocities = [0.0] * len(WAIST_JOINTS)

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

    def clamp_waist_positions(self, positions: list[float]) -> list[float]:
        clamped = []
        for i, joint in enumerate(WAIST_JOINTS):
            lower = joint.lower_limit
            upper = joint.upper_limit
            if i == WAIST_YAW_INDEX:
                lower = self.waist_yaw_lower_limit
                upper = self.waist_yaw_upper_limit
            clamped.append(clamp(positions[i], lower, upper))
        return clamped

    def publish_waist(self, positions: list[float], velocities: Optional[list[float]] = None) -> None:
        self.waist_pub.publish(self.make_waist_command(positions, velocities))

    def hold_current_waist(self, seconds: float = 0.5) -> None:
        if self.waist_positions is None:
            return

        end_time = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end_time:
            self.publish_waist(list(self.waist_positions))
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(self.period)

    def rotate_torso_by(
        self,
        angle_deg: float,
        waist_speed: float,
        hold_seconds: float,
        invert_waist_direction: bool,
    ) -> None:
        if self.waist_positions is None:
            raise RuntimeError("Waist state is unavailable.")

        requested_delta = normalize_angle(math.radians(angle_deg))
        if invert_waist_direction:
            requested_delta = -requested_delta

        if abs(requested_delta) < math.radians(1.0):
            self.get_logger().info("Angle is already close; torso turn skipped.")
            self.hold_current_waist(hold_seconds)
            return

        start_positions = list(self.waist_positions)
        target_positions = list(start_positions)
        unclipped_target_yaw = start_positions[WAIST_YAW_INDEX] + requested_delta
        target_positions[WAIST_YAW_INDEX] = clamp(
            unclipped_target_yaw,
            self.waist_yaw_lower_limit,
            self.waist_yaw_upper_limit,
        )
        actual_delta = target_positions[WAIST_YAW_INDEX] - start_positions[WAIST_YAW_INDEX]

        if abs(actual_delta) < math.radians(0.5):
            self.get_logger().warn(
                "Requested torso turn is outside the configured waist limit; "
                "holding current waist position."
            )
            self.hold_current_waist(hold_seconds)
            return

        if abs(actual_delta - requested_delta) > math.radians(0.5):
            self.get_logger().warn(
                f"Torso turn clipped from {math.degrees(requested_delta):+.1f} deg "
                f"to {math.degrees(actual_delta):+.1f} deg by waist limits."
            )

        duration = max(0.4, abs(actual_delta) / waist_speed)
        self.get_logger().info(
            f"Rotating torso/waist {math.degrees(actual_delta):+.1f} deg "
            f"over {duration:.2f} s. Legs are not commanded."
        )

        previous_positions = list(start_positions)
        start_time = time.monotonic()
        while rclpy.ok():
            elapsed = time.monotonic() - start_time
            alpha = smoothstep(elapsed / duration)
            positions = [
                start + (target - start) * alpha
                for start, target in zip(start_positions, target_positions)
            ]
            velocities = [
                (positions[i] - previous_positions[i]) / self.period
                for i in range(len(WAIST_JOINTS))
            ]
            self.publish_waist(positions, velocities)
            previous_positions = positions
            rclpy.spin_once(self, timeout_sec=0.0)

            if elapsed >= duration:
                break
            time.sleep(self.period)

        hold_until = time.monotonic() + hold_seconds
        while rclpy.ok() and time.monotonic() < hold_until:
            self.publish_waist(target_positions)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(self.period)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Rotate the torso toward a person and say "On my way".'
    )
    parser.add_argument(
        "positional_distance_m",
        nargs="?",
        type=float,
        help="Distance to person in meters. Stored/logged for future integration.",
    )
    parser.add_argument(
        "positional_angle_deg",
        nargs="?",
        type=float,
        help="Bearing to person in degrees, positive left.",
    )
    parser.add_argument("--distance-m", type=float, default=None)
    parser.add_argument("--angle-deg", type=float, default=None)
    parser.add_argument("--text", default="On my way")
    parser.add_argument(
        "--waist-speed",
        "--turn-speed",
        dest="waist_speed",
        type=float,
        default=0.5,
        help="Maximum waist yaw speed in rad/s. --turn-speed is kept as an alias.",
    )
    parser.add_argument(
        "--control-hz",
        "--hz",
        dest="control_hz",
        type=float,
        default=50.0,
    )
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--waist-soft-limit-deg", type=float, default=90.0)
    parser.add_argument(
        "--waist-state-topic",
        default="/aima/hal/joint/waist/state",
    )
    parser.add_argument(
        "--waist-command-topic",
        default="/aima/hal/joint/waist/command",
    )
    parser.add_argument(
        "--invert-waist-direction",
        "--invert-turn-direction",
        dest="invert_waist_direction",
        action="store_true",
        help="Use this if positive angle turns the torso the wrong way.",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Rotate only; do not call the TTS service.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned torso turn without connecting to ROS.",
    )
    return parser.parse_args(argv)


def resolve_distance_and_angle(args: argparse.Namespace) -> tuple[float, float]:
    distance = args.distance_m
    if distance is None:
        distance = args.positional_distance_m

    angle = args.angle_deg
    if angle is None:
        angle = args.positional_angle_deg

    if angle is None:
        if distance is None:
            distance = prompt_float("Distance to person in meters", default=0.0)
        angle = prompt_float("Angle to person in degrees, positive left")
    elif distance is None:
        distance = 0.0

    if distance < 0.0:
        raise ValueError("Distance must be >= 0.")
    return distance, angle


def validate_args(args: argparse.Namespace) -> None:
    if args.waist_speed <= 0.0:
        raise ValueError("--waist-speed must be > 0.")
    if args.control_hz <= 0.0:
        raise ValueError("--control-hz must be > 0.")
    if args.hold_seconds < 0.0:
        raise ValueError("--hold-seconds must be >= 0.")


def main(args=None) -> int:
    parsed = parse_args(sys.argv[1:] if args is None else args)
    validate_args(parsed)

    distance_m, angle_deg = resolve_distance_and_angle(parsed)
    normalized_angle_deg = math.degrees(normalize_angle(math.radians(angle_deg)))

    if parsed.dry_run:
        direction = -1.0 if parsed.invert_waist_direction else 1.0
        torso_delta_deg = normalized_angle_deg * direction
        print(
            f"Plan: distance={distance_m:.2f}m, "
            f"torso_yaw_delta={torso_delta_deg:+.1f}deg, "
            f"waist_speed={parsed.waist_speed:.2f}rad/s, "
            f"text={parsed.text!r}. No leg locomotion will be commanded."
        )
        return 0

    rclpy.init()
    node = TurnToPersonTts(
        waist_state_topic=parsed.waist_state_topic,
        waist_command_topic=parsed.waist_command_topic,
        waist_soft_limit_deg=parsed.waist_soft_limit_deg,
        control_hz=parsed.control_hz,
    )

    def signal_handler(sig, _frame):
        node.get_logger().info(f"Received signal {sig}; holding current waist.")
        node.hold_current_waist(seconds=0.5)
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        node.get_logger().info(
            f"Person input: distance={distance_m:.2f}m, "
            f"angle={normalized_angle_deg:+.1f}deg"
        )
        if not parsed.skip_tts:
            node.say(parsed.text)

        if not node.wait_for_waist_state(timeout_sec=5.0):
            return 2

        node.rotate_torso_by(
            angle_deg=normalized_angle_deg,
            waist_speed=parsed.waist_speed,
            hold_seconds=parsed.hold_seconds,
            invert_waist_direction=parsed.invert_waist_direction,
        )
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        rclpy.logging.get_logger("main").fatal(
            f"Program exited with exception: {exc}"
        )
        return 1
    finally:
        try:
            node.hold_current_waist(seconds=0.5)
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
