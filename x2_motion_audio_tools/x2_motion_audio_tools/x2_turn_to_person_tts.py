#!/usr/bin/env python3
"""Say "On my way", then rotate waist_yaw_joint using the AimDK example pattern.

This intentionally mirrors the official robot joint control example as closely
as possible, but creates only the waist controller and sets waist_yaw_joint from
the supplied angle.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Dict, List, Optional

import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray
from aimdk_msgs.srv import PlayTts

try:
    import ruckig
except ImportError:
    ruckig = None


SOURCE_NAME = "turn_to_person_tts"

subscriber_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

publisher_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


class JointArea(Enum):
    WAIST = "WAIST"


@dataclass
class JointInfo:
    name: str
    lower_limit: float
    upper_limit: float
    kp: float
    kd: float


robot_model: Dict[JointArea, List[JointInfo]] = {
    JointArea.WAIST: [
        JointInfo("waist_yaw_joint", -3.43, 2.382, 20.0, 4.0),
        JointInfo("waist_pitch_joint", -0.314, 0.314, 20.0, 4.0),
        JointInfo("waist_roll_joint", -0.488, 0.488, 20.0, 4.0),
    ],
}


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


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


class PlayTtsClient(Node):
    def __init__(self) -> None:
        super().__init__("x2_turn_to_person_tts_speaker")
        self.client = self.create_client(PlayTts, "/aimdk_5Fmsgs/srv/PlayTts")

    def say(self, text: str) -> bool:
        while not self.client.wait_for_service(timeout_sec=2.0):
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

            future = self.client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.25)
            if future.done():
                break
            self.get_logger().info(f"Retrying TTS request... [{attempt}]")

        response = future.result() if future is not None and future.done() else None
        if response is None:
            self.get_logger().error("TTS service call failed or timed out.")
            return False

        if response.tts_resp.is_success:
            self.get_logger().info("TTS sent successfully.")
            return True

        self.get_logger().error("TTS request failed.")
        return False


class JointControllerNode(Node):
    """Waist controller copied from the AimDK joint-control example."""

    def __init__(
        self,
        node_name: str,
        sub_topic: str,
        pub_topic: str,
        area: JointArea,
        dofs: int,
        control_period_sec: float,
    ) -> None:
        super().__init__(node_name)
        if ruckig is None:
            raise ImportError(
                "ruckig is required. Install it on the robot with: "
                "python3 -m pip install ruckig"
            )

        self.lock = Lock()
        self.joint_info = robot_model[area]
        self.dofs = dofs
        self.ruckig = ruckig.Ruckig(dofs, control_period_sec)
        self.input = ruckig.InputParameter(dofs)
        self.output = ruckig.OutputParameter(dofs)
        self.ruckig_initialized = False

        self.input.current_position = [0.0] * dofs
        self.input.current_velocity = [0.0] * dofs
        self.input.current_acceleration = [0.0] * dofs
        self.input.max_velocity = [1.0] * dofs
        self.input.max_acceleration = [1.0] * dofs
        self.input.max_jerk = [25.0] * dofs

        self.sub = self.create_subscription(
            JointStateArray,
            sub_topic,
            self.joint_state_callback,
            subscriber_qos,
        )
        self.pub = self.create_publisher(
            JointCommandArray,
            pub_topic,
            publisher_qos,
        )

    def joint_state_callback(self, _msg: JointStateArray) -> None:
        self.ruckig_initialized = True

    def control_callback(self, joint_idx: int) -> None:
        while self.ruckig.update(self.input, self.output) in [
            ruckig.Result.Working,
            ruckig.Result.Finished,
        ]:
            self.input.current_position = self.output.new_position
            self.input.current_velocity = self.output.new_velocity
            self.input.current_acceleration = self.output.new_acceleration

            tolerance = 1e-6
            current_p = self.output.new_position[joint_idx]
            if abs(current_p - self.input.target_position[joint_idx]) < tolerance:
                break

            cmd = JointCommandArray()
            for i, joint_info in enumerate(self.joint_info):
                joint = JointCommand()
                joint.name = joint_info.name
                joint.position = clamp(
                    self.output.new_position[i],
                    joint_info.lower_limit,
                    joint_info.upper_limit,
                )
                joint.velocity = self.output.new_velocity[i]
                joint.effort = 0.0
                joint.stiffness = joint_info.kp
                joint.damping = joint_info.kd
                cmd.joints.append(joint)

            self.pub.publish(cmd)

    def set_target_position(self, joint_name: str, position: float) -> None:
        target_positions = [0.0] * self.dofs
        joint_idx = 0
        for i, joint in enumerate(self.joint_info):
            if joint.name == joint_name:
                target_positions[i] = clamp(position, joint.lower_limit, joint.upper_limit)
                joint_idx = i

        self.input.target_position = target_positions
        self.input.target_velocity = [0.0] * self.dofs
        self.input.target_acceleration = [0.0] * self.dofs
        self.control_callback(joint_idx)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Say "On my way", then set waist_yaw_joint to the input angle.'
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
        help="Waist yaw target in degrees, positive left.",
    )
    parser.add_argument("--distance-m", type=float, default=None)
    parser.add_argument("--angle-deg", type=float, default=None)
    parser.add_argument("--text", default="On my way")
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
    )
    parser.add_argument("--control-period-sec", type=float, default=0.002)
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=0.5,
        help="Short delay after creating the waist publisher/subscriber.",
    )
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
        angle = prompt_float("Waist yaw target in degrees, positive left")
    elif distance is None:
        distance = 0.0

    if distance < 0.0:
        raise ValueError("Distance must be >= 0.")
    return distance, angle


def validate_args(args: argparse.Namespace) -> None:
    if args.control_period_sec <= 0.0:
        raise ValueError("--control-period-sec must be > 0.")
    if args.settle_sec < 0.0:
        raise ValueError("--settle-sec must be >= 0.")


def main(args=None) -> int:
    parsed = parse_args(sys.argv[1:] if args is None else args)
    validate_args(parsed)

    distance_m, angle_deg = resolve_distance_and_angle(parsed)
    yaw_target_rad = normalize_angle(math.radians(angle_deg))
    if parsed.invert_waist_direction:
        yaw_target_rad = -yaw_target_rad

    if parsed.dry_run:
        print(
            f"Plan: distance={distance_m:.2f}m, "
            f"waist_yaw_joint={math.degrees(yaw_target_rad):+.1f}deg, "
            f"text={parsed.text!r}"
        )
        return 0

    rclpy.init()
    speaker = PlayTtsClient()
    waist_node = JointControllerNode(
        "waist_node",
        parsed.waist_state_topic,
        parsed.waist_command_topic,
        JointArea.WAIST,
        3,
        parsed.control_period_sec,
    )

    def signal_handler(sig, _frame):
        waist_node.get_logger().info(f"Received signal {sig}; stopping.")
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        speaker.get_logger().info(
            f"Person input: distance={distance_m:.2f}m, angle={angle_deg:+.1f}deg"
        )
        if not parsed.skip_tts:
            speaker.say(parsed.text)

        end_time = time.monotonic() + parsed.settle_sec
        while rclpy.ok() and time.monotonic() < end_time:
            rclpy.spin_once(waist_node, timeout_sec=0.02)

        waist_node.set_target_position("waist_yaw_joint", yaw_target_rad)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        rclpy.logging.get_logger("main").fatal(
            f"Program exited with exception: {exc}"
        )
        return 1
    finally:
        speaker.destroy_node()
        waist_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
