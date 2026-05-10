#!/usr/bin/env python3
"""Keyboard teleop for the X2 head joints."""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

try:
    from aimdk_msgs.msg import (
        JointCommand,
        JointCommandArray,
        JointStateArray,
        MessageHeader,
    )

    AIMDK_AVAILABLE = True
except ImportError:
    AIMDK_AVAILABLE = False


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
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


HEAD_JOINTS = [
    JointInfo("head_yaw_joint", -0.366, 0.366, 20.0, 2.0),
    JointInfo("head_pitch_joint", -0.3838, 0.3838, 20.0, 2.0),
]
HEAD_YAW_INDEX = 0
HEAD_PITCH_INDEX = 1


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def move_toward(value: float, target: float, max_delta: float) -> float:
    if target > value + max_delta:
        return value + max_delta
    if target < value - max_delta:
        return value - max_delta
    return target


def bool_param(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class X2HeadKeyboardTeleop(Node):
    """Move head yaw/pitch with arrow keys from a terminal."""

    def __init__(self) -> None:
        super().__init__("x2_head_keyboard_teleop")

        if not AIMDK_AVAILABLE:
            self.get_logger().fatal(
                "aimdk_msgs not available. This node must run on the Agibot X2."
            )
            raise RuntimeError("aimdk_msgs not available")

        self.declare_parameter("head_state_topic", "/aima/hal/joint/head/state")
        self.declare_parameter("head_command_topic", "/aima/hal/joint/head/command")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("yaw_step_deg", 2.0)
        self.declare_parameter("pitch_step_deg", 2.0)
        self.declare_parameter("max_velocity", 0.12)
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("soft_yaw_limit_deg", 20.0)
        self.declare_parameter("soft_pitch_limit_deg", 18.0)
        self.declare_parameter("publish_hold", True)

        self.head_state_topic = self.get_parameter("head_state_topic").value
        self.head_command_topic = self.get_parameter("head_command_topic").value
        self.dry_run = bool_param(self.get_parameter("dry_run").value)
        self.yaw_step_rad = self.deg_param("yaw_step_deg")
        self.pitch_step_rad = self.deg_param("pitch_step_deg")
        self.max_velocity = float(self.get_parameter("max_velocity").value)
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.control_period_sec = 1.0 / max(control_rate_hz, 0.1)
        self.publish_hold = bool_param(self.get_parameter("publish_hold").value)

        self.yaw_lower_limit, self.yaw_upper_limit = self.soft_limits(
            HEAD_JOINTS[HEAD_YAW_INDEX],
            "soft_yaw_limit_deg",
        )
        self.pitch_lower_limit, self.pitch_upper_limit = self.soft_limits(
            HEAD_JOINTS[HEAD_PITCH_INDEX],
            "soft_pitch_limit_deg",
        )

        self.head_positions: Optional[list[float]] = None
        self.command_positions: Optional[list[float]] = None
        self.target_positions: Optional[list[float]] = None
        self.last_state_warning = 0.0
        self.last_dry_run_log = 0.0
        self.should_exit = False

        self.create_subscription(
            JointStateArray,
            self.head_state_topic,
            self.head_state_callback,
            SENSOR_QOS,
        )
        self.command_pub = self.create_publisher(
            JointCommandArray,
            self.head_command_topic,
            RELIABLE_QOS,
        )
        self.create_timer(self.control_period_sec, self.control_loop)
        self.create_timer(0.02, self.keyboard_loop)

        self.print_help()
        self.get_logger().info(
            "Head keyboard teleop started: "
            f"state={self.head_state_topic}, command={self.head_command_topic}, "
            f"dry_run={self.dry_run}"
        )

    def deg_param(self, name: str) -> float:
        return float(self.get_parameter(name).value) * 3.141592653589793 / 180.0

    def soft_limits(self, joint: JointInfo, param_name: str) -> tuple[float, float]:
        soft_limit_rad = self.deg_param(param_name)
        if soft_limit_rad <= 0.0:
            return joint.lower_limit, joint.upper_limit
        return max(joint.lower_limit, -soft_limit_rad), min(
            joint.upper_limit, soft_limit_rad
        )

    def print_help(self) -> None:
        print(
            "\nX2 head keyboard teleop\n"
            "  Left/Right arrows: yaw left/right\n"
            "  Up/Down arrows: pitch up/down\n"
            "  Space: hold current target\n"
            "  c: center softly\n"
            "  q or Ctrl-C: quit\n",
            flush=True,
        )

    def head_state_callback(self, msg: JointStateArray) -> None:
        by_name = {
            joint.name: joint
            for joint in msg.joints
            if getattr(joint, "name", "")
        }

        if all(joint.name in by_name for joint in HEAD_JOINTS):
            self.head_positions = [
                float(by_name[joint.name].position) for joint in HEAD_JOINTS
            ]
        elif len(msg.joints) >= len(HEAD_JOINTS):
            self.head_positions = [
                float(msg.joints[i].position) for i in range(len(HEAD_JOINTS))
            ]
        else:
            return

        self.head_positions = self.clamp_head_positions(self.head_positions)
        if self.command_positions is None:
            self.command_positions = list(self.head_positions)
        if self.target_positions is None:
            self.target_positions = list(self.head_positions)

    def keyboard_loop(self) -> None:
        key = read_key()
        if key is None:
            return

        if key in {"q", "\x03"}:
            self.should_exit = True
            rclpy.shutdown()
            return

        if self.target_positions is None:
            self.get_logger().warn(
                "Ignoring key press until head joint state has been received.",
                throttle_duration_sec=1.0,
            )
            return

        target = list(self.target_positions)
        if key in {"left", "a"}:
            target[HEAD_YAW_INDEX] += self.yaw_step_rad
        elif key in {"right", "d"}:
            target[HEAD_YAW_INDEX] -= self.yaw_step_rad
        elif key in {"up", "w"}:
            target[HEAD_PITCH_INDEX] += self.pitch_step_rad
        elif key in {"down", "s"}:
            target[HEAD_PITCH_INDEX] -= self.pitch_step_rad
        elif key == "c":
            target = [0.0, 0.0]
        elif key == " ":
            target = list(self.command_positions or target)
        else:
            return

        self.target_positions = self.clamp_head_positions(target)
        print(
            "target yaw={:+.1f}deg pitch={:+.1f}deg".format(
                self.target_positions[HEAD_YAW_INDEX] * 180.0 / 3.141592653589793,
                self.target_positions[HEAD_PITCH_INDEX] * 180.0 / 3.141592653589793,
            ),
            flush=True,
        )

    def control_loop(self) -> None:
        if self.head_positions is None:
            now = time.monotonic()
            if now - self.last_state_warning > 3.0:
                self.last_state_warning = now
                self.get_logger().warn(
                    f"No head joint state messages yet on {self.head_state_topic}"
                )
            return

        if self.target_positions is None:
            return

        next_positions, velocities = self.next_head_step(self.target_positions)
        moving = any(abs(vel) > 1e-6 for vel in velocities)
        if moving or self.publish_hold:
            self.publish_head_command(next_positions, velocities)
        self.command_positions = list(next_positions)

    def next_head_step(
        self, target_positions: list[float]
    ) -> tuple[list[float], list[float]]:
        start_positions = self.command_positions or self.head_positions
        if start_positions is None:
            return self.clamp_head_positions(target_positions), [0.0] * len(
                HEAD_JOINTS
            )

        max_delta = self.max_velocity * self.control_period_sec
        positions = [
            move_toward(start_positions[i], target_positions[i], max_delta)
            for i in range(len(HEAD_JOINTS))
        ]
        positions = self.clamp_head_positions(positions)
        velocities = [
            (positions[i] - start_positions[i]) / self.control_period_sec
            for i in range(len(HEAD_JOINTS))
        ]
        return positions, velocities

    def publish_head_command(
        self, positions: list[float], velocities: list[float]
    ) -> None:
        if self.dry_run:
            now = time.monotonic()
            if now - self.last_dry_run_log > 0.5:
                self.last_dry_run_log = now
                self.get_logger().info(
                    "dry_run command: yaw={:+.3f}, pitch={:+.3f}".format(
                        positions[HEAD_YAW_INDEX],
                        positions[HEAD_PITCH_INDEX],
                    )
                )
            return

        self.command_pub.publish(self.make_head_command(positions, velocities))

    def make_head_command(
        self, positions: list[float], velocities: list[float]
    ) -> JointCommandArray:
        cmd = JointCommandArray()
        try:
            cmd.header = MessageHeader()
            cmd.header.stamp = self.get_clock().now().to_msg()
        except Exception:
            pass

        positions = self.clamp_head_positions(positions)
        for i, joint_info in enumerate(HEAD_JOINTS):
            joint = JointCommand()
            joint.name = joint_info.name
            joint.position = positions[i]
            joint.velocity = velocities[i]
            joint.effort = 0.0
            joint.stiffness = joint_info.kp
            joint.damping = joint_info.kd
            cmd.joints.append(joint)

        return cmd

    def clamp_head_positions(self, positions: list[float]) -> list[float]:
        clamped = list(positions)
        clamped[HEAD_YAW_INDEX] = clamp(
            clamped[HEAD_YAW_INDEX], self.yaw_lower_limit, self.yaw_upper_limit
        )
        clamped[HEAD_PITCH_INDEX] = clamp(
            clamped[HEAD_PITCH_INDEX],
            self.pitch_lower_limit,
            self.pitch_upper_limit,
        )
        return clamped


def read_key() -> Optional[str]:
    if not sys.stdin.isatty():
        return None

    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None

    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch

    ready, _, _ = select.select([sys.stdin], [], [], 0.01)
    if not ready:
        return ch
    ch2 = sys.stdin.read(1)
    ready, _, _ = select.select([sys.stdin], [], [], 0.01)
    if not ready:
        return ch
    ch3 = sys.stdin.read(1)

    if ch2 == "[":
        return {
            "A": "up",
            "B": "down",
            "C": "right",
            "D": "left",
        }.get(ch3)
    return None


def main(args=None):
    rclpy.init(args=args)
    old_settings = None
    node = None
    if sys.stdin.isatty():
        old_settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())

    try:
        node = X2HeadKeyboardTeleop()
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        if old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
