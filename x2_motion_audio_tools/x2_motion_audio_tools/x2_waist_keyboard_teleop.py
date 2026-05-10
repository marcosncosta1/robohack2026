#!/usr/bin/env python3
"""Keyboard teleop for the X2 waist joints."""

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
    import ruckig
except ImportError:
    ruckig = None

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


WAIST_JOINTS = [
    JointInfo("waist_yaw_joint", -3.43, 2.382, 20.0, 4.0),
    JointInfo("waist_pitch_joint", -0.314, 0.314, 20.0, 4.0),
    JointInfo("waist_roll_joint", -0.488, 0.488, 20.0, 4.0),
]
WAIST_YAW_INDEX = 0
WAIST_PITCH_INDEX = 1
WAIST_ROLL_INDEX = 2


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


class X2WaistKeyboardTeleop(Node):
    """Move waist yaw/pitch/roll with keyboard input from a terminal."""

    def __init__(self) -> None:
        super().__init__("x2_waist_keyboard_teleop")

        if not AIMDK_AVAILABLE:
            self.get_logger().fatal(
                "aimdk_msgs not available. This node must run on the Agibot X2."
            )
            raise RuntimeError("aimdk_msgs not available")

        self.declare_parameter("waist_state_topic", "/aima/hal/joint/waist/state")
        self.declare_parameter("waist_command_topic", "/aima/hal/joint/waist/command")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("yaw_step_deg", 3.0)
        self.declare_parameter("pitch_step_deg", 1.0)
        self.declare_parameter("roll_step_deg", 1.0)
        self.declare_parameter("use_ruckig", True)
        self.declare_parameter("max_velocity", 1.0)
        self.declare_parameter("max_acceleration", 1.0)
        self.declare_parameter("max_jerk", 25.0)
        self.declare_parameter("control_rate_hz", 500.0)
        self.declare_parameter("soft_yaw_limit_deg", 35.0)
        self.declare_parameter("soft_pitch_limit_deg", 12.0)
        self.declare_parameter("soft_roll_limit_deg", 12.0)
        self.declare_parameter("publish_hold", True)

        self.waist_state_topic = self.get_parameter("waist_state_topic").value
        self.waist_command_topic = self.get_parameter("waist_command_topic").value
        self.dry_run = bool_param(self.get_parameter("dry_run").value)
        self.yaw_step_rad = self.deg_param("yaw_step_deg")
        self.pitch_step_rad = self.deg_param("pitch_step_deg")
        self.roll_step_rad = self.deg_param("roll_step_deg")
        self.use_ruckig = bool_param(self.get_parameter("use_ruckig").value)
        self.max_velocity = float(self.get_parameter("max_velocity").value)
        self.max_acceleration = float(self.get_parameter("max_acceleration").value)
        self.max_jerk = float(self.get_parameter("max_jerk").value)
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.control_period_sec = 1.0 / max(control_rate_hz, 0.1)
        self.publish_hold = bool_param(self.get_parameter("publish_hold").value)

        self.joint_lower_limits = [joint.lower_limit for joint in WAIST_JOINTS]
        self.joint_upper_limits = [joint.upper_limit for joint in WAIST_JOINTS]
        self.apply_soft_limit(WAIST_YAW_INDEX, "soft_yaw_limit_deg")
        self.apply_soft_limit(WAIST_PITCH_INDEX, "soft_pitch_limit_deg")
        self.apply_soft_limit(WAIST_ROLL_INDEX, "soft_roll_limit_deg")

        self.waist_positions: Optional[list[float]] = None
        self.waist_velocities: Optional[list[float]] = None
        self.command_positions: Optional[list[float]] = None
        self.target_positions: Optional[list[float]] = None
        self.ruckig_controller = None
        self.ruckig_input = None
        self.ruckig_output = None
        self.ruckig_initialized = False
        self.last_state_warning = 0.0
        self.last_dry_run_log = 0.0
        self.last_ruckig_warning = 0.0
        self.should_exit = False

        if self.use_ruckig:
            if ruckig is None:
                self.get_logger().warn(
                    "ruckig is not installed; using velocity-limited fallback."
                )
                self.use_ruckig = False
            else:
                self.ruckig_controller = ruckig.Ruckig(
                    len(WAIST_JOINTS), self.control_period_sec
                )
                self.ruckig_input = ruckig.InputParameter(len(WAIST_JOINTS))
                self.ruckig_output = ruckig.OutputParameter(len(WAIST_JOINTS))
                self.ruckig_input.max_velocity = [self.max_velocity] * len(
                    WAIST_JOINTS
                )
                self.ruckig_input.max_acceleration = [
                    self.max_acceleration
                ] * len(WAIST_JOINTS)
                self.ruckig_input.max_jerk = [self.max_jerk] * len(WAIST_JOINTS)

        self.create_subscription(
            JointStateArray,
            self.waist_state_topic,
            self.waist_state_callback,
            SENSOR_QOS,
        )
        self.command_pub = self.create_publisher(
            JointCommandArray,
            self.waist_command_topic,
            RELIABLE_QOS,
        )
        self.create_timer(self.control_period_sec, self.control_loop)
        self.create_timer(0.02, self.keyboard_loop)

        self.print_help()
        self.get_logger().info(
            "Waist keyboard teleop started: "
            f"state={self.waist_state_topic}, command={self.waist_command_topic}, "
            f"dry_run={self.dry_run}, use_ruckig={self.use_ruckig}, "
            f"period={self.control_period_sec:.4f}s"
        )

    def deg_param(self, name: str) -> float:
        return float(self.get_parameter(name).value) * 3.141592653589793 / 180.0

    def apply_soft_limit(self, joint_index: int, param_name: str) -> None:
        soft_limit_rad = self.deg_param(param_name)
        if soft_limit_rad <= 0.0:
            return
        joint = WAIST_JOINTS[joint_index]
        self.joint_lower_limits[joint_index] = max(joint.lower_limit, -soft_limit_rad)
        self.joint_upper_limits[joint_index] = min(joint.upper_limit, soft_limit_rad)

    def print_help(self) -> None:
        print(
            "\nX2 waist keyboard teleop\n"
            "  Left/Right arrows: yaw left/right\n"
            "  Up/Down arrows: pitch forward/back\n"
            "  q/e: roll left/right\n"
            "  c: center softly\n"
            "  Space: hold current target\n"
            "  x or Ctrl-C: quit\n",
            flush=True,
        )

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

        self.waist_positions = self.clamp_waist_positions(self.waist_positions)
        if self.command_positions is None:
            self.command_positions = list(self.waist_positions)
        if self.target_positions is None:
            self.target_positions = list(self.waist_positions)
        if self.use_ruckig and not self.ruckig_initialized:
            self.initialize_ruckig()

    def initialize_ruckig(self) -> None:
        if (
            self.ruckig_input is None
            or self.waist_positions is None
            or self.waist_velocities is None
        ):
            return

        self.ruckig_input.current_position = list(self.waist_positions)
        self.ruckig_input.current_velocity = list(self.waist_velocities)
        self.ruckig_input.current_acceleration = [0.0] * len(WAIST_JOINTS)
        self.ruckig_input.target_position = list(
            self.target_positions or self.waist_positions
        )
        self.ruckig_input.target_velocity = [0.0] * len(WAIST_JOINTS)
        self.ruckig_input.target_acceleration = [0.0] * len(WAIST_JOINTS)
        self.ruckig_initialized = True
        self.get_logger().info("Ruckig waist trajectory initialized from joint state.")

    def keyboard_loop(self) -> None:
        key = read_key()
        if key is None:
            return

        if key in {"x", "\x03"}:
            self.should_exit = True
            rclpy.shutdown()
            return

        if self.target_positions is None:
            self.get_logger().warn(
                "Ignoring key press until waist joint state has been received.",
                throttle_duration_sec=1.0,
            )
            return

        target = list(self.target_positions)
        if key in {"left", "a"}:
            target[WAIST_YAW_INDEX] += self.yaw_step_rad
        elif key in {"right", "d"}:
            target[WAIST_YAW_INDEX] -= self.yaw_step_rad
        elif key in {"up", "w"}:
            target[WAIST_PITCH_INDEX] += self.pitch_step_rad
        elif key in {"down", "s"}:
            target[WAIST_PITCH_INDEX] -= self.pitch_step_rad
        elif key == "q":
            target[WAIST_ROLL_INDEX] += self.roll_step_rad
        elif key == "e":
            target[WAIST_ROLL_INDEX] -= self.roll_step_rad
        elif key == "c":
            target = [0.0, 0.0, 0.0]
        elif key == " ":
            target = list(self.command_positions or target)
        else:
            return

        self.target_positions = self.clamp_waist_positions(target)
        print(
            "target yaw={:+.1f}deg pitch={:+.1f}deg roll={:+.1f}deg".format(
                self.target_positions[WAIST_YAW_INDEX] * 180.0 / 3.141592653589793,
                self.target_positions[WAIST_PITCH_INDEX] * 180.0 / 3.141592653589793,
                self.target_positions[WAIST_ROLL_INDEX] * 180.0 / 3.141592653589793,
            ),
            flush=True,
        )

    def control_loop(self) -> None:
        if self.waist_positions is None:
            now = time.monotonic()
            if now - self.last_state_warning > 3.0:
                self.last_state_warning = now
                self.get_logger().warn(
                    f"No waist joint state messages yet on {self.waist_state_topic}"
                )
            return

        if self.target_positions is None:
            return

        if self.use_ruckig:
            next_positions, velocities = self.next_ruckig_step(self.target_positions)
        else:
            next_positions, velocities = self.next_waist_step(self.target_positions)
        moving = any(abs(vel) > 1e-6 for vel in velocities)
        if moving or self.publish_hold:
            self.publish_waist_command(next_positions, velocities)
        self.command_positions = list(next_positions)

    def next_ruckig_step(
        self, target_positions: list[float]
    ) -> tuple[list[float], list[float]]:
        if (
            self.ruckig_controller is None
            or self.ruckig_input is None
            or self.ruckig_output is None
            or not self.ruckig_initialized
        ):
            return self.next_waist_step(target_positions)

        self.ruckig_input.target_position = self.clamp_waist_positions(
            target_positions
        )
        self.ruckig_input.target_velocity = [0.0] * len(WAIST_JOINTS)
        self.ruckig_input.target_acceleration = [0.0] * len(WAIST_JOINTS)
        self.ruckig_input.max_velocity = [self.max_velocity] * len(WAIST_JOINTS)
        self.ruckig_input.max_acceleration = [self.max_acceleration] * len(
            WAIST_JOINTS
        )
        self.ruckig_input.max_jerk = [self.max_jerk] * len(WAIST_JOINTS)

        result = self.ruckig_controller.update(
            self.ruckig_input, self.ruckig_output
        )
        if result not in [ruckig.Result.Working, ruckig.Result.Finished]:
            now = time.monotonic()
            if now - self.last_ruckig_warning > 3.0:
                self.last_ruckig_warning = now
                self.get_logger().warn(
                    "Ruckig waist trajectory failed; using velocity-limited fallback."
                )
            return self.next_waist_step(target_positions)

        positions = self.clamp_waist_positions(list(self.ruckig_output.new_position))
        velocities = list(self.ruckig_output.new_velocity)
        self.ruckig_input.current_position = positions
        self.ruckig_input.current_velocity = velocities
        self.ruckig_input.current_acceleration = list(
            self.ruckig_output.new_acceleration
        )
        return positions, velocities

    def next_waist_step(
        self, target_positions: list[float]
    ) -> tuple[list[float], list[float]]:
        start_positions = self.command_positions or self.waist_positions
        if start_positions is None:
            return self.clamp_waist_positions(target_positions), [0.0] * len(
                WAIST_JOINTS
            )

        max_delta = self.max_velocity * self.control_period_sec
        positions = [
            move_toward(start_positions[i], target_positions[i], max_delta)
            for i in range(len(WAIST_JOINTS))
        ]
        positions = self.clamp_waist_positions(positions)
        velocities = [
            (positions[i] - start_positions[i]) / self.control_period_sec
            for i in range(len(WAIST_JOINTS))
        ]
        return positions, velocities

    def publish_waist_command(
        self, positions: list[float], velocities: list[float]
    ) -> None:
        if self.dry_run:
            now = time.monotonic()
            if now - self.last_dry_run_log > 0.5:
                self.last_dry_run_log = now
                self.get_logger().info(
                    "dry_run command: yaw={:+.3f}, pitch={:+.3f}, roll={:+.3f}".format(
                        positions[WAIST_YAW_INDEX],
                        positions[WAIST_PITCH_INDEX],
                        positions[WAIST_ROLL_INDEX],
                    )
                )
            return

        self.command_pub.publish(self.make_waist_command(positions, velocities))

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

    def clamp_waist_positions(self, positions: list[float]) -> list[float]:
        clamped = list(positions)
        for i in range(len(WAIST_JOINTS)):
            clamped[i] = clamp(
                clamped[i],
                self.joint_lower_limits[i],
                self.joint_upper_limits[i],
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
        node = X2WaistKeyboardTeleop()
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
