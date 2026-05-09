#!/usr/bin/env python3
"""AgiBot X2: walk forward/back, then raise both arms.

This intentionally uses two control paths:
- forward/back walking uses the same MC velocity interface as the working
  keyboard example;
- arm raise uses the HAL arm joint command interface and Ruckig trajectory
  interpolation from the joint-control example.

Run the arm section only when the robot is stable and your SDK safety procedure
for HAL joint control is satisfied.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from aimdk_msgs.msg import CommonState, JointCommand, JointCommandArray, JointStateArray
from aimdk_msgs.msg import McControlArea, McPresetMotion, RequestHeader
from aimdk_msgs.msg import McLocomotionVelocity, MessageHeader
from aimdk_msgs.srv import SetMcInputSource, SetMcPresetMotion

try:
    import ruckig
except ImportError:
    ruckig = None


SOURCE_NAME = "node"

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


ARM_JOINTS: List[JointInfo] = [
    JointInfo("left_shoulder_pitch_joint", -3.08, 2.04, 20.0, 2.0),
    JointInfo("left_shoulder_roll_joint", -0.061, 2.993, 20.0, 2.0),
    JointInfo("left_shoulder_yaw_joint", -2.556, 2.556, 20.0, 2.0),
    JointInfo("left_elbow_joint", -2.3556, 0.0, 20.0, 2.0),
    JointInfo("left_wrist_yaw_joint", -2.556, 2.556, 20.0, 2.0),
    JointInfo("left_wrist_pitch_joint", -0.558, 0.558, 20.0, 2.0),
    JointInfo("left_wrist_roll_joint", -1.571, 0.724, 20.0, 2.0),
    JointInfo("right_shoulder_pitch_joint", -3.08, 2.04, 20.0, 2.0),
    JointInfo("right_shoulder_roll_joint", -2.993, 0.061, 20.0, 2.0),
    JointInfo("right_shoulder_yaw_joint", -2.556, 2.556, 20.0, 2.0),
    JointInfo("right_elbow_joint", -2.3556, 0.0, 20.0, 2.0),
    JointInfo("right_wrist_yaw_joint", -2.556, 2.556, 20.0, 2.0),
    JointInfo("right_wrist_pitch_joint", -0.558, 0.558, 20.0, 2.0),
    JointInfo("right_wrist_roll_joint", -0.724, 1.571, 20.0, 2.0),
]


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def smoothstep(alpha: float) -> float:
    alpha = clamp(alpha, 0.0, 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * alpha)


class ForwardBackRaiseArms(Node):
    def __init__(self):
        super().__init__("forward_back_raise_arms")

        self.velocity_pub = self.create_publisher(
            McLocomotionVelocity, "/aima/mc/locomotion/velocity", 10
        )
        self.arm_pub = self.create_publisher(
            JointCommandArray, "/aima/hal/joint/arm/command", PUBLISHER_QOS
        )
        self.input_source_client = self.create_client(
            SetMcInputSource, "/aimdk_5Fmsgs/srv/SetMcInputSource"
        )
        self.preset_motion_client = self.create_client(
            SetMcPresetMotion, "/aimdk_5Fmsgs/srv/SetMcPresetMotion"
        )
        self.create_subscription(
            JointStateArray,
            "/aima/hal/joint/arm/state",
            self.arm_state_callback,
            SUBSCRIBER_QOS,
        )

        self.arm_positions: Optional[List[float]] = None
        self.arm_velocities: Optional[List[float]] = None
        self.last_arm_command: Optional[List[float]] = None

    def arm_state_callback(self, msg: JointStateArray) -> None:
        by_name = {joint.name: joint for joint in msg.joints}
        if not all(joint.name in by_name for joint in ARM_JOINTS):
            if len(msg.joints) < len(ARM_JOINTS):
                return

            # Some SDK builds publish the arm state without useful names. Keep
            # the official model order as a fallback, but prefer names whenever
            # available because a wrong joint order makes the arms jitter badly.
            self.arm_positions = [
                msg.joints[i].position for i in range(len(ARM_JOINTS))
            ]
            self.arm_velocities = [
                msg.joints[i].velocity for i in range(len(ARM_JOINTS))
            ]
            return

        self.arm_positions = [
            by_name[joint.name].position for joint in ARM_JOINTS
        ]
        self.arm_velocities = [
            by_name[joint.name].velocity for joint in ARM_JOINTS
        ]

    def register_input_source(self) -> bool:
        self.get_logger().info("Registering input source...")

        timeout_sec = 8.0
        start = self.get_clock().now().nanoseconds / 1e9

        while not self.input_source_client.wait_for_service(timeout_sec=2.0):
            now = self.get_clock().now().nanoseconds / 1e9
            if now - start > timeout_sec:
                self.get_logger().error("Waiting for service timed out")
                return False
            self.get_logger().info("Waiting for input source service...")

        req = SetMcInputSource.Request()
        req.action.value = 1001
        req.input_source.name = SOURCE_NAME
        req.input_source.priority = 40
        req.input_source.timeout = 1000

        for i in range(8):
            req.request.header.stamp = self.get_clock().now().to_msg()
            future = self.input_source_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.25)
            if future.done():
                break
            self.get_logger().info(f"trying to register input source... [{i}]")

        if not future.done():
            self.get_logger().error("Service call failed or timed out")
            return False

        try:
            resp = future.result()
            state = resp.response.state.value
            self.get_logger().info(
                f"Input source set successfully: state={state}, "
                f"task_id={resp.response.task_id}"
            )
            return True
        except Exception as exc:
            self.get_logger().error(f"Service exception: {exc}")
            return False

    def send_preset_motion(self, area_id: int, motion_id: int) -> bool:
        self.get_logger().info(
            f"Sending preset motion request: area={area_id}, motion={motion_id}"
        )

        timeout_sec = 8.0
        start = self.get_clock().now().nanoseconds / 1e9

        while not self.preset_motion_client.wait_for_service(timeout_sec=2.0):
            now = self.get_clock().now().nanoseconds / 1e9
            if now - start > timeout_sec:
                self.get_logger().error("Waiting for preset motion service timed out")
                return False
            self.get_logger().info("Waiting for preset motion service...")

        request = SetMcPresetMotion.Request()
        request.header = RequestHeader()
        request.area = McControlArea()
        request.motion = McPresetMotion()
        request.area.value = area_id
        request.motion.value = motion_id
        request.interrupt = True

        for i in range(8):
            request.header.stamp = self.get_clock().now().to_msg()
            future = self.preset_motion_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.25)
            if future.done():
                break
            self.get_logger().info(f"trying preset motion request... [{i}]")

        if not future.done():
            self.get_logger().error("Preset motion service call timed out")
            return False

        response = future.result()
        if response is None:
            self.get_logger().error("Preset motion service returned no response")
            return False

        if response.response.header.code == 0:
            self.get_logger().info(
                f"Preset motion accepted: task_id={response.response.task_id}"
            )
            return True

        if response.response.state.value == CommonState.RUNNING:
            self.get_logger().info(
                f"Preset motion running: task_id={response.response.task_id}"
            )
            return True

        self.get_logger().error(
            "Preset motion rejected. Make sure the robot is in Stable Stand "
            "before running the preset arm raise."
        )
        return False

    def publish_velocity(
        self,
        forward_velocity: float,
        lateral_velocity: float = 0.0,
        angular_velocity: float = 0.0,
    ) -> None:
        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = SOURCE_NAME
        msg.forward_velocity = forward_velocity
        msg.lateral_velocity = lateral_velocity
        msg.angular_velocity = angular_velocity
        self.velocity_pub.publish(msg)

    def hold_velocity(self, forward_velocity: float, seconds: float, hz: float) -> None:
        period = 1.0 / hz
        end_time = time.monotonic() + seconds
        self.get_logger().info(
            f"Commanding forward_velocity={forward_velocity:.2f} m/s "
            f"for {seconds:.2f} s"
        )

        while rclpy.ok() and time.monotonic() < end_time:
            self.publish_velocity(forward_velocity)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    def stop_velocity(self, seconds: float, hz: float) -> None:
        period = 1.0 / hz
        end_time = time.monotonic() + seconds
        self.get_logger().info(f"Stopping velocity for {seconds:.2f} s")

        while rclpy.ok() and time.monotonic() < end_time:
            self.publish_velocity(0.0, 0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    def run_forward_back(
        self,
        forward_speed: float,
        backward_speed: float,
        forward_seconds: float,
        backward_seconds: float,
        pause_seconds: float,
        hz: float,
    ) -> None:
        self.hold_velocity(forward_speed, forward_seconds, hz)
        self.stop_velocity(pause_seconds, hz)
        self.hold_velocity(backward_speed, backward_seconds, hz)
        self.stop_velocity(pause_seconds, hz)

    def wait_for_arm_state(self, timeout_sec: float) -> bool:
        self.get_logger().info("Waiting for arm joint state...")
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.arm_positions is not None:
                self.get_logger().info("Arm joint state received.")
                return True
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().error("Timed out waiting for arm joint state.")
        return False

    def arm_target_from_current(self, arm_angle_deg: float) -> List[float]:
        if self.arm_positions is None:
            raise RuntimeError("Arm state is unavailable.")

        target_by_name: Dict[str, float] = {
            "left_shoulder_pitch_joint": -math.radians(arm_angle_deg),
            "left_shoulder_roll_joint": 0.25,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.20,
            "left_wrist_yaw_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": -math.radians(arm_angle_deg),
            "right_shoulder_roll_joint": -0.25,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.20,
            "right_wrist_yaw_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
        }

        target = list(self.arm_positions)
        for i, joint in enumerate(ARM_JOINTS):
            if joint.name in target_by_name:
                target[i] = clamp(
                    target_by_name[joint.name],
                    joint.lower_limit,
                    joint.upper_limit,
                )
        return target

    def make_arm_command(self, positions: List[float], velocities: List[float]) -> JointCommandArray:
        cmd = JointCommandArray()
        try:
            cmd.header = MessageHeader()
            cmd.header.stamp = self.get_clock().now().to_msg()
        except Exception:
            pass

        for i, joint_info in enumerate(ARM_JOINTS):
            joint = JointCommand()
            joint.name = joint_info.name
            joint.position = clamp(
                positions[i],
                joint_info.lower_limit,
                joint_info.upper_limit,
            )
            joint.velocity = velocities[i]
            joint.effort = 0.0
            joint.stiffness = joint_info.kp
            joint.damping = joint_info.kd
            cmd.joints.append(joint)

        return cmd

    def publish_arm_pose(self, positions: List[float], velocities: Optional[List[float]] = None) -> None:
        if velocities is None:
            velocities = [0.0] * len(ARM_JOINTS)
        self.last_arm_command = list(positions)
        self.arm_pub.publish(self.make_arm_command(positions, velocities))

    def raise_arms_with_ruckig(
        self,
        arm_angle_deg: float,
        control_hz: float,
        hold_seconds: float,
    ) -> None:
        if self.arm_positions is None:
            raise RuntimeError("Arm state is unavailable.")

        target = self.arm_target_from_current(arm_angle_deg)
        period = 1.0 / control_hz

        if ruckig is None:
            self.get_logger().warning(
                "ruckig Python package not found; using cosine interpolation fallback."
            )
            self.raise_arms_with_fallback(target, move_seconds=4.0, control_hz=control_hz)
        else:
            self.get_logger().info("Raising arms with Ruckig trajectory.")
            dofs = len(ARM_JOINTS)
            otg = ruckig.Ruckig(dofs, period)
            inp = ruckig.InputParameter(dofs)
            out = ruckig.OutputParameter(dofs)

            inp.current_position = list(self.arm_positions)
            inp.current_velocity = list(self.arm_velocities or [0.0] * dofs)
            inp.current_acceleration = [0.0] * dofs
            inp.target_position = target
            inp.target_velocity = [0.0] * dofs
            inp.target_acceleration = [0.0] * dofs
            inp.max_velocity = [0.8] * dofs
            inp.max_acceleration = [1.0] * dofs
            inp.max_jerk = [25.0] * dofs

            while rclpy.ok():
                result = otg.update(inp, out)
                if result not in [ruckig.Result.Working, ruckig.Result.Finished]:
                    raise RuntimeError("Ruckig trajectory generation failed.")

                self.publish_arm_pose(out.new_position, out.new_velocity)
                inp.current_position = out.new_position
                inp.current_velocity = out.new_velocity
                inp.current_acceleration = out.new_acceleration
                rclpy.spin_once(self, timeout_sec=0.0)

                if result == ruckig.Result.Finished:
                    break
                time.sleep(period)

        if hold_seconds > 0.0:
            self.get_logger().info(f"Holding arms for {hold_seconds:.2f} s.")
            hold_until = time.monotonic() + hold_seconds
            while rclpy.ok() and time.monotonic() < hold_until:
                self.publish_arm_pose(target)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(period)
        else:
            self.get_logger().info("Arm target reached; not holding with repeated HAL commands.")

    def raise_arms_with_fallback(
        self,
        target: List[float],
        move_seconds: float,
        control_hz: float,
    ) -> None:
        if self.arm_positions is None:
            raise RuntimeError("Arm state is unavailable.")

        start = list(self.arm_positions)
        period = 1.0 / control_hz
        start_time = time.monotonic()

        while rclpy.ok():
            elapsed = time.monotonic() - start_time
            alpha = smoothstep(elapsed / move_seconds)
            positions = [
                s + (t - s) * alpha
                for s, t in zip(start, target)
            ]
            self.publish_arm_pose(positions)
            rclpy.spin_once(self, timeout_sec=0.0)

            if elapsed >= move_seconds:
                break
            time.sleep(period)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move forward/back with MC velocity, then raise arms with HAL joint control."
    )
    parser.add_argument(
        "--phase",
        choices=("full", "walk", "arms"),
        default="full",
        help="Run the full sequence, only walking, or only the arm raise.",
    )
    parser.add_argument("--forward-speed", type=float, default=0.20)
    parser.add_argument("--backward-speed", type=float, default=-0.20)
    parser.add_argument("--forward-seconds", type=float, default=2.0)
    parser.add_argument("--backward-seconds", type=float, default=2.0)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--velocity-hz", type=float, default=20.0)
    parser.add_argument("--arm-control-hz", type=float, default=500.0)
    parser.add_argument("--arm-angle-deg", type=float, default=90.0)
    parser.add_argument("--arm-hold-seconds", type=float, default=0.0)
    parser.add_argument(
        "--arm-mode",
        choices=("preset", "hal"),
        default="preset",
        help="Use MC preset arm raise by default; HAL is experimental.",
    )
    parser.add_argument(
        "--preset-wait-seconds",
        type=float,
        default=2.0,
        help="Time to stand still before requesting the preset arm raise.",
    )
    parser.add_argument(
        "--skip-arms",
        action="store_true",
        help="Only run the forward/back walking part. Same as --phase walk.",
    )
    parser.add_argument(
        "--no-arm-prompt",
        action="store_true",
        help="Do not pause before publishing HAL arm joint commands.",
    )
    return parser.parse_args(argv)


def main(args=None) -> None:
    parsed = parse_args(sys.argv[1:] if args is None else args)

    rclpy.init()
    node = ForwardBackRaiseArms()

    def signal_handler(sig, _frame):
        node.get_logger().info(f"Received signal {sig}; stopping velocity.")
        node.stop_velocity(seconds=0.5, hz=parsed.velocity_hz)
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if parsed.skip_arms:
            parsed.phase = "walk"

        if parsed.phase in ("full", "walk"):
            if not node.register_input_source():
                return

            node.run_forward_back(
                forward_speed=parsed.forward_speed,
                backward_speed=parsed.backward_speed,
                forward_seconds=parsed.forward_seconds,
                backward_seconds=parsed.backward_seconds,
                pause_seconds=parsed.pause_seconds,
                hz=parsed.velocity_hz,
            )

        if parsed.phase == "walk":
            node.get_logger().info("Walk phase complete.")
            return

        if parsed.arm_mode == "preset":
            if parsed.phase == "full":
                node.stop_velocity(parsed.preset_wait_seconds, parsed.velocity_hz)

            # AimDK preset table: Raise both hands = motion 1010, area 3.
            node.send_preset_motion(area_id=3, motion_id=1010)
            return

        if not parsed.no_arm_prompt:
            node.get_logger().warning(
                "Next step publishes low-level HAL arm joint commands. "
                "Only continue when the robot is stable and your HAL safety "
                "procedure is satisfied. Press Enter to continue."
            )
            input()

        if not node.wait_for_arm_state(timeout_sec=5.0):
            return

        node.raise_arms_with_ruckig(
            arm_angle_deg=parsed.arm_angle_deg,
            control_hz=parsed.arm_control_hz,
            hold_seconds=parsed.arm_hold_seconds,
        )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        rclpy.logging.get_logger("main").fatal(
            f"Program exited with exception: {exc}"
        )
    finally:
        try:
            node.stop_velocity(seconds=0.5, hz=parsed.velocity_hz)
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
