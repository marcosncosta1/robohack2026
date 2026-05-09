#!/usr/bin/env python3
"""Turn toward a person coordinate and say "On my way" with X2 TTS.

Input convention:
  - angle is degrees from straight ahead, positive to the robot's left
  - distance is accepted for future face-recognition integration, but turning
    only needs the angle
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from typing import Optional

import rclpy
import rclpy.logging
from rclpy.node import Node

from aimdk_msgs.msg import McLocomotionVelocity, MessageHeader, TtsPriorityLevel
from aimdk_msgs.srv import PlayTts, SetMcInputSource


SOURCE_NAME = "turn_to_person_tts"


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


class TurnToPersonTts(Node):
    def __init__(self) -> None:
        super().__init__("x2_turn_to_person_tts")
        self.velocity_pub = self.create_publisher(
            McLocomotionVelocity,
            "/aima/mc/locomotion/velocity",
            10,
        )
        self.input_source_client = self.create_client(
            SetMcInputSource,
            "/aimdk_5Fmsgs/srv/SetMcInputSource",
        )
        self.tts_client = self.create_client(
            PlayTts,
            "/aimdk_5Fmsgs/srv/PlayTts",
        )

    def register_input_source(self) -> bool:
        self.get_logger().info("Registering locomotion input source...")
        timeout_sec = 8.0
        start = self.get_clock().now().nanoseconds / 1e9

        while not self.input_source_client.wait_for_service(timeout_sec=2.0):
            now = self.get_clock().now().nanoseconds / 1e9
            if now - start > timeout_sec:
                self.get_logger().error("Waiting for input source service timed out.")
                return False
            self.get_logger().info("Waiting for input source service...")

        request = SetMcInputSource.Request()
        request.action.value = 1001
        request.input_source.name = SOURCE_NAME
        request.input_source.priority = 40
        request.input_source.timeout = 1000

        future = None
        for attempt in range(8):
            request.request.header.stamp = self.get_clock().now().to_msg()
            future = self.input_source_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.25)
            if future.done():
                break
            self.get_logger().info(
                f"Trying to register input source... [{attempt + 1}/8]"
            )

        if future is None or not future.done():
            self.get_logger().error("Input source registration failed or timed out.")
            return False

        try:
            response = future.result()
            state = response.response.state.value
            self.get_logger().info(
                f"Input source registered: state={state}, "
                f"task_id={response.response.task_id}"
            )
            return True
        except Exception as exc:
            self.get_logger().error(f"Input source registration exception: {exc}")
            return False

    def say(self, text: str) -> bool:
        self.get_logger().info("Waiting for TTS service...")
        while not self.tts_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("TTS service unavailable, waiting...")

        request = PlayTts.Request()
        request.tts_req.text = text
        request.tts_req.domain = "turn_to_person_tts"
        request.tts_req.trace_id = "turn_to_person"
        request.tts_req.is_interrupted = True
        request.tts_req.priority_weight = 0
        request.tts_req.priority_level = TtsPriorityLevel()
        request.tts_req.priority_level.value = 6

        self.get_logger().info(f"Sending TTS: {text}")
        future = None
        for attempt in range(8):
            request.header.header.stamp = self.get_clock().now().to_msg()
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

    def publish_velocity(self, angular_velocity: float = 0.0) -> None:
        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = SOURCE_NAME
        msg.forward_velocity = 0.0
        msg.lateral_velocity = 0.0
        msg.angular_velocity = float(angular_velocity)
        self.velocity_pub.publish(msg)

    def stop(self, seconds: float = 0.5, hz: float = 20.0) -> None:
        period = 1.0 / hz
        end_time = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end_time:
            self.publish_velocity(0.0)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    def turn_by(
        self,
        angle_deg: float,
        turn_speed: float,
        hz: float,
        invert_turn_direction: bool,
    ) -> None:
        angle_rad = normalize_angle(math.radians(angle_deg))
        if abs(angle_rad) < math.radians(1.0):
            self.get_logger().info("Angle is already close to straight ahead; no turn.")
            self.stop()
            return

        direction = 1.0 if angle_rad > 0.0 else -1.0
        if invert_turn_direction:
            direction *= -1.0

        duration = abs(angle_rad) / turn_speed
        angular_velocity = direction * turn_speed
        period = 1.0 / hz

        self.get_logger().info(
            f"Turning toward person: angle={math.degrees(angle_rad):+.1f} deg, "
            f"angular_velocity={angular_velocity:+.2f} rad/s, "
            f"duration={duration:.2f} s"
        )

        end_time = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < end_time:
            self.publish_velocity(angular_velocity)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

        self.stop()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Turn toward a person and say "On my way" using X2 TTS.'
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
    parser.add_argument("--turn-speed", type=float, default=0.35)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument(
        "--invert-turn-direction",
        action="store_true",
        help="Use this if positive angular velocity turns the robot the wrong way.",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Turn only; do not call the TTS service.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned turn without connecting to ROS.",
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


def main(args=None) -> int:
    parsed = parse_args(sys.argv[1:] if args is None else args)
    if parsed.turn_speed <= 0.0:
        raise ValueError("--turn-speed must be > 0.")
    if parsed.hz <= 0.0:
        raise ValueError("--hz must be > 0.")

    distance_m, angle_deg = resolve_distance_and_angle(parsed)
    normalized_angle_deg = math.degrees(normalize_angle(math.radians(angle_deg)))

    if parsed.dry_run:
        duration = abs(math.radians(normalized_angle_deg)) / parsed.turn_speed
        print(
            f"Plan: distance={distance_m:.2f}m, "
            f"turn={normalized_angle_deg:+.1f}deg, "
            f"duration={duration:.2f}s, text={parsed.text!r}"
        )
        return 0

    rclpy.init()
    node = TurnToPersonTts()

    def signal_handler(sig, _frame):
        node.get_logger().info(f"Received signal {sig}; stopping turn.")
        node.stop()
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if not node.register_input_source():
            return 2

        node.get_logger().info(
            f"Person input: distance={distance_m:.2f}m, "
            f"angle={normalized_angle_deg:+.1f}deg"
        )
        if not parsed.skip_tts:
            node.say(parsed.text)

        node.turn_by(
            angle_deg=normalized_angle_deg,
            turn_speed=parsed.turn_speed,
            hz=parsed.hz,
            invert_turn_direction=parsed.invert_turn_direction,
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
            node.stop()
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
