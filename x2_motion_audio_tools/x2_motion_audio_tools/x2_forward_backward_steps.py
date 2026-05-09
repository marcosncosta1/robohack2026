#!/usr/bin/env python3
"""AgiBot X2: walk forward a few steps, then walk backward.

This is based on the official keyboard locomotion example, but it runs a fixed
motion sequence instead of reading keys.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import rclpy
import rclpy.logging
from rclpy.node import Node

from aimdk_msgs.msg import McLocomotionVelocity, MessageHeader
from aimdk_msgs.srv import SetMcInputSource


SOURCE_NAME = "node"


class ForwardBackwardSteps(Node):
    def __init__(self):
        super().__init__("forward_backward_steps")
        self.publisher = self.create_publisher(
            McLocomotionVelocity, "/aima/mc/locomotion/velocity", 10
        )
        self.client = self.create_client(
            SetMcInputSource, "/aimdk_5Fmsgs/srv/SetMcInputSource"
        )

    def register_input_source(self) -> bool:
        self.get_logger().info("Registering input source...")

        timeout_sec = 8.0
        start = self.get_clock().now().nanoseconds / 1e9

        while not self.client.wait_for_service(timeout_sec=2.0):
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
            future = self.client.call_async(req)
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
        self.publisher.publish(msg)

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

    def stop(self, seconds: float, hz: float) -> None:
        period = 1.0 / hz
        end_time = time.monotonic() + seconds
        self.get_logger().info(f"Stopping for {seconds:.2f} s")

        while rclpy.ok() and time.monotonic() < end_time:
            self.publish_velocity(0.0, 0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    def run_sequence(
        self,
        forward_speed: float,
        backward_speed: float,
        forward_seconds: float,
        backward_seconds: float,
        pause_seconds: float,
        repeats: int,
        hz: float,
    ) -> None:
        for repeat in range(repeats):
            self.get_logger().info(f"Starting repeat {repeat + 1}/{repeats}")
            self.hold_velocity(forward_speed, forward_seconds, hz)
            self.stop(pause_seconds, hz)
            self.hold_velocity(backward_speed, backward_seconds, hz)
            self.stop(pause_seconds, hz)

        self.get_logger().info("Forward/backward sequence complete")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk forward a few steps, then backward using MC velocity control."
    )
    parser.add_argument("--forward-speed", type=float, default=0.20)
    parser.add_argument("--backward-speed", type=float, default=-0.20)
    parser.add_argument("--forward-seconds", type=float, default=2.5)
    parser.add_argument("--backward-seconds", type=float, default=2.5)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--hz", type=float, default=20.0)
    return parser.parse_args(argv)


def main(args=None) -> None:
    parsed = parse_args(sys.argv[1:] if args is None else args)

    rclpy.init()
    node = ForwardBackwardSteps()

    def signal_handler(sig, _frame):
        node.get_logger().info(f"Received signal {sig}; stopping robot")
        node.stop(seconds=0.5, hz=parsed.hz)
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if not node.register_input_source():
            return

        node.run_sequence(
            forward_speed=parsed.forward_speed,
            backward_speed=parsed.backward_speed,
            forward_seconds=parsed.forward_seconds,
            backward_seconds=parsed.backward_seconds,
            pause_seconds=parsed.pause_seconds,
            repeats=parsed.repeats,
            hz=parsed.hz,
        )
    except Exception as exc:
        rclpy.logging.get_logger("main").fatal(
            f"Program exited with exception: {exc}"
        )
    finally:
        try:
            node.stop(seconds=0.5, hz=parsed.hz)
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
