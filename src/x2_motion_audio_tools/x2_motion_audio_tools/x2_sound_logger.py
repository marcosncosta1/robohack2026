#!/usr/bin/env python3
"""Minimal AgiBot X2 microphone sound detector.

This does not use AWS, Bedrock, or transcription. It only subscribes to the X2
processed microphone topic and logs when incoming PCM audio is above a threshold.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from array import array

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from aimdk_msgs.msg import ProcessedAudioOutput


MIC_TOPIC = "/agent/process_audio_output"
BYTES_PER_SAMPLE = 2


def pcm_level_dbfs(pcm: bytes) -> float | None:
    if len(pcm) < BYTES_PER_SAMPLE:
        return None

    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return None

    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    if rms <= 0:
        return None
    return 20.0 * math.log10(rms / 32768.0)


class X2SoundLogger(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("x2_sound_logger")
        self.args = args
        self.packet_count = 0
        self.sound_count = 0
        self.last_sound_log = 0.0
        self.last_waiting_log = time.monotonic()
        self.status_count = 0

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=500,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(ProcessedAudioOutput, MIC_TOPIC, self.on_audio, qos)
        self.create_timer(1.0, self.on_timer)
        self.get_logger().info(
            "Waiting for mic packets on %s stream_id=%s threshold=%.1f dBFS"
            % (MIC_TOPIC, args.stream_id, args.threshold_dbfs)
        )

    def on_audio(self, msg: ProcessedAudioOutput) -> None:
        stream_id = int(msg.stream_id)
        if self.args.stream_id and stream_id != self.args.stream_id:
            return

        self.packet_count += 1
        audio = bytes(msg.audio_data)
        level = pcm_level_dbfs(audio)

        if self.packet_count == 1:
            self.get_logger().info(
                "First mic packet received: stream=%d bytes=%d" % (stream_id, len(audio))
            )

        if level is None:
            if self.args.log_packets:
                self.get_logger().info(
                    "packet stream=%d bytes=%d level=silent" % (stream_id, len(audio))
                )
            return

        now = time.monotonic()
        if self.args.log_packets:
            self.get_logger().info(
                "packet stream=%d bytes=%d level=%.1f dBFS"
                % (stream_id, len(audio), level)
            )

        if level >= self.args.threshold_dbfs and (
            now - self.last_sound_log >= self.args.cooldown_sec
        ):
            self.sound_count += 1
            self.last_sound_log = now
            self.get_logger().info(
                "SOUND #%d stream=%d level=%.1f dBFS bytes=%d"
                % (self.sound_count, stream_id, level, len(audio))
            )

    def on_timer(self) -> None:
        self.status_count += 1
        if self.packet_count == 0:
            self.get_logger().info(
                "Still waiting for mic packets. Local subscription is active; run "
                "'ros2 topic info -v /agent/process_audio_output' in another terminal "
                "and check Subscription count."
            )
        elif self.status_count % 5 == 0:
            self.get_logger().info(
                "status: packets=%d sounds=%d" % (self.packet_count, self.sound_count)
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal X2 mic sound detector.")
    parser.add_argument(
        "--stream-id",
        type=int,
        default=int(os.environ.get("X2_MIC_STREAM_ID", "0")),
        help="1=onboard mic, 2=external mic, 0=accept any stream.",
    )
    parser.add_argument(
        "--threshold-dbfs",
        type=float,
        default=-45.0,
        help="Print SOUND when chunk RMS is at or above this dBFS value.",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=0.5,
        help="Minimum seconds between SOUND logs.",
    )
    parser.add_argument(
        "--log-packets",
        action="store_true",
        help="Log every mic packet level, useful when tuning threshold.",
    )
    return parser.parse_args()


def main() -> None:
    rclpy.init()
    node = X2SoundLogger(parse_args())
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
