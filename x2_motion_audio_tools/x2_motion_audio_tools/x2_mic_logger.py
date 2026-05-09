#!/usr/bin/env python3
"""Log AgiBot X2 microphone activity, optionally with AWS Transcribe words."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from array import array

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from aimdk_msgs.msg import ProcessedAudioOutput


MIC_TOPIC = "/agent/process_audio_output"

NO_SPEECH = 0
SPEECH_START = 1
SPEECHING = 2
SPEECH_END = 3

VAD_NAMES = {
    NO_SPEECH: "no speech",
    SPEECH_START: "speech start",
    SPEECHING: "speaking",
    SPEECH_END: "speech end",
}

SAMPLE_RATE_HZ = 16_000
BYTES_PER_SAMPLE = 2
TRANSCRIBE_CHUNK_BYTES = 3_200


def pcm_level_dbfs(pcm: bytes) -> float | None:
    """Return approximate dBFS for signed 16-bit PCM, or None for silence/no data."""
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


async def transcribe_pcm(pcm: bytes, *, region: str, language_code: str) -> str:
    try:
        from amazon_transcribe.client import TranscribeStreamingClient
        from amazon_transcribe.handlers import TranscriptResultStreamHandler
        from amazon_transcribe.model import TranscriptEvent
    except ImportError as exc:
        raise RuntimeError(
            "Missing amazon-transcribe. Run: python3 -m pip install amazon-transcribe"
        ) from exc

    class Handler(TranscriptResultStreamHandler):
        def __init__(self, output_stream: object) -> None:
            super().__init__(output_stream)
            self.parts: list[str] = []

        async def handle_transcript_event(self, event: TranscriptEvent) -> None:
            for result in event.transcript.results:
                if result.is_partial or not result.alternatives:
                    continue
                text = result.alternatives[0].transcript.strip()
                if text:
                    self.parts.append(text)

    client = TranscribeStreamingClient(region=region)
    stream = await client.start_stream_transcription(
        language_code=language_code,
        media_sample_rate_hz=SAMPLE_RATE_HZ,
        media_encoding="pcm",
    )
    handler = Handler(stream.output_stream)

    async def send_audio() -> None:
        for offset in range(0, len(pcm), TRANSCRIBE_CHUNK_BYTES):
            await stream.input_stream.send_audio_event(
                audio_chunk=pcm[offset : offset + TRANSCRIBE_CHUNK_BYTES]
            )
        await stream.input_stream.end_stream()

    await asyncio.gather(send_audio(), handler.handle_events())
    return " ".join(handler.parts).strip()


class X2MicLogger(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("x2_mic_logger")
        self.args = args
        self.buffer: list[bytes] = []
        self.recording = False
        self.last_vad: int | None = None
        self.chunk_count = 0

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100,
        )
        self.create_subscription(ProcessedAudioOutput, MIC_TOPIC, self.on_audio, qos)
        self.get_logger().info(
            "Listening on %s stream_id=%s transcribe=%s"
            % (MIC_TOPIC, args.stream_id, args.transcribe)
        )

    def on_audio(self, msg: ProcessedAudioOutput) -> None:
        stream_id = int(msg.stream_id)
        if self.args.stream_id and stream_id != self.args.stream_id:
            return

        vad = int(msg.audio_vad_state.value)
        audio = bytes(msg.audio_data)
        level = pcm_level_dbfs(audio)
        vad_name = VAD_NAMES.get(vad, "unknown")

        if vad != self.last_vad:
            level_text = "no audio" if level is None else "%.1f dBFS" % level
            self.get_logger().info(
                "stream=%d vad=%s bytes=%d level=%s"
                % (stream_id, vad_name, len(audio), level_text)
            )
            self.last_vad = vad
        elif self.args.verbose_levels and audio:
            self.chunk_count += 1
            if self.chunk_count % self.args.level_every == 0:
                level_text = "silent" if level is None else "%.1f dBFS" % level
                self.get_logger().info(
                    "stream=%d vad=%s bytes=%d level=%s"
                    % (stream_id, vad_name, len(audio), level_text)
                )

        if not self.args.transcribe:
            return

        if vad == SPEECH_START:
            self.buffer = [audio] if audio else []
            self.recording = True
            return

        if vad == SPEECHING and self.recording:
            if audio:
                self.buffer.append(audio)
            return

        if vad == SPEECH_END and self.recording:
            if audio:
                self.buffer.append(audio)
            pcm = b"".join(self.buffer)
            self.buffer = []
            self.recording = False
            duration_sec = len(pcm) / (SAMPLE_RATE_HZ * BYTES_PER_SAMPLE)
            self.get_logger().info("utterance ended: %.2fs, transcribing..." % duration_sec)
            self.transcribe_and_log(pcm)
            return

        if vad == NO_SPEECH:
            self.buffer = []
            self.recording = False

    def transcribe_and_log(self, pcm: bytes) -> None:
        try:
            text = asyncio.run(
                transcribe_pcm(
                    pcm,
                    region=self.args.aws_region,
                    language_code=self.args.language_code,
                )
            )
        except Exception as exc:  # noqa: BLE001 - this is a diagnostic logger.
            self.get_logger().error("transcription failed: %s" % exc)
            return

        if text:
            self.get_logger().info("words: %s" % text)
        else:
            self.get_logger().info("words: <none>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log AgiBot X2 mic VAD and audio levels.")
    parser.add_argument(
        "--stream-id",
        type=int,
        default=int(os.environ.get("X2_MIC_STREAM_ID", "2")),
        help="1=onboard mic, 2=external mic, 0=accept any stream.",
    )
    parser.add_argument(
        "--verbose-levels",
        action="store_true",
        help="Keep logging audio level every few chunks, not only VAD changes.",
    )
    parser.add_argument(
        "--level-every",
        type=int,
        default=20,
        help="With --verbose-levels, log every N audio chunks.",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Send each utterance to Amazon Transcribe and log recognized words.",
    )
    parser.add_argument(
        "--aws-region",
        default=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")),
    )
    parser.add_argument(
        "--language-code",
        default=os.environ.get("TRANSCRIBE_LANGUAGE_CODE", "en-US"),
    )
    return parser.parse_args()


def main() -> None:
    rclpy.init()
    node = X2MicLogger(parse_args())
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
