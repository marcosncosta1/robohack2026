#!/usr/bin/env python3
"""AgiBot X2 voice assistant: mic -> AWS STT -> Bedrock -> X2 TTS.

Run this on a machine that can see the X2 ROS graph after sourcing ROS 2 and
the AimDK workspace. The X2 microphone topic publishes 16 kHz, 16-bit, mono
PCM with VAD events; this node buffers one utterance, sends it to Amazon
Transcribe, sends the resulting text to Amazon Bedrock, then speaks the reply
through the X2 PlayTts service.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
import os
import queue
import threading
import time
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from aimdk_msgs.msg import ProcessedAudioOutput
from aimdk_msgs.srv import PlayTts


MIC_TOPIC = "/agent/process_audio_output"
TTS_SERVICE = "/aimdk_5Fmsgs/srv/PlayTts"

NO_SPEECH = 0
SPEECH_START = 1
SPEECHING = 2
SPEECH_END = 3

SAMPLE_RATE_HZ = 16_000
BYTES_PER_SAMPLE = 2
CHANNELS = 1
TRANSCRIBE_CHUNK_BYTES = 3_200


@dataclass(frozen=True)
class SpeechSegment:
    stream_id: int
    pcm: bytes
    duration_sec: float


class AwsVoiceBackend:
    """Owns Amazon Transcribe and Bedrock calls."""

    def __init__(
        self,
        *,
        bedrock_region: str,
        transcribe_region: str,
        model_id: str,
        language_code: str,
        system_prompt: str,
        max_history_turns: int,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self.transcribe_region = transcribe_region
        self.language_code = language_code
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.messages: deque[dict[str, object]] = deque(
            maxlen=max(2, max_history_turns * 2)
        )

        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=bedrock_region,
            config=Config(connect_timeout=10, read_timeout=90, retries={"max_attempts": 3}),
        )

    async def _transcribe_pcm_async(self, pcm: bytes) -> str:
        try:
            from amazon_transcribe.client import TranscribeStreamingClient
            from amazon_transcribe.handlers import TranscriptResultStreamHandler
            from amazon_transcribe.model import TranscriptEvent
        except ImportError as exc:
            raise RuntimeError(
                "Missing amazon-transcribe. Install scripts/requirements-x2-voice.txt "
                "in the Python environment that runs this node."
            ) from exc

        class SegmentTranscriptHandler(TranscriptResultStreamHandler):
            def __init__(self, output_stream: object) -> None:
                super().__init__(output_stream)
                self.parts: list[str] = []

            async def handle_transcript_event(
                self, transcript_event: TranscriptEvent
            ) -> None:
                for result in transcript_event.transcript.results:
                    if result.is_partial:
                        continue
                    if not result.alternatives:
                        continue
                    text = result.alternatives[0].transcript.strip()
                    if text:
                        self.parts.append(text)

        client = TranscribeStreamingClient(region=self.transcribe_region)
        stream = await client.start_stream_transcription(
            language_code=self.language_code,
            media_sample_rate_hz=SAMPLE_RATE_HZ,
            media_encoding="pcm",
        )
        handler = SegmentTranscriptHandler(stream.output_stream)

        async def write_audio() -> None:
            for offset in range(0, len(pcm), TRANSCRIBE_CHUNK_BYTES):
                chunk = pcm[offset : offset + TRANSCRIBE_CHUNK_BYTES]
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
            await stream.input_stream.end_stream()

        await asyncio.gather(write_audio(), handler.handle_events())
        return " ".join(handler.parts).strip()

    def transcribe_pcm(self, pcm: bytes) -> str:
        return asyncio.run(self._transcribe_pcm_async(pcm))

    def ask_bedrock(self, user_text: str) -> str:
        user_message = {"role": "user", "content": [{"text": user_text}]}
        self.messages.append(user_message)

        response = self.bedrock.converse(
            modelId=self.model_id,
            system=[{"text": self.system_prompt}],
            messages=list(self.messages),
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

        message = response["output"]["message"]
        text_blocks = [
            block["text"]
            for block in message.get("content", [])
            if isinstance(block, dict) and "text" in block
        ]
        reply = " ".join(part.strip() for part in text_blocks if part.strip()).strip()
        if not reply:
            reply = "Sorry, I did not get a usable response."

        self.messages.append({"role": "assistant", "content": [{"text": reply}]})
        return reply


class X2BedrockVoiceAssistant(Node):
    def __init__(self, args: argparse.Namespace, backend: AwsVoiceBackend) -> None:
        super().__init__("x2_bedrock_voice_assistant")
        self.args = args
        self.backend = backend

        self.buffers: dict[int, list[bytes]] = defaultdict(list)
        self.recording: dict[int, bool] = defaultdict(bool)
        self.segments: queue.Queue[SpeechSegment] = queue.Queue(maxsize=args.queue_size)
        self.stop_event = threading.Event()
        self.busy = threading.Event()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=args.qos_depth,
        )
        self.create_subscription(
            ProcessedAudioOutput,
            MIC_TOPIC,
            self._on_audio,
            qos,
        )

        self.tts_client = self.create_client(PlayTts, TTS_SERVICE)
        self._wait_for_tts_service()

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        self.get_logger().info(
            "Listening on %s stream_id=%s; Bedrock model=%s"
            % (MIC_TOPIC, args.stream_id, args.model_id)
        )

    def _wait_for_tts_service(self) -> None:
        while rclpy.ok() and not self.tts_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for X2 TTS service %s..." % TTS_SERVICE)

    def _on_audio(self, msg: ProcessedAudioOutput) -> None:
        stream_id = int(msg.stream_id)
        if self.args.stream_id and stream_id != self.args.stream_id:
            return
        if self.args.mute_input_while_busy and self.busy.is_set():
            return

        vad_state = int(msg.audio_vad_state.value)
        audio = bytes(msg.audio_data)

        if vad_state == SPEECH_START:
            self.buffers[stream_id] = []
            self.recording[stream_id] = True
            if audio:
                self.buffers[stream_id].append(audio)
            return

        if vad_state == SPEECHING and self.recording[stream_id]:
            if audio:
                self.buffers[stream_id].append(audio)
            return

        if vad_state == SPEECH_END and self.recording[stream_id]:
            if audio:
                self.buffers[stream_id].append(audio)
            pcm = b"".join(self.buffers.pop(stream_id, []))
            self.recording[stream_id] = False
            self._enqueue_segment(stream_id, pcm)
            return

        if vad_state == NO_SPEECH:
            self.buffers.pop(stream_id, None)
            self.recording[stream_id] = False

    def _enqueue_segment(self, stream_id: int, pcm: bytes) -> None:
        duration_sec = len(pcm) / (SAMPLE_RATE_HZ * BYTES_PER_SAMPLE * CHANNELS)
        if duration_sec < self.args.min_duration_sec:
            self.get_logger().debug("Ignoring %.2fs utterance as too short" % duration_sec)
            return

        segment = SpeechSegment(stream_id=stream_id, pcm=pcm, duration_sec=duration_sec)
        try:
            self.segments.put_nowait(segment)
            self.get_logger().info(
                "Queued %.2fs speech segment from stream %d" % (duration_sec, stream_id)
            )
        except queue.Full:
            self.get_logger().warning("Speech queue is full; dropping utterance")

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                segment = self.segments.get(timeout=0.2)
            except queue.Empty:
                continue

            self.busy.set()
            try:
                self._handle_segment(segment)
            except (BotoCoreError, ClientError, RuntimeError, KeyError, ValueError) as exc:
                self.get_logger().error("Voice pipeline failed: %s" % exc)
                self._speak("Sorry, I had trouble connecting to the voice service.")
            except Exception as exc:  # noqa: BLE001 - keep the robot loop alive.
                self.get_logger().error("Unexpected voice pipeline failure: %s" % exc)
                self._speak("Sorry, something went wrong.")
            finally:
                if self.args.post_tts_mute_sec > 0:
                    time.sleep(self.args.post_tts_mute_sec)
                self.busy.clear()
                self.segments.task_done()

    def _handle_segment(self, segment: SpeechSegment) -> None:
        self.get_logger().info("Transcribing %.2fs of PCM audio..." % segment.duration_sec)
        user_text = self.backend.transcribe_pcm(segment.pcm)
        if not user_text:
            self.get_logger().info("Transcribe returned no text")
            self._speak("I did not catch that.")
            return

        self.get_logger().info("User said: %s" % user_text)
        reply = self.backend.ask_bedrock(user_text)
        self.get_logger().info("Bedrock replied: %s" % reply)
        self._speak(reply)

    def _speak(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if len(text) > self.args.max_tts_chars:
            text = text[: self.args.max_tts_chars].rstrip() + "..."

        request = PlayTts.Request()
        request.header.header.stamp = self.get_clock().now().to_msg()
        request.tts_req.text = text
        request.tts_req.priority_level.value = self.args.tts_priority
        request.tts_req.priority_weight = self.args.tts_priority_weight
        request.tts_req.domain = "x2_bedrock_voice_assistant"
        request.tts_req.trace_id = uuid.uuid4().hex
        request.tts_req.is_interrupted = self.args.interrupt_same_priority

        future = self.tts_client.call_async(request)
        deadline = time.monotonic() + self.args.tts_timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)

        if not future.done():
            self.get_logger().error("Timed out waiting for TTS service response")
            return False

        response = future.result()
        if response is None:
            self.get_logger().error("TTS service returned no response")
            return False

        if not response.tts_resp.is_success:
            self.get_logger().error("TTS failed: %s" % response.tts_resp.error_message)
            return False

        self.get_logger().info(
            "TTS accepted, estimated duration %d ms"
            % response.tts_resp.estimated_duration
        )
        return True

    def stop(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=2.0)


def env_default(*names: str, fallback: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen on the AgiBot X2 mic, call AWS Bedrock, and speak via X2 TTS."
    )
    default_region = env_default("AWS_REGION", "AWS_DEFAULT_REGION", fallback="us-east-1")
    parser.add_argument("--aws-region", default=default_region)
    parser.add_argument(
        "--bedrock-region",
        default=os.environ.get("BEDROCK_REGION"),
        help="Defaults to --aws-region.",
    )
    parser.add_argument(
        "--transcribe-region",
        default=os.environ.get("TRANSCRIBE_REGION"),
        help="Defaults to --aws-region.",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-2-lite-v1:0"),
        help="Amazon Bedrock model ID or inference profile ID.",
    )
    parser.add_argument(
        "--language-code",
        default=os.environ.get("TRANSCRIBE_LANGUAGE_CODE", "en-US"),
        help="Amazon Transcribe language code, for example en-US or de-CH.",
    )
    parser.add_argument(
        "--stream-id",
        type=int,
        default=int(os.environ.get("X2_MIC_STREAM_ID", "2")),
        help="1=onboard mic, 2=external mic, 0=accept any stream.",
    )
    parser.add_argument("--qos-depth", type=int, default=200)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--min-duration-sec", type=float, default=0.35)
    parser.add_argument("--post-tts-mute-sec", type=float, default=1.2)
    parser.add_argument(
        "--mute-input-while-busy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-history-turns", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tts-chars", type=int, default=650)
    parser.add_argument("--tts-priority", type=int, default=6)
    parser.add_argument("--tts-priority-weight", type=int, default=50)
    parser.add_argument("--tts-timeout-sec", type=float, default=12.0)
    parser.add_argument(
        "--interrupt-same-priority",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--system-prompt",
        default=os.environ.get(
            "VOICE_SYSTEM_PROMPT",
            "You are speaking through an AgiBot X2 robot. "
            "Answer clearly, briefly, and helpfully. Keep responses suitable for text to speech.",
        ),
    )
    args = parser.parse_args()
    args.bedrock_region = args.bedrock_region or args.aws_region
    args.transcribe_region = args.transcribe_region or args.aws_region
    return args


def main() -> None:
    args = parse_args()
    backend = AwsVoiceBackend(
        bedrock_region=args.bedrock_region,
        transcribe_region=args.transcribe_region,
        model_id=args.model_id,
        language_code=args.language_code,
        system_prompt=args.system_prompt,
        max_history_turns=args.max_history_turns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    rclpy.init()
    node = X2BedrockVoiceAssistant(args, backend)
    try:
        rclpy.spin(node)
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
