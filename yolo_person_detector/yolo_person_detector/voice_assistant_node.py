#!/usr/bin/env python3
"""
Voice Assistant Node (AWS Bedrock)

Listens to the AgiBot X2 microphone, transcribes via Bedrock (Voxtral), generates
a response with Claude, and speaks back via the robot's TTS service.

Audio source: /agent/process_audio_output (ProcessedAudioOutput, 16 kHz S16LE mono
with built-in VAD state).
STT: Voxtral Mini 3B (Bedrock)
LLM: Claude Sonnet 4.6 (Bedrock)
TTS: Robot's PlayTts service

Dependencies:
    pip install boto3
"""

import base64
import json
import os
import re
import struct
import threading
import time
from collections import defaultdict
from typing import Dict, List

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from aimdk_msgs.msg import ProcessedAudioOutput
from aimdk_msgs.srv import PlayTts

import boto3

# ── Config ────────────────────────────────────────────────────────────────────

MIC_TOPIC = "/agent/process_audio_output"
TTS_SERVICE = "/aimdk_5Fmsgs/srv/PlayTts"

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

# VAD state values published by the X2 audio pipeline
NO_SPEECH = 0
SPEECH_START = 1
SPEECHING = 2
SPEECH_END = 3

# Stream filter: 1 = onboard mic, 2 = external mic, 0 = accept any stream
DEFAULT_STREAM_ID = int(os.environ.get("X2_MIC_STREAM_ID", "2"))

# Minimum utterance length (seconds) we bother sending to STT
MIN_UTTERANCE_SEC = 0.35
QOS_DEPTH = 500

# AWS Bedrock config
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
STT_MODEL_ID = "mistral.voxtral-mini-3b-2507"
LLM_MODEL_ID = "anthropic.claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a helpful assistant embedded in a humanoid robot. "
    "Keep responses concise and conversational (1-3 sentences). "
    "You are assisting patients in a healthcare setting."
)

# Wake keywords (any match triggers AI response). Set to an empty list to
# always respond.
WAKE_KEYWORDS: List[str] = ["robot", "hey robot", "hello"]

# ──────────────────────────────────────────────────────────────────────────────


class VoiceAssistantNode(Node):
    def __init__(self):
        super().__init__("voice_assistant_node")

        self.stream_id_filter = DEFAULT_STREAM_ID

        self.bedrock = boto3.client("bedrock-runtime", region_name=REGION)

        # Per-stream buffers driven by VAD state
        self._buffers: Dict[int, List[bytes]] = defaultdict(list)
        self._recording: Dict[int, bool] = defaultdict(bool)

        # Thread safety for the ASR/LLM pipeline
        self._asr_lock = threading.Lock()
        self._asr_busy = False

        # Mute mic during TTS playback
        self._tts_mute_until = 0.0

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=QOS_DEPTH,
        )

        self.sub = self.create_subscription(
            ProcessedAudioOutput,
            MIC_TOPIC,
            self._audio_callback,
            qos,
        )

        self.tts_client = self.create_client(PlayTts, TTS_SERVICE)
        self.get_logger().info(f"Waiting for TTS service: {TTS_SERVICE} ...")
        if self.tts_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().info("TTS service ready")
        else:
            self.get_logger().warn("TTS service not available")

        self.get_logger().info(
            f"Voice assistant started on {MIC_TOPIC} "
            f"(stream_id filter={self.stream_id_filter}, qos depth={QOS_DEPTH}), "
            f"wake words: {WAKE_KEYWORDS or '<always on>'}"
        )

    # ── Audio capture (VAD-driven) ────────────────────────────────────────────

    def _audio_callback(self, msg: ProcessedAudioOutput):
        try:
            if time.monotonic() < self._tts_mute_until:
                return

            stream_id = int(msg.stream_id)
            if self.stream_id_filter and stream_id != self.stream_id_filter:
                return

            vad_state = int(getattr(msg.audio_vad_state, "value", msg.audio_vad_state))
            audio = bytes(msg.audio_data)

            if vad_state == SPEECH_START:
                self._buffers[stream_id] = [audio] if audio else []
                self._recording[stream_id] = True
                return

            if vad_state == SPEECHING and self._recording[stream_id]:
                if audio:
                    self._buffers[stream_id].append(audio)
                return

            if vad_state == SPEECH_END and self._recording[stream_id]:
                if audio:
                    self._buffers[stream_id].append(audio)
                pcm = b"".join(self._buffers.pop(stream_id, []))
                self._recording[stream_id] = False
                self._handle_utterance(pcm)
                return

            if vad_state == NO_SPEECH:
                self._buffers.pop(stream_id, None)
                self._recording[stream_id] = False
        except Exception as e:
            self.get_logger().error(f"Audio callback error: {e}")

    def _handle_utterance(self, pcm: bytes):
        duration_sec = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        if duration_sec < MIN_UTTERANCE_SEC:
            self.get_logger().debug(f"Ignoring {duration_sec:.2f}s utterance (too short)")
            return
        self.get_logger().info(f"✅ Speech end — {duration_sec:.2f}s, processing...")
        self._run_asr_async(pcm)

    # ── ASR via Bedrock Voxtral (threaded) ────────────────────────────────────

    def _run_asr_async(self, pcm: bytes):
        with self._asr_lock:
            if self._asr_busy:
                self.get_logger().warn("ASR busy, skipping")
                return
            self._asr_busy = True

        t = threading.Thread(target=self._run_asr, args=(pcm,), daemon=True)
        t.start()

    def _run_asr(self, pcm: bytes):
        try:
            text = self._transcribe(pcm)
            if not text or len(text) < 2:
                self.get_logger().debug("Transcription too short, ignoring")
                return
            self.get_logger().info(f"📝 Transcript: {text}")
            self._match_and_respond(text)
        except Exception as e:
            self.get_logger().error(f"ASR error: {e}")
        finally:
            with self._asr_lock:
                self._asr_busy = False

    def _transcribe(self, pcm: bytes) -> str:
        """Transcribe PCM audio via Bedrock Voxtral Mini 3B."""
        if len(pcm) < SAMPLE_RATE:  # < 0.5 s at 16 kHz mono 16-bit would be too short
            return ""

        wav_data = self._pcm_to_wav(pcm)
        audio_b64 = base64.b64encode(wav_data).decode("utf-8")

        body = json.dumps({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": {"data": audio_b64, "format": "wav"}},
                    {"type": "text", "text": "Transcribe this audio exactly. Return only the transcription."},
                ],
            }],
            "max_tokens": 500,
        })

        resp = self.bedrock.invoke_model(
            modelId=STT_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        result = json.loads(resp["body"].read())
        if "choices" in result:
            return result["choices"][0]["message"]["content"].strip()
        if "content" in result:
            return result["content"][0]["text"].strip()
        return ""

    # ── Intent matching ───────────────────────────────────────────────────────

    def _match_and_respond(self, text: str):
        lower = text.lower().strip()

        if WAKE_KEYWORDS and not any(kw.lower() in lower for kw in WAKE_KEYWORDS):
            self.get_logger().debug(f"No wake word, ignoring: {text}")
            return

        question = text
        for kw in WAKE_KEYWORDS:
            question = re.sub(re.escape(kw), "", question, flags=re.IGNORECASE)
        question = question.strip(" ,.!?")

        if not question:
            self._play_tts("I'm here. How can I help you?")
            return

        self.get_logger().info(f"[Question] {question}")
        self._run_ai_async(question)

    # ── LLM via Bedrock Claude (threaded) ─────────────────────────────────────

    def _run_ai_async(self, question: str):
        t = threading.Thread(target=self._run_ai, args=(question,), daemon=True)
        t.start()

    def _run_ai(self, question: str):
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 300,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": question}],
            })

            resp = self.bedrock.invoke_model(
                modelId=LLM_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )

            result = json.loads(resp["body"].read())
            answer = result["content"][0]["text"].strip()

            if answer:
                self.get_logger().info(f"💬 Response: {answer}")
                self._play_tts(answer)
            else:
                self._play_tts("Sorry, I didn't catch that. Could you repeat?")

        except Exception as e:
            self.get_logger().error(f"LLM error: {e}")
            self._play_tts("Sorry, I'm having trouble connecting. Please try again.")

    # ── TTS via robot service ─────────────────────────────────────────────────

    def _play_tts(self, text: str):
        if not self.tts_client.service_is_ready():
            self.get_logger().warn("TTS service unavailable, skipping")
            return

        # Estimate playback duration (~3 words/sec), mute mic during playback
        word_count = len(text.split())
        estimated_sec = word_count / 3.0 + 1.5
        self._tts_mute_until = time.monotonic() + estimated_sec
        self.get_logger().info(f"Muting mic for {estimated_sec:.1f}s")

        req = PlayTts.Request()
        req.tts_req.text = text
        req.tts_req.domain = "voice_assistant"
        req.tts_req.trace_id = "voice_assist"
        req.tts_req.is_interrupted = True
        req.tts_req.priority_weight = 0
        req.tts_req.priority_level.value = 6

        req.header.header.stamp = self.get_clock().now().to_msg()
        future = self.tts_client.call_async(req)
        future.add_done_callback(self._on_tts_done)

    def _on_tts_done(self, future):
        try:
            resp = future.result()
            if resp and resp.tts_resp.is_success:
                self.get_logger().info("🔊 TTS playback complete")
            else:
                self.get_logger().warn("TTS playback failed")
        except Exception as e:
            self.get_logger().error(f"TTS callback error: {e}")

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _pcm_to_wav(pcm_data: bytes) -> bytes:
        """Wrap raw PCM in a WAV header (16 kHz, 16-bit, mono)."""
        data_size = len(pcm_data)
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', 36 + data_size, b'WAVE',
            b'fmt ', 16, 1, 1,
            16000, 32000, 2, 16,
            b'data', data_size,
        )
        return header + pcm_data


# ──────────────────────────────────────────────────────────────────────────────


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VoiceAssistantNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
