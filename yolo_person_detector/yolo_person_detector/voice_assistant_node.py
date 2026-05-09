#!/usr/bin/env python3
"""
Voice Assistant Node (AWS Bedrock)

Listens to microphone audio, transcribes via Bedrock (Voxtral), generates a
response with Claude, and speaks back via the robot's TTS service.

Audio source: /aima/hal/audio/capture (raw multi-channel PCM, 16kHz S16LE, channel 3)
STT: Voxtral Mini 3B (Bedrock)
LLM: Claude Sonnet 4.6 (Bedrock)
TTS: Robot's PlayTts service

Dependencies:
    pip install numpy boto3
"""

import base64
import json
import os
import struct
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from aimdk_msgs.msg import AudioCapture
from aimdk_msgs.srv import PlayTts

import boto3

# ── Config ────────────────────────────────────────────────────────────────────

TTS_SERVICE = "/aimdk_5Fmsgs/srv/PlayTts"
SAMPLE_RATE = 16000
MIC_CHANNEL = 3

# Energy VAD parameters
ENERGY_THRESHOLD = 80
SPEECH_FRAMES_MIN = 5
SILENCE_FRAMES_END = 20
MAX_SPEECH_FRAMES = 200

# AWS Bedrock config
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
STT_MODEL_ID = "mistral.voxtral-mini-3b-2507"
LLM_MODEL_ID = "anthropic.claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a helpful assistant embedded in a humanoid robot. "
    "Keep responses concise and conversational (1-3 sentences). "
    "You are assisting patients in a healthcare setting."
)

# Wake keywords (any match triggers AI response)
WAKE_KEYWORDS = ["robot", "hey robot", "hello"]

# ──────────────────────────────────────────────────────────────────────────────


class VoiceAssistantNode(Node):
    def __init__(self):
        super().__init__("voice_assistant_node")

        self.bedrock = boto3.client("bedrock-runtime", region_name=REGION)

        # VAD state
        self._speech_buf: list[np.ndarray] = []
        self._silence_count = 0
        self._speech_count = 0
        self._recording = False

        # Thread safety
        self._asr_lock = threading.Lock()
        self._asr_busy = False

        # Mute mic during TTS playback
        self._tts_mute_until = 0.0

        self.inited = False
        self.mic_channels = 0
        self.ref_channels = 0

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=500,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        self.sub = self.create_subscription(
            AudioCapture,
            "/aima/hal/audio/capture",
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
            f"🤖 Voice assistant started (threshold={ENERGY_THRESHOLD}), "
            f"wake words: {WAKE_KEYWORDS}"
        )

    # ── Audio capture & energy VAD ────────────────────────────────────────────

    def _audio_callback(self, msg: AudioCapture):
        if time.monotonic() < self._tts_mute_until:
            return

        if not self.inited:
            self.mic_channels = msg.mic_channels
            self.ref_channels = msg.ref_channels
            total = self.mic_channels + self.ref_channels
            self.get_logger().info(
                f"Audio format: mic={self.mic_channels} ref={self.ref_channels} "
                f"total={total} rate={SAMPLE_RATE} channel={MIC_CHANNEL}"
            )
            self.inited = True

        raw = np.frombuffer(bytes(msg.data.data), dtype=np.int16)
        total_ch = self.mic_channels + self.ref_channels
        if total_ch == 0 or len(raw) % total_ch != 0:
            return

        samples = raw.reshape(-1, total_ch)
        frame = samples[:, MIC_CHANNEL].copy()
        rms = int(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

        is_speech = rms > ENERGY_THRESHOLD

        if not self._recording:
            if is_speech:
                self._speech_count += 1
                self._speech_buf.append(frame)
                if self._speech_count >= SPEECH_FRAMES_MIN:
                    self._recording = True
                    self._silence_count = 0
            else:
                self._speech_count = 0
                self._speech_buf.clear()
        else:
            self._speech_buf.append(frame)
            if is_speech:
                self._silence_count = 0
            else:
                self._silence_count += 1

            if (self._silence_count >= SILENCE_FRAMES_END
                    or len(self._speech_buf) >= MAX_SPEECH_FRAMES):
                pcm_int16 = np.concatenate(self._speech_buf).astype(np.int16)
                self._speech_buf = []
                self._recording = False
                self._silence_count = 0
                self._speech_count = 0
                self._run_asr_async(pcm_int16)

    # ── ASR via Bedrock Voxtral (threaded) ────────────────────────────────────

    def _run_asr_async(self, pcm_int16: np.ndarray):
        with self._asr_lock:
            if self._asr_busy:
                self.get_logger().warn("ASR busy, skipping")
                return
            self._asr_busy = True

        t = threading.Thread(target=self._run_asr, args=(pcm_int16,), daemon=True)
        t.start()

    def _run_asr(self, pcm_int16: np.ndarray):
        try:
            text = self._transcribe(pcm_int16)
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

    def _transcribe(self, pcm_int16: np.ndarray) -> str:
        """Transcribe PCM audio via Bedrock Voxtral Mini 3B."""
        speech_bytes = pcm_int16.tobytes()
        if len(speech_bytes) < 16000:
            return ""

        wav_data = self._pcm_to_wav(speech_bytes)
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

        # Check for wake keyword
        if not any(kw.lower() in lower for kw in WAKE_KEYWORDS):
            self.get_logger().debug(f"No wake word, ignoring: {text}")
            return

        # Strip wake words to get the question
        question = text
        for kw in WAKE_KEYWORDS:
            question = question.replace(kw, "").replace(kw.lower(), "")
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
        """Wrap raw PCM in a WAV header (16kHz, 16-bit, mono)."""
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
