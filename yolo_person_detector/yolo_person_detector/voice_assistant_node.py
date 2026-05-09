"""
Voice Assistant ROS2 Node.

Subscribes to /agent/process_audio_output to receive noise-suppressed audio,
transcribes speech via Bedrock (Voxtral), generates a response with Claude,
and speaks it back via Amazon Polly.

Pipeline: Mic PCM → Bedrock STT → Bedrock LLM → Polly TTS → Audio playback
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from aimdk_msgs.msg import ProcessedAudioOutput
import os
import json
import base64
import struct
import tempfile
import boto3
from collections import defaultdict
from typing import Dict, List

# QoS for audio subscription (deep queue to avoid missing VAD bursts)
AUDIO_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=500,
)

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
STT_MODEL_ID = "mistral.voxtral-mini-3b-2507"
LLM_MODEL_ID = "anthropic.claude-sonnet-4-6"
POLLY_VOICE = "Danielle"
POLLY_ENGINE = "neural"

SYSTEM_PROMPT = (
    "You are a helpful assistant embedded in a humanoid robot. "
    "Keep responses concise and conversational (1-3 sentences). "
    "You are assisting patients in a healthcare setting."
)


class VoiceAssistantNode(Node):
    """ROS2 node that provides voice interaction via AWS Bedrock and Polly."""

    def __init__(self):
        super().__init__('voice_assistant_node')

        self.audio_buffers: Dict[int, List[bytes]] = defaultdict(list)
        self.recording_state: Dict[int, bool] = defaultdict(bool)

        self.bedrock = boto3.client("bedrock-runtime", region_name=REGION)
        self.polly = boto3.client("polly", region_name=REGION)

        self.subscription = self.create_subscription(
            ProcessedAudioOutput,
            '/agent/process_audio_output',
            self._audio_callback,
            AUDIO_QOS,
        )

        self.get_logger().info("🤖 Voice assistant started. Listening for speech...")

    def _audio_callback(self, msg: ProcessedAudioOutput):
        """Handle incoming audio with VAD state machine."""
        try:
            stream_id = msg.stream_id
            vad_state = msg.audio_vad_state.value
            audio_data = bytes(msg.audio_data)

            if vad_state == 1:  # Speech start
                self.get_logger().info("🎤 Speech start detected")
                self.audio_buffers[stream_id].clear()
                self.recording_state[stream_id] = True
                if audio_data:
                    self.audio_buffers[stream_id].append(audio_data)

            elif vad_state == 2:  # Speech in progress
                if self.recording_state[stream_id] and audio_data:
                    self.audio_buffers[stream_id].append(audio_data)

            elif vad_state == 3:  # Speech end
                self.get_logger().info("✅ Speech end — processing...")
                if self.recording_state[stream_id] and audio_data:
                    self.audio_buffers[stream_id].append(audio_data)
                if self.recording_state[stream_id] and self.audio_buffers[stream_id]:
                    self._process_speech(stream_id)
                self.recording_state[stream_id] = False

            elif vad_state == 0:  # No speech
                self.recording_state[stream_id] = False

        except Exception as e:
            self.get_logger().error(f"Error in audio callback: {e}")

    def _process_speech(self, stream_id: int):
        """Full pipeline: PCM → transcription → LLM → TTS → playback."""
        pcm_data = b''.join(self.audio_buffers[stream_id])
        duration = len(pcm_data) / (16000 * 2)
        self.get_logger().info(f"Processing {duration:.1f}s of audio...")

        transcript = self._transcribe(pcm_data)
        if not transcript:
            self.get_logger().warn("No transcription returned.")
            return
        self.get_logger().info(f"📝 Transcript: {transcript}")

        response = self._generate_response(transcript)
        if not response:
            self.get_logger().warn("No LLM response returned.")
            return
        self.get_logger().info(f"💬 Response: {response}")

        self._speak(response)

    def _transcribe(self, pcm_data: bytes) -> str:
        """Send PCM audio to Voxtral Mini 3B for transcription."""
        try:
            wav_data = self._pcm_to_wav(pcm_data)
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

        except Exception as e:
            self.get_logger().error(f"Transcription error: {e}")
            return ""

    def _generate_response(self, transcript: str) -> str:
        """Send transcript to Claude for a response."""
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 300,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": transcript}],
            })

            resp = self.bedrock.invoke_model(
                modelId=LLM_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )

            result = json.loads(resp["body"].read())
            return result["content"][0]["text"].strip()

        except Exception as e:
            self.get_logger().error(f"LLM inference error: {e}")
            return ""

    def _speak(self, text: str):
        """Convert text to speech with Polly and play."""
        try:
            resp = self.polly.synthesize_speech(
                Text=text,
                OutputFormat="pcm",
                VoiceId=POLLY_VOICE,
                Engine=POLLY_ENGINE,
                SampleRate="16000",
            )

            audio_stream = resp["AudioStream"].read()
            self.get_logger().info(f"🔊 Playing response ({len(audio_stream)} bytes)...")
            self._play_pcm(audio_stream)

        except Exception as e:
            self.get_logger().error(f"TTS/playback error: {e}")

    def _play_pcm(self, pcm_data: bytes):
        """Play PCM audio (16kHz, 16-bit, mono)."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
            stream.write(pcm_data)
            stream.stop_stream()
            stream.close()
            p.terminate()
        except ImportError:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(self._pcm_to_wav(pcm_data))
            tmp.close()
            os.system(f"afplay {tmp.name}")
            os.unlink(tmp.name)

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


def main(args=None):
    rclpy.init(args=args)
    node = VoiceAssistantNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down voice assistant...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
