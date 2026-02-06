"""
VAD Segmenter: Silero VAD + endpointing for accurate utterance boundaries.

Produces finalized utterance segments for high STT accuracy.

Features:
- Pre-roll buffer (keeps consonants)
- Configurable speech start/end thresholds
- Async event stream for real-time processing
- Automatic 16kHz resampling

Install: pip install sounddevice scipy torch
"""

import asyncio
import time
import hashlib
from dataclasses import dataclass
from typing import Any, AsyncIterator, List

import numpy as np
import torch


@dataclass
class UtteranceSegment:
    """Finalized utterance segment ready for STT."""

    audio16k: np.ndarray  # float32 mono @ 16kHz
    sr: int  # sample rate (16000)
    t0: float  # start timestamp
    t1: float  # end timestamp
    duration_sec: float  # duration in seconds
    rms: float  # RMS level
    vad_stats: dict[str, Any]  # VAD metadata
    audio_sha256: str  # content hash for replay


def _rms(x: np.ndarray) -> float:
    """Compute RMS level."""
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _sha256_bytes(b: bytes) -> str:
    """SHA256 hash of bytes."""
    return hashlib.sha256(b).hexdigest()


class VADSegmenter:
    """
    Voice Activity Detection with endpointing.

    Uses Silero VAD for speech detection and produces finalized
    utterance segments optimized for accuracy-first STT.

    Parameters tuned for accuracy:
    - pre_roll_ms: 400ms (preserves consonants)
    - start_ms: 120ms (confirmed speech start)
    - end_silence_ms: 700ms (clean utterance boundaries)

    Example:
        vad = VADSegmenter()
        async for event in vad.events():
            if event["type"] == "utterance":
                segment = event["segment"]
                # Send to STT
    """

    def __init__(
        self,
        *,
        mic_sr: int = 48000,
        frame_ms: int = 20,
        pre_roll_ms: int = 400,
        start_ms: int = 120,
        end_silence_ms: int = 700,
        min_utt_ms: int = 300,
        max_utt_s: float = 12.0,
        vad_threshold: float = 0.5,
        device: str = "cpu",
    ):
        """
        Args:
            mic_sr: Microphone sample rate
            frame_ms: Frame duration in ms
            pre_roll_ms: Pre-roll buffer to preserve consonants
            start_ms: Speech start threshold
            end_silence_ms: Silence needed to finalize utterance
            min_utt_ms: Minimum utterance length
            max_utt_s: Maximum utterance length (forces split)
            vad_threshold: VAD probability threshold
            device: PyTorch device
        """
        self.mic_sr = mic_sr
        self.frame_ms = frame_ms
        self.frame_n = int(mic_sr * frame_ms / 1000)
        self.pre_roll_n = int(mic_sr * pre_roll_ms / 1000)
        self.start_frames = max(1, int(start_ms / frame_ms))
        self.end_frames = max(1, int(end_silence_ms / frame_ms))
        self.min_frames = max(1, int(min_utt_ms / frame_ms))
        self.max_frames = int(max_utt_s * 1000 / frame_ms)
        self.vad_threshold = vad_threshold
        self.device = device

        self._q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)
        self._stop = False

        # Load Silero VAD
        self._vad_model = None
        self._stream = None

    def _load_vad(self):
        """Lazy load Silero VAD model."""
        if self._vad_model is None:
            self._vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self._vad_model.to(self.device)
        return self._vad_model

    def stop(self):
        """Stop capture."""
        self._stop = True
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def _callback(self, indata, frames, time_info, status):
        """Audio callback for sounddevice."""
        x = indata[:, 0].copy().astype(np.float32)
        try:
            self._q.put_nowait(x)
        except asyncio.QueueFull:
            pass

    def _start_stream(self):
        """Start microphone capture."""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("sounddevice not installed. Run: pip install sounddevice")

        self._stream = sd.InputStream(
            samplerate=self.mic_sr,
            channels=1,
            blocksize=self.frame_n,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def _to16k(self, x: np.ndarray) -> np.ndarray:
        """Resample to 16kHz."""
        if self.mic_sr == 16000:
            return x.astype(np.float32)

        try:
            from scipy.signal import resample_poly
        except ImportError:
            raise ImportError("scipy not installed. Run: pip install scipy")

        g = np.gcd(self.mic_sr, 16000)
        up = 16000 // g
        down = self.mic_sr // g
        return resample_poly(x, up, down).astype(np.float32)

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        """
        Async generator yielding VAD events.

        Yields:
            {"type": "speech_start"} - Speech detected
            {"type": "utterance", "segment": UtteranceSegment} - Finalized utterance
            {"type": "stop"} - Capture stopped
        """
        self._load_vad()
        self._start_stream()
        self._reset_state()

        while not self._stop:
            try:
                frame = await asyncio.wait_for(self._q.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            # Process frame using the sync method logic
            evt = self._process_frame(frame)
            if evt:
                if evt["type"] == "utterance":
                    yield evt
                elif evt["type"] == "speech_start":
                    yield evt

        yield {"type": "stop"}

    def _reset_state(self):
        """Reset VAD state."""
        self._ring = np.zeros(self.pre_roll_n, dtype=np.float32)
        self._ring_idx = 0
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._cur_frames = []
        self._t0 = None

        # For synchronous processing, accumulate events
        self._event_buffer: List[dict[str, Any]] = []

    def process(self, chunk: np.ndarray) -> List[dict[str, Any]]:
        """
        Synchronous processing for threaded pipelines.

        Args:
            chunk: Audio chunk

        Returns:
            A list of event dicts ({"type": "speech_start"} or {"type": "utterance", "audio": ...}).
            Returns an empty list if no events occurred.
        """
        self._load_vad()

        # Determine if we should reset (first call)
        if not hasattr(self, "_in_speech"):
            self._reset_state()

        # Accumulate into a buffer and process in standard frame sizes
        if not hasattr(self, "_proc_buf"):
            self._proc_buf = np.array([], dtype=np.float32)

        self._proc_buf = np.concatenate([self._proc_buf, chunk])

        # Clear event buffer for this call
        self._event_buffer = []

        # Process in chunks of self.frame_n
        while len(self._proc_buf) >= self.frame_n:
            frame = self._proc_buf[: self.frame_n]
            self._proc_buf = self._proc_buf[self.frame_n :]

            events = self._process_frame(frame)
            if events:
                self._event_buffer.extend(events)

        return self._event_buffer

    def _process_frame(self, frame: np.ndarray) -> List[dict[str, Any]]:
        """Internal frame processing logic. Returns a list of events."""
        events: List[dict[str, Any]] = []

        # Update pre-roll ring buffer
        n = frame.shape[0]
        if n >= self.pre_roll_n:
            self._ring[:] = frame[-self.pre_roll_n :]
            self._ring_idx = 0
        else:
            end = self._ring_idx + n
            if end <= self.pre_roll_n:
                self._ring[self._ring_idx : end] = frame
            else:
                first = self.pre_roll_n - self._ring_idx
                self._ring[self._ring_idx :] = frame[:first]
                self._ring[: end - self.pre_roll_n] = frame[first:]
            self._ring_idx = (self._ring_idx + n) % self.pre_roll_n

        # VAD on 16kHz frame
        frame16 = self._to16k(frame)
        wav = torch.from_numpy(frame16).unsqueeze(0)

        # Ensure VAD model is on device
        if next(self._vad_model.parameters()).device != torch.device(self.device):
            self._vad_model.to(self.device)

        with torch.no_grad():
            p = float(self._vad_model(wav, 16000).item())

        is_speech = p > self.vad_threshold

        if not self._in_speech:
            if is_speech:
                self._speech_run += 1
            else:
                self._speech_run = 0

            if self._speech_run >= self.start_frames:
                self._in_speech = True
                self._silence_run = 0
                self._speech_run = 0
                self._t0 = time.time()

                # Include pre-roll
                if self._ring_idx == 0:
                    pre = self._ring.copy()
                else:
                    pre = np.concatenate(
                        [self._ring[self._ring_idx :], self._ring[: self._ring_idx]]
                    )
                self._cur_frames = [pre]
                return [{"type": "speech_start"}]
        else:
            self._cur_frames.append(frame)
            if is_speech:
                self._silence_run = 0
            else:
                self._silence_run += 1

            # Finalize utterance
            total_frames = len(self._cur_frames)
            if self._silence_run >= self.end_frames or total_frames >= self.max_frames:
                self._in_speech = False
                t1 = time.time()

                # Drop if too short
                if total_frames < self.min_frames:
                    self._cur_frames = []
                    return []

                audio = np.concatenate(self._cur_frames).astype(np.float32)
                audio16 = self._to16k(audio)

                seg = UtteranceSegment(
                    audio16k=audio16,
                    sr=16000,
                    t0=self._t0 if self._t0 else (t1 - len(audio) / self.mic_sr),
                    t1=t1,
                    duration_sec=float(audio16.shape[0] / 16000.0),
                    rms=_rms(audio16),
                    vad_stats={
                        "p_last": p,
                        "end_frames": self.end_frames,
                        "silence_run": self._silence_run,
                        "max_frames": self.max_frames,
                    },
                    audio_sha256=_sha256_bytes(audio16.tobytes()),
                )
                self._cur_frames = []
                self._silence_run = 0
                return [{"type": "utterance", "segment": seg}]

        return []


class SimulatedVADSegmenter:
    """
    Simulated VAD for testing without microphone.

    Generates fake utterance segments from audio files or silence.
    """

    def __init__(self, audio_chunks: list[np.ndarray] | None = None):
        """
        Args:
            audio_chunks: List of audio arrays to simulate as utterances
        """
        self.audio_chunks = audio_chunks or []
        self._buffer = np.array([], dtype=np.float32)
        self._frame_count = 0

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield simulated events."""
        for chunk in self.audio_chunks:
            yield {"type": "speech_start"}
            await asyncio.sleep(0.1)

            seg = UtteranceSegment(
                audio16k=chunk.astype(np.float32),
                sr=16000,
                t0=time.time() - len(chunk) / 16000,
                t1=time.time(),
                duration_sec=len(chunk) / 16000,
                rms=_rms(chunk),
                vad_stats={"simulated": True},
                audio_sha256=_sha256_bytes(chunk.tobytes()),
            )
            yield {"type": "utterance", "segment": seg}
            await asyncio.sleep(0.1)

        yield {"type": "stop"}

    def process(self, chunk: np.ndarray) -> List[dict[str, Any]]:
        """Sync process method."""
        # Simple simulation
        self._buffer = np.concatenate([self._buffer, chunk])

        events = []

        # Emit speech_start halfway to threshold - simple logic
        # We need a latch to prevent repeated start events per utterance
        if not hasattr(self, "_sent_start"):
            self._sent_start = False

        if len(self._buffer) > 16000 and not self._sent_start:
            events.append({"type": "speech_start"})
            self._sent_start = True

        # Every 32000 samples (2s), emit an utterance
        if len(self._buffer) > 32000:
            utt = self._buffer[:32000]
            self._buffer = self._buffer[32000:]
            events.append({"type": "utterance", "audio": utt.astype(np.float32)})
            self._sent_start = False

        return events

    def stop(self):
        pass
