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
from typing import Any, AsyncIterator

import numpy as np
import torch


@dataclass
class UtteranceSegment:
    """Finalized utterance segment ready for STT."""
    audio16k: np.ndarray      # float32 mono @ 16kHz
    sr: int                   # sample rate (16000)
    t0: float                 # start timestamp
    t1: float                 # end timestamp
    duration_sec: float       # duration in seconds
    rms: float                # RMS level
    vad_stats: dict[str, Any] # VAD metadata
    audio_sha256: str         # content hash for replay


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
        
        # Pre-roll ring buffer
        ring = np.zeros(self.pre_roll_n, dtype=np.float32)
        ring_idx = 0
        
        in_speech = False
        speech_run = 0
        silence_run = 0
        
        cur_frames = []
        t0 = None
        
        while not self._stop:
            try:
                frame = await asyncio.wait_for(self._q.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            
            # Update pre-roll ring buffer
            n = frame.shape[0]
            if n >= self.pre_roll_n:
                ring[:] = frame[-self.pre_roll_n:]
                ring_idx = 0
            else:
                end = ring_idx + n
                if end <= self.pre_roll_n:
                    ring[ring_idx:end] = frame
                else:
                    first = self.pre_roll_n - ring_idx
                    ring[ring_idx:] = frame[:first]
                    ring[: end - self.pre_roll_n] = frame[first:]
                ring_idx = (ring_idx + n) % self.pre_roll_n
            
            # VAD on 16kHz frame
            frame16 = self._to16k(frame)
            wav = torch.from_numpy(frame16).unsqueeze(0)
            with torch.no_grad():
                p = float(self._vad_model(wav, 16000).item())
            
            is_speech = p > self.vad_threshold
            
            if not in_speech:
                if is_speech:
                    speech_run += 1
                else:
                    speech_run = 0
                
                if speech_run >= self.start_frames:
                    in_speech = True
                    silence_run = 0
                    speech_run = 0
                    t0 = time.time()
                    
                    # Include pre-roll
                    if ring_idx == 0:
                        pre = ring.copy()
                    else:
                        pre = np.concatenate([ring[ring_idx:], ring[:ring_idx]])
                    cur_frames = [pre]
                    yield {"type": "speech_start"}
            else:
                cur_frames.append(frame)
                if is_speech:
                    silence_run = 0
                else:
                    silence_run += 1
                
                # Finalize utterance
                total_frames = len(cur_frames)
                if silence_run >= self.end_frames or total_frames >= self.max_frames:
                    in_speech = False
                    t1 = time.time()
                    
                    # Drop if too short
                    if total_frames < self.min_frames:
                        cur_frames = []
                        continue
                    
                    audio = np.concatenate(cur_frames).astype(np.float32)
                    audio16 = self._to16k(audio)
                    
                    seg = UtteranceSegment(
                        audio16k=audio16,
                        sr=16000,
                        t0=t0 if t0 else (t1 - len(audio) / self.mic_sr),
                        t1=t1,
                        duration_sec=float(audio16.shape[0] / 16000.0),
                        rms=_rms(audio16),
                        vad_stats={
                            "p_last": p,
                            "end_frames": self.end_frames,
                            "silence_run": silence_run,
                            "max_frames": self.max_frames,
                        },
                        audio_sha256=_sha256_bytes(audio16.tobytes()),
                    )
                    cur_frames = []
                    silence_run = 0
                    yield {"type": "utterance", "segment": seg}
        
        yield {"type": "stop"}


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
        self._idx = 0
    
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
    
    def stop(self):
        pass
