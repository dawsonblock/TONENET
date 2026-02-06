"""
WebRTC VAD wrapper for low-latency voice activity detection.

Lighter alternative to Silero VAD, good for simple utterance gating.

Install: pip install webrtcvad
"""

from dataclasses import dataclass

import numpy as np


def _to_mono_float32(x: np.ndarray) -> np.ndarray:
    """Convert to mono float32."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Simple linear resampling."""
    if sr_in == sr_out:
        return x
    n_in = len(x)
    if n_in == 0:
        return x
    dur = n_in / float(sr_in)
    n_out = int(round(dur * sr_out))
    if n_out <= 1:
        return np.zeros((0,), dtype=np.float32)
    t_in = np.linspace(0.0, dur, num=n_in, endpoint=False)
    t_out = np.linspace(0.0, dur, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32, copy=False)


def _float32_to_int16_pcm(x: np.ndarray) -> bytes:
    """Convert float32 to 16-bit PCM bytes."""
    x = np.clip(x, -1.0, 1.0)
    i16 = (x * 32767.0).astype(np.int16)
    return i16.tobytes()


@dataclass
class WebRTCVADConfig:
    """WebRTC VAD configuration."""
    aggressiveness: int = 2          # 0..3 (3 = most aggressive speech-only)
    sample_rate: int = 16000         # webrtcvad: 8000/16000/32000/48000
    frame_ms: int = 20               # 10/20/30
    start_frames: int = 3            # Consecutive speech frames to start
    end_silence_frames: int = 8      # Consecutive non-speech frames to end


class WebRTCVADGate:
    """
    WebRTC VAD gate for utterance segmentation.
    
    Lighter than Silero, good for simple speech/non-speech gating.
    Feed audio chunks and get speech start/end events.
    
    Example:
        gate = WebRTCVADGate(WebRTCVADConfig(), input_sample_rate=48000)
        
        for audio_chunk in mic_stream:
            started, ended, active = gate.push(audio_chunk)
            if started:
                print("User started speaking")
            if ended:
                print("User stopped speaking")
    """
    
    def __init__(self, cfg: WebRTCVADConfig, input_sample_rate: int):
        if cfg.frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30")
        if cfg.sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("sample_rate must be 8000/16000/32000/48000")
        
        try:
            import webrtcvad
        except ImportError:
            raise ImportError("webrtcvad not installed. Run: pip install webrtcvad")
        
        self.cfg = cfg
        self.input_sample_rate = int(input_sample_rate)
        self.vad = webrtcvad.Vad(int(cfg.aggressiveness))
        
        self._speech_run = 0
        self._silence_run = 0
        self._active = False
        
        self._buf_16k = np.zeros((0,), dtype=np.float32)
        self._frame_len = int(cfg.sample_rate * cfg.frame_ms / 1000)
    
    @property
    def speech_active(self) -> bool:
        """Check if speech is currently active."""
        return self._active
    
    def reset(self) -> None:
        """Reset state (call after utterance ends)."""
        self._speech_run = 0
        self._silence_run = 0
        self._active = False
        self._buf_16k = np.zeros((0,), dtype=np.float32)
    
    def push(self, audio_f32: np.ndarray) -> tuple[bool, bool, bool]:
        """
        Push audio and get speech events.
        
        Args:
            audio_f32: float32 audio chunk (any sample rate)
        
        Returns:
            (speech_started, speech_ended, speech_active)
        """
        x = _to_mono_float32(audio_f32)
        
        # Resample to VAD sample rate
        x16 = _resample_linear(x, self.input_sample_rate, self.cfg.sample_rate)
        
        if x16.size:
            self._buf_16k = np.concatenate([self._buf_16k, x16], axis=0)
        
        speech_started = False
        speech_ended = False
        
        while self._buf_16k.size >= self._frame_len:
            frame = self._buf_16k[:self._frame_len]
            self._buf_16k = self._buf_16k[self._frame_len:]
            
            pcm = _float32_to_int16_pcm(frame)
            is_speech = self.vad.is_speech(pcm, self.cfg.sample_rate)
            
            if is_speech:
                self._speech_run += 1
                self._silence_run = 0
            else:
                self._silence_run += 1
                self._speech_run = 0
            
            if not self._active and self._speech_run >= self.cfg.start_frames:
                self._active = True
                speech_started = True
                self._silence_run = 0
            
            if self._active and self._silence_run >= self.cfg.end_silence_frames:
                self._active = False
                speech_ended = True
                self._speech_run = 0
        
        return speech_started, speech_ended, self._active


class MockWebRTCVAD:
    """Mock VAD for testing without webrtcvad."""
    
    def __init__(self, cfg: WebRTCVADConfig | None = None, input_sample_rate: int = 16000):
        self.cfg = cfg or WebRTCVADConfig()
        self.input_sample_rate = input_sample_rate
        self._active = False
        self._frame_count = 0
    
    @property
    def speech_active(self) -> bool:
        return self._active
    
    def reset(self) -> None:
        self._active = False
        self._frame_count = 0
    
    def push(self, audio_f32: np.ndarray) -> tuple[bool, bool, bool]:
        self._frame_count += 1
        # Simulate: speech every 10 frames, lasts 5 frames
        cycle = self._frame_count % 15
        
        speech_started = (cycle == 5)
        speech_ended = (cycle == 10)
        
        if speech_started:
            self._active = True
        if speech_ended:
            self._active = False
        
        return speech_started, speech_ended, self._active
