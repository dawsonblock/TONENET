"""
Unified TTS backend with multiple engines.

Engines (ordered by speed):
1. Piper - fast, stable, good quality
2. Kokoro - fast, higher quality  
3. XTTS - voice cloning, heavy/slow

Install:
    pip install piper-tts          # Piper
    pip install TTS                # XTTS
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TTSProtocol(Protocol):
    """TTS interface."""
    def synthesize(self, text: str) -> tuple[np.ndarray, int]: ...


@dataclass
class TTSBackendConfig:
    """Unified TTS configuration."""
    engine: str = "piper"           # piper | kokoro | xtts
    
    # Piper settings
    piper_model: str = "en_US-lessac-medium"
    piper_cmd: str = "piper"
    
    # XTTS settings
    xtts_speaker_wav: str | None = None
    xtts_language: str = "en"
    xtts_device: str = "cpu"


class PiperTTS:
    """
    Piper TTS - fast and reliable.
    
    Uses the piper CLI for synthesis. Low latency, good quality.
    Best default choice for realtime voice agents.
    
    Install: pip install piper-tts
    
    Example:
        tts = PiperTTS(TTSBackendConfig())
        audio, sr = tts.synthesize("Hello world")
    """
    
    def __init__(self, cfg: TTSBackendConfig):
        self.cfg = cfg
    
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesize text to audio.
        
        Args:
            text: Text to speak
        
        Returns:
            (audio_array, sample_rate)
        """
        if not text.strip():
            return np.zeros(0, dtype=np.float32), 22050
        
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile not installed. Run: pip install soundfile")
        
        with tempfile.TemporaryDirectory() as td:
            out_wav = os.path.join(td, "out.wav")
            
            cmd = [
                self.cfg.piper_cmd,
                "--model", self.cfg.piper_model,
                "--output_file", out_wav,
            ]
            
            p = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            _stdout, stderr = p.communicate(text)
            
            if p.returncode != 0:
                raise RuntimeError(f"Piper failed: {stderr[:2000]}")
            
            if not os.path.exists(out_wav):
                raise RuntimeError("Piper did not produce output file")
            
            audio, sr = sf.read(out_wav, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            
            return audio, sr


class XTTSTTS:
    """
    XTTS v2 - high quality with voice cloning.
    
    Heavy dependencies, slower inference. Use when you need
    voice cloning or maximum quality.
    
    Install: pip install TTS
    
    Example:
        tts = XTTSTTS(TTSBackendConfig(
            xtts_speaker_wav="./voice.wav"
        ))
        audio, sr = tts.synthesize("Hello world")
    """
    
    def __init__(self, cfg: TTSBackendConfig):
        self.cfg = cfg
        self._tts = None
    
    def _load_model(self):
        if self._tts is None:
            try:
                from TTS.api import TTS
            except ImportError:
                raise ImportError("TTS not installed. Run: pip install TTS")
            
            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self._tts.to(self.cfg.xtts_device)
        return self._tts
    
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not text.strip():
            return np.zeros(0, dtype=np.float32), 24000
        
        tts = self._load_model()
        
        wav = tts.tts(
            text=text,
            speaker_wav=self.cfg.xtts_speaker_wav,
            language=self.cfg.xtts_language,
        )
        
        audio = np.asarray(wav, dtype=np.float32)
        return audio, 24000


class MockTTSBackend:
    """Mock TTS for testing."""
    
    def __init__(self, cfg: TTSBackendConfig | None = None):
        self.cfg = cfg
        self.call_count = 0
    
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.call_count += 1
        # Generate silence proportional to text
        samples = len(text) * 100
        return np.zeros(samples, dtype=np.float32), 24000


def create_tts(cfg: TTSBackendConfig | None = None, mock: bool = False) -> TTSProtocol:
    """
    Factory to create TTS backend.
    
    Args:
        cfg: Configuration
        mock: Use mock for testing
    
    Returns:
        TTS instance
    """
    cfg = cfg or TTSBackendConfig()
    
    if mock:
        return MockTTSBackend(cfg)
    
    if cfg.engine == "xtts":
        return XTTSTTS(cfg)
    else:
        return PiperTTS(cfg)
