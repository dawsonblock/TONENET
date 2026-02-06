"""
Unified STT backend with INT8 compute optimized for Mac M2.

Supports:
- faster-whisper (CTranslate2) - primary
- whisper.cpp (Metal) - optional low-latency

INT8 compute is critical for Mac CPU performance.

Install: pip install faster-whisper ctranslate2
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class STTProtocol(Protocol):
    """STT interface."""
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


@dataclass
class STTBackendConfig:
    """Unified STT configuration."""
    engine: str = "faster_whisper"  # faster_whisper | whisper_cpp
    model: str = "large-v3-turbo"   # large-v3 | large-v3-turbo | medium
    compute: str = "int8"           # int8 (Mac CPU) | float16 | float32
    beam: int = 1                   # 1 = fast, 5 = accurate
    language: str = "en"
    device: str = "cpu"
    
    # whisper.cpp specific
    whisper_cpp_cmd: str = "whisper-cli"
    whisper_cpp_model_path: str = ""


class FasterWhisperSTT:
    """
    Faster-Whisper STT with INT8 compute for Mac M2.
    
    Uses CTranslate2 backend which supports INT8 quantization,
    giving significant speedup on CPU without GPU.
    
    Example:
        stt = FasterWhisperSTT(STTBackendConfig(
            model="large-v3-turbo",
            compute="int8",
            beam=1
        ))
        text = stt.transcribe(audio, 16000)
    """
    
    def __init__(self, cfg: STTBackendConfig):
        self.cfg = cfg
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "faster-whisper not installed. Run: pip install faster-whisper ctranslate2"
                )
            
            self._model = WhisperModel(
                self.cfg.model,
                device=self.cfg.device,
                compute_type=self.cfg.compute,
            )
        return self._model
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: float32 mono audio
            sample_rate: Sample rate (16000 recommended)
        
        Returns:
            Transcribed text
        """
        model = self._load_model()
        
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        segments, _info = model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam,
            vad_filter=False,  # We do VAD upstream
        )
        
        text = "".join(seg.text for seg in segments).strip()
        return text


class WhisperCppSTT:
    """
    Whisper.cpp STT via CLI (Metal accelerated on Mac).
    
    Lowest latency option for Mac M2 but requires whisper.cpp installed.
    
    Install: brew install whisper-cpp
    """
    
    def __init__(self, cfg: STTBackendConfig):
        self.cfg = cfg
        if not cfg.whisper_cpp_model_path:
            raise ValueError("whisper_cpp_model_path required for whisper_cpp engine")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        import os
        import subprocess
        import tempfile
        
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile not installed. Run: pip install soundfile")
        
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "audio.wav")
            sf.write(wav_path, audio, sample_rate, subtype="PCM_16")
            
            out_prefix = os.path.join(td, "out")
            
            cmd = [
                self.cfg.whisper_cpp_cmd,
                "-m", os.path.expanduser(self.cfg.whisper_cpp_model_path),
                "-f", wav_path,
                "-l", self.cfg.language,
                "-of", out_prefix,
                "-otxt",
            ]
            
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            
            txt_path = out_prefix + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            
            return (p.stdout or "").strip()


class MockSTTBackend:
    """Mock STT for testing."""
    
    def __init__(self, cfg: STTBackendConfig | None = None):
        self.cfg = cfg
        self.call_count = 0
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        self.call_count += 1
        if audio is None:
            duration = 0.0
        else:
            duration = len(audio) / sample_rate
        return f"[Mock STT {self.call_count}: {duration:.1f}s]"


def create_stt(cfg: STTBackendConfig | None = None, mock: bool = False) -> STTProtocol:
    """
    Factory to create STT backend.
    
    Args:
        cfg: Configuration (uses defaults if None)
        mock: Use mock backend for testing
    
    Returns:
        STT instance
    """
    cfg = cfg or STTBackendConfig()
    
    if mock:
        return MockSTTBackend(cfg)
    
    if cfg.engine == "whisper_cpp":
        return WhisperCppSTT(cfg)
    else:
        return FasterWhisperSTT(cfg)
