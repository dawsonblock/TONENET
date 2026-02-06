"""
Whisper.cpp STT wrapper for Mac M2 Metal acceleration.

Uses the whisper.cpp CLI for best practical accuracy/latency on Apple Silicon.
This is the recommended STT for accuracy-first Mac M2 pipelines.

Install:
    brew install whisper-cpp
    # Download model: ggml-large-v3.bin to ~/models/whisper/
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np


@dataclass
class WhisperCppConfig:
    """Configuration for whisper.cpp CLI."""
    whisper_cmd: str = "whisper-cli"  # brew whisper-cpp provides this
    model_path: str = ""               # e.g. ~/models/whisper/ggml-large-v3.bin
    language: str = "en"
    threads: int = 4
    use_gpu: bool = True               # Uses Metal on Mac when built with it
    extra_args: tuple[str, ...] = ()


class WhisperCppSTT:
    """
    Whisper.cpp speech-to-text via CLI.
    
    Best practical choice for accuracy + speed on Mac M2 (Metal accelerated).
    
    Example:
        stt = WhisperCppSTT(WhisperCppConfig(
            model_path="~/models/whisper/ggml-large-v3.bin"
        ))
        text = stt.transcribe(audio_f32, 16000)
    """
    
    def __init__(self, cfg: WhisperCppConfig):
        if not cfg.model_path:
            raise ValueError("model_path is required for whisper.cpp")
        self.cfg = cfg
        self._model_path = os.path.expanduser(cfg.model_path)
    
    def transcribe(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text using whisper.cpp.
        
        Args:
            audio_f32: float32 mono audio
            sample_rate: Sample rate of audio
        
        Returns:
            Transcribed text
        """
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile not installed. Run: pip install soundfile")
        
        audio_f32 = np.asarray(audio_f32, dtype=np.float32)
        if audio_f32.ndim == 2:
            audio_f32 = audio_f32.mean(axis=1)
        
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "utt.wav")
            sf.write(wav_path, audio_f32, sample_rate, subtype="PCM_16")
            
            out_prefix = os.path.join(td, "out")
            
            cmd = [
                self.cfg.whisper_cmd,
                "-m", self._model_path,
                "-f", wav_path,
                "-l", self.cfg.language,
                "-t", str(int(self.cfg.threads)),
                "-of", out_prefix,
                "-otxt",
            ]
            
            if not self.cfg.use_gpu:
                cmd.append("-ng")
            
            cmd.extend(self.cfg.extra_args)
            
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            
            # Prefer generated .txt file
            txt_path = out_prefix + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read().strip()
            
            # Fallback: parse stdout
            return (p.stdout or "").strip()


class MockWhisperCppSTT:
    """Mock for testing without whisper.cpp installed."""
    
    def __init__(self, cfg: WhisperCppConfig | None = None):
        self.cfg = cfg
        self.call_count = 0
    
    def transcribe(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> str:
        self.call_count += 1
        if audio_f32 is None:
            duration = 0.0
        else:
            duration = len(audio_f32) / sample_rate
        return f"[Mock whisper.cpp {self.call_count}: {duration:.1f}s]"


def get_whispercpp_stt(
    model_path: str,
    language: str = "en",
    use_gpu: bool = True,
    mock: bool = False
) -> WhisperCppSTT:
    """
    Factory for whisper.cpp STT.
    
    Args:
        model_path: Path to ggml model file
        language: Language code
        use_gpu: Use Metal acceleration
        mock: Use mock for testing
    
    Returns:
        STT instance
    """
    cfg = WhisperCppConfig(
        model_path=model_path,
        language=language,
        use_gpu=use_gpu
    )
    if mock:
        return MockWhisperCppSTT(cfg)
    return WhisperCppSTT(cfg)
