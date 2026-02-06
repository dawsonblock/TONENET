"""
XTTS TTS wrapper with sentence chunking and barge-in support.

Uses Coqui TTS for voice cloning and high-quality synthesis.

Install: pip install TTS sounddevice
"""

import re
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class XTTSVoice:
    """XTTS voice configuration."""
    reference_wav: str | None = None  # Path to voice cloning reference


@dataclass
class TTSConfig:
    """XTTS TTS configuration."""
    device: str = "cpu"            # cpu/mps/cuda
    voice: XTTSVoice | None = None
    
    # Quality settings
    temperature: float = 0.55
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    
    # Timing
    sentence_pause_ms: int = 200
    sample_rate: int = 24000


class XTTSEngine:
    """
    XTTS text-to-speech with sentence chunking and barge-in.
    
    Features:
    - Voice cloning from reference audio
    - Sentence-by-sentence synthesis
    - Immediate stop on barge-in
    
    Example:
        voice = XTTSVoice(reference_wav="./voice.wav")
        tts = XTTSEngine(TTSConfig(voice=voice))
        
        for chunk in tts.split_sentences(text):
            audio, meta = tts.synthesize(chunk)
            tts.play(audio)
    """
    
    def __init__(self, cfg: TTSConfig | None = None):
        """
        Args:
            cfg: TTS configuration
        """
        self.cfg = cfg or TTSConfig()
        self._tts = None
        self._play_lock = threading.Lock()
        self._stop_flag = threading.Event()
    
    def _load_model(self):
        """Lazy load XTTS model."""
        if self._tts is None:
            try:
                from TTS.api import TTS
            except ImportError:
                raise ImportError(
                    "Coqui TTS not installed. Run: pip install TTS"
                )
            
            # XTTS v2 model
            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self._tts.to(self.cfg.device)
        return self._tts
    
    def stop(self):
        """Stop playback immediately (for barge-in)."""
        self._stop_flag.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
    
    def split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences for chunked synthesis.
        
        Keeps chunks short to reduce slur and enable faster barge-in.
        
        Args:
            text: Full response text
        
        Returns:
            List of sentence chunks
        """
        text = text.strip()
        if not text:
            return []
        
        # Split on sentence boundaries
        parts = re.split(r"(?<=[.!?])\s+", text)
        
        # Cap chunk length
        out = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p) <= 220:
                out.append(p)
            else:
                # Hard wrap long sentences
                start = 0
                while start < len(p):
                    out.append(p[start : start + 200].strip())
                    start += 200
        return out
    
    def synthesize(self, text: str) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Synthesize text to audio.
        
        Args:
            text: Text to speak
        
        Returns:
            (audio_array, metadata)
        """
        self._stop_flag.clear()
        tts = self._load_model()
        
        ref_wav = None
        if self.cfg.voice:
            ref_wav = self.cfg.voice.reference_wav
        
        # Synthesize
        wav = tts.tts(
            text=text,
            speaker_wav=ref_wav,
            language="en",
            temperature=self.cfg.temperature,
            length_penalty=self.cfg.length_penalty,
            repetition_penalty=self.cfg.repetition_penalty,
        )
        
        wav = np.asarray(wav, dtype=np.float32)
        
        meta = {
            "model": "xtts_v2",
            "sr": self.cfg.sample_rate,
            "temperature": self.cfg.temperature,
            "length_penalty": self.cfg.length_penalty,
            "repetition_penalty": self.cfg.repetition_penalty,
            "speaker_wav": ref_wav,
            "text_length": len(text),
            "audio_samples": len(wav),
        }
        
        return wav, meta
    
    def play(self, wav: np.ndarray):
        """
        Play audio (blocks until done or stopped).
        
        Args:
            wav: Audio array to play
        """
        if wav is None or wav.size == 0:
            return
        
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice not installed. Run: pip install sounddevice"
            )
        
        with self._play_lock:
            if self._stop_flag.is_set():
                return
            
            sd.play(wav, self.cfg.sample_rate, blocking=False)
            
            # Wait until done or stop requested
            while sd.get_stream().active:
                if self._stop_flag.is_set():
                    sd.stop()
                    return
                sd.sleep(20)


class MockXTTSEngine:
    """Mock TTS for testing without Coqui TTS."""
    
    def __init__(self, cfg: TTSConfig | None = None):
        self.cfg = cfg or TTSConfig()
        self.call_count = 0
        self._stop_flag = threading.Event()
    
    def stop(self):
        self._stop_flag.set()
    
    def split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    
    def synthesize(self, text: str) -> tuple[np.ndarray, dict[str, Any]]:
        self.call_count += 1
        
        # Generate silence proportional to text length
        duration_samples = len(text) * 100
        wav = np.zeros(duration_samples, dtype=np.float32)
        
        return wav, {"mock": True, "text": text[:50]}
    
    def play(self, wav: np.ndarray):
        # No-op for mock
        pass


def get_tts(
    reference_wav: str | None = None,
    device: str = "cpu",
    mock: bool = False
) -> XTTSEngine:
    """
    Factory to get TTS instance.
    
    Args:
        reference_wav: Path to voice cloning reference audio
        device: cpu/mps/cuda
        mock: Use mock TTS for testing
    
    Returns:
        TTS instance
    """
    voice = XTTSVoice(reference_wav=reference_wav) if reference_wav else None
    cfg = TTSConfig(device=device, voice=voice)
    
    if mock:
        return MockXTTSEngine(cfg)
    return XTTSEngine(cfg)
