"""
Whisper STT wrapper with accuracy-first settings.

Uses faster-whisper for speed on Apple Silicon.
Optimized for utterance-level transcription (not streaming).

Install: pip install faster-whisper
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class STTConfig:
    """Whisper STT configuration."""
    model_name: str = "large-v3"
    device: str = "cpu"           # cpu/mps/cuda
    compute_type: str = "float16" # float16/int8/float32
    language: str = "en"
    
    # Accuracy-first decode settings
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    temperature_fallback: float = 0.2
    condition_on_previous_text: bool = True


class WhisperSTT:
    """
    Whisper speech-to-text with accuracy-first settings.
    
    Uses faster-whisper for efficient inference on Mac M2.
    Optimized for finalized utterances (not streaming partials).
    
    Example:
        stt = WhisperSTT(STTConfig(model_name="large-v3-turbo"))
        text, meta = stt.transcribe(audio16k)
    """
    
    def __init__(self, cfg: STTConfig | None = None):
        """
        Args:
            cfg: STT configuration (defaults to accuracy-first settings)
        """
        self.cfg = cfg or STTConfig()
        self._model = None
        self._prev_text: str = ""
    
    def _load_model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "faster-whisper not installed. Run: pip install faster-whisper"
                )
            
            # Note: MPS support depends on faster-whisper build
            # Falls back to CPU if MPS not available
            device = self.cfg.device
            compute = self.cfg.compute_type
            if device == "mps":
                # faster-whisper may not support MPS directly
                device = "cpu"
                compute = "float32"
            
            self._model = WhisperModel(
                self.cfg.model_name,
                device=device,
                compute_type=compute,
            )
        return self._model
    
    def transcribe(
        self,
        audio16k: np.ndarray,
        prompt: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Transcribe audio to text.
        
        Args:
            audio16k: float32 mono audio @ 16kHz
            prompt: Optional context prompt
        
        Returns:
            (transcribed_text, metadata)
        """
        _ = self._load_model()  # Eager load before _decode
        
        # Use previous text for context
        ctx_prompt = prompt
        if ctx_prompt is None and self.cfg.condition_on_previous_text:
            ctx_prompt = self._prev_text if self._prev_text else None
        
        # First pass with temperature=0.0
        text, meta = self._decode(
            audio16k,
            temperature=self.cfg.temperature,
            prompt=ctx_prompt
        )
        
        # Fallback with higher temperature if empty
        if not text.strip() and self.cfg.temperature_fallback is not None:
            text2, meta2 = self._decode(
                audio16k,
                temperature=self.cfg.temperature_fallback,
                prompt=ctx_prompt
            )
            if text2.strip():
                text = text2
                meta["fallback"] = meta2
        
        # Update context for next transcription
        if self.cfg.condition_on_previous_text and text.strip():
            self._prev_text = (self._prev_text + " " + text).strip()[-600:]
        
        return text.strip(), meta
    
    def _decode(
        self,
        audio16k: np.ndarray,
        temperature: float,
        prompt: str | None
    ) -> tuple[str, dict[str, Any]]:
        """Run Whisper decode."""
        model = self._load_model()
        
        segments, info = model.transcribe(
            audio16k,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            best_of=self.cfg.best_of,
            temperature=temperature,
            initial_prompt=prompt,
            vad_filter=False,  # We already have VAD
        )
        
        out = []
        seg_meta = []
        for s in segments:
            out.append(s.text)
            seg_meta.append({
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text
            })
        
        metadata = {
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "temperature": temperature,
            "beam_size": self.cfg.beam_size,
            "best_of": self.cfg.best_of,
            "segments": seg_meta,
        }
        
        return "".join(out), metadata
    
    def clear_context(self):
        """Clear rolling context."""
        self._prev_text = ""


class MockWhisperSTT:
    """Mock STT for testing without faster-whisper."""
    
    def __init__(self, cfg: STTConfig | None = None):
        self.cfg = cfg or STTConfig()
        self.call_count = 0
    
    def transcribe(
        self,
        audio16k: np.ndarray,
        prompt: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        self.call_count += 1
        
        # Handle None for easy testing
        if audio16k is None:
            duration = 0.0
        else:
            duration = len(audio16k) / 16000
        return f"[Mock transcript {self.call_count}: {duration:.1f}s]", {
            "mock": True,
            "duration_sec": duration
        }
    
    def clear_context(self):
        pass


def get_stt(
    model_name: str = "large-v3",
    device: str = "cpu",
    language: str = "en",
    mock: bool = False
) -> WhisperSTT:
    """
    Factory to get STT instance.
    
    Args:
        model_name: Whisper model (large-v3, large-v3-turbo, medium, etc.)
        device: cpu/mps/cuda
        language: Language code
        mock: Use mock STT for testing
    
    Returns:
        STT instance
    """
    cfg = STTConfig(model_name=model_name, device=device, language=language)
    if mock:
        return MockWhisperSTT(cfg)
    return WhisperSTT(cfg)
