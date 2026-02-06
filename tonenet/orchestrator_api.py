"""
Audio Orchestrator API for duplex voice agent.

This is the main interface matching the expected snippet API:
- AudioOrchestrator.encode(audio) -> codes
- AudioOrchestrator.emit_text(text, speaker_id) -> audio
- AudioOrchestrator.transcribe(audio) -> text
- AudioOrchestrator.duplex_step(audio_in) -> audio_out

Integrates: codec + STT + TTS + watermark + replay
"""

import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import torch

from .core import ToneNetCodec
from .streaming import StreamingToneNet
from .watermark import embed_watermark
from .replay import save_trace
from .core.tokens import pack_codes, normalize_codes


class AudioOrchestrator:
    """
    Main orchestrator for duplex voice agent.
    
    Provides the complete API for:
    - Speech recognition (STT)
    - Speech synthesis (TTS)
    - Codec encode/decode
    - Watermarking and replay
    
    Example:
        orch = AudioOrchestrator()
        
        # Transcribe incoming audio
        text = orch.transcribe(audio)
        
        # Generate response audio
        response = orch.emit_text("Hello!", speaker_id="operator")
        
        # Or run full duplex step
        response = orch.duplex_step(audio, reasoner=my_llm)
    """
    
    def __init__(
        self,
        device: str = "cpu",
        seed: int = 0,
        stt_model: str = "base",
        tts_voice: str = "guy",
        use_mock_stt: bool = False,
        use_mock_tts: bool = False,
        watermark_strength: float = 1e-3,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            device: "cpu" or "cuda"
            seed: Random seed for determinism
            stt_model: Whisper model size (tiny, base, small, medium, large-v2)
            tts_voice: TTS voice name
            use_mock_stt: Use mock STT (no faster-whisper required)
            use_mock_tts: Use mock TTS (no edge-tts required)
            watermark_strength: Watermark embedding strength
            config: Additional configuration
        """
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        
        config = config or {}
        
        # Core codec
        self.codec = ToneNetCodec().to(device).eval()
        
        # Streaming decoder
        self.streamer = StreamingToneNet(
            device=device,
            seed=seed,
            chunk_frames=config.get("chunk_frames", 5)
        )
        
        # STT
        from .stt import create_stt, MockSTTBackend
        if use_mock_stt:
            self.stt = MockSTTBackend()
        else:
            self.stt = create_stt()
        
        # TTS
        from .tts import create_tts, MockTTSBackend
        if use_mock_tts:
            self.tts = MockTTSBackend()
        else:
            self.tts = create_tts()
        
        # Watermark
        self.watermark_strength = watermark_strength
        
        # Event log
        self._event_log = []
    
    # =========================================
    # Core codec operations
    # =========================================
    
    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to codec tokens.
        
        Args:
            audio: Audio tensor [1, 1, T] or [1, T] or [T]
        
        Returns:
            Packed codes tensor [1, Q, T_frames]
        """
        # Normalize audio shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        codes = self.codec.encode(audio.to(self.device))
        return pack_codes(codes)
    
    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode codec tokens to audio.
        
        Args:
            codes: Packed codes [B, Q, T] or list format
        
        Returns:
            Audio tensor [1, 1, T]
        """
        codes = normalize_codes(codes)
        return self.codec.decode(codes)
    
    # =========================================
    # STT operations
    # =========================================
    
    def transcribe(self, audio: torch.Tensor) -> str:
        """
        Transcribe audio to text using STT.
        
        Args:
            audio: Audio tensor
        
        Returns:
            Transcribed text
        """
        text = self.stt.transcribe(audio)
        
        self._log_event("transcribe", {
            "audio_samples": audio.numel(),
            "text_length": len(text)
        })
        
        return text
    
    # =========================================
    # TTS operations
    # =========================================
    
    def emit_text(
        self,
        text: str,
        speaker_id: str = "default",
        apply_watermark: bool = True
    ) -> torch.Tensor:
        """
        Synthesize text to audio using TTS.
        
        Args:
            text: Text to speak
            speaker_id: Speaker identifier for watermarking
            apply_watermark: Whether to embed watermark
        
        Returns:
            Audio tensor [1, 1, T]
        """
        # TTS synthesis
        audio = self.tts.speak(text)
        
        # Apply watermark
        if apply_watermark:
            audio = embed_watermark(audio, speaker_id, self.watermark_strength)
        
        self._log_event("emit_text", {
            "text": text[:100],
            "speaker_id": speaker_id,
            "audio_samples": audio.numel()
        })
        
        return audio
    
    def emit_tokens(
        self,
        tokens: torch.Tensor,
        speaker_id: str = "default",
        apply_watermark: bool = True
    ) -> torch.Tensor:
        """
        Decode tokens to audio (legacy API).
        
        Args:
            tokens: Codec tokens
            speaker_id: Speaker identifier
            apply_watermark: Whether to embed watermark
        
        Returns:
            Audio tensor
        """
        audio = self.decode(tokens)
        
        if apply_watermark:
            audio = embed_watermark(audio, speaker_id, self.watermark_strength)
        
        return audio
    
    # =========================================
    # Duplex operations
    # =========================================
    
    def duplex_step(
        self,
        audio_in: torch.Tensor,
        reasoner: Optional[Callable[[str], str]] = None,
        speaker_id: str = "operator"
    ) -> tuple:
        """
        Full duplex step: audio in -> transcribe -> reason -> speak.
        
        Args:
            audio_in: Incoming audio tensor
            reasoner: Function that takes text and returns response text
                      If None, uses echo (repeats input)
            speaker_id: Speaker ID for output
        
        Returns:
            (transcript, response_text, audio_out)
        """
        # Step 1: Transcribe
        transcript = self.transcribe(audio_in)
        
        if not transcript.strip():
            return transcript, "", None
        
        # Step 2: Reason (or echo)
        if reasoner is not None:
            response_text = reasoner(transcript)
        else:
            response_text = f"I heard: {transcript}"
        
        # Step 3: Synthesize
        audio_out = self.emit_text(response_text, speaker_id=speaker_id)
        
        self._log_event("duplex_step", {
            "transcript": transcript[:100],
            "response": response_text[:100]
        })
        
        return transcript, response_text, audio_out
    
    # =========================================
    # Event logging
    # =========================================
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log event for replay/debug."""
        self._event_log.append({
            "ts": time.time(),
            "type": event_type,
            "seed": self.seed,
            **data
        })
    
    def get_event_log(self):
        """Get all logged events."""
        return self._event_log.copy()
    
    def save_trace(self, path: str):
        """Save event log to file."""
        import json
        with open(path, "w") as f:
            json.dump(self._event_log, f, indent=2)


# Convenience function for quick access
def create_orchestrator(**kwargs) -> AudioOrchestrator:
    """Create orchestrator with defaults."""
    return AudioOrchestrator(**kwargs)
