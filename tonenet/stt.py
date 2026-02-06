"""
Speech-to-Text (STT) integration using faster-whisper.

Provides streaming and batch transcription for the duplex agent.

Install: pip install faster-whisper
"""

import torch
from typing import Iterator
from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    """Single transcript segment with timing."""
    text: str
    start: float
    end: float
    confidence: float


class StreamingSTT:
    """
    Speech-to-text using faster-whisper.
    
    Supports both batch and streaming transcription.
    
    Example:
        stt = StreamingSTT(model_size="base")
        text = stt.transcribe(audio_tensor)
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "float16",
        language: str = "en"
    ):
        """
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2)
            device: "cuda", "cpu", or "auto"
            compute_type: "float16", "int8", "float32"
            language: Language code (e.g., "en", "es", "fr")
        """
        self.model_size = model_size
        self.language = language
        self._model = None
        self._device = device
        self._compute_type = compute_type
        
        # Buffer for streaming
        self._buffer: list[torch.Tensor] = []
        self._sample_rate = 16000  # Whisper expects 16kHz
    
    def _load_model(self):
        """Lazy load the whisper model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                
                device = self._device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self._model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=self._compute_type if device == "cuda" else "float32"
                )
            except ImportError:
                raise ImportError(
                    "faster-whisper not installed. Run: pip install faster-whisper"
                )
        return self._model
    
    def transcribe(
        self,
        audio: torch.Tensor,
        return_segments: bool = False
    ) -> str | list[TranscriptSegment]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio tensor [1, T] or [T] at any sample rate
            return_segments: If True, return list of segments with timing
        
        Returns:
            Transcribed text or list of segments
        """
        model = self._load_model()
        
        # Prepare audio
        if audio.dim() > 1:
            audio = audio.squeeze()
        audio_np = audio.cpu().numpy()
        
        # Transcribe
        segments, info = model.transcribe(
            audio_np,
            language=self.language,
            beam_size=5,
            vad_filter=True
        )
        
        if return_segments:
            result = []
            for seg in segments:
                result.append(TranscriptSegment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    confidence=seg.avg_logprob
                ))
            return result
        else:
            return " ".join(seg.text.strip() for seg in segments)
    
    def transcribe_file(self, path: str) -> str:
        """Transcribe audio file."""
        model = self._load_model()
        segments, _ = model.transcribe(path, language=self.language)
        return " ".join(seg.text.strip() for seg in segments)
    
    def push_audio(self, chunk: torch.Tensor):
        """Push audio chunk to streaming buffer."""
        if chunk.dim() > 1:
            chunk = chunk.squeeze()
        self._buffer.append(chunk)
    
    def get_transcript(self, clear_buffer: bool = True) -> str:
        """
        Transcribe buffered audio.
        
        Args:
            clear_buffer: Clear buffer after transcription
        
        Returns:
            Transcribed text
        """
        if not self._buffer:
            return ""
        
        audio = torch.cat(self._buffer, dim=0)
        text = self.transcribe(audio)
        
        if clear_buffer:
            self._buffer.clear()
        
        return text
    
    def stream_transcribe(
        self,
        audio_iterator: Iterator[torch.Tensor],
        chunk_duration_sec: float = 3.0
    ) -> Iterator[str]:
        """
        Stream transcription from audio iterator.
        
        Yields partial transcripts as audio comes in.
        
        Args:
            audio_iterator: Iterator yielding audio chunks
            chunk_duration_sec: How often to transcribe
        
        Yields:
            Transcribed text chunks
        """
        buffer = []
        buffer_samples = 0
        samples_per_chunk = int(chunk_duration_sec * self._sample_rate)
        
        for chunk in audio_iterator:
            if chunk.dim() > 1:
                chunk = chunk.squeeze()
            buffer.append(chunk)
            buffer_samples += chunk.shape[0]
            
            if buffer_samples >= samples_per_chunk:
                audio = torch.cat(buffer, dim=0)
                text = self.transcribe(audio)
                if text.strip():
                    yield text
                buffer.clear()
                buffer_samples = 0
        
        # Final chunk
        if buffer:
            audio = torch.cat(buffer, dim=0)
            text = self.transcribe(audio)
            if text.strip():
                yield text


class MockSTT:
    """
    Mock STT for testing without faster-whisper installed.
    """
    
    def __init__(self, **kwargs):
        self.call_count = 0
    
    def transcribe(self, audio: torch.Tensor, **kwargs) -> str:
        self.call_count += 1
        return f"[Mock transcript {self.call_count}]"
    
    def transcribe_file(self, path: str) -> str:
        return f"[Mock transcript for {path}]"


def get_stt(model_size: str = "base", mock: bool = False) -> StreamingSTT:
    """
    Factory function to get STT instance.
    
    Args:
        model_size: Whisper model size
        mock: If True, return mock STT (for testing)
    
    Returns:
        STT instance
    """
    if mock:
        return MockSTT()
    return StreamingSTT(model_size=model_size)
