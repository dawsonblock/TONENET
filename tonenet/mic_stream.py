"""
Microphone streaming adapter for ToneNet.

Provides real-time audio capture chunked for the codec's expected format.
"""

import queue
import threading
from typing import Iterator, Optional
import torch


class MicStream:
    """
    Real-time microphone stream for codec input.
    
    Captures audio in chunks compatible with ToneNet's 24kHz sample rate.
    
    Example:
        mic = MicStream()
        for chunk in mic.stream():
            codes = codec.encode(chunk)
            # process codes
    
    Requires: pip install sounddevice
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        chunk_duration_ms: int = 100,
        device: Optional[int] = None,
        channels: int = 1
    ):
        """
        Args:
            sample_rate: Sample rate (24000 for ToneNet)
            chunk_duration_ms: Chunk duration in milliseconds
            device: Audio device ID (None for default)
            channels: Number of channels (1 for mono)
        """
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self.device = device
        self.channels = channels
        
        self._running = False
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stream = None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice."""
        if status:
            pass  # Could log status
        # Convert to tensor format [1, 1, samples]
        audio = torch.from_numpy(indata[:, 0].copy()).float()
        audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        self._queue.put(audio)
    
    def start(self):
        """Start audio capture."""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("pip install sounddevice")
        
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_samples,
            device=self.device,
            channels=self.channels,
            callback=self._audio_callback
        )
        self._stream.start()
    
    def stop(self):
        """Stop audio capture."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
    
    def get_chunk(self, timeout: float = 1.0) -> Optional[torch.Tensor]:
        """
        Get next audio chunk.
        
        Returns:
            Audio tensor [1, 1, T] or None if timeout
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stream(self, max_chunks: Optional[int] = None) -> Iterator[torch.Tensor]:
        """
        Generator for continuous audio chunks.
        
        Args:
            max_chunks: Maximum chunks to yield (None for infinite)
        
        Yields:
            Audio tensors [1, 1, T]
        """
        self.start()
        try:
            count = 0
            while self._running:
                chunk = self.get_chunk()
                if chunk is not None:
                    yield chunk
                    count += 1
                    if max_chunks and count >= max_chunks:
                        break
        finally:
            self.stop()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class SimulatedMicStream:
    """
    Simulated microphone for testing (no sounddevice required).
    
    Generates random noise or plays from a file.
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        chunk_duration_ms: int = 100,
        audio_file: Optional[str] = None
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self.audio_file = audio_file
        self._audio_data: Optional[torch.Tensor] = None
        self._position = 0
        
        if audio_file:
            self._load_file(audio_file)
    
    def _load_file(self, path: str):
        """Load audio file."""
        try:
            import torchaudio
            audio, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            self._audio_data = audio.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, T]
        except Exception:
            self._audio_data = None
    
    def get_chunk(self) -> torch.Tensor:
        """Get next audio chunk."""
        if self._audio_data is not None:
            # From file
            end = self._position + self.chunk_samples
            if end > self._audio_data.shape[-1]:
                self._position = 0
                end = self.chunk_samples
            chunk = self._audio_data[..., self._position:end]
            self._position = end
            return chunk
        else:
            # Random noise
            return torch.randn(1, 1, self.chunk_samples) * 0.1
    
    def stream(self, max_chunks: int = 100) -> Iterator[torch.Tensor]:
        """Generate audio chunks."""
        for _ in range(max_chunks):
            yield self.get_chunk()
