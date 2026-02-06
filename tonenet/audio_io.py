"""
Non-blocking audio I/O for Mac M2.

Uses sounddevice (PortAudio) which works reliably on macOS.
Queue-based architecture prevents mic starvation.

Install: pip install sounddevice soundfile
"""

import queue
import threading
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class AudioIOConfig:
    """Audio I/O configuration."""
    sample_rate: int = 16000
    block_ms: int = 20
    channels: int = 1
    dtype: str = "float32"
    max_queue_size: int = 200
    device: int | str | None = None  # None = default device


class MicStream:
    """
    Non-blocking microphone stream with queue-based buffering.
    
    Works reliably on macOS (uses sounddevice/PortAudio).
    Chunks are pushed to a queue so the audio callback never blocks.
    
    Example:
        mic = MicStream()
        for chunk in mic:
            process(chunk)  # float32 @ 16kHz
    """
    
    def __init__(self, cfg: AudioIOConfig | None = None):
        self.cfg = cfg or AudioIOConfig()
        self.blocksize = int(self.cfg.sample_rate * self.cfg.block_ms / 1000)
        self.q: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.cfg.max_queue_size)
        self._stop = threading.Event()
    
    def _callback(self, indata, frames, time, status):
        """Audio callback - runs in separate thread."""
        if status:
            # Drop on overflow to keep pipeline alive
            return
        
        # Extract mono float32
        x = indata[:, 0].astype(np.float32, copy=False)
        
        try:
            self.q.put_nowait(x)
        except queue.Full:
            # Drop oldest to prevent memory growth
            try:
                self.q.get_nowait()
                self.q.put_nowait(x)
            except queue.Empty:
                pass
    
    def stop(self):
        """Stop the stream."""
        self._stop.set()
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over audio chunks."""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("sounddevice not installed. Run: pip install sounddevice")
        
        with sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype=self.cfg.dtype,
            blocksize=self.blocksize,
            callback=self._callback,
            device=self.cfg.device,
        ):
            while not self._stop.is_set():
                try:
                    yield self.q.get(timeout=0.1)
                except queue.Empty:
                    continue


class AudioOutput:
    """
    Audio output with interrupt support.
    
    Example:
        out = AudioOutput()
        out.play(audio, sample_rate)
        out.stop()  # Interrupt playback
    """
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._playing = False
    
    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing
    
    def stop(self):
        """Stop playback immediately."""
        self._stop.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
    
    def play(self, audio: np.ndarray, sample_rate: int | None = None, blocking: bool = True):
        """
        Play audio.
        
        Args:
            audio: float32 audio array
            sample_rate: Sample rate (uses default if None)
            blocking: Wait for playback to complete
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("sounddevice not installed. Run: pip install sounddevice")
        
        self._stop.clear()
        
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio[:, 0]
        
        sr = sample_rate or self.sample_rate
        
        with self._lock:
            self._playing = True
        
        try:
            sd.play(audio, sr, blocking=False)
            
            if blocking:
                while sd.get_stream().active:
                    if self._stop.is_set():
                        sd.stop()
                        break
                    sd.sleep(10)
        finally:
            with self._lock:
                self._playing = False


class MockMicStream:
    """Mock mic stream for testing."""
    
    def __init__(self, cfg: AudioIOConfig | None = None):
        self.cfg = cfg or AudioIOConfig()
        self._count = 0
        self._max = 50
    
    def stop(self):
        self._max = 0
    
    def __iter__(self) -> Iterator[np.ndarray]:
        blocksize = int(self.cfg.sample_rate * self.cfg.block_ms / 1000)
        while self._count < self._max:
            self._count += 1
            yield np.zeros(blocksize, dtype=np.float32)
