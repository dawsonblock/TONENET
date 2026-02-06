"""
Audio player with barge-in (interrupt) support.

Plays audio while allowing immediate stop when user starts speaking.

Install: pip install sounddevice
"""

import threading
from typing import Callable

import numpy as np


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


class AudioPlayer:
    """
    Audio player with barge-in support.
    
    Allows immediate stop of playback when user starts speaking,
    essential for natural duplex conversation.
    
    Example:
        player = AudioPlayer()
        
        # In TTS thread
        player.play_blocking(audio, 24000)
        
        # In VAD thread (when speech detected)
        player.stop()  # Immediate stop
    """
    
    def __init__(self, sample_rate: int = 24000):
        """
        Args:
            sample_rate: Output sample rate
        """
        self.sample_rate = int(sample_rate)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._playing = False
        self._on_stop: Callable[[], None] | None = None
    
    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        with self._lock:
            return self._playing
    
    def set_on_stop(self, callback: Callable[[], None] | None):
        """Set callback for when playback is stopped."""
        self._on_stop = callback
    
    def stop(self) -> None:
        """
        Stop playback immediately (for barge-in).
        
        Safe to call from any thread.
        """
        self._stop.set()
        with self._lock:
            if self._playing:
                try:
                    import sounddevice as sd
                    sd.stop()
                except Exception:
                    pass
        
        if self._on_stop:
            try:
                self._on_stop()
            except Exception:
                pass
    
    def play_blocking(
        self,
        audio_f32: np.ndarray,
        sample_rate: int | None = None
    ) -> bool:
        """
        Play audio, blocking until done or stopped.
        
        Args:
            audio_f32: float32 audio array
            sample_rate: Sample rate (uses player's rate if None)
        
        Returns:
            True if played completely, False if stopped early
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("sounddevice not installed. Run: pip install sounddevice")
        
        self._stop.clear()
        
        x = np.asarray(audio_f32, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        
        # Resample if needed
        sr_in = sample_rate or self.sample_rate
        if int(sr_in) != self.sample_rate and x.size:
            x = _resample_linear(x, int(sr_in), self.sample_rate)
        
        if x.size == 0:
            return True
        
        with self._lock:
            self._playing = True
        
        completed = True
        try:
            sd.play(x, self.sample_rate, blocking=False)
            while sd.get_stream().active:
                if self._stop.is_set():
                    sd.stop()
                    completed = False
                    break
                sd.sleep(10)
        finally:
            with self._lock:
                self._playing = False
        
        return completed
    
    def play_async(
        self,
        audio_f32: np.ndarray,
        sample_rate: int | None = None,
        on_complete: Callable[[], None] | None = None
    ) -> threading.Thread:
        """
        Play audio in background thread.
        
        Args:
            audio_f32: Audio to play
            sample_rate: Sample rate
            on_complete: Callback when playback finishes
        
        Returns:
            Thread handle
        """
        def _play():
            self.play_blocking(audio_f32, sample_rate)
            if on_complete:
                try:
                    on_complete()
                except Exception:
                    pass
        
        t = threading.Thread(target=_play, daemon=True)
        t.start()
        return t
