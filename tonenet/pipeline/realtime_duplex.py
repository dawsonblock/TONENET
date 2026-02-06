"""
Real-time duplex voice agent with threaded queue architecture.

Pipeline:
    Mic → VAD → STT → Reasoner → TTS → Speaker
     ↑                                    ↓
     └──────── Barge-in interrupt ────────┘

Features:
- Non-blocking queue-based architecture
- Barge-in (TTS stops when user speaks)
- Configurable backends
- Thread isolation

Usage:
    python -m tonenet.realtime_duplex --mock
"""

import argparse
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from .audio_io import MicStream, AudioOutput, AudioIOConfig, MockMicStream, MockAudioOutput
from ..vad import VADSegmenter, SimulatedVADSegmenter
from ..stt import create_stt, STTBackendConfig, MockSTTBackend
from ..tts import create_tts, TTSBackendConfig, MockTTSBackend
from ..agent import create_reasoner, ReasonerConfig, EchoReasoner


@dataclass
class DuplexConfig:
    """Full duplex configuration."""
    sample_rate: int = 16000
    max_utterance_sec: float = 18.0
    min_utterance_sec: float = 0.25
    
    # Queue sizes
    audio_queue_size: int = 400
    utterance_queue_size: int = 20
    text_queue_size: int = 20


class RealtimeDuplex:
    """
    Production-grade realtime duplex voice agent.
    
    Uses threaded queues to prevent any component from
    blocking the mic capture loop.
    
    Example:
        duplex = RealtimeDuplex(
            stt=create_stt(mock=True),
            tts=create_tts(mock=True),
            reasoner=create_reasoner()
        )
        duplex.run()
    """
    
    def __init__(
        self,
        stt=None,
        tts=None,
        vad=None,
        reasoner=None,
        audio_output=None,
        cfg: DuplexConfig | None = None,
        mock: bool = False,
    ):
        self.cfg = cfg or DuplexConfig()
        
        # Components
        if mock:
            self.stt = MockSTTBackend()
            self.tts = MockTTSBackend()
            self.vad = SimulatedVADSegmenter()
            self.reasoner = EchoReasoner()
        else:
            self.stt = stt or create_stt()
            self.tts = tts or create_tts()
            self.vad = vad or VADSegmenter()
            self.reasoner = reasoner or create_reasoner()
        
        if audio_output:
            self.audio_out = audio_output
        elif mock:
            self.audio_out = MockAudioOutput()
        else:
            self.audio_out = AudioOutput()
        self.mock = mock
        
        # Queues
        self.q_audio: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=self.cfg.audio_queue_size
        )
        self.q_utt: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=self.cfg.utterance_queue_size
        )
        self.q_text: queue.Queue[str] = queue.Queue(
            maxsize=self.cfg.text_queue_size
        )
        
        # State
        self._stop = threading.Event()
        self._barge_in = threading.Event()
        
        # Stats
        self.utterances_processed = 0
        self.barge_ins = 0
    
    def stop(self):
        """Stop all threads."""
        self._stop.set()
        self.audio_out.stop()
    
    def _mic_thread(self, mic_stream):
        """Capture mic audio and push to queue."""
        for chunk in mic_stream:
            if self._stop.is_set():
                return
            try:
                self.q_audio.put_nowait(chunk)
            except queue.Full:
                # Drop oldest
                try:
                    self.q_audio.get_nowait()
                    self.q_audio.put_nowait(chunk)
                except queue.Empty:
                    pass
    
    def _vad_thread(self):
        """Process audio chunks through VAD, emit utterances."""
        buf = []
        last_emit = time.time()
        
        while not self._stop.is_set():
            try:
                chunk = self.q_audio.get(timeout=0.1)
            except queue.Empty:
                continue
            
            buf.append(chunk)
            audio = np.concatenate(buf)
            
            # Check for utterance
            utt = self.vad.process(audio)
            
            if utt is not None and len(utt) > 0:
                # Barge-in: stop TTS if playing
                if self.audio_out.is_playing:
                    self._barge_in.set()
                    self.audio_out.stop()
                    self.barge_ins += 1
                
                # Check duration
                dur = len(utt) / self.cfg.sample_rate
                if dur >= self.cfg.min_utterance_sec:
                    try:
                        self.q_utt.put_nowait(utt)
                    except queue.Full:
                        pass
                
                buf = []
                last_emit = time.time()
            
            # Hard reset for long silence
            if time.time() - last_emit > 12 and len(audio) > self.cfg.sample_rate * 15:
                buf = []
                last_emit = time.time()
    
    def _stt_thread(self):
        """Transcribe utterances to text."""
        while not self._stop.is_set():
            try:
                utt = self.q_utt.get(timeout=0.1)
            except queue.Empty:
                continue
            
            text = self.stt.transcribe(utt, self.cfg.sample_rate)
            text = (text or "").strip()
            
            if text:
                try:
                    self.q_text.put_nowait(text)
                except queue.Full:
                    pass
    
    def _tts_thread(self):
        """Generate and play TTS responses."""
        while not self._stop.is_set():
            try:
                text = self.q_text.get(timeout=0.1)
            except queue.Empty:
                continue
            
            self._barge_in.clear()
            
            # Get response
            reply = self.reasoner.respond(text)
            if not reply:
                continue
            
            # Synthesize
            audio, sr = self.tts.synthesize(reply)
            
            if self._barge_in.is_set():
                continue
            
            # Play
            self.audio_out.play(audio, sr, blocking=True)
            self.utterances_processed += 1
    
    def run(self, mic_stream=None):
        """
        Run the duplex pipeline.
        
        Args:
            mic_stream: Optional custom mic stream
        """
        # Create mic stream
        if mic_stream is None:
            if self.mock:
                mic_stream = MockMicStream(AudioIOConfig(sample_rate=self.cfg.sample_rate))
            else:
                mic_stream = MicStream(AudioIOConfig(sample_rate=self.cfg.sample_rate))
        
        # Start threads
        threads = [
            threading.Thread(target=self._mic_thread, args=(mic_stream,), daemon=True),
            threading.Thread(target=self._vad_thread, daemon=True),
            threading.Thread(target=self._stt_thread, daemon=True),
            threading.Thread(target=self._tts_thread, daemon=True),
        ]
        
        for t in threads:
            t.start()
        
        print("RealtimeDuplex running. Press Ctrl+C to stop.")
        
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()
            print("\nStopped.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="ToneNet Realtime Duplex Voice Agent")
    parser.add_argument("--mock", action="store_true", help="Use mock components")
    parser.add_argument("--stt-model", default="large-v3-turbo", help="STT model")
    parser.add_argument("--stt-compute", default="int8", help="STT compute type")
    parser.add_argument("--tts-engine", default="piper", help="TTS engine")
    args = parser.parse_args()
    
    if args.mock:
        duplex = RealtimeDuplex(mock=True)
    else:
        stt_cfg = STTBackendConfig(model=args.stt_model, compute=args.stt_compute)
        tts_cfg = TTSBackendConfig(engine=args.tts_engine)
        
        duplex = RealtimeDuplex(
            stt=create_stt(stt_cfg),
            tts=create_tts(tts_cfg),
        )
    
    duplex.run()


if __name__ == "__main__":
    main()
