"""
Real-time voice agent with VAD, streaming TTS, and barge-in.

This is the recommended pipeline for accuracy-first duplex voice agents.

Pipeline:
    Mic → VAD → STT (on utterance end) → LLM → Chunked TTS → Speaker
            ↓                                        ↑
        Barge-in (stop TTS when user speaks) ────────┘

Features:
- VAD gates STT calls (no transcribing fan noise)
- Sentence chunking starts TTS early
- Barge-in interrupts TTS playback when user speaks
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import numpy as np

from .vad_webrtc import WebRTCVADGate, WebRTCVADConfig, MockWebRTCVAD
from .text_chunker import sentence_chunks
from .audio_player import AudioPlayer


# Type aliases
STTProtocol = Callable[[np.ndarray, int], str]  # (audio, sr) -> text
TTSProtocol = Callable[[str], tuple[np.ndarray, int]]  # text -> (audio, sr)
LLMProtocol = Callable[[str], str | Iterable[str]]  # text -> response or stream


@dataclass
class RealtimeAgentConfig:
    """Real-time voice agent configuration."""
    vad: WebRTCVADConfig = field(default_factory=WebRTCVADConfig)
    max_utt_sec: float = 18.0    # Hard cap on utterance length
    min_utt_sec: float = 0.25    # Ignore tiny noises
    tts_sample_rate: int = 24000


class RealtimeVoiceAgent:
    """
    Accuracy-first real-time voice agent.
    
    Key features:
    - VAD gates STT calls (reduces noise, cuts latency)
    - Sentence chunking starts TTS on first sentence
    - Barge-in stops TTS when user speaks
    
    Example:
        def my_stt(audio, sr):
            return whisper.transcribe(audio)
        
        def my_tts(text):
            return edge_tts.synthesize(text), 24000
        
        def my_llm(text):
            return openai.chat(text)
        
        agent = RealtimeVoiceAgent(
            stt=my_stt,
            tts=my_tts,
            llm_respond=my_llm,
            cfg=RealtimeAgentConfig()
        )
        agent.process_loop(mic_stream)
    """
    
    def __init__(
        self,
        stt: STTProtocol,
        tts: TTSProtocol,
        llm_respond: LLMProtocol,
        cfg: RealtimeAgentConfig | None = None,
        mock_vad: bool = False,
    ):
        """
        Args:
            stt: Speech-to-text function (audio, sr) -> text
            tts: Text-to-speech function (text) -> (audio, sr)
            llm_respond: LLM function (text) -> response string or stream
            cfg: Agent configuration
            mock_vad: Use mock VAD for testing
        """
        self.stt = stt
        self.tts = tts
        self.llm_respond = llm_respond
        self.cfg = cfg or RealtimeAgentConfig()
        self.mock_vad = mock_vad
        
        self.player = AudioPlayer(sample_rate=self.cfg.tts_sample_rate)
        
        self._barge_in = threading.Event()
        self._stop = threading.Event()
        
        # Stats
        self.utterances_processed = 0
        self.barge_ins = 0
    
    def stop(self):
        """Stop the agent."""
        self._stop.set()
        self.player.stop()
    
    def _run_tts_stream(self, text_stream: Iterable[str]) -> None:
        """Run TTS sentence-by-sentence with barge-in support."""
        for chunk in sentence_chunks(text_stream):
            if self._barge_in.is_set():
                break
            
            audio, sr = self.tts(chunk)
            
            if self._barge_in.is_set():
                break
            
            self.player.play_blocking(audio, sr)
    
    def process_loop(
        self,
        mic_stream: Iterable[tuple[np.ndarray, int]],
        on_utterance: Callable[[str, str], None] | None = None
    ) -> None:
        """
        Main processing loop.
        
        Args:
            mic_stream: Iterable of (audio_chunk, sample_rate) tuples
            on_utterance: Optional callback (user_text, agent_text) for logging
        """
        gate: WebRTCVADGate | MockWebRTCVAD | None = None
        utt_audio: list[np.ndarray] = []
        utt_t0: float | None = None
        mic_sr: int | None = None
        
        tts_thread: threading.Thread | None = None
        
        for audio, sr in mic_stream:
            if self._stop.is_set():
                break
            
            # Initialize VAD on first chunk
            if gate is None:
                mic_sr = int(sr)
                if self.mock_vad:
                    gate = MockWebRTCVAD(self.cfg.vad, input_sample_rate=mic_sr)
                else:
                    gate = WebRTCVADGate(self.cfg.vad, input_sample_rate=mic_sr)
            
            sr = int(sr)
            
            started, ended, active = gate.push(audio)
            
            # Barge-in: user speaks while TTS is playing
            if started:
                self._barge_in.set()
                self.player.stop()
                self.barge_ins += 1
            
            # Collect utterance audio
            if active:
                if utt_t0 is None:
                    utt_t0 = time.time()
                    utt_audio = []
                utt_audio.append(np.asarray(audio, dtype=np.float32))
                
                # Hard cap on utterance length
                if utt_t0 and (time.time() - utt_t0) > self.cfg.max_utt_sec:
                    ended = True
            
            # Process completed utterance
            if ended:
                self._barge_in.clear()
                
                if utt_t0 is None:
                    gate.reset()
                    continue
                
                dur = time.time() - utt_t0
                utt_t0 = None
                
                # Skip short noises
                if dur < self.cfg.min_utt_sec:
                    gate.reset()
                    continue
                
                # Concatenate utterance audio
                if utt_audio:
                    full = np.concatenate(utt_audio, axis=0)
                else:
                    full = np.zeros((0,), dtype=np.float32)
                utt_audio = []
                
                # STT
                text = self.stt(full, sr)
                text = (text or "").strip()
                if not text:
                    gate.reset()
                    continue
                
                # LLM response
                resp = self.llm_respond(text)
                
                # Normalize to stream
                if isinstance(resp, str):
                    def _one_shot() -> Iterator[str]:
                        yield resp
                    text_stream: Iterable[str] = _one_shot()
                else:
                    text_stream = resp
                
                # Collect response for callback
                if on_utterance:
                    collected = list(text_stream)
                    full_response = "".join(collected)
                    on_utterance(text, full_response)
                    text_stream = iter(collected)
                
                # Spawn TTS thread (allows mic loop to continue for barge-in)
                tts_thread = threading.Thread(
                    target=self._run_tts_stream,
                    args=(text_stream,),
                    daemon=True
                )
                tts_thread.start()
                
                self.utterances_processed += 1
                gate.reset()


def create_realtime_agent(
    stt: STTProtocol,
    tts: TTSProtocol,
    llm_respond: LLMProtocol,
    mock: bool = False
) -> RealtimeVoiceAgent:
    """
    Factory for creating a real-time voice agent.
    
    Args:
        stt: STT function
        tts: TTS function
        llm_respond: LLM function
        mock: Use mock VAD for testing
    
    Returns:
        Configured agent
    """
    return RealtimeVoiceAgent(
        stt=stt,
        tts=tts,
        llm_respond=llm_respond,
        mock_vad=mock
    )
