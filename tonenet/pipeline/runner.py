"""
Full duplex voice agent runner.

Orchestrates: VAD → STT → Agent → TTS with barge-in support.

Features:
- Async pipeline with bounded queues
- Turn state machine (LISTENING/THINKING/SPEAKING)
- Barge-in cancel (TTS stops when user speaks)
- JSONL ledger for replay

Example:
    python -m tonenet.duplex_runner --device cpu --mock
"""

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Awaitable

from ..vad import VADSegmenter, UtteranceSegment, SimulatedVADSegmenter
from ..stt import WhisperCppSTT as WhisperSTT, WhisperCppConfig as STTConfig, MockWhisperCppSTT as MockWhisperSTT
from ..tts import XTTSEngine, TTSConfig, XTTSVoice, MockXTTSEngine
from .ledger import JsonlLedger


@dataclass
class Turn:
    """Single conversation turn."""
    seg: UtteranceSegment
    stt_text: str
    agent_text: str
    ts: float


# Type for agent callback
AgentCallback = Callable[[str], Awaitable[str]]


async def echo_agent(text: str) -> str:
    """Default echo agent - repeats input."""
    return f"You said: {text}"


class DuplexRunner:
    """
    Real-time duplex voice agent orchestrator.
    
    Manages the full pipeline:
    - Mic → VAD (speech detection + endpointing)
    - VAD → STT (accurate transcription)
    - STT → Agent (reasoning)
    - Agent → TTS (speech synthesis)
    
    Supports:
    - Barge-in: TTS stops immediately when user speaks
    - Turn management: Only one speaker at a time
    - Ledger: All events logged for replay
    
    Example:
        async def my_agent(text: str) -> str:
            return openai.chat(text)
        
        runner = DuplexRunner(agent_callback=my_agent)
        await runner.run()
    """
    
    def __init__(
        self,
        *,
        vad: VADSegmenter | None = None,
        stt: WhisperSTT | None = None,
        tts: XTTSEngine | None = None,
        ledger: JsonlLedger | None = None,
        agent_callback: AgentCallback | None = None,
        sentence_pause_ms: int = 200,
        mock: bool = False,
    ):
        """
        Args:
            vad: VAD segmenter (created if None)
            stt: STT engine (created if None)
            tts: TTS engine (created if None)
            ledger: Event ledger (created if None)
            agent_callback: Async function (text -> response)
            sentence_pause_ms: Pause between TTS sentences
            mock: Use mock components for testing
        """
        # Create default components
        if mock:
            self.vad = SimulatedVADSegmenter()
            self.stt = MockWhisperSTT()
            self.tts = MockXTTSEngine()
        else:
            self.vad = vad or VADSegmenter()
            self.stt = stt or WhisperSTT()
            self.tts = tts or XTTSEngine()
        
        self.ledger = ledger or JsonlLedger("duplex_ledger.jsonl")
        self.agent = agent_callback or echo_agent
        self.sentence_pause_ms = sentence_pause_ms
        
        # Async queues
        self._stt_q: asyncio.Queue[UtteranceSegment] = asyncio.Queue(maxsize=8)
        self._stop = asyncio.Event()
        
        # Barge-in signal
        self._barge_in = asyncio.Event()
        
        # Current speaking task
        self._speak_task: asyncio.Task | None = None
        
        # State
        self.state = "LISTENING"
        self.turns: list[Turn] = []
    
    def signal_barge_in(self):
        """Signal that user started speaking (cancels TTS)."""
        self._barge_in.set()
        self.tts.stop()
        self.state = "LISTENING"
    
    async def run(self):
        """
        Run the duplex pipeline.
        
        Blocks until stopped or error.
        """
        # Start workers
        stt_worker = asyncio.create_task(self._stt_worker())
        
        try:
            await self._audio_loop()
        finally:
            self._stop.set()
            stt_worker.cancel()
            try:
                await stt_worker
            except asyncio.CancelledError:
                pass
    
    def stop(self):
        """Stop the pipeline."""
        self._stop.set()
        self.vad.stop()
    
    async def _audio_loop(self):
        """
        Main audio loop: VAD events → queue for STT.
        """
        async for evt in self.vad.events():
            if self._stop.is_set():
                break
            
            if evt["type"] == "speech_start":
                # Barge-in: cancel TTS if speaking
                self.signal_barge_in()
                self.ledger.append({"type": "speech_start"})
            
            elif evt["type"] == "utterance":
                seg: UtteranceSegment = evt["segment"]
                
                # Log segment
                self.ledger.append({
                    "type": "segment",
                    "t0": seg.t0,
                    "t1": seg.t1,
                    "duration_sec": seg.duration_sec,
                    "rms": seg.rms,
                    "vad_stats": seg.vad_stats,
                    "audio_sha256": seg.audio_sha256,
                    "sr": seg.sr,
                })
                
                # Queue for STT
                try:
                    self._stt_q.put_nowait(seg)
                except asyncio.QueueFull:
                    self.ledger.append({
                        "type": "drop",
                        "reason": "stt_queue_full",
                        "audio_sha256": seg.audio_sha256,
                    })
            
            elif evt["type"] == "stop":
                break
    
    async def _stt_worker(self):
        """
        STT worker: transcribe segments → agent → TTS.
        """
        while not self._stop.is_set():
            try:
                seg = await asyncio.wait_for(
                    self._stt_q.get(),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            
            # Clear barge-in for this turn
            self._barge_in.clear()
            
            # Transcribe
            self.state = "THINKING"
            st = time.time()
            stt_text, stt_meta = self.stt.transcribe(seg.audio16k)
            stt_dt = time.time() - st
            
            self.ledger.append({
                "type": "stt",
                "audio_sha256": seg.audio_sha256,
                "text": stt_text,
                "latency_sec": stt_dt,
                "meta": stt_meta,
            })
            
            if not stt_text.strip():
                self.state = "LISTENING"
                continue
            
            # Agent response
            try:
                agent_text = await self.agent(stt_text)
            except Exception as e:
                agent_text = f"Error: {e}"
            
            self.ledger.append({
                "type": "agent",
                "audio_sha256": seg.audio_sha256,
                "stt_text": stt_text,
                "agent_text": agent_text,
            })
            
            # Record turn
            turn = Turn(
                seg=seg,
                stt_text=stt_text,
                agent_text=agent_text,
                ts=time.time()
            )
            self.turns.append(turn)
            
            # Cancel any previous speak task
            if self._speak_task and not self._speak_task.done():
                self.tts.stop()
                self._speak_task.cancel()
                try:
                    await self._speak_task
                except asyncio.CancelledError:
                    pass
            
            # Speak response
            self.state = "SPEAKING"
            self._speak_task = asyncio.create_task(
                self._speak(agent_text, seg.audio_sha256)
            )
    
    async def _speak(self, text: str, audio_sha256: str):
        """
        Speak text with sentence chunking and barge-in support.
        """
        chunks = self.tts.split_sentences(text)
        
        self.ledger.append({
            "type": "tts_plan",
            "audio_sha256": audio_sha256,
            "chunks": chunks,
        })
        
        for i, chunk in enumerate(chunks):
            # Check for barge-in
            if self._barge_in.is_set():
                self.ledger.append({
                    "type": "tts_abort",
                    "audio_sha256": audio_sha256,
                    "reason": "barge_in",
                    "chunk_index": i,
                })
                return
            
            # Synthesize and play
            audio, meta = self.tts.synthesize(chunk)
            
            self.ledger.append({
                "type": "tts_chunk",
                "audio_sha256": audio_sha256,
                "chunk_index": i,
                "text": chunk,
                "audio_samples": len(audio),
            })
            
            self.tts.play(audio)
            
            # Pause between sentences
            await asyncio.sleep(self.sentence_pause_ms / 1000.0)
        
        self.state = "LISTENING"


async def run_duplex(
    agent_callback: AgentCallback | None = None,
    device: str = "cpu",
    language: str = "en",
    voice_wav: str | None = None,
    ledger_path: str = "duplex_ledger.jsonl",
    mock: bool = False,
):
    """
    Convenience function to run duplex agent.
    
    Args:
        agent_callback: Async function (text -> response)
        device: cpu/mps/cuda
        language: Language code
        voice_wav: Path to voice cloning reference
        ledger_path: Path to JSONL ledger
        mock: Use mock components
    """
    ledger = JsonlLedger(ledger_path)
    
    if mock:
        runner = DuplexRunner(
            ledger=ledger,
            agent_callback=agent_callback,
            mock=True,
        )
    else:
        vad = VADSegmenter(device=device)
        stt = WhisperSTT(STTConfig(device=device, language=language))
        
        voice = XTTSVoice(reference_wav=voice_wav) if voice_wav else None
        tts = XTTSEngine(TTSConfig(device=device, voice=voice))
        
        runner = DuplexRunner(
            vad=vad,
            stt=stt,
            tts=tts,
            ledger=ledger,
            agent_callback=agent_callback,
        )
    
    print(f"Starting duplex agent (device={device}, mock={mock})...")
    print("Press Ctrl+C to stop\n")
    
    try:
        await runner.run()
    except KeyboardInterrupt:
        runner.stop()
        print("\nStopped.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="ToneNet Duplex Voice Agent")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--language", default="en")
    parser.add_argument("--voice-wav", default=None, help="Voice cloning reference")
    parser.add_argument("--ledger", default="duplex_ledger.jsonl")
    parser.add_argument("--mock", action="store_true", help="Use mock components")
    args = parser.parse_args()
    
    asyncio.run(run_duplex(
        device=args.device,
        language=args.language,
        voice_wav=args.voice_wav,
        ledger_path=args.ledger,
        mock=args.mock,
    ))


if __name__ == "__main__":
    main()
