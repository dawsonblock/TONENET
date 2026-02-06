#!/usr/bin/env python3
"""
Full duplex voice demo with real STT/TTS.

Usage:
    # With mock STT/TTS (no dependencies)
    python examples/duplex_full.py --mock
    
    # With real STT/TTS
    pip install faster-whisper edge-tts
    python examples/duplex_full.py
    
    # With custom LLM reasoner
    python examples/duplex_full.py --reasoner openai
"""

import argparse
import torch
from typing import Optional


def echo_reasoner(text: str) -> str:
    """Simple echo reasoner."""
    return f"You said: {text}"


def get_openai_reasoner(model: str = "gpt-4o-mini"):
    """Get OpenAI-based reasoner."""
    try:
        from openai import OpenAI
        client = OpenAI()
        
        def reasoner(text: str) -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant. Keep responses brief."},
                    {"role": "user", "content": text}
                ],
                max_tokens=100
            )
            return response.choices[0].message.content
        
        return reasoner
    except Exception as e:
        print(f"OpenAI not available: {e}")
        return echo_reasoner


def run_simulated(use_mock: bool = True, n_iterations: int = 3):
    """Run with simulated audio."""
    print("=" * 50)
    print("ToneNet Full Duplex Demo (Simulated)")
    print("=" * 50)
    
    from tonenet.orchestrator_api import AudioOrchestrator
    
    orch = AudioOrchestrator(
        use_mock_stt=use_mock,
        use_mock_tts=use_mock
    )
    
    print(f"Mock mode: {use_mock}")
    print()
    
    for i in range(n_iterations):
        print(f"--- Iteration {i+1} ---")
        
        # Simulate incoming audio (1 second)
        audio_in = torch.randn(1, 1, 24000)
        
        # Run duplex step
        transcript, response, audio_out = orch.duplex_step(
            audio_in,
            reasoner=echo_reasoner
        )
        
        print(f"Transcript: {transcript}")
        print(f"Response: {response}")
        if audio_out is not None:
            print(f"Audio out: {audio_out.shape}")
        print()
    
    print("Event log:")
    for event in orch.get_event_log():
        print(f"  {event['type']}: {event.get('text', event.get('transcript', ''))[:50]}")
    
    print("\nDone!")


def run_live(use_mock: bool = False, reasoner_type: str = "echo"):
    """Run with live microphone."""
    print("=" * 50)
    print("ToneNet Full Duplex Demo (Live)")
    print("=" * 50)
    
    from tonenet.orchestrator_api import AudioOrchestrator
    from tonenet.mic_stream import MicStream
    
    orch = AudioOrchestrator(
        use_mock_stt=use_mock,
        use_mock_tts=use_mock
    )
    
    # Get reasoner
    if reasoner_type == "openai":
        reasoner = get_openai_reasoner()
    else:
        reasoner = echo_reasoner
    
    print(f"Reasoner: {reasoner_type}")
    print("Starting microphone...")
    print("Speak something, press Ctrl+C to stop")
    print()
    
    mic = MicStream(chunk_duration_ms=2000)  # 2 second chunks
    
    try:
        for i, audio_chunk in enumerate(mic.stream(max_chunks=10)):
            print(f"\n[Chunk {i+1}] Listening...")
            
            # Run duplex
            transcript, response, audio_out = orch.duplex_step(
                audio_chunk,
                reasoner=reasoner
            )
            
            if transcript.strip():
                print(f"You: {transcript}")
                print(f"Bot: {response}")
            else:
                print("(silence)")
    
    except KeyboardInterrupt:
        print("\nStopped")
    
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="ToneNet Full Duplex Demo")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock STT/TTS (no dependencies)")
    parser.add_argument("--live", action="store_true",
                        help="Use live microphone instead of simulated audio")
    parser.add_argument("--reasoner", default="echo",
                        choices=["echo", "openai"],
                        help="Reasoner type")
    parser.add_argument("-n", type=int, default=3,
                        help="Number of iterations (simulated mode)")
    args = parser.parse_args()
    
    if args.live:
        run_live(use_mock=args.mock, reasoner_type=args.reasoner)
    else:
        run_simulated(use_mock=args.mock, n_iterations=args.n)


if __name__ == "__main__":
    main()
