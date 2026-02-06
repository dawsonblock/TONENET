#!/usr/bin/env python3
"""
Duplex voice demo using ToneNet's actual APIs.

This example shows the correct way to use the codec's encode/decode
with the orchestrator's policy and watermarking.

IMPORTANT: The codec returns List[Tensor] (one per quantizer).
This demo handles that correctly.

Usage:
    python examples/duplex_demo.py
    python examples/duplex_demo.py --simulated  # No microphone required
"""

import argparse
import torch

from tonenet import ToneNetCodec, AudioOrchestrator
from tonenet.tokens import pack_codes, get_code_info


def run_simulated_duplex(n_iterations: int = 5):
    """
    Run duplex loop with simulated audio (no mic required).
    
    Demonstrates: encode -> process -> decode with policy/watermark.
    """
    print("=" * 50)
    print("ToneNet Duplex Demo (Simulated)")
    print("=" * 50)
    
    # Initialize
    codec = ToneNetCodec()
    orch = AudioOrchestrator(config={
        "policy": {"allowed_speakers": ["operator", "user"]}
    })
    
    print(f"Codec: {codec}")
    print(f"Latency: {orch.streamer.latency_ms:.0f}ms")
    print()
    
    for i in range(n_iterations):
        print(f"--- Iteration {i+1} ---")
        
        # Simulate incoming audio (1 second)
        audio_in = torch.randn(1, 1, 24000)
        print(f"Input audio: {audio_in.shape}")
        
        # Encode to codes (returns List[Tensor])
        with torch.no_grad():
            codes = codec.encode(audio_in)
        
        # Get info about codes
        info = get_code_info(codes)
        print(f"Encoded: {info['n_quantizers']} quantizers, {info['n_frames']} frames")
        
        # Pack for storage/logging if needed
        packed = pack_codes(codes)
        print(f"Packed shape: {packed.shape}")
        
        # --- PROCESSING WOULD GO HERE ---
        # In a real system, you'd:
        # 1. Run speech recognition
        # 2. Process with LLM/reasoner
        # 3. Generate response tokens
        # For now, we just echo back
        response_codes = codes
        
        # Emit through orchestrator (applies policy + watermark)
        try:
            audio_out = orch.emit_tokens(response_codes, speaker_id="operator")
            print(f"Output audio: {audio_out.shape}")
        except RuntimeError as e:
            print(f"Emission blocked: {e}")
            continue
        
        print()
    
    print("Done!")


def run_live_duplex():
    """
    Run duplex loop with real microphone.
    
    Requires: pip install sounddevice
    """
    print("=" * 50)
    print("ToneNet Duplex Demo (Live)")
    print("=" * 50)
    
    try:
        from tonenet.mic_stream import MicStream
    except ImportError:
        print("ERROR: pip install sounddevice")
        return
    
    codec = ToneNetCodec()
    orch = AudioOrchestrator(config={
        "policy": {"allowed_speakers": ["operator", "user"]}
    })
    
    print("Starting microphone...")
    print("Press Ctrl+C to stop")
    print()
    
    mic = MicStream(chunk_duration_ms=100)
    
    try:
        for i, audio_chunk in enumerate(mic.stream(max_chunks=50)):
            with torch.no_grad():
                codes = codec.encode(audio_chunk)
            
            info = get_code_info(codes)
            print(f"[{i:3d}] {info['n_frames']} frames", end="\r")
            
            # Echo back with watermark
            audio_out = orch.emit_tokens(codes, speaker_id="operator")
            
    except KeyboardInterrupt:
        print("\nStopped")
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="ToneNet Duplex Demo")
    parser.add_argument("--simulated", action="store_true", 
                        help="Use simulated audio instead of microphone")
    parser.add_argument("-n", type=int, default=5,
                        help="Number of iterations (simulated mode)")
    args = parser.parse_args()
    
    if args.simulated:
        run_simulated_duplex(n_iterations=args.n)
    else:
        run_live_duplex()


if __name__ == "__main__":
    main()
