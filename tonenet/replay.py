"""
Deterministic replay for reproducible audio synthesis.

Records seeds, tokens, and metadata; regenerates identical audio later.
"""

import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from .core import ToneNetCodec


def save_trace(
    path: str,
    tokens: torch.Tensor,
    seed: int,
    meta: Dict[str, Any]
) -> None:
    """
    Save replay trace to file.
    
    Args:
        path: Output path (.trace)
        tokens: Token tensor
        seed: Random seed used
        meta: Additional metadata dict
    """
    torch.save({
        "tokens": tokens.cpu(),
        "seed": seed,
        "meta": meta
    }, path)


def replay_trace(
    path: str,
    out_wav: Optional[str] = None,
    device: str = "cpu"
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Replay trace to regenerate identical audio.
    
    Args:
        path: Input trace file
        out_wav: Optional output WAV path
        device: Device for decode
    
    Returns:
        (audio_tensor, metadata)
    """
    obj = torch.load(path, map_location="cpu")
    tokens = obj["tokens"]
    seed = obj["seed"]
    meta = obj["meta"]
    
    # Restore deterministic state
    torch.manual_seed(seed)
    
    codec = ToneNetCodec().to(device).eval()
    
    with torch.no_grad():
        # tokens may be a list or single tensor
        if isinstance(tokens, list):
            codes = [t.to(device) for t in tokens]
        else:
            codes = [tokens.to(device)]
        audio = codec.decode(codes)
    
    if out_wav:
        try:
            import soundfile as sf
            sf.write(out_wav, audio.squeeze().cpu().numpy(), 24000)
        except ImportError:
            import wave
            import numpy as np
            audio_np = (audio.squeeze().cpu().numpy() * 32767).astype(np.int16)
            with wave.open(out_wav, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes(audio_np.tobytes())
    
    return audio, meta


class TraceRecorder:
    """Context manager for trace recording."""
    
    def __init__(self, path: str, seed: int = 0):
        self.path = path
        self.seed = seed
        self.tokens = []
        self.meta = {}
    
    def __enter__(self):
        torch.manual_seed(self.seed)
        return self
    
    def record(self, tokens: torch.Tensor, **meta):
        """Record tokens and metadata."""
        self.tokens.append(tokens.cpu())
        self.meta.update(meta)
    
    def __exit__(self, *args):
        if self.tokens:
            all_tokens = torch.cat(self.tokens, dim=-1)
            save_trace(self.path, all_tokens, self.seed, self.meta)
