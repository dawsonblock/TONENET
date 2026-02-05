"""
Streaming executor for real-time framewise audio decode.

Emits audio continuously from token frames with bounded latency
and deterministic replay capability.
"""

import torch
from collections import deque
from typing import Optional, List
from .codec import ToneNetCodec


class StreamingToneNet:
    """
    Real-time streaming decoder with bounded latency.
    
    Example:
        streamer = StreamingToneNet(chunk_frames=5)
        streamer.push_tokens(tokens)
        audio = streamer.pop_audio()  # emits every 5/75 = 66ms
    """
    
    def __init__(
        self,
        device: str = "cpu",
        frame_rate: int = 75,
        chunk_frames: int = 5,
        seed: int = 0
    ):
        """
        Args:
            device: 'cpu' or 'cuda'
            frame_rate: Codec frame rate (default 75 Hz)
            chunk_frames: Frames to buffer before emit (latency = chunk_frames/frame_rate)
            seed: Random seed for deterministic output
        """
        self.codec = ToneNetCodec().to(device).eval()
        self.device = device
        self.frame_rate = frame_rate
        self.chunk_frames = chunk_frames
        self.buf: deque = deque()
        self.seed = seed
        torch.manual_seed(seed)
        
        # Latency in milliseconds
        self.latency_ms = (chunk_frames / frame_rate) * 1000
    
    @torch.no_grad()
    def push_tokens(self, token_chunk: torch.Tensor):
        """
        Push token frames into buffer.
        
        Args:
            token_chunk: Token tensor, shape varies by codec config
        """
        self.buf.append(token_chunk.to(self.device))
    
    @torch.no_grad()
    def pop_audio(self) -> Optional[torch.Tensor]:
        """
        Pop and decode audio if enough frames buffered.
        
        Returns:
            Audio tensor or None if insufficient frames
        """
        if not self.buf:
            return None
        
        tokens = torch.cat(list(self.buf), dim=-1)
        
        if tokens.shape[-1] < self.chunk_frames:
            return None
        
        # Split: emit chunk, keep remainder
        emit = tokens[..., :self.chunk_frames]
        remain = tokens[..., self.chunk_frames:]
        
        self.buf.clear()
        if remain.numel() > 0:
            self.buf.append(remain)
        
        # Decode the chunk
        # Note: decode expects list of code tensors per quantizer layer
        audio = self.codec.decode([emit])
        return audio
    
    def flush(self) -> Optional[torch.Tensor]:
        """Decode all remaining buffered tokens."""
        if not self.buf:
            return None
        
        tokens = torch.cat(list(self.buf), dim=-1)
        self.buf.clear()
        
        if tokens.numel() == 0:
            return None
        
        audio = self.codec.decode([tokens])
        return audio
    
    def reset(self, seed: Optional[int] = None):
        """Reset buffer and optionally reseed."""
        self.buf.clear()
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)
