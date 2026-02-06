"""
Streaming executor for real-time framewise audio decode.

Emits audio continuously from token frames with bounded latency
and deterministic replay capability.

IMPORTANT: This handles multi-quantizer codes correctly.
Codes from the codec are List[Tensor[B,T]], one per quantizer.
"""

import torch
from typing import Optional, List, Union
from .codec import ToneNetCodec
from .tokens import normalize_codes, get_code_info


class StreamingToneNet:
    """
    Real-time streaming decoder with bounded latency.
    
    Properly handles multi-quantizer codes (list format).
    
    Example:
        streamer = StreamingToneNet(chunk_frames=5)
        codes = codec.encode(audio)  # list of 8 tensors
        streamer.push_codes(codes)
        audio = streamer.pop_audio()  # emits every 5/75 = 66ms
    """
    
    def __init__(
        self,
        device: str = "cpu",
        frame_rate: int = 75,
        chunk_frames: int = 5,
        seed: int = 0,
        n_quantizers: int = 8
    ):
        """
        Args:
            device: 'cpu' or 'cuda'
            frame_rate: Codec frame rate (default 75 Hz)
            chunk_frames: Frames to buffer before emit (latency = chunk_frames/frame_rate)
            seed: Random seed for deterministic output
            n_quantizers: Number of quantizer layers (default 8)
        """
        self.codec = ToneNetCodec().to(device).eval()
        self.device = device
        self.frame_rate = frame_rate
        self.chunk_frames = chunk_frames
        self.n_quantizers = n_quantizers
        self.seed = seed
        torch.manual_seed(seed)
        
        # Buffer: one list per quantizer layer
        self.buffers: List[List[torch.Tensor]] = [[] for _ in range(n_quantizers)]
        
        # Latency in milliseconds
        self.latency_ms = (chunk_frames / frame_rate) * 1000
    
    @torch.no_grad()
    def push_codes(self, codes: Union[List[torch.Tensor], torch.Tensor]):
        """
        Push codec codes into buffer.
        
        Args:
            codes: Either List[Tensor[B,T]] (one per quantizer) 
                   or Tensor[B,Q,T] (packed format)
        """
        codes = normalize_codes(codes)
        
        # Pad if fewer quantizers provided
        while len(codes) < self.n_quantizers:
            codes.append(torch.zeros_like(codes[0]))
        
        for i, code in enumerate(codes[:self.n_quantizers]):
            self.buffers[i].append(code.to(self.device))
    
    # Legacy alias
    def push_tokens(self, token_chunk: Union[List[torch.Tensor], torch.Tensor]):
        """Legacy alias for push_codes."""
        self.push_codes(token_chunk)
    
    def _concat_buffer(self, buf: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Concatenate buffer tensors along time axis."""
        if not buf:
            return None
        return torch.cat(buf, dim=-1)
    
    def _get_buffered_frames(self) -> int:
        """Get number of frames currently buffered."""
        if not self.buffers[0]:
            return 0
        concat = self._concat_buffer(self.buffers[0])
        return concat.shape[-1] if concat is not None else 0
    
    @torch.no_grad()
    def pop_audio(self) -> Optional[torch.Tensor]:
        """
        Pop and decode audio if enough frames buffered.
        
        Returns:
            Audio tensor or None if insufficient frames
        """
        n_frames = self._get_buffered_frames()
        
        if n_frames < self.chunk_frames:
            return None
        
        # Collect and split each quantizer buffer
        emit_codes: List[torch.Tensor] = []
        
        for i in range(self.n_quantizers):
            concat = self._concat_buffer(self.buffers[i])
            if concat is None:
                # Empty quantizer, create zeros
                emit_codes.append(torch.zeros(1, self.chunk_frames, device=self.device, dtype=torch.long))
                continue
            
            # Split: emit chunk, keep remainder
            emit = concat[..., :self.chunk_frames]
            remain = concat[..., self.chunk_frames:]
            
            emit_codes.append(emit)
            
            # Update buffer with remainder
            self.buffers[i].clear()
            if remain.numel() > 0:
                self.buffers[i].append(remain)
        
        # Decode the chunk (list format)
        audio = self.codec.decode(emit_codes)
        return audio
    
    def flush(self) -> Optional[torch.Tensor]:
        """Decode all remaining buffered frames."""
        n_frames = self._get_buffered_frames()
        
        if n_frames == 0:
            return None
        
        emit_codes: List[torch.Tensor] = []
        
        for i in range(self.n_quantizers):
            concat = self._concat_buffer(self.buffers[i])
            if concat is None:
                continue
            emit_codes.append(concat)
            self.buffers[i].clear()
        
        if not emit_codes:
            return None
        
        audio = self.codec.decode(emit_codes)
        return audio
    
    def reset(self, seed: Optional[int] = None):
        """Reset buffers and optionally reseed."""
        for buf in self.buffers:
            buf.clear()
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)
    
    @property
    def buffered_frames(self) -> int:
        """Number of frames currently buffered."""
        return self._get_buffered_frames()
    
    @property
    def buffered_duration_ms(self) -> float:
        """Duration of buffered audio in milliseconds."""
        return (self._get_buffered_frames() / self.frame_rate) * 1000
