"""
Token Language Model for low-bitrate intelligibility boost.

Predicts likely token sequences to restore linguistic clarity
at very low bitrates where codec-only speech loses intelligibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TokenLanguageModel(nn.Module):
    """
    Lightweight Transformer for token sequence modeling.
    
    Used for:
    - Token denoising
    - Low-bitrate enhancement
    - Predictive fill during streaming gaps
    """
    
    def __init__(
        self,
        vocab_size: int = 1024,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_len: int = 4096,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.out = nn.Linear(d_model, vocab_size)
        
        # Causal mask for autoregressive modeling
        self.register_buffer("causal_mask", None)
    
    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Token indices [B, T]
        
        Returns:
            Logits [B, T, vocab_size]
        """
        B, T = x.shape
        
        h = self.embed(x) + self.pos[:, :T]
        
        mask = self._get_causal_mask(T, x.device)
        h = self.net(h, mask=mask)
        
        return self.out(h)
    
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute autoregressive loss.
        
        Args:
            x: Token sequence [B, T]
        
        Returns:
            Cross-entropy loss
        """
        logits = self.forward(x[:, :-1])  # [B, T-1, V]
        target = x[:, 1:]  # [B, T-1]
        
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target.reshape(-1)
        )
    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            prompt: Starting tokens [1, T]
            max_new: Max new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
        
        Returns:
            Generated sequence [1, T + max_new]
        """
        self.eval()
        tokens = prompt.clone()
        
        for _ in range(max_new):
            logits = self.forward(tokens)[:, -1, :]  # [1, V]
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens
    
    @torch.no_grad()
    def refine(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Refine/denoise token sequence.
        
        Args:
            tokens: Noisy token sequence [B, T]
        
        Returns:
            Refined tokens [B, T]
        """
        logits = self.forward(tokens)
        return torch.argmax(logits, dim=-1)


class StreamingLM:
    """Streaming wrapper for continuous token refinement."""
    
    def __init__(self, model: TokenLanguageModel, window: int = 256):
        self.model = model
        self.window = window
        self.buf: Optional[torch.Tensor] = None
    
    @torch.no_grad()
    def step(self, new_tokens: torch.Tensor) -> torch.Tensor:
        """
        Process new tokens in streaming fashion.
        
        Args:
            new_tokens: New token chunk [1, T_new]
        
        Returns:
            Refined last token [1, 1]
        """
        if self.buf is None:
            self.buf = new_tokens
        else:
            self.buf = torch.cat([self.buf, new_tokens], dim=-1)
            # Keep window size
            self.buf = self.buf[..., -self.window:]
        
        logits = self.model(self.buf)
        return torch.argmax(logits[:, -1:], dim=-1)
    
    def reset(self):
        self.buf = None
