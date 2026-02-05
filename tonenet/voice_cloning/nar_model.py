"""
Non-autoregressive refinement model for voice cloning.

Generates remaining quantizer layers in parallel,
conditioned on first layer codes + text + speaker.
"""

import torch
import torch.nn as nn
from typing import List


class VoiceCloningNAR(nn.Module):
    """
    Non-autoregressive transformer for remaining quantizer layers.
    Parallel generation conditioned on first layer + text + speaker.
    """

    def __init__(
        self,
        num_quantizers: int = 7,  # Remaining after AR
        codebook_size: int = 1024,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        text_dim: int = 512,
        speaker_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size

        # Embeddings
        self.code_embed = nn.Embedding(codebook_size, d_model)
        self.layer_embed = nn.Embedding(num_quantizers, d_model)

        # Context projection
        self.context_proj = nn.Linear(d_model + text_dim + speaker_dim, d_model)

        # Transformer encoder (non-causal, parallel)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, d_model * 4,
                dropout=dropout, batch_first=True
            ),
            num_layers=n_layers
        )

        # Output heads (one per quantizer layer)
        self.heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size)
            for _ in range(num_quantizers)
        ])

    def forward(
        self,
        first_codes: torch.Tensor,
        text_emb: torch.Tensor,
        speaker_emb: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Parallel generation of remaining quantizer layers.

        Args:
            first_codes: (B, T) first quantizer codes (from AR)
            text_emb: (B, T_text, text_dim) text embeddings
            speaker_emb: (B, speaker_dim) speaker embedding

        Returns:
            all_codes: list of (B, T) codes for each remaining layer
        """
        B, T = first_codes.shape

        # Embed first layer codes
        code_emb = self.code_embed(first_codes)  # (B, T, d_model)

        # Interpolate text embeddings to match code length
        text_emb_interp = nn.functional.interpolate(
            text_emb.transpose(1, 2),
            size=T,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # (B, T, text_dim)

        # Expand speaker embedding
        speaker_emb_exp = speaker_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, speaker_dim)

        # Combine context
        context = torch.cat([code_emb, text_emb_interp, speaker_emb_exp], dim=-1)
        context = self.context_proj(context)  # (B, T, d_model)

        # Generate each remaining layer in parallel
        all_codes = []

        for q in range(self.num_quantizers):
            # Layer-specific embedding
            layer_id = torch.full((B, T), q, device=first_codes.device, dtype=torch.long)
            layer_emb = self.layer_embed(layer_id)

            # Combine context with layer embedding
            x = context + layer_emb

            # Transform
            x = self.transformer(x)

            # Predict codes for this layer
            logits = self.heads[q](x)  # (B, T, codebook_size)
            codes = torch.argmax(logits, dim=-1)
            all_codes.append(codes)

        return all_codes
