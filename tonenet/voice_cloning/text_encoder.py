"""
Text/Phoneme encoder for voice cloning.

Uses transformer architecture to encode phoneme sequences
into contextual embeddings for speech synthesis.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        return x + self.pe[:x.size(1)]


class PhonemeEncoder(nn.Module):
    """
    Converts text/phonemes to embeddings.
    Uses character-level or phoneme-level encoding.
    """

    def __init__(
        self,
        vocab_size: int = 256,  # Phoneme vocabulary
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, d_model * 4,
                dropout=dropout, batch_first=True
            ),
            num_layers=n_layers
        )

    def forward(self, phonemes: torch.Tensor):
        """
        Args:
            phonemes: (B, T) phoneme indices

        Returns:
            embeddings: (B, T, d_model)
        """
        x = self.embedding(phonemes) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        return self.transformer(x)
