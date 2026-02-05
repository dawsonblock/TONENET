"""
Autoregressive language model for voice cloning.

Generates first quantizer codes conditioned on
text embeddings and speaker identity.
"""

import torch
import torch.nn as nn


class VoiceCloningAR(nn.Module):
    """
    Autoregressive transformer that generates first quantizer codes
    conditioned on text and speaker.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        codebook_size: int = 1024,
        d_model: int = 1024,
        n_layers: int = 12,
        n_heads: int = 16,
        max_seq_len: int = 2000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model

        # Embeddings
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.code_embed = nn.Embedding(codebook_size + 2, d_model)  # +2 for BOS/EOS
        self.speaker_proj = nn.Linear(256, d_model)

        # Positional encoding
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Transformer decoder (causal)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, n_heads, d_model * 4,
                dropout=dropout, batch_first=True
            ),
            num_layers=n_layers
        )

        # Output head
        self.code_head = nn.Linear(d_model, codebook_size)

        # Special tokens
        self.bos_token = codebook_size
        self.eos_token = codebook_size + 1

    def forward(self, text_tokens: torch.Tensor, speaker_emb: torch.Tensor, previous_codes: torch.Tensor):
        """
        Training forward (teacher forcing).

        Args:
            text_tokens: (B, T_text) phoneme indices
            speaker_emb: (B, 256) speaker embedding
            previous_codes: (B, T_code) previous code indices

        Returns:
            logits: (B, T_code, codebook_size)
        """
        # Create causal mask
        T = previous_codes.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=previous_codes.device),
            diagonal=1
        ).bool()

        # Embeddings
        text_emb = self.text_embed(text_tokens)  # (B, T_text, d_model)
        speaker_emb_proj = self.speaker_proj(speaker_emb).unsqueeze(1)  # (B, 1, d_model)

        # Add speaker to text context (memory)
        memory = text_emb + speaker_emb_proj

        # Code embeddings with position
        code_emb = self.code_embed(previous_codes)  # (B, T_code, d_model)
        positions = torch.arange(T, device=previous_codes.device)
        code_emb = code_emb + self.pos_embed(positions).unsqueeze(0)

        # Causal generation
        output = self.transformer(
            tgt=code_emb,
            memory=memory,
            tgt_mask=causal_mask
        )

        logits = self.code_head(output)
        return logits

    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,
        speaker_emb: torch.Tensor,
        max_length: int = 1000,
        temperature: float = 0.8
    ):
        """
        Autoregressive generation (inference).

        Args:
            text_tokens: (B, T_text) phoneme indices
            speaker_emb: (B, 256) speaker embedding
            max_length: maximum sequence length
            temperature: sampling temperature

        Returns:
            generated: (B, T) generated codes
        """
        B = text_tokens.size(0)
        device = text_tokens.device

        # Start with BOS token
        generated = torch.full((B, 1), self.bos_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            logits = self.forward(text_tokens, speaker_emb, generated)
            logits = logits[:, -1, :] / temperature

            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_code = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_code], dim=1)

            # Check for EOS
            if (next_code >= self.codebook_size).all():
                break

        return generated[:, 1:]  # Remove BOS
