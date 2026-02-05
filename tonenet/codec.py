"""
ToneNetCodec - Complete neural audio codec.

Combines encoder, residual vector quantizer, and harmonic decoder
into an end-to-end compression/decompression system.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

from .encoder import ToneNetEncoder
from .quantizer import ResidualVectorQuantizer
from .decoder import HarmonicDecoder


class ToneNetCodec(nn.Module):
    """
    Complete neural audio codec with explicit harmonic modeling.
    Non-pseudoscience: standard VQ-VAE with interpretable decoder.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        latent_dim: int = 256,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        num_harmonics: int = 64,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim

        self.encoder = ToneNetEncoder(
            latent_dim=latent_dim,
            sample_rate=sample_rate
        )

        self.quantizer = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            latent_dim=latent_dim,
            commitment_cost=commitment_cost
        )

        self.decoder = HarmonicDecoder(
            latent_dim=latent_dim,
            num_harmonics=num_harmonics,
            sample_rate=sample_rate
        )

    def forward(
        self,
        x: torch.Tensor,
        n_quantizers: Optional[int] = None,
        return_harmonic: bool = True
    ):
        """
        Args:
            x: (B, 1, T) input waveform
            n_quantizers: number of quantizer layers (variable bitrate)
            return_harmonic: whether to compute harmonic parameters

        Returns:
            reconstructed: (B, 1, T) output waveform
            outputs: dict with codes, losses, and harmonic parameters
        """
        # Encode
        z = self.encoder(x)

        # Quantize
        z_q, indices, vq_loss = self.quantizer(z, n_quantizers)

        # Decode
        f0, H, phi, noise, vocoder_residual = self.decoder(z_q)

        # Synthesize
        waveform = self.decoder.synthesize(
            f0, H, phi, noise, vocoder_residual,
            sample_rate=self.sample_rate
        )
        reconstructed = waveform.unsqueeze(1)  # (B, 1, T)

        # Match input length
        if reconstructed.shape[-1] != x.shape[-1]:
            reconstructed = F.interpolate(
                reconstructed, size=x.shape[-1],
                mode='linear', align_corners=False
            )

        outputs = {
            'vq_loss': vq_loss,
            'f0': f0,
            'H': H,
            'phi': phi,
            'noise': noise,
            'indices': indices,
            'latent': z_q
        }

        return reconstructed, outputs

    def encode(self, x: torch.Tensor, n_quantizers: Optional[int] = None) -> List[torch.Tensor]:
        """Get discrete codes for transmission/storage."""
        z = self.encoder(x)
        return self.quantizer.encode(z, n_quantizers)

    def decode(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct from discrete codes."""
        z_q = self.quantizer.decode(indices)
        f0, H, phi, noise, vocoder_residual = self.decoder(z_q)
        waveform = self.decoder.synthesize(f0, H, phi, noise, vocoder_residual)
        return waveform.unsqueeze(1)

    def get_bitrate(self, n_quantizers: Optional[int] = None) -> Dict[str, float]:
        """Calculate honest bitrate."""
        if n_quantizers is None:
            n_quantizers = self.quantizer.num_quantizers

        hop_length = self.sample_rate // 75  # 75 Hz frame rate
        bits_per_code = math.log2(self.quantizer.codebook_size)
        bits_per_frame = n_quantizers * bits_per_code
        frame_rate = self.sample_rate / hop_length

        bitrate = frame_rate * bits_per_frame

        return {
            'bitrate_bps': bitrate,
            'bitrate_kbps': bitrate / 1000,
            'frame_rate_hz': frame_rate,
            'bits_per_frame': bits_per_frame,
            'hop_length': hop_length,
            'compression_ratio_16bit': (self.sample_rate * 16) / bitrate
        }
