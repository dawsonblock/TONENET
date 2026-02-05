"""
Causal encoder for streaming audio processing.

Features:
- Causal convolutions (no future leakage)
- GroupNorm for stability
- Downsampling to 75 Hz frame rate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CausalConv1d(nn.Module):
    """
    Causal convolution for real-time processing.
    Ensures output at time t depends only on inputs <= t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation
        )

    def forward(self, x: torch.Tensor):
        # x: (B, C, T)
        x = F.pad(x, (self.padding, 0))  # Pad left (past) only
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual block with causal convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: int = 1
    ):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor):
        residual = x
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class ToneNetEncoder(nn.Module):
    """
    Causal encoder for streaming audio.
    Produces latent representations at reduced frame rate.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = None,
        latent_dim: int = 256,
        sample_rate: int = 24000,
        hop_length: int = 320  # 75 Hz frame rate at 24kHz
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_rate = sample_rate / hop_length
        self.latent_dim = latent_dim

        # Initial causal conv
        self.input_conv = CausalConv1d(in_channels, channels[0], kernel_size=7)

        # Downsampling blocks with residual connections
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            stride = 2
            kernel = 2 * stride
            self.blocks.append(nn.Sequential(
                ResidualBlock(channels[i]),
                nn.Conv1d(
                    channels[i], channels[i + 1],
                    kernel_size=kernel,
                    stride=stride,
                    padding=0
                ),
                nn.GroupNorm(8, channels[i + 1]),
                nn.SiLU()
            ))

        # Final projection to latent
        self.latent_conv = CausalConv1d(channels[-1], latent_dim, kernel_size=3)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, T) waveform
        Returns:
            z: (B, T_latent, latent_dim) where T_latent = T // hop_length
        """
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        z = self.latent_conv(x)
        return z.transpose(1, 2)  # (B, T_latent, latent_dim)

    def get_receptive_field(self) -> int:
        """Calculate receptive field in samples."""
        # Simplified calculation
        return 5 * self.hop_length  # Approximate
