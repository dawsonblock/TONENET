"""
Harmonic decoder for interpretable audio synthesis.

Features:
- Explicit f0, harmonic amplitudes, phases, noise outputs
- Additive synthesis for waveform generation
- Optional neural vocoder residual
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HarmonicDecoder(nn.Module):
    """
    Decoder that outputs explicit harmonic parameters.
    Enables interpretable editing of f0, harmonics, and phases.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_harmonics: int = 64,
        sample_rate: int = 24000,
        frame_rate: int = 75
    ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.hop_length = sample_rate // frame_rate
        self.hidden_dim = hidden_dim

        # Temporal modeling
        self.lstm = nn.LSTM(
            latent_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Output heads
        # f0: log-scaled for perceptual uniformity (50 Hz to 8 kHz)
        self.f0_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

        # Harmonic amplitudes: log-scale for dynamic range
        self.amp_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_harmonics)
        )

        # Phases: periodic
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_harmonics)
        )

        # Aperiodic/noise component for breathy sounds
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Optional: neural vocoder head for residual (not just harmonics)
        self.vocoder_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        )

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, T, latent_dim) quantized latent
        Returns:
            f0: (B, T) fundamental frequency in Hz
            H: (B, T, num_harmonics) harmonic amplitudes (linear, positive)
            phi: (B, T, num_harmonics) harmonic phases [0, 2π]
            noise: (B, T) aperiodic energy ratio [0, 1]
            vocoder_residual: (B, 1, T*hop_length) optional neural residual
        """
        B, T, _ = z.shape

        # Temporal modeling
        x, _ = self.lstm(z)  # (B, T, hidden_dim)

        # f0: 50 Hz to ~8 kHz (log-scaled)
        f0_logits = self.f0_head(x).squeeze(-1)  # (B, T)
        f0 = 50.0 * torch.exp(f0_logits)
        f0 = torch.clamp(f0, 20.0, 8000.0)

        # Harmonic amplitudes: positive, unnormalized (preserves loudness)
        H = torch.exp(self.amp_head(x))  # (B, T, num_harmonics)
        H = torch.clamp(H, 1e-6, 1e3)  # Prevent explosion

        # Phases: [0, 2π]
        phi = torch.sigmoid(self.phase_head(x)) * 2 * math.pi

        # Noise component
        noise = self.noise_head(x).squeeze(-1)  # (B, T)

        # Optional vocoder residual for non-harmonic content
        x_conv = x.transpose(1, 2)  # (B, hidden_dim, T)
        vocoder_out = self.vocoder_conv(x_conv)  # (B, 1, T)
        vocoder_residual = F.interpolate(
            vocoder_out, size=T * self.hop_length,
            mode='linear', align_corners=False
        )

        return f0, H, phi, noise, vocoder_residual

    def synthesize(
        self,
        f0: torch.Tensor,
        H: torch.Tensor,
        phi: torch.Tensor,
        noise: torch.Tensor,
        vocoder_residual: Optional[torch.Tensor] = None,
        sample_rate: Optional[int] = None
    ):
        """
        Additive synthesis from harmonic parameters.

        Args:
            f0: (B, T) fundamental frequency
            H: (B, T, K) harmonic amplitudes
            phi: (B, T, K) harmonic phases
            noise: (B, T) noise ratio
            vocoder_residual: optional neural residual
            sample_rate: output sample rate

        Returns:
            waveform: (B, num_samples)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        B, T = f0.shape
        K = H.shape[-1]
        hop_length = sample_rate // self.frame_rate
        num_samples = T * hop_length

        # Time axis
        t = torch.arange(num_samples, device=f0.device).float() / sample_rate
        t = t.unsqueeze(0).expand(B, -1)  # (B, num_samples)

        # Interpolate f0 to sample rate
        f0_contour = F.interpolate(
            f0.unsqueeze(1), size=num_samples, mode='linear', align_corners=False
        ).squeeze(1)  # (B, num_samples)

        # Cumulative phase for continuous oscillators (prevents phase jumps)
        phase_cum = torch.cumsum(2 * math.pi * f0_contour / sample_rate, dim=1)

        # Generate harmonics
        waveform = torch.zeros(B, num_samples, device=f0.device)
        for k in range(min(K, self.num_harmonics)):
            # Harmonic frequency is (k+1) * f0
            harmonic_phase = phase_cum * (k + 1) + \
                             F.interpolate(
                                 phi[:, :, k].unsqueeze(1),
                                 size=num_samples, mode='linear', align_corners=False
                             ).squeeze(1)

            # Amplitude envelope
            harmonic_amp = F.interpolate(
                H[:, :, k].unsqueeze(1),
                size=num_samples, mode='linear', align_corners=False
            ).squeeze(1)

            waveform += harmonic_amp * torch.sin(harmonic_phase)

        # Add noise component (breathy sounds)
        noise_amp = F.interpolate(
            noise.unsqueeze(1), size=num_samples, mode='linear', align_corners=False
        ).squeeze(1)
        noise_signal = torch.randn_like(waveform) * noise_amp
        waveform = waveform * (1 - noise_amp) + noise_signal

        # Add neural vocoder residual if provided
        if vocoder_residual is not None:
            waveform = waveform + vocoder_residual.squeeze(1)

        # Normalize to prevent clipping
        waveform = waveform / (waveform.abs().max(dim=-1, keepdim=True)[0] + 1e-8)

        return waveform
