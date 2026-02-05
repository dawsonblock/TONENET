"""
Perceptual loss functions for neural audio codec training.

Based on Yamamoto et al., 2020 (multi-resolution STFT loss)
and standard mel-spectrogram reconstruction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss (Yamamoto et al., 2020).
    Industry standard for neural audio codecs.
    """

    def __init__(
        self,
        fft_sizes: List[int] = None,
        hop_sizes: List[int] = None,
        window_sizes: List[int] = None
    ):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [512, 1024, 2048]
        if hop_sizes is None:
            hop_sizes = [128, 256, 512]
        if window_sizes is None:
            window_sizes = [512, 1024, 2048]

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.window_sizes = window_sizes

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: target waveform (B, 1, T)
            y: predicted waveform (B, 1, T)
        """
        loss = 0.0

        for n_fft, hop_length, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.window_sizes
        ):
            window = torch.hann_window(win_length, device=x.device)

            # STFT
            X = torch.stft(
                x.squeeze(1), n_fft, hop_length, win_length,
                window, return_complex=True
            )
            Y = torch.stft(
                y.squeeze(1), n_fft, hop_length, win_length,
                window, return_complex=True
            )

            # Magnitude loss (log-compressed for perceptual uniformity)
            X_mag = torch.abs(X)
            Y_mag = torch.abs(Y)
            loss += F.l1_loss(
                torch.log(X_mag + 1e-7),
                torch.log(Y_mag + 1e-7)
            )

            # Spectral convergence (L2 norm ratio)
            loss += torch.norm(X_mag - Y_mag, p='fro') / (torch.norm(X_mag, p='fro') + 1e-8)

        return loss / len(self.fft_sizes)


class MelSpectrogramLoss(nn.Module):
    """Mel-spectrogram loss with standard mel scale."""

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        n_mels: int = 80,
        f_min: float = 0,
        f_max: Optional[float] = None
    ):
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels

        if f_max is None:
            f_max = sample_rate // 2

        # Standard mel scale (Stevens & Volkmann, 1940)
        mel_min = 2595 * np.log10(1 + f_min / 700)
        mel_max = 2595 * np.log10(1 + f_max / 700)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        freq_points = 700 * (10 ** (mel_points / 2595) - 1)

        # Create mel filterbank
        bins = np.floor((n_fft + 1) * freq_points / sample_rate).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1))

        for i in range(n_mels):
            # Rising slope
            for j in range(bins[i], bins[i + 1]):
                if bins[i + 1] != bins[i]:
                    fb[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
            # Falling slope
            for j in range(bins[i + 1], bins[i + 2]):
                if bins[i + 2] != bins[i + 1]:
                    fb[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])

        self.register_buffer('fb', torch.from_numpy(fb).float())

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        X = self._melspec(x)
        Y = self._melspec(y)
        return F.l1_loss(torch.log(X + 1e-7), torch.log(Y + 1e-7))

    def _melspec(self, x: torch.Tensor):
        window = torch.hann_window(self.n_fft, device=x.device)
        stft = torch.stft(
            x.squeeze(1), self.n_fft, self.n_fft // 4, self.n_fft,
            window, return_complex=True
        )
        spec = torch.abs(stft) ** 2
        mel = torch.matmul(self.fb, spec)
        return mel
