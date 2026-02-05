"""
ECAPA-TDNN speaker encoder.

State-of-the-art for speaker verification/identification.
Extracts fixed-dimensional embeddings that capture voice identity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Channel attention mechanism."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor):
        # x: (B, C, T)
        b, c, t = x.size()
        y = x.mean(dim=2)  # Global average pooling: (B, C)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.unsqueeze(2)


class SE_Res2Block(nn.Module):
    """Squeeze-Excitation Res2Net block with dilation."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=dilation, dilation=dilation, groups=channels
        )
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=1)
        self.se = SqueezeExcitation(channels)

    def forward(self, x: torch.Tensor):
        residual = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.se(x)
        return F.relu(x + residual)


class AttentiveStatisticsPooling(nn.Module):
    """Learnable attention-based statistics pooling."""

    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        # x: (B, C, T)
        # Attention weights
        w = torch.softmax(self.attention(x), dim=2)  # (B, 1, T)

        # Weighted mean and std
        mean = (x * w).sum(dim=2)  # (B, C)
        var = ((x - mean.unsqueeze(2)) ** 2 * w).sum(dim=2)
        std = torch.sqrt(var + 1e-8)

        return torch.cat([mean, std], dim=1)  # (B, 2*C)


class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN speaker encoder.
    State-of-the-art for speaker verification/identification.
    More efficient than WavLM for voice cloning.
    """

    def __init__(self, embedding_dim: int = 256, channels: int = 512):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv1d(80, channels, kernel_size=5, padding=2)

        # SE-Res2Blocks with different dilation rates
        self.block1 = SE_Res2Block(channels, kernel_size=3, dilation=2)
        self.block2 = SE_Res2Block(channels, kernel_size=3, dilation=3)
        self.block3 = SE_Res2Block(channels, kernel_size=3, dilation=4)

        # Multi-layer feature aggregation
        self.mfa = nn.Conv1d(channels * 3, channels * 3, kernel_size=1)

        # Attentive statistics pooling
        self.asp = AttentiveStatisticsPooling(channels * 3)

        # Final embedding
        self.fc = nn.Sequential(
            nn.Linear(channels * 3 * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, mel_spectrogram: torch.Tensor):
        """
        Args:
            mel_spectrogram: (B, 80, T) mel features

        Returns:
            embedding: (B, embedding_dim) speaker vector (L2-normalized)
        """
        x = F.relu(self.conv1(mel_spectrogram))

        # Multi-scale feature extraction
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        # Concatenate multi-scale features
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.mfa(x)

        # Attentive statistics pooling
        x = self.asp(x)  # (B, channels*3*2)

        # Embedding
        embedding = self.fc(x)
        return F.normalize(embedding, dim=1)
