"""
Training framework for ToneNet codec.

Includes:
- Combined loss function (time + STFT + mel + VQ)
- AdamW optimizer with cosine annealing
- Validation with audio quality metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .codec import ToneNetCodec
from .losses import MultiResolutionSTFTLoss, MelSpectrogramLoss
from .metrics import compute_snr, compute_stoi, compute_pesq


class ToneNetTrainer:
    """Complete training framework with validation."""

    def __init__(
        self,
        model: ToneNetCodec,
        device: str = 'cuda',
        lr: float = 3e-4,
        lambda_vq: float = 1.0,
        lambda_stft: float = 1.0,
        lambda_mel: float = 0.5
    ):
        self.device = device
        self.model = model.to(device)

        self.stft_loss = MultiResolutionSTFTLoss().to(device)
        self.mel_loss = MelSpectrogramLoss().to(device)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

        self.lambda_vq = lambda_vq
        self.lambda_stft = lambda_stft
        self.lambda_mel = lambda_mel

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        batch = batch.to(self.device)

        # Forward
        reconstructed, outputs = self.model(batch)

        # Losses
        time_loss = F.l1_loss(reconstructed, batch)
        stft_loss = self.stft_loss(batch, reconstructed)
        mel_loss = self.mel_loss(batch, reconstructed)
        vq_loss = outputs['vq_loss']

        total_loss = (
            time_loss +
            self.lambda_stft * stft_loss +
            self.lambda_mel * mel_loss +
            self.lambda_vq * vq_loss
        )

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'time': time_loss.item(),
            'stft': stft_loss.item(),
            'mel': mel_loss.item(),
            'vq': vq_loss.item()
        }

    def validate(self, dataloader: Any) -> Dict[str, float]:
        """Validation with metrics."""
        self.model.eval()
        metrics = {'snr': [], 'stoi': [], 'pesq': []}

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                reconstructed, _ = self.model(batch)

                # Move to CPU for metrics
                ref = batch.cpu().numpy().squeeze()
                deg = reconstructed.cpu().numpy().squeeze()

                # Handle batch dimension
                if ref.ndim == 1:
                    ref = ref[np.newaxis, :]
                    deg = deg[np.newaxis, :]

                for r, d in zip(ref, deg):
                    metrics['snr'].append(compute_snr(r, d))
                    stoi = compute_stoi(r, d)
                    if stoi is not None:
                        metrics['stoi'].append(stoi)
                    pesq = compute_pesq(r, d)
                    if pesq is not None:
                        metrics['pesq'].append(pesq)

        return {
            'snr': np.mean(metrics['snr']) if metrics['snr'] else 0.0,
            'stoi': np.mean(metrics['stoi']) if metrics['stoi'] else None,
            'pesq': np.mean(metrics['pesq']) if metrics['pesq'] else None
        }

    def step_scheduler(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()

    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """Save training checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            **kwargs
        }, path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint
