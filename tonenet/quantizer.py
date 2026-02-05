"""
Residual Vector Quantizer (RVQ) for neural audio compression.

Based on VQ-VAE (Oord et al., 2017) with EMA codebook updates
and multi-layer residual quantization (DAC/Mimi style).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE with EMA codebook updates.
    Based on Oord et al. 2017, with modern stability improvements.
    """

    def __init__(
        self,
        codebook_size: int = 1024,
        latent_dim: int = 256,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # EMA-based codebook for stability
        self.register_buffer('embeddings', torch.randn(codebook_size, latent_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_w', self.embeddings.clone())

        # Initialize with uniform random
        nn.init.uniform_(self.embeddings, -1 / self.codebook_size, 1 / self.codebook_size)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, T, latent_dim) encoder outputs
        Returns:
            quantized: (B, T, latent_dim)
            indices: (B, T) codebook indices
            loss: VQ commitment loss
        """
        B, T, D = z.shape
        assert D == self.latent_dim, f"Expected {self.latent_dim}, got {D}"

        # Flatten for quantization
        z_flat = z.reshape(-1, D)

        # Compute L2 distances to all codebook vectors
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2<z,e>
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)  # ||z||^2
            - 2 * torch.matmul(z_flat, self.embeddings.t())  # -2<z,e>
            + torch.sum(self.embeddings ** 2, dim=1)  # ||e||^2
        )

        # Find nearest neighbors
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.codebook_size).float()

        # Quantize
        quantized_flat = torch.matmul(encodings, self.embeddings)
        quantized = quantized_flat.view(B, T, D)

        # EMA update during training
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(encodings, dim=0)

            # Laplace smoothing
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.codebook_size * self.epsilon) * n
            )

            # Update embeddings
            dw = torch.matmul(encodings.t(), z_flat)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
            self.embeddings.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)

        # Losses
        commitment_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator for gradients
        quantized = z + (quantized - z).detach()

        return quantized, indices.view(B, T), loss

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookup codebook entries by indices."""
        return self.embeddings[indices]


class ResidualVectorQuantizer(nn.Module):
    """
    Multi-scale residual quantization (Mimi/DAC style).
    Uses multiple VQ layers for progressive refinement.
    """

    def __init__(
        self,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        latent_dim: int = 256,
        commitment_cost: float = 0.25,
        shared_codebook: bool = False
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.shared_codebook = shared_codebook

        # Separate codebooks per quantizer (better quality)
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, latent_dim, commitment_cost)
            for _ in range(num_quantizers)
        ])

    def forward(self, z: torch.Tensor, n_quantizers: Optional[int] = None):
        """
        Args:
            z: (B, T, latent_dim)
            n_quantizers: number of quantizers to use (for variable bitrate)
        Returns:
            quantized: (B, T, latent_dim) sum of all quantizer outputs
            all_indices: list of (B, T) indices per quantizer
            total_loss: sum of VQ losses
        """
        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        residual = z
        all_indices = []
        total_loss = 0.0

        for i, quantizer in enumerate(self.quantizers[:n_quantizers]):
            quantized, indices, loss = quantizer(residual)
            residual = residual - quantized  # Residual for next quantizer
            all_indices.append(indices)
            total_loss = total_loss + loss

        # Final quantized is sum of all layers
        quantized = z - residual

        return quantized, all_indices, total_loss

    def encode(self, z: torch.Tensor, n_quantizers: Optional[int] = None) -> List[torch.Tensor]:
        """Get discrete codes only (no gradients)."""
        with torch.no_grad():
            _, indices, _ = self.forward(z, n_quantizers)
        return indices

    def decode(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct from discrete codes."""
        quantized = 0
        for i, idx in enumerate(indices):
            q = self.quantizers[i].get_codebook_entry(idx)
            quantized = quantized + q
        return quantized
