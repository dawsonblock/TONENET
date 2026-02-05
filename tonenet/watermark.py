"""
Speaker identity watermarking for audio traceability.

Embeds a weak, deterministic spectral pattern keyed by speaker ID.
Survives typical compression and enables post-hoc attribution.
"""

import torch
import hashlib
from typing import Tuple


def _key_to_phase(key: str, n: int = 8) -> torch.Tensor:
    """Convert speaker key to deterministic phase offsets."""
    h = hashlib.sha256(key.encode()).digest()
    vals = torch.tensor(list(h), dtype=torch.float32)[:n]
    return (vals / 255.0) * 2 * torch.pi


def embed_watermark(
    audio: torch.Tensor,
    speaker_id: str,
    strength: float = 1e-3
) -> torch.Tensor:
    """
    Embed watermark pattern into audio.
    
    Args:
        audio: Audio tensor [B, 1, T] or [B, T]
        speaker_id: Speaker identifier string
        strength: Watermark amplitude (default 1e-3 = -60dB)
    
    Returns:
        Watermarked audio (same shape as input)
    """
    squeeze_out = False
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
        squeeze_out = True
    
    B, C, T = audio.shape
    device = audio.device
    
    phase = _key_to_phase(speaker_id, 8).to(device)
    t = torch.arange(T, device=device).float() / T
    
    # Generate watermark signal (sum of sinusoids at known frequencies)
    mark = torch.zeros(T, device=device)
    for k in range(len(phase)):
        mark = mark + torch.sin(2 * torch.pi * (k + 1) * t + phase[k])
    mark = mark / len(phase)
    
    result = audio + strength * mark.view(1, 1, -1)
    
    if squeeze_out:
        result = result.squeeze(1)
    
    return result


def detect_watermark(
    audio: torch.Tensor,
    speaker_id: str
) -> float:
    """
    Detect watermark correlation score.
    
    Args:
        audio: Audio tensor [B, 1, T] or [B, T]
        speaker_id: Speaker identifier to check
    
    Returns:
        Correlation score (higher = more likely match)
    """
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
    
    B, C, T = audio.shape
    device = audio.device
    
    phase = _key_to_phase(speaker_id, 8).to(device)
    t = torch.arange(T, device=device).float() / T
    
    # Generate reference watermark
    ref = torch.zeros(T, device=device)
    for k in range(len(phase)):
        ref = ref + torch.sin(2 * torch.pi * (k + 1) * t + phase[k])
    ref = ref / len(phase)
    
    # Correlation score
    score = torch.mean(audio * ref.view(1, 1, -1))
    return float(score)


def verify_watermark(
    audio: torch.Tensor,
    speaker_id: str,
    threshold: float = 1e-5
) -> Tuple[bool, float]:
    """
    Verify if audio contains watermark for given speaker.
    
    Returns:
        (is_match, score)
    """
    score = detect_watermark(audio, speaker_id)
    return score > threshold, score
