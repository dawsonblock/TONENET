"""
Audio quality evaluation metrics.

Includes:
- SNR (Signal-to-Noise Ratio)
- STOI (Short-Time Objective Intelligibility) - requires pystoi
- PESQ (Perceptual Evaluation of Speech Quality) - requires pesq
"""

import numpy as np
from typing import Optional


def compute_snr(reference: np.ndarray, degraded: np.ndarray) -> float:
    """
    Signal-to-noise ratio in dB.
    
    Args:
        reference: Clean reference signal
        degraded: Degraded/reconstructed signal
    
    Returns:
        SNR in dB (higher is better)
    """
    noise = reference - degraded
    signal_power = np.sum(reference ** 2)
    noise_power = np.sum(noise ** 2) + 1e-10
    return 10 * np.log10(signal_power / noise_power)


def compute_stoi(
    reference: np.ndarray, 
    degraded: np.ndarray, 
    fs: int = 16000
) -> Optional[float]:
    """
    Short-Time Objective Intelligibility.
    
    Args:
        reference: Clean reference signal
        degraded: Degraded/reconstructed signal
        fs: Sample rate (default 16kHz)
    
    Returns:
        STOI score [0, 1] or None if pystoi not installed
    """
    try:
        from pystoi import stoi
        return stoi(reference, degraded, fs, extended=False)
    except ImportError:
        return None


def compute_pesq(
    reference: np.ndarray, 
    degraded: np.ndarray, 
    fs: int = 16000
) -> Optional[float]:
    """
    Perceptual Evaluation of Speech Quality.
    
    Args:
        reference: Clean reference signal
        degraded: Degraded/reconstructed signal
        fs: Sample rate (16000 or 8000)
    
    Returns:
        PESQ score [-0.5, 4.5] or None if pesq not installed
    """
    try:
        from pesq import pesq
        mode = 'wb' if fs == 16000 else 'nb'
        return pesq(fs, reference, degraded, mode)
    except ImportError:
        return None
