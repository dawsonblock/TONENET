"""
ToneNet v2.0 - Neural Audio Codec with Harmonic Modeling

A research-grade neural audio codec featuring:
- Residual Vector Quantization (8×1024 codebook)
- Causal encoder for streaming
- Harmonic decoder for interpretable synthesis
- Variable bitrate (0.75-6 kbps)
- Deterministic audio pipeline with policy/ledger

Example:
    >>> from tonenet import ToneNetCodec
    >>> import torch
    >>> 
    >>> model = ToneNetCodec()
    >>> audio = torch.randn(1, 1, 24000)  # 1 second @ 24kHz
    >>> reconstructed, outputs = model(audio)
    >>> codes = model.encode(audio)
    >>> decoded = model.decode(codes)
"""

__version__ = "2.0.0"

from .codec import ToneNetCodec
from .quantizer import VectorQuantizer, ResidualVectorQuantizer
from .encoder import ToneNetEncoder, CausalConv1d, ResidualBlock
from .decoder import HarmonicDecoder
from .losses import MultiResolutionSTFTLoss, MelSpectrogramLoss
from .metrics import compute_snr, compute_stoi, compute_pesq
from .controller import ClassicalController
from .trainer import ToneNetTrainer
from .deployment import ToneNetDeployment, export_model, verify_model
from .audio import AudioCodec, compress_audio, decompress_audio, reconstruct_audio

# Pipeline modules
from .streaming import StreamingToneNet
from .watermark import embed_watermark, detect_watermark, verify_watermark
from .replay import save_trace, replay_trace, TraceRecorder
from .token_lm import TokenLanguageModel, StreamingLM
from .orchestrator import AudioOrchestrator, AudioPolicy, AudioLedger

__all__ = [
    # Core
    "ToneNetCodec",
    "ToneNetEncoder",
    "HarmonicDecoder",
    "VectorQuantizer",
    "ResidualVectorQuantizer",
    # Layers
    "CausalConv1d",
    "ResidualBlock",
    # Losses
    "MultiResolutionSTFTLoss",
    "MelSpectrogramLoss",
    # Metrics
    "compute_snr",
    "compute_stoi",
    "compute_pesq",
    # Audio
    "AudioCodec",
    "compress_audio",
    "decompress_audio",
    "reconstruct_audio",
    # Pipeline
    "StreamingToneNet",
    "embed_watermark",
    "detect_watermark",
    "verify_watermark",
    "save_trace",
    "replay_trace",
    "TraceRecorder",
    "TokenLanguageModel",
    "StreamingLM",
    "AudioOrchestrator",
    "AudioPolicy",
    "AudioLedger",
    # Training/Deployment
    "ToneNetTrainer",
    "ToneNetDeployment",
    "ClassicalController",
    "export_model",
    "verify_model",
]

