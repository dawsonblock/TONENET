"""
ToneNet Voice Cloning - Extension for text-to-speech with voice cloning.

This subpackage adds:
- ECAPA-TDNN speaker encoder for voice identity extraction
- Phoneme encoder for text processing
- AR/NAR models for code generation
- Complete voice cloning pipeline

Example:
    >>> from tonenet.voice_cloning import ToneNetVoiceCloner
    >>> import torch
    >>> 
    >>> cloner = ToneNetVoiceCloner()
    >>> ref_audio = torch.randn(1, 24000 * 5)  # 5 second reference
    >>> audio, info = cloner.clone_voice("Hello world", ref_audio)
"""

from .speaker_encoder import ECAPA_TDNN, SE_Res2Block, SqueezeExcitation, AttentiveStatisticsPooling
from .text_encoder import PhonemeEncoder, PositionalEncoding
from .ar_model import VoiceCloningAR
from .nar_model import VoiceCloningNAR
from .voice_cloner import ToneNetVoiceCloner

__all__ = [
    # Speaker
    "ECAPA_TDNN",
    "SE_Res2Block",
    "SqueezeExcitation",
    "AttentiveStatisticsPooling",
    # Text
    "PhonemeEncoder",
    "PositionalEncoding",
    # Models
    "VoiceCloningAR",
    "VoiceCloningNAR",
    # Pipeline
    "ToneNetVoiceCloner",
]
