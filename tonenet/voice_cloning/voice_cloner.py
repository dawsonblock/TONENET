"""
Complete voice cloning system built on ToneNet v2.0 codec.

Integrates speaker encoder, text encoder, AR+NAR models, and harmonic codec
into a full text-to-speech with voice cloning pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

from ..codec import ToneNetCodec
from .speaker_encoder import ECAPA_TDNN
from .text_encoder import PhonemeEncoder
from .ar_model import VoiceCloningAR
from .nar_model import VoiceCloningNAR


class ToneNetVoiceCloner(nn.Module):
    """
    Complete voice cloning system built on ToneNet v2.0.
    Integrates speaker encoder, text encoder, AR+NAR models, and harmonic codec.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        codec_num_quantizers: int = 8,
        codec_codebook_size: int = 1024,
        ar_d_model: int = 1024,
        ar_n_layers: int = 12,
        speaker_dim: int = 256,
        text_dim: int = 512,
        freeze_codec: bool = True
    ):
        super().__init__()
        self.sample_rate = sample_rate

        # 1. Speaker encoder (reference audio → speaker embedding)
        self.speaker_encoder = ECAPA_TDNN(embedding_dim=speaker_dim)

        # 2. Text encoder (phonemes → text embeddings)
        self.text_encoder = PhonemeEncoder(d_model=text_dim)

        # 3. Mel spectrogram transform (for speaker encoder)
        if HAS_TORCHAUDIO:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                win_length=1024,
                hop_length=256,
                n_mels=80,
                power=2.0
            )
        else:
            self.mel_transform = None

        # 4. AR language model (generates first quantizer)
        self.ar_model = VoiceCloningAR(
            d_model=ar_d_model,
            n_layers=ar_n_layers,
            codebook_size=codec_codebook_size
        )

        # 5. NAR refinement (generates remaining quantizers)
        self.nar_model = VoiceCloningNAR(
            num_quantizers=codec_num_quantizers - 1,
            codebook_size=codec_codebook_size,
            d_model=512,
            text_dim=text_dim,
            speaker_dim=speaker_dim
        )

        # 6. ToneNet codec (decoder only for inference)
        self.codec = ToneNetCodec(
            sample_rate=sample_rate,
            num_quantizers=codec_num_quantizers,
            codebook_size=codec_codebook_size,
            num_harmonics=64
        )

        # Freeze codec (pre-trained, not trained end-to-end)
        if freeze_codec:
            for param in self.codec.parameters():
                param.requires_grad = False

    def extract_speaker(self, reference_audio: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from 3-10 second reference audio.

        Args:
            reference_audio: (B, T) waveform, 24kHz

        Returns:
            speaker_emb: (B, 256) speaker vector
        """
        if self.mel_transform is None:
            raise RuntimeError("torchaudio is required for speaker extraction")

        # Compute mel spectrogram
        mel = self.mel_transform(reference_audio)  # (B, 80, T)
        mel = torch.log(mel + 1e-6)  # Log compression

        # Extract embedding
        speaker_emb = self.speaker_encoder(mel)
        return speaker_emb

    def text_to_phonemes(self, text: str) -> torch.Tensor:
        """
        Convert text to phoneme indices.
        (Simplified - use phonemizer library in production)

        Args:
            text: Input text string

        Returns:
            tokens: (1, T) phoneme indices
        """
        # Placeholder: character-level encoding
        # In production: use phonemizer or g2p library
        tokens = torch.tensor([
            [ord(c) % 256 for c in text]
        ], dtype=torch.long)
        return tokens

    @torch.no_grad()
    def clone_voice(
        self,
        text: str,
        reference_audio: torch.Tensor,
        max_length: int = 1000,
        temperature: float = 0.8
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Clone voice: synthesize text in target speaker's voice.

        Args:
            text: String to synthesize
            reference_audio: (1, T) reference waveform, 3-10 seconds
            max_length: Maximum output length in frames
            temperature: Sampling temperature for AR model

        Returns:
            audio: (1, T_out) synthesized waveform
            info: Dict with intermediate outputs
        """
        self.eval()
        device = reference_audio.device

        # Step 1: Extract speaker embedding
        speaker_emb = self.extract_speaker(reference_audio)

        # Step 2: Encode text
        phonemes = self.text_to_phonemes(text).to(device)
        text_emb = self.text_encoder(phonemes)  # (1, T_text, 512)

        # Step 3: AR generation of first quantizer
        first_codes = self.ar_model.generate(
            phonemes,
            speaker_emb,
            max_length=max_length,
            temperature=temperature
        )

        # Clamp to valid codebook range
        first_codes = torch.clamp(first_codes, 0, self.ar_model.codebook_size - 1)

        # Step 4: NAR generation of remaining quantizers
        remaining_codes = self.nar_model(first_codes, text_emb, speaker_emb)

        # Combine all codes
        all_codes = [first_codes] + remaining_codes

        # Step 5: Decode with ToneNet harmonic codec
        audio = self.codec.decode(all_codes)

        info = {
            'speaker_emb': speaker_emb,
            'first_codes': first_codes,
            'all_codes': all_codes,
            'text_emb': text_emb
        }

        return audio, info

    def forward(
        self,
        text: str,
        reference_audio: torch.Tensor,
        target_audio: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Training forward (for end-to-end training).

        Args:
            text: Text to synthesize
            reference_audio: Reference speaker audio
            target_audio: Target audio for supervision

        Returns:
            total_loss: Combined training loss
            metrics: Dict with individual loss components
        """
        device = reference_audio.device

        # Extract speaker
        speaker_emb = self.extract_speaker(reference_audio)

        # Encode text
        phonemes = self.text_to_phonemes(text).to(device)
        text_emb = self.text_encoder(phonemes)

        # Encode target audio with codec (teacher forcing)
        target_audio_expanded = target_audio.unsqueeze(1) if target_audio.dim() == 2 else target_audio
        with torch.no_grad():
            target_codes = self.codec.encode(target_audio_expanded)

        # Prepare target codes for AR (add BOS)
        B, T = target_codes[0].shape
        bos = torch.full((B, 1), self.ar_model.bos_token, device=device, dtype=torch.long)
        ar_input = torch.cat([bos, target_codes[0][:, :-1]], dim=1)

        # AR loss (first quantizer)
        ar_logits = self.ar_model(phonemes, speaker_emb, ar_input)
        ar_loss = F.cross_entropy(
            ar_logits.reshape(-1, ar_logits.size(-1)),
            target_codes[0].reshape(-1)
        )

        # NAR loss (remaining quantizers)
        nar_codes = self.nar_model(target_codes[0], text_emb, speaker_emb)
        nar_loss = 0.0
        for q, codes in enumerate(nar_codes):
            nar_loss = nar_loss + F.cross_entropy(
                # We need logits for loss, so recompute
                codes.float().unsqueeze(-1).expand(-1, -1, self.ar_model.codebook_size),
                target_codes[q + 1]
            )

        total_loss = ar_loss + nar_loss

        return total_loss, {
            'ar_loss': ar_loss.item(),
            'nar_loss': nar_loss.item() if isinstance(nar_loss, torch.Tensor) else nar_loss,
            'total_loss': total_loss.item()
        }
