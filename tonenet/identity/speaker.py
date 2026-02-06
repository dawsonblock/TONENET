"""
Voice identity control and cloning guard.

Uses pretrained ECAPA-TDNN speaker embeddings for real verification.

Provides:
- Speaker embedding extraction (pretrained or mock)
- Voice verification with real thresholds
- Cloning detection and prevention
- Identity locking for safety

Install: pip install speechbrain (for real embeddings)
"""

import time
from typing import Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpeakerProfile:
    """Stored speaker identity profile."""
    id: str
    name: str
    embedding: torch.Tensor
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    locked: bool = False  # If locked, cannot be cloned


class PretrainedSpeakerEmbedder(nn.Module):
    """
    Speaker embedding using pretrained ECAPA-TDNN from SpeechBrain.
    
    This provides real, meaningful speaker embeddings trained on VoxCeleb.
    """
    
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self._model = None
        self.embed_dim = 192  # ECAPA-TDNN output dim
        self._sample_rate = 16000
    
    def _load_model(self):
        """Lazy load the pretrained model."""
        if self._model is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self._model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa",
                    run_opts={"device": self.device}
                )
            except ImportError:
                raise ImportError(
                    "speechbrain not installed. Run: pip install speechbrain"
                )
        return self._model
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from raw audio.
        
        Args:
            audio: Raw audio tensor [B, T] at 16kHz
        
        Returns:
            [B, 192] speaker embedding
        """
        model = self._load_model()
        
        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # [B, 1, T] -> [B, T]
        
        # Get embeddings
        with torch.no_grad():
            embeddings = model.encode_batch(audio.to(self.device))
        
        return F.normalize(embeddings.squeeze(1), dim=-1)
    
    def embed_file(self, path: str) -> torch.Tensor:
        """Embed audio from file."""
        model = self._load_model()
        with torch.no_grad():
            embedding = model.encode_batch(
                model.load_audio(path).unsqueeze(0)
            )
        return F.normalize(embedding.squeeze(), dim=-1)


class MockSpeakerEmbedder(nn.Module):
    """
    Mock speaker embedder for testing without speechbrain.
    
    Uses deterministic hashing to produce consistent embeddings.
    """
    
    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.embed_dim = embed_dim
        self._sample_rate = 16000
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Generate deterministic embedding from audio."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        batch_size = audio.shape[0]
        
        # Deterministic: use audio statistics as seed
        embeddings = []
        for i in range(batch_size):
            # Use audio content to generate consistent embedding
            seed = int(audio[i].abs().sum().item() * 1000) % (2**31)
            torch.manual_seed(seed)
            emb = torch.randn(self.embed_dim)
            embeddings.append(F.normalize(emb, dim=0))
        
        return torch.stack(embeddings)


def get_speaker_embedder(
    use_pretrained: bool = True,
    device: str = "cpu"
) -> nn.Module:
    """
    Factory to get speaker embedder.
    
    Args:
        use_pretrained: Use pretrained ECAPA-TDNN (requires speechbrain)
        device: Device for model
    
    Returns:
        Speaker embedder module
    """
    if use_pretrained:
        try:
            return PretrainedSpeakerEmbedder(device=device)
        except ImportError:
            print("Warning: speechbrain not available, using mock embedder")
            return MockSpeakerEmbedder()
    return MockSpeakerEmbedder()


# Alias for backward compatibility
SpeakerEmbedder = MockSpeakerEmbedder


class IdentityGuard:
    """
    Voice identity control and cloning prevention.
    
    Uses pretrained speaker embeddings for real verification.
    
    Features:
    - Speaker registration and verification
    - Cloning detection
    - Identity locking
    
    Example:
        guard = IdentityGuard(use_pretrained=True)
        guard.register_speaker("user1", "Alice", audio_tensor)
        is_match, score = guard.verify_speaker("user1", test_audio)
    """
    
    # Real-world thresholds based on ECAPA-TDNN performance
    DEFAULT_SIMILARITY_THRESHOLD = 0.25  # EER threshold
    DEFAULT_CLONE_THRESHOLD = 0.70  # High confidence clone
    
    def __init__(
        self,
        use_pretrained: bool = True,
        device: str = "cpu",
        similarity_threshold: float | None = None,
        clone_alert_threshold: float | None = None
    ):
        self.embedder = get_speaker_embedder(use_pretrained, device)
        self.profiles: dict[str, SpeakerProfile] = {}
        
        # Use validated thresholds
        self.similarity_threshold = (
            similarity_threshold 
            if similarity_threshold is not None 
            else self.DEFAULT_SIMILARITY_THRESHOLD
        )
        self.clone_alert_threshold = (
            clone_alert_threshold
            if clone_alert_threshold is not None
            else self.DEFAULT_CLONE_THRESHOLD
        )
        
        # Locked speakers (cannot be cloned)
        self.locked_ids: set[str] = set()
        
        # Audit log
        self.audit_log: list[dict[str, Any]] = []
    
    def _log(self, event: str, **data):
        """Log audit event."""
        self.audit_log.append({
            "timestamp": time.time(),
            "event": event,
            **data
        })
    
    def register_speaker(
        self,
        speaker_id: str,
        name: str,
        reference_audio: torch.Tensor,
        locked: bool = False,
        **metadata
    ) -> SpeakerProfile:
        """
        Register new speaker identity from audio.
        
        Args:
            speaker_id: Unique speaker ID
            name: Display name
            reference_audio: Reference audio [1, T] at 16kHz
            locked: Prevent cloning of this speaker
            **metadata: Additional metadata
        
        Returns:
            Speaker profile
        """
        with torch.no_grad():
            embedding = self.embedder(reference_audio).squeeze(0)
        
        profile = SpeakerProfile(
            id=speaker_id,
            name=name,
            embedding=embedding,
            created_at=time.time(),
            metadata=metadata,
            locked=locked
        )
        
        self.profiles[speaker_id] = profile
        
        if locked:
            self.locked_ids.add(speaker_id)
        
        self._log("speaker_registered", speaker_id=speaker_id, locked=locked)
        
        return profile
    
    def verify_speaker(
        self,
        speaker_id: str,
        audio: torch.Tensor
    ) -> tuple[bool, float]:
        """
        Verify speaker identity from audio.
        
        Args:
            speaker_id: Claimed speaker ID
            audio: Test audio [1, T] at 16kHz
        
        Returns:
            (is_match, similarity_score)
        """
        if speaker_id not in self.profiles:
            return False, 0.0
        
        profile = self.profiles[speaker_id]
        
        with torch.no_grad():
            embedding = self.embedder(audio).squeeze(0)
        
        similarity = F.cosine_similarity(
            embedding.unsqueeze(0),
            profile.embedding.unsqueeze(0)
        ).item()
        
        is_match = similarity >= self.similarity_threshold
        
        self._log(
            "speaker_verified",
            speaker_id=speaker_id,
            match=is_match,
            similarity=round(similarity, 4)
        )
        
        return is_match, similarity
    
    def detect_clone_attempt(
        self,
        audio: torch.Tensor
    ) -> list[tuple[str, float]]:
        """
        Detect if audio attempts to clone a locked speaker.
        
        Args:
            audio: Test audio
        
        Returns:
            List of (speaker_id, similarity) for potential clones
        """
        if not self.locked_ids:
            return []
        
        with torch.no_grad():
            embedding = self.embedder(audio).squeeze(0)
        
        alerts = []
        for speaker_id in self.locked_ids:
            profile = self.profiles.get(speaker_id)
            if not profile:
                continue
            
            similarity = F.cosine_similarity(
                embedding.unsqueeze(0),
                profile.embedding.unsqueeze(0)
            ).item()
            
            if similarity >= self.clone_alert_threshold:
                alerts.append((speaker_id, similarity))
                self._log(
                    "clone_attempt_detected",
                    speaker_id=speaker_id,
                    similarity=round(similarity, 4)
                )
        
        return alerts
    
    def check_emission_allowed(
        self,
        audio: torch.Tensor,
        claimed_speaker: str
    ) -> tuple[bool, str]:
        """
        Check if emission is allowed (not cloning locked speaker).
        
        Returns:
            (allowed, reason)
        """
        # Check for clone attempts
        clone_alerts = self.detect_clone_attempt(audio)
        
        for speaker_id, sim in clone_alerts:
            if speaker_id != claimed_speaker:
                return False, f"clone_blocked:{speaker_id}"
        
        # Verify claimed identity if registered
        if claimed_speaker in self.profiles:
            is_match, sim = self.verify_speaker(claimed_speaker, audio)
            if not is_match:
                return False, f"identity_mismatch:{sim:.3f}"
        
        return True, "ok"


class VoiceMorpher:
    """
    Voice morphing between speaker identities.
    
    Note: This is a placeholder interface. Real morphing requires
    either a voice conversion model or embedding-conditioned codec.
    """
    
    def __init__(self, guard: IdentityGuard):
        self.guard = guard
    
    def morph_embedding(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Interpolate between speaker embeddings.
        
        Args:
            source_embedding: Source speaker embedding
            target_embedding: Target speaker embedding
            alpha: Morph factor (0=source, 1=target)
        
        Returns:
            Morphed embedding
        """
        alpha = max(0.0, min(1.0, alpha))
        morphed = (1 - alpha) * source_embedding + alpha * target_embedding
        return F.normalize(morphed, dim=-1)
    
    def can_morph_to(self, target_id: str) -> tuple[bool, str]:
        """Check if morphing to target is allowed."""
        if target_id in self.guard.locked_ids:
            return False, f"Target speaker {target_id} is locked"
        if target_id not in self.guard.profiles:
            return False, f"Target speaker {target_id} not registered"
        return True, "ok"
