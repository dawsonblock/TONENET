"""
Voice identity control and cloning guard.

Provides:
- Speaker embedding extraction
- Voice morphing between identities
- Cloning detection and prevention
- Identity locking for safety
"""

import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple, Set
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    locked: bool = False  # If locked, cannot be cloned


class SpeakerEmbedder(nn.Module):
    """
    Lightweight speaker embedding extractor.
    
    Produces fixed-dim embedding from audio tokens.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        embed_dim: int = 256,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        self.embed_dim = embed_dim
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from tokens.
        
        Args:
            tokens: [B, T] token indices
        
        Returns:
            [B, embed_dim] speaker embedding
        """
        # One-hot encode
        one_hot = F.one_hot(tokens.long() % 1024, num_classes=1024).float()
        
        # Mean pool over time
        pooled = one_hot.mean(dim=1)  # [B, 1024]
        
        # Project to embedding
        embedding = self.net(pooled)
        
        # L2 normalize
        return F.normalize(embedding, dim=-1)


class IdentityGuard:
    """
    Voice identity control and cloning prevention.
    
    Features:
    - Speaker registration and verification
    - Voice morphing with bounds
    - Cloning detection
    - Identity locking
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        clone_alert_threshold: float = 0.95
    ):
        self.embedder = SpeakerEmbedder()
        self.profiles: Dict[str, SpeakerProfile] = {}
        self.similarity_threshold = similarity_threshold
        self.clone_alert_threshold = clone_alert_threshold
        
        # Locked speakers (cannot be cloned)
        self.locked_ids: Set[str] = set()
        
        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
    
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
        reference_tokens: torch.Tensor,
        locked: bool = False,
        **metadata
    ) -> SpeakerProfile:
        """
        Register new speaker identity.
        
        Args:
            speaker_id: Unique speaker ID
            name: Display name
            reference_tokens: Reference audio tokens
            locked: Prevent cloning of this speaker
            **metadata: Additional metadata
        
        Returns:
            Speaker profile
        """
        with torch.no_grad():
            embedding = self.embedder(reference_tokens.unsqueeze(0)).squeeze(0)
        
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
        tokens: torch.Tensor
    ) -> Tuple[bool, float]:
        """
        Verify speaker identity from tokens.
        
        Returns:
            (is_match, similarity_score)
        """
        if speaker_id not in self.profiles:
            return False, 0.0
        
        profile = self.profiles[speaker_id]
        
        with torch.no_grad():
            embedding = self.embedder(tokens.unsqueeze(0)).squeeze(0)
        
        similarity = F.cosine_similarity(
            embedding.unsqueeze(0),
            profile.embedding.unsqueeze(0)
        ).item()
        
        is_match = similarity >= self.similarity_threshold
        
        self._log(
            "speaker_verified",
            speaker_id=speaker_id,
            match=is_match,
            similarity=similarity
        )
        
        return is_match, similarity
    
    def detect_clone_attempt(
        self,
        tokens: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """
        Detect if tokens attempt to clone a locked speaker.
        
        Returns:
            List of (speaker_id, similarity) for potential clone attempts
        """
        if not self.locked_ids:
            return []
        
        with torch.no_grad():
            embedding = self.embedder(tokens.unsqueeze(0)).squeeze(0)
        
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
                    similarity=similarity
                )
        
        return alerts
    
    def check_emission_allowed(
        self,
        tokens: torch.Tensor,
        claimed_speaker: str
    ) -> Tuple[bool, str]:
        """
        Check if emission is allowed (not cloning locked speaker).
        
        Returns:
            (allowed, reason)
        """
        # Check for clone attempts
        clone_alerts = self.detect_clone_attempt(tokens)
        
        for speaker_id, sim in clone_alerts:
            if speaker_id != claimed_speaker:
                return False, f"clone_blocked:{speaker_id}"
        
        # Verify claimed identity if registered
        if claimed_speaker in self.profiles:
            is_match, sim = self.verify_speaker(claimed_speaker, tokens)
            if not is_match:
                return False, f"identity_mismatch:{sim:.3f}"
        
        return True, "ok"


class VoiceMorpher:
    """
    Voice morphing between speaker identities.
    
    Applies controlled identity transformation with bounds.
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
    
    def apply_to_tokens(
        self,
        tokens: torch.Tensor,
        source_id: str,
        target_id: str,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Apply voice morphing to tokens.
        
        Note: This is a placeholder. Full implementation would
        modify token distribution based on speaker embeddings.
        """
        # Check if target is locked
        if target_id in self.guard.locked_ids:
            raise ValueError(f"Cannot morph to locked speaker: {target_id}")
        
        source = self.guard.profiles.get(source_id)
        target = self.guard.profiles.get(target_id)
        
        if not source or not target:
            return tokens
        
        # Placeholder: simple token offset based on morph
        morphed_emb = self.morph_embedding(
            source.embedding,
            target.embedding,
            alpha
        )
        
        # Would apply actual token transformation here
        # For now, add small deterministic offset
        offset = int(alpha * 10) % 1024
        morphed_tokens = (tokens + offset) % 1024
        
        return morphed_tokens
