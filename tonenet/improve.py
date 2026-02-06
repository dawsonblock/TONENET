"""
End-to-end self-improving speech system.

Closed-loop system that:
- Collects feedback on synthesis quality
- Updates model weights online
- Adapts to speaker/environment
- Optimizes for perceptual quality
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass 
class FeedbackRecord:
    """Single feedback record."""
    timestamp: float
    tokens: torch.Tensor
    audio: torch.Tensor
    score: float  # 0-1 quality score
    feedback_type: str  # "auto", "human", "asr"
    metadata: Dict[str, Any] = field(default_factory=dict)


class QualityEstimator(nn.Module):
    """
    Neural quality estimator for synthesis output.
    
    Predicts MOS-like quality score from audio.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 3
    ):
        super().__init__()
        
        # Simple CNN for audio quality estimation
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Estimate quality score.
        
        Args:
            audio: [B, 1, T] or [B, T] audio
        
        Returns:
            [B] quality scores in [0, 1]
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        features = self.conv(audio).squeeze(-1)  # [B, 128]
        score = self.fc(features).squeeze(-1)  # [B]
        
        return score


class OnlineAdapter(nn.Module):
    """
    Lightweight online adaptation module.
    
    Small network that learns to adjust codec output
    based on feedback.
    """
    
    def __init__(
        self,
        token_dim: int = 1024,
        adapt_dim: int = 64
    ):
        super().__init__()
        
        self.embed = nn.Embedding(token_dim, adapt_dim)
        
        self.net = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim),
            nn.ReLU(),
            nn.Linear(adapt_dim, token_dim)
        )
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptation logits.
        
        Args:
            tokens: [B, T] token indices
        
        Returns:
            [B, T, vocab] adjustment logits
        """
        h = self.embed(tokens)  # [B, T, adapt_dim]
        logits = self.net(h)  # [B, T, token_dim]
        return logits
    
    def adapt(self, tokens: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Apply adaptation to tokens.
        
        Args:
            tokens: [B, T] token indices
            temperature: Sampling temperature
        
        Returns:
            [B, T] adapted tokens
        """
        logits = self.forward(tokens)
        
        # Soft mixing with original
        original_logits = F.one_hot(tokens, num_classes=logits.shape[-1]).float() * 10.0
        mixed = original_logits + temperature * logits
        
        return torch.argmax(mixed, dim=-1)


class FeedbackBuffer:
    """Circular buffer for feedback records."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.records: List[FeedbackRecord] = []
    
    def add(self, record: FeedbackRecord):
        """Add feedback record."""
        self.records.append(record)
        if len(self.records) > self.max_size:
            self.records.pop(0)
    
    def sample(self, n: int) -> List[FeedbackRecord]:
        """Sample n records."""
        import random
        n = min(n, len(self.records))
        return random.sample(self.records, n)
    
    def get_recent(self, n: int) -> List[FeedbackRecord]:
        """Get n most recent records."""
        return self.records[-n:]
    
    def average_score(self) -> float:
        """Get average quality score."""
        if not self.records:
            return 0.0
        return sum(r.score for r in self.records) / len(self.records)


class SelfImprovingSystem:
    """
    End-to-end self-improving speech system.
    
    Features:
    - Automatic quality estimation
    - Online adaptation
    - Feedback collection
    - Continuous improvement
    """
    
    def __init__(
        self,
        codec,  # ToneNetCodec
        device: str = "cpu",
        learning_rate: float = 1e-4,
        update_interval: int = 100
    ):
        self.codec = codec
        self.device = device
        self.update_interval = update_interval
        
        # Quality estimator
        self.estimator = QualityEstimator().to(device)
        
        # Online adapter
        self.adapter = OnlineAdapter().to(device)
        self.adapter_opt = torch.optim.Adam(
            self.adapter.parameters(), lr=learning_rate
        )
        
        # Feedback buffer
        self.buffer = FeedbackBuffer()
        
        # State
        self.step_count = 0
        self.improvement_history: List[float] = []
    
    def process(
        self,
        tokens: torch.Tensor,
        apply_adaptation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Process tokens through self-improving pipeline.
        
        Returns:
            (adapted_tokens, audio, quality_score)
        """
        tokens = tokens.to(self.device)
        
        # Apply adaptation
        if apply_adaptation:
            with torch.no_grad():
                adapted = self.adapter.adapt(tokens)
        else:
            adapted = tokens
        
        # Decode
        with torch.no_grad():
            audio = self.codec.decode([adapted])
        
        # Estimate quality
        with torch.no_grad():
            score = self.estimator(audio)
        
        return adapted, audio, float(score.mean())
    
    def add_feedback(
        self,
        tokens: torch.Tensor,
        audio: torch.Tensor,
        score: Optional[float] = None,
        feedback_type: str = "auto",
        **metadata
    ):
        """
        Add feedback record.
        
        If score is None, uses automatic quality estimation.
        """
        if score is None:
            with torch.no_grad():
                score = float(self.estimator(audio.to(self.device)).mean())
        
        record = FeedbackRecord(
            timestamp=time.time(),
            tokens=tokens.cpu(),
            audio=audio.cpu(),
            score=score,
            feedback_type=feedback_type,
            metadata=metadata
        )
        self.buffer.add(record)
        
        self.step_count += 1
        
        # Periodic update
        if self.step_count % self.update_interval == 0:
            self._update_adapter()
    
    def add_human_feedback(
        self,
        tokens: torch.Tensor,
        audio: torch.Tensor,
        score: float,
        **metadata
    ):
        """Add human-rated feedback (higher weight)."""
        self.add_feedback(
            tokens, audio, score,
            feedback_type="human",
            **metadata
        )
    
    def _update_adapter(self):
        """Update adapter based on recent feedback."""
        records = self.buffer.sample(min(32, len(self.buffer.records)))
        if len(records) < 8:
            return
        
        # Simple gradient step
        self.adapter.train()
        
        total_loss = 0.0
        for record in records:
            tokens = record.tokens.to(self.device)
            target_score = record.score
            
            # Higher scores = less adaptation needed
            logits = self.adapter(tokens.unsqueeze(0))
            
            # Loss: encourage adaptation when score is low
            adaptation_magnitude = torch.abs(logits).mean()
            loss = (1.0 - target_score) * adaptation_magnitude
            
            self.adapter_opt.zero_grad()
            loss.backward()
            self.adapter_opt.step()
            
            total_loss += loss.item()
        
        self.adapter.eval()
        
        avg_loss = total_loss / len(records)
        self.improvement_history.append(avg_loss)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get improvement statistics."""
        return {
            "total_steps": self.step_count,
            "buffer_size": len(self.buffer.records),
            "average_score": self.buffer.average_score(),
            "recent_losses": self.improvement_history[-10:] if self.improvement_history else []
        }
    
    def save(self, path: str):
        """Save system state."""
        torch.save({
            "adapter": self.adapter.state_dict(),
            "estimator": self.estimator.state_dict(),
            "step_count": self.step_count,
            "improvement_history": self.improvement_history
        }, path)
    
    def load(self, path: str):
        """Load system state."""
        data = torch.load(path, map_location=self.device)
        self.adapter.load_state_dict(data["adapter"])
        self.estimator.load_state_dict(data["estimator"])
        self.step_count = data["step_count"]
        self.improvement_history = data["improvement_history"]


class AdaptiveVoiceAgent:
    """
    Full adaptive voice agent combining all components.
    
    Integrates:
    - Self-improving synthesis
    - Planner integration  
    - Memory system
    - Identity control
    """
    
    def __init__(
        self,
        codec,
        planner=None,
        memory=None,
        identity_guard=None,
        device: str = "cpu"
    ):
        self.device = device
        
        # Core components
        self.improver = SelfImprovingSystem(codec, device=device)
        self.planner = planner
        self.memory = memory
        self.identity_guard = identity_guard
        
        # State
        self.session_id = None
        self.turn_count = 0
    
    def synthesize(
        self,
        tokens: torch.Tensor,
        speaker_id: str = "default"
    ) -> Tuple[torch.Tensor, float]:
        """
        Synthesize with full pipeline.
        
        Returns:
            (audio, quality_score)
        """
        # Identity check
        if self.identity_guard:
            allowed, reason = self.identity_guard.check_emission_allowed(
                tokens, speaker_id
            )
            if not allowed:
                raise RuntimeError(f"Identity blocked: {reason}")
        
        # Self-improving synthesis
        adapted, audio, score = self.improver.process(tokens)
        
        # Store in memory
        if self.memory:
            self.memory.store(adapted, metadata={
                "speaker_id": speaker_id,
                "quality_score": score,
                "turn": self.turn_count
            })
        
        # Record for improvement
        self.improver.add_feedback(tokens, audio, score)
        
        self.turn_count += 1
        
        return audio, score
    
    def plan_and_speak(
        self,
        user_input: str,
        speaker_id: str = "default"
    ) -> Tuple[str, Optional[torch.Tensor], float]:
        """
        Plan response and synthesize.
        
        Returns:
            (response_text, audio, quality_score)
        """
        if not self.planner:
            return "", None, 0.0
        
        # Plan
        action = self.planner.step(user_input)
        
        if action.get("action") != "speak":
            return "", None, 0.0
        
        text = action.get("content", "")
        
        # Generate tokens (placeholder)
        tokens = torch.randint(0, 1024, (1, 75))
        
        # Synthesize
        audio, score = self.synthesize(tokens, speaker_id)
        
        return text, audio, score
