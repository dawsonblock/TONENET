"""
Audio pipeline orchestrator.

Binds: tokens → LM refine → gate/ledger → streaming decode → watermark → replay

Deterministic, policy-gated, traceable audio emission system.

IMPORTANT: Codec returns List[Tensor[B,T]] (one per quantizer).
Use tokens.normalize_codes() to handle both formats.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Set
import torch

from .codec import ToneNetCodec
from .streaming import StreamingToneNet
from .watermark import embed_watermark
from .replay import save_trace


class AudioPolicy:
    """Policy enforcement for audio emission."""
    
    def __init__(
        self,
        max_seconds: float = 10.0,
        max_rms: float = 0.2,
        allowed_speakers: Optional[Set[str]] = None
    ):
        self.max_seconds = max_seconds
        self.max_rms = max_rms
        self.allowed_speakers = set(allowed_speakers) if allowed_speakers else None
    
    def check(self, duration_sec: float, speaker_id: str) -> tuple:
        """
        Check if emission is allowed.
        
        Returns:
            (allowed: bool, reason: str)
        """
        if duration_sec > self.max_seconds:
            return False, "duration_exceeded"
        
        if self.allowed_speakers is not None and speaker_id not in self.allowed_speakers:
            return False, "speaker_not_allowed"
        
        return True, "ok"


class AudioLedger:
    """Tamper-evident append-only ledger for audio emissions."""
    
    def __init__(self, path: str = "audio_ledger.jsonl"):
        self.path = Path(path)
        self.prev_hash = "0" * 64
    
    def append(self, record: Dict[str, Any]) -> str:
        """Append record with hash chain."""
        record = record.copy()
        record["prev_hash"] = self.prev_hash
        
        # Compute hash
        h = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest()
        record["hash"] = h
        
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        self.prev_hash = h
        return h


class AudioOrchestrator:
    """
    Main orchestrator for deterministic audio pipeline.
    
    Example:
        orch = AudioOrchestrator(config={"policy": {"allowed_speakers": ["operator"]}})
        audio = orch.emit_tokens(tokens, speaker_id="operator")
    """
    
    def __init__(
        self,
        device: str = "cpu",
        seed: int = 0,
        config: Optional[Dict[str, Any]] = None
    ):
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        
        config = config or {}
        
        # Core codec
        self.codec = ToneNetCodec().to(device).eval()
        
        # Streaming decoder
        self.streamer = StreamingToneNet(
            device=device,
            seed=seed,
            chunk_frames=config.get("chunk_frames", 5)
        )
        
        # Policy
        policy_cfg = config.get("policy", {})
        self.policy = AudioPolicy(
            max_seconds=policy_cfg.get("max_seconds", 10.0),
            max_rms=policy_cfg.get("max_rms", 0.2),
            allowed_speakers=policy_cfg.get("allowed_speakers")
        )
        
        # Ledger
        self.ledger = AudioLedger(
            config.get("ledger_path", "audio_ledger.jsonl")
        )
        
        # Watermark strength
        self.watermark_strength = config.get("watermark_strength", 1e-3)
        
        # Token LM (optional)
        self.lm = None
        lm_path = config.get("token_lm_path")
        if lm_path and Path(lm_path).exists():
            from .token_lm import TokenLanguageModel
            self.lm = TokenLanguageModel().to(device).eval()
            self.lm.load_state_dict(torch.load(lm_path, map_location=device))
    
    def _hash(self, tensor: torch.Tensor) -> str:
        """Compute SHA256 hash of tensor."""
        return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
    
    def _rms(self, audio: torch.Tensor) -> float:
        """Compute RMS of audio."""
        return float(torch.sqrt(torch.mean(audio ** 2)))
    
    def emit_tokens(
        self,
        tokens: torch.Tensor,
        speaker_id: str = "default",
        out_trace: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """
        Emit audio from tokens with full pipeline.
        
        Args:
            tokens: Codec token tensor
            speaker_id: Speaker identifier
            out_trace: Optional path to save replay trace
        
        Returns:
            Audio tensor or None if rejected
        """
        # Estimate duration from tokens
        duration_sec = tokens.shape[-1] / 75.0
        
        # Policy check
        ok, reason = self.policy.check(duration_sec, speaker_id)
        if not ok:
            self.ledger.append({
                "ts": time.time(),
                "status": "rejected",
                "reason": reason,
                "speaker_id": speaker_id
            })
            raise RuntimeError(f"Policy rejected: {reason}")
        
        # Optional LM refinement
        if self.lm is not None:
            with torch.no_grad():
                tokens = self.lm.refine(tokens)
        
        # Decode
        with torch.no_grad():
            if isinstance(tokens, list):
                codes = [t.to(self.device) for t in tokens]
            else:
                codes = [tokens.to(self.device)]
            audio = self.codec.decode(codes)
        
        # Watermark
        audio = embed_watermark(audio, speaker_id, self.watermark_strength)
        
        # RMS clamp
        rms = self._rms(audio)
        if rms > self.policy.max_rms:
            audio = audio * (self.policy.max_rms / rms)
        
        # Log to ledger
        record = {
            "ts": time.time(),
            "status": "emitted",
            "speaker_id": speaker_id,
            "duration_sec": duration_sec,
            "rms": rms,
            "seed": self.seed,
            "token_hash": self._hash(tokens if isinstance(tokens, torch.Tensor) else tokens[0])
        }
        self.ledger.append(record)
        
        # Optional trace save
        if out_trace:
            save_trace(out_trace, tokens, self.seed, record)
        
        return audio
    
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to tokens."""
        with torch.no_grad():
            return self.codec.encode(audio.to(self.device))
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens to audio (without policy/watermark)."""
        with torch.no_grad():
            if isinstance(tokens, list):
                codes = [t.to(self.device) for t in tokens]
            else:
                codes = [tokens.to(self.device)]
            return self.codec.decode(codes)


def load_config(path: str) -> Dict[str, Any]:
    """Load orchestrator config from JSON."""
    with open(path) as f:
        return json.load(f)
