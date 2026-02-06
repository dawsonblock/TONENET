"""
Speech Reasoner module.

Lightweight transformation layer between STT and TTS.
Can be extended to:
- Intent classification
- Tool routing
- LLM integration
- Memory/context injection
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch
import torch.nn as nn


class ReasonerProtocol(Protocol):
    """Reasoner interface."""
    def respond(self, text: str) -> str: ...


@dataclass
class ReasonerConfig:
    """Reasoner configuration."""
    mode: str = "echo"              # echo | llm | neural
    system_prompt: str = "You are a helpful voice assistant."
    max_tokens: int = 150
    
    # LLM backend (for mode=llm)
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None


class EchoReasoner:
    """Simple echo for testing."""
    
    def __init__(self, cfg: ReasonerConfig | None = None):
        self.cfg = cfg or ReasonerConfig()
    
    def respond(self, text: str) -> str:
        return f"You said: {text}"


class NeuralReasoner(nn.Module):
    """
    Lightweight neural transformation in token space.
    
    Can be trained for specific tasks like:
    - Intent classification
    - Response routing
    - Style transfer
    """
    
    def __init__(self, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens.float())


class LLMReasoner:
    """
    LLM-based reasoner using OpenAI-compatible API.
    
    Example:
        reasoner = LLMReasoner(ReasonerConfig(
            llm_api_key="sk-...",
            llm_model="gpt-4o-mini"
        ))
        response = reasoner.respond("Hello!")
    """
    
    def __init__(self, cfg: ReasonerConfig):
        self.cfg = cfg
        self._client = None
        self._history: list[dict[str, str]] = []
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai not installed. Run: pip install openai")
            
            self._client = OpenAI(
                api_key=self.cfg.llm_api_key,
                base_url=self.cfg.llm_base_url,
            )
        return self._client
    
    def respond(self, text: str) -> str:
        client = self._get_client()
        
        # Add user message
        self._history.append({"role": "user", "content": text})
        
        # Build messages
        messages = [{"role": "system", "content": self.cfg.system_prompt}]
        messages.extend(self._history[-10:])  # Keep last 10 turns
        
        response = client.chat.completions.create(
            model=self.cfg.llm_model,
            messages=messages,
            max_tokens=self.cfg.max_tokens,
        )
        
        reply = response.choices[0].message.content or ""
        
        # Add to history
        self._history.append({"role": "assistant", "content": reply})
        
        return reply
    
    def clear_history(self):
        """Clear conversation history."""
        self._history = []


class CallbackReasoner:
    """Reasoner that uses a custom callback function."""
    
    def __init__(self, callback: Callable[[str], str]):
        self.callback = callback
    
    def respond(self, text: str) -> str:
        return self.callback(text)


def create_reasoner(
    cfg: ReasonerConfig | None = None,
    callback: Callable[[str], str] | None = None
) -> ReasonerProtocol:
    """
    Factory to create reasoner.
    
    Args:
        cfg: Configuration
        callback: Custom callback (overrides cfg.mode)
    
    Returns:
        Reasoner instance
    """
    if callback:
        return CallbackReasoner(callback)
    
    cfg = cfg or ReasonerConfig()
    
    if cfg.mode == "llm":
        return LLMReasoner(cfg)
    else:
        return EchoReasoner(cfg)
