"""
Planner LLM integration for autonomous voice agent.

Bridges LLM planners to audio pipeline with:
- Intent extraction
- Response generation
- Action planning
- Deterministic execution
"""

import json
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import torch

from .orchestrator import AudioOrchestrator


@dataclass
class PlannerAction:
    """Single planner action."""
    action_type: str  # "speak", "listen", "wait", "query"
    content: str = ""
    speaker_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerState:
    """Current planner state."""
    turn: int = 0
    context: List[str] = field(default_factory=list)
    memory_refs: List[str] = field(default_factory=list)
    pending_actions: List[PlannerAction] = field(default_factory=list)


class BasePlannerLLM:
    """
    Base class for planner LLM integration.
    Override `generate` for specific LLM backends.
    """
    
    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or self._default_prompt()
    
    def _default_prompt(self) -> str:
        return """You are a voice agent planner. Generate actions in JSON format:
{"action": "speak", "content": "...", "speaker": "operator"}
{"action": "listen"}
{"action": "wait", "duration": 1.0}
{"action": "query", "question": "..."}

Be concise. Output one action per response."""
    
    def generate(self, context: str, **kwargs) -> str:
        """Override with actual LLM call."""
        raise NotImplementedError
    
    def parse_action(self, response: str) -> Optional[PlannerAction]:
        """Parse LLM response into action."""
        try:
            # Try JSON parse
            data = json.loads(response.strip())
            return PlannerAction(
                action_type=data.get("action", "speak"),
                content=data.get("content", ""),
                speaker_id=data.get("speaker", "default"),
                metadata=data
            )
        except json.JSONDecodeError:
            # Fallback: treat as speech content
            return PlannerAction(
                action_type="speak",
                content=response.strip()
            )


class LocalPlannerLLM(BasePlannerLLM):
    """
    Local transformer-based planner.
    Uses lightweight model for low-latency planning.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.device = device
        self.model = None
        self.tokenizer = None
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, path: str):
        """Load local model (placeholder for actual loading)."""
        # Would load actual model here
        pass
    
    def generate(self, context: str, max_tokens: int = 100) -> str:
        """Generate response from context."""
        if self.model is None:
            # Fallback for demo
            return json.dumps({
                "action": "speak",
                "content": "I understand. Processing your request.",
                "speaker": "operator"
            })
        
        # Actual generation would go here
        return ""


class APIPlannerLLM(BasePlannerLLM):
    """
    API-backed planner (OpenAI, Anthropic, etc.)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.openai.com/v1/chat/completions",
        model: str = "gpt-4",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
    
    def generate(self, context: str, max_tokens: int = 100) -> str:
        """Call API for generation."""
        if not self.api_key:
            # Fallback
            return json.dumps({
                "action": "speak",
                "content": "API key not configured.",
                "speaker": "operator"
            })
        
        try:
            import requests
            response = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": context}
                    ],
                    "max_tokens": max_tokens
                }
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return json.dumps({"action": "speak", "content": f"Error: {e}"})


class VoiceAgentPlanner:
    """
    Full voice agent with LLM planner integration.
    
    Handles: listen → plan → speak loop with deterministic execution.
    """
    
    def __init__(
        self,
        planner: BasePlannerLLM,
        orchestrator: AudioOrchestrator,
        max_context: int = 10
    ):
        self.planner = planner
        self.orch = orchestrator
        self.state = PlannerState()
        self.max_context = max_context
        
        # Action handlers
        self.handlers: Dict[str, Callable] = {
            "speak": self._handle_speak,
            "listen": self._handle_listen,
            "wait": self._handle_wait,
            "query": self._handle_query,
        }
    
    def add_context(self, text: str, role: str = "user"):
        """Add to conversation context."""
        self.state.context.append(f"{role}: {text}")
        if len(self.state.context) > self.max_context:
            self.state.context.pop(0)
    
    def get_context_string(self) -> str:
        """Build context string for planner."""
        return "\n".join(self.state.context)
    
    def plan_next(self) -> PlannerAction:
        """Get next action from planner."""
        context = self.get_context_string()
        response = self.planner.generate(context)
        action = self.planner.parse_action(response)
        return action or PlannerAction(action_type="wait")
    
    def execute(self, action: PlannerAction) -> Dict[str, Any]:
        """Execute a planner action."""
        handler = self.handlers.get(action.action_type, self._handle_unknown)
        return handler(action)
    
    def _handle_speak(self, action: PlannerAction) -> Dict[str, Any]:
        """Handle speak action."""
        # Generate tokens from text (placeholder)
        # In real system: text → phonemes → tokens
        tokens = torch.randint(0, 1024, (1, 75))  # ~1 second
        
        audio = self.orch.emit_tokens(
            tokens,
            speaker_id=action.speaker_id
        )
        
        self.add_context(action.content, role="assistant")
        
        return {
            "action": "speak",
            "content": action.content,
            "audio_length": audio.shape[-1] if audio is not None else 0
        }
    
    def _handle_listen(self, action: PlannerAction) -> Dict[str, Any]:
        """Handle listen action (placeholder)."""
        return {"action": "listen", "status": "waiting"}
    
    def _handle_wait(self, action: PlannerAction) -> Dict[str, Any]:
        """Handle wait action."""
        duration = action.metadata.get("duration", 1.0)
        time.sleep(duration)
        return {"action": "wait", "duration": duration}
    
    def _handle_query(self, action: PlannerAction) -> Dict[str, Any]:
        """Handle query action."""
        return {
            "action": "query",
            "question": action.metadata.get("question", "")
        }
    
    def _handle_unknown(self, action: PlannerAction) -> Dict[str, Any]:
        """Handle unknown action."""
        return {"action": "unknown", "type": action.action_type}
    
    def step(self, user_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Run one step of agent loop.
        
        Args:
            user_input: Optional user message
        
        Returns:
            Action result
        """
        if user_input:
            self.add_context(user_input, role="user")
        
        action = self.plan_next()
        result = self.execute(action)
        
        self.state.turn += 1
        return result
    
    def run_loop(
        self,
        input_fn: Callable[[], Optional[str]],
        max_turns: int = 100
    ):
        """
        Run continuous agent loop.
        
        Args:
            input_fn: Function returning user input or None
            max_turns: Maximum turns
        """
        for _ in range(max_turns):
            user_input = input_fn()
            result = self.step(user_input)
            
            if result.get("action") == "stop":
                break
            
            yield result
