"""
ToneNet v2.0 - Neural Audio Codec & Real-Time Voice Agent

Organized into subpackages:
- core: Neural audio codec (VQ-VAE)
- stt: Speech-to-text backends
- tts: Text-to-speech backends
- vad: Voice activity detection
- pipeline: Real-time duplex pipelines
- agent: Reasoning and planning
- identity: Speaker identification
"""

__version__ = "2.0.0"

# Core codec
from .core import (
    ToneNetCodec,
    ToneNetEncoder,
    HarmonicDecoder,
    VectorQuantizer,
    ResidualVectorQuantizer,
    MultiResolutionSTFTLoss,
    AudioCodec,
    compress_audio,
    decompress_audio,
)

# STT backends
from .stt import (
    FasterWhisperSTT,
    STTBackendConfig,
    create_stt,
    MockSTTBackend,
    WhisperCppSTT,
    WhisperCppConfig,
    MockWhisperCppSTT,
)

# TTS backends
from .tts import (
    PiperTTS,
    XTTSTTS,
    TTSBackendConfig,
    create_tts,
    MockTTSBackend,
    XTTSEngine,
)

# VAD
from .vad import (
    VADSegmenter,
    UtteranceSegment,
    SimulatedVADSegmenter,
    WebRTCVADGate,
    WebRTCVADConfig,
    MockWebRTCVAD,
)

# Pipeline
from .pipeline import (
    MicStream,
    AudioOutput,
    AudioIOConfig,
    MockMicStream,
    AudioPlayer,
    RealtimeDuplex,
    DuplexConfig,
    DuplexRunner,
    run_duplex,
    Turn,
    RealtimeVoiceAgent,
    RealtimeAgentConfig,
    create_realtime_agent,
    JsonlLedger,
    sentence_chunks,
    split_text_to_sentences,
)

# Agent
from .agent import (
    EchoReasoner,
    LLMReasoner,
    NeuralReasoner,
    ReasonerConfig,
    create_reasoner,
    VoiceAgentPlanner,
    SemanticMemoryGraph,
)

# Identity
from .identity import (
    IdentityGuard,
    SpeakerProfile,
)

# Streaming & orchestration
from .streaming import StreamingToneNet
from .orchestrator_api import AudioOrchestrator

# Utilities
from .watermark import embed_watermark, detect_watermark
from .improve import SelfImprovingSystem, AdaptiveVoiceAgent
from .mesh import AudioMeshNode, MeshCoordinator

__all__ = [
    # Core
    "ToneNetCodec",
    "ToneNetEncoder",
    "HarmonicDecoder",
    "VectorQuantizer",
    "ResidualVectorQuantizer",
    "MultiResolutionSTFTLoss",
    "AudioCodec",
    "compress_audio",
    "decompress_audio",
    # STT
    "FasterWhisperSTT",
    "STTBackendConfig",
    "create_stt",
    "MockSTTBackend",
    "WhisperCppSTT",
    "WhisperCppConfig",
    "MockWhisperCppSTT",
    # TTS
    "PiperTTS",
    "XTTSTTS",
    "TTSBackendConfig",
    "create_tts",
    "MockTTSBackend",
    "XTTSEngine",
    # VAD
    "VADSegmenter",
    "UtteranceSegment",
    "SimulatedVADSegmenter",
    "WebRTCVADGate",
    "WebRTCVADConfig",
    "MockWebRTCVAD",
    # Pipeline
    "MicStream",
    "AudioOutput",
    "AudioIOConfig",
    "MockMicStream",
    "AudioPlayer",
    "RealtimeDuplex",
    "DuplexConfig",
    "DuplexRunner",
    "run_duplex",
    "Turn",
    "RealtimeVoiceAgent",
    "RealtimeAgentConfig",
    "create_realtime_agent",
    "JsonlLedger",
    "sentence_chunks",
    "split_text_to_sentences",
    # Agent
    "EchoReasoner",
    "LLMReasoner",
    "NeuralReasoner",
    "ReasonerConfig",
    "create_reasoner",
    "VoiceAgentPlanner",
    "SemanticMemoryGraph",
    # Identity
    "IdentityGuard",
    "SpeakerProfile",
    # Streaming
    "StreamingToneNet",
    "AudioOrchestrator",
    # Utils
    "embed_watermark",
    "detect_watermark",
    "SelfImprovingSystem",
    "AdaptiveVoiceAgent",
    "AudioMeshNode",
    "MeshCoordinator",
]
