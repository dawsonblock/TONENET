<p align="center">
  <h1 align="center">🎵 ToneNet v2.0</h1>
  <p align="center">
    <strong>Neural Audio Codec with Harmonic Modeling & Voice Agent System</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#voice-agent">Voice Agent</a>
  </p>
</p>

---

## Features

| Feature | Description |
|---------|-------------|
| 🎯 **Ultra-Low Bitrate** | 0.75–6 kbps with variable quantization |
| 🌊 **Harmonic Decoder** | Interpretable synthesis (f0 + harmonics + phases) |
| ⚡ **Streaming Ready** | Causal encoder, 75 Hz frame rate |
| 🔐 **Identity Control** | Voice cloning guard + watermarking |
| 🧠 **Self-Improving** | Online quality adaptation |
| 🌐 **Multi-Agent Mesh** | Networked voice coordination |

---

## Installation

```bash
git clone https://github.com/dawsonblock/TONENET.git
cd TONENET
pip install -e .
```

**Requirements:** Python 3.9+, PyTorch 2.0+

---

## Quick Start

### Basic Codec

```python
from tonenet import ToneNetCodec
import torch

codec = ToneNetCodec()
audio = torch.randn(1, 1, 24000)  # 1 second @ 24kHz

# Encode → Decode
codes = codec.encode(audio)
reconstructed = codec.decode(codes)

# Variable bitrate
codes_low = codec.encode(audio, n_quantizers=1)   # 0.75 kbps
codes_high = codec.encode(audio, n_quantizers=8)  # 6.0 kbps
```

### Audio File Compression

```python
from tonenet import compress_audio, decompress_audio

compress_audio("input.wav", "compressed.tnc", n_quantizers=4)
decompress_audio("compressed.tnc", "output.wav")
```

### Streaming Decode

```python
from tonenet import StreamingToneNet

streamer = StreamingToneNet(chunk_frames=5)  # 66ms latency
streamer.push_tokens(tokens)
audio = streamer.pop_audio()
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ToneNet v2.0                             │
├──────────────────┬──────────────────┬───────────────────────────┤
│   CORE CODEC     │     PIPELINE     │      VOICE AGENT          │
├──────────────────┼──────────────────┼───────────────────────────┤
│ encoder.py       │ streaming.py     │ planner.py                │
│ decoder.py       │ watermark.py     │ memory.py                 │
│ quantizer.py     │ replay.py        │ identity.py               │
│ codec.py         │ token_lm.py      │ mesh.py                   │
│ losses.py        │ orchestrator.py  │ improve.py                │
└──────────────────┴──────────────────┴───────────────────────────┘
```

### Core Codec (5.6M params)

- **Encoder**: Causal CNN with strides [5,4,4,4] → 320x downsample → 75 Hz
- **RVQ**: 8 quantizers × 1024 codebook entries
- **Decoder**: Harmonic synthesis (f0, amplitudes, phases, noise)

### Bitrate

| Quantizers | Bitrate | Use Case |
|------------|---------|----------|
| 1 | 0.75 kbps | Ultra-low bandwidth |
| 4 | 3.0 kbps | Voice streaming |
| 8 | 6.0 kbps | High quality |

---

## Voice Agent

Full autonomous voice agent with:

### Planner Integration

```python
from tonenet import VoiceAgentPlanner, LocalPlannerLLM, AudioOrchestrator

planner = LocalPlannerLLM()
orch = AudioOrchestrator()
agent = VoiceAgentPlanner(planner, orch)

result = agent.step("Hello, how are you?")
```

### Semantic Memory

```python
from tonenet import SemanticMemoryGraph

memory = SemanticMemoryGraph()
node_id = memory.store(tokens, {"speaker": "user", "text": "hello"})
similar = memory.search(query_tokens, top_k=5)
```

### Identity Guard

```python
from tonenet import IdentityGuard

guard = IdentityGuard()
guard.register_speaker("operator", "Operator", reference_tokens, locked=True)

# Verify identity
is_match, score = guard.verify_speaker("operator", tokens)

# Detect cloning attempts
alerts = guard.detect_clone_attempt(suspicious_tokens)
```

### Self-Improving System

```python
from tonenet import AdaptiveVoiceAgent, ToneNetCodec

agent = AdaptiveVoiceAgent(codec=ToneNetCodec())
audio, quality = agent.synthesize(tokens, speaker_id="operator")
agent.improver.add_human_feedback(tokens, audio, score=0.9)
```

### Multi-Agent Mesh

```python
from tonenet import AudioMeshNode

node = AudioMeshNode(node_id="agent1", port=7700)
node.start()
node.connect_peer("agent2", "192.168.1.10", 7701)
node.send_tokens(tokens, target_id="agent2")
```

---

## Module Reference

| Module | Classes/Functions |
|--------|-------------------|
| `codec` | `ToneNetCodec` |
| `streaming` | `StreamingToneNet` |
| `watermark` | `embed_watermark`, `detect_watermark` |
| `replay` | `save_trace`, `replay_trace` |
| `token_lm` | `TokenLanguageModel`, `StreamingLM` |
| `orchestrator` | `AudioOrchestrator`, `AudioPolicy`, `AudioLedger` |
| `planner` | `VoiceAgentPlanner`, `LocalPlannerLLM`, `APIPlannerLLM` |
| `memory` | `SemanticMemoryGraph`, `CrossModalMemory` |
| `identity` | `IdentityGuard`, `VoiceMorpher` |
| `mesh` | `AudioMeshNode`, `MeshCoordinator` |
| `improve` | `SelfImprovingSystem`, `AdaptiveVoiceAgent` |

---

## Training

### Token LM (Distributed)

```bash
torchrun --nproc_per_node=2 -m tonenet.train_lm --data tokens.pt --steps 100000
```

### Export

```python
from tonenet.export import export_codec_onnx, export_torchscript

export_codec_onnx("tonenet.onnx")
export_torchscript("tonenet.pt")
```

---

## License

MIT License © 2026 Dawson Block
