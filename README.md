<p align="center">
  <h1 align="center">🎵 ToneNet v2.0</h1>
  <p align="center">
    <strong>Neural Audio Codec with Harmonic Modeling & Voice Cloning</strong>
  </p>
  <p align="center">
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#voice-cloning">Voice Cloning</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/pytorch-2.0+-red.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/bitrate-0.75--6_kbps-purple.svg" alt="Bitrate">
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Residual VQ** | 8×1024 codebook with EMA updates |
| 📉 **Ultra-low bitrate** | 0.75-6 kbps variable rate |
| 🔊 **Harmonic decoder** | Explicit f0, harmonics, phases for interpretability |
| ⚡ **Streaming-ready** | Causal convolutions, 75 Hz frame rate |
| 🎤 **Voice cloning** | ECAPA-TDNN speaker encoder + AR/NAR generation |
| 📦 **File compression** | Compress/decompress audio files to `.tnc` format |

---

## 🚀 Installation

```bash
git clone https://github.com/dawsonblock/TONENET.git
cd TONENET
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[full]"  # Includes STOI, PESQ metrics + phonemizer
```

---

## 🎬 Quick Start

### Basic Codec

```python
from tonenet import ToneNetCodec
import torch

model = ToneNetCodec()
audio = torch.randn(1, 1, 24000)  # 1 second @ 24kHz

# Encode → discrete codes
codes = model.encode(audio)

# Decode → reconstructed audio
reconstructed = model.decode(codes)

# Full forward with harmonic outputs
recon, outputs = model(audio)
print(f"f0: {outputs['f0'].mean():.1f} Hz")
print(f"Harmonics: {outputs['H'].shape}")
```

### Variable Bitrate

```python
# Trade quality for compression
for n_q in [1, 4, 8]:
    info = model.get_bitrate(n_quantizers=n_q)
    print(f"{n_q} quantizers: {info['bitrate_kbps']:.2f} kbps")
# Output:
# 1 quantizers: 0.75 kbps
# 4 quantizers: 3.00 kbps  
# 8 quantizers: 6.00 kbps
```

### File Compression

```python
from tonenet import AudioCodec

codec = AudioCodec(n_quantizers=4)

# Compress any audio file
codec.compress("input.wav", "compressed.tnc")

# Decompress back to audio
codec.decompress("compressed.tnc", "output.wav")
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ToneNet v2.0                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Audio (24kHz) ──▶ [Causal Encoder] ──▶ Latent (75Hz)      │
│                           │                                 │
│                           ▼                                 │
│                    [RVQ: 8×1024]                           │
│                           │                                 │
│                           ▼                                 │
│                  [Harmonic Decoder]                        │
│                           │                                 │
│            ┌──────────────┼──────────────┐                 │
│            ▼              ▼              ▼                 │
│          f0 (Hz)    Harmonics (64)    Noise                │
│            │              │              │                 │
│            └──────────────┴──────────────┘                 │
│                           │                                 │
│                    [Additive Synth]                        │
│                           │                                 │
│                           ▼                                 │
│                   Reconstructed Audio                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Specifications

| Parameter | Value |
|-----------|-------|
| Sample Rate | 24 kHz |
| Frame Rate | 75 Hz |
| Latent Dim | 256 |
| Codebook | 1024 per layer |
| Quantizers | 1-8 (variable) |
| Harmonics | 64 |
| Bitrate | 0.75-6 kbps |
| Compression | 64-512× vs 16-bit PCM |

---

## 🎤 Voice Cloning

ToneNet includes an experimental voice cloning extension:

```python
from tonenet.voice_cloning import ToneNetVoiceCloner
import torch

cloner = ToneNetVoiceCloner()

# 5 second reference audio
reference = torch.randn(1, 24000 * 5)

# Clone voice
audio, info = cloner.clone_voice(
    text="Hello, this is a voice clone.",
    reference_audio=reference
)
```

**Architecture:**

- **ECAPA-TDNN** speaker encoder (256-dim embeddings)
- **Transformer** text encoder (phoneme → embeddings)
- **AR model** for first quantizer generation
- **NAR model** for parallel refinement

---

## 📁 Project Structure

```
tonenet/
├── __init__.py          # Package exports
├── codec.py             # ToneNetCodec main class
├── encoder.py           # Causal CNN encoder
├── decoder.py           # Harmonic decoder + synthesis
├── quantizer.py         # VQ-VAE with RVQ
├── losses.py            # Multi-STFT, Mel losses
├── metrics.py           # SNR, STOI, PESQ
├── audio.py             # File compression utilities
├── trainer.py           # Training framework
├── controller.py        # PI controller
├── deployment.py        # Export utilities
└── voice_cloning/
    ├── speaker_encoder.py  # ECAPA-TDNN
    ├── text_encoder.py     # Phoneme encoder
    ├── ar_model.py         # Autoregressive LM
    ├── nar_model.py        # Non-autoregressive
    └── voice_cloner.py     # Complete pipeline
```

---

## 🔬 Training

```python
from tonenet import ToneNetCodec, ToneNetTrainer

model = ToneNetCodec()
trainer = ToneNetTrainer(model, device='cuda')

for batch in dataloader:
    losses = trainer.train_step(batch)
    print(f"Loss: {losses['loss']:.4f}")
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with 🎵 by <a href="https://github.com/dawsonblock">Dawson Block</a></sub>
</p>
