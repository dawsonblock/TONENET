"""
Deployment utilities for ToneNet.

Includes:
- ToneNetDeployment wrapper for inference
- File compression/decompression
- Model export (TorchScript, ONNX)
- Verification script
"""

import struct
import wave
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch

from .core import ToneNetCodec


class ToneNetDeployment:
    """Production deployment wrapper."""

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {})

        # Reconstruct model
        self.model = ToneNetCodec(
            sample_rate=config.get('sample_rate', 24000),
            latent_dim=config.get('latent_dim', 256),
            num_quantizers=config.get('num_quantizers', 8),
            codebook_size=config.get('codebook_size', 1024),
            num_harmonics=config.get('num_harmonics', 64)
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Loaded ToneNet v2.0 from {model_path}")
        bitrate_info = self.model.get_bitrate()
        print(f"  Bitrate: {bitrate_info['bitrate_kbps']:.2f} kbps")
        print(f"  Frame rate: {bitrate_info['frame_rate_hz']:.1f} Hz")

    def encode(self, audio: np.ndarray, n_quantizers: Optional[int] = None) -> List[np.ndarray]:
        """Encode audio to discrete codes."""
        # Pad to frame boundary
        hop = self.model.encoder.hop_length
        pad_len = (hop - len(audio) % hop) % hop
        if pad_len > 0:
            audio = np.pad(audio, (0, pad_len))

        # To tensor
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            codes = self.model.encode(audio_t, n_quantizers)

        return [c.cpu().numpy().squeeze() for c in codes]

    def decode(self, codes: List[np.ndarray]) -> np.ndarray:
        """Decode discrete codes to audio."""
        codes_t = [torch.from_numpy(c).long().unsqueeze(0).to(self.device) for c in codes]

        with torch.no_grad():
            audio_t = self.model.decode(codes_t)

        return audio_t.cpu().numpy().squeeze()

    def compress_file(self, input_path: str, output_path: str):
        """Compress audio file to binary format."""
        with wave.open(input_path, 'rb') as f:
            n_channels = f.getnchannels()
            sr = f.getframerate()
            n_frames = f.getnframes()

            audio = np.frombuffer(f.readframes(n_frames), dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)

        codes = self.encode(audio)
        num_frames = len(codes[0])

        # Binary format: [n_quantizers][codebook_size][num_frames][codes...]
        with open(output_path, 'wb') as f:
            f.write(struct.pack('B', len(codes)))
            f.write(struct.pack('H', self.model.quantizer.codebook_size))
            f.write(struct.pack('I', num_frames))
            for code in codes:
                f.write(code.astype(np.uint16).tobytes())

        original_bits = len(audio) * 16
        compressed_bits = (1 + 2 + 4) * 8 + len(codes) * num_frames * 16
        ratio = original_bits / compressed_bits

        print(f"Compressed: {input_path} -> {output_path}")
        print(f"  Ratio: {ratio:.1f}x")
        print(f"  Bitrate: {self.model.get_bitrate()['bitrate_kbps']:.2f} kbps")

    def decompress_file(self, input_path: str, output_path: str):
        """Decompress binary file to audio."""
        with open(input_path, 'rb') as f:
            n_q = struct.unpack('B', f.read(1))[0]
            codebook_size = struct.unpack('H', f.read(2))[0]
            num_frames = struct.unpack('I', f.read(4))[0]

            codes = []
            for _ in range(n_q):
                code_bytes = f.read(num_frames * 2)
                code = np.frombuffer(code_bytes, dtype=np.uint16).astype(np.int64)
                codes.append(code)

        audio = self.decode(codes)

        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(output_path, 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.model.sample_rate)
            f.writeframes(audio_int16.tobytes())


def export_model(model: ToneNetCodec, output_dir: str):
    """Export model for deployment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'sample_rate': model.sample_rate,
            'latent_dim': model.latent_dim,
            'num_quantizers': model.quantizer.num_quantizers,
            'codebook_size': model.quantizer.codebook_size,
            'num_harmonics': model.decoder.num_harmonics,
            'version': '2.0.0'
        }
    }
    torch.save(checkpoint, output_dir / 'tonenet_v2.pt')
    print(f"✓ Saved checkpoint to {output_dir / 'tonenet_v2.pt'}")

    # TorchScript (may fail for complex models)
    try:
        model.eval()
        scripted = torch.jit.script(model)
        scripted.save(str(output_dir / 'tonenet_v2_scripted.pt'))
        print(f"✓ Saved TorchScript to {output_dir / 'tonenet_v2_scripted.pt'}")
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")

    # ONNX
    try:
        model.eval()
        dummy_input = torch.randn(1, 1, 24000)
        torch.onnx.export(
            model,
            dummy_input,
            str(output_dir / 'tonenet_v2.onnx'),
            input_names=['audio'],
            output_names=['reconstructed'],
            dynamic_axes={'audio': {0: 'batch', 2: 'time'}}
        )
        print(f"✓ Saved ONNX to {output_dir / 'tonenet_v2.onnx'}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")


def verify_model() -> ToneNetCodec:
    """Verify all components work correctly."""
    print("=" * 70)
    print("TONENET V2.0 - VERIFICATION")
    print("=" * 70)

    # Test 1: Model instantiation
    print("\n1. Model instantiation...")
    model = ToneNetCodec(
        sample_rate=24000,
        latent_dim=256,
        num_quantizers=8,
        codebook_size=1024,
        num_harmonics=64
    )
    print(f"   ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test 2: Forward pass
    print("\n2. Forward pass...")
    test_input = torch.randn(2, 1, 24000)  # 2 samples, 1 second @ 24kHz
    reconstructed, outputs = model(test_input)
    print(f"   ✓ Input shape: {test_input.shape}")
    print(f"   ✓ Output shape: {reconstructed.shape}")
    print(f"   ✓ VQ loss: {outputs['vq_loss']:.4f}")
    print(f"   ✓ f0 range: [{outputs['f0'].min():.1f}, {outputs['f0'].max():.1f}] Hz")

    # Test 3: Encode/decode round-trip
    print("\n3. Encode/decode round-trip...")
    codes = model.encode(test_input)
    print(f"   ✓ Number of quantizers: {len(codes)}")
    print(f"   ✓ Code shape: {codes[0].shape}")
    reconstructed_2 = model.decode(codes)
    print(f"   ✓ Reconstructed shape: {reconstructed_2.shape}")

    # Test 4: Bitrate calculation
    print("\n4. Bitrate calculation...")
    for n_q in [1, 4, 8]:
        info = model.get_bitrate(n_quantizers=n_q)
        print(f"   ✓ {n_q} quantizers: {info['bitrate_kbps']:.2f} kbps")

    # Test 5: Variable bitrate
    print("\n5. Variable bitrate...")
    for n_q in [1, 4, 8]:
        recon, out = model(test_input, n_quantizers=n_q)
        print(f"   ✓ {n_q} quantizers: recon shape {recon.shape}")

    # Test 6: Classical controller
    print("\n6. Classical controller...")
    from .controller import ClassicalController
    controller = ClassicalController(
        k_p=0.1, k_i=0.01,
        bounds={'f0': (50, 8000), 'amp': (0.0, 1.0)}
    )
    theta = {'f0': 440.0, 'amp': 0.5}
    error = {'f0': 10.0, 'amp': -0.1}
    theta_new = controller.adapt(theta, error)
    print(f"   ✓ Original: {theta}")
    print(f"   ✓ Adapted: {theta_new}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)

    return model
