#!/usr/bin/env python3
"""
ToneNet v2.0 - Basic Usage Examples

Demonstrates:
1. Model instantiation and forward pass
2. Encode/decode round-trip
3. Variable bitrate
4. Bitrate calculation
5. Harmonic parameter access
"""

import torch
from tonenet import ToneNetCodec, ToneNetTrainer, ClassicalController


def main():
    print("=" * 70)
    print("TONENET V2.0 - BASIC USAGE EXAMPLES")
    print("=" * 70)
    
    # 1. Create model
    print("\n1. Creating ToneNet Codec...")
    model = ToneNetCodec(
        sample_rate=24000,
        latent_dim=256,
        num_quantizers=8,
        codebook_size=1024,
        num_harmonics=64
    )
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. Generate test audio
    print("\n2. Processing test audio...")
    audio = torch.randn(2, 1, 24000)  # 2 samples, 1 second @ 24kHz
    
    # Forward pass
    reconstructed, outputs = model(audio)
    print(f"   Input shape:  {audio.shape}")
    print(f"   Output shape: {reconstructed.shape}")
    print(f"   VQ Loss:      {outputs['vq_loss']:.4f}")
    
    # 3. Access harmonic parameters
    print("\n3. Harmonic parameters:")
    print(f"   f0 range:     [{outputs['f0'].min():.1f}, {outputs['f0'].max():.1f}] Hz")
    print(f"   H shape:      {outputs['H'].shape} (amplitudes)")
    print(f"   phi shape:    {outputs['phi'].shape} (phases)")
    print(f"   noise range:  [{outputs['noise'].min():.3f}, {outputs['noise'].max():.3f}]")
    
    # 4. Encode/decode round-trip
    print("\n4. Encode/decode round-trip:")
    codes = model.encode(audio)
    print(f"   Number of code layers: {len(codes)}")
    print(f"   Code shape per layer:  {codes[0].shape}")
    
    decoded = model.decode(codes)
    print(f"   Decoded shape: {decoded.shape}")
    
    # 5. Variable bitrate
    print("\n5. Variable bitrate:")
    for n_q in [1, 2, 4, 8]:
        recon, out = model(audio, n_quantizers=n_q)
        info = model.get_bitrate(n_quantizers=n_q)
        print(f"   {n_q} quantizers: {info['bitrate_kbps']:.2f} kbps "
              f"({info['compression_ratio_16bit']:.0f}x compression)")
    
    # 6. Classical controller demo
    print("\n6. Classical controller (parameter adaptation):")
    controller = ClassicalController(
        k_p=0.1, k_i=0.01,
        bounds={'f0': (50, 8000), 'amp': (0.0, 1.0)}
    )
    theta = {'f0': 440.0, 'amp': 0.5}
    error = {'f0': 10.0, 'amp': -0.1}
    theta_new = controller.adapt(theta, error)
    print(f"   Original: f0={theta['f0']:.1f} Hz, amp={theta['amp']:.2f}")
    print(f"   Adapted:  f0={theta_new['f0']:.1f} Hz, amp={theta_new['amp']:.2f}")
    
    # 7. Training step demo
    print("\n7. Training step demo:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = ToneNetTrainer(model, device=device, lr=3e-4)
    batch = torch.randn(4, 1, 24000)
    losses = trainer.train_step(batch)
    print(f"   Device: {device}")
    print(f"   Total loss: {losses['loss']:.4f}")
    print(f"   Time loss:  {losses['time']:.4f}")
    print(f"   STFT loss:  {losses['stft']:.4f}")
    print(f"   Mel loss:   {losses['mel']:.4f}")
    print(f"   VQ loss:    {losses['vq']:.4f}")
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
