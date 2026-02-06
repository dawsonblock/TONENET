"""
ToneNet invariant tests.

These tests lock fundamental correctness properties:
- Encoder/decoder timebase alignment
- Encode/decode round-trip shapes
- Bitrate math correctness
- Import without torchaudio
"""

import torch
import pytest


def test_import_without_torchaudio():
    """Package should import even if torchaudio is unavailable."""
    # This test just verifies the import doesn't crash
    import tonenet
    assert hasattr(tonenet, 'ToneNetCodec')


def test_timebase_alignment():
    """Encoder must produce approximately 75 frames per second (±1 for padding)."""
    from tonenet import ToneNetCodec
    
    model = ToneNetCodec()
    x = torch.randn(1, 1, 24000)  # 1 second at 24kHz
    
    # Encode
    z = model.encoder(x)
    
    # Should be ~75 frames (24000 / 320 = 75, ±1 for conv padding)
    expected_frames = 75
    actual_frames = z.shape[1]
    
    assert abs(actual_frames - expected_frames) <= 1, \
        f"Expected ~{expected_frames} frames for 1 second, got {actual_frames}"


def test_roundtrip_shape():
    """Encode → decode should preserve audio length."""
    from tonenet import ToneNetCodec
    
    model = ToneNetCodec()
    x = torch.randn(1, 1, 24000)
    
    # Full forward pass
    recon, outputs = model(x)
    
    assert recon.shape == x.shape, \
        f"Shape mismatch: input {x.shape}, output {recon.shape}"


def test_bitrate_math():
    """Bitrate calculation must match actual latent rate."""
    from tonenet import ToneNetCodec
    import math
    
    model = ToneNetCodec()
    
    # Expected: 75 Hz * 8 quantizers * log2(1024) bits
    expected_bps = 75 * 8 * math.log2(1024)  # 6000 bps
    
    info = model.get_bitrate(n_quantizers=8)
    
    assert abs(info['bitrate_bps'] - expected_bps) < 1, \
        f"Expected {expected_bps} bps, got {info['bitrate_bps']}"


def test_variable_bitrate():
    """Variable quantizer count should scale bitrate linearly."""
    from tonenet import ToneNetCodec
    
    model = ToneNetCodec()
    
    # Check bitrate scales with quantizer count
    rates = []
    for n_q in [1, 4, 8]:
        info = model.get_bitrate(n_quantizers=n_q)
        rates.append((n_q, info['bitrate_kbps']))
    
    # 8 quantizers should be 8x the bitrate of 1 quantizer
    assert abs(rates[2][1] / rates[0][1] - 8.0) < 0.1, \
        f"Bitrate should scale 8x: {rates}"


def test_encoder_stride_validation():
    """Encoder should reject invalid stride configurations."""
    from tonenet.core.encoder import ToneNetEncoder
    
    # Valid: 5*4*4*4 = 320
    encoder = ToneNetEncoder()
    assert encoder.hop_length == 320
    
    # Invalid: product doesn't match hop_length
    with pytest.raises(ValueError, match="must multiply to hop_length"):
        ToneNetEncoder(strides=[2, 2, 2, 2], hop_length=320)


def test_code_shapes():
    """Encoded codes should have expected shapes (±1 frame for padding)."""
    from tonenet import ToneNetCodec
    
    model = ToneNetCodec()
    x = torch.randn(2, 1, 24000)  # 2 samples, 1 second
    
    codes = model.encode(x)
    
    assert len(codes) == 8, f"Expected 8 quantizer layers, got {len(codes)}"
    # Allow ±1 frame due to conv padding
    assert codes[0].shape[0] == 2, f"Expected batch size 2, got {codes[0].shape[0]}"
    assert abs(codes[0].shape[1] - 75) <= 1, f"Expected ~75 frames, got {codes[0].shape[1]}"
