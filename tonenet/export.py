"""
ONNX export for hardware-agnostic inference.

Exports ToneNet codec to ONNX format for deployment on various runtimes.
"""

import torch
from pathlib import Path
from typing import Optional
from .codec import ToneNetCodec


class EncoderWrapper(torch.nn.Module):
    """Wrapper for encoder-only export."""
    
    def __init__(self, codec: ToneNetCodec):
        super().__init__()
        self.encoder = codec.encoder
        self.quantizer = codec.quantizer
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        z = self.encoder(audio)
        z_q, indices, _ = self.quantizer(z)
        # Return first quantizer indices for simplicity
        return indices[0] if isinstance(indices, list) else indices


class DecoderWrapper(torch.nn.Module):
    """Wrapper for decoder-only export."""
    
    def __init__(self, codec: ToneNetCodec):
        super().__init__()
        self.quantizer = codec.quantizer
        self.decoder = codec.decoder
    
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        # Simplified: single quantizer codes input
        z_q = self.quantizer.decode_codes([codes])
        f0, H, phi, noise, vocoder_residual = self.decoder(z_q)
        return self.decoder.synthesize(f0, H, phi, noise, vocoder_residual)


def export_encoder_onnx(
    path: str = "tonenet_encoder.onnx",
    sample_rate: int = 24000,
    opset_version: int = 17
) -> None:
    """
    Export encoder to ONNX.
    
    Args:
        path: Output ONNX file path
        sample_rate: Sample rate
        opset_version: ONNX opset version
    """
    codec = ToneNetCodec(sample_rate=sample_rate).eval()
    wrapper = EncoderWrapper(codec)
    
    dummy = torch.randn(1, 1, sample_rate)  # 1 second
    
    torch.onnx.export(
        wrapper,
        dummy,
        path,
        input_names=["audio"],
        output_names=["codes"],
        opset_version=opset_version,
        dynamic_axes={
            "audio": {2: "samples"},
            "codes": {1: "frames"}
        }
    )
    print(f"Exported encoder: {path}")


def export_codec_onnx(
    path: str = "tonenet_codec.onnx",
    sample_rate: int = 24000,
    opset_version: int = 17
) -> None:
    """
    Export full codec (encode+decode) to ONNX.
    
    Note: Full codec export may have limitations due to
    complex decoder synthesis. Use encoder-only for production.
    
    Args:
        path: Output ONNX file path
        sample_rate: Sample rate
        opset_version: ONNX opset version
    """
    model = ToneNetCodec(sample_rate=sample_rate).eval()
    
    dummy = torch.randn(1, 1, sample_rate)  # 1 second
    
    # Export in trace mode for dynamic operations
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["audio"],
        output_names=["reconstructed"],
        opset_version=opset_version,
        dynamic_axes={
            "audio": {2: "samples"},
            "reconstructed": {2: "samples"}
        }
    )
    print(f"Exported codec: {path}")


def export_torchscript(
    path: str = "tonenet_codec.pt",
    sample_rate: int = 24000
) -> None:
    """
    Export codec to TorchScript.
    
    Args:
        path: Output TorchScript file path
        sample_rate: Sample rate
    """
    model = ToneNetCodec(sample_rate=sample_rate).eval()
    
    dummy = torch.randn(1, 1, sample_rate)
    
    # Use tracing
    traced = torch.jit.trace(model, dummy)
    traced.save(path)
    print(f"Exported TorchScript: {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["onnx", "torchscript", "both"], default="both")
    parser.add_argument("--encoder-only", action="store_true")
    args = parser.parse_args()
    
    if args.format in ["onnx", "both"]:
        if args.encoder_only:
            export_encoder_onnx()
        else:
            export_codec_onnx()
    
    if args.format in ["torchscript", "both"]:
        export_torchscript()
