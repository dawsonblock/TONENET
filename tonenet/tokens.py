"""
Token packing/unpacking utilities.

The ToneNet codec returns a list of tensors (one per quantizer layer).
These utilities provide a canonical way to convert between:
- list[Tensor[B,T]] - native codec format
- Tensor[B,Q,T]     - packed format for storage/logging/models
"""

import torch
from typing import List, Union


def pack_codes(codes: List[torch.Tensor]) -> torch.Tensor:
    """
    Pack list of per-quantizer codes into single tensor.
    
    Args:
        codes: List of tensors, each [B, T] containing codebook indices
               Length = number of quantizers (Q)
    
    Returns:
        Tensor of shape [B, Q, T]
    
    Example:
        >>> codes = codec.encode(audio)  # list of 8 tensors
        >>> packed = pack_codes(codes)   # [1, 8, 75]
    """
    if not codes:
        raise ValueError("Cannot pack empty codes list")
    
    # Stack along new dimension (quantizer axis)
    return torch.stack(codes, dim=1)


def unpack_codes(packed: torch.Tensor) -> List[torch.Tensor]:
    """
    Unpack tensor back to list of per-quantizer codes.
    
    Args:
        packed: Tensor of shape [B, Q, T]
    
    Returns:
        List of Q tensors, each [B, T]
    
    Example:
        >>> packed = torch.load("codes.pt")  # [1, 8, 75]
        >>> codes = unpack_codes(packed)     # list of 8 tensors
        >>> audio = codec.decode(codes)
    """
    if packed.dim() != 3:
        raise ValueError(f"Expected [B,Q,T] shape, got {packed.shape}")
    
    # Split along quantizer dimension
    return [packed[:, i, :] for i in range(packed.shape[1])]


def normalize_codes(
    codes: Union[torch.Tensor, List[torch.Tensor]]
) -> List[torch.Tensor]:
    """
    Normalize codes to canonical list format.
    
    Accepts either:
    - List[Tensor[B,T]] - returns as-is
    - Tensor[B,T]       - wraps in single-element list
    - Tensor[B,Q,T]     - unpacks to list
    
    Returns:
        List of tensors in codec-native format
    """
    if isinstance(codes, list):
        return codes
    
    if codes.dim() == 2:
        # Single quantizer tensor [B, T]
        return [codes]
    
    if codes.dim() == 3:
        # Packed format [B, Q, T]
        return unpack_codes(codes)
    
    raise ValueError(f"Cannot normalize codes with shape {codes.shape}")


def codes_to_tensor(
    codes: Union[torch.Tensor, List[torch.Tensor]]
) -> torch.Tensor:
    """
    Convert codes to packed tensor format.
    
    Accepts either list or tensor format.
    
    Returns:
        Tensor[B, Q, T]
    """
    if isinstance(codes, torch.Tensor):
        if codes.dim() == 2:
            return codes.unsqueeze(1)  # [B, 1, T]
        return codes  # already [B, Q, T]
    
    return pack_codes(codes)


def get_code_info(
    codes: Union[torch.Tensor, List[torch.Tensor]]
) -> dict:
    """
    Get metadata about code tensor.
    
    Returns:
        dict with keys: batch_size, n_quantizers, n_frames, device
    """
    if isinstance(codes, list):
        if not codes:
            return {"batch_size": 0, "n_quantizers": 0, "n_frames": 0}
        first = codes[0]
        return {
            "batch_size": first.shape[0],
            "n_quantizers": len(codes),
            "n_frames": first.shape[1],
            "device": str(first.device)
        }
    
    if codes.dim() == 2:
        return {
            "batch_size": codes.shape[0],
            "n_quantizers": 1,
            "n_frames": codes.shape[1],
            "device": str(codes.device)
        }
    
    return {
        "batch_size": codes.shape[0],
        "n_quantizers": codes.shape[1],
        "n_frames": codes.shape[2],
        "device": str(codes.device)
    }
