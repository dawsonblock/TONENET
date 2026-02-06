"""
Pytest configuration for ToneNet tests.

Fixes:
- Torch thread cap to prevent hangs in constrained environments
- Adds repo root to sys.path for import stability
"""

import os
import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Thread cap for Torch - prevents hangs in containers/CI
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)

# Common fixtures can go here
import pytest

@pytest.fixture
def sample_audio():
    """1 second of random audio at 24kHz."""
    return torch.randn(1, 1, 24000)

@pytest.fixture
def sample_tokens():
    """Sample codec tokens (single quantizer)."""
    return torch.randint(0, 1024, (1, 75))

@pytest.fixture  
def sample_codes():
    """Sample codec codes (8 quantizers, list format)."""
    return [torch.randint(0, 1024, (1, 75)) for _ in range(8)]
