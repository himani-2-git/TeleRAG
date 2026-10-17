"""Test package initialization."""
# Ensure PyTorch DLLs initialize cleanly before other native libraries on Windows
try:
    import torch
except ImportError:
    pass