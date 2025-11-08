"""Positional encoding."""
import torch
import math

def get_sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Generate sinusoidal positional encoding. Returns (max_len, d_model)."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def add_positional(x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """Add positional encoding to input. x: (B, T, d_model), pe: (max_len, d_model)"""
    seq_len = x.size(1)
    return x + pe[:seq_len, :].to(x.device)
