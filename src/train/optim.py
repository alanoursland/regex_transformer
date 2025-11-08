"""Optimizer utilities."""
import torch

def build_optimizer(model, lr=1e-3, weight_decay=0.01):
    """Build AdamW optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def clip_gradients(model, max_norm=1.0):
    """Clip gradients by global norm."""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
