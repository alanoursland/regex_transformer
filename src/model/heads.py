"""Task prediction heads."""
import torch
import torch.nn as nn

class NextTokenHead(nn.Module):
    """Next-token prediction head."""
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, T, d_model) -> (B, T, vocab_size)"""
        return self.proj(x)


class ClassHead(nn.Module):
    """State class prediction head."""
    
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, T, d_model) -> (B, T, num_classes)"""
        return self.proj(x)
