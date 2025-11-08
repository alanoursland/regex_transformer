"""MLP module."""
import torch
import torch.nn as nn

class MLP(nn.Module):
    """Two-layer MLP with GELU activation."""
    
    def __init__(self, d_model: int, d_mlp: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, T, d_model) -> (B, T, d_model)"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
