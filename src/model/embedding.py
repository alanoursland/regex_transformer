"""Token embeddings."""
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """Token embedding layer."""
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass. tokens: (B, T) -> (B, T, d_model)"""
        return self.embedding(tokens)
