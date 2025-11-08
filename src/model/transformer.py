"""Transformer block and full model."""
import torch
import torch.nn as nn
from .config import ModelConfig
from .embedding import TokenEmbedding
from .positional import get_sinusoidal_positional_encoding, add_positional
from .attention import MultiHeadAttention
from .mlp import MLP
from .heads import NextTokenHead, ClassHead

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass. x: (B, T, d_model), attn_mask: (B, T)"""
        # Pre-norm attention
        x = x + self.dropout(self.attn(self.ln1(x), attn_mask))
        # Pre-norm MLP
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class RegexTransformer(nn.Module):
    """Single-layer transformer for regex tasks."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        
        # Positional encoding
        if config.positional_type == "sin":
            self.register_buffer(
                "pos_enc",
                get_sinusoidal_positional_encoding(config.max_seq_len, config.d_model)
            )
        else:
            self.pos_enc = None
        
        # Transformer block (single layer)
        self.block = TransformerBlock(config.d_model, config.n_heads, config.d_mlp, config.dropout)
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Task heads
        self.next_token_head = NextTokenHead(config.d_model, config.vocab_size)
        self.class_head = ClassHead(config.d_model, config.num_classes)
        
        # Weight tying
        if config.tie_weights:
            self.next_token_head.proj.weight = self.token_emb.embedding.weight
    
    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            tokens: (B, T) token IDs
            attn_mask: (B, T) bool tensor, True for valid positions
        
        Returns:
            dict with:
                - next_token_logits: (B, T, vocab_size)
                - class_logits: (B, T, num_classes)
        """
        # Token embeddings
        x = self.token_emb(tokens)  # (B, T, d_model)
        
        # Add positional encoding
        if self.pos_enc is not None:
            x = add_positional(x, self.pos_enc)
        
        # Transformer block
        x = self.block(x, attn_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Task heads
        next_token_logits = self.next_token_head(x)
        class_logits = self.class_head(x)
        
        return {
            "next_token_logits": next_token_logits,
            "class_logits": class_logits,
        }
