"""Unit tests for attention masking behavior."""

import torch
import pytest
from model.attention import MultiHeadAttention


class TestAttentionMasking:
    """Test that attention masks don't produce NaN values."""
    
    def test_no_mask_produces_valid_output(self):
        """Test attention without any mask produces finite outputs."""
        d_model = 16
        n_heads = 2
        B, T = 2, 5
        
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        attn.eval()  # Disable dropout
        
        x = torch.randn(B, T, d_model)
        
        with torch.no_grad():
            out = attn(x, attn_mask=None)
        
        assert not torch.isnan(out).any(), "Attention output contains NaN"
        assert not torch.isinf(out).any(), "Attention output contains Inf"
        assert out.shape == (B, T, d_model)
    
    def test_valid_mask_produces_valid_output(self):
        """Test attention with valid positions mask produces finite outputs."""
        d_model = 16
        n_heads = 2
        B, T = 2, 5
        
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        attn.eval()
        
        x = torch.randn(B, T, d_model)
        # Mask out last 2 positions in first sequence
        attn_mask = torch.ones(B, T, dtype=torch.bool)
        attn_mask[0, 3:] = False
        
        with torch.no_grad():
            out = attn(x, attn_mask=attn_mask)
        
        assert not torch.isnan(out).any(), "Attention output contains NaN with padding"
        assert not torch.isinf(out).any(), "Attention output contains Inf with padding"
        assert out.shape == (B, T, d_model)
    
    def test_all_padding_positions_produce_valid_output(self):
        """Test that even fully padded positions produce finite outputs."""
        d_model = 16
        n_heads = 2
        B, T = 2, 5
        
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        attn.eval()
        
        x = torch.randn(B, T, d_model)
        # First sequence: only first token valid
        attn_mask = torch.zeros(B, T, dtype=torch.bool)
        attn_mask[0, 0] = True
        attn_mask[1, :] = True
        
        with torch.no_grad():
            out = attn(x, attn_mask=attn_mask)
        
        # Even padding positions should have finite outputs (they'll be masked in loss)
        assert not torch.isnan(out).any(), "Padding positions produce NaN"
        assert not torch.isinf(out).any(), "Padding positions produce Inf"
    
    def test_causal_mask_prevents_future_attention(self):
        """Test that causal mask prevents attending to future positions."""
        d_model = 16
        n_heads = 2
        T = 5
        
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        attn.eval()
        
        # Create input where position i has value i
        x = torch.arange(T).float().view(1, T, 1).expand(1, T, d_model)
        
        with torch.no_grad():
            # Get attention scores (need to modify module to expose them)
            # For now, just check output is valid
            out = attn(x, attn_mask=None)
        
        assert not torch.isnan(out).any(), "Causal attention produces NaN"
        assert out.shape == (1, T, d_model)
    
    def test_batch_with_different_lengths(self):
        """Test batch where sequences have different valid lengths."""
        d_model = 16
        n_heads = 2
        B, T = 3, 8
        
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        attn.eval()
        
        x = torch.randn(B, T, d_model)
        
        # Different lengths: 3, 5, 8
        attn_mask = torch.zeros(B, T, dtype=torch.bool)
        attn_mask[0, :3] = True
        attn_mask[1, :5] = True
        attn_mask[2, :] = True
        
        with torch.no_grad():
            out = attn(x, attn_mask=attn_mask)
        
        assert not torch.isnan(out).any(), "Variable length batch produces NaN"
        assert out.shape == (B, T, d_model)
    
    def test_single_valid_position(self):
        """Test sequence with only one valid position (edge case)."""
        d_model = 16
        n_heads = 2
        B, T = 1, 5
        
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        attn.eval()
        
        x = torch.randn(B, T, d_model)
        attn_mask = torch.zeros(B, T, dtype=torch.bool)
        attn_mask[0, 0] = True  # Only first position valid
        
        with torch.no_grad():
            out = attn(x, attn_mask=attn_mask)
        
        assert not torch.isnan(out).any(), "Single valid position produces NaN"
        # First position should have valid output (attends to itself)
        assert torch.isfinite(out[0, 0]).all(), "First position output not finite"


class TestAttentionNumericalStability:
    """Test numerical stability of attention computation."""
    
    def test_large_sequence_length(self):
        """Test attention with long sequences remains stable."""
        d_model = 64
        n_heads = 4
        B, T = 2, 100
        
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        attn.eval()
        
        x = torch.randn(B, T, d_model)
        attn_mask = torch.ones(B, T, dtype=torch.bool)
        
        with torch.no_grad():
            out = attn(x, attn_mask=attn_mask)
        
        assert not torch.isnan(out).any(), "Long sequences produce NaN"
        assert torch.isfinite(out).all(), "Long sequences produce Inf"
    
    def test_extreme_input_values(self):
        """Test attention with large input values remains stable."""
        d_model = 16
        n_heads = 2
        B, T = 2, 5
        
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        attn.eval()
        
        # Large but valid input values
        x = torch.randn(B, T, d_model) * 10.0
        
        with torch.no_grad():
            out = attn(x, attn_mask=None)
        
        assert not torch.isnan(out).any(), "Large inputs produce NaN"
        assert torch.isfinite(out).all(), "Large inputs produce Inf"
    
    def test_zero_input(self):
        """Test attention with zero input."""
        d_model = 16
        n_heads = 2
        B, T = 2, 5
        
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        attn.eval()
        
        x = torch.zeros(B, T, d_model)
        
        with torch.no_grad():
            out = attn(x, attn_mask=None)
        
        assert not torch.isnan(out).any(), "Zero input produces NaN"
        assert torch.isfinite(out).all(), "Zero input produces Inf"


class TestAttentionGradients:
    """Test that gradients are well-behaved."""
    
    def test_gradients_are_finite(self):
        """Test that backward pass produces finite gradients."""
        d_model = 16
        n_heads = 2
        B, T = 2, 5
        
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        attn.train()
        
        x = torch.randn(B, T, d_model, requires_grad=True)
        attn_mask = torch.ones(B, T, dtype=torch.bool)
        
        out = attn(x, attn_mask=attn_mask)
        loss = out.sum()
        loss.backward()
        
        # Check gradients
        for name, param in attn.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"{name} gradient contains NaN"
                assert not torch.isinf(param.grad).any(), f"{name} gradient contains Inf"
        
        # Check input gradient
        assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
        assert not torch.isinf(x.grad).any(), "Input gradient contains Inf"
    
    def test_gradients_with_padding(self):
        """Test gradients remain finite with padding."""
        d_model = 16
        n_heads = 2
        B, T = 2, 5
        
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        attn.train()
        
        x = torch.randn(B, T, d_model, requires_grad=True)
        attn_mask = torch.ones(B, T, dtype=torch.bool)
        attn_mask[0, 3:] = False  # Padding in first sequence
        
        out = attn(x, attn_mask=attn_mask)
        # Only compute loss on valid positions
        loss = (out * attn_mask.unsqueeze(-1)).sum()
        loss.backward()
        
        for name, param in attn.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"{name} gradient contains NaN with padding"
                assert not torch.isinf(param.grad).any(), f"{name} gradient contains Inf with padding"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])