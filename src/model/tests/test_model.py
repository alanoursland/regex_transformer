"""Tests for model components."""
import pytest
import torch
from ..config import ModelConfig
from ..transformer import RegexTransformer
from ..embedding import TokenEmbedding
from ..attention import MultiHeadAttention
from ..mlp import MLP
from ..transformer import TransformerBlock

def test_model_config():
    """Test model configuration."""
    cfg = ModelConfig(vocab_size=100, num_classes=3, num_states=10)
    assert cfg.d_model % cfg.n_heads == 0

def test_model_config_validation():
    """Test config validation."""
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=100, num_classes=3, num_states=10, d_model=63, n_heads=4)

def test_token_embedding():
    """Test token embedding."""
    emb = TokenEmbedding(vocab_size=100, d_model=64)
    tokens = torch.randint(0, 100, (2, 10))
    out = emb(tokens)
    assert out.shape == (2, 10, 64)

def test_attention_shapes():
    """Test attention output shapes."""
    attn = MultiHeadAttention(d_model=64, n_heads=4)
    x = torch.randn(2, 10, 64)
    out = attn(x)
    assert out.shape == (2, 10, 64)

def test_attention_with_mask():
    """Test attention with padding mask."""
    attn = MultiHeadAttention(d_model=64, n_heads=4)
    x = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, 5:] = False  # Mask out last 5 positions of first sequence
    out = attn(x, mask)
    assert out.shape == (2, 10, 64)

def test_mlp_shapes():
    """Test MLP shapes."""
    mlp = MLP(d_model=64, d_mlp=256)
    x = torch.randn(2, 10, 64)
    out = mlp(x)
    assert out.shape == (2, 10, 64)

def test_transformer_block():
    """Test transformer block."""
    block = TransformerBlock(d_model=64, n_heads=4, d_mlp=256)
    x = torch.randn(2, 10, 64)
    out = block(x)
    assert out.shape == (2, 10, 64)

def test_full_model():
    """Test full model forward pass."""
    cfg = ModelConfig(vocab_size=100, num_classes=3, num_states=10, d_model=64, n_heads=4)
    model = RegexTransformer(cfg)
    
    tokens = torch.randint(0, 100, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.bool)
    
    outputs = model(tokens, mask)
    
    assert "next_token_logits" in outputs
    assert "class_logits" in outputs
    assert outputs["next_token_logits"].shape == (2, 10, 100)
    assert outputs["class_logits"].shape == (2, 10, 3)

def test_model_determinism():
    """Test that model produces deterministic outputs."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=50, num_classes=3, num_states=10, d_model=32, n_heads=2)
    model1 = RegexTransformer(cfg)
    model1.eval()
    
    torch.manual_seed(42)
    model2 = RegexTransformer(cfg)
    model2.eval()
    
    tokens = torch.randint(0, 50, (1, 5))
    
    out1 = model1(tokens)
    out2 = model2(tokens)
    
    assert torch.allclose(out1["next_token_logits"], out2["next_token_logits"])

def test_model_gradient_flow():
    """Test that gradients flow properly."""
    cfg = ModelConfig(vocab_size=50, num_classes=3, num_states=10, d_model=32, n_heads=2)
    model = RegexTransformer(cfg)
    
    tokens = torch.randint(0, 50, (2, 5))
    outputs = model(tokens)
    
    loss = outputs["next_token_logits"].sum() + outputs["class_logits"].sum()
    loss.backward()
    
    # Check that some gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad

def test_weight_tying():
    """Test weight tying option."""
    cfg = ModelConfig(vocab_size=50, num_classes=3, num_states=10, tie_weights=True)
    model = RegexTransformer(cfg)
    
    assert model.next_token_head.proj.weight is model.token_emb.embedding.weight
