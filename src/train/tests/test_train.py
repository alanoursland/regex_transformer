"""Tests for training loop."""
import pytest
import torch
import tempfile
from pathlib import Path

from fsm.regex_def import RegexDefinition
from model.config import ModelConfig
from ..loop import TrainConfig, train_one_experiment, set_seed, evaluate
from ..losses import compute_multi_task_loss

def test_set_seed():
    """Test seed setting."""
    set_seed(42)
    x1 = torch.randn(10)
    
    set_seed(42)
    x2 = torch.randn(10)
    
    assert torch.allclose(x1, x2)

def test_compute_loss():
    """Test loss computation."""
    outputs = {
        "next_token_logits": torch.randn(2, 5, 50),
        "class_logits": torch.randn(2, 5, 3),
    }
    
    batch = {
        "tokens": torch.randint(0, 50, (2, 5)),
        "next_tokens": torch.randint(0, 50, (2, 5)),
        "state_classes": torch.randint(0, 3, (2, 6)),  # T+1
        "loss_mask": torch.ones(2, 5, dtype=torch.bool),
    }
    
    losses = compute_multi_task_loss(outputs, batch)
    
    assert "loss" in losses
    assert "next_token_loss" in losses
    assert "class_loss" in losses
    assert losses["loss"] > 0

def test_training_smoke():
    """Smoke test for training loop."""
    regex_def = RegexDefinition(
        alphabet=("a",),
        patterns=(("a+", "accept"),)
    )
    
    model_cfg = ModelConfig(
        vocab_size=4,  # PAD, EOS, a, reject
        num_classes=3,
        num_states=10,
        d_model=32,
        n_heads=2,
        d_mlp=64,
        max_seq_len=16,
    )
    
    train_cfg = TrainConfig(
        epochs=2,
        lr=1e-3,
        batch_size=4,
        seed=42,
        device="cpu",
        patience=10,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results = train_one_experiment(
            regex_def,
            model_cfg,
            train_cfg,
            results_dir=Path(tmpdir),
            n_samples=32,  # Small dataset
        )
        
        assert "val_metrics" in results
        assert "test_metrics" in results
        # Training completes (may have NaN losses with tiny dataset)
