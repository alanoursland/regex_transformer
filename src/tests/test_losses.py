"""Unit tests for loss computation."""

import torch
import pytest
from train.losses import compute_multi_task_loss


class TestLossComputation:
    """Test multi-task loss computation."""
    
    def test_basic_loss_is_finite(self):
        """Test that basic loss computation produces finite values."""
        B, T, V, C = 2, 5, 4, 3
        
        outputs = {
            "next_token_logits": torch.randn(B, T, V),
            "class_logits": torch.randn(B, T, C),
        }
        
        batch = {
            "tokens": torch.randint(0, V, (B, T)),
            "next_tokens": torch.randint(0, V, (B, T)),
            "state_classes": torch.randint(0, C, (B, T+1)),
            "loss_mask": torch.ones(B, T, dtype=torch.bool),
        }
        
        losses = compute_multi_task_loss(outputs, batch)
        
        assert not torch.isnan(losses["loss"]), "Total loss is NaN"
        assert not torch.isinf(losses["loss"]), "Total loss is Inf"
        assert not torch.isnan(losses["next_token_loss"]), "Next token loss is NaN"
        assert not torch.isnan(losses["class_loss"]), "Class loss is NaN"
        assert losses["loss"].item() >= 0, "Loss should be non-negative"
    
    def test_loss_with_partial_masking(self):
        """Test loss with some positions masked out."""
        B, T, V, C = 2, 5, 4, 3
        
        outputs = {
            "next_token_logits": torch.randn(B, T, V),
            "class_logits": torch.randn(B, T, C),
        }
        
        batch = {
            "tokens": torch.randint(0, V, (B, T)),
            "next_tokens": torch.randint(0, V, (B, T)),
            "state_classes": torch.randint(0, C, (B, T+1)),
            "loss_mask": torch.zeros(B, T, dtype=torch.bool),
        }
        # Only first 3 positions valid in each sequence
        batch["loss_mask"][:, :3] = True
        
        losses = compute_multi_task_loss(outputs, batch)
        
        assert not torch.isnan(losses["loss"]), "Loss is NaN with partial mask"
        assert torch.isfinite(losses["loss"]), "Loss not finite with partial mask"
    
    def test_loss_with_single_valid_position(self):
        """Test loss with only one valid position per sequence."""
        B, T, V, C = 2, 5, 4, 3
        
        outputs = {
            "next_token_logits": torch.randn(B, T, V),
            "class_logits": torch.randn(B, T, C),
        }
        
        batch = {
            "tokens": torch.randint(0, V, (B, T)),
            "next_tokens": torch.randint(0, V, (B, T)),
            "state_classes": torch.randint(0, C, (B, T+1)),
            "loss_mask": torch.zeros(B, T, dtype=torch.bool),
        }
        batch["loss_mask"][:, 0] = True  # Only first position
        
        losses = compute_multi_task_loss(outputs, batch)
        
        assert not torch.isnan(losses["loss"]), "Loss is NaN with single valid position"
        assert torch.isfinite(losses["loss"]), "Loss not finite with single valid position"
    
    def test_loss_with_correct_predictions(self):
        """Test that perfect predictions give low loss."""
        B, T, V, C = 2, 5, 4, 3
        
        # Create targets
        next_targets = torch.randint(0, V, (B, T))
        class_targets = torch.randint(0, C, (B, T+1))
        
        # Create perfect logits (very high scores for correct class)
        next_logits = torch.full((B, T, V), -10.0)
        class_logits = torch.full((B, T, C), -10.0)
        
        for b in range(B):
            for t in range(T):
                next_logits[b, t, next_targets[b, t]] = 10.0
                class_logits[b, t, class_targets[b, t+1]] = 10.0
        
        outputs = {
            "next_token_logits": next_logits,
            "class_logits": class_logits,
        }
        
        batch = {
            "tokens": torch.zeros(B, T, dtype=torch.long),
            "next_tokens": next_targets,
            "state_classes": class_targets,
            "loss_mask": torch.ones(B, T, dtype=torch.bool),
        }
        
        losses = compute_multi_task_loss(outputs, batch)
        
        # Perfect predictions should have very low loss
        assert losses["loss"].item() < 0.1, f"Perfect predictions have high loss: {losses['loss'].item()}"
    
    def test_loss_with_wrong_predictions(self):
        """Test that wrong predictions give high loss."""
        B, T, V, C = 2, 5, 4, 3
        
        # Create targets
        next_targets = torch.zeros(B, T, dtype=torch.long)  # All predict class 0
        class_targets = torch.zeros(B, T+1, dtype=torch.long)
        
        # Create wrong logits (high scores for wrong classes)
        next_logits = torch.full((B, T, V), -10.0)
        class_logits = torch.full((B, T, C), -10.0)
        
        # Put high scores on wrong classes
        next_logits[:, :, 1] = 10.0  # Predict class 1, but target is 0
        class_logits[:, :, 1] = 10.0
        
        outputs = {
            "next_token_logits": next_logits,
            "class_logits": class_logits,
        }
        
        batch = {
            "tokens": torch.zeros(B, T, dtype=torch.long),
            "next_tokens": next_targets,
            "state_classes": class_targets,
            "loss_mask": torch.ones(B, T, dtype=torch.bool),
        }
        
        losses = compute_multi_task_loss(outputs, batch)
        
        # Wrong predictions should have high loss
        assert losses["loss"].item() > 1.0, f"Wrong predictions have low loss: {losses['loss'].item()}"
    
    def test_loss_weights_affect_magnitude(self):
        """Test that lambda weights properly scale loss components."""
        B, T, V, C = 2, 5, 4, 3
        
        outputs = {
            "next_token_logits": torch.randn(B, T, V),
            "class_logits": torch.randn(B, T, C),
        }
        
        batch = {
            "tokens": torch.randint(0, V, (B, T)),
            "next_tokens": torch.randint(0, V, (B, T)),
            "state_classes": torch.randint(0, C, (B, T+1)),
            "loss_mask": torch.ones(B, T, dtype=torch.bool),
        }
        
        # Compute with different weights
        losses1 = compute_multi_task_loss(outputs, batch, lambda_next=1.0, lambda_class=0.0)
        losses2 = compute_multi_task_loss(outputs, batch, lambda_next=0.0, lambda_class=1.0)
        losses3 = compute_multi_task_loss(outputs, batch, lambda_next=1.0, lambda_class=1.0)
        
        # Check that weights work
        assert torch.allclose(losses1["loss"], losses1["next_token_loss"])
        assert torch.allclose(losses2["loss"], losses2["class_loss"])
        # Total should be sum when both weights are 1
        expected = losses1["next_token_loss"] + losses2["class_loss"]
        assert torch.allclose(losses3["loss"], expected, atol=1e-5)


class TestLossEdgeCases:
    """Test edge cases in loss computation."""
    
    def test_empty_mask_does_not_crash(self):
        """Test that empty mask (all False) doesn't crash."""
        B, T, V, C = 2, 5, 4, 3
        
        outputs = {
            "next_token_logits": torch.randn(B, T, V),
            "class_logits": torch.randn(B, T, C),
        }
        
        batch = {
            "tokens": torch.randint(0, V, (B, T)),
            "next_tokens": torch.randint(0, V, (B, T)),
            "state_classes": torch.randint(0, C, (B, T+1)),
            "loss_mask": torch.zeros(B, T, dtype=torch.bool),  # All masked!
        }
        
        losses = compute_multi_task_loss(outputs, batch)
        
        # Should not crash, loss should be 0 or very small
        assert not torch.isnan(losses["loss"]), "Loss is NaN with empty mask"
        assert torch.isfinite(losses["loss"]), "Loss not finite with empty mask"
    
    def test_ignore_index_works(self):
        """Test that -1 targets are properly ignored."""
        B, T, V, C = 2, 5, 4, 3
        
        outputs = {
            "next_token_logits": torch.randn(B, T, V),
            "class_logits": torch.randn(B, T, C),
        }
        
        batch = {
            "tokens": torch.randint(0, V, (B, T)),
            "next_tokens": torch.full((B, T), -1, dtype=torch.long),  # All -1
            "state_classes": torch.randint(0, C, (B, T+1)),
            "loss_mask": torch.ones(B, T, dtype=torch.bool),
        }
        
        losses = compute_multi_task_loss(outputs, batch)
        
        # -1 targets should be ignored, shouldn't cause NaN
        assert not torch.isnan(losses["loss"]), "Loss is NaN with -1 targets"
        assert torch.isfinite(losses["loss"]), "Loss not finite with -1 targets"
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        B, T, V, C = 1, 5, 4, 3
        
        outputs = {
            "next_token_logits": torch.randn(B, T, V),
            "class_logits": torch.randn(B, T, C),
        }
        
        batch = {
            "tokens": torch.randint(0, V, (B, T)),
            "next_tokens": torch.randint(0, V, (B, T)),
            "state_classes": torch.randint(0, C, (B, T+1)),
            "loss_mask": torch.ones(B, T, dtype=torch.bool),
        }
        
        losses = compute_multi_task_loss(outputs, batch)
        
        assert not torch.isnan(losses["loss"]), "Loss is NaN with batch_size=1"
        assert torch.isfinite(losses["loss"]), "Loss not finite with batch_size=1"


class TestLossGradients:
    """Test that loss produces valid gradients."""
    
    def test_loss_backward_produces_finite_gradients(self):
        """Test that backward pass on loss produces finite gradients."""
        B, T, V, C = 2, 5, 4, 3
        
        # Create model parameters
        next_logits = torch.randn(B, T, V, requires_grad=True)
        class_logits = torch.randn(B, T, C, requires_grad=True)
        
        outputs = {
            "next_token_logits": next_logits,
            "class_logits": class_logits,
        }
        
        batch = {
            "tokens": torch.randint(0, V, (B, T)),
            "next_tokens": torch.randint(0, V, (B, T)),
            "state_classes": torch.randint(0, C, (B, T+1)),
            "loss_mask": torch.ones(B, T, dtype=torch.bool),
        }
        
        losses = compute_multi_task_loss(outputs, batch)
        losses["loss"].backward()
        
        # Check gradients are finite
        assert next_logits.grad is not None
        assert not torch.isnan(next_logits.grad).any(), "next_logits gradient contains NaN"
        assert not torch.isinf(next_logits.grad).any(), "next_logits gradient contains Inf"
        
        assert class_logits.grad is not None
        assert not torch.isnan(class_logits.grad).any(), "class_logits gradient contains NaN"
        assert not torch.isinf(class_logits.grad).any(), "class_logits gradient contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])