"""Unit tests for data collation and masking."""

import torch
import pytest
from data.collate import collate_batch
from data.dataset import FsmDataset
from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex


class TestCollation:
    """Test batch collation logic."""
    
    def test_collate_single_sequence(self):
        """Test collating a single sequence."""
        examples = [
            {
                "tokens": torch.tensor([0, 1, 0]),
                "next_tokens": torch.tensor([1, 0, -1]),
                "states": torch.tensor([0, 1, 2, 3]),
                "state_classes": torch.tensor([0, 1, 1, 2]),
                "mask": torch.ones(3, dtype=torch.bool),
            }
        ]
        
        batch = collate_batch(examples, pad_id=2, eos_id=3)
        
        assert batch["tokens"].shape == (1, 3)
        assert batch["attn_mask"].shape == (1, 3)
        assert batch["loss_mask"].shape == (1, 3)
        assert not torch.isnan(batch["tokens"]).any()
    
    def test_collate_multiple_sequences_same_length(self):
        """Test collating sequences of same length."""
        examples = [
            {
                "tokens": torch.tensor([0, 1, 0]),
                "next_tokens": torch.tensor([1, 0, -1]),
                "states": torch.tensor([0, 1, 2, 3]),
                "state_classes": torch.tensor([0, 1, 1, 2]),
                "mask": torch.ones(3, dtype=torch.bool),
            },
            {
                "tokens": torch.tensor([1, 0, 1]),
                "next_tokens": torch.tensor([0, 1, -1]),
                "states": torch.tensor([0, 2, 3, 4]),
                "state_classes": torch.tensor([0, 1, 2, 2]),
                "mask": torch.ones(3, dtype=torch.bool),
            },
        ]
        
        batch = collate_batch(examples, pad_id=2, eos_id=3)
        
        assert batch["tokens"].shape == (2, 3)
        assert batch["attn_mask"].shape == (2, 3)
        assert batch["loss_mask"].shape == (2, 3)
        # All positions should be marked as valid for attention
        assert batch["attn_mask"].all()
    
    def test_collate_different_lengths(self):
        """Test collating sequences of different lengths."""
        examples = [
            {
                "tokens": torch.tensor([0, 1]),
                "next_tokens": torch.tensor([1, -1]),
                "states": torch.tensor([0, 1, 2]),
                "state_classes": torch.tensor([0, 1, 2]),
                "mask": torch.ones(2, dtype=torch.bool),
            },
            {
                "tokens": torch.tensor([1, 0, 1, 0]),
                "next_tokens": torch.tensor([0, 1, 0, -1]),
                "states": torch.tensor([0, 2, 3, 4, 5]),
                "state_classes": torch.tensor([0, 1, 2, 2, 1]),
                "mask": torch.ones(4, dtype=torch.bool),
            },
        ]
        
        batch = collate_batch(examples, pad_id=2, eos_id=3)
        
        # Should be padded to max length (4)
        assert batch["tokens"].shape == (2, 4)
        assert batch["attn_mask"].shape == (2, 4)
        assert batch["loss_mask"].shape == (2, 4)
        
        # First sequence: attn_mask should be [T, T, F, F]
        assert batch["attn_mask"][0, :2].all()
        assert not batch["attn_mask"][0, 2:].any()
        
        # Second sequence: all valid
        assert batch["attn_mask"][1].all()
    
    def test_loss_mask_has_valid_positions(self):
        """Test that loss_mask has at least some True values."""
        examples = [
            {
                "tokens": torch.tensor([0, 1, 0, 1]),
                "next_tokens": torch.tensor([1, 0, 1, -1]),
                "states": torch.tensor([0, 1, 2, 3, 4]),
                "state_classes": torch.tensor([0, 1, 1, 2, 2]),
                "mask": torch.ones(4, dtype=torch.bool),
            }
        ]
        
        batch = collate_batch(examples, pad_id=2, eos_id=3)
        
        # loss_mask should have at least one True value
        assert batch["loss_mask"].any(), "loss_mask has no valid positions!"
        assert batch["loss_mask"].sum().item() > 0, "loss_mask sum is 0!"
    
    def test_padding_positions_masked(self):
        """Test that padded positions are properly masked."""
        examples = [
            {
                "tokens": torch.tensor([0, 1]),
                "next_tokens": torch.tensor([1, -1]),
                "states": torch.tensor([0, 1, 2]),
                "state_classes": torch.tensor([0, 1, 2]),
                "mask": torch.ones(2, dtype=torch.bool),
            },
            {
                "tokens": torch.tensor([1, 0, 1, 0, 1]),
                "next_tokens": torch.tensor([0, 1, 0, 1, -1]),
                "states": torch.tensor([0, 2, 3, 4, 5, 6]),
                "state_classes": torch.tensor([0, 1, 2, 2, 1, 0]),
                "mask": torch.ones(5, dtype=torch.bool),
            },
        ]
        
        batch = collate_batch(examples, pad_id=2, eos_id=3)
        
        # Padded positions should have attn_mask=False
        assert not batch["attn_mask"][0, 2:].any(), "Padding not masked in attn_mask"
        
        # Padded positions should have loss_mask=False
        assert not batch["loss_mask"][0, 2:].any(), "Padding not masked in loss_mask"


class TestDatasetIntegration:
    """Integration tests for dataset -> collation pipeline."""
    
    def test_fsm_dataset_produces_valid_shapes(self):
        """Test that FsmDataset produces correctly shaped outputs."""
        regex_def = RegexDefinition(
            alphabet=("a", "b"),
            patterns=(("a+", "accept"),)
        )
        fsm = compile_regex(regex_def)
        
        samples = [[0, 0, 0], [1, 0, 1]]  # Token IDs
        dataset = FsmDataset(fsm, samples, "train", eos_id=None)
        
        sample = dataset[0]
        
        assert "tokens" in sample
        assert "next_tokens" in sample
        assert "states" in sample
        assert "state_classes" in sample
        
        # tokens and next_tokens should be same length
        assert len(sample["tokens"]) == len(sample["next_tokens"])
        
        # states should be length + 1 (includes start state)
        assert len(sample["states"]) == len(sample["tokens"]) + 1
        assert len(sample["state_classes"]) == len(sample["tokens"]) + 1
    
    def test_fsm_dataset_with_eos(self):
        """Test that FsmDataset correctly appends EOS."""
        regex_def = RegexDefinition(
            alphabet=("a", "b"),
            patterns=(("a+", "accept"),)
        )
        fsm = compile_regex(regex_def)
        
        samples = [[0, 0, 0]]  # "aaa"
        dataset = FsmDataset(fsm, samples, "train", eos_id=2)
        
        sample = dataset[0]
        
        # Should have EOS appended
        assert sample["tokens"][-1].item() == 2, "EOS not appended"
        
        # Length should be original + 1
        assert len(sample["tokens"]) == 4  # [0, 0, 0, 2]
    
    def test_end_to_end_batch_creation(self):
        """Test full pipeline from FSM to batch."""
        regex_def = RegexDefinition(
            alphabet=("a", "b"),
            patterns=(("a+", "accept"),)
        )
        fsm = compile_regex(regex_def)
        
        samples = [[0, 0], [0, 0, 0], [1, 0]]
        dataset = FsmDataset(fsm, samples, "train", eos_id=2)
        
        # Get batch
        examples = [dataset[i] for i in range(len(samples))]
        batch = collate_batch(examples, pad_id=3, eos_id=2)
        
        # Check shapes
        B = len(samples)
        max_len = max(len(s) for s in samples) + 1  # +1 for EOS
        
        assert batch["tokens"].shape == (B, max_len)
        assert batch["attn_mask"].shape == (B, max_len)
        assert batch["loss_mask"].shape == (B, max_len)
        
        # Check that loss_mask has valid positions
        assert batch["loss_mask"].sum() > 0, "No valid positions in loss_mask!"
        
        # Check no NaN
        assert not torch.isnan(batch["tokens"]).any()


class TestMaskingLogic:
    """Test specific masking logic edge cases."""
    
    def test_single_token_sequence(self):
        """Test sequence with only one token."""
        examples = [
            {
                "tokens": torch.tensor([0]),
                "next_tokens": torch.tensor([-1]),
                "states": torch.tensor([0, 1]),
                "state_classes": torch.tensor([0, 1]),
                "mask": torch.ones(1, dtype=torch.bool),
            }
        ]
        
        batch = collate_batch(examples, pad_id=2, eos_id=3)
        
        # Should have valid output
        assert batch["attn_mask"].any(), "Single token has no valid attn positions"
        # loss_mask might be False for the position (predicting after sequence)
        # but should not crash
        assert batch["loss_mask"].shape == (1, 1)
    
    def test_empty_sequence_handling(self):
        """Test that empty sequences are handled gracefully."""
        # This is an edge case - in practice shouldn't happen, but test robustness
        examples = [
            {
                "tokens": torch.tensor([]),
                "next_tokens": torch.tensor([]),
                "states": torch.tensor([0]),
                "state_classes": torch.tensor([0]),
                "mask": torch.ones(0, dtype=torch.bool),
            }
        ]
        
        # This might raise an error or handle gracefully
        # Main thing is it shouldn't produce NaN
        try:
            batch = collate_batch(examples, pad_id=2, eos_id=3)
            assert not torch.isnan(batch["tokens"]).any()
        except (ValueError, RuntimeError) as e:
            # It's OK to raise an error for empty sequences
            pass
    
    def test_all_padding_batch(self):
        """Test batch where first sequence is very short."""
        examples = [
            {
                "tokens": torch.tensor([0]),
                "next_tokens": torch.tensor([-1]),
                "states": torch.tensor([0, 1]),
                "state_classes": torch.tensor([0, 1]),
                "mask": torch.ones(1, dtype=torch.bool),
            },
            {
                "tokens": torch.tensor([0, 1, 0, 1, 0]),
                "next_tokens": torch.tensor([1, 0, 1, 0, -1]),
                "states": torch.tensor([0, 1, 2, 3, 4, 5]),
                "state_classes": torch.tensor([0, 1, 2, 1, 2, 0]),
                "mask": torch.ones(5, dtype=torch.bool),
            },
        ]
        
        batch = collate_batch(examples, pad_id=2, eos_id=3)
        
        # First sequence should have mostly padding
        assert batch["attn_mask"][0, 0] == True, "First position should be valid"
        assert batch["attn_mask"][0, 1:].sum() == 0, "Rest should be padding"
        
        # But second sequence should be fully valid
        assert batch["attn_mask"][1].all(), "Second sequence should be fully valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])