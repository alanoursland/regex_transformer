"""Tests for PyTorch dataset wrapper."""

import pytest
import torch

from ...fsm.regex_def import RegexDefinition
from ...fsm.compile import compile_regex
from ..dataset import FsmDataset, collate_fn


def test_dataset_basic():
    """Test basic dataset functionality."""
    regex_def = RegexDefinition(
        alphabet=('a', 'b'),
        patterns=(('a+', 'accept'),)
    )

    fsm = compile_regex(regex_def)

    # Create simple samples
    samples = [[0, 0], [0, 0, 0], [0]]  # 'aa', 'aaa', 'a'

    dataset = FsmDataset(fsm, samples, split="train")

    # Check length
    assert len(dataset) == 3

    # Check first sample
    item = dataset[0]
    assert "tokens" in item
    assert "next_tokens" in item
    assert "states" in item
    assert "state_classes" in item
    assert "mask" in item


def test_dataset_alignment():
    """Test that tokens, states, and labels are correctly aligned."""
    regex_def = RegexDefinition(
        alphabet=('a',),
        patterns=(('a+', 'accept'),)
    )

    fsm = compile_regex(regex_def)
    samples = [[0, 0, 0]]  # 'aaa'

    dataset = FsmDataset(fsm, samples, split="train")
    item = dataset[0]

    # Tokens should have length 3
    assert len(item["tokens"]) == 3

    # States should have length 4 (one more than tokens)
    assert len(item["states"]) == 4

    # Next tokens should have length 3
    assert len(item["next_tokens"]) == 3

    # Verify next_tokens is shifted
    assert torch.equal(item["next_tokens"][:-1], item["tokens"][1:])


def test_dataset_with_eos():
    """Test dataset with EOS token."""
    regex_def = RegexDefinition(
        alphabet=('a',),
        patterns=(('a+', 'accept'),)
    )

    fsm = compile_regex(regex_def)
    samples = [[0, 0]]  # 'aa'

    dataset = FsmDataset(fsm, samples, split="train", eos_id=999)
    item = dataset[0]

    # Should have EOS appended
    assert item["tokens"][-1] == 999
    assert len(item["tokens"]) == 3  # Original 2 + EOS


def test_collate_fn():
    """Test collate function pads correctly."""
    regex_def = RegexDefinition(
        alphabet=('a',),
        patterns=(('a+', 'accept'),)
    )

    fsm = compile_regex(regex_def)
    samples = [[0], [0, 0], [0, 0, 0]]  # Different lengths

    dataset = FsmDataset(fsm, samples, split="train")

    # Get batch
    batch_items = [dataset[i] for i in range(3)]
    batched = collate_fn(batch_items)

    # All sequences should be padded to max length (3)
    assert batched["tokens"].shape == (3, 3)
    assert batched["states"].shape == (3, 4)  # States is one longer
    assert batched["mask"].shape == (3, 3)

    # Verify mask is correct
    assert batched["mask"][0, 0] == True  # First token of first sample
    assert batched["mask"][0, 1] == False  # Padded position
