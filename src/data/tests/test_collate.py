"""Tests for collation and batching."""

import pytest
import torch

from ...fsm.regex_def import RegexDefinition
from ...fsm.compile import compile_regex
from dataset import FsmDataset
from ..collate import collate_batch


def test_collate_basic():
    """Test basic collation."""
    regex_def = RegexDefinition(
        alphabet=("a",),
        patterns=(("a+", "accept"),),
    )

    fsm = compile_regex(regex_def)
    samples = [[0], [0, 0]]  # 'a', 'aa'

    dataset = FsmDataset(fsm, samples, split="train")

    # Get examples
    examples = [dataset[0], dataset[1]]

    # Collate
    batch = collate_batch(examples, pad_id=999, eos_id=998)

    # Check shapes
    assert batch["tokens"].shape == (2, 2)  # Padded to max length
    assert batch["next_tokens"].shape == (2, 2)
    assert batch["states"].shape == (2, 3)  # States is T+1
    assert batch["attn_mask"].shape == (2, 2)
    assert batch["loss_mask"].shape == (2, 2)


def test_collate_padding():
    """Test that shorter sequences are padded correctly."""
    regex_def = RegexDefinition(
        alphabet=("a",),
        patterns=(("a+", "accept"),),
    )

    fsm = compile_regex(regex_def)
    samples = [[0], [0, 0, 0]]  # 'a', 'aaa' (different lengths)

    dataset = FsmDataset(fsm, samples, split="train")
    examples = [dataset[0], dataset[1]]

    PAD_ID = 999
    batch = collate_batch(examples, pad_id=PAD_ID, eos_id=998)

    # First sequence should be padded
    assert batch["tokens"][0, 0] == 0  # 'a'
    assert batch["tokens"][0, 1] == PAD_ID  # padding
    assert batch["tokens"][0, 2] == PAD_ID  # padding

    # Second sequence should not be padded
    assert batch["tokens"][1, 0] == 0  # 'a'
    assert batch["tokens"][1, 1] == 0  # 'a'
    assert batch["tokens"][1, 2] == 0  # 'a'


def test_collate_attention_mask():
    """Test that attention mask is correct."""
    regex_def = RegexDefinition(
        alphabet=("a",),
        patterns=(("a+", "accept"),),
    )

    fsm = compile_regex(regex_def)
    samples = [[0], [0, 0, 0]]

    dataset = FsmDataset(fsm, samples, split="train")
    examples = [dataset[0], dataset[1]]

    batch = collate_batch(examples, pad_id=999, eos_id=998)

    # First sequence: length 1, so mask should be [True, False, False]
    assert batch["attn_mask"][0, 0] == True
    assert batch["attn_mask"][0, 1] == False
    assert batch["attn_mask"][0, 2] == False

    # Second sequence: length 3, so mask should be [True, True, True]
    assert batch["attn_mask"][1, 0] == True
    assert batch["attn_mask"][1, 1] == True
    assert batch["attn_mask"][1, 2] == True


def test_collate_loss_mask():
    """Test that loss mask excludes last position."""
    regex_def = RegexDefinition(
        alphabet=("a",),
        patterns=(("a+", "accept"),),
    )

    fsm = compile_regex(regex_def)
    samples = [[0, 0]]  # 'aa'

    dataset = FsmDataset(fsm, samples, split="train")
    examples = [dataset[0]]

    batch = collate_batch(examples, pad_id=999, eos_id=998)

    # Sequence has length 2
    # Loss mask should be [True, False] (last position excluded)
    assert batch["loss_mask"][0, 0] == True
    assert batch["loss_mask"][0, 1] == False


def test_collate_empty_batch():
    """Test collating an empty batch."""
    # Empty batches should raise ValueError
    with pytest.raises(ValueError):
        collate_batch([], pad_id=999, eos_id=998)
