"""Tests for DataLoader construction."""

import pytest
import torch

from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex
from data.dataset import FsmDataset
from ..tokenizer import Vocab
from ..loader import make_dataloaders


def test_make_dataloaders_basic():
    """Test basic dataloader creation."""
    regex_def = RegexDefinition(
        alphabet=("a",),
        patterns=(("a+", "accept"),),
    )

    fsm = compile_regex(regex_def)
    samples_train = [[0], [0, 0], [0, 0, 0]]
    samples_val = [[0, 0]]

    datasets = {
        "train": FsmDataset(fsm, samples_train, split="train"),
        "val": FsmDataset(fsm, samples_val, split="val"),
    }

    vocab = Vocab.from_alphabet(regex_def.alphabet)

    dataloaders = make_dataloaders(
        datasets, vocab, batch_size=2, seed=42, num_workers=0
    )

    assert "train" in dataloaders
    assert "val" in dataloaders

    # Check one batch from train
    batch = next(iter(dataloaders["train"]))
    assert "tokens" in batch
    assert "attn_mask" in batch


def test_dataloader_reproducibility():
    """Test that same seed produces same batch order."""
    regex_def = RegexDefinition(
        alphabet=("a",),
        patterns=(("a+", "accept"),),
    )

    fsm = compile_regex(regex_def)
    samples = [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0]]

    datasets = {
        "train": FsmDataset(fsm, samples, split="train"),
    }

    vocab = Vocab.from_alphabet(regex_def.alphabet)

    # Create two dataloaders with same seed
    dl1 = make_dataloaders(datasets, vocab, batch_size=2, seed=42)["train"]
    dl2 = make_dataloaders(datasets, vocab, batch_size=2, seed=42)["train"]

    # Get first batch from each
    batch1 = next(iter(dl1))
    batch2 = next(iter(dl2))

    # Should be identical (same shuffle order)
    assert torch.equal(batch1["tokens"], batch2["tokens"])


def test_dataloader_no_shuffle_for_val():
    """Test that validation loader doesn't shuffle."""
    regex_def = RegexDefinition(
        alphabet=("a",),
        patterns=(("a+", "accept"),),
    )

    fsm = compile_regex(regex_def)
    samples = [[0], [0, 0], [0, 0, 0]]

    datasets = {
        "val": FsmDataset(fsm, samples, split="val"),
    }

    vocab = Vocab.from_alphabet(regex_def.alphabet)

    # Create two val dataloaders
    dl1 = make_dataloaders(datasets, vocab, batch_size=2, seed=42)["val"]
    dl2 = make_dataloaders(datasets, vocab, batch_size=2, seed=99)["val"]

    # Get first batch from each
    batch1 = next(iter(dl1))
    batch2 = next(iter(dl2))

    # Should be identical (no shuffling for val)
    assert torch.equal(batch1["tokens"], batch2["tokens"])
