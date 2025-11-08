"""Tests for deterministic splitting."""

import pytest
from ..split import split_of


def test_split_deterministic():
    """Test that same tokens always map to same split."""
    tokens = [1, 2, 3, 4, 5]

    split1 = split_of(tokens)
    split2 = split_of(tokens)

    assert split1 == split2


def test_split_different_tokens():
    """Test that different tokens can map to different splits."""
    # Generate many different token sequences
    splits = []
    for i in range(100):
        tokens = [i, i+1, i+2]
        splits.append(split_of(tokens))

    # Should have at least 2 different splits in 100 samples
    assert len(set(splits)) >= 2


def test_split_no_overlap():
    """Test that a given sequence only maps to one split."""
    tokens = [1, 2, 3]
    split = split_of(tokens)

    # Verify it's one of the valid splits
    assert split in ["train", "val", "test"]


def test_split_distribution():
    """Test that splits roughly follow 70/15/15 distribution."""
    splits = []
    for i in range(1000):
        tokens = [i, i % 7, i % 13]
        splits.append(split_of(tokens))

    train_count = splits.count("train")
    val_count = splits.count("val")
    test_count = splits.count("test")

    # Allow some tolerance (±10%)
    assert 600 <= train_count <= 800  # ~70%
    assert 50 <= val_count <= 250     # ~15%
    assert 50 <= test_count <= 250    # ~15%


def test_split_empty_tokens():
    """Test that empty token list has a consistent split."""
    split1 = split_of([])
    split2 = split_of([])

    assert split1 == split2
    assert split1 in ["train", "val", "test"]
