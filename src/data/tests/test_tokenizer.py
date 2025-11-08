"""Tests for vocabulary and tokenization."""

import pytest
from ..tokenizer import Vocab


def test_vocab_from_alphabet():
    """Test creating vocab from alphabet."""
    alphabet = ("a", "b", "c")
    vocab = Vocab.from_alphabet(alphabet)

    # Check special tokens
    assert vocab.pad_id == 0
    assert vocab.eos_id == 1
    assert vocab.itos[0] == "<PAD>"
    assert vocab.itos[1] == "<EOS>"

    # Check alphabet symbols
    assert vocab.stoi["a"] == 2
    assert vocab.stoi["b"] == 3
    assert vocab.stoi["c"] == 4
    assert len(vocab) == 5  # PAD, EOS, a, b, c


def test_vocab_encode_decode():
    """Test encoding and decoding."""
    alphabet = ("a", "b")
    vocab = Vocab.from_alphabet(alphabet)

    # Encode
    text = "aba"
    ids = vocab.encode(text)
    assert ids == [2, 3, 2]  # a=2, b=3, a=2

    # Decode
    decoded = vocab.decode(ids)
    assert decoded == text


def test_vocab_decode_skips_special_tokens():
    """Test that decode skips special tokens."""
    alphabet = ("a",)
    vocab = Vocab.from_alphabet(alphabet)

    # Encode with special tokens mixed in
    ids = [0, 2, 1, 2, 0]  # PAD, a, EOS, a, PAD
    decoded = vocab.decode(ids)

    # Should only decode 'a' tokens
    assert decoded == "aa"


def test_vocab_validate_alphabet():
    """Test alphabet validation."""
    alphabet = ("a", "b")
    vocab = Vocab.from_alphabet(alphabet)

    # Valid alphabet
    vocab.validate_alphabet(alphabet)

    # Invalid alphabet (has 'c' not in vocab)
    with pytest.raises(ValueError, match="not in vocabulary"):
        vocab.validate_alphabet(("a", "b", "c"))


def test_vocab_without_special_tokens():
    """Test creating vocab without special tokens."""
    alphabet = ("x", "y")
    vocab = Vocab.from_alphabet(alphabet, add_special_tokens=False)

    assert vocab.pad_id is None
    assert vocab.eos_id is None
    assert vocab.stoi["x"] == 0
    assert vocab.stoi["y"] == 1
    assert len(vocab) == 2


def test_vocab_stable_ids():
    """Test that IDs are stable across multiple creations."""
    alphabet = ("a", "b", "c")

    vocab1 = Vocab.from_alphabet(alphabet)
    vocab2 = Vocab.from_alphabet(alphabet)

    assert vocab1.stoi == vocab2.stoi
    assert vocab1.itos == vocab2.itos
