"""Tests for RegexDefinition."""

import pytest
from ..regex_def import RegexDefinition


def test_basic_regex_def():
    """Test basic regex definition creation."""
    regex_def = RegexDefinition(
        alphabet=('a', 'b'),
        patterns=(('a+', 'accept'),)
    )

    assert regex_def.alphabet == ('a', 'b')
    assert regex_def.patterns == (('a+', 'accept'),)


def test_invalid_alphabet_multi_char():
    """Test that multi-character alphabet symbols are rejected."""
    with pytest.raises(ValueError, match="single characters"):
        RegexDefinition(
            alphabet=('a', 'ab', 'c'),
            patterns=(('a+', 'accept'),)
        )


def test_duplicate_alphabet():
    """Test that duplicate alphabet symbols are rejected."""
    with pytest.raises(ValueError, match="duplicate"):
        RegexDefinition(
            alphabet=('a', 'b', 'a'),
            patterns=(('a+', 'accept'),)
        )


def test_invalid_regex():
    """Test that invalid regex patterns are rejected."""
    with pytest.raises(ValueError, match="Invalid regex"):
        RegexDefinition(
            alphabet=('a', 'b'),
            patterns=(('[', 'accept'),)  # Unclosed bracket
        )


def test_duplicate_class_names():
    """Test that duplicate class names are rejected."""
    with pytest.raises(ValueError, match="Duplicate class"):
        RegexDefinition(
            alphabet=('a', 'b'),
            patterns=(
                ('a+', 'accept'),
                ('b+', 'accept'),  # Duplicate class name
            )
        )


def test_empty_patterns():
    """Test that at least one pattern is required."""
    with pytest.raises(ValueError, match="at least one pattern"):
        RegexDefinition(
            alphabet=('a', 'b'),
            patterns=()
        )
