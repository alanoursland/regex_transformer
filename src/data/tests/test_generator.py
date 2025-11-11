"""Tests for data generator."""

import pytest
import numpy as np

from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex
from ..generator import GenConfig, generate_corpus, sample_target
from ..feasibility import can_reach_tables


def test_genconfig_defaults():
    """Test GenConfig default values."""
    cfg = GenConfig(L_min=1, L_max=10)

    assert cfg.L_min == 1
    assert cfg.L_max == 10
    assert cfg.p_class is not None
    assert cfg.reject_mix is not None
    assert abs(sum(cfg.p_class.values()) - 1.0) < 1e-6


def test_genconfig_validation():
    """Test that GenConfig validates probabilities."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        GenConfig(p_class={"accept": 0.5, "reject": 0.3})


def test_sample_target():
    """Test target sampling."""
    regex_def = RegexDefinition(
        alphabet=('a',),
        patterns=(('a+', 'accept'),)
    )

    fsm = compile_regex(regex_def)
    cfg = GenConfig(L_min=1, L_max=10)
    canReach = can_reach_tables(fsm, cfg.L_max)
    rng = np.random.default_rng(42)

    # Sample multiple targets
    targets = [sample_target(cfg, fsm, canReach, rng) for _ in range(100)]

    # Check that we got various lengths
    lengths = [L for _, L in targets]
    assert min(lengths) >= cfg.L_min
    assert max(lengths) <= cfg.L_max

    # Check that we got various classes
    class_ids = [c for c, _ in targets]
    assert len(set(class_ids)) > 1  # At least 2 different classes


def test_generate_corpus_simple():
    """Test corpus generation on simple pattern."""
    regex_def = RegexDefinition(
        alphabet=('a',),
        patterns=(('a+', 'accept'),)
    )

    fsm = compile_regex(regex_def)
    cfg = GenConfig(L_min=1, L_max=5, p_class={"accept": 1.0})

    samples, class_names, report = generate_corpus(fsm, cfg, n_samples=10, seed=42)

    # Check we got 10 samples
    assert len(samples) == 10
    assert len(class_names) == 10

    # All should be accept
    assert all(c == "accept" for c in class_names)

    # Lengths should be in range
    for tokens in samples:
        assert cfg.L_min <= len(tokens) <= cfg.L_max

    # Verify samples actually match the FSM
    for tokens in samples:
        states = fsm.trace(tokens)
        final_state = states[-1]
        final_class = fsm.classify_name(final_state)
        assert final_class == "accept"


def test_generate_corpus_report():
    """Test that corpus generation produces valid report."""
    regex_def = RegexDefinition(
        alphabet=('a', 'b'),
        patterns=(('a*b*', 'accept'),)
    )

    fsm = compile_regex(regex_def)

    # Build p_class from FSM's actual classes
    p_class = {cls: 1.0/len(fsm.classes) for cls in fsm.classes}
    cfg = GenConfig(L_min=1, L_max=5, p_class=p_class)

    samples, class_names, report = generate_corpus(fsm, cfg, n_samples=20, seed=42)

    # Check report structure
    assert report.n_samples > 0
    assert len(report.length_histogram) > 0
    assert len(report.class_histogram) > 0
    assert report.edge_coverage >= 0
    assert report.state_coverage >= 0
    assert 0.0 <= report.retry_rate <= 1.0


def test_generate_corpus_deterministic():
    """Test that corpus generation is deterministic with same seed."""
    regex_def = RegexDefinition(
        alphabet=('a',),
        patterns=(('a+', 'accept'),)
    )

    fsm = compile_regex(regex_def)
    cfg = GenConfig(L_min=1, L_max=5)

    samples1, class_names1, _ = generate_corpus(fsm, cfg, n_samples=10, seed=42)
    samples2, class_names2, _ = generate_corpus(fsm, cfg, n_samples=10, seed=42)

    assert samples1 == samples2
    assert class_names1 == class_names2


def test_generate_corpus_mixed_classes():
    """Test corpus generation with mixed classes."""
    regex_def = RegexDefinition(
        alphabet=('a', 'b'),
        patterns=(('a+', 'accept'),)
    )

    fsm = compile_regex(regex_def)
    cfg = GenConfig(
        L_min=1,
        L_max=5,
        p_class={"accept": 0.5, "incomplete": 0.25, "reject": 0.25}
    )

    samples, class_names, report = generate_corpus(fsm, cfg, n_samples=50, seed=42)

    # Should have multiple classes
    unique_classes = set(class_names)
    assert len(unique_classes) >= 2

    # Report should show class distribution
    assert len(report.class_histogram) >= 2
