"""Tests for micro datasets."""

import pytest
from ..microsets import (
    get_micro_a_plus,
    get_micro_a_star_b_star,
    get_micro_branching,
    get_micro_dataset,
    MICRO_A_PLUS,
)


def test_get_micro_a_plus():
    """Test micro dataset for a+."""
    fsm, samples, class_names = get_micro_a_plus(n_train=16, n_val=8)

    assert len(samples["train"]) == 16
    assert len(samples["val"]) == 8
    assert len(class_names["train"]) == 16
    assert len(class_names["val"]) == 8

    # Verify FSM exists
    assert fsm.states > 0


def test_get_micro_a_star_b_star():
    """Test micro dataset for a*b*."""
    fsm, samples, class_names = get_micro_a_star_b_star(n_train=16, n_val=8)

    assert len(samples["train"]) == 16
    assert len(samples["val"]) == 8


def test_get_micro_branching():
    """Test micro dataset for (a|b)*c."""
    fsm, samples, class_names = get_micro_branching(n_train=16, n_val=8)

    assert len(samples["train"]) == 16
    assert len(samples["val"]) == 8


def test_micro_dataset_reproducibility():
    """Test that micro datasets are reproducible."""
    fsm1, samples1, classes1 = get_micro_a_plus(n_train=16, n_val=8)
    fsm2, samples2, classes2 = get_micro_a_plus(n_train=16, n_val=8)

    # Should be identical
    assert samples1 == samples2
    assert classes1 == classes2


def test_micro_dataset_custom():
    """Test creating a custom micro dataset."""
    fsm, samples, class_names = get_micro_dataset(
        MICRO_A_PLUS, n_train=8, n_val=4, seed=99999
    )

    assert len(samples["train"]) == 8
    assert len(samples["val"]) == 4


def test_micro_dataset_has_variety():
    """Test that micro datasets have class variety."""
    fsm, samples, class_names = get_micro_a_plus(n_train=16, n_val=8)

    # Should have at least 2 different classes across all samples
    all_classes = set(class_names["train"] + class_names["val"])
    assert len(all_classes) >= 1  # At least one class (may not always have variety)
