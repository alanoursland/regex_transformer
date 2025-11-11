"""Tests for feasibility DP."""

import pytest
import numpy as np

from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex
from ..feasibility import can_reach_tables, feasible


def test_feasibility_simple_chain():
    """Test feasibility on simple a+ pattern."""
    regex_def = RegexDefinition(
        alphabet=('a',),
        patterns=(('a+', 'accept'),)
    )

    fsm = compile_regex(regex_def)
    canReach = can_reach_tables(fsm, L_max=5)

    # Find accept class ID
    accept_id = fsm.classes.index('accept')

    # From start state, we should be able to reach accept in 1,2,3,4,5 steps
    for L in [1, 2, 3, 4, 5]:
        assert feasible(canReach, fsm.start, L, accept_id), \
            f"Should be able to reach accept in {L} steps"

    # Should NOT be able to reach accept in 0 steps (start is incomplete)
    assert not feasible(canReach, fsm.start, 0, accept_id), \
        "Should not reach accept in 0 steps from incomplete start"


def test_feasibility_helper_bounds():
    """Test that feasible() handles out-of-bounds gracefully."""
    regex_def = RegexDefinition(
        alphabet=('a',),
        patterns=(('a', 'accept'),)
    )

    fsm = compile_regex(regex_def)
    canReach = can_reach_tables(fsm, L_max=5)

    # Out of bounds should return False
    assert not feasible(canReach, -1, 1, 0)  # Invalid state
    assert not feasible(canReach, 0, -1, 0)  # Invalid time
    assert not feasible(canReach, 0, 1, -1)  # Invalid class
    assert not feasible(canReach, 0, 100, 0)  # Time too large
    assert not feasible(canReach, 1000, 1, 0)  # State too large


def test_feasibility_two_classes():
    """Test feasibility with multiple classes."""
    regex_def = RegexDefinition(
        alphabet=('a', 'b'),
        patterns=(('a+', 'accept'),)
    )

    fsm = compile_regex(regex_def)
    canReach = can_reach_tables(fsm, L_max=5)

    accept_id = fsm.classes.index('accept')
    reject_id = fsm.classes.index('reject')

    # Should be able to reach accept with 'a's
    assert feasible(canReach, fsm.start, 1, accept_id)

    # Should be able to reach reject with 'b'
    assert feasible(canReach, fsm.start, 1, reject_id)


def test_feasibility_shape():
    """Test that canReach has correct shape."""
    regex_def = RegexDefinition(
        alphabet=('a', 'b'),
        patterns=(('a*b*', 'accept'),)
    )

    fsm = compile_regex(regex_def)
    L_max = 10
    canReach = can_reach_tables(fsm, L_max)

    # Check shape
    assert canReach.shape[0] == fsm.states
    assert canReach.shape[1] == L_max + 1
    assert canReach.shape[2] == len(fsm.classes)

    # Check dtype
    assert canReach.dtype == bool
