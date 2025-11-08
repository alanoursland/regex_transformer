"""Tests for quota manager."""

import pytest
from ..quotas import QuotaManager


def test_quota_init():
    """Test quota manager initialization."""
    quota = QuotaManager(num_states=10, num_tokens=5)

    assert quota.num_states == 10
    assert quota.num_tokens == 5
    assert len(quota.edge_hits) == 0
    assert len(quota.state_hits) == 0


def test_edge_weight_decreases():
    """Test that edge weight decreases as edge is used."""
    quota = QuotaManager(num_states=10, num_tokens=2)

    # Initial weight
    weight1 = quota.edge_weight(0, 0)

    # Use the edge
    quota.update_path([0, 1], [0])

    # Weight should decrease
    weight2 = quota.edge_weight(0, 0)
    assert weight2 < weight1


def test_state_weight_decreases():
    """Test that state weight decreases as state is visited."""
    quota = QuotaManager(num_states=10, num_tokens=2)

    # Initial weight
    weight1 = quota.state_weight(5)

    # Visit the state
    quota.update_path([5, 6], [0])

    # Weight should decrease
    weight2 = quota.state_weight(5)
    assert weight2 < weight1


def test_update_path():
    """Test that update_path correctly updates counts."""
    quota = QuotaManager(num_states=10, num_tokens=2)

    # Update with a path
    states = [0, 1, 2]
    tokens = [0, 1]

    quota.update_path(states, tokens)

    # Check state counts
    assert quota.state_hits[0] == 1
    assert quota.state_hits[1] == 1
    assert quota.state_hits[2] == 1

    # Check edge counts
    assert quota.edge_hits[(0, 0)] == 1
    assert quota.edge_hits[(1, 1)] == 1


def test_summary():
    """Test that summary returns correct stats."""
    quota = QuotaManager(num_states=10, num_tokens=2)

    # Initial summary
    summary = quota.summary()
    assert summary["edges_covered"] == 0
    assert summary["states_covered"] == 0

    # Add some paths
    quota.update_path([0, 1], [0])
    quota.update_path([1, 2], [1])

    summary = quota.summary()
    assert summary["edges_covered"] == 2
    assert summary["states_covered"] == 3
    assert summary["total_edge_hits"] == 2
    assert summary["total_state_hits"] == 4  # 0,1,1,2
