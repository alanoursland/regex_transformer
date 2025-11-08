"""Coverage quota management for balanced edge/state sampling."""

from collections import defaultdict
from typing import Dict, Tuple, List
import numpy as np


class QuotaManager:
    """
    Tracks edge and state coverage to encourage balanced exploration.

    Uses inverse-sqrt frequency weighting to prioritize under-covered transitions.
    """

    def __init__(self, num_states: int, num_tokens: int):
        """
        Initialize quota manager.

        Args:
            num_states: Number of states in the FSM
            num_tokens: Number of tokens in the alphabet
        """
        self.num_states = num_states
        self.num_tokens = num_tokens

        # Edge counts: (state, token_id) -> count
        self.edge_hits: Dict[Tuple[int, int], int] = defaultdict(int)

        # State visit counts
        self.state_hits: Dict[int, int] = defaultdict(int)

    def edge_weight(self, state: int, token_id: int) -> float:
        """
        Compute sampling weight for an edge.

        Uses inverse-sqrt to prefer under-covered edges.

        Args:
            state: Source state
            token_id: Token/character index

        Returns:
            Weight (higher = prefer this edge)
        """
        count = self.edge_hits[(state, token_id)]
        # Add 1 to avoid division by zero
        return 1.0 / np.sqrt(count + 1.0)

    def state_weight(self, state: int) -> float:
        """
        Compute sampling weight for a state.

        Args:
            state: State ID

        Returns:
            Weight (higher = prefer this state)
        """
        count = self.state_hits[state]
        return 1.0 / np.sqrt(count + 1.0)

    def update_path(self, states: List[int], tokens: List[int]) -> None:
        """
        Update coverage counts after sampling a path.

        Args:
            states: List of states visited (length = len(tokens) + 1)
            tokens: List of tokens (length = len(states) - 1)
        """
        # Update state counts
        for state in states:
            self.state_hits[state] += 1

        # Update edge counts
        for i, token_id in enumerate(tokens):
            state = states[i]
            self.edge_hits[(state, token_id)] += 1

    def summary(self) -> Dict:
        """
        Get summary statistics.

        Returns:
            Dict with coverage stats
        """
        num_edges_covered = len(self.edge_hits)
        num_states_covered = len(self.state_hits)

        return {
            "edges_covered": num_edges_covered,
            "states_covered": num_states_covered,
            "total_edge_hits": sum(self.edge_hits.values()),
            "total_state_hits": sum(self.state_hits.values()),
        }
