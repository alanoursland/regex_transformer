"""Feasibility dynamic programming for FSM reachability.

Computes whether a given state can reach a terminal state of a specific
class within exactly t steps.
"""

import numpy as np
from typing import Tuple

from fsm.dfa import FSM


def can_reach_tables(fsm: FSM, L_max: int) -> np.ndarray:
    """
    Compute reachability table using backward DP.

    Args:
        fsm: The finite state machine
        L_max: Maximum sequence length to consider

    Returns:
        Boolean array of shape (states, L_max+1, num_classes)
        where canReach[s, t, c] = True if from state s, we can reach
        a state of class c in exactly t steps.
    """
    num_states = fsm.states
    num_classes = len(fsm.classes)

    # canReach[state, t, class_id]
    canReach = np.zeros((num_states, L_max + 1, num_classes), dtype=bool)

    # Base case: t=0, a state can "reach" its own class in 0 steps
    for state in range(num_states):
        class_id = fsm.state_class[state]
        canReach[state, 0, class_id] = True

    # DP: for each time step t from 1 to L_max
    for t in range(1, L_max + 1):
        for state in range(num_states):
            for token_id in range(len(fsm.alphabet)):
                next_state = fsm.delta[(state, token_id)]
                # If next_state can reach class c in t-1 steps,
                # then current state can reach it in t steps
                for class_id in range(num_classes):
                    if canReach[next_state, t - 1, class_id]:
                        canReach[state, t, class_id] = True

    return canReach


def feasible(canReach: np.ndarray, state: int, t: int, class_id: int) -> bool:
    """
    Helper to check if a state can reach a class in exactly t steps.

    Args:
        canReach: Pre-computed reachability table
        state: Current state
        t: Number of steps
        class_id: Target class ID

    Returns:
        True if feasible
    """
    if t < 0 or t >= canReach.shape[1]:
        return False
    if state < 0 or state >= canReach.shape[0]:
        return False
    if class_id < 0 or class_id >= canReach.shape[2]:
        return False

    return canReach[state, t, class_id]
