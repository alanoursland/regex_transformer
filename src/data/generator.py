"""Main data generation orchestration."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Literal
from collections import Counter, defaultdict
import numpy as np

from ..fsm.dfa import FSM
from .feasibility import can_reach_tables, feasible
from .quotas import QuotaManager
from .telemetry import build_report, DataReport


@dataclass
class GenConfig:
    """
    Configuration for data generation.

    Attributes:
        L_min: Minimum sequence length
        L_max: Maximum sequence length
        p_class: Dict mapping class_name -> probability (should sum to 1.0)
        reject_mix: Dict mapping reject subtype -> probability (for reject samples)
        attempts_per_sample: Max attempts before resampling target
        beam_size: Beam size for forward generator
    """

    L_min: int = 1
    L_max: int = 10
    p_class: Dict[str, float] = None
    reject_mix: Dict[str, float] = None
    attempts_per_sample: int = 100
    beam_size: int = 16

    def __post_init__(self):
        """Set defaults and validate."""
        if self.p_class is None:
            # Default: equal probability for all classes
            self.p_class = {"accept": 0.4, "incomplete": 0.3, "reject": 0.3}

        if self.reject_mix is None:
            # Default reject subtypes (only used if reject class is sampled)
            self.reject_mix = {
                "overrun": 0.4,
                "premature": 0.3,
                "wrong_alt": 0.2,
                "illegal_step": 0.1,
            }

        # Validate probabilities
        if abs(sum(self.p_class.values()) - 1.0) > 1e-6:
            raise ValueError(f"p_class must sum to 1.0, got {sum(self.p_class.values())}")

        if abs(sum(self.reject_mix.values()) - 1.0) > 1e-6:
            raise ValueError(
                f"reject_mix must sum to 1.0, got {sum(self.reject_mix.values())}"
            )


def sample_target(
    cfg: GenConfig, fsm: FSM, canReach: np.ndarray, rng: np.random.Generator
) -> Tuple[int, int]:
    """
    Sample a target (class_id, length) pair.

    Args:
        cfg: Generation config
        fsm: FSM
        canReach: Reachability table
        rng: Random generator

    Returns:
        (class_id, L) tuple
    """
    # Sample class
    class_names = list(cfg.p_class.keys())
    class_probs = [cfg.p_class[name] for name in class_names]
    class_name = rng.choice(class_names, p=class_probs)
    class_id = fsm.classes.index(class_name)

    # Sample length
    L = rng.integers(cfg.L_min, cfg.L_max + 1)

    return class_id, L


def _build_predecessors(fsm: FSM) -> Dict[int, List[Tuple[int, int]]]:
    """
    Build predecessor index: pred[q] = [(p, a)] where delta(p, a) = q.

    Args:
        fsm: FSM

    Returns:
        Dict mapping state -> list of (predecessor_state, token_id)
    """
    predecessors = defaultdict(list)

    for (state, token_id), next_state in fsm.delta.items():
        predecessors[next_state].append((state, token_id))

    return dict(predecessors)


def backward(
    fsm: FSM,
    class_id: int,
    L: int,
    quota: QuotaManager,
    canReach: np.ndarray,
    predecessors: Dict[int, List[Tuple[int, int]]],
    rng: np.random.Generator,
) -> Optional[List[int]]:
    """
    Generate a sequence via backward sampling.

    Args:
        fsm: FSM
        class_id: Target class ID
        L: Target length
        quota: Quota manager
        canReach: Reachability table
        predecessors: Predecessor index
        rng: Random generator

    Returns:
        List of token IDs, or None if infeasible
    """
    # Find terminal states of the target class
    terminal_states = [
        s for s in range(fsm.states) if fsm.state_class[s] == class_id
    ]

    if not terminal_states:
        return None

    # Filter to states reachable in exactly L steps from start
    reachable_terminals = [
        s for s in terminal_states if feasible(canReach, fsm.start, L, class_id)
    ]

    if not reachable_terminals:
        return None

    # Weight by state coverage (prefer under-visited states)
    weights = np.array([quota.state_weight(s) for s in reachable_terminals])
    weights = weights / weights.sum()

    # Choose terminal state
    current_state = rng.choice(reachable_terminals, p=weights)

    # Walk backward L steps
    path_states = [current_state]
    path_tokens = []

    for t in range(L, 0, -1):
        # Find feasible predecessors
        preds = predecessors.get(current_state, [])

        # Filter by feasibility: (prev_state, token) is feasible if
        # prev_state can reach class_id in t-1 steps
        feasible_preds = [
            (prev_state, token_id)
            for prev_state, token_id in preds
            if feasible(canReach, prev_state, t - 1, class_id)
        ]

        if not feasible_preds:
            return None  # Dead end

        # Weight by edge coverage
        weights = np.array(
            [quota.edge_weight(prev_state, token_id) for prev_state, token_id in feasible_preds]
        )
        weights = weights / weights.sum()

        # Choose predecessor
        idx = rng.choice(len(feasible_preds), p=weights)
        prev_state, token_id = feasible_preds[idx]

        path_states.insert(0, prev_state)
        path_tokens.insert(0, token_id)
        current_state = prev_state

    # Verify we ended at start state
    if path_states[0] != fsm.start:
        return None

    return path_tokens


def forward(
    fsm: FSM,
    class_id: int,
    L: int,
    quota: QuotaManager,
    canReach: np.ndarray,
    rng: np.random.Generator,
    beam: int = 16,
) -> Optional[List[int]]:
    """
    Generate a sequence via forward beam search.

    Args:
        fsm: FSM
        class_id: Target class ID
        L: Target length
        quota: Quota manager
        canReach: Reachability table
        rng: Random generator
        beam: Beam size

    Returns:
        List of token IDs, or None if not found
    """
    # Start with initial state
    candidates = [(fsm.start, [], 0.0)]  # (state, tokens, score)

    for step in range(L):
        new_candidates = []

        for state, tokens, score in candidates:
            remaining = L - len(tokens)

            # Try each token
            for token_id in range(len(fsm.alphabet)):
                next_state = fsm.delta[(state, token_id)]

                # Check feasibility: can we reach target class in remaining-1 steps?
                if not feasible(canReach, next_state, remaining - 1, class_id):
                    continue

                new_tokens = tokens + [token_id]
                # Score: negative coverage bonus (lower is better)
                new_score = score - quota.edge_weight(state, token_id)

                new_candidates.append((next_state, new_tokens, new_score))

        if not new_candidates:
            return None  # No feasible path

        # Keep top beam candidates
        new_candidates.sort(key=lambda x: x[2])  # Sort by score
        candidates = new_candidates[:beam]

    # Find any candidate that reached target class
    for state, tokens, score in candidates:
        if fsm.state_class[state] == class_id and len(tokens) == L:
            return tokens

    return None


def generate_sample(
    fsm: FSM,
    cfg: GenConfig,
    quota: QuotaManager,
    canReach: np.ndarray,
    predecessors: Dict[int, List[Tuple[int, int]]],
    rng: np.random.Generator,
    reject_subtype_counter: Counter,
) -> Optional[Tuple[List[int], str]]:
    """
    Generate a single sample.

    Args:
        fsm: FSM
        cfg: Generation config
        quota: Quota manager
        canReach: Reachability table
        predecessors: Predecessor index
        rng: Random generator
        reject_subtype_counter: Counter for reject subtypes

    Returns:
        (tokens, class_name) or None if failed
    """
    for attempt in range(cfg.attempts_per_sample):
        # Sample target
        class_id, L = sample_target(cfg, fsm, canReach, rng)
        class_name = fsm.classes[class_id]

        # Try backward first
        tokens = backward(fsm, class_id, L, quota, canReach, predecessors, rng)

        # Fall back to forward if backward fails
        if tokens is None:
            tokens = forward(fsm, class_id, L, quota, canReach, rng, cfg.beam_size)

        if tokens is not None:
            # Update quota
            states = fsm.trace(tokens)
            quota.update_path(states, tokens)

            # Track reject subtype if applicable
            if class_name == "reject":
                # For now, just mark as "overrun" (TODO: implement proper subtypes)
                reject_subtype_counter["overrun"] += 1

            return tokens, class_name

    return None  # Failed after all attempts


def generate_corpus(
    fsm: FSM, cfg: GenConfig, n_samples: int, seed: int = 42
) -> Tuple[List[List[int]], List[str], DataReport]:
    """
    Generate a corpus of samples from an FSM.

    Args:
        fsm: The finite state machine
        cfg: Generation configuration
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        (samples, class_names, report) where:
            - samples: List of token sequences
            - class_names: List of class names (parallel to samples)
            - report: DataReport with telemetry
    """
    rng = np.random.default_rng(seed)

    # Build reachability table
    canReach = can_reach_tables(fsm, cfg.L_max)

    # Build predecessor index
    predecessors = _build_predecessors(fsm)

    # Initialize quota manager
    quota = QuotaManager(fsm.states, len(fsm.alphabet))

    # Track reject subtypes
    reject_subtype_counter = Counter()

    # Generate samples
    samples = []
    class_names = []
    failed_attempts = 0

    for i in range(n_samples):
        result = generate_sample(
            fsm, cfg, quota, canReach, predecessors, rng, reject_subtype_counter
        )

        if result is not None:
            tokens, class_name = result
            samples.append(tokens)
            class_names.append(class_name)
        else:
            failed_attempts += 1
            # Generate a fallback sample (just use backward with retries)
            # For now, try again with different target
            for retry in range(10):
                class_id, L = sample_target(cfg, fsm, canReach, rng)
                tokens = backward(fsm, class_id, L, quota, canReach, predecessors, rng)
                if tokens is not None:
                    class_name = fsm.classes[class_id]
                    samples.append(tokens)
                    class_names.append(class_name)
                    states = fsm.trace(tokens)
                    quota.update_path(states, tokens)
                    break

    # Build report
    report = build_report(
        samples, class_names, quota.summary(), reject_subtype_counter, failed_attempts
    )

    return samples, class_names, report
