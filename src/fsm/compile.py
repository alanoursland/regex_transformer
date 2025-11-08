"""Compile regex patterns into minimized DFAs."""

import re
from typing import Set, Dict, Tuple, FrozenSet, List, Optional
from collections import defaultdict, deque

from .regex_def import RegexDefinition
from .dfa import FSM


def compile_regex(regex_def: RegexDefinition) -> FSM:
    """
    Compile a RegexDefinition into a minimized DFA.

    Process: regex → NFA → DFA → minimized DFA

    Args:
        regex_def: Regex definition with alphabet and patterns

    Returns:
        Minimized FSM
    """
    # Build NFA using Python's re module for each pattern
    # Then convert to DFA via subset construction
    # Then minimize using Hopcroft's algorithm

    # For simplicity in Phase 1, we'll build a combined DFA directly
    # by using Python's re module to simulate the NFA

    alphabet = regex_def.alphabet
    patterns = regex_def.patterns

    # Build DFA via subset construction
    nfa_states, nfa_transitions, nfa_start, nfa_accepts = _build_nfa_from_patterns(
        regex_def
    )

    # Convert NFA to DFA via subset construction
    dfa_states, dfa_transitions, dfa_start, dfa_state_classes = _subset_construction(
        alphabet=alphabet,
        nfa_states=nfa_states,
        nfa_transitions=nfa_transitions,
        nfa_start=nfa_start,
        nfa_accepts=nfa_accepts,
        patterns=patterns,
    )

    # Minimize DFA
    min_states, min_transitions, min_start, min_state_classes = _minimize_dfa(
        alphabet=alphabet,
        dfa_states=dfa_states,
        dfa_transitions=dfa_transitions,
        dfa_start=dfa_start,
        dfa_state_classes=dfa_state_classes,
    )

    # Build final FSM with explicit reject state
    return _build_fsm(
        alphabet=alphabet,
        states=min_states,
        transitions=min_transitions,
        start=min_start,
        state_classes=min_state_classes,
        patterns=patterns,
    )


def _build_nfa_from_patterns(
    regex_def: RegexDefinition,
) -> Tuple[Set[int], Dict[Tuple[int, Optional[str]], Set[int]], int, Dict[int, str]]:
    """
    Build an NFA using Python's re module.

    This is a pragmatic approach: we use re.match() to simulate NFA behavior
    rather than implementing Thompson's construction from scratch.

    Returns:
        (states, transitions, start_state, accept_states_map)
        transitions: (state, char_or_None) -> set of next states
        accept_states_map: state -> class_name for accepting states
    """
    # For Phase 1, we'll use a simple simulation-based approach
    # Build a state graph by exploring all possible string prefixes

    alphabet = regex_def.alphabet
    patterns = regex_def.patterns

    # Compile regex patterns
    compiled_patterns = [(re.compile(f"^{pattern}$"), class_name) for pattern, class_name in patterns]

    # States are represented by string prefixes
    # We'll explore breadth-first up to a reasonable depth
    MAX_DEPTH = 20  # Limit exploration depth for Phase 1

    state_to_id: Dict[str, int] = {"": 0}  # Empty string is start state
    id_to_state: Dict[int, str] = {0: ""}
    next_id = 1

    transitions: Dict[Tuple[int, Optional[str]], Set[int]] = defaultdict(set)
    accept_states: Dict[int, str] = {}

    # BFS exploration
    queue = deque([""])
    visited = {""}

    while queue:
        prefix = queue.popleft()
        if len(prefix) > MAX_DEPTH:
            continue

        state_id = state_to_id[prefix]

        # Check if this prefix matches any pattern
        for compiled_re, class_name in compiled_patterns:
            if compiled_re.match(prefix):
                accept_states[state_id] = class_name
                break

        # Explore transitions
        for char in alphabet:
            next_prefix = prefix + char

            if next_prefix not in state_to_id:
                state_to_id[next_prefix] = next_id
                id_to_state[next_id] = next_prefix
                next_id += 1

            next_id_val = state_to_id[next_prefix]
            transitions[(state_id, char)].add(next_id_val)

            if next_prefix not in visited:
                visited.add(next_prefix)
                queue.append(next_prefix)

    states = set(state_to_id.values())
    start = 0

    return states, transitions, start, accept_states


def _subset_construction(
    alphabet: Tuple[str, ...],
    nfa_states: Set[int],
    nfa_transitions: Dict[Tuple[int, Optional[str]], Set[int]],
    nfa_start: int,
    nfa_accepts: Dict[int, str],
    patterns: Tuple[Tuple[str, str], ...],
) -> Tuple[List[int], Dict[Tuple[int, int], int], int, Dict[int, str]]:
    """
    Convert NFA to DFA via subset construction.

    Returns:
        (dfa_states, dfa_transitions, dfa_start, dfa_state_classes)
    """
    # DFA states are subsets of NFA states
    # We represent them as frozen sets

    start_set = frozenset([nfa_start])
    dfa_start_id = 0

    subset_to_id: Dict[FrozenSet[int], int] = {start_set: 0}
    id_to_subset: Dict[int, FrozenSet[int]] = {0: start_set}
    next_id = 1

    dfa_transitions: Dict[Tuple[int, int], int] = {}
    dfa_state_classes: Dict[int, str] = {}

    # Determine class for start set
    dfa_state_classes[0] = _classify_subset(start_set, nfa_accepts)

    # BFS to build DFA
    queue = deque([start_set])
    visited = {start_set}

    while queue:
        current_subset = queue.popleft()
        current_id = subset_to_id[current_subset]

        # For each character in alphabet
        for char_idx, char in enumerate(alphabet):
            # Compute next subset
            next_subset = set()
            for nfa_state in current_subset:
                next_states = nfa_transitions.get((nfa_state, char), set())
                next_subset.update(next_states)

            next_subset_frozen = frozenset(next_subset)

            # Add new DFA state if needed
            if next_subset_frozen not in subset_to_id:
                subset_to_id[next_subset_frozen] = next_id
                id_to_subset[next_id] = next_subset_frozen
                dfa_state_classes[next_id] = _classify_subset(next_subset_frozen, nfa_accepts)
                next_id += 1

                if next_subset_frozen not in visited:
                    visited.add(next_subset_frozen)
                    queue.append(next_subset_frozen)

            next_id_val = subset_to_id[next_subset_frozen]
            dfa_transitions[(current_id, char_idx)] = next_id_val

    dfa_states = list(range(len(subset_to_id)))
    return dfa_states, dfa_transitions, dfa_start_id, dfa_state_classes


def _classify_subset(subset: FrozenSet[int], nfa_accepts: Dict[int, str]) -> str:
    """
    Classify a DFA state (subset of NFA states).

    Returns the class name if any state in the subset is accepting,
    otherwise returns "incomplete" (or "reject" if we know it's a dead end).
    """
    for nfa_state in subset:
        if nfa_state in nfa_accepts:
            return nfa_accepts[nfa_state]

    # Not accepting - could be incomplete or reject
    # For now, return "incomplete" - we'll add explicit reject later
    return "incomplete"


def _minimize_dfa(
    alphabet: Tuple[str, ...],
    dfa_states: List[int],
    dfa_transitions: Dict[Tuple[int, int], int],
    dfa_start: int,
    dfa_state_classes: Dict[int, str],
) -> Tuple[List[int], Dict[Tuple[int, int], int], int, Dict[int, str]]:
    """
    Minimize DFA using Hopcroft's algorithm.

    For Phase 1, we'll use a simpler but correct approach:
    partition refinement based on equivalence classes.
    """
    # Initial partition: separate by class
    class_to_states: Dict[str, Set[int]] = defaultdict(set)
    for state, class_name in dfa_state_classes.items():
        class_to_states[class_name].add(state)

    partitions = list(class_to_states.values())

    # Refine partitions until stable
    changed = True
    while changed:
        changed = False
        new_partitions = []

        for partition in partitions:
            # Try to split this partition
            if len(partition) == 1:
                new_partitions.append(partition)
                continue

            # Group states by their transition signatures
            signatures: Dict[Tuple, Set[int]] = defaultdict(set)

            for state in partition:
                sig = []
                for char_idx in range(len(alphabet)):
                    next_state = dfa_transitions.get((state, char_idx))
                    # Find which partition the next state belongs to
                    next_partition_idx = None
                    for p_idx, p in enumerate(partitions):
                        if next_state in p:
                            next_partition_idx = p_idx
                            break
                    sig.append(next_partition_idx)

                signatures[tuple(sig)].add(state)

            if len(signatures) > 1:
                changed = True
                new_partitions.extend(signatures.values())
            else:
                new_partitions.append(partition)

        partitions = new_partitions

    # Build minimized DFA
    # Map old states to new representative states
    state_to_partition: Dict[int, int] = {}
    partition_representatives = []

    for p_idx, partition in enumerate(partitions):
        representative = min(partition)  # Choose canonical representative
        partition_representatives.append(representative)
        for state in partition:
            state_to_partition[state] = p_idx

    # Build minimized transitions
    min_transitions: Dict[Tuple[int, int], int] = {}
    for (state, char_idx), next_state in dfa_transitions.items():
        min_state = state_to_partition[state]
        min_next_state = state_to_partition[next_state]
        min_transitions[(min_state, char_idx)] = min_next_state

    # Build minimized state classes
    min_state_classes = {}
    for p_idx, rep in enumerate(partition_representatives):
        min_state_classes[p_idx] = dfa_state_classes[rep]

    min_start = state_to_partition[dfa_start]
    min_states = list(range(len(partitions)))

    return min_states, min_transitions, min_start, min_state_classes


def _build_fsm(
    alphabet: Tuple[str, ...],
    states: List[int],
    transitions: Dict[Tuple[int, int], int],
    start: int,
    state_classes: Dict[int, str],
    patterns: Tuple[Tuple[str, str], ...],
) -> FSM:
    """
    Build final FSM with explicit reject state.

    Ensures the transition function is total (all transitions defined).
    """
    # Collect all class names, add "reject" if not present
    class_names = set(state_classes.values())
    class_names.add("reject")
    classes_list = sorted(class_names)  # Canonical ordering
    class_to_id = {name: idx for idx, name in enumerate(classes_list)}

    # Check if we need to add an explicit reject state
    has_reject = "reject" in state_classes.values()

    if not has_reject:
        # Add explicit reject state
        reject_id = len(states)
        states.append(reject_id)
        state_classes[reject_id] = "reject"

        # Add self-loops for reject state
        for char_idx in range(len(alphabet)):
            transitions[(reject_id, char_idx)] = reject_id
    else:
        # Find existing reject state
        reject_id = None
        for state, class_name in state_classes.items():
            if class_name == "reject":
                reject_id = state
                break

    # Fill in missing transitions (point to reject)
    for state in states:
        for char_idx in range(len(alphabet)):
            if (state, char_idx) not in transitions:
                transitions[(state, char_idx)] = reject_id

    # Build state_class list
    state_class_list = [class_to_id[state_classes[s]] for s in states]

    return FSM(
        states=len(states),
        alphabet=alphabet,
        start=start,
        delta=transitions,
        classes=tuple(classes_list),
        state_class=state_class_list,
        reject=reject_id,
    )
