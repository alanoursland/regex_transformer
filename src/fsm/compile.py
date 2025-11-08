"""Compile regex patterns into minimized DFAs."""

import re
from typing import Set, Dict, Tuple, FrozenSet, List, Optional
from collections import defaultdict, deque

from .regex_def import RegexDefinition
from .dfa import FSM


def compile_regex(regex_def: RegexDefinition) -> FSM:
    """
    Compile a RegexDefinition into a minimized DFA.

    Process: Build DFA directly using regex matching

    Args:
        regex_def: Regex definition with alphabet and patterns

    Returns:
        FSM
    """
    alphabet = regex_def.alphabet
    patterns = regex_def.patterns

    # Build DFA directly by exploring reachable states
    dfa_states, dfa_transitions, dfa_start, dfa_state_classes = _build_dfa_direct(
        regex_def
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


def _build_dfa_direct(
    regex_def: RegexDefinition,
) -> Tuple[List[int], Dict[Tuple[int, int], int], int, Dict[int, str]]:
    """
    Build DFA directly by BFS over string prefixes.

    Uses Python's re module to check pattern matching.
    Limits exploration depth to keep state space manageable.
    """
    alphabet = regex_def.alphabet
    patterns = regex_def.patterns

    # Compile patterns
    # Note: Users should use explicit parentheses for alternation (e.g., '(a|b)' not 'a|b')
    # to avoid precedence issues with anchors
    compiled_patterns = [
        (re.compile(f"^{pattern}$"), re.compile(f"^{pattern}"), class_name)
        for pattern, class_name in patterns
    ]

    # State = string reached so far
    # We'll use strings as state IDs initially
    state_to_id: Dict[str, int] = {"": 0}
    id_to_state: Dict[int, str] = {0: ""}
    next_id = 1

    transitions: Dict[Tuple[int, int], int] = {}
    state_classes: Dict[int, str] = {}

    # Classify start state
    state_classes[0] = _classify_string("", compiled_patterns, alphabet)

    # BFS
    MAX_LENGTH = 10  # Limit string length for Phase 1
    queue = deque([""])
    visited = {""}

    while queue:
        current_str = queue.popleft()
        current_id = state_to_id[current_str]

        # Try each character
        for char_idx, char in enumerate(alphabet):
            next_str = current_str + char

            # Only explore up to MAX_LENGTH
            if len(next_str) > MAX_LENGTH:
                # Map to a special "too_long" state (we'll map to reject later)
                # For now, just skip
                continue

            # Add state if new
            if next_str not in state_to_id:
                state_to_id[next_str] = next_id
                id_to_state[next_id] = next_str
                state_classes[next_id] = _classify_string(next_str, compiled_patterns, alphabet)
                next_id += 1

            next_id_val = state_to_id[next_str]
            transitions[(current_id, char_idx)] = next_id_val

            # Add to queue if not visited
            if next_str not in visited and len(next_str) <= MAX_LENGTH:
                visited.add(next_str)
                queue.append(next_str)

    dfa_states = list(range(len(state_to_id)))
    return dfa_states, transitions, 0, state_classes


def _classify_string(s: str, compiled_patterns: List[Tuple], alphabet: Tuple[str, ...]) -> str:
    """
    Classify a string based on patterns.

    Returns class name if fully matched, "incomplete" if prefix of a match,
    or "reject" if neither.
    """
    # Check if fully matches any pattern
    for full_pattern, prefix_pattern, class_name in compiled_patterns:
        if full_pattern.match(s):
            return class_name

    # Check if we can extend this string to eventually match
    if _can_extend_to_match(s, compiled_patterns, alphabet):
        return "incomplete"

    return "reject"


def _can_extend_to_match(s: str, compiled_patterns: List[Tuple], alphabet: Tuple[str, ...], max_depth: int = 10) -> bool:
    """
    Check if a string can be extended to match any pattern.

    Uses BFS-style exploration up to max_depth.
    """
    if len(s) >= max_depth:
        return False

    for char in alphabet:
        extended = s + char

        # Check if this extension matches
        for full_pattern, _, _ in compiled_patterns:
            if full_pattern.match(extended):
                return True

        # Recursively check if we can extend further
        if _can_extend_to_match(extended, compiled_patterns, alphabet, max_depth):
            return True

    return False


def _minimize_dfa(
    alphabet: Tuple[str, ...],
    dfa_states: List[int],
    dfa_transitions: Dict[Tuple[int, int], int],
    dfa_start: int,
    dfa_state_classes: Dict[int, str],
) -> Tuple[List[int], Dict[Tuple[int, int], int], int, Dict[int, str]]:
    """
    Minimize DFA using partition refinement.
    """
    # Initial partition: separate by class
    class_to_states: Dict[str, Set[int]] = defaultdict(set)
    for state, class_name in dfa_state_classes.items():
        class_to_states[class_name].add(state)

    partitions = list(class_to_states.values())

    # Refine partitions until stable
    changed = True
    max_iterations = 100
    iteration = 0

    while changed and iteration < max_iterations:
        iteration += 1
        changed = False
        new_partitions = []

        for partition in partitions:
            if len(partition) == 1:
                new_partitions.append(partition)
                continue

            # Group states by transition signature
            signatures: Dict[Tuple, Set[int]] = defaultdict(set)

            for state in partition:
                sig = []
                for char_idx in range(len(alphabet)):
                    next_state = dfa_transitions.get((state, char_idx))

                    # Find which partition contains next_state
                    next_partition_idx = None
                    if next_state is not None:
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
    state_to_partition: Dict[int, int] = {}
    partition_representatives = []

    for p_idx, partition in enumerate(partitions):
        representative = min(partition)
        partition_representatives.append(representative)
        for state in partition:
            state_to_partition[state] = p_idx

    # Build minimized transitions
    min_transitions: Dict[Tuple[int, int], int] = {}
    for (state, char_idx), next_state in dfa_transitions.items():
        if next_state is not None:
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
    Build final FSM with explicit reject state and total transition function.
    """
    # Collect all class names
    class_names = set(state_classes.values())
    class_names.add("reject")
    classes_list = sorted(class_names)
    class_to_id = {name: idx for idx, name in enumerate(classes_list)}

    # Find or create reject state
    reject_id = None
    for state, class_name in state_classes.items():
        if class_name == "reject":
            reject_id = state
            break

    if reject_id is None:
        # Add explicit reject state
        reject_id = len(states)
        states.append(reject_id)
        state_classes[reject_id] = "reject"

    # Ensure reject state has self-loops
    for char_idx in range(len(alphabet)):
        transitions[(reject_id, char_idx)] = reject_id

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
