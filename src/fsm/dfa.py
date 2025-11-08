"""FSM (DFA) representation and core operations."""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Sequence


@dataclass
class FSM:
    """
    Deterministic Finite State Machine.

    Formally: FSM = (Q, Σ, δ, q₀, C)

    Attributes:
        states: Number of states (states are 0-indexed integers)
        alphabet: Tuple of single-character strings
        start: Start state (integer index)
        delta: Transition function: (state_id, token_id) -> next_state_id
        classes: Tuple of class names (ordered)
        state_class: List mapping state_id -> class_id
        reject: Explicit reject state ID
    """
    states: int
    alphabet: Tuple[str, ...]
    start: int
    delta: Dict[Tuple[int, int], int]
    classes: Tuple[str, ...]
    state_class: List[int]
    reject: int

    def __post_init__(self):
        """Validate FSM structure."""
        # Check start state is valid
        if not (0 <= self.start < self.states):
            raise ValueError(f"Start state {self.start} out of range [0, {self.states})")

        # Check reject state is valid
        if not (0 <= self.reject < self.states):
            raise ValueError(f"Reject state {self.reject} out of range [0, {self.states})")

        # Check state_class has correct length
        if len(self.state_class) != self.states:
            raise ValueError(
                f"state_class length {len(self.state_class)} != states {self.states}"
            )

        # Check all class IDs are valid
        for state_id, class_id in enumerate(self.state_class):
            if not (0 <= class_id < len(self.classes)):
                raise ValueError(
                    f"State {state_id} has invalid class_id {class_id}, "
                    f"must be in [0, {len(self.classes)})"
                )

        # Check transitions are total (all (state, token) pairs defined)
        num_alphabet = len(self.alphabet)
        expected_transitions = self.states * num_alphabet

        if len(self.delta) != expected_transitions:
            raise ValueError(
                f"Expected {expected_transitions} transitions, got {len(self.delta)}"
            )

        # Check all transition states are valid
        for (state, token_id), next_state in self.delta.items():
            if not (0 <= state < self.states):
                raise ValueError(f"Invalid source state in delta: {state}")
            if not (0 <= token_id < num_alphabet):
                raise ValueError(f"Invalid token_id in delta: {token_id}")
            if not (0 <= next_state < self.states):
                raise ValueError(f"Invalid next_state in delta: {next_state}")

        # Check reject state has self-loops
        for token_id in range(num_alphabet):
            if self.delta.get((self.reject, token_id)) != self.reject:
                raise ValueError(
                    f"Reject state {self.reject} must self-loop on all inputs"
                )

    def step(self, state: int, token_id: int) -> int:
        """
        Single-step state transition.

        Args:
            state: Current state ID
            token_id: Input token ID (index into alphabet)

        Returns:
            Next state ID
        """
        return self.delta[(state, token_id)]

    def classify(self, state: int) -> int:
        """
        Get the class ID of a state.

        Args:
            state: State ID

        Returns:
            Class ID (index into self.classes)
        """
        return self.state_class[state]

    def classify_name(self, state: int) -> str:
        """
        Get the class name of a state.

        Args:
            state: State ID

        Returns:
            Class name string
        """
        return self.classes[self.state_class[state]]

    def trace(self, tokens: Sequence[int]) -> List[int]:
        """
        Trace the sequence of states visited while processing tokens.

        Args:
            tokens: Sequence of token IDs

        Returns:
            List of state IDs, length = len(tokens) + 1
            states[0] is the start state (before consuming any input)
            states[i] is the state after consuming tokens[i-1]
        """
        states = [self.start]
        current = self.start

        for token_id in tokens:
            current = self.step(current, token_id)
            states.append(current)

        return states

    def classify_string(self, tokens: Sequence[int]) -> str:
        """
        Classify an entire input sequence.

        Args:
            tokens: Sequence of token IDs

        Returns:
            Class name of the final state
        """
        states = self.trace(tokens)
        final_state = states[-1]
        return self.classify_name(final_state)

    def char_to_token_id(self, char: str) -> int:
        """Convert a character to its token ID."""
        try:
            return self.alphabet.index(char)
        except ValueError:
            raise ValueError(f"Character {char!r} not in alphabet {self.alphabet}")

    def tokens_from_string(self, s: str) -> List[int]:
        """Convert a string to a list of token IDs."""
        return [self.char_to_token_id(c) for c in s]

    def string_from_tokens(self, tokens: Sequence[int]) -> str:
        """Convert a list of token IDs to a string."""
        return ''.join(self.alphabet[tid] for tid in tokens)
