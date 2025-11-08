"""Regex definition dataclass with validation."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import json


@dataclass(frozen=True)
class RegexDefinition:
    """
    Defines a regex matching problem with explicit alphabet and labeled patterns.

    Attributes:
        alphabet: Tuple of valid input characters (e.g., ('a', 'b', 'c'))
        patterns: Tuple of (regex_pattern, class_name) pairs
    """
    alphabet: Tuple[str, ...]
    patterns: Tuple[Tuple[str, str], ...]

    def __post_init__(self):
        """Validate the regex definition."""
        # Check alphabet contains only single characters
        for char in self.alphabet:
            if not isinstance(char, str) or len(char) != 1:
                raise ValueError(f"Alphabet must contain single characters, got: {char!r}")

        # Check for duplicate alphabet symbols
        if len(self.alphabet) != len(set(self.alphabet)):
            raise ValueError("Alphabet contains duplicate symbols")

        # Validate each pattern
        seen_classes = set()
        for pattern, class_name in self.patterns:
            # Check regex syntax validity
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern {pattern!r}: {e}")

            # Check class name uniqueness
            if class_name in seen_classes:
                raise ValueError(f"Duplicate class name: {class_name}")
            seen_classes.add(class_name)

            # Verify pattern only uses alphabet symbols (basic check)
            # This is a simplified check - actual validation happens during compilation
            # We allow regex metacharacters here

        if not self.patterns:
            raise ValueError("Must have at least one pattern")


def load_regex_def(path: Path) -> RegexDefinition:
    """
    Load a RegexDefinition from a JSON file.

    Expected format:
    {
        "alphabet": ["a", "b", "c"],
        "patterns": [
            ["a+", "accept"],
            ["b*", "loop"]
        ]
    }
    """
    with open(path, 'r') as f:
        data = json.load(f)

    return RegexDefinition(
        alphabet=tuple(data['alphabet']),
        patterns=tuple(tuple(p) for p in data['patterns'])
    )


def save_regex_def(regex_def: RegexDefinition, path: Path) -> None:
    """Save a RegexDefinition to a JSON file."""
    data = {
        'alphabet': list(regex_def.alphabet),
        'patterns': [list(p) for p in regex_def.patterns]
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
