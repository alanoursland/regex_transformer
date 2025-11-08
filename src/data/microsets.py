"""Micro datasets for overfit sanity checks."""

from typing import List, Dict, Tuple
import numpy as np

from ..fsm.dfa import FSM
from ..fsm.regex_def import RegexDefinition
from ..fsm.compile import compile_regex
from .generator import GenConfig, generate_corpus


def get_micro_dataset(
    regex_def: RegexDefinition,
    n_train: int = 16,
    n_val: int = 8,
    seed: int = 12345,
) -> Tuple[FSM, Dict[str, List[List[int]]], Dict[str, List[str]]]:
    """
    Generate a fixed micro dataset for a given regex.

    Args:
        regex_def: Regex definition
        n_train: Number of training samples
        n_val: Number of validation samples
        seed: Fixed seed for reproducibility

    Returns:
        (fsm, samples_by_split, class_names_by_split) where:
            - fsm: Compiled FSM
            - samples_by_split: Dict mapping split -> list of token sequences
            - class_names_by_split: Dict mapping split -> list of class names
    """
    # Compile FSM
    fsm = compile_regex(regex_def)

    # Build p_class from FSM classes
    p_class = {cls: 1.0 / len(fsm.classes) for cls in fsm.classes}

    # Generate corpus
    cfg = GenConfig(L_min=1, L_max=8, p_class=p_class)

    total_samples = n_train + n_val
    samples, class_names, _ = generate_corpus(fsm, cfg, total_samples, seed=seed)

    # Split deterministically
    samples_by_split = {
        "train": samples[:n_train],
        "val": samples[n_train:],
    }

    class_names_by_split = {
        "train": class_names[:n_train],
        "val": class_names[n_train:],
    }

    return fsm, samples_by_split, class_names_by_split


# Pre-defined micro datasets for common patterns
MICRO_A_PLUS = RegexDefinition(
    alphabet=("a",),
    patterns=(("a+", "accept"),),
)

MICRO_A_STAR_B_STAR = RegexDefinition(
    alphabet=("a", "b"),
    patterns=(("a*b*", "accept"),),
)

MICRO_BRANCHING = RegexDefinition(
    alphabet=("a", "b", "c"),
    patterns=(("(a|b)*c", "accept"),),
)


def get_micro_a_plus(n_train: int = 16, n_val: int = 8) -> Tuple:
    """Get micro dataset for pattern a+."""
    return get_micro_dataset(MICRO_A_PLUS, n_train, n_val, seed=12345)


def get_micro_a_star_b_star(n_train: int = 16, n_val: int = 8) -> Tuple:
    """Get micro dataset for pattern a*b*."""
    return get_micro_dataset(MICRO_A_STAR_B_STAR, n_train, n_val, seed=12346)


def get_micro_branching(n_train: int = 16, n_val: int = 8) -> Tuple:
    """Get micro dataset for pattern (a|b)*c."""
    return get_micro_dataset(MICRO_BRANCHING, n_train, n_val, seed=12347)
