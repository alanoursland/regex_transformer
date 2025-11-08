"""FSM core: regex compilation, DFA operations, and serialization."""

from .regex_def import RegexDefinition
from .dfa import FSM
from .compile import compile_regex
from .serialize import save_fsm, load_fsm

__all__ = [
    "RegexDefinition",
    "FSM",
    "compile_regex",
    "save_fsm",
    "load_fsm",
]
