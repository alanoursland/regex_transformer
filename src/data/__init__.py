"""Data generation package for FSM-based datasets."""

from .generator import generate_corpus, GenConfig
from .dataset import FsmDataset
from .split import split_of

__all__ = ['generate_corpus', 'GenConfig', 'FsmDataset', 'split_of']
