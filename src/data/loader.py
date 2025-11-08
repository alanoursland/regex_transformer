"""DataLoader construction with proper seeding."""

from typing import Dict, Tuple, Callable
import torch
from torch.utils.data import DataLoader
import numpy as np

from .dataset import FsmDataset
from .collate import collate_batch
from .tokenizer import Vocab


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """
    Initialize worker with deterministic seed.

    Args:
        worker_id: Worker ID from PyTorch
        base_seed: Base seed for reproducibility
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_dataloaders(
    datasets: Dict[str, FsmDataset],
    vocab: Vocab,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for train/val/test splits.

    Args:
        datasets: Dict mapping split name -> FsmDataset
        vocab: Vocabulary for collation
        batch_size: Batch size
        seed: Random seed for reproducibility
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        prefetch_factor: Number of batches to prefetch per worker

    Returns:
        Dict mapping split name -> DataLoader
    """
    # Create deterministic generator for shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Create collate function with vocab
    def collate_fn(examples):
        return collate_batch(examples, vocab.pad_id, vocab.eos_id)

    # Create worker init function with seed
    def worker_init(worker_id):
        return worker_init_fn(worker_id, base_seed=seed)

    dataloaders = {}

    for split, dataset in datasets.items():
        shuffle = split == "train"

        # Configure DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            generator=generator if shuffle else None,
            worker_init_fn=worker_init if num_workers > 0 else None,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

        dataloaders[split] = dataloader

    return dataloaders
