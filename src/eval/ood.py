"""Out-of-distribution evaluation utilities."""
from typing import List, Dict
import torch
from torch.utils.data import Subset

def length_ood_split(dataset, max_train_len: int):
    """
    Split dataset into in-distribution and OOD by length.
    
    Args:
        dataset: FsmDataset
        max_train_len: Maximum length seen during training
    
    Returns:
        (id_indices, ood_indices)
    """
    id_indices = []
    ood_indices = []
    
    for i, sample in enumerate(dataset.samples):
        if len(sample) <= max_train_len:
            id_indices.append(i)
        else:
            ood_indices.append(i)
    
    return id_indices, ood_indices


def compute_ood_gap(id_metrics: Dict, ood_metrics: Dict) -> Dict:
    """
    Compute generalization gap between ID and OOD.
    
    Returns:
        Dict with gap metrics
    """
    return {
        "token_acc_gap": id_metrics.get("token_acc", 0) - ood_metrics.get("token_acc", 0),
        "class_acc_gap": id_metrics.get("class_acc", 0) - ood_metrics.get("class_acc", 0),
        "nll_gap": ood_metrics.get("nll", float('inf')) - id_metrics.get("nll", float('inf')),
    }
