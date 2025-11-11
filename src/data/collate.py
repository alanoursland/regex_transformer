"""Batching, padding, and masking utilities."""

from typing import List, Dict
import torch


def collate_batch(
    examples: List[Dict[str, torch.Tensor]],
    pad_id: int,
    eos_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of examples with proper padding and masking.

    Args:
        examples: List of dicts from FsmDataset.__getitem__
        pad_id: Padding token ID
        eos_id: End-of-sequence token ID

    Returns:
        Dict with batched tensors:
            - tokens: (B, T) input tokens
            - next_tokens: (B, T) target tokens (shifted)
            - states: (B, T+1) FSM states
            - state_classes: (B, T+1) state class IDs
            - attn_mask: (B, T) attention mask (True for valid, False for PAD)
            - loss_mask: (B, T) loss mask (False for PAD and last position)
    """
    batch_size = len(examples)
    max_len = max(len(ex["tokens"]) for ex in examples)

    # Initialize tensors
    tokens_batch = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    next_tokens_batch = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    states_batch = torch.full(
        (batch_size, max_len + 1), 0, dtype=torch.long
    )  # States is T+1
    state_classes_batch = torch.full(
        (batch_size, max_len + 1), 0, dtype=torch.long
    )  # State classes is T+1
    attn_mask_batch = torch.zeros((batch_size, max_len), dtype=torch.bool)
    loss_mask_batch = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, ex in enumerate(examples):
        seq_len = len(ex["tokens"])

        # Copy tokens
        tokens_batch[i, :seq_len] = ex["tokens"]
        next_tokens_batch[i, :seq_len] = ex["next_tokens"]

        # Copy states (length is seq_len + 1)
        states_batch[i, : seq_len + 1] = ex["states"]
        state_classes_batch[i, : seq_len + 1] = ex["state_classes"]

        # Attention mask: True for valid positions
        attn_mask_batch[i, :seq_len] = True

        # Loss mask: True only where a next-token target exists (exclude last position)
        if seq_len > 1:
            loss_mask_batch[i, :seq_len - 1] = True
        # if seq_len is 0 or 1, leave loss_mask all False

    return {
        "tokens": tokens_batch,
        "next_tokens": next_tokens_batch,
        "states": states_batch,
        "state_classes": state_classes_batch,
        "attn_mask": attn_mask_batch,
        "loss_mask": loss_mask_batch,
    }
