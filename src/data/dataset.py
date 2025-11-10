"""PyTorch Dataset wrapper for FSM-generated data."""

from typing import List, Dict, Literal, Optional
import torch
from torch.utils.data import Dataset

from fsm.dfa import FSM


class FsmDataset(Dataset):
    """
    PyTorch Dataset for FSM-traced sequences.

    Computes state traces and labels on-the-fly from token sequences.
    """

    def __init__(
        self,
        fsm: FSM,
        samples: List[List[int]],
        split: Literal["train", "val", "test"],
        eos_id: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            fsm: The finite state machine for tracing
            samples: List of token sequences (each is a list of token IDs)
            split: Split name (for filtering if needed)
            eos_id: Optional end-of-sequence token ID (appended if provided)
        """
        self.fsm = fsm
        self.samples = samples
        self.split = split
        self.eos_id = eos_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample as tensors.

        Returns:
            Dict with keys:
                - tokens: [seq_len] input tokens
                - next_tokens: [seq_len] next tokens (shifted by 1)
                - states: [seq_len+1] FSM states
                - state_classes: [seq_len+1] state class IDs
                - mask: [seq_len] mask for valid positions
        """
        tokens_list = self.samples[idx]

        # Trace through FSM (original tokens only, not EOS)
        states = self.fsm.trace(tokens_list)

        # Optionally append EOS to tokens after tracing
        if self.eos_id is not None:
            tokens_list = tokens_list + [self.eos_id]
            # Extend states by repeating final state for EOS
            states = states + [states[-1]]

        # Create tensors
        tokens = torch.tensor(tokens_list, dtype=torch.long)
        states_tensor = torch.tensor(states, dtype=torch.long)

        # State classes
        state_classes = torch.tensor(
            [self.fsm.state_class[s] for s in states], dtype=torch.long
        )

        # Next tokens (shift tokens by 1, pad with -1 at end)
        next_tokens = torch.cat([tokens[1:], torch.tensor([-1], dtype=torch.long)])

        # Mask (all True for now, can be used for padding in batches)
        mask = torch.ones(len(tokens), dtype=torch.bool)

        return {
            "tokens": tokens,
            "next_tokens": next_tokens,
            "states": states_tensor,
            "state_classes": state_classes,
            "mask": mask,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Pads sequences to max length in batch.

    Args:
        batch: List of sample dicts from __getitem__

    Returns:
        Batched dict with padded tensors
    """
    # Find max length in batch
    max_len = max(len(item["tokens"]) for item in batch)

    # Pad each item
    batch_tokens = []
    batch_next_tokens = []
    batch_states = []
    batch_state_classes = []
    batch_masks = []

    for item in batch:
        seq_len = len(item["tokens"])
        pad_len = max_len - seq_len

        # Pad tokens
        tokens = torch.cat([item["tokens"], torch.zeros(pad_len, dtype=torch.long)])
        batch_tokens.append(tokens)

        # Pad next_tokens
        next_tokens = torch.cat(
            [item["next_tokens"], torch.full((pad_len,), -1, dtype=torch.long)]
        )
        batch_next_tokens.append(next_tokens)

        # Pad states (states has length seq_len + 1)
        states = torch.cat(
            [item["states"], torch.zeros(pad_len, dtype=torch.long)]
        )
        batch_states.append(states)

        # Pad state_classes
        state_classes = torch.cat(
            [item["state_classes"], torch.zeros(pad_len, dtype=torch.long)]
        )
        batch_state_classes.append(state_classes)

        # Pad mask
        mask = torch.cat([item["mask"], torch.zeros(pad_len, dtype=torch.bool)])
        batch_masks.append(mask)

    return {
        "tokens": torch.stack(batch_tokens),
        "next_tokens": torch.stack(batch_next_tokens),
        "states": torch.stack(batch_states),
        "state_classes": torch.stack(batch_state_classes),
        "mask": torch.stack(batch_masks),
    }
