"""Vocabulary and tokenization for FSM-based datasets."""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Vocab:
    """
    Vocabulary mapping between characters and integer IDs.

    Attributes:
        itos: Index-to-string mapping (list where index is token ID)
        stoi: String-to-index mapping (dict)
        pad_id: ID for padding token
        eos_id: ID for end-of-sequence token
        bos_id: Optional ID for beginning-of-sequence token
    """

    itos: List[str]
    stoi: dict[str, int]
    pad_id: int
    eos_id: int
    bos_id: Optional[int] = None

    @classmethod
    def from_alphabet(
        cls,
        alphabet: Tuple[str, ...],
        add_special_tokens: bool = True,
    ) -> "Vocab":
        """
        Build vocabulary from FSM alphabet.

        Args:
            alphabet: Tuple of characters from FSM
            add_special_tokens: Whether to add PAD/EOS tokens

        Returns:
            Vocab instance
        """
        itos = []
        stoi = {}

        # Add special tokens first for stable IDs
        if add_special_tokens:
            itos.append("<PAD>")
            itos.append("<EOS>")
            stoi["<PAD>"] = 0
            stoi["<EOS>"] = 1
            pad_id = 0
            eos_id = 1
        else:
            pad_id = None
            eos_id = None

        # Add alphabet symbols
        start_idx = len(itos)
        for i, char in enumerate(alphabet):
            itos.append(char)
            stoi[char] = start_idx + i

        return cls(
            itos=itos,
            stoi=stoi,
            pad_id=pad_id,
            eos_id=eos_id,
            bos_id=None,  # Not using BOS for now
        )

    def encode(self, text: str) -> List[int]:
        """
        Convert string to list of token IDs.

        Args:
            text: Input string

        Returns:
            List of token IDs
        """
        return [self.stoi[char] for char in text]

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs to string.

        Skips special tokens (PAD, EOS, BOS).

        Args:
            ids: List of token IDs

        Returns:
            Decoded string
        """
        chars = []
        for token_id in ids:
            # Skip special tokens
            if token_id == self.pad_id:
                continue
            if token_id == self.eos_id:
                continue
            if self.bos_id is not None and token_id == self.bos_id:
                continue

            if 0 <= token_id < len(self.itos):
                chars.append(self.itos[token_id])

        return "".join(chars)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.itos)

    def validate_alphabet(self, alphabet: Tuple[str, ...]) -> None:
        """
        Validate that all alphabet symbols are in vocab.

        Args:
            alphabet: FSM alphabet to validate

        Raises:
            ValueError: If any symbol is missing
        """
        for char in alphabet:
            if char not in self.stoi:
                raise ValueError(f"Alphabet symbol {char!r} not in vocabulary")
