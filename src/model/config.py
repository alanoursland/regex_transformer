"""Model configuration."""
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    """Transformer model configuration."""
    vocab_size: int
    num_classes: int
    num_states: int
    d_model: int = 64
    n_heads: int = 4
    d_mlp: int = 256
    dropout: float = 0.1
    max_seq_len: int = 128
    tie_weights: bool = False
    positional_type: Literal["sin", "none"] = "sin"

    def __post_init__(self):
        """Validate configuration."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads")
        if self.d_model < 1 or self.n_heads < 1 or self.max_seq_len < 1:
            raise ValueError("Invalid config values")
