# Regex Transformer

Train single-layer transformers on regex patterns to analyze learned FSM structure.

## Features

**Complete Pipeline:**
- ✅ FSM compilation from regex patterns
- ✅ Balanced dataset generation with coverage tracking
- ✅ Single-layer transformer with multi-task learning
- ✅ Training loop with early stopping
- ✅ Comprehensive evaluation metrics
- ✅ Out-of-distribution (OOD) analysis

**80 Passing Tests:**
- 15 FSM core tests
- 45 data generation & loading tests
- 11 model architecture tests
- 3 training loop tests
- 4 evaluation tests
- 2 end-to-end integration tests

## Quick Start

```python
from src.fsm.regex_def import RegexDefinition
from src.model.config import ModelConfig
from src.train.loop import train_one_experiment, TrainConfig
from pathlib import Path

# Define regex
regex_def = RegexDefinition(
    alphabet=("a", "b"),
    patterns=(("a+", "accept"),)
)

# Configure model
model_cfg = ModelConfig(
    vocab_size=4,  # PAD, EOS, a, b
    num_classes=3,  # accept, incomplete, reject
    num_states=20,
    d_model=64,
    n_heads=4,
)

# Train
train_cfg = TrainConfig(
    epochs=10,
    lr=1e-3,
    batch_size=32,
    seed=42,
)

results = train_one_experiment(
    regex_def,
    model_cfg,
    train_cfg,
    results_dir=Path("results/experiment"),
    n_samples=1000,
)

print(f"Val accuracy: {results['val_metrics']['token_acc']:.3f}")
print(f"Test accuracy: {results['test_metrics']['token_acc']:.3f}")
```

## Architecture

**FSM Module** (`src/fsm/`):
- Regex → DFA compilation with minimization
- State tracing and classification
- Serialization with checksums

**Data Module** (`src/data/`):
- Feasibility-based generation (backward/forward sampling)
- Coverage-aware quota management
- PyTorch Dataset with on-the-fly labeling
- Deterministic train/val/test splitting

**Model Module** (`src/model/`):
- Single-layer transformer
- Multi-head causal self-attention
- Dual task heads (next-token + state-class prediction)
- Weight tying support

**Training Module** (`src/train/`):
- Multi-task loss with masking
- AdamW optimizer with gradient clipping
- Checkpointing with early stopping
- Deterministic seeding

**Evaluation Module** (`src/eval/`):
- Token and class accuracy
- NLL and perplexity
- Length-binned metrics
- OOD gap computation

## Testing

```bash
# Run all tests
pytest

# Run specific modules
pytest src/fsm/tests/
pytest src/data/tests/
pytest src/model/tests/
pytest tests/  # Integration tests

# Run with coverage
pytest --cov=src --cov-report=term-missing
```

## Results Structure

```
results/experiment/
├── metrics.json       # Final evaluation metrics
├── best.pt           # Best checkpoint by validation loss
└── logs/             # Training logs
```

## Reproducibility

All experiments are fully deterministic with fixed seeds:
- Python random
- NumPy random
- PyTorch (CPU/CUDA)
- Data generation
- DataLoader shuffling

## Implementation Notes

**Phase 1 Limitations:**
- Single-layer transformer only
- Max sequence length limited during generation (default: 10)
- Simplified regex patterns (use explicit parentheses for alternation)
- CPU/single-GPU training only

**Design Choices:**
- Direct DFA construction (not Thompson's NFA)
- Partition refinement minimization
- Inverse-sqrt coverage weighting
- Pre-norm transformer architecture

## Project Structure

```
regex_transformer/
├── src/
│   ├── fsm/          # FSM core (15 tests)
│   ├── data/         # Data generation (45 tests)
│   ├── model/        # Transformer (11 tests)
│   ├── train/        # Training loop (3 tests)
│   └── eval/         # Evaluation (4 tests)
├── tests/            # Integration tests (2 tests)
├── notes/            # Milestone planning docs
└── README.md         # This file
```

## License

Research project - see individual files for details.
