# Regex Transformer API Documentation

## Overview

This codebase implements a complete pipeline for training transformers on regular expression patterns and analyzing whether they learn interpretable finite state machine structures.

The API is organized into 5 main modules:
1. **FSM** - Compile regex patterns into deterministic finite automata
2. **Data** - Generate balanced training datasets from FSMs
3. **Model** - Single-layer transformer architecture
4. **Train** - Training loop with multi-task learning
5. **Eval** - Evaluation metrics and analysis

---

## Quick Start Example

```python
from pathlib import Path
from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex
from model.config import ModelConfig
from train.loop import train_one_experiment, TrainConfig

# 1. Define a regex pattern
regex_def = RegexDefinition(
    alphabet=("a", "b"),
    patterns=(("a+b+", "accept"),)
)

# 2. Configure model
model_cfg = ModelConfig(
    vocab_size=4,  # PAD, EOS, a, b
    num_classes=3,  # accept, incomplete, reject
    num_states=20,
    d_model=64,
    n_heads=4,
    max_seq_len=20,
)

# 3. Train
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
    results_dir=Path("results/my_experiment"),
    n_samples=1000,
)

print(f"Final validation accuracy: {results['val_metrics']['token_acc']:.3f}")
```

---

## Module 1: FSM - Finite State Machine Compilation

### Core Classes

#### `RegexDefinition`
Defines a regex matching problem with explicit alphabet and labeled patterns.

```python
from fsm.regex_def import RegexDefinition

regex_def = RegexDefinition(
    alphabet=("a", "b", "c"),
    patterns=(
        ("a+", "accept"),
        ("b*c", "loop"),
    )
)
```

**Attributes:**
- `alphabet: Tuple[str, ...]` - Valid input characters (single chars only)
- `patterns: Tuple[Tuple[str, str], ...]` - List of (regex_pattern, class_name) pairs

**Notes:**
- Use explicit parentheses for alternation: `(a|b)` not `a|b`
- Patterns are Python regex syntax
- Class names must be unique

**Methods:**
```python
# Load/save from JSON
from pathlib import Path
regex_def = load_regex_def(Path("pattern.json"))
save_regex_def(regex_def, Path("output.json"))
```

JSON format:
```json
{
  "alphabet": ["a", "b"],
  "patterns": [
    ["a+", "accept"],
    ["b*", "loop"]
  ]
}
```

---

#### `FSM`
Deterministic Finite State Machine representation.

```python
from fsm.compile import compile_regex

fsm = compile_regex(regex_def)
```

**Attributes:**
- `states: int` - Number of states
- `alphabet: Tuple[str, ...]` - Input alphabet
- `start: int` - Start state ID (always 0-indexed)
- `delta: Dict[Tuple[int, int], int]` - Transition function: (state_id, token_id) → next_state_id
- `classes: Tuple[str, ...]` - Class names (ordered)
- `state_class: List[int]` - Mapping from state_id → class_id
- `reject: int` - Explicit reject state ID

**Key Methods:**

```python
# Single-step transition
next_state = fsm.step(current_state, token_id)

# Get class of a state
class_id = fsm.classify(state)
class_name = fsm.classify_name(state)

# Trace full execution
token_ids = [0, 1, 0]  # 'a', 'b', 'a'
states = fsm.trace(token_ids)  # Returns [start, s1, s2, s3]

# Classify a string
result = fsm.classify_string([0, 1])  # Returns class name like "accept"

# Helper conversions
token_id = fsm.char_to_token_id('a')
token_ids = fsm.tokens_from_string("aba")
string = fsm.string_from_tokens([0, 1, 0])
```

**Example:**

```python
from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex

# Define regex
regex_def = RegexDefinition(
    alphabet=("a", "b"),
    patterns=(("ab", "accept"),)
)

# Compile to FSM
fsm = compile_regex(regex_def)

# Test strings
print(fsm.classify_string(fsm.tokens_from_string("ab")))   # "accept"
print(fsm.classify_string(fsm.tokens_from_string("a")))    # "incomplete"
print(fsm.classify_string(fsm.tokens_from_string("ba")))   # "reject"

# Trace execution
states = fsm.trace(fsm.tokens_from_string("ab"))
for i, state in enumerate(states):
    print(f"Step {i}: state={state}, class={fsm.classify_name(state)}")
```

---

### FSM Serialization

```python
from fsm.serialize import save_fsm, load_fsm
from pathlib import Path

# Save FSM
save_fsm(fsm, Path("my_fsm.json"))

# Load FSM
loaded_fsm = load_fsm(Path("my_fsm.json"))
```

---

## Module 2: Data - Dataset Generation

### Core Functions

#### `generate_corpus`
Generate balanced training data from an FSM.

```python
from data.generator import generate_corpus, GenConfig

# Configure generation
gen_cfg = GenConfig(
    L_min=1,
    L_max=10,
    p_class={
        "accept": 0.4,
        "incomplete": 0.3,
        "reject": 0.3,
    },
    attempts_per_sample=100,
)

# Generate samples
samples, class_names, report = generate_corpus(
    fsm,
    gen_cfg,
    n_samples=1000,
    seed=42
)

# samples: List[Tuple[List[int], int]]  # (token_ids, class_id)
# class_names: List[str]
# report: DataReport with statistics
```

**GenConfig Attributes:**
- `L_min: int` - Minimum sequence length
- `L_max: int` - Maximum sequence length
- `p_class: Dict[str, float]` - Class distribution (must sum to 1.0)
- `reject_mix: Dict[str, float]` - Reject subtype distribution
- `attempts_per_sample: int` - Max generation attempts
- `beam_size: int` - Beam size for forward sampling

---

#### `FsmDataset`
PyTorch Dataset with FSM-based labeling.

```python
from data.dataset import FsmDataset

dataset = FsmDataset(
    fsm,
    samples,  # List of (token_ids, class_id)
    split="train"  # "train", "val", or "test"
)

# Access a sample
item = dataset[0]
# Returns dict with keys:
#   - tokens: tensor of token IDs
#   - next_tokens: tensor of next-token targets
#   - class_labels: tensor of per-position class labels
#   - state_labels: tensor of per-position state labels
```

---

#### `make_dataloaders`
Create PyTorch DataLoaders with batching.

```python
from data.loader import make_dataloaders
from data.tokenizer import Vocab

# Create vocabulary
vocab = Vocab.from_alphabet(("a", "b"))

# Create datasets
datasets = {
    "train": FsmDataset(fsm, train_samples, "train"),
    "val": FsmDataset(fsm, val_samples, "val"),
    "test": FsmDataset(fsm, test_samples, "test"),
}

# Make dataloaders
dataloaders = make_dataloaders(
    datasets,
    vocab,
    batch_size=32,
    seed=42
)

# Use in training loop
for batch in dataloaders["train"]:
    tokens = batch["tokens"]  # Shape: [batch_size, seq_len]
    next_tokens = batch["next_tokens"]
    class_labels = batch["class_labels"]
    state_labels = batch["state_labels"]
    attn_mask = batch["attn_mask"]
```

---

### Vocabulary

```python
from data.tokenizer import Vocab

# Create from alphabet
vocab = Vocab.from_alphabet(("a", "b", "c"))

# Special tokens
vocab.pad_id  # Padding token
vocab.eos_id  # End-of-sequence token

# Size
len(vocab)  # vocab_size including special tokens
```

---

## Module 3: Model - Transformer Architecture

### Core Classes

#### `ModelConfig`
Configuration for the transformer model.

```python
from model.config import ModelConfig

config = ModelConfig(
    vocab_size=4,        # Number of tokens (including PAD, EOS)
    num_classes=3,       # Number of class labels
    num_states=20,       # Number of FSM states (for state prediction head)
    d_model=64,          # Model dimension
    n_heads=4,           # Number of attention heads
    d_ff=256,            # FFN hidden dimension
    max_seq_len=20,      # Maximum sequence length
    dropout=0.1,
    weight_tie=False,    # Tie input/output embeddings
    use_state_head=True, # Enable state prediction head
)
```

---

#### `RegexTransformer`
Single-layer transformer with multi-task heads.

```python
from model.transformer import RegexTransformer

model = RegexTransformer(config)

# Forward pass
outputs = model(tokens, attn_mask)

# outputs is a dict with keys:
#   - "next_token_logits": [batch, seq_len, vocab_size]
#   - "class_logits": [batch, seq_len, num_classes]
#   - "state_logits": [batch, seq_len, num_states] (if use_state_head=True)
```

**Example:**

```python
import torch

# Create input
batch_size = 2
seq_len = 5
tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
attn_mask = torch.ones(batch_size, seq_len)

# Forward
outputs = model(tokens, attn_mask)

# Access predictions
next_token_probs = torch.softmax(outputs["next_token_logits"], dim=-1)
class_predictions = torch.argmax(outputs["class_logits"], dim=-1)
```

---

## Module 4: Train - Training Loop

### Core Functions

#### `train_one_experiment`
Complete training pipeline from regex to trained model.

```python
from train.loop import train_one_experiment, TrainConfig
from pathlib import Path

results = train_one_experiment(
    regex_def,
    model_cfg,
    train_cfg,
    results_dir=Path("results/experiment_001"),
    n_samples=1000,
)

# Returns dict with:
#   - "val_metrics": validation metrics dict
#   - "test_metrics": test metrics dict
#   - "best_epoch": int
```

---

#### `TrainConfig`
Training hyperparameters.

```python
train_cfg = TrainConfig(
    epochs=10,
    lr=1e-3,
    batch_size=32,
    seed=42,
    device="cpu",  # or "cuda"
    log_every=10,
    patience=5,  # Early stopping patience
)
```

---

### Loss Functions

```python
from train.losses import compute_multi_task_loss

losses = compute_multi_task_loss(
    outputs,  # Model outputs dict
    batch,    # Batch dict with targets
    lambda_next=1.0,   # Next-token loss weight
    lambda_class=1.0,  # Class loss weight
    lambda_state=0.5,  # State loss weight
)

# Returns dict with:
#   - "loss": total weighted loss
#   - "next_token_loss": CE loss for next token
#   - "class_loss": CE loss for class labels
#   - "state_loss": CE loss for state labels (if applicable)
```

---

## Module 5: Eval - Evaluation Metrics

### Core Functions

#### `compute_metrics`
Evaluate model on a dataset.

```python
from eval.metrics import compute_metrics

metrics = compute_metrics(model, dataloader, device="cpu")

# Returns dict with:
#   - "token_acc": token-level accuracy
#   - "class_acc": class prediction accuracy
#   - "state_acc": state prediction accuracy
#   - "nll": negative log-likelihood
#   - "perplexity": exp(nll)
```

---

## Complete Example: From Scratch

```python
from pathlib import Path
from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex
from data.generator import generate_corpus, GenConfig
from data.dataset import FsmDataset
from data.loader import make_dataloaders
from data.tokenizer import Vocab
from model.config import ModelConfig
from model.transformer import RegexTransformer
from train.loop import TrainConfig, set_seed
from train.losses import compute_multi_task_loss
from train.optim import build_optimizer, clip_gradients
from eval.metrics import compute_metrics
import torch

# 1. Define regex
regex_def = RegexDefinition(
    alphabet=("a", "b"),
    patterns=(("a+b+", "accept"),)
)

# 2. Compile to FSM
fsm = compile_regex(regex_def)
print(f"FSM has {fsm.states} states")

# 3. Generate data
gen_cfg = GenConfig(L_min=1, L_max=10)
samples, class_names, report = generate_corpus(fsm, gen_cfg, n_samples=1000, seed=42)
print(f"Generated {len(samples)} samples")

# 4. Create datasets
n_train = int(0.8 * len(samples))
n_val = int(0.1 * len(samples))

datasets = {
    "train": FsmDataset(fsm, samples[:n_train], "train"),
    "val": FsmDataset(fsm, samples[n_train:n_train+n_val], "val"),
    "test": FsmDataset(fsm, samples[n_train+n_val:], "test"),
}

# 5. Create dataloaders
vocab = Vocab.from_alphabet(regex_def.alphabet)
dataloaders = make_dataloaders(datasets, vocab, batch_size=32, seed=42)

# 6. Build model
model_cfg = ModelConfig(
    vocab_size=len(vocab),
    num_classes=len(fsm.classes),
    num_states=fsm.states,
    d_model=64,
    n_heads=4,
)
model = RegexTransformer(model_cfg)

# 7. Train
set_seed(42)
optimizer = build_optimizer(model, lr=1e-3)

for epoch in range(10):
    model.train()
    for batch in dataloaders["train"]:
        tokens = batch["tokens"]
        attn_mask = batch["attn_mask"]
        
        outputs = model(tokens, attn_mask)
        losses = compute_multi_task_loss(outputs, batch)
        
        optimizer.zero_grad()
        losses["loss"].backward()
        clip_gradients(model, max_norm=1.0)
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        val_metrics = compute_metrics(model, dataloaders["val"], "cpu")
    
    print(f"Epoch {epoch}: val_acc={val_metrics['token_acc']:.3f}")

# 8. Final test
model.eval()
with torch.no_grad():
    test_metrics = compute_metrics(model, dataloaders["test"], "cpu")

print(f"Test accuracy: {test_metrics['token_acc']:.3f}")
```

---

## File Organization

```
src/
├── fsm/              # FSM compilation
│   ├── regex_def.py  # RegexDefinition class
│   ├── compile.py    # compile_regex()
│   ├── dfa.py        # FSM class
│   └── serialize.py  # save/load FSM
│
├── data/             # Data generation
│   ├── generator.py  # generate_corpus()
│   ├── dataset.py    # FsmDataset
│   ├── loader.py     # make_dataloaders()
│   ├── tokenizer.py  # Vocab
│   ├── feasibility.py
│   ├── quotas.py
│   └── telemetry.py
│
├── model/            # Transformer architecture
│   ├── config.py     # ModelConfig
│   ├── transformer.py # RegexTransformer
│   ├── attention.py
│   ├── embedding.py
│   ├── heads.py
│   └── mlp.py
│
├── train/            # Training
│   ├── loop.py       # train_one_experiment()
│   ├── losses.py     # compute_multi_task_loss()
│   ├── optim.py      # optimizers
│   └── checkpoint.py # save/load checkpoints
│
└── eval/             # Evaluation
    ├── metrics.py    # compute_metrics()
    └── ood.py        # OOD analysis
```

---

## Testing

All modules have comprehensive test coverage (80 tests total):

```bash
# Run all tests
pytest

# Run specific module
pytest src/fsm/tests/
pytest src/data/tests/
pytest src/model/tests/

# Run with coverage
pytest --cov=src --cov-report=term-missing
```

---

## Common Patterns

### Pattern 1: Just compile and test an FSM

```python
from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex

regex_def = RegexDefinition(
    alphabet=("a", "b"),
    patterns=(("a+", "accept"),)
)

fsm = compile_regex(regex_def)

# Test some strings
test_strings = ["a", "aa", "b", "ab", ""]
for s in test_strings:
    tokens = fsm.tokens_from_string(s) if s else []
    result = fsm.classify_string(tokens)
    print(f"{s!r:5} -> {result}")
```

### Pattern 2: Generate data without training

```python
from data.generator import generate_corpus, GenConfig

gen_cfg = GenConfig(L_min=1, L_max=5, p_class={"accept": 0.5, "reject": 0.5})
samples, class_names, report = generate_corpus(fsm, gen_cfg, n_samples=100, seed=42)

# Inspect samples
for tokens, class_id in samples[:5]:
    s = fsm.string_from_tokens(tokens)
    class_name = class_names[class_id]
    print(f"{s!r} -> {class_name}")
```

### Pattern 3: Quick experiment with defaults

```python
from train.loop import train_one_experiment
from model.config import ModelConfig
from train.loop import TrainConfig
from pathlib import Path

# Minimal config - use sensible defaults
results = train_one_experiment(
    regex_def,
    ModelConfig(),  # Uses default hyperparameters
    TrainConfig(),  # Uses default training settings
    results_dir=Path("results/quick_test"),
)
```

---

## Notes and Limitations

**Phase 1 Constraints:**
- Single-layer transformer only
- Maximum sequence length limited (default: 10 in generation, 20 in model)
- Simplified regex patterns (use explicit parentheses)
- CPU/single-GPU only

**Design Choices:**
- Direct DFA construction (not Thompson's NFA)
- Partition refinement for minimization
- Pre-norm transformer architecture
- Multi-task learning with next-token + class + state prediction

**Data Generation:**
- Uses backward sampling for exact (class, length) targets
- Includes coverage tracking and quota management
- Supports reject subtypes (overrun, premature, wrong_alt, illegal_step)