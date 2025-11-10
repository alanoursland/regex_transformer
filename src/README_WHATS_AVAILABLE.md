# Regex Transformer - What You Can Run Right Now

## Quick Demos (No Setup Required)

### 1. FSM Compilation Demo
See regex patterns compiled to finite state machines:

```bash
cd /path/to/regex_transformer
python demo.py
```

**Shows:**
- 4 different regex patterns (a+, a*b*, (a|b)+, ab)
- FSM compilation statistics
- Test strings with accept/reject/incomplete classification
- State traces showing FSM execution step-by-step

**Sample Output:**
```
Pattern: a+
'a'    -> ✓ accept      | 0 → 1
'aa'   -> ✓ accept      | 0 → 1 → 2
'b'    -> ✗ reject      | 0 → 11
```

---

### 2. Interactive FSM Explorer
Play with regex patterns interactively:

```bash
cd /path/to/regex_transformer
python fsm_explorer.py
```

**Features:**
- Choose preset patterns or enter custom regex
- View transition tables
- Test strings interactively
- See state-by-state execution traces

**Example Session:**
```
Choice [1-5]: 1        # Choose "a+" pattern
Options: 
  1. Show transition table
  2. Test strings interactively  ← try this one
  3. Test example strings
  
Test string: aaa
  ✓ 'aaa' -> accept
     Trace: [0] --a--> [1] --a--> [2] --a--> [3]
```

---

### 3. FSM Smoke Test
Verify core FSM functionality:

```bash
cd /path/to/regex_transformer
python -m fsm.smoke_test
```

**Tests:**
- Regex compilation
- String classification
- State tracing
- Serialization/deserialization

---

## What the Codebase Contains

### ✅ Implemented (80 passing tests)

1. **FSM Compilation** (`src/fsm/`)
   - Regex → DFA compilation
   - State minimization
   - Classification (accept/incomplete/reject)
   - Serialization

2. **Data Generation** (`src/data/`)
   - Balanced dataset generation from FSMs
   - Coverage tracking
   - Train/val/test splitting
   - PyTorch Dataset integration

3. **Transformer Model** (`src/model/`)
   - Single-layer transformer
   - Multi-head attention
   - Multi-task learning heads (next-token, class, state)
   - Weight tying support

4. **Training Loop** (`src/train/`)
   - Multi-task loss
   - Checkpointing
   - Early stopping
   - Deterministic seeding

5. **Evaluation** (`src/eval/`)
   - Token/class/state accuracy
   - NLL and perplexity
   - Length-binned metrics

### ❌ Not Yet Implemented

1. **FSM → QKV Construction** (your experiment plan)
   - Direct encoding of FSM into attention matrices
   - Equivalence verification
   - This is what you want to test!

2. **Interpretability Analysis**
   - Attention pattern visualization
   - Linear probes for state extraction
   - Residual stream decomposition

---

## Your Next Steps (Based on Experiment Plan)

### Option 1: Verify Your Theory (Recommended First)

Implement the FSM→QKV construction to test if attention CAN represent FSMs:

```bash
# Files to create:
src/model/fsm_construction.py      # construct_qkv_from_fsm()
src/model/tests/test_construction.py  # Equivalence tests
```

**Time estimate:** 2-3 hours  
**Payoff:** Validates core theoretical claim before training anything

### Option 2: Train a Model (Use Existing Code)

The infrastructure is ready - just run an experiment:

```python
from train.loop import train_one_experiment
from pathlib import Path

results = train_one_experiment(
    regex_def,
    model_cfg,
    train_cfg,
    results_dir=Path("results/my_test"),
    n_samples=1000,
)
```

**Time estimate:** 30 minutes (for simple pattern)  
**Payoff:** See if transformers LEARN to do FSMs

### Option 3: Just Explore

Play with the demos and FSM explorer to build intuition about:
- How FSMs work
- What state traces look like
- How different patterns compile
