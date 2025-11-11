# Experiment 3: Transformer Training and Weight Analysis

## Abstract

Having established that FSMs can be perfectly encoded into attention matrices (Experiment 2), we now investigate whether transformers trained via gradient descent discover this encoding. We present a comprehensive training methodology for single-layer transformers on regex classification tasks, along with analysis tools to compare learned weights against the theoretical FSM construction. This experiment bridges the gap between representational capacity (what transformers can represent) and learnability (what they actually learn through training).

## 1. Introduction

### 1.1 Research Question

Experiments 1 and 2 demonstrated that:
1. FSMs correctly implement regex patterns
2. FSMs can be exactly encoded as attention value matrices

The critical remaining question is: **Do transformers trained on regex tasks converge to the FSM-like encoding, or do they discover alternative representations?**

### 1.2 Experimental Goals

1. **Train transformers** on regex classification tasks using gradient descent
2. **Extract learned weights** from trained models
3. **Compare learned weights** to theoretical FSM construction
4. **Measure similarity** using multiple metrics (cosine similarity, correlation, Frobenius distance)
5. **Analyze behavior** to determine if learned models match FSM predictions

### 1.3 Significance

This experiment addresses fundamental questions about neural network learning:
- Do neural networks find theoretically optimal solutions?
- How similar are learned representations to hand-crafted encodings?
- Can transformers discover symbolic algorithms through gradient descent?

## 2. Methodology

### 2.1 Training Pipeline

#### 2.1.1 Data Generation

For each regex pattern, we generate training samples using controlled data generation:

**Generation Parameters:**
```python
GenConfig(
    L_min=1,              # Minimum sequence length
    L_max=20,             # Maximum sequence length
    p_class={             # Class distribution
        'accept': 1/3,
        'incomplete': 1/3,
        'reject': 1/3
    }
)
```

**Sample Structure:**
- Total samples: 2000 (default), configurable up to 10000+
- Train/Val/Test split: 80/10/10
- Balanced class distribution across accept/incomplete/reject states
- Variable sequence lengths to prevent length-based learning

**Generation Process:**
1. Compile FSM from regex pattern
2. Sample target (class, length) pairs
3. Generate sequences that reach target classifications
4. Validate all samples against FSM ground truth
5. Tokenize and batch for training

#### 2.1.2 Model Architecture

**Single-Layer Transformer:**
```python
ModelConfig(
    d_model=64,           # Embedding dimension
    n_heads=4,            # Attention heads
    d_mlp=256,            # MLP hidden dimension
    dropout=0.1,          # Dropout rate
    max_seq_len=128,      # Maximum sequence length
    positional_type="sin" # Sinusoidal positional encoding
)
```

**Architecture Components:**
1. **Token Embedding:** Maps alphabet symbols to d_model dimensions
2. **Positional Encoding:** Sinusoidal encoding for position information
3. **Transformer Block:**
   - Multi-head self-attention (causal)
   - Layer normalization (pre-norm)
   - Feed-forward MLP
   - Residual connections
4. **Task Heads:**
   - Next token prediction head
   - State classification head (accept/incomplete/reject)

#### 2.1.3 Training Configuration

**Optimization:**
```python
TrainConfig(
    epochs=50,            # Training epochs
    batch_size=32,        # Batch size
    lr=1e-3,              # Learning rate (Adam)
    patience=10,          # Early stopping patience
    device="cuda"         # GPU acceleration
)
```

**Loss Function:**
- Multi-task loss combining:
  - Cross-entropy for state classification
  - Cross-entropy for next token prediction
  - Weighted combination to balance objectives

**Regularization:**
- Dropout (0.1) in attention and MLP
- Gradient clipping (max norm = 1.0)
- Early stopping based on validation loss

**Optimization Strategy:**
- Adam optimizer with default β parameters
- No learning rate scheduling (fixed rate)
- Gradient clipping for stability

### 2.2 Weight Extraction and Comparison

#### 2.2.1 Learned Weight Extraction

From trained transformer, extract value projection matrices:

```python
V_learned = model.block.attn.v_proj.weight  # (d_model, d_model)
V_reshaped = V_learned.view(n_heads, d_head, d_model)
```

For comparison with FSM construction:
- Extract submatrix corresponding to first n_states dimensions
- Each head corresponds to one alphabet symbol
- Compare head-by-head against constructed value matrices

#### 2.2.2 Theoretical Construction Baseline

Load pre-computed FSM construction:

```python
construction = torch.load("fsm_construction.pt")
V_constructed = construction['V']  # [n_heads][n_states][n_states]
```

Constructed during training to ensure identical FSM reference.

#### 2.2.3 Similarity Metrics

**1. Cosine Similarity** (directional alignment):
```
cosine_sim = (V_learned · V_constructed) / (||V_learned|| ||V_constructed||)
```
- Range: [-1, 1], higher is better
- Measures if weights point in same direction
- Invariant to magnitude scaling

**2. Correlation** (linear relationship):
```
corr = pearson_correlation(flatten(V_learned), flatten(V_constructed))
```
- Range: [-1, 1], higher is better
- Measures linear dependence
- Captures structural similarity

**3. Frobenius Distance** (magnitude difference):
```
frob_dist = ||V_learned - V_constructed||_F
relative_error = frob_dist / ||V_constructed||_F
```
- Range: [0, ∞], lower is better
- Measures element-wise differences
- Normalized for scale comparison

**4. Per-Head Analysis:**
- Compute all metrics for each attention head separately
- Identify which heads (symbols) are learned better
- Aggregate statistics across heads

#### 2.2.4 Behavioral Analysis

Beyond weight comparison, test if learned model matches FSM behavior:

**Test Procedure:**
1. Select test strings covering pattern variants
2. Execute FSM to get state trace and classification
3. Execute trained model to get predictions
4. Compare state-by-state and final classifications

**Metrics:**
- State trace agreement: Do predicted states match FSM states?
- Classification accuracy: Does final prediction match FSM classification?
- Per-position accuracy: How many positions match exactly?

## 3. Experimental Infrastructure

### 3.1 Training Script

**Location:** `src/experiments/train_model.py`

**Usage:**
```bash
python -m experiments.train_model --pattern "a+" --epochs 50 --n_samples 2000 --batch_size 32 --device cuda --seed 42 --train_classes accept --lambda_class 0.0
```

**Full Parameter Reference:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pattern` | str | required | Regex pattern to learn (e.g., "a+", "a*b*") |
| `--alphabet` | str[] | ["a", "b"] | Alphabet symbols |
| `--n_samples` | int | 2000 | Number of training samples |
| `--min_len` | int | 1 | Minimum sequence length |
| `--max_len` | int | 20 | Maximum sequence length |
| `--d_model` | int | 64 | Model embedding dimension |
| `--n_heads` | int | 4 | Number of attention heads |
| `--d_mlp` | int | 256 | MLP hidden dimension |
| `--dropout` | float | 0.1 | Dropout rate |
| `--epochs` | int | 50 | Training epochs |
| `--batch_size` | int | 32 | Batch size |
| `--lr` | float | 1e-3 | Learning rate |
| `--patience` | int | 10 | Early stopping patience |
| `--device` | str | auto | Device (cuda/cpu) |
| `--seed` | int | 42 | Random seed |
| `--tie_weights` | flag | false | Tie embedding and output weights |
| `--positional_type` | str | "sin" | Positional encoding (sin/none) |

**Output:**

Results saved to `src/results/{pattern}/{timestamp}/`:
- `best.pt` - Best model checkpoint (lowest validation loss)
- `final.pt` - Final model checkpoint
- `fsm_construction.pt` - Theoretical FSM construction baseline
- `metadata.json` - Complete experiment configuration
- `history.json` - Training history (loss, metrics per epoch)
- `metrics.json` - Final validation and test metrics
- `regex_def.json` - Regex definition for reproducibility

### 3.2 Weight Comparison Script

**Location:** `src/experiments/compare_weights.py`

**Usage:**
```bash
python -m experiments.compare_weights \
  src/results/aplus/20241110_153000 \
  --checkpoint best.pt \
  --visualize_heads 0 1 \
  --test_strings a aa aaa b ab ba
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results_dir` | str | required | Path to training results directory |
| `--checkpoint` | str | "best.pt" | Checkpoint to analyze (best.pt/final.pt) |
| `--visualize_heads` | int[] | none | Heads to visualize (0-indexed) |
| `--test_strings` | str[] | defaults | Test strings for behavioral analysis |

**Output:**

Console display:
```
Aggregate Metrics:
  Mean Cosine Similarity: 0.7234
  Mean Correlation:       0.6891
  Mean Frobenius Dist:    0.4521

Per-Head Analysis:
  Head 0 (symbol 'a'):
    Cosine Similarity: 0.8123
    Correlation:       0.7654
    Relative Error:    0.3201

  Head 1 (symbol 'b'):
    Cosine Similarity: 0.6345
    Correlation:       0.6128
    Relative Error:    0.5841

Behavioral Analysis:
  ✓ 'a'    FSM: [0,1] (accept)    Model: [0,1] (accept)
  ✓ 'aa'   FSM: [0,1,2] (accept)  Model: [0,1,2] (accept)
  ✗ 'b'    FSM: [0,11] (reject)   Model: [0,11] (reject)
```

Saved to `weight_comparison.json`:
```json
{
  "comparison_metrics": {
    "mean_similarity": 0.7234,
    "mean_correlation": 0.6891,
    "per_head": [...]
  },
  "behavior_analysis": [...]
}
```

### 3.3 Results Browser

**Location:** `src/experiments/list_results.py`

**Usage:**
```bash
# List all experiments
python -m experiments.list_results

# Filter by pattern
python -m experiments.list_results --pattern aplus

# Find best result by metric
python -m experiments.list_results --best --metric class_acc
```

## 4. Replication Instructions

### 4.1 Environment Setup

**Requirements:**
```bash
# Core dependencies
pip install torch torchvision  # PyTorch with CUDA
pip install numpy              # Numerical operations

# Already available in repository
# - FSM compilation (src/fsm/)
# - Data generation (src/data/)
# - Model architecture (src/model/)
# - Training loop (src/train/)
```

**Verify GPU:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 4.2 Quick Start

**Train on simple pattern:**
```bash
python -m experiments.train_model --pattern "a+" --epochs 50
```

**Monitor training:**
```
Epoch   1/50 | Train: 0.8234 | Val NLL: 0.6543 | Val Tok Acc: 0.812 | Val Cls Acc: 0.765
Epoch   5/50 | Train: 0.4521 | Val NLL: 0.3891 | Val Tok Acc: 0.891 | Val Cls Acc: 0.843
...
Early stopping at epoch 35

Results saved to: src/results/aplus/20241110_153000
```

**Compare weights:**
```bash
python -m experiments.compare_weights src/results/aplus/20241110_153000
```

### 4.3 Comprehensive Experiment Suite

**Test patterns:**
```bash
# Simple repetition
python -m experiments.train_model --pattern "a+" --epochs 50 --device cuda

# Kleene star concatenation
python -m experiments.train_model --pattern "a*b*" --epochs 50 --device cuda

# Alternation
python -m experiments.train_model --pattern "(a|b)+" --epochs 50 --device cuda

# Exact sequence
python -m experiments.train_model --pattern "ab" --epochs 50 --device cuda
```

**Hyperparameter sweep:**
```bash
# Vary model size
for d_model in 32 64 128; do
  python -m experiments.train_model \
    --pattern "a+" \
    --d_model $d_model \
    --epochs 50 \
    --device cuda
done

# Vary number of heads
for n_heads in 2 4 8; do
  python -m experiments.train_model \
    --pattern "(a|b)+" \
    --n_heads $n_heads \
    --epochs 50 \
    --device cuda
done

# Vary dataset size
for n_samples in 1000 2000 5000 10000; do
  python -m experiments.train_model \
    --pattern "a*b*" \
    --n_samples $n_samples \
    --epochs 50 \
    --device cuda
done
```

### 4.4 Analysis Workflow

**1. Train model:**
```bash
python -m experiments.train_model \
  --pattern "a+" \
  --epochs 100 \
  --n_samples 5000 \
  --device cuda
```

Note the output path: `src/results/aplus/20241110_153000`

**2. Compare weights:**
```bash
python -m experiments.compare_weights \
  src/results/aplus/20241110_153000 \
  --visualize_heads 0 1
```

**3. Review metrics:**
```bash
cat src/results/aplus/20241110_153000/metrics.json
cat src/results/aplus/20241110_153000/weight_comparison.json
```

**4. Find best result:**
```bash
python -m experiments.list_results --best --metric class_acc
```

## 5. Expected Results and Interpretation

### 5.1 Training Metrics

**Successful Training Indicators:**
- Val Token Accuracy > 0.85 (model predicts next tokens well)
- Val Class Accuracy > 0.80 (model classifies states correctly)
- Convergence within 30-50 epochs
- Val loss decreases steadily

**Poor Training Indicators:**
- Val accuracy < 0.70 after 50 epochs
- Val loss increases (overfitting)
- Train/val accuracy gap > 0.15 (generalization issues)

### 5.2 Weight Similarity Interpretation

**Cosine Similarity:**
- **>0.9:** Very high alignment - learned weights closely match FSM encoding
- **0.7-0.9:** Moderate alignment - similar direction but different magnitude
- **0.5-0.7:** Weak alignment - learned representation differs from FSM
- **<0.5:** No alignment - model found alternative representation

**Relative Error:**
- **<0.1:** Nearly identical to FSM construction
- **0.1-0.3:** Close approximation with minor differences
- **0.3-0.6:** Moderate differences, similar structure
- **>0.6:** Substantially different from FSM construction

### 5.3 Behavioral Analysis Interpretation

**Perfect Behavioral Match (✓✓✓):**
- All test strings match FSM predictions
- Model has learned FSM semantics
- May use different internal representation

**Partial Behavioral Match (✓✗):**
- Some strings match, others don't
- Model partially learned pattern
- May need more training or different architecture

**No Behavioral Match (✗✗✗):**
- Predictions consistently differ from FSM
- Model failed to learn pattern
- Check training metrics, data generation, architecture

### 5.4 Research Questions Answered

**Q1: Do transformers learn FSM-like weights?**
- Answer depends on similarity metrics
- High cosine similarity (>0.8) suggests yes
- Low similarity suggests alternative representation

**Q2: Does behavioral equivalence require weight similarity?**
- Models can match behavior without matching weights
- Different representations can implement same function
- Perfect behavior + low weight similarity = alternative encoding discovered

**Q3: What factors affect convergence to FSM encoding?**
- Model capacity (d_model, n_heads)
- Dataset size and diversity
- Training duration and regularization
- Pattern complexity

## 6. Troubleshooting

### 6.1 Training Issues

**Problem:** Model not learning (accuracy stuck at ~0.33)
- **Solution:** Check data generation (are samples valid?)
- **Solution:** Increase training samples (try 5000+)
- **Solution:** Reduce learning rate (try 5e-4)
- **Solution:** Increase model capacity (d_model=128)

**Problem:** Overfitting (train acc high, val acc low)
- **Solution:** Increase dropout (try 0.2 or 0.3)
- **Solution:** Reduce model size
- **Solution:** Add more training samples
- **Solution:** Earlier stopping (reduce patience)

**Problem:** Out of memory on GPU
- **Solution:** Reduce batch size (try 16 or 8)
- **Solution:** Reduce d_model or d_mlp
- **Solution:** Reduce max_seq_len

### 6.2 Comparison Issues

**Problem:** Similarity metrics very low (<0.3)
- **Expected:** Transformers may learn different representations
- **Check:** Is behavioral match good? (more important than weight match)
- **Try:** Different architecture (n_heads = alphabet_size)
- **Try:** Longer training (more epochs)

**Problem:** Weight shapes don't match for comparison
- **Check:** Model d_model >= FSM num_states
- **Check:** Model n_heads >= alphabet_size
- **Solution:** Comparison extracts relevant submatrices automatically

### 6.3 Replication Issues

**Problem:** Different results with same seed
- **Possible:** GPU non-determinism
- **Solution:** Set deterministic flags (at cost of performance)
- **Accept:** Minor variation is expected

**Problem:** Can't find training results
- **Check:** Look in `src/results/{pattern}/{timestamp}/`
- **Solution:** Use `list_results.py` to browse
- **Check:** Timestamp in output at end of training

## 7. Future Directions

### 7.1 Architecture Variations

- **Deeper models:** 2-3 transformer layers
- **Different attention:** Relative positional encoding
- **No positional encoding:** Test if model learns from transitions alone
- **Varying d_model:** Test if capacity affects convergence

### 7.2 Training Variations

- **Curriculum learning:** Start with simple patterns, increase complexity
- **Regularization toward FSM:** Add auxiliary loss encouraging FSM-like weights
- **Different optimizers:** SGD, AdamW, different schedules
- **Pre-training:** Initialize weights close to FSM construction

### 7.3 Analysis Extensions

- **Attention pattern visualization:** Heatmaps of attention weights
- **Gradient analysis:** How does gradient flow through network?
- **Intermediate checkpoint analysis:** Track weight evolution during training
- **Probe classifiers:** Can linear probes recover FSM states from embeddings?

### 7.4 Pattern Complexity

- **More complex patterns:** (a|b)*c, a+b+c+, nested operators
- **Larger alphabets:** 3-5 symbols
- **Longer sequences:** max_len = 50-100
- **Real-world patterns:** Email, URL, date validation

## 8. Conclusion

This experiment provides infrastructure and methodology for investigating whether transformers learn FSM-like representations through gradient descent. The combination of training scripts, weight comparison tools, and behavioral analysis enables systematic exploration of the relationship between representational capacity and learned representations.

**Key Contributions:**
1. **Reproducible training pipeline** with comprehensive logging
2. **Multi-metric weight comparison** framework
3. **Behavioral validation** against FSM ground truth
4. **Complete replication instructions** for independent verification

**Research Impact:**
- Bridges theory (FSMs can be encoded) and practice (what is learned)
- Provides empirical evidence for/against convergence to optimal encodings
- Establishes methodology for comparing learned vs. hand-crafted representations
- Informs understanding of what neural networks learn vs. what they can represent

**Next Steps:**
After running experiments, analyze results to determine:
- Do transformers converge to FSM-like weights?
- What architectural choices affect convergence?
- Can we guide learning toward optimal encodings?
- What alternative representations do models discover?

These questions form the foundation for understanding how neural networks learn symbolic algorithms and whether theoretical capacity translates to practical learnability.
