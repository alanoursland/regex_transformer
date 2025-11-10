# Experiment Scripts

This directory contains scripts for training transformer models and comparing learned weights to FSM constructions.

## Scripts

### 1. `train_model.py` - Train a Transformer on Regex Pattern

Trains a single-layer transformer to learn FSM behavior from a regex pattern.

**Basic Usage:**
```bash
python -m experiments.train_model --pattern "a+" --epochs 50
```

**Full Options:**
```bash
python -m experiments.train_model \
  --pattern "a*b*" \
  --alphabet a b \
  --n_samples 5000 \
  --min_len 1 \
  --max_len 20 \
  --d_model 64 \
  --n_heads 4 \
  --d_mlp 256 \
  --dropout 0.1 \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-3 \
  --patience 10 \
  --device cuda \
  --seed 42
```

**Output:**

Results are saved in `src/results/{pattern}/{timestamp}/`:
- `best.pt` - Best model checkpoint (lowest validation loss)
- `final.pt` - Final model checkpoint
- `fsm_construction.pt` - Theoretical QKV matrices from FSM
- `metadata.json` - Full experiment configuration
- `history.json` - Training history (loss, metrics per epoch)
- `metrics.json` - Final validation and test metrics
- `regex_def.json` - Regex definition for reproducibility

**Examples:**

Train on simple pattern:
```bash
python -m experiments.train_model --pattern "a+" --epochs 50
```

Train on more complex pattern with GPU:
```bash
python -m experiments.train_model \
  --pattern "(a|b)+" \
  --n_samples 10000 \
  --epochs 100 \
  --device cuda
```

Train with specific architecture:
```bash
python -m experiments.train_model \
  --pattern "a*b*" \
  --d_model 128 \
  --n_heads 2 \
  --d_mlp 512
```

---

### 2. `compare_weights.py` - Compare Learned vs Constructed Weights

Loads a trained model and compares its learned attention weights to the theoretically constructed FSM weights.

**Basic Usage:**
```bash
python -m experiments.compare_weights src/results/aplus/20241110_153000
```

**Full Options:**
```bash
python -m experiments.compare_weights \
  src/results/aplus/20241110_153000 \
  --checkpoint best.pt \
  --visualize_heads 0 1 \
  --test_strings a aa aaa b ab ba
```

**What It Does:**

1. **Weight Comparison:**
   - Extracts learned value matrices from the transformer
   - Compares them to constructed FSM value matrices
   - Computes cosine similarity, correlation, Frobenius distance
   - Reports per-head and aggregate metrics

2. **Behavioral Analysis:**
   - Runs test strings through both FSM and transformer
   - Compares state sequences and classifications
   - Reports whether behavior matches

**Output:**

Console output shows:
- Aggregate similarity metrics
- Per-head comparison statistics
- Optional: Side-by-side weight matrix visualization
- Behavioral test results (FSM vs learned)

Also saves `weight_comparison.json` in the results directory with:
- All comparison metrics
- Per-head analysis
- Behavioral test results

**Examples:**

Basic comparison:
```bash
python -m experiments.compare_weights src/results/aplus/20241110_153000
```

Compare with visualization:
```bash
python -m experiments.compare_weights \
  src/results/aplus/20241110_153000 \
  --visualize_heads 0 1
```

Test specific strings:
```bash
python -m experiments.compare_weights \
  src/results/astarb/20241110_154500 \
  --test_strings "" a b aa bb ab aabb ba
```

---

### 3. `test_data_generation.py` - Test Data Generation Pipeline

Tests that data generation works correctly (requires numpy).

**Usage:**
```bash
python -m experiments.test_data_generation
```

This verifies:
- FSM compilation works
- Data generation produces valid samples
- Generated samples are correctly labeled
- Multiple patterns can be generated

**Note:** This requires numpy and other dependencies. If they're not available, skip this test - the training script will catch any issues.

---

## Results Directory Structure

```
src/results/
├── aplus/                           # Pattern: a+
│   ├── 20241110_153000/            # Timestamp
│   │   ├── best.pt                 # Best checkpoint
│   │   ├── final.pt                # Final checkpoint
│   │   ├── fsm_construction.pt     # FSM QKV matrices
│   │   ├── metadata.json           # Config
│   │   ├── history.json            # Training history
│   │   ├── metrics.json            # Final metrics
│   │   ├── regex_def.json          # Regex definition
│   │   └── weight_comparison.json  # Comparison results
│   └── 20241110_160000/            # Another run
│       └── ...
├── astarb/                         # Pattern: a*b*
│   └── ...
└── aoorbplus/                      # Pattern: (a|b)+
    └── ...
```

## Typical Workflow

1. **Train a model:**
   ```bash
   python -m experiments.train_model --pattern "a+" --epochs 50 --device cuda
   ```

2. **Note the results path** (printed at end of training):
   ```
   Results saved to: src/results/aplus/20241110_153000
   ```

3. **Compare weights:**
   ```bash
   python -m experiments.compare_weights src/results/aplus/20241110_153000
   ```

4. **Analyze results:**
   - Check `weight_comparison.json` for similarity metrics
   - High cosine similarity (>0.8) suggests model learned FSM-like weights
   - Low relative error (<0.5) suggests close approximation
   - Behavioral match indicates correct classification

## Interpreting Results

### Weight Comparison Metrics

- **Cosine Similarity** (0 to 1): How similar the direction of weight vectors is
  - >0.9: Very similar
  - 0.7-0.9: Moderately similar
  - <0.7: Different

- **Correlation** (-1 to 1): Linear relationship between weights
  - >0.8: Strong positive correlation
  - 0.5-0.8: Moderate correlation
  - <0.5: Weak correlation

- **Relative Error** (0 to inf): Normalized distance between matrices
  - <0.1: Very close
  - 0.1-0.5: Reasonably close
  - >0.5: Different

### Behavioral Analysis

Perfect match (✓) means:
- Transformer predicts exact same states as FSM
- Classification (accept/reject/incomplete) matches

Mismatch (✗) means:
- Model hasn't fully learned the FSM
- May need more training, different architecture, or regularization

## Requirements

- PyTorch (with CUDA for GPU training)
- NumPy
- The rest of the `src/` package dependencies

## Tips

1. **Start simple:** Train on `a+` or `(a|b)+` first to verify setup
2. **Use GPU:** Training is much faster with `--device cuda`
3. **Increase samples:** For complex patterns, use `--n_samples 5000` or more
4. **Monitor overfitting:** Check validation loss in history.json
5. **Compare early and late:** Run comparison on both early and final checkpoints
6. **Try different architectures:** Vary `--d_model`, `--n_heads` to see what works best

## Troubleshooting

**Training is slow:**
- Use GPU with `--device cuda`
- Reduce `--n_samples` or `--max_len`
- Increase `--batch_size`

**Model not learning:**
- Increase `--epochs` and `--patience`
- Increase `--n_samples`
- Try different `--lr` (learning rate)
- Check if pattern is too complex for architecture

**Weights don't match:**
- This is expected! The model learns its own representation
- Focus on behavioral match rather than weight similarity
- Try simpler patterns first
- Consider if the model has enough capacity (d_model, n_heads)

**Out of memory:**
- Reduce `--batch_size`
- Reduce `--d_model` or `--d_mlp`
- Reduce `--max_len`
