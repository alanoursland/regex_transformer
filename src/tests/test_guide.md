# Unit Tests for Regex Transformer

## Created Tests

### 1. `test_attention_masking.py` - Attention Layer Tests
Tests that attention doesn't produce NaN values:

**TestAttentionMasking**
- ✓ No mask produces valid output
- ✓ Valid mask produces valid output
- ✓ All padding positions produce valid output (critical - this will fail with current bug)
- ✓ Causal mask works
- ✓ Batch with different lengths
- ✓ Single valid position edge case

**TestAttentionNumericalStability**
- ✓ Large sequence length
- ✓ Extreme input values
- ✓ Zero input

**TestAttentionGradients**
- ✓ Gradients are finite
- ✓ Gradients finite with padding

### 2. `test_losses.py` - Loss Computation Tests
Tests that loss computation is robust:

**TestLossComputation**
- ✓ Basic loss is finite
- ✓ Loss with partial masking
- ✓ Loss with single valid position
- ✓ Perfect predictions give low loss
- ✓ Wrong predictions give high loss
- ✓ Lambda weights work correctly

**TestLossEdgeCases**
- ✓ Empty mask doesn't crash
- ✓ Ignore index (-1) works
- ✓ Batch size one

**TestLossGradients**
- ✓ Backward produces finite gradients

### 3. `test_collation.py` - Data Pipeline Tests
Tests that data collation produces valid batches:

**TestCollation**
- ✓ Single sequence
- ✓ Multiple sequences same length
- ✓ Different lengths (padding)
- ✓ Loss mask has valid positions (critical)
- ✓ Padding positions masked

**TestDatasetIntegration**
- ✓ FsmDataset produces valid shapes
- ✓ FsmDataset with EOS
- ✓ End-to-end batch creation

**TestMaskingLogic**
- ✓ Single token sequence
- ✓ Empty sequence handling
- ✓ All padding batch

## Running Tests

```bash
# Run all tests
cd src
pytest tests/test_attention_masking.py tests/test_losses.py tests/test_collation.py -v

# Run specific test class
pytest tests/test_attention_masking.py::TestAttentionMasking -v

# Run with coverage
pytest tests/ --cov=model --cov=train --cov=data --cov-report=html
```

## Expected Failures (Before Fix)

With the current bug in `model/attention.py` line 46:

**WILL FAIL:**
- `test_all_padding_positions_produce_valid_output` - NaN from masking FROM padding
- `test_valid_mask_produces_valid_output` - NaN with any padding

**After fix (removing line 46):**
- All tests should pass

## What These Tests Catch

1. **NaN in forward pass** - Catches attention masking bug
2. **NaN in loss** - Catches loss computation issues
3. **Empty loss_mask** - Catches collation bug where no positions are valid
4. **Gradient explosions** - Catches numerical instability
5. **Edge cases** - Single positions, empty batches, etc.

## Coverage

These tests cover:
- ✓ Attention layer forward pass
- ✓ Attention masking logic
- ✓ Loss computation
- ✓ Loss masking
- ✓ Data collation
- ✓ Padding handling
- ✓ Gradient computation

Not covered yet:
- Full model end-to-end
- Training loop
- Data generation
- FSM construction

## Adding More Tests

To add tests for other components:

```python
# tests/test_model.py
def test_full_model_forward():
    """Test full model produces valid outputs."""
    from model.transformer import RegexTransformer
    from model.config import ModelConfig
    
    config = ModelConfig(vocab_size=4, num_classes=3, ...)
    model = RegexTransformer(config)
    
    tokens = torch.randint(0, 4, (2, 5))
    attn_mask = torch.ones(2, 5, dtype=torch.bool)
    
    outputs = model(tokens, attn_mask)
    
    assert not torch.isnan(outputs["next_token_logits"]).any()
    assert not torch.isnan(outputs["class_logits"]).any()
```

## Integration with CI

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install torch pytest pytest-cov
      - name: Run tests
        run: |
          cd src
          pytest tests/ -v --cov=. --cov-report=xml
```

## Debugging Failed Tests

If a test fails:

1. Run with `-vv` for verbose output:
   ```bash
   pytest tests/test_attention_masking.py::TestAttentionMasking::test_all_padding_positions_produce_valid_output -vv
   ```

2. Add print statements in test:
   ```python
   def test_something():
       out = model(x)
       print(f"Output range: [{out.min()}, {out.max()}]")
       print(f"Has NaN: {torch.isnan(out).any()}")
       assert not torch.isnan(out).any()
   ```

3. Use debugger:
   ```bash
   pytest --pdb tests/test_attention_masking.py
   ```

## Test Philosophy

These tests follow the principle:
- **Test behavior, not implementation**
- Focus on: Does it produce valid outputs? Are gradients finite?
- Not: Are the internal computations exactly as expected?

This catches bugs like the attention masking issue where the implementation looked plausible but produced NaN in edge cases.