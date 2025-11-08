````markdown
# Milestone 4 — Model Foundation
**Goal:** Implement a clean, single-layer transformer with minimal dependencies and task heads for next-token and state/class prediction.  
**Dependencies:** M1 (FSM core), M2 (data gen), M3 (dataset/masks).

---

## 4.1 Package Structure & Scaffolding
- [ ] Create `src/model/` with:
  - `config.py`, `embedding.py`, `positional.py`, `attention.py`, `mlp.py`, `transformer.py`, `heads.py`, `instrumentation.py`, `tests/`
- [ ] Ensure no cyclic imports (model modules import only from stdlib/torch).

---

## 4.2 Model Configuration (`config.py`)
- [ ] Define `ModelConfig` dataclass:
  - [ ] `vocab_size`, `num_classes`, `num_states`
  - [ ] `d_model`, `n_heads`, `d_mlp`, `dropout`
  - [ ] `max_seq_len`, `tie_weights: bool`
  - [ ] `factorized_embedding: bool`, `factor_rank`, `l1_u: float` (optional)
  - [ ] `positional_type: Literal["sin", "none"]`
- [ ] Validation: divisibility (`d_model % n_heads == 0`), bounds (>=1), `max_seq_len > 0`.

---

## 4.3 Embeddings (`embedding.py`)
- [ ] **Token embedding**: `nn.Embedding(vocab_size, d_model)`
- [ ] **Factorized option (optional)**:
  - [ ] `U ∈ R^{vocab×r}`, `V ∈ R^{r×d_model}`, return `U @ V[index]`
  - [ ] Expose L1 penalty term on `U` (to be summed into loss later)
- [ ] Initialization (xavier/uniform) and device/dtype propagation.
- [ ] Unit tests: shape correctness, factorized path parity (forward only).

---

## 4.4 Positional Encoding (`positional.py`)
- [ ] Implement **sinusoidal positional encoding** (static, cache up to `max_seq_len`).
- [ ] Provide `add_positional(x_btd)` utility (no learned params).
- [ ] Unit tests: reproducible values, broadcast to batch, truncation at `T`.

---

## 4.5 Attention (`attention.py`)
- [ ] Implement **scaled dot-product causal self-attention**:
  - [ ] Inputs: `(B, T, d_model)` → outputs `(B, T, d_model)`
  - [ ] Project to Q,K,V: `d_head = d_model // n_heads`
  - [ ] Causal mask: upper triangular `-inf` on logits beyond t
  - [ ] **Padding mask** support: mask out PAD positions (from `attn_mask`)
  - [ ] Dropout on attention weights (configurable)
- [ ] Return attention weights optionally for instrumentation.
- [ ] Unit tests:
  - [ ] Shapes & dtype
  - [ ] Causality: zero influence from future tokens (numerical test)
  - [ ] Masking: PAD positions neither attend nor are attended to

---

## 4.6 MLP (`mlp.py`)
- [ ] 2-layer MLP with activation (GELU) and dropout:
  - [ ] `Linear(d_model → d_mlp) → GELU → Dropout → Linear(d_mlp → d_model)`
- [ ] Unit tests: shapes, no NaNs, respects device/dtype.

---

## 4.7 Single Transformer Block (`transformer.py`)
- [ ] **Pre-norm** architecture:
  - [ ] `x = x + Attn(LN(x))`
  - [ ] `x = x + MLP(LN(x))`
- [ ] Configurable dropout in residual paths (can be 0.0 by default).
- [ ] Causal and padding masks threaded through to attention.
- [ ] Unit tests:
  - [ ] Forward pass shape `(B,T,d_model)`
  - [ ] Deterministic output given fixed seed & inputs
  - [ ] Gradient flows (no detached paths)

---

## 4.8 Task Heads (`heads.py`)
- [ ] **Next-token head**: `Linear(d_model → vocab_size)` (weight tying optional)
- [ ] **Class head**: `Linear(d_model → num_classes)` (per-position state class)
- [ ] **State head (optional)**: `Linear(d_model → num_states)`
- [ ] **Weight tying** (if enabled):
  - [ ] Share token embedding weight with output projection for next-token
  - [ ] Bias handling when tying (use separate bias or none)
- [ ] Unit tests:
  - [ ] Shapes: `(B,T,vocab_size)`, `(B,T,num_classes)`, `(B,T,num_states)`
  - [ ] Tying: check `head.W is embedding.weight` (or `.data_ptr()` equality)

---

## 4.9 Full Model Wrapper
- [ ] `class SingleLayerTransformer(nn.Module)`:
  - [ ] Compose: Embedding → AddPos → SingleBlock → Heads
  - [ ] Forward signature:
    ```python
    def forward(self, tokens_bxt: LongTensor, attn_mask_bxt: BoolTensor,
                return_attn: bool = False) -> dict:
        # returns {"next_logits", "class_logits", "state_logits?", "attn?"}
    ```
  - [ ] Handle EOS/PAD masks (passed in) without internal data loading logic.
- [ ] Unit tests:
  - [ ] End-to-end forward on dummy batch
  - [ ] `return_attn=True` yields a list or tensor of head maps `(B, n_heads, T, T)`

---

## 4.10 Numerical Stability & Init
- [ ] Initialize Q,K,V and output projections with Xavier uniform; MLP similarly.
- [ ] Scale attention by `1/√d_head`.
- [ ] Ensure LayerNorm eps (e.g., `1e-5`) set explicitly.
- [ ] Optional gradient clipping will be handled in M5 (training loop).

---

## 4.11 Instrumentation Hooks (`instrumentation.py`)
- [ ] Lightweight context manager / flag to capture:
  - [ ] Per-head attention maps on selected batches
  - [ ] Hidden states pre/post block
- [ ] Save to a simple Python dict (no heavy trackers).
- [ ] Unit tests: hook toggles, shapes captured, no overhead when disabled.

---

## 4.12 Masks & Integration Contracts
- [ ] Define expected masks from dataloader:
  - [ ] `attn_mask_bxt: bool` (True where token is valid)
  - [ ] Model ensures: invalid (PAD) positions produce logits but are **masked in loss** (M5); attention excludes PAD both ways.
- [ ] Test with crafted batch containing PAD to verify correct behavior.

---

## 4.13 Performance Sanity (Micro)
- [ ] Micro-benchmark:
  - [ ] Forward time on `(B=16, T=64, d_model=128, n_heads=4)`
  - [ ] Memory footprint check (no unexpected allocations)
- [ ] Ensure no `torch.cuda.synchronize()` except in benchmarks.

---

## 4.14 Documentation
- [ ] Docstrings for all public modules/classes/functions:
  - [ ] expected shapes, dtypes, masks
  - [ ] config fields and defaults
- [ ] Short section in README describing the model stack and masks.

---

## 4.15 Tests & Completion Criteria
- [ ] **Unit tests** pass for: embeddings, positional, attention, MLP, block, heads, wrapper.
- [ ] **Causality test** proves no leakage from future tokens.
- [ ] **Masking test** proves PAD is excluded in attention.
- [ ] **Weight tying** verified when enabled.
- [ ] **Determinism** with fixed seeds for forward pass.
- [ ] CI (or local `pytest`) green for CPU; optional CUDA smoke if available.

**Completion Criteria:**  
✅ `SingleLayerTransformer` produces correct-shaped logits, respects causal & PAD masks, supports optional weight tying, and exposes attention via instrumentation hooks.  
✅ All unit tests pass; simple micro-benchmark shows acceptable latency/memory.  
✅ Code documented; ready for integration into M5 training loop.
````
