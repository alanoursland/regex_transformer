# Milestone 5 — Training Loop
**Goal:** Implement a deterministic, lightweight training & evaluation loop with multi-task losses, checkpointing, and basic logging.  
**Dependencies:** M1 (FSM), M2 (data gen), M3 (dataset/masks), M4 (model).

---

## 5.1 Package Structure & Scaffolding
- [ ] Create `src/train/` with:
  - `loop.py`, `losses.py`, `optim.py`, `scheduler.py` (optional), `checkpoint.py`, `logging.py`, `hooks.py`, `tests/`
- [ ] Ensure modules import only stdlib/torch/utils; no heavy frameworks.

---

## 5.2 Losses (`losses.py`)
- [ ] Implement **multi-task loss**:
  - [ ] `next-token` loss: `CrossEntropy(logits_next, target_next)` masked on PAD.
  - [ ] `class` loss: `CrossEntropy(logits_class, target_class)` (mask last position).
  - [ ] `state` loss (optional): `CrossEntropy(logits_state, target_state)` (masked like class).
  - [ ] Weighted sum with `λ_next`, `λ_class`, `λ_state`.
- [ ] Add optional **L1 penalty** for factorized embeddings (`l1_u`) exposed by model.
- [ ] Return dict of component losses + total.
- [ ] Tests:
  - [ ] Masking correctness (no loss on PAD/invalid positions).
  - [ ] Weights properly scale component losses.
  - [ ] Deterministic outputs with fixed tensors.

---

## 5.3 Optimizer & (Optional) Scheduler (`optim.py`, `scheduler.py`)
- [ ] Implement `build_optimizer(model, lr, weight_decay, betas)` → AdamW.
- [ ] Optional LR schedule: linear warmup → cosine/linear decay (configurable).
- [ ] Gradient clipping utility `clip_grad_norm_(model.parameters(), max_norm)`.
- [ ] Tests:
  - [ ] Optimizer state dict save/load round-trip.
  - [ ] Scheduler step count and warmup boundaries correct.

---

## 5.4 Checkpointing (`checkpoint.py`)
- [ ] Implement:
  - [ ] `save_ckpt(path, model, optimizer, epoch, step, metrics, seed, extra:dict)`
  - [ ] `load_ckpt(path, model, optimizer=None)` returning meta dict
- [ ] Atomic write (tmp + rename); device-agnostic (`map_location="cpu"`).
- [ ] Track best model by validation metric (e.g., token accuracy).
- [ ] Tests:
  - [ ] Resume training produces identical subsequent metrics with same seed.
  - [ ] Best-checkpoint selection logic correct.

---

## 5.5 Logging (Lightweight) (`logging.py`)
- [ ] Implement minimal console logging + JSONL logging:
  - [ ] Per-epoch: train/val losses, accuracies, LR, grad-norm (optional).
  - [ ] Write `metrics.json` (final summary) and `logs/history.jsonl` (per step/epoch).
- [ ] No remote trackers; all local files under results folder.
- [ ] Tests: JSON schema sanity; paths created; append-only behavior.

---

## 5.6 Hooks & Evaluation Stubs (`hooks.py`)
- [ ] Define simple callback hooks:
  - [ ] `on_epoch_start/end`, `on_step_end`, `on_validation_end`.
  - [ ] Optional capture of a batch’s attention maps/hidden states for inspection.
- [ ] Provide `evaluate(model, dataloaders, masks_cfg)` wrapper that:
  - [ ] Computes token/class accuracies and NLL on val/test.
  - [ ] Returns dict compatible with logging & checkpoint “best” logic.
- [ ] Tests: Hook execution order; evaluation determinism.

---

## 5.7 Training Loop (`loop.py`)
- [ ] Implement `train_one_experiment(exp)`:
  1. **Seeding**: set global seed (Python/NumPy/Torch, determinism flags).
  2. **FSM/Data**: compile FSM; generate/load corpus; build vocab/datasets/loaders.
  3. **Model**: build model from `ModelConfig`; send to device.
  4. **Optim**: build optimizer (+ scheduler optional).
  5. **Train**:
     - Iterate epochs:
       - For each batch: forward → compute multi-task loss → backward → clip grads → optimizer step → scheduler step.
       - Track running averages; log per N steps.
     - End of epoch: run validation via `evaluate`; update best checkpoint.
     - **Early stopping** on patience over validation metric.
  6. **Test**: evaluate final and best checkpoint on test set (record both).
  7. **Artifacts**: save `metrics.json`, checkpoints, logs, and environment info.
- [ ] Device handling: accept `device` arg; support CPU/GPU transparently.
- [ ] Mixed precision (optional, off by default) with `torch.cuda.amp.autocast` & GradScaler.
- [ ] Tests:
  - [ ] One-epoch smoke on micro dataset runs end-to-end.
  - [ ] Early stopping triggers when metric stalls.
  - [ ] Best-checkpoint differs from last when appropriate.

---

## 5.8 Metrics Computation (Integration)
- [ ] Integrate with `eval/metrics.py` for:
  - [ ] Token accuracy, class accuracy, (optional) state accuracy, NLL.
  - [ ] Length-binned accuracy (for quick OOD signal).
- [ ] Ensure masking aligns with dataset’s `loss_mask_bxt`.
- [ ] Tests: Known small batch with hand-computed metrics.

---

## 5.9 Results Directory & Provenance
- [ ] Write artifacts to `src/results/<experiment_id>/`:
  - [ ] `config.py` (experiment definition snapshot)
  - [ ] `fsm.pkl/json`
  - [ ] `checkpoints/last.pt`, `checkpoints/best.pt`
  - [ ] `metrics.json`, `logs/history.jsonl`
  - [ ] `env.json` (seed, torch/cuda versions, git commit)
- [ ] Verify idempotency: re-running same experiment overwrites or versions correctly.
- [ ] Tests: file existence; JSON validity; env info captured.

---

## 5.10 Determinism & Reproducibility Audit
- [ ] With fixed seed:
  - [ ] Same initial weights (save their hash).
  - [ ] Same batch order and masks.
  - [ ] Identical losses/metrics up to floating-point tolerance.
- [ ] Provide a `--deterministic` flag to toggle PyTorch deterministic algorithms; document trade-offs.
- [ ] Tests: run twice → identical artifacts (except timestamps).

---

## 5.11 CLI Entrypoint (Optional)
- [ ] `python -m experiments.run --exp <name> --device cpu/cuda --epochs N`
- [ ] Flags: batch size, lr, patience, deterministic, save-every, amp.
- [ ] Print location of results directory on completion.

---

## 5.12 Documentation
- [ ] Docstrings for `train_one_experiment`, losses, optimizer builders, checkpoint helpers.
- [ ] README section: “How to train an experiment end-to-end” with example command.
- [ ] Notes on determinism, AMP, and early stopping.

---

## 5.13 Completion Criteria
✅ End-to-end training completes on at least two regex experiments; produces `best.pt`, `last.pt`, `metrics.json`, and logs.  
✅ Loss masking correct; component losses/weights validated.  
✅ Deterministic runs reproduce metrics within tolerance; resume-from-checkpoint works.  
✅ Early stopping & best-checkpoint selection verified on validation metric.  
✅ Minimal logging present; no external services; artifacts written under results directory.
