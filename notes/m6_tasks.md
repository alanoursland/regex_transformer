# Milestone 6 — Evaluation Framework
**Goal:** Provide a unified, deterministic evaluation pipeline with core metrics, OOD tests, and lightweight visualizations saved per experiment.  
**Dependencies:** M1–M5 (FSM, data, datasets, model, training loop).

---

## 6.1 Package Structure & Scaffolding
- [ ] Create `src/eval/` with:
  - `metrics.py`, `ood.py`, `viz.py`, `aggregate.py`, `reports.py`, `tests/`
- [ ] Ensure eval code is **read-only** w.r.t. training artifacts (no mutation).
- [ ] Keep imports acyclic (eval depends on `model/`, `data/`, `fsm/`, not vice versa).

---

## 6.2 Core Metrics (`metrics.py`)
- [ ] Implement **per-position** metrics:
  - [ ] Next-token **accuracy** and **NLL** (masked on PAD).
  - [ ] State-class **accuracy** (Accept/Incomplete/Reject) with last-position mask.
  - [ ] (Optional) State-ID **accuracy** (if state head enabled).
- [ ] Implement **sequence-level** metrics:
  - [ ] Acceptance decision accuracy (full-string).
  - [ ] Per-sequence NLL / perplexity summary.
- [ ] Implement **length-binned** metrics:
  - [ ] Accuracy and NLL grouped by length buckets.
- [ ] Implement **coverage-aware** metrics:
  - [ ] Edge coverage accuracy: fraction of traversed `(q,a)` transitions predicted correctly.
- [ ] Mask handling consistent with dataset masks.
- [ ] Tests:
  - [ ] Small synthetic batch with **hand-computed** metrics.
  - [ ] Mask correctness (PAD/last position excluded appropriately).
  - [ ] Determinism with fixed inputs.

---

## 6.3 OOD Evaluation (`ood.py`)
- [ ] **Length-OOD**:
  - [ ] Build evaluation splits with `L > L_train_max`.
  - [ ] Report in-distribution vs OOD metrics and the **generalization gap**.
- [ ] **Edge-OOD**:
  - [ ] Respect experiment config for held-out transitions; evaluate exclusively on sequences that traverse those edges.
- [ ] **Regex-OOD** (optional Phase 1):
  - [ ] Evaluate on unseen-but-related regex families if provided.
- [ ] Tests:
  - [ ] Splits constructed deterministically from corpus / generator config.
  - [ ] Gaps computed correctly (ID vs OOD).

---

## 6.4 Visualizations (`viz.py`)
- [ ] **Accuracy vs length** line plots (ID and OOD).
- [ ] **Confusion matrices** for state-class predictions.
- [ ] **Attention maps** (optional): render a few batches if attention capture is available (M4 instrumentation).
- [ ] **Graph overlay** (optional): overlay average attention adjacency on FSM edges for quick visual sanity.
- [ ] Save to `src/results/<exp_id>/visualizations/*.png`.
- [ ] Tests:
  - [ ] Plot functions execute without error and create files.
  - [ ] Axis labels/titles present; shapes match data.

---

## 6.5 Reports & Artifact Writing (`reports.py`)
- [ ] Write `metrics.json` (final aggregates) with:
  - [ ] Overall metrics (token acc/NLL, class acc, acceptance acc).
  - [ ] Length-binned metrics.
  - [ ] OOD metrics + generalization gaps.
  - [ ] Coverage-aware summaries (edge/state coverage accuracy).
- [ ] Write `analysis.md` (human-readable summary with links to plots).
- [ ] Append results to `logs/history.jsonl` for longitudinal tracking.
- [ ] Tests:
  - [ ] JSON schema sanity; totals match per-split computations.
  - [ ] All files land under the experiment result directory.

---

## 6.6 Evaluation Orchestration
- [ ] Implement `evaluate(model, datasets, fsm, device)` wrapper:
  - [ ] Runs core metrics on `val` and `test`.
  - [ ] Optionally runs OOD suites per config.
  - [ ] Returns a dict ready for checkpoint “best” selection and report writing.
- [ ] Implement CLI helper (optional):
  - [ ] `python -m eval.run --exp_id <id> --checkpoint best` to re-evaluate any saved model deterministically.
- [ ] Tests:
  - [ ] Re-evaluating the same checkpoint produces identical metrics within tolerance.

---

## 6.7 Aggregation Across Experiments (`aggregate.py`)
- [ ] Load multiple `metrics.json` files and produce comparison tables:
  - [ ] Compare by model size, length ranges, regex families.
  - [ ] Export `aggregate.csv` / `aggregate.json` and summary plots.
- [ ] Tests:
  - [ ] Handles missing keys gracefully.
  - [ ] Sorting/filtering stable and deterministic.

---

## 6.8 Performance & Determinism
- [ ] Ensure evaluation uses `torch.no_grad()` and `model.eval()`.
- [ ] Pin device behavior; support CPU/GPU; disable AMP for determinism (unless explicitly allowed).
- [ ] Provide seed setting for any stochastic sampling in eval (normally none).
- [ ] Tests:
  - [ ] CPU and CUDA (if available) parity within numeric tolerance.

---

## 6.9 Documentation
- [ ] Docstrings for public APIs: `evaluate`, metric functions, OOD builders, and plotting helpers.
- [ ] README section: **“Evaluating an experiment”** with example commands and artifact paths.
- [ ] Notes on OOD split definitions and interpretation caveats.

---

## 6.10 Completion Criteria
✅ `evaluate()` produces deterministic `metrics.json` with core metrics and (if configured) OOD results.  
✅ Visualizations saved under `visualizations/` and referenced in `analysis.md`.  
✅ OOD gaps computed and reported; length-binned plots present.  
✅ Aggregation scripts compare multiple experiments cleanly.  
✅ Unit/integration tests pass for at least two regexes; re-evaluation of the same checkpoint matches prior results.
