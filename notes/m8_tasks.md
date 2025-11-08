```markdown
# Milestone 8 — Validation & Testing
**Goal:** Establish right-sized, research-grade tests that guarantee correctness, determinism, and basic performance without over-engineering.  
**Dependencies:** M1–M7.

---

## 8.1 Test Matrix & Infra
- [ ] Define a **test matrix** doc listing what is covered (FSM, data gen, dataset, model, train, eval, experiments) and what is intentionally not (e.g., distributed training).
- [ ] Create `tests/` package-wide `conftest.py` with shared fixtures:
  - [ ] Toy regex fixtures: `a+`, `a*b*`, `(a|b)c`, small branching DFA
  - [ ] Tiny corpora (train/val/test) seeded and reproducible
  - [ ] Vocab fixture with `PAD`/`EOS`
- [ ] Add `pytest.ini` for warnings-as-errors (selective), slow-test markers, and random-order plugin (optional).
- [ ] Document how to run fast subset: `pytest -m "not slow"`.

---

## 8.2 FSM Tests (M1)
- [ ] **Transition correctness**: for each toy regex, verify `δ(q,a)` vs hand-computed truth.
- [ ] **Class semantics**: `classify(state)` matches Accept/Incomplete/Reject expectations.
- [ ] **Trace equivalence**: `fsm.trace(tokens)` agrees with `re.fullmatch` acceptance.
- [ ] **Serialization round-trip**: `save_fsm` → `load_fsm` preserves structure (hash/equals).
- [ ] **Property tests**: undefined transitions route to explicit reject; reject self-loops.

---

## 8.3 Data Generation Tests (M2)
- [ ] **Feasibility DP**: `canReach[q][t][class]` matches hand cases; unreachable targets fail fast.
- [ ] **Backward generator**: hits exact `(class, length)` for reachable targets.
- [ ] **Forward/beam**: finds sequences when backward can’t; prefers under-hit edges with quotas.
- [ ] **Reject subtypes**: each subtype produces genuine reject strings; ratios approximate config.
- [ ] **Distribution check**: empirical `(class, length)` histograms within tolerance of targets.
- [ ] **Coverage**: non-zero counts for most `(q,a)` in non-trivial FSMs.

---

## 8.4 Dataset / Collate / Loader Tests (M3)
- [ ] **On-the-fly labels**: `next_tokens`/`class_ids` align with `fsm.trace`.
- [ ] **Padding & masks**: PAD positions excluded from attention and losses.
- [ ] **Loader determinism**: same seed → identical batch order (including worker seeding).
- [ ] **Edge cases**: empty string (if allowed), length-1, max length; PAD not inside sequences.

---

## 8.5 Model Tests (M4)
- [ ] **Causality**: no influence from future tokens (numerical mask test).
- [ ] **Masking**: PAD tokens neither attend nor are attended.
- [ ] **Shapes & dtype**: end-to-end forward returns correctly shaped logits.
- [ ] **Weight tying**: verify shared params for next-token head if enabled.
- [ ] **Deterministic forward** with fixed seed/inputs.

---

## 8.6 Training Loop Tests (M5)
- [ ] **Loss masking**: no loss from PAD / invalid positions; component weights applied.
- [ ] **Checkpoint resume**: resume reproduces identical metrics/trajectory (within tol).
- [ ] **Early stopping**: triggers after patience on a crafted plateau run.
- [ ] **Micro overfit**: model overfits ≤32-sample dataset (loss → ~0, acc → ~1.0).

---

## 8.7 Evaluation Tests (M6)
- [ ] **Core metrics**: token acc, class acc, NLL match hand-computed small batch.
- [ ] **Length-binned** metrics: buckets sum to overall within rounding.
- [ ] **OOD splits**: length-OOD builder isolates `L > L_train_max`; gap computed correctly.
- [ ] **Viz smoke**: plot functions run headless and write files with expected shapes/labels.

---

## 8.8 Experiment Framework Tests (M7)
- [ ] **End-to-end run**: `a+` config completes → results folder contains expected artifacts.
- [ ] **Resume**: re-run with `--resume` continues deterministically.
- [ ] **Overrides**: CLI overrides reflected in saved `config.py`/`env.json`.

---

## 8.9 Reproducibility & Determinism Audit
- [ ] **Seed plumbing**: one seed controls Python/NumPy/Torch and data gen; verify recorded in `env.json`.
- [ ] **Repeatability**: two identical runs produce identical metrics (within FP tol) and same split assignments.
- [ ] **Deterministic flag**: when enabled, PyTorch deterministic algorithms set; document trade-offs.

---

## 8.10 Performance & Resource Sanity (Right-Sized)
- [ ] **Throughput smoke**: measure a forward+backward pass on `(B=16, T=64)`; assert under a lenient threshold (CI-safe).
- [ ] **Memory cap**: ensure no unexpected allocations; batch OOM handled with clear error.
- [ ] **No network calls**: verify tests run offline; artifacts local.

---

## 8.11 CI Wiring (Lightweight)
- [ ] GitHub Actions (or equivalent):
  - [ ] Python 3.10/3.11 job, CPU-only
  - [ ] Cache `pip` deps
  - [ ] Run `pytest -q`, collect coverage (optional)
  - [ ] Upload test artifacts on failure (logs, last metrics)
- [ ] Badge (optional) documented in README.

---

## 8.12 Test Utilities & Fixtures
- [ ] **Golden files**: small expected JSONs (metrics, telemetry) to diff against.
- [ ] **Randomness helpers**: fixture to freeze RNG and restore after test.
- [ ] **Tmp results dir**: fixture that auto-cleans to avoid residue.

---

## 8.13 Documentation
- [ ] `TESTING.md`:
  - [ ] How to run fast vs full test suites
  - [ ] What we test and what we intentionally do not
  - [ ] Reproducibility notes and deterministic flag usage
- [ ] Inline docstrings for non-obvious assertions and test rationale.

---

## 8.14 Completion Criteria
✅ All unit and integration tests pass locally and in CI (CPU).  
✅ End-to-end experiment (`a+`) runs deterministically; artifacts present and valid.  
✅ Micro overfit test passes; early stopping and resume verified.  
✅ Metrics/viz functions produce correct outputs; OOD split builder validated.  
✅ TESTING.md documents scope and commands; no flaky tests under default settings.
```
