```markdown
# Milestone 7 — Experiment Framework
**Goal:** Make experiments fully reproducible and one-command runnable (regex → FSM → data → model → evaluation → artifacts).  
**Dependencies:** M1–M6 (FSM, data gen, datasets, model, training, evaluation).

---

## 7.1 Package Structure & Scaffolding
- [ ] Create `src/experiments/` with:
  - [ ] `base.py` (experiment schema / protocol)
  - [ ] `registry.py` (optional name→factory map)
  - [ ] `configs/` (plain Python experiment definitions)
  - [ ] `run.py` (CLI entrypoint)
  - [ ] `tests/`
- [ ] Confirm imports are acyclic (experiments import modules, nothing imports experiments).

---

## 7.2 Experiment Schema (Code == Config)
- [ ] Define `Experiment` dataclass / protocol in `base.py`:
  - [ ] Fields: `name`, `regex_def`, `gen_cfg`, `model_cfg`, `train_cfg`, `seed`, `device`
  - [ ] `def build_id(self) -> str` (e.g., `"{name}_{shortgit}_{seed}"`)
- [ ] Provide helpers:
  - [ ] `def with_overrides(exp, **kwargs) -> Experiment` (copy/replace fields)
  - [ ] `def to_repr(exp) -> str` (for provenance snapshot)
- [ ] Tests:
  - [ ] Round-trip copy with overrides preserves immutables
  - [ ] `build_id` stable given same inputs

---

## 7.3 Experiment Definitions (configs/)
- [ ] Add minimal baseline configs:
  - [ ] `a_plus.py`, `ab_star.py`, `a_or_b_then_c.py`
- [ ] Include examples for:
  - [ ] Length-OOD setup (train ≤ L*, test > L*)
  - [ ] Edge-OOD holdout (specify excluded transitions)
- [ ] Tests:
  - [ ] Importing config returns a valid `Experiment`
  - [ ] `regex_def` compiles and seeds propagate

---

## 7.4 Run Orchestration (`run.py`)
- [ ] Implement `main(exp: Experiment)` high-level flow:
  1. Set global seed / determinism flags
  2. Compile regex → FSM; save FSM
  3. Generate data (or load cached); save telemetry
  4. Build datasets/dataloaders
  5. Build model from `model_cfg`
  6. Train using M5 loop; save checkpoints & logs
  7. Evaluate using M6; write `metrics.json` and plots
  8. Save provenance snapshot (exp repr, git commit, env)
- [ ] Support **resume** (if results dir exists, load last/best checkpoint)
- [ ] Return final metrics dict and results path
- [ ] Tests:
  - [ ] End-to-end run completes on tiny config
  - [ ] Resume continues deterministically

---

## 7.5 Results Directory Layout
- [ ] Create `src/results/<exp_id>/` with:
  - [ ] `config.py` (exact experiment definition snapshot)
  - [ ] `fsm.json` or `fsm.pkl`
  - [ ] `data/` (optional: saved corpus; always telemetry)
  - [ ] `checkpoints/` (`best.pt`, `last.pt`)
  - [ ] `metrics.json`, `logs/history.jsonl`, `env.json`
  - [ ] `visualizations/` (plots from evaluation)
- [ ] Atomic writes (tmp→rename) for JSON & checkpoints
- [ ] Tests:
  - [ ] All files present post-run
  - [ ] JSON schemas valid; paths reproducible

---

## 7.6 CLI Interface
- [ ] Implement `python -m experiments.run --exp <name> [--device cpu|cuda] [--seed N] [--resume] [--deterministic]`
- [ ] Allow overrides:
  - [ ] `--epochs`, `--batch-size`, `--lr`, `--L-min`, `--L-max`, `--p-accept` etc.
- [ ] Print results directory on completion; exit non-zero on failure
- [ ] Tests:
  - [ ] CLI end-to-end smoke on `a_plus`
  - [ ] Overrides reflected in saved `config.py`/`env.json`

---

## 7.7 Caching & Reuse (Optional but Useful)
- [ ] Data cache: if corpus with same `(regex, gen_cfg, seed)` exists, reuse (hash key)
- [ ] Checkpoint resume: if `best.pt` exists, skip training unless `--force-train`
- [ ] Tests:
  - [ ] Cache hit reduces generation time; artifacts identical

---

## 7.8 Provenance & Environment Capture
- [ ] Save:
  - [ ] `seed`, `torch/numpy/python` versions, CUDA info
  - [ ] Git commit hash and dirty flag
  - [ ] Exact experiment `repr` and effective overrides
- [ ] Tests:
  - [ ] Re-running with same seed/commit yields identical metrics within tolerance

---

## 7.9 Minimal Registry (Optional)
- [ ] `registry.py`: `register("a_plus", make_a_plus)`; `get(name) -> Experiment`
- [ ] Supports dynamic discovery for CLI `--exp`
- [ ] Tests:
  - [ ] Unknown name → clear error; known name builds experiment

---

## 7.10 Aggregation & Comparison Hooks
- [ ] Provide small script `experiments/compare.py` to load multiple `metrics.json` and print a table (model size, acc, OOD gaps)
- [ ] Export CSV/JSON summary to `src/results/aggregate/`
- [ ] Tests:
  - [ ] Handles missing metrics keys gracefully

---

## 7.11 Documentation
- [ ] `README` section: **“Running an Experiment”**
  - [ ] Example commands
  - [ ] Results directory tour
  - [ ] Reproducibility notes (seeds, deterministic flag)
- [ ] Docstrings on `Experiment` fields; describe expected types and invariants

---

## 7.12 Completion Criteria
✅ `python -m experiments.run --exp <name>` executes full pipeline and saves artifacts deterministically.  
✅ Config snapshots, seeds, environment, and commit hash recorded under results.  
✅ Resume and caching work as expected (when enabled).  
✅ At least two example configs produce valid metrics and plots.  
✅ Tests pass: end-to-end, resume, CLI overrides, and artifact checks.
```
