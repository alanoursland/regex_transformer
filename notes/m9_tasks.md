# Milestone 9 — Research Readiness
**Goal:** Final polish for Phase 1 so experiments are easy to run, inspect, and share. Add lightweight instrumentation, examples, docs, and baseline results—without growing scope.  
**Dependencies:** M1–M8.

---

## 9.1 Instrumentation (Lightweight Only)
- [ ] Add simple hooks to capture:
  - [ ] Per-head attention maps (subset of batches via flag).
  - [ ] Hidden states pre/post block for one batch per epoch.
- [ ] Toggle via config (`capture_attn_every`, `capture_hidden_every`); default **off**.
- [ ] Save small `.pt`/`.npz` blobs under `visualizations/raw/` with an index JSON.
- [ ] Provide convenience loaders to render a quick heatmap from a saved blob.
- [ ] Tests: hooks disabled → zero overhead; enabled → shapes correct; files written.

---

## 9.2 Baseline Experiments (Reproducible)
- [ ] Ship 3–4 canonical configs in `experiments/configs/`:
  - [ ] `a_plus.py` (sanity)
  - [ ] `a_star_b_star.py` (two-clause)
  - [ ] `(a|b)*c.py` (branch → terminal)
  - [ ] Optional: `balanced_alt.py` (confusable alternation)
- [ ] For each baseline:
  - [ ] Generate datasets, train once, evaluate.
  - [ ] Commit **metrics only** (JSON) under `results/baselines/<name>/metrics.json` (git-checked).
  - [ ] Document expected ranges (acc/NLL) in README.
- [ ] Script `make_baselines.py` to rebuild baselines deterministically.

---

## 9.3 Example Workflows
- [ ] `examples/01_run_experiment.md` — from config to results folder.
- [ ] `examples/02_make_dataset.md` — compile FSM, generate corpus, inspect telemetry.
- [ ] `examples/03_eval_and_plots.md` — re-evaluate a checkpoint and render plots.
- [ ] Keep examples *short*; link to deeper docs rather than duplicating.

---

## 9.4 Minimal Notebooks (Optional)
- [ ] `notebooks/inspect_attention.ipynb` — load one saved attention blob and plot.
- [ ] `notebooks/length_generalization.ipynb` — load `metrics.json`, draw acc vs length.
- [ ] No training in notebooks; read-only analysis. Save generated figures back to the experiment folder.

---

## 9.5 Documentation Pass
- [ ] Update top-level `README.md`:
  - [ ] Project overview & scope (Phase 1).
  - [ ] Quickstart (single command to run an experiment).
  - [ ] Results directory tour with file purposes.
  - [ ] Reproducibility policy (seeds, deterministic flag).
  - [ ] Non-features / out-of-scope reminder.
- [ ] `docs/ARCHITECTURE.md` — one-pager diagram of data/model/eval flow.
- [ ] `docs/DATA.md` — how generation works, class/length balance, telemetry fields.
- [ ] Trim any over-long docs; keep everything pragmatic and current.

---

## 9.6 Results Hygiene & Curation
- [ ] Ensure every experiment writes:
  - [ ] `metrics.json`, `telemetry.json`, `env.json`, `config.py`, `checkpoints/`, `visualizations/`.
- [ ] Add `results/clean.py` to prune large raw tensors (keep last N artifacts).
- [ ] Add `results/compare.py` to print a small table for multiple runs.
- [ ] Verify `gitignore` excludes large artifacts; keep only baseline metrics checked in.

---

## 9.7 Release Checklist (Internal)
- [ ] `python -m experiments.run --exp a_plus` completes on CPU in < few minutes.
- [ ] Two reruns with same seed produce identical metrics (within FP tol).
- [ ] Micro overfit script passes (from M8).
- [ ] OOD length split runs and reports a gap (even if small).
- [ ] Lint/type check (optional `ruff/black/mypy` minimal settings) — **no blocking bikeshedding**.

---

## 9.8 Packaging & Version Stamps (Lightweight)
- [ ] Add `__version__` in a single `src/version.py`; embed commit hash in `env.json`.
- [ ] Simple `setup.cfg`/`pyproject.toml` for local installs (no publishing).
- [ ] Script `tools/print_env.py` to dump env/versions for bug reports.

---

## 9.9 Guardrails Against Scope Creep
- [ ] Confirm flags default to **Phase 1 scope** (single layer, small data).
- [ ] Warn (don’t implement) if user asks for:
  - [ ] Multi-layer models
  - [ ] Complex regex (backrefs/lookahead)
  - [ ] Distributed training
- [ ] Add clear error messages pointing to “Non-Features / Out of Scope”.

---

## 9.10 Final Code Review & Cleanup
- [ ] Pass over all public APIs: docstrings, argument names, invariants.
- [ ] Remove dead code, TODOs older than this milestone, and experimental branches.
- [ ] Ensure tests reference only current APIs; delete legacy fixtures.
- [ ] Freeze baseline configs and record their seeds in `docs/BASELINES.md`.

---

## 9.11 Completion Criteria
✅ Baseline configs run out-of-the-box, produce deterministic metrics and plots.  
✅ Lightweight instrumentation available and off by default.  
✅ Concise, accurate docs and examples guide a new user from zero to results.  
✅ Results hygiene tools present; repository stays small and reproducible.  
✅ No scope creep beyond Phase 1; clear errors for out-of-scope requests.
