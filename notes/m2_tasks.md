# Milestone 2 — Data Generation Core
**Goal:** Generate balanced, auditable datasets directly from FSMs with deterministic splitting and telemetry.  
**Dependencies:** Milestone 1 (FSM compile/trace/serialize).

---

## 2.1 Package & Scaffolding
- [ ] Create `src/data/` with modules:
  - `feasibility.py`, `generator.py`, `quotas.py`, `dataset.py`, `telemetry.py`, `split.py`, `tests/`
- [ ] Add minimal `__init__.py` and placeholder docstrings.
- [ ] Ensure imports are acyclic (`data` depends on `fsm`, not vice versa).

---

## 2.2 Feasibility DP (Booleans, not counting)
- [ ] Implement `can_reach_tables(fsm: FSM, L_max: int) -> np.ndarray` returning `canReach[state, t, class_id] ∈ {0,1}`.
- [ ] Validate correctness on toy FSMs (e.g., single-chain, small branching).
- [ ] Add helper `feasible(state:int, t:int, class_id:int) -> bool`.
- [ ] Tests:
  - [ ] Terminal class feasibility matches hand-computed truth.
  - [ ] Monotonicity sanity (if infeasible at `t`, then infeasible at `t-1` for same end-class unless staying put is allowed).

---

## 2.3 Coverage Quotas
- [ ] Implement `QuotaManager` in `quotas.py`:
  - [ ] Track per-edge `hits[(state, token_id)]` and per-state `hits_state[state]`.
  - [ ] Provide `edge_weight(state, token_id)` with inverse-sqrt frequency bonus.
  - [ ] `update_path(states, tokens)` to bump counts post-sample.
- [ ] Configurable targets (min counts) for edges/states; expose simple summary API.
- [ ] Tests:
  - [ ] Weight decreases as an edge is used more.
  - [ ] `update_path` correctly updates both state and edge counts.

---

## 2.4 Target Specification
- [ ] Define `GenConfig` in `generator.py`:
  - [ ] `L_min, L_max`, `p_class` (A/I/R), `reject_mix` (illegal/overrun/premature/wrong_alt), `attempts_per_sample`.
- [ ] Implement `sample_target(cfg, rng) -> (class_id, L)` with validation that at least one state is feasible for `(L, class)`.

---

## 2.5 Backward Generator (Fast Path)
- [ ] Implement `backward(fsm, class_id, L, quota, reach, rng) -> Optional[List[int]]`:
  - [ ] Choose terminal state from requested class (weighted by under-covered predecessors).
  - [ ] Back-walk `L` steps using predecessor index; enforce feasibility at each step.
  - [ ] Reverse collected tokens to emit forward string.
- [ ] Precompute `predecessors[q] = [(p, a)]` for efficiency.
- [ ] Tests:
  - [ ] Exact hit of `(class, L)` on reachable cases.
  - [ ] Distribution sanity: with uniform tie-breaking, length histogram is uniform when sampling `L` uniformly.

---

## 2.6 Forward Generator (Best-First / Small Beam)
- [ ] Implement `forward(fsm, class_id, L, quota, reach, rng, beam=16) -> Optional[List[int]]`:
  - [ ] Maintain candidates `(state, seq, score)` up to depth `L`.
  - [ ] Prune infeasible children using `reach`.
  - [ ] Score = `-coverage_bonus` (+ optional small random tie-break).
  - [ ] Return any sequence satisfying `(class, L)` (prefer lower score).
- [ ] Tests:
  - [ ] Finds valid sequences when backward fails.
  - [ ] Uses under-covered edges more often than a uniform walk.

---

## 2.7 Reject Subtype Crafting
- [ ] Implement subtype emitters:
  - [ ] `illegal_step`: follow a feasible prefix, then pick undefined `(q,a)` exactly once; stop.
  - [ ] `overrun`: extend beyond last acceptable repetition into reject sink.
  - [ ] `premature`: stop at prefix where accept remains reachable (for whole-string acceptance tasks).
  - [ ] `wrong_alt`: force late branch contrary to ground-truth disambiguation.
- [ ] Integrate subtype selection via `cfg.reject_mix`.
- [ ] Tests:
  - [ ] Each subtype actually produces reject-class strings.
  - [ ] Mixture adheres (approx.) to configured ratios over many samples.

---

## 2.8 Top-Level Sampling Orchestrator
- [ ] Implement `generate_sample(...)`:
  - [ ] Draw target `(class, L)`.
  - [ ] Try `backward`; on failure, try `forward`; for Reject, craft subtype if needed.
  - [ ] On success, call `quota.update_path(states, tokens)`.
  - [ ] If no sample after `attempts_per_sample`, resample target and log a counter.
- [ ] Implement `generate_corpus(fsm, cfg, n_samples, seed)` that:
  - [ ] Builds `reach`, `QuotaManager`, and RNG.
  - [ ] Produces a list of token sequences and a `DataReport`.
- [ ] Tests:
  - [ ] Corpus class/length histograms close to targets (tolerance).
  - [ ] Non-zero coverage on most edges for non-trivial FSMs.

---

## 2.9 Deterministic Split
- [ ] Create `split.py` with `split_of(tokens) -> {"train","val","test"}` using `xxhash64(bytes(tokens)) % 1000`.
- [ ] Provide helper to stratify counts per split while preserving determinism.
- [ ] Tests:
  - [ ] Same string always maps to same split.
  - [ ] No string appears in multiple splits.

---

## 2.10 Telemetry & Reporting
- [ ] Implement `DataReport` in `telemetry.py` capturing:
  - [ ] Length histogram, class histogram.
  - [ ] Per-state and per-edge coverage counts.
  - [ ] Reject subtype counts.
  - [ ] Failed-attempts/retry rate.
- [ ] Add `to_json()` and pretty `print_report()`.
- [ ] Tests:
  - [ ] JSON round-trip.
  - [ ] Non-empty, internally consistent totals.

---

## 2.11 PyTorch Dataset Wrapper
- [ ] Implement `FsmDataset` in `dataset.py`:
  - [ ] Accept `(fsm, samples, split, eos_id)`.
  - [ ] Compute `states = fsm.trace(tokens)`, `next_tokens`, and `class_ids` on the fly.
  - [ ] Return dict of tensors; keep batch-first compatibility.
- [ ] Tests:
  - [ ] Alignment sanity (`len(next)==len(tokens)` and class labels match pre-next state).
  - [ ] Deterministic outputs for fixed sample.

---

## 2.12 CLI / Script for Data Export (Optional)
- [ ] Simple script to compile FSM, generate corpus, and write:
  - [ ] `samples.jsonl` (tokens as ints), `telemetry.json`, and `fsm.pkl/json`.
- [ ] Useful for inspection without training.

---

## 2.13 Documentation
- [ ] Docstrings for public functions and configs (`GenConfig`, `QuotaManager`, `generate_corpus`).
- [ ] Short `README` section: how to generate a dataset from an FSM; expected artifacts; reproducibility note.

---

## 2.14 Completion Criteria
✅ `generate_corpus` produces balanced (class, length) datasets deterministically from an FSM.  
✅ Coverage quotas and reject subtypes are active and reflected in `telemetry.json`.  
✅ Deterministic train/val/test split via string hash; no leakage.  
✅ `FsmDataset` returns correctly aligned tensors with on-the-fly labels.  
✅ Unit and integration tests pass for at least two regexes (e.g., `a+`, `a*b*`), including a branching case.  
✅ (Optional) Export script writes inspectable artifacts.
