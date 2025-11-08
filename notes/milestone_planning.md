# Milestone Planning Document

This plan defines progressive implementation milestones for the Phase 1 system, mapping concrete features to each development stage.  
Each milestone represents an independently runnable, validated subset of the full architecture described in the Feature Design and Technical Design documents.  
There are no fixed timelines—milestones are ordered by dependency and complexity.

---

## **Milestone 1 — FSM Core**

**Goal:** Implement and verify the formal foundations (regex → FSM).  

**Features**
- Regex parsing and validation (`RegexDefinition`)  
- NFA → DFA → minimized DFA compilation  
- Explicit reject-state creation and total transition table  
- FSM dataclass with `step`, `classify`, and `trace` methods  
- Serialization (save/load JSON)  
- Unit tests:
  - Transition correctness  
  - Class assignment accuracy  
  - Regex equivalence for reference patterns  

**Outcome:**  
Deterministic FSMs can be compiled, traced, and saved; correctness validated through tests.

---

## **Milestone 2 — Data Generation Core**

**Goal:** Generate balanced, auditable datasets directly from FSMs.  

**Features**
- Feasibility DP (`canReach[q][t][class]`) for pruning unreachable samples  
- Backward generator for exact (class, length) targets  
- Quota manager for coverage bonuses and statistics  
- Hash-based train/val/test split (deterministic)  
- Telemetry collection (length/class histograms, edge/state coverage)  
- Reject subtype crafting (illegal, overrun, premature, wrong alt)  

**Outcome:**  
Synthetic sequences with balanced distributions and known FSM trajectories.  
Datasets reproducible and instrumented for coverage analysis.

---

## **Milestone 3 — Dataset and Labeling**

**Goal:** Integrate FSM-based labeling and create a PyTorch-ready dataset.  

**Features**
- `FsmDataset` with:
  - Tokens, next-token targets, and per-position class labels  
  - On-the-fly label regeneration (`fsm.trace`)  
- Dataset sanity tests (alignment, reproducibility)  
- Configurable dataloader wrapper for batching  

**Outcome:**  
Model-ready dataset objects producing complete, deterministic batches.

---

## **Milestone 4 — Model Foundation**

**Goal:** Implement the baseline transformer block and task heads.  

**Features**
- `ModelConfig` with all structural parameters  
- Single-block transformer (`SingleBlock`) with causal mask  
- Factorized embeddings (`U@V`, optional L1 regularization)  
- Multi-head output: next-token, class, and state predictions  
- Weight tying option (input ↔ output embeddings)  
- Forward-pass smoke tests (shape, dtype, loss sanity)  

**Outcome:**  
Trainable forward graph verified; ready for integration with data pipeline.

---

## **Milestone 5 — Training Loop**

**Goal:** Enable end-to-end training and checkpointing.  

**Features**
- Multi-task loss function (`λ_next`, `λ_class`, `λ_state`)  
- Optimizer and scheduler setup (AdamW, optional LR warmup)  
- Training loop with gradient clipping and early stopping  
- Checkpoint saving and resume logic  
- Metric logging (loss curves, accuracy, validation results)  

**Outcome:**  
Experiments can train models deterministically and produce checkpoints and metrics.

---

## **Milestone 6 — Evaluation Framework**

**Goal:** Provide a unified evaluation and visualization pipeline.  

**Features**
- Core metrics: token accuracy, class accuracy, acceptance accuracy, NLL  
- OOD tests (length-OOD, edge-OOD, regex-OOD)  
- Visualization utilities:
  - Accuracy vs. length curves  
  - Confusion matrices  
  - Attention heatmaps  
- JSON metrics output and standardized plotting scripts  

**Outcome:**  
Quantitative and visual diagnostics reproducibly computed for each experiment.

---

## **Milestone 7 — Experiment Framework**

**Goal:** Automate reproducible experiment execution and result organization.  

**Features**
- Python-defined experiment objects (code == config)  
- Experiment runner (regex → FSM → data → model → evaluation)  
- Results directory structure (`src/results/<experiment_id>/`)  
- Automatic saving of config, seeds, FSM, telemetry, metrics, and plots  
- Logging of Git commit and environment metadata  

**Outcome:**  
Single command runs an entire experiment and produces a complete, isolated artifact folder.

---

## **Milestone 8 — Validation and Testing**

**Goal:** Validate system correctness and reproducibility end-to-end.  

**Features**
- Unit tests for FSM, data generation, and metrics  
- Smoke test for model overfitting on tiny dataset  
- End-to-end test using simple regex (`a+`)  
- Reproducibility audit (identical artifacts under same seed)  
- Coverage and balance verification  

**Outcome:**  
The system passes deterministic correctness tests; baseline experiments reproducibly succeed.

---

## **Milestone 9 — Research Readiness**

**Goal:** Final integration and documentation for Phase 1 research use.  

**Features**
- Lightweight hooks for attention and activation capture  
- README and usage documentation  
- Example experiments (`a*`, `a*b*`, `(a|b)*c`)  
- Baseline results and analysis notebooks  
- Code review for adherence to guidelines and scope  

**Outcome:**  
Phase 1 framework complete: fully runnable, interpretable, and reproducible, ready for Phase 2 interpretability studies.

---
