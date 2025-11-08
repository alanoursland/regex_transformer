# Technical Design Document (TDD)

> Implements the Phase-1 feature design (regex→FSM→data→model→training→evaluation→experiments) with a lean, testable codebase that is easy to iterate on and instrument for Phase-2 interpretability. 

---

## 0. Scope & Non-Goals

* **In scope (Phase 1):** minimal FSM compiler, balanced synthetic data generation, single-layer transformer with dual heads, simple training loop, deterministic experiments, core metrics/plots. 
* **Out of scope (Phase 1):** multi-layer models, complex regex constructs (backrefs, lookahead/behind), large-scale or distributed training, extraction algorithms, dashboards. (See “Non-Features / Out of Scope”.)

---

## 1. Repository Layout

```
src/
  fsm/
    regex_def.py          # RegexDefinition dataclass & validation
    compile.py            # regex -> NFA -> DFA -> minimized DFA
    dfa.py                # FSM (Q, Σ, δ, q0, C) representation & ops
    serialize.py          # save/load (JSON) with versioning
    viz.py                # networkx/graphviz helpers (optional)
    tests/                # unit tests for FSM core
  data/
    generator.py          # targeted/backward/search hybrid generator
    quotas.py             # coverage counters & weights
    feasibility.py        # DP reachability booleans canReach[q][t][class]
    dataset.py            # PyTorch Dataset + on-the-fly tracing
    telemetry.py          # data histograms, coverage, subtype ratios
    tests/                # generation & split determinism tests
  model/
    config.py             # ModelConfig & autosizing helpers
    embedding.py          # (optional) factorized embeddings U@V
    transformer.py        # single-layer block (attn, MLP, resid, norm)
    heads.py              # next-token/class/state heads (weight tying opt.)
    instrumentation.py    # simple hooks for attn/activations
    tests/                # smoke tests (shapes, forward pass)
  train/
    loop.py               # train/eval loops, early stop, checkpointing
    losses.py             # multi-task loss combinator & masks
    optim.py              # optimizer, schedulers (if any)
  eval/
    metrics.py            # token acc, class acc, state acc, NLL, OOD gaps
    ood.py                # length/edge/regex-OOD splits
    viz.py                # plots (matplotlib) & attention maps
  experiments/
    base.py               # Experiment base class
    run.py                # CLI entrypoint to run experiments
    configs/              # small Python files defining experiments
  results/                # gitignored, per-experiment artifacts
  utils/
    hashing.py            # xxhash64 for split
    rng.py                # unified seed helper (python/numpy/torch)
    io.py                 # atomic writes, json helpers, paths
README.md
```

Principles: small modules, no cyclic imports, explicit data flow, local files only.

---

## 2. Data Types & Conventions

* **States**: zero-based `int` indices; explicit **reject** state included in `Q`.
* **Alphabet Σ**: ordered tuple/list of single-char strings; tokenizer maps to `int`.
* **Classes C**: map `state_id -> class_id` with a stable `List[str]` index.
* **Tokens**: `[int]` sequence over `Σ ∪ {EOS, PAD}`.
* **Seeds**: single `int` threaded through Python/NumPy/PyTorch and generator.
* **Tensors**: batch-first `(B, T, …)`; dtype `float32`; device passed explicitly.

---

## 3. FSM Compiler & Runtime

### 3.1 Interfaces

```python
# src/fsm/regex_def.py
@dataclass(frozen=True)
class RegexDefinition:
    alphabet: Tuple[str, ...]
    patterns: Tuple[Tuple[str, str], ...]  # (pattern, class_name)

# src/fsm/dfa.py
@dataclass
class FSM:
    states: int
    alphabet: Tuple[str, ...]
    start: int
    delta: Dict[Tuple[int, int], int]      # (state, token_id) -> next_state
    classes: Tuple[str, ...]               # ordered class names
    state_class: List[int]                 # len=states, maps state_id->class_id
    reject: int                            # explicit reject state id

    def step(self, state: int, token_id: int) -> int: ...
    def classify(self, state: int) -> int: ...
    def trace(self, tokens: Sequence[int]) -> List[int]: ...
```

### 3.2 Compilation

* **Regex → NFA** (Thompson).
* **NFA → DFA** (subset construction).
* **DFA minimization** (Hopcroft).
* **Class propagation** preserving accept/incomplete/reject semantics.
* **Explicit reject**: undefined transitions map to `reject`; `reject` self-loops.
* **Validation**: totality over Σ, reachability, class consistency. 

### 3.3 Serialization

```python
def save_fsm(fsm: FSM, path: Path) -> None:  ...
def load_fsm(path: Path) -> FSM:            ...
```

JSON with metadata (regex definition, state/transition counts, classes, version).

---

## 4. Data Generation & Dataloader

### 4.1 Goals (recap)

* Direct control over **(class, length)** distributions;
* Good **state/edge coverage**;
* **Negative** diversity;
* **Deterministic** and **auditable**;
* **Labels recomputed** by `fsm.trace()` in O(L). 

### 4.2 Feasibility DP (booleans)

```python
# src/data/feasibility.py
def can_reach_tables(fsm: FSM, L_max: int) -> np.ndarray:
    """
    returns canReach[state, t, class_id] ∈ {0,1}
    meaning: from 'state', some length-t suffix ends in 'class_id'
    """
```

* Small DP only for feasibility pruning (not full counting).

### 4.3 Quotas / Coverage

```python
# src/data/quotas.py
class QuotaManager:
    def __init__(self, fsm: FSM, target_edge_hits: int, target_state_hits: int):
        self.edge_hits = Counter()
        self.state_hits = Counter()
    def edge_weight(self, s: int, a: int) -> float:  # inverse sqrt freq
        ...
    def update_path(self, states: Sequence[int], tokens: Sequence[int]) -> None: ...
```

### 4.4 Generators

**Top-level interface**

```python
# src/data/generator.py
@dataclass
class GenConfig:
    L_min: int; L_max: int
    p_class: Dict[str, float]     # e.g. {"accept":.4,"incomplete":.2,"reject":.4}
    reject_mix: Dict[str, float]  # illegal, overrun, premature, wrong_alt
    attempts_per_sample: int = 8

def sample_target(cfg: GenConfig, rng: Random) -> Tuple[int, int]: ...
def generate_sample(fsm: FSM, cfg: GenConfig, quota: QuotaManager, rng: Random,
                    reach: np.ndarray) -> Optional[List[int]]:
    # try backward(); then forward(); then craft reject subtype
```

**Backward (fast path)**

```python
def backward(fsm: FSM, class_id: int, L: int, quota: QuotaManager,
             reach: np.ndarray, rng: Random) -> Optional[List[int]]:
    # choose terminal state in class; backwalk with predecessor index
```

**Forward (best-first / small beam)**

```python
def forward(fsm: FSM, class_id: int, L: int, quota: QuotaManager,
            reach: np.ndarray, rng: Random, beam: int = 16) -> Optional[List[int]]:
    # keep candidates (state, seq, g) with feasibility; score by -coverage_bonus
```

**Reject crafting**

* **illegal step**: follow valid path then choose undefined `(q,a)` once.
* **overrun**: extend beyond allowed repetition into reject sink.
* **premature**: stop where accept remains reachable (if task labels end-of-string).
* **wrong_alt**: drive into late incorrect branch.

### 4.5 Splits & Telemetry

```python
# src/utils/hashing.py
def split_of(tokens: Sequence[int]) -> str:  # "train"|"val"|"test"
    h = xxhash64(bytes(tokens)).intdigest() % 1000
    return "train" if h < 900 else "val" if h < 950 else "test"

# src/data/telemetry.py
@dataclass
class DataReport:
    length_hist: Dict[int, int]
    class_hist: Dict[str, int]
    state_cov: Dict[int, int]
    edge_cov: Dict[Tuple[int,int], int]
    reject_subtypes: Dict[str, int]
```

### 4.6 PyTorch Dataset (on-the-fly labels)

```python
# src/data/dataset.py
class FsmDataset(Dataset):
    def __init__(self, fsm: FSM, samples: Sequence[List[int]], split: str, eos_id: int):
        ...
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        states = self.fsm.trace(tokens)
        next_tokens = tokens[1:] + [eos_id]
        class_ids = [self.fsm.classify(s) for s in states[:-1]]
        return {
            "tokens": torch.tensor(tokens),
            "next":   torch.tensor(next_tokens),
            "class":  torch.tensor(class_ids),
        }
```

---

## 5. Model

### 5.1 Config

```python
# src/model/config.py
@dataclass
class ModelConfig:
    vocab_size: int
    num_classes: int
    num_states: int
    d_model: int
    n_heads: int
    d_mlp: int
    max_seq_len: int
    lambda_next: float = 1.0
    lambda_class: float = 1.0
    lambda_state: float = 0.0
    dropout: float = 0.0
    factorized_embedding: bool = False
    factor_rank: int = 0
    l1_u: float = 0.0
```

### 5.2 Modules

```python
# src/model/embedding.py
class FactorizedEmbedding(nn.Module):
    # E = U @ V with L1 on U (optional)
    ...

# src/model/transformer.py
class SingleBlock(nn.Module):
    def __init__(self, cfg: ModelConfig): ...
    def forward(self, x): ...  # causal self-attn + MLP (pre-norm)

# src/model/heads.py
class Heads(nn.Module):
    def __init__(self, cfg: ModelConfig, tie_weights: bool = True): ...
    def forward(self, h):
        return logits_next, logits_class, logits_state
```

* **Causal mask** always on.
* **Weight tying** optional (next-token ↔ token embedding).
* **Instrumentation hooks** to capture attention weights and hidden states for chosen batches.

---

## 6. Training

### 6.1 Losses & Optimizer

```python
# src/train/losses.py
def multitask_loss(logits, targets, mask, cfg: ModelConfig):
    ln = F.cross_entropy(logits.next[mask], targets.next[mask])
    lc = F.cross_entropy(logits.cls[mask], targets.cls[mask])
    ls = F.cross_entropy(logits.sta[mask], targets.sta[mask])
    return cfg.lambda_next*ln + cfg.lambda_class*lc + cfg.lambda_state*ls
```

* **Optimizer**: AdamW; optional LR warmup/decay (simple linear).
* **Gradient clipping**: optional (e.g., 1.0).
* **Early stop** on validation accuracy.

### 6.2 Loop & Checkpointing

```python
# src/train/loop.py
def train_one_experiment(exp) -> Dict[str, Any]:
    # seed, build FSM, data, model, optimizer
    # train epochs -> eval -> save artifacts
```

Artifacts:

* `checkpoints/epoch_{k}.pt`
* `metrics.json` (per split)
* `telemetry.json` (data report)
* `visualizations/*.png` (optional)

---

## 7. Evaluation

* **Token accuracy**, **class accuracy**, **(optional) state accuracy**, **NLL**.
* **Length-OOD**: test on lengths > train max.
* **Edge-OOD**: hold out edges during train, test on them.
* **Regex-OOD** (optional in Phase 1).
* **Plots**: acc vs length; confusion matrices; simple attention heatmaps.
  (See Section 7 in design; implemented here as `eval/metrics.py` & `eval/viz.py`.) 

---

## 8. Experiments

### 8.1 Definition (Code, not YAML)

```python
# src/experiments/base.py
class Experiment(Protocol):
    name: str
    regex: RegexDefinition
    gen: GenConfig
    model: ModelConfig
    train: Dict[str, Any]  # epochs, batch_size, lr, etc.
    seed: int
```

### 8.2 Runner

```python
# src/experiments/run.py
def main(exp: Experiment):
    # compile regex -> fsm
    # generate datasets (+telemetry)
    # build/train/eval model
    # write artifacts under src/results/<exp_id>/
```

`<exp_id> = f"{exp.name}_{timestamp}_{shortgit}"`

---

## 9. Testing (Right-Sized)

* **FSM**: transitions, classify, trace, compile/minimize round-trip; property-based checks on small patterns.
* **Generator**: deterministic given seed; class/length histograms match target; edge/state coverage non-zero.
* **Dataset**: alignment (tokens/next/class) and mask shapes.
* **Model**: forward shape/sanity; tiny overfit (10 samples) converges.
* **End-to-end**: `a+` pipeline runs and saves artifacts.

---

## 10. Reproducibility

* Single `seed` → `random`, `numpy`, `torch` (`torch.use_deterministic_algorithms(True)` when feasible).
* Save: seed, `torch.__version__`, CUDA info, and Git commit in results.
* Hash-based stable splits by raw string; no leakage across splits.

---

## 11. Risks & Mitigations

* **Skewed data** (length/class): *Mitigation*: targeted (class, L) + feasibility DP.
* **Poor coverage** of edges: *Mitigation*: inverse-frequency coverage bonus + quotas.
* **Over-engineering**: *Mitigation*: single-layer, minimal deps, code==config.
* **Training instability on tiny models**: *Mitigation*: conservative LR, pre-norm, small batch, gradient clip.

---

## 12. Implementation Order (Milestones)

1. **FSM core**: compile/minimize/trace + unit tests.
2. **Feasibility DP & backward generator** (+ telemetry).
3. **Dataset & splits** (on-the-fly labels).
4. **Model (single block) & heads** (+ smoke tests).
5. **Train loop** (losses, checkpoints, metrics).
6. **Evaluation plots** (acc vs length; confusion matrices).
7. **Experiment runner** & two baseline experiments (`a*`, `a*b*`).
8. **Coverage quotas & reject subtypes** (data improvements).
9. **Instrument hooks** (attention capture for samples).

Stop here for Phase 1 baselines; iterate only if needed to meet success criteria. 

---

## 13. Success Criteria (Phase 1)

* ≥95% token **and** class accuracy on held-out test for simple regexes;
* Clean end-to-end runs with deterministic artifacts;
* Telemetry verifies requested distributions & coverage;
* Tiny overfit sanity passes.

---

### Appendix: Minimal Public APIs (summary)

```python
# FSM
fsm = compile_regex(regex_def)         # -> FSM
states = fsm.trace(tokens)             # List[int]
save_fsm(fsm, path); fsm = load_fsm(path)

# Data
reach = can_reach_tables(fsm, L_max)
sample = generate_sample(fsm, gen_cfg, quota, rng, reach)
dataset = FsmDataset(fsm, samples, split="train", eos_id=vocab.eos)

# Model
model = build_model(cfg)
logits = model(batch_tokens)

# Train/Eval
train_one_experiment(exp)
metrics = evaluate(model, datasets, fsm)
```

---

**This TDD is intentionally small, explicit, and testable.** It implements the feature design faithfully while keeping engineering surface minimal so Phase-2 interpretability can plug in cleanly later. 
