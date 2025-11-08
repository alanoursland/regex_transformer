# Milestone 3 — Dataset & Labeling
**Goal:** Provide robust, PyTorch-ready datasets and loaders with **on-the-fly FSM labeling**, correct padding/masking, deterministic batching, and sanity utilities.  
**Dependencies:** Milestone 1 (FSM core), Milestone 2 (generation & splits).

---

## 3.1 Package Structure & Scaffolding
- [ ] Confirm/extend `src/data/` with:
  - `tokenizer.py` (vocab & special tokens)
  - `dataset.py` (datasets, caching hooks)
  - `collate.py`  (batching, padding, masks)
  - `loader.py`   (DataLoader constructors, seeding)
  - `microsets.py` (tiny fixed sets for overfit sanity)
  - `tests/` (unit/integration tests)

---

## 3.2 Vocabulary & Tokenization (`tokenizer.py`)
- [ ] Define `Vocab` with:
  - [ ] `itos`/`stoi`, stable ordering for Σ
  - [ ] Special tokens: `PAD`, `EOS` (and optionally `BOS`)
- [ ] Validation:
  - [ ] All regex alphabet symbols are in vocab
  - [ ] Integer IDs are contiguous and stable
- [ ] Tests:
  - [ ] Round-trip str↔ids
  - [ ] Special tokens behave consistently

---

## 3.3 Dataset with On-the-Fly Labels (`dataset.py`)
- [ ] Implement `FsmDataset`:
  - [ ] Inputs: `fsm`, `samples: List[List[int]]`, `split`, `eos_id`
  - [ ] Compute per-sample:
    - [ ] `tokens` (int ids, no PAD inside)
    - [ ] `next_tokens = tokens[1:] + [EOS]`
    - [ ] `states = fsm.trace(tokens)` (length = len(tokens))
    - [ ] `class_ids = [fsm.classify(s) for s in states[:-1]]`
  - [ ] Return dict of tensors per item
- [ ] Optional label cache:
  - [ ] Toggle `cache_labels=True|False` (memoize by xxhash of tokens)
- [ ] Tests:
  - [ ] Alignment (`len(next)==len(tokens)`, `len(class)==len(tokens)-1`)
  - [ ] Determinism (same item → same labels across runs)
  - [ ] Cache correctness & memory bounds (basic)

---

## 3.4 Collation, Padding & Masks (`collate.py`)
- [ ] Implement `collate_batch(examples, pad_id, eos_id)`:
  - [ ] Left-align sequences, right-pad with `PAD`
  - [ ] Produce:
    - [ ] `tokens_bxt` (B, T)
    - [ ] `next_bxt`   (B, T)
    - [ ] `class_bxt`  (B, T) with last position masked or filled safely
    - [ ] `attn_mask_bxt` (True for valid tokens, False for PAD)
    - [ ] `loss_mask_bxt` (mask out PAD positions and last position for class loss)
- [ ] Validate EOS handling:
  - [ ] `next_tokens` uses EOS only at final valid position
- [ ] Tests:
  - [ ] Variable-length batches pad correctly
  - [ ] Masks zero-out losses on PAD & invalid positions
  - [ ] Attention mask matches valid tokens

---

## 3.5 DataLoader Construction & Seeding (`loader.py`)
- [ ] Implement `make_dataloaders(fsm, corpus, vocab, batch_size, seed, num_workers=0)`:
  - [ ] Deterministic `generator=torch.Generator().manual_seed(seed)`
  - [ ] Worker seeding via `worker_init_fn(seed)`
  - [ ] `shuffle=True` for train, `False` for val/test
  - [ ] Pin memory & prefetch factor configurable
- [ ] Provide small **bucketed sampler** (optional) to coarsely group by length for efficiency
- [ ] Tests:
  - [ ] Dataloader reproducibility (same seed → same batch order)
  - [ ] Single-worker vs multi-worker consistency (allowing known PyTorch caveats)

---

## 3.6 Micro Datasets for Sanity (`microsets.py`)
- [ ] Define fixed tiny sets (e.g., 8–32 sequences) per regex:
  - [ ] `micro_train`, `micro_val` for `a+`, `a*b*`, a branching case
- [ ] Guarantee immutability (hard-coded or seeded with fixed RNG)
- [ ] Tests:
  - [ ] Model can overfit `micro_train` (hook will be used in later milestone)

---

## 3.7 Dataset Telemetry Hooks (Lightweight)
- [ ] Add `summarize_dataset(dataset)`:
  - [ ] Length hist, class hist, min/median/max length
  - [ ] (Optional) quick FSM coverage from the samples
- [ ] Add pretty printer for quick inspection
- [ ] Tests:
  - [ ] Sums match number of samples
  - [ ] No PAD counted as symbols

---

## 3.8 Edge Cases & Invariants
- [ ] Empty string handling (`""`) when allowed by regex/FSM:
  - [ ] Ensure `next_tokens` inserts EOS properly
  - [ ] Masks exclude nonsensical positions
- [ ] Extremely short (`len=1`) and max-length sequences
- [ ] Disallow PAD inside sequences (only at batch pad)
- [ ] Assert tokens ∈ vocab and transitions are defined for all `(state, token)` except explicit reject logic
- [ ] Tests covering each edge case

---

## 3.9 Minimal CLI / Script (Optional)
- [ ] Script: compile FSM → load corpus → build dataset → dump one collated batch summary
- [ ] Aids quick manual QA before training code exists

---

## 3.10 Documentation
- [ ] Docstrings for `Vocab`, `FsmDataset`, `collate_batch`, `make_dataloaders`
- [ ] Short README section:
  - [ ] How to construct datasets from a generated corpus
  - [ ] Expected shapes and masks
  - [ ] Reproducibility notes

---

## 3.11 Completion Criteria
✅ `FsmDataset` returns correctly aligned tensors with on-the-fly labels.  
✅ `collate_batch` pads and masks correctly; tests validate EOS/PAD semantics.  
✅ Dataloaders are deterministic given a seed (including worker seeding).  
✅ Micro datasets available and reproducible for overfit sanity.  
✅ Telemetry summaries match corpus stats; no leakage or PAD misuse.  
✅ All unit/integration tests pass for at least two regexes (one linear, one branching).
