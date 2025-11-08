# Milestone 1 — FSM Core  
**Goal:** Implement and verify the foundational FSM system (regex → NFA → DFA → minimized DFA) that all later components depend on.

---

## **Task List**

### 1.1 Project Setup
- [ ] Create `src/fsm/` package and `tests/` subdirectory.  
- [ ] Initialize minimal Python environment (`requirements.txt` or `pyproject.toml`):  
  - `python>=3.10`, `pytest`, `networkx` (optional, for visualization).  
- [ ] Add stub `__init__.py` files and empty module templates.

---

### 1.2 Regex Definition Module (`regex_def.py`)
- [ ] Implement `RegexDefinition` dataclass with:
  - `alphabet: Tuple[str, ...]`  
  - `patterns: Tuple[Tuple[str, str], ...]` (pattern + class label)
- [ ] Add validation logic:
  - Ensure regex syntax validity (`re.compile` check).  
  - Ensure all tokens appear in `alphabet`.  
  - Check unique class names and disjoint patterns.  
- [ ] Add helper function: `def load_regex_def(path: Path) -> RegexDefinition`.

---

### 1.3 FSM Data Structure (`dfa.py`)
- [ ] Define `FSM` dataclass with:
  - `states: int`, `alphabet: Tuple[str, ...]`, `start: int`
  - `delta: Dict[Tuple[int, int], int]`  → transition function  
  - `state_class: List[int]`             → per-state class ID  
  - `classes: Tuple[str, ...]`, `reject: int`
- [ ] Implement core methods:
  - `step(state, token_id) -> int`
  - `classify(state) -> int`
  - `trace(tokens: Sequence[int]) -> List[int]`
- [ ] Implement consistency checks:
  - All transitions defined (total DFA).  
  - Reject state self-loops on all inputs.  
  - Class IDs within valid range.

---

### 1.4 FSM Compilation (`compile.py`)
- [ ] Implement **regex → NFA** (Thompson construction).  
- [ ] Implement **NFA → DFA** (subset construction).  
- [ ] Implement **DFA minimization** (Hopcroft algorithm).  
- [ ] Add explicit reject state for undefined transitions.  
- [ ] Map accepting states to their class labels.  
- [ ] Validate correctness:
  - Each input symbol leads to exactly one next state.  
  - All accepting states correctly classified.  
  - Start state reachable; no dead states unless reject.

---

### 1.5 Serialization & Loading (`serialize.py`)
- [ ] Implement JSON serialization (`save_fsm(fsm, path)` / `load_fsm(path)`).  
- [ ] Include metadata:
  - Alphabet, number of states, regex patterns, version stamp.  
- [ ] Validate checksum or hash for consistency testing.

---

### 1.6 Visualization (Optional, `viz.py`)
- [ ] Use `networkx` or `graphviz` to plot FSM:
  - States colored by class (accept/incomplete/reject).  
  - Directed edges labeled by token.  
- [ ] Useful for debugging and data sanity checks.

---

### 1.7 Testing (`tests/`)
- [ ] Unit tests for FSM construction:
  - Compile small regexes (`a+`, `(ab)*`, `a|b`, `a*b*`).  
  - Verify `fsm.trace()` matches regex acceptance using `re.fullmatch`.  
- [ ] Test reject state transitions and class labels.  
- [ ] Round-trip test: `fsm == load_fsm(save_fsm(fsm))`.  
- [ ] Property test: all transitions valid over Σ and deterministic.

---

### 1.8 Documentation & Sanity Checks
- [ ] Docstrings for all public functions/classes.  
- [ ] Inline comments for algorithm steps (esp. DFA minimization).  
- [ ] Quick smoke script: compile a regex, print transitions, classify strings.

---

### **Milestone 1 Completion Criteria**
✅ Regex definitions compile deterministically into minimized, valid FSMs.  
✅ All FSM transitions and classifications verified via tests.  
✅ FSMs can be serialized/deserialized without loss.  
✅ Optional visualization renders a correct state graph.  
✅ Code passes linting and tests (`pytest src/fsm/tests`).
