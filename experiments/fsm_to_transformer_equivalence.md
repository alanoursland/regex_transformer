# Experiment Plan: FSM-to-Transformer Equivalence

## Core Hypothesis

A finite state machine (FSM) compiled from a regex can be **directly encoded** into Query-Key-Value (QKV) attention matrices such that:
1. The FSM correctly implements the regex
2. Transformer attention using the constructed QKV matrices produces identical state transitions as the FSM
3. This proves transformers CAN represent FSMs (whether they LEARN to is a separate question)

## Why This Matters

If this works, it proves:
- Transformers have the **capacity** to represent finite automata
- There exists a **constructive proof** (we can build the weights)
- The learned weights in Phase 2 have a **ground truth target** to compare against

If this fails, the whole ML training experiment is built on sand.

---

## Experiment 1: Verify FSM Correctness

**Goal:** Prove the FSM compilation from regex is correct.

### Test Cases

For each regex pattern, verify the FSM produces correct accept/reject decisions:

**Simple Patterns:**
```python
patterns = [
    ("a+", ["a", "aa", "aaa"], ["", "b", "ab", "ba"]),
    ("a*", ["", "a", "aa"], ["b", "ab"]),
    ("ab", ["ab"], ["", "a", "b", "ba", "aba"]),
    ("a*b*", ["", "a", "b", "ab", "aab", "abb"], ["ba", "aba"]),
]
```

**Medium Patterns:**
```python
patterns = [
    ("(a|b)+", ["a", "b", "ab", "ba", "aaa"], ["", "c", "abc"]),
    ("a+b+", ["ab", "aab", "abb"], ["", "a", "b", "ba", "aba"]),
]
```

### Implementation

```python
# File: tests/test_fsm_correctness.py

from fsm.regex_def import RegexDefinition
from fsm.compile import compile_fsm

def test_fsm_accepts_valid_strings():
    """Verify FSM accepts all strings matching the regex"""
    regex_def = RegexDefinition(
        alphabet=("a", "b"),
        patterns=(("a+", "accept"),)
    )
    fsm = compile_fsm(regex_def)
    
    # Should accept
    assert fsm.classify("a") == "accept"
    assert fsm.classify("aa") == "accept"
    assert fsm.classify("aaa") == "accept"
    
    # Should reject
    assert fsm.classify("") == "reject"
    assert fsm.classify("b") == "reject"
    assert fsm.classify("ab") == "reject"

def test_fsm_state_trace():
    """Verify FSM produces correct state sequence"""
    regex_def = RegexDefinition(
        alphabet=("a", "b"),
        patterns=(("ab", "accept"),)
    )
    fsm = compile_fsm(regex_def)
    
    trace = fsm.trace("ab")
    # Should transition: start -> (after 'a') -> (after 'ab')
    assert len(trace) == 3  # initial + 2 characters
    assert trace[-1].classification == "accept"
```

### Success Criteria

- ✅ All test patterns correctly classified
- ✅ State traces are deterministic and complete
- ✅ FSM serialization/deserialization preserves behavior

**Time Estimate:** 30 minutes (tests probably already exist in the codebase)

---

## Experiment 2: FSM-to-QKV Construction

**Goal:** Implement the construction that maps an FSM to QKV matrices.

### Theoretical Construction

Given an FSM with states `S = {s0, s1, ..., sn}` and alphabet `Σ`:

**Key Idea:** 
- Each position's embedding encodes "current state after processing prefix"
- Attention queries "what state am I in?"
- Keys respond "I transition you to state X on symbol Y"
- Values encode "the next state"

**Construction Recipe:**

```
For FSM with:
  - n states
  - alphabet size |Σ|
  
Transformer dimensions:
  - d_model = n (one dimension per state)
  - n_heads = |Σ| (one head per input symbol)
  
Position i embedding: one-hot vector of current state
  e_i = [0, 0, ..., 1, ..., 0]  (1 at position j if in state s_j)

For head h corresponding to symbol σ:
  Q_h @ e_i = e_i  (query: "what state am I in?")
  
  K_h @ e_j = { e_j  if j == i-1
              { 0    otherwise
  (key: only previous position responds)
  
  V_h @ e_j = e_k where δ(state(j), σ) = state_k
  (value: encode the transition result)
  
Attention output at position i:
  = V_h @ softmax(Q_h @ e_i · K_h @ e_{i-1})
  = next state encoding
```

### Implementation

```python
# File: src/model/fsm_construction.py

import torch
from fsm.dfa import DFA
from typing import Dict, List

def construct_qkv_from_fsm(fsm: DFA, alphabet: List[str]) -> Dict[str, torch.Tensor]:
    """
    Construct QKV matrices that exactly implement the FSM.
    
    Args:
        fsm: The deterministic finite automaton
        alphabet: List of symbols (determines number of heads)
    
    Returns:
        Dictionary with keys 'Q', 'K', 'V' containing the constructed matrices
        
    Theory:
        - d_model = num_states (one-hot state encoding)
        - n_heads = len(alphabet) (one head per symbol)
        - Each head h processes symbol alphabet[h]
        - Attention at position i looks back to i-1
        - Output encodes δ(state[i-1], symbol[i])
    """
    num_states = len(fsm.states)
    num_heads = len(alphabet)
    d_model = num_states
    d_head = d_model // num_heads
    
    # Initialize matrices
    Q = torch.zeros(num_heads, d_model, d_head)
    K = torch.zeros(num_heads, d_model, d_head)
    V = torch.zeros(num_heads, d_model, d_head)
    
    for h, symbol in enumerate(alphabet):
        # Query: identity (ask "what state am I in?")
        Q[h] = torch.eye(d_model, d_head)
        
        # Key: identity (respond with state)
        K[h] = torch.eye(d_model, d_head)
        
        # Value: encode transition function
        for s_idx, state in enumerate(fsm.states):
            next_state = fsm.transitions.get((state, symbol))
            if next_state is not None:
                next_idx = fsm.states.index(next_state)
                V[h, s_idx, :] = torch.zeros(d_head)
                V[h, s_idx, next_idx] = 1.0  # one-hot encoding of next state
    
    return {"Q": Q, "K": K, "V": V}

def fsm_forward_pass(qkv: Dict[str, torch.Tensor], 
                     input_sequence: List[int],
                     alphabet: List[str]) -> torch.Tensor:
    """
    Execute the constructed QKV matrices on an input sequence.
    
    Args:
        qkv: Dictionary with Q, K, V matrices
        input_sequence: List of token indices
        alphabet: Symbol list (for head selection)
    
    Returns:
        Tensor of shape [seq_len, d_model] with state encodings at each position
    """
    Q, K, V = qkv["Q"], qkv["K"], qkv["V"]
    seq_len = len(input_sequence)
    d_model = Q.shape[1]
    
    # Initial state (assume state 0 is start)
    states = torch.zeros(seq_len + 1, d_model)
    states[0, 0] = 1.0  # one-hot: start state
    
    for i, token in enumerate(input_sequence):
        symbol = alphabet[token]
        head = alphabet.index(symbol)
        
        # Attention: current position queries previous position
        q = Q[head] @ states[i]  # [d_head]
        k = K[head] @ states[i]  # [d_head]
        
        # Simplified attention (only look at previous position)
        attn = torch.softmax(q @ k.T / (k.shape[0] ** 0.5), dim=-1)
        
        # Apply value transform
        v = V[head] @ states[i]
        states[i + 1] = attn * v
    
    return states[1:]  # exclude initial state
```

### Test Cases

```python
# File: tests/test_fsm_qkv_construction.py

def test_qkv_construction_simple():
    """Test QKV construction on simple regex: a+"""
    regex_def = RegexDefinition(
        alphabet=("a",),
        patterns=(("a+", "accept"),)
    )
    fsm = compile_fsm(regex_def)
    qkv = construct_qkv_from_fsm(fsm, ["a"])
    
    # Verify dimensions
    assert qkv["Q"].shape[0] == 1  # one head
    assert qkv["K"].shape[0] == 1
    assert qkv["V"].shape[0] == 1

def test_qkv_equivalence():
    """Test that QKV forward pass matches FSM trace"""
    regex_def = RegexDefinition(
        alphabet=("a", "b"),
        patterns=(("ab", "accept"),)
    )
    fsm = compile_fsm(regex_def)
    qkv = construct_qkv_from_fsm(fsm, ["a", "b"])
    
    # Test sequence: "ab"
    input_seq = [0, 1]  # indices for 'a', 'b'
    
    # FSM trace
    fsm_trace = fsm.trace("ab")
    fsm_states = [t.state for t in fsm_trace]
    
    # QKV forward pass
    qkv_states = fsm_forward_pass(qkv, input_seq, ["a", "b"])
    
    # Convert QKV output to state indices (argmax of one-hot)
    qkv_state_indices = torch.argmax(qkv_states, dim=-1).tolist()
    
    # Compare
    assert qkv_state_indices == fsm_states[1:]  # exclude initial state
```

### Success Criteria

- ✅ QKV construction produces valid matrices (no NaN, proper shapes)
- ✅ Forward pass produces one-hot encodings (or close to it)
- ✅ State transitions match FSM exactly for all test cases

**Time Estimate:** 2-3 hours (implement + debug construction logic)

---

## Experiment 3: Equivalence Verification

**Goal:** Systematically verify FSM and QKV produce identical results.

### Test Suite

```python
# File: tests/test_equivalence.py

import pytest

@pytest.mark.parametrize("regex,test_strings", [
    ("a+", ["a", "aa", "aaa", "aaaa"]),
    ("a*b*", ["", "a", "b", "ab", "aab", "abb", "aabb"]),
    ("(a|b)+", ["a", "b", "ab", "ba", "aba", "bab"]),
    ("a+b+", ["ab", "aab", "abb", "aaabbb"]),
])
def test_fsm_qkv_equivalence(regex, test_strings):
    """Verify FSM and QKV produce identical state sequences"""
    alphabet = ("a", "b")
    regex_def = RegexDefinition(alphabet=alphabet, patterns=((regex, "accept"),))
    fsm = compile_fsm(regex_def)
    qkv = construct_qkv_from_fsm(fsm, list(alphabet))
    
    for test_str in test_strings:
        # FSM execution
        fsm_trace = fsm.trace(test_str)
        fsm_states = [t.state_idx for t in fsm_trace][1:]  # exclude initial
        
        # QKV execution  
        input_seq = [alphabet.index(c) for c in test_str]
        qkv_output = fsm_forward_pass(qkv, input_seq, list(alphabet))
        qkv_states = torch.argmax(qkv_output, dim=-1).tolist()
        
        # Assert equivalence
        assert qkv_states == fsm_states, \
            f"Mismatch for '{test_str}': FSM={fsm_states}, QKV={qkv_states}"
```

### Success Criteria

- ✅ 100% equivalence on all test cases
- ✅ Works for empty string, single character, and longer sequences
- ✅ Works for alternation, kleene star, and concatenation

**Time Estimate:** 1 hour (assuming construction is correct)

---

## Experiment 4: Complexity Analysis

**Goal:** Understand the limitations of the construction.

### Questions to Answer

1. **Does this scale?**
   - How many states does `(a|b)*c` produce?
   - What's d_model for realistic regex patterns?

2. **What about overlapping heads?**
   - Current design: one head per symbol
   - Could we use fewer heads with shared computation?

3. **What about position encoding?**
   - Current design assumes causal attention handles position
   - Do we need explicit positional embeddings?

### Tests

```python
def test_complexity_scaling():
    """Measure FSM size for increasingly complex patterns"""
    patterns = [
        "a+",
        "a+b+", 
        "(a|b)+",
        "(a|b)*c",
        "(a|b|c)+",
    ]
    
    for pattern in patterns:
        regex_def = RegexDefinition(alphabet=("a", "b", "c"), patterns=((pattern, "accept"),))
        fsm = compile_fsm(regex_def)
        print(f"{pattern}: {len(fsm.states)} states, {len(fsm.transitions)} transitions")
```

**Time Estimate:** 30 minutes

---

## Summary: Execution Order

**Phase 1: Validate Foundation (1 hour)**
1. Run existing FSM tests (if they exist)
2. Add missing FSM correctness tests
3. Verify FSM compilation is correct

**Phase 2: Implement Construction (3 hours)**
1. Write `construct_qkv_from_fsm()` function
2. Write `fsm_forward_pass()` function  
3. Debug on single test case (`a+`)

**Phase 3: Verify Equivalence (2 hours)**
1. Write equivalence test suite
2. Fix any mismatches
3. Test on progressively complex patterns

**Phase 4: Analysis (1 hour)**
1. Measure complexity scaling
2. Document limitations
3. Write up findings

**Total Time Estimate: 7 hours**

---

## Success Definition

**Minimum Success:**
- FSM correctly implements simple regex (a+, ab)
- QKV construction produces valid matrices
- Equivalence holds for at least one pattern

**Full Success:**
- FSM handles all standard regex operators
- QKV equivalence verified on 10+ diverse patterns  
- Scaling analysis shows construction is practical
- Clear documentation of what works and what doesn't

**Stretch Goal:**
- Implement multi-head optimization (fewer heads than alphabet size)
- Handle epsilon transitions
- Extend to more complex regex features

---

## Next Steps After This Experiment

If equivalence holds:
✅ **Phase 2:** Train transformer and see if it LEARNS these weights
✅ **Phase 3:** Extract learned weights and compare to construction
✅ **Phase 4:** Publish findings

If equivalence fails:
❌ Debug construction logic
❌ Revisit theoretical assumptions
❌ Consider alternative attention mechanisms

This experiment is the **foundation**. Everything else builds on this.