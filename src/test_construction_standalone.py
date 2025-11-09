#!/usr/bin/env python
"""Standalone test for FSM→QKV construction (no torch dependency)."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Direct imports to avoid torch in __init__.py
from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex  
from fsm.dfa import FSM


# Inline construction code to avoid import issues
def construct_qkv_from_fsm(fsm):
    """Construct value matrices that implement FSM transitions."""
    num_states = fsm.states
    num_heads = len(fsm.alphabet)
    
    V = []
    for head_idx in range(num_heads):
        token_id = head_idx
        head_V = []
        for state in range(num_states):
            next_state = fsm.step(state, token_id)
            one_hot = [0.0] * num_states
            one_hot[next_state] = 1.0
            head_V.append(one_hot)
        V.append(head_V)
    
    initial_state = [0.0] * num_states
    initial_state[fsm.start] = 1.0
    
    return {'V': V, 'initial_state': initial_state}


def matrix_vector_mult(matrix, vector):
    """Multiply matrix by vector (result[i] = sum_j matrix[j][i] * vector[j])."""
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows > 0 else 0
    result = [0.0] * n_cols
    for j in range(n_rows):
        for i in range(n_cols):
            result[i] += matrix[j][i] * vector[j]
    return result


def argmax(vector):
    """Return index of maximum value."""
    return max(range(len(vector)), key=lambda i: vector[i])


def fsm_forward_pass(qkv, tokens, fsm):
    """Execute QKV on input sequence."""
    V = qkv['V']
    states = [qkv['initial_state']]
    predicted_ids = [fsm.start]
    
    for token_id in tokens:
        prev_state = states[-1]
        next_state = matrix_vector_mult(V[token_id], prev_state)
        states.append(next_state)
        predicted_ids.append(argmax(next_state))
    
    return states, predicted_ids


def test_pattern(name, alphabet, pattern, test_strings):
    """Test a single regex pattern."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"Pattern: {pattern}")
    print(f"Alphabet: {alphabet}\n")
    
    # Compile FSM
    regex_def = RegexDefinition(alphabet=alphabet, patterns=((pattern, "accept"),))
    fsm = compile_regex(regex_def)
    print(f"FSM: {fsm.states} states\n")
    
    # Construct QKV
    qkv = construct_qkv_from_fsm(fsm)
    print(f"QKV constructed: {len(qkv['V'])} heads")
    print()
    
    # Test strings
    all_match = True
    for test_str in test_strings:
        tokens = fsm.tokens_from_string(test_str) if test_str else []
        
        # FSM execution
        fsm_states = fsm.trace(tokens)
        fsm_class = fsm.classify_name(fsm_states[-1])
        
        # QKV execution
        _, qkv_states = fsm_forward_pass(qkv, tokens, fsm)
        qkv_class = fsm.classify_name(qkv_states[-1])
        
        match = fsm_states == qkv_states
        symbol = "✓" if match else "✗"
        
        display_str = repr(test_str) if test_str else "''"
        print(f"{symbol} {display_str:8} | FSM: {fsm_states} → {fsm_class}")
        print(f"  {' '*8} | QKV: {qkv_states} → {qkv_class}")
        
        if not match:
            print(f"  MISMATCH!")
            all_match = False
        print()
    
    return all_match


def main():
    print("\n" + "="*70)
    print(" FSM → QKV EQUIVALENCE TEST")
    print("="*70)
    print("\nTheory: FSM transitions can be directly encoded as value matrices")
    print("in multi-head attention, where each head processes one symbol.\n")
    
    results = []
    
    # Test 1
    results.append(test_pattern(
        "One or more 'a's",
        ("a", "b"),
        "a+",
        ["", "a", "aa", "aaa", "b", "ab"]
    ))
    
    # Test 2
    results.append(test_pattern(
        "a's then b's",
        ("a", "b"),
        "a*b*",
        ["", "a", "b", "ab", "aab", "abb", "ba"]
    ))
    
    # Test 3
    results.append(test_pattern(
        "Exact 'ab'",
        ("a", "b"),
        "ab",
        ["", "a", "b", "ab", "ba", "aba"]
    ))
    
    # Test 4
    results.append(test_pattern(
        "One or more of a or b",
        ("a", "b"),
        "(a|b)+",
        ["", "a", "b", "ab", "ba", "aaa", "bbb"]
    ))
    
    # Summary
    print("="*70)
    print(" SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}\n")
    
    if all(results):
        print("✓ ALL TESTS PASSED!")
        print("\nCONCLUSION:")
        print("  • FSMs CAN be directly encoded into attention matrices")
        print("  • QKV construction produces EXACT state transitions")
        print("  • This validates the theoretical foundation")
        print("\nNext step: Train transformers and see if they LEARN these weights\n")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nThe construction needs refinement.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())