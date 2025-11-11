#!/usr/bin/env python
"""Test FSM to QKV construction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex
from model.fsm_construction import construct_qkv_from_fsm, fsm_forward_pass


def assert_equivalence(finite_state_machine, test_strings, verbose=True):
    """Test if FSM and QKV construction produce identical results."""
    qkv = construct_qkv_from_fsm(finite_state_machine)
    
    all_passed = True
    
    for test_str in test_strings:
        # FSM execution
        tokens = fsm.tokens_from_string(test_str) if test_str else []
        fsm_states = fsm.trace(tokens)
        fsm_final_class = fsm.classify_name(fsm_states[-1])
        
        # QKV execution
        qkv_encodings, qkv_states = fsm_forward_pass(qkv, tokens, finite_state_machine)
        qkv_final_class = fsm.classify_name(qkv_states[-1])
        
        # Compare
        match = fsm_states == qkv_states
        
        if verbose:
            status = "✓" if match else "✗"
            print(f"{status} '{test_str}'")
            print(f"  FSM states: {fsm_states}")
            print(f"  QKV states: {qkv_states}")
            print(f"  FSM class:  {fsm_final_class}")
            print(f"  QKV class:  {qkv_final_class}")
            if not match:
                print(f"  MISMATCH!")
            print()
        
        if not match:
            all_passed = False
    
    return all_passed


def main():
    print("=" * 70)
    print("FSM → QKV CONSTRUCTION TEST")
    print("=" * 70)
    print()
    
    # Test 1: Simple pattern a+
    print("TEST 1: Pattern 'a+' (one or more a's)")
    print("-" * 70)
    regex_def = RegexDefinition(
        alphabet=("a", "b"),
        patterns=(("a+", "accept"),)
    )
    finite_state_machine = compile_regex(regex_def)
    
    test_strings = ["", "a", "aa", "aaa", "b", "ab", "ba"]
    passed = assert_equivalence(finite_state_machine, test_strings, verbose=True)
    
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print()
    
    # Test 2: Pattern a*b*
    print("TEST 2: Pattern 'a*b*' (a's then b's)")
    print("-" * 70)
    regex_def = RegexDefinition(
        alphabet=("a", "b"),
        patterns=(("a*b*", "accept"),)
    )
    finite_state_machine = compile_regex(regex_def)
    
    test_strings = ["", "a", "b", "aa", "bb", "ab", "aabb", "ba"]
    passed2 = assert_equivalence(finite_state_machine, test_strings, verbose=True)
    
    print(f"Result: {'PASS' if passed2 else 'FAIL'}")
    print()
    
    # Test 3: Pattern ab
    print("TEST 3: Pattern 'ab' (exact sequence)")
    print("-" * 70)
    regex_def = RegexDefinition(
        alphabet=("a", "b"),
        patterns=(("ab", "accept"),)
    )
    finite_state_machine = compile_regex(regex_def)
    
    test_strings = ["", "a", "b", "ab", "ba", "aba"]
    passed3 = assert_equivalence(finite_state_machine, test_strings, verbose=True)
    
    print(f"Result: {'PASS' if passed3 else 'FAIL'}")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = passed and passed2 and passed3
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print()
        print("CONCLUSION: FSMs can be directly encoded into QKV matrices.")
        print("The attention mechanism successfully implements state transitions.")
    else:
        print("✗ SOME TESTS FAILED")
        print()
        print("This indicates the construction needs refinement.")
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())