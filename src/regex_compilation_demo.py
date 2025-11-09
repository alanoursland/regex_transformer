#!/usr/bin/env python
"""Quick demo of FSM compilation from regex patterns."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex


def demo_pattern(name, alphabet, pattern, test_strings):
    """Demo a single regex pattern."""
    print("=" * 70)
    print(f"DEMO: {name}")
    print("=" * 70)
    print(f"Pattern:  {pattern}")
    print(f"Alphabet: {alphabet}")
    print()
    
    # Compile
    regex_def = RegexDefinition(
        alphabet=alphabet,
        patterns=((pattern, "accept"),)
    )
    fsm = compile_regex(regex_def)
    
    print(f"Compiled FSM: {fsm.states} states, {len(fsm.classes)} classes")
    print()
    
    # Test strings
    print("Test Results:")
    print(f"{'String':<12} | {'Result':<12} | State Trace")
    print("-" * 70)
    
    for test_str in test_strings:
        tokens = fsm.tokens_from_string(test_str) if test_str else []
        classification = fsm.classify_string(tokens)
        states = fsm.trace(tokens)
        
        # Format
        display_str = repr(test_str) if test_str else "''"
        symbol = "✓" if classification == "accept" else "✗"
        trace_str = " → ".join(str(s) for s in states)
        
        print(f"{display_str:<12} | {symbol} {classification:<10} | {trace_str}")
    
    print()


def main():
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║              REGEX TO FSM COMPILATION DEMO                       ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Demo 1: Simple repetition
    demo_pattern(
        "One or more 'a's",
        alphabet=("a", "b"),
        pattern="a+",
        test_strings=["", "a", "aa", "aaa", "b", "ab", "ba"]
    )
    
    # Demo 2: Concatenation
    demo_pattern(
        "a's then b's",
        alphabet=("a", "b"),
        pattern="a*b*",
        test_strings=["", "a", "b", "aa", "bb", "ab", "aabb", "ba"]
    )
    
    # Demo 3: Alternation
    demo_pattern(
        "a or b (one or more)",
        alphabet=("a", "b"),
        pattern="(a|b)+",
        test_strings=["", "a", "b", "ab", "ba", "aaa", "bbb", "ababab"]
    )
    
    # Demo 4: Specific sequence
    demo_pattern(
        "Exact sequence 'ab'",
        alphabet=("a", "b"),
        pattern="ab",
        test_strings=["", "a", "b", "ab", "ba", "aab", "abb", "aba"]
    )
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Each regex pattern was compiled into a deterministic finite automaton.")
    print("The FSM classifies strings as:")
    print("  • accept      - string matches the pattern")
    print("  • incomplete  - string is a valid prefix (can be extended to match)")
    print("  • reject      - string cannot match (even with extensions)")
    print()
    print("State traces show the sequence of FSM states visited while processing")
    print("each input string character by character.")
    print()


if __name__ == "__main__":
    main()