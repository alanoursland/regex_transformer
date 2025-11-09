#!/usr/bin/env python
"""Interactive CLI for exploring regex FSMs."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex


def print_banner():
    print("=" * 60)
    print("  REGEX FSM EXPLORER")
    print("=" * 60)
    print()


def get_regex_from_user():
    """Get regex pattern and alphabet from user."""
    print("Enter alphabet (space-separated, single chars):")
    alphabet_input = input("> ").strip()
    alphabet = tuple(alphabet_input.split())
    
    print("\nEnter regex pattern (use Python regex syntax):")
    print("  Examples: a+, a*b*, (a|b)+, ab*c")
    pattern = input("> ").strip()
    
    return alphabet, pattern


def show_fsm_stats(fsm):
    """Display FSM statistics."""
    print("\n" + "=" * 60)
    print("FSM STATISTICS")
    print("=" * 60)
    print(f"States:     {fsm.states}")
    print(f"Alphabet:   {fsm.alphabet}")
    print(f"Classes:    {fsm.classes}")
    print(f"Start:      {fsm.start}")
    print(f"Reject:     {fsm.reject}")
    print()


def show_transition_table(fsm):
    """Display the transition table."""
    print("=" * 60)
    print("TRANSITION TABLE")
    print("=" * 60)
    
    # Header
    header = "State | " + " | ".join(f"{char:>3}" for char in fsm.alphabet) + " | Class"
    print(header)
    print("-" * len(header))
    
    # Rows
    for state in range(fsm.states):
        transitions = []
        for token_id in range(len(fsm.alphabet)):
            next_state = fsm.step(state, token_id)
            transitions.append(f"{next_state:>3}")
        
        class_name = fsm.classify_name(state)
        row = f"{state:>5} | " + " | ".join(transitions) + f" | {class_name}"
        print(row)
    print()


def test_strings_interactive(fsm):
    """Test strings interactively."""
    print("=" * 60)
    print("STRING TESTER")
    print("=" * 60)
    print("Enter strings to test (or 'done' to finish):")
    print()
    
    while True:
        test_str = input("Test string (or 'done'): ").strip()
        
        if test_str.lower() == 'done':
            break
        
        # Validate characters
        invalid_chars = [c for c in test_str if c not in fsm.alphabet]
        if invalid_chars:
            print(f"  ❌ Invalid characters: {set(invalid_chars)}")
            continue
        
        # Classify
        tokens = fsm.tokens_from_string(test_str) if test_str else []
        classification = fsm.classify_string(tokens)
        
        # Trace
        states = fsm.trace(tokens)
        
        # Display result
        symbol = "✓" if classification == "accept" else "✗"
        print(f"  {symbol} '{test_str}' -> {classification}")
        
        # Show trace
        print(f"     Trace: ", end="")
        for i, state in enumerate(states):
            if i == 0:
                print(f"[{state}]", end="")
            else:
                char = test_str[i-1]
                print(f" --{char}--> [{state}]", end="")
        print()
        print()


def batch_test_examples(fsm):
    """Test a batch of example strings."""
    print("=" * 60)
    print("EXAMPLE TESTS")
    print("=" * 60)
    
    # Generate some test strings
    alphabet = list(fsm.alphabet)
    test_strings = [
        "",
        alphabet[0] if len(alphabet) > 0 else "",
        alphabet[0] * 2 if len(alphabet) > 0 else "",
        alphabet[0] * 3 if len(alphabet) > 0 else "",
    ]
    
    if len(alphabet) > 1:
        test_strings.extend([
            alphabet[1],
            alphabet[0] + alphabet[1],
            alphabet[1] + alphabet[0],
            alphabet[0] * 2 + alphabet[1],
            alphabet[0] + alphabet[1] * 2,
        ])
    
    print(f"{'String':<15} | Result")
    print("-" * 30)
    
    for test_str in test_strings:
        # Validate
        if not all(c in fsm.alphabet for c in test_str):
            continue
        
        tokens = fsm.tokens_from_string(test_str) if test_str else []
        classification = fsm.classify_string(tokens)
        
        display_str = repr(test_str) if test_str else "''"
        symbol = "✓" if classification == "accept" else "✗"
        print(f"{display_str:<15} | {symbol} {classification}")
    print()


def main():
    print_banner()
    
    # Preset examples
    print("Choose a preset or enter custom:")
    print("  1. a+        (one or more 'a's)")
    print("  2. a*b*      (zero or more a's, then zero or more b's)")
    print("  3. (a|b)+    (one or more a's or b's)")
    print("  4. a+b+      (one or more a's, then one or more b's)")
    print("  5. custom")
    print()
    
    choice = input("Choice [1-5]: ").strip()
    
    if choice == "1":
        alphabet = ("a", "b")
        pattern = "a+"
    elif choice == "2":
        alphabet = ("a", "b")
        pattern = "a*b*"
    elif choice == "3":
        alphabet = ("a", "b")
        pattern = "(a|b)+"
    elif choice == "4":
        alphabet = ("a", "b")
        pattern = "a+b+"
    else:
        alphabet, pattern = get_regex_from_user()
    
    # Create and compile
    print(f"\nCompiling regex '{pattern}' with alphabet {alphabet}...")
    
    try:
        regex_def = RegexDefinition(
            alphabet=alphabet,
            patterns=((pattern, "accept"),)
        )
        
        fsm = compile_regex(regex_def)
        print("✓ Compilation successful!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    # Show FSM
    show_fsm_stats(fsm)
    
    # Menu
    while True:
        print("=" * 60)
        print("OPTIONS")
        print("=" * 60)
        print("  1. Show transition table")
        print("  2. Test strings interactively")
        print("  3. Test example strings")
        print("  4. New regex")
        print("  5. Exit")
        print()
        
        option = input("Choice [1-5]: ").strip()
        print()
        
        if option == "1":
            show_transition_table(fsm)
        elif option == "2":
            test_strings_interactive(fsm)
        elif option == "3":
            batch_test_examples(fsm)
        elif option == "4":
            return main()  # Restart
        elif option == "5":
            print("Goodbye!")
            return 0
        else:
            print("Invalid choice. Try again.\n")


if __name__ == "__main__":
    sys.exit(main())