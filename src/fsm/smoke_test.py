#!/usr/bin/env python
"""Quick smoke test for FSM core functionality."""

from pathlib import Path
import tempfile

from .regex_def import RegexDefinition
from .compile import compile_regex
from .serialize import save_fsm, load_fsm


def main():
    print("=== FSM Core Smoke Test ===\n")

    # 1. Define a regex
    print("1. Creating regex definition for pattern 'a+'")
    regex_def = RegexDefinition(
        alphabet=('a', 'b'),
        patterns=(('a+', 'accept'),)
    )
    print(f"   Alphabet: {regex_def.alphabet}")
    print(f"   Patterns: {regex_def.patterns}\n")

    # 2. Compile to FSM
    print("2. Compiling to FSM...")
    fsm = compile_regex(regex_def)
    print(f"   States: {fsm.states}")
    print(f"   Classes: {fsm.classes}")
    print(f"   Reject state: {fsm.reject}\n")

    # 3. Test string classification
    print("3. Testing string classification:")
    test_strings = ['', 'a', 'aa', 'aaa', 'b', 'ab']
    for s in test_strings:
        tokens = fsm.tokens_from_string(s)
        classification = fsm.classify_string(tokens)
        print(f"   '{s}' -> {classification}")
    print()

    # 4. Test trace
    print("4. Tracing string 'aa':")
    tokens = fsm.tokens_from_string('aa')
    states = fsm.trace(tokens)
    print(f"   Tokens: {tokens}")
    print(f"   States: {states}")
    for i, state in enumerate(states):
        class_name = fsm.classify(state)
        if i == 0:
            print(f"   Start at state {state} ({class_name})")
        else:
            print(f"   After '{tokens[i-1]}' -> state {state} ({class_name})")
    print()

    # 5. Test serialization
    print("5. Testing serialization...")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_fsm(fsm, path, regex_def)
        print(f"   Saved to {path}")

        loaded_fsm = load_fsm(path)
        print(f"   Loaded back successfully")

        # Verify
        test_str = 'aaa'
        tokens = fsm.tokens_from_string(test_str)
        original_result = fsm.classify_string(tokens)
        loaded_result = loaded_fsm.classify_string(tokens)
        assert original_result == loaded_result
        print(f"   Verification: '{test_str}' -> {loaded_result} (matches original)\n")

    print("=== All smoke tests passed! ===")


if __name__ == "__main__":
    main()
