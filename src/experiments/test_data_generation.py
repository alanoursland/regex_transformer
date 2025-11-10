#!/usr/bin/env python
"""Test data generation for training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex
from data.generator import generate_corpus, GenConfig


def test_data_generation():
    """Test data generation pipeline."""

    print("="*70)
    print("Testing Data Generation")
    print("="*70)

    # Test pattern
    pattern = "a+"
    alphabet = ("a", "b")

    print(f"\nPattern: {pattern}")
    print(f"Alphabet: {alphabet}")

    # Compile FSM
    regex_def = RegexDefinition(
        alphabet=alphabet,
        patterns=((pattern, "accept"),)
    )
    fsm = compile_regex(regex_def)

    print(f"\nFSM compiled:")
    print(f"  States: {fsm.states}")
    print(f"  Classes: {fsm.classes}")

    # Generate data
    n_samples = 100
    print(f"\nGenerating {n_samples} samples...")

    p_class = {cls: 1.0/len(fsm.classes) for cls in fsm.classes}
    gen_cfg = GenConfig(
        L_min=1,
        L_max=10,
        p_class=p_class
    )

    samples, class_names, report = generate_corpus(
        fsm, gen_cfg, n_samples, seed=42
    )

    print(f"  Generated: {len(samples)} samples")
    print(f"\nClass distribution:")
    for cls, count in report['class_distribution'].items():
        print(f"  {cls}: {count}")

    print(f"\nFirst 10 samples:")
    for i, (tokens, length, class_id) in enumerate(samples[:10]):
        class_name = fsm.classes[class_id]
        string = "".join(fsm.alphabet[t] for t in tokens[:length])
        print(f"  {i+1:2d}. '{string:10s}' (len={length:2d}, class={class_name})")

    # Verify samples
    print(f"\nVerifying samples...")
    errors = 0
    for tokens, length, class_id in samples:
        # Check classification
        fsm_states = fsm.trace(tokens[:length])
        actual_class = fsm.classify(fsm_states[-1])
        expected_class = fsm.classes[class_id]

        if actual_class != expected_class:
            errors += 1
            if errors <= 5:  # Show first few errors
                string = "".join(fsm.alphabet[t] for t in tokens[:length])
                print(f"  ERROR: '{string}' - expected {expected_class}, got {actual_class}")

    if errors == 0:
        print(f"  ✓ All {len(samples)} samples verified correctly!")
    else:
        print(f"  ✗ Found {errors} errors")

    return errors == 0


def test_multiple_patterns():
    """Test data generation with multiple patterns."""

    print("\n" + "="*70)
    print("Testing Multiple Patterns")
    print("="*70)

    patterns = [
        "a+",
        "a*b*",
        "(a|b)+",
        "ab",
    ]

    all_passed = True

    for pattern in patterns:
        print(f"\nPattern: {pattern}")

        regex_def = RegexDefinition(
            alphabet=("a", "b"),
            patterns=((pattern, "accept"),)
        )
        fsm = compile_regex(regex_def)

        p_class = {cls: 1.0/len(fsm.classes) for cls in fsm.classes}
        gen_cfg = GenConfig(L_min=1, L_max=8, p_class=p_class)

        try:
            samples, class_names, report = generate_corpus(
                fsm, gen_cfg, 50, seed=42
            )
            print(f"  ✓ Generated {len(samples)} samples")
            print(f"    Classes: {dict(report['class_distribution'])}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            all_passed = False

    return all_passed


def main():
    print()

    # Run tests
    test1_passed = test_data_generation()
    test2_passed = test_multiple_patterns()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if test1_passed and test2_passed:
        print("✓ All tests passed!")
        print("\nData generation is working correctly.")
        print("You can now run the training script:")
        print("  python -m experiments.train_model --pattern 'a+' --epochs 10")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
