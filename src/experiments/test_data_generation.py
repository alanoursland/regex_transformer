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
    for cls, count in report.class_histogram.items():
        print(f"  {cls}: {count}")

    print("\nFirst 10 samples:")
    for i, (tokens, class_name) in enumerate(zip(samples[:10], class_names[:10]), start=1):
        length = len(tokens)
        string = "".join(fsm.alphabet[t] for t in tokens)
        print(f"  {i:2d}. '{string:10s}' (len={length:2d}, class={class_name})")

    print("\nVerifying samples...")
    errors = 0

    # Build mapping from class name → ID
    cls_to_id = {name: i for i, name in enumerate(fsm.classes)}

    for tokens, class_name in zip(samples, class_names):
        length = len(tokens)
        class_id = cls_to_id[class_name]

        # Check classification
        fsm_states = fsm.trace(tokens)
        actual_id = fsm.classify(fsm_states[-1])
        expected_class = fsm.classes[class_id]
        actual_class = fsm.classes[actual_id]

        if actual_id != class_id:
            errors += 1
            if errors <= 5:
                string = "".join(fsm.alphabet[t] for t in tokens)
                print(f"  ERROR: '{string}' - expected {class_id} ({expected_class}), got {actual_id} ({actual_class})")
                assert False
                
    if errors == 0:
        print(f"  ✓ All {len(samples)} samples verified correctly!")
    else:
        print(f"  ✗ Found {errors} errors")

    assert errors == 0


def test_accept_only_generation():
    """Test that generating only accept samples works correctly."""
    
    print("\n" + "="*70)
    print("Testing Accept-Only Generation (CRITICAL TEST)")
    print("="*70)
    
    pattern = "a+"
    alphabet = ("a", "b")
    
    print(f"\nPattern: {pattern}")
    print(f"Generating ONLY accept samples...")
    
    regex_def = RegexDefinition(
        alphabet=alphabet,
        patterns=((pattern, "accept"),)
    )
    fsm = compile_regex(regex_def)
    
    # ONLY accept samples - THIS IS WHAT --train_classes accept DOES
    p_class = {'accept': 1.0}
    gen_cfg = GenConfig(L_min=1, L_max=10, p_class=p_class)
    
    samples, class_names, report = generate_corpus(
        fsm, gen_cfg, 100, seed=42
    )
    
    print(f"  Generated: {len(samples)} samples")
    print(f"  Class distribution: {dict(report.class_histogram)}")
    
    print("\nFirst 10 samples:")
    for i, tokens in enumerate(samples[:10], start=1):
        string = "".join(fsm.alphabet[t] for t in tokens)
        states = fsm.trace(tokens)
        actual_class = fsm.classify_name(states[-1])
        print(f"  {i:2d}. '{string:10s}' -> {actual_class}")
    
    # Verify ALL are accept
    print("\nVerifying all samples are valid accept strings...")
    errors = 0
    for i, tokens in enumerate(samples):
        string = "".join(fsm.alphabet[t] for t in tokens)
        states = fsm.trace(tokens)
        actual_class = fsm.classify_name(states[-1])
        
        if actual_class != 'accept':
            errors += 1
            if errors <= 5:
                print(f"  ERROR: Sample {i} '{string}' classified as {actual_class}, not accept")
                assert False
        
        # For a+, string should only contain 'a'
        if 'b' in string:
            errors += 1
            if errors <= 5:
                print(f"  ERROR: Sample {i} '{string}' contains 'b' but pattern is a+")
                assert False
        
        if len(string) == 0:
            errors += 1
            if errors <= 5:
                print(f"  ERROR: Sample {i} is empty but pattern is a+")
                assert False

    if errors == 0:
        print(f"  ✓ All {len(samples)} samples are valid accept samples!")
    else:
        print(f"  ✗ Found {errors} errors in accept-only generation")
        print(f"\n  THIS IS THE BUG: Generator creates invalid samples even with p_class={{'accept': 1.0}}")
    
    assert errors == 0


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
            print(f"    Classes: {dict(report.class_histogram)}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            assert False


def main():
    print()

    # Run tests
    test1_passed = test_data_generation()
    test2_passed = test_accept_only_generation()  # THE CRITICAL TEST
    test3_passed = test_multiple_patterns()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if test1_passed and test2_passed and test3_passed:
        print("✓ All tests passed!")
        print("\nData generation is working correctly.")
        print("You can now run the training script:")
        print("  python -m experiments.train_model --pattern 'a+' --train_classes accept --lambda_class 0.0")
        return 0
    else:
        print("✗ Some tests failed")
        if not test2_passed:
            print("\n  CRITICAL: test_accept_only_generation() failed!")
            print("  This means --train_classes accept is generating invalid samples.")
        return 1


if __name__ == "__main__":
    sys.exit(main())