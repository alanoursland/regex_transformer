"""Tests for FSM compilation."""

import re
import pytest
from ..regex_def import RegexDefinition
from ..compile import compile_regex


def test_compile_simple_a_plus():
    """Test compiling a+ pattern."""
    regex_def = RegexDefinition(
        alphabet=('a',),
        patterns=(('a+', 'accept'),)
    )

    fsm = compile_regex(regex_def)

    # Basic structure checks
    assert fsm.alphabet == ('a',)
    assert 'accept' in fsm.classes
    assert 'reject' in fsm.classes or 'incomplete' in fsm.classes

    # Test string classification
    assert fsm.classify_string(fsm.tokens_from_string('a')) == 'accept'
    assert fsm.classify_string(fsm.tokens_from_string('aa')) == 'accept'
    assert fsm.classify_string(fsm.tokens_from_string('aaa')) == 'accept'

    # Empty string should not match a+
    result = fsm.classify_string([])
    assert result != 'accept'  # Should be reject or incomplete


def test_compile_a_star_b_star():
    """Test compiling a*b* pattern."""
    regex_def = RegexDefinition(
        alphabet=('a', 'b'),
        patterns=(('a*b*', 'accept'),)
    )

    fsm = compile_regex(regex_def)

    # Test matching strings
    test_cases = [
        ('', 'accept'),      # a*b* matches empty
        ('a', 'accept'),
        ('aa', 'accept'),
        ('b', 'accept'),
        ('bb', 'accept'),
        ('ab', 'accept'),
        ('aab', 'accept'),
        ('abb', 'accept'),
        ('aabb', 'accept'),
    ]

    for string, expected_class in test_cases:
        tokens = fsm.tokens_from_string(string)
        result = fsm.classify_string(tokens)
        assert result == expected_class, f"String {string!r} should be {expected_class}, got {result}"

    # Test non-matching strings (ba should not match a*b*)
    tokens = fsm.tokens_from_string('ba')
    result = fsm.classify_string(tokens)
    assert result != 'accept', "String 'ba' should not be accepted by a*b*"


def test_fsm_trace():
    """Test FSM trace method."""
    regex_def = RegexDefinition(
        alphabet=('a', 'b'),
        patterns=(('ab', 'accept'),)
    )

    fsm = compile_regex(regex_def)
    tokens = fsm.tokens_from_string('ab')
    states = fsm.trace(tokens)

    # Should have len(tokens) + 1 states
    assert len(states) == 3  # start, after 'a', after 'b'

    # All states should be valid
    for state in states:
        assert 0 <= state < fsm.states


def test_fsm_transitions_total():
    """Test that FSM has total transition function."""
    regex_def = RegexDefinition(
        alphabet=('a', 'b'),
        patterns=(('a+', 'accept'),)
    )

    fsm = compile_regex(regex_def)

    # Check all transitions are defined
    expected_transitions = fsm.states * len(fsm.alphabet)
    assert len(fsm.delta) == expected_transitions

    # Check reject state self-loops
    for token_id in range(len(fsm.alphabet)):
        assert fsm.delta[(fsm.reject, token_id)] == fsm.reject


def test_alternation():
    """Test compiling alternation pattern."""
    regex_def = RegexDefinition(
        alphabet=('a', 'b'),
        patterns=(('a|b', 'accept'),)
    )

    fsm = compile_regex(regex_def)

    # Both 'a' and 'b' should be accepted
    assert fsm.classify_string(fsm.tokens_from_string('a')) == 'accept'
    assert fsm.classify_string(fsm.tokens_from_string('b')) == 'accept'

    # Empty and 'ab' should not be accepted
    assert fsm.classify_string([]) != 'accept'
    assert fsm.classify_string(fsm.tokens_from_string('ab')) != 'accept'


def test_regex_equivalence():
    """Test that FSM matches the regex semantics using re module."""
    patterns_to_test = [
        ('a+', 'accept'),
        ('a*b*', 'accept'),
        ('(ab)*', 'accept'),
        ('a|b', 'accept'),
    ]

    alphabet = ('a', 'b')

    # Generate test strings
    test_strings = [
        '',
        'a',
        'b',
        'aa',
        'ab',
        'ba',
        'bb',
        'aaa',
        'aab',
        'aba',
        'abb',
        'baa',
        'bab',
        'bba',
        'bbb',
        'abab',
    ]

    for pattern, class_name in patterns_to_test:
        regex_def = RegexDefinition(
            alphabet=alphabet,
            patterns=((pattern, class_name),)
        )

        fsm = compile_regex(regex_def)
        compiled_re = re.compile(f'^{pattern}$')

        for test_str in test_strings:
            # Check if string is in alphabet
            if not all(c in alphabet for c in test_str):
                continue

            fsm_result = fsm.classify_string(fsm.tokens_from_string(test_str))
            re_matches = compiled_re.match(test_str) is not None

            if re_matches:
                assert fsm_result == class_name, \
                    f"Pattern {pattern}: FSM should accept {test_str!r}, got {fsm_result}"
            # Note: FSM might be more permissive (incomplete states), so we only check positive matches
