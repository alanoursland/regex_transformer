"""Tests for FSM serialization."""

import tempfile
import pytest
from pathlib import Path

from ..regex_def import RegexDefinition
from ..compile import compile_regex
from ..serialize import save_fsm, load_fsm


def test_save_load_roundtrip():
    """Test that FSM can be saved and loaded without loss."""
    regex_def = RegexDefinition(
        alphabet=('a', 'b'),
        patterns=(('a+', 'accept'),)
    )

    original_fsm = compile_regex(regex_def)

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_fsm(original_fsm, path, regex_def)

        # Load back
        loaded_fsm = load_fsm(path)

        # Verify structure matches
        assert loaded_fsm.states == original_fsm.states
        assert loaded_fsm.alphabet == original_fsm.alphabet
        assert loaded_fsm.start == original_fsm.start
        assert loaded_fsm.delta == original_fsm.delta
        assert loaded_fsm.classes == original_fsm.classes
        assert loaded_fsm.state_class == original_fsm.state_class
        assert loaded_fsm.reject == original_fsm.reject

        # Verify behavior matches
        test_strings = ['', 'a', 'aa', 'aaa']
        for s in test_strings:
            tokens = original_fsm.tokens_from_string(s)
            assert loaded_fsm.classify_string(tokens) == original_fsm.classify_string(tokens)


def test_checksum_validation():
    """Test that checksum mismatch is detected."""
    regex_def = RegexDefinition(
        alphabet=('a',),
        patterns=(('a+', 'accept'),)
    )

    fsm = compile_regex(regex_def)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_fsm(fsm, path, regex_def)

        # Manually corrupt the file
        import json
        with open(path, 'r') as f:
            data = json.load(f)

        # Modify FSM data
        data['fsm']['start'] = 999

        with open(path, 'w') as f:
            json.dump(data, f)

        # Try to load - should fail checksum
        with pytest.raises(ValueError, match="Checksum mismatch"):
            load_fsm(path)


def test_atomic_write():
    """Test that failed writes don't corrupt existing files."""
    regex_def = RegexDefinition(
        alphabet=('a',),
        patterns=(('a', 'accept'),)
    )

    fsm = compile_regex(regex_def)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        # First successful write
        save_fsm(fsm, path)

        # Verify temp file doesn't exist
        temp_path = path.with_suffix(path.suffix + '.tmp')
        assert not temp_path.exists()

        # Verify main file exists
        assert path.exists()
