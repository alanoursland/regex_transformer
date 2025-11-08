"""FSM serialization and deserialization."""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

from .dfa import FSM
from .regex_def import RegexDefinition


def save_fsm(fsm: FSM, path: Path, regex_def: Optional[RegexDefinition] = None) -> None:
    """
    Save an FSM to a JSON file.

    Args:
        fsm: FSM to save
        path: Output file path
        regex_def: Optional original regex definition for reference
    """
    # Convert FSM to dict
    data: Dict[str, Any] = {
        "version": "0.1.0",
        "fsm": {
            "states": fsm.states,
            "alphabet": list(fsm.alphabet),
            "start": fsm.start,
            "delta": {f"{s},{t}": next_s for (s, t), next_s in fsm.delta.items()},
            "classes": list(fsm.classes),
            "state_class": fsm.state_class,
            "reject": fsm.reject,
        },
        "metadata": {
            "num_states": fsm.states,
            "num_transitions": len(fsm.delta),
            "alphabet_size": len(fsm.alphabet),
            "num_classes": len(fsm.classes),
        },
    }

    # Add regex definition if provided
    if regex_def is not None:
        data["regex_definition"] = {
            "alphabet": list(regex_def.alphabet),
            "patterns": [list(p) for p in regex_def.patterns],
        }

    # Compute checksum
    canonical_json = json.dumps(data["fsm"], sort_keys=True)
    checksum = hashlib.sha256(canonical_json.encode()).hexdigest()
    data["checksum"] = checksum

    # Write to file
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp then rename
    temp_path = path.with_suffix(path.suffix + '.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        temp_path.rename(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def load_fsm(path: Path) -> FSM:
    """
    Load an FSM from a JSON file.

    Args:
        path: Input file path

    Returns:
        Reconstructed FSM
    """
    with open(path, 'r') as f:
        data = json.load(f)

    fsm_data = data["fsm"]

    # Verify checksum if present
    if "checksum" in data:
        canonical_json = json.dumps(fsm_data, sort_keys=True)
        actual_checksum = hashlib.sha256(canonical_json.encode()).hexdigest()
        if actual_checksum != data["checksum"]:
            raise ValueError(f"Checksum mismatch in {path}")

    # Reconstruct delta dict with correct key types
    delta = {}
    for key_str, next_s in fsm_data["delta"].items():
        s, t = key_str.split(',')
        delta[(int(s), int(t))] = next_s

    fsm = FSM(
        states=fsm_data["states"],
        alphabet=tuple(fsm_data["alphabet"]),
        start=fsm_data["start"],
        delta=delta,
        classes=tuple(fsm_data["classes"]),
        state_class=fsm_data["state_class"],
        reject=fsm_data["reject"],
    )

    return fsm
