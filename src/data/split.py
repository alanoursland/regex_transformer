"""Deterministic train/val/test splitting via hashing."""

from typing import List, Literal
import hashlib


def split_of(tokens: List[int]) -> Literal["train", "val", "test"]:
    """
    Deterministically assign a sample to a split based on content hash.

    Uses a 70/15/15 split via hash modulo 1000.

    Args:
        tokens: List of token IDs

    Returns:
        Split name: "train", "val", or "test"
    """
    # Convert tokens to bytes for hashing
    # Use 4 bytes per token (little-endian int32)
    token_bytes = b''.join(t.to_bytes(4, byteorder='little', signed=True) for t in tokens)

    # Use SHA256 hash (fast and available in stdlib)
    hash_digest = hashlib.sha256(token_bytes).digest()

    # Take first 8 bytes and interpret as uint64
    hash_val = int.from_bytes(hash_digest[:8], byteorder='little')

    # Map to split (70% train, 15% val, 15% test)
    bucket = hash_val % 1000

    if bucket < 700:
        return "train"
    elif bucket < 850:
        return "val"
    else:
        return "test"
