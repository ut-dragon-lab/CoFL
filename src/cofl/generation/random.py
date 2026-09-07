"""Scheduling-independent seeds and isolated global random state."""

import hashlib
import json
import random
from contextlib import contextmanager

import numpy as np


def unit_seed(seed: int, method: str, key: str) -> int:
    payload = json.dumps([seed, method, key], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


@contextmanager
def seeded_random(seed: int):
    """Scope Python and NumPy global RNGs while providing a modern generator.

    Adapters seed simulators separately using the same unit seed. Process-level
    sharding is supported; this context is intentionally not thread-safe.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield np.random.default_rng(seed)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
