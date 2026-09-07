"""The portable CoFL payload inside a standard Lightning checkpoint."""

import torch

SCHEMA_VERSION = 2


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("cofl", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported CoFL checkpoint schema: start a new run; older checkpoints "
            "do not contain the data and random states required for exact resume"
        )
    return checkpoint
