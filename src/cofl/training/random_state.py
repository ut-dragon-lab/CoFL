"""Checkpoint process RNGs using tensors and weights-only-safe Python values."""

import random

import numpy as np
import torch


def capture_random_state(*, include_cuda=True):
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all()
        if include_cuda and torch.cuda.is_initialized() else [],
    }


def restore_random_state(state):
    random.setstate(state["python"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the checkpoint's visible CUDA device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def gather_rank_states(state, world_size):
    """Checkpoint hooks run on all ranks; retain each rank's independent RNGs."""
    if world_size == 1:
        return [state]
    states = [None] * world_size
    torch.distributed.all_gather_object(states, state)
    return states
