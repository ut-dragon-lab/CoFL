"""Storage-independent Cartesian field and replay-label checks."""

from __future__ import annotations

import numpy as np


def validate_frame(frame, grid_shape):
    """Fail on corrupt payloads; ignored action labels remain explicit diagnostics."""
    valid = np.asarray(frame["slot_valid"])
    k = len(valid)
    expected = (k, 2) + tuple(grid_shape)
    if np.shape(frame["v_K"]) != expected:
        raise ValueError("Sector field shape differs from the configured Cartesian grid")
    trajectory = np.asarray(frame["dp_traj_K"])
    if trajectory.ndim != 3 or trajectory.shape[0] != k or trajectory.shape[-1] != 2:
        raise ValueError("Slot trajectories must have shape [K, horizon+1, 2]")
    if np.shape(frame["trajectory_delta"]) != (k, trajectory.shape[1] - 1, 2):
        raise ValueError("Trajectory delta shape is inconsistent")
    for name in (
        "v_K",
        "dp_traj_K",
        "trajectory_delta",
        "path_len_K",
        "depth",
        "position",
        "rotation_xyzw",
    ):
        if not np.isfinite(np.asarray(frame[name])).all():
            raise ValueError("Nonfinite generated array: " + name)
    if np.any(np.asarray(frame["depth"]) < 0) or np.any(np.asarray(frame["path_len_K"]) < 0):
        raise ValueError("Metric depth and path length must be nonnegative")
    if not np.allclose(np.diff(trajectory, axis=1), frame["trajectory_delta"], atol=1e-6):
        raise ValueError("Trajectory deltas do not reconstruct the stored trajectory")
    if np.any(trajectory[:, 0] != 0):
        raise ValueError("Body-frame trajectories must begin at the origin")
    for name in ("texts", "family_K", "cmd_records", "slot_actions", "path_len_K"):
        if len(frame[name]) != k:
            raise ValueError("Slot metadata length differs: " + name)
    if int(frame["action_id"]) not in (0, 1, 2, 3):
        raise ValueError("Executed action is outside the navigation action vocabulary")
    if (
        not np.isin(valid, (0, 1)).all()
        or not np.isin(frame["slot_actions"], (-100, 0, 1, 2, 3)).all()
    ):
        raise ValueError("Invalid slot validity or action label")
    for slot in np.flatnonzero(valid):
        if not str(frame["texts"][slot]).strip() or frame["cmd_records"][slot] is None:
            raise ValueError("Valid slots require language and command provenance")
    # Failed/unallocated candidates remain metadata. Numeric payload in an invalid
    # slot would be lost by sparse native emission, so reject that inconsistency.
    for slot in np.flatnonzero(~valid.astype(bool)):
        if np.any(frame["v_K"][slot]) or np.any(trajectory[slot]) or frame["path_len_K"][slot] != 0:
            raise ValueError("Invalid slot contains an unaccounted numeric target")
    for name in ("bev_mask", "bev_walkable"):
        if np.shape(frame[name]) != tuple(grid_shape):
            raise ValueError("Auxiliary navigation mask shape differs from field grid")
    return {
        "valid_slots": int(valid.sum()),
        "ignored_action_slots": int(
            np.sum(valid.astype(bool) & (np.asarray(frame["slot_actions"]) == -100))
        ),
    }
