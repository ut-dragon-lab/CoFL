"""Derive four-action labels and trajectory deltas from body-frame paths.

STOP requires an explicit terminal family or a completed toward-object
approach. Other commands use the trajectory direction at the first
1 m radial crossing, or its farthest point for shorter paths."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

ACTION_STOP = 0
ACTION_MOVE_FORWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
ACTION_IGNORE = -100
_TANGENT_PROBE_RADIUS_M = 1.0
_FORWARD_HALF_RAD = math.radians(15.0)
_STOP_SNAP_PATH_EPS_M = 0.01


def _derive_action_id(cart: np.ndarray, *, path_length_m: float, family: str | None) -> int:
    """Bucket the dijkstra cart trajectory into STOP / FWD / TURN_LEFT / TURN_RIGHT.

    cart : (H+1, 2) float32  — (x_fwd, y_lft) metres; cart[0] = (0, 0).
    path_length_m : float    — total dijkstra polyline length in metres
                                (NOT the cart radius — they only coincide
                                for a perfectly straight path).
    family : Optional[str]   — the slot's command family. Drives the STOP
                                eligibility — see module docstring.

    STOP rules (apply uniformly to anchor and alternatives):
      * ``family == "terminal"`` → always STOP (the builder marks the
        anchor with this family on the last sub-instruction frame, and
        the sampler may emit explicit terminal alternatives).
      * ``family == "toward_object"`` AND ``path_length_m`` collapses to
        ~0 → STOP (the upstream stop-snap fired, agent is inside the
        approach ring around the object).
      * Anything else → fall through to FWD/TURN bucketing. There is no
        physical-distance fallback. A short region_enter / go_past path
        still gets a directional label.

    For FWD/TURN, walk the cart and find the first sample with cart
    radius ≥ ``_TANGENT_PROBE_RADIUS_M``; bucket on the yaw at that
    sample. Truly degenerate trajectories (all-zero cart) return
    ``ACTION_IGNORE`` — the slot is left out of the discrete-action
    loss rather than mislabelled.
    """
    fam = str(family) if family is not None else ""
    if fam == "terminal":
        return ACTION_STOP
    if fam == "toward_object" and float(path_length_m) < _STOP_SNAP_PATH_EPS_M:
        return ACTION_STOP
    rs = np.hypot(cart[:, 0], cart[:, 1])
    above = np.where(rs >= _TANGENT_PROBE_RADIUS_M)[0]
    if above.size > 0:
        i = int(above[0])
    else:
        i = int(np.argmax(rs))
    dx = float(cart[i, 0])
    dy = float(cart[i, 1])
    if dx == 0.0 and dy == 0.0:
        return ACTION_IGNORE
    theta = math.atan2(dy, dx)
    if abs(theta) <= _FORWARD_HALF_RAD:
        return ACTION_MOVE_FORWARD
    return ACTION_TURN_LEFT if theta > 0.0 else ACTION_TURN_RIGHT


def compute_frame_baseline_labels(
    *,
    dp_traj_cart_K: np.ndarray,
    path_length_m_K: np.ndarray,
    slot_valid_K: np.ndarray,
    family_K: Sequence[str | None],
) -> dict[str, np.ndarray]:
    """Compute action-id labels + dp_traj delta for one frame's K slots.

    Every slot — including the anchor — is bucketed from its dijkstra cart
    trajectory using :func:`_derive_action_id`. The anchor no longer takes
    a separate ``gt_action`` override: its STOP signal must come from the
    ``family == "terminal"`` flag set by the builder at the terminal frame
    of each episode. This keeps the action-token target for k=0 identical
    in source to the alternatives, so the status head supervised by
    ``(action_id_anchor == ACTION_STOP)`` aligns directly with the
    action-token loss for the anchor.

    Invalid (slot_valid=0) slots get a zero delta and ``ACTION_IGNORE``.
    """
    K = int(dp_traj_cart_K.shape[0])
    H1 = int(dp_traj_cart_K.shape[1])
    H = H1 - 1
    if len(family_K) != K:
        raise ValueError(f"family_K length {len(family_K)} != K {K}; pass one entry per slot.")
    action_K = np.full((K,), ACTION_IGNORE, dtype=np.int64)
    delta_K = np.zeros((K, H, 2), dtype=np.float32)
    for k in range(K):
        if int(slot_valid_K[k]) == 0:
            continue
        cart = np.asarray(dp_traj_cart_K[k], dtype=np.float32)
        action_K[k] = _derive_action_id(
            cart, path_length_m=float(path_length_m_K[k]), family=family_K[k]
        )
        delta_K[k] = np.diff(cart, axis=0).astype(np.float32)
    return {"dp_traj_cart_delta": delta_K, "action_id": action_K}


__all__ = [
    "ACTION_IGNORE",
    "ACTION_MOVE_FORWARD",
    "ACTION_STOP",
    "ACTION_TURN_LEFT",
    "ACTION_TURN_RIGHT",
    "compute_frame_baseline_labels",
]
