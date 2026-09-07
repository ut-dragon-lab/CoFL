"""
SecVLA benchmark continuous navigation definitions, migrated into CoFL.

Continuous navigation metrics — exclusive to agents that output smooth
velocity commands rather than discrete {FORWARD, TURN_LEFT, TURN_RIGHT, STOP}.

These metrics quantify properties that discrete agents simply cannot exhibit:
  - Heading Smoothness (HS):   mean change in trajectory heading
  - Velocity Consistency (VC): speed profile uniformity
  - Execution Fidelity (EF):  match between predicted and executed path
  - Blocked-Step Rate (BSR):   fraction of steps blocked by navmesh
  - Heading Alignment (HA):   velocity direction vs path tangent
  - Continuous SPL (cSPL):    SPL variant using arc-length integration

All inputs use world-frame 3D positions (metres) or per-step scalar values.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .metrics import GeodesicFn, success_rate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arc_length_xz(path: np.ndarray) -> float:
    """Total arc length in the XZ navigation plane."""
    if len(path) < 2:
        return 0.0
    segs = np.diff(path[:, [0, 2]], axis=0)
    return float(np.sum(np.linalg.norm(segs, axis=1)))


# ---------------------------------------------------------------------------
# Heading Smoothness (HS)
# ---------------------------------------------------------------------------


def heading_smoothness(pred_path: np.ndarray) -> float:
    """HS: trajectory heading-change smoothness ∈ [0, 1].

    HS = 1 - mean(|Δheading_t|) / π

    A perfectly straight path gives HS=1; a zigzag gives HS→0.
    Discrete agents that turn 15° every other step typically score ~0.95
    but with abrupt heading changes; a smooth curve at the same mean
    angle change scores identically in HS but looks very different in VC.
    """
    path = np.asarray(pred_path, dtype=np.float64)
    if len(path) < 3:
        return 1.0
    vecs = np.diff(path[:, [0, 2]], axis=0)  # [N-1, 2] XZ segments
    norms = np.linalg.norm(vecs, axis=1)
    valid = norms > 1e-6
    if valid.sum() < 2:
        return 1.0
    vecs = vecs[valid]
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    diffs = np.diff(angles)
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
    mean_change = float(np.mean(np.abs(diffs)))
    return float(max(0.0, 1.0 - mean_change / np.pi))


# ---------------------------------------------------------------------------
# Velocity Consistency (VC)
# ---------------------------------------------------------------------------


def velocity_consistency(linear_speeds_mps: Sequence[float]) -> float:
    """VC: speed profile uniformity ∈ [0, 1].

    VC = 1 - std(v) / (mean(v) + ε)

    A constant-speed agent scores VC=1; stop-start motion scores VC→0.
    Discrete agents produce binary speed (0 or FORWARD_STEP/dt), giving
    low VC when they mix MOVE_FORWARD and TURN steps.
    """
    v = np.asarray(linear_speeds_mps, dtype=np.float64)
    v = v[v >= 0]
    if len(v) == 0:
        return 1.0
    mu = float(np.mean(v))
    sigma = float(np.std(v))
    return float(max(0.0, 1.0 - sigma / (mu + 1e-6)))


# ---------------------------------------------------------------------------
# Execution Fidelity (EF)
# ---------------------------------------------------------------------------


def execution_fidelity(
    executed_path: np.ndarray,
    predicted_path_cart: np.ndarray,
) -> float:
    """EF: closeness of executed trajectory to model's predicted trajectory ∈ [0, 1].

    EF = 1 - RMSE(executed_xz, predicted_xz) / (path_len + ε)

    Both paths are resampled to the same number of points before comparison.
    A perfect match gives EF=1; a divergent execution gives EF→0.

    Args:
        executed_path:       [N, 3] actual world positions
        predicted_path_cart: [T, 2] (x_fwd, y_left) model prediction in
                             camera frame — converted to relative XZ offsets
                             from the start of each step, then accumulated.
    """
    exec_xz = np.asarray(executed_path, dtype=np.float64)[:, [0, 2]]
    pred_cart = np.asarray(predicted_path_cart, dtype=np.float64)

    # Resample both to the shorter length
    n = min(len(exec_xz), len(pred_cart))
    if n < 2:
        return 1.0

    exec_rs = _resample_2d(exec_xz, n)
    pred_rs = _resample_2d(pred_cart, n)

    # Align origins
    exec_rs -= exec_rs[0]
    pred_rs -= pred_rs[0]

    rmse = float(np.sqrt(np.mean(np.sum((exec_rs - pred_rs) ** 2, axis=1))))
    path_len = float(np.sum(np.linalg.norm(np.diff(exec_xz, axis=0), axis=1)))
    return float(max(0.0, 1.0 - rmse / (path_len + 1e-3)))


def _resample_2d(path: np.ndarray, n: int) -> np.ndarray:
    """Resample a 2D path to n points by arc-length parameterisation."""
    diffs = np.diff(path, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(path[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    idx = np.searchsorted(cum, targets, side="right") - 1
    idx = np.clip(idx, 0, len(path) - 2)
    alpha = np.where(seg_lens[idx] > 1e-9, (targets - cum[idx]) / seg_lens[idx], 0.0)
    return path[idx] * (1 - alpha[:, None]) + path[idx + 1] * alpha[:, None]


# ---------------------------------------------------------------------------
# Blocked-Step Rate (BSR)
# ---------------------------------------------------------------------------


def blocked_step_rate(blocked_steps: Sequence[int], total_steps: int) -> float:
    """BSR: fraction of steps where navmesh blocked forward motion ∈ [0, 1].

    Args:
        blocked_steps: list of substep-block counts per step (0 = not blocked)
        total_steps:   total number of action steps taken
    """
    if total_steps <= 0:
        return 0.0
    n_blocked = sum(1 for b in blocked_steps if b > 0)
    return float(n_blocked / total_steps)


# ---------------------------------------------------------------------------
# Heading Alignment (HA)
# ---------------------------------------------------------------------------


def heading_alignment(
    velocities_world_xz: np.ndarray,
    path_tangents_xz: np.ndarray,
) -> float:
    """HA: mean cosine similarity between velocity and path tangent ∈ [-1, 1].

    HA = mean(cos(angle(v_t, tangent_t)))

    HA=1 means the robot always moves along the planned path direction.
    HA<0 means the robot is often moving against the path direction.

    Args:
        velocities_world_xz:  [N, 2] executed velocity vectors (XZ)
        path_tangents_xz:     [N, 2] local path tangent at each step (XZ)
    """
    v = np.asarray(velocities_world_xz, dtype=np.float64)
    t = np.asarray(path_tangents_xz, dtype=np.float64)
    n = min(len(v), len(t))
    if n == 0:
        return 1.0
    v, t = v[:n], t[:n]
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    t_norm = np.linalg.norm(t, axis=1, keepdims=True)
    valid = (v_norm[:, 0] > 1e-6) & (t_norm[:, 0] > 1e-6)
    if valid.sum() == 0:
        return 1.0
    cos_sim = np.sum((v[valid] / v_norm[valid]) * (t[valid] / t_norm[valid]), axis=1)
    return float(np.mean(cos_sim))


# ---------------------------------------------------------------------------
# Continuous SPL (cSPL)
# ---------------------------------------------------------------------------


def continuous_spl(
    executed_path: np.ndarray,
    ref_path: np.ndarray,
    goal: np.ndarray,
    success_threshold_m: float = 3.0,
    *,
    geodesic_fn: GeodesicFn | None = None,
    stop_reason: str = "stop_action",
) -> float:
    """cSPL: reference-length weighted success on the dense executed path.

    cSPL = SR × ref_arc_len_xz / max(exec_arc_len_xz, ref_arc_len_xz)

    SR uses the same required navmesh distance and stop eligibility as the
    classic metrics. The reference-length numerator is a custom convention;
    standard SPL instead uses the navmesh shortest distance from the start.
    """
    pred = np.asarray(executed_path, dtype=np.float64)
    ref = np.asarray(ref_path, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)

    sr = success_rate(pred, g, success_threshold_m, stop_reason, geodesic_fn)
    if sr == 0.0:
        return 0.0

    exec_len = _arc_length_xz(pred)
    ref_len = _arc_length_xz(ref)
    shortest = max(ref_len, 1e-3)
    return sr * shortest / max(exec_len, shortest)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def compute_episode_metrics(
    executed_path: np.ndarray,
    ref_path: np.ndarray,
    goal: np.ndarray,
    linear_speeds_mps: Sequence[float],
    blocked_steps: Sequence[int],
    predicted_path_cart: np.ndarray | None = None,
    velocities_world_xz: np.ndarray | None = None,
    path_tangents_xz: np.ndarray | None = None,
    success_threshold_m: float = 3.0,
    *,
    geodesic_fn: GeodesicFn | None = None,
    stop_reason: str = "stop_action",
    dense_path: np.ndarray | None = None,
) -> dict:
    """Compute all continuous-nav metrics for a single episode.

    Args:
        executed_path:       [N, 3] world positions
        ref_path:            [M, 3] reference path positions
        goal:                [3]   goal position
        linear_speeds_mps:   [N]   per-step executed linear speed
        blocked_steps:       [N]   per-step navmesh block count
        predicted_path_cart: [T, 2] optional model prediction (x_fwd, y_left)
        velocities_world_xz: [N, 2] optional executed velocity vectors
        path_tangents_xz:    [N, 2] optional path tangent at each step
        geodesic_fn:         required navmesh distance callable for cSPL success
        stop_reason:         same stop eligibility as classic SR
        dense_path:          complete plant XYZ path for cSPL, including terminal pose

    Returns:
        dict with keys: HS, VC, BSR, cSPL, and optionally EF, HA
    """
    result: dict = {
        "HS": heading_smoothness(executed_path),
        "VC": velocity_consistency(linear_speeds_mps),
        "BSR": blocked_step_rate(blocked_steps, len(blocked_steps)),
        "cSPL": continuous_spl(
            executed_path if dense_path is None else dense_path,
            ref_path,
            goal,
            success_threshold_m,
            geodesic_fn=geodesic_fn,
            stop_reason=stop_reason,
        ),
    }

    if predicted_path_cart is not None and len(predicted_path_cart) >= 2:
        result["EF"] = execution_fidelity(executed_path, predicted_path_cart)

    if velocities_world_xz is not None and path_tangents_xz is not None:
        result["HA"] = heading_alignment(velocities_world_xz, path_tangents_xz)

    return result


def aggregate_metrics(episode_metrics: list[dict]) -> dict:
    """Mean ± std over a list of per-episode metric dicts."""
    if not episode_metrics:
        return {}
    keys = set().union(*episode_metrics)
    result = {}
    for k in sorted(keys):
        vals = [m[k] for m in episode_metrics if k in m]
        if vals:
            result[f"{k}_mean"] = float(np.mean(vals))
            result[f"{k}_std"] = float(np.std(vals))
    return result
