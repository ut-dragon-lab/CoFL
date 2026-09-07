"""
VLN-CE navigation metric definitions, without simulator imports.

The shared online runner supplies navmesh geodesics, executed XYZ positions,
and the official GT_PATH locations. NE/OS/SR/SPL/nDTW/SDTW follow the Habitat
and VLN-CE definitions. The runner's continuous control and sampling protocol
are recorded separately; this module does not execute the official evaluator.
CLS is retained as a custom legacy XZ coverage metric, not official CLS.

References:
  - Anderson et al., Vision-and-Language Navigation (R2R, 2018)
  - Ku et al., Room-Across-Rooms (RxR, 2020)
  - Ilharco et al., General Evaluation for Instruction Conditioned Navigation (2019)

All functions operate on world-frame 3D positions (metres).
Trajectories are lists/arrays of [x, y, z] waypoints.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# A geodesic-distance callable provided by the environment: takes two
# world-frame xyz points and returns the shortest-path distance (metres)
# on the navmesh. Positive infinity denotes an unreachable goal. Online
# navigation scores require this callable; there is no Euclidean fallback.
GeodesicFn = Callable[[np.ndarray, np.ndarray], float]


def _require_geodesic(geodesic_fn):
    if not callable(geodesic_fn):
        raise ValueError("Online navigation metrics require a navmesh geodesic_fn")


def _xyz_point(point):
    value = np.asarray(point, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Geodesic metrics require a finite XYZ point [3]")
    return value


def _xyz_path(path):
    value = np.asarray(path, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
        raise ValueError("Geodesic metrics require a nonempty finite XYZ path [N,3]")
    return value


def _dist(
    a: np.ndarray,
    b: np.ndarray,
    geodesic_fn: GeodesicFn | None = None,
) -> float:
    """Navmesh distance in metres; unreachable points retain positive infinity."""
    _require_geodesic(geodesic_fn)
    distance = float(geodesic_fn(_xyz_point(a), _xyz_point(b)))
    if np.isnan(distance) or distance < 0:
        raise ValueError("Navmesh geodesic distance must be nonnegative or positive infinity")
    return distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path_length(path: np.ndarray) -> float:
    """Executed arc length in XYZ, as used by Habitat SPL."""
    if len(path) < 2:
        return 0.0
    segs = np.diff(path, axis=0)
    return float(np.sum(np.linalg.norm(segs, axis=1)))


def _xz_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance in the XZ navigation plane."""
    return float(np.linalg.norm(np.array(a)[[0, 2]] - np.array(b)[[0, 2]]))


def _xyz_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean XYZ distance used by the official VLN-CE NDTW measure."""
    return float(np.linalg.norm(np.asarray(b) - np.asarray(a)))


def _dtw(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """Exact XYZ Dynamic Time Warping with linear auxiliary memory."""
    n, m = len(path_a), len(path_b)
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        for j in range(1, m + 1):
            cost = _xyz_dist(path_a[i - 1], path_b[j - 1])
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return float(previous[m])


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def navigation_error(
    pred_path: np.ndarray,
    goal: np.ndarray,
    geodesic_fn: GeodesicFn | None = None,
) -> float:
    """NE: navmesh distance from the actual final position to the goal (metres)."""
    return _dist(_xyz_path(pred_path)[-1], goal, geodesic_fn=geodesic_fn)


def oracle_success(
    path: np.ndarray,
    goal: np.ndarray,
    threshold_m: float = 3.0,
    geodesic_fn: GeodesicFn | None = None,
) -> float:
    """OS: any recorded XYZ pose is within the navmesh distance threshold.

    Scan the complete plant path, with early exit on success. Exact repeated
    poses share a query; no temporal subsampling can discard a close approach.
    """
    _require_geodesic(geodesic_fn)
    p = np.asarray(path, dtype=np.float64)
    g = _xyz_point(goal)
    if p.size == 0:
        return 0.0
    seen = set()
    for xyz in _xyz_path(p):
        key = tuple(xyz)
        if key in seen:
            continue
        seen.add(key)
        if _dist(xyz, g, geodesic_fn=geodesic_fn) < threshold_m:
            return 1.0
    return 0.0


# Reasons that represent the policy emitting a STOP action. Budget exhaustion,
# cancellation and failures never substitute for STOP, even at the goal.
# 'status_head' / 'action_head_stop' are stops fired by the model's discrete
# action head; 'trajectory_endpoint' is the legacy fallback that stops when
# the predicted trajectory endpoint collapses near the agent (used on
# checkpoints without an action head).
_STOP_ACTION_REASONS = {
    "stop_action",
    "stop",
    "agent_stop",
    "status_head",
    "action_head_stop",
    "trajectory_endpoint",
    # STOP drainout (benchmark/agents/secvla.py): the inner planner fired
    # STOP, the wrapper coasted on the captured trajectory, and the drain
    # finished (endpoint reached, tick budget exhausted, or no-progress
    # detector tripped). All three cases are a "final stop" that should
    # be SR-eligible — distance to goal still gates it.
    "stop_drain_done",
}


def success_rate(
    pred_path: np.ndarray,
    goal: np.ndarray,
    threshold_m: float = 3.0,
    stop_reason: str = "stop_action",
    geodesic_fn: GeodesicFn | None = None,
) -> float:
    """SR: explicit policy STOP with final navmesh distance < ``threshold_m``.

    Distance is always navmesh geodesic; unreachable goals are unsuccessful.
    Timeout/max_steps are failures even when the agent has reached the goal.
    """
    _require_geodesic(geodesic_fn)
    if stop_reason not in _STOP_ACTION_REASONS:
        return 0.0
    return float(_dist(_xyz_path(pred_path)[-1], goal, geodesic_fn) < threshold_m)


def spl(
    pred_path: np.ndarray,
    ref_path: np.ndarray,
    goal: np.ndarray,
    threshold_m: float = 3.0,
    stop_reason: str = "stop_action",
    geodesic_fn: GeodesicFn | None = None,
) -> float:
    """SPL: Success weighted by (normalised inverse) Path Length.

    SPL = SR × shortest_path_length / max(pred_len, shortest_len)

    The shortest-path length ``L*`` is queried as the geodesic distance
    from the trajectory's starting position to the goal. An unreachable start
    never substitutes the reference route length for the navmesh distance.
    """
    sr = success_rate(pred_path, goal, threshold_m, stop_reason, geodesic_fn=geodesic_fn)
    if sr == 0.0:
        return 0.0
    pred_len = _path_length(pred_path)
    shortest = _dist(pred_path[0], goal, geodesic_fn)
    if not np.isfinite(shortest):
        return 0.0
    if shortest == 0.0:
        return float(pred_len == 0.0)
    return sr * shortest / max(pred_len, shortest)


def ndtw(
    pred_path: np.ndarray,
    gt_path: np.ndarray,
    success_threshold_m: float = 3.0,
    *,
    fdtw: bool = True,
) -> float:
    """Official VLN-CE nDTW path fidelity ∈ [0, 1].

    nDTW = exp(-DTW(executed_xyz, gt_locations) / (d_t × len(gt_locations)))

    Only consecutive identical executed positions are removed, exactly as in
    the official measure. GT points retain their supplied multiplicity. The
    default uses official FDTW=True (fastdtw's default radius); FDTW=False uses
    exact DTW. GT_PATH locations must be supplied, not episode.reference_path.
    """
    pred, gt = _xyz_path(pred_path), _xyz_path(gt_path)
    if not np.isfinite(success_threshold_m) or success_threshold_m <= 0:
        raise ValueError("nDTW success threshold must be finite and positive")
    keep = np.r_[True, np.any(pred[1:] != pred[:-1], axis=1)]
    pred = pred[keep]
    if fdtw:
        try:
            from fastdtw import fastdtw
        except ImportError as error:
            raise ImportError(
                "Official FDTW=True nDTW requires fastdtw; install CoFL's online dependencies"
            ) from error
        dtw_val = float(fastdtw(pred, gt, dist=_xyz_dist)[0])
    else:
        dtw_val = _dtw(pred, gt)
    return float(np.exp(-dtw_val / (success_threshold_m * len(gt))))


def cls(pred_path: np.ndarray, ref_path: np.ndarray, success_threshold_m: float = 3.0) -> float:
    """Legacy custom CLS: binary XZ coverage times exponential length penalty.

    Retained for continuity with past CoFL runs; this is not the official
    Coverage weighted Length Score formula from Jain et al. (2019).

    CLS = PC × exp(-|pl - rpl| / rpl)
    where PC = mean coverage of reference path by predicted path.
    """
    # Path coverage: for each ref point, find nearest pred point distance
    coverages = []
    for rp in ref_path:
        dists = [_xz_dist(rp, pp) for pp in pred_path]
        coverages.append(float(min(dists)))

    # PC = fraction of ref points covered within threshold
    pc = float(np.mean([c <= success_threshold_m for c in coverages]))

    pred_len = _path_length(pred_path[:, [0, 2]])
    ref_len = _path_length(ref_path[:, [0, 2]])
    if ref_len < 1e-3:
        return pc
    len_score = float(np.exp(-abs(pred_len - ref_len) / ref_len))
    return pc * len_score


def sdtw(
    pred_path: np.ndarray,
    gt_path: np.ndarray,
    goal: np.ndarray,
    success_threshold_m: float = 3.0,
    stop_reason: str = "stop_action",
    geodesic_fn: GeodesicFn | None = None,
    *,
    fdtw: bool = True,
) -> float:
    """Official success weighted DTW: SDTW = nDTW × SR."""
    return ndtw(pred_path, gt_path, success_threshold_m, fdtw=fdtw) * success_rate(
        pred_path, goal, success_threshold_m, stop_reason, geodesic_fn=geodesic_fn
    )


# ---------------------------------------------------------------------------
# Aggregate over episode list
# ---------------------------------------------------------------------------


def compute_episode_metrics(
    pred_path: np.ndarray,
    ref_path: np.ndarray,
    goal: np.ndarray,
    success_threshold_m: float = 3.0,
    stop_reason: str = "stop_action",
    dense_path: np.ndarray | None = None,
    geodesic_fn: GeodesicFn | None = None,
    *,
    gt_path: np.ndarray,
    ndtw_fdtw: bool = True,
) -> dict:
    """Compute official VLN-CE definitions plus the legacy custom CLS metric.

    Args:
        pred_path:           [N, 3] executed world-frame positions (planner-
                             tick sampled).
        ref_path:            [M, 3] episode.reference_path, for legacy CLS only
        goal:                [3]   goal position
        success_threshold_m: distance threshold for success (default 3.0 m)
        stop_reason:         how the episode ended; only explicit policy STOP
                             is eligible for SR/SPL/SDTW.
        dense_path:          [K, 3] optional complete plant-tick XYZ path,
                             including the final executed position. Supplies
                             NE/OS/SR/SPL/nDTW/SDTW; malformed paths are rejected.
        geodesic_fn:         required ``(p1, p2) -> float`` navmesh distance
                             callable, typically ``env.geodesic_distance``.
                             Positive infinity means unreachable.
        gt_path:             required [L, 3] official GT_PATH locations for nDTW;
                             never silently replaced by episode.reference_path.
        ndtw_fdtw:           official NDTW.FDTW switch (default True).

    Returns:
        dict with keys: NE, OS, SR, SPL, nDTW, SDTW, CLS
    """
    _require_geodesic(geodesic_fn)
    pred, ref, g = _xyz_path(pred_path), _xyz_path(ref_path), _xyz_point(goal)
    executed = pred if dense_path is None else _xyz_path(dense_path)
    distances = {}

    def distance(source, target):
        key = (tuple(source), tuple(target))
        if key not in distances:
            distances[key] = _dist(source, target, geodesic_fn)
        return distances[key]

    sr = success_rate(executed, g, success_threshold_m, stop_reason, distance)
    fidelity = ndtw(executed, gt_path, success_threshold_m, fdtw=ndtw_fdtw)

    return {
        "NE": navigation_error(executed, g, geodesic_fn=distance),
        "OS": oracle_success(executed, g, success_threshold_m, geodesic_fn=distance),
        "SR": sr,
        "SPL": spl(executed, ref, g, success_threshold_m, stop_reason, geodesic_fn=distance),
        "nDTW": fidelity,
        "SDTW": fidelity * sr,
        "CLS": cls(pred, ref, success_threshold_m),
    }


def aggregate_metrics(episode_metrics: list[dict]) -> dict:
    """Mean ± std over a list of per-episode metric dicts."""
    if not episode_metrics:
        return {}
    keys = episode_metrics[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in episode_metrics if k in m]
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_std"] = float(np.std(vals))
    return result
