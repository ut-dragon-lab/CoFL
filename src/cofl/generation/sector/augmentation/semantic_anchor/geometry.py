"""Navmesh visibility, object projection, stair-height gating and path feasibility."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..scene_analyzer.navigation import _geodesic_distance, _is_navigable, _snap_point


def _collect_stairs_aabbs(semantic_scene: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the list of (min, max) AABBs for every stairs region OR
    object in the scene.  Used by ``_snap_with_stairs_gate`` to decide
    whether a candidate point is allowed to snap to a different y than
    the agent's current floor.
    """
    out: list[tuple[np.ndarray, np.ndarray]] = []
    if semantic_scene is None:
        return out
    try:
        from ..scene_analyzer.regions import _get_region_category, _region_aabb
    except Exception:
        return out
    for region in getattr(semantic_scene, "regions", None) or []:
        try:
            cat = _get_region_category(region)
        except Exception:
            cat = None
        if cat is None or cat.lower() != "stairs":
            continue
        (mn, mx) = _region_aabb(region)
        if mn is not None and mx is not None:
            out.append((mn, mx))
    for obj in getattr(semantic_scene, "objects", None) or []:
        cat_obj = getattr(obj, "category", None)
        cat_name: str | None = None
        if cat_obj is not None:
            try:
                cat_name = cat_obj.name()
            except Exception:
                cat_name = (
                    getattr(cat_obj, "name", None)
                    if isinstance(getattr(cat_obj, "name", None), str)
                    else None
                )
        if not cat_name or str(cat_name).strip().lower() != "stairs":
            continue
        aabb = getattr(obj, "aabb", None)
        if aabb is None:
            continue
        center = getattr(aabb, "center", None)
        sizes = getattr(aabb, "sizes", None)
        if center is None or sizes is None:
            continue
        try:
            c = np.asarray(center, dtype=np.float64).reshape(3)
            s = np.asarray(sizes, dtype=np.float64).reshape(3)
            out.append((c - s * 0.5, c + s * 0.5))
        except Exception:
            continue
    return out


def _point_in_stairs_xz(
    point: np.ndarray, stairs_aabbs: list[tuple[np.ndarray, np.ndarray]] | None, pad: float = 0.2
) -> bool:
    """Is ``point``'s (x, z) inside any stairs AABB (XZ projection, padded)?"""
    if not stairs_aabbs:
        return False
    x = float(point[0])
    z = float(point[2])
    for mn, mx in stairs_aabbs:
        if mn[0] - pad <= x <= mx[0] + pad and mn[2] - pad <= z <= mx[2] + pad:
            return True
    return False


def _snap_with_stairs_gate(
    pathfinder: Any,
    point: np.ndarray,
    fallback_y: float,
    *,
    stairs_aabbs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    y_tol: float = 0.5,
    pad: float = 0.2,
) -> np.ndarray | None:
    """Accept a snap on the current floor, or on stairs that connect floor heights."""
    snapped = _snap_point(pathfinder, point)
    if snapped is None or not np.all(np.isfinite(snapped)):
        return None
    if _point_in_stairs_xz(snapped, stairs_aabbs, pad) or _point_in_stairs_xz(
        point, stairs_aabbs, pad
    ):
        return snapped
    if abs(float(snapped[1]) - float(fallback_y)) <= y_tol:
        return snapped
    return None


def _region_entry_point(
    pathfinder: Any,
    agent_position: np.ndarray,
    goal: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
) -> np.ndarray:
    """Find the first waypoint on the navmesh path agent → goal that lies
    inside the target region's AABB — geometrically the "doorway" of the
    region from the agent's standpoint.

    Used for direction classification (left/right/forward) of region
    candidates: the entry point reflects which side the agent must turn
    toward to reach the room, regardless of where ``goal`` itself sits
    inside the room (which can be deep along an off-axis lateral
    direction for large rooms).

    Falls back to ``goal`` when the path is unavailable / empty / no
    waypoint lies inside the AABB (e.g. agent already inside, or the
    navmesh path doesn't actually pierce the AABB).
    """
    g = np.asarray(goal, dtype=np.float32)
    if pathfinder is None:
        return g
    try:
        import habitat_sim

        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(agent_position, dtype=np.float32)
        sp.requested_end = g
        if not pathfinder.find_path(sp):
            return g
        pts = np.asarray(sp.points, dtype=np.float64).reshape(-1, 3)
    except Exception:
        return g
    for pt in pts:
        if (
            aabb_min[0] <= pt[0] <= aabb_max[0]
            and aabb_min[1] <= pt[1] <= aabb_max[1]
            and (aabb_min[2] <= pt[2] <= aabb_max[2])
        ):
            return pt.astype(np.float32)
    return g


def _project_object_onto_navmesh(
    pathfinder: Any,
    agent_position: np.ndarray,
    object_centroid: np.ndarray,
    max_dist_m: float,
    *,
    step_m: float = 0.1,
) -> np.ndarray | None:
    """Project ``object_centroid`` to the navmesh along the line from
    centroid → agent, returning the FIRST navigable point encountered.

    This is "the navmesh point closest to the object on the side the agent
    sees it from".  Compared with vanilla ``_snap_point`` (nearest-neighbour
    in any direction), it guarantees the projection is on the agent-facing
    side of the object — important for wall-mounted things (picture,
    mirror, tv_monitor) where snap_point can pick a navmesh cell on the
    wrong side of the wall.

    The line is traced in the XZ plane at the agent's floor height, so a
    centroid at ceiling/wall height (y ≈ 1.6 m for wall-mounted objects)
    still produces a floor-level projection.

    Returns ``None`` if no navigable point is found within ``max_dist_m``
    of the centroid — in that case the object is structurally unreachable
    from this floor / agent direction and any toward_object candidate
    pointing at it is dropped.
    """
    if pathfinder is None or max_dist_m <= 0.0:
        return None
    a = np.asarray(agent_position, dtype=np.float64).reshape(3)
    c = np.asarray(object_centroid, dtype=np.float64).reshape(3)
    delta = a - c
    delta[1] = 0.0
    dist = float(np.linalg.norm(delta))
    if dist < 0.001:
        return None
    direction = delta / dist
    trace = min(dist, float(max_dist_m))
    n_steps = max(1, int(math.ceil(trace / max(step_m, 0.001))))
    for i in range(n_steps + 1):
        d = float(i) * float(step_m)
        if d > trace:
            break
        p = np.array([c[0] + d * direction[0], a[1], c[2] + d * direction[2]], dtype=np.float64)
        if _is_navigable(pathfinder, p.astype(np.float32)):
            return p.astype(np.float32)
    return None


def _has_line_of_sight(
    pathfinder: Any,
    start: np.ndarray,
    end: np.ndarray,
    *,
    geo_slack_m: float = 0.5,
    max_snap_drift_m: float = 1.5,
    stairs_aabbs: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[bool, dict[str, float]]:
    """Geometry-grounded "agent can see ``end`` from ``start``" check.

    Algorithm (one geometric truth source: the navmesh):

    1. Snap ``end`` to the nearest navigable point — object centres often
       sit inside furniture / on walls / at the ceiling and would fail
       ``find_path`` without snapping.  When the snap drifts more than
       ``max_snap_drift_m`` the object is structurally unreachable from
       this floor → fail.
    2. ``find_path(start, snapped)`` returns the navmesh shortest path
       length ``geo``.  Compare to the snapped Euclidean ``euc``:
       * ``geo - euc > geo_slack_m``  ⇒  the path bends around something
         (a **wall** by definition, since the navmesh IS the floor plan).
         → LOS fails.
       * ``geo ≈ euc``                ⇒  straight unobstructed shot.
         → LOS passes.

    No per-sample ``is_navigable`` raster pass is needed: the navmesh
    dijkstra already encodes wall topology, and a per-sample check
    over-rejects on grid quantisation near furniture edges.
    """
    if pathfinder is None:
        return (False, {"geo_m": float("nan"), "euc_m": float("nan")})
    s = np.asarray(start, dtype=np.float64).reshape(3)
    e_raw = np.asarray(end, dtype=np.float64).reshape(3).copy()
    if not _point_in_stairs_xz(e_raw, stairs_aabbs):
        e_raw[1] = s[1]
    snapped = _snap_with_stairs_gate(pathfinder, e_raw, float(s[1]), stairs_aabbs=stairs_aabbs)
    if snapped is None or not np.all(np.isfinite(snapped)):
        return (False, {"geo_m": float("nan"), "euc_m": float("nan")})
    snap_drift = float(np.linalg.norm(snapped[[0, 2]] - e_raw[[0, 2]]))
    if snap_drift > max_snap_drift_m:
        return (
            False,
            {
                "geo_m": float("nan"),
                "euc_m": float(np.linalg.norm(e_raw - s)),
                "snap_drift": snap_drift,
            },
        )
    e = snapped.astype(np.float64)
    if not _point_in_stairs_xz(e, stairs_aabbs):
        e[1] = s[1]
    geo = _geodesic_distance(pathfinder, s, e)
    euc = float(np.linalg.norm(e - s))
    stats = {"geo_m": float(geo), "euc_m": euc, "snap_drift": snap_drift}
    if not math.isfinite(geo):
        return (False, stats)
    if geo - euc > geo_slack_m:
        return (False, stats)
    return (True, stats)


def _corridor_clear(
    pathfinder: Any,
    start: np.ndarray,
    end: np.ndarray,
    *,
    step_m: float = 0.3,
    tol: float = 0.5,
    stairs_aabbs: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> bool:
    """Return True iff EVERY sampled point along the segment ``start -> end``
    is navigable. A single navigable endpoint isn't enough — that point can
    sit past or beside an obstacle while the corridor is blocked.

    Samples include both endpoints. For a 1.5 m ray with step 0.3 this checks
    {0.0, 0.3, 0.6, 0.9, 1.2, 1.5} m along the ray.
    """
    if pathfinder is None:
        return False
    s = np.asarray(start, dtype=np.float64).reshape(3)
    e = np.asarray(end, dtype=np.float64).reshape(3)
    seg = e - s
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-06:
        return _is_navigable(pathfinder, s.astype(np.float32), tol=tol)
    n_steps = max(1, int(math.ceil(seg_len / max(step_m, 0.001))))
    for i in range(n_steps + 1):
        a = float(i) / float(n_steps)
        p_xyz = s + a * seg
        if _point_in_stairs_xz(p_xyz, stairs_aabbs):
            snapped = _snap_with_stairs_gate(
                pathfinder, p_xyz, float(s[1]), stairs_aabbs=stairs_aabbs
            )
            if snapped is None:
                return False
            p = snapped.astype(np.float32)
        else:
            p = p_xyz.astype(np.float32)
        if not _is_navigable(pathfinder, p, tol=tol):
            return False
    return True


def _sample_forward_far_point(
    pathfinder: Any,
    agent_position: np.ndarray,
    agent_heading: float,
    *,
    r_min_m: float,
    r_max_m: float,
    theta_half_rad: float,
    corridor_step_m: float,
    max_attempts: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]] | None:
    """Sample a "walk forward" goal: pick a random point inside a tight
    cone in front of the agent (r ∈ [r_min, r_max] euclidean,
    θ ∈ ±theta_half_rad around forward), accept iff every sample along
    the agent→target segment is navigable.

    The θ window is intentionally small (a few degrees) so the
    candidate is unambiguously "forward" rather than a turn — its only
    role is to add r/heading variation across frames.

    Returns ``(target_xyz, stats)`` on success or ``None`` after
    ``max_attempts`` failures.
    """
    if pathfinder is None or max_attempts <= 0:
        return None
    if r_max_m <= 0.0 or r_max_m < r_min_m:
        return None
    a = np.asarray(agent_position, dtype=np.float64).reshape(3)
    for _ in range(int(max_attempts)):
        r = float(rng.uniform(float(r_min_m), float(r_max_m)))
        theta = float(rng.uniform(-float(theta_half_rad), float(theta_half_rad)))
        h2 = float(agent_heading) + theta
        ux = math.sin(h2)
        uz = -math.cos(h2)
        target = np.array([a[0] + r * ux, a[1], a[2] + r * uz], dtype=np.float64)
        if not _corridor_clear(pathfinder, a, target, step_m=float(corridor_step_m)):
            continue
        return (
            target.astype(np.float32),
            {"forward_r_m": float(r), "forward_theta_rad": float(theta)},
        )
    return None


def _path_in_fov(
    pathfinder: Any,
    start: np.ndarray,
    end: np.ndarray,
    forward_w: np.ndarray,
    *,
    half_fov_rad: float,
    half_goal_fov_rad: float | None = None,
    sample_step_m: float = 0.4,
    min_in_fov_frac: float = 0.6,
) -> tuple[bool, dict[str, float]]:
    """FOV gate for region targets.  Returns ``(ok, stats)``.

    Replaces the older detour/curvature channel-quality gate.  The model
    only ever sees flow in the polar grid's angular range (±half_fov), so:

    1. The goal must lie inside ``half_goal_fov_rad`` from forward (defaults
       to ``half_fov_rad`` when omitted).  Setting this wider than the
       polar grid lets ``turn_X_enter`` targets (off-axis by construction)
       through — the boundary polar cells still point sideways, encoding
       "turn far left/right and enter".
    2. The geodesic path's projection onto the agent frame should mostly
       stay inside ``half_fov_rad``.  Brief excursions (path enters a
       side corridor and comes back) are tolerated via ``min_in_fov_frac``;
       setting that to 0 disables this check entirely.

    Occlusion (walls between agent and goal) is **not** a blocker — the
    BEV dijkstra already wraps walls — as long as the path stays in FOV.

    Stats: ``goal_angle_deg`` (signed |angle| at goal, ≥0),
    ``in_fov_frac`` (fraction of resampled waypoints inside FOV).
    """
    if half_goal_fov_rad is None:
        half_goal_fov_rad = half_fov_rad
    stats: dict[str, float] = {"goal_angle_deg": float("nan"), "in_fov_frac": float("nan")}
    to_goal = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    to_goal[1] = 0.0
    n = float(np.linalg.norm(to_goal))
    if n < 0.001:
        return (False, stats)
    fwd = np.asarray(forward_w, dtype=np.float64)
    cos_g = float(np.clip(np.dot(to_goal / n, fwd), -1.0, 1.0))
    goal_ang = math.acos(cos_g)
    stats["goal_angle_deg"] = math.degrees(goal_ang)
    if goal_ang > half_goal_fov_rad:
        return (False, stats)
    if min_in_fov_frac <= 0.0:
        stats["in_fov_frac"] = 1.0
        return (True, stats)
    if pathfinder is None:
        stats["in_fov_frac"] = 1.0
        return (True, stats)
    try:
        import habitat_sim

        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(start, dtype=np.float32)
        sp.requested_end = np.asarray(end, dtype=np.float32)
        if not pathfinder.find_path(sp):
            stats["in_fov_frac"] = 1.0
            return (True, stats)
        pts = np.asarray(sp.points, dtype=np.float64).reshape(-1, 3)
    except Exception:
        stats["in_fov_frac"] = 1.0
        return (True, stats)
    if pts.shape[0] < 2:
        stats["in_fov_frac"] = 1.0
        return (True, stats)
    start64 = np.asarray(start, dtype=np.float64)
    in_cnt = 0
    tot_cnt = 0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        seg[1] = 0.0
        d = float(np.linalg.norm(seg))
        if d < 1e-06:
            continue
        n_steps = max(1, int(math.ceil(d / sample_step_m)))
        u = seg / d
        for k in range(1, n_steps + 1):
            p = a + k / n_steps * d * u
            v = p - start64
            v[1] = 0.0
            nn = float(np.linalg.norm(v))
            if nn < 0.001:
                continue
            cos_p = float(np.clip(np.dot(v / nn, fwd), -1.0, 1.0))
            tot_cnt += 1
            if math.acos(cos_p) <= half_fov_rad:
                in_cnt += 1
    if tot_cnt == 0:
        stats["in_fov_frac"] = 1.0
        return (True, stats)
    frac = in_cnt / tot_cnt
    stats["in_fov_frac"] = float(frac)
    return (frac >= min_in_fov_frac, stats)
