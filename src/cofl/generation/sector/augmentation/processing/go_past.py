"""Pick the dijkstra source for a ``go_past`` alternative.

The sampler emits a *placeholder* goal at the object centroid; the actual
GT source for "going past X" must be a cell that lies on the agent's far
side of the object on the navmesh. This module finds such a cell via a
two-source Dijkstra path-threading filter so the resulting trajectory
provably routes through the object's neighbourhood before terminating.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def pick_go_past_goal_bev(
    *,
    frame_ctx: dict[str, Any],
    obj_centroid_world: np.ndarray,
    alt_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Choose a visible goal beyond the landmark using two geodesic distances.

    A candidate must be at least go_past_geo_excess_min_m from the object,
    and routing agent-to-object-to-candidate may exceed the direct distance
    by at most go_past_path_through_obj_tol_m plus object snap drift.
    Return a uniformly selected world point, or None if no candidate qualifies."""
    M_free = frame_ctx["M_free"]
    graph = frame_ctx["graph"]
    bev_fwd = frame_ctx["bev_fwd"]
    bev_lft = frame_ctx["bev_lft"]
    bev_res = float(frame_ctx["bev_resolution"])
    base_pos = np.asarray(frame_ctx["base_pos"], dtype=np.float64).reshape(3)
    forward_w = np.asarray(frame_ctx["forward_w"], dtype=np.float64).reshape(3)
    left_w = np.asarray(frame_ctx["left_w"], dtype=np.float64).reshape(3)
    (H, W) = M_free.shape
    if H == 0 or W == 0:
        return None
    fwd0 = float(bev_fwd[0, 0])
    lft0 = float(bev_lft[0, 0])

    def _snap_to_walkable(target_row: int, target_col: int) -> tuple[int, int] | None:
        (walk_rs, walk_cs) = np.where(M_free)
        if walk_rs.size == 0:
            return None
        idx = int(np.argmin((walk_rs - target_row) ** 2 + (walk_cs - target_col) ** 2))
        return (int(walk_rs[idx]), int(walk_cs[idx]))

    agent_target_row = int(round((0.0 - fwd0) / bev_res))
    agent_target_col = int(round((0.0 - lft0) / bev_res))
    snapped = _snap_to_walkable(agent_target_row, agent_target_col)
    if snapped is None:
        return None
    (agent_row, agent_col) = snapped
    obj_w = np.asarray(obj_centroid_world, dtype=np.float64).reshape(3)
    delta = obj_w - base_pos
    delta[1] = 0.0
    fwd2 = np.array([forward_w[0], forward_w[2]], dtype=np.float64)
    lft2 = np.array([left_w[0], left_w[2]], dtype=np.float64)
    d2 = np.array([delta[0], delta[2]], dtype=np.float64)
    obj_fwd_local = float(np.dot(d2, fwd2))
    obj_lft_local = float(np.dot(d2, lft2))
    obj_target_row = int(round((obj_fwd_local - fwd0) / bev_res))
    obj_target_col = int(round((obj_lft_local - lft0) / bev_res))
    obj_target_row_clamped = max(0, min(H - 1, obj_target_row))
    obj_target_col_clamped = max(0, min(W - 1, obj_target_col))
    snapped = _snap_to_walkable(obj_target_row_clamped, obj_target_col_clamped)
    if snapped is None:
        return None
    (obj_row, obj_col) = snapped
    snap_drift_cells = math.hypot(obj_row - obj_target_row, obj_col - obj_target_col)
    snap_drift_m = float(snap_drift_cells) * bev_res
    from ...fields.global_geodesic import _dijkstra_from_cell

    (_, D_a, _) = _dijkstra_from_cell(graph, M_free, (agent_row, agent_col), res=bev_res)
    (_, D_o, _) = _dijkstra_from_cell(graph, M_free, (obj_row, obj_col), res=bev_res)
    geo_to_obj_m = float(D_a[obj_row, obj_col])
    if not math.isfinite(geo_to_obj_m):
        return None
    excess_min_m = float(alt_cfg.get("go_past_geo_excess_min_m", 2.0))
    path_tol_base_m = float(alt_cfg.get("go_past_path_through_obj_tol_m", 0.5))
    path_tol_m = path_tol_base_m + snap_drift_m
    path_excess = np.full_like(D_a, np.inf)
    reachable = np.isfinite(D_a) & np.isfinite(D_o)
    path_excess[reachable] = geo_to_obj_m + D_o[reachable] - D_a[reachable]
    valid_mask = (
        M_free
        & np.isfinite(D_a)
        & np.isfinite(D_o)
        & (D_o >= excess_min_m)
        & (path_excess <= path_tol_m)
    )
    (valid_rs, valid_cs) = np.where(valid_mask)
    if valid_rs.size == 0:
        return None
    idx = int(rng.integers(0, valid_rs.size))
    r = int(valid_rs[idx])
    c = int(valid_cs[idx])
    cell_fwd = float(bev_fwd[r, c])
    cell_lft = float(bev_lft[r, c])
    world = base_pos + cell_fwd * forward_w + cell_lft * left_w
    return world.astype(np.float32)


__all__ = ["pick_go_past_goal_bev"]
