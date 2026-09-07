"""Height-aware global navmesh geodesics and sector-clipped predecessor paths."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt
from scipy.sparse.csgraph import dijkstra as sp_dijkstra


def _build_world_grid_extents(
    bev_x_max: float, goal_fwd: float, goal_lft: float, margin_m: float
) -> tuple[float, float, float, float]:
    """Pick a grid box that covers BEV ∪ goal + margin.

    Returns ``(fwd_min, fwd_max, lft_min, lft_max)`` in ego-frame metres.
    The forward axis stays ``>= 0`` (we never need to plan backward through
    the agent in ego frame for this use-case; allow a small back-margin so
    agent-near cells are interior rather than edge cells of the grid).
    """
    fwd_min = min(0.0, float(goal_fwd)) - float(margin_m)
    fwd_max = max(float(bev_x_max), float(goal_fwd)) + float(margin_m)
    lft_min = min(-float(bev_x_max), float(goal_lft)) - float(margin_m)
    lft_max = max(float(bev_x_max), float(goal_lft)) + float(margin_m)
    return (fwd_min, fwd_max, lft_min, lft_max)


def _ego_to_world(
    fwd_grid: np.ndarray,
    lft_grid: np.ndarray,
    base_pos: np.ndarray,
    forward_w: np.ndarray,
    left_w: np.ndarray,
) -> np.ndarray:
    """Project an ego-frame grid (fwd, lft) into world XYZ at agent floor y.

    Returns ``(H, W, 3)`` float64 world-frame positions. Y is taken from
    ``base_pos[1]`` — caller may snap downstream.
    """
    base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
    forward_w = np.asarray(forward_w, dtype=np.float64).reshape(3)
    left_w = np.asarray(left_w, dtype=np.float64).reshape(3)
    return (
        base_pos[None, None, :]
        + fwd_grid[..., None] * forward_w[None, None, :]
        + lft_grid[..., None] * left_w[None, None, :]
    )


def _rasterise_world_walkable(
    pathfinder: Any,
    world_pts_hwc: np.ndarray,
    floor_y: float,
    *,
    agent_cell: tuple[int, int] | None = None,
    horiz_drift_tol_m: float = 0.1,
    step_height_m: float = 0.25,
) -> np.ndarray:
    """Rasterize the height-connected navmesh component containing the agent.

    Accept snaps with horizontal drift <= horiz_drift_tol_m, then flood
    8-neighbor cells with adjacent height jumps below step_height_m.
    If grid quantization masks the seed, start at its nearest candidate.
    This crosses stair treads while excluding walls and separate storeys."""
    (H, W, _) = world_pts_hwc.shape
    flat = world_pts_hwc.reshape(-1, 3).astype(np.float32).copy()
    flat[:, 1] = float(floor_y)
    snap = getattr(pathfinder, "snap_point", None)
    if snap is None:
        return np.zeros((H, W), dtype=bool)
    N = H * W
    candidate_flat = np.zeros(N, dtype=bool)
    y_flat = np.full(N, np.nan, dtype=np.float32)
    tol2 = float(horiz_drift_tol_m) ** 2
    for i in range(N):
        try:
            sp = np.asarray(snap(flat[i]), dtype=np.float32).reshape(3)
        except Exception:
            continue
        if not np.all(np.isfinite(sp)):
            continue
        dx = float(sp[0]) - float(flat[i, 0])
        dz = float(sp[2]) - float(flat[i, 2])
        if dx * dx + dz * dz > tol2:
            continue
        candidate_flat[i] = True
        y_flat[i] = float(sp[1])
    candidate = candidate_flat.reshape(H, W)
    y_grid = y_flat.reshape(H, W)
    walkable = np.zeros((H, W), dtype=bool)
    if agent_cell is None:
        return walkable
    (a_r, a_c) = (int(agent_cell[0]), int(agent_cell[1]))
    if not (0 <= a_r < H and 0 <= a_c < W) or not candidate[a_r, a_c]:
        if not candidate.any():
            return walkable
        (cr, cc) = np.where(candidate)
        k = int(np.argmin((cr - a_r) ** 2 + (cc - a_c) ** 2))
        (a_r, a_c) = (int(cr[k]), int(cc[k]))
    from collections import deque

    walkable[a_r, a_c] = True
    queue = deque([(a_r, a_c)])
    step_h = float(step_height_m)
    while queue:
        (r, c) = queue.popleft()
        y_here = y_grid[r, c]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                (nr, nc) = (r + dr, c + dc)
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if walkable[nr, nc] or not candidate[nr, nc]:
                    continue
                if abs(float(y_grid[nr, nc]) - float(y_here)) >= step_h:
                    continue
                walkable[nr, nc] = True
                queue.append((nr, nc))
    return walkable


def _build_grid_graph_uniform(
    walkable: np.ndarray, res: float, *, cost_map: np.ndarray | None = None
) -> sp.csr_matrix:
    """8-connected weighted graph on a 2D walkable mask.

    When ``cost_map`` is ``None`` edges are pure geometric step lengths
    (``res`` for axis, ``res * sqrt(2)`` for diagonal). When provided, each
    edge weight is multiplied by ``0.5 * (cost_map[s] + cost_map[t])`` —
    matching the BEV-side ``_build_grid_graph`` so the global geodesic
    inherits the same CoFL-style safety-band repulsion.
    """
    (H, W) = walkable.shape
    N = H * W
    idx = np.arange(N, dtype=np.int32).reshape(H, W)
    sqrt2 = math.sqrt(2.0)
    (rows_l, cols_l, data_l) = ([], [], [])

    def _add(dy: int, dx: int, step: float) -> None:
        r0 = max(0, -dy)
        r1 = min(H, H - dy)
        c0 = max(0, -dx)
        c1 = min(W, W - dx)
        if r0 >= r1 or c0 >= c1:
            return
        (sr, sc) = (slice(r0, r1), slice(c0, c1))
        (dr, dc) = (slice(r0 + dy, r1 + dy), slice(c0 + dx, c1 + dx))
        valid = walkable[sr, sc] & walkable[dr, dc]
        if not valid.any():
            return
        s = idx[sr, sc][valid]
        d = idx[dr, dc][valid]
        if cost_map is None:
            w = np.full(s.shape[0], float(res) * step, dtype=np.float32)
        else:
            cs = cost_map[sr, sc][valid]
            ct = cost_map[dr, dc][valid]
            w = (0.5 * (cs + ct) * float(res) * step).astype(np.float32)
        rows_l.append(s)
        cols_l.append(d)
        data_l.append(w)

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        _add(dy, dx, 1.0)
    for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        _add(dy, dx, sqrt2)
    if not rows_l:
        return sp.csr_matrix((N, N), dtype=np.float32)
    r = np.concatenate(rows_l)
    c = np.concatenate(cols_l)
    d = np.concatenate(data_l).astype(np.float32)
    return sp.coo_matrix((d, (r, c)), shape=(N, N)).tocsr()


def _build_world_safety_cost_map(
    walkable: np.ndarray, safe_radius_cells: float, safety_cost_weight: float
) -> np.ndarray:
    """World-frame safety cost: ``cost = 1 + λ * max(0, ρ − D_obs)``.

    Operates in *cells* (the grid's native unit), matching CoFL's
    pixel-based formulation. ``D_obs`` is the EDT (in cells) from each
    walkable cell to the nearest real obstacle (= ``~walkable``).
    Mirrors the BEV-side ``_build_safety_cost_map`` so both grids share
    a single dimensionless parameterisation: ``safe_radius_cells`` is
    the band width, ``λ`` is the per-cell penalty slope, and the peak
    cost at the wall is ``1 + λ * safe_radius_cells``. With the
    canonical settings (``ρ=20, λ=5``), peak cost = 101 — strong enough
    to bend Dijkstra paths sharply away from real walls (one wall-edge
    step ≈ 100 free-space steps).
    """
    d_obs_cells = distance_transform_edt(walkable).astype(np.float32)
    cost = np.ones_like(d_obs_cells, dtype=np.float32)
    sr = float(safe_radius_cells)
    if sr > 0.0 and float(safety_cost_weight) > 0.0:
        near = d_obs_cells < sr
        if near.any():
            cost[near] += float(safety_cost_weight) * (sr - d_obs_cells[near])
    return cost


def _snap_goal_to_grid(
    goal_fwd: float,
    goal_lft: float,
    walkable: np.ndarray,
    fwd_vals: np.ndarray,
    lft_vals: np.ndarray,
) -> tuple[int, int] | None:
    """Find the closest walkable grid cell to (goal_fwd, goal_lft).

    Returns ``(row, col)`` or ``None`` if no cells are walkable.
    """
    if not walkable.any():
        return None
    (rows, cols) = np.where(walkable)
    df = fwd_vals[rows] - float(goal_fwd)
    dl = lft_vals[cols] - float(goal_lft)
    d2 = df * df + dl * dl
    k = int(np.argmin(d2))
    return (int(rows[k]), int(cols[k]))


def _dijkstra_from_cell(
    graph: sp.csr_matrix, walkable: np.ndarray, source_rc: tuple[int, int], res: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-source Dijkstra; ``np.inf`` / ``-9999`` outside walkable.

    Returns
    -------
    dist : (H, W) float32  distance grid (cost-weighted metres if the
                            graph carries a cost map, else pure metres).
    dist_pix_m : (H, W) float32  *geometric* path length in metres along
                            the predecessor tree (axis hops contribute
                            ``res``, diagonal hops contribute ``res*√2``).
                            Decoupled from ``dist`` so the velocity field's
                            magnitude is in pure metres even when ``dist``
                            is cost-weighted — matches CoFL's static
                            ``compute_velocity_field(geodesic_distance=
                            geodesic_pixel)``.
    pred : (H*W,) int64    flat predecessor array — ``pred[v]`` is the
                            cell index immediately *before* ``v`` on the
                            shortest path from ``source_rc`` to ``v``.
                            ``pred[source]`` is the scipy sentinel
                            ``-9999`` (the source has no predecessor); the
                            same sentinel marks unreachable cells.
                            Walking ``pred`` from any cell back to the
                            source enumerates the geodesic path.
    """
    (H, W) = walkable.shape
    N = H * W
    src = source_rc[0] * W + source_rc[1]
    if not walkable.ravel()[src]:
        inf_grid = np.full((H, W), np.inf, dtype=np.float32)
        pred_none = np.full(N, -9999, dtype=np.int64)
        return (inf_grid, inf_grid.copy(), pred_none)
    (dist, pred) = sp_dijkstra(graph, indices=src, return_predecessors=True)
    dist_flat = dist.astype(np.float32)
    pred_flat = pred.astype(np.int64)
    sqrt2 = float(np.sqrt(2.0))
    dist_pix_flat = np.full(N, np.inf, dtype=np.float32)
    dist_pix_flat[src] = 0.0
    order = np.argsort(dist_flat, kind="mergesort")
    for v in order:
        v = int(v)
        if v == src:
            continue
        if not np.isfinite(dist_flat[v]):
            break
        p = int(pred_flat[v])
        if p < 0 or not np.isfinite(dist_pix_flat[p]):
            continue
        (vy, vx) = divmod(v, W)
        (py, px) = divmod(p, W)
        manhattan = abs(vy - py) + abs(vx - px)
        step_m = float(res) * (sqrt2 if manhattan == 2 else 1.0)
        dist_pix_flat[v] = dist_pix_flat[p] + step_m
    return (dist_flat.reshape(H, W), dist_pix_flat.reshape(H, W), pred_flat)


def compute_global_geodesic_field(
    *,
    pathfinder: Any,
    base_pos: np.ndarray,
    forward_w: np.ndarray,
    left_w: np.ndarray,
    goal_world: np.ndarray,
    bev_x_max: float,
    res: float = 0.2,
    margin_m: float = 1.5,
    safe_radius_cells: float = 0.0,
    safety_cost_weight: float = 0.0,
) -> dict[str, Any]:
    """Solve geodesics on a grid covering the sector and goal plus margin.

    D is the cost-weighted distance that controls attraction direction.
    D_pix_m is geometric path length along the same predecessor tree, in
    metres, and controls vector magnitude. Edge costs use
    1 + safety_cost_weight * max(0, safe_radius_cells - obstacle_distance)
    with distances in cells. The returned pred, walkable, grid axes and
    goal_cell are required by sampling and trajectory extraction."""
    base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
    forward_w = np.asarray(forward_w, dtype=np.float64).reshape(3)
    left_w = np.asarray(left_w, dtype=np.float64).reshape(3)
    goal_world = np.asarray(goal_world, dtype=np.float64).reshape(3)
    d = goal_world - base_pos
    fwd2 = np.array([forward_w[0], forward_w[2]], dtype=np.float64)
    lft2 = np.array([left_w[0], left_w[2]], dtype=np.float64)
    d2 = np.array([d[0], d[2]], dtype=np.float64)
    goal_fwd = float(np.dot(d2, fwd2))
    goal_lft = float(np.dot(d2, lft2))
    (fwd_min, fwd_max, lft_min, lft_max) = _build_world_grid_extents(
        bev_x_max=bev_x_max, goal_fwd=goal_fwd, goal_lft=goal_lft, margin_m=margin_m
    )
    H_w = max(2, int(round((fwd_max - fwd_min) / res)))
    W_w = max(2, int(round((lft_max - lft_min) / res)))
    fwd_vals = np.linspace(fwd_min + res / 2, fwd_max - res / 2, H_w, dtype=np.float32)
    lft_vals = np.linspace(lft_min + res / 2, lft_max - res / 2, W_w, dtype=np.float32)
    (fwd_grid, lft_grid) = np.meshgrid(fwd_vals, lft_vals, indexing="ij")
    world_pts = _ego_to_world(
        fwd_grid.astype(np.float64), lft_grid.astype(np.float64), base_pos, forward_w, left_w
    )
    a_r_seed = int(round((0.0 - float(fwd_vals[0])) / res))
    a_c_seed = int(round((0.0 - float(lft_vals[0])) / res))
    a_r_seed = max(0, min(H_w - 1, a_r_seed))
    a_c_seed = max(0, min(W_w - 1, a_c_seed))
    walkable = _rasterise_world_walkable(
        pathfinder, world_pts, floor_y=float(base_pos[1]), agent_cell=(a_r_seed, a_c_seed)
    )
    goal_cell = _snap_goal_to_grid(goal_fwd, goal_lft, walkable, fwd_vals, lft_vals)
    if goal_cell is None:
        D = np.full((H_w, W_w), np.inf, dtype=np.float32)
        D_pix_m = D.copy()
        pred = np.full(H_w * W_w, -9999, dtype=np.int64)
    else:
        cost_map: np.ndarray | None = None
        if float(safe_radius_cells) > 0.0 and float(safety_cost_weight) > 0.0:
            cost_map = _build_world_safety_cost_map(
                walkable,
                safe_radius_cells=float(safe_radius_cells),
                safety_cost_weight=float(safety_cost_weight),
            )
        graph = _build_grid_graph_uniform(walkable, res=float(res), cost_map=cost_map)
        (D, D_pix_m, pred) = _dijkstra_from_cell(graph, walkable, goal_cell, res=float(res))
    return {
        "D": D,
        "D_pix_m": D_pix_m,
        "pred": pred,
        "walkable": walkable,
        "fwd_vals": fwd_vals,
        "lft_vals": lft_vals,
        "res": float(res),
        "goal_fwd": goal_fwd,
        "goal_lft": goal_lft,
        "goal_cell": goal_cell,
    }


def sample_global_field_at(
    field: dict[str, Any], fwd_query: np.ndarray, lft_query: np.ndarray
) -> np.ndarray:
    """Bilinear interpolation of the global geodesic field at ego-frame
    points ``(fwd_query, lft_query)``.

    Out-of-grid queries return ``np.inf``. Cells outside ``walkable`` are
    treated as ``np.inf``; bilinear weights renormalise across the valid
    neighbourhood so a query close to a boundary still gets a sensible
    finite value as long as ≥ 1 of the 4 corners is walkable.

    Parameters
    ----------
    field
        Output of :func:`compute_global_geodesic_field`.
    fwd_query, lft_query
        Same-shape arrays of ego-frame forward / left coordinates (m).

    Returns
    -------
    np.ndarray
        Same shape as the query — geodesic distance in metres,
        ``np.inf`` where invalid.
    """
    D = field["D"]
    walkable = field["walkable"]
    fwd_vals = field["fwd_vals"]
    lft_vals = field["lft_vals"]
    res = float(field["res"])
    (H_w, W_w) = D.shape
    fwd_q = np.asarray(fwd_query, dtype=np.float64)
    lft_q = np.asarray(lft_query, dtype=np.float64)
    out_shape = fwd_q.shape
    fwd0 = float(fwd_vals[0])
    lft0 = float(lft_vals[0])
    rf = (fwd_q - fwd0) / res
    cf = (lft_q - lft0) / res
    r0 = np.floor(rf).astype(np.int32)
    c0 = np.floor(cf).astype(np.int32)
    r1 = r0 + 1
    c1 = c0 + 1
    tr = (rf - r0).astype(np.float32)
    tc = (cf - c0).astype(np.float32)
    in_bounds = (r0 >= 0) & (c0 >= 0) & (r1 < H_w) & (c1 < W_w)
    out = np.full(out_shape, np.inf, dtype=np.float32)
    if not in_bounds.any():
        return out
    r0c = np.clip(r0, 0, H_w - 1)
    r1c = np.clip(r1, 0, H_w - 1)
    c0c = np.clip(c0, 0, W_w - 1)
    c1c = np.clip(c1, 0, W_w - 1)
    D00 = D[r0c, c0c]
    W00 = walkable[r0c, c0c]
    D01 = D[r0c, c1c]
    W01 = walkable[r0c, c1c]
    D10 = D[r1c, c0c]
    W10 = walkable[r1c, c0c]
    D11 = D[r1c, c1c]
    W11 = walkable[r1c, c1c]
    F00 = np.isfinite(D00) & W00
    F01 = np.isfinite(D01) & W01
    F10 = np.isfinite(D10) & W10
    F11 = np.isfinite(D11) & W11
    w00 = (1 - tr) * (1 - tc) * F00.astype(np.float32)
    w01 = (1 - tr) * tc * F01.astype(np.float32)
    w10 = tr * (1 - tc) * F10.astype(np.float32)
    w11 = tr * tc * F11.astype(np.float32)
    wsum = w00 + w01 + w10 + w11
    has_any = wsum > 1e-06
    val = np.zeros(out_shape, dtype=np.float32)
    val[has_any] = (
        w00[has_any] * np.where(F00[has_any], D00[has_any], 0.0)
        + w01[has_any] * np.where(F01[has_any], D01[has_any], 0.0)
        + w10[has_any] * np.where(F10[has_any], D10[has_any], 0.0)
        + w11[has_any] * np.where(F11[has_any], D11[has_any], 0.0)
    ) / wsum[has_any]
    out[in_bounds & has_any] = val[in_bounds & has_any]
    return out


def walk_global_pred_path(
    field: dict[str, Any],
    agent_fwd: float,
    agent_lft: float,
    *,
    stop_D: float | None = None,
    stop_field_key: str = "D_pix_m",
    sector_mask_bev: np.ndarray | None = None,
    bev_fwd: np.ndarray | None = None,
    bev_lft: np.ndarray | None = None,
    bev_resolution: float | None = None,
) -> tuple[np.ndarray, float]:
    """Walk the global predecessor tree toward its source.

    Stop at the optional distance threshold or a visibility-sector boundary.
    Short excursions outside visibility may be shortened only when the
    connecting segment is walkable. Returns body-frame points, geometric
    length and a termination reason distinguishing arrival, occlusion and
    unreachable geometry."""
    sector_clip = sector_mask_bev is not None
    if sector_clip:
        if bev_fwd is None or bev_lft is None or bev_resolution is None:
            raise ValueError("sector_mask_bev requires bev_fwd, bev_lft, bev_resolution")
        (H_bev, W_bev) = sector_mask_bev.shape
        bev_fwd0 = float(bev_fwd[0, 0])
        bev_lft0 = float(bev_lft[0, 0])
        bev_res_f = float(bev_resolution)
    D = field["D"]
    pred = field["pred"]
    walkable = field["walkable"]
    fwd_vals = field["fwd_vals"]
    lft_vals = field["lft_vals"]
    res = float(field["res"])
    goal_cell = field.get("goal_cell")
    (H_w, W_w) = D.shape
    N = H_w * W_w
    stop_field = field[stop_field_key]
    if not walkable.any() or goal_cell is None:
        return (
            np.array([[float(agent_fwd), float(agent_lft)]], dtype=np.float32),
            0.0,
            "degenerate",
        )
    rf = (float(agent_fwd) - float(fwd_vals[0])) / res
    cf = (float(agent_lft) - float(lft_vals[0])) / res
    a_r = int(round(rf))
    a_c = int(round(cf))
    a_r = max(0, min(H_w - 1, a_r))
    a_c = max(0, min(W_w - 1, a_c))
    if not walkable[a_r, a_c]:
        (ws_r, ws_c) = np.where(walkable)
        if ws_r.size == 0:
            return (
                np.array([[float(agent_fwd), float(agent_lft)]], dtype=np.float32),
                0.0,
                "degenerate",
            )
        k = int(np.argmin((ws_r - a_r) ** 2 + (ws_c - a_c) ** 2))
        (a_r, a_c) = (int(ws_r[k]), int(ws_c[k]))
    cells: list[tuple[int, int]] = []
    chain_full: list[tuple[int, int, bool]] = []
    cur = a_r * W_w + a_c
    visited: set = set()
    g_idx = int(goal_cell[0]) * W_w + int(goal_cell[1]) if goal_cell is not None else -1
    term_reason = "ok"

    def _in_sector(r: int, c: int) -> bool:
        if not sector_clip:
            return True
        cell_fwd = float(fwd_vals[r])
        cell_lft = float(lft_vals[c])
        br = int(round((cell_fwd - bev_fwd0) / bev_res_f))
        bc = int(round((cell_lft - bev_lft0) / bev_res_f))
        if not (0 <= br < H_bev and 0 <= bc < W_bev):
            return False
        return bool(sector_mask_bev[br, bc])

    def _line_all_walkable(r0: int, c0: int, r1: int, c1: int) -> bool:
        """Bresenham — every cell on the segment is walkable. No length
        cap: the splice mechanism is anchored at the agent and runs at
        most once per walk, so a long-but-physically-walkable Bresenham
        line is itself the proof that the shortcut is a valid traversal."""
        dr_ = abs(r1 - r0)
        sr_ = 1 if r0 < r1 else -1
        dc_ = abs(c1 - c0)
        sc_ = 1 if c0 < c1 else -1
        err = dr_ - dc_
        (r, c) = (r0, c0)
        while True:
            if not (0 <= r < H_w and 0 <= c < W_w):
                return False
            if not walkable[r, c]:
                return False
            if r == r1 and c == c1:
                return True
            e2 = 2 * err
            if e2 > -dc_:
                err -= dc_
                r += sr_
            if e2 < dr_:
                err += dr_
                c += sc_

    while True:
        if cur in visited:
            term_reason = "cycle"
            break
        visited.add(cur)
        (r, c) = divmod(cur, W_w)
        chain_full.append((r, c, _in_sector(r, c)))
        if cur == g_idx:
            term_reason = "goal_reached"
            break
        if stop_D is not None and float(stop_field[r, c]) <= float(stop_D):
            term_reason = "stop_D"
            break
        nxt = int(pred[cur])
        if nxt < 0 or nxt >= N:
            term_reason = "unreachable"
            break
        cur = nxt
    truncated = False
    if chain_full:
        cells.append((chain_full[0][0], chain_full[0][1]))
        i = 1
        while i < len(chain_full):
            (r, c, ok) = chain_full[i]
            if ok:
                cells.append((r, c))
                i += 1
                continue
            j = i
            while j < len(chain_full) and (not chain_full[j][2]):
                j += 1
            if j >= len(chain_full):
                truncated = True
                break
            (last_r, last_c) = cells[-1]
            (re_r, re_c, _) = chain_full[j]
            if _line_all_walkable(last_r, last_c, re_r, re_c):
                cells.append((re_r, re_c))
                i = j + 1
            else:
                truncated = True
                break
    if truncated:
        if cells:
            (last_r, last_c) = cells[-1]
            last_idx = last_r * W_w + last_c
            if last_idx == g_idx:
                term_reason = "goal_reached"
            elif stop_D is not None and float(stop_field[last_r, last_c]) <= float(stop_D):
                term_reason = "stop_D"
            else:
                term_reason = "sector_clip"
        else:
            term_reason = "sector_clip"
    if len(cells) == 1:
        return (
            np.array([[float(agent_fwd), float(agent_lft)]], dtype=np.float32),
            0.0,
            term_reason,
        )
    coords = np.zeros((1 + len(cells), 2), dtype=np.float32)
    coords[0, 0] = float(agent_fwd)
    coords[0, 1] = float(agent_lft)
    for i, (r, c) in enumerate(cells):
        coords[i + 1, 0] = float(fwd_vals[r])
        coords[i + 1, 1] = float(lft_vals[c])
    diffs = np.diff(coords, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return (coords, float(seg_lens.sum()), term_reason)
