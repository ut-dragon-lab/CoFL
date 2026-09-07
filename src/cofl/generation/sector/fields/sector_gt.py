"""Cartesian sector fields: global geodesic attraction with local obstacle escape.

Grid centers span forward [0, extent] and left [-extent, extent]. Vector
components are (forward, left) in metres divided by v_norm. The velocity
field covers the full grid; bev_mask separately records walkable, visible
sector cells. Trajectories follow the same global predecessor tree, clipped
to the perceivable sector and resampled to horizon + 1 body-frame points."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .geom import as_numpy, quat_xyzw_to_rotmat
from .global_geodesic import (
    _build_grid_graph_uniform,
    _build_world_safety_cost_map,
    compute_global_geodesic_field,
    sample_global_field_at,
    walk_global_pred_path,
)


def _build_bev_grid(
    bev_x_max: float, bev_res: float
) -> tuple[np.ndarray, np.ndarray, int, int, float, float]:
    """Create a uniform Cartesian BEV grid centred on the agent.

    The agent foot ``(x=0, y=0)`` is the geometric centre of cell
    ``(row=0, col=W//2)``. Forward axis spans ``H_bev = K + 1`` cells with
    centres at ``fwd ∈ {0, bev_res, 2·bev_res, ..., K·bev_res}``; lateral
    axis spans ``W_bev = 2·K + 1`` cells with centres at ``lft ∈ {-K·bev_res,
    ..., -bev_res, 0, +bev_res, ..., +K·bev_res}``, where ``K = round(bev_x_max
    / bev_res)``.

    Why ``(0, 0)`` must be a cell centre. Under the older even-W layout,
    cell centres sat at ``±bev_res/2`` flanking the y-axis and at
    ``+bev_res/2`` ahead of agent foot, so the dijkstra path-resampler that
    starts at the agent foot and ends at the goal cell centre carried a
    half-cell residue (~ ``bev_res / √2`` ≈ 0.07 m) for any STOP-snap
    frame whose goal is the agent itself. That residue is small but
    non-zero and clusters the STOP-slot trajectory at ``(±bev_res/2,
    ±bev_res/2)`` — geometrically indistinguishable from the tail of a
    real TURN trajectory in action bucketing. Centring a cell on the
    agent foot eliminates the residue at the geometry level: ``agent_cell
    == goal_cell`` for stop-snap → dijkstra walks zero cells →
    resampler emits an exact ``(0, 0) × (H+1)`` trajectory; ``V_bev`` at
    the agent cell is the gradient at the geodesic source, which is
    naturally zero in a radially-symmetric neighbourhood. Both signals
    become exact, no special-case branches needed.

    Note that ``bev_x_max`` now denotes the *cell-centre* coordinate of
    the farthest cell, not the outer edge — the actual cell-extent
    boundary is at ``bev_x_max + bev_res/2``. For the default
    ``bev_x_max = 5.0`` / ``bev_res = 0.1`` this gives ``H_bev = 51``,
    ``W_bev = 101``, fwd-extent ``[-0.05, 5.05] m``, lft-extent
    ``[-5.05, 5.05] m``.

    The agent's pose only enters at velocity-field interpretation time;
    the grid itself is purely local (no base_pos / forward_w dependency),
    so this function is pose-independent — the same grid is reused across
    frames.

    Returns
    -------
    bev_fwd : (H_bev, W_bev) forward-axis offset of each cell centre, metres
    bev_lft : (H_bev, W_bev) left-axis offset of each cell centre, metres
    H_bev, W_bev : grid dimensions (both odd)
    fwd_origin, lft_origin : cell-centre coords of BEV[0, 0] = (0, -K·bev_res)
    """
    if bev_res <= 0:
        raise ValueError(f"bev_res must be > 0, got {bev_res}")
    if bev_x_max <= 0:
        raise ValueError(f"bev_x_max must be > 0, got {bev_x_max}")
    K = max(1, int(round(float(bev_x_max) / float(bev_res))))
    H_bev = K + 1
    W_bev = 2 * K + 1
    fwd_vals = float(bev_res) * np.arange(H_bev, dtype=np.float32)
    lft_vals = float(bev_res) * (np.arange(W_bev, dtype=np.float32) - float(K))
    (bev_lft, bev_fwd) = np.meshgrid(lft_vals, fwd_vals, indexing="xy")
    return (bev_fwd, bev_lft, H_bev, W_bev, float(fwd_vals[0]), float(lft_vals[0]))


def _rasterise_bev_walkable(
    pathfinder,
    base_pos,
    forward_w,
    left_w,
    bev_fwd,
    bev_lft,
    snap_tol_m=0.5,
    stairs_aabbs=None,
    stairs_xz_pad_m=0.2,
):
    """Probe BEV cells at floor height, or their actual tread height inside stairs."""
    world = base_pos + bev_fwd[..., None] * forward_w + bev_lft[..., None] * left_w
    points = world.reshape(-1, 3)
    in_stairs = np.zeros(len(points), dtype=bool)
    for low, high in stairs_aabbs or []:
        in_stairs |= (
            (points[:, 0] >= low[0] - stairs_xz_pad_m)
            & (points[:, 0] <= high[0] + stairs_xz_pad_m)
            & (points[:, 2] >= low[2] - stairs_xz_pad_m)
            & (points[:, 2] <= high[2] + stairs_xz_pad_m)
        )
    flags = np.zeros(len(points), dtype=bool)
    for index, point in enumerate(points):
        if in_stairs[index]:
            snapped = np.asarray(pathfinder.snap_point(point), dtype=np.float32)
            if (
                not np.isfinite(snapped).all()
                or np.linalg.norm(snapped[[0, 2]] - point[[0, 2]]) > snap_tol_m
            ):
                continue
            point = snapped
        flags[index] = pathfinder.is_navigable(point, snap_tol_m)
    return flags.reshape(bev_fwd.shape)


def _polyline_at(pts: np.ndarray, d: float) -> np.ndarray:
    """Interpolate a 3-D polyline at a nonnegative arc length."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    d = max(0.0, float(d))
    if pts.shape[0] <= 1 or d <= 1e-08:
        return pts[0].copy()
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    rem = d
    for i in range(seg.shape[0]):
        L = float(seg_len[i])
        if L < 1e-08:
            continue
        if rem <= L:
            return (pts[i] + rem / L * seg[i]).astype(np.float32)
        rem -= L
    return pts[-1].copy()


def _point_is_visible_in_sector(
    point_world: np.ndarray,
    base_pos: np.ndarray,
    forward_w: np.ndarray,
    left_w: np.ndarray,
    hfov_rad: float,
    z_max: float,
    d_col: np.ndarray | None = None,
    *,
    depth_visibility_tol_m: float,
) -> bool:
    """Check whether a world point lies inside the current visible sector."""
    d = (point_world - base_pos).astype(np.float64)
    fwd2 = np.array([forward_w[0], forward_w[2]], dtype=np.float64)
    lft2 = np.array([left_w[0], left_w[2]], dtype=np.float64)
    d2 = np.array([d[0], d[2]], dtype=np.float64)
    x_f = float(np.dot(d2, fwd2))
    y_l = float(np.dot(d2, lft2))
    r_m = math.sqrt(x_f * x_f + y_l * y_l)
    if r_m < 1e-06:
        return True
    half_fov = hfov_rad / 2.0
    theta = math.atan2(y_l, max(x_f, 1e-08))
    in_fov = x_f >= 0.0 and abs(theta) <= half_fov + 0.001 and (r_m <= z_max + 0.001)
    if not in_fov or d_col is None:
        return in_fov
    tan_half = math.tan(half_fov)
    if tan_half <= 1e-08:
        return in_fov
    W_d = int(d_col.shape[0])
    col_f = (W_d - 1) / 2.0 * (1.0 - math.tan(theta) / tan_half)
    col_f = float(np.clip(col_f, 0.0, float(W_d - 1)))
    d_limit = float(np.interp(col_f, np.arange(W_d, dtype=np.float64), d_col))
    return r_m <= d_limit + float(depth_visibility_tol_m)


def _compute_reference_progress(
    reference_path_world: np.ndarray, base_pos: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Project the current pose onto a reference polyline and return progress."""
    pts = np.asarray(reference_path_world, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] <= 1:
        return (pts, 0.0, 0.0)
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    total_len = float(seg_len.sum())
    best_s = 0.0
    best_dist2 = float("inf")
    cum_len = 0.0
    for i, L in enumerate(seg_len):
        if L <= 1e-08:
            continue
        direction = seg[i] / L
        t = float(np.clip(np.dot(base_pos - pts[i], direction) / L, 0.0, 1.0))
        proj = pts[i] + t * seg[i]
        dist2 = float(np.sum((proj - base_pos) ** 2))
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_s = cum_len + t * L
        cum_len += float(L)
    return (pts, total_len, best_s)


def _shrink_turn_radius_to_walkable(
    half_fov: float,
    side: float,
    M_free: np.ndarray,
    bev_fwd: np.ndarray,
    bev_lft: np.ndarray,
    bev_resolution: float,
    *,
    max_radius_m: float,
    min_radius_m: float,
) -> float:
    """Choose the largest walkable FoV-edge radius in grid-cell steps.

    If no candidate is free, retain the minimum one-cell translation radius."""
    res = float(bev_resolution)
    if res <= 0.0:
        return float(min_radius_m)
    H = int(M_free.shape[0])
    W = int(M_free.shape[1])
    fwd0 = float(bev_fwd[0, 0])
    lft0 = float(bev_lft[0, 0])
    radii = []
    r = float(max_radius_m)
    floor = float(min_radius_m)
    while r >= floor - 1e-06:
        radii.append(r)
        r -= res
    if not radii or radii[-1] > floor + 1e-06:
        radii.append(floor)
    for radius_m in radii:
        edge_fwd = radius_m * math.cos(half_fov)
        edge_lft = radius_m * math.sin(half_fov) * side
        rr = int(round((edge_fwd - fwd0) / res))
        cc = int(round((edge_lft - lft0) / res))
        if 0 <= rr < H and 0 <= cc < W and bool(M_free[rr, cc]):
            return float(radius_m)
    return float(min_radius_m)


def _build_reference_turn_anchor(
    reference_path_world: np.ndarray,
    base_pos: np.ndarray,
    forward_w: np.ndarray,
    left_w: np.ndarray,
    hfov_rad: float,
    *,
    sample_step_m: float,
    turn_theta_abs_tilde: float = 1.0,
    turn_radius_m: float = 0.5,
    M_free: np.ndarray | None = None,
    bev_fwd: np.ndarray | None = None,
    bev_lft: np.ndarray | None = None,
    bev_resolution: float | None = None,
    min_turn_radius_m: float = 0.1,
) -> np.ndarray | None:
    """Choose a FoV-edge turn target from the future reference heading.

    Shrink its radius to a walkable grid cell to avoid placing the anchor
    in a wall. The default radius is 0.5 m; the minimum is one grid cell."""
    (pts, total_len, best_s) = _compute_reference_progress(reference_path_world, base_pos)
    if pts.shape[0] == 0:
        return None
    probe0_s = min(total_len, best_s + max(float(sample_step_m), 0.1))
    probe1_s = min(total_len, probe0_s + max(float(sample_step_m), 0.3))
    p0 = _polyline_at(pts, probe0_s)
    p1 = _polyline_at(pts, probe1_s)
    future_vec = (p1 - p0).astype(np.float64)
    lookahead_vec = (p1 - base_pos).astype(np.float64)
    tan2 = np.array([future_vec[0], future_vec[2]], dtype=np.float64)
    look2 = np.array([lookahead_vec[0], lookahead_vec[2]], dtype=np.float64)
    if np.linalg.norm(look2) <= 1e-08:
        lookahead_fallback = (p0 - base_pos).astype(np.float64)
        look2 = np.array([lookahead_fallback[0], lookahead_fallback[2]], dtype=np.float64)
    if np.linalg.norm(tan2) <= 1e-08:
        tan2 = look2.copy()
    if np.linalg.norm(tan2) <= 1e-08:
        return None
    fwd2 = np.array([forward_w[0], forward_w[2]], dtype=np.float64)
    lft2 = np.array([left_w[0], left_w[2]], dtype=np.float64)
    lateral = float(np.dot(look2, lft2))
    forward = float(np.dot(look2, fwd2))
    tan_lateral = float(np.dot(tan2, lft2))
    tan_forward = float(np.dot(tan2, fwd2))
    if abs(lateral) > 1e-06:
        theta_tilde = float(np.sign(lateral) * float(turn_theta_abs_tilde))
    elif forward < 0.0:
        theta_tilde = float(turn_theta_abs_tilde)
    elif abs(tan_lateral) > 1e-06:
        theta_tilde = float(np.sign(tan_lateral) * float(turn_theta_abs_tilde))
    elif tan_forward < 0.0:
        theta_tilde = float(turn_theta_abs_tilde)
    else:
        theta_tilde = 0.0
    theta_rad = theta_tilde * (float(hfov_rad) / 2.0)
    radius_m = float(turn_radius_m)
    if (
        M_free is not None
        and bev_fwd is not None
        and (bev_lft is not None)
        and (bev_resolution is not None)
        and (abs(theta_rad) > 1e-06)
    ):
        side = 1.0 if theta_rad > 0.0 else -1.0
        radius_m = _shrink_turn_radius_to_walkable(
            half_fov=abs(theta_rad),
            side=side,
            M_free=M_free,
            bev_fwd=bev_fwd,
            bev_lft=bev_lft,
            bev_resolution=float(bev_resolution),
            max_radius_m=float(turn_radius_m),
            min_radius_m=float(min_turn_radius_m),
        )
    anchor = (
        base_pos
        + radius_m * math.cos(theta_rad) * forward_w
        + radius_m * math.sin(theta_rad) * left_w
    )
    return np.asarray(anchor, dtype=np.float32).reshape(3)


def _select_visible_reference_target(
    reference_path_world: np.ndarray,
    base_pos: np.ndarray,
    forward_w: np.ndarray,
    left_w: np.ndarray,
    hfov_rad: float,
    z_max: float,
    d_col: np.ndarray | None = None,
    *,
    depth_visibility_tol_m: float,
    sample_step_m: float,
    M_free: np.ndarray | None = None,
    bev_fwd: np.ndarray | None = None,
    bev_lft: np.ndarray | None = None,
    bev_resolution: float | None = None,
    M_walkable: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str]:
    """Select a stable local target on the future suffix of a reference path.

    Priority:
    1. Farthest future point that is depth-visible in the current sector.
    2. A small-radius turn anchor inferred from the future reference heading.
    3. No target found.
    """
    (pts, total_len, best_s) = _compute_reference_progress(reference_path_world, base_pos)
    if pts.shape[0] == 0:
        return (None, "none")
    sample_step_m = max(float(sample_step_m), 0.05)
    if pts.shape[0] == 1 or total_len <= 1e-08:
        only = pts[-1].copy()
        if _point_is_visible_in_sector(
            only,
            base_pos,
            forward_w,
            left_w,
            hfov_rad,
            z_max,
            d_col,
            depth_visibility_tol_m=depth_visibility_tol_m,
        ):
            return (only, "reference_path_visible_farthest")
        turn_anchor = _build_reference_turn_anchor(
            pts,
            base_pos,
            forward_w,
            left_w,
            hfov_rad,
            sample_step_m=sample_step_m,
            M_free=M_free,
            bev_fwd=bev_fwd,
            bev_lft=bev_lft,
            bev_resolution=bev_resolution,
        )
        if turn_anchor is not None:
            return (turn_anchor, "reference_path_turn_anchor")
        return (None, "none")
    n_samples = max(2, int(math.ceil(max(total_len - best_s, 0.0) / sample_step_m)) + 1)
    min_progress_m = min(sample_step_m, 0.1)
    base2 = np.array([base_pos[0], base_pos[2]], dtype=np.float64)
    fwd2 = np.array([forward_w[0], forward_w[2]], dtype=np.float64)
    lft2 = np.array([left_w[0], left_w[2]], dtype=np.float64)
    behind_reentry_tol_m = max(3.0 * sample_step_m, 0.3)
    behind_run_start_s: float | None = None

    def _line_walkable_world(a_world: np.ndarray, b_world: np.ndarray) -> bool:
        if M_walkable is None or bev_fwd is None or bev_lft is None or (bev_resolution is None):
            return False
        a3 = np.asarray(a_world, dtype=np.float64).reshape(3) - np.asarray(
            base_pos, dtype=np.float64
        ).reshape(3)
        b3 = np.asarray(b_world, dtype=np.float64).reshape(3) - np.asarray(
            base_pos, dtype=np.float64
        ).reshape(3)
        a_xz = np.array([float(a3[0]), float(a3[2])], dtype=np.float64)
        b_xz = np.array([float(b3[0]), float(b3[2])], dtype=np.float64)
        a_fwd_ego = float(np.dot(a_xz, fwd2))
        a_lft_ego = float(np.dot(a_xz, lft2))
        b_fwd_ego = float(np.dot(b_xz, fwd2))
        b_lft_ego = float(np.dot(b_xz, lft2))
        bev_res_f = float(bev_resolution)
        (H_bev, W_bev) = M_walkable.shape
        a_r = int(round((a_fwd_ego - float(bev_fwd[0, 0])) / bev_res_f))
        a_c = int(round((a_lft_ego - float(bev_lft[0, 0])) / bev_res_f))
        b_r = int(round((b_fwd_ego - float(bev_fwd[0, 0])) / bev_res_f))
        b_c = int(round((b_lft_ego - float(bev_lft[0, 0])) / bev_res_f))
        if not (0 <= b_r < H_bev and 0 <= b_c < W_bev):
            return False
        dr_ = abs(b_r - a_r)
        sr_ = 1 if a_r < b_r else -1
        dc_ = abs(b_c - a_c)
        sc_ = 1 if a_c < b_c else -1
        err = dr_ - dc_
        (rr, cc) = (a_r, a_c)
        while True:
            if not (0 <= rr < H_bev and 0 <= cc < W_bev):
                return False
            if not M_walkable[rr, cc]:
                return False
            if rr == b_r and cc == b_c:
                return True
            e2 = 2 * err
            if e2 > -dc_:
                err -= dc_
                rr += sr_
            if e2 < dr_:
                err += dr_
                cc += sc_

    case: str | None = None
    last_visible_in_interval: np.ndarray | None = None
    final_exit_target: np.ndarray | None = None
    in_invisible_segment: bool = False
    for k in range(n_samples):
        s = min(total_len, best_s + k * sample_step_m)
        if s < best_s + min_progress_m and k != n_samples - 1:
            continue
        p = _polyline_at(pts, s)
        d2 = np.array([p[0], p[2]], dtype=np.float64) - base2
        x_f = float(np.dot(d2, fwd2))
        if x_f < 0.0:
            if behind_run_start_s is None:
                behind_run_start_s = float(s)
            elif float(s) - behind_run_start_s > behind_reentry_tol_m:
                break
            continue
        behind_run_start_s = None
        in_fov = _point_is_visible_in_sector(
            p,
            base_pos,
            forward_w,
            left_w,
            hfov_rad,
            z_max,
            d_col=None,
            depth_visibility_tol_m=depth_visibility_tol_m,
        )
        if case is None:
            case = "A" if in_fov else "B-pending"
        if not in_fov:
            if case == "B-pending":
                continue
            if not in_invisible_segment:
                if last_visible_in_interval is not None:
                    final_exit_target = last_visible_in_interval
                    last_visible_in_interval = None
                in_invisible_segment = True
            continue
        if case == "B-pending":
            if not _line_walkable_world(base_pos, p):
                break
            case = "B-confirmed"
        elif in_invisible_segment:
            if final_exit_target is None or not _line_walkable_world(final_exit_target, p):
                break
            in_invisible_segment = False
        depth_vis = _point_is_visible_in_sector(
            p,
            base_pos,
            forward_w,
            left_w,
            hfov_rad,
            z_max,
            d_col,
            depth_visibility_tol_m=depth_visibility_tol_m,
        )
        if depth_vis:
            last_visible_in_interval = p
        else:
            if last_visible_in_interval is not None:
                final_exit_target = last_visible_in_interval
                last_visible_in_interval = None
            break
    if last_visible_in_interval is not None:
        return (None, "none")
    if final_exit_target is not None:
        return (final_exit_target.astype(np.float32), "reference_path_visible_farthest")
    turn_anchor = _build_reference_turn_anchor(
        pts,
        base_pos,
        forward_w,
        left_w,
        hfov_rad,
        sample_step_m=sample_step_m,
        M_free=M_free,
        bev_fwd=bev_fwd,
        bev_lft=bev_lft,
        bev_resolution=bev_resolution,
    )
    if turn_anchor is not None:
        return (turn_anchor.astype(np.float32), "reference_path_turn_anchor")
    return (None, "none")


def _compute_depth_column_limits(
    depth_image_hw: np.ndarray, *, z_max: float, depth_valid_mask_hw: np.ndarray | None = None
) -> np.ndarray:
    """Convert a depth image into conservative per-column visibility limits.

    We use a wider central band and ignore invalid depth pixels before taking
    the per-column minimum. If a column has no valid depth in the band, treat
    it as open space up to ``z_max`` rather than fully occluded.

    Habitat depth can legitimately return zero for "no hit" / "no return"
    columns in open doorways or long sightlines. Also, if depth is resized with
    bilinear filtering, invalid regions can smear into tiny positive values.
    When a validity mask is available, trust it instead of raw ``depth > 0`` so
    those interpolated pixels do not create fake near-field walls.
    """
    depth_m = np.asarray(depth_image_hw, dtype=np.float32)
    if depth_m.ndim == 3:
        depth_m = depth_m[:, :, 0]
    valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
    if depth_valid_mask_hw is not None:
        depth_valid = np.asarray(depth_valid_mask_hw)
        if depth_valid.ndim == 3:
            depth_valid = depth_valid[:, :, 0]
        depth_valid = depth_valid.astype(bool, copy=False)
        if depth_valid.shape != depth_m.shape:
            raise ValueError(
                f"depth_valid_mask_hw shape {depth_valid.shape} must match depth_image_hw shape {depth_m.shape}"
            )
        valid_mask = valid_mask & depth_valid
    (H_d, W_d) = depth_m.shape[:2]
    row_c = H_d // 2
    half_band = max(2, H_d // 14)
    band = depth_m[max(0, row_c - half_band) : min(H_d, row_c + half_band + 1), :]
    band_mask = valid_mask[max(0, row_c - half_band) : min(H_d, row_c + half_band + 1), :]
    band_valid = np.where(band_mask, band, np.inf)
    d_col = band_valid.min(axis=0)
    d_col = np.where(np.isfinite(d_col), d_col, float(z_max))
    d_col = np.clip(d_col, 0.0, float(z_max)).astype(np.float64)
    return d_col


def _compute_bev_navmesh_y_grid(
    pathfinder: Any,
    base_pos: np.ndarray,
    forward_w: np.ndarray,
    left_w: np.ndarray,
    bev_fwd: np.ndarray,
    bev_lft: np.ndarray,
    horiz_drift_tol_m: float = 0.1,
) -> np.ndarray:
    """Per-BEV-cell navmesh height grid: ``snap_point((x, base_pos.y, z))``
    at each cell, keeping the snapped Y when horizontal drift stays
    below ``horiz_drift_tol_m`` (~one BEV cell). Cells where the snap
    drifts further (= deep inside a wall / off the navmesh) get
    ``NaN``. Used by :func:`_build_visible_bev_mask_3d` to project each
    cell's actual 3D position onto the depth image — for staircase
    scenes the floor below the agent is at a lower navmesh y, and
    using ``base_pos.y`` for the projection would mis-place those
    cells onto the wrong row of the depth image.
    """
    snap = getattr(pathfinder, "snap_point", None)
    (H, W) = bev_fwd.shape
    if snap is None:
        return np.full((H, W), float(base_pos[1]), dtype=np.float32)
    world_pts = (
        base_pos[None, None, :].astype(np.float32)
        + bev_fwd[:, :, None].astype(np.float32) * forward_w[None, None, :].astype(np.float32)
        + bev_lft[:, :, None].astype(np.float32) * left_w[None, None, :].astype(np.float32)
    )
    flat = world_pts.reshape(-1, 3)
    N = flat.shape[0]
    tol2 = float(horiz_drift_tol_m) ** 2
    y_flat = np.full(N, np.nan, dtype=np.float32)
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
        y_flat[i] = float(sp[1])
    return y_flat.reshape(H, W)


def _build_visible_bev_mask_3d(
    bev_fwd: np.ndarray,
    bev_lft: np.ndarray,
    y_grid: np.ndarray,
    base_pos: np.ndarray,
    forward_w: np.ndarray,
    left_w: np.ndarray,
    hfov_rad: float,
    z_max: float,
    depth_image_hw: np.ndarray,
    *,
    depth_visibility_tol_m: float,
) -> np.ndarray:
    """Project navmesh heights into the metric depth image to test visibility.

    Unavailable navmesh heights use the agent floor height. This maintains
    a 2-D floor estimate where no 3-D surface can be queried; one-cell
    dilation accounts for grid-to-image projection rounding."""
    (H_bev, W_bev) = bev_fwd.shape
    (H_d, W_d) = depth_image_hw.shape[:2]
    half_fov = hfov_rad / 2.0
    tan_half = math.tan(half_fov)
    if tan_half <= 1e-08:
        return np.zeros((H_bev, W_bev), dtype=bool)
    fx = W_d / 2.0 / tan_half
    fy = fx
    bev_fwd_f = np.asarray(bev_fwd, dtype=np.float32)
    bev_lft_f = np.asarray(bev_lft, dtype=np.float32)
    base_x = float(base_pos[0])
    base_y = float(base_pos[1])
    base_z = float(base_pos[2])
    fw_x = float(forward_w[0])
    fw_y = float(forward_w[1])
    fw_z = float(forward_w[2])
    lw_x = float(left_w[0])
    lw_y = float(left_w[1])
    lw_z = float(left_w[2])
    x_world = base_x + bev_fwd_f * fw_x + bev_lft_f * lw_x
    z_world = base_z + bev_fwd_f * fw_z + bev_lft_f * lw_z
    y_world = np.where(np.isfinite(y_grid), y_grid, np.float32(base_y))
    dx = x_world - base_x
    dy = y_world - base_y
    dz = z_world - base_z
    z_cam = dx * fw_x + dy * fw_y + dz * fw_z
    x_cam = -(dx * lw_x + dy * lw_y + dz * lw_z)
    v_cam = -dy
    eps_z = 0.05
    safe_z = np.where(z_cam > eps_z, z_cam, np.float32(1.0))
    u_px = W_d / 2.0 + x_cam / safe_z * fx
    v_px = H_d / 2.0 + v_cam / safe_z * fy
    in_front = z_cam > eps_z
    in_frame = (u_px >= 0.0) & (u_px < W_d) & (v_px >= 0.0) & (v_px < H_d)
    u_idx = np.clip(np.round(u_px).astype(np.int64), 0, W_d - 1)
    v_idx = np.clip(np.round(v_px).astype(np.int64), 0, H_d - 1)
    depth_at = np.asarray(depth_image_hw, dtype=np.float32)[v_idx, u_idx]
    visible = (
        in_front
        & in_frame
        & (z_cam <= depth_at + float(depth_visibility_tol_m))
        & (z_cam <= float(z_max) + 0.001)
    )
    from scipy.ndimage import binary_dilation as _binary_dilation

    visible = _binary_dilation(visible, structure=np.ones((3, 3), dtype=bool))
    return visible


def _resample_path_arclength(
    coords: np.ndarray, horizon: int, *, smooth_sigma: float = 1.5
) -> np.ndarray:
    """Resample a cart polyline at uniform arc-length, then Gaussian-smooth.

    The dijkstra polyline is a sequence of BEV cell centres connected by
    axis-aligned (length ``bev_res``) or diagonal (length ``√2·bev_res``)
    segments — visually a staircase. Pure linear-along-arc resampling
    inherits these per-cell kinks, which look jagged in the QA viewer.

    To match the polar pipeline's smooth-streamline aesthetic, this
    function applies a per-axis Gaussian filter along the sample-index
    dimension (uniform after the resample). Endpoints are pinned exactly
    to ``coords[0]`` and ``coords[-1]`` so cart[0] stays at the agent foot
    and cart[-1] stays at the goal cell centre, regardless of σ.

    ``smooth_sigma`` is in *sample-index* units. With H+1 ≈ 100 samples
    over a typical 4 m path (≈ 0.04 m / sample), σ = 1.5 smooths features
    shorter than ~0.18 m wavelength — enough to round off the cell-edge
    staircase while preserving real corners (e.g. doorways, > 0.5 m
    perpendicular displacements). Set ``smooth_sigma = 0`` to disable.

    ``out[0] = coords[0]`` (agent foot), ``out[-1] = coords[-1]`` (goal
    cell centre). Degenerate paths (length < 1e-6) collapse to a constant
    trajectory at ``coords[0]``.

    Returns
    -------
    out : (horizon + 1, 2) float32
    """
    H1 = int(horizon) + 1
    if coords.shape[0] < 2:
        return np.tile(coords[:1], (H1, 1)).astype(np.float32)
    diffs = np.diff(coords, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)]).astype(np.float64)
    total_len = float(cumlen[-1])
    if total_len < 1e-06:
        return np.tile(coords[:1], (H1, 1)).astype(np.float32)
    targets = np.linspace(0.0, total_len, H1, dtype=np.float64)
    seg_idx = np.searchsorted(cumlen, targets, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(seg_lens) - 1)
    out = np.zeros((H1, 2), dtype=np.float32)
    for i in range(H1):
        s = int(seg_idx[i])
        seg_len = seg_lens[s]
        if seg_len < 1e-12:
            out[i] = coords[s]
            continue
        ratio = float((targets[i] - cumlen[s]) / seg_len)
        out[i] = coords[s] + ratio * (coords[s + 1] - coords[s])
    sigma = float(smooth_sigma)
    if sigma > 0.0 and H1 >= 3:
        from scipy.ndimage import gaussian_filter1d

        first = out[0].copy()
        last = out[-1].copy()
        out[:, 0] = gaussian_filter1d(out[:, 0], sigma=sigma, mode="nearest")
        out[:, 1] = gaussian_filter1d(out[:, 1], sigma=sigma, mode="nearest")
        out[0] = first
        out[-1] = last
    return out


def _snap_agent_cell(
    M_free: np.ndarray, fwd0: float, lft0: float, bev_res: float
) -> tuple[int, int]:
    """Find the BEV cell at the agent foot ``(0, 0)``, falling back to the
    nearest navigable cell when that cell is non-navigable.

    With the (0, 0)-centred grid (``fwd_vals[0] = 0``,
    ``lft_vals[W//2] = 0``), the cell at ``(row=0, col=W//2)`` sits exactly
    at the agent foot — a single integer-rounded lookup gives the agent's
    cell without ties. Only when that cell is non-navigable (inside a
    narrow FoV cone with zero in-cone cells at row=0, or rasterised onto
    a wall) do we fall back to a Euclidean-nearest scan over walkable
    cells.
    """
    H = int(M_free.shape[0])
    W = int(M_free.shape[1])
    target_row = int(round((0.0 - float(fwd0)) / float(bev_res)))
    target_col = int(round((0.0 - float(lft0)) / float(bev_res)))
    target_row = max(0, min(H - 1, target_row))
    target_col = max(0, min(W - 1, target_col))
    if bool(M_free[target_row, target_col]):
        return (target_row, target_col)
    (walk_rs, walk_cs) = np.where(M_free)
    if walk_rs.size == 0:
        return (target_row, target_col)
    cell_fwd = float(fwd0) + float(bev_res) * walk_rs.astype(np.float64)
    cell_lft = float(lft0) + float(bev_res) * walk_cs.astype(np.float64)
    dist2 = cell_fwd**2 + cell_lft**2
    idx = int(np.argmin(dist2))
    return (int(walk_rs[idx]), int(walk_cs[idx]))


def _compute_obstacle_distance(obs_mask: np.ndarray) -> np.ndarray:
    """Stage 3: DTF inside obstacles -> D_obs."""
    return distance_transform_edt(obs_mask).astype(np.float32)


def _compose_potential(
    free_mask: np.ndarray,
    dist_geo: np.ndarray,
    dist_obs: np.ndarray,
    goal_weight: float = 1.0,
    obs_weight: float = 5.0,
) -> np.ndarray:
    """Stage 4: Compose potential Phi."""
    (H, W) = free_mask.shape
    phi = np.zeros((H, W), dtype=np.float32)
    reachable = np.isfinite(dist_geo) & free_mask
    max_geo = 0.0
    if reachable.any():
        max_geo = float(dist_geo[reachable].max())
        phi[reachable] = goal_weight * dist_geo[reachable]
    obs_mask = ~free_mask
    if obs_mask.any():
        b_obs = abs(goal_weight * max_geo) + 50.0
        phi[obs_mask] = obs_weight * dist_obs[obs_mask] + b_obs
    return phi


def _compute_bev_escape_direction(M_free: np.ndarray) -> np.ndarray:
    """Unit escape direction (fwd, lft) for obstacle cells -> nearest free cell.

    Uses EDT in metric BEV pixel space (geometrically correct on the uniform
    grid). Free cells get (0, 0) direction; obstacle cells get a unit vector
    pointing toward the nearest free cell.

    Returns (2, H, W) float32. Used to keep V_bev continuous across the
    walkable/obstacle boundary so downstream bilinear sampling does not see
    a hard 0-vs-gradient discontinuity.
    """
    (H, W) = M_free.shape
    out = np.zeros((2, H, W), dtype=np.float32)
    infeasible = ~M_free
    if not infeasible.any() or not M_free.any():
        return out
    nearest = distance_transform_edt(infeasible, return_distances=False, return_indices=True)
    row_g = np.broadcast_to(np.arange(H, dtype=np.float32)[:, None], (H, W))
    col_g = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :], (H, W))
    dy = nearest[0].astype(np.float32) - row_g
    dx = nearest[1].astype(np.float32) - col_g
    norm = np.sqrt(dy**2 + dx**2) + 1e-08
    out[0] = np.where(infeasible, (dy / norm).astype(np.float32), np.float32(0.0))
    out[1] = np.where(infeasible, (dx / norm).astype(np.float32), np.float32(0.0))
    return out


def _masked_gaussian_blur(
    field: np.ndarray, valid_mask: np.ndarray, ksize: int, sigma: float
) -> np.ndarray:
    """Blur only within valid cells so obstacle values do not bleed across walls."""
    field_f = np.asarray(field, dtype=np.float32)
    mask_f = np.asarray(valid_mask, dtype=np.float32)
    if not np.any(mask_f > 0.0):
        return field_f.copy()
    num = cv2.GaussianBlur(field_f * mask_f, (ksize, ksize), sigma)
    den = cv2.GaussianBlur(mask_f, (ksize, ksize), sigma)
    out = field_f.copy()
    valid = den > 1e-06
    out[valid] = num[valid] / den[valid]
    return out


def _compute_bev_velocity_field(
    potential: np.ndarray,
    free_mask: np.ndarray,
    dist_m: np.ndarray,
    bev_esc: np.ndarray,
    *,
    v_norm: float,
    escape_canon_speed: float = 0.2,
    smoothing_iters: int = 2,
    smooth_ksize: int = 5,
    smooth_sigma: float = 1.0,
    eps: float = 1e-08,
) -> np.ndarray:
    """Stage 5: V*(fwd, lft) on uniform BEV. Returns (2, H, W) fp16.

    Channel 0 = v_fwd  (row direction = forward axis = +x)
    Channel 1 = v_lft  (col direction = left axis    = +y)

    Magnitude semantics
    ~~~~~~~~~~~~~~~~~~~
    Free cells: ``V(p) = direction(p) * D_geo_m(p) / v_norm`` where
    ``D_geo_m`` is the remaining geodesic distance to the goal in
    *metres* (already in metres on input — caller is responsible for
    the unit conversion). The single source of truth is the global
    cost-weighted Dijkstra: ``direction`` comes from ``-∇`` of the
    potential built on the *cost-weighted* distance; ``D_geo_m`` is the
    *geometric* path-length-along-tree (in pure metres) so the speed
    stays unit-scaled even when the safety band inflates the
    cost-weighted distance. Both signals share one source — there is
    no BEV-local Dijkstra anymore.

    Obstacle cells: filled with ``bev_esc`` (unit escape direction toward
    nearest free cell, computed t-level) scaled by ``escape_canon_speed``.
    This keeps V_bev continuous across the walkable/obstacle boundary so
    downstream bilinear sampling does not see a 0-vs-gradient discontinuity.
    The escape speed is in canonical units (typical 0.1-0.3).

    The FoV / depth-visibility mask is applied as a *loss* mask downstream
    (``bev_mask``), NOT to V_bev — V is populated everywhere on the grid.
    """
    phi = potential.astype(np.float32).copy()
    ksize = max(3, smooth_ksize | 1)
    phi[~free_mask] = 0.0
    phi = _masked_gaussian_blur(phi, free_mask, ksize, smooth_sigma)
    grad_lft = -cv2.Sobel(phi, cv2.CV_32F, 1, 0, ksize=3)
    grad_fwd = -cv2.Sobel(phi, cv2.CV_32F, 0, 1, ksize=3)
    grad_fwd[~free_mask] = 0.0
    grad_lft[~free_mask] = 0.0
    mag = np.sqrt(grad_fwd**2 + grad_lft**2) + eps
    dir_fwd = grad_fwd / mag
    dir_lft = grad_lft / mag
    for _ in range(max(0, smoothing_iters)):
        dir_fwd = _masked_gaussian_blur(dir_fwd, free_mask, ksize, smooth_sigma)
        dir_lft = _masked_gaussian_blur(dir_lft, free_mask, ksize, smooth_sigma)
        mag = np.sqrt(dir_fwd**2 + dir_lft**2) + eps
        dir_fwd /= mag
        dir_lft /= mag
        dir_fwd[~free_mask] = 0.0
        dir_lft[~free_mask] = 0.0
    inv_v_norm = 1.0 / max(float(v_norm), 1e-08)
    dist_m_safe = np.where(np.isfinite(dist_m), dist_m, 0.0).astype(np.float32)
    v_fwd = (dir_fwd * dist_m_safe * inv_v_norm).astype(np.float32)
    v_lft = (dir_lft * dist_m_safe * inv_v_norm).astype(np.float32)
    obs_mask = ~free_mask
    if obs_mask.any():
        esc_speed = float(escape_canon_speed)
        v_fwd[obs_mask] = bev_esc[0, obs_mask] * esc_speed
        v_lft[obs_mask] = bev_esc[1, obs_mask] * esc_speed
    v_fwd = np.where(np.isfinite(v_fwd), v_fwd, 0.0)
    v_lft = np.where(np.isfinite(v_lft), v_lft, 0.0)
    return np.stack([v_fwd, v_lft], axis=0).astype(np.float16)


def build_bev_gt_frame_context(
    *,
    pathfinder: Any,
    base_pos_xyz: np.ndarray,
    base_rot_xyzw: np.ndarray,
    bev_x_max: float,
    hfov_rad: float,
    bev_resolution: float = 0.1,
    safe_radius_cells: float = 5.0,
    safety_cost_weight: float = 5.0,
    depth_image_hw: np.ndarray | None = None,
    depth_valid_mask_hw: np.ndarray | None = None,
    depth_visibility_tol_m: float = 0.1,
    stairs_aabbs: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Rasterize reusable pose, navmesh and depth visibility for one frame.

    M_free is the perceivable navigable mask; M_walkable ignores visibility.
    D_obs and bev_esc supply local obstacle repulsion. The local graph is
    reused for go-past target selection. Attraction uses the global solver."""
    hfov_rad = float(hfov_rad)
    bev_x_max = float(bev_x_max)
    bev_resolution = float(bev_resolution)
    if bev_x_max <= 0:
        raise ValueError(f"bev_x_max must be > 0, got {bev_x_max}")
    if bev_resolution <= 0:
        raise ValueError(f"bev_resolution must be > 0, got {bev_resolution}")
    R_b2w = quat_xyzw_to_rotmat(as_numpy(base_rot_xyzw).astype(np.float32).reshape(4)).astype(
        np.float32
    )
    forward_w = R_b2w @ np.array([0, 0, -1], dtype=np.float32)
    left_w = -(R_b2w @ np.array([1, 0, 0], dtype=np.float32))
    base_pos = as_numpy(base_pos_xyz).astype(np.float32).reshape(3)
    (bev_fwd, bev_lft, H_bev, W_bev, fwd0, lft0) = _build_bev_grid(
        bev_x_max=bev_x_max, bev_res=bev_resolution
    )
    M_walkable = _rasterise_bev_walkable(
        pathfinder, base_pos, forward_w, left_w, bev_fwd, bev_lft, stairs_aabbs=stairs_aabbs
    )
    half_fov = hfov_rad / 2.0
    r_m_grid = np.sqrt(bev_fwd.astype(np.float32) ** 2 + bev_lft.astype(np.float32) ** 2)
    theta_grid = np.arctan2(
        bev_lft.astype(np.float32), np.maximum(bev_fwd.astype(np.float32), 1e-08)
    )
    fov_mask = (
        (np.abs(theta_grid) <= half_fov + 0.001)
        & (r_m_grid <= bev_x_max + 0.001)
        & (bev_fwd >= 0.0)
    )
    visible_bev: np.ndarray | None = None
    d_col: np.ndarray | None = None
    bev_y_grid: np.ndarray = np.full(bev_fwd.shape, float(base_pos[1]), dtype=np.float32)
    if depth_image_hw is not None:
        d_col = _compute_depth_column_limits(
            depth_image_hw, z_max=bev_x_max, depth_valid_mask_hw=depth_valid_mask_hw
        )
        bev_y_grid = _compute_bev_navmesh_y_grid(
            pathfinder, base_pos, forward_w, left_w, bev_fwd, bev_lft
        )
        visible_bev = _build_visible_bev_mask_3d(
            bev_fwd,
            bev_lft,
            bev_y_grid,
            base_pos,
            forward_w,
            left_w,
            hfov_rad,
            bev_x_max,
            depth_image_hw,
            depth_visibility_tol_m=depth_visibility_tol_m,
        )
    M_navigable = M_walkable & fov_mask
    if visible_bev is not None:
        M_navigable = M_navigable & visible_bev
    M_obs = ~M_navigable
    cost_map = _build_world_safety_cost_map(
        M_navigable, safe_radius_cells=safe_radius_cells, safety_cost_weight=safety_cost_weight
    )
    graph = _build_grid_graph_uniform(M_walkable, res=bev_resolution, cost_map=cost_map)
    D_obs = _compute_obstacle_distance(M_obs)
    bev_esc = _compute_bev_escape_direction(M_navigable)
    agent_cell = _snap_agent_cell(M_navigable, fwd0=fwd0, lft0=lft0, bev_res=bev_resolution)
    return {
        "base_pos": base_pos,
        "forward_w": forward_w,
        "left_w": left_w,
        "R_b2w": R_b2w,
        "hfov_rad": hfov_rad,
        "bev_x_max": bev_x_max,
        "bev_resolution": bev_resolution,
        "depth_visibility_tol_m": depth_visibility_tol_m,
        "safe_radius_cells": float(safe_radius_cells),
        "safety_cost_weight": float(safety_cost_weight),
        "bev_fwd": bev_fwd,
        "bev_lft": bev_lft,
        "H_bev": int(H_bev),
        "W_bev": int(W_bev),
        "fwd0": float(fwd0),
        "lft0": float(lft0),
        "M_free": M_navigable,
        "M_walkable": M_walkable,
        "M_obs": M_obs,
        "fov_mask": fov_mask,
        "visible_bev": visible_bev,
        "d_col": d_col,
        "graph": graph,
        "D_obs": D_obs,
        "bev_esc": bev_esc,
        "agent_cell": agent_cell,
        "bev_walkable": M_walkable.astype(np.uint8),
        "bev_mask": M_navigable.astype(np.uint8),
    }


def build_bev_gt_fields_from_context(
    ctx: dict[str, Any],
    *,
    goal_world_xyz: np.ndarray,
    reference_path_world_xyz: np.ndarray | None = None,
    pathfinder: Any,
    v_norm: float,
    goal_weight: float = 1.0,
    obs_weight: float = 5.0,
    smoothing_iters: int = 2,
    smooth_ksize: int = 5,
    smooth_sigma: float = 1.0,
    escape_canon_speed: float = 0.2,
    approach_radius_m: float = 0.0,
    horizon: int = 100,
    traj_smooth_sigma: float = 1.5,
    global_geodesic_res: float | None = None,
    global_geodesic_margin_m: float = 1.5,
) -> dict[str, np.ndarray]:
    """Compute one goal field and sector-clipped predecessor trajectory.

    Attraction direction uses cost-weighted global geodesic distance;
    magnitude uses geometric path length along that same tree. The approach
    ring remains centered on the original goal when the reference target
    moves to a visible intermediate. Unreachable cells use obstacle escape.
    A goal at the agent produces a zero trajectory; an immediately clipped
    path instead provides a walkable turn anchor. Invalid global geometry
    raises instead of switching to an unrelated local planner."""
    base_pos = ctx["base_pos"]
    forward_w = ctx["forward_w"]
    left_w = ctx["left_w"]
    hfov_rad = ctx["hfov_rad"]
    bev_x_max = ctx["bev_x_max"]
    bev_resolution = ctx["bev_resolution"]
    depth_tol = ctx["depth_visibility_tol_m"]
    bev_fwd = ctx["bev_fwd"]
    bev_lft = ctx["bev_lft"]
    M_free = ctx["M_free"]
    d_col = ctx["d_col"]
    D_obs = ctx["D_obs"]
    bev_esc = ctx["bev_esc"]
    agent_cell = ctx["agent_cell"]
    goal = as_numpy(goal_world_xyz).astype(np.float32).reshape(3)
    local_target = goal
    goal_source = "episode_goal"
    if reference_path_world_xyz is not None:
        (ref_target, ref_target_source) = _select_visible_reference_target(
            reference_path_world_xyz,
            base_pos,
            forward_w,
            left_w,
            hfov_rad,
            bev_x_max,
            d_col,
            depth_visibility_tol_m=depth_tol,
            sample_step_m=max(bev_resolution, 0.1),
            M_free=M_free,
            bev_fwd=bev_fwd,
            bev_lft=bev_lft,
            bev_resolution=bev_resolution,
            M_walkable=ctx["M_walkable"],
        )
        if ref_target is not None:
            local_target = ref_target
            goal_source = ref_target_source
    _d_goal_world = np.asarray(goal, dtype=np.float64).reshape(3) - np.asarray(
        base_pos, dtype=np.float64
    ).reshape(3)
    _fwd_w_xz = np.array([float(forward_w[0]), float(forward_w[2])], dtype=np.float64)
    _lft_w_xz = np.array([float(left_w[0]), float(left_w[2])], dtype=np.float64)
    _d_xz = np.array([float(_d_goal_world[0]), float(_d_goal_world[2])], dtype=np.float64)
    goal_fwd_ego = float(np.dot(_d_xz, _fwd_w_xz))
    goal_lft_ego = float(np.dot(_d_xz, _lft_w_xz))
    H_bev = int(M_free.shape[0])
    W_bev = int(M_free.shape[1])
    global_field: dict[str, Any] | None = None
    _g_res = float(bev_resolution) if global_geodesic_res is None else float(global_geodesic_res)
    _safe_r_cells = float(ctx["safe_radius_cells"])
    _safety_w = float(ctx["safety_cost_weight"])
    global_field = compute_global_geodesic_field(
        pathfinder=pathfinder,
        base_pos=base_pos,
        forward_w=forward_w,
        left_w=left_w,
        goal_world=local_target,
        bev_x_max=bev_x_max,
        res=_g_res,
        margin_m=float(global_geodesic_margin_m),
        safe_radius_cells=_safe_r_cells,
        safety_cost_weight=_safety_w,
    )
    D_global_bev = sample_global_field_at(
        global_field, np.asarray(bev_fwd, dtype=np.float64), np.asarray(bev_lft, dtype=np.float64)
    ).astype(np.float32)
    D_pix_global = global_field["D_pix_m"]
    D_pix_m_bev = sample_global_field_at(
        {**global_field, "D": D_pix_global},
        np.asarray(bev_fwd, dtype=np.float64),
        np.asarray(bev_lft, dtype=np.float64),
    ).astype(np.float32)
    d_to_goal_bev = np.sqrt(
        (np.asarray(bev_fwd, dtype=np.float32) - np.float32(goal_fwd_ego)) ** 2
        + (np.asarray(bev_lft, dtype=np.float32) - np.float32(goal_lft_ego)) ** 2
    ).astype(np.float32)
    use_ring = float(approach_radius_m) > 0.0
    if use_ring:
        inside_ring = d_to_goal_bev <= np.float32(approach_radius_m)
        D_attract = np.where(inside_ring, np.float32(0.0), D_global_bev).astype(np.float32)
    else:
        D_attract = D_global_bev
    _stop_D: float | None = float(approach_radius_m) if use_ring else None
    _stop_snap_eps_m = 0.5 * float(bev_resolution)
    _goal_dist_from_agent_m = float(
        np.linalg.norm(
            (
                np.asarray(goal_world_xyz, dtype=np.float64).reshape(3)
                - np.asarray(base_pos, dtype=np.float64).reshape(3)
            )[[0, 2]]
        )
    )
    if _goal_dist_from_agent_m < _stop_snap_eps_m:
        path_coords_global = np.array([[0.0, 0.0]], dtype=np.float32)
        path_total_len_m_global = 0.0
        walk_term_reason = "stop_snap_at_agent"
    else:
        _sector_mask = M_free
        _gf = global_field["fwd_vals"]
        _gl = global_field["lft_vals"]
        (_ff, _ll) = np.meshgrid(_gf, _gl, indexing="ij")
        d_to_goal_global = np.sqrt(
            (_ff - np.float32(goal_fwd_ego)) ** 2 + (_ll - np.float32(goal_lft_ego)) ** 2
        ).astype(np.float32)
        global_field["d_to_goal_m"] = d_to_goal_global
        _stop_field_key = "d_to_goal_m"
        (path_coords_global, path_total_len_m_global, walk_term_reason) = walk_global_pred_path(
            global_field,
            agent_fwd=0.0,
            agent_lft=0.0,
            stop_D=_stop_D,
            stop_field_key=_stop_field_key,
            sector_mask_bev=_sector_mask,
            bev_fwd=bev_fwd,
            bev_lft=bev_lft,
            bev_resolution=bev_resolution,
        )
    picked_rc: tuple[int, int] | None = None
    if path_coords_global.shape[0] >= 2:
        last_fwd = float(path_coords_global[-1, 0])
        last_lft = float(path_coords_global[-1, 1])
        _r = int(round((last_fwd - float(bev_fwd[0, 0])) / float(bev_resolution)))
        _c = int(round((last_lft - float(bev_lft[0, 0])) / float(bev_resolution)))
        if 0 <= _r < H_bev and 0 <= _c < W_bev:
            picked_rc = (_r, _c)
    elif path_coords_global.shape[0] == 1 and walk_term_reason == "sector_clip":
        half_fov = 0.5 * float(hfov_rad)
        goal_lft_m = float(global_field["goal_lft"])
        side = 1.0 if goal_lft_m >= 0.0 else -1.0
        turn_radius = _shrink_turn_radius_to_walkable(
            half_fov=half_fov,
            side=side,
            M_free=M_free,
            bev_fwd=bev_fwd,
            bev_lft=bev_lft,
            bev_resolution=float(bev_resolution),
            max_radius_m=0.5,
            min_radius_m=float(bev_resolution),
        )
        edge_fwd = turn_radius * math.cos(half_fov)
        edge_lft = turn_radius * math.sin(half_fov) * side
        path_coords_global = np.array([[0.0, 0.0], [edge_fwd, edge_lft]], dtype=np.float32)
        path_total_len_m_global = float(np.hypot(edge_fwd, edge_lft))
        _r = int(round((edge_fwd - float(bev_fwd[0, 0])) / float(bev_resolution)))
        _c = int(round((edge_lft - float(bev_lft[0, 0])) / float(bev_resolution)))
        if 0 <= _r < H_bev and 0 <= _c < W_bev:
            picked_rc = (_r, _c)
    if picked_rc is None and path_coords_global.shape[0] == 1:
        picked_rc = (int(agent_cell[0]), int(agent_cell[1]))
    if picked_rc is not None:
        goal_mask = np.zeros((H_bev, W_bev), dtype=bool)
        goal_mask[picked_rc[0], picked_rc[1]] = True
    else:
        raise RuntimeError("Global geodesic path endpoint lies outside the sector grid")
    if picked_rc is not None:
        (_pr, _pc) = picked_rc
        _x_f = float(bev_fwd[_pr, _pc])
        _y_l = float(bev_lft[_pr, _pc])
        picked_goal_world = (
            base_pos.astype(np.float32)
            + np.float32(_x_f) * forward_w.astype(np.float32)
            + np.float32(_y_l) * left_w.astype(np.float32)
        ).astype(np.float32)
    else:
        picked_goal_world = local_target.astype(np.float32)
    dist_m_for_v = D_pix_m_bev
    D_gpix_bev = dist_m_for_v / max(float(bev_resolution), 1e-08)
    reachable_free = M_free & np.isfinite(D_attract)
    potential = _compose_potential(reachable_free, D_attract, D_obs, goal_weight, obs_weight)
    if not np.array_equal(reachable_free, M_free):
        bev_esc_eff = _compute_bev_escape_direction(reachable_free)
    else:
        bev_esc_eff = bev_esc
    V_bev = _compute_bev_velocity_field(
        potential,
        reachable_free,
        dist_m_for_v,
        bev_esc_eff,
        v_norm=v_norm,
        escape_canon_speed=escape_canon_speed,
        smoothing_iters=smoothing_iters,
        smooth_ksize=smooth_ksize,
        smooth_sigma=smooth_sigma,
    )
    path_coords = path_coords_global
    path_total_len_m = float(path_total_len_m_global)
    dp_traj_cart = _resample_path_arclength(
        path_coords, horizon=int(horizon), smooth_sigma=float(traj_smooth_sigma)
    )

    def _u8(a):
        return a.astype(np.uint8)[np.newaxis]

    def _f32(a):
        return a.astype(np.float32)[np.newaxis]

    return {
        "V_bev": V_bev,
        "dp_traj_cart": dp_traj_cart,
        "dp_traj_cart_total_length_m": np.float32(path_total_len_m),
        "geodesic": _f32(D_attract),
        "geodesic_pix": _f32(D_gpix_bev),
        "potential": _f32(potential),
        "goal_mask": _u8(goal_mask),
        "goal_source": goal_source,
        "local_goal_world": local_target.astype(np.float32),
        "picked_goal_world": picked_goal_world,
        "approach_radius_m": np.float32(approach_radius_m),
        "walk_term_reason": str(walk_term_reason),
    }
