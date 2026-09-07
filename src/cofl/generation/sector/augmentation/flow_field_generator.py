"""Shared frame rasterization and goal-dependent Cartesian field generation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..fields.sector_gt import build_bev_gt_fields_from_context, build_bev_gt_frame_context
from .scene_analyzer.types import SceneContext
from .semantic_anchor.geometry import _collect_stairs_aabbs as _collect_stairs_aabbs_for_sector

_DEFAULT_CFG: dict[str, Any] = {
    "bev_x_max": 5.0,
    "bev_resolution": 0.1,
    "hfov_rad": math.pi / 2.0,
    "v_norm": 5.0,
    "horizon": 100,
    "traj_smooth_sigma": 1.5,
    "safe_radius_cells": 5.0,
    "safety_cost_weight": 5.0,
    "goal_weight": 1.0,
    "obs_weight": 5.0,
    "smoothing_iters": 2,
    "smooth_ksize": 5,
    "smooth_sigma": 1.0,
    "escape_canon_speed": 0.2,
    "global_geodesic_res": None,
    "global_geodesic_margin_m": 1.5,
    "depth_visibility_tol_m": 0.1,
}


def _load_cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(_DEFAULT_CFG)
    if cfg:
        unknown = set(cfg) - set(_DEFAULT_CFG)
        if unknown:
            raise ValueError("Unknown field generation parameters: " + ", ".join(sorted(unknown)))
        merged.update(cfg)
    return merged


def _heading_to_xyzw(heading: float) -> np.ndarray:
    """Convert heading (0 = +X, positive toward -Z) to a Habitat xyzw rotation."""
    theta = math.atan2(-math.cos(heading), math.sin(heading))
    return np.array([0.0, math.sin(theta * 0.5), 0.0, math.cos(theta * 0.5)], dtype=np.float32)


class FlowFieldResult:
    """Wraps the per-(t, k) output of ``build_bev_gt_fields_from_context``.

    Frame-level fields (``bev_walkable`` / ``bev_mask``) are goal-independent
    and live in the dict returned by :meth:`FlowFieldGenerator.precompute_frame`.
    They are NOT duplicated here.

    Attributes
    ----------
    bev_v_field : np.ndarray  (2, H_bev, W_bev) fp16
        Cartesian velocity field. Channel 0 = v_x (forward),
        channel 1 = v_y (left). Values are in canonical units
        (= v_metric / v_norm); both channels share one isotropic scale.
    dp_traj_cart : np.ndarray  (H+1, 2) fp32
        Dijkstra shortest-path GT trajectory in cart metres
        ``(x_fwd, y_lft)``. Resampled at uniform arc-length over
        ``horizon + 1`` points. cart[0] = agent foot (0, 0);
        cart[-1] = goal cell centre. Stop-snap frames (path length 0)
        produce a constant (0, 0) trajectory.
    dp_traj_cart_total_length_m : float
        Sum of segment lengths along the dijkstra polyline (metres).
        Used by the action bucketing's STOP test (length < stop_radius_m).
    raw : dict
        Full dict returned by ``build_bev_gt_fields_from_context``. Contains
        per-goal metadata (``picked_goal_world``, ``local_goal_world``,
        ``goal_source``, ``geodesic`` etc.) for QA / debug.
    """

    __slots__ = ("bev_v_field", "dp_traj_cart", "dp_traj_cart_total_length_m", "raw")

    def __init__(self, raw: dict[str, np.ndarray]) -> None:
        self.raw = raw
        v = raw["V_bev"]
        if v.ndim != 3 or v.shape[0] != 2:
            raise ValueError(f"V_bev must have shape (2, H, W); got {tuple(v.shape)}")
        if v.dtype != np.float16:
            v = v.astype(np.float16)
        self.bev_v_field = v
        traj = np.asarray(raw["dp_traj_cart"], dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError(f"dp_traj_cart must have shape (H+1, 2); got {tuple(traj.shape)}")
        self.dp_traj_cart = traj
        self.dp_traj_cart_total_length_m = float(raw["dp_traj_cart_total_length_m"])


class FlowFieldGenerator:
    """Generate fields for candidate goals using one reusable frame context."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = _load_cfg(cfg)

    def precompute_frame(
        self,
        context: SceneContext,
        *,
        depth_image_hw: np.ndarray | None = None,
        depth_valid_mask_hw: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Rasterize navmesh, metric depth visibility and obstacle geometry once per frame."""
        cfg = self.cfg
        base_pos_xyz = np.asarray(context.position, dtype=np.float32).reshape(3)
        base_rot_xyzw = _heading_to_xyzw(context.heading)
        return build_bev_gt_frame_context(
            pathfinder=context.pathfinder,
            base_pos_xyz=base_pos_xyz,
            base_rot_xyzw=base_rot_xyzw,
            bev_x_max=float(cfg["bev_x_max"]),
            hfov_rad=float(cfg["hfov_rad"]),
            bev_resolution=float(cfg["bev_resolution"]),
            safe_radius_cells=float(cfg["safe_radius_cells"]),
            safety_cost_weight=float(cfg["safety_cost_weight"]),
            depth_image_hw=depth_image_hw,
            depth_valid_mask_hw=depth_valid_mask_hw,
            depth_visibility_tol_m=float(cfg.get("depth_visibility_tol_m", 0.1)),
            stairs_aabbs=_collect_stairs_aabbs_for_sector(getattr(context, "semantic_scene", None)),
        )

    def generate_from_context(
        self,
        frame_ctx: dict[str, Any],
        context: SceneContext,
        goal_point: np.ndarray,
        *,
        reference_path_world_xyz: np.ndarray | None = None,
        approach_radius_m: float = 0.0,
    ) -> FlowFieldResult:
        """Run the goal-dependent stages using a precomputed frame context."""
        cfg = self.cfg
        raw = build_bev_gt_fields_from_context(
            frame_ctx,
            goal_world_xyz=np.asarray(goal_point, dtype=np.float32).reshape(3),
            reference_path_world_xyz=reference_path_world_xyz,
            pathfinder=context.pathfinder,
            v_norm=float(cfg["v_norm"]),
            goal_weight=float(cfg["goal_weight"]),
            obs_weight=float(cfg["obs_weight"]),
            smoothing_iters=int(cfg["smoothing_iters"]),
            smooth_ksize=int(cfg["smooth_ksize"]),
            smooth_sigma=float(cfg["smooth_sigma"]),
            escape_canon_speed=float(cfg["escape_canon_speed"]),
            approach_radius_m=float(approach_radius_m),
            horizon=int(cfg.get("horizon", 100)),
            traj_smooth_sigma=float(cfg.get("traj_smooth_sigma", 1.5)),
            global_geodesic_res=None
            if cfg["global_geodesic_res"] is None
            else float(cfg["global_geodesic_res"]),
            global_geodesic_margin_m=float(cfg.get("global_geodesic_margin_m", 1.5)),
        )
        return FlowFieldResult(raw)
