"""Value objects and default configuration for the semantic-anchor subsystem.

Exports
-------
Dataclasses
    :class:`VisibleObject` — a semantic-image object that passed the filter.
    :class:`VisibleRegion` — a semantic-image region that is currently visible.
    :class:`CandidateCommand` — one (family, goal, params) tuple consumed by
    the instruction renderer downstream.
Defaults
    ``_DEFAULT_WHITELIST`` / ``_DEFAULT_BLACKLIST`` — mpcat40 allow / deny list.
    ``_DEFAULT_FINDER_CFG`` / ``_DEFAULT_SAMPLER_CFG`` — probe / filter params.
Config loaders
    ``_load_finder_cfg`` / ``_load_sampler_cfg`` — merge user overrides on
    top of the defaults.
Display helpers
    ``_display_name`` / ``_mpcat40_of`` — renderer-friendly name look-ups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..scene_analyzer.objects import _canonical_category

_DEFAULT_WHITELIST: list[str] = [
    "chair",
    "table",
    "bed",
    "sofa",
    "cabinet",
    "sink",
    "toilet",
    "bathtub",
    "shower",
    "stairs",
    "cushion",
    "plant",
    "mirror",
    "shelving",
    "chest_of_drawers",
    "counter",
    "stool",
    "fireplace",
    "tv_monitor",
    "picture",
    "towel",
    "appliances",
    "lighting",
    "seating",
    "railing",
    "curtain",
]
_DEFAULT_BLACKLIST: list[str] = [
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "misc",
    "void",
    "objects",
    "furniture",
    "unknown",
    "remove",
    "beam",
    "column",
    "blinds",
    "board_panel",
]
_DEFAULT_FINDER_CFG: dict[str, Any] = {
    "whitelist": _DEFAULT_WHITELIST,
    "blacklist": _DEFAULT_BLACKLIST,
    "min_visible_pixels": 500,
    "min_dist_m": 0.3,
    "max_dist_m": 8.0,
    "max_geo_euc_ratio": 2.5,
    "region_min_visible_pixels": 200,
    "region_visible_aabb_tol_m": 0.5,
    "region_visible_pixel_blacklist": [
        "wall",
        "door",
        "window",
        "misc",
        "void",
        "unknown",
        "remove",
    ],
}
_DEFAULT_SAMPLER_CFG: dict[str, Any] = {
    "K": 4,
    "backup_alt_pool_size": 3,
    "go_past_path_through_obj_tol_m": 0.5,
    "object_pair_bypass_min_pixels": 100,
    "object_require_visible": True,
    "object_visible_match_m": 1.0,
    "region_center_dist_weight": 1.0,
    "region_fov_rad": math.pi / 2,
    "region_openness_search_radius_m": 2.0,
    "terminal_alt_prob": 0.0,
    "forward_r_min_m": 1.0,
    "forward_r_max_m": 3.0,
    "forward_theta_half_rad": math.radians(5.0),
    "forward_corridor_step_m": 0.3,
    "forward_max_sample_attempts": 8,
    "object_navmesh_projection_max_m": 1.0,
    "go_past_geo_excess_min_m": 2.0,
    "region_n_sample_pts": 6,
    "region_max_geo_m": 15.0,
    "region_forward_angle_deg": 25.0,
    "region_fov_margin": 0.95,
    "region_goal_fov_rad": math.pi,
    "region_path_fov_frac": 0.6,
    "object_enabled": True,
    "max_object_alts": 4,
    "object_min_dist_m": 0.5,
    "object_max_dist_m": 8.0,
    "object_floor_band_m": 2.5,
    "object_los_geo_slack_m": 0.4,
    "object_fov_rad": math.pi,
    "object_force_nav_anchors": 2,
    "region_require_visible": True,
    "region_min_visible_pixels_strict": 3000,
    "object_min_visible_pixels_strict": 500,
    "families_enabled": {
        "toward_object": True,
        "go_past": True,
        "forward": True,
        "region_enter": True,
        "turn_left_enter": True,
        "turn_right_enter": True,
        "exit_enter": True,
        "turn_left_exit_enter": True,
        "turn_right_exit_enter": True,
    },
    "framing_probs": {"plain_turn": 0.33, "combined": 0.335, "plain_enter": 0.335},
    "prioritize_families": [
        "region_enter",
        "turn_left_enter",
        "turn_right_enter",
        "exit_enter",
        "turn_left_exit_enter",
        "turn_right_exit_enter",
    ],
}
_DISPLAY_OVERRIDES: dict[str, str] = {
    "tv_monitor": "tv",
    "chest_of_drawers": "dresser",
    "shelving": "shelf",
    "appliances": "appliance",
    "lighting": "light",
    "seating": "seat",
    "board_panel": "panel",
}


def _display_name(name: str) -> str:
    """Return a renderer-friendly noun for an mpcat40 token."""
    n = name.strip().lower()
    return _DISPLAY_OVERRIDES.get(n, n.replace("_", " "))


def _mpcat40_of(obj: Any) -> str | None:
    """Best-effort lookup of an object's mpcat40 label (falls back to raw cat)."""
    cat = getattr(obj, "category", None)
    if cat is None:
        return None
    for attr in ("mpcat40_name", "mpcat40", "_mpcat40_name"):
        v = getattr(cat, attr, None)
        if v:
            try:
                s = str(v).strip().lower()
                if s and s != "unknown":
                    return s
            except Exception:
                pass
    return _canonical_category(obj)


@dataclass
class VisibleObject:
    """A single semantic-image object that passed the finder filters."""

    object_id: int
    mpcat40: str
    display_name: str
    centroid: np.ndarray
    pixel_count: int
    distance_m: float
    geodesic_m: float
    angle_rad: float


@dataclass
class VisibleRegion:
    """A semantic region that is currently visible in the camera frame.

    "Visible" = the union of object_ids in the semantic image whose
    parent region is this region covers at least
    ``region_min_visible_pixels`` of the frame. Used by the
    region-enter visibility gate to drop ``*_enter`` / ``enter_then_*``
    candidates whose target region is not actually in view.
    """

    region_name: str
    pixel_count: int
    object_ids: list[int]


@dataclass
class CandidateCommand:
    """One (family, goal, params) to be turned into a (text, flow) sample.

    ``family`` is ``None`` for anchors: anchor text comes verbatim from
    the dataset's sub-instruction stream (FG-R2R / Landmark-RxR) and
    goal comes from the matching reference-path waypoint, so neither
    template rendering nor family-based flow processing applies. The
    only metadata key the sampler reads from anchors is
    ``"is_last_subinstr"`` (drives the builder's stop-snap eligibility
    on k=0 and the sampler's `terminal` alt injection skip).
    Alternatives keep ``family: str`` since they're rendered from
    templates and the family drives ring / stop-snap behaviour.
    """

    family: str | None
    goal_xyz: np.ndarray
    kind: str = "anchor"
    object_id: int | None = None
    object_name: str | None = None
    object_mpcat40: str | None = None
    region_name: str | None = None
    cur_region_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_finder_cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge user overrides on top of :data:`_DEFAULT_FINDER_CFG`."""
    merged = dict(_DEFAULT_FINDER_CFG)
    if cfg:
        unknown = set(cfg) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown semantic generation parameters: " + ", ".join(sorted(unknown))
            )
        merged.update(cfg)
    return merged


def _load_sampler_cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge user overrides on top of :data:`_DEFAULT_SAMPLER_CFG`.

    Validates that every family name referenced by ``families_enabled`` and
    ``prioritize_families`` is actually emittable by the renderer
    (``instruction_renderer.ACTIVE_FAMILIES``).  Stale or mistyped names
    fail fast here instead of being silently ignored at sample time.
    """
    merged = dict(_DEFAULT_SAMPLER_CFG)
    if cfg:
        unknown = set(cfg) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown semantic generation parameters: " + ", ".join(sorted(unknown))
            )
        merged.update(cfg)
    from ..instruction_renderer import ACTIVE_FAMILIES

    enabled = merged.get("families_enabled") or {}
    unknown_enabled = set(enabled.keys()) - ACTIVE_FAMILIES
    if unknown_enabled:
        raise ValueError(
            f"families_enabled references unknown families {sorted(unknown_enabled)}; valid set = {sorted(ACTIVE_FAMILIES)}"
        )
    priority = merged.get("prioritize_families") or []
    unknown_priority = set(priority) - ACTIVE_FAMILIES
    if unknown_priority:
        raise ValueError(
            f"prioritize_families references unknown families {sorted(unknown_priority)}; valid set = {sorted(ACTIVE_FAMILIES)}"
        )
    return merged
