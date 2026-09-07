"""Semantic room bounds, current-room membership and navigable candidates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .navigation import _is_navigable, _snap_point


def _get_region_category(region: Any) -> str | None:
    """Extract a lowercase category/name from a SemanticRegion.

    MP3D ships several region categories as slash-joined alternatives
    (e.g. ``"spa/sauna"``, ``"living room/dining room"``,
    ``"laundryroom/mudroom"``). These are not real English phrases — the
    slash means "either of these labels apply" — and instructions like
    ``"turn left to enter the spa/sauna"`` confuse the model. Pick the
    first alternative as the canonical name; downstream renderers can use
    that single name in instruction templates.
    """
    raw: str | None = None
    cat = getattr(region, "category", None)
    if cat is not None:
        try:
            name = cat.name()
            if name and str(name).strip():
                raw = str(name).strip().lower()
        except Exception:
            pass
        if raw is None:
            name = getattr(cat, "name", None)
            if isinstance(name, str) and name.strip():
                raw = name.strip().lower()
    if raw is None:
        name = getattr(region, "name", None)
        if isinstance(name, str) and name.strip():
            raw = name.strip().lower()
    if raw is None:
        return None
    if "/" in raw:
        raw = raw.split("/", 1)[0].strip()
    return raw or None


def _region_aabb(region: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (min, max) axis-aligned bounding box for a SemanticRegion."""
    aabb = getattr(region, "aabb", None)
    if aabb is None:
        return (None, None)
    mn = getattr(aabb, "min", None)
    mx = getattr(aabb, "max", None)
    if mn is not None and mx is not None:
        try:
            return (
                np.asarray(mn, dtype=np.float64).reshape(3),
                np.asarray(mx, dtype=np.float64).reshape(3),
            )
        except Exception:
            pass
    center = getattr(aabb, "center", None)
    sizes = getattr(aabb, "sizes", None)
    if center is not None and sizes is not None:
        try:
            c = np.asarray(center, dtype=np.float64).reshape(3)
            s = np.asarray(sizes, dtype=np.float64).reshape(3)
            return (c - s * 0.5, c + s * 0.5)
        except Exception:
            pass
    return (None, None)


def _point_in_aabb(point: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
    return bool(np.all(point >= aabb_min) and np.all(point <= aabb_max))


def _find_current_region(semantic_scene: Any, position: np.ndarray) -> str | None:
    """Return the category of the smallest region whose AABB contains *position*.

    If multiple regions contain the position, we pick the one with the smallest
    AABB volume (most specific containment).
    """
    regions = getattr(semantic_scene, "regions", []) or []
    best_name: str | None = None
    best_vol: float = float("inf")
    for region in regions:
        (aabb_min, aabb_max) = _region_aabb(region)
        if aabb_min is None:
            continue
        if not _point_in_aabb(position, aabb_min, aabb_max):
            continue
        vol = float(np.prod(np.maximum(aabb_max - aabb_min, 0)))
        if vol < best_vol:
            best_vol = vol
            best_name = _get_region_category(region)
    return best_name


def _sample_navigable_candidates_in_aabb(
    pathfinder: Any,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    floor_y: float,
    n_samples: int,
    snap_tol: float,
    stairs_aabbs: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[np.ndarray]:
    """Sample room candidates, allowing height changes only inside stairs."""
    out: list[np.ndarray] = []
    size_x = float(aabb_max[0] - aabb_min[0])
    size_z = float(aabb_max[2] - aabb_min[2])
    n_side = max(2, int(math.ceil(math.sqrt(n_samples))))
    xs = np.linspace(float(aabb_min[0]) + size_x * 0.1, float(aabb_max[0]) - size_x * 0.1, n_side)
    zs = np.linspace(float(aabb_min[2]) + size_z * 0.1, float(aabb_max[2]) - size_z * 0.1, n_side)
    gate_active = stairs_aabbs is not None
    in_stairs_aabbs = stairs_aabbs or []
    for x in xs:
        for z in zs:
            pt = np.array([x, floor_y, z], dtype=np.float64)
            in_stairs = False
            if gate_active:
                for mn, mx in in_stairs_aabbs:
                    if mn[0] - 0.2 <= x <= mx[0] + 0.2 and mn[2] - 0.2 <= z <= mx[2] + 0.2:
                        in_stairs = True
                        break
            snapped = _snap_point(pathfinder, pt)
            snap_ok = (
                snapped is not None
                and _is_navigable(pathfinder, snapped, snap_tol)
                and (not gate_active or in_stairs or abs(float(snapped[1]) - float(floor_y)) <= 0.5)
            )
            if snap_ok:
                out.append(snapped)
            elif _is_navigable(pathfinder, pt, snap_tol):
                out.append(pt)
    return out
