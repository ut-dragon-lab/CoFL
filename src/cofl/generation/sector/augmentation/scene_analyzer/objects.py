"""Habitat semantic object category and bounding-box access."""

from __future__ import annotations

from typing import Any

import numpy as np


def _canonical_category(obj: Any) -> str | None:
    """Extract a lowercase category name from a habitat_sim SemanticObject."""
    cat = getattr(obj, "category", None)
    if cat is None:
        return None
    try:
        name = cat.name()
        if name:
            return str(name).strip().lower()
    except Exception:
        pass
    name = getattr(cat, "name", None)
    if isinstance(name, str) and name:
        return name.strip().lower()
    return None


def _obj_centroid(obj: Any) -> np.ndarray | None:
    """Extract the centroid of a semantic object as (3,) float64."""
    aabb = getattr(obj, "aabb", None)
    if aabb is not None:
        center = getattr(aabb, "center", None)
        if center is not None:
            try:
                return np.asarray(center, dtype=np.float64).reshape(3)
            except Exception:
                pass
        mn = getattr(aabb, "min", None)
        sz = getattr(aabb, "sizes", None)
        if mn is not None and sz is not None:
            try:
                mn_arr = np.asarray(mn, dtype=np.float64).reshape(3)
                sz_arr = np.asarray(sz, dtype=np.float64).reshape(3)
                return mn_arr + sz_arr * 0.5
            except Exception:
                pass
    obb = getattr(obj, "obb", None)
    if obb is not None:
        center = getattr(obb, "center", None)
        if center is not None:
            try:
                return np.asarray(center, dtype=np.float64).reshape(3)
            except Exception:
                pass
    return None


def _obj_aabb(obj: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (aabb_min, aabb_max) in world coords, or (None, None)."""
    aabb = getattr(obj, "aabb", None)
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
