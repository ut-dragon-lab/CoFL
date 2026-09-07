"""Render supported instruction families from grounded candidate parameters."""

from __future__ import annotations

import numpy as np

from .families import (
    _FAMILY_TEMPLATES,
    ACTIVE_FAMILIES,
    COMBINED_REGION_FAMILIES,
    DIRECTION_FAMILIES,
    OBJECT_FAMILIES,
    REGION_FAMILIES,
)
from .templates import _TURN_CORNER

_CORNER_MIN_TOTAL_HEADING_DEG = 60.0
_CORNER_MAX_ARC_LENGTH_M = 2.5
DEFAULT_CORNER_SWAP_PROB = 0.1


def is_corner_trajectory(
    cart: np.ndarray,
    *,
    min_total_heading_deg: float = _CORNER_MIN_TOTAL_HEADING_DEG,
    max_arc_length_m: float = _CORNER_MAX_ARC_LENGTH_M,
) -> bool:
    """True iff the polyline ``cart`` looks like a sharp corner.

    ``cart`` is the agent-local dijkstra trajectory shape ``(H+1, 2)`` —
    typically ``ff.dp_traj_cart`` from the flow-field generator. We sum
    the absolute heading change across all valid segments and compare
    against ``min_total_heading_deg``; if the path is also short enough
    (``≤ max_arc_length_m``), we call it a corner. Long sweeping curves
    that accumulate the same total turn are intentionally rejected.
    """
    pts = np.asarray(cart, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return False
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    valid = seg_len > 0.001
    if int(valid.sum()) < 2:
        return False
    arc_m = float(seg_len[valid].sum())
    if arc_m > float(max_arc_length_m):
        return False
    headings = np.arctan2(seg[:, 1], seg[:, 0])
    d = np.diff(headings)
    d = np.arctan2(np.sin(d), np.cos(d))
    total_change_deg = float(np.degrees(np.sum(np.abs(d))))
    return total_change_deg >= float(min_total_heading_deg)


def render(
    family: str,
    *,
    object_name: str | None = None,
    region_name: str | None = None,
    cur_region_name: str | None = None,
    corner_eligible: bool = False,
    corner_swap_prob: float = DEFAULT_CORNER_SWAP_PROB,
    rng: np.random.Generator | None = None,
) -> str:
    """Render one sub-instruction for ``family``.

    ``corner_eligible`` (caller-determined, e.g. via
    :func:`is_corner_trajectory` on the dp_traj_cart): when True AND
    family ∈ {``turn_left``, ``turn_right``}, the template pool is
    swapped to the corner variants with probability
    ``corner_swap_prob``. The swap happens before the template is
    sampled, so the chosen surface form is already from the corner
    pool when it fires.

    Raises ``ValueError`` if the family is unknown or required params are missing.
    """
    if rng is None:
        rng = np.random.default_rng()
    pool = _FAMILY_TEMPLATES.get(family)
    if not pool:
        raise ValueError(f"Unknown instruction family: {family!r}")
    if (
        corner_eligible
        and family in ("turn_left", "turn_right")
        and (float(rng.random()) < float(corner_swap_prob))
    ):
        pool = _TURN_CORNER
    template = pool[int(rng.integers(0, len(pool)))]
    params: dict[str, str] = {}
    if family in OBJECT_FAMILIES:
        if not object_name:
            raise ValueError(f"family={family!r} requires object_name")
        params["object"] = str(object_name)
    elif family in COMBINED_REGION_FAMILIES:
        if not region_name:
            raise ValueError(f"family={family!r} requires region_name (target)")
        if not cur_region_name:
            raise ValueError(f"family={family!r} requires cur_region_name")
        params["region"] = str(region_name)
        params["cur_region"] = str(cur_region_name)
    elif family in REGION_FAMILIES:
        if not region_name:
            raise ValueError(f"family={family!r} requires region_name")
        params["region"] = str(region_name)
    rendered = template.format(**params)
    if "{" in rendered or "}" in rendered:
        raise ValueError(f"Rendered instruction still has placeholders: {rendered!r}")
    return rendered


__all__ = [
    "render",
    "is_corner_trajectory",
    "DEFAULT_CORNER_SWAP_PROB",
    "ACTIVE_FAMILIES",
    "OBJECT_FAMILIES",
    "REGION_FAMILIES",
    "COMBINED_REGION_FAMILIES",
    "DIRECTION_FAMILIES",
]
