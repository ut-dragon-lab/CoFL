"""Navigability and turn-feasibility primitives built on habitat_sim's PathFinder.

Heading convention
------------------
    heading = 0  →  +X world axis
    heading > 0  →  CCW rotation (right-hand around +Y)
    At heading = π/2, the agent faces Habitat's default forward (-Z).

    forward_w = (cos(heading), 0, -sin(heading))
    left_w    = (-sin(heading), 0, -cos(heading))
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def heading_to_dirs(heading: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (forward_w, left_w) unit vectors for a given heading.

    heading = 0  →  +X world.  Positive heading = CCW around +Y.
    forward_w = (cos h, 0, -sin h)   ← rotation of +X by h around +Y
    left_w    = (-sin h, 0, -cos h)  ← forward_w rotated π/2 CCW around +Y
    """
    (c, s) = (math.cos(heading), math.sin(heading))
    forward_w = np.array([c, 0.0, -s], dtype=np.float64)
    left_w = np.array([-s, 0.0, -c], dtype=np.float64)
    return (forward_w, left_w)


def _is_navigable(pathfinder, point, tol=0.5):
    """Test navmesh occupancy using Habitat 0.1.7's vertical tolerance."""
    return bool(pathfinder.is_navigable(point, tol))


def _snap_point(pathfinder, point):
    """Return the nearest navmesh point, or None when Habitat reports NaNs."""
    snapped = np.asarray(pathfinder.snap_point(point), dtype=np.float64).reshape(3)
    return snapped if np.isfinite(snapped).all() else None


def _geodesic_distance(pathfinder, start, end):
    """Return Habitat's shortest-path distance, or infinity when unreachable."""
    from habitat_sim import ShortestPath

    path = ShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_end = np.asarray(end, dtype=np.float32)
    return float(path.geodesic_distance) if pathfinder.find_path(path) else math.inf


def _check_turn_feasible(
    pathfinder: Any,
    position: np.ndarray,
    new_dir: np.ndarray,
    *,
    probe_dist: float,
    straightness_ratio: float,
) -> bool:
    """Check whether, after rotating to *new_dir*, the agent can walk at least
    *probe_dist* metres in a roughly straight line.
    """
    goal = position + probe_dist * new_dir
    snapped = _snap_point(pathfinder, goal)
    if snapped is not None:
        goal = snapped
    euc = float(np.linalg.norm(goal - position))
    if euc < 0.0001:
        return False
    geo = _geodesic_distance(pathfinder, position, goal)
    if not math.isfinite(geo):
        return False
    return geo < straightness_ratio * euc
