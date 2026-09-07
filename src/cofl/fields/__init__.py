"""Convert dataset grid positions to the public policy query charts."""

from __future__ import annotations

import numpy as np

from cofl.data.profiles import field_query_grid, validate_geometry


def queries_for_grid(profile: str, geometry: dict) -> np.ndarray:
    """Return (H, W, 2) model queries: image (x, y), ground (theta, radius).

    Ground theta is divided by hfov/2; radius is divided by r_max_m.
    Use ``query_valid_mask`` to exclude cells outside the sector domain.
    """
    validate_geometry(geometry, profile)
    xy = field_query_grid(geometry)
    if profile == "image_field_v1":
        return xy
    theta = np.arctan2(xy[..., 1], xy[..., 0]) / (geometry["hfov_rad"] / 2)
    radius = np.linalg.norm(xy, axis=-1) * geometry["normalization_scale_m"] / geometry["r_max_m"]
    return np.stack((theta, radius), axis=-1).astype(np.float32)


def query_valid_mask(profile: str, geometry: dict) -> np.ndarray:
    queries = queries_for_grid(profile, geometry)
    if profile == "image_field_v1":
        return np.ones(queries.shape[:2], dtype=np.bool_)
    return (np.abs(queries[..., 0]) <= 1.0 + 1e-6) & (queries[..., 1] <= 1.0 + 1e-6)


__all__ = ["field_query_grid", "queries_for_grid", "query_valid_mask"]
