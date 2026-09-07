"""The public coordinate contracts of CoFLDataset v1."""

from __future__ import annotations

import math
from typing import Any

PROFILES = ("image_field_v1", "ground_sector_v1")


def geometry_for_profile(
    profile: str,
    grid_shape: tuple[int, int] = (17, 17),
    *,
    normalization_scale_m: float = 5.0,
    hfov_rad: float = math.pi / 2,
) -> dict[str, Any]:
    """Describe a field's grid and vector units independently of model queries.

    ``grid_shape`` is (rows, columns). Ground fields use a Cartesian BEV
    grid even though the policy queries a sector coordinate chart.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown dataset profile {profile!r}; expected one of {PROFILES}")
    result: dict[str, Any] = {
        "profile": profile,
        "grid_shape": list(grid_shape),
        "field_layout": "channels_height_width",
        "time_convention": "static_velocity_per_policy_time",
        "grid_alignment": "inclusive_endpoints",
    }
    if profile == "image_field_v1":
        result.update(
            coordinate_frame="image_normalized",
            observation_view="bev",
            component_axes=["right", "down"],
            grid_axes=["down", "right"],
            grid_bounds=[[0.0, 1.0], [0.0, 1.0]],
            query_axes=["right", "down"],
            query_bounds=[[0.0, 1.0], [0.0, 1.0]],
            velocity_unit="image_fraction_per_policy_time",
        )
    else:
        result.update(
            coordinate_frame="body_normalized",
            observation_view="egocentric",
            component_axes=["forward", "left"],
            grid_axes=["forward", "left"],
            grid_bounds=[[0.0, 1.0], [-1.0, 1.0]],
            query_axes=["theta_normalized", "radius_normalized"],
            query_bounds=[[-1.0, 1.0], [0.0, 1.0]],
            velocity_unit="normalized_metres_per_policy_time",
            normalization_scale_m=float(normalization_scale_m),
            r_max_m=float(normalization_scale_m),
            hfov_rad=float(hfov_rad),
        )
    validate_geometry(result, profile)
    return result


def validate_geometry(geometry: dict[str, Any], profile: str) -> None:
    """Reject ambiguous coordinates instead of guessing defaults at load time."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown dataset profile: {profile!r}")
    if geometry.get("profile") != profile:
        raise ValueError("Geometry profile does not match dataset profile")
    shape = geometry.get("grid_shape")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("geometry.grid_shape must contain (rows, columns)")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in shape):
        raise ValueError("Each grid dimension must be an integer of at least 2")
    expected = {
        "field_layout": "channels_height_width",
        "time_convention": "static_velocity_per_policy_time",
        "grid_alignment": "inclusive_endpoints",
    }
    if profile == "image_field_v1":
        expected.update(
            coordinate_frame="image_normalized",
            observation_view="bev",
            component_axes=["right", "down"],
            grid_axes=["down", "right"],
            grid_bounds=[[0.0, 1.0], [0.0, 1.0]],
            query_axes=["right", "down"],
            query_bounds=[[0.0, 1.0], [0.0, 1.0]],
            velocity_unit="image_fraction_per_policy_time",
        )
    else:
        expected.update(
            coordinate_frame="body_normalized",
            observation_view="egocentric",
            component_axes=["forward", "left"],
            grid_axes=["forward", "left"],
            grid_bounds=[[0.0, 1.0], [-1.0, 1.0]],
            query_axes=["theta_normalized", "radius_normalized"],
            query_bounds=[[-1.0, 1.0], [0.0, 1.0]],
            velocity_unit="normalized_metres_per_policy_time",
        )
        for key in ("normalization_scale_m", "r_max_m", "hfov_rad"):
            value = geometry.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"geometry.{key} must be an explicit positive number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"geometry.{key} must be finite and positive")
        if geometry["hfov_rad"] >= math.pi:
            raise ValueError("Ground sector hfov_rad must be smaller than pi")
        if geometry["r_max_m"] != geometry["normalization_scale_m"]:
            raise ValueError("v1 requires r_max_m == normalization_scale_m")
    for key, value in expected.items():
        if geometry.get(key) != value:
            raise ValueError(f"Invalid geometry.{key} for {profile}: expected {value!r}")


def field_query_grid(geometry: dict[str, Any]):
    """Return an (H, W, 2) NumPy array in field component coordinates."""
    import numpy as np

    validate_geometry(geometry, geometry.get("profile", ""))
    h, w = geometry["grid_shape"]
    first, second = geometry["grid_bounds"]
    if geometry["profile"] == "image_field_v1":
        y, x = np.meshgrid(np.linspace(*second, h), np.linspace(*first, w), indexing="ij")
        return np.stack((x, y), axis=-1).astype(np.float32)
    x, y = np.meshgrid(np.linspace(*first, h), np.linspace(*second, w), indexing="ij")
    return np.stack((x, y), axis=-1).astype(np.float32)
