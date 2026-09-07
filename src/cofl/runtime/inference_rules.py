"""Canonical CoFL-series inference policy, importable by Python 3.7 workers.

Keep this module free of array, model, and simulator dependencies. Model fields
are static; the integration schedule scales their readout, not model time input.
"""

import math
from numbers import Integral, Real

INFERENCE_RULES_VERSION = "cofl-inference-v1"
DEFAULT_FIELD_GRID_SIZE = 100
DEFAULT_FIELD_QUERY_CHUNK_SIZE = 4096
DEFAULT_NUM_STEPS = 100
DEFAULT_POLICY_DT = 0.01
DEFAULT_TIME_SCALE_POWER = 10.0
IMAGE_TIME_SCALE_BIAS = 0.5
GROUND_TIME_SCALE_BIAS = 0.2
GROUND_START_MIN_RADIUS_NORMALIZED = 1e-3
GROUND_RADIUS_EPS_M = 1e-6
MODEL_HFOV_MIN_DEG = 79.0
MODEL_HFOV_MAX_DEG = 90.0


def profile_time_scale_bias(profile):
    if profile == "image_field_v1":
        return IMAGE_TIME_SCALE_BIAS
    if profile == "ground_sector_v1":
        return GROUND_TIME_SCALE_BIAS
    raise ValueError(f"Unsupported inference profile: {profile!r}")


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def reference_start_radius(grid_size=DEFAULT_FIELD_GRID_SIZE, *, explicit=False):
    """Normalized origin floor from SecVLA's QA/App reference-start helper."""
    if isinstance(grid_size, bool) or not isinstance(grid_size, Integral) or grid_size < 2:
        raise ValueError("grid_size must be an integer >= 2")
    half_cell = 0.5 / (int(grid_size) - 1)
    return half_cell if explicit else max(GROUND_START_MIN_RADIUS_NORMALIZED, half_cell)


def inference_rule_metadata(
    profile,
    *,
    grid_size=DEFAULT_FIELD_GRID_SIZE,
    num_steps=DEFAULT_NUM_STEPS,
    policy_dt=DEFAULT_POLICY_DT,
    time_scale_bias=None,
    time_scale_power=DEFAULT_TIME_SCALE_POWER,
):
    """Return the serializable contract used by every production inference route."""
    default_bias = profile_time_scale_bias(profile)
    radius = reference_start_radius(grid_size)
    if isinstance(num_steps, bool) or not isinstance(num_steps, Integral) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    bias = _positive(
        default_bias if time_scale_bias is None else time_scale_bias, "time_scale_bias"
    )
    power = _positive(time_scale_power, "time_scale_power")
    dt = _positive(policy_dt, "policy_dt")
    ground = profile == "ground_sector_v1"
    return {
        "version": INFERENCE_RULES_VERSION,
        "profile": profile,
        "field_evaluation": "dense_query_grid",
        "field_grid_shape": [int(grid_size), int(grid_size)],
        "field_grid_axes": ["radius_normalized", "theta_normalized"]
        if ground
        else ["down", "right"],
        "field_grid_endpoints": "inclusive",
        "interpolation": "bilinear_border_align_corners_true",
        "integration_mode": "time_scaled_euler_project",
        "integration_dtype": "field_dtype",
        "num_steps": int(num_steps),
        "policy_dt": dt,
        "time_schedule": "linspace_0_1_inclusive",
        "effective_step": "policy_dt / ((1 - t) + time_scale_bias * t**time_scale_power)",
        "time_scale_bias": bias,
        "time_scale_power": power,
        "boundary": "cartesian_box_then_polar_sector_projection"
        if ground
        else "clamp_image_xy_0_1",
        "early_stop_on_zero_or_boundary": False,
        "nonfinite": "raise",
        "origin": {
            "mode": "centerline_half_radial_cell_offset" if ground else "image_center",
            "default_radius_normalized": radius if ground else None,
            "explicit_start_radius_floor_normalized": reference_start_radius(
                grid_size, explicit=True
            )
            if ground
            else None,
            "radius_epsilon_m": GROUND_RADIUS_EPS_M if ground else None,
        },
        "model_hfov": {
            "mode": "clamp_to_training_band",
            "range_deg": [MODEL_HFOV_MIN_DEG, MODEL_HFOV_MAX_DEG],
        }
        if ground
        else None,
        "geometry_hfov": "actual_camera_hfov" if ground else None,
    }
