"""NumPy evaluation with explicit profile coordinates and physical units."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


PROFILES = ("image_field_v1", "ground_sector_v1")


def _profile_scale(profile: str, geometry: Mapping[str, Any] | None) -> tuple[float, str]:
    if profile == "image_field_v1":
        return 1.0, "image_fraction"
    if profile != "ground_sector_v1":
        raise ValueError(f"Unsupported profile: {profile!r}; expected one of {PROFILES}")
    if geometry is None or "normalization_scale_m" not in geometry:
        raise ValueError("Ground evaluation requires geometry['normalization_scale_m']")
    scale = float(geometry["normalization_scale_m"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("normalization_scale_m must be finite and positive")
    return scale, "m"


def evaluate_field(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    profile: str,
    geometry: Mapping[str, Any] | None = None,
    mask: np.ndarray | None = None,
    magnitude_epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Evaluate matching ``(..., 2)`` vector fields in native profile units.

    ``mask`` has shape ``(...)``; false cells contribute to no metric. Selected
    cells must be finite. Vector and magnitude errors include zero targets.
    Angular metrics include only targets with norm above ``magnitude_epsilon``
    (in native field units). A zero prediction on a directional target receives
    a 180-degree penalty, so predicting zero cannot evade the directional score.
    Undefined aggregate metrics are ``None``; valid counts remain explicit.

    Ground vectors are multiplied by ``normalization_scale_m``. The time unit
    is policy time, not wall-clock seconds: neither profile implies m/s.
    """
    scale, coordinate_unit = _profile_scale(profile, geometry)
    if not np.isfinite(magnitude_epsilon) or magnitude_epsilon <= 0:
        raise ValueError("magnitude_epsilon must be finite and positive")
    pred = np.asarray(prediction)
    truth = np.asarray(target)
    if pred.shape != truth.shape or pred.ndim < 2 or pred.shape[-1] != 2:
        raise ValueError("prediction and target must have identical (..., 2) shapes")
    if mask is None:
        valid = np.ones(pred.shape[:-1], dtype=bool)
    else:
        valid = np.asarray(mask)
        if valid.shape != pred.shape[:-1] or valid.dtype != np.bool_:
            raise ValueError("mask must be a boolean array matching the field spatial shape")
    valid_count = int(np.count_nonzero(valid))
    # Dense image fields are usually entirely valid. Reshaping avoids two
    # advanced-index copies; sparse fields convert only the selected values.
    if valid_count == valid.size:
        p, t = pred.reshape(-1, 2), truth.reshape(-1, 2)
    else:
        p, t = pred[valid], truth[valid]
    p, t = np.asarray(p, dtype=np.float64), np.asarray(t, dtype=np.float64)
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Selected prediction and target vectors must be finite")
    pn, tn = np.linalg.norm(p, axis=-1), np.linalg.norm(t, axis=-1)
    directional = tn > magnitude_epsilon
    zero_prediction = pn <= magnitude_epsilon
    directional_count = int(np.count_nonzero(directional))
    angles = np.full(directional_count, 180.0, dtype=np.float64)
    nonzero = ~zero_prediction[directional]
    if nonzero.any():
        directed = directional & ~zero_prediction
        unit_p = p[directed] / pn[directed, None]
        unit_t = t[directed] / tn[directed, None]
        angles[nonzero] = np.degrees(
            np.arccos(np.clip(np.sum(unit_p * unit_t, axis=-1), -1.0, 1.0))
        )
    vector_error = np.linalg.norm(p - t, axis=-1) * scale
    magnitude_error = np.abs(pn - tn) * scale
    return {
        "profile": profile,
        "vector_error_unit": f"{coordinate_unit}_per_policy_time",
        "total_count": int(valid.size),
        "valid_count": valid_count,
        "masked_count": int(valid.size - valid_count),
        "directional_count": directional_count,
        "zero_target_count": valid_count - directional_count,
        "zero_prediction_directional_count": directional_count - int(np.count_nonzero(nonzero)),
        "vector_l2_mean": float(vector_error.mean()) if vector_error.size else None,
        "magnitude_error_mean": float(magnitude_error.mean()) if magnitude_error.size else None,
        "angular_error_deg_mean": float(angles.mean()) if angles.size else None,
        "directional_accuracy_30deg": float((angles < 30.0).mean()) if angles.size else None,
        "zero_prediction_angular_penalty_deg": 180.0,
        "magnitude_epsilon_native": float(magnitude_epsilon),
    }


def _trajectory(points: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or not len(result):
        raise ValueError(f"{name} must have nonempty shape (N, 2)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite coordinates")
    return result


def resample_trajectory(points: np.ndarray, num_points: int = 101) -> np.ndarray:
    """Resample a finite 2D polyline uniformly in arc length, including endpoints."""
    path = _trajectory(points, "points")
    if (
        isinstance(num_points, bool)
        or not isinstance(num_points, (int, np.integer))
        or num_points < 2
    ):
        raise ValueError("num_points must be an integer >= 2")
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=-1))]
    keep = np.r_[True, np.diff(arc) > 0]
    arc, path = arc[keep], path[keep]
    if len(path) == 1:
        return np.repeat(path, num_points, axis=0)
    query = np.linspace(0.0, arc[-1], num_points)
    return np.column_stack([np.interp(query, arc, path[:, i]) for i in range(2)])


def _trajectory_metrics(
    prediction: np.ndarray, target: np.ndarray, unit: str, num_points: int
) -> dict[str, Any]:
    pred, truth = _trajectory(prediction, "prediction"), _trajectory(target, "target")
    p, t = resample_trajectory(pred, num_points), resample_trajectory(truth, num_points)
    pred_length = float(np.linalg.norm(np.diff(pred, axis=0), axis=-1).sum())
    truth_length = float(np.linalg.norm(np.diff(truth, axis=0), axis=-1).sum())
    segments = np.diff(p, axis=0)
    segments = segments[np.linalg.norm(segments, axis=-1) > 1e-12]
    if len(segments) >= 2:
        headings = np.arctan2(segments[:, 1], segments[:, 0])
        turns = (np.diff(headings) + np.pi) % (2 * np.pi) - np.pi
        turning = float(np.degrees(np.abs(turns)).mean())
    else:
        turning = None
    return {
        "coordinate_unit": unit,
        "final_goal_error": float(np.linalg.norm(pred[-1] - truth[-1])),
        "predicted_path_length": pred_length,
        "reference_path_length": truth_length,
        "path_length_ratio": pred_length / truth_length if truth_length > 1e-12 else None,
        "arc_matched_error_mean": float(np.linalg.norm(p - t, axis=-1).mean()),
        "mean_absolute_turn_deg": turning,
        "resample_points": int(num_points),
    }


def evaluate_image_trajectory(
    prediction: np.ndarray, target: np.ndarray, *, num_points: int = 101
) -> dict[str, Any]:
    """Evaluate normalized image XY polylines; distances are image fractions.

    Out-of-image points are preserved. No clipping, calibration, collision
    scoring, or interpretation as physical navigation success is performed.
    """
    result = _trajectory_metrics(prediction, target, "image_fraction", num_points)
    result["coordinate_frame"] = "image_xy_right_down"
    return result


def evaluate_ground_trajectory(
    prediction_m: np.ndarray,
    target_m: np.ndarray,
    *,
    num_points: int = 101,
    success_radius_m: float | None = None,
) -> dict[str, Any]:
    """Evaluate Cartesian ground polylines in metres in the same fixed frame.

    Optional goal reach is Euclidean distance only, with no STOP, obstacles,
    navmesh, or geodesic assumptions. It is not a VLN success or SPL score.
    """
    result = _trajectory_metrics(prediction_m, target_m, "m", num_points)
    result["coordinate_frame"] = "ground_cartesian_same_fixed_frame"
    if success_radius_m is not None:
        if not np.isfinite(success_radius_m) or success_radius_m <= 0:
            raise ValueError("success_radius_m must be finite and positive")
        result["success_radius_m"] = float(success_radius_m)
        result["euclidean_goal_reached"] = bool(result["final_goal_error"] <= success_radius_m)
    return result


def evaluate_trajectory(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    profile: str,
    geometry: Mapping[str, Any] | None = None,
    num_points: int = 101,
) -> dict[str, Any]:
    """Evaluate native profile coordinates, converting ground coordinates to metres."""
    scale, _ = _profile_scale(profile, geometry)
    if profile == "image_field_v1":
        return evaluate_image_trajectory(prediction, target, num_points=num_points)
    return evaluate_ground_trajectory(
        np.asarray(prediction) * scale, np.asarray(target) * scale, num_points=num_points
    )
