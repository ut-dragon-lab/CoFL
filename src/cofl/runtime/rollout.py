"""Shared CoFL time-scaled raster inference and explicit legacy Euler utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Mapping  # noqa: UP035 - Callable alias must run on Python 3.7

import numpy as np

from .inference_rules import (
    DEFAULT_FIELD_GRID_SIZE,
    DEFAULT_NUM_STEPS,
    DEFAULT_POLICY_DT,
    DEFAULT_TIME_SCALE_POWER,
    GROUND_RADIUS_EPS_M,
    inference_rule_metadata,
    profile_time_scale_bias,
    reference_start_radius,
)

FieldCallable = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RolloutResult:
    """A local streamline including its initial point and policy-time stamps."""

    points: np.ndarray
    policy_times: np.ndarray
    stop_reason: str
    profile: str
    coordinate_unit: str
    coordinate_frame: str
    vector_unit: str
    query_chart: str
    field_evaluations: int
    integration_mode: str = "time_scaled_euler_project"
    inference_rules: dict[str, Any] | None = None
    clamped_steps: int = 0

    @property
    def policy_time(self) -> float:
        return float(self.policy_times[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": self.points.tolist(),
            "policy_times": self.policy_times.tolist(),
            "stop_reason": self.stop_reason,
            "profile": self.profile,
            "coordinate_unit": self.coordinate_unit,
            "coordinate_frame": self.coordinate_frame,
            "vector_unit": self.vector_unit,
            "query_chart": self.query_chart,
            "field_evaluations": self.field_evaluations,
            "integration_mode": self.integration_mode,
            "inference_rules": self.inference_rules,
            "clamped_steps": self.clamped_steps,
        }


def _geometry(geometry: Mapping[str, Any], profile: str) -> tuple[float, float, float]:
    if not isinstance(geometry, Mapping) or geometry.get("profile") != profile:
        raise ValueError(f"geometry must explicitly identify profile {profile!r}")
    expected = {
        "time_convention": "static_velocity_per_policy_time",
        "coordinate_frame": "image_normalized"
        if profile == "image_field_v1"
        else "body_normalized",
        "component_axes": ["right", "down"] if profile == "image_field_v1" else ["forward", "left"],
    }
    for key, value in expected.items():
        if key in geometry and geometry[key] != value:
            raise ValueError(f"geometry.{key} conflicts with the {profile} rollout convention")
    if profile == "image_field_v1":
        return 1.0, 1.0, math.pi / 2
    values = []
    for key in ("normalization_scale_m", "r_max_m", "hfov_rad"):
        value = geometry.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"geometry.{key} must be finite and positive")
        values.append(float(value))
    scale, rmax, hfov = values
    if hfov >= math.pi:
        raise ValueError("geometry.hfov_rad must be smaller than pi")
    if scale != rmax:
        raise ValueError("ground_sector_v1 requires r_max_m == normalization_scale_m")
    return scale, rmax, hfov


def _settings(
    field: FieldCallable,
    max_steps: int,
    policy_dt: float,
    min_field_norm: float,
    boundary: str,
    nonfinite: str,
) -> None:
    if not callable(field):
        raise TypeError("field must be callable")
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if isinstance(policy_dt, bool) or not math.isfinite(policy_dt) or policy_dt <= 0:
        raise ValueError("policy_dt must be finite and positive")
    if isinstance(min_field_norm, bool) or not math.isfinite(min_field_norm) or min_field_norm < 0:
        raise ValueError("min_field_norm must be finite and nonnegative")
    if boundary not in ("stop", "truncate"):
        raise ValueError("boundary must be 'stop' or 'truncate'")
    if nonfinite not in ("stop", "raise"):
        raise ValueError("nonfinite must be 'stop' or 'raise'")


def _boundary_fraction(
    point: np.ndarray, candidate: np.ndarray, inside: Callable[[np.ndarray], bool]
) -> float:
    """Find the last in-domain point on a segment in either convex profile domain."""
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        if inside(point + mid * (candidate - point)):
            low = mid
        else:
            high = mid
    return low


def _rollout(
    field: FieldCallable,
    start: np.ndarray,
    *,
    profile: str,
    vector_scale: float,
    query: Callable[[np.ndarray], np.ndarray],
    inside: Callable[[np.ndarray], bool],
    max_steps: int,
    policy_dt: float,
    min_field_norm: float,
    boundary: str,
    nonfinite: str,
) -> RolloutResult:
    _settings(field, max_steps, policy_dt, min_field_norm, boundary, nonfinite)
    position = np.asarray(start, dtype=np.float64)
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("The starting point must be a finite vector of shape (2,)")
    if not inside(position):
        raise ValueError("The starting point lies outside the profile domain")
    points, times = [position.copy()], [0.0]
    evaluations = 0
    reason = "max_steps"
    for _ in range(max_steps):
        value = np.asarray(field(query(points[-1]).reshape(1, 2)), dtype=np.float64)
        evaluations += 1
        if value.shape != (1, 2):
            raise ValueError("field(query) must return an array of shape (1, 2)")
        if not np.isfinite(value).all():
            if nonfinite == "raise":
                raise ValueError("field(query) returned a nonfinite vector")
            reason = "nonfinite_field"
            break
        if float(np.hypot(value[0, 0], value[0, 1])) <= min_field_norm:
            reason = "zero_field"
            break
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = points[-1] + policy_dt * vector_scale * value[0]
        if not np.isfinite(candidate).all():
            if nonfinite == "raise":
                raise ValueError("The Euler update produced a nonfinite point")
            reason = "nonfinite_step"
            break
        if not inside(candidate):
            if boundary == "truncate":
                fraction = _boundary_fraction(points[-1], candidate, inside)
                truncated = points[-1] + fraction * (candidate - points[-1])
                if fraction > 0 and not np.array_equal(truncated, points[-1]):
                    points.append(truncated)
                    times.append(times[-1] + fraction * policy_dt)
            reason = "boundary"
            break
        points.append(candidate)
        times.append(times[-1] + policy_dt)
    image = profile == "image_field_v1"
    return RolloutResult(
        points=np.asarray(points, dtype=np.float64),
        policy_times=np.asarray(times, dtype=np.float64),
        stop_reason=reason,
        profile=profile,
        coordinate_unit="image_fraction" if image else "m",
        coordinate_frame="image_xy_right_down"
        if image
        else "body_xy_forward_left_fixed_at_observation",
        vector_unit="image_fraction_per_policy_time" if image else "m_per_policy_time",
        query_chart="image_xy_normalized" if image else "sector_theta_normalized_radius_normalized",
        field_evaluations=evaluations,
        integration_mode="static_euler_policy_time",
    )


@dataclass(frozen=True)
class RasterRollout:
    """Batch paths [B,T+1,2], scheduled commands [B,T,2], projection counts.

    Commands include time scaling and precede boundary projection, so they need
    not equal actual path displacement / policy_dt. Ground units are metres.
    """

    points: np.ndarray
    velocities: np.ndarray
    clamped_steps: np.ndarray


def _integration_settings(num_steps, policy_dt, time_scale_bias, time_scale_power, dtype):
    if (
        isinstance(num_steps, (bool, np.bool_))
        or not isinstance(num_steps, (int, np.integer))
        or num_steps < 1
    ):
        raise ValueError("num_steps must be a positive integer")
    for name, value in (
        ("policy_dt", policy_dt),
        ("time_scale_bias", time_scale_bias),
        ("time_scale_power", time_scale_power),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite positive number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        with np.errstate(over="ignore", under="ignore"):
            represented = np.asarray(value, dtype=dtype)
        if not np.isfinite(represented) or represented <= 0:
            raise ValueError(f"{name} must be representable and positive in the field dtype")


def _validated_fields(fields):
    fields = np.asarray(fields)
    if fields.ndim != 4 or fields.shape[1] != 2 or min(fields.shape) < 1:
        raise ValueError("fields must have nonempty shape [B,2,H,W]")
    if fields.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("fields must use float32 or float64")
    if not np.isfinite(fields).all():
        raise ValueError("fields must contain only finite vectors")
    return fields


def _validated_starts(starts, batch, *, image, rmax=1.0, hfov=math.pi / 2, grid_size=100):
    default = starts is None
    if default:
        if image:
            raise ValueError("image starts must be provided")
        starts = np.tile([rmax * reference_start_radius(grid_size), 0.0], (batch, 1))
    starts = np.asarray(starts)
    if starts.shape != (batch, 2):
        raise ValueError("starts must have shape [B,2] matching the field batch")
    if starts.dtype.kind not in "fiu":
        raise TypeError("starts must contain numeric coordinates")
    if not np.isfinite(starts).all():
        raise ValueError("starts must contain finite coordinates")
    if image:
        if np.any((starts < 0) | (starts > 1)):
            raise ValueError("starts must be finite image coordinates in [0,1]")
        return starts
    radius = np.hypot(starts[:, 0], starts[:, 1])
    angle = np.arctan2(starts[:, 1], starts[:, 0])
    # Accept ordinary roundoff at explicit physical boundaries.
    if np.any(radius > rmax + 1e-7 * rmax) or np.any(np.abs(angle) > hfov / 2 + 1e-7):
        raise ValueError("The starting point lies outside the profile domain")
    radius = np.maximum(radius, rmax * reference_start_radius(grid_size, explicit=not default))
    radius = np.minimum(radius, rmax)
    angle = np.clip(angle, -hfov / 2, hfov / 2)
    return np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)


def _integrate_batch(
    sample,
    starts,
    *,
    image,
    dtype,
    num_steps,
    policy_dt,
    time_scale_bias,
    time_scale_power,
    vector_scale=1.0,
    rmax=1.0,
    hfov=math.pi / 2,
):
    """Single CPU tensor integration loop shared by raster and callable APIs."""
    import torch

    _integration_settings(num_steps, policy_dt, time_scale_bias, time_scale_power, dtype)
    with torch.inference_mode():
        current = torch.from_numpy(np.array(starts, dtype=dtype, copy=True))
        batch = len(starts)
        points = torch.empty((batch, num_steps + 1, 2), dtype=current.dtype, device="cpu")
        velocities = torch.empty((batch, num_steps, 2), dtype=current.dtype, device="cpu")
        clamped_steps = torch.zeros(batch, dtype=torch.int64, device="cpu")
        points[:, 0] = current
        schedule = torch.linspace(0, 1, steps=int(num_steps), dtype=current.dtype, device="cpu")
        denominators = (1 - schedule) + float(time_scale_bias) * schedule.pow(
            float(time_scale_power)
        )
        if not bool((torch.isfinite(denominators) & denominators.gt(0)).all()):
            raise ValueError("The time-scaling schedule must remain finite and positive")
        if not image:
            query = torch.stack(
                (
                    torch.atan2(current[:, 1], current[:, 0]) / (hfov / 2),
                    torch.sqrt(current[:, 0].square() + current[:, 1].square()) / rmax,
                ),
                dim=-1,
            )
        effective_steps = float(policy_dt) / denominators
        for step in range(num_steps):
            if image:
                query = current
            value = sample(query)
            metric_value = value * float(vector_scale)
            velocity = metric_value / denominators[step]
            # Ground keeps SecVLA's dt_eff multiplication order. Image keeps
            # the existing CoFL offline command-then-dt arithmetic contract.
            candidate = current + (
                float(policy_dt) * velocity if image else metric_value * effective_steps[step]
            )
            if not bool((torch.isfinite(velocity) & torch.isfinite(candidate)).all()):
                raise ValueError("The raster rollout produced a nonfinite velocity or Euler step")
            if image:
                clipped = ((candidate < 0) | (candidate > 1)).any(dim=-1)
                current = candidate.clamp(0, 1)
            else:
                forward = candidate[:, 0].clamp(0, rmax)
                left = candidate[:, 1].clamp(-rmax, rmax)
                raw_radius = torch.sqrt(forward.square() + left.square())
                radius = raw_radius.clamp_min(GROUND_RADIUS_EPS_M)
                angle = torch.atan2(left, forward)
                clipped = (
                    (candidate[:, 0] < 0)
                    | (candidate[:, 0] > rmax)
                    | (candidate[:, 1].abs() > rmax)
                    | (radius > rmax)
                    | (raw_radius < GROUND_RADIUS_EPS_M)
                    | (angle.abs() > hfov / 2)
                )
                radius_normalized = (radius / rmax).clamp(0, 1)
                theta_normalized = (angle / (hfov / 2)).clamp(-1, 1)
                query = torch.stack((theta_normalized, radius_normalized), dim=-1)
                # Preserve the original polar chart between steps, as SecVLA
                # does, and reconstruct metric positions with the actual HFOV.
                radius = radius_normalized * rmax
                angle = theta_normalized * (hfov / 2)
                current = torch.stack(
                    (radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1
                )
            clamped_steps += clipped
            points[:, step + 1] = current
            velocities[:, step] = velocity
        return RasterRollout(points.numpy(), velocities.numpy(), clamped_steps.numpy())


def _raster_sampler(fields, *, image):
    import torch
    from torch.nn import functional as F

    raster = torch.from_numpy(np.array(fields, copy=True, order="C"))
    batch = len(fields)

    def sample(query):
        coordinates = (
            2 * query - 1 if image else torch.stack((query[:, 0], 2 * query[:, 1] - 1), dim=-1)
        )
        return F.grid_sample(
            raster,
            coordinates.reshape(batch, 1, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).reshape(batch, 2)

    return sample


def rollout_image_fields(
    fields,
    starts,
    *,
    num_steps=DEFAULT_NUM_STEPS,
    policy_dt=DEFAULT_POLICY_DT,
    time_scale_bias=None,
    time_scale_power=DEFAULT_TIME_SCALE_POWER,
):
    """Batched CPU endpoint rasters [B,2,H,W] with standard image inference.

    Readout is bilinear/border/align_corners=True. The endpoint-inclusive time
    schedule scales velocities before each Euler update; XY clamps to [0,1].
    Returns every point, including zero fields and projected boundary steps.
    """
    fields = _validated_fields(fields)
    starts = _validated_starts(starts, len(fields), image=True)
    return _integrate_batch(
        _raster_sampler(fields, image=True),
        starts,
        image=True,
        dtype=fields.dtype,
        num_steps=num_steps,
        policy_dt=policy_dt,
        time_scale_bias=profile_time_scale_bias("image_field_v1")
        if time_scale_bias is None
        else time_scale_bias,
        time_scale_power=time_scale_power,
    )


def rollout_sector_fields(
    fields,
    starts_m=None,
    *,
    geometry,
    num_steps=DEFAULT_NUM_STEPS,
    policy_dt=DEFAULT_POLICY_DT,
    time_scale_bias=None,
    time_scale_power=DEFAULT_TIME_SCALE_POWER,
    field_grid_size=DEFAULT_FIELD_GRID_SIZE,
):
    """Batched CPU polar rasters [B,2,N_r,N_theta], returning metric paths.

    Cartesian normalized field vectors are scaled to metres and integrated,
    projected to the BEV box and then the actual camera's polar sector. Default
    starts use the SecVLA reference offset; explicit starts retain their angle.
    """
    fields = _validated_fields(fields)
    reference_start_radius(field_grid_size)
    if min(fields.shape[2:]) < 2:
        raise ValueError("Ground fields require at least two radial and angular samples")
    field_grid_size = int(fields.shape[2])
    scale, rmax, hfov = _geometry(geometry, "ground_sector_v1")
    starts = _validated_starts(
        starts_m, len(fields), image=False, rmax=rmax, hfov=hfov, grid_size=field_grid_size
    )
    return _integrate_batch(
        _raster_sampler(fields, image=False),
        starts,
        image=False,
        dtype=fields.dtype,
        num_steps=num_steps,
        policy_dt=policy_dt,
        time_scale_bias=profile_time_scale_bias("ground_sector_v1")
        if time_scale_bias is None
        else time_scale_bias,
        time_scale_power=time_scale_power,
        vector_scale=scale,
        rmax=rmax,
        hfov=hfov,
    )


def _standard_rollout(
    field,
    start,
    *,
    geometry,
    profile,
    max_steps,
    policy_dt,
    time_scale_bias,
    time_scale_power,
    field_grid_size,
    min_field_norm,
    nonfinite,
):
    if not callable(field):
        raise TypeError("field must be callable")
    if nonfinite != "raise":
        raise ValueError("Standard inference requires nonfinite='raise'")
    if (
        isinstance(min_field_norm, bool)
        or not isinstance(min_field_norm, Real)
        or not math.isfinite(min_field_norm)
        or min_field_norm < 0
    ):
        raise ValueError("min_field_norm must be finite and nonnegative")
    scale, rmax, hfov = _geometry(geometry, profile)
    image = profile == "image_field_v1"
    from .field_grid import RasterField

    reference_start_radius(field_grid_size)
    if isinstance(field, RasterField):
        if field.profile != profile:
            raise ValueError("The raster field profile differs from rollout geometry")
        # A concrete raster is authoritative for origin spacing and metadata;
        # field_grid_size supplies that information for generic callables.
        field_grid_size = int(field.values.shape[0])
    starts = None if start is None else np.asarray(start)[None]
    starts = _validated_starts(
        starts, 1, image=image, rmax=rmax, hfov=hfov, grid_size=field_grid_size
    )
    if isinstance(field, RasterField):
        dtype = field.values.dtype
        sample = _raster_sampler(field.values.transpose(2, 0, 1)[None], image=image)
    else:
        import torch

        dtype = np.dtype("float64")

        def sample(query):
            value = np.asarray(field(query.numpy()), dtype=dtype)
            if value.shape != (1, 2):
                raise ValueError("field(query) must return an array of shape (1, 2)")
            return torch.from_numpy(np.array(value, copy=True))

    result = _integrate_batch(
        sample,
        starts,
        image=image,
        dtype=dtype,
        num_steps=max_steps,
        policy_dt=policy_dt,
        time_scale_bias=profile_time_scale_bias(profile)
        if time_scale_bias is None
        else time_scale_bias,
        time_scale_power=time_scale_power,
        vector_scale=scale,
        rmax=rmax,
        hfov=hfov,
    )
    rules = inference_rule_metadata(
        profile,
        grid_size=field_grid_size,
        num_steps=max_steps,
        policy_dt=policy_dt,
        time_scale_bias=time_scale_bias,
        time_scale_power=time_scale_power,
    )
    if isinstance(field, RasterField):
        rules["field_grid_shape"] = list(field.values.shape[:2])
    return RolloutResult(
        points=result.points[0],
        policy_times=np.arange(max_steps + 1, dtype=np.float64) * policy_dt,
        stop_reason="max_steps",
        profile=profile,
        coordinate_unit="image_fraction" if image else "m",
        coordinate_frame="image_xy_right_down"
        if image
        else "body_xy_forward_left_fixed_at_observation",
        vector_unit="image_fraction_per_policy_time" if image else "m_per_policy_time",
        query_chart="image_xy_normalized" if image else "sector_theta_normalized_radius_normalized",
        field_evaluations=int(max_steps),
        inference_rules=rules,
        clamped_steps=int(result.clamped_steps[0]),
    )


def rollout_image_field(
    field: FieldCallable,
    *,
    geometry: Mapping[str, Any],
    start=(0.5, 0.5),
    max_steps=DEFAULT_NUM_STEPS,
    policy_dt=DEFAULT_POLICY_DT,
    min_field_norm=1e-8,
    boundary="project",
    nonfinite="raise",
    time_scale_bias=None,
    time_scale_power=DEFAULT_TIME_SCALE_POWER,
) -> RolloutResult:
    """CoFL image inference: scheduled Euler updates, clamp and continue.

    A precomputed RasterField uses the shared CPU batch integrator. Generic
    analytic callables remain supported. min_field_norm is ignored by standard
    inference: zero fields still produce T+1 points. Explicit boundary='stop'
    or 'truncate' selects the historical, nonstandard fixed-Euler utility.
    """
    _geometry(geometry, "image_field_v1")
    if boundary == "project":
        return _standard_rollout(
            field,
            start,
            geometry=geometry,
            profile="image_field_v1",
            max_steps=max_steps,
            policy_dt=policy_dt,
            time_scale_bias=time_scale_bias,
            time_scale_power=time_scale_power,
            field_grid_size=DEFAULT_FIELD_GRID_SIZE,
            min_field_norm=min_field_norm,
            nonfinite=nonfinite,
        )
    return _rollout(
        field,
        start,
        profile="image_field_v1",
        vector_scale=1.0,
        query=lambda point: point.copy(),
        inside=lambda point: bool(np.all((point >= 0) & (point <= 1))),
        max_steps=max_steps,
        policy_dt=policy_dt,
        min_field_norm=min_field_norm,
        boundary=boundary,
        nonfinite=nonfinite,
    )


def rollout_sector_field(
    field: FieldCallable,
    *,
    geometry: Mapping[str, Any],
    start_m=None,
    max_steps=DEFAULT_NUM_STEPS,
    policy_dt=DEFAULT_POLICY_DT,
    min_field_norm=1e-8,
    boundary="project",
    nonfinite="raise",
    time_scale_bias=None,
    time_scale_power=DEFAULT_TIME_SCALE_POWER,
    field_grid_size=DEFAULT_FIELD_GRID_SIZE,
) -> RolloutResult:
    """CoFL ground inference using Cartesian vectors and a polar query chart.

    Ground vectors are scaled by normalization_scale_m before scheduled metric
    Euler steps. Each step projects through the BEV box and actual-HFOV sector,
    then continues. None starts at the centerline reference offset; explicit
    valid starts keep their angle with a half-radial-cell minimum radius. Only
    explicit boundary='stop'/'truncate' requests historical fixed Euler.
    """
    scale, rmax, hfov = _geometry(geometry, "ground_sector_v1")
    if boundary == "project":
        return _standard_rollout(
            field,
            start_m,
            geometry=geometry,
            profile="ground_sector_v1",
            max_steps=max_steps,
            policy_dt=policy_dt,
            time_scale_bias=time_scale_bias,
            time_scale_power=time_scale_power,
            field_grid_size=field_grid_size,
            min_field_norm=min_field_norm,
            nonfinite=nonfinite,
        )

    def query(point):
        return np.array([math.atan2(point[1], point[0]) / (hfov / 2), math.hypot(*point) / rmax])

    def inside(point):
        return bool(math.hypot(*point) <= rmax and abs(math.atan2(point[1], point[0])) <= hfov / 2)

    return _rollout(
        field,
        (0.0, 0.0) if start_m is None else start_m,
        profile="ground_sector_v1",
        vector_scale=scale,
        query=query,
        inside=inside,
        max_steps=max_steps,
        policy_dt=policy_dt,
        min_field_norm=min_field_norm,
        boundary=boundary,
        nonfinite=nonfinite,
    )
