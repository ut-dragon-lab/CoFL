"""Small planar controller for deterministic synthetic execution examples."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PlanarState:
    """Cartesian position in metres, heading in radians counterclockwise from +X."""

    x_m: float
    y_m: float
    heading_rad: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x_m, self.y_m, self.heading_rad)):
            raise ValueError("State must be finite")

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x_m, self.y_m], dtype=np.float64)


@dataclass(frozen=True)
class VelocityCommand:
    """Forward velocity in m/s and counterclockwise yaw velocity in rad/s."""

    linear_mps: float = 0.0
    angular_radps: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.linear_mps, self.angular_radps)):
            raise ValueError("Velocity command must be finite")


def integrate_unicycle(state: PlanarState, command: VelocityCommand, dt_s: float) -> PlanarState:
    """Exact integration of a constant command in a collision-free plane."""
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    yaw = state.heading_rad
    half_turn = command.angular_radps * dt_s / 2.0
    # np.sinc(z) = sin(pi*z)/(pi*z), including its continuous value at zero.
    distance = command.linear_mps * dt_s * float(np.sinc(half_turn / math.pi))
    return PlanarState(
        state.x_m + distance * math.cos(yaw + half_turn),
        state.y_m + distance * math.sin(yaw + half_turn),
        (yaw + 2 * half_turn + math.pi) % (2 * math.pi) - math.pi,
    )


def _lookahead_point(
    position: np.ndarray, trajectory: np.ndarray, lookahead_m: float
) -> np.ndarray:
    path = np.asarray(trajectory, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or not len(path) or not np.isfinite(path).all():
        raise ValueError("trajectory must be a nonempty finite (N, 2) metre array")
    if len(path) == 1:
        return path[0]
    starts, segments = path[:-1], np.diff(path, axis=0)
    squared_lengths = np.sum(segments * segments, axis=-1)
    factors = np.zeros(len(segments), dtype=np.float64)
    nonzero = squared_lengths > 0
    factors[nonzero] = (
        np.sum((position - starts[nonzero]) * segments[nonzero], axis=-1) / squared_lengths[nonzero]
    )
    factors = np.clip(factors, 0, 1)
    projections = starts + factors[:, None] * segments
    closest = int(np.argmin(np.linalg.norm(projections - position, axis=-1)))
    point = projections[closest]
    remaining = lookahead_m
    for endpoint in path[closest + 1 :]:
        distance = float(np.linalg.norm(endpoint - point))
        if distance >= remaining and distance > 0:
            return point + (endpoint - point) * (remaining / distance)
        remaining -= distance
        point = endpoint
    return path[-1]


@dataclass(frozen=True)
class PurePursuitController:
    """Forward-only pursuit with turn-in-place for a target behind the robot.

    This controller models no wheel dynamics, acceleration limit or collision
    response. Its explicit rates and speed limits belong to the benchmark recipe.
    """

    lookahead_m: float = 0.4
    max_linear_mps: float = 0.6
    max_angular_radps: float = 1.5
    goal_radius_m: float = 0.08

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v) and v > 0
            for v in (
                self.lookahead_m,
                self.max_linear_mps,
                self.max_angular_radps,
                self.goal_radius_m,
            )
        ):
            raise ValueError("Controller distances and speed limits must be finite and positive")

    def compute(self, state: PlanarState, trajectory_m: np.ndarray) -> VelocityCommand:
        target = _lookahead_point(state.position, trajectory_m, self.lookahead_m)
        path = np.asarray(trajectory_m, dtype=np.float64)
        goal_distance = float(np.linalg.norm(path[-1] - state.position))
        if goal_distance <= self.goal_radius_m:
            return VelocityCommand()
        delta = target - state.position
        alpha = (math.atan2(delta[1], delta[0]) - state.heading_rad + math.pi) % (
            2 * math.pi
        ) - math.pi
        if abs(alpha) >= math.pi / 2:
            return VelocityCommand(0.0, math.copysign(self.max_angular_radps, alpha))
        linear = min(self.max_linear_mps, goal_distance) * max(math.cos(alpha), 0.0)
        angular = 2.0 * linear * math.sin(alpha) / max(float(np.linalg.norm(delta)), 1e-9)
        return VelocityCommand(
            linear, float(np.clip(angular, -self.max_angular_radps, self.max_angular_radps))
        )
