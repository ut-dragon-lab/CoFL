"""SecVLA ground benchmark clock, pursuit and action arbitration semantics.

These implementations are independent of Torch and Habitat. Clock time and
controller velocities are physical quantities; policy_dt only integrates the
network's static field. Control and STOP defaults follow the legacy velocity
benchmark; model replanning defaults to 5 Hz.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from cofl.runtime.inference_rules import DEFAULT_FIELD_GRID_SIZE, DEFAULT_FIELD_QUERY_CHUNK_SIZE


@dataclass(frozen=True)
class RunParameters:
    max_time_s: float = 120.0
    max_steps: int = 500
    plant_hz: float = 100.0
    controller_hz: float = 50.0
    sensor_hz: float = 30.0
    planner_hz: float = 5.0
    policy_steps: int = 100
    policy_dt: float | None = None
    field_grid_size: int = DEFAULT_FIELD_GRID_SIZE
    field_query_chunk_size: int = DEFAULT_FIELD_QUERY_CHUNK_SIZE
    stop_threshold: float = 0.0
    stop_endpoint_m: float = 0.2
    stop_pool_step: float = 0.5
    oscillation_window: int = 4
    oscillation_fwd_stop_max: float = 1.0
    turn_in_place_lookahead_m: float = 1.0
    lookahead_m: float = 0.5
    max_linear_mps: float = 0.5
    max_angular_radps: float = math.pi / 3
    rotate_threshold_deg: float = 30.0

    def __post_init__(self):
        if self.policy_dt is None:
            if (
                isinstance(self.policy_steps, bool)
                or not isinstance(self.policy_steps, int)
                or self.policy_steps < 1
            ):
                raise ValueError("policy_steps must be a positive integer")
            object.__setattr__(self, "policy_dt", 1.0 / self.policy_steps)
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(name + " must be finite")
            if value < 0 or (
                value == 0
                and name
                not in (
                    "stop_threshold",
                    "stop_endpoint_m",
                    "stop_pool_step",
                    "oscillation_window",
                    "oscillation_fwd_stop_max",
                )
            ):
                raise ValueError(name + " is outside its valid range")
        if (
            self.planner_hz > self.controller_hz
            or max(self.controller_hz, self.sensor_hz) > self.plant_hz
        ):
            raise ValueError(
                "Rates require planner_hz <= controller_hz <= plant_hz and sensor_hz <= plant_hz"
            )
        if (
            self.stop_threshold > 1
            or self.stop_pool_step > 1
            or self.oscillation_fwd_stop_max > 1
            or self.rotate_threshold_deg > 90
        ):
            raise ValueError("Invalid stop probability or rotation threshold")
        if any(
            not isinstance(getattr(self, n), int)
            for n in (
                "max_steps",
                "policy_steps",
                "oscillation_window",
                "field_grid_size",
                "field_query_chunk_size",
            )
        ):
            raise ValueError(
                "Step counts, field_grid_size and field_query_chunk_size must be integers"
            )
        if self.field_grid_size < 2:
            raise ValueError("field_grid_size must be at least 2")

    def to_dict(self):
        return asdict(self)


class SimulationClock:
    """Deadline scheduler, including the initial sensor/planner tick at t=0."""

    def __init__(self, params):
        self.params = params
        self.dt = 1.0 / params.plant_hz
        self.t_sim = 0.0
        self.first = True
        self.deadlines = {"planner": 0.0, "controller": 0.0, "sensor": 0.0}

    @property
    def is_done(self):
        return self.t_sim >= self.params.max_time_s

    def tick(self):
        if self.first:
            self.first = False
        else:
            self.t_sim += self.dt
        events = {}
        for name in self.deadlines:
            events[name] = self.t_sim + 1e-9 >= self.deadlines[name]
            if events[name]:
                self.deadlines[name] += 1.0 / getattr(self.params, name + "_hz")
        return events


def xz_basis(rotation_xyzw):
    x, y, z, w = np.asarray(rotation_xyzw, dtype=np.float64)
    forward = np.array([-2 * (x * z + y * w), -(1 - 2 * (x * x + y * y))])
    left = np.array([-(1 - 2 * (y * y + z * z)), -2 * (x * z - y * w)])
    for vector in (forward, left):
        norm = np.linalg.norm(vector)
        if norm > 1e-9:
            vector /= norm
    return forward, left


def camera_to_world(points, position, rotation_xyzw):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    forward, left = xz_basis(rotation_xyzw)
    return np.asarray(position)[[0, 2]] + points[:, :1] * forward + points[:, 1:] * left


def pursuit(position, rotation_xyzw, world_path, params):
    """Legacy ahead-filtered lookahead, cosine speed and fixed-rate yaw gate."""
    path = np.asarray(world_path, dtype=np.float64).reshape(-1, 2)
    if not len(path):
        return (0.0, 0.0), {"reason": "empty_trajectory"}
    forward, left = xz_basis(rotation_xyzw)
    delta = path - np.asarray(position)[[0, 2]]
    distances = np.linalg.norm(delta, axis=1)
    ahead = delta @ forward >= 0
    far = np.flatnonzero(ahead & (distances >= params.lookahead_m))
    candidates = np.flatnonzero(ahead)
    index = int(far[0] if len(far) else candidates[-1] if len(candidates) else len(path) - 1)
    target = delta[index]
    alpha = math.atan2(float(target @ left), float(target @ forward))
    if abs(alpha) > math.radians(params.rotate_threshold_deg):
        linear, angular = 0.0, math.copysign(params.max_angular_radps, alpha)
        mode = "rotate_in_place"
    else:
        linear = params.max_linear_mps * max(math.cos(alpha), 0.0)
        angular = float(
            np.clip(
                linear * 2 * math.sin(alpha) / max(params.lookahead_m, 1e-3),
                -params.max_angular_radps,
                params.max_angular_radps,
            )
        )
        mode = "pure_pursuit"
    return (linear, angular), {
        "mode": mode,
        "target_index": index,
        "alpha_rad": alpha,
        "target_distance_m": float(distances[index]),
    }


class ActionArbitrator:
    """STOP argmax/threshold, degenerate-flow fallback and oscillation rescue.

    The STOP pool follows the benchmark CLI defaults; drainout is disabled. A
    synthetic *control* waypoint is separate from the model trajectory shown
    in the UI. No reference path or ground-truth goal enters policy decisions.
    """

    def __init__(self, params):
        self.params = params
        self.history = deque(maxlen=max(1, params.oscillation_window))
        self.committed = None
        self.stop_pool = 0.0

    def decide(self, trajectory, actions):
        trajectory = np.asarray(trajectory, dtype=np.float64).reshape(-1, 2)
        if not len(trajectory) or not np.isfinite(trajectory).all():
            raise ValueError("Policy produced an empty or nonfinite trajectory")
        probs = None if actions is None else np.asarray(actions, dtype=np.float64)
        action = None
        if probs is not None:
            if probs.shape != (4,) or not np.isfinite(probs).all():
                raise ValueError("Action probabilities must be four finite values")
            action = int(probs.argmax())
            qualifies = action == 0 and probs[0] >= self.params.stop_threshold
            if self.params.stop_pool_step > 0:
                self.stop_pool = float(
                    np.clip(
                        self.stop_pool + self.params.stop_pool_step * (1.0 if qualifies else -0.5),
                        0,
                        1,
                    )
                )
                fire_stop = self.stop_pool >= 1.0
            else:
                fire_stop = qualifies
            if fire_stop:
                action = 0
            elif action == 0:
                action = 1 + int(probs[1:].argmax())
        endpoint = trajectory[-1]
        distance = float(np.linalg.norm(endpoint))
        angle = math.degrees(math.atan2(endpoint[1], endpoint[0])) if distance > 1e-6 else 0.0
        degenerate = self.params.stop_endpoint_m > 0 and distance < self.params.stop_endpoint_m
        rotation = action if degenerate and action in (2, 3) else None
        if not degenerate and abs(angle) > self.params.rotate_threshold_deg:
            rotation = 2 if angle > 0 else 3
        if rotation is None:
            self.committed = None
            self.history.clear()
        else:
            self.history.append((rotation, None if probs is None else float(probs[0] + probs[1])))
            if (
                self.committed is None
                and self.params.oscillation_window > 0
                and len(self.history) == self.params.oscillation_window
            ):
                directions = [d for d, _ in self.history]
                if len(set(directions)) == 2 and all(
                    p is None or p <= self.params.oscillation_fwd_stop_max for _, p in self.history
                ):
                    self.committed = 2 if directions.count(2) >= directions.count(3) else 3
        info = {
            "action_id": action,
            "endpoint_distance_m": distance,
            "flow_degenerate": degenerate,
            "oscillation_committed": self.committed,
            "reason": "track",
            "stop_pool": self.stop_pool,
        }
        override = self.committed if self.committed is not None else action if degenerate else None
        if self.committed is not None:
            info["reason"] = (
                "oscillation_committed_left"
                if self.committed == 2
                else "oscillation_committed_right"
            )
        elif action == 0:
            info["reason"] = "action_head_stop"
            return trajectory, True, info
        elif degenerate and action is None:
            info["reason"] = "trajectory_endpoint"
            return trajectory, True, info
        if override in (1, 2, 3):
            point = {1: (1.0, 0.0), 2: (0.0, 1.0), 3: (0.0, -1.0)}[override]
            if self.committed is None:
                info["reason"] = (
                    "action_head_forward_override" if override == 1 else "action_head_turn_in_place"
                )
            return (
                np.array([[0.0, 0.0], point]) * self.params.turn_in_place_lookahead_m,
                False,
                info,
            )
        return trajectory, False, info
