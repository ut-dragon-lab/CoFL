"""Public navigation policy contracts, independent of model and simulator code.

Local trajectory points are metres in the agent's horizontal frame:
``[forward, left]``. A stationary decision is HOLD unless the policy explicitly
emits STOP; an empty or zero direction never implies successful termination.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np


def _array_copy(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class EpisodeContext:
    session_id: str
    episode_id: str
    instruction: str
    instruction_mode: str
    sensors: dict
    planner_hz: float

    def __post_init__(self):
        object.__setattr__(self, "sensors", deepcopy(self.sensors))


@dataclass(frozen=True)
class ExecutionFeedback:
    mode: str
    elapsed_sim_s: float
    linear_mps: float
    angular_radps: float
    blocked_substeps: int


@dataclass(frozen=True)
class PolicyObservation:
    session_id: str
    step_index: int
    t_sim_s: float
    image: np.ndarray
    depth: np.ndarray | None
    instruction: str
    instruction_revision: int
    instruction_id: str
    feedback: ExecutionFeedback | None

    def __post_init__(self):
        object.__setattr__(self, "image", _array_copy(self.image))
        if self.depth is not None:
            object.__setattr__(self, "depth", _array_copy(self.depth))


@dataclass(frozen=True)
class NavigationDecision:
    """A trajectory to track, an explicit hold, or an explicit episode stop.

    Trajectories gain an exact ``[0, 0]`` origin when their first point is not
    already the origin. All supplied lookahead points are retained. Arrays and
    diagnostics are copied so validation never changes the caller's data.
    """

    mode: Literal["track", "hold", "stop"] = "track"
    trajectory: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in ("track", "hold", "stop"):
            raise ValueError("Navigation decision mode must be track, hold, or stop")
        if not isinstance(self.diagnostics, dict):
            raise ValueError("Navigation diagnostics must be a dictionary")  # noqa: TRY004 — invalid decision
        object.__setattr__(self, "diagnostics", deepcopy(self.diagnostics))
        if self.mode != "track":
            if self.trajectory is not None and np.asarray(self.trajectory).size:
                raise ValueError("HOLD and STOP cannot include a nonempty trajectory")
            object.__setattr__(self, "trajectory", None)
            return
        try:
            if np.iscomplexobj(self.trajectory):
                raise ValueError("Complex trajectory coordinates are not supported")
            points = np.array(self.trajectory, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "TRACK requires finite Nx2 forward/left coordinates in metres"
            ) from error
        if points.shape == (2,):
            points = points.reshape(1, 2)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points):
            raise ValueError("TRACK requires a nonempty Nx2 trajectory in metres")
        if not np.isfinite(points).all():
            raise ValueError("TRACK trajectory coordinates must be finite")
        if not np.any(np.hypot(points[:, 0], points[:, 1]) > 1e-8):
            raise ValueError("TRACK requires a nonzero point; use HOLD for no motion")
        if np.any(points[0] != 0):
            points = np.concatenate((np.zeros((1, 2)), points), axis=0)
        points.setflags(write=False)
        object.__setattr__(self, "trajectory", points)

    @classmethod
    def track(cls, points, diagnostics=None):
        return cls("track", points, {} if diagnostics is None else diagnostics)

    @classmethod
    def hold(cls, diagnostics=None):
        return cls("hold", diagnostics={} if diagnostics is None else diagnostics)

    @classmethod
    def stop(cls, diagnostics=None):
        return cls("stop", diagnostics={} if diagnostics is None else diagnostics)


def direction_to_trajectory(direction, length_m=0.25, epsilon=1e-6) -> NavigationDecision:
    """Normalize a local forward/left direction to a fixed lookahead distance.

    Directions whose norm is at most ``epsilon`` produce HOLD. The direction
    must have shape ``(2,)`` and finite real values; length and epsilon must be
    finite and strictly positive.
    """
    try:
        if np.iscomplexobj(direction):
            raise ValueError("Complex directions are not supported")
        vector = np.array(direction, dtype=np.float64, copy=True)
        length_m, epsilon = float(length_m), float(epsilon)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Direction, length_m, and epsilon must be finite real values") from error
    if vector.shape != (2,) or not np.isfinite(vector).all():
        raise ValueError("Direction must contain two finite forward/left components")
    if not np.isfinite(length_m) or length_m <= 0:
        raise ValueError("length_m must be finite and strictly positive")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive")
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        return NavigationDecision.hold()
    # Scale before taking the norm so finite large directions cannot overflow.
    scaled = vector / scale
    norm = float(np.hypot(scaled[0], scaled[1]))
    if scale <= epsilon and norm <= epsilon / scale:
        return NavigationDecision.hold()
    return NavigationDecision.track((scaled / norm) * length_m)


class NavigationPolicy(Protocol):
    def describe(self) -> dict: ...

    def begin_episode(self, context: EpisodeContext) -> None: ...

    def act(self, observation: PolicyObservation) -> NavigationDecision: ...

    def end_episode(self, reason: str) -> None: ...

    def close(self) -> None: ...
