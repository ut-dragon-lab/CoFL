"""Typed recipes for shared offline evaluation and image navigation scoring."""

import math
from dataclasses import dataclass, field
from typing import Literal

from cofl.runtime.inference_rules import (
    DEFAULT_FIELD_GRID_SIZE,
    DEFAULT_FIELD_QUERY_CHUNK_SIZE,
    DEFAULT_NUM_STEPS,
    DEFAULT_POLICY_DT,
    DEFAULT_TIME_SCALE_POWER,
    IMAGE_TIME_SCALE_BIAS,
)


@dataclass(frozen=True)
class ImageTrajectoryConfig:
    grid_size: int = DEFAULT_FIELD_GRID_SIZE
    num_steps: int = DEFAULT_NUM_STEPS
    policy_dt: float = DEFAULT_POLICY_DT
    time_scale_bias: float = IMAGE_TIME_SCALE_BIAS
    time_scale_power: float = DEFAULT_TIME_SCALE_POWER
    resample_points: int = 101

    def __post_init__(self):
        for key in ("grid_size", "num_steps", "resample_points"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (1 if key == "num_steps" else 2)
            ):
                raise ValueError(f"{key} has an invalid integer value")
        for key in ("policy_dt", "time_scale_bias", "time_scale_power"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")


@dataclass(frozen=True)
class EvaluationConfig:
    dataset: str
    checkpoint: str
    output: str
    split: str = "val"
    tasks: tuple[Literal["field", "image_navigation", "action"], ...] = ("field",)
    device: str = "cpu"
    batch_size: int = 8
    num_workers: int = 4
    multiprocessing_context: Literal["forkserver", "spawn"] = "forkserver"
    query_chunk_size: int = DEFAULT_FIELD_QUERY_CHUNK_SIZE
    precision: Literal["float32", "bfloat16", "float16"] = "float32"
    score_workers: int = 1
    profile_batches: int = 0
    seed: int = 42
    max_samples: int | None = None
    visualization_count: int = 24
    trajectory: ImageTrajectoryConfig = field(default_factory=ImageTrajectoryConfig)

    def __post_init__(self):
        for key in ("dataset", "checkpoint", "output", "split", "device"):
            if not str(getattr(self, key)).strip():
                raise ValueError(f"{key} must not be empty")
        if (
            not self.tasks
            or len(set(self.tasks)) != len(self.tasks)
            or not set(self.tasks) <= {"field", "image_navigation", "action"}
        ):
            raise ValueError("tasks must contain unique field, image_navigation, or action entries")
        for key in (
            "batch_size",
            "query_chunk_size",
            "num_workers",
            "seed",
            "visualization_count",
            "score_workers",
            "profile_batches",
        ):
            value = getattr(self, key)
            minimum = 1 if key in {"batch_size", "query_chunk_size"} else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if self.max_samples is not None and (
            isinstance(self.max_samples, bool)
            or not isinstance(self.max_samples, int)
            or self.max_samples < 1
        ):
            raise ValueError("max_samples must be a positive integer or null")
        if self.multiprocessing_context not in {"forkserver", "spawn"}:
            raise ValueError("multiprocessing_context must be forkserver or spawn")
        if self.precision not in {"float32", "bfloat16", "float16"}:
            raise ValueError("precision must be float32, bfloat16, or float16")
        if self.precision == "float16" and not str(self.device).startswith("cuda"):
            raise ValueError("float16 evaluation requires a CUDA device")
        if not isinstance(self.trajectory, ImageTrajectoryConfig):
            raise TypeError("trajectory must be an ImageTrajectoryConfig")
