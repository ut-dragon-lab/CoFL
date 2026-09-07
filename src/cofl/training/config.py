"""Typed model and optimizer arguments for LightningCLI and the Python API."""

from dataclasses import dataclass
import math
from typing import Literal

METHOD_PROFILES = {"cofl": "image_field_v1", "cofl-s": "ground_sector_v1"}


@dataclass(frozen=True)
class ModelConfig:
    method: Literal["cofl", "cofl-s"] = "cofl"
    d_model: int = 768
    num_heads: int = 8
    fusion_layers: int = 4
    decoder_layers: int = 2
    siglip_model_name: str = "google/siglip2-base-patch16-224"
    local_files_only: bool = False
    use_depth: bool = True
    depth_interpolation: Literal["nearest", "bicubic"] = "nearest"
    depth_min_m: float = 0.0
    r_max_m: float = 5.0
    hfov_rad: float = math.pi / 2
    normalization_scale_m: float = 5.0
    vision_unfreeze_last_n_layers: int = 0
    text_unfreeze_last_n_layers: int = 0

    def __post_init__(self):
        if self.method not in METHOD_PROFILES:
            raise ValueError("method must be cofl or cofl-s")
        for name in ("d_model", "num_heads", "fusion_layers", "decoder_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        for name in ("vision_unfreeze_last_n_layers", "text_unfreeze_last_n_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < -1:
                raise ValueError(f"{name} must be -1 or a nonnegative integer")
        if self.depth_interpolation not in {"nearest", "bicubic"}:
            raise ValueError("depth_interpolation must be nearest or bicubic")
        if not math.isfinite(self.depth_min_m) or self.depth_min_m < 0:
            raise ValueError("depth_min_m must be finite and nonnegative")
        for name in ("r_max_m", "normalization_scale_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.hfov_rad) or not 0 < self.hfov_rad < math.pi:
            raise ValueError("hfov_rad must be between zero and pi")
        if self.depth_min_m >= self.r_max_m:
            raise ValueError("depth_min_m must be smaller than r_max_m")


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    schedule: Literal["constant", "cosine"] = "cosine"
    warmup_steps: int = 40000
    vision_learning_rate: float | None = None
    text_learning_rate: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        for name in ("vision_learning_rate", "text_learning_rate"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be null or finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.schedule not in {"constant", "cosine"}:
            raise ValueError("schedule must be constant or cosine")
        if (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or self.warmup_steps < 0
        ):
            raise ValueError("warmup_steps must be a nonnegative integer")
