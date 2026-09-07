"""Configurable direction and magnitude objectives for vector fields."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FieldLossConfig:
    """Method-independent formulas; ``dataclasses.asdict`` produces JSON values.

    Direction averages over supervised targets whose magnitude exceeds the
    threshold. Magnitude averages over every
    supervised vector. The returned component losses are unweighted.
    """

    magnitude_loss: str = "mse"
    magnitude_normalization: str = "absolute"
    direction_weight: float = 1.0
    magnitude_weight: float = 0.5
    relative_epsilon: float = 0.1
    direction_epsilon: float = 1e-8
    direction_min_target_magnitude: float = 1e-8

    def __post_init__(self):
        for name, choices in (
            ("magnitude_loss", {"mse", "l1"}),
            ("magnitude_normalization", {"absolute", "relative"}),
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if value not in choices:
                raise ValueError(f"{name} must be one of {sorted(choices)}")
        for name in (
            "direction_weight",
            "magnitude_weight",
            "relative_epsilon",
            "direction_epsilon",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name.endswith("weight") and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            if name.endswith("epsilon") and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.direction_weight == self.magnitude_weight == 0:
            raise ValueError("At least one field loss weight must be positive")
        threshold = self.direction_min_target_magnitude
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("direction_min_target_magnitude must be a number")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("direction_min_target_magnitude must be finite and nonnegative")


def field_loss(
    prediction,
    target,
    *,
    config: FieldLossConfig | None = None,
    mask=None,
    mask_normalization: Literal["selected", "all"] = "selected",
    allow_empty: bool = False,
):
    """Return ``loss``, ``direction`` and ``magnitude`` for matching ``[..., D]`` tensors.

    ``D`` must be at least two. Masked vectors are removed from both component
    numerators and receive zero gradients. ``selected`` averages over selected
    queries (and selected eligible directions), ignoring masked nonfinite values.
    ``all`` retains the original query and eligible-direction denominators;
    every original target must then be finite, including masked targets. Masked
    predictions may remain nonfinite with either normalization.
    ``allow_empty=True`` permits an all-false mask and returns differentiable
    zero losses; an empty input tensor is still rejected.
    Relative magnitude divides the scalar norm error by the target norm plus
    ``relative_epsilon`` before applying L1 or MSE. Direction uses cosine
    similarity with norms clamped by ``direction_epsilon``.
    Input validity is checked together before the fixed-shape tensor reductions.
    """
    if config is None:
        config = FieldLossConfig()
    elif not isinstance(config, FieldLossConfig):
        raise TypeError("config must be a FieldLossConfig or None")
    if mask_normalization not in ("selected", "all"):
        raise ValueError("mask_normalization must be selected or all")
    if not isinstance(allow_empty, bool):
        raise TypeError("allow_empty must be a bool")
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("prediction and target must be torch tensors")
    if prediction.ndim < 1 or prediction.shape != target.shape or prediction.shape[-1] < 2:
        raise ValueError("prediction and target must have the same [..., D] shape with D >= 2")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating point tensors")
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    if prediction.numel() == 0:
        raise ValueError("A training batch must have at least one supervised vector")
    if mask is not None and not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch tensor or None")
    valid = torch.ones_like(target[..., 0], dtype=torch.bool) if mask is None else mask.bool()
    if valid.shape != target.shape[:-1]:
        raise ValueError("mask must match the vector leading dimensions")
    if valid.device != prediction.device:
        raise ValueError("mask must be on the same device as the vectors")
    original_target = target
    if mask is not None:
        prediction = torch.where(valid[..., None], prediction, 0)
        target = torch.where(valid[..., None], target, 0)
    supervised = valid.any()
    finite = torch.isfinite(prediction).all() & torch.isfinite(target).all()
    original_target_finite = None
    if mask_normalization == "all" and mask is not None:
        original_target_finite = torch.isfinite(original_target).all()
        finite = finite & original_target_finite
    if not bool((supervised | allow_empty) & finite):
        if not allow_empty and not bool(supervised):
            raise ValueError("A training batch must have at least one supervised vector")
        if original_target_finite is not None and not bool(original_target_finite):
            raise ValueError("All original targets must be finite with mask_normalization='all'")
        raise ValueError("Supervised vectors must be finite")
    direction_error, magnitude_error, directional = _field_errors(
        prediction, target, config, valid
    )
    if mask_normalization == "all" and mask is not None:
        direction_count = (
            original_target.norm(dim=-1) > config.direction_min_target_magnitude
        ).sum()
        query_count = valid.numel()
    else:
        direction_count, query_count = directional.sum(), valid.sum().clamp_min(1)
    direction = direction_error.sum() / direction_count.clamp_min(1)
    magnitude = magnitude_error.sum() / query_count
    total = config.direction_weight * direction + config.magnitude_weight * magnitude
    return {"loss": total, "direction": direction, "magnitude": magnitude}


def _field_errors(prediction, target, config, valid):
    """Shared per-query errors for validated, finite vectors and diagnostics.

    Callers remove invalid nonfinite vectors before entering this helper.
    Keeping the formulas here makes regional diagnostics follow the objective,
    including magnitude normalization and the direction threshold.
    """
    predicted_magnitude = torch.linalg.vector_norm(prediction, dim=-1)
    target_magnitude = torch.linalg.vector_norm(target, dim=-1)
    threshold = config.direction_min_target_magnitude
    directional = valid & (target_magnitude > threshold)
    # Unit-valued placeholders keep excluded vectors away from the cosine
    # singularity, including when epsilon is below a low-precision dtype's range.
    similarity = F.cosine_similarity(
        torch.where(directional[..., None], prediction, 1),
        torch.where(directional[..., None], target, 1),
        dim=-1,
        eps=config.direction_epsilon,
    ).clamp(-1, 1)
    direction_error = torch.where(directional, 1 - similarity, 0)
    magnitude_delta = predicted_magnitude - target_magnitude
    if config.magnitude_normalization == "relative":
        magnitude_delta = magnitude_delta / (target_magnitude + config.relative_epsilon)
    magnitude_error = (
        magnitude_delta.square() if config.magnitude_loss == "mse" else magnitude_delta.abs()
    )
    return direction_error, torch.where(valid, magnitude_error, 0), directional
