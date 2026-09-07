"""Profile-aware offline metrics and reproducible evaluation records."""

from .config import EvaluationConfig, ImageTrajectoryConfig
from .image import evaluate_image_navigation
from .metrics import (
    evaluate_field,
    evaluate_ground_trajectory,
    evaluate_image_trajectory,
    evaluate_trajectory,
    resample_trajectory,
)
from .protocol import EvaluationProtocol, ResultWriter, TimingSpec, file_sha256

__all__ = [
    "EvaluationConfig",
    "EvaluationProtocol",
    "ImageTrajectoryConfig",
    "ResultWriter",
    "TimingSpec",
    "evaluate_field",
    "evaluate_ground_trajectory",
    "evaluate_image_navigation",
    "evaluate_image_trajectory",
    "evaluate_trajectory",
    "file_sha256",
    "resample_trajectory",
]
