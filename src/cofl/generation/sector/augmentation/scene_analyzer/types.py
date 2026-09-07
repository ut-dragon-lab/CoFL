"""Pose and simulator handles shared by per-frame field generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SceneContext:
    position: np.ndarray
    heading: float
    pathfinder: Any
    semantic_scene: Any
