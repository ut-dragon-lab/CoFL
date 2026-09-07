"""Validated configuration for portable, method-specific generation recipes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


def load_mapping(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as error:
            raise ImportError(
                "YAML recipes require pip install 'cofl-navigation[generation]'"
            ) from error
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Generation configuration must be a mapping")
    return value


@dataclass(frozen=True)
class GenerationConfig:
    method: str
    dataset_id: str
    revision: str
    split: str
    source: dict[str, Any]
    pipeline: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    limit_units: int | None = None
    num_shards: int = 1
    shard_index: int = 0
    min_free_gb: float = 5.0

    def __post_init__(self):
        for name in ("method", "dataset_id", "revision", "split"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.source, dict) or any(
            not isinstance(self.source.get(key), str) or not self.source[key].strip()
            for key in ("id", "version")
        ):
            raise ValueError("source must specify nonempty id and version")
        if not isinstance(self.pipeline, dict):
            raise ValueError("pipeline must be a mapping of method options")
        if "partial" in self.source and not isinstance(self.source["partial"], bool):
            raise ValueError("source.partial must be a boolean")
        for name in ("seed", "num_shards", "shard_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.num_shards < 1 or self.shard_index >= self.num_shards:
            raise ValueError("shard_index must be in [0, num_shards)")
        if self.limit_units is not None and (
            isinstance(self.limit_units, bool)
            or not isinstance(self.limit_units, int)
            or self.limit_units < 1
        ):
            raise ValueError("limit_units must be a positive integer or null")
        if isinstance(self.min_free_gb, bool) or not isinstance(self.min_free_gb, (int, float)):
            raise ValueError("min_free_gb must be nonnegative")
        if not 0 <= self.min_free_gb < float("inf"):
            raise ValueError("min_free_gb must be finite and nonnegative")
        if "split" in self.pipeline and self.pipeline["split"] != self.split:
            raise ValueError("pipeline.split must match the top-level split")
        json.dumps(asdict(self), allow_nan=False)

    @classmethod
    def from_mapping(cls, value: dict) -> GenerationConfig:
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown generation options: {', '.join(sorted(unknown))}")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f"Invalid generation configuration: {error}") from error

    def to_dict(self) -> dict:
        return asdict(self)
