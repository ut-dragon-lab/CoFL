"""Small contracts between source-specific generators and the shared runner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from cofl.data.storage import DatasetWriter


@dataclass(frozen=True)
class GenerationUnit:
    """An independently reproducible view or episode, with its source files.

    Keys must be stable source identities, never worker or output row numbers.
    Inputs are hashed before generation and checked again before publication.
    Payload remains private to its adapter; the runner assumes no geometry.
    """

    key: str
    payload: Any = None
    inputs: tuple[Path, ...] = ()


class GenerationPipeline(Protocol):
    profile: str
    recipe_version: str
    writer_options: Mapping[str, Any]

    def units(self) -> Iterable[GenerationUnit]: ...

    def generate(self, unit: GenerationUnit, writer: DatasetWriter, *, seed: int) -> dict: ...

    def close(self) -> None: ...
