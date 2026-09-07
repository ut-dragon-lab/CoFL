"""Small service contracts for local, remote, or test implementations."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class DomainConstraint:
    """Transport-independent resource compatibility, checked before publication."""

    id: str
    methods: tuple[str, ...]
    profiles: tuple[str, ...]

    def check_dataset(self, profile: str) -> None:
        if profile not in self.profiles:
            raise ValueError(f"Profile {profile!r} is incompatible with domain {self.id!r}")

    def check_model(self, method: str, profile: str) -> None:
        self.check_dataset(profile)
        if method not in self.methods:
            raise ValueError(f"Method {method!r} is incompatible with domain {self.id!r}")


class InferenceBackend(Protocol):
    """Static field prediction; simulator modes define their own execution API."""

    def predict(
        self,
        request: Mapping,
        sample: dict | None = None,
        *,
        domain: DomainConstraint | None = None,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


class DatasetBackend(Protocol):
    def sample(self, dataset_id: str, index: int, split: str = "val") -> dict: ...

    def view(self, dataset_id: str, index: int, split: str = "val") -> dict: ...

    def close(self) -> None: ...


class RegisterableInferenceBackend(InferenceBackend, Protocol):
    """Optional local checkpoint registration capability."""

    def register(
        self, resource_id: str, path: Path, *, domain: DomainConstraint | None = None
    ) -> dict[str, Any]: ...


class RegisterableDatasetBackend(DatasetBackend, Protocol):
    """Optional native dataset registration capability."""

    def register(
        self,
        resource_id: str,
        path: Path,
        *,
        split: str = "val",
        domain: DomainConstraint | None = None,
    ) -> dict: ...
