"""Explicit local resources, resolved relative to a portable YAML configuration."""

from pathlib import Path

from pydantic import Field, model_validator

from .contracts import Contract, FloorSpec, PolicyDevice, ResourceId


class FileResource(Contract):
    path: Path
    label: str = ""
    domain_id: ResourceId | None = None


class DatasetResource(FileResource):
    split: str = "val"


class SceneResource(FileResource):
    up_axis: str = Field(default="y", pattern="^[yz]$")
    floors: list[FloorSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_floors(self):
        if len({f.id for f in self.floors}) != len(self.floors):
            raise ValueError("Floor IDs must be unique within a scene")
        for floor in self.floors:
            if not floor.min_height <= floor.elevation < floor.max_height:
                raise ValueError("Each floor needs min_height <= elevation < max_height")
        return self


class StudioConfig(Contract):
    device: PolicyDevice = "cuda:0"
    models: dict[ResourceId, FileResource] = Field(default_factory=dict)
    datasets: dict[ResourceId, DatasetResource] = Field(default_factory=dict)
    scenes: dict[ResourceId, SceneResource] = Field(default_factory=dict)
    benchmarks: dict[ResourceId, FileResource] = Field(default_factory=dict)

    def resolve(self, base_dir: Path):
        config = self.model_copy(deep=True)
        for resources in (config.models, config.datasets, config.scenes, config.benchmarks):
            for resource in resources.values():
                resource.path = (base_dir / resource.path.expanduser()).resolve()
        return config


def load_config(path: Path | None = None) -> StudioConfig:
    if path is None:
        return StudioConfig()
    import yaml

    return StudioConfig.model_validate(yaml.safe_load(path.read_text()) or {}).resolve(
        path.resolve().parent
    )
