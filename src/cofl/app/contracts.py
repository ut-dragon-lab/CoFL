"""Versioned HTTP contracts; the TypeScript client is generated from OpenAPI."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Finite = Annotated[float, Field(allow_inf_nan=False)]
Point2 = tuple[Finite, Finite]
ResourceId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
ResourceKind = Literal["model", "dataset", "benchmark"]
PolicyDevice = Annotated[str, Field(pattern=r"^(cpu|cuda:[0-9]+)$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DeviceFailure(Contract):
    id: str
    device: str
    message: str


class RuntimeDevice(Contract):
    device: PolicyDevice
    gpu_available: bool
    gpu_name: str | None = None
    failure: DeviceFailure | None = None
    can_change: bool


class DeviceSelection(Contract):
    device: PolicyDevice


class StudioClosed(Contract):
    status: Literal["closed"] = "closed"
    server_stopping: bool


class FloorSpec(Contract):
    id: str
    label: str
    elevation: Finite
    min_height: Finite
    max_height: Finite


class DomainDescriptor(Contract):
    id: ResourceId
    title: str
    description: str
    methods: list[str] = Field(min_length=1)
    profiles: list[str] = Field(min_length=1)
    resource_kinds: list[ResourceKind] = Field(default_factory=lambda: ["model", "dataset"])


class ModeDescriptor(Contract):
    id: ResourceId
    domain_id: ResourceId
    task: str
    execution: Literal["static", "session"]
    title: str
    description: str
    renderer: str
    capabilities: list[str] = Field(default_factory=list)
    default_parameters: dict[str, Any] = Field(default_factory=dict)


class ResourceDescriptor(Contract):
    id: str
    label: str


class ModelDescriptor(ResourceDescriptor):
    method: str | None = None
    profile: str | None = None
    domain_id: str | None = None


class DatasetDescriptor(ResourceDescriptor):
    split: str = "val"
    profile: str | None = None
    domain_id: str | None = None


class BenchmarkDescriptor(ResourceDescriptor):
    domain_id: str | None = None


class SceneDescriptor(ResourceDescriptor):
    url: str
    up_axis: Literal["y", "z"] = "y"
    floors: list[FloorSpec] = Field(default_factory=list)


class Catalog(Contract):
    api_version: Literal["1"] = "1"
    domains: list[DomainDescriptor]
    modes: list[ModeDescriptor]
    models: list[ModelDescriptor]
    datasets: list[DatasetDescriptor]
    scenes: list[SceneDescriptor]
    benchmarks: list[BenchmarkDescriptor]
    device: str


class BrowseEntry(Contract):
    name: str
    path: str
    is_dir: bool
    selectable: bool


class BrowseShortcut(Contract):
    id: Literal["cwd", "home", "filesystem"]
    label: str
    path: str


class ResourceBrowse(Contract):
    kind: ResourceKind
    cwd: str
    parent: str | None
    selectable: bool
    selected_path: str | None = None
    entries: list[BrowseEntry]
    shortcuts: list[BrowseShortcut]


class ResourceRegistration(Contract):
    kind: ResourceKind
    domain_id: ResourceId | None = None
    path: str = Field(min_length=1, max_length=8192)
    label: str | None = Field(default=None, max_length=160)
    split: str = Field(default="val", min_length=1, max_length=80)


class ResourceActivation(Contract):
    domain_id: ResourceId
    split: str | None = Field(default=None, min_length=1, max_length=80)


class RegisteredResource(Contract):
    kind: ResourceKind
    resource_id: ResourceId
    label: str
    path: str
    split: str | None = None
    domain_id: str | None = None
    method: str | None = None
    profile: str | None = None


class PredictionRequest(Contract):
    mode_id: ResourceId
    domain_id: ResourceId | None = None
    model_id: ResourceId
    observation_id: str = Field(min_length=1, max_length=160)
    image: str | None = Field(default=None, max_length=12_000_000)
    instruction: str = Field(min_length=1, max_length=2048)
    start: Point2 | None = Field(
        default=None,
        description="Optional selected start; null uses the profile's standard inference start",
    )
    grid_size: int = Field(default=24, ge=8, le=64)
    max_steps: int = Field(default=100, ge=1, le=500)
    policy_dt: Finite = Field(default=0.01, gt=0, le=1)
    dataset_id: ResourceId | None = None
    sample_index: int | None = Field(default=None, ge=0)
    split: str = Field(default="val", min_length=1, max_length=80)


class PredictionResult(Contract):
    observation_id: str
    model_id: str
    method: str
    profile: str
    coordinate_frame: str
    coordinate_unit: str
    vector_unit: str
    queries: list[Point2]
    vectors: list[Point2]
    trajectory: list[Point2]
    stop_reason: str
    elapsed_ms: Finite
    cache_hit: bool
    inference_rules: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    actions: list[Finite] | None = None
    action_semantics: str | None = None


class DatasetView(Contract):
    dataset_id: str
    split: str
    observation_id: str
    index: int
    count: int
    sample_id: str
    profile: str
    instruction: str
    image: str
    geometry: dict[str, Any]
    start: Point2
    coordinate_frame: str
    coordinate_unit: str
    vector_unit: str
    queries: list[Point2]
    vectors: list[Point2]
    trajectory: list[Point2] | None = None
    target_action: int | None = None
    supervised_count: int
    preview_count: int


class ErrorResponse(Contract):
    detail: str
