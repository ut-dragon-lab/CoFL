"""Typed browser adapter for the shared CoFL-S closed-loop benchmark runner."""

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from cofl.online.dynamics import RunParameters


class OnlineContract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class OnlineBenchmarkResource(OnlineContract):
    """A portable YAML recipe, separate from native validation datasets."""

    path: Path
    label: str = ""

    def resolve(self, base_dir: Path):
        return self.model_copy(update={"path": (base_dir / self.path.expanduser()).resolve()})


class OnlineBenchmark(OnlineContract):
    id: str
    label: str
    available: bool
    reason: str | None = None
    benchmark_reason: str | None = None
    episode_count: int = 0
    split: str = "val_unseen"


class OnlineEpisode(OnlineContract):
    index: int
    episode_id: str
    scene_id: str
    instruction: str


class OnlineEpisodePage(OnlineContract):
    items: list[OnlineEpisode]
    total: int
    offset: int
    limit: int


class OnlineScene(OnlineContract):
    id: str
    label: str
    scene_id: str
    spawn_position: tuple[float, float, float] | None = None
    spawn_yaw_deg: float = 0


class OnlineScenePage(OnlineContract):
    items: list[OnlineScene]
    total: int


def _default(name):
    return RunParameters.__dataclass_fields__[name].default


class OnlineSessionRequest(OnlineContract):
    benchmark_id: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=80)
    mode: Literal["benchmark", "interactive"] = "benchmark"
    scene_id: str | None = None
    start_position: tuple[float, float, float] | None = None
    start_yaw_deg: float | None = None
    instruction: str = Field(default="", max_length=2048)
    instruction_mode: Literal["oracle", "full_instruction"] | None = None
    episode_index: int = Field(default=0, ge=0)
    max_time_s: float = Field(default=_default("max_time_s"), gt=0, le=600)
    max_steps: int = Field(default=_default("max_steps"), ge=1, le=12000)
    plant_hz: float = Field(default=_default("plant_hz"), ge=1, le=200)
    controller_hz: float = Field(default=_default("controller_hz"), ge=1, le=200)
    sensor_hz: float = Field(default=_default("sensor_hz"), ge=1, le=200)
    planner_hz: float = Field(default=_default("planner_hz"), ge=0.1, le=100)
    policy_steps: int = Field(default=_default("policy_steps"), ge=1, le=500)
    policy_dt: float | None = Field(default=None, gt=0, le=1)
    field_grid_size: int = Field(default=_default("field_grid_size"), ge=2)
    field_query_chunk_size: int = Field(default=_default("field_query_chunk_size"), ge=1)
    stop_threshold: float = Field(default=_default("stop_threshold"), ge=0, le=1)
    stop_endpoint_m: float = Field(default=_default("stop_endpoint_m"), ge=0, le=2)
    stop_pool_step: float = Field(default=_default("stop_pool_step"), ge=0, le=1)
    oscillation_window: int = Field(default=_default("oscillation_window"), ge=0, le=100)
    oscillation_fwd_stop_max: float = Field(
        default=_default("oscillation_fwd_stop_max"), ge=0, le=1
    )
    turn_in_place_lookahead_m: float = Field(
        default=_default("turn_in_place_lookahead_m"), gt=0, le=5
    )
    lookahead_m: float = Field(default=_default("lookahead_m"), gt=0, le=5)
    max_linear_mps: float = Field(default=_default("max_linear_mps"), gt=0, le=2)
    max_angular_radps: float = Field(default=_default("max_angular_radps"), gt=0, le=6.3)
    rotate_threshold_deg: float = Field(default=_default("rotate_threshold_deg"), gt=0, le=90)


class OnlineCommand(OnlineContract):
    instruction: str = Field(min_length=1, max_length=2048)


class OnlineReset(OnlineContract):
    start_position: tuple[float, float, float] | None = None
    start_yaw_deg: float | None = None
    instruction: str | None = Field(default=None, max_length=2048)


class OnlineObservation(OnlineContract):
    image: str
    t_sim_s: float
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


class OnlineStep(OnlineContract):
    step_index: int
    t_sim_s: float
    image: str
    instruction: str
    sector_trajectory: list[tuple[float, float]]
    world_trajectory: list[tuple[float, float]]
    world_path: list[tuple[float, float]]
    agent_position: tuple[float, float, float]
    agent_rotation: tuple[float, float, float, float] | None = None
    instruction_info: dict[str, Any] = Field(default_factory=dict)
    epoch: int = 0
    actions: list[float] | None = None
    linear_mps: float
    angular_radps: float
    inference_ms: float
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class OnlineSession(OnlineContract):
    id: str
    domain_id: Literal["ego2d"] = "ego2d"
    mode_id: Literal["ego2d-playground", "ego2d-interactive"] = "ego2d-playground"
    mode: Literal["benchmark", "interactive"] = "benchmark"
    benchmark_id: str
    model_id: str
    episode_index: int
    status: Literal[
        "starting", "running", "waiting", "paused", "stopping", "completed", "cancelled", "failed"
    ]
    scene_id: str = ""
    current_position: tuple[float, float, float] | None = None
    current_yaw_deg: float | None = None
    observation: OnlineObservation | None = None
    command_state: dict[str, Any] = Field(default_factory=dict)
    active_instruction: dict[str, Any] | None = None
    command_log: list[dict[str, Any]] = Field(default_factory=list)
    epoch: int = 0
    reason: str | None = None
    steps: list[OnlineStep] = Field(default_factory=list)
    total_steps: int = 0
    first_available_step: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    reference_path: list[tuple[float, float]] = Field(default_factory=list)
    goal_position: tuple[float, float, float] | None = None
    geometry: dict[str, Any] = Field(default_factory=dict)


def OnlineService(models, benchmarks, device="cuda:0", **kwargs):
    """Load the orchestration layer without importing Torch or Habitat."""
    from cofl.online.service import SessionService

    return SessionService(models, benchmarks, device=device, **kwargs)


def create_router(service, *, session_creator=None, prefix="/api/v1/domains/ego2d") -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["online"])

    async def call(method, *args, **kwargs):
        try:
            from inspect import iscoroutinefunction

            if iscoroutinefunction(method):
                return await method(*args, **kwargs)
            return await run_in_threadpool(method, *args, **kwargs)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(503, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @router.get("/benchmarks", response_model=list[OnlineBenchmark])
    async def benchmarks():
        return await call(service.describe)

    @router.get("/benchmarks/{benchmark_id}/episodes", response_model=OnlineEpisodePage)
    async def episodes(
        benchmark_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)
    ):
        return await call(service.episodes, benchmark_id, offset=offset, limit=limit)

    @router.get("/benchmarks/{benchmark_id}/scenes", response_model=OnlineScenePage)
    async def scenes(benchmark_id: str):
        return await call(service.scenes, benchmark_id)

    @router.post("/sessions", response_model=OnlineSession, status_code=201)
    async def create_session(payload: OnlineSessionRequest, request: Request):
        from .resources import require_same_origin

        require_same_origin(request)
        return await call(session_creator or service.create, payload.model_dump())

    @router.get("/sessions/active", response_model=OnlineSession | None)
    async def active_session():
        return await call(service.active)

    @router.get("/sessions/{session_id}", response_model=OnlineSession)
    async def session(session_id: str, after_step: int = Query(-1, ge=-1)):
        return await call(service.get, session_id, after_step=after_step)

    @router.post("/sessions/{session_id}/stop", response_model=OnlineSession)
    async def stop_session(session_id: str, request: Request):
        from .resources import require_same_origin

        require_same_origin(request)
        return await call(service.stop, session_id)

    @router.post("/sessions/{session_id}/commands", response_model=OnlineSession)
    async def command_session(session_id: str, payload: OnlineCommand, request: Request):
        from .resources import require_same_origin

        require_same_origin(request)
        return await call(service.command, session_id, payload.instruction)

    @router.post("/sessions/{session_id}/pause", response_model=OnlineSession)
    async def pause_session(session_id: str, request: Request):
        from .resources import require_same_origin

        require_same_origin(request)
        return await call(service.pause, session_id)

    @router.post("/sessions/{session_id}/resume", response_model=OnlineSession)
    async def resume_session(session_id: str, request: Request):
        from .resources import require_same_origin

        require_same_origin(request)
        return await call(service.resume, session_id)

    @router.post("/sessions/{session_id}/reset", response_model=OnlineSession)
    async def reset_session(session_id: str, payload: OnlineReset, request: Request):
        from .resources import require_same_origin

        require_same_origin(request)
        return await call(service.reset, session_id, **payload.model_dump())

    return router
