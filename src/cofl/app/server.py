"""Compose domain-specific tasks, resource adapters, and execution protocols."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from cofl.runtime.device import DeviceManager

from .adapters import DatasetBackend, DomainConstraint, InferenceBackend
from .config import DatasetResource, FileResource, StudioConfig
from .contracts import (
    Catalog,
    DatasetView,
    DeviceSelection,
    DomainDescriptor,
    ErrorResponse,
    PredictionRequest,
    PredictionResult,
    RegisteredResource,
    ResourceActivation,
    ResourceBrowse,
    ResourceId,
    ResourceKind,
    ResourceRegistration,
    RuntimeDevice,
    StudioClosed,
)
from .lifecycle import ShutdownController, ShutdownError, ShutdownMiddleware
from .modes import StudioMode, builtin_domains, builtin_modes
from .resources import browse_resources, require_same_origin, resolved_path, resource_id


def create_app(
    config: StudioConfig | None = None,
    *,
    domains: list[DomainDescriptor] | None = None,
    modes: list[StudioMode] | None = None,
    inference: InferenceBackend | None = None,
    datasets: DatasetBackend | None = None,
    online=None,
    device_manager: DeviceManager | None = None,
    static_dir: Path | None = None,
    shutdown_callback=None,
):
    """Add domains and dedicated task protocols without changing 2D prediction."""
    config = (config or StudioConfig()).resolve(Path.cwd())
    device_manager = device_manager or DeviceManager(config.device)
    config.device = device_manager.device
    domain_list = builtin_domains() + list(domains or [])
    domain_map = {domain.id: domain for domain in domain_list}
    if len(domain_map) != len(domain_list):
        raise ValueError("Studio domain IDs must be unique")
    registered = builtin_modes() + list(modes or [])
    mode_map = {mode.descriptor.id: mode for mode in registered}
    if len(mode_map) != len(registered):
        raise ValueError("Studio mode IDs must be unique")
    for mode in registered:
        if mode.descriptor.domain_id not in domain_map:
            raise ValueError(f"Mode {mode.descriptor.id!r} references an unknown domain")
        if mode.static_input is not None and mode.descriptor.execution != "static":
            raise ValueError("Session modes cannot use the static 2D prediction adapter")
    for resources in (config.models, config.datasets, config.benchmarks):
        for resource in resources.values():
            if resource.domain_id is not None and resource.domain_id not in domain_map:
                raise ValueError(f"Resource references unknown domain {resource.domain_id!r}")
    if inference is None:
        from .inference import InferenceService

        inference = InferenceService(
            {k: v.path for k, v in config.models.items()}, device_manager=device_manager
        )
    elif hasattr(inference, "device_manager"):
        if inference.device_manager is not device_manager:
            inference.close()
        inference.device_manager = device_manager
    if datasets is None:
        from .datasets import DatasetService

        datasets = DatasetService({k: v.path for k, v in config.datasets.items()})
    from .online import OnlineService
    from .online import create_router as create_online_router

    if online is None:
        online = OnlineService(
            {k: v.path for k, v in config.models.items()},
            config.benchmarks,
            device=config.device,
            scenes=config.scenes,
            device_manager=device_manager,
        )
    elif hasattr(online, "device_manager"):
        if online.device_manager is not device_manager and online.is_running:
            raise ValueError("Stop the active session before rebinding its device manager")
        online.device_manager = device_manager

    shutdown = ShutdownController(
        [("online", online.close), ("inference", inference.close), ("datasets", datasets.close)],
        callback=shutdown_callback,
    )

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await shutdown.shutdown()

    app = FastAPI(title="CoFL Studio", version="1.0.0", lifespan=lifespan)
    app.state.config, app.state.inference, app.state.datasets = config, inference, datasets
    app.state.online = online
    app.state.device_manager = device_manager
    app.state.shutdown = shutdown
    app.add_middleware(ShutdownMiddleware, controller=shutdown)
    app.state.domains = domain_map
    app.state.modes = {key: mode.descriptor for key, mode in mode_map.items()}
    execution_gate = asyncio.Lock()
    metadata: dict[str, dict[str, dict]] = {"model": {}, "dataset": {}}
    prefix = "/api/v1"
    errors = {code: {"model": ErrorResponse} for code in (400, 403, 404, 409, 501, 503)}

    def require_open():
        if shutdown.closing:
            raise HTTPException(503, "Studio is closing or already closed")

    @app.post(prefix + "/runtime/shutdown", response_model=StudioClosed, responses=errors)
    async def close_studio(request: Request):
        require_same_origin(request)
        try:
            await shutdown.shutdown(request_server_stop=True)
        except ShutdownError as error:
            raise HTTPException(503, str(error)) from error
        return StudioClosed(server_stopping=shutdown.server_stopping)

    async def execute_policy(method, *args, **kwargs):
        def execute():
            with device_manager.execution(check=False, cleanup=inference.close):
                return method(*args, **kwargs)

        return await run_in_threadpool(execute)

    @app.get(prefix + "/runtime/device", response_model=RuntimeDevice, responses=errors)
    async def runtime_device():
        result = await run_in_threadpool(device_manager.status)
        result["can_change"] = not online.is_running and not execution_gate.locked()
        return result

    @app.post(prefix + "/runtime/device", response_model=RuntimeDevice, responses=errors)
    async def select_device(body: DeviceSelection, request: Request):
        require_same_origin(request)
        async with execution_gate:
            require_open()
            if online.is_running:
                raise HTTPException(
                    409, "Stop the active session before changing the policy device"
                )
            try:
                await run_in_threadpool(device_manager.validate, body.device)
                # Release using the old device before changing the shared selection.
                await run_in_threadpool(inference.close)
                for backend in (inference, online):
                    if not hasattr(backend, "device_manager") and hasattr(backend, "device"):
                        backend.device = body.device
                device_manager.select(body.device)
                config.device = body.device
                return await run_in_threadpool(device_manager.status)
            except RuntimeError as error:
                raise HTTPException(503, str(error)) from error

    def require_static_gpu():
        if online.is_running:
            raise HTTPException(
                409, "Stop the active online session before loading or querying a static policy"
            )

    def domain_constraint(domain_id: str | None, kind: ResourceKind | None = None):
        if domain_id is None:
            return None
        domain = domain_map.get(domain_id)
        if domain is None:
            raise HTTPException(404, "Unknown Studio domain")
        if kind is not None and kind not in domain.resource_kinds:
            raise ValueError(f"Domain {domain_id!r} does not accept {kind} resources")
        return DomainConstraint(domain.id, tuple(domain.methods), tuple(domain.profiles))

    def bindings(kind: ResourceKind):
        return {"model": config.models, "dataset": config.datasets, "benchmark": config.benchmarks}[
            kind
        ]

    def inferred_domain(kind: str, values: dict) -> str | None:
        matches = [
            domain.id
            for domain in domain_list
            if kind in domain.resource_kinds
            and values.get("profile") in domain.profiles
            and (kind != "model" or values.get("method") in domain.methods)
        ]
        return matches[0] if len(matches) == 1 else None

    def record_metadata(kind: str, identifier: str, values: dict, domain_id: str | None):
        resources = bindings(kind)
        if identifier in resources:
            metadata[kind][identifier] = values
            resources[identifier] = resources[identifier].model_copy(
                update={
                    "domain_id": domain_id or inferred_domain(kind, values),
                }
            )

    @app.get(prefix + "/catalog", response_model=Catalog)
    async def catalog():
        return Catalog(
            domains=domain_list,
            modes=[mode.descriptor for mode in registered],
            device=config.device,
            models=[
                {
                    "id": key,
                    "label": value.label or key,
                    "domain_id": value.domain_id,
                    "method": metadata["model"].get(key, {}).get("method"),
                    "profile": metadata["model"].get(key, {}).get("profile"),
                }
                for key, value in config.models.items()
            ],
            datasets=[
                {
                    "id": key,
                    "label": value.label or key,
                    "split": value.split,
                    "domain_id": value.domain_id,
                    "profile": metadata["dataset"].get(key, {}).get("profile"),
                }
                for key, value in config.datasets.items()
            ],
            scenes=[
                {
                    "id": key,
                    "label": value.label or key,
                    "url": f"{prefix}/scenes/{key}/asset",
                    "up_axis": value.up_axis,
                    "floors": value.floors,
                }
                for key, value in config.scenes.items()
            ],
            benchmarks=[
                {
                    "id": key,
                    "label": value.label or key,
                    "domain_id": value.domain_id,
                }
                for key, value in config.benchmarks.items()
            ],
        )

    @app.get(prefix + "/resources/browse", response_model=ResourceBrowse, responses=errors)
    async def resource_browser(
        request: Request,
        kind: ResourceKind,
        path: Annotated[str | None, Query(max_length=8192)] = None,
        domain_id: ResourceId | None = None,
    ):
        require_same_origin(request)
        try:
            domain_constraint(domain_id, kind)
            return await run_in_threadpool(browse_resources, kind, path)
        except PermissionError as error:
            raise HTTPException(403, f"Directory cannot be read: {error}") from error
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, OSError, RuntimeError) as error:
            raise HTTPException(400, str(error)) from error

    async def register_resource(body: ResourceRegistration, request: Request):
        require_same_origin(request)
        backend = {"model": inference, "dataset": datasets, "benchmark": online}[body.kind]
        register = getattr(backend, "register", None)
        if not callable(register):
            raise HTTPException(
                501, f"This {body.kind} backend does not support local registration"
            )
        try:
            path = resolved_path(body.path)
            async with execution_gate:
                require_open()
                if await request.is_disconnected():
                    raise HTTPException(499, "Resource registration was disconnected")
                if body.kind == "model":
                    require_static_gpu()
                resources = bindings(body.kind)
                existing = next(
                    (key for key, value in resources.items() if value.path == path), None
                )
                identifier = existing or resource_id(body.kind, path)
                if identifier in resources and resources[identifier].path != path:
                    raise ValueError("Resource ID is already assigned to a different path")
                selected_domain = body.domain_id or (
                    resources[existing].domain_id if existing else None
                )
                if body.kind == "benchmark":
                    selected_domain = selected_domain or "ego2d"
                    if selected_domain != "ego2d":
                        raise ValueError("Online benchmarks require the ego2d domain")
                constraint = domain_constraint(selected_domain, body.kind)
                label = (body.label or "").strip() or (
                    (resources[existing].label if existing else "")
                    or path.stem
                    or path.name
                    or body.kind
                )
                if body.kind == "model":
                    values = await execute_policy(register, identifier, path, domain=constraint)
                    if (
                        not isinstance(values, dict)
                        or not values.get("method")
                        or not values.get("profile")
                    ):
                        raise ValueError(
                            "Model registration must return verified method and profile"
                        )
                    if constraint is not None:
                        constraint.check_model(values["method"], values["profile"])
                    if values["method"] == "cofl-s":
                        await run_in_threadpool(online.register_model, identifier, path)
                    config.models[identifier] = FileResource(path=path, label=label)
                    record_metadata("model", identifier, values, selected_domain)
                elif body.kind == "dataset":
                    values = await run_in_threadpool(
                        register, identifier, path, split=body.split, domain=constraint
                    )
                    if not isinstance(values, dict) or not values.get("profile"):
                        raise ValueError("Dataset registration must return a verified profile")
                    if constraint is not None:
                        constraint.check_dataset(values["profile"])
                    config.datasets[identifier] = DatasetResource(
                        path=path, label=label, split=body.split
                    )
                    record_metadata("dataset", identifier, values, selected_domain)
                else:
                    await run_in_threadpool(register, identifier, path, label=label)
                    config.benchmarks[identifier] = FileResource(
                        path=path, label=label, domain_id="ego2d"
                    )
                    values = {}
                resource = bindings(body.kind)[identifier]
                return RegisteredResource(
                    kind=body.kind,
                    resource_id=identifier,
                    label=label,
                    path=str(path),
                    split=body.split if body.kind == "dataset" else None,
                    domain_id=resource.domain_id,
                    method=values.get("method"),
                    profile=values.get("profile"),
                )
        except PermissionError as error:
            raise HTTPException(403, f"Resource cannot be read: {error}") from error
        except (FileNotFoundError, IndexError) as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise HTTPException(400, str(error)) from error
        except NotImplementedError as error:
            raise HTTPException(501, str(error) or "Local registration is unavailable") from error
        except RuntimeError as error:
            raise HTTPException(503, f"Resource could not be loaded: {error}") from error

    app.post(prefix + "/resources", response_model=RegisteredResource, responses=errors)(
        register_resource
    )

    @app.post(
        prefix + "/resources/{kind}/{resource_id}/activate",
        response_model=RegisteredResource,
        responses=errors,
    )
    async def activate_resource(
        kind: ResourceKind, resource_id: ResourceId, body: ResourceActivation, request: Request
    ):
        require_same_origin(request)
        resource = bindings(kind).get(resource_id)
        if resource is None:
            raise HTTPException(404, "Unknown configured resource")
        return await register_resource(
            ResourceRegistration(
                kind=kind,
                path=str(resource.path),
                label=resource.label,
                domain_id=body.domain_id,
                split=body.split or getattr(resource, "split", "val"),
            ),
            request,
        )

    @app.get(prefix + "/scenes/{scene_id}/asset", responses=errors)
    def scene_asset(scene_id: str):
        scene = config.scenes.get(scene_id)
        if scene is None or not scene.path.is_file():
            raise HTTPException(404, "Scene file not found in the configured catalog")
        if scene.path.suffix.lower() != ".glb":
            raise HTTPException(400, "Studio scenes must be self-contained .glb files")
        return FileResponse(scene.path, media_type="model/gltf-binary")

    @app.get(
        prefix + "/datasets/{dataset_id}/samples/{index}",
        response_model=DatasetView,
        responses=errors,
    )
    async def dataset_sample(
        dataset_id: str,
        index: Annotated[int, PathParam(ge=0)],
        split: str = "val",
        domain_id: ResourceId | None = None,
    ):
        try:
            constraint = domain_constraint(domain_id, "dataset")
            view = await run_in_threadpool(datasets.view, dataset_id, index, split=split)
            if constraint is not None:
                constraint.check_dataset(view["profile"])
            record_metadata("dataset", dataset_id, {"profile": view["profile"]}, domain_id)
            return view
        except (KeyError, FileNotFoundError, IndexError) as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post(prefix + "/predict", response_model=PredictionResult, responses=errors)
    async def predict(request: PredictionRequest, http_request: Request):
        mode = mode_map.get(request.mode_id)
        if mode is None:
            raise HTTPException(404, "Unknown Studio mode")
        if mode.descriptor.execution != "static" or mode.static_input is None:
            raise HTTPException(400, "Use this mode's dedicated API for its execution protocol")
        selected_domain = mode.descriptor.domain_id
        if request.domain_id is not None and request.domain_id != selected_domain:
            raise HTTPException(400, "Prediction domain does not match the selected mode")
        constraint = domain_constraint(selected_domain)
        try:
            sample = None
            if mode.static_input == "dataset":
                if request.dataset_id is None or request.sample_index is None:
                    raise ValueError("Validation prediction requires a dataset and sample index")
                if request.image is not None:
                    raise ValueError("Validation uses the original dataset image")
            elif not request.image or request.dataset_id is not None:
                raise ValueError("Scene prediction requires a captured image, without a dataset")
            async with execution_gate:
                require_open()
                if await http_request.is_disconnected():
                    raise HTTPException(499, "Prediction request was disconnected")
                require_static_gpu()
                if mode.static_input == "dataset":
                    sample = await run_in_threadpool(
                        datasets.sample,
                        request.dataset_id,
                        request.sample_index,
                        split=request.split,
                    )
                    constraint.check_dataset(sample["profile"])
                result = await execute_policy(
                    inference.predict, request.model_dump(), sample=sample, domain=constraint
                )
                constraint.check_model(result["method"], result["profile"])
                record_metadata(
                    "model",
                    request.model_id,
                    {
                        "method": result["method"],
                        "profile": result["profile"],
                    },
                    selected_domain,
                )
                if sample is not None:
                    record_metadata(
                        "dataset",
                        request.dataset_id,
                        {"profile": sample["profile"]},
                        selected_domain,
                    )
                return result
        except (KeyError, FileNotFoundError, IndexError) as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(503, f"Policy execution failed: {error}") from error

    async def create_online_session(payload: dict):
        async with execution_gate:
            require_open()
            require_static_gpu()
            await run_in_threadpool(inference.close)
            return await run_in_threadpool(online.create, payload)

    app.include_router(create_online_router(online, session_creator=create_online_session))
    app.include_router(
        create_online_router(
            online,
            session_creator=create_online_session,
            prefix="/api/v1/modes/ego2d-playground",
        ),
        include_in_schema=False,
    )
    for mode in registered:
        if mode.router is not None:
            app.include_router(mode.router, prefix=f"{prefix}/modes/{mode.descriptor.id}")
    static = static_dir or Path(__file__).parent / "static"
    if (static / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static, html=True), name="studio")
    return app
