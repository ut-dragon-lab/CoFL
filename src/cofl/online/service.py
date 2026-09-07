"""Single active local run, bounded browser telemetry and explicit cancellation."""

from __future__ import annotations

import copy
import math
import time
import uuid
from collections import deque
from pathlib import Path
from threading import Event, Lock, RLock, Thread, current_thread

from cofl.runtime.device import DeviceManager

from .config import (
    episode_ground_truth,
    ground_truth_paths,
    load_ground_truth,
    load_recipe,
    read_episodes,
    runtime_availability,
)
from .dynamics import RunParameters
from .habitat import HabitatEnvironment
from .policy import NativePolicy
from .runner import episode_geometry, protocol, run_episode


def _visual_environment(recipe):
    """App benchmark and interactive sessions retain every scheduled sensor frame."""
    return HabitatEnvironment(recipe, render_all_sensor_frames=True)


class _ManagedPolicy:
    """Only policy execution reports CUDA failures; renderer failures remain environmental."""

    def __init__(self, policy, device_manager):
        self._policy = policy
        self._device_manager = device_manager

    def __getattr__(self, name):
        return getattr(self._policy, name)

    def predict(self, *args, **kwargs):
        with self._device_manager.execution(check=False, cleanup=self._policy.close):
            return self._policy.predict(*args, **kwargs)


class SessionService:
    def __init__(
        self,
        models,
        benchmarks,
        *,
        device="cuda:0",
        before_start=None,
        environment_factory=_visual_environment,
        policy_factory=NativePolicy,
        availability=runtime_availability,
        scenes=None,
        device_manager=None,
    ):
        self.models = {key: Path(value) for key, value in models.items()}
        self.benchmarks = dict(benchmarks)
        self._scenes = dict(scenes or {})
        self.device_manager = device_manager or DeviceManager(device)
        self.before_start = before_start
        self._environment_factory = environment_factory
        self._policy_factory = policy_factory
        self._availability = availability
        self._lock = RLock()
        self._sessions = {}
        self._active_id = None
        self._closed = False
        self._shutdown_lock = Lock()
        self._pending_closes = {}
        self._recipes = {}
        self._availability_cache = {}
        self._resource_versions = {}

    @property
    def device(self):
        return self.device_manager.device

    @device.setter
    def device(self, value):
        self.device_manager.select(value)

    @property
    def is_running(self):
        with self._lock:
            return self._active_id is not None

    def register_model(self, resource_id, path):
        with self._lock:
            if self._active_id is not None:
                raise RuntimeError("Stop the active online session before changing models")
            self.models[resource_id] = Path(path)

    def register(self, resource_id, path, *, label=None):
        path = Path(path).expanduser().resolve()
        load_recipe(path)  # Invalid YAML never becomes a visible resource.
        with self._lock:
            self.benchmarks[resource_id] = {"path": path, "label": label or path.stem}
            self._recipes.pop(resource_id, None)
            self._availability_cache.pop(resource_id, None)
            self._resource_versions[resource_id] = self._resource_versions.get(resource_id, 0) + 1
        return {"domain_id": "ego2d"}

    def _resource(self, key):
        if key not in self.benchmarks:
            raise KeyError("Unknown online benchmark: " + key)
        value = self.benchmarks[key]
        if isinstance(value, (str, Path)):
            return Path(value), key
        if isinstance(value, dict):
            return Path(value["path"]), value.get("label") or key
        return Path(value.path), value.label or key

    def _recipe(self, key):
        with self._lock:
            path, _ = self._resource(key)
            version = self._resource_versions.get(key, 0)
            cached = self._recipes.get(key)
        stat = path.stat()
        identity = (str(path), stat.st_mtime_ns, stat.st_size)
        if cached is not None and cached[0] == identity:
            return cached[1]
        recipe = load_recipe(path)
        with self._lock:
            # A registration or another reader can finish while the file is read.
            # Never replace a newer resource/cache with this older snapshot.
            current = self._recipes.get(key)
            if not self._closed and self._resource_versions.get(key, 0) == version:
                if current is cached:
                    self._recipes[key] = (identity, recipe)
                    self._availability_cache.pop(key, None)
                elif current is not None and current[0] == identity:
                    return current[1]
        return recipe

    def _available(self, key, recipe):
        with self._lock:
            cached = self._availability_cache.get(key)
            if cached is not None and cached[1] is recipe and time.monotonic() - cached[0] <= 30:
                return cached[2]
        result = self._availability(recipe)
        with self._lock:
            current = self._recipes.get(key)
            if not self._closed and current is not None and current[1] is recipe:
                self._availability_cache[key] = (time.monotonic(), recipe, result)
        return result

    def describe(self):
        with self._lock:
            resources = [(key, self._resource(key)[1]) for key in self.benchmarks]
        entries = []
        for key, label in resources:
            entry = {
                "id": key,
                "label": label,
                "available": False,
                "reason": None,
                "benchmark_reason": None,
                "episode_count": 0,
                "split": "val_unseen",
            }
            try:
                recipe = self._recipe(key)
                entry["split"] = recipe.split
                entry["available"], entry["reason"] = self._available(key, recipe)
                try:
                    entry["episode_count"] = (
                        len(read_episodes(recipe)) if recipe.dataset_data_path is not None else 0
                    )
                    if not entry["episode_count"]:
                        entry["benchmark_reason"] = "No matching benchmark episodes"
                    else:
                        for path in ground_truth_paths(recipe):
                            if not path.is_file():
                                raise FileNotFoundError(
                                    "Official GT file is unavailable: " + str(path)
                                )
                except (OSError, ValueError, KeyError) as error:
                    entry["benchmark_reason"] = str(error)
            except (OSError, ValueError, KeyError) as error:
                entry["reason"] = str(error)
            entries.append(entry)
        return entries

    def episodes(self, benchmark_id, *, offset=0, limit=50):
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError("Episode pagination requires offset >= 0 and limit within 1..200")
        raw = read_episodes(self._recipe(benchmark_id))
        items = [
            {
                "index": index,
                "episode_id": str(episode["episode_id"]),
                "scene_id": str(episode["scene_id"]),
                "instruction": episode["instruction"],
            }
            for index, episode in enumerate(raw[offset : offset + limit], start=offset)
        ]
        return {"items": items, "total": len(raw), "offset": offset, "limit": limit}

    @staticmethod
    def _yaw(rotation):
        x, y, z, w = rotation
        return math.degrees(math.atan2(2 * (w * y + x * z), 1 - 2 * (x * x + y * y)))

    @staticmethod
    def _rotation(yaw):
        if isinstance(yaw, bool) or not isinstance(yaw, (float, int)) or not math.isfinite(yaw):
            raise ValueError("start_yaw_deg must be finite")
        half = math.radians(yaw) / 2
        return [0.0, math.sin(half), 0.0, math.cos(half)]

    @staticmethod
    def _position(position):
        if position is None:
            return None
        if len(position) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
            for v in position
        ):
            raise ValueError("start_position must contain three finite world XYZ coordinates")
        return [float(v) for v in position]

    def scenes(self, benchmark_id):
        items = {}
        recipe = self._recipe(benchmark_id)
        episodes = read_episodes(recipe) if recipe.dataset_data_path is not None else []
        for episode in episodes:
            scene_id = str(episode["scene_id"])
            items.setdefault(
                scene_id,
                {
                    "id": scene_id,
                    "label": Path(scene_id).stem,
                    "scene_id": scene_id,
                    "spawn_position": self._position(episode["start_position"]),
                    "spawn_yaw_deg": self._yaw(episode["start_rotation"]),
                },
            )
        for key, value in self._scenes.items():
            if isinstance(value, (str, Path)):
                path, label = Path(value), key
            elif isinstance(value, dict):
                path, label = Path(value["path"]), value.get("label") or key
            else:
                path, label = value.path, value.label or key
            scene_id = str(path.expanduser().resolve())
            if path.suffix.lower() == ".glb":
                items.setdefault(
                    scene_id,
                    {
                        "id": scene_id,
                        "label": label,
                        "scene_id": scene_id,
                        "spawn_position": None,
                        "spawn_yaw_deg": 0.0,
                    },
                )
        return {"items": list(items.values()), "total": len(items)}

    def create(self, request):
        with self._lock:
            if self._closed:
                raise RuntimeError("Online service is closed")
        from .instructions import LiveInstructionProvider, make_instruction_provider

        request = dict(request)
        benchmark_id, model_id = request.pop("benchmark_id"), request.pop("model_id")
        index = request.pop("episode_index", 0)
        mode = request.pop("mode", "benchmark")
        scene_id = request.pop("scene_id", None)
        position = self._position(request.pop("start_position", None))
        yaw = request.pop("start_yaw_deg", None)
        instruction = request.pop("instruction", "")
        instruction_mode = request.pop("instruction_mode", None)
        if mode not in ("benchmark", "interactive"):
            raise ValueError("mode must be benchmark or interactive")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("episode_index must be a nonnegative integer")
        parameters = RunParameters(**request)
        recipe = self._recipe(benchmark_id)
        if mode == "benchmark":
            if scene_id is not None or position is not None or yaw is not None or instruction:
                raise ValueError("Benchmark scene, spawn and instructions are fixed by its episode")
            episodes = read_episodes(recipe)
            if index >= len(episodes):
                raise ValueError("episode_index is outside the benchmark")
            episode = episodes[index]
            episode_geometry(episode)
            gt_path = episode_ground_truth(load_ground_truth(recipe), episode)
            provider = make_instruction_provider(recipe, episode, mode=instruction_mode)
        else:
            choices = {item["id"]: item for item in self.scenes(benchmark_id)["items"]}
            if scene_id not in choices:
                raise ValueError("Select a scene from this environment's scene catalog")
            choice = choices[scene_id]
            episode = {
                "episode_id": "interactive",
                "scene_id": choice["scene_id"],
                "start_position": position if position is not None else choice["spawn_position"],
                "start_rotation": self._rotation(choice["spawn_yaw_deg"] if yaw is None else yaw),
                "instruction": instruction,
            }
            provider = LiveInstructionProvider(instruction)
            gt_path = None
        available, reason = self._available(benchmark_id, recipe)
        if not available:
            raise FileNotFoundError(reason)
        with self._lock:
            if self._closed:
                raise RuntimeError("Online service is closed")
            if self._active_id is not None:
                raise RuntimeError(
                    "One online session is already running; stop it before starting another"
                )
            if model_id not in self.models:
                raise KeyError("Unknown model: " + model_id)
            checkpoint = self.models[model_id]
            if not checkpoint.is_file():
                raise FileNotFoundError("Checkpoint is unavailable: " + str(checkpoint))
            key = uuid.uuid4().hex
            session = {
                "id": key,
                "benchmark_id": benchmark_id,
                "model_id": model_id,
                "episode_index": index,
                "mode": mode,
                "mode_id": "ego2d-interactive" if mode == "interactive" else "ego2d-playground",
                "scene_id": str(episode["scene_id"]),
                "current_position": None,
                "current_yaw_deg": None,
                "observation": None,
                "command_state": provider.state() if mode == "interactive" else {},
                "active_instruction": None,
                "command_log": [],
                "epoch": 0,
                "status": "starting",
                "reason": None,
                "metrics": {},
                "total_steps": 0,
                "parameters": protocol(
                    parameters,
                    provider,
                    interactive=mode == "interactive",
                    ndtw_fdtw=recipe.ndtw_fdtw,
                ),
                "reference_path": [],
                "goal_position": None,
                "geometry": {},
                "_steps": deque(maxlen=128),
                "_world_path": [],
                "_cancel": Event(),
                "_environment": None,
                "_provider": provider,
                "_gt_path": gt_path,
            }
            self._sessions[key] = session
            self._active_id = key
            # Keep bounded metadata for a local single-user app.
            for old_key in list(self._sessions)[:-16]:
                if old_key != key:
                    self._sessions.pop(old_key)
            try:
                if self.before_start is not None:
                    self.before_start()
                thread = Thread(
                    target=self._execute,
                    args=(key, recipe, checkpoint, episode, parameters),
                    name="CoFLOnlineSession",
                    daemon=True,
                )
                session["_thread"] = thread
                thread.start()
            except Exception:
                self._active_id = None
                self._sessions.pop(key, None)
                raise
            return self.get(key)

    def _execute(self, key, recipe, checkpoint, episode, parameters):
        session = self._sessions[key]
        environment = policy = None
        try:
            if session["_cancel"].is_set():
                return
            with self.device_manager.execution(check=self._policy_factory is NativePolicy):
                policy = self._policy_factory(checkpoint, device=self.device)
            policy = _ManagedPolicy(policy, self.device_manager)
            if session["_cancel"].is_set():
                return
            environment = self._environment_factory(recipe)
            with self._lock:
                session["_environment"] = environment

            def on_begin(info):
                with self._lock:
                    session.update(copy.deepcopy(info))

            def on_state(info):
                with self._lock:
                    preview_keys = {
                        "current_position",
                        "current_rotation",
                        "image",
                        "t_sim_s",
                        "epoch",
                    }
                    if info.get("revision", 0) >= session["command_state"].get("revision", 0):
                        session["command_state"] = {
                            name: copy.deepcopy(value)
                            for name, value in info.items()
                            if name not in preview_keys
                        }
                    if "image" in info:
                        session["observation"] = {
                            "image": info["image"],
                            "t_sim_s": info["t_sim_s"],
                            "position": info["current_position"],
                            "rotation": info["current_rotation"],
                        }
                        session["current_position"] = info["current_position"]
                        session["current_yaw_deg"] = self._yaw(info["current_rotation"])
                    epoch = info.get("epoch", session["epoch"])
                    if epoch != session["epoch"]:
                        session["_world_path"] = []
                        session["active_instruction"] = None
                    session["epoch"] = epoch
                    if session["status"] != "stopping":
                        session["status"] = session["command_state"].get("status", "running")

            def on_step(step):
                with self._lock:
                    session["_world_path"] = step["world_path"]
                    # Retain the full executed path once, on the newest frame.
                    # Old sensor frames do not duplicate an ever-growing trace.
                    session["_steps"].append(dict(step, world_path=[]))
                    session["total_steps"] = step["step_index"] + 1
                    session["current_position"] = step["agent_position"]
                    if step.get("agent_rotation") is not None:
                        session["current_yaw_deg"] = self._yaw(step["agent_rotation"])
                    session["active_instruction"] = step.get("instruction_info")
                    session["epoch"] = step.get("epoch", 0)
                    if session["status"] != "stopping":
                        session["status"] = session["command_state"].get("status", "running")

            result = run_episode(
                environment,
                policy,
                episode,
                parameters,
                on_step=on_step,
                on_begin=on_begin,
                abort_check=session["_cancel"].is_set,
                instruction_provider=session["_provider"],
                interactive=session["mode"] == "interactive",
                on_state=on_state,
                gt_path=session["_gt_path"],
                ndtw_fdtw=recipe.ndtw_fdtw,
            )
            with self._lock:
                for name in (
                    "parameters",
                    "geometry",
                    "reference_path",
                    "goal_position",
                    "reason",
                    "total_steps",
                    "metrics",
                ):
                    session[name] = result[name]
                session["status"] = "cancelled" if result["reason"] == "cancelled" else "completed"
        except Exception as error:  # noqa: BLE001 — isolate failures from the session worker
            with self._lock:
                session["reason"] = str(error)
                session["status"] = "failed"
        finally:
            cleanup_errors = self._close_resources((environment, policy))
            with self._lock:
                if cleanup_errors and session["status"] != "failed":
                    session.update(
                        status="failed",
                        reason="Runtime cleanup failed: " + "; ".join(cleanup_errors),
                    )
                if session["_cancel"].is_set():
                    session.update(status="cancelled", reason="cancelled")
                session["_environment"] = None
                self._active_id = None

    def _close_resources(self, resources):
        """Attempt every cleanup; retain only failed resources for an explicit retry."""
        errors = []
        seen = set()
        for resource in resources:
            if resource is None or id(resource) in seen or not hasattr(resource, "close"):
                continue
            seen.add(id(resource))
            try:
                resource.close()
            except Exception as error:  # noqa: BLE001 — finish independent cleanup and allow retry
                message = type(resource).__name__ + ": " + str(error)
                errors.append(message)
                with self._lock:
                    self._pending_closes[id(resource)] = (resource, message)
            else:
                with self._lock:
                    self._pending_closes.pop(id(resource), None)
        return errors

    def get(self, session_id, *, after_step=-1):
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("Unknown online session: " + session_id)
            session = self._sessions[session_id]
            result = {key: value for key, value in session.items() if not key.startswith("_")}
            steps = session["_steps"]
            result["steps"] = [step for step in steps if step["step_index"] > after_step]
            result["first_available_step"] = steps[0]["step_index"] if steps else 0
            result = copy.deepcopy(result)
            if result["steps"] and result["steps"][-1]["step_index"] == session["total_steps"] - 1:
                result["steps"][-1]["world_path"] = copy.deepcopy(session["_world_path"])
            return result

    def active(self):
        """Recover the current bounded snapshot after a browser refresh."""
        with self._lock:
            return None if self._active_id is None else self.get(self._active_id)

    def _control(self, session_id, action, **kwargs):
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("Unknown online session: " + session_id)
            session = self._sessions[session_id]
            if session_id != self._active_id or session["_cancel"].is_set():
                raise RuntimeError("The session is no longer active")
            if session["mode"] != "interactive":
                raise ValueError(
                    "Live commands and resets are available only in Interactive Playground"
                )
            provider = session["_provider"]
        # The provider protects inference revisions. Never take its lock while
        # holding the session lock: the runner publishes frames in the opposite order.
        state = getattr(provider, action)(**kwargs)
        with self._lock:
            if state["revision"] >= session["command_state"].get("revision", 0):
                session["command_state"] = copy.deepcopy(state)
                if session["status"] not in ("starting", "stopping", "cancelled", "failed"):
                    session["status"] = state["status"]
            session["command_log"].append(
                {
                    "action": action,
                    "revision": state["revision"],
                    "command_id": state["command_id"],
                    "instruction": state["text"],
                    "epoch": state["restart_generation"],
                    "status": state["status"],
                }
            )
            session["command_log"].sort(key=lambda event: event["revision"])
            del session["command_log"][:-256]
        return self.get(session_id)

    def command(self, session_id, instruction):
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 2048:
            raise ValueError("A live command must contain 1..2048 nonblank characters")
        return self._control(session_id, "submit", text=instruction)

    def pause(self, session_id):
        return self._control(session_id, "pause")

    def resume(self, session_id):
        return self._control(session_id, "resume")

    def reset(self, session_id, *, start_position=None, start_yaw_deg=None, instruction=None):
        position = self._position(start_position)
        rotation = None if start_yaw_deg is None else self._rotation(start_yaw_deg)
        return self._control(
            session_id, "restart", position=position, rotation=rotation, instruction=instruction
        )

    def stop(self, session_id):
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("Unknown online session: " + session_id)
            session = self._sessions[session_id]
            if session_id != self._active_id:
                return self.get(session_id)
            session["_cancel"].set()
            session["status"] = "stopping"
            environment = session["_environment"]
            provider = session["_provider"]
        errors = self._close_resources((environment, provider))
        if errors:
            raise RuntimeError("Online cancellation cleanup incomplete: " + "; ".join(errors))
        return self.get(session_id)

    def close(self, *, timeout_s=30.0):
        """Cancel owned work and release its data only after the policy thread exits.

        A timed-out call leaves the service closed to new work and retains the
        thread/resources for another close attempt. Successful calls are idempotent.
        """
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s < 0
        ):
            raise ValueError("Shutdown timeout_s must be finite and nonnegative")
        deadline = time.monotonic() + timeout_s
        if not self._shutdown_lock.acquire(timeout=timeout_s):
            raise TimeoutError("Online shutdown is still in progress; retry shutdown")
        try:
            with self._lock:
                self._closed = True
                sessions = list(self._sessions.values())
                for session in sessions:
                    session["_cancel"].set()
                    thread = session.get("_thread")
                    if thread is not None and thread.is_alive():
                        session["status"] = "stopping"
                resources = [
                    resource
                    for session in sessions
                    for resource in (session.get("_environment"), session.get("_provider"))
                ]
            # Environment.close terminates only this service's owned child process.
            # Wake the provider even when environment cleanup reports an error.
            self._close_resources(resources)
            for session in sessions:
                thread = session.get("_thread")
                if thread is None:
                    continue
                if thread is current_thread():
                    raise RuntimeError("The policy worker cannot wait for its own shutdown")
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if any(
                session.get("_thread") is not None and session["_thread"].is_alive()
                for session in sessions
            ):
                raise TimeoutError(
                    "Online policy worker is still running; resources are not fully released. Retry shutdown."
                )
            with self._lock:
                pending = [resource for resource, _ in self._pending_closes.values()]
            self._close_resources(pending)
            with self._lock:
                if self._pending_closes:
                    raise RuntimeError(
                        "Online resource cleanup incomplete: "
                        + "; ".join(message for _, message in self._pending_closes.values())
                    )
                self._sessions.clear()
                self._active_id = None
                self._recipes.clear()
                self._availability_cache.clear()
                self._resource_versions.clear()
            # Annotation files are read-only cache entries; providers no longer
            # reference them once all owned session threads have terminated.
            from .instructions import _rows

            _rows.cache_clear()
        finally:
            self._shutdown_lock.release()
