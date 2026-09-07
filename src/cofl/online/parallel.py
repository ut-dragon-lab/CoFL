"""Scene-aware spawned evaluators with one canonical, parent-owned result store.

Each child retains the ordinary episode loop and Habitat adapter. Model requests
are batched by one parent-owned policy; children only integrate its CPU fields.
Step logs and episode results are staged beside the parent-owned result store.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import signal
import tempfile
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing.connection import wait
from pathlib import Path
from threading import current_thread, main_thread

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_SCHEDULING = "episode_distinct_scene_first"


class SceneScheduler:
    """Assign one episode at a time, preferring scenes without active workers.

    Within that priority, reuse a worker's previous scene when possible, then
    choose the scene with most pending episodes. If every pending scene already
    has workers, retain the worker's loaded scene or share the scene with the
    largest pending/(active+1) ratio. Ties follow scene order in the selection.
    Unassigned episodes stay available to every worker, including helpers that
    have exhausted a smaller scene. Only the parent accesses this scheduler.
    """

    def __init__(self, indices, episodes):
        self._pending = {}
        seen = set()
        for index in indices:
            if index in seen:
                raise ValueError("Episode selection contains duplicate indices")
            seen.add(index)
            self._pending.setdefault(episodes[index]["scene_id"], deque()).append(index)
        self._active = dict.fromkeys(self._pending, 0)
        self._assignments = {}
        self._previous = {}

    @property
    def pending_count(self):
        return sum(map(len, self._pending.values()))

    @property
    def inflight_count(self):
        return len(self._assignments)

    def assign(self, worker_id):
        """Return a single-episode job or None; require the previous job completed."""
        if worker_id in self._assignments:
            raise RuntimeError(f"Worker {worker_id} already has an in-flight episode")
        candidates = [scene for scene, pending in self._pending.items() if pending]
        if not candidates:
            return None
        previous = self._previous.get(worker_id)
        unoccupied = [scene for scene in candidates if self._active[scene] == 0]
        if unoccupied:
            scene = previous if previous in unoccupied else max(
                unoccupied, key=lambda item: len(self._pending[item])
            )
        elif previous in candidates:
            scene = previous
        else:
            scene = max(
                candidates,
                key=lambda item: len(self._pending[item]) / (self._active[item] + 1),
            )
        index = self._pending[scene].popleft()
        self._assignments[worker_id] = (scene, index)
        self._active[scene] += 1
        return [index]

    def complete(self, worker_id):
        """Release the scene after the parent has committed the assigned episode."""
        if worker_id not in self._assignments:
            raise RuntimeError(f"Worker {worker_id} has no in-flight episode")
        scene, _index = self._assignments.pop(worker_id)
        self._active[scene] -= 1
        self._previous[worker_id] = scene


@contextmanager
def _thread_environment(count):
    # Spawn imports its main module before calling our worker. Set these before
    # process.start() so even early NumPy/OpenMP imports see the requested bound.
    previous = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    try:
        os.environ.update({name: str(count) for name in _THREAD_VARIABLES})
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _termination_cleanup():
    """Let a CLI SIGTERM unwind worker/store cleanup, then restore its handler."""
    if current_thread() is not main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _worker_main(connection, context, spool_root):
    import torch

    from .config import episode_ground_truth
    from .habitat import HabitatEnvironment
    from .inference import RemoteInferenceError, RemotePolicy
    from .instructions import make_instruction_provider
    from .runner import run_episode

    args, recipe = context["args"], context["recipe"]
    torch.set_num_threads(args.worker_threads)
    torch.set_num_interop_threads(args.worker_threads)
    mode = args.instruction_mode or recipe.instruction_mode
    policy = environment = None
    try:
        policy = RemotePolicy(
            connection,
            dict(context["geometry"]),
            args.parameters,
            context["worker_id"],
        )
        while True:
            job = connection.recv()
            if job is None:
                return
            for index in job:
                episode = context["episodes"][index]
                directory = Path(spool_root) / str(index)
                directory.mkdir()
                result = None
                error = None
                with (directory / "steps.jsonl").open("w", encoding="utf-8") as stream:

                    def write_step(step):
                        stream.write(
                            json.dumps(
                                dict(step, world_path=[]), ensure_ascii=False, allow_nan=False
                            )
                            + "\n"
                        )
                        stream.flush()

                    try:
                        provider = make_instruction_provider(recipe, episode, mode)
                        if environment is None:
                            environment = HabitatEnvironment(recipe)
                        started = time.perf_counter()
                        result = run_episode(
                            environment,
                            policy,
                            episode,
                            args.parameters,
                            instruction_provider=provider,
                            gt_path=episode_ground_truth(context["ground_truth"], episode),
                            ndtw_fdtw=recipe.ndtw_fdtw,
                            on_step=write_step,
                        )
                        result["execution"] = {
                            "episode_wall_s": time.perf_counter() - started,
                            "group_by_scene": True,
                            "reuse_scene": args.reuse_scene,
                            "inference_backend_requested": args.inference_backend,
                            "inference_backend": getattr(
                                policy, "active_inference_backend", "eager"
                            ),
                            "acceleration": getattr(policy, "acceleration_status", None),
                            "workers_requested": args.workers,
                            "workers_effective": context["workers_effective"],
                            "worker_id": context["worker_id"],
                            "worker_pid": os.getpid(),
                            "worker_threads": args.worker_threads,
                            "scene_scheduling": _SCHEDULING,
                            "inference_architecture": "shared_model_batched",
                            "shared_model_pid": context["shared_model_pid"],
                            "inference_batch_size": context["inference_batch_size"],
                            "batch_wait_ms": args.batch_wait_ms,
                        }
                    except (MemoryError, torch.cuda.OutOfMemoryError, RemoteInferenceError):
                        raise
                    except Exception as exception:  # noqa: BLE001 — same per-episode failure policy
                        error = f"{type(exception).__name__}: {exception}"
                        result = None
                        if environment is not None:
                            environment.close()
                            environment = None
                    finally:
                        if not args.reuse_scene and environment is not None:
                            environment.close()
                            environment = None
                payload = {"result": result, "error": error}
                (directory / "result.json").write_text(
                    json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8"
                )
                connection.send({"event": "episode", "index": index})
            # Keep the IPC event name for compatibility; it now finishes one
            # assigned episode, not an exclusive reservation of an entire scene.
            connection.send({"event": "scene_done"})
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            if policy is not None:
                policy.close()


def _worker_bootstrap(connection, context, spool_root, worker_entry):
    # Habitat inherits this group. The parent can clean the entire tree even if
    # this evaluator dies before running its adapter's finally block.
    if os.name == "posix":
        os.setsid()
    try:
        worker_entry(connection, context, spool_root)
    except Exception as exception:
        try:
            connection.send(
                {
                    "event": "fatal",
                    "error": f"{type(exception).__name__}: {exception}",
                    "resource_failure": isinstance(exception, MemoryError)
                    or type(exception).__name__ == "OutOfMemoryError",
                }
            )
        except (OSError, EOFError):
            pass  # The coordinator may already be cancelling the worker group.
        raise
    finally:
        connection.close()


@dataclass
class _Worker:
    process: object
    connection: object
    worker_id: int
    pending: set = field(default_factory=set)
    stopping: bool = False
    prediction: tuple | None = None
    last_request_id: int = -1


def _stop_workers(workers):
    # Signal all groups first; a slow renderer must not multiply the wait budget.
    for worker in workers:
        process = worker.process
        if process.pid is None:
            continue
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                if process.is_alive():
                    process.terminate()  # It may not have reached setsid yet.
        elif process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 3.0
    for worker in workers:
        if worker.process.pid is not None:
            worker.process.join(timeout=max(0, deadline - time.monotonic()))
    for worker in workers:
        process = worker.process
        if process.pid is not None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.is_alive():
                process.kill()
            process.join(timeout=1)
        worker.connection.close()


def _commit_episode(store, episodes, index, spool_root):
    directory = Path(spool_root) / str(index)
    payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    writer = store.begin_episode(episodes[index], index)
    try:
        writer.import_steps(directory / "steps.jsonl")
        if payload["error"] is not None:
            writer.fail(payload["error"], result=payload["result"])
            report = {"episode_index": index, "error": payload["error"]}
        else:
            result = payload["result"]
            writer.finish(result)
            report = {
                "episode_index": index,
                "episode_id": str(episodes[index]["episode_id"]),
                "reason": result["reason"],
                "metrics": result["metrics"],
            }
    except BaseException:
        writer.abort()
        raise
    shutil.rmtree(directory)
    print(json.dumps(report, allow_nan=False), flush=True)


def run_parallel(
    *,
    args,
    recipe,
    episodes,
    ground_truth,
    indices,
    store,
    worker_entry=None,
    policy_factory=None,
    service_factory=None,
):
    """Evaluate episodes concurrently with scene affinity and a shared policy.

    A child process crash aborts the run after committing its received results.
    Uncommitted episodes remain eligible for resume. ``worker_entry`` is an
    injectable, spawn-picklable implementation used by CPU-only integration tests.
    Policy and service factories execute only in the parent, never in children.
    """
    if args.workers < 1 or args.worker_threads < 1:
        raise ValueError("workers and worker_threads must be positive")
    if args.workers > 1 and not args.group_by_scene:
        raise ValueError("Parallel evaluation requires group_by_scene")
    indices = list(indices)
    if len(set(indices)) != len(indices):
        raise ValueError("Parallel episode selection contains duplicate indices")
    remaining = [index for index in indices if not store.is_completed(episodes[index], index)]
    if not remaining:
        return
    scheduler = SceneScheduler(remaining, episodes)
    effective = min(args.workers, len(remaining))
    batch_size = min(args.inference_batch_size or effective, effective)
    context = {
        "args": args,
        "recipe": recipe,
        "episodes": episodes,
        "ground_truth": ground_truth,
        "workers_effective": effective,
        "shared_model_pid": os.getpid(),
        "inference_batch_size": batch_size,
    }
    workers = []
    policy = service = None
    spawn = multiprocessing.get_context("spawn")
    with (
        _termination_cleanup(),
        tempfile.TemporaryDirectory(prefix=".parallel-", dir=store.output_dir) as spool_root,
    ):
        try:
            if policy_factory is None:
                from .policy import NativePolicy

                policy_factory = NativePolicy
            if service_factory is None:
                from .inference import BatchedInferenceService

                service_factory = BatchedInferenceService
            try:
                policy = policy_factory(
                    args.checkpoint,
                    device=args.device,
                    inference_backend=args.inference_backend,
                )
            except Exception as exception:
                raise RuntimeError(
                    "Shared policy initialization failed: "
                    f"{type(exception).__name__}: {exception}; no episodes were committed"
                ) from exception
            context["geometry"] = dict(policy.geometry)
            with _thread_environment(args.worker_threads):
                for worker_id in range(effective):
                    parent, child = spawn.Pipe()
                    process = spawn.Process(
                        target=_worker_bootstrap,
                        args=(
                            child,
                            dict(context, worker_id=worker_id),
                            spool_root,
                            worker_entry or _worker_main,
                        ),
                        name=f"cofl-scene-{worker_id}",
                    )
                    worker = _Worker(process, parent, worker_id)
                    workers.append(worker)
                    try:
                        process.start()
                    finally:
                        child.close()
                    job = scheduler.assign(worker_id)
                    worker.pending = set(job)
                    parent.send(job)

            # The inference thread starts after spawn, once the temporary child
            # thread-limit environment has been restored in the parent.
            service = service_factory(
                policy,
                args.parameters,
                max_batch_size=batch_size,
                batch_wait_ms=args.batch_wait_ms,
            )
            predictions = {}

            def receive(worker):
                while worker.connection.poll():
                    try:
                        event = worker.connection.recv()
                    except EOFError:
                        return
                    if event.get("event") == "predict":
                        request_id = event.get("request_id")
                        if (
                            not worker.pending
                            or worker.stopping
                            or worker.prediction is not None
                            or isinstance(request_id, bool)
                            or not isinstance(request_id, int)
                            or request_id <= worker.last_request_id
                        ):
                            raise RuntimeError("Worker sent an invalid or duplicate policy request")
                        key = (worker.worker_id, request_id)
                        worker.prediction = key
                        worker.last_request_id = request_id
                        predictions[key] = worker
                        service.submit(key, event["sample"])
                    elif event.get("event") == "episode":
                        index = event["index"]
                        if index not in worker.pending or worker.prediction is not None:
                            raise RuntimeError(f"Worker returned an unexpected episode: {index}")
                        _commit_episode(store, episodes, index, spool_root)
                        worker.pending.remove(index)
                    elif event.get("event") == "scene_done":
                        if worker.pending or worker.stopping or worker.prediction is not None:
                            raise RuntimeError("Worker ended a task with missing episode results")
                        scheduler.complete(worker.worker_id)
                        job = scheduler.assign(worker.worker_id)
                        worker.pending = set(job or ())
                        worker.stopping = job is None
                        worker.connection.send(job)
                    elif event.get("event") == "fatal":
                        advice = (
                            " Reduce --workers to lower memory demand before retrying."
                            if event.get("resource_failure")
                            else ""
                        )
                        raise RuntimeError(
                            f"Scene worker {worker.process.name} failed: {event['error']}; "
                            "uncommitted episodes can be retried with --resume." + advice
                        )
                    else:
                        raise RuntimeError(f"Unknown evaluator event: {event!r}")

            active = list(workers)
            while active:
                ready = wait(
                    [worker.connection for worker in active],
                    timeout=0.002 if predictions else 0.1,
                )
                for worker in active:
                    if worker.connection in ready:
                        receive(worker)
                try:
                    completed = service.drain()
                except Exception as exception:
                    resource_failure = isinstance(exception, MemoryError) or (
                        type(exception).__name__ == "OutOfMemoryError"
                    )
                    advice = (
                        " Reduce --inference-batch-size to lower memory demand before retrying."
                        if resource_failure
                        else ""
                    )
                    raise RuntimeError(
                        "Shared inference failed: "
                        f"{type(exception).__name__}: {exception}; uncommitted episodes "
                        "can be retried with --resume." + advice
                    ) from exception
                for key, prediction in completed:
                    worker = predictions.pop(key, None)
                    if worker is None or worker.prediction != key:
                        raise RuntimeError(f"Shared inference returned an unknown request: {key}")
                    worker.connection.send(
                        {"event": "prediction", "request_id": key[1], "result": prediction}
                    )
                    worker.prediction = None
                for worker in list(active):
                    if worker.process.is_alive():
                        continue
                    # A process may have exited after sending its last result.
                    # Drain the pipe before classifying the exit as an interruption.
                    receive(worker)
                    if not worker.stopping or worker.process.exitcode != 0:
                        raise RuntimeError(
                            f"Scene worker {worker.process.name} exited unexpectedly "
                            f"(exit code {worker.process.exitcode}); uncommitted episodes "
                            "can be retried with --resume"
                        )
                    # Clean an exited worker's group immediately, before a long
                    # remaining scene could allow its PID to be reused elsewhere.
                    _stop_workers([worker])
                    active.remove(worker)
                    workers.remove(worker)
            return service.snapshot()
        finally:
            try:
                _stop_workers(workers)
            finally:
                # Never release model weights while the service may still be
                # executing: a failed close deliberately skips policy.close().
                if service is not None:
                    service.close()
                if policy is not None:
                    policy.close()
