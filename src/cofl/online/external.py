"""External stateful policies on the existing continuous controller and plant.

The native CoFL-S entrypoint, action arbitration, protocol and batching remain
independent. External policies submit already-decided trajectories, HOLD or
STOP. Goals, reference paths and oracle progress never enter policy inputs.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import shutil
import sys
import uuid
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from .config import (
    episode_ground_truth,
    ground_truth_paths,
    load_ground_truth,
    load_recipe,
    read_episodes,
    runtime_availability,
)
from .continuous_metrics import compute_episode_metrics as continuous_metrics
from .dynamics import RunParameters, camera_to_world
from .habitat import HabitatEnvironment
from .instructions import FullInstructionProvider, make_instruction_provider
from .metrics import _xyz_path, compute_episode_metrics
from .navigation import EpisodeContext, ExecutionFeedback, NavigationDecision, PolicyObservation
from .runner import episode_geometry, image_url
from .runner import protocol as native_protocol
from .store import ResultStore, file_identity


def trajectory_protocol(parameters, *, ndtw_fdtw=True):
    """Use the existing metric conventions without claiming CoFL field inference."""
    native = native_protocol(parameters, ndtw_fdtw=ndtw_fdtw)
    names = (
        "max_steps",
        "max_time_s",
        "plant_hz",
        "controller_hz",
        "sensor_hz",
        "planner_hz",
        "lookahead_m",
        "max_linear_mps",
        "max_angular_radps",
        "rotate_threshold_deg",
        "metric_distance",
        "metric_recipe",
        "metric_distance_by_name",
        "metric_pose_sampling",
        "executed_path_sampling",
        "navigation_metric_sampling",
        "oracle_success_sampling",
        "ndtw_sampling",
        "ndtw_reference",
        "ndtw_algorithm",
        "ndtw_normalization",
        "geodesic_fallback",
        "unreachable_goal_distance",
        "nonfinite_metric_encoding",
        "success_threshold_m",
        "success_threshold_comparison",
        "timeout_counts_for_success",
        "stop_drain_max_ticks",
        "plant",
    )
    return dict(
        {name: native[name] for name in names},
        protocol="cofl-vln-trajectory-v1",
        mode="benchmark",
        policy_api="navigation-v1",
        trajectory_frame="body_forward_left_metres",
        action_interval="one_planner_tick",
        step_budget_unit="policy_decisions_including_hold_and_stop",
        stop_rule="explicit_stop_immediate",
        hold_rule="zero_motion_one_planner_tick",
        realtime_pacing=False,
    )


def load_factory(spec):
    """Load an explicitly configured module or Python file, without model imports."""
    source, name = spec.rsplit(":", 1)
    if source.endswith(".py") or "/" in source:
        path = Path(source).expanduser().resolve(strict=True)
        identity = file_identity(path, hash_content=True)
        module_name = "_cofl_external_" + hashlib.sha256(str(path).encode()).hexdigest()[:20]
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError("Cannot load policy module: " + str(path))
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    else:
        module = importlib.import_module(source)
        path = getattr(module, "__file__", None)
        identity = file_identity(path, hash_content=True) if path else {"module": source}
    factory = getattr(module, name, None)
    if not callable(factory):
        raise TypeError("External policy factory is not callable: " + spec)
    return factory, {"entrypoint": spec, "module": identity}


def _policy_description(policy):
    for name in ("describe", "begin_episode", "act", "end_episode", "close"):
        if not callable(getattr(policy, name, None)):
            raise TypeError("External policy requires " + name + "()")
    metadata = policy.describe()
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError(
            "Policy describe() must return nonempty JSON metadata including model identity"
        )
    # Descriptions participate in strict resume checks, including model assets
    # and adapter options. Reject nonfinite or unserializable metadata early.
    return json.loads(json.dumps(metadata, allow_nan=False))


def run_episode(
    environment,
    policy,
    episode,
    parameters=None,
    *,
    instruction_provider=None,
    gt_path=None,
    ndtw_fdtw=True,
    geometry=None,
    on_step=None,
    on_begin=None,
    abort_check=None,
):
    """One isolated policy session; all environment state is kept evaluator-side."""
    parameters = parameters or RunParameters()
    provider = instruction_provider or FullInstructionProvider(episode)
    if provider.interactive:
        raise ValueError("External policies currently support fixed benchmark episodes only")
    _, goal, reference = episode_geometry(episode)
    if gt_path is None:
        raise ValueError("VLN-CE benchmark requires official GT locations")
    gt_path = _xyz_path(gt_path)
    if not callable(getattr(environment, "geodesic_distance", None)):
        raise TypeError("Benchmark requires navmesh geodesic distance")
    needs_depth = getattr(policy, "needs_depth", False)
    if not isinstance(needs_depth, bool):
        raise TypeError("Policy needs_depth must be a boolean")
    cancelled = abort_check or (lambda: False)
    geometry = dict(geometry or {"hfov_rad": math.pi / 2})
    info = {
        "parameters": dict(
            trajectory_protocol(parameters, ndtw_fdtw=ndtw_fdtw), **provider.describe()
        ),
        "reference_path": reference[:, [0, 2]].tolist(),
        "goal_position": goal.tolist(),
    }
    if cancelled():
        return dict(info, reason="cancelled", total_steps=0, metrics={}, executed_path=[])
    frame = environment.start(episode, parameters, geometry)
    sensors = copy.deepcopy(frame.get("sensor_geometry", {}))
    info["parameters"]["sensors"] = sensors
    if on_begin is not None:
        on_begin(info)
    snapshot = provider.current(environment, frame, 0)
    if snapshot is None:
        raise ValueError("Benchmark instruction provider returned no instruction")
    context = EpisodeContext(
        session_id=uuid.uuid4().hex,
        episode_id=str(episode["episode_id"]),
        instruction=snapshot.text,
        instruction_mode=provider.describe()["instruction_mode"],
        sensors=sensors,
        planner_hz=parameters.planner_hz,
    )
    dense = [list(frame["state"]["position"])]
    planned, predictions, speeds, blocked, timings = [], [], [], [], []
    reason, count, feedback = "error", 0, None
    # end_episode also runs if begin_episode partially initializes and fails.
    try:
        policy.begin_episode(context)
        reason = "max_steps"
        while count < parameters.max_steps:
            if cancelled():
                reason = "cancelled"
                break
            if float(frame["t_sim_s"]) + 1e-9 >= parameters.max_time_s:
                reason = "timeout"
                break
            if count:
                snapshot = provider.current(environment, frame, count)
            if snapshot is None:
                raise ValueError("Benchmark instruction provider returned no instruction")
            observation = PolicyObservation(
                session_id=context.session_id,
                step_index=count,
                t_sim_s=float(frame["t_sim_s"]),
                image=np.asarray(frame["image"]).copy(),
                depth=np.asarray(frame["depth"]).copy() if needs_depth else None,
                instruction=snapshot.text,
                instruction_revision=snapshot.revision,
                instruction_id=snapshot.command_id,
                feedback=feedback,
            )
            started = perf_counter()
            decision = policy.act(observation)
            inference_ms = (perf_counter() - started) * 1000
            if not isinstance(decision, NavigationDecision):
                raise TypeError("External act() must return NavigationDecision")
            if cancelled():
                reason = "cancelled"
                break
            state = frame["state"]
            trajectory = decision.trajectory
            world = (
                camera_to_world(trajectory, state["position"], state["rotation"])
                if decision.mode == "track"
                else np.empty((0, 2))
            )
            if decision.mode == "track":
                following = environment.advance(world[1:])
            elif decision.mode == "hold":
                following = environment.hold()
            else:
                following = dict(frame, positions=[], linear=[], angular=[], blocked=0)
            dense.extend(following["positions"])
            linear = float(np.mean(following["linear"])) if len(following["linear"]) else 0.0
            angular = float(np.mean(following["angular"])) if len(following["angular"]) else 0.0
            feedback = ExecutionFeedback(
                mode=decision.mode,
                elapsed_sim_s=float(following["t_sim_s"]) - float(frame["t_sim_s"]),
                linear_mps=linear,
                angular_radps=angular,
                blocked_substeps=int(following["blocked"]),
            )
            step = {
                "step_index": count,
                "t_sim_s": float(frame["t_sim_s"]),
                "image": image_url(frame["image"]),
                "instruction": snapshot.text,
                "instruction_info": snapshot.to_dict(),
                "decision_mode": decision.mode,
                "trajectory_body_m": trajectory.tolist() if trajectory is not None else [],
                "world_trajectory": world.tolist(),
                "agent_position": list(state["position"]),
                "agent_rotation": list(state["rotation"]),
                "linear_mps": linear,
                "angular_radps": angular,
                "inference_ms": inference_ms,
                "diagnostics": dict(
                    decision.diagnostics, controller=following.get("controller", {})
                ),
                "blocked_substeps": int(following["blocked"]),
            }
            planned.append(state["position"])
            if len(world):
                predictions.append(world)
            speeds.append(linear)
            blocked.append(int(following["blocked"]))
            timings.append(inference_ms)
            count += 1
            if on_step is not None:
                on_step(step)
            if decision.mode == "stop":
                reason = "stop_action"
                break
            frame = following
            if frame["done"] or float(frame["t_sim_s"]) + 1e-9 >= parameters.max_time_s:
                reason = "timeout"
                break
    except BaseException:
        reason = "error"
        raise
    finally:
        # Preserve the primary exception if cleanup also fails. The run owner
        # will close the policy and abort this run on any error.
        if (primary_error := sys.exception()) is not None:
            try:
                policy.end_episode(reason)
            except Exception as cleanup_error:  # noqa: BLE001 — preserve the primary failure
                primary_error.add_note(f"Policy cleanup also failed: {cleanup_error}")
        else:
            policy.end_episode(reason)
    path = np.asarray(planned if planned else dense[:1], dtype=np.float64)
    dense_path = np.asarray(dense, dtype=np.float64)
    metrics = compute_episode_metrics(
        path,
        reference,
        goal,
        stop_reason=reason,
        dense_path=dense_path,
        geodesic_fn=environment.geodesic_distance,
        gt_path=gt_path,
        ndtw_fdtw=ndtw_fdtw,
    )
    metrics.update(
        continuous_metrics(
            path,
            reference,
            goal,
            speeds,
            blocked,
            predicted_path_cart=np.concatenate(predictions) if predictions else None,
            geodesic_fn=environment.geodesic_distance,
            stop_reason=reason,
            dense_path=dense_path,
        )
    )
    metrics.update(
        path_length_m=float(np.linalg.norm(np.diff(dense_path, axis=0), axis=1).sum()),
        inference_ms_mean=float(np.mean(timings)) if timings else 0.0,
        simulated_time_s=float(frame["t_sim_s"]),
        metric_distance="geodesic_xyz_core_and_xz_extras",
    )
    undefined = [
        key
        for key, value in metrics.items()
        if isinstance(value, (float, int)) and not np.isfinite(value)
    ]
    for key in undefined:
        metrics[key] = None
    if undefined:
        metrics["undefined_metrics"] = undefined
    return dict(
        info, reason=reason, total_steps=count, metrics=metrics, executed_path=dense_path.tolist()
    )


def run_evaluation(args):
    """Serial external entrypoint, sharing cohort selection and atomic storage."""
    from .__main__ import group_indices_by_scene, selected_indices

    if args.workers != 1 or args.inference_batch_size not in (None, 1):
        raise ValueError("External policies require one worker and one inference session")
    recipe = load_recipe(args.benchmark)
    ready, reason = runtime_availability(recipe)
    if not ready:
        raise RuntimeError(reason)
    episodes = read_episodes(recipe)
    indices = selected_indices(args, len(episodes))
    truth = load_ground_truth(recipe)
    for index in indices:
        episode_ground_truth(truth, episodes[index])
    config_path = args.policy_config.resolve() if args.policy_config else None
    config = yaml.safe_load(config_path.read_text()) if config_path else {}
    if not isinstance(config, dict):
        raise TypeError("External policy config must contain a mapping")
    factory, factory_identity = load_factory(args.policy)
    mode = args.instruction_mode or recipe.instruction_mode
    policy, environment = None, None

    def make_policy():
        return factory(
            config=copy.deepcopy(config),
            config_path=config_path,
            device=args.device,
            output_dir=args.output,
        )

    try:
        policy = make_policy()
        metadata = _policy_description(policy)
        recorded = dict(
            trajectory_protocol(args.parameters, ndtw_fdtw=recipe.ndtw_fdtw),
            instruction_mode=mode,
            uses_gt_progress=mode == "oracle",
            progress_lookahead=recipe.progress_lookahead,
            oracle_source=recipe.oracle_source,
            oracle_unavailable=recipe.oracle_unavailable,
            policy=metadata,
        )
        manifest = {
            "protocol": recorded["protocol"],
            "parameters": recorded,
            "checkpoint_identity": {"policy": metadata},
            "policy_factory_identity": factory_identity,
            "policy_config_identity": file_identity(config_path, hash_content=True)
            if config_path
            else None,
            "recipe_identity": file_identity(recipe.path, hash_content=True),
            "dataset_identity": file_identity(recipe.dataset_data_path, hash_content=True),
            "instruction_sources": {
                name: file_identity(path, hash_content=True)
                for name in ("fgr2r_json", "landmark_rxr_json")
                if (path := getattr(recipe, name)) is not None
            },
            "ground_truth_identity": [
                file_identity(path, hash_content=True) for path in ground_truth_paths(recipe)
            ],
            "habitat_config_identity": file_identity(recipe.habitat_config, hash_content=True),
            "runtime_identity": {
                name: file_identity(Path(__file__).with_name(name), hash_content=True)
                for name in (
                    "external.py",
                    "navigation.py",
                    "dynamics.py",
                    "worker.py",
                    "metrics.py",
                    "continuous_metrics.py",
                    "instructions.py",
                )
            },
            "device": args.device,
        }
        with ResultStore(args.output, manifest=manifest, resume=args.resume) as store:
            indices = [i for i in indices if not store.is_completed(episodes[i], i)]
            if args.group_by_scene:
                indices = group_indices_by_scene(indices, episodes)
            for index in indices:
                episode = episodes[index]
                writer = store.begin_episode(episode, index)
                try:
                    provider = make_instruction_provider(recipe, episode, mode)
                    if policy is None:
                        policy = make_policy()
                        if _policy_description(policy) != metadata:
                            raise ValueError("Policy identity changed while retrying the run")
                    if environment is None:
                        environment = HabitatEnvironment(recipe)
                    started = perf_counter()
                    result = run_episode(
                        environment,
                        policy,
                        episode,
                        args.parameters,
                        instruction_provider=provider,
                        gt_path=episode_ground_truth(truth, episode),
                        ndtw_fdtw=recipe.ndtw_fdtw,
                        geometry={"hfov_rad": math.radians(recipe.hfov_deg or 90.0)},
                        on_step=writer.write_step,
                    )
                    result["execution"] = {
                        "episode_wall_s": perf_counter() - started,
                        "workers": 1,
                        "reuse_scene": args.reuse_scene,
                    }
                except Exception as error:
                    # Infrastructure/model errors are not navigation outcomes.
                    # Preserve diagnostics outside committed episodes, then
                    # abort so the same episode is eligible for --resume.
                    failure_dir = args.output / "errors" / f"{index}-{uuid.uuid4().hex}"
                    failure_dir.mkdir(parents=True)
                    shutil.copy2(writer.path / "steps.jsonl", failure_dir / "steps.jsonl")
                    (failure_dir / "error.json").write_text(
                        json.dumps(
                            {
                                "episode_index": index,
                                "episode_id": str(episode["episode_id"]),
                                "error_type": type(error).__name__,
                                "error": str(error),
                            },
                            allow_nan=False,
                            indent=2,
                        )
                    )
                    writer.abort()
                    raise RuntimeError(
                        f"External episode {index} failed; diagnostics: {failure_dir}. "
                        "Fix the error and retry with --resume."
                    ) from error
                else:
                    writer.finish(result)
                    print(
                        json.dumps(
                            {
                                "episode_index": index,
                                "episode_id": str(episode["episode_id"]),
                                "reason": result["reason"],
                                "metrics": result["metrics"],
                            },
                            allow_nan=False,
                        ),
                        flush=True,
                    )
                finally:
                    if not args.reuse_scene and environment is not None:
                        environment.close()
                        environment = None
            summary = store.summary()
            print(
                json.dumps({k: v for k, v in summary.items() if k != "episodes"}, allow_nan=False),
                flush=True,
            )
        return summary
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            if policy is not None:
                policy.close()
