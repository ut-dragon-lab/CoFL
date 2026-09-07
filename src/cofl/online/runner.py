"""One policy/control loop for benchmark evaluation and live interaction."""

from __future__ import annotations

import base64
import time
from collections import deque
from io import BytesIO

import numpy as np

from .dynamics import ActionArbitrator, RunParameters, camera_to_world
from .instructions import FullInstructionProvider, LiveInstructionProvider


def image_url(image):
    from PIL import Image

    stream = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB").save(
        stream, format="JPEG", quality=85
    )
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def episode_geometry(episode, *, interactive=False):
    start = np.asarray(episode.get("start_position"), dtype=np.float64)
    if interactive and episode.get("start_position") is None:
        start = np.zeros(3)
    rotation = np.asarray(episode.get("start_rotation"), dtype=np.float64)
    if (
        start.shape != (3,)
        or rotation.shape != (4,)
        or not np.isfinite(start).all()
        or not np.isfinite(rotation).all()
        or not np.isclose(np.linalg.norm(rotation), 1.0, atol=1e-3)
    ):
        raise ValueError("Episode requires finite XYZ start and a unit XYZW quaternion")
    if interactive:
        return start, None, np.empty((0, 3))
    goals = episode.get("goals", [])
    if not goals or not isinstance(goals[0], dict):
        raise ValueError("Benchmark episode requires a goal position")
    goal = np.asarray(goals[0].get("position"), dtype=np.float64)
    reference = np.asarray(episode.get("reference_path"), dtype=np.float64)
    if (
        goal.shape != (3,)
        or reference.ndim != 2
        or reference.shape[1] != 3
        or not len(reference)
        or not np.isfinite(goal).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Benchmark requires a finite XYZ goal and reference path")
    return start, goal, reference


def protocol(parameters, instruction_provider=None, *, interactive=False, ndtw_fdtw=True):
    from cofl.runtime.inference_rules import inference_rule_metadata

    result = dict(
        parameters.to_dict(),
        protocol="cofl-s-interactive-v3" if interactive else "cofl-s-vln-benchmark-v6",
        mode="interactive" if interactive else "benchmark",
        instruction_mode="live" if interactive else "full_instruction",
        memory_mode="single_frame",
        metric_distance=None if interactive else "geodesic_xyz_core_and_xz_extras",
        metric_recipe=None if interactive else "vlnce_official_core_v1",
        metric_distance_by_name={}
        if interactive
        else {
            "NE/OS/SR": "navmesh_geodesic",
            "SPL": "navmesh_shortest_and_xyz_executed",
            "nDTW": "xyz_dtw_official_gt_locations",
            "SDTW": "ndtw_times_explicit_stop_geodesic_success",
            "CLS": "legacy_custom_xz",
            "cSPL": "navmesh_success_and_xz_reference_and_executed_lengths",
            "HS/VC/BSR/EF/HA": "xz_and_controller_samples",
        },
        metric_pose_sampling="planner_ticks",
        executed_path_sampling="plant_ticks",
        navigation_metric_sampling=None if interactive else "plant_xyz_including_terminal",
        oracle_success_sampling=None if interactive else "all_plant_xyz",
        ndtw_sampling=None if interactive else "plant_xyz_consecutive_duplicates_removed",
        ndtw_reference=None if interactive else "official_gt_path_locations",
        ndtw_algorithm=None if interactive else "fastdtw_radius_1" if ndtw_fdtw else "exact_dtw",
        ndtw_normalization=None if interactive else "success_threshold_times_gt_location_count",
        geodesic_fallback=None,
        unreachable_goal_distance=None if interactive else "positive_infinity",
        nonfinite_metric_encoding="null_with_undefined_metric_names",
        success_threshold_m=3.0,
        success_threshold_comparison="strict_less_than",
        timeout_counts_for_success=False,
        stop_drain_max_ticks=0,
        plant="rotate_translate_navmesh_snap",
        trajectory_integration="polar_grid_time_scaled_euler_project",
        inference_rules=inference_rule_metadata(
            "ground_sector_v1",
            grid_size=parameters.field_grid_size,
            num_steps=parameters.policy_steps,
            policy_dt=parameters.policy_dt,
        ),
        field_evaluation="dense_polar_grid",
        field_grid_shape=[parameters.field_grid_size, parameters.field_grid_size],
        field_grid_axes=["radius_normalized", "theta_normalized"],
        field_grid_bounds=[[0.0, 1.0], [-1.0, 1.0]],
        field_grid_alignment="inclusive_endpoints",
        field_interpolation="bilinear_border_align_corners_true",
        realtime_pacing=interactive,
        command_budget_steps=parameters.max_steps if interactive else None,
        command_budget_sim_s=parameters.max_time_s if interactive else None,
        interactive_path_limit=10000 if interactive else None,
    )
    if instruction_provider is not None:
        result.update(instruction_provider.describe())
    return result


def run_episode(
    environment,
    policy,
    episode,
    parameters=None,
    *,
    on_step=None,
    on_begin=None,
    abort_check=None,
    instruction_provider=None,
    interactive=False,
    on_state=None,
    gt_path=None,
    ndtw_fdtw=True,
):
    """Run fixed benchmark instructions or revisioned commands in one simulator.

    In live mode, waiting and pause freeze simulated time. Commands invalidate
    predictions still in flight. Benchmark callers must supply official GT
    ``locations`` via gt_path, separately from the episode's reference route.
    The caller owns and closes the adapters.
    """
    from .continuous_metrics import compute_episode_metrics as continuous_metrics
    from .metrics import compute_episode_metrics as classic_metrics

    parameters = parameters or RunParameters()
    cancelled = abort_check or (lambda: False)
    provider = instruction_provider or (
        LiveInstructionProvider() if interactive else FullInstructionProvider(episode)
    )
    if bool(provider.interactive) != interactive:
        raise ValueError("Instruction provider does not match the requested session mode")
    start, goal, reference = episode_geometry(episode, interactive=interactive)
    info = {
        "parameters": protocol(parameters, provider, interactive=interactive, ndtw_fdtw=ndtw_fdtw),
        "geometry": dict(policy.geometry),
        "reference_path": reference[:, [0, 2]].tolist(),
        "goal_position": None if goal is None else goal.tolist(),
    }
    if on_begin is not None:
        on_begin(info)
    if cancelled():
        return dict(info, reason="cancelled", total_steps=0, metrics={})
    if interactive:
        frame = environment.start(episode, parameters, policy.geometry, interactive=True)
    else:
        # Validate before spending time loading or running the simulator.
        from .metrics import _xyz_path

        if gt_path is None:
            raise ValueError("VLN-CE benchmark requires the episode's official GT locations")
        gt_path = _xyz_path(gt_path)
        if not callable(getattr(environment, "geodesic_distance", None)):
            raise ValueError("Benchmark metrics require Habitat navmesh geodesic distance")
        frame = environment.start(episode, parameters, policy.geometry)
    info["parameters"]["sensors"] = frame.get("sensor_geometry", {})
    if callable(getattr(policy, "configure_sensors", None)):
        policy.configure_sensors(frame.get("sensor_geometry", {}))
        info["geometry"] = dict(policy.geometry)
    # The simulator may snap the requested spawn onto its navmesh.
    start = np.asarray(frame["state"]["position"], dtype=np.float64)
    if on_begin is not None:
        on_begin(info)
    arbiter = ActionArbitrator(parameters)
    dense = deque([start.tolist()], maxlen=10000) if interactive else [start.tolist()]
    planned_positions, predictions, speeds, blocked_counts, inference_times = [], [], [], [], []
    reason, count, command_steps = "max_steps", 0, 0
    epoch, requested_reset, previous_revision = 0, 0, None
    previous_command_id, command_start_sim = None, float(frame["t_sim_s"])
    pace_wall, pace_sim, was_idle = time.monotonic(), float(frame["t_sim_s"]), True
    emitted_state_key = None

    def emit_state(*, force=False):
        nonlocal emitted_state_key
        if on_state is None:
            return
        status = provider.state() if interactive else {}
        key = (
            status.get("revision"),
            status.get("status"),
            status.get("reset_pending"),
            status.get("reset_error"),
            float(frame["t_sim_s"]),
            epoch,
        )
        if not force and key == emitted_state_key:
            return
        emitted_state_key = key
        on_state(
            dict(
                status,
                current_position=list(frame["state"]["position"]),
                current_rotation=list(frame["state"]["rotation"]),
                image=image_url(frame["image"]),
                t_sim_s=float(frame["t_sim_s"]),
                epoch=epoch,
            )
        )

    emit_state(force=True)
    while True:
        if cancelled():
            reason = "cancelled"
            break
        if not interactive and count >= parameters.max_steps:
            break
        if interactive:
            status = provider.state()
            if status.get("closed"):
                reason = "cancelled"
                break
            if status["restart_generation"] != requested_reset:
                requested_reset = status["restart_generation"]
                try:
                    reset_frame = environment.reset(**status["restart_pose"])
                except (ValueError, RuntimeError, OSError) as error:
                    provider.acknowledge_restart(requested_reset, error=error)
                else:
                    frame, epoch = reset_frame, requested_reset
                    dense.clear()
                    dense.append(list(frame["state"]["position"]))
                    arbiter = ActionArbitrator(parameters)
                    provider.acknowledge_restart(requested_reset)
                    was_idle = True
                emit_state(force=True)
                continue
        snapshot = provider.current(environment, frame, count)
        emit_state()
        if snapshot is None:
            was_idle = True
            provider.wait(0.1)
            continue
        if snapshot.revision != previous_revision:
            previous_revision = snapshot.revision
            arbiter = ActionArbitrator(parameters)
        if snapshot.command_id != previous_command_id:
            previous_command_id = snapshot.command_id
            command_steps = 0
            command_start_sim = float(frame["t_sim_s"])
        if interactive:
            if was_idle:
                pace_wall, pace_sim, was_idle = time.monotonic(), float(frame["t_sim_s"]), False
            delay = pace_wall + float(frame["t_sim_s"]) - pace_sim - time.monotonic()
            if delay > 0:
                provider.wait(min(0.1, delay))
                continue
            if (
                command_steps >= parameters.max_steps
                or float(frame["t_sim_s"]) - command_start_sim >= parameters.max_time_s
            ):
                provider.complete(
                    snapshot,
                    reason="max_steps" if command_steps >= parameters.max_steps else "timeout",
                )
                emit_state(force=True)
                continue
        prediction = policy.predict(frame["image"], frame["depth"], snapshot.text, parameters)
        if cancelled():
            reason = "cancelled"
            break
        trajectory = np.asarray(prediction["trajectory"], dtype=np.float64)
        control, should_stop, diagnostics = arbiter.decide(trajectory, prediction.get("actions"))
        diagnostics["field_stop_reason"] = prediction.get("field_stop_reason", "unknown")
        state = frame["state"]
        actual_world = camera_to_world(trajectory, state["position"], state["rotation"])
        control_world = camera_to_world(control, state["position"], state["rotation"])

        def execute(should_stop=should_stop, control_world=control_world):
            if should_stop:
                return {"positions": [], "linear": [], "angular": [], "blocked": 0, "done": False}
            return environment.advance(control_world[1:])

        accepted, following = provider.execute_if_current(snapshot, execute)
        if not accepted:
            continue
        dense.extend(following["positions"])
        path = np.asarray(dense, dtype=np.float64)
        linear = float(np.mean(following["linear"])) if following["linear"] else 0.0
        angular = float(np.mean(following["angular"])) if following["angular"] else 0.0
        diagnostics.update(
            controller=following.get("controller", {}),
            blocked_substeps=int(following["blocked"]),
            control_trajectory=control.tolist(),
        )
        step = {
            "step_index": count,
            "epoch": epoch,
            "t_sim_s": float(frame["t_sim_s"]),
            "image": image_url(frame["image"]),
            "instruction": snapshot.text,
            "instruction_info": snapshot.to_dict(),
            "sector_trajectory": trajectory.tolist(),
            "world_trajectory": actual_world.tolist(),
            "world_path": path[:, [0, 2]].tolist(),
            "agent_position": list(state["position"]),
            "agent_rotation": list(state["rotation"]),
            "actions": None
            if prediction.get("actions") is None
            else np.asarray(prediction["actions"]).tolist(),
            "linear_mps": linear,
            "angular_radps": angular,
            "inference_ms": float(prediction["inference_ms"]),
            "diagnostics": diagnostics,
        }
        if "inference_batch" in prediction:
            step["inference_batch"] = prediction["inference_batch"]
        count += 1
        command_steps += 1
        if not interactive:
            planned_positions.append(state["position"])
            predictions.append(actual_world)
            speeds.append(linear)
            blocked_counts.append(int(following["blocked"]))
            inference_times.append(float(prediction["inference_ms"]))
        if on_step is not None:
            on_step(step)
        if should_stop:
            if interactive:
                provider.complete(snapshot, reason=diagnostics["reason"])
                emit_state(force=True)
                continue
            reason = diagnostics["reason"]
            break
        frame = following
        emit_state()
        if frame["done"]:
            if interactive:
                raise RuntimeError("Interactive simulator unexpectedly exhausted its global clock")
            reason = "timeout"
            break
    dense_path = np.asarray(dense, dtype=np.float64)
    metrics = {}
    if not interactive:
        path = np.asarray(planned_positions if planned_positions else [start], dtype=np.float64)
        metrics = classic_metrics(
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
                blocked_counts,
                predicted_path_cart=np.concatenate(predictions) if predictions else None,
                geodesic_fn=environment.geodesic_distance,
                stop_reason=reason,
                dense_path=dense_path,
            )
        )
        metrics.update(
            path_length_m=float(np.linalg.norm(np.diff(dense_path, axis=0), axis=1).sum()),
            inference_ms_mean=float(np.mean(inference_times)) if inference_times else 0.0,
            simulated_time_s=float(frame.get("t_sim_s", 0.0)),
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
