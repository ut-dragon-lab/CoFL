"""Python 3.7-compatible Habitat-Sim worker, JSON commands and NPZ sensors.

Only simulator dependencies live in this interpreter. Model inference, dataset
storage, HTTP and Pydantic stay in the CoFL Python environment. The ground plant
uses the benchmark's rotate-then-translate/navmesh projection semantics.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from copy import copy
from pathlib import Path

import numpy as np

from .dynamics import RunParameters, SimulationClock, pursuit


def scene_path(scene_id, scenes_dir):
    source = Path(scene_id)
    if source.is_absolute():
        candidates = [source]
    else:
        root = Path(scenes_dir)
        parts = source.parts
        suffix = (
            Path(*parts[parts.index("scene_datasets") + 1 :])
            if "scene_datasets" in parts
            else source
        )
        candidates = [
            root / suffix,
            root / source,
            root / "mp3d" / source.stem / (source.stem + ".glb"),
            root / source.stem / (source.stem + ".glb"),
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Episode scene asset is unavailable: " + str(candidates[0]))


def task_settings(path):
    import yaml

    path = Path(path)
    document = yaml.safe_load(path.read_text()) or {}
    if "BASE_TASK_CONFIG_PATH" in document:
        return task_settings(path.parent / document["BASE_TASK_CONFIG_PATH"])
    if "SIMULATOR" not in document:
        raise ValueError("Habitat task YAML requires SIMULATOR sensor configuration")
    return document["SIMULATOR"]


class SimulatorRuntime:
    def __init__(self, options, episode, parameters, geometry, scratch, interactive=False):
        import habitat_sim as hs

        self.params = RunParameters(**parameters)
        self.render_all_sensor_frames = options.get("render_all_sensor_frames", False)
        if not isinstance(self.render_all_sensor_frames, bool):
            raise TypeError("render_all_sensor_frames must be a boolean")
        self.scene_id = episode["scene_id"]
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.sim = None
        settings = task_settings(options["habitat_config"])
        config = hs.SimulatorConfiguration()
        config.scene_id = str(scene_path(episode["scene_id"], options["scenes_dir"]))
        config.gpu_device_id = int(options.get("gpu_id", 0))
        if hasattr(config, "allow_sliding"):
            config.allow_sliding = True
        agent_config = hs.agent.AgentConfiguration()
        agent_config.height = float(settings.get("AGENT_0", {}).get("HEIGHT", 1.5))
        agent_config.radius = float(settings.get("AGENT_0", {}).get("RADIUS", 0.1))
        sensors = []
        self.sensor_geometry = {}
        for name, sensor_type in (("rgb", hs.SensorType.COLOR), ("depth", hs.SensorType.DEPTH)):
            values = settings.get(name.upper() + "_SENSOR", {})
            sensor = hs.CameraSensorSpec() if hasattr(hs, "CameraSensorSpec") else hs.SensorSpec()
            sensor.uuid = name
            sensor.sensor_type = sensor_type
            sensor.resolution = [int(values.get("HEIGHT", 224)), int(values.get("WIDTH", 224))]
            if min(sensor.resolution) < 1 or max(sensor.resolution) > 2048:
                raise ValueError("Online sensor sizes must be within 1..2048")
            configured_hfov = options.get("hfov_deg")
            hfov = float(
                values.get("HFOV", math.degrees(geometry["hfov_rad"]))
                if configured_hfov is None
                else configured_hfov
            )
            if not math.isfinite(hfov) or not 0 < hfov <= 180:
                raise ValueError("Online sensor HFOV must be finite and within (0, 180] degrees")
            if self.sensor_geometry and not math.isclose(
                hfov, self.sensor_geometry["rgb"]["hfov_deg"], abs_tol=1e-6
            ):
                raise ValueError("RGB and depth sensor HFOV must match for ground-sector inference")
            if hasattr(sensor, "hfov"):
                sensor.hfov = hfov
            else:
                sensor.parameters["hfov"] = str(hfov)
            sensor.position = [float(v) for v in values.get("POSITION", [0.0, 1.5, 0.0])]
            sensors.append(sensor)
            self.sensor_geometry[name] = {
                "resolution": [int(v) for v in sensor.resolution],
                "position": [float(v) for v in sensor.position],
                "hfov_deg": hfov,
            }
        agent_config.sensor_specifications = sensors
        self.depth_max = float(settings.get("DEPTH_SENSOR", {}).get("MAX_DEPTH", 10.0))
        self.sim = hs.Simulator(hs.Configuration(config, [agent_config]))
        if not self.sim.pathfinder.is_loaded:
            self.close()
            raise RuntimeError(
                "Episode navmesh is unavailable; closed-loop ground motion requires it"
            )
        self.agent = self.sim.get_agent(0)
        try:
            self.reset_episode(episode, parameters, interactive=interactive)
        except Exception:
            self.close()
            raise

    def reset_episode(self, episode, parameters, interactive=False):
        """Reset all episode state while retaining the static scene and renderer."""
        import quaternion

        if episode["scene_id"] != self.scene_id:
            raise ValueError("Episode reset requires the currently loaded scene")
        params = RunParameters(**parameters)
        pose = self.validate_pose(episode["start_position"], episode["start_rotation"])
        state = self.agent.get_state()
        state.position = np.asarray(pose["position"], dtype=np.float32)
        x, y, z, w = pose["rotation"]
        state.rotation = quaternion.quaternion(w, x, y, z)
        # This ground plant never steps Habitat physics. Restoring the agent
        # and its sensor transforms is sufficient to reset the static world.
        self.agent.set_state(state, reset_sensors=True)
        self.params = params
        self.clock = SimulationClock(params)
        self.interactive = bool(interactive)
        self.command = (0.0, 0.0)
        self.reference = np.zeros((0, 2))
        self.pending = None
        self.obs = None
        self.controller_debug = {}

    def validate_pose(self, position, rotation):
        if position is None:
            position = self.sim.pathfinder.get_random_navigable_point()
        target = np.asarray(position, dtype=np.float32)
        quaternion = np.asarray(rotation, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Spawn must contain three finite world XYZ coordinates")
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("Spawn rotation must contain four finite XYZW quaternion values")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-8:
            raise ValueError("Spawn quaternion cannot be zero")
        snapped = np.asarray(self.sim.pathfinder.snap_point(target), dtype=np.float32)
        if (
            snapped.shape != (3,)
            or not np.isfinite(snapped).all()
            or not self.sim.pathfinder.is_navigable(snapped)
            or float(np.linalg.norm(snapped[[0, 2]] - target[[0, 2]])) > 0.2
            or abs(float(snapped[1] - target[1])) > 0.2
        ):
            raise ValueError(
                "Spawn is outside the scene navmesh or on another floor; "
                "choose a navigable XYZ position within 0.2 m of its surface"
            )
        return {"ok": True, "position": snapped.tolist(), "rotation": (quaternion / norm).tolist()}

    def geodesic_distance(self, start, end):
        import habitat_sim as hs

        path = hs.ShortestPath()
        for name, point in (("requested_start", start), ("requested_end", end)):
            value = np.asarray(point, dtype=np.float32)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("Geodesic queries require finite world XYZ points")
            setattr(path, name, value)
        found = self.sim.pathfinder.find_path(path)
        distance = float(path.geodesic_distance) if found else float("inf")
        if math.isnan(distance) or distance < 0:
            raise ValueError("Navmesh geodesic distance must be nonnegative or positive infinity")
        return {"ok": True, "distance": distance if math.isfinite(distance) else None}

    def state(self):
        state = self.agent.get_state()
        q = state.rotation
        return {"position": np.asarray(state.position).tolist(), "rotation": [q.x, q.y, q.z, q.w]}

    def plant(self):
        import quaternion

        linear, angular = self.command
        state = self.agent.get_state()
        start = np.asarray(state.position, dtype=np.float32).copy()
        half = angular * self.clock.dt / 2
        rotation = state.rotation * quaternion.quaternion(math.cos(half), 0, math.sin(half), 0)
        distance = linear * self.clock.dt
        blocked = False
        if abs(distance) > 1e-9:
            forward = quaternion.rotate_vectors(rotation, np.array([0.0, 0.0, -1.0]))
            candidate = start + forward * distance
            candidate[1] = start[1]
            snapped = np.asarray(self.sim.pathfinder.snap_point(candidate), dtype=np.float32)
            valid = (
                np.isfinite(snapped).all()
                and self.sim.pathfinder.is_navigable(snapped)
                and np.linalg.norm(snapped[[0, 2]] - candidate[[0, 2]])
                <= max(0.05, abs(distance) * 1.5)
            )
            state.position = snapped if valid else start
            blocked = not valid
        state.rotation = rotation
        self.agent.set_state(state)
        end = np.asarray(state.position, dtype=np.float32)
        moved = float(np.linalg.norm(end[[0, 2]] - start[[0, 2]]))
        blocked = blocked or (abs(distance) > 1e-6 and moved < abs(distance) * 0.5)
        return end.tolist(), int(blocked)

    def _next_observation_time(self):
        """Find the last sensor sample consumed by the next planner/terminal frame.

        Only the latest sample survives an advance. Preview the same clock ticks
        so rendering can stay at that sample's original pose, even when sensor
        and planner rates do not divide evenly. No agent or sensor transform
        needs to be saved and restored. Pending planner events were already
        sampled by the preceding advance and are not ticked again here.
        """
        clock = copy(self.clock)
        clock.deadlines = dict(clock.deadlines)
        sampled_at = None
        while self.interactive or not clock.is_done:
            events = clock.tick()
            if events["sensor"]:
                sampled_at = clock.t_sim
            if events["planner"]:
                break
        return sampled_at

    def advance(self, reference=None):
        if reference is not None:
            value = np.asarray(reference, dtype=np.float64)
            if value.shape == (0,):
                # JSON cannot retain the (0, 2) shape. Only external HOLD sends
                # an empty reference; pursuit already returns zero for it.
                value = value.reshape(0, 2)
            if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
                raise ValueError("Reference trajectory must contain finite world XZ points")
            self.reference = value
            if not len(value):
                self.command = (0.0, 0.0)
        # Interactive sessions and the App's visual benchmark keep every
        # configured sensor tick. Only headless evaluation coalesces rendering.
        render_all = self.render_all_sensor_frames or self.interactive
        sampled_at = None if render_all else self._next_observation_time()
        positions, linear, angular, blocked = [], [], [], 0
        while True:
            if self.pending is not None:
                events, self.pending = self.pending, None
            else:
                if self.clock.is_done and not self.interactive:
                    break
                events = self.clock.tick()
                if events["sensor"] and (render_all or self.clock.t_sim == sampled_at):
                    self.obs = self.sim.get_sensor_observations()
                if events["planner"]:
                    self.pending = events
                    break
            if events["controller"]:
                state = self.state()
                self.command, self.controller_debug = pursuit(
                    state["position"], state["rotation"], self.reference, self.params
                )
            pos, hit = self.plant()
            positions.append(pos)
            linear.append(self.command[0])
            angular.append(self.command[1])
            blocked += hit
        image = np.asarray(self.obs["rgb"])[..., :3].astype(np.uint8)
        depth = np.clip(
            np.asarray(self.obs["depth"], dtype=np.float32).squeeze(), 0, self.depth_max
        )
        np.savez(self.scratch / "sensors.npz", image=image, depth=depth)
        return {
            "ok": True,
            "done": not self.interactive and self.clock.is_done and self.pending is None,
            "t_sim_s": self.clock.t_sim,
            "state": self.state(),
            "positions": positions,
            "linear": linear,
            "angular": angular,
            "blocked": blocked,
            "controller": self.controller_debug,
            "sensor_geometry": self.sensor_geometry,
        }

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None


def main():
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    # C++ renderer messages must not enter the JSON protocol.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    runtime = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                command = request["command"]
                if command == "initialize":
                    if runtime is not None:
                        runtime.close()
                        runtime = None
                    runtime = SimulatorRuntime(
                        request["options"],
                        request["episode"],
                        request["parameters"],
                        request["geometry"],
                        request["scratch"],
                        interactive=request.get("interactive", False),
                    )
                    response = runtime.advance()
                elif command == "reset_episode":
                    if runtime is None:
                        raise RuntimeError("Simulator has not been initialized")
                    runtime.reset_episode(
                        request["episode"],
                        request["parameters"],
                        interactive=request.get("interactive", False),
                    )
                    response = runtime.advance()
                elif command == "advance":
                    if runtime is None:
                        raise RuntimeError("Simulator has not been initialized")
                    response = runtime.advance(request["reference"])
                elif command == "validate_pose":
                    if runtime is None:
                        raise RuntimeError("Simulator has not been initialized")
                    response = runtime.validate_pose(request.get("position"), request["rotation"])
                elif command == "geodesic_distance":
                    if runtime is None:
                        raise RuntimeError("Simulator has not been initialized")
                    response = runtime.geodesic_distance(request["start"], request["end"])
                elif command == "close":
                    break
                else:
                    raise ValueError("Unknown simulator command")
                print(json.dumps(response, allow_nan=False), file=protocol, flush=True)
            except Exception as error:  # noqa: BLE001 — report errors across the process boundary
                traceback.print_exc(file=sys.stderr)
                print(
                    json.dumps(
                        {"ok": False, "error": str(error), "error_type": type(error).__name__}
                    ),
                    file=protocol,
                    flush=True,
                )
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
