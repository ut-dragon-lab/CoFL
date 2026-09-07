"""Parent-side transport for an isolated simulator, with bounded waits."""

from __future__ import annotations

import json
import selectors
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path
from threading import Lock, RLock

import numpy as np


class HabitatEnvironment:
    """Isolated simulator; visual consumers opt into every scheduled sensor frame."""

    def __init__(self, recipe, *, render_all_sensor_frames=False):
        if not isinstance(render_all_sensor_frames, bool):
            raise TypeError("render_all_sensor_frames must be a boolean")
        self.recipe = recipe
        self.render_all_sensor_frames = render_all_sensor_frames
        self._process = None
        self._resources = ExitStack()
        self._scratch = self._resources.enter_context(
            tempfile.TemporaryDirectory(prefix="cofl-online-")
        )
        self._log = self._resources.enter_context(
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8")  # noqa: SIM115 — owned by ExitStack
        )
        self._close_lock = Lock()
        self._io_lock = RLock()
        self._closed = False
        self._initialization = None
        self._scene_key = None

    def start(self, episode, parameters, geometry, *, interactive=False):
        options = dict(
            self.recipe.worker_options(), render_all_sensor_frames=self.render_all_sensor_frames
        )
        # Sensor geometry and renderer settings belong to the loaded scene;
        # timing and controller parameters can change at an episode reset.
        scene_key = json.dumps(
            [episode["scene_id"], options, geometry], sort_keys=True, allow_nan=False
        )
        with self._io_lock:
            with self._close_lock:
                if self._closed:
                    raise RuntimeError("Online session was cancelled before simulator startup")
                reuse = (
                    self._process is not None
                    and self._process.poll() is None
                    and self._scene_key == scene_key
                )
                if not reuse:
                    self._stop_process()
                    self._process = subprocess.Popen(
                        [str(self.recipe.habitat_python), "-u", "-m", "cofl.online.worker"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=self._log,
                        text=True,
                        env=self.recipe.environment(),
                    )
            payload = {
                "command": "reset_episode" if reuse else "initialize",
                # The simulator only needs the scene and spawn. RxR annotations
                # can contain NaN word timestamps; keep those host-side with the
                # instruction provider instead of serializing them over IPC.
                "episode": {
                    key: episode[key]
                    for key in ("scene_id", "start_position", "start_rotation")
                },
                "parameters": parameters.to_dict(),
                "interactive": interactive,
            }
            if not reuse:
                payload.update(options=options, geometry=geometry, scratch=self._scratch)
            try:
                frame = self._request(payload)
                initial_episode = dict(
                    episode,
                    start_position=frame["state"]["position"],
                    start_rotation=frame["state"]["rotation"],
                )
                with self._close_lock:
                    if self._closed:
                        raise RuntimeError("Online session was cancelled")
                    self._scene_key = scene_key
                    self._initialization = (initial_episode, parameters, geometry, interactive)
            except Exception:
                # A partial initialize/reset must never leak a worker or carry
                # unknown controller state into the next benchmark episode.
                with self._close_lock:
                    self._stop_process()
                raise
            return frame

    def _request(self, payload):
        with self._io_lock:
            return self._exchange(payload)

    def _exchange(self, payload):
        process = self._process
        if process is None or self._closed:
            raise RuntimeError("Simulator is closed")
        try:
            process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
            process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=90):
                    raise RuntimeError("Habitat did not respond within 90 seconds")
            line = process.stdout.readline()
            if not line:
                self._log.seek(0)
                raise RuntimeError("Habitat worker exited: " + self._log.read()[-1800:])
            response = json.loads(line)
            if not response.get("ok"):
                error_type = (
                    ValueError if response.get("error_type") == "ValueError" else RuntimeError
                )
                raise error_type("Habitat failed: " + response.get("error", "unknown error"))
            if payload["command"] in ("initialize", "reset_episode", "advance"):
                with np.load(Path(self._scratch) / "sensors.npz", allow_pickle=False) as arrays:
                    response.update(image=arrays["image"].copy(), depth=arrays["depth"].copy())
            return response
        except (OSError, ValueError) as error:
            if self._closed:
                raise RuntimeError("Online session was cancelled") from error
            raise

    def advance(self, reference):
        return self._request({"command": "advance", "reference": np.asarray(reference).tolist()})

    def hold(self):
        """An external policy waits one planner tick with zero commanded motion."""
        return self.advance(np.empty((0, 2), dtype=np.float64))

    def geodesic_distance(self, point_a, point_b):
        result = self._request(
            {"command": "geodesic_distance", "start": list(point_a), "end": list(point_b)}
        )
        return float("inf") if result["distance"] is None else float(result["distance"])

    def reset(self, *, position=None, rotation=None):
        if self._initialization is None:
            raise RuntimeError("Simulator has not been initialized")
        episode, parameters, geometry, interactive = self._initialization
        if not interactive:
            raise ValueError("Benchmark episodes cannot change their initial pose")
        target = dict(episode)
        if position is not None:
            target["start_position"] = position
        if rotation is not None:
            target["start_rotation"] = rotation
        with self._io_lock:
            # Invalid interactive poses leave the current episode usable.
            pose = self._exchange(
                {
                    "command": "validate_pose",
                    "position": target["start_position"],
                    "rotation": target["start_rotation"],
                }
            )
            target.update(start_position=pose["position"], start_rotation=pose["rotation"])
            frame = self.start(target, parameters, geometry, interactive=True)
            # Omitted reset coordinates always mean the original spawn.
            self._initialization = (episode, parameters, geometry, interactive)
            return frame

    def _stop_process(self):
        self._scene_key = None
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
        self._process = None

    def close(self):
        with self._close_lock:
            if self._closed and self._process is None:
                self._resources.close()
                return
            self._closed = True
            self._stop_process()
            self._resources.close()
