"""Episode adapter for native generation, with an optional isolated Habitat worker."""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from cofl.generation.types import GenerationUnit

from .core import episode_identity, merge_augmentation
from .sink import NativeSink


class SectorPipeline:
    profile = "ground_sector_v1"
    recipe_version = "1"
    writer_options = {
        "image_encoding": "jpeg",
        "field_dtype": "float16",
        "depth_dtype": "float16",
        "chunk_rows": 32,
        "compression_shuffle": "byte",
        "compression_level": 3,
    }

    def __init__(self, options: dict, *, base_dir: Path):
        required = {"habitat_config", "dataset_data_path", "scenes_dir", "subinstructions"}
        allowed = required | {
            "gt_actions_path",
            "habitat_python",
            "runtime_pythonpath",
            "replay_mode",
            "split",
            "augmentation",
        }
        if not isinstance(options, dict):
            raise ValueError("Sector pipeline options must be a mapping")
        if required - set(options):
            raise ValueError(
                "Missing sector pipeline options: " + ", ".join(sorted(required - set(options)))
            )
        if set(options) - allowed:
            raise ValueError(
                "Unknown sector pipeline options: " + ", ".join(sorted(set(options) - allowed))
            )
        alignment = options["subinstructions"]
        if not isinstance(alignment, dict) or set(alignment) != {"kind", "path"}:
            raise ValueError("subinstructions must contain kind and path")
        if options.get("replay_mode", "gt") == "gt" and not options.get("gt_actions_path"):
            raise ValueError("GT replay requires gt_actions_path")
        if not isinstance(options.get("runtime_pythonpath", []), list):
            raise ValueError(
                "runtime_pythonpath must be a list of explicit runtime source directories"
            )
        if not isinstance(options.get("augmentation", {}), dict):
            raise ValueError("augmentation must be a mapping")
        self.options = json.loads(json.dumps(options))
        self.base_dir = Path(base_dir).resolve()
        self._worker = None
        self._runtime = None
        self._log = None
        self._core = None
        self._units = None
        self.options.setdefault("replay_mode", "gt")
        self.options.setdefault("split", "train")
        for name in ("habitat_config", "dataset_data_path", "scenes_dir"):
            self.options[name] = str(self._path(self.options[name]))
        if self.options["replay_mode"] == "gt":
            self.options["gt_actions_path"] = str(self._path(self.options["gt_actions_path"]))
        elif self.options["replay_mode"] != "follower":
            raise ValueError("replay_mode must be gt or follower")
        alignment = self.options["subinstructions"]
        if alignment["kind"] not in ("fgr2r", "landmark_rxr"):
            raise ValueError("subinstructions.kind must be fgr2r or landmark_rxr")
        alignment["path"] = str(self._path(alignment["path"]))
        for name in ("habitat_config", "dataset_data_path"):
            if not Path(self.options[name]).is_file():
                raise FileNotFoundError(self.options[name])
        if not Path(alignment["path"]).is_file():
            raise FileNotFoundError(alignment["path"])
        self.options["runtime_pythonpath"] = [
            str(self._path(p)) for p in self.options.get("runtime_pythonpath", [])
        ]
        if self.options.get("habitat_python"):
            # A venv executable must retain its path to locate that environment's packages.
            python_path = Path(
                str(self.options["habitat_python"]).format(split=self.options["split"])
            ).expanduser()
            self.options["habitat_python"] = str((self.base_dir / python_path).absolute())
        self.augmentation = merge_augmentation(self.options.get("augmentation"))
        self.extent = float(self.augmentation["bev_x_max"])
        self.hfov = float(self.augmentation["hfov_rad"])
        steps = self.extent / float(self.augmentation["bev_resolution"])
        if not np.isclose(steps, round(steps)) or self.extent <= 0 or not 0 < self.hfov < math.pi:
            raise ValueError("Invalid inclusive Cartesian sector geometry")
        if float(self.augmentation["v_norm"]) != self.extent:
            raise ValueError("Native ground-sector profile requires v_norm == bev_x_max")
        self.grid_shape = (int(round(steps)) + 1, 2 * int(round(steps)) + 1)

    def _path(self, value):
        path = Path(str(value).format(split=self.options.get("split", "train"))).expanduser()
        return (self.base_dir / path).resolve() if not path.is_absolute() else path.resolve()

    def _scene_inputs(self, scene_id):
        scene = Path(scene_id)
        if not scene.is_absolute():
            parts = scene.parts
            if "scene_datasets" in parts:
                scene = Path(*parts[parts.index("scene_datasets") + 1 :])
            scene = Path(self.options["scenes_dir"]) / scene
        if not scene.is_file():
            raise FileNotFoundError("Episode scene asset does not exist: " + str(scene))
        # Navmesh, semantic mesh and labels are all inputs to the same replay.
        return tuple(sorted(path.resolve() for path in scene.parent.rglob("*") if path.is_file()))

    def units(self):
        if self._units is None:
            dataset = Path(self.options["dataset_data_path"])
            opener = gzip.open if dataset.suffix == ".gz" else open
            with opener(dataset, "rt", encoding="utf-8") as stream:
                document = json.load(stream)
            common = [
                dataset,
                Path(self.options["habitat_config"]),
                Path(self.options["subinstructions"]["path"]),
            ]
            if self.options["replay_mode"] == "gt":
                common.append(Path(self.options["gt_actions_path"]))
            # Config inheritance is explicit, and contributes to source identity.
            import yaml

            configuration = yaml.safe_load(Path(self.options["habitat_config"]).read_text()) or {}
            if "BASE_TASK_CONFIG_PATH" in configuration:
                base = Path(configuration["BASE_TASK_CONFIG_PATH"])
                if not base.is_absolute():
                    base = Path(self.options["habitat_config"]).parent / base
                if not base.is_file():
                    raise ValueError(
                        "Use a direct Habitat task config or a BASE_TASK_CONFIG_PATH relative to its config file"
                    )
                common.append(base.resolve())
            cache = {}
            units, keys = [], set()
            for episode in document["episodes"]:
                key = ":".join(episode_identity(episode))
                if key in keys:
                    raise ValueError("Source episode identities are not unique: " + key)
                keys.add(key)
                scene_id = episode["scene_id"]
                if scene_id not in cache:
                    cache[scene_id] = self._scene_inputs(scene_id)
                units.append(
                    GenerationUnit(key, episode, tuple(sorted(set(common + list(cache[scene_id])))))
                )
            self._units = units
        return iter(self._units)

    def _request(self, request):
        try:
            self._worker.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            self._worker.stdin.flush()
            line = self._worker.stdout.readline()
            if not line:
                self._log.seek(0)
                raise RuntimeError(
                    "Habitat worker exited before replying: " + self._log.read()[-8000:]
                )
            response = json.loads(line)
            if not response.get("ok"):
                self._log.seek(0)
                raise RuntimeError(
                    "Habitat worker failed: "
                    + response.get("error", "")
                    + "\n"
                    + self._log.read()[-8000:]
                )
            return response
        except BaseException:
            self.close(force=True)
            raise

    def _start_worker(self):
        if self._worker is not None:
            return
        env = os.environ.copy()
        package_src = Path(__file__).resolve().parents[3]
        paths = [str(package_src)] + self.options.get("runtime_pythonpath", [])
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        self._log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        self._worker = subprocess.Popen(
            [self.options["habitat_python"], "-u", "-m", "cofl.generation.sector.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            env=env,
        )
        response = self._request({"command": "initialize", "options": self.options})
        self._runtime = response["runtime"]

    def provenance(self):
        if self.options.get("habitat_python"):
            self._start_worker()
            return self._runtime
        from .worker import runtime_provenance

        return runtime_provenance()

    def generate(self, unit, writer, *, seed):
        sink = NativeSink(
            writer,
            key=unit.key,
            split=self.options["split"],
            extent=self.extent,
            hfov=self.hfov,
            grid_shape=self.grid_shape,
        )
        if not self.options.get("habitat_python"):
            if self.options.get("runtime_pythonpath"):
                raise ValueError(
                    "runtime_pythonpath requires habitat_python; install runtime dependencies for in-process replay"
                )
            if self._core is None:
                from .core import SectorCore

                self._core = SectorCore(self.options)
            summary = self._core.generate(unit.payload, seed=seed, sink=sink)
        else:
            self._start_worker()
            scratch = writer.path / ".sector-transport"
            if scratch.exists():
                raise FileExistsError("Refusing to reuse existing worker transport directory")
            try:
                response = self._request(
                    {
                        "command": "generate",
                        "episode": unit.payload,
                        "seed": int(seed),
                        "scratch": str(scratch.resolve()),
                    }
                )
                summary = response["summary"]
                if (scratch / "episode.json").exists():
                    sink.begin_episode(json.loads((scratch / "episode.json").read_text()))
                for frame_index in range(response["frames"]):
                    stem = f"frame-{frame_index:06d}"
                    frame = json.loads((scratch / (stem + ".json")).read_text())
                    with np.load(scratch / (stem + ".npz"), allow_pickle=False) as arrays:
                        frame.update({name: arrays[name] for name in arrays.files})
                    frame["rgb_blob"] = frame["rgb_blob"].tobytes()
                    sink.add_frame(frame)
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
        summary["qa"] = {"status": "passed", **sink.qa}
        return summary

    def close(self, force=False):
        worker, self._worker = self._worker, None
        if worker is not None:
            try:
                if worker.poll() is None and not force:
                    worker.stdin.write('{"command":"close"}\n')
                    worker.stdin.flush()
                    worker.wait(timeout=10)
                elif worker.poll() is None:
                    worker.terminate()
                    worker.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                worker.kill()
                worker.wait(timeout=10)
            finally:
                worker.stdin.close()
                worker.stdout.close()
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._core is not None:
            self._core.close()
            self._core = None
