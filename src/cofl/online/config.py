"""Portable benchmark recipes and read-only runtime capability checks."""

from __future__ import annotations

import gzip
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkRecipe:
    path: Path
    habitat_config: Path
    dataset_data_path: Path | None
    scenes_dir: Path
    habitat_python: Path
    runtime_pythonpath: tuple[Path, ...] = ()
    split: str = "val_unseen"
    language_filter: str = "en"
    gpu_id: int = 0
    instruction_mode: str = "oracle"
    fgr2r_json: Path | None = None
    landmark_rxr_json: Path | None = None
    progress_lookahead: int = 6
    hfov_deg: float | None = None
    gt_path: Path | None = None
    ndtw_fdtw: bool = True
    oracle_unavailable: str = "error"
    oracle_source: str = "auto"

    def worker_options(self):
        return {
            "habitat_config": str(self.habitat_config),
            "scenes_dir": str(self.scenes_dir),
            "gpu_id": self.gpu_id,
            "hfov_deg": self.hfov_deg,
        }

    def environment(self):
        env = os.environ.copy()
        source = Path(__file__).resolve().parents[2]
        paths = [str(source)] + [str(p) for p in self.runtime_pythonpath]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        return env


def load_recipe(path) -> BenchmarkRecipe:
    import yaml

    path = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ValueError("Invalid benchmark YAML: " + str(error)) from error
    if not isinstance(document, dict):
        raise ValueError("Benchmark YAML must contain a mapping")  # noqa: TRY004 — invalid document
    options = document.get("online", document)
    required = {"habitat_config", "scenes_dir"}
    allowed = required | {
        "dataset_data_path",
        "habitat_python",
        "runtime_pythonpath",
        "split",
        "language_filter",
        "gpu_id",
        "instruction_mode",
        "fgr2r_json",
        "landmark_rxr_json",
        "progress_lookahead",
        "hfov_deg",
        "gt_path",
        "ndtw_fdtw",
        "oracle_unavailable",
        "oracle_source",
    }
    if not isinstance(options, dict) or required - options.keys():
        raise ValueError("Environment YAML requires habitat_config and scenes_dir")
    if options.keys() - allowed:
        raise ValueError(
            "Unsupported benchmark options: " + ", ".join(sorted(options.keys() - allowed))
        )
    split = options.get("split", "val_unseen")
    if not isinstance(split, str) or not split:
        raise ValueError("Benchmark split must be a nonempty string")

    def resolve(value, *, executable=False):
        if not isinstance(value, str) or not value:
            raise ValueError("Benchmark paths must be nonempty strings")
        value = value.replace("{split}", split)
        result = path.parent / Path(value).expanduser()
        # Resolving a venv interpreter symlink loses its environment.
        return result.absolute() if executable else result.resolve()

    pythonpaths = options.get("runtime_pythonpath", [])
    if not isinstance(pythonpaths, list):
        raise ValueError("runtime_pythonpath must be a list")  # noqa: TRY004 — invalid document
    language = options.get("language_filter", "en")
    if not isinstance(language, str):
        raise ValueError("language_filter must be a string")  # noqa: TRY004 — invalid document
    gpu = options.get("gpu_id", 0)
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError("gpu_id must be a nonnegative integer")
    mode = options.get("instruction_mode", "oracle")
    if mode not in ("oracle", "full_instruction"):
        raise ValueError("instruction_mode must be oracle or full_instruction")
    oracle_unavailable = options.get("oracle_unavailable", "error")
    if oracle_unavailable not in ("error", "full_instruction"):
        raise ValueError("oracle_unavailable must be error or full_instruction")
    oracle_source = options.get("oracle_source", "auto")
    if oracle_source not in ("auto", "landmark_rxr"):
        raise ValueError("oracle_source must be auto or landmark_rxr")
    lookahead = options.get("progress_lookahead", 6)
    if isinstance(lookahead, bool) or not isinstance(lookahead, int) or not 0 <= lookahead <= 100:
        raise ValueError("progress_lookahead must be an integer within 0..100")
    hfov = options.get("hfov_deg")
    fdtw = options.get("ndtw_fdtw", True)
    if not isinstance(fdtw, bool):
        raise ValueError("ndtw_fdtw must be a boolean")  # noqa: TRY004 — invalid recipe
    if hfov is not None and (
        isinstance(hfov, bool) or not isinstance(hfov, (int, float)) or not 1 <= hfov <= 179
    ):
        raise ValueError("hfov_deg must be within 1..179")
    return BenchmarkRecipe(
        path=path,
        habitat_config=resolve(options["habitat_config"]),
        dataset_data_path=resolve(options["dataset_data_path"])
        if options.get("dataset_data_path") is not None
        else None,
        scenes_dir=resolve(options["scenes_dir"]),
        habitat_python=resolve(options.get("habitat_python", sys.executable), executable=True),
        runtime_pythonpath=tuple(resolve(p) for p in pythonpaths),
        split=split,
        language_filter=language,
        gpu_id=gpu,
        instruction_mode=mode,
        fgr2r_json=resolve(options["fgr2r_json"])
        if options.get("fgr2r_json") is not None
        else None,
        landmark_rxr_json=resolve(options["landmark_rxr_json"])
        if options.get("landmark_rxr_json") is not None
        else None,
        progress_lookahead=lookahead,
        hfov_deg=hfov,
        gt_path=resolve(options["gt_path"]) if options.get("gt_path") is not None else None,
        ndtw_fdtw=fdtw,
        oracle_unavailable=oracle_unavailable,
        oracle_source=oracle_source,
    )


def read_episodes(recipe: BenchmarkRecipe) -> list[dict]:
    path = recipe.dataset_data_path
    if path is None:
        raise ValueError("Benchmark unavailable: this environment has no dataset_data_path")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    episodes = document.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("Benchmark source must contain an episodes list")  # noqa: TRY004 — invalid document
    result = []
    for raw in episodes:
        if not isinstance(raw, dict):
            raise ValueError("Each benchmark episode must be a mapping")  # noqa: TRY004 — invalid document
        language = raw.get("instruction", {})
        language = language.get("language", "") if isinstance(language, dict) else ""
        if language and recipe.language_filter and not language.startswith(recipe.language_filter):
            continue
        instruction = raw.get("instruction", {})
        if isinstance(instruction, dict):
            instruction = instruction.get("instruction_text", "")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Every online episode requires a language instruction")
        if not raw.get("scene_id") or "episode_id" not in raw:
            raise ValueError("Every online episode requires scene_id and episode_id")
        result.append(
            dict(raw, instruction=instruction, instruction_data=raw.get("instruction", {}))
        )
    return result


def ground_truth_paths(recipe: BenchmarkRecipe) -> tuple[Path, ...]:
    """Resolve the official NDTW GT_PATH, including both RxR annotation roles."""
    if recipe.gt_path is None:
        raise ValueError(
            "VLN-CE benchmark requires gt_path pointing to the official *_gt.json.gz "
            "trajectory file; episode reference_path is not a substitute"
        )
    template = str(recipe.gt_path)
    if "{role}" in template:
        return tuple(Path(template.replace("{role}", role)) for role in ("guide", "follower"))
    return (recipe.gt_path,)


def validate_gt_locations(value):
    """Validate official XYZ locations without altering their sampling or duplicates."""
    if not isinstance(value, list) or not value:
        raise ValueError("Official GT locations must be a nonempty list of finite XYZ positions")
    for position in value:
        if (
            not isinstance(position, list)
            or len(position) != 3
            or any(
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not math.isfinite(component)
                for component in position
            )
        ):
            raise ValueError(
                "Official GT locations must be a nonempty list of finite XYZ positions"
            )
    return value


def load_ground_truth(recipe: BenchmarkRecipe) -> dict:
    """Read official episode-id mappings; only selected trajectories need validation."""

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate key in official GT: " + str(key))
            result[key] = value
        return result

    result = {}
    for path in ground_truth_paths(recipe):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            document = json.load(stream, object_pairs_hook=unique_object)
        if not isinstance(document, dict):
            raise ValueError("Official GT must map episode IDs to trajectory records")  # noqa: TRY004 — invalid document
        overlap = result.keys() & document.keys()
        if overlap:
            raise ValueError(
                "Ambiguous episode IDs across official GT files: " + str(sorted(overlap))
            )
        result.update(document)
    return result


def episode_ground_truth(ground_truth, episode):
    identifier = str(episode["episode_id"])
    record = ground_truth.get(identifier)
    if not isinstance(record, dict) or "locations" not in record:
        raise ValueError("Official GT trajectory is missing for episode " + identifier)
    return validate_gt_locations(record["locations"])


def runtime_availability(recipe: BenchmarkRecipe) -> tuple[bool, str | None]:
    for key in ("habitat_config", "habitat_python"):
        value = getattr(recipe, key)
        if not value.is_file():
            return False, f"Missing {key}: {value}"
    if not recipe.scenes_dir.is_dir():
        return False, f"Missing scene assets directory: {recipe.scenes_dir}"
    probe = (
        "import importlib.util,json; "
        "print(json.dumps([n for n in ('habitat_sim','numpy','quaternion','yaml') "
        "if importlib.util.find_spec(n) is None]))"
    )
    try:
        completed = subprocess.run(
            [str(recipe.habitat_python), "-c", probe],
            env=recipe.environment(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode:
            return False, "Habitat interpreter probe failed: " + completed.stderr[-600:]
        missing = json.loads(completed.stdout.strip().splitlines()[-1])
        if missing:
            return False, "Habitat runtime is unavailable; missing modules: " + ", ".join(missing)
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as error:
        return False, "Habitat runtime probe failed: " + str(error)
    return True, None
