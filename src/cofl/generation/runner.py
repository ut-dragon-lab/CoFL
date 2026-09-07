"""Generate immutable native shards with deterministic recovery and provenance."""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import shutil
from pathlib import Path

from cofl.data._io import file_sha256
from cofl.data.collection import publish_collection
from cofl.data.storage import DatasetWriter

from .config import GenerationConfig
from .io import InputFingerprints, digest_json, output_lock, tree_checksums, write_json
from .metadata import portable_metadata
from .random import seeded_random, unit_seed

LOGGER = logging.getLogger(__name__)
RUNNER_VERSION = "1"


def _read_json(path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _code_fingerprint():
    root = Path(__file__).resolve().parents[1]
    files = {}
    for folder in ("generation", "data", "fields"):
        for path in sorted((root / folder).rglob("*.py")):
            files[path.relative_to(root).as_posix()] = file_sha256(path)
    return digest_json(files)


def _environment():
    result = {"python": platform.python_version()}
    for name in (
        "numpy",
        "scipy",
        "opencv-python",
        "opencv-python-headless",
        "Pillow",
        "pyarrow",
        "zarr",
        "numcodecs",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _verify_shard(path, expected_report=None):
    report_path = path / "generation-report.json"
    if expected_report is not None and file_sha256(report_path) != expected_report:
        raise ValueError("Finalized generation report changed; cannot resume")
    report = _read_json(report_path)
    actual = tree_checksums(path, exclude=("generation-report.json",))
    if actual != report["files"]:
        raise ValueError("Finalized generation payload changed; cannot resume")
    return report


def generate_dataset(
    config: GenerationConfig, output, *, base_dir=None, resume=False, pipeline=None
):
    """Generate one view/episode per atomic shard, then publish a collection.

    Resume verifies finalized payloads and source bytes. A unit with no labels
    is reported explicitly; an exception fails the run and leaves no completed
    collection. Separate process shards use independent output directories.
    """
    from . import create_pipeline

    if isinstance(config, dict):
        config = GenerationConfig.from_mapping(config)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Generation output is nonempty; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        adapter = pipeline or create_pipeline(config, base_dir=Path(base_dir or ".").resolve())
        try:
            return _run(config, output, adapter, resume=resume)
        finally:
            adapter.close()


def _run(config, output, adapter, *, resume):
    units = sorted(adapter.units(), key=lambda unit: unit.key)
    keys = [unit.key for unit in units]
    if any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError("Generation unit keys must be nonempty strings")
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate generation unit identities")
    if not units:
        raise ValueError("No generation units matched the source and split")
    limited = units[: config.limit_units] if config.limit_units is not None else units
    selected = limited[config.shard_index :: config.num_shards]
    public_config = config.to_dict()
    public_config["pipeline"] = portable_metadata(config.pipeline)
    public_config["source"] = {
        key: value if key in {"id", "version"} else portable_metadata(value)
        for key, value in config.source.items()
    }
    environment = _environment()
    recipe = {
        "id": f"{config.method}_generation",
        "version": adapter.recipe_version,
        "runner_version": RUNNER_VERSION,
        "seed": config.seed,
        "seed_scheme": "sha256-base-method-unit-v1",
        "code_sha256": _code_fingerprint(),
        "parameters": {**public_config["pipeline"], "split": config.split},
        "environment": environment,
        "runtime": portable_metadata(adapter.provenance())
        if hasattr(adapter, "provenance")
        else {},
    }
    definition = {
        "config": public_config,
        "profile": adapter.profile,
        "recipe": recipe,
        "available_units": keys,
        "units": [unit.key for unit in selected],
    }
    identity = digest_json(definition)
    state_path = output / "generation.json"
    if state_path.exists():
        state = _read_json(state_path)
        if not resume or state.get("definition_sha256") != identity:
            raise ValueError(
                "Generation recipe, environment, source selection or code changed; use a new output revision"
            )
    else:
        leftovers = {path.name for path in output.iterdir()} - {".generation.lock"}
        if leftovers:
            raise ValueError("Output has no matching generation journal; refusing to overwrite it")
        state = {
            "format": "cofl_generation",
            "version": RUNNER_VERSION,
            "definition_sha256": identity,
            "definition": definition,
            "units": {},
        }
    state.update(status="running")
    state.pop("error", None)
    write_json(state_path, state)
    fingerprints = InputFingerprints()
    shard_paths = []
    staging_root = output / ".staging"
    staging_root.mkdir(exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)
    partial = bool(config.source.get("partial", False)) or len(selected) < len(units)
    source = {**public_config["source"], "partial": partial}

    def publish(complete):
        if shard_paths:
            publish_collection(
                output,
                shard_paths,
                profile=adapter.profile,
                dataset_id=config.dataset_id,
                revision=config.revision,
                source=source,
                recipe=recipe,
                complete=complete,
            )
            write_json(output / "collection.json", _read_json(output / "collection.json"))

    try:
        # A resumed run must not advertise completion while checking finalized shards.
        if (output / "collection.json").exists():
            manifest = _read_json(output / "collection.json")
            manifest["complete"] = False
            write_json(output / "collection.json", manifest)
        for index, unit in enumerate(selected):
            seed = unit_seed(config.seed, config.method, unit.key)
            inputs = fingerprints.describe(unit.inputs)
            unit_identity = {
                "key": unit.key,
                "seed": seed,
                "inputs": inputs,
                "definition_sha256": identity,
            }
            unit_sha = digest_json(unit_identity)
            name = digest_json(unit.key)[:24]
            destination = output / "shards" / name
            previous = state["units"].get(unit.key)
            if previous is not None and previous["unit_sha256"] != unit_sha:
                raise ValueError(
                    f"Source inputs changed for unit {unit.key}; use a new output revision"
                )
            if previous is not None and previous["status"] == "skipped":
                continue
            if destination.exists():
                report = _verify_shard(destination, (previous or {}).get("report_sha256"))
                if report["unit_sha256"] != unit_sha:
                    raise ValueError("Orphan shard does not match the current source/recipe")
                shard_paths.append(destination)
                state["units"][unit.key] = {
                    "status": "completed",
                    "unit_sha256": unit_sha,
                    "path": destination.relative_to(output).as_posix(),
                    "report_sha256": file_sha256(destination / "generation-report.json"),
                    "counts": report["counts"],
                    "summary": report["summary"],
                }
                write_json(state_path, state)
                continue
            if previous is not None:
                raise ValueError("A previously finalized generation shard is missing")
            if shutil.disk_usage(output).free < config.min_free_gb * 1024**3:
                raise OSError(f"Generation requires {config.min_free_gb:g} GiB free disk space")
            staging = staging_root / name
            if staging.exists():
                shutil.rmtree(staging)
            LOGGER.info("Generating unit %d/%d: %s", index + 1, len(selected), unit.key)
            writer = DatasetWriter(
                staging,
                profile=adapter.profile,
                dataset_id=config.dataset_id,
                revision=config.revision,
                source={**source, "unit": unit.key, "inputs": inputs},
                recipe={**recipe, "unit_seed": seed},
                **dict(adapter.writer_options),
            )
            with seeded_random(seed):
                summary = adapter.generate(unit, writer, seed=seed)
            if not isinstance(summary, dict):
                raise ValueError("Pipeline generate() must return a JSON summary mapping")
            if fingerprints.describe(unit.inputs) != inputs:
                raise ValueError("Source inputs changed while generating a unit")
            if not writer.annotations:
                if not summary.get("reason"):
                    raise ValueError("An empty generation unit must report its skip reason")
                shutil.rmtree(staging)
                state["units"][unit.key] = {
                    "status": "skipped",
                    "unit_sha256": unit_sha,
                    "inputs": inputs,
                    "summary": summary,
                }
                write_json(state_path, state)
                continue
            writer.finalize()
            # finalize() validates every payload before returning this manifest.
            counts = {
                "profile": writer.manifest["profile"],
                "schema_version": writer.manifest["schema_version"],
                **writer.manifest["counts"],
            }
            report = {
                "format": "cofl_generation_unit",
                "version": RUNNER_VERSION,
                "unit_sha256": unit_sha,
                **unit_identity,
                "counts": counts,
                "summary": summary,
                "files": tree_checksums(staging),
            }
            write_json(staging / "generation-report.json", report)
            staging.replace(destination)
            shard_paths.append(destination)
            state["units"][unit.key] = {
                "status": "completed",
                "unit_sha256": unit_sha,
                "path": destination.relative_to(output).as_posix(),
                "counts": counts,
                "report_sha256": file_sha256(destination / "generation-report.json"),
                "summary": summary,
            }
            write_json(state_path, state)
            publish(False)
        if not shard_paths and config.num_shards == 1:
            raise ValueError("All selected units were empty; see generation.json for skip reasons")
        fingerprints.verify_all()
        publish(True)
        state.update(
            status="completed",
            completed_units=len(shard_paths),
            skipped_units=len(selected) - len(shard_paths),
            empty_partition=not shard_paths,
        )
        write_json(state_path, state)
        return {
            "path": str(output),
            "status": "completed_empty_partition" if not shard_paths else state["status"],
            "profile": adapter.profile,
            "completed_units": state["completed_units"],
            "skipped_units": state["skipped_units"],
            "partial": partial,
            "counts": (
                _read_json(output / "collection.json")["counts"]
                if shard_paths
                else {"episodes": 0, "observations": 0, "annotations": 0}
            ),
        }
    except BaseException as error:
        state.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        write_json(state_path, state)
        raise
