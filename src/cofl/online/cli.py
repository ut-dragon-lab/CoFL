"""Online evaluation YAML settings with explicit command-line overrides."""

import argparse
import json
import math
from pathlib import Path

import yaml

from .dynamics import RunParameters

_DEFAULTS = {
    "benchmark": None,
    "checkpoint": None,
    "output": None,
    "episode": None,
    "start": 0,
    "stride": 1,
    "count": None,
    "instruction_mode": None,
    "device": "cpu",
    "resume": False,
    "group_by_scene": True,
    "reuse_scene": True,
    "workers": 1,
    "worker_threads": 1,
    "inference_batch_size": None,
    "batch_wait_ms": 2.0,
    "inference_backend": "auto",
    "parameters": None,
    "policy": None,
    "policy_config": None,
}


def _load_config(path):
    path = path.expanduser().resolve()
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise TypeError("Evaluation config must contain a mapping")
    unknown = document.keys() - _DEFAULTS.keys()
    if unknown:
        raise ValueError("Unknown evaluation config keys: " + ", ".join(map(str, unknown)))
    for key in ("benchmark", "checkpoint", "output", "policy_config"):
        if key in document:
            value = document[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(key + " must be a nonempty path string")
            document[key] = (path.parent / Path(value).expanduser()).resolve()
    spec = document.get("policy")
    if isinstance(spec, str) and ":" in spec:
        source, factory = spec.rsplit(":", 1)
        if source.endswith(".py") or "/" in source:
            document["policy"] = (
                str((path.parent / Path(source).expanduser()).resolve()) + ":" + factory
            )
    parameters = document.get("parameters")
    if isinstance(parameters, str):
        if not parameters.strip():
            raise ValueError("parameters must be a mapping or a nonempty JSON path")
        document["parameters"] = (path.parent / Path(parameters).expanduser()).resolve()
    elif parameters is not None and not isinstance(parameters, dict):
        raise ValueError("parameters must be a mapping or a JSON path")
    return document


def _validate(values):
    policy = values["policy"]
    if policy is not None:
        if not isinstance(policy, str) or ":" not in policy:
            raise ValueError("policy must be a module:factory or path.py:factory")
        source, factory = policy.rsplit(":", 1)
        if not source.strip() or not factory.isidentifier():
            raise ValueError("policy must name a module/path and a factory function")
        if values["workers"] != 1 or values["inference_batch_size"] not in (None, 1):
            raise ValueError(
                "External policies currently require workers=1 and inference_batch_size=1"
            )
        if values["checkpoint"] is not None:
            raise ValueError("External policy checkpoints belong in policy_config, not checkpoint")
    elif values["policy_config"] is not None:
        raise ValueError("policy_config requires an external policy")
    for key in ("start", "stride", "count", "workers", "worker_threads", "inference_batch_size"):
        value = values[key]
        if key in ("count", "inference_batch_size") and value is None:
            continue
        minimum = 0 if key == "start" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(key + f" must be an integer >= {minimum}")
    episode = values["episode"]
    if episode is not None and (
        not isinstance(episode, list)
        or not episode
        or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in episode)
    ):
        raise ValueError("episode must be a nonempty list of nonnegative integers")
    if values["instruction_mode"] not in (None, "oracle", "full_instruction"):
        raise ValueError("instruction_mode must be oracle or full_instruction")
    if not isinstance(values["device"], str) or not values["device"].strip():
        raise ValueError("device must be a nonempty string")
    for key in ("resume", "group_by_scene", "reuse_scene"):
        if not isinstance(values[key], bool):
            raise TypeError(key + " must be a boolean")
    if values["workers"] > 1 and not values["group_by_scene"]:
        raise ValueError("workers > 1 requires group_by_scene: true")
    waiting = values["batch_wait_ms"]
    if (
        isinstance(waiting, bool)
        or not isinstance(waiting, (int, float))
        or not math.isfinite(waiting)
        or waiting < 0
    ):
        raise ValueError("batch_wait_ms must be finite and nonnegative")
    if values["inference_backend"] not in ("auto", "eager", "cuda_graph"):
        raise ValueError("inference_backend must be auto, eager or cuda_graph")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate native CoFL-S on fixed VLN episodes",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config", type=Path, help="Evaluation YAML; CLI options override its values"
    )
    parser.add_argument("--benchmark", type=Path, help="Environment/benchmark YAML")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--policy", help="External policy factory: module:function or path.py:function"
    )
    parser.add_argument(
        "--policy-config", type=Path, help="External policy YAML/JSON configuration"
    )
    parser.add_argument(
        "--episode", type=int, action="append", help="Repeat for specific dataset indices"
    )
    parser.add_argument("--start", type=int, help="First dataset index (default: 0)")
    parser.add_argument("--stride", type=int, help="Dataset index stride (default: 1)")
    parser.add_argument(
        "--count", type=int, help="Limit the start/stride selection; default all episodes"
    )
    parser.add_argument("--instruction-mode", choices=("oracle", "full_instruction"))
    parser.add_argument("--device", help="Policy device (default: cpu)")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        help="Parallel environment processes sharing one batched policy; prefer distinct scenes (default: 1)",
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        help="CPU operator threads per parallel worker; leaves serial execution unchanged (default: 1)",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        help="Maximum shared-model batch size; default the effective environment-worker count",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        help="Maximum queue collection window for a partial inference batch (default: 2 ms)",
    )
    parser.add_argument(
        "--group-by-scene",
        action=argparse.BooleanOptionalAction,
        help="Prefer scene reuse and distinct scenes across workers, retaining original indices (default: true)",
    )
    parser.add_argument(
        "--reuse-scene",
        action=argparse.BooleanOptionalAction,
        help="Reuse the Habitat worker and scene between compatible episodes (default: true)",
    )
    parser.add_argument(
        "--inference-backend",
        choices=("auto", "eager", "cuda_graph"),
        help="Dense-grid execution; auto/eager use batched queries (cuda_graph is a legacy alias)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        help="Resume only an identical saved protocol (default: false)",
    )
    parser.add_argument(
        "--parameters", type=Path, help="JSON RunParameters mapping; replaces config parameters"
    )
    overrides = vars(parser.parse_args(argv))
    config_path = overrides.pop("config", None)
    try:
        configured = _load_config(config_path) if config_path is not None else {}
        _validate({**_DEFAULTS, **configured})
        values = {**_DEFAULTS, **configured, **overrides}
        _validate(values)
        required = (
            ("benchmark", "output") if values["policy"] else ("benchmark", "checkpoint", "output")
        )
        missing = [key for key in required if values[key] is None]
        if missing:
            raise ValueError("Required via --config or CLI: " + ", ".join(missing))
        for key in ("benchmark", "checkpoint", "output", "policy_config"):
            if values[key] is not None:
                values[key] = values[key].expanduser()
        parameters = values["parameters"]
        if isinstance(parameters, Path):
            parameters = json.loads(parameters.expanduser().read_text())
            if not isinstance(parameters, dict):
                raise TypeError("parameters JSON must contain a mapping")
        values["parameters"] = RunParameters(**(parameters if parameters is not None else {}))
    except (ValueError, TypeError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
    return argparse.Namespace(config=config_path, **values)
