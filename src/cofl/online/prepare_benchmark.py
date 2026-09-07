"""Materialize a pinned R2R dataset using only the Python standard library.

Run ``python -m cofl.online.prepare_benchmark --help`` for the portable entry
point. Inputs are read-only; validation finishes before output is published.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


_MANIFEST_FIELDS = {
    "schema_version", "dataset_id", "created", "description", "split", "language_filter",
    "source_episode_count", "episode_count", "sources", "fingerprint", "scene_assets", "episodes",
}


def canonical_sha256(value):
    """Hash JSON content independently of compression, whitespace and key order."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        document = json.load(stream, object_pairs_hook=_unique_object)
    canonical_sha256(document)  # Reject NaN and Infinity, including in unused records.
    return document


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _validate_manifest(document):
    _mapping(document, "dataset manifest")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 2:
        raise ValueError(
            "Unsupported dataset schema_version "
            "(expected 2; legacy benchmark manifests are not supported)"
        )
    _nonempty(document.get("dataset_id"), "dataset_id")
    if document.get("split") != "val_unseen" or document.get("language_filter") != "en":
        raise ValueError("Dataset manifest requires split val_unseen and language_filter en")
    unknown = document.keys() - _MANIFEST_FIELDS
    if unknown:
        raise ValueError("Dataset manifest must contain only data metadata; unsupported fields: "
                         + ", ".join(sorted(unknown)))
    count = _integer(document.get("episode_count"), "episode_count", 1)
    source_count = _integer(document.get("source_episode_count"), "source_episode_count", 1)
    episodes = document.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != count or count > source_count:
        raise ValueError("episodes must match episode_count and source_episode_count")
    identifiers, indices = set(), set()
    for item in episodes:
        _mapping(item, "manifest episode")
        identifier = _nonempty(item.get("episode_id"), "episode_id")
        index = _integer(item.get("source_index"), "source_index")
        if identifier in identifiers or index in indices:
            raise ValueError("Duplicate manifest episode_id or source_index")
        if index >= source_count:
            raise ValueError("source_index is outside the source dataset")
        identifiers.add(identifier)
        indices.add(index)
        _nonempty(item.get("scene_id"), "scene_id")
        _integer(item.get("trajectory_id"), "trajectory_id")
        _integer(item.get("instruction_index"), "instruction_index")
        for key in ("episode_sha256", "gt_sha256", "fgr2r_sha256"):
            digest = item.get(key)
            if (
                not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"Invalid {key} for episode {identifier}")


def _xyz(value, name):
    if (
        not isinstance(value, list) or len(value) != 3
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in value)
    ):
        raise ValueError(f"{name} must contain three finite coordinates")


def normalize_fgr2r_row(row):
    """Normalize the two official Python-literal list columns before hashing."""
    normalized = dict(_mapping(row, "FGR2R row"))
    for name in ("new_instructions", "chunk_view"):
        value = normalized.get(name)
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"Invalid FGR2R {name} literal") from error
        if not isinstance(value, list):
            raise ValueError(f"FGR2R {name} must be a list or Python literal list")
        normalized[name] = value
    return normalized


def _normalize(text):
    return " ".join(str(text).split()).lower()


def _fgr2r_index(document):
    if not isinstance(document, list):
        raise ValueError("FGR2R JSON must contain a list")
    by_route, by_text = {}, {}
    for raw in document:
        row = normalize_fgr2r_row(raw)
        key = (str(row.get("scan", "")), str(row.get("path_id", "")))
        if key in by_route:
            raise ValueError(f"Duplicate FGR2R scene/path_id: {key}")
        by_route[key] = row
        instructions = row.get("instructions")
        if not isinstance(instructions, list) or not all(isinstance(t, str) for t in instructions):
            raise ValueError("FGR2R instructions must contain strings")
        for index, text in enumerate(instructions):
            by_text.setdefault((key[0], _normalize(text)), []).append((row, index))
    return by_route, by_text


def _check_annotation(row, index, identifier):
    path = row.get("path")
    if not isinstance(path, list) or len(path) < 2:
        raise ValueError(f"FGR2R path is invalid for episode {identifier}")
    try:
        chunks, spans = row["new_instructions"][index], row["chunk_view"][index]
    except IndexError as error:
        raise ValueError(f"FGR2R sub-instructions missing for episode {identifier}") from error
    if not isinstance(chunks, list) or not chunks or not isinstance(spans, list):
        raise ValueError(f"Invalid FGR2R chunks/spans for episode {identifier}")
    if len(chunks) != len(spans):
        raise ValueError(f"FGR2R chunks/spans length mismatch for episode {identifier}")
    assigned_text = False
    for chunk, span in zip(chunks, spans):
        if not isinstance(chunk, (str, list)):
            raise ValueError(f"Invalid FGR2R chunk for episode {identifier}")
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError(f"Invalid FGR2R chunk_view span for episode {identifier}")
        for number in span:
            _integer(number, "FGR2R chunk_view index", 1)
        start, end = sorted(span)
        if end > len(path):
            raise ValueError(f"FGR2R chunk_view exceeds source path for episode {identifier}")
        text = chunk if isinstance(chunk, str) else " ".join(map(str, chunk))
        if start < end and text.strip():
            assigned_text = True
    if not assigned_text:
        raise ValueError(f"FGR2R cannot assign sub-instruction text for episode {identifier}")


def _write_json(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if path.suffix == ".gz":
        # No filename or time-dependent header: repeated preparation yields identical bytes.
        payload = gzip.compress(payload, mtime=0)
    path.write_bytes(payload)


def prepare(args):
    """Validate input identity and atomically create a new local dataset directory."""
    output = Path(args.output).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    paths = {}
    for name in ("dataset_json", "r2r", "gt", "fgr2r"):
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        if path == output or output.resolve() in path.resolve().parents:
            raise ValueError(f"Output must not contain an input file: {path}")
        paths[name] = path
    manifest = _read_json(paths["dataset_json"])
    _validate_manifest(manifest)
    dataset = _mapping(_read_json(paths["r2r"]), "R2R dataset")
    episodes = dataset.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != manifest["source_episode_count"]:
        raise ValueError("R2R episode count does not match source_episode_count")
    by_id = {}
    for index, episode in enumerate(episodes):
        _mapping(episode, "R2R episode")
        if "episode_id" not in episode:
            raise ValueError("R2R episode is missing episode_id")
        identifier = str(episode["episode_id"])
        if identifier in by_id:
            raise ValueError(f"Duplicate R2R episode_id: {identifier}")
        by_id[identifier] = (index, episode)
    ground_truth = _mapping(_read_json(paths["gt"]), "GT dataset")
    annotations = _read_json(paths["fgr2r"])
    by_route, by_text = _fgr2r_index(annotations)
    selected, selected_gt, selected_annotations, records, scenes = [], {}, {}, [], set()
    for prepared_index, item in enumerate(manifest["episodes"]):
        identifier = item["episode_id"]
        if identifier not in by_id:
            raise ValueError(f"Missing R2R episode_id: {identifier}")
        input_index, episode = by_id[identifier]
        if episode.get("scene_id") != item["scene_id"]:
            raise ValueError(f"Scene mismatch for episode {identifier}")
        if episode.get("trajectory_id") != item["trajectory_id"]:
            raise ValueError(f"Trajectory mismatch for episode {identifier}")
        if canonical_sha256(episode) != item["episode_sha256"]:
            raise ValueError(f"Episode content hash mismatch for episode {identifier}")
        instruction = episode.get("instruction")
        if isinstance(instruction, dict):
            language = instruction.get("language", "")
            if language and (not isinstance(language, str) or not language.startswith("en")):
                raise ValueError(f"Episode {identifier} would be excluded by language_filter")
            instruction = instruction.get("instruction_text")
        _nonempty(instruction, f"Instruction for episode {identifier}")
        gt_record = ground_truth.get(identifier)
        if not isinstance(gt_record, dict) or not isinstance(gt_record.get("locations"), list):
            raise ValueError(f"Missing GT locations for episode {identifier}")
        if not gt_record["locations"]:
            raise ValueError(f"Empty GT locations for episode {identifier}")
        for position in gt_record["locations"]:
            _xyz(position, f"GT location for episode {identifier}")
        if canonical_sha256(gt_record) != item["gt_sha256"]:
            raise ValueError(f"GT content hash mismatch for episode {identifier}")
        scene_path = Path(item["scene_id"])
        if scene_path.is_absolute() or ".." in scene_path.parts or scene_path.suffix != ".glb":
            raise ValueError(f"Scene must be a relative .glb path: {scene_path}")
        scene = scene_path.stem
        row = by_route.get((scene, str(item["trajectory_id"])))
        instruction_index = item["instruction_index"]
        if row is None or instruction_index >= len(row["instructions"]):
            raise ValueError(f"Missing FGR2R route/instruction for episode {identifier}")
        if _normalize(row["instructions"][instruction_index]) != _normalize(instruction):
            raise ValueError(f"FGR2R instruction text mismatch for episode {identifier}")
        candidates = by_text.get((scene, _normalize(instruction)), [])
        if len(candidates) != 1 or candidates[0] != (row, instruction_index):
            raise ValueError(f"Ambiguous FGR2R scene/instruction for episode {identifier}")
        if canonical_sha256(row) != item["fgr2r_sha256"]:
            raise ValueError(f"FGR2R content hash mismatch for episode {identifier}")
        _check_annotation(row, instruction_index, identifier)
        selected_annotations[(scene, str(item["trajectory_id"]))] = row
        scenes.add(scene_path)
        selected.append(episode)
        selected_gt[identifier] = gt_record
        records.append({"episode_id": identifier, "source_index": item["source_index"],
                        "input_index": input_index, "prepared_index": prepared_index})
    index_map = {
        "schema_version": 2,
        "dataset_id": manifest["dataset_id"],
        "source_index_to_prepared_index": {
            str(r["source_index"]): r["prepared_index"] for r in records
        },
        "input_index_to_prepared_index": {
            str(r["input_index"]): r["prepared_index"] for r in records
        },
        "episode_id_to_prepared_index": {r["episode_id"]: r["prepared_index"] for r in records},
        "episodes": records,
    }
    inputs = {}
    for name, document in (
        ("dataset_json", manifest), ("r2r", dataset), ("gt", ground_truth),
        ("fgr2r", [normalize_fgr2r_row(row) for row in annotations]),
    ):
        inputs[name] = {"path": str(paths[name]), "file_sha256": _file_sha256(paths[name]),
                        "content_sha256": canonical_sha256(document)}
    preparation = {
        "schema_version": 2,
        "dataset_id": manifest["dataset_id"],
        "dataset_sha256": canonical_sha256(manifest),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(selected),
        "source_episode_count": len(episodes),
        "scene_count": len(scenes),
        "fgr2r_row_count": len(selected_annotations),
        "inputs": inputs,
        "index_map": "episode_index_map.json",
    }
    payloads = {
        "selected_episodes.json.gz": {**dataset, "episodes": selected},
        "selected_gt.json.gz": selected_gt,
        "selected_fgr2r.json": list(selected_annotations.values()),
        "episode_index_map.json": index_map,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for filename, document in payloads.items():
            _write_json(temporary / filename, document)
        preparation["output_file_sha256"] = {
            name: _file_sha256(temporary / name) for name in payloads
        }
        _write_json(temporary / "preparation.json", preparation)
        if os.path.lexists(output):
            raise FileExistsError(f"Output already exists; choose a new directory: {output}")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return preparation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, help_text in (
        ("dataset-json", "Published schema-2 R2R dataset membership JSON"),
        ("r2r", "Original VLN-CE R2R val_unseen.json.gz"),
        ("gt", "Official VLN-CE R2R val_unseen_gt.json.gz"),
        ("fgr2r", "Fine-Grained R2R FGR2R_val_unseen.json"),
        ("output", "New destination directory; existing paths are never overwritten"),
    ):
        parser.add_argument("--" + name, type=Path, required=True, help=help_text)
    args = parser.parse_args(argv)
    try:
        result = prepare(args)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(f"Prepared {result['episode_count']} episodes: {Path(args.output).absolute()}")


if __name__ == "__main__":
    main()
