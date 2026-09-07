"""Prepare a fixed RxR dataset with matching Landmark-RxR annotations.

Only the Python standard library is required. Source files are read-only;
instruction.timed_instruction is omitted from prepared episode records.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .prepare_benchmark import (
        _file_sha256, _integer, _mapping, _nonempty, _read_json, _unique_object,
        _write_json, _xyz, canonical_sha256,
    )
else:
    from prepare_benchmark import (
        _file_sha256, _integer, _mapping, _nonempty, _read_json, _unique_object,
        _write_json, _xyz, canonical_sha256,
    )


_MANIFEST_FIELDS = {
    "schema_version", "dataset_type", "dataset_id", "created", "description", "split", "role",
    "language_filter", "source_episode_count", "source_record_count", "episode_count", "sources",
    "fingerprint", "quality_filter", "scene_assets", "episodes",
}
_EPISODE_FIELDS = {
    "episode_id", "source_index", "source_record_index", "scene_id", "trajectory_id",
    "instruction_id", "episode_sha256", "gt_sha256", "landmark_sha256",
}


def project_episode(episode):
    """Copy an episode while omitting only the unused native timed-instruction field."""
    result = dict(_mapping(episode, "RxR episode"))
    instruction = result.get("instruction")
    if isinstance(instruction, dict):
        result["instruction"] = {
            key: value for key, value in instruction.items() if key != "timed_instruction"
        }
    return result


def _read_unchecked_json(path):
    # Finiteness is checked after the RxR projection, or on selected Landmark rows.
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_object)


def _instruction_id(value):
    """Use the runtime's exact integer instruction ID matching, never episode/path aliases."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return str(int(value))
    return None


def _episode_instruction_id(episode):
    instruction = episode.get("instruction", {})
    identifier = instruction.get("instruction_id") if isinstance(instruction, dict) else None
    return _instruction_id(episode.get("instruction_id") if identifier is None else identifier)


def _validate_manifest(document):
    _mapping(document, "RxR dataset manifest")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("Unsupported RxR dataset schema_version (expected 1)")
    unknown = document.keys() - _MANIFEST_FIELDS
    if unknown:
        raise ValueError("RxR manifest must contain only data metadata; unsupported fields: "
                         + ", ".join(sorted(unknown)))
    if document.get("dataset_type") != "rxr_landmark":
        raise ValueError("RxR dataset_type must be rxr_landmark")
    _nonempty(document.get("dataset_id"), "dataset_id")
    if (
        document.get("split") != "val_unseen" or document.get("role") != "guide"
        or document.get("language_filter") != "en"
    ):
        raise ValueError("RxR manifest requires val_unseen, guide and language_filter en")
    count = _integer(document.get("episode_count"), "episode_count", 1)
    source_count = _integer(document.get("source_episode_count"), "source_episode_count", 1)
    record_count = _integer(document.get("source_record_count"), "source_record_count", 1)
    episodes = document.get("episodes")
    if (
        not isinstance(episodes, list) or len(episodes) != count
        or not count <= source_count <= record_count
    ):
        raise ValueError("episodes must match episode_count and source counts")
    identifiers, source_indices, record_indices = set(), set(), set()
    for item in episodes:
        _mapping(item, "RxR manifest episode")
        if item.keys() != _EPISODE_FIELDS:
            raise ValueError("RxR manifest episode has missing or unsupported fields")
        identifier = _nonempty(item["episode_id"], "episode_id")
        source_index = _integer(item["source_index"], "source_index")
        record_index = _integer(item["source_record_index"], "source_record_index")
        if (
            identifier in identifiers or source_index in source_indices
            or record_index in record_indices
        ):
            raise ValueError("Duplicate RxR manifest episode_id or source index")
        if source_index >= source_count or record_index >= record_count:
            raise ValueError("RxR manifest source index is outside the source dataset")
        identifiers.add(identifier)
        source_indices.add(source_index)
        record_indices.add(record_index)
        for key in ("scene_id", "trajectory_id", "instruction_id"):
            _nonempty(item[key], key)
        if _instruction_id(item["instruction_id"]) is None:
            raise ValueError(f"Invalid instruction_id for episode {identifier}")
        for key in ("episode_sha256", "gt_sha256", "landmark_sha256"):
            digest = item[key]
            if (
                not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"Invalid {key} for episode {identifier}")


def _landmark_index(annotations):
    if not isinstance(annotations, list):
        raise ValueError("Landmark-RxR JSON must contain a list")
    result = {}
    for row in annotations:
        _mapping(row, "Landmark-RxR row")
        identifier = _instruction_id(row.get("instruction_id"))
        if identifier is not None:
            result.setdefault(identifier, []).append(row)
    return result


def _validate_landmark(row, identifier):
    chunks = row.get("sub_instructions")
    if (
        not isinstance(chunks, list) or len(chunks) < 2
        or any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks)
    ):
        raise ValueError(
            f"Landmark-RxR requires at least two nonempty chunks for episode {identifier}"
        )
    path, subpaths = row.get("path"), row.get("sub_paths")
    if not isinstance(path, list) or len(path) < 2:
        raise ValueError(f"Invalid Landmark-RxR path for episode {identifier}")
    if not isinstance(subpaths, list) or len(subpaths) != len(chunks):
        raise ValueError(f"Landmark-RxR sub_paths count mismatch for episode {identifier}")
    nodes = set(map(str, path))
    for subpath in subpaths:
        if (
            not isinstance(subpath, list) or len(subpath) < 2
            or any(str(node) not in nodes for node in subpath)
        ):
            raise ValueError(f"Invalid Landmark-RxR sub_path for episode {identifier}")


def prepare(args):
    """Validate data identities and atomically publish five files into a new directory."""
    output = Path(args.output).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    paths = {}
    for name in ("dataset_json", "rxr", "gt", "landmark_rxr"):
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        if path == output or output.resolve() in path.parents:
            raise ValueError(f"Output must not contain an input file: {path}")
        paths[name] = path
    manifest = _read_json(paths["dataset_json"])
    _validate_manifest(manifest)
    raw_dataset = _mapping(_read_unchecked_json(paths["rxr"]), "RxR dataset")
    raw_episodes = raw_dataset.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != manifest["source_record_count"]:
        raise ValueError("RxR raw episode count does not match source_record_count")
    projected_dataset = {**raw_dataset, "episodes": [project_episode(row) for row in raw_episodes]}
    # Only the omitted timed field may contain nonfinite values.
    canonical_sha256(projected_dataset)
    raw_ids, english = set(), {}
    for record_index, episode in enumerate(projected_dataset["episodes"]):
        if "episode_id" not in episode:
            raise ValueError("RxR episode is missing episode_id")
        identifier = str(episode["episode_id"])
        if identifier in raw_ids:
            raise ValueError(f"Duplicate RxR episode_id: {identifier}")
        raw_ids.add(identifier)
        instruction = _mapping(episode.get("instruction"), f"RxR instruction for {identifier}")
        language = instruction.get("language", "")
        if not isinstance(language, str):
            raise ValueError(f"Invalid RxR instruction language for episode {identifier}")
        if language.startswith(manifest["language_filter"]):
            english[identifier] = (len(english), record_index, episode)
    if len(english) != manifest["source_episode_count"]:
        raise ValueError("RxR English episode count does not match source_episode_count")
    ground_truth = _mapping(_read_json(paths["gt"]), "RxR GT dataset")
    annotations = _read_unchecked_json(paths["landmark_rxr"])
    by_instruction = _landmark_index(annotations)
    selected, selected_gt, selected_annotations, records, scenes = [], {}, {}, [], set()
    for prepared_index, item in enumerate(manifest["episodes"]):
        identifier = item["episode_id"]
        if identifier not in english:
            raise ValueError(f"Missing English RxR episode_id: {identifier}")
        input_index, input_record_index, episode = english[identifier]
        if episode.get("scene_id") != item["scene_id"]:
            raise ValueError(f"Scene mismatch for episode {identifier}")
        if str(episode.get("trajectory_id")) != item["trajectory_id"]:
            raise ValueError(f"Trajectory mismatch for episode {identifier}")
        instruction_id = _instruction_id(item["instruction_id"])
        if _episode_instruction_id(episode) != instruction_id:
            raise ValueError(f"Instruction ID mismatch for episode {identifier}")
        _nonempty(episode["instruction"].get("instruction_text"), f"Instruction for {identifier}")
        if canonical_sha256(episode) != item["episode_sha256"]:
            raise ValueError(f"Projected episode content hash mismatch for episode {identifier}")
        gt_record = ground_truth.get(identifier)
        if not isinstance(gt_record, dict) or not isinstance(gt_record.get("locations"), list):
            raise ValueError(f"Missing GT locations for episode {identifier}")
        if not gt_record["locations"]:
            raise ValueError(f"Empty GT locations for episode {identifier}")
        for position in gt_record["locations"]:
            _xyz(position, f"GT location for episode {identifier}")
        if canonical_sha256(gt_record) != item["gt_sha256"]:
            raise ValueError(f"GT content hash mismatch for episode {identifier}")
        matches = by_instruction.get(instruction_id, [])
        if len(matches) != 1:
            reason = "Ambiguous" if matches else "Missing"
            raise ValueError(f"{reason} Landmark-RxR instruction_id for episode {identifier}")
        row = matches[0]
        scene = Path(item["scene_id"].replace("\\", "/")).stem
        if str(row.get("scan") or "").strip() != scene:
            raise ValueError(f"Landmark-RxR scene mismatch for episode {identifier}")
        if str(row.get("path_id")) != item["trajectory_id"]:
            raise ValueError(f"Landmark-RxR trajectory mismatch for episode {identifier}")
        if canonical_sha256(row) != item["landmark_sha256"]:
            raise ValueError(f"Landmark-RxR content hash mismatch for episode {identifier}")
        _validate_landmark(row, identifier)
        selected.append(episode)
        selected_gt[identifier] = gt_record
        selected_annotations[instruction_id] = row
        scenes.add(scene)
        records.append({
            "episode_id": identifier,
            "source_index": item["source_index"],
            "source_record_index": item["source_record_index"],
            "input_index": input_index,
            "input_record_index": input_record_index,
            "prepared_index": prepared_index,
        })
    index_map = {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        **{name + "_to_prepared_index": {
            str(record[name]): record["prepared_index"] for record in records
        } for name in ("source_index", "source_record_index", "input_index", "input_record_index")},
        "episode_id_to_prepared_index": {
            row["episode_id"]: row["prepared_index"] for row in records
        },
        "episodes": records,
    }
    selected_rows = list(selected_annotations.values())
    inputs = {}
    for name, document in (
        ("dataset_json", manifest), ("rxr", projected_dataset), ("gt", ground_truth),
    ):
        inputs[name] = {
            "path": str(paths[name]), "file_sha256": _file_sha256(paths[name]),
            "content_sha256": canonical_sha256(document),
        }
    inputs["rxr"]["content_sha256_scope"] = "dataset with instruction.timed_instruction omitted"
    inputs["landmark_rxr"] = {
        "path": str(paths["landmark_rxr"]), "file_sha256": _file_sha256(paths["landmark_rxr"]),
        "selected_content_sha256": canonical_sha256(selected_rows),
        "source_record_count": len(annotations), "selected_record_count": len(selected_rows),
    }
    preparation = {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "dataset_sha256": canonical_sha256(manifest),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(selected),
        "source_episode_count": len(english),
        "source_record_count": len(raw_episodes),
        "scene_count": len(scenes),
        "landmark_row_count": len(selected_rows),
        "episode_projection": "omit instruction.timed_instruction",
        "inputs": inputs,
        "index_map": "episode_index_map.json",
    }
    payloads = {
        "selected_episodes.json.gz": {**projected_dataset, "episodes": selected},
        "selected_gt.json.gz": selected_gt,
        "selected_landmark_rxr.json": selected_rows,
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
        ("dataset-json", "Published RxR-Landmark dataset membership JSON"),
        ("rxr", "Original RxR-CE val_unseen_guide.json.gz including all languages"),
        ("gt", "Official RxR-CE val_unseen_guide_gt.json.gz"),
        ("landmark-rxr", "Original LandmarkRxR_val_unseen.json"),
        ("output", "New destination directory; existing paths are never overwritten"),
    ):
        parser.add_argument("--" + name, required=True, type=Path, help=help_text)
    args = parser.parse_args(argv)
    try:
        result = prepare(args)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(f"Prepared {result['episode_count']} RxR episodes: {Path(args.output).absolute()}")


if __name__ == "__main__":
    main()
