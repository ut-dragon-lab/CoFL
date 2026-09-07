"""Stream joined native sample metadata without opening dense payload stores."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from ._io import file_sha256
from .collection import _local_child, manifest_path
from .profiles import PROFILES
from .storage import _validate_manifest


def _object(value, name):
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return result


def _first(records, key):
    for record in records:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _episode_context(episode, manifest):
    metadata = _object(episode["metadata_json"], "Episode metadata")
    records = [metadata]
    records.extend(
        metadata[key]
        for key in ("source_episode", "source_record")
        if isinstance(metadata.get(key), dict)
    )
    scene = _first(records, "scene_id")
    trajectory = _first(records, "trajectory_id")
    if trajectory is None:
        trajectory = _first(records, "path_id")
    source = manifest["source"]
    namespace = [manifest["profile"], source["id"], source["version"], episode["split"]]
    # Source trajectories can be replayed differently for different episodes;
    # their frame indices are not guaranteed to describe a shared time axis.
    group = [*namespace, "episode", episode["episode_id"]]
    return {
        "scene_id": scene,
        "trajectory_id": trajectory,
        "temporal_group": json.dumps(group, sort_keys=True, separators=(",", ":")),
    }


def _native_metadata(root, manifest, *, split, eligible_only):
    _validate_manifest(manifest)
    episodes = {}
    for batch in pq.ParquetFile(root / "episodes.parquet").iter_batches(
        batch_size=4096, columns=["episode_id", "split", "metadata_json"]
    ):
        for row in batch.to_pylist():
            if row["split"] == split:
                key = row["episode_id"]
                if key in episodes:
                    raise ValueError("Duplicate episode identity in native metadata")
                episodes[key] = _episode_context(row, manifest)
    if not episodes:
        return
    observations = {}
    for batch in pq.ParquetFile(root / "observations.parquet").iter_batches(
        batch_size=4096,
        columns=["observation_id", "episode_id", "frame_index", "timestamp"],
    ):
        for row in batch.to_pylist():
            if row["episode_id"] in episodes:
                key = row["observation_id"]
                if key in observations:
                    raise ValueError("Duplicate observation identity in native metadata")
                observations[key] = row
    for batch in pq.ParquetFile(root / "annotations.parquet").iter_batches(
        batch_size=4096,
        columns=[
            "annotation_id", "observation_id", "annotation_key", "instruction",
            "is_anchor", "split", "metadata_json",
        ],
    ):
        for row in batch.to_pylist():
            if row["split"] != split:
                continue
            observation = observations.get(row["observation_id"])
            if observation is None:
                raise ValueError("Annotation has no observation in its declared split")
            metadata = _object(row.pop("metadata_json"), "Annotation metadata")
            if eligible_only and not metadata.get("training_eligible", True):
                continue
            yield {
                **row,
                **observation,
                **episodes[observation["episode_id"]],
                "sample_id": row["annotation_id"],
            }


def _iter_metadata(root, *, split, eligible_only, expected_profile=None):
    manifest = json.loads(manifest_path(root).read_text(encoding="utf-8"))
    profile = manifest.get("profile")
    if expected_profile is not None and profile != expected_profile:
        raise ValueError("Collection shard profile differs from its parent")
    if manifest.get("format") != "cofl_dataset_collection":
        yield from _native_metadata(root, manifest, split=split, eligible_only=eligible_only)
        return
    if manifest.get("schema_version") != "1.0" or profile not in PROFILES:
        raise ValueError("Unsupported collection schema or profile")
    if not manifest.get("complete", False):
        raise ValueError("Dataset collection is incomplete")
    for entry in manifest["shards"]:
        child = _local_child(root, entry["path"])
        counts = entry["eligible_split_counts" if eligible_only else "split_counts"]
        expected = counts.get(split, 0)
        if not expected:
            continue
        if file_sha256(manifest_path(child)) != entry["manifest_sha256"]:
            raise ValueError("Shard manifest changed after collection publication")
        count = 0
        for row in _iter_metadata(
            child, split=split, eligible_only=eligible_only, expected_profile=profile
        ):
            count += 1
            yield row
        if count != expected:
            raise ValueError("Shard sample count differs from collection index")


def iter_sample_metadata(dataset_root, *, split, eligible_only=True):
    """Yield annotation identities joined to original frame and source metadata.

    Only Parquet columns and manifests are read. Memory is bounded by the
    selected episode/observation rows of one native shard and an annotation
    batch; dense Zarr stores are never opened. Multiple instructions retain
    distinct sample IDs and share a temporal group and original frame index.
    Temporal groups use native episodes, since source trajectory IDs alone do
    not guarantee aligned replay-step indices across episodes.
    """
    if not isinstance(split, str) or not split.strip():
        raise ValueError("An explicit nonempty split is required")
    if not isinstance(eligible_only, bool):
        raise ValueError("eligible_only must be a boolean")
    yield from _iter_metadata(Path(dataset_root), split=split, eligible_only=eligible_only)
