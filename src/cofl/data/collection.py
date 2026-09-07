"""Portable collections of native shards, opened lazily for large corpora."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import pyarrow.parquet as pq

from ._io import file_sha256
from .profiles import PROFILES
from .storage import CoFLDataset, validate_dataset


def manifest_path(root: str | Path) -> Path:
    """Select the collection index, or the manifest for a single native shard."""
    root = Path(root)
    collection = root / "collection.json"
    return collection if collection.is_file() else root / "manifest.json"


def _read_manifest(path):
    filename = manifest_path(path)
    return filename, json.loads(filename.read_text())


def _local_child(root, relative):
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Collection shards must use portable paths inside their collection")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Collection shard resolves outside its collection")
    return path


def _split_counts(path, manifest):
    if manifest.get("format") == "cofl_dataset_collection":
        return manifest["split_counts"], manifest["eligible_split_counts"]
    counts, eligible = {}, {}
    for batch in pq.ParquetFile(path / "annotations.parquet").iter_batches(
        columns=["split", "metadata_json"]
    ):
        for row in batch.to_pylist():
            split = row["split"]
            counts[split] = counts.get(split, 0) + 1
            if json.loads(row["metadata_json"]).get("training_eligible", True):
                eligible[split] = eligible.get(split, 0) + 1
    return counts, eligible


def publish_collection(
    output, shard_paths, *, profile, dataset_id, revision, source, recipe, complete=True
):
    """Publish an index only after all listed native shards have been finalized.

    Paths must be within output. Existing indexes can be atomically refreshed
    as dataset creation advances, with complete=False until its end.
    """
    output = Path(output).resolve()
    if profile not in PROFILES:
        raise ValueError("Unknown collection profile")
    previous = {}
    previous_path = output / "collection.json"
    if previous_path.is_file():
        old = json.loads(previous_path.read_text())
        if (
            old.get("profile") != profile
            or old.get("dataset_id") != dataset_id
            or old.get("revision") != revision
        ):
            raise ValueError("Cannot replace a collection with a different dataset identity")
        previous = {entry["path"]: entry for entry in old["shards"]}
    entries, splits, eligible = [], {}, {}
    any_partial = bool(source.get("partial", False))
    totals = dict(episodes=0, observations=0, annotations=0)
    seen = set()
    for path in shard_paths:
        path = Path(path).resolve()
        relative = path.relative_to(output).as_posix()
        _local_child(output, relative)
        if relative in seen:
            raise ValueError("Duplicate shard path")
        seen.add(relative)
        entry = previous.get(relative)
        if entry is not None and not complete:
            # Finalized shards are immutable. Reuse their counts while adding
            # shards, avoiding quadratic Parquet decoding during large builds.
            entry = dict(entry)
        else:
            filename, manifest = _read_manifest(path)
            if manifest["profile"] != profile:
                raise ValueError("A collection cannot mix different method profiles")
            if manifest.get("format") == "cofl_dataset_collection" and not manifest.get(
                "complete", False
            ):
                raise ValueError("Cannot publish an incomplete child collection")
            if entry is not None and file_sha256(filename) != entry["manifest_sha256"]:
                raise ValueError("Previously finalized shard changed during collection publication")
            if entry is None:
                split_counts, eligible_counts = _split_counts(path, manifest)
                entry = {
                    "path": relative,
                    "manifest_sha256": file_sha256(filename),
                    "counts": manifest["counts"],
                    "split_counts": split_counts,
                    "eligible_split_counts": eligible_counts,
                }
            else:
                entry = dict(entry)
            entry["partial"] = bool(manifest.get("source", {}).get("partial", False))
        any_partial = any_partial or entry.get("partial", False)
        for key in totals:
            totals[key] += entry["counts"][key]
        for counts, target in (
            (entry["split_counts"], splits),
            (entry["eligible_split_counts"], eligible),
        ):
            for split, count in counts.items():
                target[split] = target.get(split, 0) + count
        entries.append(entry)
    if not entries:
        raise ValueError("A collection requires at least one finalized shard")
    manifest = {
        "format": "cofl_dataset_collection",
        "schema_version": "1.0",
        "profile": profile,
        "dataset_id": dataset_id,
        "revision": revision,
        "source": {**source, "partial": any_partial},
        "recipe": recipe,
        "counts": totals,
        "split_counts": splits,
        "eligible_split_counts": eligible,
        "complete": bool(complete),
        "shards": entries,
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "collection.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "collection.json")
    return output


class DatasetCollection:
    """Separate bounded caches for Arrow metadata and decoded shard payloads."""

    def __init__(
        self,
        path,
        *,
        expected_profile=None,
        split=None,
        eligible_only=False,
        cache_size=2,
        metadata_cache_size=16,
        include_extras=True,
        include_field=True,
        include_depth=True,
        extra_keys=None,
    ):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "collection.json").read_text())
        if (
            self.manifest.get("format") != "cofl_dataset_collection"
            or self.manifest.get("schema_version") != "1.0"
        ):
            raise ValueError("Unsupported collection schema")
        self.profile = self.manifest["profile"]
        if self.profile not in PROFILES or (
            expected_profile is not None and expected_profile != self.profile
        ):
            raise ValueError("Collection profile does not match the requested method")
        if not self.manifest.get("complete", False):
            raise ValueError("Dataset collection is incomplete")
        self.split = split
        self.eligible_only = eligible_only
        self.include_extras = bool(include_extras)
        self.include_field = bool(include_field)
        self.include_depth = bool(include_depth)
        self.extra_keys = (
            None if extra_keys is None else {kind: tuple(keys) for kind, keys in extra_keys.items()}
        )
        self.cache_size = max(1, int(cache_size))
        self.metadata_cache_size = max(self.cache_size, int(metadata_cache_size))
        self._cache = OrderedDict()
        self._payload_shards = OrderedDict()
        self._entries, self._ends = [], []
        total = 0
        for entry in self.manifest["shards"]:
            _local_child(self.path, entry["path"])
            counts = entry["eligible_split_counts" if eligible_only else "split_counts"]
            length = sum(counts.values()) if split is None else counts.get(split, 0)
            if length:
                total += length
                self._entries.append(entry)
                self._ends.append(total)

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def _locate(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard = bisect_right(self._ends, index)
        entry = self._entries[shard]
        if shard not in self._cache:
            path = _local_child(self.path, entry["path"])
            filename, _ = _read_manifest(path)
            if file_sha256(filename) != entry["manifest_sha256"]:
                raise ValueError("Shard manifest changed after collection publication")
            self._cache[shard] = open_dataset(
                path,
                expected_profile=self.profile,
                split=self.split,
                eligible_only=self.eligible_only,
                include_extras=self.include_extras,
                include_field=self.include_field,
                include_depth=self.include_depth,
                extra_keys=self.extra_keys,
                cache_size=self.cache_size,
                metadata_cache_size=self.metadata_cache_size,
            )
            expected = self._ends[shard] - (self._ends[shard - 1] if shard else 0)
            if len(self._cache[shard]) != expected:
                raise ValueError("Shard sample count differs from collection index")
            if len(self._cache) > self.metadata_cache_size:
                removed, _ = self._cache.popitem(last=False)
                self._payload_shards.pop(removed, None)
        self._cache.move_to_end(shard)
        return shard, index - (self._ends[shard - 1] if shard else 0)

    def sample_id(self, index: int) -> str:
        """Read a selected annotation identity through nested shard metadata."""
        shard, local_index = self._locate(index)
        return self._cache[shard].sample_id(local_index)

    def __getitem__(self, index):
        shard, local_index = self._locate(index)
        self._payload_shards[shard] = None
        self._payload_shards.move_to_end(shard)
        while len(self._payload_shards) > self.cache_size:
            removed, _ = self._payload_shards.popitem(last=False)
            self._cache[removed].clear_payload_cache()
        return self._cache[shard][local_index]

    def clear_payload_cache(self):
        """Retain child metadata while dropping decoded arrays recursively."""
        for child in self._cache.values():
            child.clear_payload_cache()
        self._payload_shards.clear()


def open_dataset(
    path,
    *,
    expected_profile=None,
    split=None,
    eligible_only=False,
    include_extras=True,
    include_field=True,
    include_depth=True,
    extra_keys=None,
    cache_size=2,
    metadata_cache_size=16,
):
    """Open a native view, optionally projecting dense payloads before decoding.

    ``include_field=False`` omits fields and masks; ``include_depth=False`` omits
    depth and its validity mask. ``extra_keys`` selects optional arrays by record
    kind (``observation`` or ``annotation``); absent kinds select no arrays.
    ``include_extras=False`` skips all extras. Scalar/JSON metadata is retained.
    Collection caches retain metadata for 16 child shards and decoded data for
    two by default.
    """
    kwargs = dict(
        expected_profile=expected_profile,
        split=split,
        eligible_only=eligible_only,
        include_extras=include_extras,
        include_field=include_field,
        include_depth=include_depth,
        extra_keys=extra_keys,
    )
    if (Path(path) / "collection.json").is_file():
        return DatasetCollection(
            path, cache_size=cache_size, metadata_cache_size=metadata_cache_size, **kwargs
        )
    return CoFLDataset(path, **kwargs)


def validate_collection(path, *, expected_profile=None):
    dataset = DatasetCollection(path, expected_profile=expected_profile)
    for entry in dataset.manifest["shards"]:
        child = _local_child(dataset.path, entry["path"])
        filename, manifest = _read_manifest(child)
        if file_sha256(filename) != entry["manifest_sha256"]:
            raise ValueError("Shard manifest differs from its recorded hash")
        validator = (
            validate_collection
            if manifest.get("format") == "cofl_dataset_collection"
            else validate_dataset
        )
        result = validator(child, expected_profile=dataset.profile)
        if any(
            result[key] != entry["counts"][key]
            for key in ("episodes", "observations", "annotations")
        ):
            raise ValueError("Shard counts do not match collection")
    return {"profile": dataset.profile, "schema_version": "1.0", **dataset.manifest["counts"]}
