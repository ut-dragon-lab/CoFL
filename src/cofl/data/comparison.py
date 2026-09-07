"""Exact, identity-aware dataset regression checks without modifying inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zarr

from ._io import file_sha256
from .storage import CoFLDataset, validate_dataset


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _table_rows_digest(table, *, parse_geometry: bool = False) -> str:
    """Hash the canonical JSON row list using bounded Python batches."""
    digest = hashlib.sha256(b"[")
    separator = b""
    for batch in table.to_batches(max_chunksize=4096):
        for row in batch.to_pylist():
            if parse_geometry and "geometry_json" in row:
                row["geometry_json"] = json.loads(row["geometry_json"])
            digest.update(separator)
            digest.update(_json_bytes(row))
            separator = b","
    digest.update(b"]")
    return digest.hexdigest()


def _arrow_metadata(metadata) -> dict:
    return {key.hex(): value.hex() for key, value in sorted((metadata or {}).items())}


def _table_fingerprint(path: Path, identity: str, storage_columns: set[str]) -> dict:
    table = pq.read_table(path)
    schema = {
        "fields": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
                "metadata": _arrow_metadata(field.metadata),
            }
            for field in table.schema
        ],
        "metadata": _arrow_metadata(table.schema.metadata),
    }
    semantic = table.sort_by([(identity, "ascending")]).select(
        [name for name in table.column_names if name not in storage_columns]
    )
    return {
        "rows": table.num_rows,
        "schema_sha256": _digest(schema),
        "metadata_sha256": _table_rows_digest(semantic, parse_geometry=True),
        "storage_rows_sha256": _table_rows_digest(table),
    }


def _array_fingerprint(array, references: list[tuple[str, int]] | None) -> dict:
    """Read one logical image/field at a time; do not materialize a corpus."""
    header = {"shape": list(array.shape), "dtype": np.dtype(array.dtype).str}
    digest = hashlib.sha256(_json_bytes(header))

    def update(value):
        values = np.asarray(value)
        if values.dtype.hasobject:
            for item in values.flat:
                if not isinstance(item, bytes):
                    raise ValueError("Object payloads must contain encoded image bytes")
                digest.update(len(item).to_bytes(8, "big"))
                digest.update(item)
        else:
            digest.update(values.tobytes(order="C"))

    if references is None:
        if array.ndim == 0:
            update(array[()])
        else:
            for index in range(array.shape[0]):
                update(array[index])
    else:
        for identity, index in sorted(references):
            identity_bytes = identity.encode("utf-8")
            digest.update(len(identity_bytes).to_bytes(8, "big"))
            digest.update(identity_bytes)
            update(array[index])
    return {**header, "elements": int(array.size), "value_sha256": digest.hexdigest()}


def dataset_fingerprint(path: str | Path) -> dict[str, Any]:
    """Hash validated native data both semantically and as stored files.

    Semantic equality is exact: no floating-point tolerance is applied.
    Stable IDs determine dense payload order. Physical array offsets,
    Parquet row order, compression, and file encoding do not change the
    semantic digest, but do affect the independent byte digest.
    """
    path = Path(path).resolve()
    if (path / "collection.json").is_file():
        return _collection_fingerprint(path)
    counts = validate_dataset(path)
    dataset = CoFLDataset(path)
    tables = {
        "episodes": _table_fingerprint(path / "episodes.parquet", "episode_id", set()),
        "observations": _table_fingerprint(
            path / "observations.parquet", "observation_id", {"image_index", "depth_index"}
        ),
        "annotations": _table_fingerprint(
            path / "annotations.parquet", "annotation_id", {"field_index"}
        ),
    }
    references = {
        "images": [(row["observation_id"], row["image_index"]) for row in dataset.observations],
        "depths": [
            (row["observation_id"], row["depth_index"])
            for row in dataset.observations
            if row["depth_index"] is not None
        ],
        "depth_masks": [
            (row["observation_id"], row["depth_index"])
            for row in dataset.observations
            if row["depth_index"] is not None
        ],
        "fields": [(row["annotation_id"], row["field_index"]) for row in dataset.annotations],
        "masks": [(row["annotation_id"], row["field_index"]) for row in dataset.annotations],
    }
    arrays = {}
    attributes = {"/": dict(dataset.arrays.attrs)}

    def inspect(name, obj):
        attributes[name] = dict(obj.attrs)
        if isinstance(obj, zarr.Array):
            ref_name = (
                "images"
                if name.startswith("observation_extras/")
                else "fields"
                if name.startswith("annotation_extras/")
                else name
            )
            arrays[name] = _array_fingerprint(obj, references.get(ref_name))

    dataset.arrays.visititems(inspect)
    semantic = {
        "manifest_sha256": _digest(dataset.manifest),
        "tables": {
            key: {name: value for name, value in table.items() if name != "storage_rows_sha256"}
            for key, table in tables.items()
        },
        "arrays": arrays,
        "attributes_sha256": _digest(attributes),
    }
    files = {
        str(filename.relative_to(path)): file_sha256(filename)
        for filename in sorted(path.rglob("*"))
        if filename.is_file()
    }
    return {
        "path": str(path),
        "profile": dataset.profile,
        "revision": dataset.manifest["revision"],
        "source": dataset.manifest["source"],
        "recipe": dataset.manifest["recipe"],
        "counts": {key: counts[key] for key in ("episodes", "observations", "annotations")},
        "manifest_sha256": semantic["manifest_sha256"],
        "tables": tables,
        "arrays": arrays,
        "attributes_sha256": semantic["attributes_sha256"],
        "semantic_sha256": _digest(semantic),
        "sample_order_sha256": _digest([row["annotation_id"] for row in dataset.annotations]),
        "file_count": len(files),
        "files": files,
        "byte_sha256": _digest(files),
    }


def _collection_fingerprint(path: Path) -> dict[str, Any]:
    """Compare collections recursively, including the order of native shards.

    Equality is defined for the same shard layout. Different physical layouts
    require a separate identity join; no approximate equality is inferred.
    """
    from .collection import DatasetCollection, _local_child, _read_manifest

    dataset = DatasetCollection(path)
    children = []
    files = {}
    for entry in dataset.manifest["shards"]:
        child = _local_child(path, entry["path"])
        filename, _ = _read_manifest(child)
        if file_sha256(filename) != entry["manifest_sha256"]:
            raise ValueError("Collection shard manifest changed")
        result = dataset_fingerprint(child)
        if result["profile"] != dataset.profile or result["counts"] != entry["counts"]:
            raise ValueError("Collection shard profile or counts differ from its index")
        children.append((entry["path"], result))
        files.update({f"{entry['path']}/{name}": value for name, value in result["files"].items()})
    totals = {
        name: sum(result["counts"][name] for _, result in children)
        for name in ("episodes", "observations", "annotations")
    }
    if totals != dataset.manifest["counts"]:
        raise ValueError("Collection totals differ from its shards")
    tables = {}
    for name in ("episodes", "observations", "annotations"):
        tables[name] = {"rows": totals[name]}
        for key in ("schema_sha256", "metadata_sha256", "storage_rows_sha256"):
            tables[name][key] = _digest(
                [(relative, result["tables"][name][key]) for relative, result in children]
            )
    arrays = {
        f"{relative}/{name}": value
        for relative, result in children
        for name, value in result["arrays"].items()
    }
    manifest = {
        **dataset.manifest,
        "shards": [
            {key: value for key, value in entry.items() if key != "manifest_sha256"}
            for entry in dataset.manifest["shards"]
        ],
    }
    manifest_sha = _digest(manifest)
    attributes_sha = _digest(
        [(relative, result["attributes_sha256"]) for relative, result in children]
    )
    semantic = {
        "manifest_sha256": manifest_sha,
        "children": [(relative, result["semantic_sha256"]) for relative, result in children],
    }
    for filename in sorted(path.rglob("*")):
        relative = filename.relative_to(path).as_posix()
        if filename.is_file() and relative not in files:
            files[relative] = file_sha256(filename)
    return {
        "path": str(path),
        "profile": dataset.profile,
        "revision": manifest["revision"],
        "source": manifest["source"],
        "recipe": manifest["recipe"],
        "counts": totals,
        "manifest_sha256": manifest_sha,
        "tables": tables,
        "arrays": arrays,
        "attributes_sha256": attributes_sha,
        "semantic_sha256": _digest(semantic),
        "sample_order_sha256": _digest(
            [(relative, result["sample_order_sha256"]) for relative, result in children]
        ),
        "file_count": len(files),
        "files": files,
        "byte_sha256": _digest(files),
    }


def compare_datasets(left: str | Path, right: str | Path) -> dict[str, Any]:
    """Compare all metadata and dense payloads of two valid dataset revisions.

    Return a JSON-serializable report. Invalid data raises ``ValueError``;
    differing valid datasets return ``semantic_equal=False``. Inputs are
    always read-only. File timestamps and permissions are not compared.
    """
    left_result = dataset_fingerprint(left)
    right_result = dataset_fingerprint(right)
    differences = []
    if left_result["manifest_sha256"] != right_result["manifest_sha256"]:
        differences.append("manifest")
    for table in ("episodes", "observations", "annotations"):
        for key in ("rows", "schema_sha256", "metadata_sha256"):
            if left_result["tables"][table][key] != right_result["tables"][table][key]:
                differences.append(f"tables.{table}.{key}")
    for name in sorted(set(left_result["arrays"]) | set(right_result["arrays"])):
        if left_result["arrays"].get(name) != right_result["arrays"].get(name):
            differences.append(f"arrays.{name}")
    if left_result["attributes_sha256"] != right_result["attributes_sha256"]:
        differences.append("zarr_attributes")
    storage_differences = [
        name
        for name in sorted(set(left_result["files"]) | set(right_result["files"]))
        if left_result["files"].get(name) != right_result["files"].get(name)
    ]
    return {
        "comparison_version": "1",
        "semantic_equal": left_result["semantic_sha256"] == right_result["semantic_sha256"],
        "sample_order_equal": left_result["sample_order_sha256"]
        == right_result["sample_order_sha256"],
        "byte_equal": left_result["byte_sha256"] == right_result["byte_sha256"],
        "differences": differences,
        "storage_differences": storage_differences,
        "left": left_result,
        "right": right_result,
    }


def write_comparison_report(report: dict, path: str | Path) -> Path:
    """Write a report outside its compared datasets, preserving their bytes."""
    path = Path(path).resolve()
    for side in ("left", "right"):
        if path.is_relative_to(Path(report[side]["path"]).resolve()):
            raise ValueError("Comparison reports must be saved outside both input datasets")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path
