"""A small reference writer and reader for the native CoFLDataset v1 format.

Parquet stores identities and annotations. Zarr 2 stores chunked dense
arrays. This is the CoFL format, not an alternative implementation of the
LeRobot format; the optional export bridge uses LeRobot's own writer.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc, VLenBytes

from .profiles import PROFILES, validate_geometry

SCHEMA_VERSION = "1.0"


def _metadata_json(value: dict) -> str:
    if not isinstance(value, dict):
        raise ValueError("Record metadata must be a dictionary")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _decode_jpeg(value: bytes) -> np.ndarray:
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(value)) as image:
        if image.format != "JPEG":
            raise ValueError("Encoded image must be JPEG")
        return np.array(image.convert("RGB"), copy=True)


def _storage_float(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = value.astype(dtype)
    if dtype == np.dtype("float16") and not np.array_equal(result.astype(value.dtype), value):
        raise ValueError("float16 storage would change source values")
    return result


EPISODE_SCHEMA = pa.schema(
    [
        ("episode_id", pa.string()),
        ("episode_key", pa.string()),
        ("split", pa.string()),
        ("task", pa.string()),
        ("metadata_json", pa.string()),
    ]
)
OBSERVATION_SCHEMA = pa.schema(
    [
        ("observation_id", pa.string()),
        ("episode_id", pa.string()),
        ("frame_index", pa.int64()),
        ("timestamp", pa.float64()),
        ("image_index", pa.int64()),
        ("depth_index", pa.int64()),
        ("executed_action", pa.int64()),
        ("metadata_json", pa.string()),
    ]
)
ANNOTATION_SCHEMA = pa.schema(
    [
        ("annotation_id", pa.string()),
        ("observation_id", pa.string()),
        ("annotation_key", pa.string()),
        ("instruction", pa.string()),
        ("instruction_source", pa.string()),
        ("is_anchor", pa.bool_()),
        ("target_action", pa.int64()),
        ("field_index", pa.int64()),
        ("split", pa.string()),
        ("geometry_json", pa.string()),
        ("metadata_json", pa.string()),
    ]
)


def stable_id(kind: str, *parts: Any) -> str:
    """Make an identity from source identity and revision, never row order."""
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"{kind}_{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _action(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer action ID or None")
    if int(value) < 0:
        raise ValueError(f"{name} must be nonnegative; use None for unavailable labels")
    return int(value)


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("format") != "cofl_dataset" or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported CoFLDataset format or schema version")
    if manifest.get("profile") not in PROFILES:
        raise ValueError("Unknown dataset profile")
    for key in ("dataset_id", "revision"):
        _require_text(manifest.get(key), key)
    for key in ("source", "recipe"):
        value = manifest.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"manifest.{key} must include id and version")
        for required in ("id", "version"):
            _require_text(value.get(required), f"{key}.{required}")
    storage = manifest.get("storage")
    if (
        not isinstance(storage, dict)
        or storage.get("indices") != "parquet"
        or storage.get("dense") != "zarr_v2"
    ):
        raise ValueError("Native storage requires Parquet indices and Zarr v2 arrays")
    if storage.get("image_encoding") not in {"rgb", "jpeg"}:
        raise ValueError("Unsupported native image encoding")
    for key in ("field_dtype", "depth_dtype"):
        if storage.get(key) not in {"float16", "float32"}:
            raise ValueError(f"Unsupported native {key}")


def _read_table(path: Path, schema: pa.Schema) -> pa.Table:
    table = pq.read_table(path)
    for field in schema:
        if (
            field.name not in table.column_names
            or table.schema.field(field.name).type != field.type
        ):
            raise ValueError(f"{path.name} requires {field.name} with type {field.type}")
    if table["metadata_json"].null_count:
        raise ValueError(
            f"{path.name} metadata_json must contain objects; use '{{}}' for empty metadata"
        )
    return table


def _validate_payload(
    image: np.ndarray, field: np.ndarray, mask: np.ndarray, geometry: dict, profile: str
) -> None:
    validate_geometry(geometry, profile)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be uint8 (H, W, 3) RGB")
    if min(image.shape[:2]) < 1:
        raise ValueError("image dimensions must be positive")
    if field.shape != (2, *geometry["grid_shape"]):
        raise ValueError("field must have shape (2, *geometry.grid_shape)")
    if not np.issubdtype(field.dtype, np.floating) or not np.isfinite(field).all():
        raise ValueError("field must contain finite floating-point velocities")
    if mask.shape != tuple(geometry["grid_shape"]):
        raise ValueError("mask must have shape geometry.grid_shape")
    if mask.dtype != np.bool_:
        raise ValueError("mask must have boolean dtype; True means supervise this grid cell")
    if not mask.any():
        raise ValueError("annotation mask has no valid supervised cells")
    if profile == "ground_sector_v1":
        from cofl.fields import query_valid_mask

        if np.any(mask & ~query_valid_mask(profile, geometry)):
            raise ValueError("Ground supervision mask includes cells outside the policy sector")


class DatasetWriter:
    """Write a new immutable revision of a single-profile dataset.

    Dense arrays are appended immediately; metadata rows are buffered until
    ``finalize``. Existing nonempty output directories are never overwritten.
    Fields and images must each have a fixed shape within one revision.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        profile: str,
        dataset_id: str,
        revision: str,
        source: dict,
        recipe: dict,
        image_encoding: str = "rgb",
        field_dtype: str = "float32",
        depth_dtype: str = "float32",
        chunk_rows: int = 1,
        compression_shuffle: str = "bit",
        compression_level: int = 3,
    ):
        if image_encoding not in {"rgb", "jpeg"}:
            raise ValueError("image_encoding must be rgb or jpeg")
        if field_dtype not in {"float16", "float32"} or depth_dtype not in {"float16", "float32"}:
            raise ValueError("Dense floating-point storage must use float16 or float32")
        if chunk_rows < 1 or compression_shuffle not in {"bit", "byte"}:
            raise ValueError("Invalid chunk rows or compression shuffle")
        self.chunk_rows = int(chunk_rows)
        self.compressor = Blosc(
            cname="zstd",
            clevel=compression_level,
            shuffle=Blosc.SHUFFLE if compression_shuffle == "byte" else Blosc.BITSHUFFLE,
        )
        self._pending: dict[str, list] = {}
        self._append_counts: dict[str, int] = {}
        self.image_encoding = image_encoding
        self.field_dtype = np.dtype(field_dtype)
        self.depth_dtype = np.dtype(depth_dtype)
        self.path = Path(path)
        self.manifest = {
            "format": "cofl_dataset",
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "revision": revision,
            "profile": profile,
            "source": source,
            "recipe": recipe,
            "storage": {
                "indices": "parquet",
                "dense": "zarr_v2",
                "image_encoding": image_encoding,
                "field_dtype": field_dtype,
                "depth_dtype": depth_dtype,
            },
        }
        _validate_manifest(self.manifest)
        # Validate provenance JSON before allocating any files.
        self.manifest = json.loads(json.dumps(self.manifest, allow_nan=False))
        if self.path.exists() and any(self.path.iterdir()):
            raise FileExistsError(f"Dataset output must be absent or empty: {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        self.arrays = zarr.open_group(str(self.path / "arrays.zarr"), mode="w", zarr_version=2)
        self.arrays.attrs.update(format="cofl_dataset", schema_version=SCHEMA_VERSION)
        self.episodes: list[dict] = []
        self.observations: list[dict] = []
        self.annotations: list[dict] = []
        self._episodes: dict[str, dict] = {}
        self._observations: dict[str, dict] = {}
        self._annotations: set[str] = set()
        self._finalized = False
        self._extra_keys: dict[str, set[str]] = {}

    def _check_open(self) -> None:
        if self._finalized:
            raise RuntimeError("This dataset revision has already been finalized")

    def _id(self, kind: str, *parts: Any) -> str:
        return stable_id(kind, self.manifest["dataset_id"], self.manifest["revision"], *parts)

    def add_episode(
        self, episode_key: str, *, split: str, task: str, metadata: dict | None = None
    ) -> str:
        self._check_open()
        _require_text(episode_key, "episode_key")
        _require_text(split, "split")
        _require_text(task, "task")
        episode_id = self._id("episode", episode_key)
        if episode_id in self._episodes:
            raise ValueError(f"Duplicate episode key: {episode_key}")
        row = dict(episode_id=episode_id, episode_key=episode_key, split=split, task=task)
        row["metadata_json"] = _metadata_json({} if metadata is None else metadata)
        self.episodes.append(row)
        self._episodes[episode_id] = row
        return episode_id

    def _append(self, key: str, value: np.ndarray) -> int:
        if key not in self.arrays:
            self.arrays.create_dataset(
                key,
                shape=(0, *value.shape),
                chunks=(self.chunk_rows, *value.shape),
                dtype=value.dtype,
                compressor=self.compressor,
            )
        array = self.arrays[key]
        if np.dtype(array.dtype) != value.dtype:
            raise ValueError(f"{key} dtype changed within dataset: {value.dtype} != {array.dtype}")
        if array.shape[1:] != value.shape:
            raise ValueError(
                f"{key} shape changed within dataset: {value.shape} != {array.shape[1:]}"
            )
        index = self._append_counts.get(key, 0)
        self._append_counts[key] = index + 1
        pending = self._pending.setdefault(key, [])
        pending.append(np.array(value, copy=True))
        if len(pending) >= self.chunk_rows:
            array.append(np.stack(pending))
            pending.clear()
        return index

    def _append_extras(self, kind: str, index: int, extras: dict | None) -> None:
        extras = extras or {}
        keys = set(extras)
        if kind in self._extra_keys and keys != self._extra_keys[kind]:
            raise ValueError(f"{kind} extra keys must remain fixed within a shard")
        self._extra_keys[kind] = keys
        for name, value in extras.items():
            if not name.isidentifier():
                raise ValueError("Extra array names must be identifiers")
            value = np.asarray(value)
            if value.dtype.kind not in "biuf":
                raise ValueError("Extra arrays must have numeric or boolean dtype")
            actual = self._append(f"{kind}_extras/{name}", value)
            if actual != index:
                raise ValueError("Extra array index differs from its source record")

    def _append_image(self, image, encoded):
        if self.image_encoding == "rgb":
            return self._append("images", image)
        if "images" not in self.arrays:
            self.arrays.create_dataset(
                "images",
                shape=(0,),
                chunks=(64,),
                dtype=object,
                object_codec=VLenBytes(),
                compressor=None,
            )
            self.arrays["images"].attrs["decoded_shape"] = list(image.shape)
        array = self.arrays["images"]
        if list(image.shape) != list(array.attrs["decoded_shape"]):
            raise ValueError("Image shape changed within a shard")
        index = self._append_counts.get("images", 0)
        self._append_counts["images"] = index + 1
        pending = self._pending.setdefault("images", [])
        pending.append(encoded)
        if len(pending) >= 64:
            values = np.empty(len(pending), dtype=object)
            values[:] = pending
            array.append(values)
            pending.clear()
        return index

    def add_observation(
        self,
        episode_id: str,
        *,
        frame_index: int,
        image: np.ndarray,
        timestamp: float | None = None,
        executed_action: int | None = None,
        depth: np.ndarray | None = None,
        depth_valid: np.ndarray | None = None,
        metadata: dict | None = None,
        extras: dict | None = None,
    ) -> str:
        self._check_open()
        if episode_id not in self._episodes:
            raise ValueError("Observation references an unknown episode")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, (int, np.integer))
            or frame_index < 0
        ):
            raise ValueError("frame_index must be a nonnegative integer")
        if timestamp is not None and (not np.isfinite(timestamp) or timestamp < 0):
            raise ValueError("timestamp must be finite and nonnegative, or None")
        encoded = None
        if self.image_encoding == "jpeg":
            if not isinstance(image, bytes):
                raise ValueError("JPEG storage requires the original encoded bytes")
            encoded = image
            image = _decode_jpeg(image)
        image = np.asarray(image)
        if (
            image.ndim != 3
            or image.shape[-1] != 3
            or image.dtype != np.uint8
            or min(image.shape[:2]) < 1
        ):
            raise ValueError("image must be uint8 (H, W, 3) RGB")
        if depth is None and depth_valid is not None:
            raise ValueError("depth_valid requires depth")
        if depth is not None:
            depth = np.asarray(depth)
            if depth.shape != image.shape[:2] or not np.issubdtype(depth.dtype, np.floating):
                raise ValueError(
                    "depth must be floating-point metres with the RGB image's (H, W) shape"
                )
            if not np.isfinite(depth).all() or np.any(depth < 0):
                raise ValueError(
                    "depth must be finite and nonnegative; use zero for unavailable depth"
                )
            depth_valid = np.asarray(depth > 0 if depth_valid is None else depth_valid)
            if depth_valid.dtype != np.bool_ or depth_valid.shape != depth.shape:
                raise ValueError("depth_valid must be boolean with the depth image's shape")
            if np.any(depth_valid & (depth <= 0)):
                raise ValueError("Valid depth pixels must have positive metric depth")
        observation_id = self._id("observation", episode_id, int(frame_index))
        if observation_id in self._observations:
            raise ValueError("Duplicate observation for episode and frame_index")
        action = _action(executed_action, "executed_action")
        if (
            self.image_encoding == "rgb"
            and "images" in self.arrays
            and self.arrays["images"].shape[1:] != image.shape
        ):
            raise ValueError("images shape changed within dataset")
        if (
            depth is not None
            and "depths" in self.arrays
            and self.arrays["depths"].shape[1:] != depth.shape
        ):
            raise ValueError("depths shape changed within dataset")
        depth_index = None
        if depth is not None:
            depth_index = self._append("depths", _storage_float(depth, self.depth_dtype))
            self._append("depth_masks", depth_valid)
        row = dict(
            observation_id=observation_id,
            episode_id=episode_id,
            frame_index=int(frame_index),
            timestamp=timestamp,
            image_index=self._append_image(image, encoded),
            depth_index=depth_index,
            executed_action=action,
        )
        row["metadata_json"] = _metadata_json({} if metadata is None else metadata)
        self._append_extras("observation", row["image_index"], extras)
        self.observations.append(row)
        self._observations[observation_id] = row
        return observation_id

    def add_annotation(
        self,
        observation_id: str,
        *,
        annotation_key: str,
        instruction: str,
        field: np.ndarray,
        mask: np.ndarray,
        geometry: dict,
        target_action: int | None = None,
        is_anchor: bool = True,
        instruction_source: str = "source",
        metadata: dict | None = None,
        extras: dict | None = None,
    ) -> str:
        self._check_open()
        if observation_id not in self._observations:
            raise ValueError("Annotation references an unknown observation")
        for name, value in (
            ("annotation_key", annotation_key),
            ("instruction", instruction),
            ("instruction_source", instruction_source),
        ):
            _require_text(value, name)
        if not isinstance(is_anchor, bool):
            raise ValueError("is_anchor must be boolean")
        observation = self._observations[observation_id]
        episode = self._episodes[observation["episode_id"]]
        field, mask = np.asarray(field), np.asarray(mask)
        _validate_payload(
            np.empty((1, 1, 3), dtype=np.uint8), field, mask, geometry, self.manifest["profile"]
        )
        annotation_id = self._id(
            "annotation", observation_id, self.manifest["recipe"], annotation_key
        )
        if annotation_id in self._annotations:
            raise ValueError("Duplicate annotation_key for observation and recipe")
        target_action = _action(target_action, "target_action")
        geometry_json = json.dumps(geometry, sort_keys=True, allow_nan=False)
        # Check both append shapes before making either mutation.
        for key, value in (("fields", field), ("masks", mask)):
            if key in self.arrays and self.arrays[key].shape[1:] != value.shape:
                raise ValueError(f"{key} shape changed within dataset")
        field_index = self._append("fields", _storage_float(field, self.field_dtype))
        self._append("masks", mask)
        row = dict(
            annotation_id=annotation_id,
            observation_id=observation_id,
            annotation_key=annotation_key,
            instruction=instruction,
            instruction_source=instruction_source,
            is_anchor=is_anchor,
            target_action=target_action,
            field_index=field_index,
            split=episode["split"],
            geometry_json=geometry_json,
        )
        row["metadata_json"] = _metadata_json({} if metadata is None else metadata)
        self._append_extras("annotation", field_index, extras)
        self.annotations.append(row)
        self._annotations.add(annotation_id)
        return annotation_id

    def finalize(self) -> Path:
        """Flush tables and arrays, then fully validate the written dataset."""
        self._check_open()
        if not self.episodes or not self.observations or not self.annotations:
            raise ValueError("A dataset requires episodes, observations and annotations")
        for key, values in self._pending.items():
            if values:
                if key == "images" and self.image_encoding == "jpeg":
                    batch = np.empty(len(values), dtype=object)
                    batch[:] = values
                else:
                    batch = np.stack(values)
                self.arrays[key].append(batch)
                values.clear()
        self.manifest["counts"] = dict(
            episodes=len(self.episodes),
            observations=len(self.observations),
            annotations=len(self.annotations),
        )
        for filename, rows, schema in (
            ("episodes.parquet", self.episodes, EPISODE_SCHEMA),
            ("observations.parquet", self.observations, OBSERVATION_SCHEMA),
            ("annotations.parquet", self.annotations, ANNOTATION_SCHEMA),
        ):
            pq.write_table(
                pa.Table.from_pylist(rows, schema=schema), self.path / filename, compression="zstd"
            )
        (self.path / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8"
        )
        validate_dataset(self.path)
        self._finalized = True
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None and not self._finalized:
            self.finalize()


class _TableRows(Sequence):
    """Arrow-backed rows with the list-like read interface used by native APIs.

    Large JSON/string columns stay in Arrow buffers until a row is requested.
    Returned dictionaries are independent, so editing a row cannot mutate the
    dataset or a later read of the same row.
    """

    def __init__(self, table, indices=None):
        self.table = table
        self.indices = indices
        self._columns = dict(zip(table.column_names, table.columns))

    def __len__(self):
        return len(self.table) if self.indices is None else len(self.indices)

    def row_index(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return int(index if self.indices is None else self.indices[index])

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = np.arange(len(self.table)) if self.indices is None else self.indices
            return _TableRows(self.table, indices[index])
        row = self.row_index(index)
        return {name: column[row].as_py() for name, column in self._columns.items()}


class _RowLookup(Mapping):
    """Compact identity-to-row lookup; metadata dictionaries remain lazy."""

    def __init__(self, rows, key):
        self.rows = rows
        self.indices = {value: index for index, value in enumerate(rows.table[key].to_pylist())}

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __getitem__(self, key):
        return self.rows[self.indices[key]]


class CoFLDataset:
    """Read annotation samples, joining each to its shared observation.

    Returned arrays are NumPy arrays. ``field`` is (2, H, W), ``mask`` is
    (H, W), and ``image`` is uint8 (H, W, 3). Filtering samples never alters
    the stored observation sequence or turns alternatives into time steps.
    Dense projections are opt-in; the default reader returns every payload.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        expected_profile: str | None = None,
        split: str | None = None,
        eligible_only: bool = False,
        include_extras: bool = True,
        include_field: bool = True,
        include_depth: bool = True,
        extra_keys: Mapping[str, Sequence[str]] | None = None,
    ):
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"No finalized CoFLDataset manifest at {self.path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(self.manifest)
        self.profile = self.manifest["profile"]
        if expected_profile is not None and expected_profile != self.profile:
            raise ValueError(
                f"Dataset profile {self.profile!r} does not match expected profile {expected_profile!r}"
            )
        self.include_extras = bool(include_extras)
        self.include_field = bool(include_field)
        self.include_depth = bool(include_depth)
        self.extra_keys = (
            None if extra_keys is None else {kind: frozenset(keys) for kind, keys in extra_keys.items()}
        )
        self.episodes = _TableRows(
            _read_table(self.path / "episodes.parquet", EPISODE_SCHEMA)
        )
        self.observations = _TableRows(
            _read_table(self.path / "observations.parquet", OBSERVATION_SCHEMA)
        )
        self.annotations = _TableRows(
            _read_table(self.path / "annotations.parquet", ANNOTATION_SCHEMA)
        )
        self._episodes = _RowLookup(self.episodes, "episode_id")
        self._observations = _RowLookup(self.observations, "observation_id")
        if len(self._episodes) != len(self.episodes) or len(self._observations) != len(self.observations):
            raise ValueError("Duplicate episode or observation IDs")
        # Vectorized foreign-key joins inspect only identity and split columns.
        # Neither annotation JSON nor observation metadata is expanded here.
        observation_episodes = pc.index_in(
            self.observations.table["episode_id"], value_set=self.episodes.table["episode_id"]
        )
        annotation_observations = pc.index_in(
            self.annotations.table["observation_id"], value_set=self.observations.table["observation_id"],
        )
        if observation_episodes.null_count or annotation_observations.null_count:
            raise ValueError("Annotation references a missing observation or episode")
        self._observation_episode_indices = observation_episodes.to_numpy()
        self._annotation_observation_indices = annotation_observations.to_numpy()
        annotation_episodes = self._observation_episode_indices[
            self._annotation_observation_indices
        ]
        inherited_splits = pc.take(self.episodes.table["split"], pa.array(annotation_episodes))
        split_equal = pc.equal(self.annotations.table["split"], inherited_splits)
        if split_equal.null_count or (len(split_equal) and not pc.all(split_equal).as_py()):
            raise ValueError("Annotation split must inherit its source episode split")
        selected = np.ones(len(self.annotations), dtype=bool)
        if split is not None:
            selected &= pc.equal(self.annotations.table["split"], split).to_numpy()
        if eligible_only:
            # Decode eligibility only for the selected split, retaining a
            # compact index rather than expanded metadata dictionaries.
            metadata = self.annotations.table["metadata_json"]
            for index in np.flatnonzero(selected):
                selected[index] = bool(
                    json.loads(metadata[int(index)].as_py()).get("training_eligible", True)
                )
        self.samples = _TableRows(self.annotations.table, np.flatnonzero(selected))
        self.arrays = zarr.open_group(str(self.path / "arrays.zarr"), mode="r")
        if (
            self.arrays.attrs.get("format") != "cofl_dataset"
            or self.arrays.attrs.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError("Dense array metadata must match the native dataset schema")
        self._array_handles = {}
        self._extra_names = {}
        self.clear_payload_cache()

    def clear_payload_cache(self):
        """Release decoded chunks while retaining compact metadata and indices."""
        self._dense_cache = OrderedDict()
        self._dense_cache_bytes = 0
        self._image_cache = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def sample_id(self, index: int) -> str:
        """Read a selected annotation identity without decoding dense arrays."""
        return self.samples.table["annotation_id"][self.samples.row_index(index)].as_py()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        annotation_index = self.samples.row_index(index)
        observation_index = int(self._annotation_observation_indices[annotation_index])
        observation = self.observations[observation_index]
        episode = self.episodes[int(self._observation_episode_indices[observation_index])]
        geometry = json.loads(row["geometry_json"])
        validate_geometry(geometry, self.profile)
        result = {
            "sample_id": row["annotation_id"],
            "annotation_id": row["annotation_id"],
            "annotation_key": row["annotation_key"],
            "observation_id": row["observation_id"],
            "episode_id": episode["episode_id"],
            "frame_index": observation["frame_index"],
            "timestamp": observation["timestamp"],
            "profile": self.profile,
            "split": episode["split"],
            "image": self.read_image(observation["image_index"]),
            "instruction": row["instruction"],
            "instruction_source": row["instruction_source"],
            "is_anchor": row["is_anchor"],
            "geometry": geometry,
            "executed_action": observation["executed_action"],
            "target_action": row["target_action"],
            "task": episode["task"],
            "source": self.manifest["source"],
            "recipe": self.manifest["recipe"],
        }
        if self.include_field:
            result["field"] = np.asarray(self._read_dense("fields", row["field_index"]))
            result["mask"] = np.asarray(self._read_dense("masks", row["field_index"]))
        if self.include_depth and observation["depth_index"] is not None:
            result["depth"] = np.asarray(self._read_dense("depths", observation["depth_index"]))
            result["depth_valid"] = np.asarray(
                self._read_dense("depth_masks", observation["depth_index"])
            )
        for kind, record, dense_index in (
            ("episode", episode, None),
            ("observation", observation, observation["image_index"]),
            ("annotation", row, row["field_index"]),
        ):
            result[f"{kind}_metadata"] = json.loads(record["metadata_json"])
            group = f"{kind}_extras"
            if not self.include_extras:
                continue
            if kind not in self._extra_names:
                requested = None if self.extra_keys is None else self.extra_keys.get(kind, ())
                self._extra_names[kind] = (
                    tuple(
                        key
                        for key in self.arrays[group].array_keys()
                        if requested is None or key in requested
                    )
                    if (requested is None or requested) and group in self.arrays
                    else ()
                )
            if self._extra_names[kind]:
                result[group] = {
                    key: np.asarray(self._read_dense(f"{group}/{key}", dense_index))
                    for key in self._extra_names[kind]
                }
        return result

    def _read_dense(self, key: str, index: int):
        if key not in self._array_handles:
            self._array_handles[key] = self.arrays[key]
        array = self._array_handles[key]
        chunk = index // array.chunks[0]
        cache_key = (key, chunk)
        if cache_key not in self._dense_cache:
            start = chunk * array.chunks[0]
            values = np.asarray(array[start : min(start + array.chunks[0], array.shape[0])])
            values.setflags(write=False)
            size = values.nbytes
            if values.dtype.hasobject:
                size += sum(len(item) for item in values.flat)
            self._dense_cache[cache_key] = (values, size)
            self._dense_cache_bytes += size
            while self._dense_cache_bytes > 64 * 1024 * 1024 and len(self._dense_cache) > 1:
                _, (_, removed_size) = self._dense_cache.popitem(last=False)
                self._dense_cache_bytes -= removed_size
        self._dense_cache.move_to_end(cache_key)
        return self._dense_cache[cache_key][0][index % array.chunks[0]]

    def read_image(self, index: int) -> np.ndarray:
        if self.manifest["storage"]["image_encoding"] != "jpeg":
            return np.asarray(self._read_dense("images", index))
        if index not in self._image_cache:
            decoded = _decode_jpeg(bytes(self._read_dense("images", index)))
            decoded.setflags(write=False)
            self._image_cache[index] = decoded
            if len(self._image_cache) > 8:
                self._image_cache.popitem(last=False)
        self._image_cache.move_to_end(index)
        return self._image_cache[index]

    def episode_observations(self, episode_id: str) -> list[dict]:
        if episode_id not in self._episodes:
            raise KeyError(episode_id)
        episode_index = self._episodes.indices[episode_id]
        indices = np.flatnonzero(self._observation_episode_indices == episode_index)
        return sorted(
            (self.observations[int(index)] for index in indices), key=lambda row: row["frame_index"]
        )


def validate_dataset(path: str | Path, *, expected_profile: str | None = None) -> dict[str, Any]:
    """Validate identities, split inheritance, array bounds, geometry and labels."""
    dataset = CoFLDataset(path, expected_profile=expected_profile)
    counts = dict(
        episodes=len(dataset.episodes),
        observations=len(dataset.observations),
        annotations=len(dataset.annotations),
    )
    if dataset.manifest.get("counts") != counts or any(n == 0 for n in counts.values()):
        raise ValueError("Manifest counts do not match nonempty dataset tables")
    if len({row["annotation_id"] for row in dataset.annotations}) != counts["annotations"]:
        raise ValueError("Duplicate annotation IDs")
    if dataset.arrays["images"].shape[0] != counts["observations"]:
        raise ValueError("Image array count does not match observations")
    for kind, count in (
        ("observation", counts["observations"]),
        ("annotation", counts["annotations"]),
    ):
        group = f"{kind}_extras"
        if group in dataset.arrays:
            for name in dataset.arrays[group].array_keys():
                array = dataset.arrays[group][name]
                if array.shape[0] != count or np.dtype(array.dtype).kind not in "biuf":
                    raise ValueError(
                        f"{group}/{name} must contain one numeric entry per source record"
                    )
    for key in ("fields", "masks"):
        if dataset.arrays[key].shape[0] != counts["annotations"]:
            raise ValueError(f"{key} array count does not match annotations")
    for key, rows, count in (
        ("image_index", dataset.observations, counts["observations"]),
        ("field_index", dataset.annotations, counts["annotations"]),
    ):
        if sorted(row[key] for row in rows) != list(range(count)):
            raise ValueError(
                f"{key} must reference each dense row exactly once without out-of-bounds indices"
            )
    depth_indices = sorted(
        row["depth_index"] for row in dataset.observations if row["depth_index"] is not None
    )
    if depth_indices:
        if depth_indices != list(range(len(depth_indices))):
            raise ValueError("depth_index must reference each dense row exactly once")
        if any(
            dataset.arrays[key].shape[0] != len(depth_indices) for key in ("depths", "depth_masks")
        ):
            raise ValueError("Depth array counts do not match observations")
    for episode in dataset.episodes:
        expected_episode = stable_id(
            "episode",
            dataset.manifest["dataset_id"],
            dataset.manifest["revision"],
            episode["episode_key"],
        )
        if episode["episode_id"] != expected_episode:
            raise ValueError("Episode identity does not match dataset revision and source key")
        observations = dataset.episode_observations(episode["episode_id"])
        if not observations:
            raise ValueError("Episode has no observations")
        frames = [row["frame_index"] for row in observations]
        if any(frame < 0 for frame in frames) or len(set(frames)) != len(frames):
            raise ValueError("Episode frame indices must be unique and nonnegative")
        timestamps = [row["timestamp"] for row in observations]
        if any(t is not None for t in timestamps):
            if any(t is None or not np.isfinite(t) or t < 0 for t in timestamps):
                raise ValueError(
                    "An episode must have all finite timestamps or all timestamps absent"
                )
            if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
                raise ValueError("Episode timestamps must increase strictly")
        for observation in observations:
            _action(observation["executed_action"], "executed_action")
            expected = stable_id(
                "observation",
                dataset.manifest["dataset_id"],
                dataset.manifest["revision"],
                episode["episode_id"],
                observation["frame_index"],
            )
            if observation["observation_id"] != expected:
                raise ValueError(
                    "Observation identity does not match dataset revision and source frame"
                )
    for sample in dataset:
        _action(sample["target_action"], "target_action")
        _validate_payload(
            sample["image"], sample["field"], sample["mask"], sample["geometry"], dataset.profile
        )
        if "depth" in sample:
            depth, valid = sample["depth"], sample["depth_valid"]
            if (
                depth.shape != sample["image"].shape[:2]
                or valid.shape != depth.shape
                or valid.dtype != np.bool_
            ):
                raise ValueError("Depth shape or validity mask does not match observation")
            if not np.isfinite(depth).all() or np.any(depth < 0) or np.any(valid & (depth <= 0)):
                raise ValueError("Invalid metric depth values")
    for observation in dataset.observations:
        if observation["episode_id"] not in dataset._episodes:
            raise ValueError("Observation references a missing episode")
    for annotation in dataset.annotations:
        expected = stable_id(
            "annotation",
            dataset.manifest["dataset_id"],
            dataset.manifest["revision"],
            annotation["observation_id"],
            dataset.manifest["recipe"],
            annotation["annotation_key"],
        )
        if annotation["annotation_id"] != expected:
            raise ValueError("Annotation identity does not match observation and versioned recipe")
    return {"profile": dataset.profile, "schema_version": SCHEMA_VERSION, **counts}
