"""Build a deterministic, portable catalog of historical render cameras.

Only camera parameters and source-annotation checksums are exported. Missing
camera records stay null; missing views and malformed cameras are errors.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from cofl.generation.io import InputFingerprints

FORMAT = "cofl-render-cameras"
SCHEMA_VERSION = 1
CAMERA_CONVENTION = "open3d-legacy-view-control"
DATASETS = ("matterport", "scannet")
VIEWS = ("top", *(f"view{i}" for i in range(8)))
COUNT_FIELDS = ("scenes", "regions", "views", "available_cameras", "missing_cameras")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _identifier(value, *, context):
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or "/" in value or "\\" in value):
        raise ValueError(f"Invalid identifier: {context}")
    return value


def validate_camera_catalog(document) -> dict:
    """Validate the small catalog header without visiting every camera.

    Renderers validate a selected scene's records before replay. This header
    check is deliberately cheap enough to reuse with an already-loaded catalog.
    """
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise ValueError("Invalid camera catalog format")
    version = document.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError("Unsupported camera catalog schema version")
    if document.get("camera_convention") != CAMERA_CONVENTION:
        raise ValueError("Unsupported camera catalog convention")
    datasets = document.get("datasets")
    if not isinstance(datasets, dict) or not datasets or not set(datasets) <= set(DATASETS):
        raise ValueError("Camera catalog datasets must be matterport and/or scannet")
    for dataset, value in datasets.items():
        if not isinstance(value, dict):
            raise ValueError(f"Invalid camera catalog dataset: {dataset}")
        scenes, counts = value.get("scenes"), value.get("counts")
        if not isinstance(scenes, dict) or not scenes:
            raise ValueError(f"Camera catalog requires scene records: {dataset}")
        if (not isinstance(counts, dict) or set(counts) != set(COUNT_FIELDS)
                or any(type(counts[key]) is not int or counts[key] < 0 for key in COUNT_FIELDS)):
            raise ValueError(f"Invalid camera catalog counts: {dataset}")
        if (counts["scenes"] != len(scenes) or counts["regions"] < counts["scenes"]
                or counts["views"] != len(VIEWS) * counts["regions"]
                or counts["available_cameras"] + counts["missing_cameras"] != counts["views"]):
            raise ValueError(f"Inconsistent camera catalog counts: {dataset}")
    return document


def read_camera_catalog(path) -> dict:
    """Read the ordinary JSON catalog without importing Open3D."""
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle, object_pairs_hook=_unique_object)
    return validate_camera_catalog(document)


def _read_annotations(path, fingerprints):
    path = path.resolve(strict=True)
    before = fingerprints.describe([path])[0]
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != before["sha256"]:
        raise ValueError(f"Source annotations changed while reading: {path}")
    document = json.loads(payload, object_pairs_hook=_unique_object)
    fingerprints.describe([path])
    return document, before["sha256"]


def _scene_record(annotations, dataset, scan_id, source_sha256, validate_camera):
    if not isinstance(annotations, dict) or annotations.get("scan_id") != scan_id:
        raise ValueError(f"Source annotations do not match scene {dataset}/{scan_id}")
    if dataset == "matterport":
        groups = annotations.get("regions")
    else:
        groups = {"region0": annotations}
    if not isinstance(groups, dict) or not groups:
        raise ValueError(f"Source annotations require regions: {dataset}/{scan_id}")

    regions = {}
    available = missing = 0
    for region in sorted(groups):
        _identifier(region, context=f"{dataset}/{scan_id}/region")
        group = groups[region]
        views = group.get("views") if isinstance(group, dict) else None
        context = f"{dataset}/{scan_id}/{region}"
        if not isinstance(views, list) or len(views) != len(VIEWS):
            raise ValueError(f"Source annotations require all nine views: {context}")
        records = {}
        for view in views:
            view_id = view.get("view_id") if isinstance(view, dict) else None
            if not isinstance(view_id, str) or view_id not in VIEWS:
                raise ValueError(f"Invalid source view ID: {context}/{view_id}")
            if view_id in records:
                raise ValueError(f"Duplicate source view ID: {context}/{view_id}")
            records[view_id] = view
        if set(records) != set(VIEWS):
            raise ValueError(f"Source annotations require all nine views: {context}")
        region_cameras = {}
        for view_id in VIEWS:
            camera = records[view_id].get("camera")
            if camera is None:
                region_cameras[view_id] = None
                missing += 1
            else:
                region_cameras[view_id] = validate_camera(camera, context=f"{context}/{view_id}")
                available += 1
        regions[region] = region_cameras
    return {"source_annotations_sha256": source_sha256, "regions": regions}, available, missing


def _compact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _write_catalog(handle, document):
    """Keep each view on one line so changes remain practical to inspect."""
    handle.write("{\n")
    for key in ("format", "schema_version", "camera_convention"):
        handle.write(f"  {_compact(key)}: {_compact(document[key])},\n")
    handle.write('  "datasets": {\n')
    datasets = sorted(document["datasets"].items())
    for dataset_index, (dataset, value) in enumerate(datasets):
        handle.write(f"    {_compact(dataset)}: {{\n")
        handle.write(f'      "counts": {_compact(value["counts"])},\n')
        handle.write('      "scenes": {\n')
        scenes = sorted(value["scenes"].items())
        for scene_index, (scan_id, scene) in enumerate(scenes):
            handle.write(f"        {_compact(scan_id)}: {{\n")
            handle.write(f'          "source_annotations_sha256": {_compact(scene["source_annotations_sha256"])},\n')
            handle.write('          "regions": {\n')
            regions = sorted(scene["regions"].items())
            for region_index, (region, cameras) in enumerate(regions):
                handle.write(f"            {_compact(region)}: {{\n")
                for view_index, view_id in enumerate(VIEWS):
                    comma = "," if view_index + 1 < len(VIEWS) else ""
                    handle.write(f"              {_compact(view_id)}: {_compact(cameras[view_id])}{comma}\n")
                comma = "," if region_index + 1 < len(regions) else ""
                handle.write(f"            }}{comma}\n")
            handle.write("          }\n")
            comma = "," if scene_index + 1 < len(scenes) else ""
            handle.write(f"        }}{comma}\n")
        handle.write("      }\n")
        comma = "," if dataset_index + 1 < len(datasets) else ""
        handle.write(f"    }}{comma}\n")
    handle.write("  }\n}\n")


def build_camera_catalog(roots: dict[str, Path], output: Path) -> dict:
    """Export source annotations to a new catalog; existing output is an error.

    Each root contains scene directories with annotations.json. A subset of the
    two supported datasets is allowed. Input bytes and file identities must stay
    unchanged through publication. No scene assets or labels are copied.
    """
    # Keep this import inside the builder: cameras can consume catalogs too.
    from .cameras import validate_camera

    if not isinstance(roots, dict) or not roots or not set(roots) <= set(DATASETS):
        raise ValueError("Camera roots must contain matterport and/or scannet")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Camera catalog already exists: {output}")
    fingerprints = InputFingerprints()
    document = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "camera_convention": CAMERA_CONVENTION,
        "datasets": {},
    }
    for dataset in sorted(roots):
        root = Path(roots[dataset]).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
        scenes = {}
        counts = dict.fromkeys(COUNT_FIELDS, 0)
        for folder in sorted(
            path for path in root.iterdir() if path.is_dir() and path.name != ".staging"
        ):
            scan_id = _identifier(folder.name, context=f"{dataset}/scene")
            annotations, source_sha256 = _read_annotations(folder / "annotations.json", fingerprints)
            source_id = annotations.get("scan_id") if isinstance(annotations, dict) else None
            if isinstance(source_id, str) and source_id in scenes:
                raise ValueError(f"Duplicate source scene ID: {dataset}/{source_id}")
            scene, available, missing = _scene_record(
                annotations, dataset, scan_id, source_sha256, validate_camera,
            )
            scenes[scan_id] = scene
            counts["scenes"] += 1
            counts["regions"] += len(scene["regions"])
            counts["views"] += available + missing
            counts["available_cameras"] += available
            counts["missing_cameras"] += missing
        if not scenes:
            raise ValueError(f"No source scene annotations found: {root}")
        document["datasets"][dataset] = {"counts": counts, "scenes": scenes}
    validate_camera_catalog(document)
    fingerprints.verify_all()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            _write_catalog(handle, document)
            handle.flush()
            os.fsync(handle.fileno())
        fingerprints.verify_all()
        checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
        size_bytes = temporary.stat().st_size
        # Same-directory link publication is atomic and cannot replace output.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "output": str(output),
        "sha256": checksum,
        "size_bytes": size_bytes,
        "datasets": {dataset: value["counts"] for dataset, value in document["datasets"].items()},
    }
