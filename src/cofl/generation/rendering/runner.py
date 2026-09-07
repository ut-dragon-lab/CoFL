"""Render official scene assets into the image generator's source layout.

Publication is per scene, including every Matterport region. Historical output
directories without receipts are never adopted or modified by this command.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from cofl.data._io import file_sha256
from cofl.generation.io import (
    InputFingerprints, digest_json, output_lock, tree_checksums, write_json,
)
from cofl.generation.random import seeded_random, unit_seed

from .cameras import camera_source, load_cameras

VIEWS = ("top", *(f"view{i}" for i in range(8)))


def _renderer(dataset):
    return importlib.import_module(f"cofl.generation.rendering.{dataset}")


def _source_files(root, dataset, scan_id):
    scene = root / scan_id
    if dataset == "matterport":
        regions = scene / "region_segmentations" / scan_id / "region_segmentations"
        meshes = sorted(regions.glob("region*.ply"))
        if not meshes:
            raise FileNotFoundError(f"No Matterport region meshes: {regions}")
        paths = [
            path for mesh in meshes
            for path in (mesh, mesh.with_suffix(".semseg.json"), mesh.with_suffix(".fsegs.json"))
        ]
    else:
        paths = [scene / f"{scan_id}{suffix}" for suffix in (
            "_vh_clean_2.ply", ".aggregation.json", "_vh_clean_2.0.010000.segs.json", ".txt",
        )]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required rendering source is missing: {path}")
    return paths


def _validate_scene(path, dataset):
    """Validate renderer-owned files before exposing them to ImagePipeline."""
    annotations = json.loads((path / "annotations.json").read_text())
    if dataset == "matterport":
        regions = annotations.get("regions", {})
        actual = {p.name for p in path.iterdir() if p.is_dir() and p.name.startswith("region")}
        if not regions or set(regions) != actual:
            raise ValueError("Rendered scene has missing region annotations")
        groups = []
        for name, value in regions.items():
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("Invalid rendered region name")
            groups.append((path / name, value.get("views", [])))
    else:
        groups = [(path, annotations.get("views", []))]
    for folder, views in groups:
        if len(views) != len(VIEWS) or {v.get("view_id") for v in views} != set(VIEWS):
            raise ValueError(f"Rendered scene requires all nine views: {folder}")
        for view in views:
            name = view["view_id"]
            if view.get("image") != f"{name}.png" or view.get("mask_npy") != f"{name}_mask.npy":
                raise ValueError(f"Invalid rendered view filenames: {folder}/{name}")
            camera = view.get("camera", {})
            for key, shape in (("intrinsic", (3, 3)), ("extrinsic", (4, 4))):
                matrix = np.asarray(camera.get(key), dtype=float)
                if matrix.shape != shape or not np.isfinite(matrix).all():
                    raise ValueError(f"Missing or invalid camera {key}: {folder}/{name}")
            with Image.open(folder / f"{name}.png") as rgb:
                rgb.load()
                shape = (rgb.height, rgb.width)
                if rgb.mode != "RGB" or min(shape) < 2:
                    raise ValueError(f"Invalid rendered RGB image: {folder}/{name}")
            # This is a freshly generated, local renderer-owned pickle payload.
            payload = np.load(folder / f"{name}_mask.npy", allow_pickle=True).item()
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid rendered semantic payload: {folder}/{name}")
            mask, labels = payload.get("mask"), payload.get("label_map")
            if (not isinstance(mask, np.ndarray) or mask.shape != shape
                    or not np.issubdtype(mask.dtype, np.integer)
                    or not isinstance(labels, dict)
                    or any(not isinstance(k, (int, np.integer)) or not isinstance(v, str)
                           for k, v in labels.items())):
                raise ValueError(f"Invalid rendered semantic mask: {folder}/{name}")
    return {"view_directories": len(groups), "views": len(groups) * len(VIEWS)}


def _environment():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "Pillow", "open3d"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _code_fingerprint():
    folder = Path(__file__).resolve().parent
    paths = [*sorted(folder.glob("*.py")), folder.parent / "io.py", folder.parent / "random.py"]
    return digest_json({p.relative_to(folder.parent).as_posix(): file_sha256(p) for p in paths})


def render_sources(
    dataset, root, output, *, scan_ids=None, seed=42, resume=False,
    cameras_from=None, bundled_cameras=False,
):
    """Render complete scenes with scoped seeds and verified per-scene resume.

    cameras_from accepts a scene-annotations directory or a combined catalog;
    bundled_cameras uses the optional catalog in a local checkout. Both replay the
    recorded intrinsics/extrinsics without sampling. Otherwise the seed fixes
    new camera sampling. Neither mode
    guarantees identical OpenGL pixels on another renderer stack.
    """
    if dataset not in {"matterport", "scannet"}:
        raise ValueError("Rendering dataset must be matterport or scannet")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Rendering seed must be a nonnegative integer")
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Scene source directory does not exist: {root}")
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Rendering source and output directories must not overlap")
    camera_root = camera_source(cameras_from, bundled_cameras=bundled_cameras)
    if camera_root is not None:
        if output.is_relative_to(camera_root) or camera_root.is_relative_to(output):
            raise ValueError("Rendering output and camera source directories must not overlap")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Rendering output is nonempty; use --resume: {output}")
    adapter = _renderer(dataset)
    available = sorted(adapter.find_all_scans(root))
    selected = available if scan_ids is None else sorted(scan_ids)
    if (not selected or len(selected) != len(set(selected))
            or any(not isinstance(s, str) or Path(s).name != s or s in {".", ".."}
                   for s in selected)):
        raise ValueError("Select at least one unique, valid scan ID")
    missing = set(selected) - set(available)
    if missing:
        raise ValueError(f"Unknown or unavailable scan IDs: {sorted(missing)}")
    sources = {s: _source_files(root, dataset, s) for s in selected}
    fingerprints = InputFingerprints()
    cameras = {}
    if camera_root is not None:
        catalog = None
        if camera_root.is_file():
            from .catalog import read_camera_catalog

            fingerprints.describe((camera_root,))
            catalog = read_camera_catalog(camera_root)
            fingerprints.describe((camera_root,))
        # Validate all selected records before starting the first scene. Hash
        # before loading, then check again so a file changed during this read
        # cannot produce an output with another version's receipt.
        for scan_id in selected:
            camera_file = (
                camera_root / scan_id / "annotations.json" if catalog is None else camera_root
            )
            fingerprints.describe((camera_file,))
            regions = (
                [p.stem for p in sources[scan_id] if p.suffix == ".ply"]
                if dataset == "matterport" else ["region0"]
            )
            cameras[scan_id] = load_cameras(
                camera_file, dataset, scan_id=scan_id, regions=regions, catalog=catalog,
            )
            fingerprints.describe((camera_file,))
            sources[scan_id].append(camera_file)
    definition = {
        "schema_version": 1, "dataset": dataset, "source_root": str(root),
        "scans": selected, "seed": seed, "seed_scheme": "sha256-base-method-unit-v1",
        "resolution": [640, 480], "code_sha256": _code_fingerprint(),
        "environment": _environment(),
        "camera_mode": "replay" if camera_root is not None else "sampled",
        "cameras_from": str(camera_root) if camera_root is not None else None,
        "bundled_cameras": bool(bundled_cameras),
    }
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        journal_path = output / "rendering.json"
        if journal_path.is_file():
            journal = json.loads(journal_path.read_text())
            if not resume or journal.get("definition") != definition:
                raise ValueError("Rendering recipe, code or environment changed; use a new output")
        else:
            if any(p.name not in {".generation.lock", "rendering.json.tmp"}
                   for p in output.iterdir()):
                raise ValueError("Cannot resume rendered assets without a rendering.json receipt")
            journal = {"definition": definition, "scenes": {}}
        journal["status"] = "running"
        write_json(journal_path, journal)
        try:
            for scan_id in selected:
                inputs = fingerprints.describe(sources[scan_id])
                scene_seed = unit_seed(seed, f"render-{dataset}", scan_id)
                identity = digest_json({
                    "dataset": dataset, "scan_id": scan_id, "seed": scene_seed,
                    "inputs": inputs, "code_sha256": definition["code_sha256"],
                    "environment": definition["environment"],
                })
                destination = output / scan_id
                previous = journal["scenes"].get(scan_id)
                if previous is not None and previous["identity"] != identity:
                    raise ValueError(f"Rendering source changed: {scan_id}")
                if destination.exists():
                    receipt = destination / "render-report.json"
                    if not receipt.is_file():
                        raise ValueError(f"Rendered scene has no receipt: {scan_id}")
                    if previous and previous.get("report_sha256") != file_sha256(receipt):
                        raise ValueError(f"Rendered scene receipt changed: {scan_id}")
                    report = json.loads(receipt.read_text())
                    if report.get("identity") != identity:
                        raise ValueError(f"Rendering source or identity changed: {scan_id}")
                    if report.get("files") != tree_checksums(destination, exclude=("render-report.json",)):
                        raise ValueError(f"Rendered payload changed: {scan_id}")
                else:
                    if previous is not None:
                        raise ValueError(f"Completed rendered scene is missing: {scan_id}")
                    staging = output / ".staging" / scan_id
                    if staging.exists():
                        shutil.rmtree(staging)
                    staging.mkdir(parents=True)
                    with seeded_random(scene_seed):
                        try:
                            if camera_root is None:
                                adapter.render_scan(scan_id, root, staging)
                            else:
                                adapter.render_scan(scan_id, root, staging, cameras=cameras[scan_id])
                        except ModuleNotFoundError as error:
                            if error.name == "open3d":
                                raise ImportError(
                                    "Scene rendering requires the rendering extra: "
                                    "uv sync --locked --extra rendering"
                                ) from error
                            raise
                    counts = _validate_scene(staging, dataset)
                    if dataset == "matterport":
                        expected = {p.stem for p in sources[scan_id] if p.suffix == ".ply"}
                        actual = {p.name for p in staging.iterdir() if p.is_dir()}
                        if actual != expected:
                            raise ValueError(f"Renderer omitted source regions: {scan_id}")
                    fingerprints.describe(sources[scan_id])
                    report = {
                        "identity": identity, "scan_id": scan_id, "seed": scene_seed,
                        "inputs": inputs, **counts, "files": tree_checksums(staging),
                    }
                    write_json(staging / "render-report.json", report)
                    staging.replace(destination)
                journal["scenes"][scan_id] = {
                    "identity": identity,
                    "report_sha256": file_sha256(destination / "render-report.json"),
                }
                write_json(journal_path, journal)
            fingerprints.verify_all()
            journal["status"] = "completed"
            write_json(journal_path, journal)
        except BaseException:
            journal["status"] = "failed"
            write_json(journal_path, journal)
            raise
    return {"path": str(output), "dataset": dataset, "scenes": len(selected), "status": "completed"}
