"""Load and apply the exact camera records exported by legacy Open3D.

Legacy ViewControl can export nonorthogonal up/front rows in its extrinsic
matrix. Preserve those records: they are not necessarily rigid SE(3) poses.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import numpy as np

VIEWS = ("top", *(f"view{i}" for i in range(8)))
WIDTH, HEIGHT = 640, 480


def validate_camera(camera, *, context="camera"):
    """Return canonical JSON-compatible parameters without changing matrices."""
    if not isinstance(camera, dict):
        raise ValueError(f"Missing camera record: {context}")
    width, height = camera.get("width", WIDTH), camera.get("height", HEIGHT)
    if (isinstance(width, bool) or not isinstance(width, int)
            or isinstance(height, bool) or not isinstance(height, int)
            or (width, height) != (WIDTH, HEIGHT)):
        raise ValueError(f"Camera replay requires {WIDTH}x{HEIGHT} images: {context}")
    matrices = {}
    for key, shape in (("intrinsic", (3, 3)), ("extrinsic", (4, 4))):
        try:
            matrix = np.asarray(camera.get(key), dtype=np.float64)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid camera {key}: {context}") from error
        if matrix.shape != shape or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid camera {key}: {context}")
        matrices[key] = matrix
    intrinsic, extrinsic = matrices["intrinsic"], matrices["extrinsic"]
    if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-10, rtol=0)
            or abs(intrinsic[0, 1]) > 1e-10 or abs(intrinsic[1, 0]) > 1e-10):
        raise ValueError(f"Invalid camera intrinsic calibration: {context}")
    if (not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or abs(np.linalg.det(extrinsic[:3, :3])) < 1e-12):
        raise ValueError(f"Invalid or singular camera extrinsic: {context}")
    return {"width": width, "height": height, **{k: v.tolist() for k, v in matrices.items()}}


def bundled_camera_path():
    """Locate an optional historical camera catalog in a local checkout."""
    path = Path(str(files("cofl.generation.rendering").joinpath("assets", "cameras.json")))
    if not path.is_file():
        raise FileNotFoundError(
            "The historical CoFL camera catalog is not included in this release. "
            "Supply a licensed camera catalog with --cameras-from."
        )
    return path


def camera_source(cameras_from=None, *, bundled_cameras=False):
    """Resolve an explicit camera directory/catalog or the packaged catalog."""
    if bundled_cameras and cameras_from is not None:
        raise ValueError("Select either bundled cameras or cameras_from, not both")
    path = bundled_camera_path() if bundled_cameras else (
        None if cameras_from is None else Path(cameras_from).expanduser().resolve()
    )
    if path is not None and not (path.is_file() or path.is_dir()):
        raise FileNotFoundError(f"Camera source does not exist: {path}")
    return path


def load_cameras(path, dataset, *, scan_id, regions, catalog=None):
    """Require every source region/view; never fall back to sampled cameras.

    Accept a scene annotations file or one combined catalog. A caller reading
    many scenes can pass a document already validated by read_camera_catalog
    as catalog, avoiding reparsing the complete file for each scene.
    """
    from .catalog import _unique_object, validate_camera_catalog

    path = Path(path)
    annotations = catalog if catalog is not None else json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
    )
    if isinstance(annotations, dict) and "format" in annotations:
        validate_camera_catalog(annotations)
        try:
            scene = annotations["datasets"][dataset]["scenes"][scan_id]
        except KeyError as error:
            raise ValueError(f"Camera catalog has no {dataset} scene {scan_id}: {path}") from error
        if (not isinstance(scene, dict) or not isinstance(scene.get("regions"), dict)
                or not scene["regions"]
                or any(not isinstance(region, str) or not isinstance(views, dict)
                       for region, views in scene["regions"].items())):
            raise ValueError(f"Invalid camera catalog region records: {dataset}/{scan_id}")
        annotations = {
            "scan_id": scan_id,
            "regions": {
                region: {"views": [
                    {"view_id": view, "camera": camera} for view, camera in views.items()
                ]} for region, views in scene["regions"].items()
            },
        }
        # The catalog gives both datasets an explicit region layer.
        grouped = True
    else:
        grouped = False
    if not isinstance(annotations, dict) or annotations.get("scan_id") != scan_id:
        raise ValueError(f"Camera annotations do not match scan {scan_id}: {path}")
    if dataset == "matterport" or (dataset == "scannet" and grouped):
        groups = annotations.get("regions")
    elif dataset == "scannet":
        groups = {"region0": annotations}
    else:
        raise ValueError("Camera dataset must be matterport or scannet")
    if not isinstance(groups, dict) or set(groups) != set(regions):
        raise ValueError(f"Camera regions do not match source meshes: {path}")
    result = {}
    for region in sorted(groups):
        group = groups[region]
        views = group.get("views") if isinstance(group, dict) else None
        if (not isinstance(views, list) or len(views) != len(VIEWS)
                or any(not isinstance(v, dict) or not isinstance(v.get("view_id"), str)
                       for v in views)
                or {v["view_id"] for v in views} != set(VIEWS)):
            raise ValueError(f"Camera records require all nine views: {path}/{region}")
        records = {v["view_id"]: v for v in views}
        result[region] = {
            view: validate_camera(records[view].get("camera"), context=f"{path}/{region}/{view}")
            for view in VIEWS
        }
    return result


def resolve_render_cameras(cameras, cameras_from, bundled_cameras, *, dataset, scan_id, regions):
    """Share the direct renderer API with CLI-backed catalog loading."""
    if cameras is not None:
        if cameras_from is not None or bundled_cameras:
            raise ValueError("Pass camera records or a camera source, not both")
        return cameras
    source = camera_source(cameras_from, bundled_cameras=bundled_cameras)
    if source is None:
        return None
    path = source / scan_id / "annotations.json" if source.is_dir() else source
    return load_cameras(path, dataset, scan_id=scan_id, regions=regions)


def apply_camera(control, camera):
    """Apply a legacy record and fail if ViewControl cannot retain its values."""
    import open3d as o3d

    camera = validate_camera(camera)
    intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64)
    extrinsic = np.asarray(camera["extrinsic"], dtype=np.float64)
    parameters = o3d.camera.PinholeCameraParameters()
    parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        camera["width"], camera["height"], intrinsic,
    )
    parameters.extrinsic = extrinsic
    if not control.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True):
        raise ValueError("Open3D rejected the recorded camera parameters")
    actual = control.convert_to_pinhole_camera_parameters()
    if (actual.intrinsic.width != camera["width"] or actual.intrinsic.height != camera["height"]
            or not np.allclose(actual.intrinsic.intrinsic_matrix, intrinsic, rtol=1e-10, atol=1e-8)
            or not np.allclose(actual.extrinsic, extrinsic, rtol=1e-10, atol=1e-8)):
        raise ValueError("Open3D changed the recorded camera parameters during replay")
