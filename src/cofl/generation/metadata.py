"""Portable metadata for generation manifests and simulator workers."""

import hashlib
from pathlib import PurePath, PurePosixPath, PureWindowsPath

import numpy as np


def portable_metadata(value):
    """Convert NumPy values and replace absolute paths with stable identities.

    Paths retain only their basename and a digest of the original path string;
    file contents and the filesystem are never accessed. Relative paths, URLs
    and ordinary instruction text remain unchanged. Mapping keys become strings.
    """
    if isinstance(value, dict):
        return {str(key): portable_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_metadata(item) for item in value]
    if isinstance(value, np.ndarray):
        return portable_metadata(value.tolist())
    if isinstance(value, np.generic):
        return portable_metadata(value.item())
    if isinstance(value, PurePath):
        value = str(value)
    if isinstance(value, str):
        path = PurePosixPath(value)
        if not path.is_absolute():
            path = PureWindowsPath(value)
        if path.is_absolute():
            return {
                "name": path.name,
                "source_path_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            }
    return value
