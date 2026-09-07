from __future__ import annotations

from typing import Any

import numpy as np


def as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def quat_to_xyzw(rotation):
    """Convert a Habitat numpy quaternion or an explicit xyzw sequence."""
    if isinstance(rotation, (np.ndarray, list, tuple)):
        return np.asarray(rotation, dtype=np.float32).reshape(4)
    return np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float32)


def quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to 3x3 rotation matrix.

    Assumes quaternion represents rotation from local -> world.
    """
    q = as_numpy(q_xyzw).astype(np.float64).reshape(4)
    (x, y, z, w) = q.tolist()
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    (xx, yy, zz) = (x * x * s, y * y * s, z * z * s)
    (xy, xz, yz) = (x * y * s, x * z * s, y * z * s)
    (wx, wy, wz) = (w * x * s, w * y * s, w * z * s)
    R = np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R.astype(np.float32)
