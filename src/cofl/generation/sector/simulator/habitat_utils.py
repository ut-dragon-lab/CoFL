"""The supported Habitat 0.1.7 task configuration and sensor contract."""

from __future__ import annotations

import numpy as np

from ..fields.geom import quat_to_xyzw


def load_task_config(config_path):
    """Load a task YAML on the installed VLN-CE extension defaults."""
    import habitat_extensions.task  # noqa: F401  (registers VLN-CE datasets)
    from habitat_extensions.config.default import _C

    config = _C.clone()
    config.defrost()
    config.merge_from_file(str(config_path))
    config.freeze()
    return config


def get_pose(env):
    state = env.sim.get_agent_state()
    return np.asarray(state.position, dtype=np.float32).reshape(3), quat_to_xyzw(state.rotation)


def obs_get_rgb_depth_semantic(obs):
    """Read the configured RGB, metric depth and semantic sensors."""
    rgb = np.asarray(obs["rgb"])
    depth = np.asarray(obs["depth"])
    semantic = np.asarray(obs["semantic"])
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or depth.ndim != 2 or semantic.ndim != 2:
        raise ValueError("Habitat sensors must return RGB HWC, depth HW and semantic HW arrays")
    return rgb, depth, semantic
