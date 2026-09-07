"""Explicit Habitat 0.1.7 sensor and replay configuration."""

from __future__ import annotations

import logging
import math


def build_habitat_env(
    habitat_config_path, split, dataset_data_path=None, scenes_dir=None, hfov_deg=None
):
    import habitat

    from .habitat_utils import load_task_config

    if not dataset_data_path or not scenes_dir or hfov_deg is None:
        raise ValueError("Habitat replay requires explicit dataset, scene assets and sensor HFOV")
    rounded_hfov = int(round(float(hfov_deg)))
    if not math.isclose(float(hfov_deg), rounded_hfov, abs_tol=1e-6):
        raise ValueError(
            "Habitat 0.1.7 requires an integer-degree HFOV matching the field geometry"
        )
    cfg = load_task_config(habitat_config_path)
    for name in ("DATASET", "TASK", "SIMULATOR"):
        if not hasattr(cfg, name):
            raise ValueError("Habitat task config is missing " + name)
    cfg.defrost()
    cfg.DATASET.DATA_PATH = str(dataset_data_path).format(split=split)
    cfg.DATASET.SCENES_DIR = str(scenes_dir)
    cfg.DATASET.SPLIT = split
    cfg.DATASET.set_new_allowed(True)
    cfg.DATASET.CONTENT_SCENES = ["*"]
    cfg.DATASET.EPISODES_ALLOWED = ["*"]
    if str(cfg.DATASET.TYPE) == "RxR-VLN-CE-v1":
        cfg.DATASET.LANGUAGES = ["*"]
        if (
            not hasattr(cfg.DATASET, "ROLES")
            or len(cfg.DATASET.ROLES) != 1
            or cfg.DATASET.ROLES[0] == "*"
        ):
            raise ValueError(
                "A single RxR source file requires exactly one DATASET.ROLES entry in its task config"
            )
    # Generation reads language from episodes and uses an explicit follower.
    cfg.TASK.MEASUREMENTS = []
    cfg.TASK.SENSORS = []
    sim = cfg.SIMULATOR
    for name in ("RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"):
        if not hasattr(sim, name):
            raise ValueError("Habitat simulator config is missing " + name)
        getattr(sim, name).HFOV = rounded_hfov
    sim.SEMANTIC_SENSOR.WIDTH = 224
    sim.SEMANTIC_SENSOR.HEIGHT = 224
    sim.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    sim.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    cfg.freeze()
    logging.getLogger(__name__).info(
        "Replay sensors configured: HFOV=%s, metric depth, semantic=224", rounded_hfov
    )
    return habitat.Env(config=cfg)
