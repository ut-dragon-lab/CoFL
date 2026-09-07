"""CoFLDataset v1: shared observations with explicit field annotations."""

from .profiles import PROFILES, field_query_grid, geometry_for_profile, validate_geometry
from importlib import import_module

_LAZY = {
    "DatasetCollection": ".collection",
    "open_dataset": ".collection",
    "publish_collection": ".collection",
    "validate_collection": ".collection",
    "SCHEMA_VERSION": ".storage",
    "CoFLDataset": ".storage",
    "DatasetWriter": ".storage",
    "stable_id": ".storage",
    "validate_dataset": ".storage",
    "LeRobotCoFLDataset": ".lerobot_bridge",
    "export_lerobot": ".lerobot_bridge",
    "compare_datasets": ".comparison",
    "dataset_fingerprint": ".comparison",
    "write_comparison_report": ".comparison",
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(name)
    value = getattr(import_module(_LAZY[name], __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "DatasetCollection",
    "open_dataset",
    "publish_collection",
    "validate_collection",
    "CoFLDataset",
    "DatasetWriter",
    "PROFILES",
    "SCHEMA_VERSION",
    "field_query_grid",
    "geometry_for_profile",
    "stable_id",
    "validate_dataset",
    "validate_geometry",
    "LeRobotCoFLDataset",
    "export_lerobot",
    "compare_datasets",
    "dataset_fingerprint",
    "write_comparison_report",
]
