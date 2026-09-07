"""Use a saved field policy to generate a trajectory from one observation."""

import json
from pathlib import Path

import torch

from cofl.data.collection import open_dataset
from cofl.evaluation.protocol import file_sha256
from cofl.runtime.field_grid import predict_field_grid
from cofl.runtime.inference_rules import (
    DEFAULT_FIELD_GRID_SIZE,
    DEFAULT_FIELD_QUERY_CHUNK_SIZE,
    inference_rule_metadata,
)
from cofl.runtime.rollout import rollout_image_field, rollout_sector_field

from .config import METHOD_PROFILES
from .policy import encode_samples, load_policy, query_policy_kwargs


def rollout_sample(
    dataset_root,
    checkpoint_path,
    output_path,
    *,
    split="val",
    sample_index=0,
    start=None,
    max_steps=100,
    policy_dt=0.01,
    device="cpu",
):
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Rollout output already exists: {output}")
    model, config = load_policy(checkpoint_path, device=device)
    dataset = open_dataset(
        dataset_root, expected_profile=METHOD_PROFILES[config.method], split=split
    )
    if sample_index < 0 or sample_index >= len(dataset):
        raise ValueError("sample-index is outside the selected dataset split")
    sample = dataset[sample_index]
    with torch.inference_mode():
        context = encode_samples(model, config, [sample], device)
        query_context = model.prepare_query_context(context)
        query_kwargs = query_policy_kwargs(
            config, [sample], device=device, dtype=torch.float32, inference=True
        )

        field = predict_field_grid(
            model,
            query_context,
            query_kwargs,
            geometry=sample["geometry"],
            device=device,
            grid_size=DEFAULT_FIELD_GRID_SIZE,
            query_chunk_size=DEFAULT_FIELD_QUERY_CHUNK_SIZE,
        )

        kwargs = {
            "geometry": sample["geometry"],
            "max_steps": max_steps,
            "policy_dt": policy_dt,
            "boundary": "project",
            "nonfinite": "raise",
        }
        if config.method == "cofl":
            result = rollout_image_field(
                field, start=(0.5, 0.5) if start is None else start, **kwargs
            )
        else:
            result = rollout_sector_field(
                field, start_m=start, field_grid_size=DEFAULT_FIELD_GRID_SIZE, **kwargs
            )
    record = {
        "method": config.method,
        "profile": sample["profile"],
        "sample_id": sample["sample_id"],
        "instruction": sample["instruction"],
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "field_grid": field.metadata,
        "inference_rules": inference_rule_metadata(
            sample["profile"],
            grid_size=DEFAULT_FIELD_GRID_SIZE,
            num_steps=max_steps,
            policy_dt=policy_dt,
        ),
        "rollout": result.to_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record
