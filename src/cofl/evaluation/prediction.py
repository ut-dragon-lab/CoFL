"""One batch encoding serves field reconstruction, image rollout and actions."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.profiler import record_function

from cofl.runtime.field_grid import field_grid_queries
from cofl.training.policy import encode_prepared, query_policy_kwargs


@dataclass
class BatchPredictions:
    fields: list[np.ndarray | None]
    navigation_fields: np.ndarray | None
    actions: np.ndarray | None


@torch.inference_mode()
def predict_batch(policy, model_config, batch, config):
    device = next(policy.parameters()).device
    precision = config.precision
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        precision
    ]
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "float32"):
        return _predict_batch(policy, model_config, batch, config, device)


def _predict_batch(policy, model_config, batch, config, device):
    samples = batch["samples"]
    with record_function("cofl.encode"):
        context = encode_prepared(policy, model_config, batch["inputs"], device)
    image_grid = None
    if "image_navigation" in config.tasks:
        image_grid = field_grid_queries("image_field_v1", grid_size=config.trajectory.grid_size)
    field_queries = (
        [sample["query_grid"][sample["mask"]] for sample in samples]
        if "field" in config.tasks
        else [None] * len(samples)
    )
    field_counts = [0 if values is None else len(values) for values in field_queries]
    navigation_count = 0 if image_grid is None else len(image_grid)
    count = max(field_counts) + navigation_count
    predictions = None
    if count:
        with record_function("cofl.transfer_queries"):
            if "field" not in config.tasks:
                # Navigation uses the same lattice for every item. Upload it
                # once, and let the decoder consume a batch view.
                queries = torch.from_numpy(image_grid).to(device)[None].expand(len(samples), -1, -1)
            else:
                queries = np.zeros((len(samples), count, 2), dtype=np.float32)
                for index, (values, field_count) in enumerate(zip(field_queries, field_counts)):
                    queries[index, :field_count] = values
                    if image_grid is not None:
                        queries[index, field_count : field_count + navigation_count] = image_grid
                queries = torch.from_numpy(queries).to(device)
        kwargs = query_policy_kwargs(
            model_config, samples, device=device, dtype=queries.dtype, inference=True
        )
        with record_function("cofl.prepare_query_context"):
            query_context = policy.prepare_query_context(context)
        with record_function("cofl.query"):
            output = torch.empty((len(samples), count, 2), dtype=torch.float32, device=device)
            for start in range(0, count, config.query_chunk_size):
                chunk = slice(start, start + config.query_chunk_size)
                output[:, chunk] = policy.query(queries[:, chunk], query_context, **kwargs)
        with record_function("cofl.transfer_predictions"):
            predictions = output.cpu().numpy()
    fields, rasters = [], []
    for index, (sample, field_count) in enumerate(zip(samples, field_counts)):
        field = None
        if "field" in config.tasks:
            field = np.full((*sample["mask"].shape, 2), np.nan, dtype=np.float32)
            if field_count:
                field[sample["mask"]] = predictions[index, :field_count]
        fields.append(field)
        if image_grid is not None:
            size = config.trajectory.grid_size
            rasters.append(
                predictions[index, field_count : field_count + len(image_grid)]
                .reshape(size, size, 2)
                .transpose(2, 0, 1)
            )
    actions = (
        policy.action_logits(context).float().cpu().numpy() if "action" in config.tasks else None
    )
    return BatchPredictions(fields, np.stack(rasters) if rasters else None, actions)
