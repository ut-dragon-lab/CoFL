"""Count-weighted vector-field validation with distributed TorchMetrics states."""

import torch
from torch.nn import functional as F
from torchmetrics import Metric

from .losses import FieldLossConfig, _field_errors
from .regions import REGION_NAMES, UNKNOWN


@torch.no_grad()
def region_statistics(prediction, target, query_regions, config):
    """Return detached [region, statistic] sums; never change the objective.

    Columns are direction error, magnitude error, directional count and query
    count. Inputs have already passed the normal objective's validity checks.
    Missing region labels explicitly belong to unknown.
    """
    prediction, target = prediction.detach(), target.detach()
    if query_regions is None:
        query_regions = torch.full_like(target[..., 0], UNKNOWN, dtype=torch.int8)
    if query_regions.shape != target.shape[:-1]:
        raise ValueError("Query regions must match the vector leading dimensions")
    valid = torch.ones_like(target[..., 0], dtype=torch.bool)
    direction, magnitude, directional = _field_errors(prediction, target, config, valid)
    membership = query_regions[..., None] == torch.arange(
        len(REGION_NAMES), device=target.device
    )
    axes = tuple(range(target.ndim - 1))
    return torch.stack(
        (
            torch.where(membership, direction[..., None], 0).sum(axes, dtype=torch.float64),
            torch.where(membership, magnitude[..., None], 0).sum(axes, dtype=torch.float64),
            (membership & directional[..., None]).sum(axes, dtype=torch.float64),
            membership.sum(axes, dtype=torch.float64),
        ),
        dim=-1,
    )


def region_values(statistics, config):
    """Regional means and contributions using distinct global denominators.

    Contributions of the four disjoint regions sum to the global field loss.
    gt_obstacle is an overlapping diagnostic union of obstacle and occluded;
    it must not be added to that partition again. Empty regions have zero
    values and counts, which means no supervision rather than perfect quality.
    """
    global_sums = statistics.sum(dim=0)
    direction_count, query_count = global_sums[2:].clamp_min(1)
    global_loss = (
        config.direction_weight * global_sums[0] / direction_count
        + config.magnitude_weight * global_sums[1] / query_count
    )
    sums = torch.cat((statistics, (statistics[1] + statistics[2])[None]), dim=0)
    direction = sums[:, 0] / sums[:, 2].clamp_min(1)
    magnitude = sums[:, 1] / sums[:, 3].clamp_min(1)
    contribution = (
        config.direction_weight * sums[:, 0] / direction_count
        + config.magnitude_weight * sums[:, 1] / query_count
    )
    values = {
        "direction": direction,
        "magnitude": magnitude,
        "loss": config.direction_weight * direction + config.magnitude_weight * magnitude,
        "loss_contribution": contribution,
        "loss_fraction": contribution / global_loss.clamp_min(torch.finfo(statistics.dtype).tiny),
        "queries": sums[:, 3],
        "query_fraction": sums[:, 3] / query_count,
        "directional_count": sums[:, 2],
    }
    return {
        f"regions/{name}/{key}": value[index]
        for index, name in enumerate((*REGION_NAMES, "gt_obstacle"))
        for key, value in values.items()
    }


class FieldMetrics(Metric):
    full_state_update = False

    def __init__(self, config: FieldLossConfig, action_weight: float):
        super().__init__()
        self.config = config
        self.action_weight = action_weight
        self.add_state(
            "region_totals",
            default=torch.zeros(len(REGION_NAMES), 4, dtype=torch.float64),
            dist_reduce_fx="sum",
            persistent=False,
        )
        for name in (
            "direction",
            "magnitude",
            "vector",
            "angle",
            "action",
            "correct",
            "queries",
            "directions",
            "angles",
            "actions",
        ):
            self.add_state(
                name, default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
            )

    def update(
        self, prediction, target, losses, scales, logits=None, labels=None, query_regions=None
    ):
        self.region_totals += region_statistics(prediction, target, query_regions, self.config)
        target_norm = target.norm(dim=-1)
        prediction_norm = prediction.norm(dim=-1)
        directions = (target_norm > self.config.direction_min_target_magnitude).sum()
        count = target_norm.numel()
        self.direction += losses["direction"].double() * directions
        self.magnitude += losses["magnitude"].double() * count
        self.directions += directions
        self.queries += count
        self.vector += ((prediction - target).norm(dim=-1) * scales[:, None]).double().sum()
        directional = target_norm > 1e-8
        cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-8).clamp(-1, 1)
        angle = torch.rad2deg(torch.acos(cosine))
        angle = torch.where(prediction_norm <= 1e-8, 180.0, angle)
        self.angle += torch.where(directional, angle, 0).double().sum()
        self.angles += directional.sum()
        if logits is not None:
            valid = labels.ne(-100)
            self.action += F.cross_entropy(
                logits, labels, ignore_index=-100, reduction="sum"
            ).double()
            self.correct += (logits.argmax(-1).eq(labels) & valid).sum()
            self.actions += valid.sum()

    def compute(self):
        direction = self.direction / self.directions.clamp_min(1)
        magnitude = self.magnitude / self.queries.clamp_min(1)
        action = self.action / self.actions.clamp_min(1)
        field = self.config.direction_weight * direction + self.config.magnitude_weight * magnitude
        return {
            "direction": direction,
            "magnitude": magnitude,
            "loss": field,
            "total": field + self.action_weight * action,
            "vector_l2_mean": self.vector / self.queries.clamp_min(1),
            "angular_error_deg_mean": self.angle / self.angles.clamp_min(1),
            "action": action,
            "action_accuracy": self.correct / self.actions.clamp_min(1),
            "queries": self.queries,
            "directional_count": self.directions,
            "angular_count": self.angles,
            "action_count": self.actions,
            **region_values(self.region_totals, self.config),
        }
