"""CoFL objectives and portable assets within the standard Lightning lifecycle."""

import math
from dataclasses import asdict
from typing import Literal

import lightning.pytorch as pl
import torch
from torch.nn import functional as F
from transformers import get_scheduler

from cofl.models.backbone import SigLIPBackbone

from .checkpoint import SCHEMA_VERSION, read_checkpoint
from .config import METHOD_PROFILES, ModelConfig, OptimizerConfig
from .losses import FieldLossConfig, field_loss
from .metrics import FieldMetrics, region_statistics, region_values
from .policy import build_policy, encode_prepared, query_policy
from .random_state import capture_random_state, gather_rank_states, restore_random_state
from .regions import FREE, OBSTACLE, OCCLUDED
from .resume import checkpoint_resume_error, prepare_loop_checkpoint, resume_at_next_epoch
from .sampling import interpolate_field_groups


class CoFLModule(pl.LightningModule):
    def __init__(
        self,
        model: ModelConfig = ModelConfig(),
        optimizer: OptimizerConfig = OptimizerConfig(),
        field_loss: FieldLossConfig = FieldLossConfig(),
        action_loss_weight: float = 0.3,
        checkpoint_path: str | None = None,
        mask_gt_obstacle: bool = False,
        masked_loss_normalization: Literal["all", "selected"] = "all",
    ):
        super().__init__()
        self.model_config = ModelConfig(**model) if isinstance(model, dict) else model
        self.optimizer_config = (
            OptimizerConfig(**optimizer) if isinstance(optimizer, dict) else optimizer
        )
        self.loss_config = (
            FieldLossConfig(**field_loss) if isinstance(field_loss, dict) else field_loss
        )
        if not math.isfinite(action_loss_weight) or action_loss_weight < 0:
            raise ValueError("action_loss_weight must be finite and nonnegative")
        self.action_loss_weight = action_loss_weight
        if not isinstance(mask_gt_obstacle, bool):
            raise TypeError("mask_gt_obstacle must be a boolean")
        if masked_loss_normalization not in {"all", "selected"}:
            raise ValueError("masked_loss_normalization must be all or selected")
        self.mask_gt_obstacle = mask_gt_obstacle
        self.masked_loss_normalization = masked_loss_normalization
        # The baseline needs no mask configuration. Persist every enabled
        # ablation explicitly so it cannot be changed during exact resume.
        mask_config = (
            {
                "mask_gt_obstacle": True,
                "masked_loss_normalization": masked_loss_normalization,
            }
            if mask_gt_obstacle else {}
        )
        self.save_hyperparameters(
            {
                "model": asdict(self.model_config),
                "optimizer": asdict(self.optimizer_config),
                "field_loss": asdict(self.loss_config),
                "action_loss_weight": action_loss_weight,
                **mask_config,
            }
        )
        backbone = None
        if checkpoint_path is not None:
            checkpoint = read_checkpoint(checkpoint_path)
            backbone = SigLIPBackbone.from_assets(checkpoint["cofl"]["backbone_assets"])
        self.policy = build_policy(self.model_config, backbone=backbone)
        self.validation_metrics = FieldMetrics(self.loss_config, action_loss_weight)
        self._resume_random_state = None
        self._resume_next_epoch = False

    def forward(self, batch):
        context = encode_prepared(self.policy, self.model_config, batch["inputs"], self.device)
        samples = [{"geometry": geometry} for geometry in batch["geometry"]]
        prediction = query_policy(
            self.policy, self.model_config, batch["queries"], context, samples
        )
        logits = (
            self.policy.action_logits(context) if self.model_config.method == "cofl-s" else None
        )
        return prediction, logits

    def _objective(self, batch, *, training=False):
        prediction, logits = self(batch)
        target = batch["targets"]
        if target is None:
            target = interpolate_field_groups(batch["fields"], batch["queries"])
        mask = self._training_query_mask(batch, target) if training else None
        losses = field_loss(
            prediction.float(), target.float(), config=self.loss_config,
            mask=mask,
            mask_normalization=self.masked_loss_normalization if mask is not None else "selected",
            allow_empty=mask is not None,
        )
        total = losses["loss"]
        if logits is not None:
            labels = batch["target_actions"]
            action = F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
            action = action / labels.ne(-100).sum().clamp_min(1)
            losses["action"] = action
            total = total + self.action_loss_weight * action
        losses["total"] = total
        return prediction, target, logits, losses

    def _training_query_mask(self, batch, target):
        if not self.mask_gt_obstacle:
            return None
        regions = batch.get("query_regions")
        if regions is None:
            raise ValueError("mask_gt_obstacle requires known query region labels")
        if regions.shape != target.shape[:-1] or regions.device != target.device:
            raise ValueError("Query regions must match target shape and device")
        known = (regions == FREE) | (regions == OBSTACLE) | (regions == OCCLUDED)
        if not bool(known.all()):
            raise ValueError(
                "mask_gt_obstacle requires known query region labels; unknown regions "
                "cannot be included in this ablation"
            )
        return regions == FREE

    def training_step(self, batch, batch_idx):
        prediction, target, _, losses = self._objective(batch, training=True)
        statistics = region_statistics(
            prediction.float(), target.float(), batch.get("query_regions"), self.loss_config
        )
        diagnostics = region_values(statistics, self.loss_config)
        if self.mask_gt_obstacle:
            # Regional diagnostics keep measuring every original query. Their
            # contributions sum to this full-field reference, while train/loss
            # is the masked field objective actually used for optimization.
            sums = statistics.sum(dim=0)
            full_direction = sums[0] / sums[2].clamp_min(1)
            full_magnitude = sums[1] / sums[3].clamp_min(1)
            full_loss = (
                self.loss_config.direction_weight * full_direction
                + self.loss_config.magnitude_weight * full_magnitude
            )
            diagnostics.update({
                "full/direction": full_direction,
                "full/magnitude": full_magnitude,
                "full/loss": full_loss,
                "full/total": full_loss + self.action_loss_weight
                * losses.get("action", full_loss.new_zeros(())).detach(),
                "supervision/queries": statistics[FREE, 3],
                "supervision/query_fraction": statistics[FREE, 3] / sums[3].clamp_min(1),
                "supervision/directional_count": statistics[FREE, 2],
                "supervision/directional_fraction": statistics[FREE, 2] / sums[2].clamp_min(1),
            })
        self.log_dict(
            {
                f"train/{key}": value
                for key, value in {**losses, **self._region_logs(diagnostics)}.items()
            },
            on_step=True,
            on_epoch=False,
            batch_size=len(batch["sample_ids"]),
        )
        return losses["total"]

    def _region_logs(self, values):
        if self.model_config.method == "cofl":
            return {
                key: value
                for key, value in values.items()
                if not key.startswith(("regions/occluded/", "regions/gt_obstacle/"))
            }
        return values

    def validation_step(self, batch, batch_idx):
        prediction, target, logits, losses = self._objective(batch)
        scales = torch.tensor(
            [
                geometry["normalization_scale_m"] if self.model_config.method == "cofl-s" else 1.0
                for geometry in batch["geometry"]
            ],
            device=self.device,
        )
        self.validation_metrics.update(
            prediction.float(),
            target.float(),
            losses,
            scales,
            logits,
            batch["target_actions"],
            batch.get("query_regions"),
        )

    def on_validation_epoch_end(self):
        values = self._region_logs(self.validation_metrics.compute())
        if self.model_config.method == "cofl":
            for name in ("action", "action_accuracy", "action_count"):
                values.pop(name)
        self.log_dict(
            {f"val/{key}": value for key, value in values.items() if key != "total"},
            prog_bar=False,
            sync_dist=False,
        )
        self.log("val/total", values["total"], prog_bar=True, sync_dist=False)
        self.validation_metrics.reset()

    def setup(self, stage):
        data = self.trainer.datamodule
        if data is None:
            return
        if data.profile != METHOD_PROFILES[self.model_config.method]:
            raise ValueError("Dataset profile does not match the model")
        if self.model_config.method == "cofl-s":
            for key in ("r_max_m", "normalization_scale_m"):
                if not math.isclose(
                    data.geometry[key], getattr(self.model_config, key), rel_tol=1e-6
                ):
                    raise ValueError(f"Dataset {key} does not match the model")

    def on_train_start(self):
        from lightning.pytorch.loggers import WandbLogger

        if self._resume_next_epoch:
            resume_at_next_epoch(self.trainer)
            self._resume_next_epoch = False

        for logger in self.loggers:
            if isinstance(logger, WandbLogger) and self.trainer.is_global_zero:
                logger.experiment.summary.update(
                    {
                        "parameters/total": sum(p.numel() for p in self.policy.parameters()),
                        "parameters/trainable": sum(
                            p.numel() for p in self.policy.parameters() if p.requires_grad
                        ),
                    }
                )
                logger.log_image(
                    "train/examples",
                    images=self.trainer.datamodule.example_images(),
                    step=self.global_step,
                )
        # Model/optimizer construction and loader restoration have finished.
        # Data augmentation owns separate RNGs, including with zero workers.
        if self._resume_random_state is not None:
            restore_random_state(self._resume_random_state)
            self._resume_random_state = None

    def configure_optimizers(self):
        config = self.optimizer_config
        backbone = self.policy.backbone
        vision = list(backbone.vision_encoder.parameters())
        text = list(backbone.text_encoder.parameters())
        backbone_ids = {id(parameter) for parameter in (*vision, *text)}
        # Keep the original downstream order for legacy frozen checkpoints.
        downstream = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]
        groups = []
        for name, parameters, learning_rate in (
            ("policy", downstream, config.learning_rate),
            ("siglip_vision", vision, config.vision_learning_rate),
            ("siglip_text", text, config.text_learning_rate),
        ):
            trainable = [parameter for parameter in parameters if parameter.requires_grad]
            if trainable:
                groups.append({
                    "name": name,
                    "params": trainable,
                    "lr": config.learning_rate if learning_rate is None else learning_rate,
                })
        optimizer = torch.optim.AdamW(
            groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = get_scheduler(
            "constant_with_warmup" if config.schedule == "constant" else "cosine",
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=int(self.trainer.estimated_stepping_batches),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint):
        data = self.trainer.datamodule
        next_epoch = prepare_loop_checkpoint(self.trainer, checkpoint)
        checkpoint["cofl"] = {
            "schema_version": SCHEMA_VERSION,
            "backbone_assets": self.policy.backbone.export_assets(),
            "dataset_identity": data.identity if data is not None else None,
            "training_protocol": self._training_protocol(),
            "next_epoch": next_epoch,
            "resume_error": checkpoint_resume_error(self.trainer),
            "random_states": gather_rank_states(
                capture_random_state(include_cuda=self.device.type == "cuda"),
                self.trainer.world_size,
            ),
        }

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("cofl", {}).get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "Checkpoint lacks exact resume state: start a new training run"
            )
        saved_hparams = dict(checkpoint["hyper_parameters"])
        # Earlier schema-2 checkpoints predate the independent tower options.
        # Only fill those new defaults; all other resume settings stay exact.
        for section, defaults in (
            ("model", {
                "vision_unfreeze_last_n_layers": 0,
                "text_unfreeze_last_n_layers": 0,
            }),
            ("optimizer", {"vision_learning_rate": None, "text_learning_rate": None}),
        ):
            saved_hparams[section] = {**defaults, **saved_hparams[section]}
        if dict(self.hparams) != saved_hparams:
            raise ValueError(
                "Resume requires the checkpoint model, loss and optimizer configuration"
            )
        data = self.trainer.datamodule if self._trainer is not None else None
        identity = checkpoint["cofl"]["dataset_identity"]
        if data is not None and data.identity != identity:
            raise ValueError(
                "Checkpoint dataset or validation sampling differs from this run, "
                "or training loader settings changed"
            )
        if (
            self._trainer is not None
            and self.trainer.state.fn == pl.trainer.states.TrainerFn.FITTING
        ):
            saved = checkpoint["cofl"]
            if saved["resume_error"]:
                raise ValueError(saved["resume_error"])
            if saved.get("training_protocol") != self._training_protocol():
                raise ValueError("Resume requires the checkpoint training protocol and step horizon")
            if data is None or type(data).__qualname__ not in checkpoint:
                raise ValueError("Checkpoint lacks training DataLoader state: start a new run")
            states = saved.get("random_states", [])
            if len(states) != self.trainer.world_size:
                raise ValueError("Checkpoint lacks random state for every training rank")
            self._resume_random_state = states[self.trainer.global_rank]
            self._resume_next_epoch = saved["next_epoch"]

    def _training_protocol(self):
        trainer = self.trainer
        return {
            "world_size": trainer.world_size,
            "device_type": trainer.strategy.root_device.type,
            "accumulate_grad_batches": trainer.accumulate_grad_batches,
            "precision": str(trainer.precision),
            "max_epochs": trainer.max_epochs,
            "max_steps": trainer.max_steps,
            "limit_train_batches": (
                type(trainer.limit_train_batches).__name__, trainer.limit_train_batches
            ),
            "limit_val_batches": (
                type(trainer.limit_val_batches).__name__, trainer.limit_val_batches
            ),
            "val_check_interval": (
                type(trainer.val_check_interval).__name__, trainer.val_check_interval
            ),
            "check_val_every_n_epoch": trainer.check_val_every_n_epoch,
            "gradient_clip_val": trainer.gradient_clip_val,
            "gradient_clip_algorithm": str(trainer.gradient_clip_algorithm),
        }
