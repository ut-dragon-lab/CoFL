"""Lightning's configuration, command dispatch and checkpoint resume entry point."""

import sys

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import ProgressBar
from lightning.pytorch.cli import LightningCLI

from .callbacks import ResumeProgressBar
from .data import CoFLDataModule
from .module import CoFLModule

BatchCountOrFraction = int | float


class CoFLTrainer(Trainer):
    """Trainer arguments distinguish batch counts from epoch fractions."""

    def __init__(
        self,
        *,
        limit_train_batches: BatchCountOrFraction | None = None,
        limit_val_batches: BatchCountOrFraction | None = None,
        limit_test_batches: BatchCountOrFraction | None = None,
        limit_predict_batches: BatchCountOrFraction | None = None,
        overfit_batches: BatchCountOrFraction = 0.0,
        val_check_interval: BatchCountOrFraction | None = None,
        **kwargs,
    ):
        if kwargs.get("enable_progress_bar", True):
            callbacks = kwargs.get("callbacks")
            callbacks = (
                [callbacks] if isinstance(callbacks, Callback) else list(callbacks or [])
            )
            if not any(isinstance(callback, ProgressBar) for callback in callbacks):
                kwargs["callbacks"] = [*callbacks, ResumeProgressBar()]
        super().__init__(
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            val_check_interval=val_check_interval,
            **kwargs,
        )


class CoFLLightningCLI(LightningCLI):
    def before_instantiate_classes(self):
        config = self.config.get(self.subcommand, self.config)
        checkpoint = config.get("ckpt_path")
        if checkpoint is not None:
            # Construct the backbone from embedded assets before Trainer restores
            # weights and optimizer state; no pretrained cache is required.
            config.model.checkpoint_path = str(checkpoint)


def main(argv=None):
    previous = sys.argv
    try:
        if argv is not None:
            sys.argv = [previous[0], *argv]
        return CoFLLightningCLI(
            CoFLModule,
            CoFLDataModule,
            trainer_class=CoFLTrainer,
            seed_everything_default=42,
            auto_configure_optimizers=False,
            load_from_checkpoint_support=False,
            save_config_kwargs={"overwrite": True},
        )
    finally:
        sys.argv = previous
