"""Small lifecycle hooks alongside Lightning's standard checkpoint and loggers."""

from pathlib import Path

from lightning.pytorch import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar


class ResumeProgressBar(TQDMProgressBar):
    """Initialize the epoch display even when Lightning resumes inside an epoch."""

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        bar = self.train_progress_bar
        total = self.total_train_batches
        bar.total = None if total == float("inf") else total
        # Mid-epoch resume skips on_train_epoch_start. A checkpoint saved in
        # on_train_batch_end has processed the batch but not yet completed it.
        # Read that position for display without changing Lightning's loop.
        position = (
            trainer.fit_loop.epoch_loop.batch_progress.current.processed
            if trainer.fit_loop.restarted_mid_epoch
            else 0
        )
        bar.n = bar.initial = position
        bar.set_description(f"Epoch {trainer.current_epoch}")


class WarmupEarlyStopping(EarlyStopping):
    """Begin the standard patience counter after the learning-rate warmup."""

    def __init__(
        self,
        start_step: int = 40000,
        monitor: str = "val/total",
        patience: int = 10,
        min_delta: float = 0.0,
    ):
        if start_step < 0:
            raise ValueError("start_step must be nonnegative")
        super().__init__(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            mode="min",
            check_on_train_epoch_end=False,
        )
        self.start_step = start_step

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_step >= self.start_step:
            super().on_validation_end(trainer, pl_module)


class FinalCheckpoint(Callback):
    """Keep the final optimizer update in the existing standard last checkpoint."""

    def on_train_end(self, trainer, pl_module):
        callback = next(
            (
                c
                for c in trainer.checkpoint_callbacks
                if isinstance(c, ModelCheckpoint) and c.save_last
            ),
            None,
        )
        if callback is not None:
            path = Path(callback.dirpath) / "last.ckpt"
            callback.last_model_path = str(path)
            trainer.save_checkpoint(path)
