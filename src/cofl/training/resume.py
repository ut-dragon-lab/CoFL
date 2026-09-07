"""Checkpoint boundaries for the pinned Lightning training loop.

Lightning saves inside hooks, before their progress counters are committed.
Keep these corrections on the checkpoint copy, never on the running baseline.
"""


def prepare_loop_checkpoint(trainer, checkpoint):
    fit = checkpoint["loops"]["fit_loop"]
    # The DataModule owns per-rank loader snapshots. Lightning's duplicate
    # CombinedLoader snapshot only contains rank zero.
    fit["state_dict"].pop("combined_loader", None)
    epochs = fit["epoch_progress"]
    next_epoch = epochs["current"]["processed"] > epochs["current"]["completed"]
    if next_epoch:
        for scope in ("current", "total"):
            epochs[scope]["completed"] = epochs[scope]["processed"]
        checkpoint["epoch"] = epochs["current"]["completed"]
        return True

    loop = trainer.fit_loop.epoch_loop
    batches = fit["epoch_loop.batch_progress"]["current"]
    if (
        trainer.training
        and batches["processed"] > batches["completed"]
        and loop._should_check_val_fx(trainer.fit_loop._data_fetcher)
    ):
        # This batch has finished but its scheduled validation has not begun.
        # Encode the first validation batch as pending, so Lightning resumes
        # validation before fetching the next training batch.
        validation = fit["epoch_loop.val_loop.batch_progress"]
        validation["is_last_batch"] = False
        for scope in ("current", "total"):
            completed = 0 if scope == "current" else validation[scope]["completed"]
            validation[scope].update(
                ready=completed + 1, started=completed + 1,
                processed=completed, completed=completed,
            )
    return False


def resume_at_next_epoch(trainer):
    # A completed epoch needs the ordinary new-epoch reset. Lightning 2.5's
    # iteration-based restart path otherwise adds a batch to the old epoch.
    # The restored loader already points at the next epoch's first batch.
    loop = trainer.fit_loop.epoch_loop
    loop.restarting = False
    loop.reset_restart_stage()
    loop.val_loop.reset_restart_stage()


def checkpoint_resume_error(trainer):
    """Reject custom saves whose unfinished work is absent from checkpoints."""
    loop = trainer.fit_loop.epoch_loop
    if trainer.validating:
        if loop.val_loop.batch_progress.current.completed < sum(trainer.num_val_batches):
            return "Resume requires a checkpoint saved after validation finished"
    batches = loop.batch_progress.current
    if batches.ready > batches.processed:
        return "Resume requires a checkpoint saved after the training batch finished"
    if loop._should_accumulate():
        return "Resume requires a completed optimizer step, not partial gradient accumulation"
    return None
