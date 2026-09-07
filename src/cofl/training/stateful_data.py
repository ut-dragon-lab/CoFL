"""Serializable worker collation and consumed-batch snapshots for training."""

from copy import deepcopy

import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler as _DistributedSampler


DATA_STATE_VERSION = 1


class CollatedDataset(Dataset):
    """Expose the same collator object to TorchData's worker dataset snapshots.

    TorchData snapshots the dataset after fetching and collating each batch;
    it does not call state_dict on a separate collate_fn. Pickling dataset and
    collate_fn together preserves this shared reference in each worker.
    """

    def __init__(self, dataset, collator):
        self.dataset = dataset
        self.collator = collator

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def state_dict(self):
        return self.collator.state_dict()

    def load_state_dict(self, state):
        self.collator.load_state_dict(state)


class StatefulDistributedSampler(_DistributedSampler):
    """Let TorchData restore the permutation epoch before worker prefetching."""

    def state_dict(self):
        return {**super().state_dict(), "epoch": self.epoch}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.set_epoch(state["epoch"])


class CoFLStatefulDataLoader(StatefulDataLoader):
    """Preserve worker seeding and normalize checkpoints at an epoch boundary.

    The sampler uses its own generator. TorchData records that sampler state,
    but its loader generator (used to seed workers) needs separate storage.
    Lightning stops at len(loader) without requesting StopIteration, so an
    exhausted iterator must be advanced once when resuming at that boundary.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_worker_generator = None
        self._pending_sampler_generator = None
        self._pending_epoch_complete = False
        self._restored_epoch_unconsumed = False

    def state_dict(self):
        state = super().state_dict()
        return {
            "version": DATA_STATE_VERSION,
            "loader": deepcopy(state),
            "worker_generator": self.generator.get_state().clone(),
            "sampler_generator": (
                self.sampler.generator.get_state().clone()
                if hasattr(self.sampler, "generator") else None
            ),
            "epoch_complete": self._iterator._num_yielded >= len(self),
        }

    def load_state_dict(self, state):
        if state.get("version") != DATA_STATE_VERSION:
            raise ValueError("Training dataloader checkpoint protocol is unsupported")
        loader_state = deepcopy(state["loader"])
        # We handle the completed epoch explicitly, including the sampler's
        # lazy fast-forward. TorchData's automatic reset would skip that step.
        loader_state["_iterator_finished"] = False
        super().load_state_dict(loader_state)
        self._pending_worker_generator = state["worker_generator"].clone()
        self._pending_sampler_generator = state["sampler_generator"]
        self._pending_epoch_complete = state["epoch_complete"]
        self._restored_epoch_unconsumed = False

    def __iter__(self):
        if self._restored_epoch_unconsumed:
            if self._iterator._num_yielded == 0:
                # Restoring validation at the last training batch can finish
                # the old epoch after setup_data already prepared the next.
                # Reuse that unconsumed iterator when Lightning then enters
                # the next epoch, preserving its prefetched worker batches.
                return self._iterator
            self._restored_epoch_unconsumed = False
        iterator = super().__iter__()
        if self._pending_worker_generator is None:
            return iterator
        worker_state = self._pending_worker_generator
        self._pending_worker_generator = None
        if self._pending_epoch_complete:
            self._pending_epoch_complete = False
            # This consumes no data; the saved epoch has already yielded all
            # batches. It finishes the sampler's deferred cursor restoration.
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError("Completed training epoch restored an unconsumed batch")
            self.generator.set_state(worker_state)
            if self._pending_sampler_generator is not None:
                # RandomSampler generates an additional empty-tail permutation
                # on StopIteration. Lightning's length-based loop may never
                # have requested it, so keep the actual saved generator state.
                self.sampler.generator.set_state(self._pending_sampler_generator)
            if isinstance(self.sampler, StatefulDistributedSampler):
                self.sampler.set_epoch(self.sampler.epoch + 1)
            iterator = super().__iter__()
            self._restored_epoch_unconsumed = True
        else:
            # Constructing a replacement iterator draws a base seed. Undo only
            # that extra draw; sampler randomness has a separate generator.
            self.generator.set_state(worker_state)
        self._pending_sampler_generator = None
        return iterator
