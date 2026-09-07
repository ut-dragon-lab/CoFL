"""Native CoFL datasets prepared by Lightning and standard DataLoader workers."""

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchdata.stateful_dataloader.sampler import RandomSampler

from cofl.data import open_dataset
from cofl.data.collection import manifest_path

from .policy import action_labels, prepare_inputs
from .regions import REGION_EXTRA_KEYS, prepare_query_regions
from .sampling import prepare_field_groups, prepare_queries
from .stateful_data import (
    DATA_STATE_VERSION,
    CoFLStatefulDataLoader,
    CollatedDataset,
    StatefulDistributedSampler,
)


@dataclass
class CoFLCollator:
    """A worker owns a processor and CPU tensors, never pretrained model weights.

    Training owns isolated NumPy and CPU Torch random streams, serialized by
    the dataset wrapper after collation. Model RNG is never consumed.
    Validation queries are keyed by sample identity and never augmented.
    """

    processor_files: dict = field(repr=False)
    max_text_length: int = 64
    num_queries: int = 1000
    query_sampling: str = "continuous"
    stratified_grid_size: int = 10
    color_jitter: float = 0.0
    use_depth: bool = False
    validation: bool = False
    seed: int = 42
    _processor: object = field(default=None, init=False, repr=False)
    _numpy_rng: object = field(default=None, init=False, repr=False)
    _torch_state: object = field(default=None, init=False, repr=False)

    def _initialize_rng(self):
        if self._numpy_rng is not None:
            return
        identity = json.dumps(["cofl-train", self.seed, torch.initial_seed()])
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
        self._numpy_rng = np.random.default_rng(seed)
        self._torch_state = torch.Generator().manual_seed(seed).get_state()

    def state_dict(self):
        # Worker startup snapshots happen before the first batch. Do not seed
        # here: this may run in the parent before worker-specific seeding.
        return {
            "numpy": (
                None if self._numpy_rng is None else deepcopy(self._numpy_rng.bit_generator.state)
            ),
            "torch": None if self._torch_state is None else self._torch_state.clone(),
        }

    def load_state_dict(self, state):
        if state["numpy"] is None:
            self._numpy_rng, self._torch_state = None, None
            return
        self._numpy_rng = np.random.default_rng(0)
        self._numpy_rng.bit_generator.state = deepcopy(state["numpy"])
        self._torch_state = state["torch"].clone()

    def __call__(self, samples):
        if self._processor is None:
            from cofl.models.backbone import processor_from_files

            self._processor = processor_from_files(self.processor_files)
        if self.validation:
            prepared = []
            for sample in samples:
                identity = json.dumps([self.seed, sample["sample_id"]], separators=(",", ":"))
                seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
                prepared.append(
                    prepare_queries(
                        [sample],
                        self.num_queries,
                        np.random.default_rng(seed),
                        mode=self.query_sampling,
                        grid_size=self.stratified_grid_size,
                    )
                )
            queries = torch.cat([item[0] for item in prepared])
            targets = None if prepared[0][1] is None else torch.cat([item[1] for item in prepared])
        else:
            self._initialize_rng()
            queries, targets = prepare_queries(
                samples,
                self.num_queries,
                self._numpy_rng,
                mode=self.query_sampling,
                grid_size=self.stratified_grid_size,
            )
        fields = prepare_field_groups(samples, dtype=queries.dtype) if targets is None else []
        input_options = dict(max_text_length=self.max_text_length, use_depth=self.use_depth)
        if self.validation:
            inputs = prepare_inputs(self._processor, samples, color_jitter=0.0, **input_options)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self._torch_state)
                inputs = prepare_inputs(
                    self._processor, samples, color_jitter=self.color_jitter, **input_options
                )
                self._torch_state = torch.get_rng_state()
        return {
            "inputs": inputs,
            "queries": queries,
            "query_regions": prepare_query_regions(samples, queries, mode=self.query_sampling),
            "targets": targets,
            "fields": fields,
            "target_actions": action_labels(samples, "cpu"),
            "geometry": [sample["geometry"] for sample in samples],
            "sample_ids": [sample["sample_id"] for sample in samples],
            "observation_ids": [sample["observation_id"] for sample in samples],
            "episode_ids": [sample["episode_id"] for sample in samples],
        }


class CoFLDataModule(pl.LightningDataModule):
    """Split and collate native data using the standard Lightning loader lifecycle.

    Training snapshots consumed sample order, queries and augmentation in
    every worker; validation remains independently keyed by sample identity.
    """

    def __init__(
        self,
        dataset: str,
        batch_size: int = 32,
        num_workers: int = 4,
        multiprocessing_context: Literal["forkserver", "spawn"] = "forkserver",
        num_queries: int = 1000,
        query_sampling: str = "continuous",
        stratified_grid_size: int = 10,
        color_jitter: float = 0.15,
        validation_split: str = "val",
        validation_batch_size: int = 16,
        validation_num_queries: int = 1000,
        validation_max_samples: int | None = None,
        seed: int = 42,
        allow_partial_dataset: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        for name in (
            "batch_size",
            "num_queries",
            "stratified_grid_size",
            "validation_batch_size",
            "validation_num_queries",
        ):
            value = getattr(self.hparams, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("num_workers", "seed"):
            value = getattr(self.hparams, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if validation_max_samples is not None and (
            isinstance(validation_max_samples, bool)
            or not isinstance(validation_max_samples, int)
            or validation_max_samples < 1
        ):
            raise ValueError("validation_max_samples must be positive or null")
        if query_sampling not in {"grid", "continuous"}:
            raise ValueError("query_sampling must be grid or continuous")
        if multiprocessing_context not in {"forkserver", "spawn"}:
            raise ValueError("multiprocessing_context must be forkserver or spawn")
        if not math.isfinite(color_jitter) or not 0 <= color_jitter <= 1:
            raise ValueError("color_jitter must be between zero and one")
        if validation_split not in {"val", "val_seen", "val_unseen"}:
            raise ValueError("validation_split must be val, val_seen, or val_unseen")
        self.train_dataset = None
        self.validation_dataset = None
        self.profile = None
        self.geometry = None
        self.identity = None
        self._train_loader = None
        self._pending_train_loader = None

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return
        root = Path(self.hparams.dataset)
        train_dataset = open_dataset(
            root, split="train",
            eligible_only=True, extra_keys=REGION_EXTRA_KEYS,
        )
        if not len(train_dataset):
            raise ValueError("The eligible training split is empty")
        if train_dataset.manifest.get("source", {}).get("partial", False) and not (
            self.hparams.allow_partial_dataset
        ):
            raise ValueError("Partial datasets require allow_partial_dataset=true")
        profile = train_dataset.profile
        geometry = train_dataset[0]["geometry"]
        train_dataset.clear_payload_cache()
        validation = open_dataset(
            root,
            expected_profile=profile,
            split=self.hparams.validation_split,
            eligible_only=True,
            extra_keys=REGION_EXTRA_KEYS,
        )
        if not len(validation):
            raise ValueError("The eligible validation split is empty")
        candidates = np.arange(len(validation), dtype=np.int64)
        count = min(len(candidates), self.hparams.validation_max_samples or len(candidates))
        indices = (
            candidates
            if count == len(candidates)
            else np.sort(
                np.random.default_rng(self.hparams.seed).choice(candidates, count, False)
            )
        )
        self.train_dataset = train_dataset
        self.validation_dataset = Subset(validation, indices.tolist())
        self.profile, self.geometry = profile, geometry
        self.identity = {
            "training_data_protocol": DATA_STATE_VERSION,
            "dataset_manifest_sha256": hashlib.sha256(manifest_path(root).read_bytes()).hexdigest(),
            "profile": self.profile,
            "training_samples": len(self.train_dataset),
            "validation_split": self.hparams.validation_split,
            "validation_samples": count,
            "validation_subset_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
            "validation_seed": self.hparams.seed,
            "validation_num_queries": self.hparams.validation_num_queries,
            "query_sampling": self.hparams.query_sampling,
            "stratified_grid_size": self.hparams.stratified_grid_size,
            "training_batch_size": self.hparams.batch_size,
            "training_num_queries": self.hparams.num_queries,
            "training_color_jitter": self.hparams.color_jitter,
            "training_seed": self.hparams.seed,
            "num_workers": self.hparams.num_workers,
            "multiprocessing_context": self.hparams.multiprocessing_context,
            "validation_batch_size": self.hparams.validation_batch_size,
        }

    def _collator(self, *, validation):
        module = self.trainer.lightning_module
        model_config = module.model_config
        assets = module.policy.backbone.export_assets()
        return CoFLCollator(
            processor_files=assets["processor_files"],
            max_text_length=assets["max_text_length"],
            num_queries=(
                self.hparams.validation_num_queries if validation else self.hparams.num_queries
            ),
            query_sampling=self.hparams.query_sampling,
            stratified_grid_size=self.hparams.stratified_grid_size,
            color_jitter=self.hparams.color_jitter,
            use_depth=model_config.method == "cofl-s" and model_config.use_depth,
            validation=validation,
            seed=self.hparams.seed,
        )

    def _loader(self, *, validation):
        workers = self.hparams.num_workers
        collator = self._collator(validation=validation)
        kwargs = dict(
            batch_size=self.hparams.validation_batch_size
            if validation
            else self.hparams.batch_size,
            num_workers=workers,
            collate_fn=collator,
            pin_memory=self.trainer.strategy.root_device.type == "cuda",
            persistent_workers=workers > 0,
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )
        if workers:
            kwargs.update(
                multiprocessing_context=self.hparams.multiprocessing_context, prefetch_factor=2
            )
        if validation:
            return DataLoader(self.validation_dataset, shuffle=False, **kwargs)
        dataset = CollatedDataset(self.train_dataset, collator)
        world_size = getattr(self.trainer, "world_size", 1)
        batches = math.ceil(math.ceil(len(dataset) / world_size) / self.hparams.batch_size)
        limit = getattr(self.trainer, "limit_train_batches", 1.0)
        truncated = limit < batches if isinstance(limit, int) else limit < 1.0
        if truncated:
            raise ValueError(
                "Exact training resume requires complete DataLoader epochs; "
                "set trainer.limit_train_batches=1.0 or an integer covering all "
                f"{batches} training batches. Use trainer.max_steps for short runs."
            )
        if world_size > 1:
            sampler = StatefulDistributedSampler(
                dataset, num_replicas=world_size, rank=self.trainer.global_rank,
                shuffle=True, seed=self.hparams.seed,
            )
        else:
            sampler = RandomSampler(
                dataset, generator=torch.Generator().manual_seed(self.hparams.seed)
            )
        return CoFLStatefulDataLoader(dataset, sampler=sampler, snapshot_every_n_steps=1, **kwargs)

    def train_dataloader(self):
        if self._train_loader is None:
            self._train_loader = self._loader(validation=False)
            if self._pending_train_loader is not None:
                self._train_loader.load_state_dict(self._pending_train_loader)
                self._pending_train_loader = None
        return self._train_loader

    def val_dataloader(self):
        return self._loader(validation=True)

    def state_dict(self):
        snapshot = (
            self._pending_train_loader
            if self._train_loader is None
            else self._train_loader.state_dict()
        )
        world_size = getattr(self.trainer, "world_size", 1)
        snapshots = [None] * world_size
        if world_size > 1:
            torch.distributed.all_gather_object(snapshots, snapshot)
        else:
            snapshots[0] = snapshot
        return {"version": DATA_STATE_VERSION, "world_size": world_size, "rank_states": snapshots}

    def load_state_dict(self, state):
        world_size = getattr(self.trainer, "world_size", 1)
        if state.get("version") != DATA_STATE_VERSION:
            raise ValueError("Training dataloader checkpoint protocol is unsupported")
        if state["world_size"] != world_size or len(state["rank_states"]) != world_size:
            raise ValueError("Training dataloader resume requires the same world size")
        snapshot = state["rank_states"][getattr(self.trainer, "global_rank", 0)]
        if snapshot is None:
            raise ValueError("Checkpoint has no resumable training dataloader state")
        if self._train_loader is None:
            self._pending_train_loader = deepcopy(snapshot)
        else:
            self._train_loader.load_state_dict(snapshot)

    def example_images(self):
        """Read up to five original training views in the parent process."""
        if self.train_dataset is None:
            self.setup("fit")
        return [
            self.train_dataset[index]["image"] for index in range(min(5, len(self.train_dataset)))
        ]
