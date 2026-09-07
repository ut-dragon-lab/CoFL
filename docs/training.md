# Training

CoFL and CoFL-S use PyTorch Lightning for training, validation, checkpoint
selection and logging. `CoFLModule` contains the policy and losses;
`CoFLDataModule` reads native datasets and samples field supervision.
`ModelConfig` and `OptimizerConfig` describe model and optimizer arguments.
LightningCLI parses the YAML configuration and command-line overrides.

## Install and launch

Follow the [uv installation instructions](../README.md#installation), then run
from the repository root. The Linux x86_64 environment pins Python 3.12.12,
PyTorch 2.5.1, Torchvision 0.20.1, TorchData 0.10.0, Lightning 2.5.5,
TorchMetrics 1.8.2 and Transformers 4.57.1. Its default `dev` and `cu124` dependency groups include
the complete training runtime.

```bash
uv sync --locked
uv run --locked wandb login
uv run --locked cofl train --config configs/cofl_formal.yaml
uv run --locked cofl train --config configs/cofl_s_formal.yaml
```

Run one command per available GPU. The supplied recipes read `datasets/cofl`
and `datasets/cofl-s` and write to `outputs/cofl` and `outputs/cofl-s`.
Prepared [dataset downloads](data-release.md) and release checkpoints are still
pending. Supply the native datasets before launching; synchronization does not
download them. A dataset root can be one native dataset or a complete collection
of shards; validation uses a held-out split in the same root.

```bash
uv run --locked cofl data validate datasets/cofl
uv run --locked cofl train --config configs/cofl_formal.yaml --data.dataset /path/to/cofl
uv run --locked cofl train --help
uv run --locked cofl train --config configs/cofl_formal.yaml --print_config
```

`uv run --locked cofl train` forwards its arguments to LightningCLI's `fit` command. Settings
use standard dotted names, for example `--data.batch_size 16` or
`--trainer.accelerator cpu`. Configuration files and explicit arguments follow
jsonargparse's left-to-right override order. To choose a different output
location, edit `trainer.default_root_dir` and the logger's `save_dir` in a
copy of the recipe. The checkpoint callbacks use Lightning's logger directory.

For CPU execution, use the `cpu` dependency group for synchronization and
commands, and override the recipe's accelerator:

```bash
uv sync --locked --no-group cu124 --group cpu
uv run --locked --no-group cu124 --group cpu cofl train \
  --config configs/cofl_formal.yaml --trainer.accelerator cpu
```

Initial training loads pretrained SigLIP assets. Set
`model.model.local_files_only: true` to require an existing cache, or set
`model.model.siglip_model_name` to a local pretrained directory. Resuming a
CoFL checkpoint reconstructs the backbone from its embedded assets.

## SigLIP encoder optimization

Both formal recipes train the complete vision and text encoders. The two towers
have independent layer counts and learning rates:

```yaml
model:
  model:
    vision_unfreeze_last_n_layers: -1
    text_unfreeze_last_n_layers: -1
  optimizer:
    learning_rate: 1.0e-4
    vision_learning_rate: 1.0e-5
    text_learning_rate: 1.0e-5
```

For each tower, `0` freezes all parameters, a positive count trains only the
last N Transformer blocks, and `-1` trains the entire tower, including embeddings
and output layers. A positive count cannot exceed that tower's depth. Partial
unfreezing leaves the final normalization and pooling/projection layers frozen.

`learning_rate` controls the downstream policy. Each tower's learning rate
controls only its trainable parameters; `null` inherits `learning_rate`. A frozen
tower has no optimizer group, so its learning rate takes effect only after it
is unfrozen. All active groups share the configured warmup and decay schedule
and weight decay. The learning-rate monitor names them `policy`, `siglip_vision`
and `siglip_text`.

For example, add `--model.model.text_unfreeze_last_n_layers 2
--model.optimizer.text_learning_rate 5e-6` to a training command to train only
the final two text blocks. Set both layer counts to `0` for a frozen baseline.
Python `ModelConfig` defaults to both towers frozen for compatibility with
earlier runs; the formal YAML files explicitly train both towers.

Checkpoint assets and hyperparameters preserve the independent settings.
Exact resume with `--ckpt_path` requires the saved unfreezing and optimizer
configuration. Start a new run when changing them. Earlier frozen checkpoints
without these fields still load and resume with the frozen defaults.

## Formal recipes

| Setting | CoFL | CoFL-S |
| --- | --- | --- |
| Profile | `image_field_v1` | `ground_sector_v1` |
| Epochs | 10 | 10 |
| Batch size / queries per sample | 32 / 1,000 | 32 / 1,000 |
| Width / heads / fusion layers / decoder layers | 768 / 8 / 4 / 2 | 768 / 8 / 4 / 2 |
| Learning rate / weight decay | 1e-4 / 1e-5 | 1e-4 / 1e-5 |
| Schedule / warmup | cosine / 40,000 steps | cosine / 40,000 steps |
| Gradient norm clipping | 0.5 | 0.5 |
| Depth | RGB | metric RGB-D, bicubic resize |
| Action loss weight | unused | 0.3 |
| Validation split / sample limit | `val` / all | `val_unseen` / 8,192 |

Both recipes use seed 42, four data workers, brightness/contrast jitter of
0.15 and float32 training. `seed_everything` controls Lightning's global seed;
`data.seed` controls data sampling and the fixed validation subset. Set both
when changing the experiment seed.

Data workers use `data.multiprocessing_context: forkserver` by default; `spawn`
is also supported. These follow PyTorch's supported start methods for CUDA
multiprocessing. See the [PyTorch multiprocessing guidance](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/multiprocessing.rst#cuda-in-multiprocessing).

Use `trainer.max_epochs` or `trainer.max_steps` for the training horizon.
These have Lightning's standard meaning; there is no separate execution cap.
Choose the complete horizon before starting: resume requires the saved horizon
so that the learning-rate schedule continues with the same definition.
The supplied validation interval is 5,000 training batches across epoch
boundaries (`check_val_every_n_epoch: null`). With one optimizer and no gradient
accumulation, a training batch is one optimizer step.

The CoFL-S validation subset is fixed by its seed. Its 8,192-sample limit is a
training monitor, not a full validation benchmark. Set
`data.validation_max_samples: null` for the complete eligible split.
Partial datasets require `data.allow_partial_dataset: true`.

For CoFL image-field and image-trajectory evaluation, use
`uv run --locked cofl evaluate --config configs/evaluation/cofl.yaml`, supplying
the selected checkpoint. This recipe scores all eligible annotations and native
field cells by default. Training validation uses sampled queries and may use a
smaller annotation subset, so its monitor values follow a different sampling
protocol.

For CoFL-S evaluation, prepare the R2R-1200 or RxR-1600 selected subset and the
Habitat assets using the [evaluation configuration guide](../configs/evaluation/README.md),
then run the corresponding online recipe:

```bash
uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_r2r_online.yaml
uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_rxr_online.yaml
```

These recipes measure executed navigation on fixed custom subsets selected for
this task; neither subset is the official full split. CoFL-S training validation
continues to monitor sampled field and action losses. See
[evaluation](benchmarks.md) for metric definitions and result scope.

The formal training recipes use one GPU. Lightning's default
distributed sampler can repeat samples to balance validation across devices;
use a single device for validation that must visit each sample once.

## Supervision and geometry

Training retains the final short batch. Continuous image queries use
stratified normalized image coordinates and bilinear readout with half-pixel
sampling. Sector queries sample physical area uniformly using a radius
proportional to `sqrt(U)`; their targets are interpolated from the Cartesian
field grid. `data.query_sampling: grid` samples valid stored grid locations.

Ground decoder geometry is explicit in `model.model.r_max_m`,
`model.model.hfov_rad` and `model.model.normalization_scale_m`. Radius and
vector normalization must match the dataset; samples supply their camera HFOV.
Field output components remain body-frame forward/left vectors.

Metric depth is valid when finite, above `depth_min_m`, within `r_max_m`, and
allowed by the stored mask. Invalid depth is zeroed, normalized by `r_max_m`,
resized with `depth_interpolation` and clamped to `[0, 1]`. Validity masks use
nearest-neighbor resizing.

`model.field_loss` selects the same method-independent objective for either
policy:

```yaml
model:
  field_loss:
    magnitude_loss: mse
    magnitude_normalization: absolute
    direction_weight: 1.0
    magnitude_weight: 0.5
    relative_epsilon: 0.1
    direction_epsilon: 1.0e-8
    direction_min_target_magnitude: 1.0e-8
```

Direction loss averages `1 - cosine(prediction, target)` over targets above
the magnitude threshold. Zero targets supervise magnitude only. Magnitude
loss applies MSE or L1 to the difference of vector norms, not their components.
Relative normalization divides that difference by the target norm plus
`relative_epsilon` before applying MSE or L1. Masked vectors are removed before
arithmetic; supervised values must be finite.

CoFL-S adds action cross-entropy weighted by `model.action_loss_weight`.
Annotation targets 0–3 are supervised; null targets use the ignore index `-100`.

## Losses by query region

Both methods automatically log regional diagnostics under
`train/regions/<region>/<metric>` and `val/regions/<region>/<metric>`.
For example, compare `val/regions/free/direction` with
`val/regions/obstacle/direction`, and compare their `query_fraction` and
`loss_fraction` to see whether one region dominates supervision.

The regions come from stored navigation masks at the sampled query positions:

| Region | CoFL | CoFL-S |
| --- | --- | --- |
| `free` | Inside `navigation_mask` | Inside `bev_mask`: visible and walkable |
| `obstacle` | Outside `navigation_mask` | Outside `bev_walkable`: non-walkable |
| `occluded` | Not logged | Walkable but outside `bev_mask` |
| `unknown` | Navigation mask unavailable | Available masks cannot determine the region |
| `gt_obstacle` | Not logged | Union of `obstacle` and `occluded` |

CoFL-S also accepts imported `source_bev_mask` and `source_bev_walkable`.
Its `gt_obstacle` matches the GT's complement of visible free space where both
masks are available; it includes walkable space excluded by visibility and
rasterized sector boundaries. It is an overlapping aggregate, so do not add
it to the other regions when computing totals. CoFL's image navigation mask
does not provide a separate visibility distinction.

Region masks use nearest-cell readout in the same coordinate system as field
supervision. Continuous field targets remain bilinearly interpolated, so a
query at a mask boundary receives one region label even if its target blends
adjacent cells. The annotation supervision `mask` does not define occupancy.
If masks are missing, queries without an unambiguous label stay `unknown`;
their losses still contribute to standard training and to global metrics.
The optional training ablation below requires known region labels instead.

| Metric | Meaning |
| --- | --- |
| `direction` | Mean direction loss over eligible targets in this region |
| `magnitude` | Mean configured magnitude loss over queries in this region |
| `loss` | Weighted sum of this region's direction and magnitude means |
| `loss_contribution` | This region's contribution to the global field loss |
| `loss_fraction` | Contribution divided by the global field loss |
| `queries` | Number of sampled queries in this region |
| `query_fraction` | Region query count divided by all sampled queries |
| `directional_count` | Region targets above the direction magnitude threshold |

`direction` and `magnitude` use the same loss settings as the objective,
including relative magnitude normalization when configured. Regional `loss`
measures error within the region; it does not account for how often that
region was sampled. `loss_contribution` instead divides regional error sums
by the corresponding **global** counts before applying the loss weights:

```text
loss_contribution = direction_weight * region_direction_error_sum / global_directional_count
                 + magnitude_weight * region_magnitude_error_sum / global_query_count
```

The disjoint regions' contributions sum to the full-query field loss, up to
floating-point roundoff: `train/loss` for ordinary training, `train/full/loss`
for the obstacle-mask ablation below, and `val/loss` for validation. They exclude
CoFL-S action classification loss, which is included in the corresponding
`total`. When the full-query field loss is zero,
`loss_fraction` is zero. An empty region has zero loss and zero counts;
this means it had no supervision, not that its predictions were perfect.
Check `directional_count` too: a region with zero-vector targets can supervise
magnitude while having no direction supervision.

Training diagnostics describe the current batch on each rank, following the
existing per-step training logs; they are not a synchronized epoch summary.
Validation accumulates error sums and counts over all validation batches and
distributed ranks before computing means and fractions. This preserves correct
weighting for short batches and for regions with different query counts.

Regional diagnostics themselves do not change query sampling, random-number
streams, loss weights, optimizer updates or checkpoint selection. No
configuration change is needed to enable them, and their metric states are not
stored in checkpoints. Regional history cannot be reconstructed from old
scalar logs. The checkpoint format has separately changed to support exact
resume; runs made with the old format must start fresh as described below.

## Training without GT obstacle supervision

Set `model.mask_gt_obstacle: true` to remove GT obstacle queries from the
training field loss. For CoFL this removes queries outside `navigation_mask`.
For CoFL-S it removes both `obstacle` and `occluded`, retaining only the visible,
walkable `free` region. This is a supervision ablation: query positions and
their random draws stay the same, and CoFL-S action classification keeps its
existing loss and weight. Validation still evaluates every sampled query with
the original objective. Unknown training-region labels cause an error when
masking is enabled, because missing masks cannot establish which queries to
retain. The default is `model.mask_gt_obstacle: false`.

The overlay recipes enable this ablation and give each method a separate
output directory and W&B run name. Apply the overlay **after** the matching
formal recipe:

```bash
uv run --locked cofl train \
  --config configs/cofl_formal.yaml \
  --config configs/ablations/cofl_mask_gt_obstacle.yaml \
  --data.dataset /path/to/cofl

uv run --locked cofl train \
  --config configs/cofl_s_formal.yaml \
  --config configs/ablations/cofl_s_mask_gt_obstacle.yaml \
  --data.dataset /path/to/cofl-s
```

Select an available GPU using the usual `CUDA_VISIBLE_DEVICES` environment
variable before launching. Start these runs fresh without `--ckpt_path` or a
previous W&B run ID. Keep the baseline dataset, seeds, optimizer, batch size,
query sampling, training horizon and validation protocol the same. The
overlays inherit the current formal settings, including action weight and
checkpoint/early-stopping monitor `val/total`; they only add masking and a
separate output destination. A checkpoint from an unmasked run cannot resume
into the masked objective.

`model.masked_loss_normalization` controls the denominators after masking:

| Value | Direction denominator | Magnitude denominator |
| --- | --- | --- |
| `all` (default and supplied overlays) | All originally eligible directional targets | All sampled queries |
| `selected` | Retained eligible directional targets | Retained queries |

Both settings remove the excluded error terms from the numerator. `all`
preserves each retained query's original contribution. As fewer queries
supervise the field, its gradient can become smaller relative to CoFL-S action
loss. `selected` averages over the retained supervision and therefore also
rescales the field gradient; the direction and magnitude scale changes can
differ because they use separate counts. An empty retained region gives zero
field loss with either setting. Treat normalization as part of the experiment,
and use a separate output/run when comparing the two settings.

With masking enabled, the training logs distinguish the optimized objective
from the full-query reference:

| Log | Meaning |
| --- | --- |
| `train/direction`, `train/magnitude`, `train/loss`, `train/total` | Actual masked objective used for optimization |
| `train/full/direction`, `train/full/magnitude`, `train/full/loss`, `train/full/total` | Original objective evaluated on all sampled queries |
| `train/supervision/queries`, `train/supervision/directional_count` | Retained query count and retained eligible directional-target count |
| `train/supervision/query_fraction` | Fraction of sampled queries retained for field supervision |
| `train/supervision/directional_fraction` | Fraction of originally eligible directional targets retained |
| `train/regions/<region>/*` | Region diagnostics over all queries, including excluded regions |

Regional `loss_contribution` and `loss_fraction` refer to `train/full/loss`
during the ablation, so their partition remains interpretable. Validation
keys keep their original definitions. Use `val/regions/free/loss` as the main
comparison for whether free-space field prediction improves, with its
direction and magnitude components alongside `val/loss`, obstacle-region
errors and CoFL-S action metrics to show tradeoffs. A lower `train/loss` alone
does not establish an improvement: removing supervision and choosing its
normalization directly change that scalar.

## Validation, checkpointing and resume

Lightning performs its standard sanity validation before training. The formal
recipes use `ModelCheckpoint` to retain the lowest `val/total` and save the
latest state every 1,000 optimizer steps. `WarmupEarlyStopping` starts patience
checks after its configured `start_step`, which matches the warmup in the
recipes. Patience counts
validation checks; `min_delta` controls the improvement needed to reset it.
Best-checkpoint selection independently tracks any lower monitored value.

Each run retains two full checkpoints. With the supplied W&B logger, Lightning
uses `outputs/<method>/CoFL/<run-id>/checkpoints/`:

| File | Purpose |
| --- | --- |
| `best.ckpt` | Lowest monitored validation loss |
| `last.ckpt` | Latest periodic save, also updated when training ends normally |

Both include the full policy, frozen SigLIP weights, model configurations and
processor/tokenizer assets. Lightning also saves optimizer, scheduler, loop
and callback state. Checkpoint schema **2** additionally preserves the training
data position and random states needed to continue inside an epoch:

- The training loader uses TorchData's `StatefulDataLoader`, preserving sampler
  progress and worker snapshots instead of starting a new traversal at the
  beginning of the dataset.
- Each training collator saves its own NumPy and CPU Torch random streams for
  query sampling and image augmentation. These streams are independent of
  model randomness, including when `num_workers: 0`.
- Every training rank saves its Python, NumPy and CPU Torch random states;
  CUDA training additionally saves CUDA random states. They are restored after model and loader setup so that
  subsequent stochastic model operations, including dropout, continue from the
  checkpoint.

The default progress bar also restores the epoch label, total batch count and
saved batch position. Subsequent epochs start their displays at zero normally.
This restores the display from Lightning's saved progress without changing
the loop counters in the progress callback. Checkpoint restoration also handles
pending validation and completed epoch boundaries, so validation is not skipped
and the next epoch starts at the correct batch.

`FinalCheckpoint` updates `last.ckpt` at normal training completion, including
early stopping. Periodic checkpoints represent completed optimizer updates.
A crash or Ctrl+C resumes from the most recent completed save; Ctrl+C does
not trigger an additional checkpoint, so updates since that save are replayed.
Custom checkpoints taken during an unfinished training batch, partway through
validation, or before a gradient accumulation window reaches an optimizer
update are rejected for training resume. Use the standard checkpoint callbacks
at completed optimizer updates or validation end.

**Breaking change:** schema **1** checkpoints are rejected by the current
checkpoint loader, including for inference. They lack the state required by
this resume protocol, and no compatibility or migration path is provided.
Start a new run with the updated code and no `--ckpt_path`. Use a new output
directory and W&B run ID to keep the new training history separate. A fresh
run initializes SigLIP from the configured pretrained assets as usual.

To resume a **schema 2** run, use its saved resolved configuration, an explicit
checkpoint file and the same W&B run ID copied from its URL:

```bash
uv run --locked cofl train --config outputs/cofl/config.yaml \
  --ckpt_path "outputs/cofl/CoFL/<run-id>/checkpoints/last.ckpt" \
  --trainer.logger.init_args.id "<run-id>" --trainer.logger.init_args.resume allow
```

The model is reconstructed from that checkpoint before Lightning restores the
training state; no pretrained-model directory, Hugging Face cache or model
network request is needed. The installed runtime and training dataset are
still required.

Resume validates the saved protocol and rejects incompatible changes. Keep
the model, loss and optimizer configuration; dataset revision and validation
subset; batch sizes, worker count and multiprocessing context; query sampling,
augmentation and seeds; device type, world size, precision and gradient accumulation;
gradient clipping; training horizon and batch limits; and validation schedule
fixed. Changing `max_steps` or `max_epochs` to extend an existing run is not
supported by this exact-resume protocol. Use the original planned horizon
when restarting an interrupted run.
Training must traverse the full loader each epoch (`limit_train_batches: 1.0`);
truncating an epoch with `limit_train_batches` is not supported. Use `max_steps`
to set a short training horizon instead.

The saved data and random states support continuation of the same sequence
of samples, queries and augmentations under the same runtime and protocol.
Keep the hardware, software and selected kernels unchanged when comparing
numerical results. GPU kernels can be nondeterministic; saving their random
states does not guarantee bitwise equality across hardware or environment
changes.

Inference uses either checkpoint without a separate model configuration:

```python
from cofl.training.policy import load_policy

policy, model_config = load_policy("outputs/cofl/CoFL/<run-id>/checkpoints/best.ckpt", device="cuda")
```

## Weights & Biases

The recipes configure `lightning.pytorch.loggers.WandbLogger` with project
`CoFL`, online logging and separate method names. W&B chooses the logged-in
account by default; set `trainer.logger.init_args.entity` for a team. The
default logger generates a run ID at runtime, which is not written back into
the saved CLI configuration. To continue the same online run, pass its `id`
and `resume: allow` as shown above, or set them in your configuration. A new
experiment should use a new ID.

Set `trainer.logger.init_args.offline: true` to write SDK data locally for
later `uv run --locked wandb sync`, or set `trainer.logger: false` and remove `LearningRateMonitor` from the
callback list to disable logging. Offline
sessions do not provide the same server-side resume behavior as online runs.

WandbLogger handles model metrics, hyperparameters and media. The W&B SDK
collects available hardware utilization, memory, environment metadata and
console output. Available sensors depend on the runtime. `log_model: false`
keeps checkpoints local; uploading model artifacts is an explicit logger
setting. Training and validation charts follow Lightning's global step.
`LearningRateMonitor` records the optimizer learning rate each step.

These configurations define training runs. Published navigation results still
require the assets and execution protocols in [benchmarks](benchmarks.md).
