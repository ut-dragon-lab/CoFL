# Evaluation and benchmark protocols

CoFL uses offline image-field and image-navigation evaluation, producing
per-annotation results, aggregate scores, plots and a local HTML report.
CoFL-S uses closed-loop Habitat/VLN evaluation on R2R and RxR through the
runner shared by Studio and `python -m cofl.online`;
see the [online setup and batch CLI](app.md#online-environment-and-benchmark-configuration)
and [protocol definitions](app.md#shared-protocol-and-comparison-scope).
These tools alone do not establish reproduction of a published result.

The [R2R-1200](../benchmarks/r2r_1200/README.md) and
[RxR-1600](../benchmarks/rxr_1600/README.md) datasets supply fixed episode
IDs and prepared annotations for the custom subsets selected for this task.
Neither is the official full split. Model and runtime settings belong to the
[evaluation configurations](../configs/evaluation/README.md). Reference scores
and their evaluation conditions are documented separately in the
[R2R evaluation reference](../reproduce/r2r_1200.md) and
[RxR evaluation reference](../reproduce/rxr_1600.md). Those scores apply to the
selected subsets.

Native CoFL neural prediction entry points follow the shared
[`cofl-inference-v1` rules](inference.md). These include complete-grid
trajectory inference, scheduled integration with boundary projection, and
CoFL-S model-only HFOV conditioning and a ground-start radial floor. Offline
evaluation records `offline_policy_v2`; prior output directories cannot be
resumed under the new protocol.

Prepare each dataset under `prepared/r2r_1200` or `prepared/rxr_1600`, set scene
and interpreter paths in its `configs/evaluation/r2r_1200.yaml` or
`configs/evaluation/rxr_1600.yaml` runtime recipe, and supply the checkpoint
configured by the corresponding model evaluation YAML:

```bash
uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_r2r_online.yaml
uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_rxr_online.yaml
```

The first loads the 1,200 prepared R2R episodes. The second loads 1,600
RxR val_unseen **guide-English** episodes (`en-IN` and `en-US`).
Prepared directories contain data, not evaluation settings. RxR uses its native
79° camera, official guide GT trajectories and Landmark-RxR sub-instructions.
Its `oracle_source: landmark_rxr` and `oracle_unavailable: error` settings
require usable Landmark annotations; native timed and full instructions are
not fallbacks. The separate legacy `local/rxr-online.yaml` Studio recipe retains
the full 3,669 English episodes and its broader fallback behavior. Both canonical
online configurations use
`cofl-inference-v1` and benchmark protocol v6, with independently configured
checkpoints and separate output directories. These are online evaluation recipes; the offline
dataset evaluator below has a different scoring unit and protocol.

Online YAMLs define the checkpoint, benchmark recipe, output, episode selection
and run parameters. Each YAML resolves its relative paths against its own file;
explicit CLI options override the settings. For a custom data preparation
directory, update the three annotation paths in the corresponding runtime recipe. See the
[configuration index](../configs/evaluation/README.md),
[RxR assets and App setup](app.md#rxr-guide-english-setup) and
the [online CLI guide](app.md#headless-benchmark-runs-and-resume) for configuration
paths, overrides and resume behavior.

The headless online CLI can evaluate independent episodes in spawned environment
processes with one shared policy:

```bash
uv run --locked python -m cofl.online \
  --config configs/evaluation/cofl_s_r2r_online.yaml \
  --workers 2 --inference-batch-size 2 --worker-threads 1 --batch-wait-ms 2 \
  --output artifacts/online/cofl-s-r2r-1200-workers2
```

The R2R evaluation recipe fixes workers and batch size to 1; the command above
overrides both for an execution comparison. RxR's evaluation recipe sets both
to 16. The CLI defaults to `workers: 1`, `worker_threads: 1`,
`inference_batch_size: null` and `batch_wait_ms: 2.0`; evaluation YAMLs may override
these values. Multiple environment workers require `group_by_scene: true`.
The parent assigns one episode at a time, preferring scenes with no other active
worker and reusing each worker's loaded scene where possible. When all pending
scenes have workers, additional workers may process different episodes in the
same `.glb`. Workers that finish a short scene can help with the remaining ones.
The process count is capped by pending episodes, not distinct scenes. Each
process owns an independent simulator, including when scenes match.
The parent loads one policy, dynamically batches ready observations in its
inference thread, and alone commits results to the shared output.

The maximum batch size defaults to the effective environment-worker count;
`--inference-batch-size` can lower it. Full batches dispatch immediately, while
partial batches wait at most the configured additional `--batch-wait-ms` window.
Workers do not synchronize at a common environment step. Requests arriving during
inference queue for the next batch. The shared policy uses `device`, while each
Habitat instance uses the recipe's `gpu_id`. Extra simulators and larger batches
consume additional memory even though the model is loaded once. Compare 1, 2 and
4 workers on an identical cohort and inspect observed batch sizes and throughput;
these options do not imply a measured acceleration or linear scaling.

Batch shape can change floating-point accumulation, so fixed-frame comparisons
need stated numerical tolerances and complete closed-loop comparisons; bitwise
identity to batch size 1 is not guaranteed. Worker, CPU thread and batch settings
can change on a compatible v6 `--resume` after the previous process releases its
output lock. The inference protocol, clocks, grids, episode controls and selected
episode identities are preserved, with batching recorded as execution metadata. See the
[online CLI guide](app.md#headless-benchmark-runs-and-resume) for details. These
online scene workers are separate from the offline evaluator's data-loading
and scoring workers described below.

## Run a CoFL offline evaluation

Use the [locked uv environment](../README.md#installation) from the repository
root. The evaluator needs a native dataset and a complete CoFL checkpoint;
public dataset and checkpoint downloads are still pending. With those assets:

```bash
uv sync --locked
uv run --locked cofl evaluate --config configs/evaluation/cofl.yaml \
  --dataset datasets/cofl --checkpoint checkpoints/cofl/best.ckpt \
  --output artifacts/evaluation/cofl
```

The checkpoint contains the backbone weights and processor assets. Evaluation
does not require a model download, W&B authentication or Habitat, and does not
invoke the Lightning training framework.

For CPU evaluation, select the CPU dependency group on both commands and set
the evaluation device explicitly:

```bash
uv sync --locked --no-group cu124 --group cpu
uv run --locked --no-group cu124 --group cpu cofl evaluate \
  --config configs/evaluation/cofl.yaml --device cpu \
  --dataset datasets/cofl --checkpoint checkpoints/cofl/best.ckpt \
  --output artifacts/evaluation/cofl-cpu
```

| Recipe | Default split | Tasks | Batch size / loader workers |
| --- | --- | --- | --- |
| `configs/evaluation/cofl.yaml` | `val` | Field errors and image navigation | 16 / 8 |

The CoFL recipe selects all eligible annotations by default (`max_samples: null`),
with CUDA, `forkserver` workers, query chunks of 4096, FP32 inference, one
background scoring worker, seed 42, and 24 figures. Override YAML values with
underscore-separated flags:

```bash
uv run --locked cofl evaluate --config configs/evaluation/cofl.yaml \
  --max_samples 128 --batch_size 4 --num_workers 0 \
  --visualization_count 8 --output artifacts/evaluation/cofl-subset
```

A limited cohort is chosen without replacement using the seed, then sorted
into dataset order. The scoring unit is an annotation, so multiple instructions
for one observation remain separate samples. The recipe records the selected
cohort and whether the source is partial. Completing a selected subset does
not mean that the full source dataset was evaluated.

## Throughput and profiling

The evaluator encodes each batch once and reuses projected attention keys and
values across decoder chunks. Data readers select only the field, depth and
navigation arrays needed by the tasks, and cache read-only Zarr handles and
fixed query grids. CPU scoring runs in a standard thread pool while the next
batch is predicted. At most `score_workers + 1` scoring batches are retained;
results commit in dataset order on the main thread. Set `score_workers: 0`
for serial scoring. Increasing `num_workers` only changes data loading.
Fresh-run figures reuse selected field predictions; resume computes any
missing figure fields in batches.

`precision` accepts `float32` (default), `float16` (CUDA only), or `bfloat16`.
The latter two use PyTorch autocast and retain FP32 Fourier-coordinate math.
They can accelerate inference but change numerical predictions, so precision
is part of the frozen recipe and cannot change when resuming. Evaluate the
same cohort in a new directory when comparing precision settings:

```bash
uv run --locked cofl evaluate --config configs/evaluation/cofl.yaml \
  --dataset datasets/cofl --checkpoint checkpoints/cofl/best.ckpt \
  --precision float16 --max_samples 128 --visualization_count 0 \
  --output artifacts/evaluation/cofl-fp16-subset
```

Field reconstruction and navigation are different workloads. For a fully
valid 224 by 224 native field and the recipe's 100 by 100 navigation raster,
the combined task queries 60,176 points per sample. Navigation alone queries
10,000. Use `--tasks '["image_navigation"]'` for a navigation-only benchmark;
this deliberately omits field scores. Changing the navigation grid also
changes the rollout recipe, not just execution speed.

Every `executions.jsonl` entry includes `startup_seconds` and a `stages`
object: data wait, prediction, scoring, scoring wait, commit, evaluation loop,
and summary durations. Prediction includes encoding, all field queries and
the transfer back to CPU. Scoring is the sum of worker durations and overlaps
prediction; **stage durations must not be added to infer wall time**. The
`seconds` field measures the loop and summary, excluding startup,
figures and report generation. Throughput comparisons must use the same
cohort, checkpoint, tasks, grids, precision and timing boundary, with other
GPU workloads excluded.

For operator traces, set `profile_batches` to a small positive count. After
one warmup batch (zero for a one-batch cohort), PyTorch Profiler records up to
that many batches and writes `profile.json`, viewable in Perfetto. Profiling
adds overhead and should be disabled for throughput measurements. The trace
covers the main inference thread and its GPU operators. Thread-pool scoring
has separate wall-time counters; use serial scoring to include its operators
in the same trace:

```bash
uv run --locked cofl evaluate --config configs/evaluation/cofl.yaml \
  --dataset datasets/cofl --checkpoint checkpoints/cofl/best.ckpt \
  --max_samples 64 --score_workers 0 --profile_batches 3 \
  --visualization_count 0 --output artifacts/evaluation/cofl-profile
```

The output directory is bound to its protocol fingerprint. A different
source, checkpoint, cohort or scoring recipe requires a separate directory.

## Field scores

CoFL fields are scored at image pixel centers on the native dataset grid.
The sample's field mask is intersected with the valid query domain. The rollout
raster described below is a separate grid and does not replace this scoring
grid.

| Metric | Definition |
| --- | --- |
| `vector_l2_mean` | Mean Euclidean vector difference over selected cells |
| `magnitude_error_mean` | Mean absolute difference of vector magnitudes over selected cells |
| `angular_error_deg_mean` | Mean angular error where the target has a direction |
| `directional_accuracy_30deg` | Fraction of directional targets with angular error strictly below 30 degrees |

The directional threshold is `1e-8` in native normalized field units. Targets
at or below it are excluded from angular scores and retained in vector and
magnitude errors. A prediction at or below it against a directional target
receives 180 degrees of error. Empty cohorts produce `null` with explicit
counts. Masked-out nonfinite values are ignored; invalid selected ground truth
raises an error. A nonfinite model prediction is recorded as a failed sample.

The `image_field_v1` components are image right/down, with vector errors in
`image_fraction_per_policy_time`. Policy time is an integration parameter.
The low-level NumPy API also supports direct comparison of arrays shaped
`(..., 2)`:

```python
import numpy as np
from cofl.evaluation import evaluate_field

metrics = evaluate_field(
    np.moveaxis(predicted_field, 0, -1),
    np.moveaxis(sample["field"], 0, -1),
    profile=sample["profile"],
    geometry=sample["geometry"],
    mask=sample["mask"],
)
```

## CoFL image navigation

This task requires an independent goal and a reference trajectory in
`annotation_extras.goal` and `annotation_extras.trajectory_state`, plus
`observation_extras.navigation_mask`. The navigation mask marks walkable cells
with `True`; it is distinct from the field scoring mask. Missing labels fail
validation. The start is the reference trajectory's first point. The goal is
used for scoring and does not steer the rollout or trigger an early stop.
When present, `annotation_extras.trajectory_length` or `trajectory_valid`
selects the nonempty valid trajectory prefix; both declarations must agree.

The CoFL YAML recipe uses the shared inference implementation to query a
100 by 100 raster including both endpoints of
`[0, 1]` on each axis. It reads that raster with bilinear interpolation, border
padding, and `align_corners=True`. The following schedule applies 100 Euler
steps, retaining the initial point and every updated point:

```text
t[k] = k / 99                              # k = 0, ..., 99
command = field(position) / (1 - t[k] + 0.5 * t[k]**10)
position = clamp(position + 0.01 * command, 0, 1)
```

The schedule, step count, and policy-time increment are explicit configuration
values. There is no early goal or zero-field stopping. `clamped_steps` records
boundary clamping. Collision metrics score the returned, clamped path; a
clamped update alone is not an obstacle collision.
`ImageTrajectoryConfig` also defaults to a 100 by 100 raster. The resolved
recipe is saved with each run.

This is the image branch of the same algorithm used by App and checkpoint
rollout. CoFL-S online planning uses the corresponding ground branch with
bias `0.2`, the actual camera's polar/Cartesian geometry and sector boundary
projection. Its controller and Habitat metrics remain a separate task; see
the [complete inference rules](inference.md).

| Score | Definition and unit |
| --- | --- |
| FGE (`fge`) | Euclidean distance from the predicted endpoint to the independent goal, in image fractions |
| CR (`cr`) | Per-path unsafe indicator: obstacle contact or an out-of-bounds point; the aggregate mean is the unsafe-path fraction |
| PLR (`plr`) | Predicted/reference polyline-length ratio after independent 101-point arc-length resampling |
| Curv (`curv`) | Mean absolute wrapped heading change between nonzero resampled segments, in radians per turn |

Obstacle contact checks every grid cell touched by the original path segments,
including contacts along cell edges and corners. This supercover test avoids
missing a narrow obstacle between resampled points. Out-of-bounds points are
reported independently and count as unsafe; the scoring function does not
clip them. This checks a point path, without a robot footprint or metric
clearance model.

Both lengths in PLR come from the resampled paths. A reference length below
`1e-8` gives `plr: null` and `degenerate_reference: true`. Segments at or below
`1e-8` have no heading; a path with fewer than two nonzero resampled segments
has `curv: null` and `turn_count: 0`. Undefined scores do not become perfect
zeros. Failed model predictions remain in the cohort with `cr: 1` and the
other navigation scores undefined.

## Results, aggregation, and resume

`results.sqlite` is the authoritative result store. Each inference batch is
committed in one SQLite transaction, with unique sample IDs. Repeating the
same command skips committed samples, including recorded failures. Resume
requires matching checkpoint bytes, source code, dataset manifest identity,
selected cohort, and mathematical recipe. The dataset fingerprint covers the
native manifest structure; evaluation does not rehash every dense payload.
Keep dataset contents immutable during a run.

`protocol.json` and `recipe.json` carry the resolved inference rules and
`cofl-inference-v1` identity. The recipe also identifies which tasks apply
trajectory integration; field-only tasks do not generate an unused trajectory.

Batch size, device, worker count, and query chunk size are recorded for each
execution and can be adjusted on resume. Floating-point results can vary
slightly with these execution settings. Use one writer process per output
directory.

| Output | Purpose |
| --- | --- |
| `results.sqlite` | Durable per-sample metrics, diagnostics, and protocol identity |
| `protocol.json`, `recipe.json`, `config.json` | Frozen scoring identity, selection/settings, and current execution configuration |
| `results.jsonl`, `metrics.csv` | Rebuildable record and tabular exports |
| `summary.json` | Overall, per-scene, and per-category aggregates with coverage counts |
| `report.html`, `figures/` | Local report and plots; keep the figure directory beside the HTML |
| `executions.jsonl` | Execution settings, elapsed time, and completion status |

`summary.json` wraps the aggregate object as follows; the CLI prints the inner
summary without the large scene/category tables:

```text
protocol_sha256
sample_count
summary
  selected_samples, evaluated_samples, complete
  overall, by_scene, by_category
```

Field headline scores pool selected vectors: vector/magnitude errors use valid
cell counts, and angular scores use directional-target counts. Their
`sample_statistics` also report equal-annotation mean, sample standard
deviation (`ddof=1`), count, and undefined count. Navigation scores use equal
annotation weights. Action accuracy pools labeled predictions. The same
aggregation applies within scene and category groups.

`successful_samples` means that scoring completed with finite predictions; it
is not a navigation success rate. `complete` means every selected annotation
has a committed record, including failures. Reports expose failed and
undefined counts so they cannot silently disappear from averages. Exports can
be rebuilt from SQLite after interruption, even if an exported JSONL file was
truncated.

## Other runtime and protocol tools

`cofl.runtime.rollout_image_field` and `rollout_sector_field` implement the shared
[`cofl-inference-v1` integration rules](inference.md) through a NumPy callable
interface: time-scaled Euler updates, boundary projection and all configured
steps. Inference callers supply the canonical 100 by 100 `RasterField` for
bilinear sampling; these integration functions do not query the decoder. The sector
version converts metric forward/left positions into sector queries, then
scales returned Cartesian vectors into metric displacement. It never adds
Cartesian vectors directly to polar coordinates. These local integration APIs
do not refresh observations or run a closed-loop navigation controller.

The generic `evaluate_image_trajectory` and `evaluate_ground_trajectory` APIs
compare endpoint positions, original polyline lengths, arc-resampled point
distances, and heading changes in degrees. Their endpoint reference and length
conventions differ from the task-specific FGE/PLR/Curv definitions above.
Ground `euclidean_goal_reached` is an optional distance test without STOP
semantics, not VLN SR or SPL.

```bash
uv run --locked cofl benchmark synthetic-ground --output artifacts/evaluation/synthetic --seed 0
```

This small execution demo runs three obstacle-free trajectories with an
analytic, known-goal field. Planning runs at 5 Hz; pure pursuit and a unicycle
plant run at 20 Hz. It writes states, commands, timestamps, and Euclidean
endpoint summaries. It requires a fresh output directory and does not load a
learned policy or report a paper benchmark.

`EvaluationProtocol` and `TimingSpec` also describe fixed-step, teleport, and
wall-time execution with explicit planner/controller/plant timing. Instruction
modes distinguish full instructions, provided annotations, oracle assistance,
learned progress, and known goals. These metadata types do not supply the
corresponding environment runner. The offline CLI uses `provided_annotation`.
