# Evaluation configurations

Run commands from the CoFL repository root. CoFL-S uses online Habitat
navigation evaluation on R2R and RxR. CoFL uses offline image-field and
image-navigation evaluation on prepared native data.
R2R-1200 and RxR-1600 are fixed custom subsets selected for this task; neither
is the official full split.

| Configuration | Purpose |
| --- | --- |
| [`cofl_s_r2r_online.yaml`](cofl_s_r2r_online.yaml) | R2R online model, output, execution settings and `RunParameters` |
| [`r2r_1200.yaml`](r2r_1200.yaml) | R2R-1200 data paths, scene/interpreter paths, oracle options and simulator sensors |
| [`cofl_s_rxr_online.yaml`](cofl_s_rxr_online.yaml) | RxR online model, output, execution settings and `RunParameters` for RxR-1600 |
| [`rxr_1600.yaml`](rxr_1600.yaml) | RxR-1600 data paths, scene/interpreter paths, strict Landmark oracle and simulator sensors |
| [`cofl.yaml`](cofl.yaml) | Offline CoFL image fields and image navigation on native `val` data |

## R2R online

1. Follow the [R2R-1200 data preparation guide](../../benchmarks/r2r_1200/README.md)
   to create the five data outputs under `prepared/r2r_1200`.
2. Supply the model checkpoint and a compatible Habitat environment with
   Matterport3D `.glb` scenes and `.navmesh` files. Edit `scenes_dir` and
   `habitat_python` in `r2r_1200.yaml` for your installation. Their defaults are
   `data/scene_datasets` and `.habitat/bin/python` under the repository root.
3. Run the evaluation:

```bash
uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_r2r_online.yaml
```

The evaluation YAML points to `benchmark: r2r_1200.yaml`. That runtime recipe
contains an `online` block referencing `selected_episodes.json.gz`,
`selected_gt.json.gz` and `selected_fgr2r.json`, plus the `SIMULATOR` sensor
configuration in the same file. The prepared directory supplies data only;
no evaluation YAML is generated or inherited from it.

The default checkpoint is
`checkpoints/cofl-s/best.ckpt`, and results go to
`artifacts/online/cofl-s-r2r-1200`. Override the checkpoint location with
`--checkpoint /path/to/best.ckpt`. To evaluate the same reference model, use
the matching checkpoint identity in the
[evaluation reference](../../reproduce/r2r_1200.md); public weights are still pending.

Each YAML resolves relative paths against its own directory. Explicit CLI
options override YAML settings, with CLI paths resolved from the working
directory. For a custom data preparation directory, edit the three annotation
paths in `r2r_1200.yaml`; there is no prepared `evaluation.yaml` to select.

Append `--resume` to continue the same compatible result directory. For a smoke
run use `--count 1 --output artifacts/online/r2r_1200_smoke`. Episode IDs are
preserved, but CLI indices refer to the prepared dataset's `0..1199` sequence.

The shipped R2R evaluation uses protocol `cofl-s-vln-benchmark-v6`,
`cofl-inference-v1` and metric recipe `vlnce_official_core_v1`. It fixes FGR2R
oracle progress, single-frame RGB-D input, 5/50/30/100 Hz
planner/controller/sensor/plant clocks, 500 planner predictions and a
120-second cap. Navigation success requires explicit STOP and navmesh distance
strictly below 3 m. Full `RunParameters` live in `cofl_s_r2r_online.yaml`; camera,
scene and oracle settings live in `r2r_1200.yaml`.

The default execution is eager with 16 environment workers and batch size 16.
For a parallel-execution comparison, explicitly override both `--workers` and
`--inference-batch-size` and use a separate output directory. See the
[online CLI guide](../../docs/app.md#headless-benchmark-runs-and-resume) for options
and resume compatibility.

## RxR online

RxR-1600 supplies identities and annotations for the selected subset.
Reference scores and their evaluation conditions belong to the separate
[RxR-1600 evaluation reference](../../reproduce/rxr_1600.md).

1. Follow the [RxR-1600 data preparation guide](../../benchmarks/rxr_1600/README.md)
   to create the five annotation outputs under `prepared/rxr_1600`.
2. Supply the checkpoint, Matterport3D `.glb` and `.navmesh` assets, and a
   compatible separate Habitat environment. Set `scenes_dir` and
   `habitat_python` in `rxr_1600.yaml`; the defaults are `data/scene_datasets`
   and `.habitat/bin/python` under the repository root.
3. Run the evaluation:

```bash
uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_rxr_online.yaml
```

The evaluation YAML references `benchmark: rxr_1600.yaml`, which contains
the `online` data/runtime paths and `SIMULATOR` configuration. It evaluates
1,600 prepared guide-English episodes, using `oracle_source: landmark_rxr`
and `oracle_unavailable: error`. Missing or unusable Landmark annotations
raise an error; native timed instructions and full instructions are not fallbacks.
The older `local/rxr-online.yaml` remains a separate full-English Studio recipe.

The default checkpoint is `checkpoints/cofl-s/best.ckpt`;
use `--checkpoint` to override its location. Public checkpoint downloads remain
pending. Results go to `artifacts/online/cofl-s-rxr-1600`, with 16 environment
workers and a maximum inference batch size of 16. To run serially, append
`--workers 1 --inference-batch-size 1`; execution settings belong to this
evaluation YAML, not the dataset manifest.

RxR keeps 640 × 480 RGB/depth sensors, a 79° HFOV and 0.88 m sensor height.
It uses protocol v6, `cofl-inference-v1`, metric recipe
`vlnce_official_core_v1`, oracle progress lookahead 6, 5/50/30/100 Hz clocks,
500 planner predictions and a 120-second cap. Complete model/control settings
are in `cofl_s_rxr_online.yaml`; scene, sensor and oracle settings are in
`rxr_1600.yaml`. See [RxR setup](../../docs/app.md#rxr-guide-english-setup).

For a custom preparation directory, update the three annotation paths in
`rxr_1600.yaml`. Relative paths and CLI overrides follow the same rules as R2R.
Prepared indices are `0..1599`; original IDs and source indices are recorded
in `episode_index_map.json`. Use a separate output for a smoke run, for example
`--count 1 --workers 1 --inference-batch-size 1 --output artifacts/online/rxr_1600_smoke`.
Append `--resume` only for a compatible result directory.

## CoFL offline evaluation

`cofl.yaml` evaluates CoFL image fields and image navigation on native `val`
data. Supply the dataset and complete checkpoint, then run:

```bash
uv run --locked cofl evaluate --config configs/evaluation/cofl.yaml
```

Its dataset, checkpoint and result paths are resolved from the working directory.
Use `--dataset`, `--checkpoint` and `--output` to override them. The CoFL-S
evaluation entrypoints are the R2R and RxR online commands above.
