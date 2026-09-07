# Paper reproduction status

This directory distinguishes software verification from paper reproduction.

The first model/data release targets CoFL-S and its R2R/RxR training and
validation collection. CoFL image-field artifacts are deferred; they are not
part of the current publication bundle. See [release scope](../docs/data-release.md).

The [uv setup](../README.md#installation) creates an independent Linux x86_64
environment with Python 3.12.12 and locked dependencies. From the repository
root, the CLI check can run on CPU without release datasets, checkpoints,
W&B credentials or a pretrained-model download:

```bash
uv sync --locked --no-group cu124 --group cpu
uv run --locked --no-group cu124 --group cpu cofl --help
```

The full internal Python and browser test suites are not included in the public
release. Public CI checks the environment, CLI entrypoints, generated API types,
frontend unit tests and builds, and package builds. Formal training and evaluation
still need the assets listed below. The Habitat worker also keeps its own
simulator environment.

| Work | Reference | Available software | Still needed for paper reproduction |
| --- | --- | --- | --- |
| CoFL | [arXiv:2603.02854](https://arxiv.org/abs/2603.02854) | Image-field decoder, shared RGB/language encoder, native data tools, Lightning training, batched field and image-navigation evaluation with reports, and rendered-scene generation | Release checkpoints, published dataset/split revisions, matched generation/evaluation settings, baseline runs and result verification |
| CoFL-S | [arXiv:2607.02222](https://arxiv.org/abs/2607.02222) | Sector-field decoder, shared single-frame RGB-D/language encoder, action head, native data tools, VLN episode generation, Lightning training and validation, a closed-loop Habitat/VLN runner, and fixed R2R-1200/RxR-1600 manifest/preparation workflows | Release checkpoints, published generated dataset/splits, a verified historical simulator environment, matched baselines and fresh result verification |

[`docs/generation.md`](../docs/generation.md) describes the runnable source-to-data
adapters, required assets, and the scope of generation verification. Habitat
rendering during dataset generation does not constitute navigation benchmark
evaluation.

CoFL now includes the upstream Matterport3D/ScanNet mesh renderers as
`cofl data render`; see [rendering](../docs/rendering.md). This completes the
available source-to-dataset code path, but does not establish exact historical
dataset regeneration. Existing native CoFL data was migrated with its legacy
labels preserved. The historical camera catalog is outside the first asset
release; a separately supplied licensed catalog can be replayed with `--cameras-from`;
missing camera records, historical label choices, empty-view retention, sample
order and action variants still require explicit reconciliation; see the
[dataset reproduction boundary](../docs/generation.md#historical-datasets-versus-a-new-deterministic-generation).

A paper reproduction release must pair its configuration with a dataset
revision and split identity, label-generation version, checkpoint, environment,
evaluation protocol, expected results, and table-generation command.

The CoFL offline evaluation recipe is
[`evaluation/cofl.yaml`](../configs/evaluation/cofl.yaml). It scores image fields,
image-space goals, trajectories and point-path collisions, and records
dataset/cohort, checkpoint, code and recipe identities in the SQLite result
store, with CSV/JSONL exports and a local HTML report. See
[metric definitions](../docs/benchmarks.md).

CoFL-S evaluation uses the R2R-1200 and RxR-1600 online entrypoints below; see the
[online setup and CLI](../docs/app.md#online-environment-and-benchmark-configuration).
Protocol `cofl-s-vln-benchmark-v6` uses metric recipe `vlnce_official_core_v1`
and shared `cofl-inference-v1` rules:
NE/OS/SR/SPL/nDTW/SDTW follow the official core definitions, with strict STOP
success, XYZ path length and official `gt_path` trajectories. Extra metrics
remain available, with SM renamed HS and StR renamed BSR. Continuous control,
oracle instructions and plant-step sampling remain explicit experimental
conditions; matching formulas does not establish paper reproduction. Earlier
v2–v5 results cannot be resumed or merged into a v6 output directory. See the
[full comparison scope](../docs/app.md#shared-protocol-and-comparison-scope).

The [R2R-1200 dataset](../benchmarks/r2r_1200/README.md) fixes a custom subset
selected for this task and its annotation identities. It is not the official
full split. Its preparation command produces data only. The separate
[R2R-1200 evaluation reference](r2r_1200.md) records stored metric aggregates
for that subset, checkpoint identity, execution conditions and environment
limitations. The [evaluation configuration guide](../configs/evaluation/README.md)
covers model and runtime setup.

The [RxR-1600 dataset](../benchmarks/rxr_1600/README.md) likewise supplies fixed
IDs and annotation preparation for the custom subset selected for this task.
It is not the official full split. Its manifest contains data identities only;
scores for that subset and their evaluation conditions belong to the separate
[RxR-1600 evaluation reference](rxr_1600.md).

After preparing the data and configuring evaluation assets, run:

```bash
uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_r2r_online.yaml
uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_rxr_online.yaml
```

Each entrypoint defines its checkpoint and run parameters and references its
own `r2r_1200.yaml` or `rxr_1600.yaml` runtime recipe. The RxR configuration uses
1,600 prepared guide-English episodes, prepared indices `0..1599`, and a strict
Landmark oracle. Its new results go to `artifacts/online/cofl-s-rxr-1600`.
Preparing the dataset does not execute a new evaluation or establish a score.

Training uses the public LightningCLI recipes
[`cofl_formal.yaml`](../configs/cofl_formal.yaml) and
[`cofl_s_formal.yaml`](../configs/cofl_s_formal.yaml). Record the resolved
configuration, package versions and dataset identities with each experiment.
`best.ckpt` and `last.ckpt` contain full policy assets and Lightning training
state. Standard checkpoint continuation does not restore arbitrary worker
augmentation streams bit for bit.

Internal CPU integration tests exercise both policy profiles through the actual CLI,
checkpoint continuation and inference with small local SigLIP assets. These
tests establish software behavior; they are not training-quality or navigation
benchmark results. Public release checkpoints remain listed as pending above.
