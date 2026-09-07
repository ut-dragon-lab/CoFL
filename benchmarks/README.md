# Available benchmark surfaces

Run commands from the repository root after the
[uv environment setup](../README.md#installation). CoFL offline policy evaluation
requires a prepared native dataset and a complete checkpoint; release downloads
are pending. The synthetic execution example uses analytic inputs and requires
neither asset.

| Surface | Entry point | Scope |
| --- | --- | --- |
| CoFL offline policy evaluation | `uv run --locked cofl evaluate --config configs/evaluation/cofl.yaml` | Dense field errors, FGE, CR, PLR, Curv, and visual reports |
| R2R-1200 data | [Prepare annotations](r2r_1200/README.md) | Fixed episode IDs, source versions and annotation files; no model settings or results |
| CoFL-S R2R online evaluation | `uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_r2r_online.yaml` | Model and execution configuration using the R2R-1200 data and separate runtime recipe |
| RxR-1600 data | [Prepare annotations](rxr_1600/README.md) | Fixed custom guide-English subset; data identities and annotations only |
| CoFL-S RxR online evaluation | `uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_rxr_online.yaml` | Model and execution configuration using RxR-1600 with a strict Landmark oracle |
| Offline field metrics | `cofl.evaluation.evaluate_field` | CoFL image fields |
| Offline trajectory metrics | `cofl.evaluation.evaluate_image_trajectory` | Image polyline geometry with explicit units |
| Synthetic ground execution | `uv run --locked cofl benchmark synthetic-ground --output results/synthetic` | Analytic known-goal planner, pursuit controller, unicycle |

See [protocol documentation](../docs/benchmarks.md) for masks, zero-vector
handling, units, instruction assistance, timing and result persistence.
The [online guide](../docs/app.md#online-environment-and-benchmark-configuration)
describes the Habitat interpreter, scene assets, episode data and recipe needed
for closed-loop evaluation. Its current protocol is
`cofl-s-vln-benchmark-v6` with metric recipe `vlnce_official_core_v1` and shared
`cofl-inference-v1` rules; older v2–v5 results require a separate output
directory and cannot be resumed into a v6 run. Official `gt_path` trajectories
are required for nDTW/SDTW.
The core scores use official VLN-CE definitions; additional scores include
HS (Heading Smoothness), BSR (Blocked-Step Rate), VC, EF, HA, CLS and cSPL.
See the [comparison scope](../docs/app.md#shared-protocol-and-comparison-scope)
for distance, executed-path sampling and success conventions.

The [evaluation configuration index](../configs/evaluation/README.md) explains
the dataset-named online entrypoints. The `r2r_1200.yaml` and `rxr_1600.yaml`
runtime recipes supply prepared data paths, scene assets, the Habitat
interpreter and sensors; their evaluation YAMLs define the checkpoints and
run parameters. Each YAML resolves relative paths from its location, and CLI
options override evaluation settings.

The CoFL offline recipe accepts dataset, checkpoint, output and execution
overrides. Its results include per-annotation records, scene/category summaries,
CSV tables and an HTML report. CoFL-S uses the two online evaluation entrypoints
above for executed navigation trajectories and episode metrics.

The top-level JSON files are machine-readable protocol descriptions. The synthetic
runner embeds its executable numerical recipe in `summary.json`; these
descriptions are not CLI configuration files. External simulator assets,
publication checkpoints and a verified paper reproduction package are not
bundled. Matching core metric definitions does not establish the same control,
instruction-assistance and sampling conditions or a completed benchmark result.
The [R2R-1200](r2r_1200/dataset.json) and
[RxR-1600](rxr_1600/dataset.json) manifests define fixed custom subsets selected
for this task. Neither is the official full split. Their preparation commands
produce data only. Reference scores and evaluation conditions are documented
separately in the [R2R-1200 evaluation reference](../reproduce/r2r_1200.md) and
[RxR-1600 evaluation reference](../reproduce/rxr_1600.md); the scores apply to
these selected subsets.
