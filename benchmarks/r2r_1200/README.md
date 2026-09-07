# R2R-1200 dataset

[`dataset.json`](dataset.json) identifies a fixed selected subset of 1,200 R2R-CE
`val_unseen` episodes. It provides original episode IDs, source versions and
annotation fingerprints so evaluations can use the same data. It is not the
official full split. This subset supplies the fixed inputs for our navigation
task; see the separate [evaluation reference](../../reproduce/r2r_1200.md)
for the model and evaluation protocol.

The dataset package contains no model, evaluation settings or results.

## Source annotations

- **R2R episodes and official GT:** use `val_unseen/val_unseen.json.gz` and
  `val_unseen/val_unseen_gt.json.gz` from `R2R_VLNCE_v1-3_preprocessed`, linked by
  [VLN-CE](https://github.com/jacobkrantz/VLN-CE) through its
  [upstream archive](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view).
- **FGR2R:** use `data/FGR2R_val_unseen.json` from
  [Fine-Grained-R2R at commit `305ed4cd219d79306bc38650899edc14cfdcfb69`](https://github.com/YicongHong/Fine-Grained-R2R/tree/305ed4cd219d79306bc38650899edc14cfdcfb69).
  All 531 source rows referenced by the selected episodes match this upstream
  version; no annotation patch is needed.

The preparation tool checks selected records using the canonical fingerprint
rules in `dataset.json`. Unrelated FGR2R rows and JSON formatting do not affect
those checks. Upstream annotations are obtained separately under their
providers' terms.

## Prepare the data

From the CoFL repository root, with CoFL installed:

```bash
uv run --locked python -m cofl.online.prepare_benchmark \
  --dataset-json benchmarks/r2r_1200/dataset.json \
  --r2r /data/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz \
  --gt /data/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz \
  --fgr2r /data/Fine-Grained-R2R/data/FGR2R_val_unseen.json \
  --output prepared/r2r_1200
```

This command uses the Python standard library. Data preparation needs only the
manifest and source annotations; it does not load a checkpoint, scene assets or
a simulator environment.

| Output | Contents |
| --- | --- |
| `selected_episodes.json.gz` | The 1,200 original episode objects in manifest order |
| `selected_gt.json.gz` | Their matching official GT records |
| `selected_fgr2r.json` | The 531 matched FGR2R rows with normalized annotation representations |
| `episode_index_map.json` | Prepared index, original episode ID and source-index correspondence |
| `preparation.json` | Data preparation provenance and input/output identities |

Original episode IDs are preserved. Prepared indices are `0..1199`, distinct
from source indices `0..1838`; use the index map to translate them. The selected
annotation files are self-contained, so subsequent evaluation does not depend
on the original annotation input paths.

## Use in evaluation

In a CoFL checkout, `configs/evaluation/cofl_s_r2r_online.yaml` defines the model
evaluation, and `configs/evaluation/r2r_1200.yaml` supplies the runtime and data
paths. Configure those evaluation assets, then run from the repository root:

```bash
uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_r2r_online.yaml
```

The [evaluation configuration guide](../../configs/evaluation/README.md#r2r-online)
provides setup details. Checkpoint, scenes, simulator settings, execution
parameters and result output belong to evaluation. If you choose a preparation
directory other than `prepared/r2r_1200`, update the three data paths in the
runtime recipe. These configuration files are supplied by the CoFL
repository/package, not by the compact data bundle.

## Distribution

The dataset files are included in source distributions and under
`share/cofl/benchmarks/r2r_1200` in wheel installations. The compact data bundle
contains this README, `dataset.json` and a standalone `prepare_benchmark.py`.
In that bundle, use `python prepare_benchmark.py --dataset-json dataset.json`
with the same `--r2r`, `--gt`, `--fgr2r` and `--output` arguments above.
Evaluation code and configurations are supplied by the CoFL repository/package.
