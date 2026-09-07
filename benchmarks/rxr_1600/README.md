# RxR-1600 dataset

[`dataset.json`](dataset.json), identified as `rxr-1600-v1`, provides a fixed
selected subset of **1,600 RxR-CE `val_unseen` guide-English episodes** with their
original Landmark-RxR annotations, spanning 11 scenes. The JSON is a data
provider: it records episode IDs, source indices and annotation fingerprints.
It contains no model settings, execution parameters or evaluation results.

This selected subset supplies the fixed inputs for our navigation task. It is
a custom benchmark rather than the official full split. The model, evaluation
protocol and reference metrics are documented separately in the
[evaluation reference](../../reproduce/rxr_1600.md).

## Source annotations

- **RxR episodes and official GT:** `val_unseen/val_unseen_guide.json.gz` and
  `val_unseen/val_unseen_guide_gt.json.gz` from `RxR_VLNCE_v0`, linked by
  [VLN-CE](https://github.com/jacobkrantz/VLN-CE) through its
  [upstream archive](https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view).
- **Landmark-RxR:** `LandmarkRxR_val_unseen.json` from the
  [official project](https://github.com/hekj/Landmark-RxR) and its
  [annotation download](https://www.dropbox.com/sh/muwe42gzj4qxr4m/AABAGMvCL3odeRpDb_z_Ghu9a?dl=0).

Obtain these annotations separately under their providers' terms. Preparation
checks each selected episode, official GT record and complete Landmark row
against its canonical fingerprint. Formatting and unrelated source records do
not change the selected records' canonical identity.

## Dataset contract

- Episodes retain their original IDs. Prepared indices are `0..1599`.
- `source_index` indexes the original 3,669 English guide episodes;
  `source_record_index` indexes the complete 11,006-record guide file.
  These indices differ from prepared indices and from episode/instruction IDs.
- Each selected episode has at least two nonempty Landmark sub-instructions,
  matched by instruction ID, scene and trajectory. Their normalized concatenated
  words preserve the full instruction, and contiguous sub-paths cover its route.
  The manifest's `quality_filter` records these base annotation eligibility rules.
  The exact selected episodes are defined by its fixed ID list.
- Sub-instructions have variable lengths. The dataset does not impose a
  universal 64-token limit; text truncation is part of the model's input settings.
- Preparation omits only `instruction.timed_instruction`, whose unused native
  timestamps may be non-finite. All other episode fields and all selected GT
  and Landmark records remain unchanged.

## Prepare with CoFL

From the CoFL repository root:

```bash
uv run --locked python -m cofl.online.prepare_rxr_benchmark \
  --dataset-json benchmarks/rxr_1600/dataset.json \
  --rxr /data/RxR_VLNCE_v0/val_unseen/val_unseen_guide.json.gz \
  --gt /data/RxR_VLNCE_v0/val_unseen/val_unseen_guide_gt.json.gz \
  --landmark-rxr /data/Landmark-RxR/LandmarkRxR_val_unseen.json \
  --output prepared/rxr_1600
```

Choose an output directory that does not exist. The preparer validates the
inputs before atomically publishing these five files:

| Output | Contract |
| --- | --- |
| `selected_episodes.json.gz` | 1,600 projected episode objects in manifest order |
| `selected_gt.json.gz` | Complete official GT records keyed by the selected episode IDs |
| `selected_landmark_rxr.json` | The 1,600 complete matched Landmark rows |
| `episode_index_map.json` | Prepared indices, original IDs and original source indices |
| `preparation.json` | Preparation provenance and input/output identities |

The five outputs supply the annotation inputs for evaluation. The original
source annotation paths are no longer needed after preparation. Preparing data
requires no checkpoint, scene assets, simulator, GPU or stored model outcomes.

## Prepare from the standalone data bundle

The compact ZIP contains `dataset.json`, this README, `prepare_rxr_benchmark.py`
and its `prepare_benchmark.py` helper. Keep both scripts together. With Python
3.10 or newer, no third-party packages are required:

```bash
python prepare_rxr_benchmark.py \
  --dataset-json dataset.json \
  --rxr /data/RxR_VLNCE_v0/val_unseen/val_unseen_guide.json.gz \
  --gt /data/RxR_VLNCE_v0/val_unseen/val_unseen_guide_gt.json.gz \
  --landmark-rxr /data/Landmark-RxR/LandmarkRxR_val_unseen.json \
  --output prepared/rxr_1600
```

The selected IDs are already frozen in the manifest. Preparation requires only
this manifest, the two preparation scripts and the source annotations.

## Evaluate separately

The CoFL repository/package supplies the evaluation implementation and two
configuration files:

| File | Responsibility |
| --- | --- |
| `configs/evaluation/rxr_1600.yaml` | Prepared annotation paths, strict Landmark oracle, scenes/interpreter and Habitat sensors |
| `configs/evaluation/cofl_s_rxr_online.yaml` | Checkpoint, execution settings, controller/planner parameters and result output |

Configure the model, scene assets and Habitat environment following the
[evaluation guide](../../configs/evaluation/README.md#rxr-online), then run:

```bash
uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_rxr_online.yaml
```

The evaluation resolves `prepared/rxr_1600` by default and requires Landmark
sub-instructions without native/full-instruction fallback. It writes results
to the evaluation output directory, independently of this data provider. A
different preparation location requires changing the runtime recipe's three
annotation paths. The data preparer does not generate evaluation settings.

Source distributions include this data manifest and README; wheel installations
place them under `share/cofl/benchmarks/rxr_1600`. Dataset preparation reproduces
the fixed inputs; numerical evaluation reproducibility also depends on the
checkpoint and runtime protocol described in the separate reproduction note.
