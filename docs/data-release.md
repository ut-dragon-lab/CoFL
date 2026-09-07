# CoFL-S model and dataset release

The first asset release contains the **CoFL-S** checkpoint and its matching
`ground_sector_v1` dataset. CoFL image-field weights, image-field datasets and
ScanNet-derived assets are outside this release.

| Artifact | Contents | Hosting |
| --- | --- | --- |
| CoFL-S model | Sanitized complete checkpoint, model card, license, upstream notices and checksums | Hugging Face model repository |
| CoFL-S dataset | R2R/RxR `train` and `val_unseen`, native shards, original collection manifests, dataset card and source terms | Separate Hugging Face dataset repository |

The training collection is `cofl-s-training-v1`, revision `20260907-v1`.
Its 4,042 training shards contain 3,758,511 eligible training samples. Preserve
the complete validation collection too: the training recipe selects 8,192
validation samples from it. That number is not the size of the validation
release. Model and dataset publication status are tracked separately; a prepared
local model file does not mean the dataset is already uploaded. The
[model repository](https://huggingface.co/lhk66666/CoFL-S) contains the released
checkpoint and matching source snapshot. See the
[dataset card](https://huggingface.co/datasets/lhk66666/CoFL-S-Dataset) for the
current archive-upload status and access form.

The standard training path is to download a prepared dataset, verify its
checksums, unpack it and start training. Generation is provided for custom
experiments and dataset development.

## Hosting and organization

Use a Hugging Face dataset repository for the versioned download and dataset
card. Retain R2R/RxR source identity and their train/validation splits. Keep the
checkpoint in its model repository so users can download either artifact
independently.

Distribute the native Parquet/Zarr shards in approximately 1–5 GB archive files,
with a manifest listing archive names, sizes, checksums, split membership and
relative extraction paths. This is a packaging choice; the extracted dataset
uses the documented [native schema](dataset-format.md). Group existing shards
without changing sample IDs, labels or split assignments.

Each release should include:

- A fixed revision and the root collection manifests.
- Prepared RGB, metric depth where used, instructions, fields and action labels.
- Archive checksums and commands to download selected splits and unpack them.
- The generation recipe, geometry, counts, eligible counts and source attribution.
- Applicable license and access terms.

Hugging Face downloads support a fixed revision and file selection. A release
can therefore be downloaded by split without running a simulator. Publish the
actual repository IDs and archive manifest with the data release; no download
endpoint is advertised before it exists.
[Hugging Face download documentation](https://huggingface.co/docs/huggingface_hub/guides/download)

Large public datasets should confirm their storage allocation with the host.
Free public Hub storage is best-effort, and paid/research allocations have
separate conditions. Avoid uploading every Zarr chunk as an individual Hub
file; use archive packaging and follow the current repository/file limits.
[Hugging Face storage policy](https://huggingface.co/docs/hub/storage-limits)

Hugging Face can assign a DOI to a dataset release. A second full-size archive
is optional; a DOI alone does not require mirroring the corpus to another host.
[Hugging Face DOI documentation](https://huggingface.co/docs/hub/doi)

## Source-data access

Hosting does not grant redistribution rights. Final access settings must follow
the source agreements accepted for the assets used to create each release.

| Source | Release consideration |
| --- | --- |
| Matterport3D | Data and derived information are subject to the academic-data agreement. Distribution must carry its terms and, where required, obtain and record affirmative acceptance. |
| RxR | Annotation licensing is separate from the Matterport3D scene license. |
| R2R and alignment annotations | Keep their provenance and applicable annotation terms alongside the scene agreement. A code license does not determine the image/depth license. |

Sources: [Matterport3D agreement](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf),
[Matterport current academic-data terms](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data),
[RxR license](https://github.com/google-research-datasets/RxR#license),
[R2R simulator license notes](https://github.com/peteanderson80/Matterport3DSimulator#license).

A gated repository can collect license acceptance and access records when
redistribution is authorized. It does not replace permission from the source
provider. Model-weight distribution also needs its applicable source terms.
[Hugging Face gated datasets](https://huggingface.co/docs/hub/datasets-gated)

For the complete RGB-D training-data release, prepare a public dataset page
with automatic access requests. Configure explicit source-agreement acceptance
and the information/consent required by the applicable agreement. Dataset-card
form fields customize that request; the repository's access setting must also
be enabled before data files become public. Preserve the acceptance records.

Publication preparation includes a full metadata privacy scan, provenance and
license verification, archive inventory and account-storage check. Package
existing files without changing sample IDs, split membership, values, order or
manifest bytes. If sanitization requires changing a manifest, publish a new
identity and document its relationship to the checkpoint instead of claiming
the old hash still matches.
