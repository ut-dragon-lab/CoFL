# Third-party code and asset notices

The repository's Apache-2.0 license covers the contributors' original source
code. Separately licensed third-party code and assets retain their own terms.
Model checkpoints and training datasets have separate release licenses.

## ScanNet renderer adaptations

`src/cofl/generation/rendering/scannet.py` and `postprocess.py` retain the
upstream MIT permission notice and the 2017 copyright attribution to Angela
Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser and Matthias
Niessner. Their complete notices remain in the respective source files.

This code license does not grant access or redistribution rights to ScanNet
scene data. The first asset release excludes CoFL's scene datasets and
historical camera catalog.

## Studio dependencies

Studio dependencies are identified by `app/package.json` and pinned in
`app/package-lock.json`. Their licenses remain with their respective packages.
When browser assets are built, the decoder bundle includes the Three.js license
and the upstream license/notice files copied with the Basis and Draco libraries.
Builds generate `THIRD_PARTY_LICENSES.txt` for JavaScript packages included in
the browser bundles. The build fails if a bundled package lacks a license notice.
Notices absent from an npm archive are retained in `app/licenses`, with their
upstream URLs and the corresponding npm package versions in `sources.json`.

## Model and dataset assets

SigLIP2 components, Matterport3D observations, R2R/RxR annotations and alignment
sources are covered by the model/dataset repositories' own LICENSE and
THIRD_PARTY_NOTICES files. Their terms are not replaced by the source-code
license here. The first published model/data artifacts are for CoFL-S.
