# Studio components

Studio uses React, Radix Themes, TanStack Query, React Three Fiber, Drei,
Three.js/three-stdlib, Lucide, and openapi-fetch directly. Resolved versions
are in `package-lock.json`; their source and licenses ship in the npm packages.

`scripts/copy-decoders.mjs` copies the unchanged Basis Universal and Draco
runtime assets distributed with Three.js into the built app. These decoders
run locally; Studio does not fetch decoder code from a CDN. Their upstream
README/license files are copied along with the assets and Three.js license.

`scene/basis.ts` is a GLTFLoader extension adapter for Habitat's historical
`GOOGLE_texture_basis` extension. It delegates decoding to Binomial's bundled
Basis Universal WASM runtime; ordinary glTF, Draco and KTX2 parsing stays in
the third-party loaders.

References:
- https://github.com/mrdoob/three.js
- https://github.com/BinomialLLC/basis_universal
- https://github.com/google/draco
