# cuju-grf-media

The media of [Google Research Football](https://github.com/google-research/football) that the
software renderer of [cuju](https://github.com/cuju-football/engine) reads, the models, textures,
menu images and font of GRF's frames. The files are those of GRF's commit 3d9e754, copied byte for
byte, so the frames that cuju draws from them match GRF's.

The package holds data alone and is installed with cuju's extra `pixels`.

```sh
pip install 'cuju-engine[pixels]'
```

`cuju_grf_media/manifest.json` names each file, its path in GRF, its size, its SHA-256 and its
license, and `cuju_grf_media/SHA256SUMS` lists the SHA-256 of every file for `sha256sum -c`. The
models and textures descend from Gameplay Football by Bastiaan Konings Schuiling, dedicated to the
public domain under the Unlicense, with Google's changes under the Apache License 2.0, and the font
Alegreya Sans SC is licensed under the SIL Open Font License 1.1. NOTICE gives the details.
