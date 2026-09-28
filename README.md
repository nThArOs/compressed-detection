# compressed-detection

Compressed-domain video analysis for drone detection, following the CoViAR idea (Wu et al., CVPR 2018): work on I-frames, motion vectors and residuals instead of fully decoded frames. CPU only.

## Extraction

Videos are re-encoded to H.264 with a GOP of 12, no B-frames and a single reference frame, then decoded with PyAV. Motion vectors come from FFmpeg (`export_mvs`). Residuals are computed as the current frame minus the previous frame warped by the motion vectors.

```bash
docker compose build
docker compose run --rm extract data/videos/clip.mp4
docker compose run --rm --entrypoint python extract scripts/mosaic.py
```

Output in `results/compressed/<video>/`:

- `frames/` decoded frames, `000012_I.jpg`, `000013_P.jpg`
- `residuals/` residual images (128 = zero)
- `mv/` motion vectors drawn on the frame
- `flow/` motion field as color (hue = direction, brightness = magnitude)
- `raw/` residual (int16) and motion vectors as `.npz`, if `save.raw` is enabled
- `decoded.mp4`, `residual.mp4`, `mv.mp4`, `flow.mp4`
- `mosaic.jpg` frame / motion / residual side by side (`scripts/mosaic.py`)
- `frames.csv` frame type, number of vectors, residual MAE

Parameters are in `configs/compressed.yaml`.

## Datasets

One folder per dataset in `data/`, converted to MOTChallenge format in `data/mot/<dataset>/<split>/<sequence>/` (`gt.txt`, `seqinfo.ini`, `video.mp4` encoded with the settings above). Common class list in `configs/mot.yaml`, statistics in `results/datasets.json`.

| Dataset | Content | Splits | License |
| --- | --- | --- | --- |
| [VisDrone2019-MOT](https://github.com/VisDrone/VisDrone-Dataset) | vehicles and pedestrians from a drone | train 56, val 7, test-dev 17 | research only |
| [UAVDT](https://sites.google.com/view/grli-uavdt/) | vehicles from a drone | train 30, test 20 | research only |
| [DUT Anti-UAV](https://github.com/wangdongdut/DUT-Anti-UAV) | drones, single target | 20 sequences | Apache-2.0 |

```bash
docker compose run --rm --entrypoint python extract scripts/download_datasets.py
docker compose run --rm --entrypoint python extract scripts/convert_mot.py
```

VisDrone is on Google Drive and often hits the download quota: download the zips in a browser and put them in `data/visdrone_mot/archives/`.
