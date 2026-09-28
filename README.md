# compressed-detection

Compressed-domain video analysis for drone detection, following the CoViAR idea (Wu et al., CVPR 2018): work on I-frames, motion vectors and residuals instead of fully decoded frames. CPU only.

## Extraction

Videos are re-encoded to H.264 with a GOP of 12, no B-frames and a single reference frame, then decoded with PyAV. Motion vectors come from FFmpeg (`export_mvs`). Residuals are computed as the current frame minus the previous frame warped by the motion vectors.

```bash
docker compose build
docker compose run --rm extract data/videos/clip.mp4
```

Output in `results/compressed/<video>/`:

- `frames/` decoded frames, `000012_I.jpg`, `000013_P.jpg`
- `residuals/` residual images (128 = zero)
- `mv/` motion vectors drawn on the frame
- `raw/` residual (int16) and motion vectors as `.npz`, if `save.raw` is enabled
- `decoded.mp4`, `residual.mp4`, `mv.mp4`
- `frames.csv` frame type, number of vectors, residual MAE

Parameters are in `configs/compressed.yaml`.
