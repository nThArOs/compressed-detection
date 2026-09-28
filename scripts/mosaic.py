"""Frame / motion vectors / residual mosaic from the output of compressed_video.py."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from common import ROOT, load_config

COLUMNS = [("frames", ".jpg", "frame"), ("flow", ".png", "motion vectors"),
           ("residuals", ".png", "residual")]


def tile(path, width, label=None):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])),
                     interpolation=cv2.INTER_AREA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
        cv2.putText(img, label, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return img


def mosaic(video_dir, rows, width):
    stems = sorted(p.stem for p in (video_dir / "residuals").glob("*.png"))
    if not stems:
        return None
    picks = [stems[i] for i in np.linspace(0, len(stems) - 1, min(rows, len(stems))).astype(int)]
    lines = []
    for r, stem in enumerate(picks):
        cells = []
        for c, (folder, ext, name) in enumerate(COLUMNS):
            label = name if r == 0 else None
            if c == 0:
                label = f"{stem}  {name}" if r == 0 else stem
            cells.append(tile(video_dir / folder / f"{stem}{ext}", width, label))
        lines.append(np.hstack(cells))
    return np.vstack(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="*", help="video names in the output dir (default: all)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "compressed.yaml"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_root = ROOT / cfg["output_dir"]
    dirs = [out_root / v for v in args.videos] or sorted(d for d in out_root.iterdir() if d.is_dir())
    for d in dirs:
        img = mosaic(d, cfg["mosaic"]["rows"], cfg["mosaic"]["tile_width"])
        if img is None:
            print(f"{d.name}: no residuals, run compressed_video.py with save.residuals enabled")
            continue
        cv2.imwrite(str(d / "mosaic.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{d.name}: {d.relative_to(ROOT) / 'mosaic.jpg'}")


if __name__ == "__main__":
    main()
