"""Render tracker output next to the residual: ground truth in green, tracks with their id."""
import argparse

import av
import cv2
import numpy as np

from common import ROOT, load_config
from compressed_video import VideoWriter, motion_field, past_vectors, residual


def color(tid):
    rng = np.random.default_rng(int(tid))
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def load(path):
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 8))
    return np.loadtxt(path, delimiter=",", ndmin=2)[:, :8]


def draw(img, gt, trk, f, scale):
    for r in gt[(gt[:, 0] == f) & (gt[:, 6] == 1)]:
        x, y, w, h = (r[2:6] * scale).astype(int)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for r in trk[trk[:, 0] == f]:
        x, y, w, h = (r[2:6] * scale).astype(int)
        c = color(r[1])
        cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
        cv2.putText(img, f"{int(r[1])} {r[6]:.2f}", (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, c, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="folder in results/tracking/")
    parser.add_argument("dataset")
    parser.add_argument("split")
    parser.add_argument("sequences", nargs="+")
    parser.add_argument("--width", type=int, default=960, help="width of each panel")
    parser.add_argument("--preview", action="store_true",
                        help="light version for the report: 480 px panels, crf 30")
    args = parser.parse_args()
    if args.preview:
        args.width = 480

    cfg = load_config(ROOT / "configs" / "track.yaml")
    mot = load_config(ROOT / "configs" / "mot.yaml")
    gain = load_config(ROOT / "configs" / "compressed.yaml")["residual_gain"]
    out_dir = ROOT / "results" / "videos" / ("previews" if args.preview else "") / args.method / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq in args.sequences:
        seq_dir = ROOT / mot["output_dir"] / args.dataset / args.split / seq
        gt = load(seq_dir / "gt.txt")
        trk = load(ROOT / cfg["output_dir"] / args.method / args.dataset / args.split / f"{seq}.txt")

        container = av.open(str(seq_dir / "video.mp4"))
        stream = container.streams.video[0]
        stream.codec_context.options = {"flags2": "+export_mvs"}
        w, h = stream.codec_context.width, stream.codec_context.height
        scale = args.width / w
        size = (args.width, round(h * scale) // 2 * 2)
        grid = np.dstack(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)))
        writer = VideoWriter(out_dir / f"{seq}.mp4", stream.average_rate or 30, 2 * size[0], size[1],
                             crf=30 if args.preview else None)

        prev = None
        for i, frame in enumerate(container.decode(stream)):
            img = frame.to_ndarray(format="bgr24")
            mvs = past_vectors(frame) if prev is not None else None
            res = (np.clip(128 + gain * residual(img, prev, motion_field(mvs, h, w), grid), 0, 255)
                   .astype(np.uint8) if mvs is not None and len(mvs) else np.full_like(img, 128))
            left, right = cv2.resize(img, size), cv2.resize(res, size, interpolation=cv2.INTER_AREA)
            draw(left, gt, trk, i + 1, scale)
            draw(right, gt, trk, i + 1, scale)
            fs = max(0.45, args.width / 1370)
            cv2.putText(left, f"{seq}  #{i + 1}  {args.method}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(right, "residual", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(np.hstack([left, right]))
            prev = img
        container.close()
        writer.close()
        print(f"{seq}: {out_dir.relative_to(ROOT) / f'{seq}.mp4'}")


if __name__ == "__main__":
    main()
