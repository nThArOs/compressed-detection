"""Build YOLO datasets (residual and RGB) from the MOT videos, split by sequence."""
import argparse
import json
import random

import av
import cv2
import numpy as np
import yaml

from common import ROOT, load_config
from compressed_video import ResidualSource


def sequence_splits(name, dcfg, mot_root, seed):
    def seqs(split):
        return sorted(p.name for p in (mot_root / name / split).iterdir() if p.is_dir())

    rng = random.Random(seed)
    if "random_split" in dcfg:
        pool = seqs(dcfg["source_split"])
        rng.shuffle(pool)
        n_val, n_test = dcfg["random_split"]["val"], dcfg["random_split"]["test"]
        out = {"val": pool[:n_val], "test": pool[n_val:n_val + n_test],
               "train": pool[n_val + n_test:]}
        return {k: [(dcfg["source_split"], s) for s in sorted(v)] for k, v in out.items()}

    out = {k: [(src, s) for s in seqs(src)] for k, src in dcfg["splits"].items()}
    if "random_split_from_train" in dcfg:
        train = out["train"][:]
        rng.shuffle(train)
        n_val = dcfg["random_split_from_train"]["val"]
        out["val"], out["train"] = sorted(train[:n_val]), sorted(train[n_val:])
    return out


def load_gt(path):
    gt = np.loadtxt(path, delimiter=",", ndmin=2) if path.stat().st_size else np.zeros((0, 9))
    return gt


def yolo_labels(boxes, w, h):
    lines = []
    for x, y, bw, bh, c in boxes:
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + bw, w), min(y + bh, h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
    return "".join(lines)


def process_sequence(seq_dir, split, out, cfg, class_idx):
    gt = load_gt(seq_dir / "gt.txt")
    jpg = [cv2.IMWRITE_JPEG_QUALITY, cfg["jpeg_quality"]]
    container = av.open(str(seq_dir / "video.mp4"))
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}
    w, h = stream.codec_context.width, stream.codec_context.height
    source = ResidualSource(w, h, cfg.get("residual_mode", "rgb"), cfg["residual_gain"])

    n = 0
    for i, frame in enumerate(container.decode(stream)):
        # the residual of every frame is needed to follow the stream, only one in frame_step is kept
        img = frame.to_ndarray(format="bgr24")
        f = i + 1
        residual_img, _, _ = source.step(frame, img, compute=f % cfg["frame_step"] == 0)
        if f % cfg["frame_step"] == 0 and residual_img is not None:
            rows = gt[gt[:, 0] == f]
            keep = rows[(rows[:, 6] == 1) & np.isin(rows[:, 7], list(class_idx))]
            boxes = [(*r[2:6], class_idx[int(r[7])]) for r in keep]
            ignore = rows[rows[:, 6] == 0]

            images = {"residual": residual_img, "rgb": img.copy()}
            stem = f"{seq_dir.name}_{f:06d}"
            for mod in cfg["modalities"]:
                im = images[mod]
                for x, y, bw, bh in ignore[:, 2:6].astype(int):
                    im[max(y, 0):y + bh, max(x, 0):x + bw] = 128 if mod == "residual" else 114
                d = out / mod
                cv2.imwrite(str(d / "images" / split / f"{stem}.jpg"), im, jpg)
                (d / "labels" / split / f"{stem}.txt").write_text(yolo_labels(boxes, w, h))
            n += 1
    container.close()
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help="names from configs/residual.yaml (default: all)")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "residual.yaml")
    mot = load_config(ROOT / "configs" / "mot.yaml")
    mot_cls = {name: i for i, name in enumerate(mot["classes"], 1)}
    mot_root = ROOT / mot["output_dir"]

    for name in args.datasets or list(cfg["datasets"]):
        dcfg = cfg["datasets"][name]
        class_idx = {mot_cls[c]: i for i, c in enumerate(dcfg["classes"])}
        out = ROOT / cfg["output_dir"] / name
        splits = sequence_splits(name, dcfg, mot_root, cfg["split_seed"])

        for mod in cfg["modalities"]:
            for split in splits:
                (out / mod / "images" / split).mkdir(parents=True, exist_ok=True)
                (out / mod / "labels" / split).mkdir(parents=True, exist_ok=True)
            (out / mod / "data.yaml").write_text(yaml.safe_dump({
                "path": f"/app/{cfg['output_dir']}/{name}/{mod}",
                **{s: f"images/{s}" for s in splits},
                "names": dict(enumerate(dcfg["classes"]))}, sort_keys=False))

        counts = {}
        for split, seqs in splits.items():
            counts[split] = 0
            for src, seq in seqs:
                n = process_sequence(mot_root / name / src / seq, split, out, cfg, class_idx)
                counts[split] += n
                print(f"{name}/{split}/{seq}: {n} frames")
        (out / "splits.json").write_text(json.dumps(
            {"sequences": {k: [f"{src}/{s}" for src, s in v] for k, v in splits.items()},
             "frames": counts}, indent=2))
        print(f"{name}: {counts}")


if __name__ == "__main__":
    main()
