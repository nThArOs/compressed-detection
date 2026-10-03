"""Recall of the tracker boxes per object speed and size, from the MOT files of a method and the ground truth.

A ground-truth object is found in a frame when a tracker box of confidence >= --conf overlaps it with IoU >= 0.5.
Speed is the apparent displacement of the object centre, in pixels per frame, over its neighbouring frames; it
includes the camera motion. Relative speed subtracts the median apparent displacement of all objects of the frame,
a rough camera estimate.
"""
import argparse
import json

import numpy as np

from common import ROOT, load_config, save_json


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    y2 = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / np.maximum(a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter, 1e-6)


def motion(gt):
    """Per ground-truth row: apparent speed and speed relative to the frame median, from central differences."""
    gt = gt[np.lexsort((gt[:, 0], gt[:, 1]))]
    centre = gt[:, 2:4] + gt[:, 4:6] / 2
    vel = np.full((len(gt), 2), np.nan)
    same = gt[1:, 1] == gt[:-1, 1]
    step = (centre[1:] - centre[:-1]) / np.maximum(gt[1:, 0] - gt[:-1, 0], 1)[:, None]
    vel[1:][same] = step[same]
    vel[:-1][same] = np.where(np.isnan(vel[:-1][same]), step[same], (vel[:-1][same] + step[same]) / 2)
    speed = np.hypot(*vel.T)
    rel = np.full(len(gt), np.nan)
    for f in np.unique(gt[:, 0]):
        rows = gt[:, 0] == f
        v = vel[rows]
        ok = ~np.isnan(v[:, 0])
        if ok.sum() >= 3:
            rel[rows] = np.hypot(*(v - np.nanmedian(v, axis=0)).T)
    return gt, speed, rel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", nargs="+", help="folders in results/tracking/")
    parser.add_argument("--dataset", default="uavdt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequences", default="data/yolo/uavdt/splits.json")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--out", default="results/recall_by_motion.json")
    args = parser.parse_args()

    mot = load_config(ROOT / "configs" / "mot.yaml")
    seqs = [s.split("/")[1] for s in json.loads((ROOT / args.sequences).read_text())["sequences"][args.split]]
    speed_edges, rel_edges, size_edges = [1, 3, 6, 12], [1, 3, 6, 12], [20, 30, 45, 70]
    result = {}
    for method in args.method:
        rows = []  # speed, relative speed, size, found
        for seq in seqs:
            gt = np.loadtxt(ROOT / mot["output_dir"] / args.dataset / args.split / seq / "gt.txt", delimiter=",", ndmin=2)
            gt = gt[gt[:, 6] == 1]
            gt, speed, rel = motion(gt)
            trk = np.loadtxt(ROOT / "results" / "tracking" / method / args.dataset / args.split / f"{seq}.txt", delimiter=",", ndmin=2)
            trk = trk[trk[:, 6] >= args.conf] if len(trk) else trk
            found = np.zeros(len(gt), bool)
            by_frame = {int(f): trk[trk[:, 0] == f][:, 2:6] for f in np.unique(trk[:, 0])} if len(trk) else {}
            for f in np.unique(gt[:, 0]):
                idx = np.where(gt[:, 0] == f)[0]
                boxes = by_frame.get(int(f))
                if boxes is not None and len(boxes):
                    found[idx] = (iou(gt[idx, 2:6], boxes) >= 0.5).any(axis=1)
            size = np.sqrt(gt[:, 4] * gt[:, 5])
            rows.append(np.column_stack([speed, rel, size, found]))
        data = np.vstack(rows)

        def table(col, edges):
            out = {}
            bounds = [-np.inf] + edges + [np.inf]
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                sel = (data[:, col] >= lo) & (data[:, col] < hi) & ~np.isnan(data[:, col])
                label = f"<{hi:g}" if lo == -np.inf else (f">={lo:g}" if hi == np.inf else f"{lo:g}-{hi:g}")
                out[label] = {"objects": int(sel.sum()), "recall": round(float(data[sel, 3].mean()), 3) if sel.any() else None}
            return out
        result[method] = {"all": round(float(data[:, 3].mean()), 3), "speed_px_per_frame": table(0, speed_edges),
                          "relative_speed_px_per_frame": table(1, rel_edges), "size_px": table(2, size_edges)}
    save_json({"conf": args.conf, "dataset": args.dataset, "methods": result}, ROOT / args.out)
    for key, name in (("speed_px_per_frame", "apparent speed"), ("relative_speed_px_per_frame", "relative speed"), ("size_px", "size")):
        print(f"\nrecall by {name} (px), conf >= {args.conf}")
        labels = list(next(iter(result.values()))[key])
        print(f"{'':10s}" + "".join(f"{m[:22]:>24s}" for m in result))
        for lab in labels:
            print(f"{lab:10s}" + "".join(f"{result[m][key][lab]['recall']!s:>14s} ({result[m][key][lab]['objects']:>6d})" for m in result))
    print("\nall:", {m: result[m]["all"] for m in result})


if __name__ == "__main__":
    main()
