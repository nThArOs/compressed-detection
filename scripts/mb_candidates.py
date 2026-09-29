"""Level 1 without training: candidate regions from the macroblock SAD map, and their recall."""
import argparse
import json
import time

import cv2
import numpy as np

from common import ROOT, hardware_info, load_config, save_json


def candidates(sad, intra, cfg):
    med = np.median(sad)
    mad = np.median(np.abs(sad - med)) + 1e-6
    mask = sad > max(med + cfg["k_mad"] * mad, cfg["min_sad"])
    if cfg["use_intra"]:
        mask |= intra
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    gh, gw = sad.shape
    boxes = []
    for x, y, w, h, _ in stats[1:n]:
        if w * h > cfg["max_area"] * gh * gw:
            continue
        m = cfg["margin"]
        x0, y0 = max(x - m, 0), max(y - m, 0)
        boxes.append((x0, y0, min(x + w + m, gw) - x0, min(y + h + m, gh) - y0))
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequences", help="splits.json: also report the held-out test sequences")
    args = parser.parse_args()

    ecfg = load_config(ROOT / "configs" / "encoder.yaml")
    ccfg, mb = ecfg["candidates"], ecfg["mb"]
    mot = load_config(ROOT / "configs" / "mot.yaml")
    maps_dir = ROOT / ecfg["output_dir"] / args.dataset / args.split
    out_dir = ROOT / "results" / "candidates" / args.dataset / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seq = {}
    for path in sorted(maps_dir.glob("*.npz")):
        m = dict(np.load(path))
        w, h = m["size"]
        gt = np.loadtxt(ROOT / mot["output_dir"] / args.dataset / args.split / path.stem / "gt.txt",
                        delimiter=",", ndmin=2)
        gt = gt[gt[:, 6] == 1]
        lines, found, n_gt, n_cand, area = [], 0, 0, 0, 0.0
        t0 = time.time()
        boxes = []
        for i, t in enumerate(m["types"]):
            f = i + 1
            if t == "P":
                boxes = candidates(m["sad"][i], m["intra"][i], ccfg)
            # I-frames: no encoder statistics, the previous candidates are kept
            px = [(x * mb, y * mb, bw * mb, bh * mb) for x, y, bw, bh in boxes]
            lines += [f"{f},{x},{y},{bw},{bh}\n" for x, y, bw, bh in px]
            n_cand += len(px)
            area += sum(bw * bh for _, _, bw, bh in px) / (w * h)
            for gx, gy, gw, gh in gt[gt[:, 0] == f][:, 2:6]:
                cx, cy = gx + gw / 2, gy + gh / 2
                n_gt += 1
                found += any(x <= cx < x + bw and y <= cy < y + bh for x, y, bw, bh in px)
        ms = 1000 * (time.time() - t0) / len(m["types"])
        (out_dir / f"{path.stem}.txt").write_text("".join(lines))
        n = len(m["types"])
        per_seq[path.stem] = {"frames": n, "gt": n_gt, "found": found,
                              "recall": round(found / max(n_gt, 1), 3),
                              "candidates_per_frame": round(n_cand / n, 2),
                              "area_pct": round(100 * area / n, 2), "ms_per_frame": round(ms, 2)}
        print(f"{path.stem}: recall {per_seq[path.stem]['recall']}, "
              f"{per_seq[path.stem]['candidates_per_frame']} candidates/frame, "
              f"{per_seq[path.stem]['area_pct']}% of the frame")

    def total(seqs):
        s = [per_seq[k] for k in seqs]
        frames = sum(x["frames"] for x in s)
        return {"sequences": len(s), "frames": frames,
                "recall": round(sum(x["found"] for x in s) / max(sum(x["gt"] for x in s), 1), 3),
                "candidates_per_frame": round(sum(x["candidates_per_frame"] * x["frames"] for x in s) / frames, 2),
                "area_pct": round(sum(x["area_pct"] * x["frames"] for x in s) / frames, 2),
                "ms_per_frame": round(sum(x["ms_per_frame"] * x["frames"] for x in s) / frames, 2)}

    res = {"dataset": args.dataset, "split": args.split, "config": ecfg,
           "all": total(per_seq), "sequences": per_seq, "hardware": hardware_info()}
    if args.sequences:
        test = json.loads((ROOT / args.sequences).read_text())["sequences"]["test"]
        res["heldout"] = total([s.split("/")[1] for s in test])
    save_json(res, ROOT / "results" / f"candidates_{args.dataset}_{args.split}.json")
    print("all:", res["all"])
    if "heldout" in res:
        print("held-out:", res["heldout"])


if __name__ == "__main__":
    main()
