"""HOTA / CLEAR / Identity metrics per dataset and class, with TrackEval."""
import argparse
import json

import numpy as np

# TrackEval still uses the aliases removed in NumPy 2
np.float, np.int = float, int
from trackeval.metrics import CLEAR, HOTA, Identity  # noqa: E402

from common import ROOT, hardware_info, load_config, save_json  # noqa: E402
from detection_metrics import bootstrap_hota, sequence_stats, summarize  # noqa: E402

METRICS = [HOTA(), CLEAR(), Identity()]


def load(path):
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 8))
    return np.loadtxt(path, delimiter=",", ndmin=2)[:, :8]


def iou(a, b):
    a, b = a[:, None, :], b[None, :, :]
    x1, y1 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2])
    y2 = np.minimum(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter, inter / (a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter)


def sequence_data(gt, trk, class_id, n_frames, ignore_ioa):
    gt_c = gt[(gt[:, 7] == class_id) & (gt[:, 6] == 1)]
    trk_c = trk[trk[:, 7] == class_id]
    ignore = gt[gt[:, 6] == 0]
    gt_map = {v: i for i, v in enumerate(np.unique(gt_c[:, 1]))}
    trk_map = {v: i for i, v in enumerate(np.unique(trk_c[:, 1]))}

    data = {"num_timesteps": n_frames, "num_gt_ids": len(gt_map), "num_tracker_ids": len(trk_map),
            "gt_ids": [], "tracker_ids": [], "similarity_scores": []}
    n_gt = n_trk = 0
    for f in range(1, n_frames + 1):
        g, t, ig = gt_c[gt_c[:, 0] == f], trk_c[trk_c[:, 0] == f], ignore[ignore[:, 0] == f]
        if len(t) and len(ig):
            inter, _ = iou(t[:, 2:6], ig[:, 2:6])
            t = t[(inter / (t[:, 4:5] * t[:, 5:6])).max(axis=1) <= ignore_ioa]
        data["gt_ids"].append(np.array([gt_map[i] for i in g[:, 1]], dtype=int))
        data["tracker_ids"].append(np.array([trk_map[i] for i in t[:, 1]], dtype=int))
        data["similarity_scores"].append(iou(g[:, 2:6], t[:, 2:6])[1] if len(g) and len(t)
                                         else np.zeros((len(g), len(t))))
        n_gt, n_trk = n_gt + len(g), n_trk + len(t)
    data["num_gt_dets"], data["num_tracker_dets"] = n_gt, n_trk
    return data


def summary(res):
    h, c, i = res["HOTA"], res["CLEAR"], res["Identity"]
    return {"HOTA": round(100 * float(np.mean(h["HOTA"])), 2),
            "DetA": round(100 * float(np.mean(h["DetA"])), 2),
            "AssA": round(100 * float(np.mean(h["AssA"])), 2),
            "MOTA": round(100 * float(c["MOTA"]), 2),
            "IDF1": round(100 * float(i["IDF1"]), 2),
            "IDSW": int(c["IDSW"]), "FP": int(c["CLR_FP"]), "FN": int(c["CLR_FN"]),
            "GT": int(c["CLR_TP"] + c["CLR_FN"]), **detection_scores(c)}


def detection_scores(c):
    tp, fp, fn = float(c["CLR_TP"]), float(c["CLR_FP"]), float(c["CLR_FN"])
    return {"Precision": round(100 * tp / max(tp + fp, 1), 2), "Recall": round(100 * tp / max(tp + fn, 1), 2),
            "F1": round(100 * 2 * tp / max(2 * tp + fp + fn, 1), 2)}


def confusion(per_class):
    # matching is done per class, so off-diagonal class pairs are not measured; background is
    # what nobody annotated (false positives) or what the tracker missed (false negatives)
    names = list(per_class)
    n = len(names)
    matrix = [[0] * (n + 1) for _ in range(n + 1)]
    for i, name in enumerate(names):
        c = per_class[name]["CLEAR"]
        matrix[i][i] = int(c["CLR_TP"])
        matrix[i][n] = int(c["CLR_FN"])
        matrix[n][i] = int(c["CLR_FP"])
    matrix[n][n] = None
    return {"labels": [*names, "background"], "matrix": matrix, "rows": "ground truth", "columns": "prediction"}


def operational(stats, thresholds, ecfg, hota_seq_res):
    out = summarize(stats, thresholds, ecfg["size_edges"], ecfg["bootstrap"])
    if hota_seq_res:
        out["intervals"]["mean.HOTA"] = bootstrap_hota(hota_seq_res, METRICS[0], ecfg["bootstrap"])
    return out


def evaluate(method, name, split, cfg, mot, cls, only=None):
    dcfg = cfg["datasets"][name]
    gt_root = ROOT / mot["output_dir"] / name / split
    trk_root = ROOT / cfg["output_dir"] / method / name / split
    seqs = sorted(p for p in gt_root.iterdir() if p.is_dir() and (only is None or p.name in only))

    ecfg = cfg["eval"]
    thresholds = [round(t, 2) for t in np.arange(*ecfg["thresholds"])]
    per_class, last_seq_res, stats = {}, {}, []
    for seq in seqs:
        info = dict(l.split("=", 1) for l in (seq / "seqinfo.ini").read_text().splitlines() if "=" in l)
        stats.append(sequence_stats(load(seq / "gt.txt"), load(trk_root / f"{seq.name}.txt"),
                                    [cls[c] for c in dcfg["eval_classes"]], int(info["seqLength"]),
                                    float(info.get("frameRate", 30)), ecfg["ignore_ioa"], thresholds))
    for cname in dcfg["eval_classes"]:
        seq_res = {}
        for seq in seqs:
            n = int(next(l for l in (seq / "seqinfo.ini").read_text().splitlines()
                         if l.startswith("seqLength")).split("=")[1])
            data = sequence_data(load(seq / "gt.txt"), load(trk_root / f"{seq.name}.txt"),
                                 cls[cname], n, cfg["eval"]["ignore_ioa"])
            seq_res[seq.name] = {type(m).__name__: m.eval_sequence(data) for m in METRICS}
        per_class[cname] = {type(m).__name__: m.combine_sequences(
            {s: r[type(m).__name__] for s, r in seq_res.items()}) for m in METRICS}
        last_seq_res = {s: r["HOTA"] for s, r in seq_res.items()}

    mean = {type(m).__name__: m.combine_classes_class_averaged(
        {c: r[type(m).__name__] for c, r in per_class.items()}) for m in METRICS}

    timing = json.loads((trk_root / "timing.json").read_text())
    done = [timing["sequences"][s.name] for s in seqs]
    frames = sum(s["frames"] for s in done)
    seconds = sum(s["seconds"] for s in done)
    return {"method": method, "dataset": name, "split": split,
            "sequences": [s.name for s in seqs],
            "frames": frames, "fps": round(frames / seconds, 1),
            "mean": summary(mean), "classes": {c: summary(r) for c, r in per_class.items()},
            "confusion_matrix": confusion(per_class),
            **operational(stats, thresholds, ecfg, last_seq_res if len(dcfg["eval_classes"]) == 1 else None),
            "settings": {k: v for k, v in timing.items() if k != "sequences"},
            "hardware": hardware_info()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="folder in results/tracking/")
    parser.add_argument("datasets", nargs="*", help="default: all in configs/track.yaml")
    parser.add_argument("--sequences", help="splits.json from make_yolo_dataset.py: evaluate "
                        "its test sequences only, results tagged _heldout")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "track.yaml")
    mot = load_config(ROOT / "configs" / "mot.yaml")
    cls = {name: i for i, name in enumerate(mot["classes"], 1)}

    only, tag = None, ""
    if args.sequences:
        test = json.loads((ROOT / args.sequences).read_text())["sequences"]["test"]
        only, tag = {s.split("/")[1] for s in test}, "_heldout"

    for name in args.datasets or list(cfg["datasets"]):
        for split in cfg["datasets"][name]["splits"]:
            if not (ROOT / cfg["output_dir"] / args.method / name / split).exists():
                continue
            res = evaluate(args.method, name, split, cfg, mot, cls, only)
            save_json(res, ROOT / "results" / f"metrics_{name}_{split}_{args.method}{tag}.json")
            m = res["mean"]
            print(f"{name}/{split}: HOTA {m['HOTA']}  MOTA {m['MOTA']}  IDF1 {m['IDF1']}  "
                  f"{res['fps']} fps")


if __name__ == "__main__":
    main()
