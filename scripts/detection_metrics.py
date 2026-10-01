"""Operational detection metrics on MOT files: false alarms per hour (a false alarm is a predicted track
whose boxes mostly match no annotated object), recall by object size, delay to
the first detection, precision and recall against the confidence threshold, bootstrap intervals."""
import numpy as np

IOU = 0.5


def iou(a, b):
    a, b = a[:, None, :], b[None, :, :]
    x1, y1 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2])
    y2 = np.minimum(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter)


def match(gt, trk):
    """Greedy one-to-one matching, best IoU first. Returns a boolean per gt box and per tracker box."""
    g, t = np.zeros(len(gt), bool), np.zeros(len(trk), bool)
    if not len(gt) or not len(trk):
        return g, t
    m = iou(gt[:, 2:6], trk[:, 2:6])
    for gi, ti in zip(*np.unravel_index(np.argsort(-m, axis=None), m.shape)):
        if m[gi, ti] < IOU:
            break
        if not g[gi] and not t[ti]:
            g[gi] = t[ti] = True
    return g, t


def drop_ignored(trk, ignore, ioa):
    if not len(trk) or not len(ignore):
        return trk
    a, b = trk[:, None, 2:6], ignore[None, :, 2:6]
    x1, y1 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2])
    y2 = np.minimum(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return trk[(inter / (trk[:, None, 4] * trk[:, None, 5])).max(axis=1) <= ioa]


def sequence_stats(gt, trk, class_ids, n_frames, fps, ignore_ioa, thresholds):
    gt_c = gt[np.isin(gt[:, 7], class_ids) & (gt[:, 6] == 1)]
    ignore = gt[gt[:, 6] == 0]
    trk_c = trk[np.isin(trk[:, 7], class_ids)]
    sizes, found, first_seen, first_hit, track_hits = [], [], {}, {}, {}
    tp = fp = fa_frames = 0
    curve = np.zeros((len(thresholds), 3))
    for f in range(1, n_frames + 1):
        g = gt_c[gt_c[:, 0] == f]
        t = drop_ignored(trk_c[trk_c[:, 0] == f], ignore[ignore[:, 0] == f], ignore_ioa)
        hit_g, hit_t = match(g, t)
        tp, fp = tp + int(hit_t.sum()), fp + int((~hit_t).sum())
        fa_frames += int((~hit_t).any())
        for tid, hit in zip(t[:, 1].astype(int), hit_t):
            n_hit, n_all = track_hits.get(tid, (0, 0))
            track_hits[tid] = (n_hit + int(hit), n_all + 1)
        sizes += list(np.sqrt(g[:, 4] * g[:, 5]))
        found += list(hit_g)
        for row, hit in zip(g, hit_g):
            first_seen.setdefault(int(row[1]), f)
            if hit:
                first_hit.setdefault(int(row[1]), f)
        for k, th in enumerate(thresholds):
            hg, ht = match(g, t[t[:, 6] >= th])
            curve[k] += (ht.sum(), (~ht).sum(), (~hg).sum())
    delays = [(first_hit[i] - first_seen[i]) / fps for i in first_seen if i in first_hit]
    return {"tp": tp, "fp": fp, "fn": len(found) - int(np.sum(found)), "seconds": n_frames / fps,
            "sizes": np.array(sizes), "found": np.array(found, bool), "delays": delays,
            "tracks": len(first_seen), "missed_tracks": len(first_seen) - len(first_hit), "curve": curve,
            "false_tracks": sum(1 for n_hit, n_all in track_hits.values() if n_hit < 0.5 * n_all),
            "predicted_tracks": len(track_hits), "fa_frames": fa_frames, "frames": n_frames}


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def summarize(stats, thresholds, size_edges, n_boot=1000, seed=0):
    tp, fp, fn = (sum(s[k] for s in stats) for k in ("tp", "fp", "fn"))
    hours = sum(s["seconds"] for s in stats) / 3600
    sizes = np.concatenate([s["sizes"] for s in stats])
    found = np.concatenate([s["found"] for s in stats])
    delays = np.array([d for s in stats for d in s["delays"]])
    edges = [0, *size_edges, np.inf]
    names = ["small", *(["medium"] if len(size_edges) == 2 else [f"medium{i + 1}" for i in range(len(size_edges) - 1)]),
             "large"]
    by_size, count_by_size = {}, {}
    for name, lo, hi in zip(names, edges, edges[1:]):
        sel = (sizes >= lo) & (sizes < hi)
        count_by_size[name] = int(sel.sum())
        by_size[name] = round(100 * float(found[sel].mean()), 2) if sel.any() else None
    curve = sum(s["curve"] for s in stats)
    p = curve[:, 0] / np.maximum(curve[:, 0] + curve[:, 1], 1)
    r = curve[:, 0] / np.maximum(curve[:, 0] + curve[:, 2], 1)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = [stats[i] for i in rng.integers(0, len(stats), len(stats))]
        boot.append(100 * f1(*(sum(s[k] for s in pick) for k in ("tp", "fp", "fn"))))
    return {
        "operational": {
            "false_alarms_per_hour": round(sum(s["false_tracks"] for s in stats) / hours, 1) if hours else None,
            "false_tracks": sum(s["false_tracks"] for s in stats),
            "predicted_tracks": sum(s["predicted_tracks"] for s in stats),
            "time_with_false_alarm_pct": round(100 * sum(s["fa_frames"] for s in stats)
                                               / max(sum(s["frames"] for s in stats), 1), 2),
            "video_hours": round(hours, 3),
            "detection_delay_s": {k: round(float(v), 2) for k, v in (
                ("median", np.median(delays)), ("mean", delays.mean()), ("p90", np.percentile(delays, 90)))}
            if len(delays) else None,
            "tracks": sum(s["tracks"] for s in stats),
            "tracks_never_detected": sum(s["missed_tracks"] for s in stats),
        },
        "slices": {"recall_by_size": by_size, "objects_by_size": count_by_size, "size_edges_px": list(size_edges)},
        "curves": {"threshold": {"x": [round(float(t), 2) for t in thresholds],
                                 "precision": [round(100 * float(v), 2) for v in p],
                                 "recall": [round(100 * float(v), 2) for v in r],
                                 "f1": [round(100 * float(v), 2) for v in 2 * p * r / np.maximum(p + r, 1e-9)]}},
        "intervals": {"mean.F1": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)]},
    }


def bootstrap_hota(seq_results, metric, n_boot=1000, seed=0):
    """95 % interval of HOTA when sequences are resampled with replacement."""
    names = list(seq_results)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = {f"{names[i]}#{k}": seq_results[names[i]] for k, i in enumerate(rng.integers(0, len(names), len(names)))}
        out.append(100 * float(np.mean(metric.combine_sequences(pick)["HOTA"])))
    return [round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]
