"""Tracking that calls the detector on I-frames only and follows the boxes on P-frames with the stream's motion vectors.

The GOP length decides how often the detector runs: the stream is encoded again with --gop (or read as is). The
residual model, if given, runs every --residual-every P-frames to catch moving objects that appeared since the
last I-frame and to correct drift; it never ends a track, since a static object has no residual.
Output is the same MOTChallenge format as track.py, so eval_mot.py reads it unchanged.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import av
import numpy as np
from av.video.frame import PictureType
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from common import ROOT, load_config, save_json
from compressed_video import ResidualSource, motion_field_fast, past_vectors, reencode


def iou_matrix(a, b):
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


class Tracks:
    def __init__(self, new_conf, max_missed, min_iou):
        self.items, self.next_id = [], 1
        self.new_conf, self.max_missed, self.min_iou = new_conf, max_missed, min_iou

    def move(self, flow, w, h):
        """Shift each box by the median motion vector under it; flow points to the reference frame, so the box moves by -flow."""
        for t in self.items:
            for _ in range(2):  # the second pass samples where the first one moved the box
                x1, y1 = max(int(round(t["box"][0])), 0), max(int(round(t["box"][1])), 0)
                x2, y2 = min(max(int(round(t["box"][2])), x1 + 1), w), min(max(int(round(t["box"][3])), y1 + 1), h)
                if x2 <= x1 or y2 <= y1:
                    break
                dx, dy = np.median(flow[y1:y2, x1:x2].reshape(-1, 2), axis=0)
                t["box"] = [t["box"][0] - dx, t["box"][1] - dy, t["box"][2] - dx, t["box"][3] - dy]
        self.items = [t for t in self.items if t["box"][2] > 0 and t["box"][3] > 0 and t["box"][0] < w and t["box"][1] < h]

    def update(self, boxes, confs, classes, end_tracks):
        """Match detections to tracks by IoU. end_tracks: an unmatched track counts a miss (a detector that sees everything)."""
        matched_t, matched_d = set(), set()
        if self.items and len(boxes):
            iou = iou_matrix([t["box"] for t in self.items], boxes)
            for ti, di in zip(*linear_sum_assignment(1 - iou)):
                if iou[ti, di] >= self.min_iou:
                    self.items[ti].update(box=list(map(float, boxes[di])), conf=float(confs[di]), cls=int(classes[di]), missed=0)
                    matched_t.add(ti)
                    matched_d.add(di)
        if end_tracks:
            for ti, t in enumerate(self.items):
                if ti not in matched_t:
                    t["missed"] += 1
            self.items = [t for t in self.items if t["missed"] <= self.max_missed]
        for di in range(len(boxes)):
            if di not in matched_d and confs[di] >= self.new_conf:
                self.items.append({"id": self.next_id, "box": list(map(float, boxes[di])), "conf": float(confs[di]),
                                   "cls": int(classes[di]), "missed": 0})
                self.next_id += 1


def detect(model, img, tcfg, class_ids):
    r = model.predict(img, imgsz=tcfg["imgsz"], conf=tcfg["conf"], classes=list(class_ids), device="cpu", verbose=False)[0].boxes
    return r.xyxy.numpy(), r.conf.numpy(), r.cls.numpy().astype(int)


def track_sequence(video, rgb_model, res_model, tcfg, class_ids, args, res_mode, gain):
    container = av.open(str(video))
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}
    w, h = stream.codec_context.width, stream.codec_context.height
    source = ResidualSource(w, h, res_mode, gain) if res_model else None
    tracks = Tracks(args.new_conf, args.max_missed, args.min_iou)
    lines, calls, n_p = [], {"rgb": 0, "residual": 0}, 0
    t0 = time.time()
    for n, frame in enumerate(container.decode(stream), 1):
        is_i = PictureType(frame.pict_type).name == "I"
        res_image = None
        if source is not None:
            # in luma mode following the stream is cheap, in rgb mode it converts every frame
            res_image, _, mvs = source.step(frame, compute=not is_i)
        else:
            mvs = None if is_i else past_vectors(frame)
        if is_i:
            boxes, confs, cls = detect(rgb_model, frame.to_ndarray(format="bgr24"), tcfg, class_ids)
            tracks.update(boxes, confs, cls, end_tracks=True)
            calls["rgb"] += 1
        else:
            n_p += 1
            if mvs is not None and len(mvs):
                tracks.move(motion_field_fast(mvs, h, w), w, h)
            if res_model is not None and res_image is not None and n_p % args.residual_every == 0:
                boxes, confs, cls = detect(res_model, res_image, tcfg, class_ids)
                tracks.update(boxes, confs, cls, end_tracks=False)
                calls["residual"] += 1
        for t in tracks.items:
            x1, y1, x2, y2 = t["box"]
            lines.append(f"{n},{t['id']},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{t['conf']:.3f},{class_ids[t['cls']]},-1,-1\n")
    container.close()
    return "".join(lines), n, time.time() - t0, calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="output folder name in results/tracking/")
    parser.add_argument("dataset")
    parser.add_argument("--rgb-model", required=True, help="detector run on I-frames")
    parser.add_argument("--residual-model", help="detector run on some P-frames, on the residual image")
    parser.add_argument("--residual-every", type=int, default=4, help="run the residual detector every N P-frames")
    parser.add_argument("--classes", nargs="+", required=True, help="model class names, in model order")
    parser.add_argument("--sequences", help="splits.json from make_yolo_dataset.py: test sequences only")
    parser.add_argument("--config", default="configs/track.yaml", help="detector settings (baseline section)")
    parser.add_argument("--gop", type=int, help="encode the videos again with this GOP length")
    parser.add_argument("--reencode", help="extra encoder settings as k=v,k=v, like track.py")
    parser.add_argument("--new-conf", type=float, default=0.25, help="minimum confidence to start a track")
    parser.add_argument("--max-missed", type=int, default=2, help="I-frame detections a track may miss before it ends")
    parser.add_argument("--min-iou", type=float, default=0.3)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    mot = load_config(ROOT / "configs" / "mot.yaml")
    rcfg = load_config(ROOT / "configs" / "residual.yaml")
    cls = {name: i for i, name in enumerate(mot["classes"], 1)}
    dcfg = cfg["datasets"][args.dataset]
    class_ids = {i: cls[c] for i, c in enumerate(args.classes)}

    rgb_model = YOLO(str(ROOT / args.rgb_model))
    res_model = YOLO(str(ROOT / args.residual_model)) if args.residual_model else None

    if args.sequences:
        todo = [tuple(s.split("/")) for s in json.loads((ROOT / args.sequences).read_text())["sequences"]["test"]]
    else:
        todo = [(split, p.name) for split in dcfg["splits"]
                for p in sorted((ROOT / mot["output_dir"] / args.dataset / split).iterdir()) if p.is_dir()]

    enc = None
    if args.gop or args.reencode:
        settings = dict(kv.split("=", 1) for kv in (args.reencode or "").split(",") if kv)
        enc = {**load_config(ROOT / "configs" / "compressed.yaml")["reencode"], **settings}
        if args.gop:
            enc["gop"] = args.gop
    tmp = Path(tempfile.mkdtemp())

    timing = {}
    for i, (split, seq) in enumerate(todo, 1):
        out = ROOT / cfg["output_dir"] / args.method / args.dataset / split
        out.mkdir(parents=True, exist_ok=True)
        video = ROOT / mot["output_dir"] / args.dataset / split / seq / "video.mp4"
        if enc:
            reencode(video, tmp / f"{seq}.mp4", enc)
            video = tmp / f"{seq}.mp4"
        text, n, sec, calls = track_sequence(video, rgb_model, res_model, cfg["baseline"], class_ids, args,
                                             rcfg.get("residual_mode", "rgb"), rcfg["residual_gain"])
        (out / f"{seq}.txt").write_text(text)
        timing.setdefault(split, {})[seq] = {"frames": n, "seconds": round(sec, 2), "detector_calls": calls}
        print(f"[{i}/{len(todo)}] {args.dataset}/{split}/{seq}: {n} frames, {n / sec:.1f} fps, calls {calls}", flush=True)

    for split, seqs in timing.items():
        save_json({"method": args.method, "input": "gop", "rgb_model": args.rgb_model, "residual_model": args.residual_model,
                   "gop": args.gop, "residual_every": args.residual_every if res_model else None,
                   **cfg["baseline"], "sequences": seqs},
                  ROOT / cfg["output_dir"] / args.method / args.dataset / split / "timing.json")


if __name__ == "__main__":
    main()
