"""Detector + ByteTrack on the RGB frames or on the residual, output in MOTChallenge format."""
import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
from ultralytics import YOLO

from common import ROOT, load_config, save_json
from compressed_video import motion_field, past_vectors, residual


def frames(video, mode, gain):
    container = av.open(str(video))
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}
    w, h = stream.codec_context.width, stream.codec_context.height
    grid = np.dstack(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)))
    prev = None
    for frame in container.decode(stream):
        img = frame.to_ndarray(format="bgr24")
        if mode == "rgb":
            yield img
        else:
            mvs = past_vectors(frame) if prev is not None else None
            # I-frames have no residual: an empty frame, the tracker coasts through it
            yield (np.clip(128 + gain * residual(img, prev, motion_field(mvs, h, w), grid), 0, 255)
                   .astype(np.uint8) if mvs is not None and len(mvs) else np.full_like(img, 128))
        prev = img
    container.close()


def track_sequence(model, video, mode, tcfg, class_ids, gain):
    model.predictor = None  # fresh tracker state per sequence
    lines, n = [], 0
    t0 = time.time()
    for n, img in enumerate(frames(video, mode, gain), 1):
        b = model.track(img, persist=True, tracker=tcfg["tracker"], conf=tcfg["conf"],
                        imgsz=tcfg["imgsz"], classes=list(class_ids), device="cpu",
                        verbose=False)[0].boxes
        if b.id is None:
            continue
        for (x1, y1, x2, y2), tid, conf, c in zip(b.xyxy.tolist(), b.id.int().tolist(),
                                                  b.conf.tolist(), b.cls.int().tolist()):
            lines.append(f"{n},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},"
                         f"{conf:.3f},{class_ids[c]},-1,-1\n")
    return "".join(lines), n, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="output folder name in results/tracking/")
    parser.add_argument("dataset")
    parser.add_argument("--input", choices=["rgb", "residual"], default="rgb")
    parser.add_argument("--model", help="default: the dataset model in configs/track.yaml")
    parser.add_argument("--classes", nargs="+", help="model class names, in model order "
                        "(default: class_map in configs/track.yaml)")
    parser.add_argument("--sequences", help="splits.json from make_yolo_dataset.py: test sequences only")
    parser.add_argument("--config", default="configs/track.yaml", help="tracker settings")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    mot = load_config(ROOT / "configs" / "mot.yaml")
    gain = load_config(ROOT / "configs" / "residual.yaml")["residual_gain"]
    cls = {name: i for i, name in enumerate(mot["classes"], 1)}
    dcfg = cfg["datasets"][args.dataset]

    model = YOLO(str(ROOT / (args.model or dcfg["model"])))
    class_map = dict(enumerate(args.classes)) if args.classes else dcfg["class_map"]
    class_ids = {k: cls[v] for k, v in class_map.items()}

    if args.sequences:
        test = json.loads((ROOT / args.sequences).read_text())["sequences"]["test"]
        todo = [tuple(s.split("/")) for s in test]
    else:
        todo = [(split, p.name) for split in dcfg["splits"]
                for p in sorted((ROOT / mot["output_dir"] / args.dataset / split).iterdir()) if p.is_dir()]

    timing = {}
    for i, (split, seq) in enumerate(todo, 1):
        out = ROOT / cfg["output_dir"] / args.method / args.dataset / split
        out.mkdir(parents=True, exist_ok=True)
        text, n, sec = track_sequence(model, ROOT / mot["output_dir"] / args.dataset / split / seq /
                                      "video.mp4", args.input, cfg["baseline"], class_ids, gain)
        (out / f"{seq}.txt").write_text(text)
        timing.setdefault(split, {})[seq] = {"frames": n, "seconds": round(sec, 2)}
        print(f"[{i}/{len(todo)}] {args.dataset}/{split}/{seq}: {n} frames, {n / sec:.1f} fps", flush=True)

    for split, seqs in timing.items():
        save_json({"method": args.method, "input": args.input, "model": args.model or dcfg["model"],
                   **cfg["baseline"], "sequences": seqs},
                  ROOT / cfg["output_dir"] / args.method / args.dataset / split / "timing.json")


if __name__ == "__main__":
    main()
