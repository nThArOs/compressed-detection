"""Baseline: RGB detector on every frame + ByteTrack, output in MOTChallenge format."""
import argparse
import time

from ultralytics import YOLO

from common import ROOT, load_config, save_json

METHOD = "baseline_rgb"


def track_sequence(model, video, bcfg, class_ids):
    model.predictor = None  # fresh tracker state per sequence
    lines, frames = [], 0
    t0 = time.time()
    for frames, r in enumerate(model.track(source=str(video), stream=True, persist=True,
                                           tracker=bcfg["tracker"], conf=bcfg["conf"],
                                           imgsz=bcfg["imgsz"], classes=list(class_ids),
                                           device="cpu", verbose=False), 1):
        b = r.boxes
        if b.id is None:
            continue
        for (x1, y1, x2, y2), tid, conf, c in zip(b.xyxy.tolist(), b.id.int().tolist(),
                                                  b.conf.tolist(), b.cls.int().tolist()):
            lines.append(f"{frames},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},"
                         f"{conf:.3f},{class_ids[c]},-1,-1\n")
    return "".join(lines), frames, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help="names from configs/track.yaml (default: all)")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "track.yaml")
    mot = load_config(ROOT / "configs" / "mot.yaml")
    cls = {name: i for i, name in enumerate(mot["classes"], 1)}

    for name in args.datasets or list(cfg["datasets"]):
        dcfg = cfg["datasets"][name]
        model = YOLO(str(ROOT / dcfg["model"]))
        class_ids = {k: cls[v] for k, v in dcfg["class_map"].items()}
        for split in dcfg["splits"]:
            out = ROOT / cfg["output_dir"] / METHOD / name / split
            out.mkdir(parents=True, exist_ok=True)
            timing = {}
            for seq in sorted(p for p in (ROOT / mot["output_dir"] / name / split).iterdir()
                              if p.is_dir()):
                text, frames, sec = track_sequence(model, seq / "video.mp4", cfg["baseline"],
                                                   class_ids)
                (out / f"{seq.name}.txt").write_text(text)
                timing[seq.name] = {"frames": frames, "seconds": round(sec, 2)}
                print(f"{name}/{split}/{seq.name}: {frames} frames, {frames / sec:.1f} fps")
            save_json({"method": METHOD, "model": dcfg["model"], **cfg["baseline"],
                       "sequences": timing}, out / "timing.json")


if __name__ == "__main__":
    main()
