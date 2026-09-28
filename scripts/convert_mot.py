"""Convert VisDrone-MOT, UAVDT and DUT Anti-UAV to MOTChallenge format with an H.264 video per sequence."""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2

from common import ROOT, load_config, save_json


def visdrone(cfg, cls):
    for split, root in cfg["splits"].items():
        root = ROOT / root
        for seq in sorted((root / "sequences").iterdir()):
            rows = []
            for line in (root / "annotations" / f"{seq.name}.txt").read_text().split():
                f, tid, x, y, w, h, score, cat = map(int, line.split(",")[:8])
                if score == 0 or cat == 0:
                    rows.append((f, -1, x, y, w, h, 0, -1))
                else:
                    rows.append((f, tid, x, y, w, h, 1, cls[cfg["categories"][cat]]))
            yield split, seq.name, seq, rows


def uavdt(cfg, cls):
    gt_dir = ROOT / cfg["gt"]
    for seq in sorted((ROOT / cfg["images"]).iterdir()):
        rows = []
        for line in (gt_dir / f"{seq.name}_gt_whole.txt").read_text().split():
            f, tid, x, y, w, h, _, _, cat = map(int, line.split(","))
            rows.append((f, tid, x, y, w, h, 1, cls[cfg["categories"][cat]]))
        for line in (gt_dir / f"{seq.name}_gt_ignore.txt").read_text().split():
            f, _, x, y, w, h = map(int, line.split(",")[:6])
            rows.append((f, -1, x, y, w, h, 0, -1))
        yield ("test" if seq.name in cfg["test"] else "train"), seq.name, seq, rows


def dut_anti_uav(cfg, cls):
    gt_dir = ROOT / cfg["gt"]
    for seq in sorted((ROOT / cfg["images"]).iterdir()):
        rows = []
        for f, line in enumerate((gt_dir / f"{seq.name}_gt.txt").read_text().splitlines(), 1):
            x, y, w, h = map(int, line.split())
            if w > 0 and h > 0:
                rows.append((f, 1, x, y, w, h, 1, cls["drone"]))
        yield "test", seq.name, seq, rows


READERS = {"visdrone_mot": visdrone, "uavdt": uavdt, "dut_anti_uav": dut_anti_uav}


def encode(img_dir, pattern, fps, out, enc):
    gop = str(enc["gop"])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-start_number", "1", "-i", str(img_dir / pattern),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", str(enc["crf"]), "-g", gop, "-keyint_min", gop,
                    "-sc_threshold", "0", "-bf", str(enc["bframes"]), "-refs", str(enc["refs"]),
                    str(out)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help="names from configs/mot.yaml (default: all)")
    parser.add_argument("--no-video", action="store_true", help="annotations only")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "mot.yaml")
    enc = load_config(ROOT / "configs" / "compressed.yaml")["reencode"]
    cls = {name: i for i, name in enumerate(cfg["classes"], 1)}
    out_root = ROOT / cfg["output_dir"]

    stats = {}
    for name in args.datasets or list(cfg["datasets"]):
        dcfg = cfg["datasets"][name]
        for split, seq_name, img_dir, rows in READERS[name](dcfg, cls):
            out = out_root / name / split / seq_name
            out.mkdir(parents=True, exist_ok=True)
            frames = sorted(img_dir.glob("*.jpg"))
            h, w = cv2.imread(str(frames[0])).shape[:2]

            rows.sort()
            (out / "gt.txt").write_text("".join(f"{f},{t},{x},{y},{bw},{bh},{c},{k},-1\n"
                                                for f, t, x, y, bw, bh, c, k in rows))
            (out / "seqinfo.ini").write_text(
                f"[Sequence]\nname={seq_name}\nimDir={img_dir.relative_to(ROOT).as_posix()}\n"
                f"frameRate={dcfg['fps']}\nseqLength={len(frames)}\nimWidth={w}\nimHeight={h}\n"
                f"imExt=.jpg\n")
            if not args.no_video and not (out / "video.mp4").exists():
                encode(img_dir, dcfg["frame_pattern"], dcfg["fps"], out / "video.mp4", enc)

            s = stats.setdefault(name, {}).setdefault(split, {
                "sequences": 0, "frames": 0, "boxes": 0, "tracks": 0, "classes": Counter()})
            valid = [r for r in rows if r[6] == 1]
            s["sequences"] += 1
            s["frames"] += len(frames)
            s["boxes"] += len(valid)
            s["tracks"] += len({r[1] for r in valid})
            s["classes"].update(cfg["classes"][r[7] - 1] for r in valid)
            print(f"{name}/{split}/{seq_name}: {len(frames)} frames, {len(valid)} boxes")

        for split, s in stats[name].items():
            (out_root / name / split / "seqmap.txt").write_text(
                "name\n" + "".join(f"{p.name}\n" for p in sorted((out_root / name / split).iterdir())
                                   if p.is_dir()))
            s["classes"] = dict(s["classes"].most_common())

    path = ROOT / "results" / "datasets.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    save_json(previous | stats, path)


if __name__ == "__main__":
    main()
