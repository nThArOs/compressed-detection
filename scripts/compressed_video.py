"""Extract I-frames, motion vectors and residuals from compressed videos (CoViAR-style)."""
import argparse
import csv
import gc
import subprocess
import time
from pathlib import Path

import av
import cv2
import numpy as np
from av.video.frame import PictureType

from common import ROOT, hardware_info, load_config, save_json

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def reencode(src, dst, cfg):
    gop = str(cfg["gop"])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(cfg["crf"]),
                    "-g", gop, "-keyint_min", gop, "-sc_threshold", "0",
                    "-bf", str(cfg["bframes"]), "-refs", str(cfg["refs"]), str(dst)], check=True)


class VideoWriter:
    def __init__(self, path, fps, w, h, crf=None):
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=fps,
                                                options={"crf": str(crf)} if crf else None)
        self.stream.width, self.stream.height, self.stream.pix_fmt = w, h, "yuv420p"

    def write(self, img):
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


_reads = 0


def past_vectors(frame):
    global _reads
    # PyAV's motion vector side data sits in reference cycles the collector reaches too late:
    # without this a stream grows by about 0.1 MB per frame until the container is killed
    _reads += 1
    if _reads % 100 == 0:
        gc.collect()
    sd = frame.side_data.get("MOTION_VECTORS")
    if sd is None:
        return None
    mvs = sd.to_ndarray()
    # blocks predicted from a future frame (B-frames) are ignored
    return mvs[mvs["source"] < 0]


def motion_field(mvs, h, w):
    # per-pixel offset to the reference frame, filled block by block
    flow = np.zeros((h, w, 2), np.float32)
    scale = mvs["motion_scale"].astype(np.float32)
    dx, dy = mvs["motion_x"] / scale, mvs["motion_y"] / scale
    x0 = np.clip(mvs["dst_x"] - mvs["w"] // 2, 0, w)
    y0 = np.clip(mvs["dst_y"] - mvs["h"] // 2, 0, h)
    x1 = np.clip(x0 + mvs["w"], 0, w)
    y1 = np.clip(y0 + mvs["h"], 0, h)
    for a, b, c, d, u, v in zip(x0.tolist(), y0.tolist(), x1.tolist(), y1.tolist(),
                                dx.tolist(), dy.tolist()):
        flow[b:d, a:c] = (u, v)
    return flow


def residual(cur, ref, flow, grid):
    pred = cv2.remap(ref, grid[..., 0] + flow[..., 0], grid[..., 1] + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cur.astype(np.int16) - pred.astype(np.int16)


def flow_image(flow, max_mag):
    # hue = direction, brightness = magnitude, black = static
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 255 / max_mag, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_vectors(img, mvs, min_mag):
    out = img.copy()
    mag = np.hypot(mvs["motion_x"], mvs["motion_y"]) / mvs["motion_scale"]
    for mv in mvs[mag >= min_mag]:
        cv2.arrowedLine(out, (int(mv["src_x"]), int(mv["src_y"])),
                        (int(mv["dst_x"]), int(mv["dst_y"])), (0, 255, 0), 1, tipLength=0.3)
    return out


def process(path, cfg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    src = path
    if cfg["reencode"]["enabled"]:
        src = out_dir / f"source_gop{cfg['reencode']['gop']}.mp4"
        reencode(path, src, cfg["reencode"])

    save = cfg["save"]
    for key in ("frames", "residuals", "mv", "flow", "raw"):
        if save[key]:
            (out_dir / key).mkdir(exist_ok=True)
    jpg = [cv2.IMWRITE_JPEG_QUALITY, cfg["jpeg_quality"]]
    gain = cfg["residual_gain"]

    container = av.open(str(src))
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}
    fps = stream.average_rate or 25
    w, h = stream.codec_context.width, stream.codec_context.height
    grid = np.dstack(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)))
    writers = {}
    if save["videos"]:
        writers = {k: VideoWriter(out_dir / f"{k}.mp4", fps, w, h)
                   for k in ("decoded", "residual", "mv", "flow")}

    rows, types, maes = [], {}, []
    prev = None
    t0 = time.time()
    for i, frame in enumerate(container.decode(stream)):
        img = frame.to_ndarray(format="bgr24")
        ptype = PictureType(frame.pict_type).name
        types[ptype] = types.get(ptype, 0) + 1
        mvs = past_vectors(frame) if prev is not None else None

        res = mv_img = None
        flow_img = np.zeros_like(img)
        if mvs is not None and len(mvs):
            flow = motion_field(mvs, h, w)
            res = residual(img, prev, flow, grid)
            mv_img = draw_vectors(img, mvs, cfg["mv_min_magnitude"])
            flow_img = flow_image(flow, cfg["flow_max_magnitude"])
            maes.append(float(np.abs(res).mean()))
        res_img = (np.full_like(img, 128) if res is None
                   else np.clip(128 + gain * res, 0, 255).astype(np.uint8))

        stem = f"{i:06d}_{ptype}"
        if save["frames"]:
            cv2.imwrite(str(out_dir / "frames" / f"{stem}.jpg"), img, jpg)
        if res is not None and save["residuals"]:
            cv2.imwrite(str(out_dir / "residuals" / f"{stem}.png"), res_img)
        if mv_img is not None and save["mv"]:
            cv2.imwrite(str(out_dir / "mv" / f"{stem}.jpg"), mv_img, jpg)
        if res is not None and save["flow"]:
            cv2.imwrite(str(out_dir / "flow" / f"{stem}.png"), flow_img)
        if res is not None and save["raw"]:
            np.savez_compressed(out_dir / "raw" / f"{stem}.npz", residual=res, mvs=mvs)
        if writers:
            writers["decoded"].write(img)
            writers["residual"].write(res_img)
            writers["mv"].write(img if mv_img is None else mv_img)
            writers["flow"].write(flow_img)

        rows.append({"frame": i, "time": round(float(frame.time or 0), 3), "type": ptype,
                     "mvs": 0 if mvs is None else len(mvs),
                     "residual_mae": round(maes[-1], 2) if res is not None else ""})
        prev = img

    container.close()
    for wr in writers.values():
        wr.close()

    with open(out_dir / "frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "source": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
        "resolution": [w, h],
        "frames": len(rows),
        "frame_types": types,
        "mean_mvs_per_frame": round(sum(r["mvs"] for r in rows) / max(len(rows), 1), 1),
        "mean_residual_mae": round(sum(maes) / len(maes), 2) if maes else None,
        "fps": round(len(rows) / (time.time() - t0), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="video file or folder of videos")
    parser.add_argument("--config", default=str(ROOT / "configs" / "compressed.yaml"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_root = ROOT / cfg["output_dir"]
    p = Path(args.source)
    sources = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT) if p.is_dir() else [p]

    summary = {"config": cfg, "hardware": hardware_info(), "videos": {}}
    for src in sources:
        stats = process(src, cfg, out_root / src.stem)
        summary["videos"][src.stem] = stats
        print(f"{src.stem:20s} {stats['frames']} frames {stats['frame_types']}, "
              f"residual MAE {stats['mean_residual_mae']}, {stats['fps']} fps")

    save_json(summary, out_root / "summary.json")


if __name__ == "__main__":
    main()
