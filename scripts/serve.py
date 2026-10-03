"""Live detection on a video stream (file played in a loop, or RTSP), metrics on /metrics."""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import av
import cv2
import numpy as np
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from ultralytics import YOLO

from common import ROOT, limit_threads, load_config, tune_onnx
from compressed_video import ResidualSource, past_vectors

SECONDS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2.5, 5)
LATENCY = Histogram("inference_latency_seconds", "Model inference time per frame", buckets=SECONDS)
STAGE = Histogram("stage_latency_seconds", "Time per frame and stage", ["stage"], buckets=SECONDS)
FRAMES = Counter("inference_requests_total", "Frames processed")
ERRORS = Counter("inference_errors_total", "Frames that failed")
CONFIDENCE = Histogram("prediction_confidence", "Detection confidence",
                       buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1))
PER_FRAME = Histogram("predictions_per_input", "Detections per frame", buckets=(0, 1, 2, 3, 5, 10, 20))
FPS = Gauge("frames_per_second", "Processing rate over the last frames")
# cheap descriptors of the input, compared over time by the platform to detect drift without labels
MOTION = Histogram("input_motion_magnitude", "Mean motion vector length per frame, pixels",
                   buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16))
RESIDUAL = Histogram("input_residual_energy", "Mean absolute residual per frame",
                     buckets=(0.5, 1, 2, 3, 5, 8, 12, 20, 40))
BRIGHTNESS = Histogram("input_brightness", "Mean luma per frame", buckets=(32, 64, 96, 128, 160, 192, 224))

latest = {"frame": 0, "boxes": [], "fps": 0.0}
preview = {"frame": 0, "source": None, "input": None, "boxes": [], "cache": {}}
lock = threading.Lock()


def frames(source, mode, gain):
    container = av.open(source)
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}
    w, h = stream.codec_context.width, stream.codec_context.height
    residuals = ResidualSource(w, h, load_config(ROOT / "configs" / "residual.yaml").get("residual_mode", "rgb"), gain)
    decoded = iter(container.decode(stream))
    first = True
    while True:
        t0 = time.perf_counter()
        frame = next(decoded, None)
        if frame is None:
            break
        img = frame.to_ndarray(format="bgr24")
        STAGE.labels("decode").observe(time.perf_counter() - t0)
        BRIGHTNESS.observe(float(img.mean()))
        if mode == "rgb":
            mvs = None if first else past_vectors(frame)
            out = res = None
        else:
            t0 = time.perf_counter()
            out, res, mvs = residuals.step(frame, img)
            STAGE.labels("residual").observe(time.perf_counter() - t0)
        first = False
        if mvs is not None and len(mvs):
            scale = mvs["motion_scale"].astype(np.float32)
            MOTION.observe(float(np.hypot(mvs["motion_x"] / scale, mvs["motion_y"] / scale).mean()))
            if res is not None:
                RESIDUAL.observe(float(np.abs(res).mean()))
        if mode == "rgb":
            yield img, img
        else:
            yield img, out if out is not None else np.full_like(img, 128)
    container.close()


def process(model, args, source, img, state):
    t0 = time.perf_counter()
    try:
        result = model.predict(img, conf=args.conf, imgsz=args.imgsz, device="cpu", verbose=False)[0]
        boxes = result.boxes
        if args.classes:
            boxes = boxes[[result.names[int(c)] in args.classes for c in boxes.cls.tolist()]]
    except Exception:
        ERRORS.inc()
        return
    dt = time.perf_counter() - t0
    if not state["tuned"]:
        tune_onnx(model, state["threads"])
        state["tuned"] = True
    LATENCY.observe(dt)
    STAGE.labels("inference").observe(dt)
    FRAMES.inc()
    confs = boxes.conf.tolist()
    for c in confs:
        CONFIDENCE.observe(c)
    PER_FRAME.observe(len(confs))
    state["n"] += 1
    now = time.time()
    state["recent"] = [t for t in state["recent"] if t > now - 10] + [now]
    recent = state["recent"]
    fps = len(recent) / max(recent[-1] - recent[0], 1e-6) if len(recent) > 1 else 0.0
    FPS.set(fps)
    latest.update(frame=state["n"], fps=round(fps, 2), boxes=[
        {"xyxy": [round(v, 1) for v in b], "conf": round(c, 3)} for b, c in zip(boxes.xyxy.tolist(), confs)])
    with lock:
        preview.update(frame=state["n"], source=source, input=img, boxes=latest["boxes"], cache={})


def run(args):
    threads = limit_threads()
    print(f"torch threads: {threads}", flush=True)
    model = YOLO(args.model, task="detect")
    state = {"n": 0, "recent": [], "threads": threads, "tuned": False}
    gain = load_config(ROOT / "configs" / "residual.yaml")["residual_gain"]
    sources = [s.strip() for s in args.source.split(",") if s.strip()]
    while True:
        for path in sources:
            for source, img in frames(path, args.input, gain):
                process(model, args, source, img, state)
        if not args.loop:
            break


def render(view, width):
    with lock:
        key = (view, width)
        if key in preview["cache"]:
            return preview["cache"][key]
        img, boxes = preview[view], preview["boxes"]
    if img is None:
        return None
    t0 = time.perf_counter()
    # shrink first, then draw: the preview must stay cheap next to inference
    scale = width / img.shape[1] if width and img.shape[1] > width else 1.0
    out = cv2.resize(img, (width, round(img.shape[0] * scale)), interpolation=cv2.INTER_LINEAR) if scale < 1 else img.copy()
    for b in boxes:
        x1, y1, x2, y2 = (int(v * scale) for v in b["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (52, 104, 235), 2)
        cv2.putText(out, f"{b['conf']:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 104, 235), 1)
    data = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
    STAGE.labels("preview").observe(time.perf_counter() - t0)
    with lock:
        preview["cache"][key] = data
    return data


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/metrics":
            return self.reply(200, generate_latest(), CONTENT_TYPE_LATEST)
        if self.path == "/health":
            return self.reply(200, {"status": "ok"})
        if self.path == "/latest":
            return self.reply(200, latest)
        if self.path.startswith("/frame.jpg"):
            query = dict(q.split("=", 1) for q in self.path.partition("?")[2].split("&") if "=" in q)
            view = "source" if query.get("view") == "source" else "input"
            data = render(view, int(query.get("width", 960)))
            return self.reply(200, data, "image/jpeg") if data else self.reply(503, {"detail": "no frame yet"})
        self.reply(404, {"detail": "not found"})

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--source", default="data/mot/dut_anti_uav/test/video01/video.mp4",
                        help="video file or RTSP URL, several separated by commas are played in turn")
    parser.add_argument("--input", choices=["rgb", "residual"], default="residual")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--classes", nargs="+", help="class names to keep (default: all)")
    parser.add_argument("--no-loop", dest="loop", action="store_false")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"serving {args.model} on :{args.port}, source {args.source}, input {args.input}", flush=True)
    run(args)


if __name__ == "__main__":
    main()
