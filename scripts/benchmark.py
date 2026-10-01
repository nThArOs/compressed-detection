"""Latency, throughput and memory of the full pipeline (decode, residual, inference) on one video,
with the CPU limits of the container."""
import argparse
import os
import resource
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from common import ROOT, hardware_info, limit_threads, load_config, model_size, save_json, tune_onnx
from serve import frames


def percentiles(values):
    v = np.array(values) * 1000
    return {"p50": round(float(np.percentile(v, 50)), 1), "p95": round(float(np.percentile(v, 95)), 1),
            "p99": round(float(np.percentile(v, 99)), 1), "mean": round(float(v.mean()), 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", choices=["rgb", "residual"], default="residual")
    parser.add_argument("--source", default="data/mot/dut_anti_uav/test/video01/video.mp4")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default="results/bench_dut_anti_uav_{input}_platform.json")
    args = parser.parse_args()

    threads = limit_threads()
    gain = load_config(ROOT / "configs" / "residual.yaml")["residual_gain"]
    t0 = time.perf_counter()
    model = YOLO(args.model, task="detect")
    stream = frames(str(ROOT / args.source), args.input, gain)
    _, first = next(stream)
    model.predict(first, imgsz=args.imgsz, device="cpu", verbose=False)
    cold_start = time.perf_counter() - t0
    tune_onnx(model, threads)

    inference, end_to_end, decode_and_input = [], [], []
    n = 0
    t_frame = time.perf_counter()
    t_start = None
    for _, img in stream:
        t_ready = time.perf_counter()
        t1 = time.perf_counter()
        model.predict(img, imgsz=args.imgsz, device="cpu", verbose=False)
        t2 = time.perf_counter()
        n += 1
        if n > args.warmup:
            t_start = t_start or t_frame
            inference.append(t2 - t1)
            decode_and_input.append(t_ready - t_frame)
            end_to_end.append(t2 - t_frame)
        t_frame = time.perf_counter()
        if n >= args.frames + args.warmup:
            break
    total = time.perf_counter() - t_start

    params_m, gflops = model_size(model, args.model, args.imgsz)
    out = Path(ROOT / args.out.format(input=args.input))
    save_json({
        "model": args.model,
        "input": args.input,
        "profile": os.environ.get("MLOPS_PROFILE"),
        "measured": False,
        "frames": len(inference),
        "threads": threads,
        "cold_start_ms": round(cold_start * 1000, 1),
        "latency_ms": percentiles(inference),
        "end_to_end_ms": percentiles(end_to_end),
        "stages_ms": {"decode_and_input": round(1000 * float(np.mean(decode_and_input)), 1),
                      "inference": round(1000 * float(np.mean(inference)), 1)},
        "fps": round(len(inference) / total, 2),
        "ram_peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "model_mb": round(Path(args.model).stat().st_size / 1e6, 2),
        "params_m": round(params_m, 2),
        **({"gflops": round(float(gflops), 2)} if gflops is not None else {}),
        "hardware": {**hardware_info(), "cpus_available": threads},
    }, out)
    print(f"{args.input}: p95 {percentiles(inference)['p95']} ms, {len(inference) / total:.2f} fps, "
          f"{threads} threads, cold start {cold_start:.1f} s")


if __name__ == "__main__":
    main()
