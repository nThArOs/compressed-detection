"""Render the residual dumped by the patched FFmpeg (tools/ffmpeg-residual) as a video.

Usage: python scripts/h264_residual.py <video.res> <width> <height> <out.mp4> [--gain 4] [--fps 25]
Pixel = 128 + gain * residual; intra macroblocks of P-frames are tinted blue, I-frames are mid grey.
"""
import argparse
from pathlib import Path

import numpy as np

from compressed_video import VideoWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("res")
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    parser.add_argument("out")
    parser.add_argument("--gain", type=float, default=4)
    parser.add_argument("--fps", type=float, default=25)
    args = parser.parse_args()

    mbw, mbh = -(-args.width // 16), -(-args.height // 16)
    pw, ph = mbw * 16, mbh * 16
    res_bytes, flag_bytes = pw * ph * 2, mbw * mbh
    data = np.memmap(args.res, np.uint8, "r")
    frames = len(data) // (res_bytes + flag_bytes)
    writer = VideoWriter(Path(args.out), args.fps, args.width, args.height)

    def all_intra(i):
        base = i * (res_bytes + flag_bytes) + res_bytes
        return bool(np.frombuffer(data[base:base + flag_bytes], np.uint8).all())

    # the decoder reports the first picture twice
    first = 1 if frames > 1 and all_intra(0) and all_intra(1) else 0
    for i in range(first, frames):
        base = i * (res_bytes + flag_bytes)
        res = np.frombuffer(data[base:base + res_bytes], np.int16).reshape(ph, pw)
        intra = np.frombuffer(data[base + res_bytes:base + res_bytes + flag_bytes], np.uint8).reshape(mbh, mbw)
        gray = np.clip(128 + args.gain * res.astype(np.float32), 0, 255).astype(np.uint8)
        img = np.dstack([gray, gray, gray])
        if not intra.all():
            mask = np.kron(intra, np.ones((16, 16), np.uint8)).astype(bool)
            img[mask] = (160, 90, 60)  # BGR, blue tint
        else:
            img[:] = 128
        writer.write(img[:args.height, :args.width])
    writer.close()
    print(f"{frames - first} frames, {args.out}")


if __name__ == "__main__":
    main()
