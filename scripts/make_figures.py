"""Figures of the README: RGB / motion / residual side by side, one GOP, confidence of the RGB and residual detectors."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

from common import ROOT

OUT = ROOT / "docs"
COLORS = {"rgb": "#2a6fdb", "residual": "#d9622b", "i": "#1f2933", "p": "#9aa5b1", "mv": "#2f9e44"}


def drone_center(residual, win=9):
    """The residual is flat grey except where something moved: the strongest blurred response is the drone."""
    d = np.abs(residual.astype(np.float32) - 128).sum(axis=2)
    kernel = np.ones(win, np.float32) / win
    d = np.apply_along_axis(lambda r: np.convolve(r, kernel, "same"), 1, d)
    d = np.apply_along_axis(lambda c: np.convolve(c, kernel, "same"), 0, d)
    y, x = np.unravel_index(int(d.argmax()), d.shape)
    return int(x), int(y)


def flows(video, frames, half):
    base = ROOT / "results" / "compressed" / video
    fig, axes = plt.subplots(len(frames), 5, figsize=(15, 2.6 * len(frames)), squeeze=False)
    titles = ["decoded P-frame (RGB)", "motion vectors", "residual", "RGB, zoom", "residual, zoom"]
    for row, n in zip(axes, frames):
        rgb = np.array(Image.open(base / "frames" / f"{n:06d}_P.jpg"))
        mv = np.array(Image.open(base / "flow" / f"{n:06d}_P.png"))
        res = np.array(Image.open(base / "residuals" / f"{n:06d}_P.png"))
        x, y = drone_center(res)
        x0, y0 = min(max(x - half, 0), rgb.shape[1] - 2 * half), min(max(y - half, 0), rgb.shape[0] - 2 * half)
        crop = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        for ax, img in zip(row, [rgb, mv, res, rgb[crop], res[crop]]):
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].add_patch(Rectangle((x0, y0), 2 * half, 2 * half, fill=False, ec="#2fdb4e", lw=1.5))
        row[0].set_ylabel(f"frame {n}", fontsize=11)
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "flows.png", dpi=110)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, tc="white", fs=10):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs)


def gop(length=12, residual_every=4):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(-0.3, length + 1.3)
    ax.set_ylim(-3.9, 3.6)
    ax.axis("off")
    for k in range(length + 1):
        is_i = k % length == 0
        box(ax, k + 0.05, 0, 0.9, 0.9, "I" if is_i else "P", COLORS["i"] if is_i else COLORS["p"], fs=13)
        ax.text(k + 0.5, -0.4, str(k), ha="center", fontsize=10, color="#52606d")
    ax.text(length + 0.5, 1.15, "next GOP", ha="center", fontsize=10, color="#52606d")
    ax.text(0, 3.45, "One GOP of %d frames" % length, fontsize=14, weight="bold")
    ax.add_patch(FancyArrowPatch((0.5, 0.95), (0.5, 2.1), arrowstyle="-|>", mutation_scale=16, color=COLORS["rgb"]))
    box(ax, 0, 2.15, 5.6, 1.0, "YOLO on the decoded I-frame\n(full image, RGB detector)", COLORS["rgb"], fs=11)
    for k in range(residual_every, length, residual_every):
        ax.add_patch(FancyArrowPatch((k + 0.5, 0.95), (k + 0.5, 1.75), arrowstyle="-|>", mutation_scale=12, color=COLORS["residual"]))
        ax.plot(k + 0.5, 1.9, "o", color=COLORS["residual"], ms=8)
    ax.text(length + 1.2, 2.6, "optional residual detector: every %d P-frames,\nor earlier when the residual shows a lot of motion\n(catches objects that appeared after the I-frame)" % residual_every,
            ha="right", va="center", fontsize=10.5, color=COLORS["residual"])
    ax.add_patch(FancyArrowPatch((1.0, -0.95), (length - 0.2, -0.95), arrowstyle="-|>", mutation_scale=16, color=COLORS["mv"]))
    ax.text(length / 2 + 0.4, -1.65, "P-frames: no detector, each box is moved by the median motion vector under it\n(vectors read from the stream)", ha="center", fontsize=11, color=COLORS["mv"])
    ax.text(length / 2 + 0.4, -2.75, "detector calls: 1 per %d frames instead of %d" % (length, length), ha="center", fontsize=12, weight="bold")
    ax.text(length / 2 + 0.4, -3.4, "the next I-frame resets the tracks and removes the accumulated drift", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "gop.png", dpi=110)
    plt.close(fig)


def confidence(rgb, residual):
    data = {}
    for name, path in [("rgb", rgb), ("residual", residual)]:
        data[name] = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(axes, ["precision", "recall", "f1"], ["precision", "recall", "F1"]):
        for name, d in data.items():
            c = d["curves"]["threshold"]
            ys = [np.nan if v is None else v for v in c[key]]
            ax.plot(c["x"], ys, color=COLORS[name], lw=2.2, label=f"{name} ({d['settings']['model'].split('/')[-1]})")
        ax.set_title(f"{title} against confidence threshold")
        ax.set_xlabel("confidence threshold")
        ax.set_ylabel("%")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "confidence.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="clip")
    ap.add_argument("--frames", type=int, nargs="+", default=[59, 119])
    ap.add_argument("--half", type=int, default=48, help="half size of the zoom window, in pixels")
    ap.add_argument("--rgb", default="results/metrics_dut_anti_uav_test_platform_rgb_heldout.json")
    ap.add_argument("--residual", default="results/metrics_dut_anti_uav_test_platform_residual_heldout.json")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    flows(args.video, args.frames, args.half)
    gop()
    confidence(ROOT / args.rgb, ROOT / args.residual)
