"""Per-macroblock maps a hardware encoder would export: motion vector, SAD and intra flag."""
import argparse
import time

import av
import cv2
import numpy as np

from common import ROOT, load_config


def cell_field(mvs, h4, w4):
    # one vector per 4x4 cell, the smallest H.264 partition, filled without a Python loop
    field = np.zeros((h4, w4, 2), np.float32)
    covered = np.zeros((h4, w4), bool)
    mvs = mvs[mvs["source"] < 0]
    if not len(mvs):
        return field, covered
    nx, ny = mvs["w"].astype(int) // 4, mvs["h"].astype(int) // 4
    x0 = (mvs["dst_x"].astype(int) - mvs["w"] // 2) // 4
    y0 = (mvs["dst_y"].astype(int) - mvs["h"] // 2) // 4
    counts = nx * ny
    blk = np.repeat(np.arange(len(mvs)), counts)
    i = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    cx, cy = x0[blk] + i % nx[blk], y0[blk] + i // nx[blk]
    ok = (cx >= 0) & (cx < w4) & (cy >= 0) & (cy < h4)
    scale = mvs["motion_scale"].astype(np.float32)
    field[cy[ok], cx[ok], 0] = (mvs["motion_x"] / scale)[blk[ok]]
    field[cy[ok], cx[ok], 1] = (mvs["motion_y"] / scale)[blk[ok]]
    covered[cy[ok], cx[ok]] = True
    return field, covered


def pad_to(a, h, w):
    return np.pad(a, ((0, h - a.shape[0]), (0, w - a.shape[1])), mode="edge")


def sequence_maps(seq_dir, img_dir, cfg):
    mb = cfg["mb"]
    container = av.open(str(seq_dir / "video.mp4"))
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}
    w, h = stream.codec_context.width, stream.codec_context.height
    H, W = -(-h // mb) * mb, -(-w // mb) * mb
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    sources = sorted(img_dir.glob("*.jpg"))

    sad, mv, intra, types = [], [], [], []
    prev = None
    for i, frame in enumerate(container.decode(stream)):
        # reference = what the encoder reconstructed, current = what the sensor produced
        recon = pad_to(frame.to_ndarray(format="gray"), H, W)
        src = cv2.imread(str(sources[i]), cv2.IMREAD_GRAYSCALE)
        src = pad_to(src[:h, :w], H, W).astype(np.int16)
        side = frame.side_data.get("MOTION_VECTORS")
        is_p = prev is not None and side is not None
        types.append("P" if is_p else "I")
        if is_p:
            field4, covered4 = cell_field(side.to_ndarray(), H // 4, W // 4)
            flow = cv2.resize(field4, (W, H), interpolation=cv2.INTER_NEAREST)
            pred = cv2.remap(prev, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
            err = np.abs(src - pred).reshape(H // mb, mb, W // mb, mb).mean(axis=(1, 3))
            sad.append(np.clip(np.rint(err * cfg["sad_scale"]), 0, 255).astype(np.uint8))
            k = mb // 4
            mv.append(np.clip(np.rint(field4.reshape(H // mb, k, W // mb, k, 2).mean(axis=(1, 3))),
                              -127, 127).astype(np.int8))
            intra.append(~covered4.reshape(H // mb, k, W // mb, k).any(axis=(1, 3)))
        else:
            sad.append(np.zeros((H // mb, W // mb), np.uint8))
            mv.append(np.zeros((H // mb, W // mb, 2), np.int8))
            intra.append(np.zeros((H // mb, W // mb), bool))
        prev = recon
    container.close()
    return {"sad": np.stack(sad), "mv": np.stack(mv), "intra": np.stack(intra),
            "types": np.array(types), "size": np.array([w, h])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "encoder.yaml")
    mot = load_config(ROOT / "configs" / "mot.yaml")
    root = ROOT / mot["output_dir"] / args.dataset / args.split
    out = ROOT / cfg["output_dir"] / args.dataset / args.split
    out.mkdir(parents=True, exist_ok=True)

    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        info = dict(l.split("=", 1) for l in (seq_dir / "seqinfo.ini").read_text().splitlines() if "=" in l)
        t0 = time.time()
        maps = sequence_maps(seq_dir, ROOT / info["imDir"], cfg)
        np.savez_compressed(out / f"{seq_dir.name}.npz", **maps)
        n = len(maps["types"])
        print(f"{args.dataset}/{args.split}/{seq_dir.name}: {n} frames, grid {maps['sad'].shape[1:]}, "
              f"{n / (time.time() - t0):.1f} fps")


if __name__ == "__main__":
    main()
