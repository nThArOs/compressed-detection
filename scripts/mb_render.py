"""Preview video of level 1: frame with candidates (yellow) and ground truth (green), SAD map next to it."""
import argparse

import cv2
import numpy as np

from common import ROOT, load_config
from compressed_video import VideoWriter


def boxes_by_frame(path, cols):
    if not path.exists() or path.stat().st_size == 0:
        return {}
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    return {int(f): a[a[:, 0] == f][:, 1:1 + cols] for f in np.unique(a[:, 0])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("sequences", nargs="+")
    parser.add_argument("--split", default="test")
    parser.add_argument("--width", type=int, default=480)
    args = parser.parse_args()

    ecfg = load_config(ROOT / "configs" / "encoder.yaml")
    mot = load_config(ROOT / "configs" / "mot.yaml")
    out_dir = ROOT / "results" / "videos" / "previews" / "level1_sad" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq in args.sequences:
        seq_dir = ROOT / mot["output_dir"] / args.dataset / args.split / seq
        info = dict(l.split("=", 1) for l in (seq_dir / "seqinfo.ini").read_text().splitlines() if "=" in l)
        frames = sorted((ROOT / info["imDir"]).glob("*.jpg"))
        m = dict(np.load(ROOT / ecfg["output_dir"] / args.dataset / args.split / f"{seq}.npz"))
        gt = np.loadtxt(seq_dir / "gt.txt", delimiter=",", ndmin=2)
        gt = {int(f): gt[(gt[:, 0] == f) & (gt[:, 6] == 1)][:, 2:6] for f in np.unique(gt[:, 0])}
        cand = boxes_by_frame(ROOT / "results" / "candidates" / args.dataset / args.split / f"{seq}.txt", 4)

        w, h = m["size"]
        s = args.width / w
        size = (args.width, round(h * s) // 2 * 2)
        writer = VideoWriter(out_dir / f"{seq}.mp4", int(info["frameRate"]), 2 * size[0], size[1], crf=30)
        vmax = max(np.percentile(m["sad"][m["types"] == "P"], 99.5), 1)
        for i, path in enumerate(frames[:len(m["types"])]):
            f = i + 1
            img = cv2.resize(cv2.imread(str(path)), size, interpolation=cv2.INTER_AREA)
            heat = np.clip(m["sad"][i].astype(np.float32) / vmax * 255, 0, 255).astype(np.uint8)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
            heat[m["intra"][i]] = (255, 255, 255)
            gh, gw = m["sad"].shape[1:]
            heat = cv2.resize(heat, (round(gw * 16 * s), round(gh * 16 * s)),
                              interpolation=cv2.INTER_NEAREST)[:size[1], :size[0]]
            heat = cv2.copyMakeBorder(heat, 0, size[1] - heat.shape[0], 0, size[0] - heat.shape[1],
                                      cv2.BORDER_CONSTANT)
            for panel in (img, heat):
                for x, y, bw, bh in cand.get(f, []):
                    cv2.rectangle(panel, (int(x * s), int(y * s)), (int((x + bw) * s), int((y + bh) * s)),
                                  (0, 220, 255), 1)
                for x, y, bw, bh in gt.get(f, []):
                    cv2.rectangle(panel, (int(x * s), int(y * s)), (int((x + bw) * s), int((y + bh) * s)),
                                  (0, 255, 0), 1)
            cv2.putText(img, f"{seq} #{f} {m['types'][i]}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(heat, "encoder SAD per macroblock (white = intra)", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(np.hstack([img, heat]))
        writer.close()
        print(f"{seq}: {(out_dir / f'{seq}.mp4').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
