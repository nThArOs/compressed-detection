"""errors.json with every missed object and false alarm of an evaluation, plus thumbnails of a sample.

Each thumbnail shows the decoded frame and the model input side by side around the box: green for a
missed object, red for a false alarm. Ids are stable across versions so two evaluations can be compared:
a missed object by its annotation id, a false alarm by its position on a 16 px grid.
"""
import json
import random
import shutil

import cv2
import numpy as np

from common import ROOT, load_config

PER_KIND = 150
HEIGHT = 160


def _id(seq, e):
    if e["kind"] == "fn":
        return f"{seq}:{e['frame']}:fn:{e['object']}"
    x, y, w, h = e["box"]
    return f"{seq}:{e['frame']}:fp:{round((x + w / 2) / 16)}:{round((y + h / 2) / 16)}"


def _thumb(frame, model_input, box, color):
    x, y, w, h = box
    side = int(max(4 * max(w, h), 160))
    cx, cy = x + w / 2, y + h / 2
    H, W = frame.shape[:2]
    x0 = int(np.clip(cx - side / 2, 0, max(W - side, 0)))
    y0 = int(np.clip(cy - side / 2, 0, max(H - side, 0)))
    tiles = []
    for img in (frame, model_input):
        crop = img[y0:y0 + side, x0:x0 + side].copy()
        cv2.rectangle(crop, (int(x - x0), int(y - y0)), (int(x - x0 + w), int(y - y0 + h)), color, 2)
        tiles.append(cv2.resize(crop, (round(HEIGHT * crop.shape[1] / crop.shape[0]), HEIGHT)))
    return np.hstack([tiles[0], np.full((HEIGHT, 4, 3), 255, np.uint8), tiles[1]])


def write_gallery(errors, out, mode, seed=0):
    from serve import frames  # same decoding and residual as the service

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    gain = load_config(ROOT / "configs" / "residual.yaml")["residual_gain"]
    examples = []
    for seq, found in errors.items():
        for e in found:
            x, y, w, h = e["box"]
            examples.append({"id": _id(seq.name, e), "sequence": seq.name, "size": round(float(np.sqrt(w * h)), 1),
                             "image": None, **e, "_seq": seq})
    rng = random.Random(seed)
    for kind in ("fn", "fp"):
        pool = [e for e in examples if e["kind"] == kind]
        for e in rng.sample(pool, min(PER_KIND, len(pool))):
            e["image"] = f"{e['id'].replace(':', '_')}.jpg"
    wanted = {}
    for e in examples:
        if e["image"]:
            wanted.setdefault(e["_seq"], {}).setdefault(e["frame"], []).append(e)
    for seq, by_frame in wanted.items():
        for n, (frame, model_input) in enumerate(frames(str(seq / "video.mp4"), mode, gain), 1):
            for e in by_frame.get(n, []):
                color = (60, 180, 60) if e["kind"] == "fn" else (40, 40, 220)
                cv2.imwrite(str(out / e["image"]), _thumb(frame, model_input, e["box"], color),
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
            if n >= max(by_frame):
                break
    for e in examples:
        del e["_seq"]
    (out / "errors.json").write_text(json.dumps({"input": mode, "examples": examples}))
    print(f"errors: {sum(e['kind'] == 'fn' for e in examples)} missed, "
          f"{sum(e['kind'] == 'fp' for e in examples)} false alarms, "
          f"{sum(bool(e['image']) for e in examples)} thumbnails in {out}")
