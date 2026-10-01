"""Train YOLO on one dataset and modality from data/yolo/, copy the best weights to models/.

Inside the platform, MLOPS_DATASETS lists the mounted datasets: the YOLO ones are merged, with the union
of their classes in mount order, so a run trains on exactly the datasets the operator checked.
"""
import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import yaml
from ultralytics import YOLO, settings

from common import ROOT, hardware_info, load_config, save_json


def mounted_yolo(mod):
    """Folders of the mounted datasets that have a YOLO data.yaml for this modality."""
    pairs = [x.split("=", 1) for x in os.environ.get("MLOPS_DATASETS", "").split(",") if "=" in x]
    return [(name, ROOT / mount / mod) for name, mount in pairs if (ROOT / mount / mod / "data.yaml").is_file()]


def merge(sources, out):
    """Build one YOLO dataset from several: links to the images, labels remapped to the union of classes."""
    names = []
    for _, root in sources:
        names += [n for n in yaml.safe_load((root / "data.yaml").read_text())["names"].values() if n not in names]
    shutil.rmtree(out, ignore_errors=True)
    splits = set()
    for dataset, root in sources:
        own = yaml.safe_load((root / "data.yaml").read_text())["names"]
        remap = {str(i): str(names.index(n)) for i, n in own.items()}
        for split in ("train", "val", "test"):
            images = root / "images" / split
            if not images.is_dir():
                continue
            splits.add(split)
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            for img in images.iterdir():
                stem = f"{dataset}_{img.stem}"
                (out / "images" / split / f"{stem}{img.suffix}").symlink_to(img)
                label = root / "labels" / split / f"{img.stem}.txt"
                lines = label.read_text().splitlines() if label.is_file() else []
                (out / "labels" / split / f"{stem}.txt").write_text(
                    "".join(f"{remap[cls]} {box}\n" for cls, box in (l.split(" ", 1) for l in lines if l.strip())))
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out), **{s: f"images/{s}" for s in sorted(splits)}, "names": dict(enumerate(names))},
        sort_keys=False))
    print(f"merged {', '.join(d for d, _ in sources)}: classes {names}", flush=True)
    return out / "data.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("modalities", nargs="+", help="residual, rgb")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--tag", help="suffix for the run, model and result names")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    # inside the platform, Ultralytics logs every epoch to the run the platform created
    settings.update({"mlflow": bool(os.environ.get("MLFLOW_RUN_ID"))})
    data_root = ROOT / load_config(ROOT / "configs" / "residual.yaml")["output_dir"]

    for mod in args.modalities:
        name = f"{args.dataset}_{mod}" + (f"_{args.tag}" if args.tag else "")
        sources = mounted_yolo(mod)
        if len(sources) > 1:
            data = merge(sources, Path(tempfile.gettempdir()) / "merged" / name)
        elif sources:
            data = sources[0][1] / "data.yaml"
        else:
            data = data_root / args.dataset / mod / "data.yaml"
        model = YOLO(str(ROOT / cfg["init"]))
        t0 = time.time()
        model.train(data=str(data), epochs=cfg["epochs"],
                    imgsz=cfg["imgsz"], batch=cfg["batch"], patience=cfg["patience"],
                    workers=cfg["workers"], cache=cfg["cache"], fraction=cfg.get("fraction", 1.0), device="cpu",
                    project=str(ROOT / cfg["project"]), name=name, exist_ok=True, verbose=False)
        best = ROOT / cfg["project"] / name / "weights" / "best.pt"
        shutil.copy(best, ROOT / "models" / f"{name}.pt")

        m = YOLO(str(best)).val(data=str(data),
                                split="test", imgsz=cfg["imgsz"], device="cpu", verbose=False)
        save_json({"dataset": args.dataset, "datasets": [d for d, _ in sources] or [args.dataset], "modality": mod, "train": cfg,
                   "train_hours": round((time.time() - t0) / 3600, 2),
                   "test": {"precision": round(float(m.box.mp), 3), "recall": round(float(m.box.mr), 3),
                            "map50": round(float(m.box.map50), 3),
                            "map50_95": round(float(m.box.map), 3)},
                   "hardware": hardware_info()},
                  ROOT / "results" / f"detect_{name}.json")
        print(f"{name}: test mAP50 {m.box.map50:.3f}")


if __name__ == "__main__":
    main()
