"""Train YOLO on one dataset and modality from data/yolo/, copy the best weights to models/."""
import argparse
import shutil
import time

from ultralytics import YOLO

from common import ROOT, hardware_info, load_config, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("modalities", nargs="+", help="residual, rgb")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "train.yaml")
    data_root = ROOT / load_config(ROOT / "configs" / "residual.yaml")["output_dir"]

    for mod in args.modalities:
        name = f"{args.dataset}_{mod}"
        model = YOLO(str(ROOT / cfg["init"]))
        t0 = time.time()
        model.train(data=str(data_root / args.dataset / mod / "data.yaml"), epochs=cfg["epochs"],
                    imgsz=cfg["imgsz"], batch=cfg["batch"], patience=cfg["patience"],
                    workers=cfg["workers"], cache=cfg["cache"], device="cpu",
                    project=str(ROOT / cfg["project"]), name=name, exist_ok=True, verbose=False)
        best = ROOT / cfg["project"] / name / "weights" / "best.pt"
        shutil.copy(best, ROOT / "models" / f"{name}.pt")

        m = YOLO(str(best)).val(data=str(data_root / args.dataset / mod / "data.yaml"),
                                split="test", imgsz=cfg["imgsz"], device="cpu", verbose=False)
        save_json({"dataset": args.dataset, "modality": mod, "train": cfg,
                   "train_hours": round((time.time() - t0) / 3600, 2),
                   "test": {"precision": round(float(m.box.mp), 3), "recall": round(float(m.box.mr), 3),
                            "map50": round(float(m.box.map50), 3),
                            "map50_95": round(float(m.box.map), 3)},
                   "hardware": hardware_info()},
                  ROOT / "results" / f"detect_{name}.json")
        print(f"{name}: test mAP50 {m.box.map50:.3f}")


if __name__ == "__main__":
    main()
