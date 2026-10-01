"""Export a detector to ONNX, optionally quantized to INT8 with calibration frames from the YOLO dataset."""
import argparse
import random
import re
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from common import ROOT, load_config, save_json


class Calibration:
    def __init__(self, images, input_name, imgsz):
        self.images = iter(images)
        self.input_name = input_name
        self.imgsz = imgsz

    def get_next(self):
        path = next(self.images, None)
        if path is None:
            return None
        img = cv2.resize(cv2.imread(str(path)), (self.imgsz, self.imgsz))[:, :, ::-1]
        return {self.input_name: np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 255}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", choices=["rgb", "residual"], default="residual")
    parser.add_argument("--format", choices=["onnx", "onnx-int8"], default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--calibration", type=int, default=64, help="frames used to calibrate INT8")
    args = parser.parse_args()

    out = ROOT / "models" / f"export_dut_anti_uav_{args.input}.onnx"
    # the platform mounts the source model read-only and Ultralytics writes next to it
    work = Path(tempfile.mkdtemp())
    source = work / Path(args.model).name
    shutil.copy(args.model, source)
    exported = Path(YOLO(str(source)).export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True))
    if args.format == "onnx":
        shutil.move(exported, out)
    else:
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process

        prepared = exported.with_name(exported.stem + "_prep.onnx")
        quant_pre_process(str(exported), str(prepared))
        data_root = ROOT / load_config(ROOT / "configs" / "residual.yaml")["output_dir"] / "dut_anti_uav" / args.input
        images = sorted((data_root / "images" / "train").glob("*.jpg"))
        random.Random(0).shuffle(images)
        import onnx

        graph = onnx.load(str(prepared)).graph
        # the detection head turns into zeros when its outputs are quantized: keep it in float
        layers = [int(m.group(1)) for n in graph.node if (m := re.match(r"/model\.(\d+)/", n.name))]
        head = f"/model.{max(layers)}/"
        keep = [n.name for n in graph.node if n.name.startswith(head)]
        quantize_static(str(prepared), str(out), Calibration(images[:args.calibration], graph.input[0].name, args.imgsz),
                        quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax, per_channel=True, nodes_to_exclude=keep)
        original, quantized = onnx.load(str(exported)), onnx.load(str(out))
        quantized.metadata_props.extend(original.metadata_props)
        onnx.save(quantized, str(out))
    shutil.rmtree(work, ignore_errors=True)
    save_json({"format": args.format, "source": args.model, "model_mb": round(out.stat().st_size / 1e6, 2),
               "calibration_frames": args.calibration if args.format == "onnx-int8" else 0},
              ROOT / "results" / f"export_dut_anti_uav_{args.input}.json")
    print(f"{args.format}: {out.name}, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
