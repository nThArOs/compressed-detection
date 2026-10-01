import json
import os
import platform
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / "configs" / "compressed.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def hardware_info():
    return {"platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count()}


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def available_cpus():
    # docker --cpus sets a cgroup quota, os.cpu_count() still reports every core of the host
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, round(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def limit_threads():
    import torch

    n = available_cpus()
    torch.set_num_threads(n)
    return n


def tune_onnx(model, threads):
    """Ultralytics opens ONNX models with one thread per host core; reopen the session within the CPU quota."""
    backend = getattr(getattr(model, "predictor", None), "model", None)
    backend = getattr(backend, "backend", backend)
    session = vars(backend).get("session") if backend is not None else None
    if session is None:
        return
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    backend.session = ort.InferenceSession(session._model_path, options, providers=["CPUExecutionProvider"])


def model_size(model, path, imgsz):
    """Parameters in millions and GFLOPs; GFLOPs are only known for PyTorch weights."""
    if str(path).endswith(".onnx"):
        import onnx
        import numpy as np

        params = sum(int(np.prod(t.dims)) for t in onnx.load(str(path)).graph.initializer)
        return params / 1e6, None
    from ultralytics.utils.torch_utils import get_flops

    return sum(p.numel() for p in model.model.parameters()) / 1e6, get_flops(model.model, imgsz)
