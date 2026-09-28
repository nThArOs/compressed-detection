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
