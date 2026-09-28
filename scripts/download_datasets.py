"""Download and extract tracking datasets into data/<dataset>/."""
import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

import gdown

from common import ROOT, load_config


def extract(archive, dest):
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
    elif archive.suffix.lower() in (".rar", ".7z"):
        shutil.unpack_archive(archive, dest)
    else:
        raise ValueError(f"unknown archive format: {archive}")


def download(name, cfg, keep):
    root = ROOT / "data" / name
    archives = root / "archives"
    archives.mkdir(parents=True, exist_ok=True)

    for part in cfg.get("manual", []):
        if not (archives / part).exists():
            print(f"{name}: {part} missing, download it from {cfg['url']} into {archives}")

    for part, spec in cfg.get("files", {}).items():
        done = root / f".{part}.done"
        if done.exists():
            continue
        spec = spec if isinstance(spec, dict) else {"id": spec}
        file_id, local = spec["id"], archives / spec.get("archive", "-")
        if local.exists():
            extract(local, root / part)
            done.touch()
            continue
        try:
            path = gdown.download(id=file_id, output=str(archives) + "/", quiet=False, resume=True)
        except gdown.exceptions.FileURLRetrievalError:
            path = None
        if path is None:
            print(f"{name}/{part}: download failed (Drive quota?), retry later or get it from "
                  f"https://drive.google.com/uc?id={file_id} into {archives}")
            continue
        extract(Path(path), root / part)
        done.touch()
        if not keep:
            Path(path).unlink()

    for part in cfg.get("manual", []):
        done = root / f".{Path(part).stem}.done"
        if (archives / part).exists() and not done.exists():
            extract(archives / part, root)
            done.touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help="names from configs/datasets.yaml (default: all)")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "datasets.yaml")
    for name in args.datasets or list(cfg):
        download(name, cfg[name], args.keep_archives)
        print(f"{name}: data/{name}")


if __name__ == "__main__":
    main()
