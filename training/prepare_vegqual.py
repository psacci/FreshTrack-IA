"""Verify and unpack the CC BY 4.0 VegQual YOLO dataset.

Usage: python training/prepare_vegqual.py
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import stat
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_URL = "https://ndownloader.figshare.com/files/62799448"
ARCHIVE_MD5 = "de19cbfe7d9792c5db37b6c3af0458e1"
SOURCE_ROOT = "VegQual/VegMining"
EXPECTED_CLASSES = (
    "defected_Potato",
    "defected_bitter_gourd",
    "defected_brinjal",
    "defected_capsicum",
    "defected_onion",
    "defected_pointed_gourd",
    "defected_tomato",
    "fresh_bitter_gourd",
    "fresh_brinjal",
    "fresh_capsicum",
    "fresh_onion",
    "fresh_pointed_gourd",
    "fresh_potato",
    "fresh_tomato",
)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading VegQual to {path} (about 192 MB)...")
    urllib.request.urlretrieve(ARCHIVE_URL, path)


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        name = member.filename.replace("\\", "/")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"Symlinks are not allowed in dataset archives: {member.filename}")
        target = (root / Path(*relative.parts)).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Dataset archive member escapes its destination: {member.filename}")
    archive.extractall(destination)


def read_classes(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^names:\s*(\[.*\])\s*$", text)
    if not match:
        raise ValueError(f"Could not read the class list from {path}")
    classes = ast.literal_eval(match.group(1))
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError(f"Invalid class list in {path}")
    result = tuple(classes)
    if result != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected VegQual classes in {path}: {result}")
    return result


def validate_dataset(root: Path, classes: tuple[str, ...]) -> dict[str, int]:
    counts = {}
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        images = root / folder / "images"
        labels = root / folder / "labels"
        if not images.is_dir() or not labels.is_dir():
            raise ValueError(f"Missing {folder}/images or {folder}/labels in {root}")
        image_paths = list(images.glob("*.jpg"))
        if not image_paths:
            raise ValueError(f"No JPG images found in {images}")
        for label_path in labels.glob("*.txt"):
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                values = line.split()
                if len(values) != 5:
                    raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
                try:
                    class_id = int(values[0])
                    coords = [float(value) for value in values[1:]]
                except ValueError as exc:
                    raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}") from exc
                if class_id < 0 or class_id >= len(classes) or any(not 0 <= value <= 1 for value in coords):
                    raise ValueError(f"Out-of-range YOLO label at {label_path}:{line_number}")
        counts[split] = len(image_paths)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "raw" / "VegQual_latest.zip")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset" / "detection" / "vegqual")
    args = parser.parse_args()

    if args.out.exists():
        print(f"Output already exists: {args.out}. Move it aside before preparing another copy.")
        return 1
    if not args.archive.exists():
        download_archive(args.archive)
    actual_md5 = md5_file(args.archive)
    if actual_md5 != ARCHIVE_MD5:
        print(f"Archive checksum mismatch: expected {ARCHIVE_MD5}, got {actual_md5}.")
        return 1

    args.out.mkdir(parents=True)
    try:
        with zipfile.ZipFile(args.archive) as archive:
            safe_extract(archive, args.out)

        source_root = args.out / SOURCE_ROOT
        original_yaml = source_root / "data.yaml"
        classes = read_classes(original_yaml)
        counts = validate_dataset(source_root, classes)
        yaml = (
            f"path: {json.dumps(str(source_root.resolve()))}\n"
            "train: train/images\n"
            "val: valid/images\n"
            "test: test/images\n"
            f"names: {list(classes)!r}\n"
        )
        (args.out / "data.yaml").write_text(yaml, encoding="utf-8")
        (args.out / "ATTRIBUTION.txt").write_text(
            "VegQual dataset, version 4, by Tarek Rahman et al.\n"
            "Dataset article: https://figshare.com/articles/dataset/"
            "A_Multiclass_Dataset_for_Real-Time_Fresh_and_Defective_Vegetables/30596084\n"
            "Dataset license: Creative Commons Attribution 4.0 International "
            "(https://creativecommons.org/licenses/by/4.0/).\n"
            "The dataset labels seven vegetables as fresh or defected; these labels "
            "describe visible dataset categories and are not food-safety judgments.\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(args.out, ignore_errors=True)
        raise

    print(f"VegQual prepared at {args.out}")
    print(f"Images by split: {counts}")
    print(f"Train with: python training/train_yolo.py --data {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
