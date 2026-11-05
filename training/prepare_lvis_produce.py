"""Validate the LVIS produce download and write a corrected YOLO config.

The upstream YAML points validation at the test set. This creates a local
configuration that uses the actual train, validation, and held-out test splits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "data" / "raw" / "lvis-fruits-vegetables" / "LVIS_Fruits_And_Vegetables"
DEFAULT_CONFIG = ROOT / "data" / "raw" / "lvis-fruits-vegetables" / "freshtrack_data.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
EXPECTED_CLASSES = 63


def _read_classes(path: Path) -> list[str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read dataset YAML at {path}: {exc}") from exc
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, dict):
        try:
            classes = [names[i] if i in names else names[str(i)] for i in range(len(names))]
        except KeyError as exc:
            raise ValueError(f"Class IDs in {path} must be contiguous from zero.") from exc
    elif isinstance(names, list):
        classes = names
    else:
        raise ValueError(f"Dataset YAML at {path} must define names as a list or mapping.")
    if len(classes) != EXPECTED_CLASSES or not all(isinstance(name, str) and name.strip() for name in classes):
        raise ValueError(f"Expected {EXPECTED_CLASSES} non-empty class names in {path}; found {len(classes)}.")
    normalized = {name.strip().lower() for name in classes}
    if "banana" not in normalized or not any("cucumber" in name for name in normalized):
        raise ValueError(f"Dataset classes must include banana and cucumber; found them missing in {path}.")
    return [name.strip() for name in classes]


def _validate_split(source: Path, split: str, image_dir: str) -> int:
    images_root = source / "images" / split
    labels_root = source / "labels" / split
    actual_images = images_root / image_dir if image_dir else images_root
    images = sorted(path for path in actual_images.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images found for the {split} split under {actual_images}.")
    seen_stems: set[Path] = set()
    for image in images:
        relative = image.relative_to(images_root).with_suffix(".txt")
        label = labels_root / relative
        if not label.is_file():
            raise ValueError(f"Missing annotation for {image}: expected {label}.")
        seen_stems.add(relative)
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Invalid YOLO annotation at {label}:{line_number}.")
            try:
                class_id = int(values[0])
                coordinates = [float(value) for value in values[1:]]
            except ValueError as exc:
                raise ValueError(f"Invalid YOLO annotation at {label}:{line_number}.") from exc
            if class_id < 0 or class_id >= EXPECTED_CLASSES or any(not 0 <= value <= 1 for value in coordinates):
                raise ValueError(f"Out-of-range YOLO annotation at {label}:{line_number}.")
    extra_labels = {
        path.relative_to(labels_root)
        for path in labels_root.rglob("*.txt")
    } - seen_stems
    if extra_labels:
        raise ValueError(f"Found annotations without matching images in {split}: {next(iter(extra_labels))}.")
    return len(images)


def prepare_dataset(source: Path, out_config: Path) -> dict[str, int]:
    source = source.resolve()
    yaml_path = source / "data.yaml"
    if not source.is_dir() or not yaml_path.is_file():
        raise FileNotFoundError(f"Expected an extracted dataset with data.yaml at {source}.")
    classes = _read_classes(yaml_path)
    counts = {
        "train": _validate_split(source, "train", "train"),
        "val": _validate_split(source, "val", "val"),
        "test": _validate_split(source, "test", ""),
    }

    out_config.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "path": str(source),
        "train": "images/train/train",
        "val": "images/val/val",
        "test": "images/test",
        "names": classes,
    }
    out_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    attribution = (
        "LVIS Fruits and Vegetables dataset by Henning Heyen.\n"
        "Kaggle dataset: https://www.kaggle.com/datasets/henningheyen/lvis-fruits-and-vegetables-dataset\n"
        "License listed by Kaggle: MIT.\n"
        "The dataset is a subset of LVIS. Review the original LVIS image licenses and "
        "the dataset's third-party image terms before redistribution.\n"
        "This preparation fixes the upstream YAML's validation path: validation uses "
        "images/val/val and the 180-image test split is kept separate.\n"
    )
    (out_config.parent / "ATTRIBUTION.txt").write_text(attribution, encoding="utf-8")
    report = {"source": str(source), "classes": classes, "images_by_split": counts}
    (out_config.parent / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        counts = prepare_dataset(args.source, args.out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset preparation failed: {exc}")
        return 1
    print(f"Validated 63 classes; images by split: {counts}")
    print(f"Training config: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
