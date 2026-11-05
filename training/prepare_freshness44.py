"""Merge a locally downloaded Freshness44 export with FreshTrack's 3-level fruit set.

Download and extract Freshness44 from Kaggle first. The Kaggle archive is about
7.4 GB and requires your own Kaggle account. This script never handles Kaggle
credentials.

Usage:
  python training/prepare_dataset.py
  kaggle datasets download -d siavash93/freshness44 -p data/raw/freshness44 --unzip
  python training/prepare_freshness44.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OUTPUT_CLASSES = ("fresh", "aging", "spoiled")


def parse_quality_folder(name: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(.+?)[_\s-]+(fresh|rotten)", name.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    food = re.sub(r"[_\s]+", " ", match.group(1).strip().lower())
    label = "fresh" if match.group(2).lower() == "fresh" else "spoiled"
    return (food, label) if food else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_original_dataset(base: Path, output: Path, manifest: list[dict]) -> set[str]:
    manifest_path = base / "manifest.csv"
    if not manifest_path.is_file():
        raise ValueError(f"Missing {manifest_path}; run training/prepare_dataset.py first.")
    seen_hashes: set[str] = set()
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            relative = Path(row["path"])
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) < 3
                or relative.parts[0] not in ("train", "val", "test")
                or relative.parts[1] != row.get("label")
                or relative.parts[0] != row.get("split")
            ):
                raise ValueError(f"Unsafe path in base freshness manifest: {row['path']}")
            source = base / relative
            if not source.is_file() or row.get("label") not in OUTPUT_CLASSES:
                raise ValueError(f"Invalid entry in base freshness manifest: {row}")
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            seen_hashes.add(sha256_file(source))
            manifest.append({
                "path": relative.as_posix(),
                "split": row["split"],
                "label": row["label"],
                "fruit": row.get("fruit", "").strip().lower().replace("_", " "),
                "source": "Fruit-freshness-detection-dataset",
                "source_file": row.get("image_id", ""),
            })
    return seen_hashes


def prepare_freshness44(source: Path, base: Path, output: Path) -> dict[str, int]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --out path.")
    if not source.is_dir():
        raise FileNotFoundError(f"Freshness44 folder not found: {source}")

    class_dirs = []
    for folder in source.rglob("*"):
        if folder.is_dir() and parse_quality_folder(folder.name):
            class_dirs.append(folder)
    if not class_dirs:
        raise ValueError(
            f"No class folders ending in _Fresh or _Rotten were found under {source}."
        )

    output.mkdir(parents=True)
    manifest: list[dict] = []
    try:
        known_hashes = _copy_original_dataset(base, output, manifest)
        grouped: dict[tuple[str, str], list[tuple[str, Path]]] = defaultdict(list)
        for folder in sorted(class_dirs):
            parsed = parse_quality_folder(folder.name)
            if parsed is None:
                continue
            food, label = parsed
            for image in sorted(folder.rglob("*")):
                if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                    digest = sha256_file(image)
                    if digest in known_hashes:
                        continue
                    known_hashes.add(digest)
                    grouped[(food, label)].append((digest, image))

        counts = {split: 0 for split in ("train", "val", "test")}
        foods = set()
        for (food, label), entries in sorted(grouped.items()):
            if not entries:
                continue
            entries.sort(key=lambda pair: pair[0])
            train_end = round(len(entries) * 0.70)
            val_end = train_end + round(len(entries) * 0.15)
            for index, (digest, source_file) in enumerate(entries):
                split = "train" if index < train_end else "val" if index < val_end else "test"
                filename = f"{re.sub(r'[^a-z0-9]+', '_', food)}_{digest[:16]}{source_file.suffix.lower()}"
                relative = Path(split) / label / filename
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target)
                manifest.append({
                    "path": relative.as_posix(),
                    "split": split,
                    "label": label,
                    "fruit": food,
                    "source": "Freshness44",
                    "source_file": source_file.relative_to(source).as_posix(),
                })
                counts[split] += 1
                foods.add(food)

        if not foods:
            raise ValueError("No Freshness44 images were found in supported class folders.")
        for split in ("train", "val", "test"):
            for label in OUTPUT_CLASSES:
                if not any(row["split"] == split and row["label"] == label for row in manifest):
                    raise ValueError(f"The combined dataset has no {label} examples in its {split} split.")

        fields = ["path", "split", "label", "fruit", "source", "source_file"]
        with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(manifest)
        (output / "ATTRIBUTION.txt").write_text(
            "Freshness44 (22 fruit/vegetable types; fresh and rotten labels)\n"
            "Kaggle dataset: https://www.kaggle.com/datasets/siavash93/freshness44\n"
            "License: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).\n"
            "Cite the Freshness44 paper linked on the Kaggle dataset page.\n\n"
            "Original FreshTrack fruit-freshness dataset:\n"
            "https://github.com/zijianchen98/Fruit-freshness-detection-dataset\n"
            "License: Apache-2.0. Its 'normal' class remains mapped to 'aging'.\n\n"
            "Freshness44 provides binary fresh/rotten labels, mapped here to fresh/spoiled. "
            "These are visual dataset labels, not food-safety judgments. The merged split "
            "is deterministic and image-level; test on separate household photos before relying on it.\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "raw" / "freshness44",
        help="extracted Freshness44 folder containing *_Fresh and *_Rotten class directories",
    )
    parser.add_argument("--base", type=Path, default=settings.DATASET_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / "dataset" / "freshness_expanded")
    args = parser.parse_args()

    try:
        counts = prepare_freshness44(args.source, args.base, args.out)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"Dataset preparation failed: {exc}")
        return 1
    print(f"Expanded freshness dataset prepared at {args.out}")
    print(f"Freshness44 images by split: {counts}")
    print("Train: python training/train_freshness.py --data dataset/freshness_expanded")
    print("Evaluate: python training/evaluate_freshness.py --data dataset/freshness_expanded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
