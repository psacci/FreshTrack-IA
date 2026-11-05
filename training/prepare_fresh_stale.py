"""Merge Kaggle's fresh/stale produce images with FreshTrack's base dataset.

Download and extract the dataset with your own Kaggle account first:
  kaggle datasets download -d raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables \
    -p data/raw/fresh-stale --unzip

The source's "stale" label is mapped to FreshTrack's "aging" class, not
"spoiled". Near-duplicate image clusters stay within one split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from training.prepare_freshness44 import (  # noqa: E402
    IMAGE_SUFFIXES,
    OUTPUT_CLASSES,
    _copy_original_dataset,
    sha256_file,
)

SOURCE_NAME = "Fresh and Stale Images of Fruits and Vegetables (Kaggle)"
SOURCE_URL = "https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables"
SOURCE_LICENSE = "CC0: Public Domain (as listed by Kaggle)"
FOODS = {
    "apple": ("apple", "apples"),
    "banana": ("banana", "bananas"),
    "bitter gourd": ("bittergourd", "bitter_gourd", "bitter-gourd", "karela"),
    "capsicum": ("capsicum", "bell pepper", "bell_pepper"),
    "orange": ("orange", "oranges"),
    "tomato": ("tomato", "tomatoes"),
}
LABELS = {"fresh": "fresh", "stale": "aging"}
PHASH_THRESHOLD = 6
PHASH_BANDS = 8
_DCT_SIZE = 32
_positions = np.arange(_DCT_SIZE, dtype=np.float32)
_frequencies = np.arange(_DCT_SIZE, dtype=np.float32)[:, None]
_DCT_BASIS = np.cos(np.pi * (2 * _positions + 1) * _frequencies / (2 * _DCT_SIZE))
_DCT_BASIS[0] *= np.sqrt(1 / _DCT_SIZE)
_DCT_BASIS[1:] *= np.sqrt(2 / _DCT_SIZE)


def parse_image_path(path: Path, root: Path) -> tuple[str, str] | None:
    parts = [path.stem, *path.relative_to(root).parts[:-1]]
    normalized_parts = [
        re.sub(r"(?<=[a-z])(?=[A-Z])", " ", part).lower().replace("_", " ").replace("-", " ")
        for part in parts
    ]
    text = " ".join(normalized_parts)
    labels = {
        label for label in ("fresh", "stale")
        if re.search(rf"\b{label}\b", text)
    }
    if len(labels) != 1:
        return None

    foods = []
    for food, aliases in FOODS.items():
        if any(
            re.search(rf"\b{re.escape(alias.replace('_', ' ').replace('-', ' '))}\b", text)
            for alias in aliases
        ):
            foods.append(food)
    if len(foods) != 1:
        return None
    return foods[0], next(iter(labels))


def perceptual_hash(path: Path) -> int:
    with Image.open(path) as image:
        gray = np.asarray(
            image.convert("L").resize((_DCT_SIZE, _DCT_SIZE), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    low = (_DCT_BASIS @ gray @ _DCT_BASIS.T)[:8, :8].flatten()
    median = float(np.median(low[1:]))
    result = 0
    for value in low[1:]:
        result = (result << 1) | int(value > median)
    return result


def _band_keys(value: int) -> tuple[int, ...]:
    return tuple((value >> (band * 8)) & 0xFF for band in range(PHASH_BANDS))


def _find(parent: list[int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _union(parent: list[int], first: int, second: int) -> None:
    first_root, second_root = _find(parent, first), _find(parent, second)
    if first_root != second_root:
        parent[second_root] = first_root


def _copy_candidates(
    source: Path,
    base: Path,
    output: Path,
    known_hashes: set[str],
    manifest: list[dict],
) -> tuple[dict[str, int], int, int]:
    candidates = []
    seen_source_hashes: set[str] = set()
    duplicates = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parsed = parse_image_path(path, source)
        if parsed is None:
            continue
        digest = sha256_file(path)
        if digest in known_hashes or digest in seen_source_hashes:
            duplicates += 1
            continue
        seen_source_hashes.add(digest)
        try:
            with Image.open(path) as image:
                image.verify()
            image_hash = perceptual_hash(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read dataset image {path}: {exc}") from exc
        food, source_label = parsed
        candidates.append({
            "path": path,
            "digest": digest,
            "hash": image_hash,
            "food": food,
            "source_label": source_label,
            "label": LABELS[source_label],
        })

    if not candidates:
        raise ValueError(f"No supported Fresh/Stale image folders were found under {source}.")

    # Existing base images are fixed in their original splits. Exclude new
    # images that closely match them so a duplicate cannot cross the split.
    base_hashes: dict[int, list[int]] = {}
    for split in ("train", "val", "test"):
        for label in OUTPUT_CLASSES:
            for image_path in (base / split / label).glob("*"):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    value = perceptual_hash(image_path)
                    for band, key in enumerate(_band_keys(value)):
                        base_hashes.setdefault((band << 8) | key, []).append(value)

    retained = []
    index: dict[int, list[int]] = {}
    parent: list[int] = []
    hashes: list[int] = []
    for candidate in candidates:
        value = candidate["hash"]
        keys = _band_keys(value)
        matches_base = any(
            (value ^ other).bit_count() <= PHASH_THRESHOLD
            for band, key in enumerate(keys)
            for other in base_hashes.get((band << 8) | key, ())
        )
        if matches_base:
            duplicates += 1
            continue

        match_ids = {
            index_id
            for band, key in enumerate(keys)
            for index_id in index.get((band << 8) | key, ())
            if (value ^ hashes[index_id]).bit_count() <= PHASH_THRESHOLD
        }
        new_id = len(hashes)
        parent.append(new_id)
        hashes.append(value)
        for index_id in match_ids:
            _union(parent, new_id, index_id)
        for band, key in enumerate(keys):
            index.setdefault((band << 8) | key, []).append(new_id)
        retained.append(candidate)

    grouped: dict[int, list[dict]] = {}
    for index_id, candidate in enumerate(retained):
        grouped.setdefault(_find(parent, index_id), []).append(candidate)

    group_split = {}
    for root_id, members in grouped.items():
        identity = min(member["digest"] for member in members)
        bucket = int(hashlib.sha256(identity.encode("ascii")).hexdigest()[:8], 16) / 0xFFFFFFFF
        group_split[root_id] = "train" if bucket < 0.70 else "val" if bucket < 0.85 else "test"

    counts = {split: 0 for split in ("train", "val", "test")}
    for index_id, candidate in enumerate(retained):
        split = group_split[_find(parent, index_id)]
        digest = candidate["digest"]
        relative = Path(split) / candidate["label"] / f"{candidate['food']}_{digest[:16]}{candidate['path'].suffix.lower()}"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate["path"], destination)
        manifest.append({
            "path": relative.as_posix(),
            "split": split,
            "label": candidate["label"],
            "fruit": candidate["food"],
            "source": SOURCE_NAME,
            "source_file": candidate["path"].relative_to(source).as_posix(),
        })
        known_hashes.add(digest)
        counts[split] += 1

    found = {(item["food"], item["source_label"]) for item in candidates}
    expected = {(food, label) for food in FOODS for label in LABELS}
    missing = expected - found
    if missing:
        names = ", ".join(f"{food}/{label}" for food, label in sorted(missing))
        raise ValueError(f"Dataset is missing expected food/label folders: {names}")
    return counts, duplicates, len(grouped)


def prepare_fresh_stale(source: Path, base: Path, output: Path) -> dict[str, int]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --out path.")
    if not source.is_dir():
        raise FileNotFoundError(f"Fresh/Stale dataset folder not found: {source}")
    for split in ("train", "val", "test"):
        if not (base / split).is_dir():
            raise FileNotFoundError(f"Base dataset split not found: {base / split}")

    output.mkdir(parents=True)
    manifest: list[dict] = []
    try:
        known_hashes = _copy_original_dataset(base, output, manifest)
        counts, duplicates, groups = _copy_candidates(source, base, output, known_hashes, manifest)
        if not all(counts.values()):
            raise ValueError(
                f"Fresh/Stale data did not populate every split ({counts}). "
                "The near-duplicate grouping may have left too few independent images."
            )

        fields = ["path", "split", "label", "fruit", "source", "source_file"]
        with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(manifest)
        (output / "ATTRIBUTION.txt").write_text(
            f"{SOURCE_NAME}\n"
            f"Dataset: {SOURCE_URL}\n"
            f"License listed by Kaggle: {SOURCE_LICENSE}.\n"
            "The Kaggle page says some class images were obtained from the separate "
            "Fruits fresh and rotten dataset and some were extracted from videos. "
            "Check the upstream sources and terms before redistribution. The CC0 listing "
            "does not automatically resolve rights in third-party content.\n"
            "The source's 'stale' label is mapped to FreshTrack's 'aging' class; it is "
            "not equivalent to rotten/spoiled and is not a food-safety judgment.\n"
            f"Perceptual duplicate grouping threshold: {PHASH_THRESHOLD} bits of a 63-bit pHash. "
            "This reduces, but cannot guarantee elimination of, leakage from related video frames.\n\n"
            "Original FreshTrack fruit-freshness dataset: "
            "https://github.com/zijianchen98/Fruit-freshness-detection-dataset\n"
            "License: Apache-2.0. Its 'normal' class remains mapped to 'aging'.\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    print(f"Near-duplicate image groups retained: {groups}; exact/near duplicates skipped: {duplicates}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "raw" / "fresh-stale")
    parser.add_argument("--base", type=Path, default=settings.DATASET_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / "dataset" / "freshness_stale_expanded")
    args = parser.parse_args()
    try:
        counts = prepare_fresh_stale(args.source, args.base, args.out)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"Dataset preparation failed: {exc}")
        return 1
    print(f"Fresh/Stale dataset prepared at {args.out}")
    print(f"Additional images by split: {counts}")
    print(f"Check it with: python training/check_freshness_dataset.py --data {args.out}")
    print(f"Train with: python training/train_freshness.py --data {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
