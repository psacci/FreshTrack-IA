"""Sanity-check dataset/freshness before training.

Checks folder layout, class counts, unreadable images, class imbalance, and
exact duplicate files shared between splits. Exits with code 1 on errors.

Usage:  python training/check_freshness_dataset.py [--data dataset/freshness]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
CLASSES = ("fresh", "aging", "spoiled")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_PER_CLASS = {"train": 50, "val": 10, "test": 10}


def short(path: Path) -> str:
    """Path relative to the project when possible, otherwise as given."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "dataset" / "freshness")
    args = ap.parse_args()

    errors, warnings = [], []
    counts: dict[str, dict[str, int]] = {}
    seen_hash: dict[str, str] = {}
    unreadable = 0

    for split in SPLITS:
        counts[split] = {}
        for cls in CLASSES:
            folder = args.data / split / cls
            if not folder.is_dir():
                errors.append(f"Missing folder: {short(folder)}")
                counts[split][cls] = 0
                continue
            files = [p for p in folder.iterdir() if p.suffix.lower() in EXTS]
            counts[split][cls] = len(files)
            if len(files) < MIN_PER_CLASS[split]:
                errors.append(f"{split}/{cls} has only {len(files)} images (need at least {MIN_PER_CLASS[split]}).")
            for path in files:
                try:
                    with Image.open(path) as im:
                        im.verify()
                except Exception:  # noqa: BLE001
                    unreadable += 1
                    errors.append(f"Unreadable image: {short(path)}")
                    continue
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                other = seen_hash.get(digest)
                if other and other != split:
                    errors.append(f"Identical file in {other} and {split}: {path.name}")
                seen_hash.setdefault(digest, split)

    for split in SPLITS:
        values = [v for v in counts[split].values() if v]
        if values and max(values) / min(values) > 4:
            warnings.append(f"{split} is imbalanced ({max(values)} vs {min(values)}). Training uses class weights to compensate.")

    print(f"{'split':<7}" + "".join(f"{c:>9}" for c in CLASSES) + f"{'total':>9}")
    for split in SPLITS:
        row = [counts[split][c] for c in CLASSES]
        print(f"{split:<7}" + "".join(f"{n:>9}" for n in row) + f"{sum(row):>9}")

    report = {"counts": counts, "unreadable": unreadable, "errors": errors, "warnings": warnings}
    if args.data.exists():
        (args.data / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    if errors:
        print("\nDataset check FAILED. Fix the errors above (or run training/prepare_dataset.py).")
        return 1
    print("\nDataset check passed. Next: python training/train_freshness.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
