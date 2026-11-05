"""Build dataset/freshness/ from a public fruit-freshness dataset.

Source: zijianchen98/Fruit-freshness-detection-dataset (Apache-2.0)
        3,202 photos, 7,087 labelled fruits (apple, banana, orange) in Pascal VOC format,
        each tagged fresh / normal / rotten.

What this script does
  1. Clones the dataset with git (no Kaggle account needed).
  2. Crops every labelled fruit out of its photo.
  3. Maps labels:  fresh -> fresh,  normal -> aging,  rotten -> spoiled.
  4. Finds near-duplicate crops (same fruit photographed or augmented several times,
     including mirrored copies) and groups them with every other crop from the same photo.
  5. Splits train/val/test BY GROUP, so a fruit and its look-alikes never sit in two splits.
     Without this, test accuracy would be inflated by memorised copies. Class shares
     stay about equal across splits.
  6. Writes dataset/freshness/manifest.csv.

Usage:  python training/prepare_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/zijianchen98/Fruit-freshness-detection-dataset.git"
LABEL_MAP = {"fresh": "fresh", "normal": "aging", "rotten": "spoiled"}
CLASSES = ("fresh", "aging", "spoiled")
N_FOLDS = 20


def get_source(source: Path) -> Path:
    if (source / "Annotations").exists():
        return source
    source.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning dataset into {source} (about 350 MB, one time) ...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(source)], check=True)
    return source


def read_objects(source: Path) -> list[dict]:
    objects = []
    for xml_path in sorted((source / "Annotations").glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        image_id = xml_path.stem
        size = root.find("size")
        xml_size = (int(size.find("width").text), int(size.find("height").text)) if size is not None else None
        for idx, obj in enumerate(root.findall("object")):
            name = obj.find("name").text.strip().lower()
            condition, _, fruit = name.partition("_")
            if condition not in LABEL_MAP or not fruit:
                continue
            box = obj.find("bndbox")
            coords = tuple(int(float(box.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax"))
            objects.append(dict(image_id=image_id, idx=idx, fruit=fruit, label=LABEL_MAP[condition],
                                box=coords, xml_size=xml_size))
    return objects


def dhash(img: Image.Image) -> np.ndarray:
    """256-bit difference hash packed into 32 bytes. Similar-looking images get similar hashes."""
    small = np.asarray(img.convert("L").resize((17, 16), Image.LANCZOS), dtype=np.int16)
    return np.packbits((small[:, 1:] > small[:, :-1]).flatten())


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(query: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Number of differing bits between one hash and every hash in `pool`."""
    return _POPCOUNT[np.bitwise_xor(pool, query)].sum(axis=1)


def group_ids(objects: list[dict], hashes: np.ndarray, flipped: np.ndarray, threshold: int) -> np.ndarray:
    """Union-find: crops share a group if they come from the same photo or look
    near-identical (also when mirrored). `threshold` is in bits out of 256."""
    n = len(objects)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    first_of_photo: dict[str, int] = {}
    for i, o in enumerate(objects):
        union(i, first_of_photo.setdefault(o["image_id"], i))
    for i in range(n - 1):
        close = np.minimum(hamming(hashes[i], hashes[i + 1:]), hamming(flipped[i], hashes[i + 1:]))
        for j in np.nonzero(close <= threshold)[0]:
            union(i, i + 1 + int(j))
    return np.array([find(i) for i in range(n)])


def split_by_group(objects: list[dict], groups: np.ndarray, val_frac: float, test_frac: float, seed: int) -> None:
    """Assign objects[i]['split'] so no group is shared and class shares stay balanced."""
    y = np.array([f"{o['label']}_{o['fruit']}" for o in objects])
    k_test, k_val = round(test_frac * N_FOLDS), round(val_frac * N_FOLDS)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (_, held_out) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
        name = "test" if fold < k_test else "val" if fold < k_test + k_val else "train"
        for i in held_out:
            objects[i]["split"] = name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=ROOT / "data" / "raw" / "fruit-freshness")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset" / "freshness")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-side", type=int, default=32, help="skip crops smaller than this (pixels)")
    ap.add_argument("--pad", type=float, default=0.04, help="extra border around each box")
    ap.add_argument("--dup-threshold", type=int, default=20, help="max differing bits (of 256) counted as near-duplicate")
    ap.add_argument("--force", action="store_true", help="overwrite an existing train/val/test")
    args = ap.parse_args()

    existing = [args.out / s for s in ("train", "val", "test", "_staging") if (args.out / s).exists()]
    if existing and not args.force:
        print("dataset/freshness already has data. Re-run with --force to rebuild it.")
        return 1
    for path in existing:
        shutil.rmtree(path)

    source = get_source(args.source)
    objects = read_objects(source)
    print(f"Read {len(objects)} labelled fruits from {len({o['image_id'] for o in objects})} photos.")

    # Pass 1: crop every fruit into a staging folder and hash it.
    staging = args.out / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    kept, hashes, flipped = [], [], []
    skipped_small = skipped_size = 0
    by_image: dict[str, list[dict]] = {}
    for o in objects:
        by_image.setdefault(o["image_id"], []).append(o)
    for image_id, objs in by_image.items():
        path = source / "JPEGImages" / f"{image_id}.jpg"
        if not path.exists():
            continue
        with Image.open(path) as im:
            im = im.convert("RGB")
            for o in objs:
                if o["xml_size"] and o["xml_size"] != im.size:
                    skipped_size += 1
                    continue
                x1, y1, x2, y2 = o["box"]
                px, py = int((x2 - x1) * args.pad), int((y2 - y1) * args.pad)
                box = (max(0, x1 - px), max(0, y1 - py), min(im.width, x2 + px), min(im.height, y2 + py))
                if min(box[2] - box[0], box[3] - box[1]) < args.min_side:
                    skipped_small += 1
                    continue
                crop = im.crop(box)
                o["name"] = f"{o['fruit']}_{image_id}_{o['idx']}.jpg"
                crop.save(staging / o["name"], quality=95)
                hashes.append(dhash(crop))
                flipped.append(dhash(crop.transpose(Image.FLIP_LEFT_RIGHT)))
                kept.append(o)

    print("Grouping near-duplicate crops ...")
    groups = group_ids(kept, np.array(hashes, dtype=np.uint8), np.array(flipped, dtype=np.uint8), args.dup_threshold)
    print(f"{len(kept)} crops form {len(set(groups.tolist()))} groups (largest: {np.bincount(np.unique(groups, return_inverse=True)[1]).max()}).")
    split_by_group(kept, groups, args.val, args.test, args.seed)

    # Pass 2: move crops into their split/class folders and write the manifest.
    manifest = []
    for o in kept:
        rel = Path(o["split"]) / o["label"] / o["name"]
        (args.out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging / o["name"]), args.out / rel)
        manifest.append(dict(path=rel.as_posix(), split=o["split"], label=o["label"],
                             fruit=o["fruit"], image_id=o["image_id"]))
    shutil.rmtree(staging)

    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "split", "label", "fruit", "image_id"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nSkipped: {skipped_small} too small, {skipped_size} with mismatched image size.\n")
    print(f"{'split':<7}" + "".join(f"{c:>9}" for c in CLASSES) + f"{'total':>9}")
    for split in ("train", "val", "test"):
        counts = [sum(1 for m in manifest if m["split"] == split and m["label"] == c) for c in CLASSES]
        print(f"{split:<7}" + "".join(f"{n:>9}" for n in counts) + f"{sum(counts):>9}")
    print(f"\nWrote {len(manifest)} crops and manifest.csv to {args.out}")
    print("Next: python training/check_freshness_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
