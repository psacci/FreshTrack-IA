"""Evaluate the trained freshness model on the held-out test split.

Reports accuracy, precision, recall, F1, a confusion matrix, a classification
report and per-fruit accuracy. Everything is written to
models/freshness_classification/ so the app's About page can show the real numbers.

Optional: --extra-dir <folder> evaluates on your OWN photos too. Use the same layout
(<folder>/fresh, <folder>/aging, <folder>/spoiled). This "real-world" result is the
most honest number for your paper.

Usage:  python training/evaluate_freshness.py
        python training/evaluate_freshness.py --extra-dir dataset/own_photos
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from modules.freshness_classifier import build_model  # noqa: E402


def predict_folder(model, folder: Path, classes: list[str], size: int, device: str, batch: int, workers: int):
    tf = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize(settings.IMAGENET_MEAN, settings.IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(folder, tf)
    if sorted(ds.classes) != sorted(classes):
        raise SystemExit(f"{folder} must contain exactly these class folders: {classes} (found {ds.classes})")
    remap = {ds.class_to_idx[c]: classes.index(c) for c in ds.classes}
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds += model(images.to(device)).argmax(1).cpu().tolist()
            targets += [remap[int(t)] for t in labels]
    paths = [Path(p) for p, _ in ds.samples]
    return np.array(targets), np.array(preds), paths


def summarize(targets, preds, classes):
    p, r, f, support = precision_recall_fscore_support(targets, preds, labels=range(len(classes)), zero_division=0)
    macro = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
    return {
        "n": int(len(targets)),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(targets, preds, labels=range(len(classes))).tolist(),
        "classes": list(classes),
    }


def save_confusion_png(matrix, classes, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array(matrix)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(m, cmap="Greens")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(m[i, j]), ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=settings.DATASET_DIR)
    ap.add_argument("--ckpt", type=Path, default=settings.FRESHNESS_CHECKPOINT)
    ap.add_argument("--extra-dir", type=Path, default=None, help="your own photos, same folder layout")
    ap.add_argument("--out-dir", type=Path, default=None, help="report directory (defaults to <checkpoint>_evaluation)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--note", default="", help="free text stored in metrics.json (e.g. how the model was trained)")
    args = ap.parse_args()

    if not args.ckpt.exists():
        print(f"No checkpoint at {args.ckpt}. Train first: python training/train_freshness.py")
        return 1
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = list(ckpt["classes"])
    size = int(ckpt.get("image_size", settings.FRESHNESS_IMAGE_SIZE))
    device = settings.get_device()
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    out_dir = args.out_dir or args.ckpt.parent / f"{args.ckpt.stem}_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets, preds, paths = predict_folder(model, args.data / "test", classes, size, device, args.batch, args.workers)
    result = summarize(targets, preds, classes)

    # Accuracy per fruit, using the manifest written by prepare_dataset.py.
    manifest = args.data / "manifest.csv"
    if manifest.exists():
        fruit_of = {Path(r["path"]).name: r["fruit"] for r in csv.DictReader(open(manifest, encoding="utf-8"))}
        by_fruit: dict[str, list[bool]] = {}
        for path, t, p in zip(paths, targets, preds):
            fruit = fruit_of.get(path.name)
            if fruit:
                by_fruit.setdefault(fruit, []).append(bool(t == p))
        result["per_fruit_accuracy"] = {f: {"n": len(v), "accuracy": float(np.mean(v))} for f, v in sorted(by_fruit.items())}

    wrong = [(str(p.relative_to(args.data)), classes[t], classes[q]) for p, t, q in zip(paths, targets, preds) if t != q]
    with open(out_dir / "misclassified.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "actual", "predicted"])
        w.writerows(wrong)

    metrics = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "device": device, "checkpoint": args.ckpt.name,
        "checkpoint_created": ckpt.get("created"),
        "trained_epoch": ckpt.get("epoch"), "pretrained_backbone": ckpt.get("pretrained"),
        "note": args.note, "test": result,
    }
    save_confusion_png(result["confusion_matrix"], classes, out_dir / "confusion_matrix.png", "Test split")

    if args.extra_dir:
        t2, p2, _ = predict_folder(model, args.extra_dir, classes, size, device, args.batch, args.workers)
        metrics["real_world"] = summarize(t2, p2, classes)
        save_confusion_png(metrics["real_world"]["confusion_matrix"], classes, out_dir / "confusion_matrix_real_world.png", "Your own photos")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(targets, preds, target_names=classes, digits=3, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    print(f"Test images: {result['n']}  |  accuracy {result['accuracy']:.3f}  |  macro-F1 {result['macro_f1']:.3f}\n")
    print(report)
    print("Confusion matrix (rows = actual, columns = predicted):", classes)
    for row, name in zip(result["confusion_matrix"], classes):
        print(f"  {name:<8}", row)
    if "per_fruit_accuracy" in result:
        print("\nPer fruit:", {k: round(v["accuracy"], 3) for k, v in result["per_fruit_accuracy"].items()})
    if "real_world" in metrics:
        print(f"\nYour own photos: {metrics['real_world']['n']} images, accuracy {metrics['real_world']['accuracy']:.3f}")
    print(f"\nSaved metrics.json, confusion_matrix.png, classification_report.txt, misclassified.csv to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
