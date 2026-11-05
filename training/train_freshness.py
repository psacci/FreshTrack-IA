"""Train the MobileNetV2 freshness classifier with transfer learning.

Two phases
  1. Warm-up:   backbone frozen, only the new final layer learns (fast, stable).
  2. Fine-tune: whole network learns with a small learning rate and cosine decay.

The checkpoint with the best validation macro-F1 is saved to
models/freshness_classification/mobilenetv2_freshness.pt

Usage:  python training/train_freshness.py
        python training/train_freshness.py --epochs 12 --batch 32
Needs:  dataset/freshness/{train,val}/{fresh,aging,spoiled}/  (see prepare_dataset.py)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from modules.freshness_classifier import build_model  # noqa: E402


def make_transforms(size: int):
    norm = transforms.Normalize(settings.IMAGENET_MEAN, settings.IMAGENET_STD)
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        # No hue shift: colour (green vs brown) is exactly what signals freshness.
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        norm,
    ])
    evaluate = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])
    return train, evaluate


def limit_per_class(ds: datasets.ImageFolder, per_class: int, seed: int) -> Subset:
    rng = random.Random(seed)
    keep = []
    for cls_idx in range(len(ds.classes)):
        idxs = [i for i, t in enumerate(ds.targets) if t == cls_idx]
        rng.shuffle(idxs)
        keep += idxs[:per_class]
    return Subset(ds, sorted(keep))


def run_epoch(model, loader, device, criterion, optimizer=None, backbone_eval=False):
    training = optimizer is not None
    model.train(training)
    if training and backbone_eval:
        model.features.eval()  # keep BatchNorm statistics steady while the backbone is frozen
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds += logits.argmax(1).cpu().tolist()
            targets += labels.cpu().tolist()
    n = len(targets)
    acc = float(np.mean(np.array(preds) == np.array(targets)))
    return total_loss / n, acc, float(f1_score(targets, preds, average="macro", zero_division=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=settings.DATASET_DIR)
    ap.add_argument("--out", type=Path, default=settings.FRESHNESS_CHECKPOINT)
    ap.add_argument("--warmup-epochs", type=int, default=3, help="epochs with the backbone frozen")
    ap.add_argument("--epochs", type=int, default=12, help="fine-tuning epochs")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-4, help="fine-tuning learning rate")
    ap.add_argument("--patience", type=int, default=4, help="stop fine-tuning after this many epochs without improvement")
    ap.add_argument("--image-size", type=int, default=settings.FRESHNESS_IMAGE_SIZE)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-per-class", type=int, default=0, help="debug: use only N training images per class")
    ap.add_argument("--no-pretrained", action="store_true", help="debug: skip ImageNet weights")
    ap.add_argument("--init-from", type=Path, default=None, help="continue from an earlier checkpoint's weights")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for split in ("train", "val"):
        if not (args.data / split).is_dir():
            print(f"Missing {args.data / split}. Run training/prepare_dataset.py first.")
            return 1

    device = settings.get_device()
    train_tf, eval_tf = make_transforms(args.image_size)
    train_ds = datasets.ImageFolder(args.data / "train", train_tf)
    val_ds = datasets.ImageFolder(args.data / "val", eval_tf)
    if train_ds.classes != val_ds.classes:
        print("train and val have different class folders:", train_ds.classes, val_ds.classes)
        return 1
    classes = train_ds.classes
    manifest_path = args.data / "manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
            foods = sorted({
                row["fruit"].strip().lower().replace("_", " ")
                for row in csv.DictReader(manifest_file)
                if row.get("fruit", "").strip()
            })
    else:
        foods = list(settings.FRESHNESS_FRUITS)
    counts = np.bincount(train_ds.targets, minlength=len(classes))
    print(f"Device: {device} | classes: {classes} | foods: {foods} | train per class: {counts.tolist()} | val images: {len(val_ds)}")

    # Rare classes get a bigger say in the loss so the model does not ignore them.
    weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    train_set = limit_per_class(train_ds, args.limit_per_class, args.seed) if args.limit_per_class else train_ds
    loader_args = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=(device == "cuda"),
                       persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    model = build_model(len(classes), pretrained=not args.no_pretrained)
    if args.init_from:
        previous = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if list(previous["classes"]) != list(classes):
            print("The checkpoint's classes do not match the dataset:", previous["classes"], classes)
            return 1
        model.load_state_dict(previous["state_dict"])
        print(f"Continuing from {args.init_from}")
    model = model.to(device)
    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def record(phase, epoch, tr, va):
        row = dict(phase=phase, epoch=epoch, train_loss=tr[0], train_acc=tr[1], val_loss=va[0], val_acc=va[1], val_macro_f1=va[2])
        history.append(row)
        print(f"[{phase:<9}] epoch {epoch:>2} | train loss {tr[0]:.3f} acc {tr[1]:.3f} | val loss {va[0]:.3f} acc {va[1]:.3f} macro-F1 {va[2]:.3f}"
              f" | {time.time() - started:.0f}s")

    def save_if_best(epoch, val_f1):
        nonlocal best_f1, best_epoch, stale
        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            torch.save({
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "classes": list(classes), "image_size": args.image_size, "epoch": epoch,
                "val_macro_f1": val_f1, "created": datetime.now().isoformat(timespec="seconds"),
                "pretrained": not args.no_pretrained,
                "foods": foods,
            }, args.out)
            foods_path = args.out.with_suffix(".foods.json")
            foods_path.write_text(json.dumps({"foods": foods}, indent=2), encoding="utf-8")
            return True
        stale += 1
        return False

    # Phase 1: only the new classifier layer learns.
    for p in model.features.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr_head, weight_decay=1e-4)
    for epoch in range(1, args.warmup_epochs + 1):
        tr = run_epoch(model, train_loader, device, criterion, opt, backbone_eval=True)
        va = run_epoch(model, val_loader, device, criterion)
        record("warm-up", epoch, tr, va)
        save_if_best(epoch, va[2])

    # Phase 2: fine-tune everything.
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    stale = 0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, criterion, opt)
        va = run_epoch(model, val_loader, device, criterion)
        sched.step()
        record("fine-tune", epoch, tr, va)
        if save_if_best(args.warmup_epochs + epoch, va[2]):
            print("            saved a new best checkpoint")
        if stale >= args.patience:
            print(f"No improvement for {args.patience} epochs. Stopping early.")
            break

    log = dict(history=history, best_val_macro_f1=best_f1, best_epoch=best_epoch, device=device,
               seconds=round(time.time() - started), classes=list(classes), train_counts=counts.tolist(),
               args={k: str(v) for k, v in vars(args).items()})
    (args.out.parent / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        x = range(1, len(history) + 1)
        ax[0].plot(x, [h["train_loss"] for h in history], label="train")
        ax[0].plot(x, [h["val_loss"] for h in history], label="validation")
        ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(x, [h["train_acc"] for h in history], label="train")
        ax[1].plot(x, [h["val_acc"] for h in history], label="validation")
        ax[1].plot(x, [h["val_macro_f1"] for h in history], label="validation macro-F1")
        ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(args.out.parent / "training_curves.png", dpi=140)
    except Exception as exc:  # noqa: BLE001 - plots are optional
        print("Could not save training curves:", exc)

    print(f"\nBest validation macro-F1 {best_f1:.3f} at epoch {best_epoch}. Checkpoint: {args.out}")
    print("Next: python training/evaluate_freshness.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
