"""Evaluate the LVIS produce detector on its held-out test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402


def class_average_precision(box_metrics, class_id: int) -> tuple[float, float] | None:
    """Return AP50 and AP50-95 for a class, accounting for metrics' compact class axis."""
    class_indices = list(box_metrics.ap_class_index)
    if class_id not in class_indices:
        return None
    metric_index = class_indices.index(class_id)
    return float(box_metrics.ap50[metric_index]), float(box_metrics.maps[class_id])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "raw" / "lvis-fruits-vegetables" / "freshtrack_data.yaml",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "models" / "food_detection" / "freshtrack_lvis.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON report path (defaults beside the checkpoint)",
    )
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset config not found: {args.data}")
    if not args.weights.is_file():
        parser.error(f"Detector checkpoint not found: {args.weights}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.data), split="test", device=settings.get_device())
    names = metrics.names
    names_by_id = names if isinstance(names, dict) else dict(enumerate(names))
    class_ids = {
        str(name).strip().lower(): int(class_id)
        for class_id, name in names_by_id.items()
    }

    print(f"Held-out test mAP50: {metrics.box.map50:.4f}")
    print(f"Held-out test mAP50-95: {metrics.box.map:.4f}")
    per_class = {}
    for label in ("cucumber/cuke", "banana"):
        class_id = class_ids.get(label)
        if class_id is None:
            print(f"{label}: class missing from checkpoint")
            continue
        values = class_average_precision(metrics.box, class_id)
        if values is None:
            print(f"{label}: no instances in held-out test set")
            continue
        ap50, ap = values
        print(f"{label}: AP50={ap50:.4f}, AP50-95={ap:.4f}")
        per_class[label] = {"ap50": ap50, "ap50_95": ap}

    report_path = args.out or args.weights.with_name(f"{args.weights.stem}_test_metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "weights": str(args.weights),
        "dataset": str(args.data),
        "split": "test",
        "images": len(metrics.box.image_metrics),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
    }, indent=2), encoding="utf-8")
    print(f"Saved detector evaluation to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
