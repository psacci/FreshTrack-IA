"""Optional: train a custom YOLOv8 detector on more food classes.

The app works without this. It uses the COCO-pretrained yolov8n.pt and accepts
apple, banana, orange, carrot, broccoli, cake, donut, sandwich, hot dog and pizza.
To recognise more foods you need your own labelled photos.

Steps
  1. Put photos in dataset/detection/images/{train,val}/ and YOLO-format labels
     (one .txt per image) in dataset/detection/labels/{train,val}/.
  2. Copy dataset/detection/data.yaml.example to dataset/detection/data.yaml and list your classes.
  3. Run:  python training/train_yolo.py --epochs 60
  4. The best weights are copied to models/food_detection/custom_food.pt (or --out).
     Point the app at them with FRESHTRACK_YOLO=<checkpoint path>
     and add the new class names to YOLO_TO_CATEGORY in config/settings.py.
     Use --weights to continue fine-tuning a prior checkpoint.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "dataset" / "detection" / "data.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", type=Path, help="initial weights; defaults to the COCO-pretrained YOLOv8n")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "food_detection" / "custom_food.pt",
                    help="output checkpoint path; defaults to the existing custom detector path")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"Missing {args.data}. Copy data.yaml.example and edit it first (see this file's header).")
        return 1
    from ultralytics import YOLO

    device = settings.get_device()
    default_weights = settings.MODELS_DIR / "food_detection" / "yolov8n.pt"
    initial_weights = args.weights or (default_weights if default_weights.exists() else Path("yolov8n.pt"))
    model = YOLO(str(initial_weights))
    results = model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                          device=device, project=str(ROOT / "runs"), name=args.out.stem, exist_ok=True)
    best = Path(results.save_dir) / "weights" / "best.pt"
    target = args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, target)
    print(f"Saved {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
