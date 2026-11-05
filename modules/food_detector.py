"""YOLO food detection (COCO-pretrained by default, optionally custom-trained).

COCO only knows a handful of foods. We keep the ones it really contains
(see `settings.YOLO_TO_CATEGORY`); custom checkpoints may add trained classes.
Nothing is ever invented.
"""
from __future__ import annotations

import importlib.util
import logging
import re
import threading
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from config import settings
from modules.status import ModelStatus

log = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]        # x1, y1, x2, y2 in pixels
    category: str
    source: str = "local"
    quality: Optional[str] = None


def _food_metadata(raw_label: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve plain food names and VegQual's quality-prefixed class names."""
    label = re.sub(r"[_\s]+", " ", str(raw_label).strip().lower())
    quality = None
    for prefix, value in (("fresh ", "fresh"), ("defected ", "defective")):
        if label.startswith(prefix):
            quality = value
            label = label[len(prefix):]
            break
    label = settings.YOLO_LABEL_ALIASES.get(label, label)
    category = settings.YOLO_TO_CATEGORY.get(label)
    if category is None:
        return None, None, None
    return label, category, quality


def filter_detections(
    raw: Iterable[tuple[str, float, Sequence[float]]],
    min_conf: float = settings.DETECTION_MIN_CONFIDENCE,
) -> list[Detection]:
    """Keep supported food classes above the confidence floor, best first."""
    kept = []
    for label, conf, box in raw:
        food, category, quality = _food_metadata(label)
        if food is None or category is None or conf < min_conf:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        kept.append(Detection(food, float(conf), (x1, y1, x2, y2), category, quality=quality))
    return sorted(kept, key=lambda d: -d.confidence)


def count_by_label(detections: Iterable[Detection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return counts


class FoodDetector:
    def __init__(self, weights=None) -> None:
        self.weights = weights or settings.YOLO_WEIGHTS
        self._model = None
        self._world_model = None
        self._device = "cpu"
        self._inference_lock = threading.Lock()

    def status(self) -> ModelStatus:
        if importlib.util.find_spec("ultralytics") is None:
            return ModelStatus(False, "Ultralytics is not installed. Run: pip install -r requirements.txt")
        if self.weights.exists():
            if self.weights.name == "yolov8n.pt":
                return ModelStatus(True, "YOLOv8n ready (COCO-pretrained)")
            return ModelStatus(True, f"Custom food detector ready ({self.weights.name})")
        if self.weights.name != "yolov8n.pt":
            return ModelStatus(False, f"Configured food detector checkpoint is missing: {self.weights}")
        return ModelStatus(True, "YOLOv8n will download on first scan (about 6 MB)")

    def _load(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        self._device = settings.get_device()
        path = self.weights
        if not path.exists():
            if path.name != "yolov8n.pt":
                raise FileNotFoundError(f"Configured food detector checkpoint is missing: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from ultralytics.utils.downloads import attempt_download_asset
                attempt_download_asset(str(path))
            except Exception as exc:  # noqa: BLE001
                log.warning("Direct weight download failed (%s); letting Ultralytics handle it.", exc)
        self._model = YOLO(str(path) if path.exists() else "yolov8n.pt")
        return self._model

    def detect(self, image_bgr: np.ndarray, min_conf: Optional[float] = None) -> list[Detection]:
        """Run detection. Returns an empty list when nothing supported is found."""
        min_conf = settings.DETECTION_MIN_CONFIDENCE if min_conf is None else min_conf
        with self._inference_lock:
            model = self._load()
            try:
                results = model.predict(image_bgr, conf=min_conf, device=self._device, verbose=False)
            except Exception as exc:  # noqa: BLE001 - e.g. an MPS op not supported
                log.warning("Detection on %s failed (%s). Retrying on CPU.", self._device, exc)
                self._device = "cpu"
                results = model.predict(image_bgr, conf=min_conf, device="cpu", verbose=False)

        raw = []
        for result in results:
            names = result.names
            for box in result.boxes:
                raw.append((names[int(box.cls[0])], float(box.conf[0]), box.xyxy[0].tolist()))
        return filter_detections(raw, min_conf)

    def detect_extended(self, image_bgr: np.ndarray, min_conf: Optional[float] = None) -> list[Detection]:
        """Use a local open-vocabulary model for foods outside the main detector."""
        min_conf = settings.DETECTION_MIN_CONFIDENCE if min_conf is None else min_conf
        with self._inference_lock:
            if self._world_model is None:
                from ultralytics import YOLOWorld

                weights = settings.WORLD_FOOD_WEIGHTS
                if not weights.exists():
                    weights.parent.mkdir(parents=True, exist_ok=True)
                    from ultralytics.utils.downloads import attempt_download_asset
                    attempt_download_asset(str(weights))
                self._world_model = YOLOWorld(str(weights))
                self._world_model.set_classes(list(settings.WORLD_FOOD_PROMPTS))
            model = self._world_model
            try:
                results = model.predict(image_bgr, conf=min_conf, device=self._device, verbose=False)
            except Exception as exc:  # noqa: BLE001 - e.g. an MPS op not supported
                log.warning("Extended detection on %s failed (%s). Retrying on CPU.", self._device, exc)
                self._device = "cpu"
                results = model.predict(image_bgr, conf=min_conf, device="cpu", verbose=False)

        raw = []
        for result in results:
            names = result.names
            for box in result.boxes:
                raw.append((names[int(box.cls[0])], float(box.conf[0]), box.xyxy[0].tolist()))
        detections = filter_detections(raw, min_conf)
        return [
            Detection(det.label, det.confidence, det.box, det.category, source="open-vocabulary")
            for det in detections
        ]
