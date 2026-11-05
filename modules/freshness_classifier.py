"""MobileNetV2 visual freshness classifier (Fresh / Aging / Visible spoilage).

Honest-by-design:
* No checkpoint -> the app says the model is not trained. No fake predictions.
* Low confidence -> the prediction is flagged "uncertain".
* It judges visible appearance only. It cannot see bacteria or toxins, and
  "Fresh" is never proof that food is safe to eat.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from config import settings
from modules.status import ModelStatus

log = logging.getLogger(__name__)


@dataclass
class FreshnessPrediction:
    label: str                       # fresh | aging | spoiled
    confidence: float                # 0-1
    probabilities: dict[str, float]
    uncertain: bool

    @property
    def display(self) -> str:
        return settings.FRESHNESS_LABELS[self.label]


def build_model(num_classes: int = len(settings.FRESHNESS_CLASSES), pretrained: bool = False):
    """MobileNetV2 with the last layer replaced for our classes."""
    import torch.nn as nn
    from torchvision import models

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, num_classes))
    return model


def eval_transform(size: int = settings.FRESHNESS_IMAGE_SIZE):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(settings.IMAGENET_MEAN, settings.IMAGENET_STD),
    ])


def is_supported_fruit(name: str) -> bool:
    return name.strip().lower() in settings.FRESHNESS_FRUITS


def load_metrics(path: Path = settings.FRESHNESS_METRICS) -> Optional[dict]:
    """Real evaluation results written by training/evaluate_freshness.py, if any."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_checkpoint_metrics(checkpoint: Path) -> Optional[dict]:
    """Load evaluation metrics only when they identify the requested checkpoint."""
    checkpoint = Path(checkpoint)
    paths = (
        checkpoint.parent / f"{checkpoint.stem}_evaluation" / "metrics.json",
        checkpoint.parent / "metrics.json",
    )
    for path in paths:
        metrics = load_metrics(path)
        if metrics and metrics.get("checkpoint") == checkpoint.name:
            return metrics
    return None


class FreshnessClassifier:
    def __init__(self, checkpoint: Optional[Path] = None) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else settings.FRESHNESS_CHECKPOINT
        self._model = None
        self._classes = settings.FRESHNESS_CLASSES
        self._supported_foods: Optional[tuple[str, ...]] = None
        self._size = settings.FRESHNESS_IMAGE_SIZE
        self._device = "cpu"

    def supported_foods(self) -> tuple[str, ...]:
        if self._supported_foods is not None:
            return self._supported_foods

        foods_path = self.checkpoint.with_suffix(".foods.json")
        if foods_path.exists():
            try:
                data = json.loads(foods_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Could not read freshness food metadata from {foods_path}: {exc}") from exc
            foods = data.get("foods") if isinstance(data, dict) else None
            if not isinstance(foods, list) or not all(isinstance(food, str) for food in foods):
                raise RuntimeError(f"Freshness food metadata in {foods_path} must contain a list of food names.")
            self._supported_foods = tuple(sorted({
                re.sub(r"[_\s]+", " ", food.strip().lower())
                for food in foods
                if food.strip()
            }))
        else:
            self._supported_foods = tuple(settings.FRESHNESS_FRUITS)
        return self._supported_foods

    def supports_food(self, name: str) -> bool:
        normalized = re.sub(r"[_\s]+", " ", name.strip().lower())
        return normalized in self.supported_foods()

    def status(self) -> ModelStatus:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ImportError:
            return ModelStatus(False, "PyTorch is not installed. Run: pip install -r requirements.txt")
        if not self.checkpoint.exists():
            return ModelStatus(
                False,
                "The freshness model has not been trained yet. "
                "Follow documentation/MODEL_TRAINING_GUIDE.md, then restart the app.",
            )
        return ModelStatus(True, "Freshness model ready")

    def _load(self):
        if self._model is not None:
            return self._model
        import torch

        self._device = settings.get_device()
        ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._classes = tuple(ckpt.get("classes", self._classes))
        checkpoint_foods = ckpt.get("foods")
        if isinstance(checkpoint_foods, list) and all(isinstance(food, str) for food in checkpoint_foods):
            self._supported_foods = tuple(sorted({
                re.sub(r"[_\s]+", " ", food.strip().lower())
                for food in checkpoint_foods
                if food.strip()
            }))
        self._size = int(ckpt.get("image_size", self._size))
        model = build_model(len(self._classes), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        self._model = model.to(self._device).eval()
        return self._model

    def preprocess(self, image_rgb: np.ndarray):
        """RGB array -> normalised tensor of shape [1, 3, size, size] (size comes from the checkpoint)."""
        from PIL import Image

        tensor = eval_transform(self._size)(Image.fromarray(image_rgb).convert("RGB"))
        return tensor.unsqueeze(0)

    def predict(self, image_rgb: np.ndarray) -> FreshnessPrediction:
        state = self.status()
        if not state.available:
            raise RuntimeError(state.message)
        import torch

        model = self._load()
        with torch.no_grad():
            logits = model(self.preprocess(image_rgb).to(self._device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        best = int(probs.argmax())
        confidence = float(probs[best])
        return FreshnessPrediction(
            label=self._classes[best],
            confidence=confidence,
            probabilities={c: float(p) for c, p in zip(self._classes, probs)},
            uncertain=confidence < settings.FRESHNESS_MIN_CONFIDENCE,
        )
