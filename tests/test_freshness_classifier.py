import numpy as np
import pytest

from config import settings
from modules.freshness_classifier import (
    FreshnessClassifier,
    is_supported_fruit,
    load_checkpoint_metrics,
    load_metrics,
)

torch = pytest.importorskip("torch")


def test_missing_checkpoint_is_reported_and_blocks_predictions(tmp_path):
    clf = FreshnessClassifier(tmp_path / "nope.pt")
    status = clf.status()
    assert not status.available and "not been trained" in status.message
    with pytest.raises(RuntimeError):
        clf.predict(np.zeros((64, 64, 3), np.uint8))


def test_preprocessing_shape_and_range():
    clf = FreshnessClassifier()
    t = clf.preprocess(np.random.randint(0, 255, (123, 77, 3), dtype=np.uint8))
    assert tuple(t.shape) == (1, 3, 224, 224) and t.dtype == torch.float32


def test_prediction_output_format_with_a_tiny_checkpoint(tmp_path):
    from modules.freshness_classifier import build_model

    model = build_model(pretrained=False)
    path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": model.state_dict(), "classes": list(settings.FRESHNESS_CLASSES)}, path)
    pred = FreshnessClassifier(path).predict(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    assert pred.label in settings.FRESHNESS_CLASSES
    assert abs(sum(pred.probabilities.values()) - 1) < 1e-4
    assert 0 <= pred.confidence <= 1 and isinstance(pred.uncertain, bool)


def test_supported_fruit_list():
    assert is_supported_fruit("Apple") and is_supported_fruit(" banana ")
    assert not is_supported_fruit("lettuce")


def test_trained_food_manifest_expands_supported_freshness_types(tmp_path):
    checkpoint = tmp_path / "custom.pt"
    checkpoint.with_suffix(".foods.json").write_text(
        '{"foods":["tomato","bitter_gourd","cucumber"]}', encoding="utf-8"
    )
    clf = FreshnessClassifier(checkpoint)
    assert clf.supports_food("Bitter Gourd")
    assert clf.supports_food("cucumber")
    assert not clf.supports_food("egg")


def test_invalid_trained_food_manifest_is_reported(tmp_path):
    checkpoint = tmp_path / "custom.pt"
    checkpoint.with_suffix(".foods.json").write_text("not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Could not read freshness food metadata"):
        FreshnessClassifier(checkpoint).supports_food("tomato")


def test_metrics_missing_returns_none(tmp_path):
    assert load_metrics(tmp_path / "metrics.json") is None


def test_checkpoint_metrics_must_match_checkpoint_name(tmp_path):
    checkpoint = tmp_path / "active.pt"
    (tmp_path / "metrics.json").write_text(
        '{"checkpoint":"other.pt","test":{"accuracy":0.99}}',
        encoding="utf-8",
    )
    assert load_checkpoint_metrics(checkpoint) is None

    report = tmp_path / "active_evaluation"
    report.mkdir()
    (report / "metrics.json").write_text(
        '{"checkpoint":"active.pt","test":{"accuracy":0.91}}',
        encoding="utf-8",
    )
    assert load_checkpoint_metrics(checkpoint)["test"]["accuracy"] == 0.91
