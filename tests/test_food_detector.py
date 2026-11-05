import numpy as np
import pytest

from modules.food_detector import FoodDetector, count_by_label, filter_detections
from modules.image_processor import draw_detections


RAW = [
    ("apple", 0.91, (10.2, 10, 100, 120)),
    ("banana", 0.60, (150, 20, 260, 90)),
    ("person", 0.99, (0, 0, 50, 50)),      # not food: must be dropped
    ("laptop", 0.88, (0, 0, 50, 50)),      # not food: must be dropped
    ("orange", 0.20, (5, 5, 50, 50)),      # below the confidence floor
    ("apple", 0.75, (300, 30, 380, 110)),
]


def test_only_supported_food_above_threshold_is_kept():
    dets = filter_detections(RAW, min_conf=0.35)
    assert [d.label for d in dets] == ["apple", "apple", "banana"]
    assert all(d.category == "Fruits & Vegetables" for d in dets)


def test_output_format():
    d = filter_detections(RAW)[0]
    assert d.label == "apple" and d.confidence == 0.91
    assert d.box == (10, 10, 100, 120) and all(isinstance(v, int) for v in d.box)


def test_counts():
    assert count_by_label(filter_detections(RAW)) == {"apple": 2, "banana": 1}


def test_no_detection_returns_empty_list():
    assert filter_detections([]) == []
    assert filter_detections([("chair", 0.9, (0, 0, 5, 5))]) == []


def test_recognizes_more_coco_foods_at_default_confidence():
    dets = filter_detections([
        ("sandwich", 0.30, (0, 0, 50, 50)),
        ("hot dog", 0.27, (50, 0, 100, 50)),
        ("pizza", 0.25, (100, 0, 150, 50)),
    ])
    assert [det.label for det in dets] == ["sandwich", "hot dog", "pizza"]
    assert [det.category for det in dets] == ["Bakery", "Meat & Seafood", "Bakery"]


def test_normalizes_vegqual_food_and_preserves_its_dataset_quality_label():
    dets = filter_detections([
        ("fresh_tomato", 0.88, (0, 0, 30, 30)),
        ("defected_bitter_gourd", 0.76, (40, 0, 90, 50)),
    ])
    assert [det.label for det in dets] == ["tomato", "bitter gourd"]
    assert [det.quality for det in dets] == ["fresh", "defective"]
    assert all(det.category == "Fruits & Vegetables" for det in dets)


def test_accepts_remote_food_names_without_expanding_coco_model_classes():
    dets = filter_detections([
        ("cucumber", 0.9, (0, 0, 50, 50)),
        ("egg", 0.8, (50, 0, 100, 50)),
    ])
    assert [det.label for det in dets] == ["cucumber", "egg"]
    assert [det.category for det in dets] == ["Fruits & Vegetables", "Dairy"]


def test_every_local_world_food_prompt_has_a_category_and_hotdog_alias():
    from config import settings
    from modules.food_detector import _food_metadata

    assert all(prompt in settings.YOLO_TO_CATEGORY for prompt in settings.WORLD_FOOD_PROMPTS)
    hotdog = filter_detections([("hotdog", 0.8, (0, 0, 40, 30))])[0]
    assert (hotdog.label, hotdog.category) == ("hot dog", "Meat & Seafood")
    assert _food_metadata("ice cream")[:2] == ("ice cream", "Dairy")


def test_extended_detector_uses_local_world_model_for_meat_and_dairy(tmp_path, monkeypatch):
    from config import settings

    weight_path = tmp_path / "world.pt"
    weight_path.touch()
    monkeypatch.setattr(settings, "WORLD_FOOD_WEIGHTS", weight_path)

    class FakeBox:
        def __init__(self, class_index, confidence, coords):
            self.cls = np.array([class_index])
            self.conf = np.array([confidence])
            self.xyxy = np.array([coords])

    class FakeWorldModel:
        def set_classes(self, classes):
            self.classes = classes

        def predict(self, image, conf, device, verbose):
            names = {index: label for index, label in enumerate(self.classes)}
            return [type("Result", (), {
                "names": names,
                "boxes": [
                    FakeBox(self.classes.index("beef steak"), 0.91, (0, 0, 50, 50)),
                    FakeBox(self.classes.index("milk"), 0.84, (50, 0, 100, 50)),
                ],
            })()]

    monkeypatch.setattr("ultralytics.YOLOWorld", lambda path: FakeWorldModel())
    detections = FoodDetector().detect_extended(np.zeros((100, 100, 3), dtype=np.uint8))

    assert [(det.label, det.category, det.source) for det in detections] == [
        ("beef steak", "Meat & Seafood", "open-vocabulary"),
        ("milk", "Dairy", "open-vocabulary"),
    ]


def test_normalizes_lvis_food_aliases():
    dets = filter_detections([
        ("cucumber/cuke", 0.9, (0, 0, 50, 50)),
        ("bean curd/tofu", 0.8, (50, 0, 100, 50)),
        ("bell pepper/capsicum", 0.7, (100, 0, 150, 50)),
        ("edible corn/corn/maize", 0.6, (150, 0, 200, 50)),
    ])
    assert [det.label for det in dets] == ["cucumber", "tofu", "capsicum", "corn"]
    assert [det.category for det in dets] == [
        "Fruits & Vegetables", "Dry & Packaged", "Fruits & Vegetables", "Fruits & Vegetables",
    ]


def test_annotated_image_shape_and_input_untouched():
    img = np.full((200, 400, 3), 255, np.uint8)
    out = draw_detections(img, filter_detections(RAW))
    assert out.shape == img.shape and (img == 255).all()
    assert (out != 255).any()


def test_detector_status_does_not_load_the_model():
    detector = FoodDetector()
    assert detector.status().message
    assert detector._model is None


def test_missing_custom_checkpoint_is_not_reported_as_ready(tmp_path):
    detector = FoodDetector(weights=tmp_path / "freshtrack_lvis.pt")
    status = detector.status()
    assert not status.available
    assert "checkpoint is missing" in status.message


def test_default_detector_prefers_lvis_checkpoint_when_available(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
    lvis_weights = tmp_path / "food_detection" / "freshtrack_lvis.pt"
    lvis_weights.parent.mkdir()
    lvis_weights.touch()

    assert settings._default_yolo_weights() == lvis_weights


def test_default_detector_uses_coco_until_lvis_checkpoint_exists(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)

    assert settings._default_yolo_weights() == tmp_path / "food_detection" / "yolov8n.pt"
