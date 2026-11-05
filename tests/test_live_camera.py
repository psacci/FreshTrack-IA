import numpy as np
from av import VideoFrame

from modules.food_detector import Detection
from modules.freshness_classifier import FreshnessPrediction
from modules.live_camera import LiveFoodProcessor, suggest_food_from_text
from ui import scan


def test_suggests_common_food_names_from_general_package_text():
    assert suggest_food_from_text("MILK | FULL CREAM") == ("Full Cream Milk", "Dairy")
    assert suggest_food_from_text("GREEK YOGURT") == ("Yogurt", "Dairy")
    assert suggest_food_from_text("WHOLE GRAIN BREAD") == ("Bread", "Bakery")
    assert suggest_food_from_text("MYSTERY BRAND") == (None, "Dry & Packaged")
    assert suggest_food_from_text("") == (None, "Dry & Packaged")


def test_live_processor_returns_video_and_reads_current_frame_text():
    class Detector:
        def detect(self, image):
            return [Detection("apple", 0.9, (5, 5, 50, 50), "Fruits & Vegetables")]

    class OCR:
        def read_lines(self, gray, psm):
            class Line:
                text = "FULL CREAM MILK"
                confidence = 91

            return [Line()]

    class Classifier:
        def supports_food(self, name):
            return name == "apple"

        def predict(self, image):
            return FreshnessPrediction(
                label="fresh",
                confidence=0.93,
                probabilities={"fresh": 0.93, "aging": 0.05, "spoiled": 0.02},
                uncertain=False,
            )

    processor = LiveFoodProcessor(Detector(), Classifier(), OCR(), freshness_enabled=True, ocr_enabled=True)
    frame = VideoFrame.from_ndarray(np.zeros((80, 120, 3), dtype=np.uint8), format="bgr24")

    output = processor.recv(frame)

    assert output.to_ndarray(format="bgr24").shape == (80, 120, 3)
    text, observations = processor.latest_observation()
    assert text == "FULL CREAM MILK"
    label, prediction = observations[0][0].label, observations[0][1]
    assert label == "apple"
    assert prediction is not None and prediction.display == "Fresh"


def test_live_processor_does_not_draw_ocr_status_over_camera():
    class Detector:
        def detect(self, image):
            return []

    class OCR:
        def read_lines(self, gray, psm):
            class Line:
                text = "MYSTERY BRAND"
                confidence = 91

            return [Line()]

    frame = VideoFrame.from_ndarray(np.zeros((80, 120, 3), dtype=np.uint8), format="bgr24")
    processor = LiveFoodProcessor(Detector(), object(), OCR(), freshness_enabled=False, ocr_enabled=True)

    output = processor.recv(frame).to_ndarray(format="bgr24")

    assert np.count_nonzero(output) == 0


def test_live_freshness_skips_unsupported_food_classes():
    class Detector:
        def detect(self, image):
            return [Detection("carrot", 0.9, (5, 5, 50, 50), "Fruits & Vegetables")]

    class Classifier:
        def predict(self, image):
            raise AssertionError("unsupported foods must not be freshness-classified")

    class OCR:
        def read_lines(self, gray, psm):
            return []

    processor = LiveFoodProcessor(Detector(), Classifier(), OCR(), freshness_enabled=True, ocr_enabled=False)
    frame = VideoFrame.from_ndarray(np.zeros((80, 120, 3), dtype=np.uint8), format="bgr24")

    processor.recv(frame)

    assert processor.latest_freshness() == [("carrot", None)]


def test_live_freshness_is_not_run_for_dairy():
    class Detector:
        def detect(self, image):
            return [Detection("milk", 0.9, (5, 5, 50, 50), "Dairy")]

    class Classifier:
        def supports_food(self, name):
            raise AssertionError("freshness checks are only for fruits and vegetables")

    class OCR:
        def read_lines(self, gray, psm):
            return []

    processor = LiveFoodProcessor(Detector(), Classifier(), OCR(), freshness_enabled=True, ocr_enabled=False)
    frame = VideoFrame.from_ndarray(np.zeros((80, 120, 3), dtype=np.uint8), format="bgr24")

    processor.recv(frame)

    assert processor.latest_freshness() == [("milk", None)]


def test_live_processor_keeps_result_and_freshness_until_new_food_is_detected():
    apple = Detection("apple", 0.9, (5, 5, 50, 50), "Fruits & Vegetables")
    cucumber = Detection("cucumber", 0.85, (10, 10, 55, 55), "Fruits & Vegetables")

    class Detector:
        results = [[apple], [], [cucumber]]

        def detect(self, image):
            return self.results.pop(0)

    class Classifier:
        def supports_food(self, name):
            return name == "apple"

        def predict(self, image):
            return FreshnessPrediction(
                label="fresh",
                confidence=0.93,
                probabilities={"fresh": 0.93, "aging": 0.05, "spoiled": 0.02},
                uncertain=False,
            )

    class OCR:
        def read_lines(self, gray, psm):
            return []

    processor = LiveFoodProcessor(Detector(), Classifier(), OCR(), freshness_enabled=True, ocr_enabled=False)
    frame = VideoFrame.from_ndarray(np.zeros((80, 120, 3), dtype=np.uint8), format="bgr24")

    processor.recv(frame)
    _, initial = processor.latest_observation()
    assert initial[0][0] == apple
    assert initial[0][1] is not None
    assert initial[0][1].display == "Fresh"
    initial_thumbnail = processor.latest_thumbnail()
    assert initial_thumbnail is not None

    processor._next_detection = 0
    empty_frame = processor.recv(frame).to_ndarray(format="bgr24")
    _, retained = processor.latest_observation()
    assert not np.count_nonzero(empty_frame)
    assert retained == initial
    assert processor.latest_thumbnail() == initial_thumbnail

    processor._next_detection = 0
    processor.recv(frame)
    _, replaced = processor.latest_observation()
    assert [detection.label for detection, _ in replaced] == ["cucumber"]
    assert replaced[0][1] is None
    assert processor.latest_thumbnail() is not None


def test_live_freshness_info_includes_label_and_model_confidence():
    detection = Detection("apple", 0.9, (5, 5, 50, 50), "Fruits & Vegetables")
    prediction = FreshnessPrediction(
        label="fresh",
        confidence=0.93,
        probabilities={"fresh": 0.93, "aging": 0.05, "spoiled": 0.02},
        uncertain=False,
    )

    message = scan._live_freshness_info(detection, prediction)

    assert message == (
        "Freshness appearance: Fresh (93% model confidence). Not a food-safety test."
    )


def test_live_freshness_info_explains_unsupported_food(monkeypatch):
    class Classifier:
        def status(self):
            class Status:
                available = True

            return Status()

        def supports_food(self, name):
            return False

    monkeypatch.setattr(scan, "get_classifier", lambda: Classifier())
    detection = Detection("cucumber", 0.85, (5, 5, 50, 50), "Fruits & Vegetables")

    message = scan._live_freshness_info(detection, None)

    assert message == (
        "Freshness appearance is not available for cucumber; "
        "this food is not in the loaded freshness model."
    )


def test_live_processor_extracts_dates_for_package_camera():
    class Detector:
        def detect(self, image):
            return []

    class Classifier:
        pass

    class OCR:
        def read_lines(self, gray, psm):
            class Line:
                text = "EXP 25 JUN 2027"
                confidence = 91

            return [Line()]

    processor = LiveFoodProcessor(Detector(), Classifier(), OCR(), freshness_enabled=False, ocr_enabled=True)
    frame = VideoFrame.from_ndarray(np.zeros((80, 120, 3), dtype=np.uint8), format="bgr24")

    processor.recv(frame)

    raw_text, candidates = processor.latest_date_observation()
    assert raw_text == "EXP 25 JUN 2027"
    assert len(candidates) == 1
    assert candidates[0].label_type == "expiration"
    assert candidates[0].parsed.isoformat() == "2027-06-25"
