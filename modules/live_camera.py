"""Live browser-camera food detection and package-label text reading."""
from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
from av import VideoFrame
from streamlit_webrtc import VideoProcessorBase

from modules.expiration_date_reader import DateCandidate, TesseractEngine, parse_lines
from modules.freshness_classifier import FreshnessClassifier, FreshnessPrediction
from modules.food_detector import Detection, FoodDetector
from modules.image_processor import bgr_to_rgb, crop_box, draw_detections, ocr_variants, thumbnail_jpeg
from modules.roboflow_workflow import RoboflowWorkflow, merge_detections

log = logging.getLogger(__name__)

_MAX_DETECTION_SIDE = 640
_DETECTION_INTERVAL = 0.5
_FRESHNESS_INTERVAL = 1.5
_OCR_INTERVAL = 2.0
_ROBOFLOW_INTERVAL = 3.0
_REMOTE_RESULT_MAX_AGE = 8.0
_PRODUCT_RESULT_MAX_AGE = 4.5


_FOOD_TEXT_CATEGORIES = {
    "apple": ("Apple", "Fruits & Vegetables"),
    "banana": ("Banana", "Fruits & Vegetables"),
    "orange": ("Orange", "Fruits & Vegetables"),
    "tomato": ("Tomato", "Fruits & Vegetables"),
    "cucumber": ("Cucumber", "Fruits & Vegetables"),
    "carrot": ("Carrot", "Fruits & Vegetables"),
    "lettuce": ("Lettuce", "Fruits & Vegetables"),
    "broccoli": ("Broccoli", "Fruits & Vegetables"),
    "potato": ("Potato", "Fruits & Vegetables"),
    "milk": ("Milk", "Dairy"),
    "yogurt": ("Yogurt", "Dairy"),
    "yoghurt": ("Yogurt", "Dairy"),
    "cheese": ("Cheese", "Dairy"),
    "butter": ("Butter", "Dairy"),
    "cream": ("Cream", "Dairy"),
    "egg": ("Eggs", "Dairy"),
    "bread": ("Bread", "Bakery"),
    "cake": ("Cake", "Bakery"),
    "donut": ("Donut", "Bakery"),
    "pizza": ("Pizza", "Bakery"),
    "sandwich": ("Sandwich", "Bakery"),
    "chicken": ("Chicken", "Meat & Seafood"),
    "beef": ("Beef", "Meat & Seafood"),
    "pork": ("Pork", "Meat & Seafood"),
    "tuna": ("Tuna", "Meat & Seafood"),
    "salmon": ("Salmon", "Meat & Seafood"),
    "rice": ("Rice", "Dry & Packaged"),
    "pasta": ("Pasta", "Dry & Packaged"),
    "noodles": ("Noodles", "Dry & Packaged"),
    "cereal": ("Cereal", "Dry & Packaged"),
    "beans": ("Beans", "Canned & Preserved"),
    "corn": ("Corn", "Fruits & Vegetables"),
}


def suggest_food_from_text(text: str) -> tuple[Optional[str], str]:
    """Suggest a common food name and category from OCR; leave unknown text editable."""
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if not normalized:
        return None, "Dry & Packaged"

    if re.search(r"\bfull\s+cream\b", normalized) and re.search(r"\bmilk\b", normalized):
        return "Full Cream Milk", "Dairy"

    for keyword in sorted(_FOOD_TEXT_CATEGORIES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(keyword)}\b", normalized):
            name, category = _FOOD_TEXT_CATEGORIES[keyword]
            return name, category

    return None, "Dry & Packaged"


class LiveFoodProcessor(VideoProcessorBase):
    def __init__(
        self,
        detector: FoodDetector,
        classifier: FreshnessClassifier,
        ocr_engine: TesseractEngine,
        freshness_enabled: bool,
        ocr_enabled: bool,
        roboflow_detector: Optional[RoboflowWorkflow] = None,
    ) -> None:
        self._detector = detector
        self._classifier = classifier
        self._ocr_engine = ocr_engine
        self._roboflow_detector = roboflow_detector
        self._remote_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="freshtrack-roboflow") if roboflow_detector else None
        self._remote_future: Optional[Future[list[Detection]]] = None
        self._remote_detections: list[Detection] = []
        self._remote_shape: Optional[tuple[int, int]] = None
        self._remote_scale = 1.0
        self._remote_image_data: Optional[bytes] = None
        self._remote_read_at = 0.0
        self._remote_error: Optional[str] = None
        self._freshness_enabled = freshness_enabled
        self._ocr_enabled = ocr_enabled
        self._lock = threading.Lock()
        self._detections: list[Detection] = []
        self._freshness_predictions: list[Optional[FreshnessPrediction]] = []
        self._presented_detections: list[Detection] = []
        self._presented_freshness_predictions: list[Optional[FreshnessPrediction]] = []
        self._presented_remote_detections: list[Detection] = []
        self._presented_image_data: Optional[bytes] = None
        self._detection_shape: Optional[tuple[int, int]] = None
        self._latest_text = ""
        self._latest_dates: list[DateCandidate] = []
        self._latest_date_image_data: Optional[bytes] = None
        self._latest_read_at = 0.0
        self._detection_error: Optional[str] = None
        self._freshness_error: Optional[str] = None
        self._ocr_error: Optional[str] = None
        self._next_detection = 0.0
        self._next_freshness = 0.0
        self._next_ocr = 0.0
        self._next_remote = 0.0

    def latest_observation(self) -> tuple[str, list[tuple[Detection, Optional[FreshnessPrediction]]]]:
        with self._lock:
            now = time.monotonic()
            text = self._latest_text if now - self._latest_read_at <= _PRODUCT_RESULT_MAX_AGE else ""
            observations = list(zip(self._presented_detections, self._presented_freshness_predictions))
            observations.extend((detection, None) for detection in self._presented_remote_detections)
            return text, observations

    def latest_date_observation(self) -> tuple[str, list[DateCandidate]]:
        with self._lock:
            if time.monotonic() - self._latest_read_at > _PRODUCT_RESULT_MAX_AGE:
                return "", []
            return self._latest_text, list(self._latest_dates)

    def latest_date_thumbnail(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_date_image_data

    def latest_freshness(self) -> list[tuple[str, Optional[FreshnessPrediction]]]:
        with self._lock:
            return list(zip(
                (d.label for d in self._detections),
                self._freshness_predictions,
            ))

    def latest_thumbnail(self) -> Optional[bytes]:
        with self._lock:
            return self._presented_image_data

    def latest_errors(self) -> list[str]:
        with self._lock:
            return [
                message for message in (
                    self._detection_error,
                    self._freshness_error,
                    self._ocr_error,
                    self._remote_error,
                ) if message
            ]

    def recv(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        now = time.monotonic()
        height, width = image.shape[:2]
        self._poll_remote(now)
        self._schedule_remote(image, now)

        if now >= self._next_detection:
            self._next_detection = now + _DETECTION_INTERVAL
            scale = min(1.0, _MAX_DETECTION_SIDE / max(height, width))
            resized = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
            try:
                detections = self._detector.detect(resized)
                scaled = [
                    Detection(
                        det.label,
                        det.confidence,
                        tuple(round(value / scale) for value in det.box),
                        det.category,
                        det.source,
                        det.quality,
                    )
                    for det in detections
                ]
                thumbnail = thumbnail_jpeg(image) if scaled else None
                with self._lock:
                    prior_predictions: dict[str, list[Optional[FreshnessPrediction]]] = {}
                    for prior_detection, prediction in zip(
                        self._presented_detections,
                        self._presented_freshness_predictions,
                    ):
                        prior_predictions.setdefault(prior_detection.label, []).append(prediction)
                    predictions = []
                    for detection in scaled:
                        matching_predictions = prior_predictions.get(detection.label, [])
                        predictions.append(matching_predictions.pop(0) if matching_predictions else None)
                    self._detections = scaled
                    self._freshness_predictions = predictions
                    self._detection_shape = (height, width)
                    if scaled:
                        self._presented_detections = list(scaled)
                        self._presented_freshness_predictions = list(predictions)
                        self._presented_remote_detections = []
                        self._presented_image_data = thumbnail
                    self._detection_error = None
            except Exception as exc:  # noqa: BLE001 - keep video flowing and show the inference failure
                log.exception("Live food detection failed")
                with self._lock:
                    self._detection_error = f"Food detection failed: {exc}"

        if self._freshness_enabled and now >= self._next_freshness:
            self._next_freshness = now + _FRESHNESS_INTERVAL
            with self._lock:
                detections_to_classify = list(self._detections)
            predictions: list[Optional[FreshnessPrediction]] = []
            freshness_error = None
            for detection in detections_to_classify:
                try:
                    if detection.category != "Fruits & Vegetables" or not self._classifier.supports_food(detection.label):
                        predictions.append(None)
                        continue
                    crop = crop_box(image, detection.box)
                    predictions.append(self._classifier.predict(bgr_to_rgb(crop)))
                except Exception as exc:  # noqa: BLE001 - display the model failure rather than hiding it
                    log.exception("Live freshness classification failed for %s", detection.label)
                    predictions.append(None)
                    freshness_error = f"Live freshness assessment failed: {exc}"
            with self._lock:
                if detections_to_classify == self._detections:
                    self._freshness_predictions = predictions
                    if detections_to_classify == self._presented_detections:
                        self._presented_freshness_predictions = list(predictions)
                self._freshness_error = freshness_error

        if self._ocr_enabled and now >= self._next_ocr:
            self._next_ocr = now + _OCR_INTERVAL
            try:
                y1, y2 = round(height * 0.1), round(height * 0.9)
                x1, x2 = round(width * 0.1), round(width * 0.9)
                label_area = image[y1:y2, x1:x2]
                gray = cv2.cvtColor(label_area, cv2.COLOR_BGR2GRAY)
                lines = []
                variants = ocr_variants(gray)
                for variant_name, variant in variants:
                    lines.extend(self._ocr_engine.read_lines(variant, psm=11))
                    if variant_name == "contrast":
                        lines.extend(self._ocr_engine.read_lines(variant, psm=6))
                        lines.extend(self._ocr_engine.read_lines(variant, psm=7))
                unique_lines = list(dict.fromkeys(line.text for line in lines if line.text.strip()))
                text = " | ".join(unique_lines)[:180]
                dates = parse_lines([(line.text, line.confidence) for line in lines])
                date_image_data = thumbnail_jpeg(image) if dates else None
                with self._lock:
                    self._latest_text = text
                    self._latest_dates = dates
                    if dates:
                        self._latest_date_image_data = date_image_data
                    self._latest_read_at = time.monotonic()
                    self._ocr_error = None
            except Exception as exc:  # noqa: BLE001 - show OCR failure without stopping live detection
                log.exception("Live package-text recognition failed")
                with self._lock:
                    self._ocr_error = f"Package text reading failed: {exc}"
                    self._latest_text = ""
                    self._latest_dates = []
                    self._latest_read_at = 0.0

        with self._lock:
            local_detections = list(self._detections)
            local_predictions = list(self._freshness_predictions)
            detections = list(local_detections)
            if time.monotonic() - self._remote_read_at <= _REMOTE_RESULT_MAX_AGE:
                remote_detections = list(self._remote_detections)
                remote_shape = self._remote_shape
                detections.extend(remote_detections)
            else:
                remote_detections = []
                remote_shape = None
            shape = self._detection_shape

        if shape and shape != (height, width):
            local_detections = _scale_detections(local_detections, shape, (height, width))
        if remote_shape and remote_shape != (height, width):
            remote_detections = _scale_detections(remote_detections, remote_shape, (height, width))
        detections = merge_detections(local_detections, remote_detections)

        output = image.copy()
        if detections:
            predictions_by_detection = {
                (det.source, det.label, det.box): prediction
                for det, prediction in zip(local_detections, local_predictions)
            }
            annotated = []
            for detection in detections:
                prediction = predictions_by_detection.get((detection.source, detection.label, detection.box))
                label = detection.label
                if detection.quality:
                    label = f"{label} - dataset label {detection.quality}"
                elif detection.source == "roboflow":
                    label = f"{label} - cloud"
                if prediction is None:
                    pass
                elif prediction.uncertain:
                    label = f"{detection.label} (freshness unsure)"
                else:
                    label = f"{detection.label} - {prediction.display}"
                annotated.append(Detection(label, detection.confidence, detection.box, detection.category))
            # draw_detections returns RGB; convert back for the WebRTC frame.
            output = cv2.cvtColor(draw_detections(output, annotated), cv2.COLOR_RGB2BGR)

        return VideoFrame.from_ndarray(output, format="bgr24")

    def _poll_remote(self, now: float) -> None:
        with self._lock:
            future = self._remote_future
        if future is None or not future.done():
            return
        try:
            remote_detections = future.result()
            remote_error = None
        except Exception as exc:  # noqa: BLE001 - surfaced in the live video status
            log.exception("Roboflow live inference failed")
            remote_detections = []
            remote_error = f"Roboflow inference failed: {exc}"
        with self._lock:
            if self._remote_future is not future:
                return
            self._remote_future = None
            self._remote_detections = [
                Detection(
                    detection.label,
                    detection.confidence,
                    tuple(round(value / self._remote_scale) for value in detection.box),
                    detection.category,
                    detection.source,
                    detection.quality,
                )
                for detection in remote_detections
            ]
            if self._remote_detections:
                self._presented_detections = []
                self._presented_freshness_predictions = []
                self._presented_remote_detections = list(self._remote_detections)
                self._presented_image_data = self._remote_image_data
            self._remote_read_at = now
            self._remote_error = remote_error

    def _schedule_remote(self, image: np.ndarray, now: float) -> None:
        if self._remote_executor is None or self._roboflow_detector is None:
            return
        with self._lock:
            if self._remote_future is not None or now < self._next_remote:
                return
            height, width = image.shape[:2]
            scale = min(1.0, _MAX_DETECTION_SIDE / max(height, width))
            resized = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
            self._remote_shape = (height, width)
            self._remote_scale = scale
            self._remote_image_data = thumbnail_jpeg(image)
            self._remote_future = self._remote_executor.submit(self._roboflow_detector.detect, resized)
            self._next_remote = now + _ROBOFLOW_INTERVAL


def _scale_detections(
    detections: list[Detection],
    from_shape: tuple[int, int],
    to_shape: tuple[int, int],
) -> list[Detection]:
    from_height, from_width = from_shape
    to_height, to_width = to_shape
    return [
        Detection(
            detection.label,
            detection.confidence,
            (
                round(detection.box[0] * to_width / from_width),
                round(detection.box[1] * to_height / from_height),
                round(detection.box[2] * to_width / from_width),
                round(detection.box[3] * to_height / from_height),
            ),
            detection.category,
            detection.source,
            detection.quality,
        )
        for detection in detections
    ]
