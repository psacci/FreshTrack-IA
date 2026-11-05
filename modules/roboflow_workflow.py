"""Opt-in client for Roboflow Workflows image inference."""
from __future__ import annotations

import base64
import json
import math
from dataclasses import replace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import cv2
import numpy as np

from config import settings
from modules.food_detector import Detection, filter_detections


class RoboflowWorkflow:
    def __init__(
        self,
        api_key: str,
        workspace: str = settings.ROBOFLOW_WORKSPACE,
        workflow_id: str = settings.ROBOFLOW_WORKFLOW_ID,
        api_url: str = settings.ROBOFLOW_API_URL,
        image_input: str = settings.ROBOFLOW_IMAGE_INPUT,
        parameters_json: str = settings.ROBOFLOW_WORKFLOW_PARAMETERS,
        timeout: float = 15.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Set ROBOFLOW_API_KEY in your local environment before using Roboflow.")
        if not workspace.strip() or not workflow_id.strip() or not image_input.strip():
            raise ValueError("Roboflow workspace, workflow ID, and image input name must not be empty.")
        self._api_key = api_key.strip()
        self.workspace = workspace.strip()
        self.workflow_id = workflow_id.strip()
        self.api_url = api_url.rstrip("/")
        self.image_input = image_input.strip()
        self.timeout = timeout
        try:
            parameters = json.loads(parameters_json) if parameters_json.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError("ROBOFLOW_WORKFLOW_PARAMETERS must be a JSON object.") from exc
        if not isinstance(parameters, dict):
            raise ValueError("ROBOFLOW_WORKFLOW_PARAMETERS must be a JSON object.")
        self.parameters = parameters

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        ok, encoded = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise RuntimeError("Could not encode the image for Roboflow inference.")
        return self.detect_bytes(encoded.tobytes(), image_bgr.shape[:2])

    def detect_bytes(self, image_bytes: bytes, image_shape: tuple[int, int]) -> list[Detection]:
        if not image_bytes:
            raise ValueError("Cannot send an empty image to Roboflow.")
        height, width = image_shape
        if height <= 0 or width <= 0:
            raise ValueError("Image dimensions must be positive.")

        inputs: dict[str, Any] = {
            self.image_input: {
                "type": "base64",
                "value": base64.b64encode(image_bytes).decode("ascii"),
            }
        }
        if self.image_input in self.parameters:
            raise ValueError("ROBOFLOW_WORKFLOW_PARAMETERS must not overwrite the image input.")
        inputs.update(self.parameters)
        payload = json.dumps({"inputs": inputs}).encode("utf-8")
        endpoint = (
            f"{self.api_url}/infer/workflows/"
            f"{quote(self.workspace, safe='')}/{quote(self.workflow_id, safe='')}"
        )
        request = Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Roboflow inference returned HTTP {exc.code}. Check the workflow and API key.") from exc
        except URLError as exc:
            raise RuntimeError("Could not connect to Roboflow. Check your internet connection and API URL.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Roboflow returned an invalid JSON response.") from exc

        raw = []
        for prediction in _prediction_records(result):
            label = _first_string(prediction, ("class", "label", "class_name", "name"))
            confidence = _first_number(prediction, ("confidence", "score"))
            box = _box(prediction, width, height)
            if label and confidence is not None and box is not None:
                raw.append((label, confidence, box))
        return [
            replace(detection, source="roboflow")
            for detection in filter_detections(raw, min_conf=0.0)
        ]


def merge_detections(
    local: list[Detection],
    remote: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Combine providers without showing the same class/box twice."""
    merged = list(local)
    for candidate in remote:
        duplicate_index = next(
            (
                index for index, current in enumerate(merged)
                if current.label == candidate.label and _iou(current.box, candidate.box) >= iou_threshold
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
        elif candidate.confidence > merged[duplicate_index].confidence:
            merged[duplicate_index] = candidate
    return sorted(merged, key=lambda detection: -detection.confidence)


def _prediction_records(value: Any):
    """Find object predictions in the nested output shapes used by Workflows."""
    if isinstance(value, dict):
        has_box = (
            all(key in value for key in ("x1", "y1", "x2", "y2"))
            or all(key in value for key in ("x_min", "y_min", "x_max", "y_max"))
            or all(key in value for key in ("x", "y", "width", "height"))
            or isinstance(value.get("box"), (dict, list, tuple))
            or isinstance(value.get("bounding_box"), (dict, list, tuple))
            or isinstance(value.get("bbox"), (dict, list, tuple))
        )
        if _first_string(value, ("class", "label", "class_name", "name")) and has_box:
            yield value
            return
        for child in value.values():
            yield from _prediction_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _prediction_records(child)


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def _first_string(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_number(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _box(record: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float] | None:
    bounds = record.get("box") or record.get("bounding_box") or record.get("bbox")
    if isinstance(bounds, dict):
        record = {**record, **bounds}

    if all(key in record for key in ("x1", "y1", "x2", "y2")):
        values = tuple(record[key] for key in ("x1", "y1", "x2", "y2"))
        center_format = False
    elif all(key in record for key in ("x_min", "y_min", "x_max", "y_max")):
        values = tuple(record[key] for key in ("x_min", "y_min", "x_max", "y_max"))
        center_format = False
    elif all(key in record for key in ("x", "y", "width", "height")):
        values = tuple(record[key] for key in ("x", "y", "width", "height"))
        center_format = True
    elif isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        values = tuple(bounds)
        center_format = False
    else:
        return None

    try:
        x1, y1, x2, y2 = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None

    normalized = max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1
    if center_format:
        x1, y1, x2, y2 = x1 - x2 / 2, y1 - y2 / 2, x1 + x2 / 2, y1 + y2 / 2
    if normalized:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1, x2 = min(max(x1, 0), width), min(max(x2, 0), width)
    y1, y2 = min(max(y1, 0), height), min(max(y2, 0), height)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
