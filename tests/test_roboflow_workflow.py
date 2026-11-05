import json
from urllib.error import HTTPError
from unittest.mock import patch

import numpy as np
import pytest

from modules.food_detector import Detection
from modules.roboflow_workflow import RoboflowWorkflow, merge_detections


class FakeResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.value).encode()


def test_sends_base64_image_and_workflow_parameters_with_bearer_auth():
    result = [{"predictions": {"predictions": [
        {"class": "cucumber", "confidence": 0.91, "x": 50, "y": 40, "width": 40, "height": 20},
        {"class": "unmapped", "confidence": 0.99, "x": 50, "y": 40, "width": 40, "height": 20},
    ]}}]
    client = RoboflowWorkflow(
        "test-secret",
        workspace="my-workspace",
        workflow_id="food-flow",
        parameters_json='{"classes":"egg, cucumber"}',
    )
    with patch("modules.roboflow_workflow.urlopen", return_value=FakeResponse(result)) as open_url:
        detections = client.detect_bytes(b"jpeg-bytes", (100, 200))

    request = open_url.call_args.args[0]
    payload = json.loads(request.data)
    assert request.full_url.endswith("/infer/workflows/my-workspace/food-flow")
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert payload["inputs"]["image"] == {
        "type": "base64",
        "value": "anBlZy1ieXRlcw==",
    }
    assert payload["inputs"]["classes"] == "egg, cucumber"
    assert [(d.label, d.confidence, d.box, d.source) for d in detections] == [
        ("cucumber", 0.91, (30, 30, 70, 50), "roboflow"),
    ]


def test_parses_normalized_boxes_and_clips_to_image_bounds():
    client = RoboflowWorkflow("test-secret")
    response = FakeResponse([
        {"predictions": [
            {"class": "egg", "confidence": 0.8, "x": 0.95, "y": 0.5, "width": 0.3, "height": 0.4}
        ]}
    ])
    with patch("modules.roboflow_workflow.urlopen", return_value=response):
        detections = client.detect_bytes(b"image", (100, 200))
    assert detections[0].box == (160, 30, 200, 70)


def test_requires_key_and_valid_parameter_json():
    with pytest.raises(ValueError, match="ROBOFLOW_API_KEY"):
        RoboflowWorkflow("")
    with pytest.raises(ValueError, match="JSON object"):
        RoboflowWorkflow("test-secret", parameters_json="[]")


def test_http_errors_do_not_include_credentials():
    client = RoboflowWorkflow("test-secret")
    error = HTTPError("https://serverless.roboflow.com", 401, "unauthorized", {}, None)
    with patch("modules.roboflow_workflow.urlopen", side_effect=error):
        with pytest.raises(RuntimeError, match="HTTP 401") as raised:
            client.detect_bytes(b"image", (10, 10))
    assert "test-secret" not in str(raised.value)


def test_merge_keeps_the_highest_confidence_duplicate_and_new_classes():
    local = [Detection("apple", 0.8, (0, 0, 50, 50), "Fruits & Vegetables")]
    remote = [
        Detection("apple", 0.9, (2, 2, 48, 48), "Fruits & Vegetables", source="roboflow"),
        Detection("egg", 0.85, (60, 0, 100, 40), "Dairy", source="roboflow"),
    ]
    merged = merge_detections(local, remote)
    assert len(merged) == 2
    assert merged[0].label == "apple" and merged[0].source == "roboflow"
    assert merged[1].label == "egg"
