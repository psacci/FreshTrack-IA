"""Image helpers: loading, cropping, drawing boxes, and OCR pre-processing."""
from __future__ import annotations

import io
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 1800

# Box colours (BGR) match the app's palette.
_LEAF = (97, 148, 46)
_INK = (42, 48, 20)


def load_image(data: bytes, max_side: int = MAX_SIDE) -> np.ndarray:
    """Decode uploaded bytes into a BGR array, honouring phone-camera rotation."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def thumbnail_jpeg(img: np.ndarray, max_side: int = 320) -> bytes:
    """Encode a small JPEG preview for an inventory item."""
    height, width = img.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    preview = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    success, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not success:
        raise RuntimeError("Could not encode the food photo thumbnail.")
    return encoded.tobytes()


def crop_box(img: np.ndarray, box: Sequence[int], pad: float = 0.04) -> np.ndarray:
    """Crop a bounding box (x1, y1, x2, y2) with a little padding."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    return img[y1:y2, x1:x2]


def draw_detections(img: np.ndarray, detections: Iterable) -> np.ndarray:
    """Return an RGB copy of the image with numbered boxes and labels."""
    out = img.copy()
    thickness = max(2, round(max(out.shape[:2]) / 400))
    scale = max(0.6, max(out.shape[:2]) / 1200)
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det.box
        cv2.rectangle(out, (x1, y1), (x2, y2), _LEAF, thickness)
        text = f"{idx}. {det.label} {det.confidence * 100:.0f}%"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        top = max(0, y1 - th - base - 6)
        cv2.rectangle(out, (x1, top), (x1 + tw + 10, top + th + base + 6), _LEAF, -1)
        cv2.putText(out, text, (x1 + 5, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return bgr_to_rgb(out)


def ocr_variants(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Build a few cleaned-up grayscale versions of a packaging photo for OCR.

    Printed dates are small and often low-contrast, so we upscale, boost local
    contrast, reduce noise and try both soft and hard thresholding. The reader
    runs OCR on each and keeps what it finds.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    longest = max(gray.shape[:2])
    if longest < 1600:
        factor = 1600 / longest
        gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    if gray.mean() < 110:  # light text on a dark label
        gray = cv2.bitwise_not(gray)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    boosted = clahe.apply(gray)
    denoised = cv2.medianBlur(boosted, 3)

    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 12
    )
    return [("contrast", denoised), ("otsu", otsu), ("adaptive", adaptive)]
