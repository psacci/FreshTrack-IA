"""Read printed date information from packaging photos.

Design rules (they are tested):
* The date *type* comes only from the printed label next to it. A manufacturing
  or packing date is never turned into an expiration date.
* Ambiguous numeric dates (05/06/2027) are never guessed. Both readings are
  returned and the user picks one.
* If no label is found the type is "unknown" and the user must choose it.
* If OCR fails we say so. We never invent a date.
"""
from __future__ import annotations

import calendar
import logging
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Sequence

import numpy as np

from config import settings
from modules.image_processor import ocr_variants
from modules.status import ModelStatus

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Label keywords -> date type
# --------------------------------------------------------------------------
_KEYWORD_PATTERNS = [
    ("best_before", r"BEST[\s\-]*BEFORE(?:[\s\-]*END)?|BEST[\s\-]*BY|\bBBE\b"),
    ("use_by", r"USE[\s\-]*BY|USE[\s\-]*BEFORE"),
    ("sell_by", r"SELL[\s\-]*BY"),
    ("expiration", r"\bEXP(?:IRY|IRATION|IRES|IRED)?(?:[\s\-]*DATE)?\b"),
    ("manufacturing", r"\bMFG\b|\bMFD\b|MANUFACTUR(?:ED|ING|E)(?:[\s\-]*(?:ON|DATE))?|DATE[\s\-]*OF[\s\-]*MANUFACTURE|\bDOM\b"),
    ("packing", r"PACK(?:ED|ING|AGED)(?:[\s\-]*(?:ON|DATE))?|\bPKD\b"),
]
_KEYWORDS = [(t, re.compile(p)) for t, p in _KEYWORD_PATTERNS]

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_MONTH = (
    r"(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|"
    r"AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)"
)

_RE_NUMERIC = re.compile(r"(?<![\d])(\d{1,4})\s*([./\-])\s*(\d{1,2})\s*\2\s*(\d{2,4})(?!\d)")
_RE_D_MON_Y = re.compile(r"(?<!\d)(\d{1,2})\s*[\-/.\s]?\s*" + _MONTH + r"\.?\s*[\-/.,\s]?\s*(\d{4}|\d{2})(?!\d)")
_RE_MON_D_Y = re.compile(_MONTH + r"\.?\s*(\d{1,2})(?:ST|ND|RD|TH)?\s*,?\s*(\d{4})(?!\d)")
_RE_MON_Y = re.compile(_MONTH + r"\.?\s*[\-/.,\s]?\s*(\d{4})(?!\d)")
_RE_NUM_MONTH_Y = re.compile(r"(?<![\d/.\-])(\d{1,2})\s*([/.\-])\s*(\d{4})(?!\d)")
_RE_COMPACT = re.compile(r"(?<!\d)(\d{8})(?!\d)")

YEAR_MIN, YEAR_MAX = 2000, 2099


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------
@dataclass
class DateCandidate:
    label_type: str                       # one of settings.DATE_TYPES or "unknown"
    label_text: Optional[str]             # the printed label, e.g. "BEST BEFORE"
    raw_text: str                         # the date text as read
    line_text: str
    candidates: list[date]                # 1 = clear, 2 = ambiguous, 0 = unreadable
    precision: str = "day"                # "day" or "month"
    ocr_confidence: Optional[float] = None
    note: str = ""

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    @property
    def parsed(self) -> Optional[date]:
        return self.candidates[0] if len(self.candidates) == 1 else None


@dataclass
class OCRLine:
    text: str
    confidence: Optional[float]


@dataclass
class OCRResult:
    status: str                           # ok | no_dates | no_text | engine_unavailable
    message: str
    raw_text: str = ""
    candidates: list[DateCandidate] = field(default_factory=list)
    engine: str = ""


# --------------------------------------------------------------------------
# Text normalisation and parsing (pure functions, easy to test)
# --------------------------------------------------------------------------
def _normalize(text: str) -> str:
    s = text.upper().replace("\u2013", "-").replace("\u2014", "-").replace("|", "1")
    # Common OCR slips inside numbers: O->0, I/L->1 when surrounded by digits or separators.
    s = re.sub(r"(?<=[\d/.\-])O(?=[\d/.\-])", "0", s)
    s = re.sub(r"(?<=[\d/.\-])[IL](?=[\d/.\-])", "1", s)
    return s


def _valid_date(y: int, m: int, d: int) -> Optional[date]:
    if not (YEAR_MIN <= y <= YEAR_MAX):
        return None
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _year(text: str) -> Optional[int]:
    if len(text) == 4:
        return int(text)
    if len(text) == 2:
        return 2000 + int(text)
    return None


def _end_of_month(y: int, m: int) -> Optional[date]:
    if not (YEAR_MIN <= y <= YEAR_MAX and 1 <= m <= 12):
        return None
    return date(y, m, calendar.monthrange(y, m)[1])


def _two_readings(a: int, b: int, y: int) -> list[date]:
    """Day-first and month-first readings of a/b/y, unique and valid."""
    found = []
    for candidate in (_valid_date(y, b, a), _valid_date(y, a, b)):  # DMY, then MDY
        if candidate and candidate not in found:
            found.append(candidate)
    return found


@dataclass
class _RawDate:
    start: int
    end: int
    text: str
    candidates: list[date]
    precision: str
    note: str
    needs_label: bool = False             # compact numbers only count next to a label


def _find_dates(line: str) -> list[_RawDate]:
    taken: list[tuple[int, int]] = []
    found: list[_RawDate] = []

    def free(span: tuple[int, int]) -> bool:
        return all(span[1] <= s or span[0] >= e for s, e in taken)

    def add(m: re.Match, cands: list[date], precision: str = "day", note: str = "", needs_label: bool = False) -> None:
        taken.append(m.span())
        found.append(_RawDate(m.start(), m.end(), m.group(0).strip(), cands, precision, note, needs_label))

    for m in _RE_NUMERIC.finditer(line):
        if not free(m.span()):
            continue
        g1, g3, g4 = m.group(1), m.group(3), m.group(4)
        if len(g1) == 4 and len(g4) <= 2:                       # YYYY-MM-DD
            c = _valid_date(int(g1), int(g3), int(g4))
            add(m, [c] if c else [])
        elif len(g1) <= 2:                                      # DD/MM/YY(YY) or MM/DD/YY(YY)
            y = _year(g4)
            cands = _two_readings(int(g1), int(g3), y) if y else []
            note = "Two-digit year read as 20xx." if len(g4) == 2 and cands else ""
            add(m, cands, note=note)
        else:
            add(m, [])

    for m in _RE_D_MON_Y.finditer(line):
        if free(m.span()):
            y = _year(m.group(3))
            c = _valid_date(y, _MONTHS[m.group(2)[:3]], int(m.group(1))) if y else None
            note = "Two-digit year read as 20xx." if len(m.group(3)) == 2 and c else ""
            add(m, [c] if c else [], note=note)

    for m in _RE_MON_D_Y.finditer(line):
        if free(m.span()):
            c = _valid_date(int(m.group(3)), _MONTHS[m.group(1)[:3]], int(m.group(2)))
            add(m, [c] if c else [])

    for m in _RE_MON_Y.finditer(line):
        if free(m.span()):
            c = _end_of_month(int(m.group(2)), _MONTHS[m.group(1)[:3]])
            add(m, [c] if c else [], "month", "Only a month and year are printed, so the last day of that month is used. Please confirm.")

    for m in _RE_NUM_MONTH_Y.finditer(line):
        if free(m.span()):
            c = _end_of_month(int(m.group(3)), int(m.group(1)))
            add(m, [c] if c else [], "month", "Only a month and year are printed, so the last day of that month is used. Please confirm.")

    for m in _RE_COMPACT.finditer(line):
        if not free(m.span()):
            continue
        s = m.group(1)
        cands = []
        ymd = _valid_date(int(s[:4]), int(s[4:6]), int(s[6:]))
        if ymd and s[:2] in ("20",):
            cands = [ymd]
        else:
            cands = _two_readings(int(s[:2]), int(s[2:4]), int(s[4:]))
        add(m, cands, needs_label=True)

    found.sort(key=lambda r: r.start)
    return found


@dataclass
class _Keyword:
    start: int
    end: int
    date_type: str
    text: str


def _find_keywords(line: str) -> list[_Keyword]:
    hits: list[_Keyword] = []
    for date_type, pattern in _KEYWORDS:
        for m in pattern.finditer(line):
            hits.append(_Keyword(m.start(), m.end(), date_type, m.group(0).strip()))
    # Drop overlapping hits, keeping the longer one.
    hits.sort(key=lambda k: (k.start, -(k.end - k.start)))
    kept: list[_Keyword] = []
    for hit in hits:
        if kept and hit.start < kept[-1].end:
            continue
        kept.append(hit)
    return kept


def parse_lines(lines: Sequence[tuple[str, Optional[float]]]) -> list[DateCandidate]:
    """Find dates in OCR lines and attach the label that belongs to each."""
    results: list[DateCandidate] = []
    carried: Optional[_Keyword] = None   # label at the end of the previous line

    for text, conf in lines:
        line = _normalize(text)
        keywords = _find_keywords(line)
        dates = _find_dates(line)

        for d in dates:
            owner: Optional[_Keyword] = None
            before = [k for k in keywords if k.end <= d.start]
            if before:
                k = max(before, key=lambda k: k.end)
                if not any(o is not d and o.start >= k.end and o.end <= d.start for o in dates):
                    owner = k
            if owner is None:
                after = [k for k in keywords if k.start >= d.end and k.start - d.end <= 15]
                if after:
                    owner = min(after, key=lambda k: k.start)
            if owner is None and carried is not None and not keywords:
                owner = carried

            if d.needs_label and owner is None:
                continue  # an 8-digit number with no label is more likely a batch code
            if not d.candidates and not owner:
                continue  # unreadable and unlabelled: just noise

            results.append(DateCandidate(
                label_type=owner.date_type if owner else "unknown",
                label_text=owner.text if owner else None,
                raw_text=d.text, line_text=text.strip(), candidates=d.candidates,
                precision=d.precision, ocr_confidence=conf, note=d.note,
            ))

        if keywords and not dates:
            carried = keywords[-1]
        else:
            carried = None

    return _dedupe(results)


def _dedupe(items: list[DateCandidate]) -> list[DateCandidate]:
    best: dict[tuple, DateCandidate] = {}
    for item in items:
        key = (item.label_type, tuple(item.candidates), re.sub(r"\s+", "", item.raw_text))
        old = best.get(key)
        if old is None or (item.ocr_confidence or 0) > (old.ocr_confidence or 0):
            best[key] = item
    return list(best.values())


def _detail_regions(image: np.ndarray) -> list[np.ndarray]:
    """Return overlapping image regions so small package print can be upscaled for OCR."""
    height, width = image.shape[:2]
    if max(height, width) < 1000 or min(height, width) < 600:
        return []

    region_height, region_width = round(height * 0.6), round(width * 0.6)
    y_starts = (0, height - region_height)
    x_starts = (0, width - region_width)
    return [
        image[y:y + region_height, x:x + region_width]
        for y in y_starts
        for x in x_starts
    ]


def extract_from_text(text: str) -> list[DateCandidate]:
    """Convenience wrapper: parse a block of plain text (no confidence values)."""
    return parse_lines([(line, None) for line in text.splitlines() if line.strip()])


# --------------------------------------------------------------------------
# OCR engines (replaceable)
# --------------------------------------------------------------------------
class OCREngine(ABC):
    name = "ocr"

    @abstractmethod
    def status(self) -> ModelStatus: ...

    @abstractmethod
    def read_lines(self, gray: np.ndarray, psm: int = 6) -> list[OCRLine]: ...


class TesseractEngine(OCREngine):
    """Tesseract via pytesseract. The Tesseract program itself must be installed."""

    name = "Tesseract"
    _MAC_PATHS = ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract")

    def __init__(self, cmd: Optional[str] = None) -> None:
        import os
        self.cmd = cmd or os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
        if not self.cmd:
            self.cmd = next((p for p in self._MAC_PATHS if shutil.which(p) or __import__("os").path.exists(p)), None)

    def status(self) -> ModelStatus:
        if not self.cmd:
            return ModelStatus(False, "Tesseract is not installed. On macOS run: brew install tesseract")
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = self.cmd
            version = pytesseract.get_tesseract_version()
            return ModelStatus(True, f"Tesseract {version} ready")
        except Exception as exc:  # noqa: BLE001 - surface any setup problem to the UI
            return ModelStatus(False, f"Tesseract could not start: {exc}")

    def read_lines(self, gray: np.ndarray, psm: int = 6) -> list[OCRLine]:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = self.cmd
        data = pytesseract.image_to_data(
            gray, lang="eng", config=f"--oem 3 --psm {psm}", output_type=pytesseract.Output.DICT
        )
        grouped: dict[tuple, list[tuple[str, float]]] = {}
        for i, word in enumerate(data["text"]):
            word = (word or "").strip()
            if not word:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            grouped.setdefault(key, []).append((word, float(data["conf"][i])))
        lines = []
        for key in sorted(grouped):
            words = grouped[key]
            confs = [c for _, c in words if c >= 0]
            lines.append(OCRLine(" ".join(w for w, _ in words), sum(confs) / len(confs) if confs else None))
        return lines


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def read_expiration_dates(
    image_bgr: np.ndarray,
    engine: Optional[OCREngine] = None,
    psms: Sequence[int] = (6, 11),
) -> OCRResult:
    """Run OCR on a packaging photo and return labelled date candidates."""
    engine = engine or TesseractEngine()
    state = engine.status()
    if not state.available:
        return OCRResult("engine_unavailable", state.message, engine=engine.name)

    merged: list[DateCandidate] = []
    text_blocks: list[str] = []
    any_text = False
    for variant_name, gray in ocr_variants(image_bgr):
        for psm in psms:
            try:
                lines = engine.read_lines(gray, psm)
            except Exception as exc:  # noqa: BLE001
                log.warning("OCR failed on %s/psm%s: %s", variant_name, psm, exc)
                continue
            if lines:
                any_text = True
                text_blocks.append("\n".join(l.text for l in lines))
            merged.extend(parse_lines([(l.text, l.confidence) for l in lines]))

    candidates = _dedupe(merged)
    if not candidates:
        for region in _detail_regions(image_bgr):
            for variant_name, gray in ocr_variants(region):
                for psm in psms:
                    try:
                        lines = engine.read_lines(gray, psm)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("OCR failed on detail %s/psm%s: %s", variant_name, psm, exc)
                        continue
                    if lines:
                        any_text = True
                        text_blocks.append("\n".join(line.text for line in lines))
                    merged.extend(parse_lines([(line.text, line.confidence) for line in lines]))
        candidates = _dedupe(merged)

    candidates.sort(key=lambda c: -(c.ocr_confidence or 0))
    if not any_text:
        return OCRResult("no_text", "No readable text found. Try a sharper, closer photo with even light.", engine=engine.name)
    raw_text = "\n".join(dict.fromkeys(text_blocks))
    if not candidates:
        return OCRResult(
            "no_dates",
            "Text was detected, but no recognizable date was read. Try a closer photo of the printed date, or enter it by hand.",
            raw_text=raw_text,
            engine=engine.name,
        )
    return OCRResult("ok", f"Found {len(candidates)} possible date(s). Please check them.", raw_text, candidates, engine.name)
