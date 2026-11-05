import shutil
from datetime import date

import cv2
import numpy as np
import pytest

from modules.expiration_date_reader import (
    OCREngine, OCRLine, extract_from_text, parse_lines, read_expiration_dates,
)
from modules.status import ModelStatus


def one(text):
    found = extract_from_text(text)
    assert len(found) == 1, found
    return found[0]


# ------------------------------------------------------------ label types
@pytest.mark.parametrize("text,expected", [
    ("EXP 05 JUN 2027", "expiration"),
    ("EXPIRY DATE: 05 JUN 2027", "expiration"),
    ("EXPIRATION DATE 2027-06-05", "expiration"),
    ("BEST BEFORE 05 JUN 2027", "best_before"),
    ("BEST BY JUN 5, 2027", "best_before"),
    ("USE BY 05 JUN 2027", "use_by"),
    ("SELL BY 05-JUN-2027", "sell_by"),
    ("MFG 05 JUN 2026", "manufacturing"),
    ("MANUFACTURED ON 05 JUN 2026", "manufacturing"),
    ("DATE OF MANUFACTURE 05 JUN 2026", "manufacturing"),
    ("PACKED ON 05 JUN 2026", "packing"),
])
def test_label_classification(text, expected):
    c = one(text)
    assert c.label_type == expected
    assert c.parsed in (date(2027, 6, 5), date(2026, 6, 5))
    assert not c.ambiguous


def test_manufacturing_date_is_never_labelled_expiration():
    c = one("MFG 01 JAN 2026")
    assert c.label_type == "manufacturing" and c.label_type != "expiration"


def test_two_dates_on_one_line_get_their_own_labels():
    found = extract_from_text("MFG 01 JAN 2026 EXP 01 JAN 2027")
    by_type = {c.label_type: c.parsed for c in found}
    assert by_type == {"manufacturing": date(2026, 1, 1), "expiration": date(2027, 1, 1)}


def test_label_on_previous_line_is_used():
    c = one("BEST BEFORE END\n05 JUN 2027")
    assert c.label_type == "best_before" and c.parsed == date(2027, 6, 5)


def test_date_without_a_label_is_unknown_type():
    c = one("05 JUN 2027")
    assert c.label_type == "unknown" and c.label_text is None


# ---------------------------------------------------------- date formats
@pytest.mark.parametrize("text", [
    "EXP 2027/06/05", "EXP 2027-06-05", "EXP 5 JUN 2027", "EXP 05JUN2027",
    "EXP JUN 5 2027", "EXP 05.JUNE.2027", "EXP 20270605", "EXP 13/06/2027 ",
])
def test_unambiguous_formats(text):
    c = one(text)
    assert not c.ambiguous and c.parsed is not None


def test_day_over_12_resolves_the_order():
    assert one("EXP 25/06/2027").parsed == date(2027, 6, 25)
    assert one("EXP 06/25/2027").parsed == date(2027, 6, 25)


def test_ambiguous_date_is_never_guessed():
    c = one("EXP 05/06/2027")
    assert c.ambiguous and c.parsed is None
    assert set(c.candidates) == {date(2027, 6, 5), date(2027, 5, 6)}
    assert c.raw_text == "05/06/2027"


def test_same_day_and_month_is_not_ambiguous():
    assert one("EXP 05/05/2027").parsed == date(2027, 5, 5)


def test_month_year_only_is_flagged():
    c = one("BEST BEFORE END DEC 2027")
    assert c.precision == "month" and c.parsed == date(2027, 12, 31) and "confirm" in c.note.lower()


def test_ocr_digit_slips_are_repaired_inside_numbers():
    assert one("EXP 2O27-O6-O5").parsed == date(2027, 6, 5)


# ------------------------------------------------------------- bad input
@pytest.mark.parametrize("text", [
    "", "   ", "LOT 4829 B7", "NET WT 500 G", "32/13/2027 ", "the quick brown fox",
])
def test_garbage_gives_no_dates(text):
    assert extract_from_text(text) == []


def test_invalid_calendar_date_is_not_invented():
    found = extract_from_text("EXP 31/02/2027")
    assert all(not c.candidates for c in found)


def test_unlabelled_eight_digit_number_is_treated_as_batch_code():
    assert extract_from_text("BATCH 12062027") == []


def test_confidence_is_carried_from_the_ocr_line():
    found = parse_lines([("EXP 05 JUN 2027", 87.5)])
    assert found[0].ocr_confidence == 87.5


# ----------------------------------------------------- engine plumbing
class _Down(OCREngine):
    name = "down"
    def status(self): return ModelStatus(False, "not installed")
    def read_lines(self, gray, psm=6): raise AssertionError("must not run")


class _Fake(OCREngine):
    name = "fake"
    def __init__(self, lines): self.lines = lines
    def status(self): return ModelStatus(True, "ok")
    def read_lines(self, gray, psm=6): return self.lines


BLANK = np.full((200, 400, 3), 255, np.uint8)


def test_missing_engine_is_reported_not_crashed():
    r = read_expiration_dates(BLANK, _Down())
    assert r.status == "engine_unavailable" and r.candidates == []


def test_no_text_and_no_dates_statuses():
    assert read_expiration_dates(BLANK, _Fake([])).status == "no_text"
    assert read_expiration_dates(BLANK, _Fake([OCRLine("HELLO WORLD", 90)])).status == "no_dates"


def test_engine_results_flow_through_parser():
    r = read_expiration_dates(BLANK, _Fake([OCRLine("USE BY 05 JUN 2027", 91)]))
    assert r.status == "ok" and r.candidates[0].label_type == "use_by"


def test_detail_crops_retry_small_package_dates_when_full_image_misses():
    class _CropOnlyDate(_Fake):
        def __init__(self):
            self.calls = 0

        def read_lines(self, gray, psm=6):
            self.calls += 1
            text = "BRAND TEXT" if self.calls <= 6 else "EXP 25 JUN 2027"
            return [OCRLine(text, 90)]

    image = np.full((1000, 1000, 3), 255, np.uint8)
    result = read_expiration_dates(image, _CropOnlyDate())

    assert result.status == "ok"
    assert result.candidates[0].label_type == "expiration"
    assert result.candidates[0].parsed == date(2027, 6, 25)


# ------------------------------------------------- real Tesseract (if any)
@pytest.mark.integration
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")
def test_real_ocr_on_synthetic_label():
    img = np.full((260, 900, 3), 255, np.uint8)
    cv2.putText(img, "MFG 01 JAN 2026", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "EXP 15 JUN 2027", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3, cv2.LINE_AA)
    r = read_expiration_dates(img)
    assert r.status == "ok"
    found = {c.label_type: c.parsed for c in r.candidates}
    assert found.get("expiration") == date(2027, 6, 15)
    assert found.get("manufacturing") == date(2026, 1, 1)
