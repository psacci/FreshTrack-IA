from datetime import date

from modules.formatting import format_quantity, friendly_date, location_phrase


def test_quantity_wording():
    assert format_quantity(1, "pcs") == "1 pc"
    assert format_quantity(4, "pcs") == "4 pcs"
    assert format_quantity(2, "pack") == "2 packs"
    assert format_quantity(1, "can") == "1 can"
    assert format_quantity(500, "g") == "500 g"
    assert format_quantity(1.5, "L") == "1.5 L"


def test_location_wording():
    assert location_phrase("Counter") == "on the counter"
    assert location_phrase("Fridge") == "in the fridge"


def test_year_only_shown_for_far_dates():
    today = date(2026, 11, 7)
    assert friendly_date(date(2026, 11, 8), today) == "Sun Nov 8"
    assert friendly_date(date(2027, 9, 3), today).endswith("2027")
