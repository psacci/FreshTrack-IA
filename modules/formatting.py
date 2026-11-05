"""Small text helpers shared by the recommendation engine and the UI."""
from __future__ import annotations

from datetime import date
from typing import Optional

_PLURAL_UNITS = {"pack": "packs", "can": "cans", "bottle": "bottles"}
_LOCATION_PHRASE = {
    "Fridge": "in the fridge",
    "Freezer": "in the freezer",
    "Pantry": "in the pantry",
    "Counter": "on the counter",
}


def format_quantity(qty: float, unit: str) -> str:
    """'1 pc', '4 pcs', '1 pack', '2 packs', '500 g'."""
    number = f"{qty:g}"
    if unit == "pcs":
        return f"{number} pc" if qty == 1 else f"{number} pcs"
    if unit in _PLURAL_UNITS and qty != 1:
        return f"{number} {_PLURAL_UNITS[unit]}"
    return f"{number} {unit}"


def location_phrase(location: str) -> str:
    return _LOCATION_PHRASE.get(location, f"in {location.lower()}")


def friendly_date(d: date, today: Optional[date] = None) -> str:
    """'Sat Nov 8'. The year is added when the date is far away."""
    today = today or date.today()
    text = f"{d:%a} {d:%b} {d.day}"
    if abs((d - today).days) > 150:
        text += f", {d.year}"
    return text
