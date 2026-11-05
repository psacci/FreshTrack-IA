"""Inventory logic: validation + coordination between the database, alerts and ranking."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

from config import settings
from modules import alert_manager, recommendation_engine
from modules.database_manager import DatabaseManager


@dataclass
class InventoryEntry:
    item: dict
    dates: list[dict]
    alert: alert_manager.AlertResult
    rec: recommendation_engine.Recommendation


def _require_iso(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Not a valid date: {value!r}") from exc


class InventoryManager:
    def __init__(self, db: Optional[DatabaseManager] = None) -> None:
        self.db = db or DatabaseManager()

    # ------------------------------------------------------------ validation
    @staticmethod
    def _validate_item(fields: dict) -> dict:
        clean = dict(fields)
        name = (clean.get("name") or "").strip()
        if not name:
            raise ValueError("Give the item a name.")
        clean["name"] = name
        if clean.get("date_added") is not None:
            clean["date_added"] = _require_iso(clean["date_added"])
        if clean.get("category") not in settings.FOOD_CATEGORIES:
            raise ValueError("Pick one of the food categories.")
        try:
            quantity = float(clean.get("quantity", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("Quantity must be a number.") from exc
        if quantity < 0:
            raise ValueError("Quantity cannot be negative.")
        clean["quantity"] = quantity
        label = clean.get("freshness_label")
        if label is not None and label not in settings.FRESHNESS_CLASSES:
            raise ValueError("Unknown freshness label.")
        return clean

    # ----------------------------------------------------------------- items
    def add_item(self, dates: Optional[Sequence[dict]] = None, **fields) -> int:
        clean = self._validate_item(fields)
        item_id = self.db.insert_item(**clean)
        for d in dates or []:
            self.add_date(item_id, **d)
        return item_id

    def update_item(self, item_id: int, **fields) -> None:
        current = self.db.get_item(item_id)
        if current is None:
            raise KeyError(f"Item {item_id} does not exist.")
        merged = {**current, **fields}
        self.db.update_item(item_id, **self._validate_item(merged))

    def delete_item(self, item_id: int) -> None:
        self.db.delete_item(item_id)

    def archive_item(self, item_id: int, status: str) -> None:
        self.db.archive_item(item_id, status)

    def record_freshness(self, item_id: int, label: str, confidence: float) -> None:
        self.update_item(item_id, freshness_label=label, freshness_confidence=confidence)

    # ----------------------------------------------------------------- dates
    def add_date(
        self,
        item_id: int,
        date_type: str,
        date_value: str,
        source: str = "manual",
        verified: bool = True,
        original_label: Optional[str] = None,
        original_text: Optional[str] = None,
        ocr_confidence: Optional[float] = None,
    ) -> int:
        if date_type not in settings.DATE_TYPES:
            raise ValueError(f"Unknown date type: {date_type}")
        return self.db.add_date(
            item_id, date_type, _require_iso(date_value), source, original_label,
            original_text, ocr_confidence, "verified" if verified else "unverified",
        )

    # ------------------------------------------------------------- read side
    def soon_days(self) -> int:
        return int(self.db.get_setting("soon_days", str(settings.DEFAULT_SOON_DAYS)))

    def set_soon_days(self, days: int) -> None:
        self.db.set_setting("soon_days", str(max(1, int(days))))

    def entries(
        self,
        search: str = "",
        category: Optional[str] = None,
        location: Optional[str] = None,
        today: Optional[date] = None,
    ) -> list[InventoryEntry]:
        soon = self.soon_days()
        items = self.db.list_items(search, category, location)
        by_item: dict[int, list[dict]] = {}
        for d in self.db.list_dates():
            by_item.setdefault(d["item_id"], []).append(d)
        result = []
        for item in items:
            dates = by_item.get(item["id"], [])
            alert = alert_manager.evaluate_item(dates, today, soon)
            rec = recommendation_engine.recommend_item(item, dates, today, soon)
            result.append(InventoryEntry(item, dates, alert, rec))
        return result

    def ranked(self, today: Optional[date] = None) -> recommendation_engine.RankedInventory:
        soon = self.soon_days()
        by_item: dict[int, list[dict]] = {}
        for d in self.db.list_dates():
            by_item.setdefault(d["item_id"], []).append(d)
        pairs = [(i, by_item.get(i["id"], [])) for i in self.db.list_items()]
        return recommendation_engine.rank(pairs, today, soon)


BUCKET_RANK = {"use_first": 0, "review": 1, "limited": 2, "ok": 3}


def group_entries(entries: Sequence[InventoryEntry]) -> dict[str, list[InventoryEntry]]:
    """Group inventory entries by recommendation bucket, most important first."""
    groups: dict[str, list[InventoryEntry]] = {k: [] for k in BUCKET_RANK}
    for entry in entries:
        groups[entry.rec.bucket].append(entry)
    groups["use_first"].sort(key=lambda e: -(e.rec.score or 0))
    groups["ok"].sort(key=lambda e: -(e.rec.score or 0))
    groups["review"].sort(key=lambda e: (not e.rec.spoiled, e.rec.alert.severity != "danger", e.rec.alert.days_remaining or 0))
    return groups
