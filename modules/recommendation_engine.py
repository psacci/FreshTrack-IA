"""Turns inventory + alerts + fuzzy scores into a ranked, explained to-do list.

Safety rules baked in:
* Visibly spoiled items and items past an expiration/use-by date are never put
  on the "use first" list. They go to "review" with plain wording.
* Items with no usable information get no invented score ("limited").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Sequence

from config import settings
from modules import alert_manager as alerts
from modules.formatting import format_quantity
from modules.fuzzy_engine import compute_priority

BUCKET_USE_FIRST = "use_first"
BUCKET_REVIEW = "review"
BUCKET_LIMITED = "limited"
BUCKET_OK = "ok"


@dataclass
class Recommendation:
    item_id: int
    name: str
    category: str
    quantity: float
    unit: str
    bucket: str
    action: str
    score: Optional[float]
    level: Optional[str]
    alert: alerts.AlertResult
    reasons: list[str] = field(default_factory=list)
    spoiled: bool = False


@dataclass
class RankedInventory:
    use_first: list[Recommendation]
    review: list[Recommendation]
    limited: list[Recommendation]
    ok: list[Recommendation]

    @property
    def all(self) -> list[Recommendation]:
        return self.use_first + self.review + self.limited + self.ok


def recommend_item(
    item: dict,
    dates: Sequence[dict],
    today: Optional[date] = None,
    soon_days: int = settings.DEFAULT_SOON_DAYS,
) -> Recommendation:
    today = today or date.today()
    alert = alerts.evaluate_item(dates, today, soon_days)
    added = date.fromisoformat(item["date_added"])
    storage_days = max((today - added).days, 0)

    label = item.get("freshness_label")
    conf = item.get("freshness_confidence")
    days_remaining = alert.days_remaining if alert.status in (alerts.EXPIRED, alerts.TODAY, alerts.SOON, alerts.OK) else None
    fuzzy = compute_priority(days_remaining, label, conf, storage_days)

    reasons: list[str] = []
    if alert.status in (alerts.EXPIRED, alerts.TODAY, alerts.SOON, alerts.OK):
        reasons.append(alert.message)
    spoiled = label == "spoiled"
    if label:
        pct = f" ({conf * 100:.0f}% sure)" if conf is not None else ""
        reasons.append(f"Last freshness scan: {settings.FRESHNESS_LABELS[label].lower()}{pct}")
    if storage_days >= 7:
        reasons.append(f"Added {storage_days} days ago")
    if item.get("quantity", 1) > 1:
        reasons.append(f"{format_quantity(item['quantity'], item['unit'])} on hand")

    past_hard = alert.status == alerts.EXPIRED and alert.date_type in ("expiration", "use_by")
    past_soft = alert.status == alerts.EXPIRED and not past_hard

    if spoiled:
        bucket, action = BUCKET_REVIEW, "Looks spoiled. Do not eat it."
    elif past_hard:
        bucket, action = BUCKET_REVIEW, "Past its date. Check it before eating, or toss it."
    elif past_soft:
        bucket, action = BUCKET_REVIEW, "Past its best-before. Check the quality."
    elif fuzzy.limited:
        bucket = BUCKET_LIMITED
        action = ("Confirm its date to rank it." if alert.status == alerts.NEEDS_VERIFICATION
                  else "Add a date or scan it to rank it.")
        reasons.append(fuzzy.reason)
    else:
        level = fuzzy.level
        if level == "Urgent":
            bucket, action = BUCKET_USE_FIRST, "Use it today."
        elif level == "High":
            bucket, action = BUCKET_USE_FIRST, "Use it in the next day or two."
        elif level == "Moderate":
            bucket, action = BUCKET_USE_FIRST, "Plan to use it this week."
        else:
            bucket, action = BUCKET_OK, "No rush."

    return Recommendation(
        item_id=item["id"], name=item["name"], category=item["category"],
        quantity=item["quantity"], unit=item["unit"], bucket=bucket, action=action,
        score=fuzzy.score, level=fuzzy.level, alert=alert, reasons=reasons, spoiled=spoiled,
    )


def rank(
    entries: Sequence[tuple[dict, Sequence[dict]]],
    today: Optional[date] = None,
    soon_days: int = settings.DEFAULT_SOON_DAYS,
) -> RankedInventory:
    recs = [recommend_item(item, dates, today, soon_days) for item, dates in entries]
    use_first = sorted((r for r in recs if r.bucket == BUCKET_USE_FIRST), key=lambda r: -(r.score or 0))
    review = sorted(
        (r for r in recs if r.bucket == BUCKET_REVIEW),
        key=lambda r: (not r.spoiled, r.alert.severity != "danger", r.alert.days_remaining or 0),
    )
    limited = [r for r in recs if r.bucket == BUCKET_LIMITED]
    ok = sorted((r for r in recs if r.bucket == BUCKET_OK), key=lambda r: -(r.score or 0))
    return RankedInventory(use_first, review, limited, ok)
