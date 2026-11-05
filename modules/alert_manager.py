"""Expiration alerts.

Rules that keep the alerts honest:
* Only *verified* dates of type expiration / use-by / best-before / sell-by count.
* Manufacturing and packing dates never raise an alert.
* A date the user has not confirmed is reported as "needs verification",
  never as an automatic expiry claim.
* The wording differs per date type: a best-before date is a quality hint,
  not a safety limit.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

from config import settings

EXPIRED = "expired"
TODAY = "today"
SOON = "soon"
OK = "ok"
NO_DATE = "no_date"
NEEDS_VERIFICATION = "needs_verification"

STATUS_LABELS = {
    EXPIRED: "Past date",
    TODAY: "Due today",
    SOON: "Coming up",
    OK: "Not near its date",
    NO_DATE: "No date yet",
    NEEDS_VERIFICATION: "Date needs a check",
}


@dataclass
class AlertResult:
    status: str
    severity: str                      # danger | warn | ok | info
    message: str
    date_type: Optional[str] = None
    date_value: Optional[date] = None
    days_remaining: Optional[int] = None

    @property
    def label(self) -> str:
        return STATUS_LABELS[self.status]


def _type_name(date_type: str) -> str:
    return settings.DATE_TYPE_LABELS[date_type].lower().replace(" date", "")


def _past_message(date_type: str, days_ago: int) -> str:
    when = "today" if days_ago == 0 else f"{days_ago} day{'s' if days_ago != 1 else ''} ago"
    if date_type == "expiration":
        return f"Passed its expiration date {when}"
    if date_type == "use_by":
        return f"Passed its use-by date {when}"
    if date_type == "best_before":
        return f"Past its best-before date ({when}). That is a quality date, so check the food."
    return f"Past its sell-by date ({when}). That is a store date, so check the food."


def _upcoming_message(date_type: str, days: int) -> str:
    name = _type_name(date_type)
    if days == 0:
        return f"Its {name} date is today"
    if days == 1:
        return f"Its {name} date is tomorrow"
    return f"Its {name} date is in {days} days"


def evaluate_item(
    dates: Sequence[dict],
    today: Optional[date] = None,
    soon_days: int = settings.DEFAULT_SOON_DAYS,
) -> AlertResult:
    """Return the alert for one item from its date records."""
    today = today or date.today()
    relevant = [
        d for d in dates
        if d["date_type"] in settings.ALERT_DATE_TYPES and d["verification_status"] != "rejected"
    ]
    if not relevant:
        return AlertResult(NO_DATE, "info", "No expiration-related date saved for this item")

    verified = [d for d in relevant if d["verification_status"] == "verified"]
    if not verified:
        return AlertResult(NEEDS_VERIFICATION, "info", "A date was read but you have not confirmed it yet")

    chosen = min(verified, key=lambda d: d["date_value"])
    due = date.fromisoformat(chosen["date_value"])
    days = (due - today).days
    dtype = chosen["date_type"]

    if days < 0:
        severity = "danger" if dtype in ("expiration", "use_by") else "warn"
        return AlertResult(EXPIRED, severity, _past_message(dtype, -days), dtype, due, days)
    if days == 0:
        return AlertResult(TODAY, "warn", _upcoming_message(dtype, 0), dtype, due, 0)
    if days <= soon_days:
        return AlertResult(SOON, "warn", _upcoming_message(dtype, days), dtype, due, days)
    return AlertResult(OK, "ok", _upcoming_message(dtype, days), dtype, due, days)


def summarize(alerts: Sequence[AlertResult]) -> dict[str, int]:
    """Count alerts per status (used by the home page summary)."""
    counts = {status: 0 for status in STATUS_LABELS}
    for alert in alerts:
        counts[alert.status] += 1
    return counts
