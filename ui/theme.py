"""Look and feel: CSS injection plus small HTML builders for cards and chips."""
from __future__ import annotations

from datetime import date
from html import escape
from typing import Optional, Sequence

import streamlit as st

from config import settings
from modules.formatting import format_quantity, friendly_date, location_phrase

DATE_SHORT = {
    "expiration": "Exp",
    "use_by": "Use by",
    "best_before": "Best before",
    "sell_by": "Sell by",
}
LEVEL_CHIP = {"Urgent": "Use today", "High": "Use soon", "Moderate": "This week", "Low": "No rush"}


def inject_css() -> None:
    css = settings.STYLES_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def fmt_date(d: date) -> str:
    return friendly_date(d)


def esc(value) -> str:
    return escape(str(value if value is not None else ""))


def hero(title: str, subtitle: str = "") -> None:
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="hero"><h1>{esc(title)}</h1>{sub}</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    sub = f'<p class="section-sub">{esc(subtitle)}</p>' if subtitle else ""
    st.markdown(f'<div class="section-title">{esc(title)}</div>{sub}', unsafe_allow_html=True)


def note(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="note {kind}">{esc(text)}</div>', unsafe_allow_html=True)


def chip(text: str, tone: str = "") -> str:
    return f'<span class="chip {tone}">{esc(text)}</span>'


def render_header(pages: Sequence[str]) -> None:
    with st.container(key="topbar"):
        left, right = st.columns([1, 2], vertical_alignment="center")
        left.markdown('<div class="wordmark"><span class="dot"></span>FreshTrack</div>', unsafe_allow_html=True)
        with right.container(key="topnav"):
            st.radio("Go to", list(pages), key="page", horizontal=True, label_visibility="collapsed")


# ----------------------------------------------------------------------------
# Item cards
# ----------------------------------------------------------------------------
def tone_for(rec) -> str:
    if rec.bucket == "review":
        return "berry"
    if rec.bucket == "limited":
        return "frost"
    if rec.level in ("Urgent", "High"):
        return "citrus"
    return "leaf"


def _status_chip(rec) -> str:
    if rec.bucket == "review":
        if rec.spoiled:
            return chip("Looks spoiled", "berry")
        if rec.alert.severity == "danger":
            return chip("Past its date", "berry")
        return chip("Past best-before", "berry")
    if rec.bucket == "limited":
        return chip("Confirm date" if rec.alert.status == "needs_verification" else "Needs a date")
    tone = "citrus" if rec.level in ("Urgent", "High") else "leaf"
    return chip(LEVEL_CHIP.get(rec.level, ""), tone)


def _sticker(rec) -> str:
    a = rec.alert
    if a.date_value:
        text = f"{DATE_SHORT[a.date_type]} {fmt_date(a.date_value)}"
    elif a.status == "needs_verification":
        text = "Date unconfirmed"
    else:
        text = "No date"
    return f'<span class="sticker">{esc(text)}</span>'


def item_card(entry, lead: bool = False, reasons: int = 2, fixed: bool = False) -> str:
    """HTML for one container-style card. `entry` is an InventoryEntry."""
    rec, item = entry.rec, entry.item
    tone = tone_for(rec)
    qty = format_quantity(item["quantity"], item["unit"])
    fresh = ""
    if item.get("category") == "Fruits & Vegetables" and item.get("freshness_label") and not rec.spoiled:
        label = settings.FRESHNESS_LABELS[item["freshness_label"]]
        fresh_tone = {"fresh": "leaf", "aging": "citrus", "spoiled": "berry"}[item["freshness_label"]]
        fresh = chip(f"Looks: {label.lower()}", fresh_tone)
    why = "".join(f"<li>{esc(r)}</li>" for r in rec.reasons[:reasons] if r)
    why_html = f'<ul class="why">{why}</ul>' if why else ""
    return (
        f'<div class="item tone-{tone}{" is-lead" if lead else ""}{" fixed" if fixed else ""}"><div class="lid"></div><div class="body">'
        f'<div class="topline"><div class="chips">{_status_chip(rec)}{fresh}</div>{_sticker(rec)}</div>'
        f'<h3 class="name">{esc(item["name"])}</h3>'
        f'<p class="meta">{esc(qty)} {esc(location_phrase(item["storage_location"]))}</p>'
        f'<p class="action">{esc(rec.action)}</p>{why_html}</div></div>'
    )


def shelf(entries: Sequence, lead_first: bool = False) -> None:
    """Render entries as rows of three cards, each row on its own shelf line."""
    rows = [entries[i:i + 3] for i in range(0, len(entries), 3)]
    for row_index, row in enumerate(rows):
        lead = lead_first and row_index == 0
        cards = "".join(item_card(e, lead=lead and i == 0) for i, e in enumerate(row))
        st.markdown(f'<div class="shelf{" lead" if lead else ""}">{cards}</div>', unsafe_allow_html=True)
