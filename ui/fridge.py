"""Fridge: every item, with search, filters and per-item controls."""
from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from config import settings
from modules.inventory_manager import BUCKET_RANK
from ui import theme
from ui.state import get_inventory

SORTS = ("Use first", "Newest", "Name (A to Z)", "Date soonest")
PACKAGE_DATE_TYPES = ("expiration", "best_before")


def _sort(entries, how: str):
    if how == "Newest":
        return sorted(entries, key=lambda e: (e.item["date_added"], e.item["id"]), reverse=True)
    if how == "Name (A to Z)":
        return sorted(entries, key=lambda e: e.item["name"].lower())
    if how == "Date soonest":
        return sorted(entries, key=lambda e: e.alert.date_value or date.max)
    return sorted(entries, key=lambda e: (BUCKET_RANK[e.rec.bucket], -(e.rec.score or 0)))


def _freshness_info(item: dict) -> tuple[str, bool]:
    label = item.get("freshness_label")
    if not label:
        return "No freshness result is saved for this item yet.", False
    freshness = settings.FRESHNESS_LABELS[label]
    confidence = item.get("freshness_confidence")
    certainty = f" ({float(confidence):.0%} model confidence)" if confidence is not None else ""
    return f"Saved result: looks {freshness.lower()}{certainty}.", True


def _has_freshness_info(item: dict) -> bool:
    return item.get("category") == "Fruits & Vegetables"


def _add_form(inv) -> None:
    with st.expander("Add something by hand"):
        with st.form("fridge_add", clear_on_submit=True):
            c1, c2 = st.columns(2)
            name = c1.text_input("Name")
            category = c2.selectbox("Category", settings.FOOD_CATEGORIES)
            c3, c4, c5 = st.columns(3)
            qty = c3.number_input("How many", min_value=0.0, value=1.0, step=1.0)
            unit = c4.selectbox("Unit", settings.UNITS)
            where = c5.selectbox("Where is it?", settings.STORAGE_LOCATIONS)
            c6, c7 = st.columns(2)
            kind = c6.selectbox("Package date type", ["No date"] + list(PACKAGE_DATE_TYPES),
                                format_func=lambda k: k if k == "No date" else settings.DATE_TYPE_LABELS[k])
            when = c7.date_input("Enter selected date", value=date.today() + timedelta(days=7), format="YYYY-MM-DD")
            added = st.date_input("Added to fridge", value=date.today(), format="YYYY-MM-DD")
            if st.form_submit_button("Add to fridge", type="primary"):
                try:
                    dates = [] if kind == "No date" else [dict(date_type=kind, date_value=when.isoformat(), verified=True)]
                    inv.add_item(name=name, category=category, quantity=qty, unit=unit,
                                 storage_location=where, date_added=added.isoformat(), dates=dates)
                    inv.db.add_scan("detection", f"Added {name.strip()} by hand")
                    st.success(f"Added {name.strip()}.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))


def _manage(inv, entry) -> None:
    item = entry.item
    key = f"it{item['id']}"
    with st.popover("Manage", use_container_width=True):
        st.markdown(f"**{item['name']}**")
        date_label, date_edit = st.columns([5, 1], vertical_alignment="center")
        date_label.caption(f"Added to fridge: {theme.fmt_date(date.fromisoformat(item['date_added']))}")
        with date_edit.popover("⋯"):
            added = st.date_input(
                "Added to fridge",
                value=date.fromisoformat(item["date_added"]),
                format="YYYY-MM-DD",
                key=f"{key}_added",
            )
            if st.button("Save date", key=f"{key}_save_added"):
                inv.update_item(item["id"], date_added=added.isoformat())
                st.rerun()
        if item.get("image_data"):
            st.image(item["image_data"], width=112)
        qty = st.number_input("How many", min_value=0.0, value=float(item["quantity"]), step=1.0, key=f"{key}_q")
        category = st.selectbox(
            "Category",
            settings.FOOD_CATEGORIES,
            index=settings.FOOD_CATEGORIES.index(item["category"])
            if item["category"] in settings.FOOD_CATEGORIES else 0,
            key=f"{key}_cat",
        )
        where = st.selectbox("Where is it?", settings.STORAGE_LOCATIONS,
                             index=settings.STORAGE_LOCATIONS.index(item["storage_location"]) if item["storage_location"] in settings.STORAGE_LOCATIONS else 0,
                             key=f"{key}_w")
        notes = st.text_input("Notes", value=item["notes"] or "", key=f"{key}_n")
        if st.button("Save changes", type="primary", key=f"{key}_save"):
            inv.update_item(
                item["id"],
                quantity=qty,
                category=category,
                storage_location=where,
                notes=notes or None,
            )
            st.rerun()

        if _has_freshness_info(item):
            st.divider()
            st.markdown("**Freshness info**")
            freshness_text, has_freshness = _freshness_info(item)
            st.write(freshness_text)
            if has_freshness:
                st.caption("Appearance only; this is not a food-safety assessment.")

        st.divider()
        st.caption("Dates on file")
        if not entry.dates:
            st.write("None yet.")
        for d in entry.dates:
            state = {"verified": "confirmed", "unverified": "not confirmed", "rejected": "rejected"}[d["verification_status"]]
            c1, c2 = st.columns([3, 1])
            c1.write(f"{settings.DATE_TYPE_LABELS[d['date_type']]}: {theme.fmt_date(date.fromisoformat(d['date_value']))} ({state})")
            if d["verification_status"] == "unverified" and c2.button("Confirm", key=f"{key}_c{d['id']}"):
                inv.db.set_date_status(d["id"], "verified")
                st.rerun()
            if c2.button("Remove", key=f"{key}_d{d['id']}"):
                inv.db.delete_date(d["id"])
                st.rerun()
        kind = st.selectbox("Package date type", PACKAGE_DATE_TYPES,
                            format_func=lambda k: settings.DATE_TYPE_LABELS[k], key=f"{key}_k")
        when = st.date_input("Enter selected date", value=date.today() + timedelta(days=7), format="YYYY-MM-DD", key=f"{key}_dt")
        if st.button("Save date", key=f"{key}_sd"):
            inv.add_date(item["id"], kind, when.isoformat(), source="manual", verified=True)
            st.rerun()

        st.divider()
        consumed, delete = st.columns(2)
        if consumed.button("Mark consumed", key=f"{key}_consumed"):
            inv.archive_item(item["id"], "consumed")
            st.rerun()
        if delete.button("Delete item", key=f"{key}_del"):
            inv.archive_item(item["id"], "deleted")
            st.rerun()


def _history(inv) -> None:
    items = inv.db.list_archived_items()
    with st.expander(f"Fridge history ({len(items)})"):
        if not items:
            st.caption("Consumed and deleted items will appear here.")
            return
        for item in items:
            name, details = st.columns([3, 2])
            name.markdown(f"**{item['name']}** · {item['quantity']:g} {item['unit']}")
            status = "Consumed" if item["status"] == "consumed" else "Deleted"
            archived = item.get("archived_at") or ""
            details.caption(f"{status} · {archived[:10]}")
            if item.get("image_data"):
                name.image(item["image_data"], width=72)
            restore, remove = st.columns(2)
            if restore.button("Restore", key=f"history_restore_{item['id']}"):
                inv.db.restore_item(item["id"])
                st.rerun()
            if remove.button("Delete permanently", key=f"history_delete_{item['id']}"):
                inv.delete_item(item["id"])
                st.rerun()


def render() -> None:
    inv = get_inventory()
    theme.hero("Your fridge", "Everything you have, with the date and what to do about it.")
    _add_form(inv)

    f1, f2, f3, f4 = st.columns([2, 1.3, 1.1, 1.2])
    search = f1.text_input("Search", placeholder="Search by name or brand", label_visibility="collapsed")
    category = f2.selectbox("Category", ["All categories"] + list(settings.FOOD_CATEGORIES), label_visibility="collapsed")
    where = f3.selectbox("Location", ["Anywhere"] + list(settings.STORAGE_LOCATIONS), label_visibility="collapsed")
    how = f4.selectbox("Sort", SORTS, label_visibility="collapsed")

    entries = inv.entries(search, None if category.startswith("All") else category, None if where == "Anywhere" else where)
    if not entries:
        st.write("")
        st.markdown('<div class="empty"><h3>Nothing here yet</h3><p>Scan a photo or add something by hand to see it here.</p></div>', unsafe_allow_html=True)
        _history(inv)
        return

    entries = _sort(entries, how)
    st.caption(f"{len(entries)} item{'s' if len(entries) != 1 else ''}")
    for start in range(0, len(entries), 3):
        cols = st.columns(3)
        for col, entry in zip(cols, entries[start:start + 3]):
            with col:
                st.markdown(theme.item_card(entry, fixed=True), unsafe_allow_html=True)
                _manage(inv, entry)
        st.write("")
    _history(inv)
