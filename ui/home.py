"""Home: what to eat first, what to check, what is waiting on a date."""
from __future__ import annotations

from datetime import date

import streamlit as st

from modules.inventory_manager import group_entries
from ui import theme
from ui.state import get_inventory, goto

SCAN_LABELS = {"detection": "Photo scan", "freshness": "Freshness check", "ocr": "Date read"}


def render() -> None:
    inv = get_inventory()
    entries = inv.entries()
    today = date.today()

    if not entries:
        theme.hero("Your fridge is empty", "Scan a photo or add something by hand. FreshTrack will tell you what to eat first.")
        st.markdown('<div class="empty"><h3>Start with one thing</h3>'
                    '<p>Take a photo of a fruit, or of a package label, and it lands in your fridge with its date.</p></div>',
                    unsafe_allow_html=True)
        _recent_scans(inv)
        st.write("")
        c1, c2, _ = st.columns([1, 1.1, 6])
        c1.button("Scan food", type="primary", on_click=goto, args=("Scan",), kwargs={"scan_mode": "Food photo"})
        c2.button("Read a date", on_click=goto, args=("Scan",), kwargs={"scan_mode": "Package date"})
        return

    groups = group_entries(entries)
    use, review, limited = groups["use_first"], groups["review"], groups["limited"]

    if use:
        title = "Eat these first"
    elif review:
        title = "Check these before you eat"
    else:
        title = "Nothing urgent right now"
    theme.hero(title, f"{theme.fmt_date(today)}. {len(use)} to use soon, {len(review)} to check, {len(limited)} waiting on a date.")

    c1, c2, _ = st.columns([1, 1.1, 6])
    c1.button("Scan food", type="primary", on_click=goto, args=("Scan",), kwargs={"scan_mode": "Food photo"})
    c2.button("Read a date", on_click=goto, args=("Scan",), kwargs={"scan_mode": "Package date"})

    if use:
        theme.section("Use first", "Ranked by how soon the date is, how it looks and how long it has been in the fridge.")
        theme.shelf(use[:6], lead_first=True)

    if review:
        theme.section("Check before eating", "These look spoiled or are past a date on the label. They are never on the use-first list.")
        theme.shelf(review[:6])

    if limited:
        theme.section("Waiting on a date", "Confirm a date or scan the item and it joins the ranking.")
        rows = "".join(f'<div class="row"><span>{theme.esc(e.item["name"])}</span><span>{theme.esc(e.rec.action)}</span></div>' for e in limited[:8])
        st.markdown(f'<div class="plain-list">{rows}</div>', unsafe_allow_html=True)

    if not (use or review or limited):
        st.write("")
        theme.note("Everything you have is well within its dates. Nice.", "info")

    st.write("")
    with st.expander("Alert settings"):
        current = inv.soon_days()
        chosen = st.slider("Call a date 'coming up' this many days ahead", 1, 14, current, key="soon_slider")
        if chosen != current:
            inv.set_soon_days(chosen)
            st.rerun()
        theme.note("FreshTrack ranks what to eat first. It is not a food-safety test, and a low score never means a food is safe.", "info")

    _recent_scans(inv)


def _recent_scans(inv) -> None:
    scans = inv.db.list_scans(5)
    if scans:
        theme.section("Recent scans")
        for scan in scans:
            label = SCAN_LABELS.get(scan["scan_type"], scan["scan_type"])
            with st.expander(f"{label}: {scan['result_summary']}"):
                scan_date_display, scan_date_edit = st.columns([6, 0.4])
                scan_date_display.caption(scan["created_at"][:16].replace("T", " "))
                if scan.get("preview_image"):
                    st.image(scan["preview_image"], width=220, caption="Captured scan image")
                else:
                    st.caption("No picture was saved with this scan.")
                with scan_date_edit.popover("⋯"):
                    with st.form(f"scan_date_{scan['id']}"):
                        scan_date = st.date_input(
                            "Date",
                            value=date.fromisoformat(scan["created_at"][:10]),
                            format="YYYY-MM-DD",
                        )
                        if st.form_submit_button("Save scan date"):
                            try:
                                inv.db.update_scan_date(scan["id"], scan_date.isoformat())
                                st.rerun()
                            except ValueError as exc:
                                st.error(str(exc))
