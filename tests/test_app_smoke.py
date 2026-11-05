"""Loads every page of the Streamlit app headlessly and fails on any exception."""
from datetime import date, timedelta

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from config import settings  # noqa: E402
from modules.food_detector import Detection  # noqa: E402
from ui.fridge import _freshness_info, _has_freshness_info  # noqa: E402
from ui.scan import _group_detected_foods  # noqa: E402

PAGES = ["Home", "Scan", "Fridge", "About"]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    import streamlit as st
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "app.db")
    st.cache_resource.clear()
    at = AppTest.from_file(str(settings.BASE_DIR / "app.py"), default_timeout=90)
    yield at
    st.cache_resource.clear()


def seed():
    from modules.inventory_manager import InventoryManager
    inv = InventoryManager()
    today = date.today()
    soon = (today + timedelta(days=2)).isoformat()
    past = (today - timedelta(days=3)).isoformat()
    inv.add_item(name="Banana", category="Fruits & Vegetables", quantity=4, freshness_label="aging",
                 freshness_confidence=0.9, dates=[dict(date_type="use_by", date_value=soon, verified=True)])
    inv.add_item(name="Yogurt", category="Dairy", dates=[dict(date_type="expiration", date_value=past, verified=True)])
    inv.add_item(name="Apple", category="Fruits & Vegetables", freshness_label="spoiled", freshness_confidence=0.95)
    inv.add_item(name="Rice", category="Dry & Packaged")
    inv.add_item(name="Cheese", category="Dairy", dates=[dict(date_type="best_before", date_value=soon, verified=False,
                                                              source="ocr")])


@pytest.mark.parametrize("page", PAGES)
def test_pages_load_when_empty(app, page):
    app.session_state["page"] = page
    if page == "Scan":
        app.session_state["food_scan_mode"] = "Photo or upload"
    app.run()
    assert not app.exception, app.exception


@pytest.mark.parametrize("page", PAGES)
def test_pages_load_with_data(app, page):
    seed()
    app.session_state["page"] = page
    if page == "Scan":
        app.session_state["food_scan_mode"] = "Photo or upload"
    app.run()
    assert not app.exception, app.exception


def test_home_lists_use_first_and_review(app):
    seed()
    app.run()
    html = " ".join(m.value for m in app.markdown)
    assert "Banana" in html and "Looks spoiled" in html and "Past its date" in html
    assert "Eat these first" in html


def test_scan_package_date_mode_loads(app):
    seed()
    app.session_state["page"] = "Scan"
    app.session_state["scan_mode"] = "Package date"
    app.session_state["date_scan_mode"] = "Photo or upload"
    app.run()
    assert not app.exception, app.exception


def test_manual_add_from_fridge_page(app):
    app.session_state["page"] = "Fridge"
    app.run()
    assert not app.exception


def test_manage_can_change_item_category(app):
    seed()
    app.session_state["page"] = "Fridge"
    app.run()

    app.selectbox(key="it1_cat").select("Dairy").run()
    app.button(key="it1_save").click().run()

    from modules.database_manager import DatabaseManager
    item = DatabaseManager().get_item(1)
    assert item["category"] == "Dairy"


def test_fridge_freshness_info_formats_saved_result_and_confidence():
    text, has_result = _freshness_info({"freshness_label": "aging", "freshness_confidence": 0.9})
    assert has_result
    assert text == "Saved result: looks aging / overripe (90% model confidence)."


def test_fridge_freshness_info_explains_missing_result():
    text, has_result = _freshness_info({"freshness_label": None, "freshness_confidence": None})
    assert not has_result
    assert text == "No freshness result is saved for this item yet."


def test_fridge_shows_freshness_info_only_for_produce():
    assert _has_freshness_info({"category": "Fruits & Vegetables"})
    assert not _has_freshness_info({"category": "Dairy"})
    assert not _has_freshness_info({"category": "Meat & Seafood"})


def test_photo_scan_groups_repeated_food_detections_for_quantity():
    detections = [
        Detection("banana", 0.81, (0, 0, 10, 10), "Fruits & Vegetables"),
        Detection("banana", 0.94, (15, 0, 25, 10), "Fruits & Vegetables"),
        Detection("apple", 0.88, (30, 0, 40, 10), "Fruits & Vegetables"),
    ]

    grouped = _group_detected_foods(detections, [None, None, None])

    assert [(detection.label, detection.confidence, count) for detection, _, count in grouped] == [
        ("banana", 0.94, 2),
        ("apple", 0.88, 1),
    ]
