import sqlite3

import pytest


def test_tables_created(db):
    with db.connect() as conn:
        names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"food_items", "food_dates", "scan_history", "app_settings"} <= names


def test_insert_get_update_delete(db):
    item_id = db.insert_item(name="Milk", category="Dairy", quantity=2, unit="bottle")
    row = db.get_item(item_id)
    assert row["name"] == "Milk" and row["quantity"] == 2
    db.update_item(item_id, quantity=1, notes="opened")
    assert db.get_item(item_id)["notes"] == "opened"
    db.delete_item(item_id)
    assert db.get_item(item_id) is None


def test_item_photo_thumbnail_is_stored_in_database(db):
    thumbnail = b"\xff\xd8small jpeg preview\xff\xd9"
    item_id = db.insert_item(
        name="Apple",
        category="Fruits & Vegetables",
        image_data=thumbnail,
    )

    assert db.get_item(item_id)["image_data"] == thumbnail


def test_archived_items_leave_fridge_list_and_can_be_restored(db):
    item_id = db.insert_item(name="Banana", category="Fruits & Vegetables")

    db.archive_item(item_id, "consumed")

    assert db.list_items() == []
    archived = db.list_archived_items()
    assert len(archived) == 1
    assert archived[0]["status"] == "consumed"
    assert archived[0]["archived_at"]

    db.restore_item(item_id)

    assert db.list_items()[0]["id"] == item_id
    assert db.list_archived_items() == []


def test_archived_items_distinguish_delete_from_consumed(db):
    item_id = db.insert_item(name="Milk", category="Dairy")

    db.archive_item(item_id, "deleted")

    assert db.list_archived_items()[0]["status"] == "deleted"


def test_scan_date_can_be_changed_without_changing_time(db):
    scan_id = db.add_scan("ocr", "Read a date")
    original = db.list_scans()[0]["created_at"]

    db.update_scan_date(scan_id, "2026-11-06")

    updated = db.list_scans()[0]["created_at"]
    assert updated.startswith("2026-11-06")
    assert updated[10:] == original[10:]


def test_scan_history_stores_and_lists_captured_image(db):
    thumbnail = b"\xff\xd8scan preview\xff\xd9"
    scan_id = db.add_scan("ocr", "Expiration date saved", image_data=thumbnail)

    scan = db.list_scans()[0]

    assert scan["id"] == scan_id
    assert scan["image_data"] == thumbnail
    assert scan["preview_image"] == thumbnail


def test_scan_history_uses_linked_item_photo_when_scan_has_no_photo(db):
    thumbnail = b"\xff\xd8item preview\xff\xd9"
    item_id = db.insert_item(
        name="Apple",
        category="Fruits & Vegetables",
        image_data=thumbnail,
    )
    db.add_scan("detection", "Apple added", item_id=item_id)

    assert db.list_scans()[0]["preview_image"] == thumbnail


def test_existing_database_gets_scan_photo_column_migration(db):
    with db.connect() as conn:
        conn.execute("ALTER TABLE scan_history DROP COLUMN image_data")

    from modules.database_manager import DatabaseManager

    migrated_db = DatabaseManager(db.db_path)
    thumbnail = b"\xff\xd8migrated preview\xff\xd9"
    scan_id = migrated_db.add_scan("ocr", "Date saved", image_data=thumbnail)

    assert migrated_db.list_scans()[0]["id"] == scan_id
    assert migrated_db.list_scans()[0]["preview_image"] == thumbnail


def test_existing_database_gets_photo_column_migration(db):
    with db.connect() as conn:
        conn.execute("ALTER TABLE food_items DROP COLUMN image_data")

    from modules.database_manager import DatabaseManager

    migrated_db = DatabaseManager(db.db_path)
    item_id = migrated_db.insert_item(
        name="Cucumber",
        category="Fruits & Vegetables",
        image_data=b"jpeg",
    )

    assert migrated_db.get_item(item_id)["image_data"] == b"jpeg"


def test_multiple_dates_per_item_and_cascade(db):
    item_id = db.insert_item(name="Yogurt", category="Dairy")
    db.add_date(item_id, "manufacturing", "2026-10-01", "ocr", "MFG", "MFG 01/10/2026", 80, "verified")
    db.add_date(item_id, "expiration", "2026-12-01", "manual", verification_status="verified")
    dates = db.list_dates(item_id)
    assert [d["date_type"] for d in dates] == ["manufacturing", "expiration"]
    db.delete_item(item_id)
    assert db.list_dates(item_id) == []


def test_constraints_reject_bad_values(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_item(name="  ", category="Dairy")
    item_id = db.insert_item(name="Rice", category="Dry & Packaged")
    with pytest.raises(sqlite3.IntegrityError):
        db.add_date(item_id, "made_up_type", "2026-01-01", "manual")


def test_search_and_filters(db):
    db.insert_item(name="Apple", category="Fruits & Vegetables", storage_location="Fridge")
    db.insert_item(name="Pasta", category="Dry & Packaged", storage_location="Pantry")
    assert len(db.list_items(search="app")) == 1
    assert len(db.list_items(category="Dry & Packaged")) == 1
    assert len(db.list_items(location="Pantry")) == 1
    assert len(db.list_items()) == 2


def test_data_survives_reopening(tmp_path):
    from modules.database_manager import DatabaseManager
    path = tmp_path / "persist.db"
    DatabaseManager(path).insert_item(name="Bread", category="Bakery")
    assert DatabaseManager(path).list_items()[0]["name"] == "Bread"


def test_settings_roundtrip(db):
    assert db.get_setting("soon_days") is None
    db.set_setting("soon_days", "5")
    db.set_setting("soon_days", "7")
    assert db.get_setting("soon_days") == "7"


def test_scan_history(db):
    db.add_scan("ocr", "EXP read", {"raw": "EXP 05 JUN 2027"})
    assert db.list_scans()[0]["scan_type"] == "ocr"
