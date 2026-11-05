-- FreshTrack AI schema (SQLite)
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS food_items (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT    NOT NULL CHECK (length(trim(name)) > 0),
    category             TEXT    NOT NULL,
    brand                TEXT,
    quantity             REAL    NOT NULL DEFAULT 1 CHECK (quantity >= 0),
    unit                 TEXT    NOT NULL DEFAULT 'pcs',
    storage_location     TEXT    NOT NULL DEFAULT 'Fridge',
    date_added           TEXT    NOT NULL,            -- ISO date
    freshness_label      TEXT    CHECK (freshness_label IN ('fresh','aging','spoiled') OR freshness_label IS NULL),
    freshness_confidence REAL    CHECK (freshness_confidence BETWEEN 0 AND 1 OR freshness_confidence IS NULL),
    image_data           BLOB,
    notes                TEXT,
    status               TEXT    NOT NULL DEFAULT 'active' CHECK (status IN ('active','consumed','deleted')),
    archived_at          TEXT,
    created_at           TEXT    NOT NULL,
    updated_at           TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS food_dates (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id             INTEGER NOT NULL REFERENCES food_items(id) ON DELETE CASCADE,
    date_type           TEXT    NOT NULL CHECK (date_type IN
                          ('expiration','use_by','best_before','sell_by','manufacturing','packing')),
    date_value          TEXT    NOT NULL,             -- ISO date
    original_label      TEXT,                         -- label text as printed (e.g. "BEST BEFORE")
    original_text       TEXT,                         -- raw OCR text for the date
    source              TEXT    NOT NULL CHECK (source IN ('ocr','manual')),
    ocr_confidence      REAL    CHECK (ocr_confidence BETWEEN 0 AND 100 OR ocr_confidence IS NULL),
    verification_status TEXT    NOT NULL DEFAULT 'unverified'
                          CHECK (verification_status IN ('verified','unverified','rejected')),
    created_at          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_type      TEXT NOT NULL CHECK (scan_type IN ('detection','freshness','ocr')),
    item_id        INTEGER REFERENCES food_items(id) ON DELETE SET NULL,
    result_summary TEXT,
    detail_json    TEXT,
    status         TEXT NOT NULL DEFAULT 'completed',
    image_data     BLOB,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_category   ON food_items(category);
CREATE INDEX IF NOT EXISTS idx_items_location   ON food_items(storage_location);
CREATE INDEX IF NOT EXISTS idx_items_name       ON food_items(name);
CREATE INDEX IF NOT EXISTS idx_dates_item       ON food_dates(item_id);
CREATE INDEX IF NOT EXISTS idx_dates_value      ON food_dates(date_value);
CREATE INDEX IF NOT EXISTS idx_scans_created    ON scan_history(created_at);
