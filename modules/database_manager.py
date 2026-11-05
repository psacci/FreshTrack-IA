"""SQLite access layer. Only this module talks to the database directly."""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from config import settings

log = logging.getLogger(__name__)

ITEM_FIELDS = (
    "name", "category", "brand", "quantity", "unit", "storage_location",
    "date_added", "freshness_label", "freshness_confidence", "image_data", "notes",
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class DatabaseManager:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else settings.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    # ------------------------------------------------------------------ core
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        schema = settings.SCHEMA_PATH.read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(schema)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(food_items)")}
            if "image_data" not in columns:
                conn.execute("ALTER TABLE food_items ADD COLUMN image_data BLOB")
            if "status" not in columns:
                conn.execute(
                    "ALTER TABLE food_items ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
                )
            if "archived_at" not in columns:
                conn.execute("ALTER TABLE food_items ADD COLUMN archived_at TEXT")
            scan_columns = {row["name"] for row in conn.execute("PRAGMA table_info(scan_history)")}
            if "image_data" not in scan_columns:
                conn.execute("ALTER TABLE scan_history ADD COLUMN image_data BLOB")

    # ----------------------------------------------------------------- items
    def insert_item(self, **fields: Any) -> int:
        data = {k: fields.get(k) for k in ITEM_FIELDS}
        data["quantity"] = fields.get("quantity", 1)
        data["unit"] = fields.get("unit") or "pcs"
        data["storage_location"] = fields.get("storage_location") or "Fridge"
        data["date_added"] = fields.get("date_added") or date.today().isoformat()
        now = _now()
        cols = list(data) + ["created_at", "updated_at"]
        vals = list(data.values()) + [now, now]
        sql = f"INSERT INTO food_items ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        with self.connect() as conn:
            return conn.execute(sql, vals).lastrowid

    def update_item(self, item_id: int, **fields: Any) -> None:
        updates = {k: v for k, v in fields.items() if k in ITEM_FIELDS}
        if not updates:
            return
        updates["updated_at"] = _now()
        assignments = ", ".join(f"{k} = ?" for k in updates)
        with self.connect() as conn:
            conn.execute(f"UPDATE food_items SET {assignments} WHERE id = ?", [*updates.values(), item_id])

    def delete_item(self, item_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM food_items WHERE id = ?", (item_id,))

    def archive_item(self, item_id: int, status: str) -> None:
        if status not in {"consumed", "deleted"}:
            raise ValueError("An item can only be archived as consumed or deleted.")
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE food_items SET status = ?, archived_at = ?, updated_at = ? "
                "WHERE id = ? AND status = 'active'",
                (status, _now(), _now(), item_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Active item {item_id} does not exist.")

    def restore_item(self, item_id: int) -> None:
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE food_items SET status = 'active', archived_at = NULL, updated_at = ? "
                "WHERE id = ? AND status != 'active'",
                (_now(), item_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Archived item {item_id} does not exist.")

    def list_archived_items(self, limit: int = 100) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM food_items WHERE status != 'active' "
                "ORDER BY archived_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_item(self, item_id: int) -> Optional[dict]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM food_items WHERE id = ?", (item_id,)).fetchone()
        return dict(row) if row else None

    def list_items(
        self,
        search: str = "",
        category: Optional[str] = None,
        location: Optional[str] = None,
    ) -> list[dict]:
        clauses, params = [], []
        if search.strip():
            clauses.append("(name LIKE ? OR brand LIKE ?)")
            params += [f"%{search.strip()}%"] * 2
        if category:
            clauses.append("category = ?")
            params.append(category)
        if location:
            clauses.append("storage_location = ?")
            params.append(location)
        clauses.insert(0, "status = 'active'")
        where = f"WHERE {' AND '.join(clauses)}"
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM food_items {where} ORDER BY date_added DESC, id DESC", params
            ).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------------- dates
    def add_date(
        self,
        item_id: int,
        date_type: str,
        date_value: str,
        source: str = "manual",
        original_label: Optional[str] = None,
        original_text: Optional[str] = None,
        ocr_confidence: Optional[float] = None,
        verification_status: str = "unverified",
    ) -> int:
        with self.connect() as conn:
            return conn.execute(
                """INSERT INTO food_dates (item_id, date_type, date_value, original_label,
                   original_text, source, ocr_confidence, verification_status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (item_id, date_type, date_value, original_label, original_text,
                 source, ocr_confidence, verification_status, _now()),
            ).lastrowid

    def list_dates(self, item_id: Optional[int] = None) -> list[dict]:
        sql = "SELECT * FROM food_dates"
        params: list[Any] = []
        if item_id is not None:
            sql += " WHERE item_id = ?"
            params.append(item_id)
        sql += " ORDER BY date_value ASC"
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def set_date_status(self, date_id: int, status: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE food_dates SET verification_status = ? WHERE id = ?", (status, date_id))

    def delete_date(self, date_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM food_dates WHERE id = ?", (date_id,))

    # ----------------------------------------------------------------- scans
    def add_scan(
        self,
        scan_type: str,
        summary: str = "",
        detail: Optional[dict] = None,
        item_id: Optional[int] = None,
        status: str = "completed",
        image_data: Optional[bytes] = None,
    ) -> int:
        with self.connect() as conn:
            return conn.execute(
                """INSERT INTO scan_history
                   (scan_type, item_id, result_summary, detail_json, status, image_data, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (scan_type, item_id, summary, json.dumps(detail or {}, default=str), status, image_data, _now()),
            ).lastrowid

    def list_scans(self, limit: int = 50) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT scans.*, COALESCE(scans.image_data, items.image_data) AS preview_image
                   FROM scan_history AS scans
                   LEFT JOIN food_items AS items ON items.id = scans.item_id
                   ORDER BY scans.id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_scan_date(self, scan_id: int, scan_date: str) -> None:
        try:
            normalized_date = date.fromisoformat(scan_date).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Not a valid scan date: {scan_date!r}") from exc
        with self.connect() as conn:
            row = conn.execute(
                "SELECT created_at FROM scan_history WHERE id = ?", (scan_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Scan {scan_id} does not exist.")
            time_part = row["created_at"][10:]
            conn.execute(
                "UPDATE scan_history SET created_at = ? WHERE id = ?",
                (normalized_date + time_part, scan_id),
            )

    # -------------------------------------------------------------- settings
    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO app_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
