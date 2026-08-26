"""SQLite-backed persistence for sweep batches.

Without this, restarting the dashboard process drops all in-flight and
completed batch records -- users lose visibility into what ran and what
failed.  This module writes batch metadata and per-item state to a single
SQLite file (``<results_dir>/.sweeps.db``) so they survive restarts.

Design decisions:
- One table for batch headers, one for items.  Items are updated in-place as
  they transition from pending → running → completed/failed.
- WAL mode so the background sweep thread can write while the Flask request
  thread reads without contention.
- No ORM dependency -- plain sqlite3 from the standard library.
- The store is used *alongside* the existing in-memory SWEEP_BATCHES dict,
  not instead of it.  On startup the webapp rehydrates the in-memory dict
  from the store; all subsequent reads still hit memory.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

_DDL = """
CREATE TABLE IF NOT EXISTS sweep_batches (
    id          TEXT PRIMARY KEY,
    profile     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL   -- pending | running | done | interrupted
);

CREATE TABLE IF NOT EXISTS sweep_items (
    batch_id    TEXT NOT NULL REFERENCES sweep_batches(id),
    item_index  INTEGER NOT NULL,
    item_json   TEXT NOT NULL,  -- full item dict serialised as JSON
    PRIMARY KEY (batch_id, item_index)
);
"""


class SweepStore:
    """Thread-safe SQLite store for sweep batch state.

    All public methods acquire ``_lock`` and commit immediately so that the
    Flask thread always reads a consistent snapshot.
    """

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.executescript(_DDL)

    # ------------------------------------------------------------------
    # Write operations (called from the background sweep thread)
    # ------------------------------------------------------------------

    def save_batch(self, batch) -> None:  # type: ignore[type-arg]  # avoid circular import
        """Persist a new SweepBatch.  Call once, right after creation."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO sweep_batches (id, profile, created_at, status) VALUES (?,?,?,?)",
                    (batch.id, batch.profile_name, batch.created_at, batch.status),
                )
                for i, item in enumerate(batch.items):
                    conn.execute(
                        "INSERT OR REPLACE INTO sweep_items (batch_id, item_index, item_json) VALUES (?,?,?)",
                        (batch.id, i, json.dumps(item)),
                    )

    def update_item(self, batch_id: str, item_index: int, item: dict) -> None:
        """Overwrite one item's serialised state (called after each trial completes or fails)."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE sweep_items SET item_json=? WHERE batch_id=? AND item_index=?",
                    (json.dumps(item), batch_id, item_index),
                )

    def finish_batch(self, batch_id: str, status: str = "done") -> None:
        """Mark a batch's top-level status (called when the thread exits)."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE sweep_batches SET status=? WHERE id=?",
                    (status, batch_id),
                )

    def interrupt_running_batches(self) -> None:
        """Mark any batch still in 'running' state as 'interrupted'.

        Called on webapp startup -- a running batch from a previous process
        cannot be resumed without its background thread, so we surface it as
        interrupted rather than pretending it is still in progress.
        """
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE sweep_batches SET status='interrupted' WHERE status='running'",
                )

    # ------------------------------------------------------------------
    # Read operations (called from Flask request thread)
    # ------------------------------------------------------------------

    def load_all_batches(self) -> list[dict]:
        """Return all persisted batches with their items, newest first."""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id, profile, created_at, status FROM sweep_batches ORDER BY created_at DESC"
                ).fetchall()
                batches = []
                for row in rows:
                    items = [
                        json.loads(r["item_json"])
                        for r in conn.execute(
                            "SELECT item_json FROM sweep_items WHERE batch_id=? ORDER BY item_index",
                            (row["id"],),
                        ).fetchall()
                    ]
                    batches.append(
                        {
                            "id": row["id"],
                            "profile_name": row["profile"],
                            "created_at": row["created_at"],
                            "status": row["status"],
                            "items": items,
                        }
                    )
                return batches
