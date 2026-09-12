"""
SQLite persistence for enrolled persons and their averaged embeddings.

Schema:
    person_id       INTEGER PRIMARY KEY
    name            TEXT UNIQUE
    embedding       BLOB    -- float32 numpy array, averaged across enrollment images
    num_samples     INTEGER -- how many images contributed to the average
    created_at      TEXT    -- ISO timestamp
    thumbnail_path  TEXT    -- local-only path, never committed to git
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    embedding BLOB NOT NULL,
    num_samples INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    thumbnail_path TEXT
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(config.DB_PATH)
    try:
        conn.execute(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def add_or_update_person(
    name: str,
    embedding: np.ndarray,
    num_samples: int,
    thumbnail_path: Optional[str] = None,
) -> int:
    """Insert a new person, or update an existing one with the same name."""
    blob = embedding.astype(np.float32).tobytes()
    created_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        row = conn.execute("SELECT person_id FROM persons WHERE name = ?", (name,)).fetchone()
        if row:
            person_id = row[0]
            conn.execute(
                "UPDATE persons SET embedding=?, num_samples=?, created_at=?, thumbnail_path=? "
                "WHERE person_id=?",
                (blob, num_samples, created_at, thumbnail_path, person_id),
            )
            return person_id
        cur = conn.execute(
            "INSERT INTO persons (name, embedding, num_samples, created_at, thumbnail_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, blob, num_samples, created_at, thumbnail_path),
        )
        return cur.lastrowid


def get_all_persons() -> List[Tuple[int, str, np.ndarray, int, str, Optional[str]]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT person_id, name, embedding, num_samples, created_at, thumbnail_path "
            "FROM persons ORDER BY name"
        ).fetchall()

    result = []
    for person_id, name, blob, num_samples, created_at, thumb in rows:
        embedding = np.frombuffer(blob, dtype=np.float32)
        result.append((person_id, name, embedding, num_samples, created_at, thumb))
    return result


def get_person_embeddings_for_matching() -> List[Tuple[int, str, np.ndarray]]:
    return [(person_id, name, embedding) for person_id, name, embedding, *_ in get_all_persons()]


def delete_person(person_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM persons WHERE person_id = ?", (person_id,))


def name_exists(name: str) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM persons WHERE name = ?", (name,)).fetchone()
    return row is not None
