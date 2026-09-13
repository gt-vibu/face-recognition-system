"""
SQLite persistence for enrolled persons and their averaged embeddings.

Schema:
    person_id       INTEGER PRIMARY KEY
    name            TEXT UNIQUE
    embedding       BLOB    -- float32 reference: normalised sum of the enrollment embeddings
    num_samples     INTEGER -- how many images contributed to the reference
    created_at      TEXT    -- ISO timestamp
    thumbnail_path  TEXT    -- local-only path, never committed to git
    embedding_sum   BLOB    -- float64 un-normalised sum, for exact "add photos" updates
                            -- (NULL for rows enrolled before it existed: those are replace-only)

Enrollment photos themselves are never stored — only these vectors and one thumbnail.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional, Set, Tuple

import numpy as np

from . import config
from .enrollment import reference_from_sum

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    embedding BLOB NOT NULL,
    num_samples INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    thumbnail_path TEXT,
    embedding_sum BLOB
);
"""


class ReplaceOnlyEnrollmentError(Exception):
    """The person was enrolled before embedding sums were stored, so photos can't be added exactly."""


@contextmanager
def _connect():
    conn = sqlite3.connect(config.DB_PATH)
    try:
        conn.execute(SCHEMA)
        # Databases created before `embedding_sum` existed: add the column; old rows stay NULL.
        if "embedding_sum" not in {row[1] for row in conn.execute("PRAGMA table_info(persons)")}:
            conn.execute("ALTER TABLE persons ADD COLUMN embedding_sum BLOB")
        yield conn
        conn.commit()
    finally:
        conn.close()


def add_or_update_person(
    name: str,
    embedding: np.ndarray,
    num_samples: int,
    thumbnail_path: Optional[str] = None,
    embedding_sum: Optional[np.ndarray] = None,
) -> int:
    """Insert a new person, or replace the stored enrollment of an existing one with the same name."""
    blob = embedding.astype(np.float32).tobytes()
    sum_blob = None if embedding_sum is None else embedding_sum.astype(np.float64).tobytes()
    created_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        row = conn.execute("SELECT person_id FROM persons WHERE name = ?", (name,)).fetchone()
        if row:
            person_id = row[0]
            conn.execute(
                "UPDATE persons SET embedding=?, num_samples=?, created_at=?, thumbnail_path=?, embedding_sum=? "
                "WHERE person_id=?",
                (blob, num_samples, created_at, thumbnail_path, sum_blob, person_id),
            )
            return person_id
        cur = conn.execute(
            "INSERT INTO persons (name, embedding, num_samples, created_at, thumbnail_path, embedding_sum) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, blob, num_samples, created_at, thumbnail_path, sum_blob),
        )
        return cur.lastrowid


def add_photos_to_person(person_id: int, added_sum: np.ndarray, added_count: int) -> int:
    """
    Exact incremental enrollment: add new embeddings to a person's stored sum and count, then
    recompute the normalised reference. Returns the new sample count. The thumbnail is unchanged.
    Raises KeyError if the person doesn't exist, ReplaceOnlyEnrollmentError for pre-sum rows.
    """
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")  # read-modify-write as one transaction
        row = conn.execute(
            "SELECT embedding_sum, num_samples FROM persons WHERE person_id = ?", (person_id,)
        ).fetchone()
        if row is None:
            raise KeyError(person_id)
        if row[0] is None:
            raise ReplaceOnlyEnrollmentError(person_id)
        new_sum = np.frombuffer(row[0], dtype=np.float64) + added_sum.astype(np.float64)
        new_count = row[1] + added_count
        conn.execute(
            "UPDATE persons SET embedding=?, num_samples=?, embedding_sum=? WHERE person_id=?",
            (reference_from_sum(new_sum).tobytes(), new_count, new_sum.tobytes(), person_id),
        )
        return new_count


def ids_supporting_add_photos() -> Set[int]:
    """People whose stored sum allows exact "add photos" updates (everyone enrolled after the change)."""
    with _connect() as conn:
        return {r[0] for r in conn.execute("SELECT person_id FROM persons WHERE embedding_sum IS NOT NULL")}


def get_person_id(name: str) -> Optional[int]:
    with _connect() as conn:
        row = conn.execute("SELECT person_id FROM persons WHERE name = ?", (name,)).fetchone()
    return row[0] if row else None


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
