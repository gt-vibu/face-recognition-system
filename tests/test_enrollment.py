"""
Tests for new / add-photos / replace enrollment, run through the real API functions and SQLite code
against a temporary database. Face detection is replaced by fake embeddings (no model is loaded).
"""
import io
import os
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api  # noqa: E402
from src import config, database  # noqa: E402
from src.enrollment import embedding_sum, reference_from_sum  # noqa: E402
from src.matching import MatchStatus, best_match  # noqa: E402

RNG = np.random.default_rng(0)
REAL_LOAD = api._load  # the real image decoder, kept before the fixture replaces it


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


PERSON_A = unit(RNG.normal(size=512))
PERSON_B = unit(RNG.normal(size=512))


def photo_of(direction, noise=0.5):
    """A fake single-face photo: the person's direction plus random noise (similarity ~0.9), normalised."""
    return SimpleNamespace(filename="face.jpg", faces=[unit(direction + noise * RNG.normal(size=512) / np.sqrt(512))])


def no_face_photo():
    return SimpleNamespace(filename="wall.jpg", faces=[])


def two_face_photo():
    return SimpleNamespace(filename="group.jpg", faces=[PERSON_A, PERSON_B])


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setattr(config, "THUMB_DIR", str(tmp_path))
    # Stand-in for api._load: returns an image plus fake detected faces carrying the embeddings.
    monkeypatch.setattr(
        api, "_load",
        lambda f: (Image.new("RGB", (8, 8)), None, [SimpleNamespace(embedding=e) for e in f.faces]),
    )


def stored(person_id):
    with sqlite3.connect(config.DB_PATH) as conn:
        ref, count, total = conn.execute(
            "SELECT embedding, num_samples, embedding_sum FROM persons WHERE person_id=?", (person_id,)
        ).fetchone()
    return np.frombuffer(ref, np.float32), count, (None if total is None else np.frombuffer(total, np.float64))


def embeddings(photos):
    return [p.faces[0] for p in photos]


# 1. New person enrollment still works (same reference direction as the old normalised mean).
def test_new_enrollment_stores_sum_count_and_normalised_reference():
    photos = [photo_of(PERSON_A), photo_of(PERSON_A)]
    res = api.enroll(name="Alice", files=photos, mode="new")
    ref, count, total = stored(res["person_id"])
    assert res["num_samples"] == count == 2 and res["updated"] is False
    np.testing.assert_allclose(total, embedding_sum(embeddings(photos)))
    old_method = np.mean(embeddings(photos), axis=0)
    np.testing.assert_allclose(ref, old_method / np.linalg.norm(old_method), atol=1e-6)


# 2 + 4 + 5. Add exactly one photo: count +1, reference normalised and exactly as if enrolled together.
def test_add_exactly_one_photo_is_an_exact_update():
    first = [photo_of(PERSON_A), photo_of(PERSON_A)]
    pid = api.enroll(name="Alice", files=first, mode="new")["person_id"]
    extra = [photo_of(PERSON_A)]
    res = api.add_photos(pid, extra)
    ref, count, total = stored(pid)
    assert res["added"] == 1 and res["num_samples"] == count == 3
    np.testing.assert_allclose(total, embedding_sum(embeddings(first + extra)))
    np.testing.assert_allclose(ref, reference_from_sum(embedding_sum(embeddings(first + extra))), atol=1e-7)
    assert abs(np.linalg.norm(ref) - 1.0) < 1e-6


# 3 + 4. Add several photos, in more than one batch — the count keeps adding up.
def test_add_multiple_photos_in_batches():
    pid = api.enroll(name="Alice", files=[photo_of(PERSON_A), photo_of(PERSON_A)], mode="new")["person_id"]
    assert api.add_photos(pid, [photo_of(PERSON_A) for _ in range(3)])["num_samples"] == 5
    assert api.add_photos(pid, [photo_of(PERSON_A)])["num_samples"] == 6
    ref, count, _ = stored(pid)
    assert count == 6 and abs(np.linalg.norm(ref) - 1.0) < 1e-6
    assert [p["num_samples"] for p in api.list_persons()] == [6]


# 6. Recognition works before and after adding photos (and strangers stay Unknown).
def test_recognition_before_and_after_adding_photos():
    pid = api.enroll(name="Alice", files=[photo_of(PERSON_A), photo_of(PERSON_A)], mode="new")["person_id"]
    query = photo_of(PERSON_A).faces[0]
    before = best_match(query, database.get_person_embeddings_for_matching())
    api.add_photos(pid, [photo_of(PERSON_A), photo_of(PERSON_A)])
    after = best_match(query, database.get_person_embeddings_for_matching())
    assert before.status == after.status == MatchStatus.CONFIRMED and after.person_name == "Alice"
    assert after.similarity >= before.similarity - 0.05  # more samples: the reference stays on the person
    stranger = best_match(PERSON_B, database.get_person_embeddings_for_matching())
    assert stranger.status == MatchStatus.UNKNOWN


# 7. New-person enrollment still needs at least MIN (2) valid photos.
def test_new_enrollment_rejects_fewer_than_two_valid_photos():
    with pytest.raises(HTTPException) as e:
        api.enroll(name="Alice", files=[photo_of(PERSON_A), no_face_photo()], mode="new")
    assert e.value.status_code == 400
    assert api.list_persons() == []


# 8. No-face, multi-face and unreadable photos are rejected on both paths.
def test_invalid_photos_are_rejected():
    res = api.enroll(name="Alice", files=[photo_of(PERSON_A), two_face_photo(), no_face_photo(), photo_of(PERSON_A)], mode="new")
    assert [f["accepted"] for f in res["files"]] == [True, False, False, True] and res["num_samples"] == 2
    with pytest.raises(HTTPException) as e:  # adding: nothing usable -> rejected, count unchanged
        api.add_photos(res["person_id"], [no_face_photo(), two_face_photo()])
    assert e.value.status_code == 400 and stored(res["person_id"])[1] == 2
    unreadable = SimpleNamespace(filename="notes.txt", file=io.BytesIO(b"not an image"))
    with pytest.raises(HTTPException) as e:  # the real decoder rejects it before any face detection
        REAL_LOAD(unreadable)
    assert e.value.status_code == 400


# 9. Replace is explicit: "new" refuses an existing name, "replace" overwrites sum and count.
def test_replace_is_explicit_and_overwrites_the_enrollment():
    first = [photo_of(PERSON_A), photo_of(PERSON_A)]
    pid = api.enroll(name="Alice", files=first, mode="new")["person_id"]
    api.add_photos(pid, [photo_of(PERSON_A)])
    with pytest.raises(HTTPException) as e:
        api.enroll(name="Alice", files=[photo_of(PERSON_A), photo_of(PERSON_A)], mode="new")
    assert e.value.status_code == 409 and stored(pid)[1] == 3  # nothing replaced silently
    fresh = [photo_of(PERSON_A), photo_of(PERSON_A)]
    res = api.enroll(name="Alice", files=fresh, mode="replace")
    ref, count, total = stored(pid)
    assert res["updated"] is True and res["person_id"] == pid and count == 2
    np.testing.assert_allclose(total, embedding_sum(embeddings(fresh)))
    with pytest.raises(HTTPException) as e:
        api.enroll(name="Nobody", files=fresh, mode="replace")
    assert e.value.status_code == 404


# Existing databases: rows from before `embedding_sum` are kept, flagged replace-only, and never guessed.
def test_legacy_rows_are_replace_only_until_re_enrolled():
    with sqlite3.connect(config.DB_PATH) as conn:  # the schema as it was before this change
        conn.execute(
            "CREATE TABLE persons (person_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, "
            "embedding BLOB NOT NULL, num_samples INTEGER NOT NULL, created_at TEXT NOT NULL, thumbnail_path TEXT)"
        )
        conn.execute(
            "INSERT INTO persons (name, embedding, num_samples, created_at) VALUES (?, ?, 3, '2026-09-12')",
            ("Legacy", PERSON_A.tobytes()),
        )
    [person] = api.list_persons()  # opening the DB adds the column; the old row gets NULL
    assert person["can_add_photos"] is False and person["num_samples"] == 3
    assert best_match(PERSON_A, database.get_person_embeddings_for_matching()).status == MatchStatus.CONFIRMED
    with pytest.raises(HTTPException) as e:
        api.add_photos(person["person_id"], [photo_of(PERSON_A)])
    assert e.value.status_code == 409 and stored(person["person_id"])[1] == 3
    api.enroll(name="Legacy", files=[photo_of(PERSON_A), photo_of(PERSON_A)], mode="replace")
    assert api.list_persons()[0]["can_add_photos"] is True
    assert api.add_photos(person["person_id"], [photo_of(PERSON_A)])["num_samples"] == 3
