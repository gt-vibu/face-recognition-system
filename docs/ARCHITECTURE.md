# Architecture

## System overview

The project started as a single-process Streamlit application (Decision 001). Once that version was working and tested,
a Next.js UI was added on top of a thin FastAPI layer (Decision 008). Both UIs call the same `src/` code and share one
SQLite database; the Streamlit app is kept as an optional UI.

```
Next.js UI (frontend/)            Streamlit UI (app.py, pages/)
        ↓ /api/* proxied by Next.js           │
FastAPI (api.py) — stateless                  │
        └──────────────┬──────────────────────┘
                       ↓
src/embeddings.py   — InsightFace detection + ArcFace embedding
src/matching.py     — cosine similarity + tiered decision policy
src/database.py     — SQLite persistence
src/config.py       — thresholds, paths, retry limits
                       ↓
SQLite file (data/face_db.sqlite3)
```

## Responsibility boundaries

- **UI layer (`frontend/` Next.js; `app.py`, `pages/*.py` Streamlit)**: presentation, uploads, the per-session retry counter, and the evaluation threshold slider. No ML or SQL logic lives here — the Next.js UI never makes a recognition decision itself.
- **API layer (`api.py`)**: stateless FastAPI endpoints (status, persons, detect, enroll, identify) that call into `src/`. It passes the optional evaluation `confirmed_threshold` straight to `best_match`.
- **ML layer (`src/embeddings.py`)**: the only file that touches InsightFace. Detection and embedding extraction happen in one call since InsightFace's `FaceAnalysis.get()` does both in a single forward pass.
- **Matching layer (`src/matching.py`)**: pure functions — cosine similarity and threshold logic — no I/O, fully unit-testable without any ML dependency installed (see `tests/test_matching.py`).
- **Persistence layer (`src/database.py`)**: all SQL lives here; no other file opens a `sqlite3` connection directly.
- **Config (`src/config.py`)**: the single source of truth for thresholds, retry limits, and paths — every other module imports from here rather than hardcoding values.

## Data flow — enrollment

```
User uploads N images
  → detect_faces() per image
  → reject images with 0 or >1 faces
  → sum accepted embeddings (src/enrollment.py), normalise the sum → reference
  → store reference + embedding sum + count (never the photos)
  (add photos later: sum += new embeddings, count += n, reference re-normalised — exact)
  → database.add_or_update_person()
```

## Data flow — identification

```
User uploads 1 image
  → detect_faces() (may return multiple faces)
  → for each face: cosine_similarity vs every enrolled embedding
  → best_match() applies the tiered Confirmed/Uncertain/Unknown policy
    (optionally with the evaluation-only Confirmed threshold from the slider)
  → the UI counts an Uncertain result once per new upload, capped at MAX_IDENTIFICATION_ATTEMPTS
```

## Why no vector database / FAISS

At the scale this assignment targets (single-digit to low-tens of enrolled people), a linear scan over stored embeddings in `database.get_person_embeddings_for_matching()` is fast enough (milliseconds) and far simpler to reason about and debug than standing up an ANN index. This is called out explicitly in the README's "Future improvements" section as the correct next step at larger scale — not implemented now because it adds complexity with no present benefit.

## Error handling

- No face detected → inline error, no crash, no partial state written.
- Multiple faces during enrollment → rejected per-image with an explanation; enrollment only proceeds using unambiguous single-face images.
- Multiple faces during identification → each face processed and reported independently.
- Uncertain match → never silently promoted to a confirmed identity; explicit retry flow with a hard cap.
- Backend/model load failure (e.g. InsightFace weights fail to download) → surfaces as a Streamlit exception, or the API fails to start and the Next.js UI shows "Recognition API not reachable"; never silently swallowed, since silently returning fabricated results would be worse than a visible error.
- Unreadable upload or out-of-range evaluation threshold → the API returns HTTP 400 with a message.

## Security / privacy

See README "Privacy considerations." Practically enforced via `.gitignore` (excludes `data/` and `evaluation_data/`) rather than relying on developer discipline alone.

## Scaling considerations (not implemented, documented for completeness)

- Beyond a few hundred enrolled people: swap the linear scan in `matching.best_match` for an ANN index (FAISS `IndexFlatIP` is a drop-in start since embeddings are already L2-normalized, making inner product equivalent to cosine similarity).
- Beyond single-machine usage: SQLite would need to become a proper client-server database (e.g. Postgres) to support concurrent writers safely.
