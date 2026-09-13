# Face Recognition Identification System

A local, offline face enrollment and identification system built with ArcFace embeddings and cosine similarity —
developed for the Code Nimbus Solutions AI/ML Internship assignment.

## Overview

Enroll people from a few photos each, then identify faces in new photos against that enrolled database. Every face
gets one of three decisions — **Confirmed Match**, **Uncertain Match** or **Unknown** — instead of always returning the
closest name, so the system can honestly say "I'm not sure" or "I don't know this person."

Everything runs locally: no cloud APIs, no external calls, $0 cost.

## Architecture

The main application is a **Next.js** web UI backed by a small **FastAPI** service that calls the Python ML code
directly. The original **Streamlit** app is still included and uses the same code and database.

```
Next.js UI (frontend/)          Streamlit UI (app.py, pages/) — optional
   Dashboard · Enroll ·            Enroll · Identify · People
   Identify · People                        │
        │  /api/* (proxied by Next.js)      │
        ▼                                   │
FastAPI (api.py) — stateless HTTP layer     │
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
src/embeddings.py   SCRFD face detection + ArcFace embedding (InsightFace buffalo_l, ONNX Runtime, CPU)
src/matching.py     cosine similarity + Confirmed / Uncertain / Unknown decision
src/enrollment.py   enrollment maths (embedding sum → normalised reference; exact "add photos")
src/database.py     SQLite persistence (data/face_db.sqlite3)
src/config.py       thresholds, retry limit, enrollment limits, paths
```

- The browser only talks to Next.js; `next.config.ts` proxies `/api/*` to FastAPI, so no CORS setup is needed.
- `api.py` contains no recognition logic of its own — it calls the same `src/` functions as the Streamlit pages.
- Both UIs share `data/face_db.sqlite3`, so people enrolled in one appear in the other.

See `docs/ARCHITECTURE.md` and `docs/ML_PIPELINE.md` for more detail.

## ML pipeline

```
Image → SCRFD face detection → align → ArcFace embedding (512-d, L2-normalised)
   Enrollment:     2–5 photos, one face each → sum the embeddings → normalise → store reference + sum + count in SQLite
   Add photos:     1–5 more photos later → add to the stored sum and count → re-normalise (exact update)
   Identification: each detected face → cosine similarity vs every enrolled person → best match → decision
```

- **Model:** InsightFace `buffalo_l` (SCRFD detector + ArcFace recognition model), pretrained, inference only — no training.
- **Metric:** cosine similarity. We call it **similarity, never "confidence"** — it is not a calibrated probability.

## Recognition decision policy

| Similarity | Decision | Behaviour |
|---|---|---|
| `≥ 0.62` (`CONFIRMED_THRESHOLD`) | **Confirmed Match** | Shows the person's name and similarity. |
| `0.45 – 0.62` (`UNCERTAIN_LOWER_BOUND`–`CONFIRMED_THRESHOLD`) | **Uncertain Match** | The closest candidate is shown for context only — never as a confirmed identity. Asks for another photo. |
| `< 0.45` | **Unknown** | No enrolled identity matched. Offers "Enroll this person" — never creates an identity automatically. |

The values live in `src/config.py`. 0.62 is a deliberately conservative operating point: on a 50-person sample of
the real-photo LFW dataset it gave **90.2% confident identification with zero wrong-person confirmations and zero
strangers confirmed**; the other 9.8% were Uncertain (retry). Lower thresholds scored higher on LFW but produced
wrong-person confirmations on synthetic look-alikes, so borderline cases go to Uncertain instead. See
`docs/EVALUATION.md` for the numbers and reasoning.

### Bounded retries for Uncertain matches

- An Uncertain result uses **one attempt per newly uploaded photo**; re-renders or reruns never count.
- Confirmed and Unknown results reset the counter; a photo with no face does not use an attempt.
- After **3** Uncertain attempts (`MAX_IDENTIFICATION_ATTEMPTS`) the flow locks with
  "Identity could not be confidently verified" until **Try again**.
- The counter lives in the browser (React state in Next.js, `session_state` in Streamlit) — reloading the page resets it.

### Evaluation threshold slider (testing only)

Both UIs have a slider to try other Confirmed thresholds without editing `config.py`:

- **Next.js:** Identify page → **"Evaluation threshold — testing only"** (bottom of the panel).
- **Streamlit:** Identify page → **"Advanced — evaluation threshold (testing only)"**.

It ranges from 0.45 to 0.90 (default 0.62) and only moves the **Confirmed** boundary; the Uncertain lower bound stays
0.45. In Next.js the value is sent with the next uploaded photo (`confirmed_threshold` form field) and FastAPI passes it
to `src/matching.best_match` — the decision is always made by the backend. A notice is shown while it differs from the
default. It lasts for the browser tab only and never changes `config.py`.

## Installation

Requirements: **Python 3.10 or 3.11** (newer/pre-release versions may lack InsightFace/ONNX Runtime wheels) and
**Node.js 20+**.

```bash
git clone https://github.com/gt-vibu/face-recognition-system.git
cd face-recognition-system
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm install
```

The first run downloads the InsightFace `buffalo_l` weights (~300 MB) to `~/.insightface` — internet is needed only
for this one-time download.

**Windows note:** `pip install insightface` may need to compile native code. If it fails with a compiler error,
install the free **Microsoft C++ Build Tools** (workload "Desktop development with C++") and run
`pip install -r requirements.txt` again.

## Running

Start the API (from the repository root, with the virtual environment active):

```bash
uvicorn api:app --port 8000
```

Start the web UI (in a second terminal):

```bash
cd frontend
npm run dev
```

Open **http://localhost:3000**. If port 3000 is busy, Next.js picks the next free port. To point the UI at an API on
a different address, set `FACE_API_URL` (default `http://127.0.0.1:8000`).

Optional — the original Streamlit UI (no API needed):

```bash
streamlit run app.py
```

### Using the app

- **Enroll:** add 2–5 photos of the same person (one face per photo), enter a name, **Save person**. **3 photos with
  slightly different angles or lighting are recommended** — on LFW this raised confident identification from 90.2%
  (2 photos) to 95.9% at the same threshold.
- **Add photos later / replace:** typing a name that is already enrolled asks what saving should do —
  **Add photos** (1–5 more photos update that person's stored reference; the default) or **Replace enrollment**
  (discards it and re-enrolls from the new photos, 2–5 needed). Nothing is ever replaced silently. The original
  enrollment photos are never stored — only the face embeddings and one thumbnail.
- **Identify:** upload a photo; every detected face gets its own Confirmed / Uncertain / Unknown result with its similarity.
- **People:** see everyone enrolled, with photo, sample count and date; **Add photos** to someone (opens Enroll with
  their name filled in); delete a person (with confirmation).

## API

| Method & path | Purpose |
|---|---|
| `GET /api/status` | Thresholds, limits, enrolled count, slider range |
| `GET /api/persons` | List enrolled people (incl. `can_add_photos`) |
| `GET /api/persons/{id}/thumbnail` | A person's thumbnail |
| `DELETE /api/persons/{id}` | Delete a person |
| `POST /api/detect` | Count faces in one photo (Enroll page check) |
| `POST /api/enroll` | `name` + `files` + `mode` (`new` — refused if the name exists — or `replace`) → enroll or re-enroll a person |
| `POST /api/persons/{id}/photos` | `files` (1–5) → add photos to an existing person's enrollment |
| `POST /api/identify` | `file` (+ optional `confirmed_threshold`, 0.45–0.90) → per-face decision, similarity and face crop (numbered left to right), plus the photo with status-coloured boxes |

## Evaluation

**Real photos — LFW, 50 people × 5 photos** (2-photo enrollment, default thresholds 0.62 / 0.45):

| Metric | Result |
|---|---|
| Samples | 1,101 genuine identifications · 39,560 impostor comparisons · 1,839 unknown-person tests |
| Correctly Confirmed | **90.2%** |
| Uncertain (asked for another photo) | 9.8% |
| Confirmed as the wrong person | **0** |
| Unknown person Confirmed (false accept) | **0** |
| Similarity: genuine range / highest impostor | 0.456 – 0.895 / 0.313 (no overlap) |
| With 3 enrollment photos | 95.9% Confirmed, still 0 wrong-person / 0 false accepts |

**In-app validation** (one real person + AI-generated faces): the real person was Confirmed at 0.95 on the enrollment
photos and 0.77–0.95 on 24 of 26 edited variants (lighting, blur, rotation, occlusion; the 2 tight crops had no face
detected), while every other face scored ≤ 0.13 against them. The failures were a head-turned photo (0.42 → Unknown)
and 4 false accepts at 0.62, all from AI-generated faces the generator had duplicated.

Run `python evaluation/run_evaluation.py` (see its docstring for the `evaluation_data/` layout). `docs/EVALUATION.md`
has the real-person results (50 LFW people, 250 photos), the in-app validation (one real person plus AI-generated
faces), the threshold sweep, robustness tests and failure cases — and the caveat that the synthetic faces include
duplicated identities.

## Tests

```bash
pytest                                   # matching/decision policy, enrollment (new/add/replace), display helpers
cd frontend && npm run typecheck && npm run build
```

## Privacy considerations

- All processing is local; no image or embedding is sent to an external service.
- Enrollment photos are not retained (also when adding photos later): only the face embeddings (reference, running
  sum, count) and one thumbnail per person are stored.
- The database and thumbnails (`data/`) and evaluation photos (`evaluation_data/`) are git-ignored.
- The API listens on `127.0.0.1` and has no authentication — do not expose it on a network.
- Deleting a person removes their embedding; their thumbnail file currently stays in `data/thumbnails/`.
- This is an assignment/demo project, not a production biometric system: there is no liveness / anti-spoofing check.

## Failure cases

How the system behaves when recognition cannot or should not succeed (details, scores and test results are in
`docs/EVALUATION.md`, sections 5 and 7):

| Case | Behaviour |
|---|---|
| **No face detected** (blank image, background only, very tight crop) | "No face detected. Please upload a clearer image." — no result is guessed and no retry attempt is used. |
| **Multiple faces** | Identify: every face gets its own independent result. Enroll: a photo with more than one face is rejected ("Multiple faces"); saving a new person needs at least 2 single-face photos (1 when adding photos to an existing person). |
| **Unknown / low similarity** (< 0.45) | "No enrolled identity matched" with an **Enroll this person** action — an identity is never created automatically. |
| **Uncertain match** (0.45 – 0.62) | The closest candidate is shown for context only, never as a confirmed identity, and another photo is requested. Each new photo uses one attempt; after 3 the flow locks ("Identity could not be confidently verified") until **Try again**. |
| **Difficult pose / poor conditions** | Large head turns can drop below 0.45 (a head-turned photo scored 0.42 → Unknown). Blur, darker/brighter lighting and covered eyes or mouth lower the similarity; on the real face these variants still matched (0.77–0.95). Very small, low-resolution faces fall to Unknown rather than a wrong match. |
| **Look-alike faces** | The main false-accept risk: in testing, near-duplicate AI-generated faces were confirmed as each other. No wrong-person confirmations occurred on the 50 real LFW people. |
| **Invalid or non-image upload** | Only JPG/PNG can be selected ("Only JPG or PNG images are supported"); the API rejects unreadable files with HTTP 400. |
| **Empty database** | Every face is reported as Unknown (similarity 0.00) until someone is enrolled. |

## Limitations

- Evaluated on 50 real LFW people (mostly frontal press photos, not look-alikes), one real person's selfies and
  AI-generated faces — not yet on phone/webcam photos, look-alikes, twins or relatives.
- Large head turns can be missed (0.42 → Unknown in testing); tightly cropped faces are not detected.
- Look-alike faces are the main false-accept risk.
- **Add photos** trusts the user: new photos are not checked against the person's existing reference, so adding
  someone else's photo would dilute it (Replace enrollment fixes it). People enrolled before adding photos existed
  have no stored sum and are replace-only until re-enrolled once.
- Linear-scan matching — fine for a small number of people; FAISS or similar would be needed at scale.

## Future improvements

- A larger real evaluation (full LFW protocol, look-alikes, phone/webcam photos) before re-tuning the thresholds.
- Liveness / anti-spoofing detection; delete thumbnails together with the person.
- FAISS for larger galleries; optional webcam capture.

## Tech stack

Python, InsightFace (SCRFD + ArcFace), ONNX Runtime, OpenCV, NumPy, SQLite, FastAPI · Next.js, React, TypeScript,
Tailwind CSS, shadcn/ui (Base UI) · Streamlit (optional UI).

## Project structure

```
face-recognition-system/
├── api.py                  # FastAPI service used by the Next.js UI
├── app.py                  # Streamlit home page (optional UI)
├── pages/                  # Streamlit Enroll / Identify / People
├── src/
│   ├── config.py           # thresholds, retry limit, enrollment limits, paths
│   ├── embeddings.py       # SCRFD + ArcFace (InsightFace) wrapper
│   ├── enrollment.py       # enrollment maths: embedding sum → normalised reference
│   ├── matching.py         # cosine similarity + three-tier decision
│   └── database.py         # SQLite persistence
├── frontend/               # Next.js app (Dashboard, Enroll, Identify, People)
│   ├── app/                # pages
│   ├── components/         # shared UI components
│   └── lib/api.ts          # typed client for api.py
├── evaluation/run_evaluation.py
├── tests/test_matching.py
├── docs/                   # ARCHITECTURE, ML_PIPELINE, EVALUATION
├── data/                   # git-ignored: database + thumbnails
└── evaluation_data/        # git-ignored: evaluation photos
```
