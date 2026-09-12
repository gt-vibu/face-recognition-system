# Face Recognition Identification System

A local, offline face enrollment and identification system built with ArcFace embeddings and cosine similarity — developed for the Code Nimbus Solutions AI/ML Internship assignment.

## Overview

The system lets you enroll people from a few photos each, and later identify new photos against that enrolled database. Every decision is graded into one of three tiers — **Confirmed Match**, **Uncertain Match**, or **Unknown** — rather than a naive "always return the closest name" approach, so the system can honestly say "I'm not sure" or "I don't know this person."

Everything runs locally: no cloud APIs, no external calls, $0 cost.

## Features

- Face detection + ArcFace embedding generation (InsightFace, CPU-only)
- Cosine-similarity based 1:N matching against enrolled identities
- Tiered decision policy: **Confirmed / Uncertain / Unknown** (see below)
- Bounded retry policy for uncertain matches (no infinite retry loops)
- Multi-face identification (each face in an image handled independently)
- Enrollment, identification, and people-management pages (Streamlit)
- Delete-enrolled-person support
- Basic evaluation script producing genuine/impostor/unknown score distributions and FAR/FRR

## Architecture

```
Image upload
     ↓
Face detection (InsightFace / SCRFD)
     ↓
Crop + align
     ↓
ArcFace embedding (512-d, L2-normalized)
     ↓
   ┌─────────────┴─────────────┐
   ↓                           ↓
Enrollment path           Identification path
(average embeddings,      (cosine similarity vs.
 store in SQLite)          every stored embedding)
                                ↓
                      Best match + tiered threshold check
                                ↓
                 Confirmed Match / Uncertain Match / Unknown
```

See `docs/ARCHITECTURE.md` and `docs/ML_PIPELINE.md` for more detail.

## Model used

**InsightFace `buffalo_l`** model pack — a SCRFD detector paired with an ArcFace recognition model, run via ONNX Runtime on CPU. This is a widely-used, modern, fully offline, pretrained face-recognition approach; no training was performed (see `docs/ML_PIPELINE.md` for why that's the correct approach here).

## Similarity metric & terminology

**Cosine similarity** between L2-normalized embeddings. We deliberately call this "similarity," never "confidence" — cosine similarity is not a calibrated probability, and labeling it a confidence percentage would misrepresent what the number means.

## Recognition decision policy

| Similarity range | Status | Behavior |
|---|---|---|
| `>= CONFIRMED_THRESHOLD` | **Confirmed Match** | Show the person's name and score. No retry needed. |
| `[UNCERTAIN_LOWER_BOUND, CONFIRMED_THRESHOLD)` | **Uncertain Match** | Do NOT present a confirmed identity. Ask for another image. Bounded to `MAX_IDENTIFICATION_ATTEMPTS` (default 3) attempts total. |
| `< UNCERTAIN_LOWER_BOUND` | **Unknown** | No enrolled identity matched. Offer an "Enroll This Person" action — never auto-creates an identity. |

The two threshold values live in `src/config.py` as clearly-commented constants — they are **starting defaults**, not scientifically fixed values, and must be validated against your own evaluation data before being trusted (see "Threshold selection" below).

## Threshold selection

1. Run `evaluation/run_evaluation.py` against a small self-collected dataset (see its docstring for the expected folder layout).
2. Look at where genuine scores cluster vs. where impostor/unknown scores cluster.
3. Pick `CONFIRMED_THRESHOLD` above the impostor cluster and `UNCERTAIN_LOWER_BOUND` a bit further below it, leaving a genuine "uncertain" band in between.
4. Update `src/config.py` with the validated values and record the reasoning in `docs/DECISIONS.md`.

**Actual results from this run:** see `docs/EVALUATION.md` — filled in only after actually running the evaluation script, never fabricated.

## Installation

```bash
git clone <this-repo-url>
cd face-recognition-system
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run will download the InsightFace `buffalo_l` model weights automatically (requires internet only for this one-time download).

## Usage

```bash
streamlit run app.py
```

- **Enroll:** go to the Enroll page, enter a name, upload 2–5 clear single-person photos.
- **Identify:** go to the Identify page, upload a photo. Each detected face gets its own Confirmed/Uncertain/Unknown result.
- **People:** view and delete enrolled identities.

## Evaluation

See `docs/EVALUATION.md` for the dataset description, method, and actual results (accuracy, FAR, FRR) — run via `python evaluation/run_evaluation.py`.

## Failure cases

Documented in `docs/ML_PIPELINE.md`, including: no face detected, multiple faces, poor lighting, side profiles, occlusion (glasses/masks), low resolution, visually similar people, and duplicate enrollment attempts — each with expected vs. observed behavior and a possible improvement.

## Privacy considerations

- All processing happens locally; no image or embedding is ever sent to an external service.
- Face images, thumbnails, and the SQLite database are stored under `data/`, which is excluded from git via `.gitignore`.
- Any evaluation photos live under `evaluation_data/`, also git-ignored.
- A "Delete Person" action removes an enrolled identity's stored data.
- This is a demo/assignment project, not a production biometric system — it has no liveness/anti-spoofing detection and should not be used for real access-control decisions without further security review.

## Limitations

- Small evaluation dataset (see `docs/EVALUATION.md`) — thresholds are validated against a handful of people, not a large benchmark.
- Sensitive to poor lighting, heavy occlusion, and extreme pose.
- No liveness detection (a photo of a photo could in principle be presented).
- Linear-scan matching — fine for a handful of enrolled people, would need an approximate-nearest-neighbor index (e.g. FAISS) at larger scale.

## Future improvements

- Larger, more diverse evaluation dataset with a proper FAR/FRR-vs-threshold curve.
- Liveness/anti-spoofing detection.
- FAISS or similar for scaling beyond a few hundred enrolled identities.
- Optional webcam capture in addition to image upload.

## Tech stack

Python, InsightFace (ArcFace + SCRFD), ONNX Runtime, OpenCV, NumPy, SQLite, Streamlit.

## Project structure

```
face-recognition-system/
├── app.py                    # Home / dashboard
├── requirements.txt
├── README.md
├── .gitignore
├── pages/
│   ├── 1_Enroll.py
│   ├── 2_Identify.py
│   └── 3_People.py
├── src/
│   ├── config.py              # all tunable thresholds/paths
│   ├── embeddings.py          # InsightFace detection + embedding wrapper
│   ├── matching.py            # cosine similarity + tiered decision policy
│   └── database.py            # SQLite persistence
├── evaluation/
│   └── run_evaluation.py
├── tests/
│   └── test_matching.py
├── data/                      # git-ignored: db + thumbnails
└── docs/
    ├── ARCHITECTURE.md
    ├── ML_PIPELINE.md
    ├── EVALUATION.md
    └── DECISIONS.md
```
