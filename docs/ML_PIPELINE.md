# ML Pipeline

## Face detection

InsightFace's bundled detector (part of the `buffalo_l` pack, SCRFD-based) locates faces and returns bounding boxes plus a detection confidence score. This runs as part of the same `FaceAnalysis.get()` call that produces embeddings — see `src/embeddings.py`.

## Face embeddings

Each detected face is converted to a 512-dimensional, L2-normalized embedding vector by the ArcFace recognition model bundled in `buffalo_l`. Because the vectors are pretrained on millions of identities never seen during our enrollment, the model generalizes to brand-new people without any retraining — we are doing inference only, never training.

## Similarity metric

Cosine similarity between L2-normalized vectors (equivalent to a dot product here). ArcFace-style embeddings are trained so that *direction*, not magnitude, encodes identity, making cosine similarity the natural metric — see `src/matching.py::cosine_similarity`.

## Threshold & tiered decision policy

See README "Recognition decision policy" for the full table. Two thresholds (`CONFIRMED_THRESHOLD`, `UNCERTAIN_LOWER_BOUND`) create three bands: Confirmed / Uncertain / Unknown. Both live in `src/config.py`, are clearly marked as unvalidated defaults, and must be set from real evaluation data — see `docs/EVALUATION.md`.

## Unknown rejection

If the best similarity across all enrolled people is below `UNCERTAIN_LOWER_BOUND`, the system reports Unknown rather than forcing the nearest enrolled identity. This is the core mechanism that prevents every stranger from being misidentified as whoever happens to be closest.

## Uncertain-match retry policy

If the best similarity falls in the uncertain band, the system does **not** present a name as confirmed. It asks for another image and increments a session-scoped counter, capped at `config.MAX_IDENTIFICATION_ATTEMPTS` (default 3). After the cap is reached, the system stops requesting further images and reports "Identity could not be confidently verified" — it never loops indefinitely.

## Enrollment aggregation

Each accepted enrollment image's embedding is combined into a single representative vector per person — the normalised sum of the embeddings, which points in the same direction as the normalised average — rather than storing every raw sample. This reduces the noise contributed by any single lower-quality photo at minimal implementation cost. The un-normalised sum and the sample count are stored alongside the reference, so photos added later update it exactly (`src/enrollment.py`); the photos themselves are never stored.

## Multi-face identification

`detect_faces()` returns every detected face; the identification page runs the matching pipeline independently per face and reports each result separately, rather than returning one result for the whole image.

## Failure cases

| Case | Expected behavior | Notes / improvement |
|---|---|---|
| No face detected | Clear inline error, no crash | — |
| Multiple faces (enrollment) | Rejected with an explanation; user must supply single-person images | — |
| Multiple faces (identification) | Each face gets its own result | — |
| Very dark / poorly lit image | Similarity scores degrade, may land in Uncertain or Unknown | Could add a brightness/contrast normalization pre-processing step |
| Side profile | Lower similarity than frontal shots | Enroll with a couple of angled photos for robustness |
| Occlusion (glasses, mask, hand) | Similarity drop, possible false Unknown | Documented limitation; occlusion-robust models exist but are out of scope here |
| Low resolution | Noisier embeddings, less reliable scores | Could add a minimum-resolution check on upload |
| Visually similar people (e.g. siblings) | Known hard case — risk of false Confirmed/Uncertain overlap | Documented as an explicit limitation; would need additional signals (more enrollment photos, liveness, etc.) at scale |
| Duplicate enrollment (same name) | Never replaced silently: the user chooses **Add photos** (exact update of the stored reference) or **Replace enrollment** | — |
| Empty database | Any identification attempt returns Unknown | Handled explicitly in `matching.best_match` |

Each of these should be spot-checked manually before submission and the *actual observed* behavior (not just the expected behavior) noted here or in `docs/EVALUATION.md`.
