# Decision Log

### Decision 001 — Single-stack Streamlit instead of Next.js + FastAPI
**Context:** Assignment must be completed solo, at $0, within a very short deadline.
**Options considered:** (a) Next.js frontend + FastAPI backend as two services, (b) Streamlit single-process app, (c) plain Flask + HTML.
**Decision:** Streamlit.
**Reason:** Pure Python, built-in file upload widgets, native multi-page support, zero cross-service integration risk (no CORS, no two dev servers to keep in sync). Given the deadline, integration risk was the single biggest threat to actually finishing.
**Tradeoff:** Less visually customizable than a hand-built Next.js UI; acceptable given the assignment is graded primarily on ML correctness and documentation, not frontend polish.

### Decision 002 — InsightFace (ArcFace) as the primary embedding model
**Context:** Needed a free, pretrained, CPU-runnable face embedding model.
**Options considered:** OpenCV Haar/LBPH, `face_recognition` (dlib), InsightFace (ArcFace), DeepFace.
**Decision:** InsightFace `buffalo_l`.
**Reason:** ArcFace-style embeddings are the modern production standard; InsightFace bundles both detector and embedding model, ships pretrained ONNX weights, and runs fast on CPU.
**Tradeoff:** Slightly more setup friction than `face_recognition` on some machines; `face_recognition` is documented as the fallback in `src/embeddings.py`'s docstring.

### Decision 003 — Cosine similarity as the matching metric
**Reason:** ArcFace embeddings are trained/normalized such that direction, not magnitude, encodes identity — cosine similarity is the metric the model was designed to be compared with.

### Decision 004 — SQLite for storage
**Options considered:** JSON, CSV, pickle, `.npy`, a full vector database.
**Decision:** SQLite.
**Reason:** Serverless, built into Python, supports structured metadata alongside the embedding BLOB, and is simple to explain and defend. A vector database is unnecessary overhead at this scale (see `docs/ARCHITECTURE.md`).

### Decision 005 — Three-tier Confirmed / Uncertain / Unknown decision policy (instead of a single binary threshold)
**Context:** A single threshold either forces a match or rejects it, with no room for "this is close but not certain enough."
**Decision:** Two thresholds creating three bands, with a bounded retry loop for the middle band.
**Reason:** More honest representation of the actual evidence; prevents both over-confident misidentification and unnecessary rejection of borderline-but-correct matches.
**Tradeoff:** Slightly more state to manage (a per-session retry counter) — mitigated by keeping that state in Streamlit's `session_state` rather than the database, since it's session-scoped, not persistent.

### Decision 006 — Thresholds are configurable defaults, not hardcoded truths
**Reason:** Per the assignment's own explicit instruction, threshold values must be derived from actual evaluation data, not invented. `src/config.py` documents this in comments; `docs/EVALUATION.md` is the place those values get justified once real data is collected.

### Decision 007 — Local inference only, no cloud APIs
**Reason:** Hard $0 constraint, plus privacy: face embeddings are biometric-adjacent data and should not leave the local machine.
