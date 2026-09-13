# Decision Log

### Decision 001 — Single-stack Streamlit instead of Next.js + FastAPI
**Status:** Superseded in part by Decision 008 — Streamlit was the first working version and is still included, but the main UI is now Next.js + FastAPI.
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
**Tradeoff:** Slightly more state to manage (a per-session retry counter) — mitigated by keeping that state in the UI session (Streamlit's `session_state`; React state in Next.js, see Decision 009) rather than the database, since it's session-scoped, not persistent.

### Decision 006 — Thresholds are configurable defaults, not hardcoded truths
**Reason:** Per the assignment's own explicit instruction, threshold values must be derived from actual evaluation data, not invented. `src/config.py` documents this in comments; `docs/EVALUATION.md` is the place those values get justified once real data is collected.

### Decision 007 — Local inference only, no cloud APIs
**Reason:** Hard $0 constraint, plus privacy: face embeddings are biometric-adjacent data and should not leave the local machine.

### Decision 008 — Next.js UI over a thin FastAPI layer, after the Streamlit version worked
**Context:** Once the Streamlit version was working, tested and pushed, there was time to build a more polished UI.
**Options considered:** (a) keep Streamlit only, (b) Next.js calling a thin API over the existing `src/` code, (c) rewrite the backend.
**Decision:** (b). `api.py` (FastAPI) exposes status / persons / detect / enroll / identify endpoints and calls the same `src/` functions as the Streamlit pages; the Next.js app (`frontend/`) proxies `/api/*` to it via `next.config.ts`, so there is no CORS configuration.
**Reason:** A much more usable interface without touching the recognition code — detection, embeddings, matching, thresholds and the database are shared, and the Streamlit app keeps working unchanged against the same database.
**Tradeoff:** Two processes to run (API + web UI) and a second UI to maintain. The enrollment averaging steps exist in both `pages/1_Enroll.py` and `api.py` (identical math) rather than in one shared function.

### Decision 009 — Retry counter kept in the browser, not on the server
**Decision:** The Next.js Identify page keeps the attempt counter in React state; the API stays stateless.
**Reason:** Same behaviour as the Streamlit version (per-session, resets on reload) with no session storage or extra API state. The assignment requires the retries to be bounded, which they are.
**Tradeoff:** Reloading the page resets the counter — acceptable for a demo; a production system would enforce it server-side.

### Decision 010 — Count one attempt per new upload, not per rerun
**Context:** Testing found that a plain Streamlit rerun re-processed the same uploaded photo and used up another attempt.
**Decision:** Remember the last counted upload (`file_id`) and only count a new one; the Next.js page counts once in the upload handler.
**Reason:** The limit is meant to count photo attempts, not page refreshes. Verified in both UIs (plain rerun / re-render leaves the counter unchanged).

### Decision 011 — Evaluation threshold slider, sent to the backend
**Decision:** Both Identify pages have a testing-only slider (0.45–0.90, default 0.62) that overrides only the Confirmed threshold. In Next.js the value is sent as an optional `confirmed_threshold` field; FastAPI validates it and passes it to `src/matching.best_match`. With no value, the configured default is used.
**Reason:** Lets reviewers see the threshold trade-off on real uploads without editing `config.py`, and without a second copy of the decision logic in the frontend. The Uncertain lower bound never moves.
**Tradeoff:** One optional parameter on `best_match` and on the identify endpoint.

### Decision 012 — Keep 0.62 / 0.45 after evaluation
**Context:** The in-app validation used one real identity plus AI-generated faces; a later test used 50 real people from LFW (see `docs/EVALUATION.md`).
**Options considered:** (a) keep 0.62 / 0.45, (b) lower to 0.55 / 0.40, which scored higher on LFW.
**Decision:** (a) Keep `CONFIRMED_THRESHOLD = 0.62` and `UNCERTAIN_LOWER_BOUND = 0.45`.
**Reason:** On LFW, 0.62 gave 90.2% confident identification with zero wrong-person confirmations and zero strangers confirmed; the rest went to Uncertain. 0.55 would have reached 97.7% on LFW, but thresholds below 0.62 produced wrong-person confirmations on the synthetic look-alikes. The system is designed so that a borderline face is asked for another photo rather than confidently named. For 0.40 the evidence is one borderline synthetic case and a thin LFW margin — not enough to re-tune.
**Tradeoff:** About 1 in 10 genuine photos (2-photo enrollment) needs a retry; look-alikes, twins and phone/webcam photos remain untested.

### Decision 013 — Recommend 3 enrollment photos instead of lowering the threshold
**Decision:** The Enroll page recommends 3 photos with slightly different angles or lighting; the minimum stays 2 and up to 5 are averaged.
**Reason:** On LFW at the same 0.62 threshold, 3 photos raised confident identification from 90.2% to 95.9% (4 photos: 97.3%), still with zero wrong-person or stranger confirmations — better recognition without making the decision less conservative or touching the model.
**Tradeoff:** None in code (enrollment already averaged any number of accepted photos); only a hint.
