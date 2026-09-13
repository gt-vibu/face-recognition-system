"""
Thin HTTP API over the existing src/ modules, used by the Next.js frontend.

It reuses the same detection, embedding, matching and SQLite code as the
Streamlit pages — nothing here changes the recognition logic or thresholds.
The Uncertain-match retry counter lives in the browser (like Streamlit's
per-session state), so this API is stateless.

Run from this folder:
    uvicorn api:app --port 8000
"""
from __future__ import annotations

import base64
import io
import os
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from src import config, database
from src.embeddings import _get_app, detect_faces
from src.matching import MatchStatus, best_match


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _get_app()  # load buffalo_l once at startup instead of on the first request
    yield


app = FastAPI(title="Face Recognition API", lifespan=lifespan)

# Upper bound of the evaluation/testing threshold slider (same range as the Streamlit slider).
# The lower bound is always config.UNCERTAIN_LOWER_BOUND, so the Uncertain band can't be inverted.
EVAL_THRESHOLD_MAX = 0.90


def _load(file: UploadFile):
    """Same decoding as the Streamlit pages: PIL -> RGB array -> BGR for InsightFace."""
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail=f"'{file.filename}' is not a readable image.")
    image_np = np.array(image)
    faces = detect_faces(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return image, image_np, faces


@app.get("/api/status")
def status():
    return {
        "enrolled_count": len(database.get_all_persons()),
        "confirmed_threshold": config.CONFIRMED_THRESHOLD,
        "uncertain_lower_bound": config.UNCERTAIN_LOWER_BOUND,
        "max_identification_attempts": config.MAX_IDENTIFICATION_ATTEMPTS,
        "min_enrollment_images": config.MIN_ENROLLMENT_IMAGES,
        "max_enrollment_images": config.MAX_ENROLLMENT_IMAGES,
        "model_pack": config.MODEL_PACK,
        "evaluation_threshold_max": EVAL_THRESHOLD_MAX,
    }


@app.get("/api/persons")
def list_persons():
    return [
        {
            "person_id": person_id,
            "name": name,
            "num_samples": num_samples,
            "created_at": created_at,
            "has_thumbnail": bool(thumb and os.path.exists(thumb)),
        }
        for person_id, name, _, num_samples, created_at, thumb in database.get_all_persons()
    ]


@app.get("/api/persons/{person_id}/thumbnail")
def person_thumbnail(person_id: int):
    for pid, _, _, _, _, thumb in database.get_all_persons():
        if pid == person_id and thumb and os.path.exists(thumb):
            return FileResponse(thumb, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No thumbnail for this person.")


@app.delete("/api/persons/{person_id}")
def delete_person(person_id: int):
    if person_id not in {p[0] for p in database.get_all_persons()}:
        raise HTTPException(status_code=404, detail="Person not found.")
    database.delete_person(person_id)
    return {"deleted": person_id}


@app.post("/api/detect")
def detect(file: UploadFile = File(...)):
    """Per-photo check shown on the Enroll page before saving."""
    _, _, faces = _load(file)
    return {"filename": file.filename, "face_count": len(faces)}


@app.post("/api/enroll")
def enroll(name: str = Form(...), files: List[UploadFile] = File(...)):
    """Mirrors pages/1_Enroll.py: first MAX images, single-face only, averaged embedding."""
    name_clean = name.strip()
    files = files[: config.MAX_ENROLLMENT_IMAGES]

    accepted_embeddings, first_good_image, per_file = [], None, []
    for file in files:
        image, _, faces = _load(file)
        accepted = len(faces) == 1
        per_file.append({"filename": file.filename, "face_count": len(faces), "accepted": accepted})
        if accepted:
            accepted_embeddings.append(faces[0].embedding)
            if first_good_image is None:
                first_good_image = image

    if name_clean == "":
        raise HTTPException(status_code=400, detail="Enter a name to continue.")
    if len(accepted_embeddings) < config.MIN_ENROLLMENT_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {config.MIN_ENROLLMENT_IMAGES} valid single-face images "
            f"(got {len(accepted_embeddings)}).",
        )

    updated = database.name_exists(name_clean)
    avg_embedding = np.mean(np.stack(accepted_embeddings), axis=0)
    avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-10)

    thumbnail_path = os.path.join(config.THUMB_DIR, f"{uuid.uuid4().hex}.jpg")
    first_good_image.save(thumbnail_path)

    person_id = database.add_or_update_person(
        name_clean, avg_embedding, len(accepted_embeddings), thumbnail_path
    )
    return {
        "person_id": person_id,
        "name": name_clean,
        "num_samples": len(accepted_embeddings),
        "updated": updated,
        "files": per_file,
    }


@app.post("/api/identify")
def identify(file: UploadFile = File(...), confirmed_threshold: Optional[float] = Form(None)):
    """
    Mirrors pages/2_Identify.py's matching; retry counting is done by the caller.

    confirmed_threshold: optional evaluation/testing override (the Identify page slider),
    passed straight to src.matching.best_match. It only moves the Confirmed boundary —
    the Uncertain lower bound stays config.UNCERTAIN_LOWER_BOUND and config.py is never
    changed. When omitted, config.CONFIRMED_THRESHOLD is used.
    """
    if confirmed_threshold is not None and not (
        config.UNCERTAIN_LOWER_BOUND <= confirmed_threshold <= EVAL_THRESHOLD_MAX
    ):
        raise HTTPException(
            status_code=400,
            detail=f"confirmed_threshold must be between {config.UNCERTAIN_LOWER_BOUND} and {EVAL_THRESHOLD_MAX}.",
        )

    _, image_np, faces = _load(file)
    enrolled = database.get_person_embeddings_for_matching()

    annotated = image_np.copy()
    results = []
    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 128, 128), 2)
        cv2.putText(annotated, str(idx + 1), (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        result = best_match(face.embedding, enrolled, confirmed_threshold)
        results.append({
            "index": idx + 1,
            "status": result.status.value,
            # Candidate name for Uncertain is context only, never a confirmed identity.
            "person_name": result.person_name if result.status != MatchStatus.UNKNOWN else None,
            "similarity": round(result.similarity, 4),
        })

    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    return {
        "faces": results,
        "annotated_image": "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode() if ok else None,
        # The Confirmed threshold actually applied to this request (override or default).
        "confirmed_threshold": confirmed_threshold if confirmed_threshold is not None else config.CONFIRMED_THRESHOLD,
        "default_confirmed_threshold": config.CONFIRMED_THRESHOLD,
        "uncertain_lower_bound": config.UNCERTAIN_LOWER_BOUND,
    }
