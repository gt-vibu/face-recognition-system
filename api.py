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


# Presentation only: box colours on the annotated image (RGB), matching the UI's status colours.
BOX_COLORS = {
    MatchStatus.CONFIRMED: (22, 163, 74),   # green
    MatchStatus.UNCERTAIN: (217, 119, 6),   # amber
    MatchStatus.UNKNOWN: (220, 38, 38),     # red
}


def _draw_face_tag(image_rgb: np.ndarray, bbox, number: int, color) -> None:
    """Draw a face box plus a filled number tag, scaled to the image size so it stays readable."""
    x1, y1, x2, y2 = bbox
    scale = max(image_rgb.shape[:2]) / 1000
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, max(2, round(3 * scale)))
    label, font = str(number), cv2.FONT_HERSHEY_SIMPLEX
    font_scale, weight = max(0.55, 0.9 * scale), max(1, round(2 * scale))
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, weight)
    pad = max(3, round(6 * scale))
    tag_h = th + baseline + 2 * pad
    top = y1 - tag_h if y1 - tag_h >= 0 else max(y1, 0)  # above the box, or just inside it at the top edge
    left = max(x1, 0)
    cv2.rectangle(image_rgb, (left, top), (left + tw + 2 * pad, top + tag_h), color, -1)
    cv2.putText(image_rgb, label, (left + pad, top + pad + th), font, font_scale, (255, 255, 255), weight, cv2.LINE_AA)


def _face_crop(image_rgb: np.ndarray, bbox, size: int = 160) -> Optional[str]:
    """Square thumbnail around one face (from the original, un-annotated image) as a JPEG data URL."""
    x1, y1, x2, y2 = bbox
    h, w = image_rgb.shape[:2]
    cx, cy, half = (x1 + x2) // 2, (y1 + y2) // 2, int(0.75 * max(x2 - x1, y2 - y1))
    crop = image_rgb[max(0, cy - half):min(h, cy + half), max(0, cx - half):min(w, cx + half)]
    if crop.size == 0:
        return None
    f = size / max(crop.shape[:2])
    if f < 1:
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * f)), max(1, round(crop.shape[0] * f))),
                          interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode() if ok else None


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

    # Number faces left to right so the tags on the image read naturally (display order only).
    faces = sorted(faces, key=lambda f: f.bbox[0])
    annotated = image_np.copy()
    results = []
    for idx, face in enumerate(faces):
        result = best_match(face.embedding, enrolled, confirmed_threshold)
        _draw_face_tag(annotated, face.bbox, idx + 1, BOX_COLORS[result.status])
        results.append({
            "index": idx + 1,
            "status": result.status.value,
            # Candidate name for Uncertain is context only, never a confirmed identity.
            "person_name": result.person_name if result.status != MatchStatus.UNKNOWN else None,
            "similarity": round(result.similarity, 4),
            "face_image": _face_crop(image_np, face.bbox),
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
