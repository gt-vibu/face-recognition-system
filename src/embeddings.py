"""
Wraps InsightFace's FaceAnalysis app to provide face detection and ArcFace
embedding generation in a single forward pass.

Model: InsightFace `buffalo_l` pack (SCRFD detector + ArcFace recognition
model), executed via ONNX Runtime on CPU. Weights are downloaded
automatically on first run and cached under ~/.insightface.

FALLBACK: if InsightFace fails to install on your machine, swap this file's
internals for the `face_recognition` (dlib) library instead — it exposes
`face_recognition.face_locations()` and `face_recognition.face_encodings()`.
No other file in this project needs to change; only DetectedFace population
in `detect_faces()` below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from . import config

_app = None  # lazily-initialized InsightFace FaceAnalysis instance


@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    embedding: np.ndarray            # L2-normalized, shape (512,)
    det_score: float


def _get_app():
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis

        _app = FaceAnalysis(name=config.MODEL_PACK, providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=-1, det_size=config.DET_SIZE)  # ctx_id=-1 forces CPU
    return _app


def detect_faces(image_bgr: np.ndarray) -> List[DetectedFace]:
    """
    Run detection + embedding extraction on a BGR image (as produced by
    cv2.imread or cv2.cvtColor(..., COLOR_RGB2BGR)).
    """
    app = _get_app()
    faces = app.get(image_bgr)

    results: List[DetectedFace] = []
    for f in faces:
        embedding = f.normed_embedding.astype(np.float32)  # already L2-normalized
        bbox = tuple(int(v) for v in f.bbox)
        results.append(DetectedFace(bbox=bbox, embedding=embedding, det_score=float(f.det_score)))
    return results
