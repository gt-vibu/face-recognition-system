"""
Central configuration for the face recognition system.

IMPORTANT: CONFIRMED_THRESHOLD and UNCERTAIN_LOWER_BOUND below are STARTING
DEFAULTS ONLY. They are not scientifically validated values. Before trusting
them, run evaluation/run_evaluation.py against your own collected genuine /
impostor / unknown similarity scores and update these numbers based on what
you actually observe (see docs/EVALUATION.md).
"""
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "face_db.sqlite3")
THUMB_DIR = os.path.join(DATA_DIR, "thumbnails")

# --- Model ---
# InsightFace model pack: bundles a detector (SCRFD/RetinaFace) + ArcFace
# recognition model, run via ONNX Runtime on CPU.
MODEL_PACK = "buffalo_l"
DET_SIZE = (640, 640)

# --- Matching thresholds (see warning above) ---
# score >= CONFIRMED_THRESHOLD                              -> Confirmed Match
# UNCERTAIN_LOWER_BOUND <= score < CONFIRMED_THRESHOLD        -> Uncertain Match
# score < UNCERTAIN_LOWER_BOUND                                -> Unknown
CONFIRMED_THRESHOLD = 0.62
UNCERTAIN_LOWER_BOUND = 0.45

# --- Bounded retry policy for Uncertain Match ---
MAX_IDENTIFICATION_ATTEMPTS = 3

# --- Enrollment ---
MIN_ENROLLMENT_IMAGES = 2
MAX_ENROLLMENT_IMAGES = 5

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)
