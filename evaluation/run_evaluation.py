"""
Run a basic, honest evaluation of the recognition pipeline.

Expected folder layout (NOT committed to git — see .gitignore):

    evaluation_data/
    ├── known/
    │   ├── Alice/
    │   │   ├── enroll_1.jpg  (used to build Alice's reference embedding)
    │   │   ├── enroll_2.jpg
    │   │   ├── test_1.jpg    (held-out — NOT used to build the reference)
    │   │   └── test_2.jpg
    │   └── Bob/
    │       └── ...
    └── unknown/
        ├── stranger_1.jpg
        └── ...

Usage:
    python evaluation/run_evaluation.py

This prints genuine / impostor / unknown score distributions and the
resulting False Acceptance Rate / False Rejection Rate at the currently
configured threshold. Copy the ACTUAL printed numbers into
docs/EVALUATION.md — never fabricate results.
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.embeddings import detect_faces
from src.matching import cosine_similarity

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_data")


def embed_image(path: str):
    image = cv2.imread(path)
    if image is None:
        return None
    faces = detect_faces(image)
    if len(faces) != 1:
        return None
    return faces[0].embedding


def build_reference_embeddings(known_dir: str):
    references = {}
    for person in sorted(os.listdir(known_dir)):
        person_dir = os.path.join(known_dir, person)
        if not os.path.isdir(person_dir):
            continue
        enroll_files = sorted(glob.glob(os.path.join(person_dir, "enroll_*")))
        embeddings = [e for e in (embed_image(f) for f in enroll_files) if e is not None]
        if embeddings:
            avg = np.mean(np.stack(embeddings), axis=0)
            references[person] = avg / (np.linalg.norm(avg) + 1e-10)
        else:
            print(f"  [warning] no usable enrollment images found for '{person}'")
    return references


def summarize(label: str, scores):
    if not scores:
        print(f"{label}: no data")
        return
    arr = np.array(scores)
    print(f"{label}: n={len(arr)}  min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}")


def main():
    known_dir = os.path.join(EVAL_DIR, "known")
    unknown_dir = os.path.join(EVAL_DIR, "unknown")

    if not os.path.isdir(known_dir):
        print(f"Evaluation data not found at {known_dir}. See this file's docstring for the expected layout.")
        return

    references = build_reference_embeddings(known_dir)
    print(f"Built reference embeddings for: {list(references.keys())}\n")

    genuine_scores, impostor_scores, unknown_scores = [], [], []
    # Same scores as above, paired with a "<photo> vs <reference>" label for the detailed report.
    labelled = {"Genuine": [], "Impostor": [], "Unknown": []}

    for person in sorted(os.listdir(known_dir)):
        person_dir = os.path.join(known_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for test_file in sorted(glob.glob(os.path.join(person_dir, "test_*"))):
            embedding = embed_image(test_file)
            if embedding is None:
                print(f"  [skip] no single face detected in {test_file}")
                continue
            for ref_name, ref_embedding in references.items():
                score = cosine_similarity(embedding, ref_embedding)
                (genuine_scores if ref_name == person else impostor_scores).append(score)
                group = "Genuine" if ref_name == person else "Impostor"
                labelled[group].append((score, f"{person}/{os.path.basename(test_file)} vs {ref_name}"))

    if os.path.isdir(unknown_dir):
        for unknown_file in sorted(glob.glob(os.path.join(unknown_dir, "*"))):
            embedding = embed_image(unknown_file)
            if embedding is None:
                continue
            for ref_name, ref_embedding in references.items():
                score = cosine_similarity(embedding, ref_embedding)
                unknown_scores.append(score)
                labelled["Unknown"].append((score, f"unknown/{os.path.basename(unknown_file)} vs {ref_name}"))

    print("--- Raw score distributions ---")
    summarize("Genuine", genuine_scores)
    summarize("Impostor", impostor_scores)
    summarize("Unknown", unknown_scores)

    threshold = config.CONFIRMED_THRESHOLD
    negatives = impostor_scores + unknown_scores
    far = float(np.mean(np.array(negatives) >= threshold)) if negatives else 0.0
    frr = float(np.mean(np.array(genuine_scores) < threshold)) if genuine_scores else 0.0

    print(f"\n--- At CONFIRMED_THRESHOLD = {threshold} ---")
    print(f"False Acceptance Rate (FAR): {far:.3f}")
    print(f"False Rejection Rate (FRR): {frr:.3f}")

    print_detailed_report(labelled, genuine_scores, negatives)

    print("\nCopy these ACTUAL numbers into docs/EVALUATION.md.")


def print_detailed_report(labelled, genuine_scores, negatives):
    """Evidence for choosing thresholds: every score, a FAR/FRR sweep, and Uncertain-band counts."""
    print("\n--- Every score, sorted (highest first) ---")
    for group, rows in labelled.items():
        print(f"{group} (n={len(rows)}):")
        for score, label in sorted(rows, reverse=True):
            print(f"  {score:.4f}  {label}")

    print("\n--- Threshold sweep (FAR over impostor+unknown, FRR over genuine) ---")
    print("  threshold    FAR     FRR")
    for t in np.round(np.arange(0.30, 0.80 + 1e-9, 0.05), 2):
        far_t = float(np.mean(np.array(negatives) >= t)) if negatives else 0.0
        frr_t = float(np.mean(np.array(genuine_scores) < t)) if genuine_scores else 0.0
        print(f"  {t:9.2f}  {far_t:6.3f}  {frr_t:6.3f}")

    lo, hi = config.UNCERTAIN_LOWER_BOUND, config.CONFIRMED_THRESHOLD
    print(f"\n--- Scores in current Uncertain band ({lo} <= score < {hi}) ---")
    for group, rows in labelled.items():
        in_band = sum(1 for score, _ in rows if lo <= score < hi)
        print(f"  {group}: {in_band} of {len(rows)}")


if __name__ == "__main__":
    main()
