"""
Cosine similarity and the tiered recognition decision policy:

    Confirmed Match  -> similarity >= config.CONFIRMED_THRESHOLD
    Uncertain Match  -> config.UNCERTAIN_LOWER_BOUND <= similarity < CONFIRMED_THRESHOLD
    Unknown          -> similarity < config.UNCERTAIN_LOWER_BOUND

We report "similarity", never "confidence" — cosine similarity is not a
calibrated probability, and calling it a confidence percentage would be
misleading.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from . import config


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


class MatchStatus(str, Enum):
    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


@dataclass
class MatchResult:
    status: MatchStatus
    person_id: Optional[int]
    person_name: Optional[str]
    similarity: float


def best_match(
    query_embedding: np.ndarray,
    enrolled: List[Tuple[int, str, np.ndarray]],
) -> MatchResult:
    """
    enrolled: list of (person_id, name, embedding) for every enrolled person.
    Applies the three-tier decision policy using the thresholds in config.py.
    """
    if not enrolled:
        return MatchResult(MatchStatus.UNKNOWN, None, None, 0.0)

    scored = [
        (person_id, name, cosine_similarity(query_embedding, emb))
        for person_id, name, emb in enrolled
    ]
    person_id, name, score = max(scored, key=lambda t: t[2])

    if score >= config.CONFIRMED_THRESHOLD:
        return MatchResult(MatchStatus.CONFIRMED, person_id, name, score)
    elif score >= config.UNCERTAIN_LOWER_BOUND:
        # Candidate name is carried for display context, but callers must
        # NOT present this as a confirmed identity.
        return MatchResult(MatchStatus.UNCERTAIN, person_id, name, score)
    else:
        return MatchResult(MatchStatus.UNKNOWN, None, None, score)
