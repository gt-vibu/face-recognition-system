import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matching import MatchStatus, best_match, cosine_similarity


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_best_match_empty_database_is_unknown():
    result = best_match(np.array([1.0, 0.0]), [])
    assert result.status == MatchStatus.UNKNOWN


def test_best_match_confirmed():
    query = np.array([1.0, 0.0])
    enrolled = [(1, "Alice", np.array([1.0, 0.0]))]
    result = best_match(query, enrolled)
    assert result.status == MatchStatus.CONFIRMED
    assert result.person_name == "Alice"


def test_best_match_unknown_when_far():
    query = np.array([1.0, 0.0])
    enrolled = [(1, "Alice", np.array([0.0, 1.0]))]
    result = best_match(query, enrolled)
    assert result.status == MatchStatus.UNKNOWN


def test_best_match_uncertain_band():
    # ~40 degrees apart -> cos(40deg) ~= 0.766, tune vectors to land in band
    query = np.array([1.0, 0.0])
    enrolled = [(1, "Alice", np.array([0.55, 0.83]))]  # similarity ~0.55
    result = best_match(query, enrolled)
    assert result.status == MatchStatus.UNCERTAIN
