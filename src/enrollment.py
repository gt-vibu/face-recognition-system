"""
Enrollment maths — pure NumPy, no I/O.

A person's reference embedding is the normalised SUM of every accepted enrollment
embedding. Normalising the sum points in exactly the same direction as normalising
the mean (they differ only in length), so matching is unchanged — but keeping the
un-normalised sum and the sample count makes adding photos later an exact update:

    new_sum   = old_sum + sum(new embeddings)
    reference = new_sum / ||new_sum||

which is identical to enrolling all the photos at once. A normalised reference alone
cannot be extended exactly (its original length is lost), so the sum is stored too.
Only these vectors are kept — never the enrollment photos themselves.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def embedding_sum(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Sum of the accepted (already L2-normalised) embeddings, in float64 for exact accumulation."""
    if len(embeddings) == 0:
        raise ValueError("At least one embedding is required.")
    return np.sum(np.stack(embeddings).astype(np.float64), axis=0)


def reference_from_sum(total: np.ndarray) -> np.ndarray:
    """The stored reference: the running sum normalised to unit length (float32, like the embeddings)."""
    return (total / (np.linalg.norm(total) + 1e-10)).astype(np.float32)
