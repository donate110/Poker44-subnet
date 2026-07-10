"""Rank-vote blending of decorrelated model members.

Percentile-ranks each member's raw score before combining, instead of
averaging raw probabilities. Members trained on different algorithms (a tree
ensemble vs a neural net, say) produce differently-scaled/shaped score
distributions; rank voting is immune to that scale drift and is what a
top-scoring miner's "HG2Blend" (rank voting across a tree stack, a
monotone-constrained booster, and a PCA->MLP) uses instead of probability
averaging -- see git history for the reference.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def rank01(scores) -> np.ndarray:
    """Percentile rank in [0, 1], stable order for ties.

    Rank is undefined for a single value, but a live request can validly
    contain just one chunk -- falling back to the (clipped) raw score there
    instead of a hardcoded 0.0 avoids silently mis-scoring every
    single-chunk request as confidently human regardless of the model's
    actual output.
    """
    values = np.asarray(scores, dtype=float)
    if values.size <= 1:
        return np.clip(values, 0.0, 1.0)
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return order.astype(float) / (values.size - 1)


class RankVoteBlend:
    """Blends several predict_proba-capable members by rank, not raw probability."""

    def __init__(self, members: Sequence[Tuple[str, object]], weights: Optional[Sequence[float]] = None):
        self.members: List[Tuple[str, object]] = list(members)
        self.weights: List[float] = list(weights) if weights is not None else [1.0] * len(self.members)
        if len(self.weights) != len(self.members):
            raise ValueError("weights must match members in length")

    def predict_proba(self, X) -> np.ndarray:
        total_weight = float(sum(self.weights)) or 1.0
        n = len(X)
        blended = np.zeros(n, dtype=float)
        for (_name, model), weight in zip(self.members, self.weights):
            raw = np.asarray(model.predict_proba(X))[:, 1]
            blended += weight * rank01(raw)
        blended /= total_weight
        return np.column_stack([1.0 - blended, blended])
