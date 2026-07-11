"""Blending of decorrelated model members for the Poker44 detector.

Two strategies, and they are NOT interchangeable for this subnet:

``CalibratedAverageBlend`` (use this one) fits a per-member isotonic
calibrator ONCE, offline, on a held-out split, then at serve time averages
each member's *calibrated* probability. The result is comparable across
different, unrelated validator requests, because the mapping from raw score
to blended score is fixed at training time.

``RankVoteBlend`` percentile-ranks each member's raw score *within the
current call* before combining. This is what a top-scoring miner's
"HG2Blend" uses (rank voting across a tree stack, a monotone-constrained
booster, and a PCA->MLP), and it is immune to inter-member scale drift -- but
it is only safe when the thing being ranked is the full, final population
you'll be scored against in one shot. It is NOT safe here: the validator
accumulates predictions across many separate forward() calls into a rolling
buffer and computes reward (rank-based average precision + a threshold-swept
recall@FPR -- see poker44.score.scoring.reward) over that *pooled* buffer,
not per-call. Rank-normalizing within each call forces every request into
the same flat [0, 1] spread regardless of how separable its chunks actually
are, which throws away confidence information the pooled ranking needs.
Measured on the live leaderboard: h02 (RankVoteBlend) scored 0.309 in its
first live round despite a *better* offline held-out score than h01's plain
probability-averaging ensemble (0.958 vs 0.923), which scored 0.485 live.
Kept here for reference/completeness, not for production use.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression


def rank01(scores) -> np.ndarray:
    """Percentile rank in [0, 1], stable order for ties. See RankVoteBlend's
    caveat above before using this for anything scored across pooled calls."""
    values = np.asarray(scores, dtype=float)
    if values.size <= 1:
        return np.clip(values, 0.0, 1.0)
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return order.astype(float) / (values.size - 1)


class RankVoteBlend:
    """Blends several predict_proba-capable members by within-call rank.
    See the module docstring: not safe for this subnet's pooled-window reward."""

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


class CalibratedAverageBlend:
    """Blends several predict_proba-capable members by averaging each
    member's OWN isotonic-calibrated probability, fit once offline.

    Diversifies across decorrelated model families the same way rank voting
    does (each member gets its own probability scale, so raw-score-scale
    drift between e.g. a tree ensemble and a neural net doesn't dominate the
    blend), but the calibration mapping is fixed at fit() time -- so scores
    stay comparable across separate validator calls, which this subnet's
    pooled-buffer reward requires.
    """

    def __init__(self, members: Sequence[Tuple[str, object]], weights: Optional[Sequence[float]] = None):
        self.members: List[Tuple[str, object]] = list(members)
        self.weights: List[float] = list(weights) if weights is not None else [1.0] * len(self.members)
        if len(self.weights) != len(self.members):
            raise ValueError("weights must match members in length")
        self.calibrators_: Dict[str, IsotonicRegression] = {}

    def fit(self, X_cal, y_cal) -> "CalibratedAverageBlend":
        """Fit one isotonic calibrator per member on a held-out split (rows
        no member trained on)."""
        y = np.asarray(y_cal, dtype=float)
        for name, model in self.members:
            raw = np.asarray(model.predict_proba(X_cal))[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw, y)
            self.calibrators_[name] = iso
        return self

    def predict_proba(self, X) -> np.ndarray:
        total_weight = float(sum(self.weights)) or 1.0
        n = len(X)
        blended = np.zeros(n, dtype=float)
        for (name, model), weight in zip(self.members, self.weights):
            raw = np.asarray(model.predict_proba(X))[:, 1]
            calibrator = self.calibrators_.get(name)
            calibrated = calibrator.predict(raw) if calibrator is not None else raw
            blended += weight * np.clip(calibrated, 0.0, 1.0)
        blended /= total_weight
        return np.column_stack([1.0 - blended, blended])
