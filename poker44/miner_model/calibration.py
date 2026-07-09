"""Monotone score calibration aimed at the reward formula's 5%-FPR ceiling.

    raw score -> isotonic regression -> boundary remap (cut -> 0.5)

``poker44.score.scoring.reward`` is ``0.75 * average_precision + 0.25 *
recall_at_5pct_fpr``: a model that ranks bots above humans perfectly can
still score badly if its natural 0.5 threshold sits inside the human tail
(too many humans read as bots). Isotonic regression alone fixes calibration
but not that boundary placement. This fits both stages on a held-out split
(never the rows the base model trained on) and grid-searches the boundary
that maximizes reward while keeping FPR under the ceiling. Both stages are
monotone, so ranking -- and therefore average precision -- is preserved
exactly; only where the 0.5 cut falls moves, not the relative order of scores.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from poker44.score.scoring import reward


def _boundary_remap(scores: np.ndarray, cut: float) -> np.ndarray:
    """Monotone piecewise-linear map sending `cut` -> 0.5; order preserved."""
    s = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    cut = min(max(float(cut), 1e-6), 1.0 - 1e-6)
    out = np.where(s < cut, (s / cut) * 0.5, 0.5 + ((s - cut) / (1.0 - cut)) * 0.5)
    return np.clip(out, 0.0, 1.0)


class FprCeilingCalibrator:
    """Isotonic-calibrate, then pick the boundary maximizing reward under an FPR ceiling."""

    def __init__(self, *, max_fpr: float = 0.05, identity_blend: float = 0.05, grid_points: int = 256):
        self.max_fpr = float(max_fpr)
        self.identity_blend = float(identity_blend)
        self.grid_points = int(grid_points)
        self.grid_: Optional[np.ndarray] = None
        self.iso_y_: Optional[np.ndarray] = None
        self.cut_: float = 0.5

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> "FprCeilingCalibrator":
        raw = np.asarray(raw_scores, dtype=float)
        y = np.asarray(labels, dtype=int)

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw, y.astype(float))
        grid = np.linspace(0.0, 1.0, self.grid_points)
        # A sliver of the identity map keeps the calibration strictly monotone
        # and avoids wide flat isotonic plateaus that would tie many chunks
        # together and throw away rank information the base model actually had.
        iso_y = (1.0 - self.identity_blend) * np.clip(iso.predict(grid), 0.0, 1.0) + self.identity_blend * grid
        iso_val = np.clip(np.interp(raw, grid, iso_y), 0.0, 1.0)

        candidates = np.unique(np.quantile(iso_val, np.linspace(0.40, 0.999, 80)))
        best_key = None
        best_cut = 0.5
        for cut in candidates:
            remapped = _boundary_remap(iso_val, cut)
            rew, details = reward(remapped, y)
            if details["fpr"] >= self.max_fpr - 1e-9:
                continue
            key = (rew, details["bot_recall"])
            if best_key is None or key > best_key:
                best_key, best_cut = key, float(cut)
        if best_key is None:
            # Nothing cleared the ceiling on this split: fall back to a
            # conformal cut at the human tail so FPR doesn't blow past 0.05.
            human_scores = iso_val[y == 0]
            best_cut = float(np.quantile(human_scores, 1.0 - self.max_fpr)) if human_scores.size else 0.5

        self.grid_, self.iso_y_, self.cut_ = grid, iso_y, best_cut
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if self.grid_ is None or self.iso_y_ is None:
            return np.clip(np.asarray(raw_scores, dtype=float), 0.0, 1.0)
        raw = np.clip(np.asarray(raw_scores, dtype=float), 0.0, 1.0)
        iso_val = np.clip(np.interp(raw, self.grid_, self.iso_y_), 0.0, 1.0)
        return _boundary_remap(iso_val, self.cut_)


class CalibratedClassifier:
    """Wraps a predict_proba-capable model with an FprCeilingCalibrator behind
    a single predict_proba(), so poker44.miner_model.detector.TrainedDetector
    can load it exactly like an uncalibrated model."""

    def __init__(self, base_model: Any, calibrator: FprCeilingCalibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X) -> np.ndarray:
        raw = np.asarray(self.base_model.predict_proba(X))[:, 1]
        calibrated = self.calibrator.transform(raw)
        return np.column_stack([1.0 - calibrated, calibrated])
