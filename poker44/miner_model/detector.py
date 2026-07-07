"""Runtime scorer that loads the trained detector artifact produced by train.py."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import bittensor as bt
import joblib

from poker44.miner_model.features import chunk_features, features_to_row

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "detector.joblib"


class TrainedDetector:
    """Loads once, scores many chunks. Raises if the artifact is missing."""

    def __init__(self, artifact_path: Path = ARTIFACT_PATH):
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"No trained model at {artifact_path}. Run "
                "`python -m poker44.miner_model.train` first."
            )
        artifact = joblib.load(artifact_path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self.metadata = artifact.get("metadata", {})

    def score_chunk(self, chunk: list) -> float:
        if not chunk:
            return 0.5
        features = chunk_features(chunk)
        row = [float(features.get(name, 0.0)) for name in self.feature_names]
        proba = self.model.predict_proba([row])[0][1]
        return round(max(0.0, min(1.0, float(proba))), 6)

    def score_batch(self, chunks: List[list]) -> List[float]:
        if not chunks:
            return []
        rows = []
        empty_indices = set()
        for i, chunk in enumerate(chunks):
            if not chunk:
                empty_indices.add(i)
                rows.append([0.0] * len(self.feature_names))
                continue
            features = chunk_features(chunk)
            rows.append([float(features.get(name, 0.0)) for name in self.feature_names])
        probabilities = self.model.predict_proba(rows)[:, 1]
        return [
            0.5 if i in empty_indices else round(max(0.0, min(1.0, float(p))), 6)
            for i, p in enumerate(probabilities)
        ]


_DETECTOR: Optional[TrainedDetector] = None
_LOAD_FAILED = False


def try_load_detector() -> Optional[TrainedDetector]:
    """Best-effort singleton loader; returns None (once) if the artifact is missing or
    unloadable (e.g. trained under a different scikit-learn version than is installed)."""
    global _DETECTOR, _LOAD_FAILED
    if _DETECTOR is not None:
        return _DETECTOR
    if _LOAD_FAILED:
        return None
    try:
        _DETECTOR = TrainedDetector()
        return _DETECTOR
    except Exception as exc:
        bt.logging.warning(
            f"Trained detector unavailable ({exc!r}); falling back to heuristic scorer."
        )
        _LOAD_FAILED = True
        return None
