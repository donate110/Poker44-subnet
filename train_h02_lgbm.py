"""Train a LightGBM detector for the sn1/h02 miner.

Reuses the same public-benchmark fetch, feature extraction, and chunk-size
augmentation as poker44/miner_model/train.py (that pipeline is already
evidence-based: ROBUST_FEATURE_NAMES drops absolute bet/pot/stack magnitude
features that are 2-11 sigma out-of-distribution live vs benchmark, and
LARGE_CHUNK_SIZES covers the live-realistic 60-120 hand range). What's new
here, informed by the top-4 miners as of 2026-07-09:

- LightGBM instead of the ExtraTrees+RandomForest+HistGB ensemble the h01
  miner runs (a different model family, so h01 and h02 aren't just two
  copies of the same thing under different wallets).
- FprCeilingCalibrator (poker44/miner_model/calibration.py): isotonic
  regression + a boundary remap that grid-searches the 0.5 cut to maximize
  reward while keeping FPR under the reward formula's 5% ceiling, fit on a
  held-out split the base model never trained on. Monotone, so ranking (and
  average precision) is preserved exactly.

Usage:
    python train_h02_lgbm.py --release-dates 14 --holdout-dates 2
"""
from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

# Cosmetic: numpy rows are correctly column-aligned; this only fires because
# LightGBM stores a feature-name signature from fit() and checks it on predict.
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from poker44.miner_model.calibration import CalibratedClassifier, FprCeilingCalibrator
from poker44.miner_model.features import ROBUST_FEATURE_NAMES
from poker44.miner_model.train import (
    LARGE_CHUNK_SIZES,
    SUB_CHUNK_SIZES,
    add_concat_augmentation,
    add_sub_chunk_augmentation,
    build_examples,
    evaluate_by_chunk_size as _base_evaluate_by_chunk_size,
    featurize,
    list_release_dates,
    split_train_test,
)
from poker44.score.scoring import reward

ARTIFACT_PATH = Path(__file__).resolve().parent / "poker44" / "miner_model" / "artifacts" / "detector_h02_lgbm.joblib"
TRAINING_FEATURE_NAMES = ROBUST_FEATURE_NAMES


def _to_row(features: Dict[str, float]) -> List[float]:
    return [float(features.get(name, 0.0)) for name in TRAINING_FEATURE_NAMES]


def evaluate(model: Any, examples: List[Dict[str, Any]]) -> Dict[str, float]:
    if not examples:
        return {}
    rows = np.array([_to_row(ex["features"]) for ex in examples], dtype=float)
    labels = np.array([ex["label"] for ex in examples])
    scores = model.predict_proba(rows)[:, 1]

    rew, details = reward(scores, labels)
    return {
        "reward": rew,
        "ap_score": details["ap_score"],
        "bot_recall_at_5pct_fpr": details["bot_recall"],
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else 0.0,
        "average_precision": float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else 0.0,
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "n_examples": len(examples),
        "n_bots": int(labels.sum()),
        "n_humans": int((labels == 0).sum()),
    }


def evaluate_by_chunk_size(model: Any, raw_test_examples, *, rng: random.Random) -> Dict[str, Dict[str, float]]:
    # Delegate to train.py's implementation via a tiny adapter so both
    # pipelines report chunk-size breakdowns the same way and stay comparable.
    class _Adapter:
        def predict_proba(self, X):
            return model.predict_proba(X)

    return _base_evaluate_by_chunk_size(
        _Adapter(), raw_test_examples,
        window_sizes=SUB_CHUNK_SIZES, concat_sizes=LARGE_CHUNK_SIZES, rng=rng,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dates", type=int, default=14)
    parser.add_argument("--holdout-dates", type=int, default=2)
    parser.add_argument("--calibration-fraction", type=float, default=0.2,
                         help="Share of the (post-augmentation) training rows held out to fit the calibrator.")
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output", type=str, default=str(ARTIFACT_PATH))
    args = parser.parse_args()

    print(f"Listing up to {args.release_dates} release dates ...")
    source_dates = list_release_dates(args.release_dates)
    print(f"Training on release dates: {source_dates}")

    raw_examples = build_examples(source_dates)
    raw_train, raw_test = split_train_test(raw_examples, holdout_dates=args.holdout_dates)
    train_dates = sorted({ex["source_date"] for ex in raw_train})
    test_dates = sorted({ex["source_date"] for ex in raw_test})

    rng = random.Random(args.seed)
    augmented_train = add_sub_chunk_augmentation(raw_train, rng=rng)
    augmented_train = add_concat_augmentation(augmented_train, rng=rng)
    train_examples = featurize(augmented_train)
    test_examples = featurize(raw_test)
    print(f"Train: {len(train_examples)} examples over {train_dates} "
          f"({len(raw_train)} base groups + sub-chunk sizes {SUB_CHUNK_SIZES} "
          f"+ concatenated large-chunk sizes {LARGE_CHUNK_SIZES})")
    print(f"Test:  {len(test_examples)} examples over {test_dates} (unaugmented, original 30/40-hand groups)")
    print(f"Training on {len(TRAINING_FEATURE_NAMES)} robust features "
          f"(absolute bet/pot/stack magnitude features excluded).")

    X = np.array([_to_row(ex["features"]) for ex in train_examples], dtype=float)
    y = np.array([ex["label"] for ex in train_examples])

    # Fit-split for the base model, calibration-split held out for the
    # FprCeilingCalibrator so the boundary reflects generalization, not rows
    # the tree model memorized.
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=args.calibration_fraction, random_state=args.seed, stratify=y,
    )

    n_pos, n_neg = int(y_fit.sum()), int((y_fit == 0).sum())
    model = LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        max_depth=-1, min_child_samples=20, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=2.0, random_state=args.seed, n_jobs=-1,
        verbose=-1, class_weight="balanced" if abs(n_pos - n_neg) > 0.1 * (n_pos + n_neg) else None,
    )
    model.fit(X_fit, y_fit)

    raw_cal_scores = model.predict_proba(X_cal)[:, 1]
    calibrator = FprCeilingCalibrator(max_fpr=args.max_fpr).fit(raw_cal_scores, y_cal)
    print(f"Calibrator boundary cut={calibrator.cut_:.4f} (max_fpr={args.max_fpr})")

    calibrated_model = CalibratedClassifier(model, calibrator)

    metrics = evaluate(calibrated_model, test_examples)
    print("Held-out evaluation (original chunk sizes):", json.dumps(metrics, indent=2))

    size_metrics = evaluate_by_chunk_size(calibrated_model, raw_test, rng=random.Random(args.seed + 1))
    print("Held-out evaluation by chunk size:", json.dumps(size_metrics, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": calibrated_model,
            "feature_names": TRAINING_FEATURE_NAMES,
            "metadata": {
                "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "backend": "lightgbm",
                "train_source_dates": train_dates,
                "test_source_dates": test_dates,
                "train_rows": len(train_examples),
                "calibration_rows": int(len(X_cal)),
                "test_rows": len(test_examples),
                "sub_chunk_sizes": list(SUB_CHUNK_SIZES),
                "large_chunk_sizes": list(LARGE_CHUNK_SIZES),
                "calibrator_cut": calibrator.cut_,
                "max_fpr": args.max_fpr,
                "metrics": metrics,
                "metrics_by_chunk_size": size_metrics,
            },
        },
        output_path,
        compress=3,
    )
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
