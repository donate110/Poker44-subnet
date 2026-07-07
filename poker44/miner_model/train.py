"""Train the Poker44 reference bot-detection model on the public training benchmark.

Fetches labeled chunk groups from ``https://api.poker44.net/api/v1/benchmark``,
extracts the chunk-level feature set from ``poker44.miner_model.features``, trains a
soft-vote ensemble, and evaluates it with the validator's own reward formula
(``poker44.score.scoring.reward``) on release dates held out of training.

Usage:
    python -m poker44.miner_model.train --release-dates 14 --holdout-dates 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import requests
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from poker44.miner_model.features import FEATURE_NAMES, chunk_features, features_to_row
from poker44.score.scoring import reward

BENCHMARK_BASE_URL = "https://api.poker44.net/api/v1/benchmark"
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "detector.joblib"
CACHE_DIR = Path(__file__).resolve().parent / "artifacts" / "benchmark_cache"


def _get_json(url: str, params: Dict[str, Any] | None = None, timeout: float = 30.0) -> Dict[str, Any]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()["data"]


def list_release_dates(max_dates: int) -> List[str]:
    releases = _get_json(f"{BENCHMARK_BASE_URL}/releases", params={"limit": max_dates})["releases"]
    return [release["sourceDate"] for release in releases]


def fetch_source_date_chunks(source_date: str, *, page_limit: int = 24) -> List[Dict[str, Any]]:
    """Download (with local cache) every release-chunk record for one sourceDate."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{source_date}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())["chunks"]

    all_chunks: List[Dict[str, Any]] = []
    cursor = None
    while True:
        params: Dict[str, Any] = {"sourceDate": source_date, "limit": page_limit}
        if cursor:
            params["cursor"] = cursor
        data = _get_json(f"{BENCHMARK_BASE_URL}/chunks", params=params, timeout=60.0)
        all_chunks.extend(data["chunks"])
        cursor = data.get("nextCursor")
        if not cursor:
            break

    cache_path.write_text(json.dumps({"sourceDate": source_date, "chunks": all_chunks}))
    return all_chunks


def build_examples(source_dates: List[str]) -> List[Dict[str, Any]]:
    """One example per (chunk group, label) pair, tagged with sourceDate/split."""
    examples: List[Dict[str, Any]] = []
    for source_date in source_dates:
        for release_chunk in fetch_source_date_chunks(source_date):
            groups = release_chunk.get("chunks") or []
            labels = release_chunk.get("groundTruth") or []
            split = release_chunk.get("split")
            for group, label in zip(groups, labels):
                examples.append(
                    {
                        "features": chunk_features(group),
                        "label": int(label),
                        "source_date": source_date,
                        "split": split,
                    }
                )
    return examples


def split_train_test(
    examples: List[Dict[str, Any]], *, holdout_dates: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dates = sorted({example["source_date"] for example in examples})
    holdout = set(dates[-max(1, holdout_dates):]) if len(dates) > holdout_dates else set()
    train = [ex for ex in examples if ex["source_date"] not in holdout]
    test = [ex for ex in examples if ex["source_date"] in holdout]
    if not train or not test:
        # Not enough distinct dates for a clean date split; fall back to a random split.
        rng = np.random.default_rng(0)
        indices = rng.permutation(len(examples))
        split_at = max(1, int(len(examples) * 0.85))
        train = [examples[i] for i in indices[:split_at]]
        test = [examples[i] for i in indices[split_at:]]
    return train, test


def build_ensemble(seed: int = 0) -> VotingClassifier:
    extra_trees = ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=seed, n_jobs=-1,
    )
    random_forest = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=seed + 1, n_jobs=-1,
    )
    hist_gb = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
        random_state=seed + 2,
    )
    return VotingClassifier(
        estimators=[("extra_trees", extra_trees), ("random_forest", random_forest), ("hist_gb", hist_gb)],
        voting="soft",
    )


def evaluate(model: VotingClassifier, examples: List[Dict[str, Any]]) -> Dict[str, float]:
    if not examples:
        return {}
    rows = [features_to_row(ex["features"]) for ex in examples]
    labels = np.array([ex["label"] for ex in examples])
    scores = model.predict_proba(np.array(rows, dtype=float))[:, 1]

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dates", type=int, default=14, help="How many recent release dates to train on.")
    parser.add_argument("--holdout-dates", type=int, default=2, help="Most recent dates held out for evaluation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=str(ARTIFACT_PATH))
    args = parser.parse_args()

    print(f"Listing up to {args.release_dates} release dates from {BENCHMARK_BASE_URL} ...")
    source_dates = list_release_dates(args.release_dates)
    print(f"Training on release dates: {source_dates}")

    examples = build_examples(source_dates)
    print(f"Loaded {len(examples)} labeled chunk examples.")

    train_examples, test_examples = split_train_test(examples, holdout_dates=args.holdout_dates)
    train_dates = sorted({ex["source_date"] for ex in train_examples})
    test_dates = sorted({ex["source_date"] for ex in test_examples})
    print(f"Train: {len(train_examples)} examples over {train_dates}")
    print(f"Test:  {len(test_examples)} examples over {test_dates}")

    X_train = np.array([features_to_row(ex["features"]) for ex in train_examples], dtype=float)
    y_train = np.array([ex["label"] for ex in train_examples])

    model = build_ensemble(seed=args.seed)
    model.fit(X_train, y_train)

    metrics = evaluate(model, test_examples)
    print("Held-out evaluation:", json.dumps(metrics, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "metadata": {
                "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "train_source_dates": train_dates,
                "test_source_dates": test_dates,
                "train_rows": len(train_examples),
                "test_rows": len(test_examples),
                "metrics": metrics,
            },
        },
        output_path,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
