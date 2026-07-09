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
import random
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

from poker44.miner_model.features import ROBUST_FEATURE_NAMES, chunk_features
from poker44.score.scoring import reward

TRAINING_FEATURE_NAMES = ROBUST_FEATURE_NAMES

BENCHMARK_BASE_URL = "https://api.poker44.net/api/v1/benchmark"
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "detector.joblib"
CACHE_DIR = Path(__file__).resolve().parent / "artifacts" / "benchmark_cache"

# Every public benchmark chunk group is exactly 30 or 40 hands, but the live
# validator contract explicitly allows "one or many hands" per chunk and warns
# miners not to assume a fixed size (docs/miner.md). Training only on 30-40
# hand groups means every aggregate/quantile/signature feature is calibrated
# to that size and behaves very differently at other sizes. SUB_CHUNK_SIZES
# covers smaller windows for general size-robustness. LARGE_CHUNK_SIZES
# specifically targets 60-120 hands: a competitor who instrumented real
# live-validator capture (poker44_ml/live_capture.py in Travis861/Poker44_v1,
# the #1-by-score miner as of 2026-07-09) measured actual live chunks at
# 80-100 hands -- bigger than the benchmark, not smaller. Built via
# concatenating whole same-label groups rather than sub-windowing since no
# single benchmark group is long enough on its own.
SUB_CHUNK_SIZES = (1, 2, 3, 5, 8, 12, 20)
SUB_CHUNKS_PER_SIZE = 2
LARGE_CHUNK_SIZES = (60, 80, 100, 120)
LARGE_CHUNKS_PER_SIZE = 1


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
    """One example per (chunk group, label) pair, tagged with sourceDate/split.

    Keeps the raw hand list (not just its features) so callers can generate
    label-preserving sub-chunks of other sizes before featurizing.
    """
    examples: List[Dict[str, Any]] = []
    for source_date in source_dates:
        for release_chunk in fetch_source_date_chunks(source_date):
            groups = release_chunk.get("chunks") or []
            labels = release_chunk.get("groundTruth") or []
            split = release_chunk.get("split")
            for group, label in zip(groups, labels):
                examples.append(
                    {
                        "group": group,
                        "label": int(label),
                        "source_date": source_date,
                        "split": split,
                        "augmented": False,
                    }
                )
    return examples


def add_sub_chunk_augmentation(
    examples: List[Dict[str, Any]],
    *,
    sizes: Tuple[int, ...] = SUB_CHUNK_SIZES,
    samples_per_size: int = SUB_CHUNKS_PER_SIZE,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Add label-preserving contiguous sub-windows of each group at smaller sizes."""
    augmented: List[Dict[str, Any]] = list(examples)
    for example in examples:
        group = example["group"]
        for size in sizes:
            if size >= len(group):
                continue
            for _ in range(samples_per_size):
                start = rng.randint(0, len(group) - size)
                augmented.append(
                    {
                        "group": group[start : start + size],
                        "label": example["label"],
                        "source_date": example["source_date"],
                        "split": example["split"],
                        "augmented": True,
                    }
                )
    return augmented


def _concat_chunk(pool: List[Dict[str, Any]], size: int, rng: random.Random) -> Tuple[list, str]:
    """Concatenate whole groups (no single group reaches 60+ hands) up to `size` hands."""
    order = list(pool)
    rng.shuffle(order)
    concatenated: List[dict] = []
    idx = 0
    while len(concatenated) < size:
        concatenated.extend(order[idx % len(order)]["group"])
        idx += 1
    return concatenated[:size], order[0]["source_date"]


def add_concat_augmentation(
    examples: List[Dict[str, Any]],
    *,
    sizes: Tuple[int, ...] = LARGE_CHUNK_SIZES,
    samples_per_size: int = LARGE_CHUNKS_PER_SIZE,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Add label-preserving large chunks built by concatenating several
    same-label groups, since no single benchmark group reaches 60+ hands."""
    by_label: Dict[int, List[Dict[str, Any]]] = {}
    for example in examples:
        by_label.setdefault(example["label"], []).append(example)

    augmented: List[Dict[str, Any]] = list(examples)
    for label, pool in by_label.items():
        if len(pool) < 2:
            continue
        for size in sizes:
            for _ in range(samples_per_size):
                group, source_date = _concat_chunk(pool, size, rng)
                augmented.append(
                    {
                        "group": group,
                        "label": label,
                        "source_date": source_date,
                        "split": None,
                        "augmented": True,
                    }
                )
    return augmented


def featurize(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {**example, "features": chunk_features(example["group"])}
        for example in examples
    ]


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
    # min_samples_leaf/max_depth are deliberately conservative: the sub-chunk
    # augmentation in add_sub_chunk_augmentation() produces many highly
    # correlated rows per base group (they're overlapping windows of the same
    # 30/40-hand session), so unlimited-depth trees mostly memorize individual
    # groups instead of generalizing — and blow up the pickled artifact size
    # (400 estimators x 2 unlimited-depth forests over ~4-5k rows exceeded
    # GitHub's 100MB file limit) for no accuracy benefit.
    extra_trees = ExtraTreesClassifier(
        n_estimators=150, max_depth=16, min_samples_leaf=5, class_weight="balanced_subsample",
        random_state=seed, n_jobs=-1,
    )
    random_forest = RandomForestClassifier(
        n_estimators=150, max_depth=16, min_samples_leaf=5, class_weight="balanced_subsample",
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


def _to_row(features: Dict[str, float]) -> List[float]:
    return [float(features.get(name, 0.0)) for name in TRAINING_FEATURE_NAMES]


def evaluate(model: VotingClassifier, examples: List[Dict[str, Any]]) -> Dict[str, float]:
    if not examples:
        return {}
    rows = [_to_row(ex["features"]) for ex in examples]
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


def evaluate_by_chunk_size(
    model: VotingClassifier,
    raw_test_examples: List[Dict[str, Any]],
    *,
    window_sizes: Tuple[int, ...],
    concat_sizes: Tuple[int, ...],
    rng: random.Random,
) -> Dict[str, Dict[str, float]]:
    """Diagnostic: reward at each chunk size, staying strictly within the
    held-out test groups/dates. Sizes below a group's length are built by
    sub-windowing; sizes above it (60-120, the live range) are built by
    concatenating several held-out groups, same as the training augmentation."""
    by_size: Dict[str, Dict[str, float]] = {}
    for size in window_sizes:
        sized_examples = []
        for example in raw_test_examples:
            group = example["group"]
            if size >= len(group):
                continue
            start = rng.randint(0, len(group) - size)
            sized_examples.append({**example, "group": group[start : start + size]})
        sized_examples = featurize(sized_examples)
        metrics = evaluate(model, sized_examples)
        if metrics:
            by_size[str(size)] = metrics

    concat_examples = add_concat_augmentation(
        raw_test_examples, sizes=concat_sizes, samples_per_size=20, rng=rng
    )
    concat_only = [ex for ex in concat_examples if ex.get("augmented")]
    for size in concat_sizes:
        sized = [ex for ex in concat_only if len(ex["group"]) == size]
        metrics = evaluate(model, featurize(sized))
        if metrics:
            by_size[str(size)] = metrics

    by_size["full"] = evaluate(model, featurize(raw_test_examples))
    return by_size


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

    raw_examples = build_examples(source_dates)
    print(f"Loaded {len(raw_examples)} labeled chunk examples "
          f"(benchmark groups are always 30 or 40 hands).")

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

    X_train = np.array([_to_row(ex["features"]) for ex in train_examples], dtype=float)
    y_train = np.array([ex["label"] for ex in train_examples])

    model = build_ensemble(seed=args.seed)
    model.fit(X_train, y_train)

    metrics = evaluate(model, test_examples)
    print("Held-out evaluation (original chunk sizes):", json.dumps(metrics, indent=2))

    size_metrics = evaluate_by_chunk_size(
        model, raw_test,
        window_sizes=SUB_CHUNK_SIZES, concat_sizes=LARGE_CHUNK_SIZES,
        rng=random.Random(args.seed + 1),
    )
    print("Held-out evaluation by chunk size:", json.dumps(size_metrics, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": TRAINING_FEATURE_NAMES,
            "metadata": {
                "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "train_source_dates": train_dates,
                "test_source_dates": test_dates,
                "train_rows": len(train_examples),
                "test_rows": len(test_examples),
                "sub_chunk_sizes": list(SUB_CHUNK_SIZES),
                "large_chunk_sizes": list(LARGE_CHUNK_SIZES),
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
