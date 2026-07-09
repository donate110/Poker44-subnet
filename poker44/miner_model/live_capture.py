"""Local-only capture of real validator queries, for diagnosing the benchmark-vs-live gap.

Persists the UNLABELED chunks a validator actually sends at inference time — the real
live distribution — plus this miner's own score for each. A live query carries no
ground-truth bot/human label, so nothing captured here can be used as a supervised
training label; it's for comparing chunk-size / feature distributions against the
public benchmark, the way ``poker44/miner_model/train.py``'s ``ROBUST_FEATURE_NAMES``
exclusions were informed by a competitor doing exactly this (see git history).

Safety contract:
  * OFF by default. Enable with env POKER44_CAPTURE=1.
  * Size-capped per file (POKER44_CAPTURE_MAX_BYTES, default 250MB).
  * FAIL-SAFE: every path is wrapped so a capture error can never affect serving.
  * Output directory is gitignored and never leaves the box on its own.

ATTESTATION: capturing live traffic for diagnosis does not change the miner's
training-data statement. If captured data is ever fed into training (even unlabeled,
for domain adaptation), update POKER44_MODEL_PRIVATE_DATA_ATTESTATION truthfully.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

_LOCK = threading.Lock()
_DIR = Path(os.getenv("POKER44_CAPTURE_DIR") or Path(__file__).resolve().parent / "artifacts" / "live_capture")
_MAX_BYTES = int(os.getenv("POKER44_CAPTURE_MAX_BYTES", str(250 * 1024 * 1024)))

# Per-process state: resolved output path, size-cap latch, and a content-hash
# dedupe set. Validators may resend the same evaluation snapshot across many
# query rounds within a day; without dedupe the size cap fills with duplicates.
_state: dict[str, Any] = {"path": None, "full": False, "seen": None}


def enabled() -> bool:
    return os.getenv("POKER44_CAPTURE", "0") == "1"


def _chunk_key(chunk: Sequence[dict]) -> str:
    blob = json.dumps(chunk, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_seen(path: Path) -> set:
    seen: set = set()
    try:
        if path.exists():
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        seen.add(_chunk_key(json.loads(line).get("chunk") or []))
                    except Exception:
                        continue
    except Exception:
        pass
    return seen


def capture(chunks: Sequence[Sequence[dict]], scores: Sequence[float], uid: Any, validator: Any) -> None:
    """Append one JSONL record per chunk: {t, v, uid, n, score, chunk}.

    Input-only (no labels). Never raises — a capture failure must not affect scoring.
    """
    if not enabled() or _state["full"] or not chunks:
        return
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        if _state["path"] is None:
            _state["path"] = _DIR / f"capture_{str(uid)[:16]}.jsonl"
        path: Path = _state["path"]
        if path.exists() and path.stat().st_size >= _MAX_BYTES:
            _state["full"] = True
            return
        if _state["seen"] is None:
            _state["seen"] = _load_seen(path)
        seen: set = _state["seen"]

        ts = round(time.time(), 2)
        vtag = str(validator or "")[:8]
        lines = []
        for chunk, score in zip(chunks, scores):
            key = _chunk_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            try:
                score_value = round(float(score), 5)
            except (TypeError, ValueError):
                score_value = None
            lines.append(
                json.dumps(
                    {"t": ts, "v": vtag, "uid": str(uid), "n": len(chunk), "score": score_value, "chunk": chunk},
                    separators=(",", ":"),
                    default=str,
                )
            )
        if not lines:
            return
        payload = "\n".join(lines) + "\n"
        with _LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
    except Exception:
        pass


def batch_enabled() -> bool:
    return os.getenv("POKER44_CAPTURE_BATCH", "0") == "1"


_batch_state: dict[str, Any] = {"path": None, "full": False, "seen": None}


def _batch_key(chunks: Sequence[Sequence[dict]]) -> str:
    blob = json.dumps(chunks, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_batch_seen(path: Path) -> set:
    seen: set = set()
    try:
        if path.exists():
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        seen.add(_batch_key(json.loads(line).get("chunks") or []))
                    except Exception:
                        continue
    except Exception:
        pass
    return seen


def capture_batch(chunks: Sequence[Sequence[dict]], scores: Sequence[float], uid: Any, validator: Any) -> None:
    """Append the whole query (all chunks + scores) as one JSON record, deduped by
    whole-batch content. Separate from per-chunk capture; gated by POKER44_CAPTURE_BATCH=1
    so operators can capture batch-level shape (chunks-per-query) without doubling
    per-chunk storage. Never raises."""
    if not batch_enabled() or _batch_state["full"] or not chunks:
        return
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        if _batch_state["path"] is None:
            _batch_state["path"] = _DIR / f"batch_{str(uid)[:16]}.jsonl"
        path: Path = _batch_state["path"]
        if path.exists() and path.stat().st_size >= _MAX_BYTES:
            _batch_state["full"] = True
            return
        if _batch_state["seen"] is None:
            _batch_state["seen"] = _load_batch_seen(path)
        key = _batch_key(chunks)
        if key in _batch_state["seen"]:
            return
        _batch_state["seen"].add(key)

        out_scores = []
        for score in scores:
            try:
                out_scores.append(round(float(score), 6))
            except (TypeError, ValueError):
                out_scores.append(None)
        record = {
            "t": round(time.time(), 2),
            "v": str(validator or "")[:8],
            "uid": str(uid),
            "n_chunks": len(chunks),
            "sizes": [len(chunk) for chunk in chunks],
            "scores": out_scores,
        }
        with _LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
    except Exception:
        pass
