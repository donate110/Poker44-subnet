"""Chunk-level feature extraction for the trained Poker44 bot detector.

Operates only on the miner-visible hand schema (metadata/players/streets/actions/outcome)
as sanitized by ``poker44.validator.payload_view``. Fields that are always constant in
that sanitized view (``button_seat``, ``outcome.showdown``, ``outcome.total_pot``,
``hole_cards``, ``board_cards``, ...) are intentionally not featurized: they carry no
signal live and would just be noise learned from an unsanitized training source.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

_MEANINGFUL_ACTIONS = ("call", "check", "bet", "raise", "fold")
_AMOUNT_BUCKET_EDGES = ((0.5, "xs"), (1.0, "s"), (2.0, "m"), (5.0, "l"))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return _div(sum(values), len(values))


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return math.sqrt(max(0.0, _mean([(v - m) ** 2 for v in values])))


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = min(max(q, 0.0), 1.0) * (len(ordered) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (1 - (pos - lo)) + ordered[hi] * (pos - lo)


def _entropy(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    total = float(sum(counts.values()))
    ent = -sum((n / total) * math.log(n / total) for n in counts.values())
    return _div(ent, math.log(len(counts)))


def _max_run_share(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    longest = current = 1
    for prev, current_value in zip(values, values[1:]):
        current = current + 1 if prev == current_value else 1
        longest = max(longest, current)
    return _div(longest, len(values))


def _amount_bucket(value: float) -> str:
    if value <= 0.0:
        return "z"
    for threshold, tag in _AMOUNT_BUCKET_EDGES:
        if value <= threshold:
            return tag
    return "xl"


def _slope(values: Sequence[float]) -> float:
    """Least-squares slope of values against their index; 0.0 if undefined."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = _mean(xs)
    y_mean = _mean(values)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    return numer / denom


def _lag1_autocorr(values: Sequence[float]) -> float:
    """Pearson correlation between values[:-1] and values[1:]; 0.0 if undefined."""
    if len(values) < 3:
        return 0.0
    head, tail = values[:-1], values[1:]
    head_std, tail_std = _std(head), _std(tail)
    if head_std <= 0 or tail_std <= 0:
        return 0.0
    head_mean, tail_mean = _mean(head), _mean(tail)
    covariance = _mean([(a - head_mean) * (b - tail_mean) for a, b in zip(head, tail)])
    return max(-1.0, min(1.0, covariance / (head_std * tail_std)))


def _bigram_rate(pairs: Sequence[Tuple[str, str]], first: str, second: str) -> float:
    if not pairs:
        return 0.0
    from_first = [b for a, b in pairs if a == first]
    if not from_first:
        return 0.0
    return _div(sum(1 for b in from_first if b == second), len(from_first))


def hand_features(hand: Dict[str, Any]) -> Dict[str, float]:
    """Scalar behavioral features for a single hand."""
    metadata = hand.get("metadata") or {}
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    actions = hand.get("actions") or []

    max_seats = max(1, _i(metadata.get("max_seats"), 6))
    hero_seat = _i(metadata.get("hero_seat"), 0)
    bb = _f(metadata.get("bb"), 0.02) or 0.02

    action_types: List[str] = []
    actor_seats: List[int] = []
    street_names: List[str] = []
    amounts_bb: List[float] = []
    pot_before_bb: List[float] = []
    pot_after_bb: List[float] = []
    raise_to_count = 0
    call_to_count = 0

    for action in actions:
        if not isinstance(action, dict):
            continue
        action_types.append(str(action.get("action_type") or "").lower().strip())
        street_names.append(str(action.get("street") or "").lower().strip())
        seat = _i(action.get("actor_seat"), 0)
        if seat > 0:
            actor_seats.append(seat)
        amounts_bb.append(max(0.0, _f(action.get("normalized_amount_bb"))))
        pot_before_bb.append(max(0.0, _div(_f(action.get("pot_before")), bb)))
        pot_after_bb.append(max(0.0, _div(_f(action.get("pot_after")), bb)))
        raise_to_count += int(action.get("raise_to") is not None)
        call_to_count += int(action.get("call_to") is not None)

    stacks_bb = [
        _div(_f(player.get("starting_stack")), bb)
        for player in players
        if isinstance(player, dict)
    ]

    action_count = max(1.0, float(len(actions)))
    counts = Counter(action_types)
    meaningful = max(1, sum(counts.get(kind, 0) for kind in _MEANINGFUL_ACTIONS))
    aggressive = counts.get("bet", 0) + counts.get("raise", 0)
    passive = counts.get("call", 0) + counts.get("check", 0)
    preflop_n = sum(1 for street in street_names if street == "preflop")
    postflop_n = sum(1 for street in street_names if street not in ("", "preflop"))
    pot_delta = [max(0.0, a - b) for a, b in zip(pot_after_bb, pot_before_bb)]
    monotonic = sum(
        1 for prev, cur in zip(pot_after_bb, pot_after_bb[1:]) if cur + 1e-9 >= prev
    )

    action_bigrams = list(zip(action_types, action_types[1:]))
    street_transitions = sum(
        1 for prev, cur in zip(street_names, street_names[1:]) if prev != cur
    )

    return {
        "player_count": float(len(players)),
        "seat_utilization": _div(len(players), max_seats),
        "action_count": float(len(actions)),
        "street_count": float(len(streets)),
        "call_share": _div(counts.get("call", 0), meaningful),
        "check_share": _div(counts.get("check", 0), meaningful),
        "fold_share": _div(counts.get("fold", 0), meaningful),
        "bet_share": _div(counts.get("bet", 0), meaningful),
        "raise_share": _div(counts.get("raise", 0), meaningful),
        "aggression_share": _div(aggressive, action_count),
        "passive_share": _div(passive, action_count),
        "preflop_share": _div(preflop_n, action_count),
        "postflop_share": _div(postflop_n, action_count),
        "action_entropy": _entropy(action_types),
        "actor_entropy": _entropy(actor_seats),
        "street_entropy": _entropy(street_names),
        "unique_actor_share": _div(len(set(actor_seats)), max(1.0, float(len(players)))),
        "actor_switch_rate": _div(
            sum(1 for a, b in zip(actor_seats, actor_seats[1:]) if a != b),
            max(len(actor_seats) - 1, 1),
        ),
        "actor_run_max_share": _max_run_share(actor_seats),
        "action_run_max_share": _max_run_share(action_types),
        "amount_mean_bb": _mean(amounts_bb),
        "amount_std_bb": _std(amounts_bb),
        "amount_q90_bb": _quantile(amounts_bb, 0.9),
        "nonzero_amount_share": _div(sum(1 for v in amounts_bb if v > 0), action_count),
        "pot_before_mean_bb": _mean(pot_before_bb),
        "pot_delta_mean_bb": _mean(pot_delta),
        "pot_growth_bb": (max(pot_after_bb) - min(pot_before_bb)) if pot_after_bb and pot_before_bb else 0.0,
        "pot_monotonic_rate": _div(monotonic, max(len(pot_after_bb) - 1, 1)),
        "raise_to_share": _div(raise_to_count, action_count),
        "call_to_share": _div(call_to_count, action_count),
        "stack_mean_bb": _mean(stacks_bb),
        "stack_std_bb": _std(stacks_bb),
        "stack_iqr_bb": _quantile(stacks_bb, 0.75) - _quantile(stacks_bb, 0.25),
        "hero_action_share": _div(
            sum(1 for seat in actor_seats if seat == hero_seat and hero_seat > 0), action_count
        ),
        "street_transition_rate": _div(street_transitions, action_count),
        "raise_after_call_rate": _bigram_rate(action_bigrams, "call", "raise"),
        "fold_after_raise_rate": _bigram_rate(action_bigrams, "raise", "fold"),
        "check_after_check_rate": _bigram_rate(action_bigrams, "check", "check"),
        "raise_after_raise_rate": _bigram_rate(action_bigrams, "raise", "raise"),
        "amount_trend_slope": _slope(amounts_bb),
        "amount_lag1_autocorr": _lag1_autocorr(amounts_bb),
        "first_action_aggressive": float(
            bool(action_types) and action_types[0] in ("bet", "raise")
        ),
        "last_action_aggressive": float(
            bool(action_types) and action_types[-1] in ("bet", "raise")
        ),
    }


_EMPTY_HAND_KEYS = tuple(
    sorted(
        hand_features({"metadata": {}, "players": [], "streets": [], "actions": []}).keys()
    )
)
_AGGREGATES = ("mean", "std", "min", "max", "q10", "q50", "q90")


def _hand_signatures(hand: Dict[str, Any]) -> Tuple[tuple, tuple, tuple, tuple, tuple]:
    actions = hand.get("actions") or []
    action_types = tuple(str((a or {}).get("action_type") or "").lower().strip() for a in actions)
    actor_seq = tuple(
        _i((a or {}).get("actor_seat"), 0) for a in actions if _i((a or {}).get("actor_seat"), 0) > 0
    )
    street_seq = tuple(str((a or {}).get("street") or "").lower().strip() for a in actions)
    amount_bucket_seq = tuple(
        _amount_bucket(max(0.0, _f((a or {}).get("normalized_amount_bb")))) for a in actions
    )
    # Multiset of adjacent action-type pairs: catches hands that repeat the same
    # sub-pattern (e.g. "call->raise->fold") even when overall sequence length
    # differs, which the full-sequence signature above would treat as distinct.
    bigram_signature = tuple(sorted(Counter(zip(action_types, action_types[1:])).items()))
    return action_types, actor_seq, street_seq, amount_bucket_seq, bigram_signature


_DRIFT_KEYS = (
    "aggression_share",
    "fold_share",
    "action_entropy",
    "amount_mean_bb",
    "pot_monotonic_rate",
)


def _chunk_drift_features(per_hand: List[Dict[str, float]]) -> Dict[str, float]:
    """Second-half minus first-half means, in hand order: near zero for a
    behaviorally consistent (often scripted) actor, larger for one whose play
    shifts over the course of the session the chunk represents."""
    n = len(per_hand)
    out: Dict[str, float] = {}
    if n < 2:
        for key in _DRIFT_KEYS:
            out[f"drift_{key}"] = 0.0
        return out
    midpoint = n // 2
    first_half, second_half = per_hand[:midpoint], per_hand[midpoint:]
    for key in _DRIFT_KEYS:
        first_mean = _mean([row[key] for row in first_half])
        second_mean = _mean([row[key] for row in second_half])
        out[f"drift_{key}"] = second_mean - first_mean
    return out


def chunk_features(chunk: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate a chunk (list of hands belonging to one scoring unit) into a feature row."""
    if not chunk:
        return {"hand_count": 0.0}

    per_hand = [hand_features(hand) for hand in chunk]
    out: Dict[str, float] = {"hand_count": float(len(chunk))}
    for name in _EMPTY_HAND_KEYS:
        series = [row[name] for row in per_hand]
        out[f"{name}_mean"] = _mean(series)
        out[f"{name}_std"] = _std(series)
        out[f"{name}_min"] = min(series)
        out[f"{name}_max"] = max(series)
        out[f"{name}_q10"] = _quantile(series, 0.1)
        out[f"{name}_q50"] = _quantile(series, 0.5)
        out[f"{name}_q90"] = _quantile(series, 0.9)

    action_sigs, actor_sigs, street_sigs, amount_sigs, bigram_sigs = [], [], [], [], []
    for hand in chunk:
        a_sig, ac_sig, s_sig, amt_sig, bg_sig = _hand_signatures(hand)
        action_sigs.append(a_sig)
        actor_sigs.append(ac_sig)
        street_sigs.append(s_sig)
        amount_sigs.append(amt_sig)
        bigram_sigs.append(bg_sig)

    n = float(len(chunk))
    for tag, signatures in (
        ("action", action_sigs),
        ("actor", actor_sigs),
        ("street", street_sigs),
        ("amount_bucket", amount_sigs),
        ("action_bigram", bigram_sigs),
    ):
        out[f"signature_{tag}_top_share"] = _div(max(Counter(signatures).values()), n)
        out[f"signature_{tag}_unique_share"] = _div(len(set(signatures)), n)

    out.update(_chunk_drift_features(per_hand))

    return out


FEATURE_NAMES = sorted(
    chunk_features(
        [{"metadata": {"max_seats": 6, "hero_seat": 1}, "players": [], "streets": [],
          "actions": [{"action_type": "call", "street": "preflop", "actor_seat": 1}]}]
    ).keys()
)


# Absolute bet/pot/stack magnitude ("_bb") features and the raw (non-scale-
# invariant) bet-size trend slope split trees on benchmark-scale thresholds.
# A competitor who instrumented real live-validator capture (not just the
# public benchmark) measured these 2-11 sigma out-of-distribution between
# benchmark and live traffic (live pots/bets run at roughly half benchmark
# scale) — the single biggest evidenced source of benchmark-to-live
# generalization failure found so far. `amount_lag1_autocorr` is a Pearson
# correlation, which is scale-invariant by construction, so it is kept.
_FRAGILE_MAGNITUDE_SUBSTRING = "_bb"
_FRAGILE_MAGNITUDE_EXACT = {
    name for name in FEATURE_NAMES if name.startswith("amount_trend_slope")
}

ROBUST_FEATURE_NAMES = [
    name
    for name in FEATURE_NAMES
    if _FRAGILE_MAGNITUDE_SUBSTRING not in name and name not in _FRAGILE_MAGNITUDE_EXACT
]


def features_to_row(features: Dict[str, float]) -> List[float]:
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
