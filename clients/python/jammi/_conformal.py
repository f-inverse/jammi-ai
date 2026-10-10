"""Stateless conformal / reciprocal-rank-fusion numerics — computed locally.

These four verbs are pure functions of **caller-supplied arrays** (calibration
and test scores, observed targets, ranked id lists). The remote engine holds
none of their inputs, so a gRPC hop would only ship data the caller already has;
they are a *client/local numeric utility*, not a server round-trip. This module
is the pure-Python peer of the embedded engine's `predict/conformal.rs` and
`query/rrf.rs`, reproducing the SAME finite-sample quantile, score families,
calibration schemes (marginal, Mondrian, weighted), and fusion order so the two
transports agree byte-for-byte on shared inputs.

The conformal guarantee: under exchangeability of calibration and serving data,
the emitted sets/intervals carry marginal coverage `>= 1 - alpha` for any
underlying model and any sample size (Vovk et al. 2005; Angelopoulos & Bates
2021). The one task-specific piece is the nonconformity score; everything else
is the empirical `⌈(n+1)(1-alpha)⌉` quantile applied at serving time.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from itertools import accumulate
from typing import List, Optional, Sequence, Tuple

# Cormack's default `k_rrf` (SIGIR 2009). Robust across [40, 80]; damps the
# contribution of deep ranks so a row's top-of-list appearances dominate. Kept
# identical to the engine's `query::DEFAULT_K_RRF`.
DEFAULT_K_RRF = 60


def _validate_alpha(alpha: float) -> None:
    """Reject a miscoverage level outside the open interval ``(0, 1)``."""
    if not (alpha > 0.0 and alpha < 1.0):
        raise ValueError(f"conformal alpha must lie in (0, 1), got {alpha}")


def finite_sample_quantile(scores: Sequence[float], alpha: float) -> float:
    """The conformal threshold ``q̂`` — the ``⌈(n+1)(1-alpha)⌉``-th smallest score.

    The ``(n+1)`` finite-sample correction is what makes ``P(s_test <= q̂) >=
    1 - alpha`` hold *exactly* under exchangeability; the naive ``⌈n(1-alpha)⌉``
    order statistic under-covers by ``~1/n``. When ``⌈(n+1)(1-alpha)⌉ > n`` the
    requested level is unattainable from this few points and the threshold is
    ``+inf`` (every label admitted) — conservative, never silently wrong.

    Raises ``ValueError`` when ``scores`` is empty, ``alpha`` is out of
    ``(0, 1)``, or any score is NaN (a NaN would silently corrupt the rank).
    """
    _validate_alpha(alpha)
    if not scores:
        raise ValueError("conformal calibration requires at least one score")
    if any(math.isnan(s) for s in scores):
        raise ValueError("conformal calibration scores must be finite")
    n = len(scores)
    # rank = ⌈(n+1)(1-alpha)⌉, a 1-based order statistic index.
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        # Too few points for this level: the honest answer is +inf.
        return math.inf
    # Ascending sort; 1-based rank to 0-based index.
    return sorted(scores)[rank - 1]


def _validate_weights(weights: Sequence[float]) -> None:
    """Likelihood ratios must each be finite and non-negative."""
    if any(w < 0.0 or not math.isfinite(w) for w in weights):
        raise ValueError(
            "weighted conformal weights must be finite and non-negative"
        )


class _WeightedScores:
    """Calibration scores ascending, each with the running total of the weights
    up to and including it — what a weighted threshold walks (Tibshirani et al.
    2019). The engine's ``WeightedScores``, step for step."""

    def __init__(self, scores: Sequence[float], weights: Sequence[float]) -> None:
        if not scores:
            raise ValueError("conformal calibration requires at least one score")
        if any(math.isnan(s) for s in scores):
            raise ValueError("conformal calibration scores must be finite")
        _validate_weights(weights)
        pairs = sorted(zip(scores, weights), key=lambda pair: pair[0])
        self._sorted = [score for score, _ in pairs]
        self._cumulative = list(accumulate(weight for _, weight in pairs))

    def threshold(self, test_weight: float, alpha: float) -> float:
        """The smallest score whose cumulative weight reaches
        ``(1 - alpha)(Σw + w_test)``, or ``+inf`` when the calibration mass
        cannot — the test point's own weight sits at +inf."""
        _validate_weights([test_weight])
        total = self._cumulative[-1] + test_weight
        if total <= 0.0:
            raise ValueError("weighted conformal requires a positive total weight")
        reached = bisect_left(self._cumulative, (1.0 - alpha) * total)
        return self._sorted[reached] if reached < len(self._sorted) else math.inf


def _thresholds(
    cal_scores: Sequence[float],
    test_rows: int,
    alpha: float,
    groups: Optional[Sequence[str]],
    test_groups: Optional[Sequence[str]],
    weights: Optional[Sequence[float]],
    test_weights: Optional[Sequence[float]],
) -> List[float]:
    """The threshold each test row is held to under the calibration scheme its
    arguments name: one marginal quantile; one per cohort (``groups`` with
    ``test_groups``, a cohort unseen at calibration held to the pooled
    quantile); or one per test row's likelihood ratio (``weights`` with
    ``test_weights``)."""
    marginal = groups is None and test_groups is None
    unweighted = weights is None and test_weights is None
    if marginal and unweighted:
        return [finite_sample_quantile(cal_scores, alpha)] * test_rows
    if groups is not None and test_groups is not None and unweighted:
        _one_per_row("cohort", len(groups), len(cal_scores))
        _one_per_test_row(len(test_groups), test_rows)
        _validate_alpha(alpha)
        by_group: dict = {}
        for score, group in zip(cal_scores, groups):
            by_group.setdefault(group, []).append(score)
        quantiles = {
            group: finite_sample_quantile(scores, alpha)
            for group, scores in by_group.items()
        }
        pooled = finite_sample_quantile(cal_scores, alpha)
        return [quantiles.get(group, pooled) for group in test_groups]
    if weights is not None and test_weights is not None and marginal:
        _one_per_row("weight", len(weights), len(cal_scores))
        _one_per_test_row(len(test_weights), test_rows)
        _validate_alpha(alpha)
        calibration = _WeightedScores(cal_scores, weights)
        return [calibration.threshold(w, alpha) for w in test_weights]
    raise ValueError(
        "conformal takes `groups` with `test_groups`, or `weights` with "
        "`test_weights`, and not both"
    )


def _one_per_row(what: str, given: int, rows: int) -> None:
    if given != rows:
        raise ValueError(
            f"conformal needs one {what} per calibration row: {given} for {rows} rows"
        )


def _one_per_test_row(given: int, rows: int) -> None:
    if given != rows:
        raise ValueError(
            f"conformal needs one entry per test row: {given} for {rows} rows"
        )


def score_family(score: Optional[str]) -> str:
    """The classification score family a verb's ``score`` names (default
    ``"aps"``)."""
    family = (score or "aps").lower()
    if family not in ("lac", "aps", "raps"):
        raise ValueError(
            f"unknown classification conformal score '{score}', "
            "expected 'lac', 'aps', or 'raps'"
        )
    return family


def _aps_cumulative_mass(
    probs: Sequence[float], target: int, reg: Optional[Tuple[float, int]]
) -> float:
    """APS (and RAPS) nonconformity for one candidate class.

    The cumulative probability mass of classes ranked most- to least-probable,
    up to and including ``target``, plus the RAPS rank penalty
    ``lambda * max(0, rank_1based - k_reg)`` when ``reg`` is supplied. Ties in
    probability break by ascending class index so the ordering — and thus the
    score — is deterministic, matching the engine's ``total_cmp`` tiebreak.
    """
    # Descending probability; index ascending on ties for determinism.
    order = sorted(range(len(probs)), key=lambda c: (-probs[c], c))
    cumulative = 0.0
    for rank, cls in enumerate(order):
        cumulative += probs[cls]
        if cls == target:
            if reg is not None:
                lam, k_reg = reg
                # 1-based rank; penalize ranks beyond k_reg.
                overflow = max(0, (rank + 1) - k_reg)
                cumulative += lam * overflow
            return cumulative
    # `target` is in range (checked by the caller), so it is always found.
    return cumulative


def _validate_probabilities(probs: Sequence[float]) -> None:
    """A per-class probability row must be non-empty and NaN-free.

    The values need not sum to one — conformal reads the *ordering* and the
    true-class mass, not a normalized distribution — but a NaN would silently
    corrupt the rank.
    """
    if not probs:
        raise ValueError(
            "classification conformal requires at least one class probability"
        )
    if any(math.isnan(p) for p in probs):
        raise ValueError("classification conformal probabilities must be finite")


def _true_label_score(
    probs: Sequence[float], label: int, score: str, reg: Tuple[float, int]
) -> float:
    """The nonconformity of one calibration row evaluated at its true class."""
    _validate_probabilities(probs)
    if label >= len(probs) or label < 0:
        raise ValueError(
            f"true label {label} out of range for {len(probs)} classes"
        )
    if score == "lac":
        return 1.0 - probs[label]
    if score == "aps":
        return _aps_cumulative_mass(probs, label, None)
    # raps
    return _aps_cumulative_mass(probs, label, reg)


def _admit_classes(
    probs: Sequence[float], score: str, q: float, reg: Tuple[float, int]
) -> List[int]:
    """The classes whose nonconformity at this row is ``<= q̂``, ascending.

    An infinite threshold admits every class. For APS/RAPS a candidate's score
    is the cumulative mass up to and including it — exactly ``_true_label_score``
    with the candidate standing in for the true label.
    """
    _validate_probabilities(probs)
    if math.isinf(q):
        return list(range(len(probs)))
    if score == "lac":
        return [c for c in range(len(probs)) if 1.0 - probs[c] <= q]
    if score == "aps":
        return [
            c
            for c in range(len(probs))
            if _aps_cumulative_mass(probs, c, None) <= q
        ]
    # raps
    return [
        c for c in range(len(probs)) if _aps_cumulative_mass(probs, c, reg) <= q
    ]


def conformalize(
    calibration: Sequence[Sequence[float]],
    true_labels: Sequence[int],
    test: Sequence[Sequence[float]],
    *,
    alpha: float,
    score: Optional[str] = None,
    raps_params: Optional[Tuple[float, int]] = None,
    groups: Optional[Sequence[str]] = None,
    test_groups: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
    test_weights: Optional[Sequence[float]] = None,
) -> List[List[int]]:
    """Conformalize a classification predictor into prediction sets.

    Split (inductive) conformal: ``calibration`` holds one row of per-class
    probabilities per held-out example and ``true_labels[i]`` is the realised
    class for row ``i``; the calibration scores yield the finite-sample
    ``⌈(n+1)(1-alpha)⌉`` quantile, applied to every row of ``test`` to emit a
    prediction set with marginal coverage ``>= 1 - alpha``.

    ``score`` selects the nonconformity family: ``"lac"``, ``"aps"`` (default),
    or ``"raps"`` (regularized APS). For ``"raps"``, ``raps_params`` is the
    ``(lambda, k_reg)`` pair — the penalty weight and the 1-based rank past
    which it applies; ignored by ``"lac"``/``"aps"`` and defaulting to
    ``(0.0, 1)``.

    ``groups`` with ``test_groups`` calibrates one quantile per cohort
    (Mondrian, group-conditional coverage): ``groups[i]`` is calibration row
    ``i``'s cohort and ``test_groups[j]`` test row ``j``'s; a cohort unseen at
    calibration is held to the pooled quantile. ``weights`` with
    ``test_weights`` are likelihood ratios ``dP_test/dP_cal`` for a known
    covariate shift (weighted conformal, Tibshirani et al. 2019): each test
    row's own ratio sets its threshold. The calibration set must be disjoint
    from both the training set and ``test`` — reusing test points inflates
    coverage. Pure and deterministic: identical inputs yield identical sets.

    Returns one list of admitted class indices per row of ``test``.
    """
    family = score_family(score)
    reg = raps_params if raps_params is not None else (0.0, 1)
    if len(calibration) != len(true_labels):
        raise ValueError(
            f"classification conformal: {len(calibration)} probability rows "
            f"but {len(true_labels)} labels"
        )
    if not calibration:
        raise ValueError(
            "classification conformal requires at least one calibration row"
        )
    cal_scores = [
        _true_label_score(probs, label, family, reg)
        for probs, label in zip(calibration, true_labels)
    ]
    thresholds = _thresholds(
        cal_scores, len(test), alpha, groups, test_groups, weights, test_weights
    )
    return [_admit_classes(row, family, q, reg) for row, q in zip(test, thresholds)]


def conformalize_interval(
    predictions: Sequence[float],
    observed: Sequence[float],
    test_predictions: Sequence[float],
    *,
    alpha: float,
    groups: Optional[Sequence[str]] = None,
    test_groups: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
    test_weights: Optional[Sequence[float]] = None,
) -> List[Tuple[float, float]]:
    """Conformalize an absolute-residual regression predictor into intervals.

    The calibration nonconformity is ``|y - ŷ|`` over the ``predictions`` /
    ``observed`` held-out pairs; the finite-sample quantile ``q̂`` then yields
    the constant-width interval ``[ŷ - q̂, ŷ + q̂]`` around each
    ``test_predictions`` point, with marginal coverage ``>= 1 - alpha``.
    ``groups``/``test_groups`` and ``weights``/``test_weights`` calibrate as in
    :func:`conformalize`. The calibration set must be disjoint from the training
    set and the test points.

    Returns one ``(lower, upper)`` tuple per test row.
    """
    if len(predictions) != len(observed):
        raise ValueError(
            f"absolute-residual conformal: {len(predictions)} predictions "
            f"but {len(observed)} observations"
        )
    if not predictions:
        raise ValueError(
            "regression conformal requires at least one calibration row"
        )
    _ensure_finite(predictions, "predictions")
    _ensure_finite(observed, "observations")
    cal_scores = [abs(y - yhat) for yhat, y in zip(predictions, observed)]
    thresholds = _thresholds(
        cal_scores, len(test_predictions), alpha, groups, test_groups, weights, test_weights
    )
    return [(p - q, p + q) for p, q in zip(test_predictions, thresholds)]


def conformalize_cqr(
    lower: Sequence[float],
    upper: Sequence[float],
    observed: Sequence[float],
    test_lower: Sequence[float],
    test_upper: Sequence[float],
    *,
    alpha: float,
    groups: Optional[Sequence[str]] = None,
    test_groups: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
    test_weights: Optional[Sequence[float]] = None,
) -> List[Tuple[float, float]]:
    """Conformalize a Conformalized Quantile Regression predictor into intervals.

    The calibration nonconformity is ``max(q_lo - y, y - q_hi)`` over the
    ``lower`` / ``upper`` quantile estimates and ``observed`` targets; the
    finite-sample quantile ``q̂`` then yields the adaptive-width interval
    ``[q_lo - q̂, q_hi + q̂]`` around each ``test_lower`` / ``test_upper`` band,
    so width tracks the predictor's local uncertainty.
    ``groups``/``test_groups`` and ``weights``/``test_weights`` calibrate as in
    :func:`conformalize`. The calibration set must be disjoint from the training
    set and the test points.

    Returns one ``(lower, upper)`` tuple per test row.
    """
    if len(lower) != len(observed) or len(upper) != len(observed):
        raise ValueError(
            f"CQR conformal: {len(lower)} lower / {len(upper)} upper quantiles "
            f"but {len(observed)} observations"
        )
    if not observed:
        raise ValueError(
            "regression conformal requires at least one calibration row"
        )
    if len(test_lower) != len(test_upper):
        raise ValueError(
            "cqr conformal: 'test_lower' and 'test_upper' must have equal length"
        )
    _ensure_finite(lower, "lower quantiles")
    _ensure_finite(upper, "upper quantiles")
    _ensure_finite(observed, "observations")
    cal_scores = [
        max(lo - y, y - hi) for lo, hi, y in zip(lower, upper, observed)
    ]
    thresholds = _thresholds(
        cal_scores, len(test_lower), alpha, groups, test_groups, weights, test_weights
    )
    return [
        (lo - q, hi + q) for lo, hi, q in zip(test_lower, test_upper, thresholds)
    ]


def rrf_fuse(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k_rrf: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Fuse several ranked retrieval lists into one by reciprocal-rank fusion.

    Each entry of ``ranked_lists`` is one retriever's output: a best-first list
    of ``_row_id``s (position 0 is rank 0). A row's fused score is
    ``Σ_lists 1 / (k_rrf + rank_in_list + 1)`` — the ``+1`` makes the best rank
    contribute ``1/(k_rrf+1)``, matching the canonical 1-based-rank
    formulation while keeping the call site 0-based. Fusion is on *rank*, never
    raw score, so dense (ANN) and lexical (BM25) scales never need reconciling.

    The result is sorted by fused score descending; ties break ascending by
    ``row_id``, so the output is fully deterministic and independent of the
    order the lists are supplied in. A row repeated within a single list counts
    only at its first (best) occurrence in that list. ``k_rrf`` damps deep ranks
    (default 60, robust across 40–80).

    Returns ``(row_id, rrf_score)`` tuples.
    """
    k = float(DEFAULT_K_RRF if k_rrf is None else k_rrf)
    scores: dict = {}
    for ranked in ranked_lists:
        # First occurrence wins within a list — a retriever ranking the same
        # row twice must not double-count it.
        seen: set = set()
        for rank, row_id in enumerate(ranked):
            if row_id in seen:
                continue
            seen.add(row_id)
            scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (k + rank + 1.0)
    # Sort by fused score descending, ties ascending by row_id.
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _ensure_finite(values: Sequence[float], what: str) -> None:
    """Reject non-finite values in a regression input slice."""
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"regression conformal {what} must be finite")
