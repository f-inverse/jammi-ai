"""Conformal coverage under covariate shift — the consumer-side numerics the
book measures the engine's marginal conformal against.

The engine ships the marginal split-conformal surface (``conformalize``,
``conformalize_interval``, ``conformalize_cqr``). Whether a covariate shift can
be repaired by *weighting* the calibration set toward the test distribution
(Tibshirani et al. 2019) is the consumer's question, so it is answered here,
in plain numpy: one weighted split-conformal quantile for both the marginal
and the weighted passes — the marginal pass is the uniform-weight case, so a
coverage change is attributable to the weights alone — the kNN density-ratio
weights, and the diagnostics that explain a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Weights:
    """Test-to-calibration likelihood ratios: one per calibration row and one
    per test row. Weighted split-conformal needs both — each test row's own
    weight is the mass its quantile holds back at +∞."""

    cal: np.ndarray
    test: np.ndarray


def density_ratio(test_share: np.ndarray, neighbours: int) -> np.ndarray:
    """Each row's test-to-calibration likelihood ratio, estimated as the
    Laplace-smoothed odds of a test-era neighbour among its ``neighbours``
    nearest (a kNN density-ratio estimate; Tibshirani et al. 2019). Smoothing
    bounds the odds by ``neighbours + 1``, so no single row dominates the
    weighted quantile."""
    tests = test_share * neighbours
    return (tests + 1) / (neighbours - tests + 1)


def aps_nonconformity(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The APS nonconformity per labeled row: the cumulative softmax mass of the
    classes ranked at least as high as the true class, the true class included."""
    out = np.empty(len(labels))
    for i, label in enumerate(labels):
        cum = 0.0
        for j in np.argsort(-scores[i]):
            cum += scores[i][j]
            if j == label:
                break
        out[i] = cum
    return out


def quantiles(
    cal: np.ndarray, n_test: int, *, weights: Weights | None, alpha: float
) -> np.ndarray:
    """Each test row's split-conformal ``1−α`` quantile of the calibration
    nonconformity ``cal`` (Tibshirani et al. 2019): calibration row ``i`` holds
    mass ``w_i / (Σw + w_test)`` and the test row its own ``w_test`` at +∞, so
    the quantile is +∞ when the calibration mass cannot reach ``1−α``. With no
    weights every row weighs one and this is the ⌈(n+1)(1−α)⌉-th smallest — the
    marginal quantile."""
    w = np.ones(len(cal)) if weights is None else np.asarray(weights.cal, dtype=float)
    w_test = np.ones(n_test) if weights is None else np.asarray(weights.test, dtype=float)
    order = np.argsort(cal)
    sorted_cal = np.append(np.asarray(cal, dtype=float)[order], np.inf)
    reach = np.searchsorted(np.cumsum(w[order]), (1 - alpha) * (w.sum() + w_test))
    return sorted_cal[np.minimum(reach, len(cal))]


def score_coverage(
    cal: np.ndarray, test: np.ndarray, *, weights: Weights | None, alpha: float
) -> float:
    """The share of test rows whose nonconformity is within their conformal
    quantile — the coverage of any split-conformal set or interval built on
    this score (an absolute residual's ``ŷ ± q̂``, an APS set)."""
    return float(np.mean(test <= quantiles(cal, len(test), weights=weights, alpha=alpha)))


def aps_coverage(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    *,
    weights: Weights | None,
    alpha: float,
) -> tuple[float, float]:
    """Split-APS coverage and mean set size, marginal (``weights=None``) or
    weighted. A class is admitted while the cumulative mass up to and including
    it stays ≤ q̂ — the class that crosses q̂ is excluded; ties break by class
    index. One convention for both passes, so only the weights differ."""
    q = quantiles(
        aps_nonconformity(cal_scores, cal_labels), len(test_labels), weights=weights, alpha=alpha
    )
    covered = size = 0
    for scores, label, q_row in zip(test_scores, test_labels, q, strict=True):
        cum, admitted = 0.0, set()
        for c in sorted(range(len(scores)), key=lambda c: (-scores[c], c)):
            cum += scores[c]
            if cum <= q_row:
                admitted.add(c)
        covered += int(label in admitted)
        size += len(admitted)
    return covered / len(test_labels), size / len(test_labels)
