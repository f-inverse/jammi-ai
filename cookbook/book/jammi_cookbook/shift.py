"""Conformal coverage under covariate shift — the consumer-side numerics the
book measures the engine's marginal conformal against.

The engine ships the marginal split-conformal surface (``conformalize``,
``conformalize_interval``, ``conformalize_cqr``). Whether a covariate shift can
be repaired by *weighting* the calibration set toward the test distribution
(Tibshirani et al. 2019) is the consumer's question, so it is answered here,
in plain numpy: one self-consistent local APS routine for both the marginal
and the weighted passes (so a coverage change is attributable to the weights
alone), the kNN density-ratio weights, and the diagnostics that explain a
no-op.
"""

from __future__ import annotations

import numpy as np


def density_ratio(test_share: np.ndarray, neighbours: int) -> np.ndarray:
    """Each calibration row's test-to-calibration likelihood ratio, estimated
    as the Laplace-smoothed odds of a test-era neighbour among its
    ``neighbours`` nearest (a kNN density-ratio estimate; Tibshirani et al.
    2019). Smoothing bounds the odds by ``neighbours + 1``, so no single row
    dominates the weighted quantile."""
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


def _quantile(values: np.ndarray, weights: np.ndarray | None, alpha: float) -> float:
    """The finite-sample ``1−α`` quantile of ``values``: unweighted, the
    ⌈(n+1)(1−α)⌉-th smallest; weighted, the smallest value whose reweighted
    empirical CDF reaches ``1−α``."""
    n = len(values)
    order = np.argsort(values)
    if weights is None:
        return float(values[order][min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1])
    w = np.asarray(weights, dtype=float)
    cdf = np.cumsum((w / w.sum())[order])
    return float(values[order][min(int(np.searchsorted(cdf, 1 - alpha)), n - 1)])


def aps_coverage(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    *,
    weights: np.ndarray | None,
    alpha: float,
) -> tuple[float, float]:
    """Split-APS coverage and mean set size, marginal (``weights=None``) or
    weighted. A class is admitted while the cumulative mass up to and including
    it stays ≤ q̂ — the class that crosses q̂ is excluded; ties break by class
    index. One convention for both passes, so only the weights differ."""
    q = _quantile(aps_nonconformity(cal_scores, cal_labels), weights, alpha)
    covered = size = 0
    for scores, label in zip(test_scores, test_labels, strict=True):
        cum, admitted = 0.0, set()
        for c in sorted(range(len(scores)), key=lambda c: (-scores[c], c)):
            cum += scores[c]
            if cum <= q:
                admitted.add(c)
        covered += int(label in admitted)
        size += len(admitted)
    return covered / len(test_labels), size / len(test_labels)


def residual_coverage(
    cal_residuals: np.ndarray, test_residuals: np.ndarray, *, weights: np.ndarray, alpha: float
) -> float:
    """Coverage of the ``ŷ ± q̂`` interval whose q̂ is the weighted quantile of
    the calibration absolute residuals."""
    q = _quantile(cal_residuals, weights, alpha)
    return float(np.mean(test_residuals <= q))
