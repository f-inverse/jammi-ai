"""Covariate-shift helpers the conformal chapters build on: the weights a caller
estimates and the yardstick a coverage is read against.

The engine applies weighted split-conformal (``weights=``/``test_weights=`` on
``conformalize``, ``conformalize_interval`` and ``conformalize_cqr``);
*estimating* the likelihood ratios is the caller's, so the kNN density-ratio
estimate lives here, beside the APS nonconformity the chapters correlate with a
shift and the binomial band a measured coverage is held to.
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


def binomial_band(alpha: float, n: int) -> float:
    """Three binomial standard errors of a ``1 − alpha`` coverage measured over
    ``n`` test points: the band a valid split-conformal coverage lands within on
    all but a rare draw."""
    return 3 * float(np.sqrt(alpha * (1 - alpha) / n))
