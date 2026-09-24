"""Conformal coverage under covariate shift — the consumer-side numerics the
book measures the engine's marginal conformal against.

The engine ships the marginal split-conformal surface (``conformalize``,
``conformalize_interval``, ``conformalize_cqr``). Whether a covariate shift can
be repaired by *weighting* the calibration set toward the test distribution
(Tibshirani et al. 2019) is the consumer's question, so it is answered here,
in plain numpy: one self-consistent local APS routine for both the marginal
and the weighted passes (so a coverage change is attributable to the weights
alone), three importance-weighting schemes, and the diagnostics that explain a
no-op.
"""

from __future__ import annotations

import numpy as np

# Temperature on the nearest-centroid cosine logits.
SOFTMAX_TEMP = 8.0
# Temperature on the test-likeness weights.
WEIGHTED_TEMP = 5.0
# Neighbours for the kNN density-ratio weights.
KNN_WEIGHTS = 25


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Each row scaled to unit length."""
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)


def nearest_centroid_scores(
    embeddings: np.ndarray, labels: np.ndarray, train: np.ndarray, rows: np.ndarray, classes: int
) -> np.ndarray:
    """Softmax class scores for ``rows`` from a nearest-centroid head fitted on
    ``train``: each class's centroid is the mean unit embedding of its training
    rows, and a row's logits are its cosine to every centroid."""
    norm = unit_rows(embeddings)
    centroids = np.zeros((classes, norm.shape[1]))
    for c in range(classes):
        members = train[labels[train] == c]
        if len(members):
            centroids[c] = norm[members].mean(0)
    centroids = unit_rows(centroids)
    logits = norm[rows] @ centroids.T * SOFTMAX_TEMP
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def test_likeness(cal_embeddings: np.ndarray, test_embeddings: np.ndarray) -> np.ndarray:
    """Each calibration row's cosine to the test era's mean direction."""
    centre = test_embeddings.mean(0)
    return cal_embeddings @ (centre / (np.linalg.norm(centre) + 1e-12))


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
    for scores, label in zip(test_scores, test_labels):
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


def weight_schemes(cal_embeddings: np.ndarray, test_embeddings: np.ndarray) -> dict[str, np.ndarray]:
    """Three estimators of each calibration row's test-era likelihood ratio: a
    centroid-likeness softmax, a kNN density ratio, and a logistic-regression
    domain classifier's odds (Tibshirani et al. 2019)."""
    centroid = np.exp(WEIGHTED_TEMP * test_likeness(cal_embeddings, test_embeddings))

    pool = np.vstack([cal_embeddings, test_embeddings])
    is_test = np.concatenate([np.zeros(len(cal_embeddings)), np.ones(len(test_embeddings))])
    sims = cal_embeddings @ pool.T
    np.fill_diagonal(sims[:, : len(cal_embeddings)], -np.inf)
    k = min(KNN_WEIGHTS, len(pool) - 1)
    nearest = np.argpartition(-sims, k, axis=1)[:, :k]
    frac_test = is_test[nearest].mean(1)
    knn = (frac_test + 1e-3) / (1 - frac_test + 1e-3)

    return {"centroid": centroid, "knn": knn, "domain_lr": _domain_odds(cal_embeddings, test_embeddings)}


def _domain_odds(cal_embeddings: np.ndarray, test_embeddings: np.ndarray) -> np.ndarray:
    """A batch-gradient logistic regression of test membership on the
    embedding; the fitted odds on the calibration rows. Deterministic: fixed
    initialization and step count."""
    x = np.vstack([cal_embeddings, test_embeddings])
    t = np.concatenate([np.zeros(len(cal_embeddings)), np.ones(len(test_embeddings))])
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = (x - mu) / sd
    w = np.random.default_rng(0).normal(0, 0.01, xs.shape[1])
    b = 0.0
    for _ in range(300):
        p = 1.0 / (1.0 + np.exp(-(xs @ w + b)))
        g = p - t
        w -= 0.5 * (xs.T @ g / len(t) + 1e-3 * w)
        b -= 0.5 * g.mean()
    p_cal = np.clip(1.0 / (1.0 + np.exp(-(((cal_embeddings - mu) / sd) @ w + b))), 1e-3, 1 - 1e-3)
    return p_cal / (1 - p_cal)
