"""Hermetic unit tests for the consumer-side weighted conformal numerics.

The chapters compare the engine's marginal conformal with a weighted pass; the
comparison is only honest if the two passes are one routine that differs in
its weights, and if the weighted pass carries the test row's own mass the way
Tibshirani et al. (2019) state it.
"""

from __future__ import annotations

import numpy as np

from jammi_cookbook import shift

ALPHA = 0.1


def test_uniform_weights_are_the_marginal_quantile():
    cal = np.random.default_rng(0).random(99)
    k = int(np.ceil((len(cal) + 1) * (1 - ALPHA)))
    marginal = shift.quantiles(cal, 3, weights=None, alpha=ALPHA)
    uniform = shift.quantiles(
        cal, 3, weights=shift.Weights(cal=np.full(99, 2.0), test=np.full(3, 2.0)), alpha=ALPHA
    )
    assert np.all(marginal == np.sort(cal)[k - 1])
    assert np.array_equal(marginal, uniform)


def test_a_test_row_heavier_than_the_calibration_tail_gets_an_unbounded_quantile():
    cal = np.arange(9.0)  # (9+1)(1-α) = 9: the largest value, the most it can reach
    assert shift.quantiles(cal, 1, weights=None, alpha=ALPHA)[0] == 8.0
    heavy = shift.Weights(cal=np.ones(9), test=np.array([1.0, 5.0]))
    q = shift.quantiles(cal, 2, weights=heavy, alpha=ALPHA)
    assert q[0] == 8.0 and np.isinf(q[1])


def test_exact_weights_restore_what_a_score_aligned_shift_breaks():
    # A covariate x drives the score upward; the calibration set keeps each row with
    # probability π(x) = 1/(1+e^{2x}), leaning it toward easy rows, and the
    # likelihood ratio 1/π is known exactly. Coverage is an expectation over the
    # calibration draw, so it is averaged over draws.
    rng = np.random.default_rng(7)

    def draw():
        x_cal, x_test = rng.normal(size=400), rng.normal(size=400)
        score = lambda x: x + rng.normal(scale=0.5, size=len(x))  # noqa: E731
        kept = rng.random(400) < 1 / (1 + np.exp(2 * x_cal))
        x_cal = x_cal[kept]
        cal, test = score(x_cal), score(x_test)
        w = shift.Weights(cal=1 + np.exp(2 * x_cal), test=1 + np.exp(2 * x_test))
        return (
            shift.score_coverage(cal, test, weights=None, alpha=ALPHA),
            shift.score_coverage(cal, test, weights=w, alpha=ALPHA),
        )

    marginal, weighted = np.array([draw() for _ in range(300)]).mean(axis=0)
    # The guarantee is a floor: heavy weights leave few effective calibration
    # rows, and the test row's mass at +∞ makes the weighted pass conservative.
    assert marginal < 1 - ALPHA - 0.1
    assert 1 - ALPHA <= weighted < 1 - ALPHA + 0.05
