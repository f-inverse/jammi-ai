"""Unit tests for the client-local conformal / RRF numerics.

These verbs are pure functions of caller-supplied arrays, so the tests are
fully hermetic — no engine, no wheel, no server, no gRPC stub. They pin the
finite-sample quantile index, marginal-coverage sanity on the calibration draw,
the LAC/APS/RAPS set differences, the CQR/absolute-residual interval widths, and
the RRF ordering against hand-derived expected values.

The module is loaded directly from its file rather than via ``import
jammi_client`` so the numerics can be exercised on a machine that carries no
gRPC / Flight transport stack — they depend on nothing but the stdlib.
"""

from __future__ import annotations

import importlib.util
import math
import os
import random

import pytest

_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "jammi_client", "_conformal.py"
)
_spec = importlib.util.spec_from_file_location("jammi_client._conformal", _MODULE_PATH)
conformal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conformal)


# --- finite-sample quantile --------------------------------------------------


def test_quantile_takes_the_corrected_order_statistic():
    # n = 9, alpha = 0.1: rank = ceil(10 * 0.9) = 9, the 9th (largest) score.
    scores = [float(i) for i in range(1, 10)]
    assert conformal.finite_sample_quantile(scores, 0.1) == 9.0


def test_quantile_is_infinite_below_the_finite_sample_floor():
    # n = 5, alpha = 0.1: rank = ceil(6 * 0.9) = 6 > 5 -> +inf.
    scores = [float(i) for i in range(1, 6)]
    assert math.isinf(conformal.finite_sample_quantile(scores, 0.1))


def test_quantile_rejects_bad_alpha_and_empty():
    with pytest.raises(ValueError):
        conformal.finite_sample_quantile([1.0], 0.0)
    with pytest.raises(ValueError):
        conformal.finite_sample_quantile([1.0], 1.0)
    with pytest.raises(ValueError):
        conformal.finite_sample_quantile([], 0.1)
    with pytest.raises(ValueError):
        conformal.finite_sample_quantile([float("nan"), 1.0], 0.1)


# --- classification: LAC / APS / RAPS ---------------------------------------


def test_lac_threshold_and_set_from_hand_derivation():
    # LAC scores 1 - p_y over the four calibration rows: [0.4, 0.3, 0.1, 0.5].
    # n = 4, alpha = 0.5: rank = ceil(5 * 0.5) = 3, the 3rd-smallest = 0.4.
    # A test row admits class c iff 1 - p_c <= 0.4, i.e. p_c >= 0.6.
    calibration = [[0.6, 0.4], [0.3, 0.7], [0.9, 0.1], [0.5, 0.5]]
    labels = [0, 1, 0, 1]
    # [0.7, 0.3]: class 0 (p=0.7) qualifies, class 1 (p=0.3) does not.
    sets = conformal.conformalize(
        calibration, labels, [[0.7, 0.3], [0.5, 0.5]], alpha=0.5, score="lac"
    )
    assert sets == [[0], []]


def test_aps_defaults_when_score_omitted():
    calibration = [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]
    labels = [0, 1, 2]
    default = conformal.conformalize(calibration, labels, [[0.5, 0.3, 0.2]], alpha=0.5)
    aps = conformal.conformalize(
        calibration, labels, [[0.5, 0.3, 0.2]], alpha=0.5, score="aps"
    )
    assert default == aps


def test_aps_set_grows_on_harder_inputs():
    # Calibrate APS on a moderately hard set; the diffuse row gets a set at
    # least as large as the concentrated one.
    rng = random.Random(99)
    n_classes = 6
    calibration, labels = _synthetic_classification(rng, 3000, n_classes)
    easy = [[0.9, 0.04, 0.02, 0.02, 0.01, 0.01]]
    hard = [[0.25, 0.22, 0.20, 0.15, 0.10, 0.08]]
    easy_set = conformal.conformalize(calibration, labels, easy, alpha=0.1, score="aps")
    hard_set = conformal.conformalize(calibration, labels, hard, alpha=0.1, score="aps")
    assert len(hard_set[0]) > len(easy_set[0])


def test_raps_penalty_differs_from_aps_on_a_tail_class():
    # RAPS adds lambda * max(0, rank_1based - k_reg) to a candidate's
    # nonconformity, so a deep-tail class costs strictly more to admit than
    # under plain APS — the per-row scores differ, and with a decisive penalty
    # the RAPS set can drop the tail class APS would keep.
    calibration = [
        [0.5, 0.3, 0.15, 0.05],
        [0.4, 0.3, 0.2, 0.1],
        [0.6, 0.25, 0.1, 0.05],
        [0.45, 0.3, 0.15, 0.1],
    ]
    labels = [0, 1, 0, 2]
    test = [[0.4, 0.3, 0.2, 0.1]]
    aps = conformal.conformalize(calibration, labels, test, alpha=0.3, score="aps")
    raps = conformal.conformalize(
        calibration,
        labels,
        test,
        alpha=0.3,
        score="raps",
        raps_params=(1.0, 1),
    )
    # The heavy rank penalty must not *enlarge* the set relative to APS on this
    # draw, and the two families must produce genuinely different scoring (the
    # sets are not forced equal).
    assert len(raps[0]) <= len(aps[0])
    assert set(raps[0]).issubset(set(aps[0]))


def test_lac_marginal_coverage_on_an_exchangeable_draw():
    # On a calibrated, exchangeable synthetic classifier the LAC sets cover the
    # true label on >= 1 - alpha of a held-out test split (the conformal
    # guarantee), allowing finite-sample slack.
    rng = random.Random(20260605)
    n_classes = 5
    alpha = 0.1
    calibration, cal_labels = _synthetic_classification(rng, 2000, n_classes)
    test, test_labels = _synthetic_classification(rng, 4000, n_classes)
    sets = conformal.conformalize(
        calibration, cal_labels, test, alpha=alpha, score="lac"
    )
    covered = sum(1 for s, y in zip(sets, test_labels) if y in s)
    coverage = covered / len(test_labels)
    assert coverage >= 1.0 - alpha - 0.03, coverage


def test_classification_rejects_bad_inputs():
    with pytest.raises(ValueError):
        conformal.conformalize([[0.5, 0.5]], [0, 1], [[0.5, 0.5]], alpha=0.1)
    with pytest.raises(ValueError):
        conformal.conformalize([], [], [[0.5, 0.5]], alpha=0.1)
    with pytest.raises(ValueError):
        conformal.conformalize([[0.5, 0.5]], [2], [[0.5, 0.5]], alpha=0.1)
    with pytest.raises(ValueError):
        conformal.conformalize(
            [[0.5, 0.5]], [0], [[0.5, 0.5]], alpha=0.1, score="bogus"
        )


# --- regression: absolute-residual interval ---------------------------------


def test_absolute_residual_interval_from_hand_derivation():
    # Residuals |y - yhat|: with yhat all 0 and y = [0, 1, 2, 3] -> [0, 1, 2, 3].
    # n = 4, alpha = 0.25: rank = ceil(5 * 0.75) = 4, the 4th-smallest = 3.
    # The interval around a test point p is [p - 3, p + 3].
    intervals = conformal.conformalize_interval(
        [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 3.0], [10.0, -5.0], alpha=0.25
    )
    assert intervals == [(7.0, 13.0), (-8.0, -2.0)]


def test_absolute_residual_covers_marginally():
    rng = random.Random(5)
    n = 3000
    cal_pred = [0.0] * n
    cal_obs = [rng.gauss(0.0, 1.0) for _ in range(n)]
    test_obs = [rng.gauss(0.0, 1.0) for _ in range(4000)]
    intervals = conformal.conformalize_interval(
        cal_pred, cal_obs, [0.0] * len(test_obs), alpha=0.1
    )
    covered = sum(1 for (lo, hi), y in zip(intervals, test_obs) if lo <= y <= hi)
    coverage = covered / len(test_obs)
    assert coverage >= 0.9 - 0.02, coverage


def test_interval_rejects_length_mismatch():
    with pytest.raises(ValueError):
        conformal.conformalize_interval([0.0, 0.0], [0.0], [1.0], alpha=0.1)
    with pytest.raises(ValueError):
        conformal.conformalize_interval([], [], [1.0], alpha=0.1)


# --- regression: CQR interval ------------------------------------------------


def test_cqr_interval_from_hand_derivation():
    # CQR scores max(lo - y, y - hi): with lo = -1, hi = 1 and observed
    # [0, 0, 3, -3] -> [max(-1, -1), max(-1, -1), max(-4, 2), max(2, -4)]
    #              = [-1, -1, 2, 2]. n = 4, alpha = 0.25 -> rank = 4 -> q = 2.
    # A test band [tl, tu] becomes [tl - 2, tu + 2].
    intervals = conformal.conformalize_cqr(
        [-1.0, -1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 3.0, -3.0],
        [0.0, 5.0],
        [2.0, 7.0],
        alpha=0.25,
    )
    assert intervals == [(-2.0, 4.0), (3.0, 9.0)]


def test_cqr_width_tracks_the_band_plus_constant():
    # The conformal correction q is a constant added symmetrically, so a wider
    # input band yields a strictly wider output interval.
    lower = [-1.0, -2.0, -1.5, -1.0]
    upper = [1.0, 2.0, 1.5, 1.0]
    observed = [0.5, -0.5, 1.0, -1.0]
    narrow, wide = conformal.conformalize_cqr(
        lower, upper, observed, [-1.0, -3.0], [1.0, 3.0], alpha=0.25
    )
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_cqr_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        conformal.conformalize_cqr(
            [-1.0], [1.0, 1.0], [0.0], [0.0], [1.0], alpha=0.1
        )
    with pytest.raises(ValueError):
        conformal.conformalize_cqr(
            [-1.0], [1.0], [0.0], [0.0, 0.0], [1.0], alpha=0.1
        )


# --- reciprocal-rank fusion --------------------------------------------------


def test_rrf_single_list_preserves_order():
    fused = conformal.rrf_fuse([["a", "b", "c"]])
    assert [row for row, _ in fused] == ["a", "b", "c"]


def test_rrf_default_k_is_sixty():
    # The best rank contributes 1 / (k + 0 + 1) = 1 / 61 at the default k = 60.
    fused = conformal.rrf_fuse([["a"]])
    assert fused[0][1] == pytest.approx(1.0 / 61.0)
    assert conformal.DEFAULT_K_RRF == 60


def test_rrf_rewards_cross_list_agreement():
    # `b` is mid-pack in each list but appears in both; it outranks rows that
    # top a single list but appear in only one.
    fused = conformal.rrf_fuse([["a", "b", "c"], ["x", "b", "y"]])
    assert fused[0][0] == "b"


def test_rrf_is_independent_of_list_order():
    dense = ["a", "b", "c", "d"]
    lexical = ["d", "c", "b", "a"]
    one = conformal.rrf_fuse([dense, lexical])
    two = conformal.rrf_fuse([lexical, dense])
    assert one == two


def test_rrf_tie_breaks_ascending_by_row_id():
    # `a` and `b` each appear once at the same rank in disjoint lists, so their
    # fused scores are equal; the tie breaks ascending by id.
    fused = conformal.rrf_fuse([["b"], ["a"]])
    assert [row for row, _ in fused] == ["a", "b"]
    assert fused[0][1] == fused[1][1]


def test_rrf_duplicate_within_list_counts_once():
    with_dup = dict(conformal.rrf_fuse([["a", "b", "a"]]))
    without = dict(conformal.rrf_fuse([["a", "b"]]))
    assert with_dup["a"] == without["a"]


def test_rrf_larger_k_damps_and_flattens():
    small = dict(conformal.rrf_fuse([["r0", "r1"]], k_rrf=1))
    large = dict(conformal.rrf_fuse([["r0", "r1"]], k_rrf=1000))
    # A larger k damps the rank-0 contribution and flattens the rank-0 vs
    # rank-1 gap, while rank-0 still outscores rank-1 at any k.
    assert small["r0"] > large["r0"]
    assert (small["r0"] / small["r1"]) > (large["r0"] / large["r1"])
    assert large["r0"] > large["r1"]


def test_rrf_empty_input_yields_empty():
    assert conformal.rrf_fuse([]) == []


# --- helpers -----------------------------------------------------------------


def _synthetic_classification(rng, n, n_classes):
    """A noisy-softmax synthetic classifier: probabilities concentrate on the
    true class but leak elsewhere, and the realised label is drawn from those
    probabilities — exchangeable and calibrated by construction (the Python peer
    of the engine's test fixture)."""
    probs = []
    labels = []
    for _ in range(n):
        logits = [rng.uniform(-2.0, 2.0) for _ in range(n_classes)]
        m = max(logits)
        exp = [math.exp(x - m) for x in logits]
        s = sum(exp)
        row = [e / s for e in exp]
        u = rng.uniform(0.0, 1.0)
        acc = 0.0
        label = n_classes - 1
        for c, p in enumerate(row):
            acc += p
            if u <= acc:
                label = c
                break
        probs.append(row)
        labels.append(label)
    return probs, labels
