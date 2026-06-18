"""Cache-backed checks on the committed KV-arxiv keystone artifacts.

These run on CPU against the committed cache (no GPU, no recompute) and assert the
keystone's load-bearing invariants: the artifact files exist with the contracted
shape, the calibration/test split is disjoint, and the tier-04 honest properties.

The bidirectional A3 win: `train_context_predictor(gaussian, value_column="year")`
now **fits** (real mean ≈ 2018, real spread — not the #43 collapse), so the
previously-impossible regression-conformal workflow runs end-to-end. The honest
conformal lesson, under the dataset's time-split, holds in BOTH cruxes:

* **(A) regression** — the year-regression interval **under-covers** (≈ 0.83). The
  shift is a *location* shift: the predictor regresses to the embedding-conditioned
  mean for both eras, so cal and test |y−ŷ| residual magnitudes are ≈ equal and
  corr(|residual|, test-likeness) ≈ 0 — importance weighting is a no-op (Δ ≤ ~0.01).
* **(B) classification** — the subject-classification marginal APS **under-covers**
  (≈ 0.867); the shift is ≈ orthogonal to the APS score (corr ≈ −0.12), so none of
  the three weighting schemes repairs it — they move coverage a little (−0.001 /
  +0.022 / +0.006) and all stay below nominal.

Both are negative results plus a diagnostic, NOT a restore. If the emitted cache is
absent the heavy artifacts are skipped (the unit suite stays runnable without the
keystone run), but the golden metrics themselves, once committed, are always
asserted.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_ARXIV = contracts._dataset_dir("arxiv")
_HAVE_CACHE = (_ARXIV / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="keystone cache not emitted")


@_needs_cache
def test_regression_predictor_fits_and_workflow_runs():
    """Part A — the bidirectional A3 win: the gaussian year predictor FITS.

    The headline is that `train_context_predictor(gaussian, value_column="year")`
    now produces a real mean (≈ 2018, the dataset era) with real spread — not the
    #43 collapse (std ≈ 0.001, mean ~2163). The previously-impossible regression-
    conformal workflow runs end-to-end; 0.26.2 completed the context-predictor
    standardization this keystone surfaced.
    """
    cal_mean = contracts.golden("arxiv.tier04.reg_cal_mean").value
    test_mean = contracts.golden("arxiv.tier04.reg_test_mean").value
    assert 2015 < cal_mean < 2021, "predictor cal-mean must land in the dataset era (~2018)"
    assert 2015 < test_mean < 2021, "predictor test-mean must land in the dataset era (~2018)"
    assert contracts.golden("arxiv.tier04.reg_pred_std").value > 0.05, "predictor collapsed"
    assert contracts.golden("arxiv.tier04.reg_interval_width").value > 0


@_needs_cache
def test_regression_conformal_under_covers_and_weighting_is_a_noop():
    """Part A — the honest finding: the year-regression interval UNDER-covers.

    Under the time-split the |y−ŷ| split-conformal interval falls below nominal.
    The shift is a *location* shift, not a residual-magnitude shift: the predictor
    regresses to the embedding-conditioned mean for both eras, so cal and test
    residual magnitudes are ≈ equal and corr(|residual|, test-likeness) ≈ 0 —
    importance weighting is a no-op (Δ ≤ ~0.01). A negative result, NOT a restore.
    """
    nominal = contracts.golden("arxiv.tier04.nominal_coverage").value
    reg_cov = contracts.golden("arxiv.tier04.reg_interval_coverage").value
    assert reg_cov < nominal, "the regression interval must UNDER-cover under the time-split"

    # location-shift evidence: cal and test residual magnitudes match.
    cal_mag = contracts.golden("arxiv.tier04.reg_cal_resid_mag").value
    test_mag = contracts.golden("arxiv.tier04.reg_test_resid_mag").value
    assert abs(cal_mag - test_mag) < 0.25, "residual magnitudes must match (location shift)"

    delta = contracts.golden("arxiv.tier04.reg_weighting_delta")
    assert abs(delta.value) <= delta.tol, "regression weighting must be a no-op here"

    corr = contracts.golden("arxiv.tier04.reg_resid_corr")
    assert abs(corr.value) < 0.2, "|residual| must be ~uncorrelated with test-likeness"


@_needs_cache
def test_classification_marginal_under_covers_and_weighting_is_a_noop():
    """Part B — the honest lesson: marginal APS under-covers; weighting does not repair.

    Marginal coverage falls below nominal (non-exchangeability is real). The
    weighted-conformal coverage changes across the three test-likeness schemes are
    small (the largest is the recorded ``weighting_max_abs_delta``), and crucially
    NO scheme reaches nominal: weighting does not repair the under-coverage. The
    corr(nonconformity, test-likeness) diagnostic is small — the shift is ~orthogonal
    to the conformal score, so reweighting cannot move the quantile to nominal. This
    encodes the negative result, NOT a restore.
    """
    nominal = contracts.golden("arxiv.tier04.nominal_coverage").value
    marginal = contracts.golden("arxiv.tier04.marginal_coverage").value
    assert marginal < nominal, "marginal conformal must UNDER-cover on the graph"

    # The weighting is a no-op in the load-bearing sense: across all three schemes the
    # weighted coverage still UNDER-covers — none reaches nominal — and the movements
    # are small and not even consistently toward nominal (one scheme moves away).
    weighting = contracts.load_artifact("arxiv.tier04_weighting")
    coverages = [s["coverage"] for s in weighting["schemes"].values()]
    deltas = [s["delta_vs_marginal"] for s in weighting["schemes"].values()]
    assert all(c < nominal for c in coverages), "no weighted scheme may reach nominal"
    assert max(abs(d) for d in deltas) < 0.03, "weighting movements must stay small"
    assert min(deltas) < 0, "movements are not systematically toward nominal (≈orthogonal)"

    corr = contracts.golden("arxiv.tier04.score_shift_corr")
    assert abs(corr.value) < 0.25, "the shift must be ~orthogonal to the conformal score"


@_needs_cache
def test_golden_set_sizes_present():
    """Sharpness (mean prediction-set size) is reported for the marginal pass."""
    assert contracts.golden("arxiv.tier04.marginal_set_size").value > 0
    assert 0.0 < contracts.golden("arxiv.tier04.classifier_accuracy").value < 1.0


@_needs_cache
def test_tier_recall_gains_are_real():
    """Propagation and declared-edge fine-tune each move recall in the right way."""
    base = contracts.golden("arxiv.tier01.recall_at_10").value
    assert 0.0 <= base <= 1.0
    # the recorded deltas are internally consistent with the recorded recalls
    prop = contracts.golden("arxiv.tier02.recall_at_10").value
    assert abs((prop - base) - contracts.golden("arxiv.tier02.recall_delta").value) < 1e-6
    ft = contracts.golden("arxiv.tier03.recall_at_10").value
    assert abs((ft - base) - contracts.golden("arxiv.tier03.recall_gain_vs_base").value) < 1e-6


@_needs_cache
def test_cite_graph_is_homophilous():
    assert contracts.golden("arxiv.tier01.cite_homophily").value > 0.4


@_needs_cache
def test_cal_split_is_disjoint():
    split = contracts.load_artifact("arxiv.cal_split")
    cal, test, train = set(split["calibration"]), set(split["test"]), set(split["train"])
    assert cal and test and train
    assert cal.isdisjoint(test)
    assert cal.isdisjoint(train)
    assert test.isdisjoint(train)


@_needs_cache
def test_committed_artifacts_match_contract():
    """Every parquet/edge_table artifact loads and carries its contracted columns."""
    for name in ("arxiv.papers", "arxiv.embeddings", "arxiv.neighbor_graph",
                 "arxiv.cite_edges", "arxiv.propagated", "arxiv.tier04_predictions",
                 "arxiv.tier04_regression"):
        art = contracts.artifact(name)
        table = contracts.load_artifact(name)
        assert table.num_rows > 0
        for col in art.columns:
            assert col in table.column_names, f"{name} missing column {col}"
