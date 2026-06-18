"""Cache-backed checks for the conformal expansion vertical (08-conformal).

These run on CPU against the committed tier-04 cache (no GPU, no recompute). They
assert the per-score conformal goldens and the chapter's keystone-complement
thesis: that on a transparently-synthetic, *score-aligned* shift, weighted
split-conformal GENUINELY restores coverage — the inverse of the keystone's
orthogonal-shift no-op.

The score families:

* **LAC / APS / RAPS** classification sets — every family UNDER-covers under the
  time-split (non-exchangeability is real); on this cache APS gives the SHARPER
  (smaller) sets and LAC the higher realised coverage with larger sets; RAPS reduces
  to APS (the rank-penalty does not bite at this class count, an honest measured
  equality, not a tuned gap).
* **abs-residual / CQR** regression intervals — both under-cover; CQR is wider and
  recovers a little more coverage by inheriting the predictor's spread.

The score-aligned restore: marginal UNDER-covers (≈ 0.83) and weighted RESTORES to
≥ nominal (≈ 0.94), with corr(nonconformity, shift-feature) ≈ +0.73 — the high,
positive correlation that explains why weighting CAN move the quantile here, exactly
the complement of the keystone's ≈ −0.12.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_ARXIV = contracts._dataset_dir("arxiv")
_HAVE_CACHE = (_ARXIV / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="keystone cache not emitted")


@_needs_cache
def test_classification_scores_under_cover_and_aps_is_sharper():
    """LAC / APS / RAPS all under-cover; APS gives the sharper sets on this cache.

    The textbook ordering (LAC = smallest sets at *exact* nominal calibration) does
    NOT hold under this time-split: LAC's threshold is the more conservative one, so
    it reaches higher realised coverage (0.889) at the cost of larger sets (≈ 8.0),
    while APS is the sharper family (≈ 6.2) but under-covers more. We assert the
    measured ordering, not the idealized one.
    """
    nominal = contracts.golden("arxiv.conformal.nominal_coverage").value
    lac = contracts.golden("arxiv.conformal.lac_coverage").value
    aps = contracts.golden("arxiv.conformal.aps_coverage").value
    raps = contracts.golden("arxiv.conformal.raps_coverage").value
    assert lac < nominal and aps < nominal and raps < nominal, "all families under-cover"

    lac_size = contracts.golden("arxiv.conformal.lac_set_size").value
    aps_size = contracts.golden("arxiv.conformal.aps_set_size").value
    assert aps_size < lac_size, "APS gives the sharper (smaller) sets on this cache"
    assert lac > aps, "LAC reaches higher realised coverage (more conservative threshold)"


@_needs_cache
def test_raps_reduces_to_aps_on_this_cache():
    """The honest measured equality: RAPS == APS here (the rank-penalty does not bite)."""
    assert (
        contracts.golden("arxiv.conformal.raps_coverage").value
        == contracts.golden("arxiv.conformal.aps_coverage").value
    )
    assert (
        contracts.golden("arxiv.conformal.raps_set_size").value
        == contracts.golden("arxiv.conformal.aps_set_size").value
    )


@_needs_cache
def test_regression_intervals_under_cover_and_cqr_is_wider():
    """Abs-residual and CQR both under-cover; CQR is wider and recovers more coverage."""
    nominal = contracts.golden("arxiv.conformal.nominal_coverage").value
    iv_cov = contracts.golden("arxiv.conformal.interval_coverage").value
    iv_w = contracts.golden("arxiv.conformal.interval_width").value
    cqr_cov = contracts.golden("arxiv.conformal.cqr_coverage").value
    cqr_w = contracts.golden("arxiv.conformal.cqr_width").value
    assert iv_cov < nominal and cqr_cov < nominal, "both intervals under-cover"
    assert cqr_w > iv_w, "CQR inherits the predictor spread → wider band"
    assert cqr_cov > iv_cov, "the wider CQR band recovers a little more coverage"


@_needs_cache
def test_score_aligned_shift_genuinely_restores_coverage():
    """The keystone's complement: on a SCORE-ALIGNED shift, weighting RESTORES coverage.

    This is the inverse of the keystone's no-op test. The shift is a transparently-
    synthetic teaching device, NOT the real time-split: a calibration subsample biased
    along a real embedding covariate that correlates with the APS nonconformity. The
    correlation is high and POSITIVE (≈ +0.73, vs the keystone's ≈ −0.12), so weighted
    split-conformal moves the quantile and restores coverage.
    """
    nominal = contracts.golden("arxiv.conformal.nominal_coverage").value
    marginal = contracts.golden("arxiv.conformal.synthetic_marginal_coverage").value
    weighted = contracts.golden("arxiv.conformal.synthetic_weighted_coverage").value
    corr = contracts.golden("arxiv.conformal.synthetic_shift_corr").value

    assert marginal < nominal, "marginal must under-cover on the constructed shift"
    assert weighted >= nominal, "weighting must RESTORE coverage when the shift is score-aligned"
    assert weighted - marginal > 0.05, "the restore must be a real, sizeable move (not a no-op)"
    assert corr > 0.4, "the constructed shift must be score-aligned (high positive corr)"

    # The keystone's complement: the real time-split shift is ~orthogonal to the score
    # (this vertical's restore is the explicit inverse of that no-op).
    keystone_corr = contracts.golden("arxiv.tier04.score_shift_corr").value
    assert abs(keystone_corr) < 0.25 < corr, (
        "the keystone shift is orthogonal (no-op); this constructed shift is aligned (restore)"
    )


@_needs_cache
def test_synthetic_shift_record_matches_contract():
    """The committed synthetic-shift record loads and carries the labelled construction."""
    record = contracts.load_artifact("arxiv.conformal_synthetic_shift")
    assert record["marginal_coverage"] < record["nominal_coverage"]
    assert record["weighted_coverage"] >= record["nominal_coverage"]
    assert record["corr_nonconformity_shift_feature"] > 0.4
    assert "TRANSPARENTLY-SYNTHETIC" in record["note"], "the device must be labelled as constructed"
