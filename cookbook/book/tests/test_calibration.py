"""Cache-backed checks for the calibration & uncertainty vertical (09-calibration).

These run on CPU against the committed tier-04 gaussian predictions (no GPU, no
recompute). They assert the calibration goldens — the proper scores (CRPS / NLL),
the adaptive ECE, sharpness, central coverage, and the PIT KS statistic — and the
chapter's load-bearing finding: the predictor is **sharp but miscalibrated** under
the time-split (a non-uniform PIT), the same non-exchangeability the conformal tier
reports as under-coverage.

`eval_calibration(shape="gaussian")` is the engine surface; the chapter cross-checks
every proper score against an independent numpy closed-form fold, so a drifted engine
number fails loudly rather than silently.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_ARXIV = contracts._dataset_dir("arxiv")
_HAVE_CACHE = (_ARXIV / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="keystone cache not emitted")


@_needs_cache
def test_proper_scores_are_present_and_finite():
    """CRPS and NLL are real, finite proper-score headlines."""
    crps = contracts.golden("arxiv.calibration.crps").value
    nll = contracts.golden("arxiv.calibration.nll").value
    assert crps > 0, "CRPS is a non-negative proper score"
    assert nll > 0, "gaussian NLL is positive here"


@_needs_cache
def test_predictor_is_sharp_but_miscalibrated():
    """The honest finding: a narrow spread (sharp) with a non-uniform PIT (miscalibrated).

    Sharpness subject to calibration is the principle; this predictor fails the second
    half. The adaptive ECE is well above zero and the PIT KS statistic is far from the
    ~0 a calibrated forecast would show — the same non-exchangeability the conformal
    tier reports as under-coverage.
    """
    sharpness = contracts.golden("arxiv.calibration.sharpness").value
    ece = contracts.golden("arxiv.calibration.adaptive_ece").value
    pit_ks = contracts.golden("arxiv.calibration.pit_ks").value
    assert sharpness > 0, "the predictor has a real (non-degenerate) spread"
    assert ece > 0.05, "the adaptive ECE registers the miscalibration"
    assert pit_ks > 0.2, "the PIT is far from uniform — the predictor is miscalibrated"


@_needs_cache
def test_central_coverage_is_recorded():
    """The central-interval coverage is a real fraction in (0, 1)."""
    cov = contracts.golden("arxiv.calibration.central_coverage").value
    assert 0.0 < cov < 1.0


@_needs_cache
def test_calibration_report_matches_contract():
    """The committed calibration record loads and carries the cross-checked scores."""
    record = contracts.load_artifact("arxiv.calibration_report")
    assert record["shape"] == "gaussian"
    # the record's scores agree with the frozen goldens (within their tolerances)
    assert contracts.golden("arxiv.calibration.crps").contains(record["crps"])
    assert contracts.golden("arxiv.calibration.nll").contains(record["nll"])
    assert contracts.golden("arxiv.calibration.pit_ks").contains(record["pit_ks"])
    assert record["adaptive_ece"] > 0.05, "the record registers the miscalibration"
