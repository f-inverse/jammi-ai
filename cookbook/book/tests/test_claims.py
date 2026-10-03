"""Unit tests for the claims a chapter makes: shown when they hold, raised by
name when they do not."""

from __future__ import annotations

import numpy as np
import pytest

from jammi_cookbook.claims import ClaimFailed, claim


def test_a_holding_claim_prints_with_its_evidence(capsys):
    claim("rescore recovers recall", True, "0.875 against 0.613")
    assert capsys.readouterr().out == "✓ rescore recovers recall (0.875 against 0.613)\n"


def test_a_claim_without_evidence_prints_alone(capsys):
    claim("each publish takes the next offset", True)
    assert capsys.readouterr().out == "✓ each publish takes the next offset\n"


def test_a_failing_claim_raises_naming_it_and_prints_nothing(capsys):
    with pytest.raises(ClaimFailed, match="rescore recovers recall.*0.5 against 0.6"):
        claim("rescore recovers recall", False, "0.5 against 0.6")
    assert capsys.readouterr().out == ""


def test_a_failed_claim_is_an_assertion_error():
    with pytest.raises(AssertionError):
        claim("anything", False)


def test_a_numpy_truth_value_is_a_truth_value(capsys):
    claim("numpy comparisons work", np.float64(0.9) > np.float64(0.8))
    with pytest.raises(ClaimFailed):
        claim("and fail", np.float64(0.7) > np.float64(0.8))
