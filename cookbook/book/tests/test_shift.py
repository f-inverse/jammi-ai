"""Hermetic unit tests for the covariate-shift helpers the conformal chapters
build on: the weight estimate, the nonconformity diagnostic, and the binomial
band."""

from __future__ import annotations

import numpy as np
import pytest

from jammi_cookbook import shift

ALPHA = 0.1


def test_the_density_ratio_is_the_smoothed_odds_of_a_test_era_neighbour():
    # Of 10 neighbours, 3 test-era: odds (3 + 1) / (7 + 1) = 0.5. None: 1/11. All: 11.
    ratio = shift.density_ratio(np.array([0.3, 0.0, 1.0]), 10)
    assert ratio == pytest.approx([0.5, 1 / 11, 11.0])


def test_aps_nonconformity_is_the_mass_down_to_the_true_class():
    scores = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    assert shift.aps_nonconformity(scores, np.array([0, 2])) == pytest.approx([0.5, 1.0])


def test_the_binomial_band_is_three_standard_errors_of_the_nominal_coverage():
    # 90% nominal over 100 test points: one standard error is sqrt(0.09 / 100) = 0.03.
    assert shift.binomial_band(0.10, 100) == pytest.approx(0.09)
    # The band narrows as the square root of the test set grows.
    assert shift.binomial_band(0.10, 400) == pytest.approx(shift.binomial_band(0.10, 100) / 2)
