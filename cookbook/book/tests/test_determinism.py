"""Unit tests for the book's per-step seeds."""

from __future__ import annotations

from jammi_cookbook import determinism


def test_seeded_is_pure_and_stable():
    # Deterministic across calls and independent of PYTHONHASHSEED.
    assert determinism.seeded("tier04.predictor") == determinism.seeded("tier04.predictor")
    # Distinct call sites get distinct seeds.
    assert determinism.seeded("tier04.predictor") != determinism.seeded("tier01.subset")
    # Non-negative (usable as a numpy seed).
    assert determinism.seeded("anything") >= 0
