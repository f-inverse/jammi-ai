"""Unit tests for the determinism contract."""

from __future__ import annotations

import os

from jammi_cookbook import determinism


def test_env_pinned_on_import():
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_the_small_scale_runs_on_the_cpu():
    assert os.environ["JAMMI_GPU__DEVICE"] == "-1"


def test_seeded_is_pure_and_stable():
    # Deterministic across calls and independent of PYTHONHASHSEED.
    assert determinism.seeded("tier04.predictor") == determinism.seeded("tier04.predictor")
    # Distinct call sites get distinct seeds.
    assert determinism.seeded("tier04.predictor") != determinism.seeded("tier01.subset")
    # Non-negative (usable as a numpy seed).
    assert determinism.seeded("anything") >= 0
