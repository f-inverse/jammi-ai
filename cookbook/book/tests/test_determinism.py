"""Unit tests for the determinism contract (K0 §3)."""

from __future__ import annotations

import os

import pytest

from jammi_cookbook import determinism


def test_env_pinned_on_import():
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_seeded_is_pure_and_stable():
    # Deterministic across calls and independent of PYTHONHASHSEED.
    assert determinism.seeded("tier04.predictor") == determinism.seeded("tier04.predictor")
    # Distinct call sites get distinct seeds.
    assert determinism.seeded("tier04.predictor") != determinism.seeded("tier01.subset")
    # Non-negative (usable as a numpy seed).
    assert determinism.seeded("anything") >= 0


def test_committed_ids_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(determinism, "_IDS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="committed id list not found"):
        determinism.committed_ids("arxiv")


def test_committed_ids_reads_and_strips(monkeypatch, tmp_path):
    monkeypatch.setattr(determinism, "_IDS_DIR", tmp_path)
    (tmp_path / "arxiv.txt").write_text("a\nb\n\n  c  \n")
    assert determinism.committed_ids("arxiv") == ["a", "b", "c"]
