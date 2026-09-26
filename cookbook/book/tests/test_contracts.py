"""Unit tests for the frozen goldens: lookup, the check, and freezing."""

from __future__ import annotations

import json

import pytest

from jammi_cookbook import contracts
from jammi_cookbook.contracts import Golden
from jammi_cookbook.scale import Scale


@pytest.fixture
def goldens(monkeypatch, tmp_path):
    """An empty goldens directory, at `small` scale, with freezing off."""
    monkeypatch.setattr(contracts, "GOLDENS", tmp_path)
    monkeypatch.setenv("JAMMI_COOKBOOK_SCALE", "small")
    monkeypatch.delenv(contracts.FREEZE_ENV, raising=False)
    return tmp_path


def _write(path, entries):
    path.write_text(json.dumps(entries))


def test_golden_tolerance_band():
    g = Golden(value=0.83, tol=0.03)
    assert g.contains(0.83)
    assert g.contains(0.85)
    assert g.contains(0.80)
    assert not g.contains(0.87)
    assert not g.contains(0.79)


def test_a_per_scale_golden_is_read_at_its_scale(goldens):
    _write(goldens / "arxiv.small.json", {"tier02.recall_at_10": {"value": 0.5, "tol": 0.01}})
    _write(goldens / "arxiv.full.json", {"tier02.recall_at_10": {"value": 0.7, "tol": 0.01}})
    assert contracts.golden("arxiv.tier02.recall_at_10").value == 0.5
    assert contracts.golden("arxiv.tier02.recall_at_10", Scale.FULL).value == 0.7


def test_a_scale_free_golden_serves_every_scale(goldens):
    _write(goldens / "air.json", {"routes.count": {"value": 42.0, "tol": 0.0}})
    assert contracts.golden("air.routes.count", Scale.SMALL).value == 42.0
    assert contracts.golden("air.routes.count", Scale.FULL).value == 42.0


def test_a_dataset_both_scale_free_and_per_scale_is_refused(goldens):
    _write(goldens / "air.json", {})
    _write(goldens / "air.small.json", {})
    with pytest.raises(ValueError, match="one kind or the other"):
        contracts.golden("air.routes.count")


def test_an_unfrozen_metric_names_the_gap(goldens):
    with pytest.raises(KeyError, match="nobody"):
        contracts.golden("arxiv.tier04.marginal_coverage")


def test_a_metric_without_a_dataset_is_refused(goldens):
    with pytest.raises(ValueError, match="<dataset>.<key>"):
        contracts.golden("coverage")


def test_assert_close_returns_the_observation_or_names_the_gap(goldens):
    _write(goldens / "arxiv.small.json",
           {"tier04.marginal_coverage": {"value": 0.83, "tol": 0.03}})
    assert contracts.assert_close("arxiv.tier04.marginal_coverage", 0.85) == 0.85
    with pytest.raises(AssertionError, match="off by"):
        contracts.assert_close("arxiv.tier04.marginal_coverage", 0.95)


def test_freezing_records_the_observation_and_keeps_a_frozen_tolerance(goldens, monkeypatch):
    path = goldens / "arxiv.small.json"
    _write(path, {"kept": {"value": 1.0, "tol": 0.2}})
    monkeypatch.setenv(contracts.FREEZE_ENV, "1")
    assert contracts.assert_close("arxiv.kept", 3.0, tol=0.9) == 3.0
    assert contracts.assert_close("arxiv.fresh", 0.5, tol=0.05) == 0.5
    assert json.loads(path.read_text()) == {
        "fresh": {"value": 0.5, "tol": 0.05},
        "kept": {"value": 3.0, "tol": 0.2},
    }
