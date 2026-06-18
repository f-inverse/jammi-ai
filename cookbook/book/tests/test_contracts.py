"""Unit tests for the artifact-contract registry (K0 §1)."""

from __future__ import annotations

import json

import pytest

from jammi_cookbook import contracts
from jammi_cookbook.contracts import Golden


def test_registry_is_self_consistent():
    for name, art in contracts.ARTIFACTS.items():
        assert art.name == name
        assert art.dataset == name.split(".", 1)[0]
        assert art.kind in {
            "parquet", "edge_table", "model_id", "split",
            # scale-tier binary sidecar + plain-text artifacts (H2 cross-check)
            "usearch_bundle", "rowmap", "id_list",
        }
        assert art.filename
        # A graph artifact's declared-edge source names itself.
        if art.declared_edges is not None:
            assert art.declared_edges.edge_source == name


def test_artifact_lookup_unknown_raises():
    with pytest.raises(KeyError):
        contracts.artifact("arxiv.not_a_real_artifact")


def test_both_datasets_present():
    datasets = {art.dataset for art in contracts.ARTIFACTS.values()}
    assert {"arxiv", "air"} <= datasets


def test_golden_tolerance_band():
    g = Golden(value=0.83, tol=0.03)
    assert g.contains(0.83)
    assert g.contains(0.85)
    assert g.contains(0.80)
    assert not g.contains(0.87)
    assert not g.contains(0.79)


def test_golden_missing_file_raises_with_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(contracts, "_ARTIFACT_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="golden metrics not found"):
        contracts.golden("arxiv.tier04.marginal_coverage")


def test_assert_close_reads_committed_metric(monkeypatch, tmp_path):
    ds = tmp_path / "arxiv"
    ds.mkdir()
    (ds / "golden_metrics.json").write_text(
        json.dumps({"tier04.marginal_coverage": {"value": 0.83, "tol": 0.03}})
    )
    monkeypatch.setattr(contracts, "_ARTIFACT_ROOT", tmp_path)

    # within tolerance returns the observed value
    assert contracts.assert_close("arxiv.tier04.marginal_coverage", 0.85) == 0.85
    # outside tolerance fails loudly, naming the gap
    with pytest.raises(AssertionError, match="outside golden"):
        contracts.assert_close("arxiv.tier04.marginal_coverage", 0.95)


def test_load_artifact_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(contracts, "_ARTIFACT_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="not found"):
        contracts.load_artifact("arxiv.embeddings")
