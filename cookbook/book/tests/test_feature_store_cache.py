"""Cache-backed checks for the mutable-companion-table / feature-store vertical (C1).

These run on CPU against the committed feature cache (no GPU, no recompute) and assert
the vertical's load-bearing facts: the committed per-paper feature (citation in-degree)
loads with its contracted shape, the subject-level SUM(in_degree) JOIN aggregate matches
its frozen golden and conserves the feature mass, and — the honesty constraint — the
mutable table is recorded as APPEND-ONLY (UPDATE / DELETE / duplicate-key INSERT each
rejected on this surface).

If the emitted cache is absent the heavy artifacts are skipped, but the committed golden
metrics, once present, are always asserted.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_FS = contracts._dataset_dir("feature_store")
_HAVE_CACHE = (_FS / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="feature_store cache not emitted")


@_needs_cache
def test_feature_rows_match_contract():
    """Every paper carries a citation-in-degree feature value (0 if uncited)."""
    art = contracts.artifact("feature_store.paper_features")
    table = contracts.load_artifact("feature_store.paper_features")
    assert table.num_rows > 0
    for col in art.columns:
        assert col in table.column_names, f"missing column {col}"
    in_degree = {r["paper_id"]: r["in_degree"] for r in table.to_pylist()}
    assert all(v >= 0 for v in in_degree.values()), "in-degree is a non-negative count"
    # the feature is dense — every committed paper has a value.
    papers = {p["paper_id"] for p in contracts.load_artifact("arxiv.papers").to_pylist()}
    assert set(in_degree) == papers, "the feature column covers every paper"


@_needs_cache
def test_join_aggregate_conserves_feature_mass():
    """The subject-level SUM(in_degree) aggregate sums to the committed grand total."""
    record = contracts.load_artifact("feature_store.record")
    subject_totals = record["subject_totals"]
    assert sum(subject_totals.values()) == record["join_total_in_degree"], (
        "the subject-level JOIN aggregate must conserve the feature mass")
    # the committed grand total equals the total feature mass on the feature rows.
    table = contracts.load_artifact("feature_store.paper_features")
    assert sum(r["in_degree"] for r in table.to_pylist()) == record["join_total_in_degree"]
    # the golden grand total + top-subject total are exact (zero tolerance).
    total = contracts.golden("feature_store.total_in_degree")
    top = contracts.golden("feature_store.top_subject_total")
    assert total.tol == 0.0 and total.value == float(record["join_total_in_degree"])
    assert top.tol == 0.0 and top.value == float(record["top_subject_total"])
    assert subject_totals[record["top_subject"]] == record["top_subject_total"]


@_needs_cache
def test_surface_is_append_only():
    """The honesty constraint: UPDATE / DELETE / duplicate-key INSERT are each rejected."""
    record = contracts.load_artifact("feature_store.record")
    probe = record["append_only"]
    assert probe["update_rejected"], "UPDATE must be rejected on this surface"
    assert probe["delete_rejected"], "DELETE must be rejected on this surface"
    assert probe["duplicate_insert_rejected"], "duplicate-key INSERT must hit the UNIQUE key"
    # the populated row count is exact and matches the feature row count.
    populated = contracts.golden("feature_store.populated_rows")
    assert populated.tol == 0.0
    assert populated.value == float(record["populated_rows"]) == float(record["rows"])
