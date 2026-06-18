"""Cache-backed checks on the committed retrieval / search vertical (B1).

These run on CPU against the committed cache (no GPU, no recompute) and assert the
vertical's load-bearing invariants: every retriever's recall@10 / nDCG@10 row exists
with the contracted shape and is a real number, the dense numbers reproduce the
keystone's per-table recall, and — the HONEST finding — that RRF fusion does NOT beat
the best single arm on this same-subject target. The search-multi-table engine finding
is recorded as a structured record, not prose.

If the emitted cache is absent the heavy artifacts are skipped, but the golden metrics,
once committed, are always asserted.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_RT = contracts._dataset_dir("retrieval")
_HAVE_CACHE = (_RT / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="retrieval cache not emitted")

_DENSE = ("dense_raw", "dense_propagated")
_LEXICAL = ("lexical_bm25",)
_FUSION = ("rrf_raw_prop", "rrf_raw_lex", "rrf_prop_lex")
_ALL = _DENSE + _LEXICAL + _FUSION


@_needs_cache
def test_every_method_has_real_recall_and_ndcg_in_range():
    """Each retriever's recall@10 and nDCG@10 are real numbers in [0, 1], matching golden."""
    rows = {r["method"]: r for r in
            contracts.load_artifact("retrieval.method_metrics").to_pylist()}
    assert set(rows) == set(_ALL), f"expected every retriever, got {sorted(rows)}"
    for method, r in rows.items():
        assert 0.0 <= r["recall_at_10"] <= 1.0
        assert 0.0 <= r["ndcg_at_10"] <= 1.0
        assert contracts.golden(f"retrieval.{method}.recall_at_10").contains(r["recall_at_10"])
        assert contracts.golden(f"retrieval.{method}.ndcg_at_10").contains(r["ndcg_at_10"])


@_needs_cache
def test_dense_reproduces_keystone_per_table_recall():
    """The dense arms reproduce the keystone's frozen per-table recall (same fold)."""
    rows = {r["method"]: r for r in
            contracts.load_artifact("retrieval.method_metrics").to_pylist()}
    # raw dense == keystone tier01 recall; propagated dense == keystone tier02 recall.
    assert contracts.golden("arxiv.tier01.recall_at_10").contains(rows["dense_raw"]["recall_at_10"])
    assert contracts.golden("arxiv.tier02.recall_at_10").contains(
        rows["dense_propagated"]["recall_at_10"])


@_needs_cache
def test_lexical_is_weaker_than_dense_on_same_subject():
    """BM25 over titles is markedly weaker than dense on the same-subject target.

    Subject membership is a semantic relation that title-word overlap only partially
    captures — the precondition that makes fusion liable to hurt (a weak arm dragging
    the strong arm down).
    """
    rows = {r["method"]: r for r in
            contracts.load_artifact("retrieval.method_metrics").to_pylist()}
    best_dense = max(rows[m]["recall_at_10"] for m in _DENSE)
    assert rows["lexical_bm25"]["recall_at_10"] < best_dense, "lexical must be weaker than dense"


@_needs_cache
def test_honest_fusion_finding_fusion_does_not_help():
    """The honest finding: RRF fusion does NOT beat the best single arm here.

    Fusing in a weaker arm cannot exceed the stronger ranker it already contains. The
    record asserts fusion_helps is False and that the best fusion arm sits below the
    best single arm — reported as the data shows, never spun.
    """
    finding = contracts.load_artifact("retrieval.finding")
    assert finding["fusion_helps"] is False, "fusion must be reported as not helping"
    assert finding["best_fusion_recall_at_10"] <= finding["best_single_recall_at_10"]
    # RRF of the two dense arms beats the weaker arm but not the stronger one.
    assert finding["fusion_vs_raw"] > 0.0, "RRF(raw+prop) beats the weaker (raw) arm"
    assert finding["fusion_vs_best_dense"] < 0.0, "RRF(raw+prop) does not beat the best dense arm"
    # the recorded deltas agree with the frozen goldens.
    assert contracts.golden("retrieval.fusion_vs_best_dense").contains(
        finding["fusion_vs_best_dense"])
    assert contracts.golden("retrieval.fusion_vs_raw").contains(finding["fusion_vs_raw"])
    assert contracts.golden("retrieval.hybrid_vs_dense").contains(finding["hybrid_vs_dense"])


@_needs_cache
def test_search_multi_table_engine_finding_is_recorded():
    """The search-multi-table ambiguity is recorded as a structured engine finding.

    search(source, ...) carries no table= argument and resolves a source's single
    ready embedding table, so it is ambiguous once a source has several embedding
    tables. Recorded as a candidate for an explicit table= argument — not papered over.
    """
    sf = contracts.load_artifact("retrieval.finding")["search_finding"]
    assert sf["verb"] == "search"
    assert sf["reachable_on_cpu"] is False
    assert "table=" in sf["candidate"]
    assert "no table= argument" in sf["reason"] or "table=" in sf["reason"]


@_needs_cache
def test_committed_retrieval_artifacts_match_contract():
    art = contracts.artifact("retrieval.method_metrics")
    table = contracts.load_artifact("retrieval.method_metrics")
    assert table.num_rows == len(_ALL)
    for col in art.columns:
        assert col in table.column_names, f"missing column {col}"
