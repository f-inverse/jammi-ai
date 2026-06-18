"""Cache-backed checks for the eval vertical (chapter 14) — CPU, no remote re-diff.

These run on CPU against the committed **embedded-canonical** eval reports (no
server, no GPU, no recompute) and assert the vertical's load-bearing facts:

* **aggregates-to-golden (oracle a):** every committed report's aggregate scalar
  matches its frozen golden in ``artifacts/eval/golden_metrics.json`` — the
  golden-stability gate;
* **committed-report shape:** each report carries the structural keys the chapter
  reads (the same keys the engine's live test pins by ``_shape``), with the
  instance-minted keys (``eval_run_id`` / ``table_name``) stripped from the
  committed form;
* **the eval_compare contrast (RC3):** the self-comparison is the determinism
  anchor (every metric delta exactly 0.0, significance present) and the two-table
  comparison is genuinely non-degenerate (a non-zero recall delta with a real
  significance block).

The cross-transport remote == embedded parity is a ONE-TIME emit-side LIVE check
(recorded in ``eval.json``, continuously re-guarded by the engine's gated
``test_remote_eval_live.py``); PR CI never re-diffs two static artifacts here.

If the emitted cache is absent the report-backed checks skip, but the committed
golden metrics, once present, are always asserted.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_EVAL = contracts._dataset_dir("eval")
_HAVE_CACHE = (_EVAL / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="eval cache not emitted")

# The instance-minted keys the emit strips from the committed reports (their
# presence is pinned by the engine's live _shape check, not the committed form).
_INSTANCE_KEYS = {"eval_run_id", "table_name"}


def _report(name: str):
    return contracts.load_artifact(f"eval.{name}")


# --------------------------------------------------------------------------- #
# oracle (a): aggregates-to-golden
# --------------------------------------------------------------------------- #


@_needs_cache
def test_embeddings_aggregate_matches_golden():
    """The committed embeddings aggregate matches its frozen golden, every metric."""
    agg = _report("embeddings")["aggregate"]
    for metric in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
        contracts.assert_close(f"eval.embeddings.{metric}", float(agg[metric]))


@_needs_cache
def test_classification_aggregate_matches_golden():
    """The committed classification aggregate (accuracy/f1) matches its golden."""
    agg = _report("inference_cls")["aggregate"]
    assert agg["task"] == "classification"
    contracts.assert_close("eval.inference_cls.accuracy", float(agg["accuracy"]))
    contracts.assert_close("eval.inference_cls.f1", float(agg["f1"]))


@_needs_cache
def test_ner_aggregate_matches_golden():
    """The committed NER aggregate (precision/recall/f1) matches its golden — the
    Python + remote NER eval coverage this chapter adds (the engine's own Python
    live tests carry only classification)."""
    agg = _report("inference_ner")["aggregate"]
    assert agg["task"] == "ner"
    for metric in ("precision", "recall", "f1"):
        contracts.assert_close(f"eval.inference_ner.{metric}", float(agg[metric]))
        assert 0.0 <= agg[metric] <= 1.0, f"NER {metric} out of [0,1]"


# --------------------------------------------------------------------------- #
# committed-report shape
# --------------------------------------------------------------------------- #


@_needs_cache
def test_committed_reports_carry_expected_shape():
    """Each committed report carries the structural keys the chapter reads, with
    the instance-minted keys stripped from the committed form."""
    embeddings = _report("embeddings")
    assert set(embeddings) == {"aggregate", "per_query"}
    assert _INSTANCE_KEYS.isdisjoint(embeddings)
    assert embeddings["per_query"], "per_query must carry one record per golden query"
    for rec in embeddings["per_query"]:
        assert {"query_id", "metrics", "recall_at_ks", "distance", "cohorts"} <= set(rec)

    per_query = _report("per_query")
    assert per_query, "the persisted per-query rows must be read back"
    for rec in per_query:
        assert {"query_id", "cohorts", "metrics"} <= set(rec)
        assert "eval_run_id" not in rec  # stripped from the committed form

    for task, agg_keys in (
        ("inference_cls", {"task", "accuracy", "f1", "per_class"}),
        ("inference_ner", {"task", "precision", "recall", "f1", "per_type"}),
    ):
        report = _report(task)
        assert set(report) == {"aggregate", "per_record"}
        assert agg_keys <= set(report["aggregate"])
        assert report["per_record"], f"{task} must carry per-record predictions"


@_needs_cache
def test_per_query_sorted_by_id_byte_stable():
    """The committed per-query / embeddings per_query rows are sorted by id — the
    byte-stable normalisation the emit applies (engine output has no ORDER BY)."""
    emb_ids = [r["query_id"] for r in _report("embeddings")["per_query"]]
    assert emb_ids == sorted(emb_ids), "embeddings per_query must be sorted by query_id"
    pq_ids = [r["query_id"] for r in _report("per_query")]
    assert pq_ids == sorted(pq_ids), "per_query rows must be sorted by query_id"
    cls_ids = [r["record_id"] for r in _report("inference_cls")["per_record"]]
    assert cls_ids == sorted(cls_ids), "classification per_record must be sorted by record_id"
    ner_ids = [r["record_id"] for r in _report("inference_ner")["per_record"]]
    assert ner_ids == sorted(ner_ids), "NER per_record must be sorted by record_id"


# --------------------------------------------------------------------------- #
# the eval_compare contrast (RC3)
# --------------------------------------------------------------------------- #


@_needs_cache
def test_compare_self_is_determinism_anchor():
    """The self-comparison is the determinism anchor: the baseline carries
    delta None, the treatment's every metric delta is exactly 0.0, and the
    significance block is present (a CI collapsed onto zero)."""
    per_table = _report("compare_self")["per_table"]
    assert len(per_table) == 2
    baseline, treatment = per_table
    assert baseline["delta"] is None
    delta = treatment["delta"]
    assert delta["significance"] is not None, "self-comparison significance must be present"
    for metric in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
        assert delta[metric]["absolute"] == 0.0, metric
    contracts.assert_close("eval.compare.self_max_abs_delta", 0.0)
    contracts.assert_close("eval.compare.self_significance_present", 1.0)


@_needs_cache
def test_compare_two_table_is_non_degenerate():
    """The two-table comparison is genuinely non-degenerate: the baseline carries
    delta None, the treatment carries a NON-ZERO recall delta and a present
    significance block — a real cross-model comparison, not a self-anchor."""
    per_table = _report("compare_two")["per_table"]
    assert len(per_table) == 2
    baseline, treatment = per_table
    assert baseline["delta"] is None
    delta = treatment["delta"]
    assert delta["significance"] is not None, "two-table significance must be present"
    observed = float(delta["recall_at_k"]["absolute"])
    contracts.assert_close("eval.compare.two_recall_delta_abs", observed)
    assert observed != 0.0, "the two-table recall delta must be genuinely non-zero"
    contracts.assert_close("eval.compare.two_significance_present", 1.0)
