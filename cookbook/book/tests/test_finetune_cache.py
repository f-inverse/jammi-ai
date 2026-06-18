"""Cache-backed checks on the committed fine-tune-methods vertical (A1).

These run on CPU against the committed cache (no GPU, no recompute) and assert the
vertical's load-bearing invariants: every method's recall row exists with the
contracted shape, the recall numbers are real (in range, gain internally
consistent), the Matryoshka recall-vs-truncated-dim curve is monotone
non-increasing, and the HONEST finding — that the method choice moves recall only
within a narrow band on this same-subject supervision (the supervision caps the
gain, the tier-03 circularity contract generalized from the graph to the loss).

The numbers are asserted as the data shows them: no method is privileged, ties and
no-improvements are encoded as such. If the emitted cache is absent the heavy
artifacts are skipped, but the golden metrics, once committed, are always asserted.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_FT = contracts._dataset_dir("finetune")
_HAVE_CACHE = (_FT / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="finetune cache not emitted")

# The methods that complete and enter the apples-to-apples recall table (each a short
# LoRA run on the same subset + golden). Hard-negative mining is reported as a separate
# engine finding (it OOMs at the cookbook's corpus scale), not a recall-table row.
_METHODS = (
    "cosent", "mnrl_t0.05", "mnrl_t0.20", "triplet", "matryoshka", "graph_declared",
)


@_needs_cache
def test_every_method_has_a_real_recall_in_range():
    """Each method's recall@10 is a real number in [0, 1], gain consistent with base."""
    base = contracts.golden("finetune.base_recall_at_10").value
    assert 0.0 <= base <= 1.0
    rows = {r["method"]: r for r in contracts.load_artifact("finetune.method_recall").to_pylist()}
    assert set(rows) == set(_METHODS), f"expected all methods, got {sorted(rows)}"
    for method, r in rows.items():
        assert 0.0 <= r["recall_at_10"] <= 1.0
        # the gain column is internally consistent with the recorded base.
        assert abs((r["recall_at_10"] - base) - r["recall_gain_vs_base"]) < 1e-6
        # the recorded golden matches the committed row.
        g = contracts.golden(f"finetune.{method}.recall_at_10")
        assert g.contains(r["recall_at_10"])


@_needs_cache
def test_matryoshka_curve_is_monotone_and_truncation_retains_recall():
    """The Matryoshka curve must not RISE as the dimension shrinks; 64-d still retrieves.

    A prefix of the representation carries no more information than the whole, so
    recall@10 is monotone non-increasing in the truncation. The smallest committed
    prefix retains a substantial fraction of the full-dim recall — the Matryoshka
    win (a 12x smaller index that still works), demonstrated, not asserted.
    """
    curve = sorted(contracts.load_artifact("finetune.matryoshka_curve").to_pylist(),
                   key=lambda c: -c["dim"])
    recalls = [c["recall_at_10"] for c in curve]
    assert recalls == sorted(recalls, reverse=True), "recall must not rise as dim shrinks"
    full, smallest = recalls[0], recalls[-1]
    assert full > 0
    # the smallest prefix is not degenerate — it retains most of the full-dim recall.
    assert smallest >= 0.5 * full, "the truncated prefix must remain a usable embedding"


@_needs_cache
def test_honest_finding_method_spread_is_narrow():
    """The honest finding: the method choice moves recall only within a narrow band.

    The supervision caps the achievable gain (the tier-03 circularity contract, from
    the graph to the loss): on a same-subject signal the base geometry already
    separates, no contrastive loss / hard-negative / Matryoshka knob opens a large
    gap. The recorded spread across the WHOLE method spectrum is small — a method is
    "best" only by the literal number, not by a decisive margin.
    """
    spread = contracts.golden("finetune.method_recall_spread").value
    assert spread >= 0.0
    assert spread < 0.10, "the method spectrum must not show a large recall gap (narrow band)"

    methods = contracts.load_artifact("finetune.methods")
    assert methods["best_method"] in _METHODS
    assert methods["worst_method"] in _METHODS
    # the recorded spread agrees with the per-method recalls.
    recalls = [m["recall_at_10"] for m in methods["methods"]]
    assert abs((max(recalls) - min(recalls)) - methods["recall_spread"]) < 1e-6


@_needs_cache
def test_hard_negative_finding_is_recorded_honestly():
    """Hard-negative mining is recorded as a real result, never a fabricated recall.

    Either it completed (and carries a recall), or — the real 0.26.2 result on the
    A10G — it OOM'd at the full supervised scale, in which case the record carries the
    CUDA out-of-memory error and the smaller-corpus threshold that proves the kwarg
    works at scale (a memory-scale limit, not a broken signature). It must NOT appear
    in the apples-to-apples recall table either way.
    """
    methods = contracts.load_artifact("finetune.methods")
    hn = methods["hard_negatives"]
    assert hn["status"] in ("completed", "oom_at_full_scale")
    if hn["status"] == "oom_at_full_scale":
        assert "out of memory" in hn["error"].lower()
        assert hn["small_scale_pairs"] < hn["full_scale_pairs"]
    # never in the recall table — no fabricated recall for a run that did not happen.
    table_methods = {r["method"] for r in
                     contracts.load_artifact("finetune.method_recall").to_pylist()}
    assert "hard_neg" not in table_methods


@_needs_cache
def test_committed_artifacts_match_contract():
    for name in ("finetune.method_recall", "finetune.matryoshka_curve",
                 "finetune.emb_matryoshka"):
        art = contracts.artifact(name)
        table = contracts.load_artifact(name)
        assert table.num_rows > 0
        for col in art.columns:
            assert col in table.column_names, f"{name} missing column {col}"
