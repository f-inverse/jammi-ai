"""Cache-backed checks on the closed eval loop and the provenance rail.

These run on CPU against the committed cache (no GPU, no upstream recompute) and
assert the two K-rails properties the chapters demonstrate:

* the **closed eval loop** recomputes the conformal verdicts *live* from the
  committed per-row outputs and they match the frozen goldens — the full golden
  chain holds under real execution, not just as transcribed constants;
* the **provenance rail** reconstructs a tier-04 prediction's exact informing rows
  (its declared-citation-edge context set) from the committed cache, and
  ``rails.provenance`` extracts the audit trail.

If the emitted cache is absent the heavy artifacts are skipped; the golden
metrics, once committed, are always asserted.
"""

from __future__ import annotations

import collections

import jammi_ai
import numpy as np
import pytest

from jammi_cookbook import contracts, rails

_ARXIV = contracts._dataset_dir("arxiv")
_HAVE_CACHE = (_ARXIV / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="keystone cache not emitted")

_ALPHA = (
    round(1 - contracts.golden("arxiv.tier04.nominal_coverage").value, 4)
    if _HAVE_CACHE else 0.1
)


@_needs_cache
def test_recall_chain_reads_in_order():
    """The construct→propagate→learn recall chain is the frozen 0.538 → 0.556 → 0.548."""
    base = contracts.golden("arxiv.tier01.recall_at_10").value
    prop = contracts.golden("arxiv.tier02.recall_at_10").value
    ft = contracts.golden("arxiv.tier03.recall_at_10").value
    assert abs(base - 0.538) <= contracts.golden("arxiv.tier01.recall_at_10").tol
    assert abs(prop - 0.556) <= contracts.golden("arxiv.tier02.recall_at_10").tol
    assert abs(ft - 0.548) <= contracts.golden("arxiv.tier03.recall_at_10").tol
    assert prop > base  # propagation denoises
    assert ft > base    # the declared-edge fine-tune beats the base (circularity contract)


@_needs_cache
def test_marginal_classification_conformal_recomputes_to_golden():
    """Live engine APS marginal coverage matches the frozen 0.867 — under-covers."""
    db = jammi_ai.connect("file:///tmp/jammi_test_cel_cls")
    preds = contracts.load_artifact("arxiv.tier04_predictions").to_pylist()
    cal = [r for r in preds if r["split"] == "calibration"]
    test = [r for r in preds if r["split"] == "test"]
    sets = db.conformalize([r["scores"] for r in cal], [r["true_label"] for r in cal],
                           [r["scores"] for r in test], alpha=_ALPHA, score="aps")
    cov = float(np.mean([test[i]["true_label"] in sets[i] for i in range(len(test))]))
    # the measurement rail asserts the recomputed number against the golden
    rails.measure("arxiv.tier04.marginal_coverage", cov)
    assert cov < 1 - _ALPHA  # under-coverage — the honest lesson


@_needs_cache
def test_regression_interval_conformal_recomputes_to_golden():
    """Live conformalize_interval coverage matches the frozen 0.830 — under-covers."""
    db = jammi_ai.connect("file:///tmp/jammi_test_cel_reg")
    reg = contracts.load_artifact("arxiv.tier04_regression").to_pylist()
    cal = [r for r in reg if r["split"] == "calibration"]
    test = [r for r in reg if r["split"] == "test"]
    iv = db.conformalize_interval([r["pred_mean"] for r in cal],
                                  [float(r["true_year"]) for r in cal],
                                  [r["pred_mean"] for r in test], alpha=_ALPHA)
    ty = [r["true_year"] for r in test]
    cov = float(np.mean([lo <= ty[i] <= hi for i, (lo, hi) in enumerate(iv)]))
    rails.measure("arxiv.tier04.reg_interval_coverage", cov)
    assert cov < 1 - _ALPHA


@_needs_cache
def test_weighting_restores_neither_crux():
    """Regression weighting is an exact no-op; no classification scheme reaches nominal."""
    reg_w = contracts.load_artifact("arxiv.tier04_regression_weighting")
    assert reg_w["weighting_delta"] == 0.0  # exact no-op (location shift)

    cls_w = contracts.load_artifact("arxiv.tier04_weighting")
    covs = {n: s["coverage"] for n, s in cls_w["schemes"].items()}
    deltas = {n: s["delta_vs_marginal"] for n, s in cls_w["schemes"].items()}
    assert all(c < 1 - _ALPHA for c in covs.values())  # none reaches nominal
    assert max(deltas, key=lambda n: abs(deltas[n])) == "knn"  # kNN is the largest mover
    assert min(deltas.values()) < 0  # movements not systematically toward nominal


@_needs_cache
def test_provenance_reconstructs_informing_rows_from_cache():
    """A tier-04 prediction's exact informing rows come from the committed cite graph.

    The tier-04 context was assembled over the declared citation edges, so a
    target's context_ref is its in-pool citation neighbourhood. We reconstruct it
    from the committed cache and assert ``rails.provenance`` extracts a real trail.
    """
    preds = contracts.load_artifact("arxiv.tier04_predictions").to_pylist()
    cite = contracts.load_artifact("arxiv.cite_edges").to_pylist()
    pool = {r["paper_id"] for r in preds}
    cite_out = collections.defaultdict(list)
    for e in cite:
        cite_out[e["src"]].append(e["dst"])

    target = next(r for r in preds if r["split"] == "test"
                  and len([d for d in cite_out[r["paper_id"]] if d in pool]) >= 3)
    context_keys = [d for d in cite_out[target["paper_id"]] if d in pool]

    trail = rails.provenance({"kind": "gaussian", "source": "edges",
                              "context_ref": context_keys})
    assert trail["source"] == "edges"
    assert trail["context_ref"] == context_keys
    assert all(k in pool for k in trail["context_ref"])  # every informing row is real
