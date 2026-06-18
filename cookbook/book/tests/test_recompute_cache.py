"""Cache-backed checks for incremental-recompute + opt-in caching (SPEC-03 / W-61).

These run on CPU against the committed **embedded-canonical** matrix (no server,
no GPU, no re-execution of the engine producers) and assert the chapter's
load-bearing facts:

* **the matrix-to-golden oracle** — every committed verdict matches its frozen
  golden in ``artifacts/recompute/golden_metrics.json``: the golden-stability gate;
* **family 1 — cache-hit-reuses** — opt-in memoization reuses an exact prior
  materialisation (by table-name identity), recomputes on any full-descriptor
  change (a different ``k`` / ``min_similarity`` / ``hops``), and the
  ``UnpinnedAtInstant`` producer honestly never hits;
* **family 2 — staleness (definition_changed arm)** — a table is ``fresh`` against
  its own recorded hash and ``stale`` (reason ``definition_changed``) against a
  different one. The ``result_digest`` input-drift arm is CUT (a written rationale
  in the chapter + the record), not measured — so the test asserts it is recorded
  as a deliberate cut, never silently absent;
* **family 3 — recompute-restores** — a child recompute is byte-identical; a parent
  ``report_only`` reports the downstream-stale child without recomputing it;
  ``cascade='downstream'`` sweeps parent + child once; ``derives_from`` is one-hop
  ``result_digest`` lineage.

If the emitted cache is absent the matrix-backed checks skip; the committed golden
metrics, once present, are always asserted.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_RECOMPUTE = contracts._dataset_dir("recompute")
_HAVE_CACHE = (_RECOMPUTE / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="recompute cache not emitted")


def _matrix() -> dict:
    return contracts.load_artifact("recompute.matrix")


def _record() -> dict:
    return contracts.load_artifact("recompute.record")


# --------------------------------------------------------------------------- #
# the matrix-to-golden oracle
# --------------------------------------------------------------------------- #


@_needs_cache
def test_every_verdict_matches_golden():
    """Every committed matrix verdict matches its frozen golden — the golden the
    chapter renders against. A drift in any cell fails CI here."""
    # the boolean verdicts that must be 1.0 (the property holds)
    for metric in (
        "cache.bng_reused",
        "cache.bng_k4_recomputed",
        "cache.bng_minsim_recomputed",
        "cache.prop_reused",
        "cache.prop_hops_recomputed",
        "staleness.fresh",
        "staleness.stale_on_definition_change",
        "recompute.byte_identical",
        "recompute.report_only_recomputes_only_parent",
        "recompute.derives_from_all_result_digest",
    ):
        contracts.assert_close(f"recompute.{metric}", 1.0)
    # the unpinned producer honestly never reuses → exactly 0.0
    contracts.assert_close("recompute.cache.unpinned_reused", 0.0)
    # the lineage / cascade counts — exact, tol 0
    contracts.assert_close("recompute.recompute.downstream_stale_count", 1.0)
    contracts.assert_close("recompute.recompute.cascade_recomputed_count", 2.0)
    contracts.assert_close("recompute.recompute.derives_from_edge_count", 2.0)


# --------------------------------------------------------------------------- #
# family 1 — cache-hit-reuses (reuse observed by table-name identity)
# --------------------------------------------------------------------------- #


@_needs_cache
def test_exact_recomputation_is_reused_by_name_identity():
    """An exact prior materialisation is reused: ``build_neighbor_graph(k=3)`` under
    ``cache='use'`` returns the SAME table name as the ``cache='bypass'`` build, and
    the pinned-embedding ``propagate_embeddings(hops=1)`` reuses across two calls.
    Reuse is a name-identity boolean, never a wall-clock measurement."""
    cache = _matrix()["cache"]
    assert cache["bng_reused"] is True
    assert cache["prop_reused"] is True
    # the recorded names back the identity claim
    names = cache["names"]
    assert names["g_bypass"] == names["g_use"]
    assert names["prop_h1_a"] == names["prop_h1_b"]


@_needs_cache
def test_any_full_descriptor_change_recomputes():
    """The probe keys on the FULL producing descriptor, not just one knob: a
    different ``k``, a ``min_similarity``, or a different ``hops`` each yields a new
    materialisation (a different name) — never a stale reuse."""
    cache = _matrix()["cache"]
    assert cache["bng_k4_recomputed"] is True
    assert cache["bng_minsim_recomputed"] is True
    assert cache["prop_hops_recomputed"] is True
    names = cache["names"]
    assert names["g_k4"] != names["g_use"]
    assert names["g_minsim"] != names["g_use"]
    assert names["prop_h2"] != names["prop_h1_a"]


@_needs_cache
def test_unpinned_producer_honestly_never_reuses():
    """``generate_embeddings`` is anchored on an ``UnpinnedAtInstant`` source (no
    version surface), so ``cache='use'`` twice returns two DIFFERENT names — the
    cache is honestly OFF there, not silently broken (SPEC-03 §4.3)."""
    cache = _matrix()["cache"]
    assert cache["unpinned_reused"] is False
    assert cache["names"]["emb_a"] != cache["names"]["emb_b"]


# --------------------------------------------------------------------------- #
# family 2 — staleness (definition_changed arm; result_digest drift is CUT)
# --------------------------------------------------------------------------- #


@_needs_cache
def test_fresh_against_own_hash_stale_against_a_different_hash():
    """The ``definition_changed`` staleness arm: a table is ``fresh`` against its
    own recorded definition hash and ``stale`` against a different one, with reason
    ``definition_changed`` naming both ``recorded`` and ``current``."""
    s = _matrix()["staleness"]
    assert s["is_fresh"] is True
    assert s["is_stale_on_definition_change"] is True
    assert s["fresh_verdict"]["staleness"] == "fresh"
    stale = s["stale_verdict"]
    assert stale["staleness"] == "stale"
    reason = next(r for r in stale["reasons"] if r["reason"] == "definition_changed")
    assert "recorded" in reason and "current" in reason


@_needs_cache
def test_result_digest_input_drift_arm_is_a_recorded_cut():
    """The sibling ``result_digest`` input-drift staleness arm is CUT — a written
    rationale, never a half-shipped fake demo. The record states the cut explicitly
    so the omission is a deliberate, auditable engineering decision."""
    note = _record()["staleness_result_digest_drift_cut"]
    assert "CUT" in note
    assert "byte-identical" in note
    # the cut metric is genuinely absent from the golden (not silently faked to 0)
    with pytest.raises(KeyError):
        contracts.golden("recompute.staleness.result_digest_drift")


# --------------------------------------------------------------------------- #
# family 3 — recompute-restores (byte-identical replay, cascade, lineage)
# --------------------------------------------------------------------------- #


@_needs_cache
def test_child_recompute_is_byte_identical():
    """``recompute(child, report_only)`` re-invokes the recorded producer over the
    inputs' current (unmoved) state and yields a byte-identical output: the
    recompute's verdict against the original's recorded hash is ``match`` AND its own
    recorded definition hash equals the original's (SPEC-03 §5.1 — the complete
    descriptor replays faithfully)."""
    rc = _matrix()["recompute"]
    assert rc["child_recompute_outcome"] == "computed"
    assert rc["child_recompute_byte_identical"] is True


@_needs_cache
def test_report_only_reports_but_does_not_recompute_downstream():
    """``cascade='report_only'`` (the default) recomputes the NAMED table only and
    REPORTS the downstream-stale set (via ``derives_from_closure``); it recomputes
    none of it — the consumer decides what to do with the report (SPEC-03 §5.4)."""
    rc = _matrix()["recompute"]
    assert rc["report_only_recomputes_only_parent"] is True
    assert rc["child_in_downstream_stale"] is True
    assert rc["downstream_stale_count"] == 1


@_needs_cache
def test_cascade_downstream_sweeps_parent_and_child_once():
    """``cascade='downstream'`` is ONE bounded topological sweep on ONE explicit
    request: the named parent then its transitive dependent (the child), in order,
    each once — parent + child = 2 recomputed (SPEC-03 §5.4)."""
    rc = _matrix()["recompute"]
    assert rc["cascade_recomputed_count"] == 2


@_needs_cache
def test_derives_from_is_one_hop_result_digest_lineage():
    """``derives_from(emb)`` returns the one-hop reverse-dependency edges — every
    ready table whose recorded input anchors name ``emb`` (the neighbor-graph + the
    propagation), each a ``result_digest`` edge (the cacheable immutable-input
    lineage, SPEC-03 §3.3)."""
    rc = _matrix()["recompute"]
    assert rc["derives_from_edge_count"] == 2
    assert rc["derives_from_all_result_digest"] is True
    assert rc["derives_from_kinds"] == ["result_digest"]


# --------------------------------------------------------------------------- #
# the engine/platform boundary (recorded, names no consumer)
# --------------------------------------------------------------------------- #


@_needs_cache
def test_boundary_is_mechanism_not_the_loop():
    """The record states the SPEC-03 §7 boundary: the engine ships the bounded
    MECHANISM (one probe per call, one recompute per request, one bounded sweep);
    the scheduled / monitored recompute LOOP is the consumer's composition. Names no
    consumer."""
    boundary = _record()["boundary"].lower()
    assert "mechanism" in boundary
    assert "loop" in boundary
    assert "names no consumer" in boundary
