"""Cache-backed checks for point-in-time correctness (H4) — CPU, no re-emit.

These run on CPU against the committed point-in-time cache (no server, no GPU, no
recompute) and assert the chapter's load-bearing facts:

* **the leakage delta is real and strictly positive** — the naive (current-state)
  feature peeks at the future and reports a higher downstream AUC than the
  leakage-safe as-of feature; the committed `pit.leakage_delta = naive_auc -
  asof_auc` matches its golden and is > 0;
* **the conformal honesty result** — the leaky calibration deviates from nominal
  coverage (it breaks the guarantee), while the as-of calibration holds nominal;
* **train == serve** — the SAME `asof_join` definition produced byte-identical
  feature rows on the embedded `Database` and a live `grpc://` `RemoteDatabase`:
  `pit.train_serve_skew == 0.0`;
* **the four-verdict materialization matrix** — `verify_materialization` returns
  `match` (a result-table-input producer anchors `ResultDigest`),
  `match_with_unpinned_inputs` (a file source has no version surface), `mismatch`
  (a wrong expected definition), and `missing_manifest` (no sidecar); plus the
  within-run round-trip (the Match table verifies against its own hash).

The cross-transport skew and the verdict matrix are ONE-TIME emit-side LIVE checks
(recorded in `point_in_time.json`); PR CI reads the committed cache and asserts the
frozen verdicts/goldens, never re-drives a server. If the emitted cache is absent the
checks skip, but the committed golden metrics, once present, are always asserted.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_PIT = contracts._dataset_dir("point_in_time")
_HAVE_CACHE = (_PIT / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="point-in-time cache not emitted")


def _record() -> dict:
    return contracts.load_artifact("point_in_time.record")


# --------------------------------------------------------------------------- #
# the leakage delta — measured, strictly positive
# --------------------------------------------------------------------------- #


@_needs_cache
def test_leakage_delta_matches_golden_and_is_positive():
    """The leak the as-of join closes: the naive (current-state) feature peeks at
    the future and reports a higher AUC than the leakage-safe as-of feature. The
    committed delta matches its golden AND is strictly positive."""
    rec = _record()
    delta = rec["leak"]["leakage_delta"]
    contracts.assert_close("point_in_time.pit.leakage_delta", delta)
    contracts.assert_close("point_in_time.pit.naive_auc", rec["leak"]["naive_auc"])
    contracts.assert_close("point_in_time.pit.asof_auc", rec["leak"]["asof_auc"])
    assert delta > 0.0, "the naive (leaky) AUC must be inflated above the honest as-of AUC"
    assert rec["leak"]["naive_auc"] > rec["leak"]["asof_auc"]


@_needs_cache
def test_facts_carry_real_future_leakage():
    """The time-stamped facts are real: a majority of the committed citation edges
    are future leakage relative to the one-year horizon (the leak the as-of join
    must close is not a toy)."""
    facts = _record()["facts"]
    assert facts["n_edges"] > 0
    assert facts["n_leaked"] > 0
    assert facts["leak_fraction"] > 0.5, "most citation edges are future-relative to the horizon"


# --------------------------------------------------------------------------- #
# the conformal honesty result
# --------------------------------------------------------------------------- #


@_needs_cache
def test_leaky_calibration_breaks_nominal_asof_holds_it():
    """The killer measurement: the leaky calibration's coverage deviates from
    nominal (its over-optimistic scores break the exchangeability the guarantee
    rests on), while the as-of calibration tracks nominal within its band."""
    rec = _record()
    cov_leaky = rec["leak"]["coverage_leaky"]
    cov_asof = rec["leak"]["coverage_asof"]
    nominal = rec["leak"]["nominal_coverage"]
    contracts.assert_close("point_in_time.pit.coverage_leaky", cov_leaky)
    contracts.assert_close("point_in_time.pit.coverage_asof", cov_asof)
    contracts.assert_close("point_in_time.pit.nominal_coverage", nominal)
    # the as-of calibration holds nominal more tightly than the leaky one
    assert abs(cov_asof - nominal) <= abs(cov_leaky - nominal), (
        "the as-of calibration must track nominal at least as tightly as the leaky one"
    )
    # the leaky calibration visibly misses nominal (the broken guarantee)
    assert abs(cov_leaky - nominal) >= 0.02, "the leaky calibration must visibly miss nominal"


# --------------------------------------------------------------------------- #
# train == serve, by construction
# --------------------------------------------------------------------------- #


@_needs_cache
def test_train_equals_serve_skew_is_zero():
    """The same `asof_join` definition produced byte-identical feature rows on the
    embedded `Database` and a live `grpc://` `RemoteDatabase` — one definition, both
    paths, skew exactly zero."""
    rec = _record()
    if not rec.get("skew_measured"):
        pytest.skip("skew arm not measured (emitted with --target embedded)")
    assert rec["train_serve_skew"] == 0.0
    contracts.assert_close("point_in_time.pit.train_serve_skew", 0.0)


@_needs_cache
def test_asof_preserves_the_spine_and_matches_known_facts():
    """The as-of join preserves every spine row (left-outer) and the matched-fact
    count equals the number of papers with a positive as-of in-degree — the
    leakage-safe matches, no more."""
    rec = _record()
    matched = rec["asof"]["matched_rows"]
    contracts.assert_close("point_in_time.pit.asof_matched_rows", float(matched))
    assert matched == rec["asof"]["asof_cited_papers"], (
        "matched (non-null) rows must equal the papers with a citation known by the horizon"
    )


# --------------------------------------------------------------------------- #
# the four-verdict materialization matrix
# --------------------------------------------------------------------------- #


@_needs_cache
def test_four_verdict_materialization_matrix():
    """`verify_materialization` returns each of the four verdicts, and the committed
    verdict goldens hold: `match` (a `ResultDigest`-anchored producer),
    `match_with_unpinned_inputs` (a file source), `mismatch` (a wrong expected
    definition), `missing_manifest` (no sidecar)."""
    matrix = _record()["verdict_matrix"]
    assert matrix["match"]["verdict"] == "match"
    assert matrix["match_with_unpinned_inputs"]["verdict"] == "match_with_unpinned_inputs"
    assert matrix["mismatch"]["verdict"] == "mismatch"
    assert matrix["missing_manifest"]["verdict"] == "missing_manifest"
    for metric in (
        "pit.verdict_match",
        "pit.verdict_match_with_unpinned",
        "pit.verdict_mismatch",
        "pit.verdict_missing_manifest",
        "pit.verdict_match_roundtrip",
    ):
        contracts.assert_close(f"point_in_time.{metric}", 1.0)


@_needs_cache
def test_match_case_is_anchored_by_a_result_digest_input():
    """The honest `Match`: it is anchored by a producer reading a RESULT-TABLE input
    (the embeddings result table → a `result_digest` anchor), the only way an output
    verifies as a clean `match`. A file source would be `match_with_unpinned_inputs`."""
    matrix = _record()["verdict_matrix"]
    anchors = matrix["match_input_anchors"]
    assert anchors, "the Match case must carry at least one input anchor"
    assert all(a["kind"] == "result_digest" for a in anchors), (
        "the clean Match is anchored by ResultDigest inputs, not UnpinnedAtInstant"
    )
    # the unpinned case names its unpinned (file) source honestly
    assert matrix["match_with_unpinned_inputs"]["unpinned"], (
        "the unpinned verdict must name the file source with no version surface"
    )


@_needs_cache
def test_mismatch_carries_both_definition_hashes():
    """The `mismatch` verdict carries both the expected and the found definition
    hash, so a caller can see exactly which definition the artifact is NOT."""
    mismatch = _record()["verdict_matrix"]["mismatch"]
    assert "expected" in mismatch and "found" in mismatch
    assert mismatch["expected"] != mismatch["found"]
