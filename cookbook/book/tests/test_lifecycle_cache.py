"""Cache-backed checks for the model catalog (§3.6) — CPU, no re-emit.

These run on CPU against the committed **embedded-canonical** catalog matrix (no
server, no GPU, no recompute) and assert the chapter's load-bearing facts:

* **the matrix-to-golden oracle:** every committed verdict matches its frozen
  golden in ``artifacts/lifecycle/golden_metrics.json`` — the golden-stability gate;
* **the referential-integrity matrix:** delete-while-referenced raises the typed
  ``referenced`` class; delete-absent (strict) is ``not_found`` (the typed
  ``ModelNotFound``, NOT invalid-argument); delete-absent (``if_exists=True``) is a
  no-op; and the headline property — every model in the catalog is referenced, so
  the delete-unreferenced-succeeds cell is unreachable (a measured property of the
  catalog, where every model is trained-and-referenced);
* **the catalog surface:** a model registers as ``registered`` via the only public
  path (training); ``describe_model`` / ``list_models`` reflect it;
* **the committed matrix shape:** the model projections carry exactly the minimal
  client-facing keys with the per-run UUID ``model_id`` stripped.

The cross-transport ``remote == embedded`` parity is a ONE-TIME emit-side LIVE
check (recorded in ``lifecycle.json``, continuously re-guarded by the engine's
gated conformance + catalog tests); PR CI never re-diffs two static artifacts.

If the emitted cache is absent the matrix-backed checks skip, but the committed
golden metrics, once present, are always asserted.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_LIFECYCLE = contracts._dataset_dir("lifecycle")
_HAVE_CACHE = (_LIFECYCLE / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="lifecycle cache not emitted")

# The minimal client-facing projection keys with the per-run UUID model_id
# stripped from the committed form (the projection is model_id + these).
_PROJECTION_KEYS = {"backend", "task", "status"}


def _matrix() -> dict:
    return contracts.load_artifact("lifecycle.matrix")


def _record() -> dict:
    return contracts.load_artifact("lifecycle.lifecycle")


# --------------------------------------------------------------------------- #
# the matrix-to-golden oracle
# --------------------------------------------------------------------------- #


@_needs_cache
def test_every_catalog_verdict_matches_golden():
    """Every committed matrix verdict matches its frozen golden — the golden the
    chapter renders against. A drift in any cell fails CI here."""
    for metric in (
        "register.status_registered",
        "delete.referenced_raises",
        "delete.every_catalog_model_referenced",
        "delete.absent_strict_not_found",
        "delete.absent_if_exists_noop",
        "parity.all_observables_equal",
    ):
        contracts.assert_close(f"lifecycle.{metric}", 1.0)


# --------------------------------------------------------------------------- #
# the referential-integrity matrix (the headline)
# --------------------------------------------------------------------------- #


@_needs_cache
def test_delete_while_referenced_raises_the_typed_referenced_error():
    """delete-while-referenced raises the typed ``ModelReferenced`` guard on BOTH
    a fine-tuned model (referenced by training_jobs.output_model_id) and its base
    model (referenced by training_jobs.base_model_id) — the normalized class is
    ``referenced``, and the verdict golden holds."""
    m = _matrix()
    for cell in ("delete_referenced", "ft_delete_referenced", "base_delete_referenced"):
        assert m[cell]["raised"] is True, cell
        assert m[cell]["error_class"] == "referenced", cell
    contracts.assert_close("lifecycle.delete.referenced_raises", 1.0)


@_needs_cache
def test_every_catalog_model_is_referenced():
    """The headline referential property: every model in the catalog is
    trained-and-referenced, so the delete-unreferenced-SUCCEEDS cell of the matrix
    is unreachable here — there is no bare unreferenced model in the catalog to
    delete. A measured property of the engine's catalog, not faked."""
    assert _matrix()["every_catalog_model_is_referenced"] is True
    contracts.assert_close("lifecycle.delete.every_catalog_model_referenced", 1.0)
    # the property is recorded in the provenance record for the coordinator
    assert "trained-and-referenced" in _record()["catalog_property"].lower()


@_needs_cache
def test_delete_absent_is_not_found_not_invalid_argument():
    """delete-absent WITHOUT if_exists is the ``not_found`` class — the typed
    ``ModelNotFound`` (the wire status is NOT_FOUND, never INVALID_ARGUMENT);
    WITH if_exists=True it is a clean no-op (no raise)."""
    m = _matrix()
    assert m["delete_absent_strict"]["raised"] is True
    assert m["delete_absent_strict"]["error_class"] == "not_found"
    assert m["delete_absent_if_exists"]["raised"] is False
    assert m["delete_absent_if_exists"]["error_class"] is None
    contracts.assert_close("lifecycle.delete.absent_strict_not_found", 1.0)
    contracts.assert_close("lifecycle.delete.absent_if_exists_noop", 1.0)


# --------------------------------------------------------------------------- #
# the catalog surface
# --------------------------------------------------------------------------- #


@_needs_cache
def test_register_reflected_in_describe_and_list():
    """A fresh model registers as ``registered`` via the only public path
    (training) and is reflected by describe / list as the minimal projection."""
    m = _matrix()
    assert m["registered"]["status"] == "registered"
    contracts.assert_close("lifecycle.register.status_registered", 1.0)


@_needs_cache
def test_committed_projections_carry_the_minimal_shape():
    """The committed model projections carry exactly the minimal client-facing
    keys (with the per-run UUID model_id stripped) — the same projection both
    transports return, so no server-internal field leaks."""
    m = _matrix()
    for proj in (m["registered"], *m["list_after_register"]):
        assert set(proj) == _PROJECTION_KEYS, f"{set(proj)} != {_PROJECTION_KEYS}"
        assert "model_id" not in proj, "the per-run UUID must be stripped from the committed form"
    # registration registers TWO rows on the public path: the base model + the
    # fine-tuned model (both registered).
    assert len(m["list_after_register"]) == 2
    assert all(p["status"] == "registered" for p in m["list_after_register"])


# --------------------------------------------------------------------------- #
# the cross-transport parity verdict (recorded, asserted as a fact)
# --------------------------------------------------------------------------- #


@_needs_cache
def test_remote_equals_embedded_for_every_observable():
    """The recorded one-time live parity verdict: remote == embedded for every
    catalog observable (the model projections and the normalized delete-error
    class). Asserted as a committed fact + the golden."""
    rec = _record()
    parity = rec["parity"]
    assert parity, "the parity record must carry per-observable verdicts"
    assert all(p["equal"] for p in parity), "every observable must agree across transports"
    assert rec["parity_verdict"].startswith("remote == embedded")
    contracts.assert_close("lifecycle.parity.all_observables_equal", 1.0)
