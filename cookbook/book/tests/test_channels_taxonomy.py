"""Cache-backed oracle for the channel error taxonomy (§3.8) — CPU, no re-emit.

These run on CPU against the committed **channel error-taxonomy matrix** (no
server, no GPU, no re-drive) and assert the chapter's load-bearing facts — that
each evidence-channel failure mode maps to its CORRECT typed gRPC status code on
the wire (engine #193), not the `Internal`-for-everything a catch-all produces:

* **the taxonomy-to-golden oracle:** every committed `(mode → wire_code)` verdict
  matches its frozen golden in ``artifacts/channels/golden_metrics.json`` — the
  golden-stability gate;
* **the four typed wire codes:** duplicate → ``ALREADY_EXISTS``, unknown →
  ``NOT_FOUND``, column conflict → ``FAILED_PRECONDITION``, bad argument →
  ``INVALID_ARGUMENT`` — each measured on the ``grpc://`` transport, none of them
  ``INTERNAL`` / ``UNKNOWN`` (the #193 guarantee);
* **the embedded error-class companion:** the embedded engine raises the same
  NORMALIZED class (``duplicate`` / ``unknown`` / ``conflict`` / ``bad_argument``)
  with no wire code — the cross-transport contract is the class;
* **the client-side dtype guard:** an invalid dtype STRING is rejected client-side
  on both transports (a ``ValueError`` that never reaches the wire) — a measured
  property, distinct from the wire ``INVALID_ARGUMENT`` cell (an empty channel id);
* **the cross-transport parity verdict:** remote == embedded class for every mode,
  recorded and asserted as a committed fact.

The cross-transport ``remote == embedded`` parity is a ONE-TIME emit-side LIVE
check (recorded in ``channels_taxonomy.json``); PR CI never re-diffs two static
artifacts. If the emitted cache is absent the matrix-backed checks skip, but the
committed golden metrics, once present, are always asserted.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_CHANNELS = contracts._dataset_dir("channels")
_HAVE_CACHE = (_CHANNELS / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="channels taxonomy cache not emitted")

# The four headline failure modes and the gRPC status code each maps to under #193.
_EXPECTED_WIRE = {
    "duplicate": "ALREADY_EXISTS",
    "unknown": "NOT_FOUND",
    "column_conflict": "FAILED_PRECONDITION",
    "bad_argument": "INVALID_ARGUMENT",
}
_EXPECTED_CLASS = {
    "duplicate": "duplicate",
    "unknown": "unknown",
    "column_conflict": "conflict",
    "bad_argument": "bad_argument",
}


def _matrix() -> dict:
    return contracts.load_artifact("channels.matrix")


def _record() -> dict:
    return contracts.load_artifact("channels.channels_taxonomy")


# --------------------------------------------------------------------------- #
# the taxonomy-to-golden oracle
# --------------------------------------------------------------------------- #


@_needs_cache
def test_every_taxonomy_verdict_matches_golden():
    """Every committed taxonomy verdict matches its frozen golden — the golden the
    chapter renders against. A drift in any cell fails CI here."""
    for mode in _EXPECTED_WIRE:
        contracts.assert_close(f"channels.taxonomy.{mode}_wire_code_matches", 1.0)
        contracts.assert_close(f"channels.taxonomy.{mode}_class_parity", 1.0)
    contracts.assert_close("channels.taxonomy.all_modes_typed_not_internal", 1.0)
    contracts.assert_close("channels.taxonomy.client_dtype_guard_client_side", 1.0)
    contracts.assert_close("channels.parity.all_modes_class_equal", 1.0)


# --------------------------------------------------------------------------- #
# the four typed wire codes (the #193 headline)
# --------------------------------------------------------------------------- #


@_needs_cache
def test_each_mode_maps_to_its_typed_wire_code():
    """Each channel failure mode maps to its CORRECT typed gRPC status code on the
    grpc:// transport — duplicate→ALREADY_EXISTS, unknown→NOT_FOUND, column
    conflict→FAILED_PRECONDITION, bad argument→INVALID_ARGUMENT (the #193
    taxonomy that replaced Internal-for-everything)."""
    remote = _matrix()["remote"]
    for mode, expected in _EXPECTED_WIRE.items():
        cell = remote[mode]
        assert cell["raised"] is True, mode
        assert cell["wire_code"] == expected, f"{mode}: {cell['wire_code']} != {expected}"


@_needs_cache
def test_no_failure_collapses_to_internal():
    """The #193 guarantee: no typed failure mode collapses to INTERNAL / UNKNOWN
    on the wire — each speaks its true gRPC code."""
    remote = _matrix()["remote"]
    for mode in _EXPECTED_WIRE:
        assert remote[mode]["wire_code"] not in (None, "INTERNAL", "UNKNOWN"), mode


# --------------------------------------------------------------------------- #
# the embedded error-class companion + parity
# --------------------------------------------------------------------------- #


@_needs_cache
def test_embedded_companion_carries_the_normalized_class():
    """The embedded engine raises the same NORMALIZED error class for each mode,
    with NO wire code (it is in-process, not on the wire) — the cross-transport
    contract is the class, not the native Python exception type."""
    embedded = _matrix()["embedded"]
    for mode, expected_class in _EXPECTED_CLASS.items():
        cell = embedded[mode]
        assert cell["raised"] is True, mode
        assert cell["error_class"] == expected_class, mode
        assert cell["wire_code"] is None, f"{mode}: embedded carries no wire code"


@_needs_cache
def test_remote_equals_embedded_class_for_every_mode():
    """The recorded one-time live parity verdict: remote == embedded normalized
    error class for every channel failure mode. The two transports raise different
    native exception TYPES (embedded RuntimeError/ValueError, wire grpc.RpcError);
    the honest observable is the class."""
    rec = _record()
    parity = rec["parity"]
    assert parity, "the parity record must carry per-mode verdicts"
    assert all(p["class_equal"] for p in parity), "every mode must agree on class across transports"
    assert rec["parity_verdict"].startswith("remote == embedded")
    contracts.assert_close("channels.parity.all_modes_class_equal", 1.0)


# --------------------------------------------------------------------------- #
# the client-side dtype guard (distinct from the wire INVALID_ARGUMENT cell)
# --------------------------------------------------------------------------- #


@_needs_cache
def test_invalid_dtype_is_a_client_side_guard_not_a_wire_code():
    """An invalid column dtype STRING is rejected CLIENT-SIDE on both transports
    (a ValueError that never reaches the wire — no StatusCode), distinct from the
    wire INVALID_ARGUMENT cell (an empty channel id, server-rejected)."""
    guard = _matrix()["client_side_dtype_guard"]
    for arm in ("embedded", "remote"):
        assert guard[arm]["raised"] is True, arm
        assert guard[arm]["wire_code"] is None, f"{arm}: dtype guard must not reach the wire"
        assert guard[arm]["native_type"] == "ValueError", arm
        assert guard[arm]["error_class"] == "bad_argument", arm


# --------------------------------------------------------------------------- #
# the documented INTERNAL residual + zero deviation from #193
# --------------------------------------------------------------------------- #


@_needs_cache
def test_internal_is_the_documented_residual_and_no_deviation():
    """INTERNAL is the documented residual of the taxonomy (a genuine DB fault is
    not fabricated), and the measured taxonomy has ZERO deviation from #193 — every
    mode maps as intended."""
    rec = _record()
    assert rec["deviations"] == [], f"unexpected #193 deviation: {rec['deviations']}"
    assert "residual" in rec["internal_residual"].lower()
    # the measured taxonomy equals the #193-intended one, exactly.
    assert rec["taxonomy"] == rec["expected_taxonomy"]
