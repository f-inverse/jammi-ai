"""`jammi.open_sessions()` — the live-session registry — enumerated by route.

The client itself knows every live session, so no external
patch of `jammi.connect` is needed to catch a leak; a registry populated in the
shared constructor of every resource-owning session class answers "what is
open right now" regardless of how the caller imported or bound `connect`.

Every route that constructs a NEW resource-owning session is exercised here:

* `jammi.connect("file://…")` — the embedded dispatch factory (`_open_embedded`).
* Direct construction of `jammi.EmbeddedBackend` (hermetic: a fake native
  handle, so this route needs neither the `[embedded]` extra nor a real
  catalog — it proves the registration lives in `EmbeddedBackend.__init__`
  itself, not somewhere inside `_open_embedded`).
* `jammi.connect("grpc://…")` — the remote dispatch factory (`open_remote`).
* Direct construction of `jammi.RemoteDatabase` over a bare `grpc.insecure_channel`
  (hermetic: grpcio channels are lazy — no server is dialed at construction,
  the same fact `test_target.py` and `test_add_source_format.py` rely on).

`tenant_scope()` on both backends yields the SAME already-registered instance
(see `_embedded._TenantScope.__enter__` and `RemoteDatabase.tenant_scope`), so
it is not its own route and is not separately exercised here.

A snapshot diff over `open_sessions()` (the `WeakSet` view above) cannot see
the leak shape that matters most — a bare `jammi.connect("grpc://…")`
statement, or a local dropped at a test frame's own exit, is
refcount-collected before any `finally`/fixture-teardown code runs, so it is
already gone from the `WeakSet` by the time a diff looks. `observe()` /
`open_session_labels()` below are the EVENT- and LEDGER-backed views that see
a session for as long as it is registered, independent of whether anything
still holds a reference to it — the property a leak guard actually needs.
"""

from __future__ import annotations

import gc
import threading

import grpc
import pytest

import jammi
from jammi import EmbeddedBackend, RemoteDatabase


class _FakeNative:
    """A stand-in for the compiled `jammi_native` handle — just enough surface
    (`close(release)`) for `EmbeddedBackend` to delegate to, so the direct-
    construction route is testable without the `[embedded]` extra or a real
    catalog file."""

    def __init__(self) -> None:
        self.closed_with = None

    def close(self, release: bool = False) -> None:
        self.closed_with = release


def _dead_channel() -> grpc.Channel:
    """A lazily-constructed, never-dialed gRPC channel to a loopback address
    nothing listens on. Construction alone performs no I/O (grpcio channels
    are lazy), so this is hermetic — no server, no port collision risk beyond
    the (unused) address itself."""
    return grpc.insecure_channel("127.0.0.1:1")


# --- Route: jammi.connect("file://…") ---------------------------------------


@pytest.mark.embedded
def test_connect_file_route_appears_and_disappears(tmp_path):
    db = jammi.connect(f"file://{tmp_path}")
    try:
        assert db in jammi.open_sessions()
    finally:
        db.close()
    assert db not in jammi.open_sessions()


# --- Route: direct EmbeddedBackend construction ------------------------------


def test_direct_embedded_backend_construction_appears_and_disappears():
    db = EmbeddedBackend(_FakeNative())
    assert db in jammi.open_sessions()
    db.close()
    assert db not in jammi.open_sessions()


def test_embedded_close_is_idempotent_and_stays_absent():
    db = EmbeddedBackend(_FakeNative())
    db.close()
    assert db not in jammi.open_sessions()
    db.close()  # second close: no error, still absent
    assert db not in jammi.open_sessions()


def test_embedded_unclosed_session_disappears_once_collected():
    db = EmbeddedBackend(_FakeNative())
    assert db in jammi.open_sessions()
    del db
    gc.collect()
    assert not any(isinstance(s, EmbeddedBackend) for s in jammi.open_sessions())


# --- Route: jammi.connect("grpc://…") ----------------------------------------


def test_connect_grpc_route_appears_and_disappears():
    # No server is dialed: grpcio channels are lazy (see test_target.py /
    # test_add_source_format.py), so this is hermetic.
    db = jammi.connect("grpc://127.0.0.1:1")
    try:
        assert db in jammi.open_sessions()
    finally:
        db.close()
    assert db not in jammi.open_sessions()


# --- Route: direct RemoteDatabase construction -------------------------------


def test_direct_remote_database_construction_appears_and_disappears():
    db = RemoteDatabase(
        _dead_channel(), session_id="s-1", endpoint="127.0.0.1:1", tls=False
    )
    assert db in jammi.open_sessions()
    db.close()
    assert db not in jammi.open_sessions()


def test_remote_close_is_idempotent_and_stays_absent():
    db = RemoteDatabase(
        _dead_channel(), session_id="s-2", endpoint="127.0.0.1:1", tls=False
    )
    db.close()
    assert db not in jammi.open_sessions()
    db.close()  # second close: no error, still absent
    assert db not in jammi.open_sessions()


def test_remote_unclosed_session_disappears_once_collected():
    db = RemoteDatabase(
        _dead_channel(), session_id="s-3", endpoint="127.0.0.1:1", tls=False
    )
    assert db in jammi.open_sessions()
    del db
    gc.collect()
    assert not any(isinstance(s, RemoteDatabase) for s in jammi.open_sessions())


# --- Two open at once, across both transports --------------------------------


def test_two_sessions_open_at_once_are_both_listed():
    a = EmbeddedBackend(_FakeNative())
    b = RemoteDatabase(
        _dead_channel(), session_id="s-4", endpoint="127.0.0.1:1", tls=False
    )
    try:
        live = jammi.open_sessions()
        assert a in live
        assert b in live
    finally:
        a.close()
        b.close()
    live = jammi.open_sessions()
    assert a not in live
    assert b not in live


# --- The registry observes EVENTS, by handle and label, independent of reachability ---
#
# `jammi.open_sessions()` (above) is a `WeakSet` snapshot: it cannot see a
# session that is refcount-collected before anything ever diffs it. These
# tests exercise the ledger (`open_session_labels()`) and the event stream
# (`observe()`) instead, which see a session for as long as it is
# registered, independent of reachability.


class _Recorder:
    """A stand-in `observe()` listener: records every `(handle, label)` it is
    delivered, separately for register and unregister."""

    def __init__(self) -> None:
        self.registered: list = []
        self.unregistered: list = []

    def on_register(self, handle: int, label: str) -> None:
        self.registered.append((handle, label))

    def on_unregister(self, handle, label) -> None:
        self.unregistered.append((handle, label))


def test_observe_sees_register_and_unregister_for_grpc_connect_route():
    rec = _Recorder()
    unsubscribe = jammi.observe(rec.on_register, rec.on_unregister)
    try:
        db = jammi.connect("grpc://127.0.0.1:1")
        assert len(rec.registered) == 1
        handle, label = rec.registered[0]
        assert label == "127.0.0.1:1"
        assert rec.unregistered == []
        db.close()
        assert rec.unregistered == [(handle, label)]
    finally:
        unsubscribe()


def test_observe_sees_register_and_unregister_for_direct_construction():
    rec = _Recorder()
    unsubscribe = jammi.observe(rec.on_register, rec.on_unregister)
    try:
        db = EmbeddedBackend(_FakeNative(), label="/data/catalog-a")
        assert len(rec.registered) == 1
        handle, label = rec.registered[0]
        assert label == "/data/catalog-a"
        assert rec.unregistered == []
        db.close()
        assert rec.unregistered == [(handle, label)]
    finally:
        unsubscribe()


def test_dropped_without_close_leaves_a_register_with_no_unregister():
    """The property the leak guard needs: a session dropped WITHOUT `close()`
    still fired a register event, never an unregister — and its handle+label
    survive in `open_session_labels()` after the object itself is gone. A
    bare-statement `connect()` (no local binding held across the collection)
    is exactly the shape a snapshot diff over `open_sessions()` misses."""
    rec = _Recorder()
    unsubscribe = jammi.observe(rec.on_register, rec.on_unregister)
    try:
        jammi.connect("grpc://127.0.0.1:1")  # bare statement: no surviving ref
        gc.collect()
        assert len(rec.registered) == 1
        handle, label = rec.registered[0]
        assert label == "127.0.0.1:1"
        assert rec.unregistered == []  # never closed -> never fires
        # The session object itself is gone (collected), yet its handle+label
        # are still in the ledger — the property `open_sessions()` cannot
        # offer, and the one a leak guard actually needs. This entry is a
        # deliberate, permanent orphan (there is no session object left to
        # `close()`); it is unique to this test (a fresh handle each run) and
        # never collides with another test's assertions.
        assert (handle, label) in jammi.open_session_labels()
    finally:
        unsubscribe()


def test_open_session_labels_reports_the_printable_target():
    a = EmbeddedBackend(_FakeNative(), label="/data/catalog-b")
    b = RemoteDatabase(
        _dead_channel(), session_id="s-5", endpoint="10.0.0.9:8081", tls=False
    )
    try:
        labels = dict(jammi.open_session_labels())
        assert labels[a._session_handle] == "/data/catalog-b"
        assert labels[b._session_handle] == "10.0.0.9:8081"
    finally:
        a.close()
        b.close()


def test_handles_are_unique_across_sessions_even_after_collection():
    """Guards against a handle scheme that reuses `id()`: repeatedly construct,
    drop and collect, and construct again — no handle a still-tracked session
    ever held may be handed to a later one.

    A single construct/drop/construct pair is a coin flip on whether CPython's
    allocator happens to reuse the just-freed address for the next same-size
    object — an `id()`-backed handle scheme would only SOMETIMES collide on
    one trial. Looping many rapid open/drop cycles, and checking every new
    handle against every handle ever seen, makes the property deterministic:
    each iteration frees the address the previous session held, which is
    exactly the address CPython's per-size-class freelist reuses next, so an
    `id()`-backed handle scheme collides on some iteration with overwhelming
    reliability, while a real monotonic counter never repeats."""
    seen_handles: set = set()
    for _ in range(200):
        db = EmbeddedBackend(_FakeNative())
        handle = db._session_handle
        assert handle not in seen_handles, f"handle {handle} reused across sessions"
        seen_handles.add(handle)
        del db
        gc.collect()


def test_unsubscribe_stops_delivery():
    rec = _Recorder()
    unsubscribe = jammi.observe(rec.on_register, rec.on_unregister)
    unsubscribe()
    db = EmbeddedBackend(_FakeNative())
    db.close()
    assert rec.registered == []
    assert rec.unregistered == []
    unsubscribe()  # idempotent: a second call is a no-op, not an error


def test_concurrent_open_close_produce_balanced_events():
    rec = _Recorder()
    unsubscribe = jammi.observe(rec.on_register, rec.on_unregister)
    n_threads = 8

    def _open_and_close(i: int) -> None:
        db = EmbeddedBackend(_FakeNative(), label=f"/data/catalog-{i}")
        db.close()

    try:
        threads = [
            threading.Thread(target=_open_and_close, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(rec.registered) == n_threads
        assert len(rec.unregistered) == n_threads
        assert len(set(h for h, _ in rec.registered)) == n_threads
        assert set(rec.registered) == set(rec.unregistered)
    finally:
        unsubscribe()


class _Orphan:
    """A minimal stand-in "session" for exercising `_sessions.register()`
    directly, without a real `EmbeddedBackend`/`RemoteDatabase` — needs
    only to support weak references (a plain `object()` does not, and
    `register()` weak-references it via `_live`/`_session_handles`)."""


# --- a falsy label is never stored/delivered as "" ------------------------


def test_direct_construction_with_no_label_reports_a_non_empty_label():
    """`EmbeddedBackend(native)` (no `label=`) defaults to `label=""` at the
    call site — the registry must not deliver or store that verbatim: `''`
    collapses "nobody gave this session a label" with "a caller explicitly
    labelled it the empty string", and a leak report naming `''` reads as
    blank rather than as a real (if uninformative) name."""
    rec = _Recorder()
    unsubscribe = jammi.observe(rec.on_register, rec.on_unregister)
    try:
        db = EmbeddedBackend(_FakeNative())
        try:
            assert len(rec.registered) == 1
            handle, label = rec.registered[0]
            assert label != ""
            assert label
            assert dict(jammi.open_session_labels())[handle] == label
        finally:
            db.close()
    finally:
        unsubscribe()


def test_remote_direct_construction_with_no_label_also_reports_non_empty():
    """Same property, other transport: `RemoteDatabase`'s own registration
    call in `_database.py` always passes `endpoint`, but the registry
    itself must not depend on every future call site remembering to — the
    non-empty guarantee lives at the ONE seam (`register()`), not at each
    caller."""
    from jammi import _sessions as _sessions_module

    rec = _Recorder()
    unsubscribe = jammi.observe(rec.on_register, rec.on_unregister)
    try:
        handle = _sessions_module.register(_Orphan(), "")
        try:
            assert len(rec.registered) == 1
            _, label = rec.registered[0]
            assert label != ""
            assert label
        finally:
            _sessions_module._open_ledger.pop(handle, None)
    finally:
        unsubscribe()


# --- the non-weak ledger is bounded ----------------------------------------


def test_ledger_stays_bounded_when_thousands_of_sessions_are_dropped_without_close():
    """Refutes the unbounded shape ("N dropped sessions -> N retained
    entries, forever"): register far more
    sessions than `_LEDGER_CAP` without ever closing them and confirm the
    ledger's own size never exceeds the stated cap, rather than growing
    without bound for the life of the process. A leak DETECTOR must not
    itself retain the leaked quantity unboundedly."""
    from jammi import _sessions as _sessions_module

    before = len(_sessions_module._open_ledger)
    n = _sessions_module._LEDGER_CAP + 3000
    handles = [
        _sessions_module.register(_Orphan(), f"orphan-{i}") for i in range(n)
    ]

    after = len(_sessions_module._open_ledger)
    assert after <= _sessions_module._LEDGER_CAP, (
        f"ledger grew to {after} entries registering {n} never-closed "
        f"sessions, past its own stated cap of {_sessions_module._LEDGER_CAP}"
    )
    # The cap bites (this test's own registrations alone exceed it), not
    # merely "still small because nothing else happened to grow it" —
    # otherwise this assertion would pass vacuously on an unbounded ledger
    # too, as long as no prior test had pushed it over the cap yet.
    assert after < before + n

    # The MOST RECENTLY registered handles are the ones still present — a
    # FIFO-by-registration-order eviction, not an arbitrary one.
    assert handles[-1] in _sessions_module._open_ledger
    assert handles[0] not in _sessions_module._open_ledger
