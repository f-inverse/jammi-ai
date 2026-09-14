"""`jammi.open_sessions()` — the live-session registry — enumerated by route.

Issue #536, Z1: the client itself knows every live session, so no cookbook-side
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
"""

from __future__ import annotations

import gc
from importlib.util import find_spec

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


@pytest.mark.skipif(
    find_spec("jammi_native") is None,
    reason="needs the [embedded] extra to open a real file:// target",
)
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
