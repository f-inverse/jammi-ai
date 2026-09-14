"""The live-session registry — every open :class:`~jammi.Session`, by construction.

The client itself knows every session it opened, so it needs no cooperation
from a caller (or a test harness) to answer "what is still open": every route
that constructs a resource-owning session object — `jammi.connect`, direct
construction of :class:`~jammi.EmbeddedBackend` or
:class:`~jammi.RemoteDatabase`, and any future `open`/`from_*`/tenant-scope
helper that yields a NEW resource-owning object rather than returning an
existing one — registers here, in that object's `__init__`, the ONE seam every
route passes through. `tenant_scope()` on both backends yields the SAME
already-registered instance (see `_embedded._TenantScope` and
`RemoteDatabase.tenant_scope`), so it needs no registration of its own.

`EmbeddedBackend` and `RemoteDatabase` are the objects a leak means: each one
owns the resource its own `close()` releases (the catalog file's release
handshake; the gRPC + Flight channels) — there is no separate "session façade"
object wrapping either one, so registering the backend IS registering the
session `connect()` returned.

A `weakref.WeakSet` so a session a caller drops without closing disappears
here too, exactly when it is collected — this registry answers "what is open
right now", never "what did the caller forget to close"; a caller (or test
harness) that wants to catch a forgotten `close()` must hold its own strong
reference across the window it is checking, so the entry survives to be seen.
"""

from __future__ import annotations

import threading
import weakref
from typing import Tuple

_lock = threading.Lock()
_live: "weakref.WeakSet[object]" = weakref.WeakSet()


def register(session: object) -> None:
    """Record `session` as open. Called once, from the `__init__` of every
    class that owns a session's underlying resource."""
    with _lock:
        _live.add(session)


def unregister(session: object) -> None:
    """Record `session` as closed. Called from `close()`; a no-op if `session`
    is already absent (closed twice, or already collected), so `close()`
    stays idempotent."""
    with _lock:
        _live.discard(session)


def open_sessions() -> Tuple[object, ...]:
    """Every session currently registered as open, as a snapshot tuple.

    A diagnostic surface, not a control one: mechanism only, reflecting
    exactly the constructed-but-not-yet-closed sessions this process holds
    strong or weak references to right now. A session dropped without
    `close()` disappears from a later snapshot once garbage-collected, whether
    or not anything ever closed it — so this answers "what is open right
    now", not "what leaked"; a caller that wants to catch a forgotten
    `close()` must snapshot this, hold its own strong references across the
    window under test, and diff against a later snapshot.
    """
    with _lock:
        return tuple(_live)
