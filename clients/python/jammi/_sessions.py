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

A `weakref.WeakSet` backs :func:`open_sessions`, so a session a caller drops
without closing disappears from THAT view exactly when it is collected — a
snapshot diff over `open_sessions()` answers "what is open right now", never
"what leaked". That is unsound for exactly the shape a leak guard needs to
catch: a bare `jammi.connect(...)` statement (or a local dropped at a test
frame's exit) is refcount-collected before any `finally`/fixture-teardown
code runs, so it is gone from the WeakSet before a diff ever sees it — the
after-count reads identical to the closed case.

So this module ALSO keeps a non-weak ledger of currently-open
`(handle, label)` pairs, and fires synchronous events on register/unregister
to any subscribed observer. The ledger and the events see a session for as
long as it is registered, independent of whether anything ever held a strong
reference to look — a caller that wants to know "was this handle EVER opened
and never closed" reads `open_session_labels()` (or an `observe()` listener's
delivered `(handle, label)` pairs) rather than diffing `open_sessions()`.

`register()` assigns each session a small monotonic integer handle — never
`id()`, which a already-collected-and-reused address would silently alias to
an unrelated, later object — and returns it so a caller can correlate a
registration with its eventual unregistration.
"""

from __future__ import annotations

import itertools
import threading
import weakref
from typing import Callable, Dict, Tuple

_lock = threading.Lock()
_live: "weakref.WeakSet[object]" = weakref.WeakSet()
_handle_counter = itertools.count(1)

# The handle this module assigned each currently-registered session — read
# under `_lock`, set on `register`, popped on `unregister`. A
# `WeakKeyDictionary`, keyed by the session object itself, so an entry never
# outlives the session it names and `unregister` can look one up by identity
# without this dict itself keeping the session alive.
_session_handles: "weakref.WeakKeyDictionary[object, int]" = (
    weakref.WeakKeyDictionary()
)

# The non-weak ledger: every handle currently open, by label, independent of
# whether the session object itself is still reachable. Removed on
# `unregister`; NOT removed by garbage collection — that is the whole point
# (a dropped-without-close session stays visible here).
#
# Bounded to `_LEDGER_CAP` entries: a leak DETECTOR must not itself retain an
# unbounded copy of what it detects (an unbounded ledger retains one entry per
# session dropped without `close()`, forever, for the life of the process).
# Registering past the cap evicts the OLDEST
# still-open entry (FIFO by registration order — a plain `dict` already
# preserves insertion order since 3.7, so no separate ordering structure is
# needed) rather than growing further. This bounds `open_session_labels()`'s
# own retrospective view; it does NOT weaken detection for anything that
# actually uses this module's leak-catching property: `observe()` (the
# cookbook leak rail's mechanism — `cookbook/book/tests/conftest.py`) sees
# every register/unregister EVENT synchronously, as it fires, independent of
# the ledger's size or the cap — a subscriber watching a bounded window (one
# test, one pytest session) never needs more than that window holds. Only a
# caller asking `open_session_labels()` for a running total across a process
# that has abandoned more than `_LEDGER_CAP` sessions loses visibility into
# the oldest ones past the cap — a stated, bounded limit, not a silent one.
_LEDGER_CAP = 4096
_open_ledger: Dict[int, str] = {}

# Subscribed (on_register, on_unregister) pairs. A plain list under `_lock`;
# `observe()`'s returned unsubscriber removes by identity.
_listeners: list = []


def register(session: object, label: str) -> int:
    """Record `session` as open under `label`, returning its handle.

    Called once, as the LAST statement of the `__init__` of every class that
    owns a session's underlying resource — by the time this returns, the
    session is live in every view this module offers (`open_sessions()`,
    `open_session_labels()`, and every :func:`observe` listener), so its
    construction and its visibility here are the same event.

    `label` is the printable target the session was opened against (an
    embedded catalog location, or a remote endpoint) — never derived from
    `session` itself here, so it survives collection in the ledger and in a
    delivered event even after the session object is gone.

    A falsy `label` (`EmbeddedBackend.__init__`'s direct-construction route
    defaults to `label=""`) is never stored or
    delivered as `""` — that collapses "unlabeled" (this construction route
    passed nothing) with "labeled the empty string" (a caller explicitly
    named an empty target), and a downstream leak report naming `''` reads
    as a blank rather than as "this session was never given a label". `""`
    is replaced here, the ONE seam every construction route already passes
    through, with a non-empty placeholder carrying the session's own type
    and handle, so `open_session_labels()` and every `observe()` listener
    always see a printable, non-empty label regardless of which route
    constructed the session or whether it passed a label at all.
    """
    with _lock:
        handle = next(_handle_counter)
        if not label:
            label = f"<unlabeled {type(session).__name__} #{handle}>"
        _live.add(session)
        _session_handles[session] = handle
        _open_ledger[handle] = label
        if len(_open_ledger) > _LEDGER_CAP:
            oldest_handle = next(iter(_open_ledger))
            del _open_ledger[oldest_handle]
        listeners = list(_listeners)
    # Fired OUTSIDE the lock: a listener that itself calls back into this
    # module (e.g. `open_session_labels()`) must not deadlock on `_lock`, and
    # a slow or raising listener must not hold up another thread's
    # register/unregister.
    for on_register, _on_unregister in listeners:
        on_register(handle, label)
    return handle


def unregister(session: object) -> None:
    """Record `session` as closed. Called from `close()`; a no-op if `session`
    is already absent (closed twice, or already collected), so `close()`
    stays idempotent — a second `unregister` of the same session fires no
    event and touches nothing.
    """
    with _lock:
        handle = _session_handles.pop(session, None)
        if handle is None:
            return
        label = _open_ledger.pop(handle, None)
        _live.discard(session)
        listeners = list(_listeners)
    for _on_register, on_unregister in listeners:
        on_unregister(handle, label)


def open_sessions() -> Tuple[object, ...]:
    """Every session currently registered as open, as a snapshot tuple.

    A diagnostic surface, not a control one: mechanism only, reflecting
    exactly the constructed-but-not-yet-closed sessions this process holds
    strong or weak references to right now. A session dropped without
    `close()` disappears from a later snapshot once garbage-collected, whether
    or not anything ever closed it — so this answers "what is open right
    now", not "what leaked"; a caller that wants to catch a forgotten
    `close()` must snapshot this, hold its own strong references across the
    window under test, and diff against a later snapshot — or, to catch a
    session dropped without a surviving reference at all, use
    :func:`open_session_labels` or :func:`observe` instead, which see a
    session for as long as it is registered, independent of reachability.
    """
    with _lock:
        return tuple(_live)


def open_session_labels() -> Tuple[Tuple[int, str], ...]:
    """Every currently-open `(handle, label)` pair, independent of whether the
    session object itself is still reachable.

    Backed by a non-weak ledger, so a session dropped without `close()` stays
    listed here after it is collected — the property `open_sessions()`
    cannot offer, because its `WeakSet` sees exactly the sessions something
    still holds a reference to.
    """
    with _lock:
        return tuple(_open_ledger.items())


def observe(
    on_register: Callable[[int, str], None],
    on_unregister: Callable[[int, str], None],
) -> Callable[[], None]:
    """Subscribe to every future register/unregister event; returns an
    unsubscribe callable.

    Both callbacks run SYNCHRONOUSLY, on the thread that called `register` /
    `unregister`, outside `_lock` — so a listener sees `(handle, label)` for
    a registration or unregistration exactly as it happens, with no polling
    and no risk of missing a session that is opened and dropped-without-close
    faster than a diff could observe it. Thread-safe: two threads registering
    or unregistering concurrently each fire their own events without
    interleaving corruption, though the ORDER two threads' events arrive in
    is not itself guaranteed.

    Does not replay history: a listener sees only events fired after it
    subscribes. Idempotent unsubscribe (calling the returned callable twice
    is a no-op).
    """
    pair = (on_register, on_unregister)
    with _lock:
        _listeners.append(pair)

    def _unsubscribe() -> None:
        with _lock:
            try:
                _listeners.remove(pair)
            except ValueError:
                pass  # already unsubscribed

    return _unsubscribe
