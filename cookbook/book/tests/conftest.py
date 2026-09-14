"""Suite-wide session-lifecycle rails.

The property this file establishes, over every test in this suite:

    **every engine session a test opens is closed before the test ends — and,
    for an embedded session, before its backing directory is removed.**

Two mechanisms, because one alone is a promise rather than a rail:

* :func:`embedded` / :func:`remote` — the affordance. A test asks for a
  connected session and never owns its teardown. The embedded fixture runs on
  ``tmp_path``, which pytest does **not** remove during the run (its
  ``tmp_path_factory`` retains the last few run directories and prunes them at a
  LATER session's start, in a process that holds no engine), and the fixture
  closes the session in its own finalizer, which runs before any of that. So the
  close-then-remove ordering is structural here, not a convention.

* :func:`_no_leaked_sessions` — the rail. An autouse guard that subscribes to
  the client's own session registry (``jammi.observe``, see
  ``clients/python/jammi/_sessions.py``) for the duration of a test and fails
  it BY NAME if any session registered during the test never unregistered.
  The registry fires its events from inside the ``__init__`` of every
  resource-owning session class (``EmbeddedBackend``, ``RemoteDatabase``) and
  from ``close()`` — the ONE seam every construction route passes through, so
  it is independent of how a test bound the name it called through (a plain
  ``jammi.connect(...)`` call, an aliased import, direct backend
  construction — see that module's docstring for the full enumeration this
  guard no longer needs to reason about). It also sees a session that was
  NEVER closed even after the object itself is garbage-collected, because the
  events are delivered synchronously at register/unregister time, not read
  off a liveness snapshot: a session dropped by refcount inside the test body
  (a bare ``jammi.connect(...)`` statement, or a local gone at frame exit) is
  reclaimed before any teardown code runs, so a snapshot taken only at
  teardown would never see it open at all — the events already fired by
  then, and this fixture already recorded them. A seventh site that opens a
  session by hand and forgets to close it does not quietly race the cleanup
  again; it fails, by name, on the first run.

Why a *closed* session and not merely a dropped one: the embedded engine holds
its catalog through SQLite's ``unix-excl`` VFS, so dropping the handle releases
nothing at any bounded moment — ``close()`` is the only awaited release (see
``jammi.EmbeddedBackend.close``). A live engine re-creates catalog/journal files
inside its directory, so a ``shutil.rmtree`` racing it fails with ``OSError:
[Errno 39] Directory not empty`` between its ``scandir`` and its ``rmdir``.

Known limit, stated rather than assumed: this guard sees exactly what the
client's registry sees — every session any constructor route registers. That
is every route today: ``jammi.connect`` and direct ``EmbeddedBackend`` /
``RemoteDatabase`` construction all call the registry's ``register()`` as the
last statement of their own ``__init__`` (see ``_sessions.py``), so within
this process no import-time binding shape and no construction path escapes
it — there is no separate module-attribute patch left to alias around, and so
no standing enumerating gate over binding shapes is needed here. What this
guard does NOT reach is the non-pytest lanes (scripts, recipes, quickstart,
the executed chapter cells) — a static gate over those is filed as issue
#539.
"""

from __future__ import annotations

import pytest

# Enables the `pytester` fixture: `test_session_lifecycle_guard.py` runs a
# throwaway pytest session, through THIS conftest, in a subprocess — the
# non-vacuity control for the runtime rail below.
pytest_plugins = ["pytester"]

try:  # the suite also runs where the [embedded] extra is absent
    import jammi
except ImportError:  # pragma: no cover - exercised only on a lean install
    jammi = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# the affordance
# --------------------------------------------------------------------------- #


@pytest.fixture
def embedded(tmp_path):
    """A connected embedded engine on ``tmp_path``, closed in teardown.

    A test that also needs the catalog directory (to write a parquet beside it,
    say) asks for ``tmp_path`` as well — it is the same directory.
    """
    if jammi is None:  # pragma: no cover - lean install
        pytest.skip("jammi is not installed")
    db = jammi.connect(f"file://{tmp_path}")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def remote():
    """A remote session against the conventional local endpoint, closed in
    teardown. Opening it dials nothing — the gRPC channel is lazy — so this is
    hermetic and needs no server."""
    if jammi is None:  # pragma: no cover - lean install
        pytest.skip("jammi is not installed")
    db = jammi.connect("grpc://127.0.0.1:8081")
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# the rail
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _no_leaked_sessions(request):
    """esc-112 fix (`closes_escape: esc-112`): fail a test that leaves a jammi
    session open.

    Subscribes to `jammi.observe()` for the duration of the test: every
    session ANY construction route registers fires `on_register(handle,
    label)` synchronously, and its `close()` fires `on_unregister(handle,
    label)` — both independent of whether this fixture (or anything else)
    still holds a reference to the session object. A handle that registered
    during the test and never unregistered is a leak, reported by its label,
    whether or not the session object itself is still reachable: a session
    that was never `close()`d is a leak even if it was collected by refcount
    the moment the test dropped its last reference.

    Unsubscribes in `finally`, before anything else, so this fixture's own
    teardown never races a later listener call against the diff below.
    """
    if jammi is None:  # pragma: no cover - lean install
        yield
        return

    # A leak inside an already-failing test is noise on top of the true cause, so
    # the guard warns there and fails only when the test itself passed.
    # `session.testsfailed` is pytest's own running count and needs no hook.
    failed_before = request.session.testsfailed
    registered: dict[int, str] = {}
    unregistered: set[int] = set()

    def _on_register(handle: int, label: str) -> None:
        registered[handle] = label

    def _on_unregister(handle: int, label: str) -> None:  # noqa: ARG001
        unregistered.add(handle)

    unsubscribe = jammi.observe(_on_register, _on_unregister)
    try:
        yield
    finally:
        unsubscribe()
        leaked: dict[int, str] = {
            handle: label
            for handle, label in registered.items()
            if handle not in unregistered
        }

        if leaked:
            names = ", ".join(
                f"{label!r} (handle {handle})"
                for handle, label in sorted(leaked.items(), key=lambda kv: kv[1])
            )
            message = (
                f"{request.node.nodeid} left {len(leaked)} jammi session(s) open: "
                f"{names}. Close every session in the test (or take one from the "
                "`embedded` / `remote` fixture): an embedded engine holds its catalog "
                "until close() returns, so a directory removed under a live session "
                "races it (OSError: [Errno 39] Directory not empty)."
            )
            if request.session.testsfailed > failed_before:
                # The test already failed; do not bury its cause under this one.
                request.node.warn(pytest.PytestWarning(message))
            else:
                pytest.fail(message, pytrace=False)
