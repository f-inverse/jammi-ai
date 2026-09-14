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
  construction — see that module's docstring for the full enumeration). It
  also sees a session that was NEVER closed even after the object itself is
  garbage-collected, because the events are delivered synchronously at
  register/unregister time, not read off a liveness snapshot: a session
  dropped by refcount inside the test body (a bare ``jammi.connect(...)``
  statement, or a local gone at frame exit) is reclaimed before any teardown
  code runs, so a snapshot taken only at teardown would never see it open at
  all — the events already fired by then, and this fixture already recorded
  them. A session opened by hand and forgotten fails, by name, on the first
  run.

Why a *closed* session and not merely a dropped one: the embedded engine holds
its catalog through SQLite's ``unix-excl`` VFS, so dropping the handle releases
nothing at any bounded moment — ``close()`` is the only awaited release (see
``jammi.EmbeddedBackend.close``). A live engine re-creates catalog/journal files
inside its directory, so a ``shutil.rmtree`` racing it fails with ``OSError:
[Errno 39] Directory not empty`` between its ``scandir`` and its ``rmdir``.

Capability, not a flag: the rail also depends on the installed client having
a registry to subscribe to. Three states, checked once at import: ``jammi``
absent (a lean install — the affordance fixtures skip, and the rail is a
no-op); ``jammi`` present without ``jammi.observe`` (a client built before
the registry shipped — this happens whenever a lane installs a previously
published wheel rather than HEAD source, e.g. the nightly release-recipe leg
in ``.github/workflows/cookbook-render.yml``, which pins nothing and so
tracks the last PyPI release): the rail is INACTIVE for the WHOLE session —
every test still runs, but none of them is checked for a leak — and exactly
ONE ``pytest.PytestWarning`` names the installed version at session start, so
that silence is never silent; ``jammi`` present with ``jammi.observe``: the
rail below runs as documented.

Known limit, stated rather than assumed: the client's own registry
(``clients/python/jammi/_sessions.py``) sees every session any constructor
route registers, independent of how the caller bound the name it called
through — ``jammi.connect`` and direct ``EmbeddedBackend`` /
``RemoteDatabase`` construction both call the registry's ``register()`` as
the last statement of their own ``__init__``, so no import-time binding shape
and no construction path escapes the REGISTRY. This fixture, however, only
WATCHES the registry for the span of one test: it subscribes at that test's
own setup and reads the diff in its own ``finally``, before any coarser
fixture tears down. So the window this guard actually covers is a single
test's own fixture window, and the suite rule is that a test-opened session
is closed by that SAME test. A session registered before this fixture
subscribes — at import time, or in a module- or session-scoped fixture's own
setup — is outside the window and invisible to it; and a session that spans
tests (opened by one test, left open past that test's own teardown, and only
closed later by a different test or a coarser fixture) is reported, if it is
reported at all, against the test that OPENED it, never the one that
eventually closes it. That gap — plus the registry ledger's unbounded size
and its ``label=""`` default — is filed as issue #552. What this guard does
NOT reach at all is the non-pytest lanes (scripts, recipes, quickstart, the
executed chapter cells) — a static gate over those is filed as issue #539.
"""

from __future__ import annotations

import warnings

import pytest

# Enables the `pytester` fixture: `test_session_lifecycle_guard.py` runs a
# throwaway pytest session, through THIS conftest, in a subprocess — the
# non-vacuity control for the runtime rail below.
pytest_plugins = ["pytester"]

try:  # the suite also runs where the [embedded] extra is absent
    import jammi
except ImportError:  # pragma: no cover - exercised only on a lean install
    jammi = None  # type: ignore[assignment]

# Capability, not a flag (see module docstring): a `jammi` without the
# session registry cannot run the rail, whether or not it is installed at
# all. Checked once, here, rather than per test.
_RAIL_ACTIVE = jammi is not None and getattr(jammi, "observe", None) is not None


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


@pytest.fixture(scope="session", autouse=True)
def _warn_if_rail_inactive() -> None:
    """Session-start capability check (see module docstring's "Capability,
    not a flag" paragraph): if `jammi` is installed but predates the session
    registry, the leak rail is off for the whole run, and that has to be
    visible rather than a silent no-op repeated on every test.
    """
    if jammi is not None and not _RAIL_ACTIVE:
        version = getattr(jammi, "__version__", "unknown")
        warnings.warn(
            f"session-leak rail inactive: this jammi client (version "
            f"{version}) predates the session registry (`jammi.observe` is "
            "absent); no session leak will be reported for this run. "
            "Installing from a previously published wheel rather than HEAD "
            "source is expected to lag like this.",
            pytest.PytestWarning,
            stacklevel=1,
        )


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
    if not _RAIL_ACTIVE:  # lean install, or a client without the registry
        # `jammi` absent: nothing to observe. `jammi` present but without
        # `.observe`: `_warn_if_rail_inactive` already warned once for the
        # whole session; there is nothing this per-test fixture can check.
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
