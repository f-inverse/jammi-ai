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
test's own fixture window, and **the suite rule is that a test-opened
session is closed by that SAME test** — that rule is what makes the
per-test guard sound at all: a session opened in test A and closed in test
B is reported against A (the OPENER), never against B (the closer, which
did nothing wrong), because A's own window closes with the session still
registered and B's window never saw a `register` event for it at all, only
an `unregister` it did not ask for. `test_session_lifecycle_guard.py`'s
``test_a_session_opened_in_one_test_and_closed_in_a_later_test_is_reported_against_the_opener``
pins this by name. A session registered before this fixture subscribes — at
import time, or in a module- or session-scoped fixture's own setup — is
outside a single test's window and invisible to the per-test guard; that gap
is covered by a SECOND mechanism, process-wide rather than per-test (below):
a session-scope baseline taken at ``pytest_sessionstart`` (before collection
runs, so even an import-time ``jammi.connect(...)`` at module scope is
inside its window) and a sweep at ``pytest_sessionfinish`` that fails the
WHOLE RUN, by label, for every handle registered anywhere in the process
during the session and never unregistered by the time the session ends —
independent of which single test's window (if any) the registration fell
inside. The per-test guard stays: it attributes a leak to the exact test
that caused it, which the session-wide sweep cannot do (by the time it runs,
every test has already finished) — the two are complementary, not
redundant. What neither guard reaches at all is the non-pytest lanes
(scripts, recipes, quickstart, the executed chapter cells) — those are
covered instead by the AST gate
``ci/scripts/check_cookbook_session_lifecycle.py``, which reads
the tree statically rather than running under any test harness.
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

# Process-wide baseline for the `pytest_sessionfinish` sweep below: every handle `jammi.observe()` reports registered ANYWHERE
# in this process during the run, and every handle it reports unregistered.
# Module globals, not fixture state — `pytest_sessionstart` fires before
# `pytest_collection`, so subscribing there sees even a module-import-time
# `jammi.connect(...)` (a construction the per-test `_no_leaked_sessions`
# fixture below can never see, since collection/import happens before any
# fixture — even a session-scoped one — gets to run its own setup).
_session_registered: dict[int, str] = {}
_session_unregistered: set[int] = set()
_session_unsubscribe = None


def pytest_sessionstart(session: pytest.Session) -> None:  # noqa: ARG001
    """Subscribe to the registry for the WHOLE run, before collection —
    the process-wide half of the leak rail. Runs even
    when `_RAIL_ACTIVE` is False; the capability check below is what makes
    subscribing a no-op in that case, same as the per-test fixture."""
    global _session_unsubscribe
    if not _RAIL_ACTIVE:
        return

    def _on_register(handle: int, label: str) -> None:
        _session_registered[handle] = label

    def _on_unregister(handle: int, label: str) -> None:  # noqa: ARG001
        _session_unregistered.add(handle)

    _session_unsubscribe = jammi.observe(_on_register, _on_unregister)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """The process-wide sweep: every session registered
    anywhere in this process during the run and never unregistered by the
    time it ends fails the WHOLE RUN, by label — independent of whether any
    single test's own `_no_leaked_sessions` window ever saw it (a
    module-/session-scoped fixture's setup, or an import-time construction,
    both fall outside every per-test window and so are invisible to that
    fixture; this sweep is the only guard that reaches them). Runs after
    every test has already finished, so it cannot attribute the leak to one
    test the way the per-test guard does — it can only say the run, as a
    whole, leaked.
    """
    if not _RAIL_ACTIVE or _session_unsubscribe is None:
        return
    _session_unsubscribe()

    leaked = {
        handle: label
        for handle, label in _session_registered.items()
        if handle not in _session_unregistered
    }
    if not leaked:
        return

    names = ", ".join(
        f"{label!r} (handle {handle})"
        for handle, label in sorted(leaked.items(), key=lambda kv: kv[1])
    )
    message = (
        f"SESSION-WIDE LEAK: {len(leaked)} jammi session(s) were opened "
        f"somewhere in this process and never closed by the end of the "
        f"run, outside any single test's own fixture window: {names}. This "
        "is the process-wide sweep, not the per-test guard — it catches a "
        "session opened at import time or in a module-/session-scoped "
        "fixture's own setup, neither of which `_no_leaked_sessions` can "
        "see."
    )
    # stdout, not stderr: pytest's own capture/report machinery (and this
    # rail's own pytester-based non-vacuity test, which reads `result.
    # outlines`/`result.stdout`) both look at the process's stdout stream.
    print(f"\n{message}\n")
    session.exitstatus = pytest.ExitCode.TESTS_FAILED


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
    """Fail a test that leaves a jammi session open.

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
