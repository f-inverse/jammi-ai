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

* :func:`_no_leaked_sessions` — the rail. An autouse guard that holds a
  ``jammi.SessionWindow`` open for the duration of a test and fails it BY NAME
  if any session registered during the test never unregistered, or was closed
  after the directory it lives in was removed. The client's registry fires from
  inside the ``__init__`` of every resource-owning session class and from
  ``close()`` — the ONE seam every construction route passes through — so the
  window is independent of how a test bound the name it called through, and it
  sees a session that was NEVER closed even after the object itself is
  garbage-collected: a session dropped by refcount inside the test body is
  reclaimed before any teardown code runs, so a liveness snapshot taken at
  teardown would never see it open at all.

Why a *closed* session and not merely a dropped one: the embedded engine holds
its catalog through SQLite's ``unix-excl`` VFS, so dropping the handle releases
nothing at any bounded moment — ``close()`` is the only awaited release (see
``jammi.EmbeddedBackend.close``). A live engine re-creates catalog/journal files
inside its directory, so a ``shutil.rmtree`` racing it fails with ``OSError:
[Errno 39] Directory not empty`` between its ``scandir`` and its ``rmdir``.

The per-test window covers a single test's own fixture window, and **the suite
rule is that a test-opened session is closed by that SAME test**: a session
opened in test A and closed in test B is reported against A (the opener), never
against B, because A's window closes with the session still registered and B's
window never saw it open. A session registered outside every per-test window —
at import time, or in a module- or session-scoped fixture's own setup — is
covered by a second, process-wide window opened at ``pytest_sessionstart``
(before collection) and judged at ``pytest_sessionfinish``, which fails the
WHOLE RUN by label. The per-test guard attributes a leak to the exact test that
caused it, which the run-wide sweep cannot do; the two are complementary.

The non-pytest lanes (scripts, recipes, quickstart, the executed chapter cells)
run under ``python -m jammi.session_journal``, which judges the same two
verdicts across every process a lane starts.
"""

from __future__ import annotations

import jammi
import pytest

# Enables the `pytester` fixture: `test_session_lifecycle_guard.py` runs a
# throwaway pytest session, through THIS conftest, in a subprocess — the
# non-vacuity control for the rail below.
pytest_plugins = ["pytester"]


# --------------------------------------------------------------------------- #
# the affordance
# --------------------------------------------------------------------------- #


@pytest.fixture
def embedded(tmp_path):
    """A connected embedded engine on ``tmp_path``, closed in teardown.

    A test that also needs the catalog directory (to write a parquet beside it,
    say) asks for ``tmp_path`` as well — it is the same directory.
    """
    with jammi.connect(f"file://{tmp_path}") as db:
        yield db


@pytest.fixture
def remote():
    """A remote session against the conventional local endpoint, closed in
    teardown. Opening it dials nothing — the gRPC channel is lazy — so this is
    hermetic and needs no server."""
    with jammi.connect("grpc://127.0.0.1:8081") as db:
        yield db


# --------------------------------------------------------------------------- #
# the rail
# --------------------------------------------------------------------------- #

# The process-wide window for the `pytest_sessionfinish` sweep below. A module
# global, not fixture state — `pytest_sessionstart` fires before
# `pytest_collection`, so a window opened there sees even a module-import-time
# `jammi.connect(...)`, which no fixture — even a session-scoped one — can,
# since collection/import happens before any fixture's setup runs.
_run_window = jammi.SessionWindow()


def pytest_sessionstart(session: pytest.Session) -> None:  # noqa: ARG001
    _run_window.open()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """The process-wide sweep. Runs after every test has finished, so it cannot
    attribute a leak to one test the way the per-test guard does — it can only
    say the run, as a whole, leaked."""
    _run_window.close()
    messages = _window_messages(
        _run_window,
        "SESSION-WIDE LEAK: outside any single test's own fixture window (at import "
        "time, or in a module-/session-scoped fixture's own setup), this run",
    )
    # stdout, not stderr: pytest's own report machinery, and the rail's
    # pytester-based non-vacuity test, both read the process's stdout stream.
    for message in messages:
        print(f"\n{message}\n")
    if messages:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def _window_messages(window: jammi.SessionWindow, where: str) -> list[str]:
    """One message per verdict a closed `SessionWindow` holds, each naming its
    sessions by label. `where` is the sentence's subject: who left them."""
    messages = []
    leaked = window.leaked()
    if leaked:
        messages.append(
            f"{where} left {len(leaked)} jammi session(s) open: "
            f"{jammi.describe_sessions(leaked)}. Close every session in the test (or take "
            "one from the `embedded` / `remote` fixture): an embedded engine holds its "
            "catalog until close() returns, so a directory removed under a live session "
            "races it (OSError: [Errno 39] Directory not empty)."
        )
    late = window.closed_after_removal()
    if late:
        messages.append(
            f"{where} closed {len(late)} jammi session(s) after their directory "
            f"was removed: {jammi.describe_sessions(late)}. Close the session inside the "
            "scope that owns its directory."
        )
    return messages


@pytest.fixture(autouse=True)
def _no_leaked_sessions(request):
    """Fail a test that leaves a jammi session open, or closes one after its
    directory is gone."""
    # A leak inside an already-failing test is noise on top of the true cause, so
    # the guard warns there and fails only when the test itself passed.
    # `session.testsfailed` is pytest's own running count and needs no hook.
    failed_before = request.session.testsfailed
    window = jammi.SessionWindow().open()
    try:
        yield
    finally:
        window.close()
        message = "\n".join(_window_messages(window, request.node.nodeid))
        if message and request.session.testsfailed > failed_before:
            # The test already failed; do not bury its cause under this one.
            request.node.warn(pytest.PytestWarning(message))
        elif message:
            pytest.fail(message, pytrace=False)
