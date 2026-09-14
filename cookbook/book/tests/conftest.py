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

* :func:`_no_leaked_sessions` — the rail. An autouse guard that tracks every
  ``jammi.connect(...)`` a test makes and FAILS the test if any of those
  sessions is still open when the test returns. A seventh site that opens a
  session by hand and forgets to close it does not quietly race the cleanup
  again; it fails, by name, on the first run.

Why a *closed* session and not merely a dropped one: the embedded engine holds
its catalog through SQLite's ``unix-excl`` VFS, so dropping the handle releases
nothing at any bounded moment — ``close()`` is the only awaited release (see
``jammi.EmbeddedBackend.close``). A live engine re-creates catalog/journal files
inside its directory, so a ``shutil.rmtree`` racing it fails with ``OSError:
[Errno 39] Directory not empty`` between its ``scandir`` and its ``rmdir``.

Known limit, stated rather than assumed: the guard tracks the ``jammi.connect``
*module attribute*, so a test that bound ``from jammi import connect`` at import
time would slip past it. ``test_session_alias_gate.py`` is the standing gate for
that one precondition, over every ``.py`` module directly under this ``tests/``
directory (including this conftest and the gate module itself). This runtime
guard, plus that alias gate, is the sole rail for every session shape in this
suite's pytest lanes; neither reaches the non-pytest lanes (scripts, recipes,
quickstart, the executed chapter cells) — a static gate over those is filed as
issue #539.
"""

from __future__ import annotations

import functools
from typing import Any

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

    Autouse fixtures are set up before the test's own fixtures and finalized
    after them, so this runs its check *after* :func:`embedded` / :func:`remote`
    have closed theirs — the two mechanisms compose rather than collide.
    """
    if jammi is None:  # pragma: no cover - lean install
        yield
        return

    # A leak inside an already-failing test is noise on top of the true cause, so
    # the guard warns there and fails only when the test itself passed.
    # `session.testsfailed` is pytest's own running count and needs no hook.
    failed_before = request.session.testsfailed
    opened: list[tuple[Any, str]] = []  # strong refs: ids must not be recycled
    closed_ids: set[int] = set()
    patched: dict[type, Any] = {}
    real_connect = jammi.connect

    def _probe_close(cls: type) -> None:
        if cls in patched:
            return
        original = cls.close
        patched[cls] = original

        @functools.wraps(original)
        def _close(self, *args, **kwargs):
            closed_ids.add(id(self))
            return original(self, *args, **kwargs)

        cls.close = _close  # type: ignore[method-assign]

    @functools.wraps(real_connect)
    def _tracking_connect(target, *args, **kwargs):
        session = real_connect(target, *args, **kwargs)
        _probe_close(type(session))
        opened.append((session, str(target)))
        return session

    jammi.connect = _tracking_connect  # type: ignore[assignment]
    try:
        yield
    finally:
        jammi.connect = real_connect  # type: ignore[assignment]
        leaked = [(s, t) for s, t in opened if id(s) not in closed_ids]
        # Close them regardless: an unreleased catalog is this process's to hold,
        # and leaving it open would let one test's defect surface as another
        # test's failure.
        for session, _ in leaked:
            for cls, original in patched.items():
                if isinstance(session, cls):
                    try:
                        original(session)
                    except Exception:  # noqa: BLE001 - best-effort release
                        pass
                    break
        for cls, original in patched.items():
            cls.close = original  # type: ignore[method-assign]

        if leaked:
            targets = ", ".join(sorted(t for _, t in leaked))
            message = (
                f"{request.node.nodeid} left {len(leaked)} jammi session(s) open: "
                f"{targets}. Close every session in the test (or take one from the "
                "`embedded` / `remote` fixture): an embedded engine holds its catalog "
                "until close() returns, so a directory removed under a live session "
                "races it (OSError: [Errno 39] Directory not empty)."
            )
            if request.session.testsfailed > failed_before:
                # The test already failed; do not bury its cause under this one.
                request.node.warn(pytest.PytestWarning(message))
            else:
                pytest.fail(message, pytrace=False)
