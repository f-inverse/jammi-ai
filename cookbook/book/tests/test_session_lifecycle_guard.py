"""The runtime rail's own non-vacuity test for ``conftest.py``'s
``_no_leaked_sessions`` autouse guard.

The session-lifecycle rail had a static half too — an enumerating textual
shape gate over `cookbook/**` — but a third audit found its with-item-only
close oracle unsound on several with-item shapes it could not tell from a
real close, and the real control-flow analysis that would fix it soundly is
filed as its own unit (issue #539); the static gate was excised rather than
narrowed again. The runtime guard here is what remains, and nothing committed
proved that it bites: that a test which actually leaves a session open is
actually failed, BY NAME, at teardown — on both transports. This file is that
proof, run as a real pytest session rather than a unit test of a helper
function, because the guard's own subject is "how does the outer pytest run
report a leak" — there is no smaller unit that exercises it honestly.

The throwaway suite runs against the REAL ``conftest.py`` next to this file
(read from disk, not reimplemented), in an isolated subprocess (`pytester`,
enabled in that conftest via ``pytest_plugins = ["pytester"]``): the guard
monkeypatches ``jammi.connect`` and ``<backend class>.close`` for the
DURATION of one test, so running it out-of-process is what lets this file's
own test collection stay unaffected by that patching.

A leaked session needs no live counterpart to prove the point: the embedded
arm holds a real catalog on ``tmp_path`` (never removed here — no assertion
in this file depends on a race with `rmtree`, only on the guard's own
teardown report), and the remote arm's gRPC channel is LAZY (documented in
`conftest.py`'s own ``remote`` fixture) — connecting and never closing it
needs no server, and is exactly the shape the guard exists to catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_CONFTEST = (Path(__file__).resolve().parent / "conftest.py").read_text()

_THROWAWAY_SUITE = '''
import jammi

def test_leaked_embedded_session_is_failed_by_name(tmp_path):
    """Opens an embedded session by hand and never closes it."""
    jammi.connect(f"file://{tmp_path}")

def test_leaked_remote_session_is_failed_by_name():
    """Opens a remote session by hand and never closes it. The channel is
    lazy (conftest.py's own `remote` fixture docstring) so no server is
    needed for the leak itself to be real."""
    jammi.connect("grpc://127.0.0.1:8081")

def test_closing_control_passes(embedded, remote):
    """The positive control: both transports, taken from the shared
    fixtures, which close in their own finalizers. Proves the guard does not
    fail a test for merely having opened a session."""
    assert embedded.list_sources() == []
    assert remote is not None
'''


def test_leaked_session_fails_by_name_on_both_transports(pytester: pytest.Pytester):
    """The runtime guard's non-vacuity control: a leak on EACH transport is
    reported by name at teardown, and a test that closes properly still
    passes.

    The AUTOUSE guard's `pytest.fail()` runs in a FIXTURE FINALIZER, so
    pytest's own outcome taxonomy buckets it as an "error" (at teardown), not
    a "failed" (at call) -- the test BODY genuinely passed; what failed is the
    fixture that checks its exit state. That is a taxonomy distinction, not a
    lesser guarantee: the run's exit code is still non-zero (CI still fails),
    the short summary still names the exact node id under `ERROR`, and the
    guard's own message still carries the transport target -- asserted below,
    not assumed from the category label.
    """
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(test_the_throwaway_suite=_THROWAWAY_SUITE)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

    # Each leaked test's CALL phase passes (nothing about opening a session is
    # itself wrong); each also produces exactly one teardown ERROR. The
    # closing control is a real, separate PASS -- not credited by omission.
    result.assert_outcomes(passed=3, errors=2, failed=0)
    assert result.ret != 0, "a leaked session must fail the run (non-zero exit)"

    result.stdout.fnmatch_lines(
        ["*ERROR*test_leaked_embedded_session_is_failed_by_name*"]
    )
    result.stdout.fnmatch_lines(
        ["*ERROR*test_leaked_remote_session_is_failed_by_name*"]
    )
    # The two leak messages name the transport target, not just "a session":
    # `request.node.nodeid left 1 jammi session(s) open: file://... /
    # grpc://127.0.0.1:8081` (conftest.py's own message format).
    full = "\n".join(result.outlines)
    assert full.count("left 1 jammi session(s) open") == 2
    assert "grpc://127.0.0.1:8081" in full


def test_leaked_session_reports_exactly_once_by_name(pytester: pytest.Pytester):
    """One leak produces exactly ONE reported problem attributed to the exact
    node id -- not buried under a second, vaguer report (a bare unraisable-
    exception warning, or a duplicate failure from double-teardown)."""
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(
        test_the_throwaway_suite='''
import jammi

def test_leaked_embedded_session_is_failed_by_name(tmp_path):
    jammi.connect(f"file://{tmp_path}")
'''
    )

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

    result.assert_outcomes(passed=1, errors=1, failed=0)
    full = "\n".join(result.outlines)
    assert full.count("left 1 jammi session(s) open") == 1
    assert full.count("test_leaked_embedded_session_is_failed_by_name") >= 1
