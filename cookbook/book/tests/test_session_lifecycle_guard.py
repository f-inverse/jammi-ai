"""The runtime rail's own non-vacuity test for ``conftest.py``'s
``_no_leaked_sessions`` autouse guard.

The runtime guard is the control for every session shape in this suite —
every transport, every construction route, and every way a test's code could
have bound the name it called through. This file is the committed proof that
it bites: a test which actually leaves a session open is actually failed, BY
NAME, at teardown, regardless of transport or binding shape. It runs as a
real pytest session rather than a unit test of a helper function, because
the guard's own subject is "how does the outer pytest run report a leak" —
there is no smaller unit that exercises it honestly. A static gate over the
rest of `cookbook/**` (the non-pytest lanes: scripts, recipes, quickstart,
the executed chapter cells) is filed as issue #539.

The throwaway suites run against the REAL ``conftest.py`` next to this file
(read from disk, not reimplemented), in an isolated subprocess (`pytester`,
enabled in that conftest via ``pytest_plugins = ["pytester"]``). The guard
patches no module attribute and no class method (see
`clients/python/jammi/_sessions.py` and `conftest.py`'s own docstring): it
subscribes to that module's `observe()` events for the duration of a test.
Running it out-of-process is not load-bearing for isolation on that account —
it stays because it is still the only honest way to assert what the OUTER
pytest run reports (exit code, ERROR summary, message text) rather than a
helper function's return value.

A leaked session needs no live counterpart to prove the point: the embedded
arm holds a real catalog on ``tmp_path`` (never removed here — no assertion
in this file depends on a race with `rmtree`, only on the guard's own
teardown report), and the remote arm's gRPC channel is LAZY (documented in
`conftest.py`'s own ``remote`` fixture) — connecting and never closing it
needs no server. Every alias/construction-shape test below also uses the lazy
remote target for the same reason: the shape under test is how the NAME was
bound, not which transport it opened, so there is no need to pay for a real
embedded engine per shape.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_CONFTEST = (Path(__file__).resolve().parent / "conftest.py").read_text()

_THROWAWAY_SUITE = '''
import jammi

def test_leaked_embedded_session_is_failed_by_name(tmp_path):
    """Opens an embedded session by hand and never closes it. This is the
    bare-`jammi.connect(...)`-statement shape: the returned session is bound
    to nothing at all, so CPython collects it by refcount at the end of this
    very statement, before the test body even reaches its next line -- and
    the registry guard still catches it, because it observes the register
    event as it fires, not a later liveness snapshot."""
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


# pytest truncates its short-summary "ERROR <nodeid> - Failed: <message>"
# line to the terminal width (see COLUMNS in `runpytest_subprocess`'s
# environment): a narrow terminal drops the message entirely from that
# line, a wide one shows it in full, so counting a message SUBSTRING across
# the whole run's stdout counts a different number of occurrences per
# terminal width -- the exact defect this file's cause experiment
# (COLUMNS=80 vs COLUMNS=300, documented in the commit that added these
# helpers) reproduced: 1 occurrence at 80 columns, 2 at 300. The section
# header and body pytest writes under `=== ERRORS ===` -- `ERROR at
# teardown of <nodeid>`, one per failing fixture finalizer -- are never
# truncated by width, `-r` flags or summary formatting, so keying off that
# header is the width-independent form of "each leak is reported exactly
# once, attributed to its node id".
_TEARDOWN_ERROR_HEADER_RE = re.compile(r"^_+\s+ERROR at teardown of (.+?)\s+_+$")
_BANNER_RE = re.compile(r"^=+\s.*\s=+$")


def _teardown_error_sections(outlines: list[str]) -> list[tuple[str, list[str]]]:
    """Split a pytest run's plain-text stdout into one ``(nodeid, body
    lines)`` entry per ``ERROR at teardown of <nodeid>`` section pytest
    emits under its ``=== ERRORS ===`` banner. A list, not a dict keyed by
    name: a name appearing twice (a second, duplicate report for the same
    node id) must stay visible to the caller, not silently collapse."""
    sections: list[tuple[str, list[str]]] = []
    current_lines: list[str] | None = None
    for line in outlines:
        header = _TEARDOWN_ERROR_HEADER_RE.match(line)
        if header:
            current_lines = []
            sections.append((header.group(1), current_lines))
            continue
        if current_lines is not None:
            if _BANNER_RE.match(line):
                current_lines = None
            else:
                current_lines.append(line)
    return sections


def _section_after_banner(outlines: list[str], banner_substring: str) -> list[str]:
    """Return the body lines of the FIRST pytest banner section (a line
    matching pytest's ``=== <text> ===`` banner format) whose own text
    contains ``banner_substring``, up to (not including) the next banner.
    Used the same way `_teardown_error_sections` uses `_BANNER_RE`: keyed
    off text pytest always writes, not a width-dependent truncation."""
    lines: list[str] = []
    collecting = False
    for line in outlines:
        if _BANNER_RE.match(line):
            if collecting:
                break
            collecting = banner_substring in line
            continue
        if collecting:
            lines.append(line)
    return lines


def _assert_no_second_report_class(outlines: list[str]) -> None:
    """No leak is EVER reported through a second, vaguer channel alongside
    its own ``ERROR at teardown of <nodeid>`` section: no bare
    unraisable-exception warning, no "Exception ignored" interpreter
    traceback tail, no duplicate teardown-error section for the same node
    id, and no co-occurring entry in the warnings summary (the guard's OTHER
    arm -- an already-failing test -- legitimately reports a leak only
    through the warnings summary INSTEAD of an ERROR section; a leak that
    got both is the double-report this guards against)."""
    full = "\n".join(outlines)
    assert "PytestUnraisableExceptionWarning" not in full
    assert "Exception ignored" not in full
    names = [name for name, _ in _teardown_error_sections(outlines)]
    assert len(names) == len(set(names)), (
        f"duplicate teardown-error section for the same node id: {names!r}"
    )
    warnings_body = "\n".join(_section_after_banner(outlines, "warnings summary"))
    assert "left 1 jammi session(s) open" not in warnings_body, (
        "the leak must not ALSO be reported via the warnings summary "
        f"alongside its own ERROR at teardown section: {warnings_body!r}"
    )


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
    guard's own message still carries the target label -- asserted below,
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
    # Each leak is reported exactly once, attributed to its own node id, and
    # to neither the wrong test nor a vaguer second channel -- asserted here
    # in the width-independent form (see the module-level comment above
    # `_teardown_error_sections`).
    sections = _teardown_error_sections(result.outlines)
    names = [name for name, _ in sections]
    assert sorted(names) == sorted(
        [
            "test_leaked_embedded_session_is_failed_by_name",
            "test_leaked_remote_session_is_failed_by_name",
        ]
    ), f"expected exactly one teardown-error section per leaking test, got {names!r}"
    for name, lines in sections:
        assert "left 1 jammi session(s) open" in "\n".join(lines), (name, lines)
    # The remote leak message names the target label, not just "a session":
    # the embedded label is the artifact_dir it was opened on, the remote
    # label is the endpoint `RemoteDatabase.__init__` was given (no scheme --
    # `_sessions.register` is called with `endpoint`, not the original
    # `target` string; see `_database.py`'s own `open_remote`).
    remote_body = "\n".join(
        line
        for name, lines in sections
        if name == "test_leaked_remote_session_is_failed_by_name"
        for line in lines
    )
    assert "127.0.0.1:8081" in remote_body
    _assert_no_second_report_class(result.outlines)


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
    # Exactly one `ERROR at teardown of <nodeid>` section, attributed to the
    # exact node id, carrying the leak message -- and no second, vaguer
    # report of the same leak (asserted in the width-independent form; see
    # the module-level comment above `_teardown_error_sections`).
    sections = _teardown_error_sections(result.outlines)
    names = [name for name, _ in sections]
    assert names == [
        "test_leaked_embedded_session_is_failed_by_name"
    ], f"expected exactly one teardown-error section, got {names!r}"
    assert "left 1 jammi session(s) open" in "\n".join(sections[0][1])
    _assert_no_second_report_class(result.outlines)


# --------------------------------------------------------------------------- #
# Every import-time binding shape and construction route audit #5 executed.
#
# A guard that tracks leaks by patching the `jammi.connect` MODULE ATTRIBUTE
# for the duration of a test cannot see any of these: each one binds its own
# name at IMPORT time, before any such patch could be installed, so the
# tracking wrapper is never the function actually called and the leak passes
# silently -- an enumeration problem with no upper bound on binding shapes.
# The registry guard below pins the opposite property directly: it reads no
# bound name at all, only the register/close EVENT the session's own
# `__init__`/`close()` fire, so no binding shape changes what it sees.
# --------------------------------------------------------------------------- #

_SHAPES_SUITE = '''
import importlib

import grpc

import jammi
from jammi import EmbeddedBackend, RemoteDatabase

# shape: class-body alias
class _ClassBodyHolder:
    connect = jammi.connect

# shape: try:-nested `from jammi import connect`
try:
    from jammi import connect as _try_nested_connect
except ImportError:  # pragma: no cover - jammi is always importable here
    _try_nested_connect = None

# shape: if:-nested `from jammi import connect`
if True:
    from jammi import connect as _if_nested_connect

# shape: parenthesised multi-line import
from jammi import (
    connect as _paren_multiline_connect,
)

# shape: `from jammi import *` -- binds the bare name `connect`. Captured
# immediately under its own name: a later plain `from jammi import connect`
# (below) rebinds that same bare name, and the registry guard does not care
# which shape a session was opened through -- only this suite's own
# bookkeeping needs the two kept distinguishable.
from jammi import *  # noqa: F401,F403
_star_import_connect = connect

# shape: `import jammi as j` + `connect = j.connect`
import jammi as _j_alias
_module_alias_connect = _j_alias.connect

# shape: `getattr(jammi, "connect")`
_getattr_connect = getattr(jammi, "connect")

# shape: `importlib.import_module("jammi").connect`
_importlib_connect = importlib.import_module("jammi").connect

# shape: tuple-unpack
(_tuple_unpack_connect,) = (jammi.connect,)

# shape: default-arg -- the default is evaluated ONCE, at `def` time, so this
# is an import-time binding of the underlying function even though the name
# `connect` here is a parameter, not a module global.
def _default_arg_fn(url, connect=jammi.connect):
    return connect(url)

# shape: `from jammi import connect as c`
from jammi import connect as _import_as_c

# shape: `_connect = jammi.connect`
_connect = jammi.connect

# shape: plain column-zero `from jammi import connect`
from jammi import connect


_TARGET = "grpc://127.0.0.1:8081"  # lazy channel -- no server needed to leak it


def test_shape_class_body_alias():
    _ClassBodyHolder.connect(_TARGET)

def test_shape_try_nested_import():
    _try_nested_connect(_TARGET)

def test_shape_if_nested_import():
    _if_nested_connect(_TARGET)

def test_shape_paren_multiline_import():
    _paren_multiline_connect(_TARGET)

def test_shape_star_import():
    _star_import_connect(_TARGET)

def test_shape_module_alias_then_attr():
    _module_alias_connect(_TARGET)

def test_shape_getattr():
    _getattr_connect(_TARGET)

def test_shape_importlib_import_module():
    _importlib_connect(_TARGET)

def test_shape_tuple_unpack():
    _tuple_unpack_connect(_TARGET)

def test_shape_default_arg():
    _default_arg_fn(_TARGET)

def test_shape_import_as_c():
    _import_as_c(_TARGET)

def test_shape_underscore_bare_alias():
    _connect(_TARGET)

def test_shape_plain_column_zero_import():
    connect(_TARGET)

def test_shape_local_dropped_at_frame_exit():
    """Distinct from the bare-statement shape (see the throwaway suite
    above): `db` is a real local binding, so it stays alive for the whole
    test body and is only collected when THIS FRAME's locals are torn down
    at return -- after the test body, but still before this fixture's own
    teardown runs its check."""
    db = jammi.connect(_TARGET)
    assert db is not None

def test_shape_direct_embedded_construction(tmp_path):
    """Bypasses `jammi.connect` (and `_open_embedded`) entirely: constructs
    the resource-owning class itself. Registration happens in
    `EmbeddedBackend.__init__`, so this is visible regardless."""
    import jammi_native

    native = jammi_native.open_local(artifact_dir=str(tmp_path), config=None)
    EmbeddedBackend(native, label=str(tmp_path))

def test_shape_direct_remote_construction():
    """Bypasses `jammi.connect` (and `open_remote`) entirely: constructs
    `RemoteDatabase` itself over a lazy channel."""
    channel = grpc.insecure_channel("127.0.0.1:8081")
    RemoteDatabase(
        channel,
        session_id="direct-construction-shape",
        endpoint="127.0.0.1:8081",
        tls=False,
        auth_metadata=None,
    )

def test_shape_closing_control_passes():
    """Not every one of these bound names leaks: closing through one of them
    (the plain import) still passes -- the guard does not blanket-fail every
    test in this suite, only the ones that actually leave a session open."""
    db = connect(_TARGET)
    db.close()
'''

# Every leaking test defined in `_SHAPES_SUITE`, by name -- the 13 import-time
# binding shapes from audit #5, plus the two direct-construction routes and
# the local-dropped-at-frame-exit shape (16 total). `test_shape_closing_control_passes`
# is deliberately excluded: it must NOT appear in the ERROR summary.
_LEAKING_SHAPE_TESTS = (
    "test_shape_class_body_alias",
    "test_shape_try_nested_import",
    "test_shape_if_nested_import",
    "test_shape_paren_multiline_import",
    "test_shape_star_import",
    "test_shape_module_alias_then_attr",
    "test_shape_getattr",
    "test_shape_importlib_import_module",
    "test_shape_tuple_unpack",
    "test_shape_default_arg",
    "test_shape_import_as_c",
    "test_shape_underscore_bare_alias",
    "test_shape_plain_column_zero_import",
    "test_shape_local_dropped_at_frame_exit",
    "test_shape_direct_embedded_construction",
    "test_shape_direct_remote_construction",
)


def test_every_alias_and_construction_shape_leaks_by_name(pytester: pytest.Pytester):
    """Every shape audit #5 found (13 import-time bindings), plus a local
    dropped at frame exit and both backends' direct construction, is FAILED
    BY NAME under the current registry-based guard -- and the one test that
    actually closes its session is not.

    This is the completeness claim for Z2: the registry observes the
    CONSTRUCTOR, not a name a caller happened to call through, so there is no
    binding shape left to enumerate against -- `_no_leaked_sessions` no
    longer has a "how was `connect` spelled" precondition at all.
    """
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(test_the_shapes_suite=_SHAPES_SUITE)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

    result.assert_outcomes(
        passed=len(_LEAKING_SHAPE_TESTS) + 1,  # every CALL phase passes
        errors=len(_LEAKING_SHAPE_TESTS),  # every leaking test's teardown errors
        failed=0,
    )
    assert result.ret != 0

    full = "\n".join(result.outlines)
    for name in _LEAKING_SHAPE_TESTS:
        result.stdout.fnmatch_lines([f"*ERROR*{name}*"])
        assert name in full

    # The closing control's name must NOT appear anywhere under an ERROR line.
    error_lines = [line for line in result.outlines if "ERROR" in line]
    assert not any("test_shape_closing_control_passes" in line for line in error_lines)


def test_leak_inside_an_already_failing_test_is_warned_not_failed_again(
    pytester: pytest.Pytester,
):
    """The guard's OTHER arm (`conftest.py`'s `_no_leaked_sessions`, the
    ``request.session.testsfailed > failed_before`` branch): a test that
    already failed on its own assertion and ALSO leaks a session is reported
    as exactly one `failed` -- the test's own assertion -- plus a WARNING
    naming the leaked label, never a second `error` piled on top of the true
    cause.

    The mirror case -- the identical leak inside a test that otherwise
    PASSES -- is already covered above by
    `test_leaked_session_reports_exactly_once_by_name`: there, nothing else
    failed for the guard to defer to, so the leak itself is the teardown
    `error`. Referencing rather than duplicating it keeps the two arms next
    to their one shared fixture shape.
    """
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(
        test_the_throwaway_suite='''
import jammi

def test_already_failing_test_that_also_leaks(tmp_path):
    jammi.connect(f"file://{tmp_path}")
    assert False, "already-failing"
'''
    )

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

    # One `failed` (the assertion, at CALL), no `errors` at all -- the guard's
    # own fixture finalizer took the warn branch instead of `pytest.fail`ing a
    # second time on top of an already-failing test.
    result.assert_outcomes(failed=1, errors=0, passed=0, warnings=1)
    assert result.ret != 0, "the test's own assertion must still fail the run"

    full = "\n".join(result.outlines)
    assert "left 1 jammi session(s) open" in full
    assert "test_already_failing_test_that_also_leaks" in full

    # The leak message lives under the warnings summary, as a `PytestWarning`
    # -- not under a `FAILED`/`ERROR` short-summary line of its own (that
    # would mean it was reported as a SECOND problem, not folded into a
    # warning).
    result.stdout.fnmatch_lines(["*warnings summary*"])
    warning_lines = [
        line
        for line in result.outlines
        if "left 1 jammi session(s) open" in line
    ]
    assert warning_lines, "the leak message must appear somewhere in the run"
    assert not any(
        "FAILED" in line or line.strip().startswith("ERROR")
        for line in warning_lines
    ), "the leak message must not be reported as its own FAILED/ERROR line"

    short_summary = [
        line
        for line in result.outlines
        if line.startswith("FAILED") or line.startswith("ERROR")
    ]
    assert len(short_summary) == 1, (
        "exactly one short-summary line -- the assertion's FAILED -- and "
        f"nothing else; got {short_summary!r}"
    )
    assert short_summary[0].startswith("FAILED")


def test_leaked_session_in_a_tests_subdirectory_module_is_failed_by_name(
    pytester: pytest.Pytester,
):
    """The guard reaches a module in a SUBDIRECTORY of the test root too --
    it is not scoped to `.py` files directly under one directory (that was
    the deleted static alias gate's own stated scope limit; the registry
    guard has no such limit because it is not walking files at all, it is an
    autouse fixture that applies wherever pytest collects a test)."""
    pytester.makeconftest(_CONFTEST)
    nested = pytester.mkpydir("nested_suite")
    (nested / "test_nested_leak.py").write_text(
        '''
import jammi

def test_leak_in_a_subdirectory_module():
    jammi.connect("grpc://127.0.0.1:8081")
'''
    )

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

    result.assert_outcomes(passed=1, errors=1, failed=0)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*ERROR*test_leak_in_a_subdirectory_module*"])


# --------------------------------------------------------------------------- #
# F1: capability detection -- a jammi client without the session registry.
# --------------------------------------------------------------------------- #

# A minimal stand-in for a pre-registry `jammi` client: exposes `connect` but
# no `observe` / `open_sessions`, simulating exactly what the nightly
# release-recipe leg installs (`.github/workflows/cookbook-render.yml`'s
# `pip install jammi-ai[embedded]` from PyPI, which lags HEAD by
# construction -- see conftest.py's own "Capability, not a flag" paragraph).
_FAKE_PRE_REGISTRY_JAMMI = '''
__version__ = "0.1.0-fake-pre-registry"


class _FakeSession:
    def list_sources(self):
        return []

    def close(self):
        pass


def connect(target):
    return _FakeSession()
'''

_THROWAWAY_SUITE_FOR_FAKE_CLIENT = '''
import jammi


def test_uses_embedded_fixture(embedded):
    assert embedded.list_sources() == []


def test_opens_and_closes_cleanly():
    db = jammi.connect("grpc://127.0.0.1:8081")
    db.close()


def test_leaks_but_the_rail_is_inactive():
    """With the registry absent there is nothing to observe: this leak is
    invisible by construction (the known limit conftest.py states), not by
    a bug -- the point of this test is that the run still reports the
    capability gap once, not this leak."""
    jammi.connect("grpc://127.0.0.1:8081")
'''


def test_rail_inactive_without_the_registry_warns_once_and_runs_clean(
    pytester: pytest.Pytester,
):
    """F1: a `jammi` client that predates the session registry (`.observe`
    absent) must not error every test at setup -- against the guard before
    this capability arm existed, this exact fixture (a fake `jammi` package
    with `connect` but no `observe`) failed every test's setup with
    `AttributeError: module 'jammi' has no attribute 'observe'` (the
    nightly release-recipe leg's own symptom, reproduced by execution while
    developing this fix). The capability arm detects the missing registry
    at import and yields instead of subscribing, and the session-scoped
    fixture reports the gap exactly once.
    """
    pytester.makeconftest(_CONFTEST)
    fake_jammi = pytester.mkpydir("jammi")
    (fake_jammi / "__init__.py").write_text(_FAKE_PRE_REGISTRY_JAMMI)
    pytester.makepyfile(test_the_throwaway_suite=_THROWAWAY_SUITE_FOR_FAKE_CLIENT)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

    # No setup errors: every test's CALL phase runs, including the one that
    # leaks -- the rail cannot see it, and must not crash trying.
    result.assert_outcomes(passed=3, errors=0, failed=0, warnings=1)
    assert result.ret == 0, "a client without the registry must not fail the run"

    full = "\n".join(result.outlines)
    assert full.count("session-leak rail inactive") == 1
    assert "0.1.0-fake-pre-registry" in full
