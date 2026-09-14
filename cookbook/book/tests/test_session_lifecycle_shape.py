"""The static half of the session-lifecycle rail — the whole cookbook tree.

``conftest.py``'s runtime guard fails any TEST that leaves a session open, but it
only runs where pytest runs. The same defect lives in the emit scripts, the
recipes and the quickstart, which execute in other lanes (the book gate's
API-reference step, the nightly emits, the recipe smoke). This file is the
enumerating gate for that: it reads every ``.py`` / ``.qmd`` under ``cookbook/``
and fails, BY NAME, on the shape that caused the defect —

    an embedded engine opened on a directory that a ``tempfile.TemporaryDirectory``
    will remove, with no ``close()`` on that handle anywhere in the file.

Why that shape and not merely "a leak": a live embedded engine holds and keeps
writing its catalog (``catalog.db``, ``catalog.db-wal``, and — measured — a
``catalog.db-journal`` created *during* the cleanup by the background training
worker's poll). ``shutil.rmtree`` is scandir -> unlink -> rmdir, so a file the
engine creates after the scan makes the final ``rmdir`` fail with ``ENOTEMPTY``
(``Errno 39`` on Linux, ``66`` on macOS). ``close()`` is the engine's only
bounded release point — dropping the handle releases nothing at any bounded
moment (``jammi.EmbeddedBackend.close``), so "it goes out of scope" is not a fix.

This is a SHAPE gate, and says so: it reasons about names in one file, not about
reachability. It cannot see a handle closed in another module, and it does not
flag an engine opened on a directory nothing removes (a leak, but not a race).
The runtime guard in ``conftest.py`` is the precise oracle for the tests; this is
the net for the lanes pytest never enters.

The close oracle is WITH-ITEM-ONLY: a `handle = jammi.connect(...)` ASSIGNMENT
tainted by an enclosing `TemporaryDirectory` is an OFFENDER, full stop — no
`close()` call anywhere in the file earns it credit, including an
unconditional `try:/finally: handle.close()` immediately guarding it. The
only two shapes that are not offenders are the two `_CONNECT` never matches as
an assignment in the first place: a connect that is itself a `with`-item of
the SAME `with` statement as the `TemporaryDirectory` (`with
TemporaryDirectory() as d, jammi.connect(f"file://{d}") as db:`), and a nested
`with jammi.connect(...) as handle:` whose block lies inside the tempdir
block. In both, `Session.__exit__` closes the handle, and `with`-items unwind
in reverse, before the tempdir. This is strictly NARROWER than a close oracle
that credits a `finally:` clause: a real `finally:` close does run on every
exit path, but a purely textual search inside a `finally:` block cannot tell
an unconditional `handle.close()` from one inside an `if`, a sibling `finally`
after a connect-in-a-loop, a comment, a nested `def`, or a close on a
*different* name that merely aliases the tainted handle (`ctx.db.close()` for
handle `db`) — telling those apart needs a real control-flow walk, and this is
a line-based shape gate, not one. Rather than keep widening a textual
`finally` search one audit at a time, this gate credits nothing inside a
`finally:` and relies on the site fix (the `with`-item shape) instead; the
control-flow analysis that would let a `finally:` close count safely is filed
as its own unit, issue #539, rather than attempted here.

Nine more connect/alias shapes are invisible to ``_CONNECT`` / ``_ALIAS``
altogether, so a site using them would be invisible to the whole-file scan
regardless of the close oracle above:

- the URL bound to a variable first (`url = f"file://{d}"; jammi.connect(url)`)
- a helper function that returns an open session
- string concatenation instead of an f-string (`"file://" + str(d)`)
- a `tempfile.TemporaryDirectory()` OBJECT bound to a name, with its `.name`
  attribute read at the connect site (rather than the `as NAME` form binding
  the path directly)
- `contextlib.ExitStack().enter_context(tempfile.TemporaryDirectory())`
- a tuple-unpack alias (`(tmp, other) = (d, 1); jammi.connect(f"file://{tmp}")`)
- a catalog opened in a freshly-named SUBDIRECTORY of the temp directory
  (`sub = f"{d}/nested"; jammi.connect(f"file://{sub}")` — `_ALIAS` matches a
  bare name or `Path(...)`/`str(...)` wrapper, not an f-string expression)
- a multi-line `jammi.connect(\n    ...\n)` call — `_CONNECT` matches a SINGLE
  line, so a connect whose `f"file://{...}"` argument sits on a different
  physical line than the `jammi.connect(` token is invisible to it
- a `TemporaryDirectory(...)` with-item that itself spans lines inside a
  parenthesized `with (` header, where the closing `) as NAME` is not on the
  same physical line as `TemporaryDirectory(` — `_TEMPDIR` matches a single
  line and needs the `) as NAME` tail on that same line

None of these nine shapes exists in `cookbook/**` today (swept by hand: every
`jammi.connect` call site with a `file://` target uses the inline, single-line
f-string form this gate already sees; the remaining variable-target connects
are `--target` CLI overrides whose defaults are a `grpc://` endpoint or a
fixed, never-removed path, neither of which this gate needs to see). A
reviewer introducing one of them should not rely on this test to catch it.

A further class is out of this gate's scope BY CONSTRUCTION, not by omission:
an engine opened on a plain `tempfile.mkdtemp()` that nothing ever removes —
an executed chapter cell's own artifact directory, or a `LiveServer` fixture's
artifact directory, roughly twenty sites across `cookbook/**` — is a HELD
catalog, not a directory-removal race; `shutil.rmtree` never runs on it, so
`ENOTEMPTY` cannot occur there. This gate does not, and should not, flag it.

Known limit, stated rather than assumed (in addition to the module-crossing one
above): the taint tracker is INTRA-procedural — it does not follow a
``TemporaryDirectory`` passed as a function PARAMETER into a helper that opens
``tempfile.mkdtemp(dir=that_parameter)`` several calls deep (this is exactly the
shape ``build_recompute_cache.py`` had: `emit`'s ``work_root`` flows through
`run_cache`/`_fresh_chain` as `catalog_root` before the `mkdtemp` call that
actually derives the catalog directory). That file's two `jammi.connect(...)`
call sites are each exercised through several callers — `_fresh_chain` (five
callers) and `run_cache` (one) — for six connect-opening call sites in total,
found by hand, not by this gate, and fixed directly (each `db.close()`d before
its enclosing `TemporaryDirectory` can unwind) rather than papered over
by widening the regex into a real dataflow analysis, which is disproportionate
for a line-based shape gate. A reviewer adding a NEW multi-hop
`TemporaryDirectory` -> `mkdtemp(dir=...)` -> `jammi.connect` chain should not
rely on this test to catch it.
"""

from __future__ import annotations

import re
from pathlib import Path

_COOKBOOK = Path(__file__).resolve().parents[2]

# `with tempfile.TemporaryDirectory(...) as NAME` — NAME's directory is removed
# by rmtree at the end of that block, and only inside it.
_TEMPDIR = re.compile(r"TemporaryDirectory\([^)]*\)\s+as\s+(\w+)")
# `X = <tainted>` / `Path(<tainted>)` / `str(<tainted>)` — the alias forms the
# recipes use (`tmp_path = Path(tmp)`).
_ALIAS = re.compile(r"^\s*(\w+)\s*=\s*(?:Path|str)?\(?\s*(\w+)\s*\)?\s*$")
# `handle = jammi.connect(f"file://{...}")`, and `handle = jammi.connect(<expr> or
# f"file://{...}")` — an emit script's `--target` override falls back to a fresh
# temp catalog this way, and the f-string is not the first token after the open
# paren. This matches the ASSIGNMENT form only: a connect that is a `with`-item
# (`with jammi.connect(...) as handle`) has no `=` before `jammi.connect` and
# is invisible to this regex from the start — `Session.__exit__` closes it, and
# it unwinds BEFORE an outer `TemporaryDirectory` in the same `with` statement.
# Under the with-item-only close oracle (E1) that is exactly the point: every
# line this DOES match is, by construction, an assignment-form connect, which
# this gate always treats as an offender when tainted — see `_offending_sites`.
_CONNECT = re.compile(r"(\w+)\s*=\s*jammi\.connect\([^)]*?f[\"']file://\{([^}]+)\}")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


# A bare `with (` opener: the parenthesized multi-line form (`with (\n    A as
# a,\n    B as b,\n):`). Its items are CONTINUATION lines -- their own
# indentation is not what governs the block Python actually opens; the `):`
# closer sits back at the `with` line's own column. A `TemporaryDirectory`
# item inside this header must be tainted at the HEADER's indent, not the
# item's, or the block body (indented past the header, same as or past the
# items) never reads as "inside" it once the generic dedent-based taint-expiry
# check is applied to the `):` closer line itself (whose indent equals the
# item's-if-mistaken column) before that body is ever reached.
_PAREN_WITH_OPEN = re.compile(r"^with\s*\(\s*$")
_PAREN_WITH_CLOSE = re.compile(r"^\)\s*:")
# A malformed/unrecognized header should not blind the scan for the rest of
# the file: bail out of "inside a parenthesized header" bookkeeping past this
# many lines without a closer (every real site in this tree closes within 6).
_PAREN_WITH_MAX_LINES = 20


def _offending_sites(text: str) -> list[tuple[int, str, str]]:
    """Sites in one file where an embedded engine is opened on a temp directory
    and never closed ON EVERY EXIT PATH. Returns (line number, handle, directory
    name).

    The taint is scoped by INDENTATION, the way the block that removes the
    directory is: a name bound by `with TemporaryDirectory() as X` is tainted
    only until the file dedents back past that `with`. A same-named
    `X = tempfile.mkdtemp()` in another function is therefore not confused with
    it — that is a leak, but nothing removes its directory, so it is not this
    race and this gate does not speak to it.

    The parenthesized multi-line `with (...)：` form is parsed as ONE
    statement whose block indent is the `with` line's own column -- not the
    indent of whichever item line a `TemporaryDirectory(...)` happens to sit
    on -- so a `TemporaryDirectory` opened as a with-item there taints the
    body at the right scope even when a later edit turns its sibling connect
    item into a bare, un-closed assignment statement.
    """
    lines = text.splitlines()
    sites: list[tuple[int, str, str]] = []
    tainted: dict[str, int] = {}  # name -> indent of the `with` that binds it
    in_paren_header = False
    paren_header_indent = 0
    paren_header_started_at = 0
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        here = _indent(line) if stripped else None

        if in_paren_header:
            # Continuation lines of a parenthesized `with (`: no dedent-based
            # taint expiry (their own indent does not govern the block), and
            # only a `TemporaryDirectory` with-item is looked for -- a
            # `= jammi.connect(...)` assignment cannot appear here (a
            # with-item is `EXPR as NAME`, never an `=` binding).
            temp = _TEMPDIR.search(line)
            if temp:
                tainted[temp.group(1)] = paren_header_indent
            if stripped and _PAREN_WITH_CLOSE.match(stripped) and here <= paren_header_indent:
                in_paren_header = False
            elif lineno - paren_header_started_at > _PAREN_WITH_MAX_LINES:
                in_paren_header = False  # malformed/unrecognized: stop guessing
            continue

        if stripped:
            tainted = {n: i for n, i in tainted.items() if here > i}

        if _PAREN_WITH_OPEN.match(stripped):
            in_paren_header = True
            paren_header_indent = here
            paren_header_started_at = lineno
            continue

        temp = _TEMPDIR.search(line)
        if temp:
            tainted[temp.group(1)] = _indent(line)
        alias = _ALIAS.match(line)
        if alias and alias.group(2) in tainted:
            tainted[alias.group(1)] = tainted[alias.group(2)]

        found = _CONNECT.search(line)
        if not found:
            continue
        handle, arg = found.group(1), found.group(2)
        base = re.sub(r"^(?:str|Path)\(", "", arg).split(")")[0].split("/")[0].strip()
        if base not in tainted:
            continue  # the catalog is not inside a directory being removed
        # `_CONNECT` only ever matches the ASSIGNMENT form (`handle =
        # jammi.connect(...)`); the with-item and nested-with forms never
        # reach here at all (see `_CONNECT`'s comment). Under the
        # with-item-only close oracle (E1), an assignment-form connect on a
        # tainted directory is ALWAYS an offender — no `close()` call
        # anywhere in the file, including one in a `finally:`, earns credit.
        sites.append((lineno, handle, base))
    return sites


def test_no_embedded_engine_outlives_its_temporary_directory():
    """esc-112 fix test (`closes_escape: esc-112`): no file in the cookbook opens
    an embedded engine on a `TemporaryDirectory` without closing it ON EVERY EXIT
    PATH — the defect class, enumerated rather than remembered."""
    offenders = []
    for path in sorted(_COOKBOOK.rglob("*.py")) + sorted(_COOKBOOK.rglob("*.qmd")):
        for lineno, handle, directory in _offending_sites(
            path.read_text(errors="replace")
        ):
            offenders.append(
                f"{path.relative_to(_COOKBOOK.parent)}:{lineno}: `{handle}` is opened on "
                f"`{directory}` (a TemporaryDirectory) and never closed"
            )
    assert not offenders, (
        "an embedded engine outlives the directory rmtree is about to remove "
        "(ENOTEMPTY: Errno 39 on Linux / 66 on macOS):\n  "
        + "\n  ".join(offenders)
        + "\n\nClose the session before the directory is removed. In a script, put the "
        "connect ON the same `with` statement (`with TemporaryDirectory() as d, "
        "jammi.connect(f\'file://{d}\') as db:` — `Session.__exit__` closes, and the "
        "items unwind in reverse). In a test, take the `embedded` fixture."
    )


def test_the_gate_sees_the_shape_it_claims_to_see():
    """The gate is not vacuous: the pre-fix shape is flagged, and the close
    oracle is WITH-ITEM-ONLY (E1) — each shape is its own positive or negative
    control, not folded into one assertion."""
    bad = (
        'with tempfile.TemporaryDirectory() as d:\n'
        '    embedded = jammi.connect(f"file://{d}")\n'
        '    embedded.set_tenant("x")\n'
    )

    # E1 self-test 1/5: happy-path-only close -> RED. A plain statement after
    # the connect, with no enclosing `try`, is an assignment-form connect on a
    # tainted directory -- always an offender under the with-item-only oracle.
    happy_path_only_close = bad + '    embedded.close()\n'
    assert [s[1] for s in _offending_sites(happy_path_only_close)] == ["embedded"]

    # E1 self-test 2/5: close in an unrelated function -> RED. `handle` is a
    # same-named `close()` call in a DIFFERENT function's `try/finally`, well
    # past this block's own dedent -- it cannot run when the tainted block
    # unwinds and must not be credited to it (and would not be, under the
    # with-item-only oracle, even if it could).
    close_in_unrelated_function = (
        'def a():\n'
        '    with tempfile.TemporaryDirectory() as d:\n'
        '        handle = jammi.connect(f"file://{d}")\n'
        '        handle.set_tenant("x")\n'
        'def b():\n'
        '    try:\n'
        '        pass\n'
        '    finally:\n'
        '        handle.close()\n'
    )
    assert [s[1] for s in _offending_sites(close_in_unrelated_function)] == ["handle"]

    # E1 self-test 3/5: `finally:` close -> RED (the R1 `finally:` -> GREEN
    # case, excised). The connect is an assignment tainted by the enclosing
    # `TemporaryDirectory`; the with-item-only oracle credits nothing inside a
    # `finally:` -- an unconditional, textually unambiguous `handle.close()`
    # guarding the connect's only use is still an offender, because the
    # oracle can no longer distinguish it, in general, from a close under an
    # `if`, a sibling `finally` after a connect-in-a-loop, a comment, a nested
    # `def`, or a close on an aliasing name (`ctx.db.close()` for handle
    # `db`) -- see the module docstring and issue #539.
    finally_close = (
        'with tempfile.TemporaryDirectory() as d:\n'
        '    try:\n'
        '        embedded = jammi.connect(f"file://{d}")\n'
        '        embedded.set_tenant("x")\n'
        '    finally:\n'
        '        embedded.close()\n'
    )
    assert [s[1] for s in _offending_sites(finally_close)] == ["embedded"]

    # E1 self-test 4/5: `with`-item -> GREEN. `_CONNECT` only matches the
    # `handle = jammi.connect(...)` assignment form, so a connect that is
    # itself a `with`-item is invisible to the offender scan from the start;
    # `Session.__exit__` closes it before the tempdir unwinds either way.
    with_item = (
        'with tempfile.TemporaryDirectory() as d, jammi.connect(f"file://{d}") as db:\n'
        '    db.set_tenant("x")\n'
    )
    assert _offending_sites(with_item) == []

    # E1 self-test 5/5: nested `with ... as handle:` -> GREEN. Same reasoning
    # as the with-item form: no `=` before `jammi.connect`, so `_CONNECT`
    # never matches, and the nested block closes before the outer one unwinds.
    nested_with = (
        'with tempfile.TemporaryDirectory() as d:\n'
        '    with jammi.connect(f"file://{d}") as handle:\n'
        '        handle.set_tenant("x")\n'
    )
    assert _offending_sites(nested_with) == []

    # R1 addendum (a hole the static gate itself missed on the first fix
    # round's own output): a PARENTHESIZED multi-line `with (` header is one
    # statement whose block indent is the `with` line's own column, not
    # whichever item line a `TemporaryDirectory` happens to sit on. Exact
    # regression shape: `build_segmented_ann_cache.py`'s with-item connect
    # replaced by a bare assignment as the first body statement, no close --
    # RED. A multi-line header where BOTH items stay with-items -- GREEN.
    paren_with_regressed_to_bare_assignment = (
        'def emit(fixtures_root):\n'
        '    with (\n'
        '        tempfile.TemporaryDirectory(prefix="jammi_segmented_ann_") as artifact_dir,\n'
        '    ):\n'
        '        db = jammi.connect(f"file://{artifact_dir}")\n'
        '        db.set_tenant("x")\n'
    )
    assert [s[1] for s in _offending_sites(paren_with_regressed_to_bare_assignment)] == ["db"]

    paren_with_item_control = (
        'def emit(fixtures_root):\n'
        '    with (\n'
        '        tempfile.TemporaryDirectory(prefix="jammi_segmented_ann_") as artifact_dir,\n'
        '        jammi.connect(f"file://{artifact_dir}") as db,\n'
        '    ):\n'
        '        db.set_tenant("x")\n'
    )
    assert _offending_sites(paren_with_item_control) == []

    # Retained regressions from the taint tracker itself (aliasing, the
    # `--target` fallback shape, directory taint, and cross-function scope
    # isolation) -- unaffected by, or updated for, the E1 close oracle.
    aliased = (
        'with tempfile.TemporaryDirectory() as tmp:\n'
        '    tmp_path = Path(tmp)\n'
        '    db = jammi.connect(f"file://{str(tmp_path)}")\n'
    )
    not_a_tempdir = 'db = jammi.connect(f"file://{ARTIFACT_DIR}")\n'
    # an emit script's `--target` override: the f-string is not the first token
    # after the open paren (real defect found in build_cdc_cache.py et al.).
    fallback_target = (
        'with tempfile.TemporaryDirectory() as catalog:\n'
        '    db = jammi.connect(args.target or f"file://{catalog}")\n'
    )
    # The actual site fix (E2): the connect moves onto the SAME `with`
    # statement as the tempdir, as a with-item -- GREEN, unlike a
    # `try:/finally: db.close()` guarding the assignment form, which stays RED
    # under the with-item-only oracle (see `finally_close` above).
    fallback_target_with_item = (
        'with tempfile.TemporaryDirectory() as catalog, jammi.connect(args.target or '
        'f"file://{catalog}") as db:\n'
        '    db.set_tenant("x")\n'
    )
    other_scope = (
        'def a():\n'
        '    with tempfile.TemporaryDirectory() as catalog, '
        'jammi.connect(f"file://{catalog}") as db:\n'
        '        db.set_tenant("x")\n'
        'def b():\n'
        '    catalog = tempfile.mkdtemp()\n'
        '    other = jammi.connect(f"file://{catalog}")\n'
    )

    assert [s[1] for s in _offending_sites(aliased)] == ["db"]
    assert _offending_sites(not_a_tempdir) == []
    assert [s[1] for s in _offending_sites(fallback_target)] == ["db"]
    assert _offending_sites(fallback_target_with_item) == []
    # `a`'s `db` is a with-item, invisible to `_CONNECT`; `b`'s `catalog` is a
    # different, never-removed directory -- neither is this race.
    assert _offending_sites(other_scope) == []


def test_tests_call_connect_through_the_module():
    """No test module binds `connect` at import time.

    The runtime guard in `conftest.py` patches the `jammi.connect` module
    attribute, so a module-level `from jammi import connect` would hold a
    pre-patch reference and slip past it. Rather than leave that as an assumption
    in a docstring, it is failed here.
    """
    alias = re.compile(r"\s*from jammi import .*\bconnect\b")
    # the detector, on a positive and a negative control — a guard whose pattern
    # matched nothing at all would pass this test for the wrong reason
    assert alias.match("from jammi import connect, Capability")
    assert not alias.match("from jammi import Capability, Session")

    offenders = [
        f"{p.name}:{n}"
        for p in sorted(Path(__file__).resolve().parent.glob("*.py"))
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if alias.match(line)
    ]
    assert not offenders, (
        "these test modules alias `jammi.connect` at import time, which escapes the "
        f"conftest leak guard: {offenders}"
    )
