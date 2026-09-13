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

Known limit, stated rather than assumed (in addition to the module-crossing one
above): the taint tracker is INTRA-procedural — it does not follow a
``TemporaryDirectory`` passed as a function PARAMETER into a helper that opens
``tempfile.mkdtemp(dir=that_parameter)`` several calls deep (this is exactly the
shape ``build_recompute_cache.py`` had: `emit`'s ``work_root`` flows through
`run_cache`/`_fresh_chain` as `catalog_root` before the `mkdtemp` call that
actually derives the catalog directory). That file's four connect sites were
found by hand, not by this gate, and are fixed directly (each `db.close()`d
before its enclosing `TemporaryDirectory` can unwind) rather than papered over
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
# paren. The `with jammi.connect(…) as handle` form needs no match:
# `Session.__exit__` closes, and it unwinds BEFORE an outer `TemporaryDirectory`
# in the same `with` statement.
_CONNECT = re.compile(r"(\w+)\s*=\s*jammi\.connect\([^)]*?f[\"']file://\{([^}]+)\}")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _offending_sites(text: str) -> list[tuple[int, str, str]]:
    """Sites in one file where an embedded engine is opened on a temp directory
    and never closed. Returns (line number, handle, directory name).

    The taint is scoped by INDENTATION, the way the block that removes the
    directory is: a name bound by `with TemporaryDirectory() as X` is tainted
    only until the file dedents back past that `with`. A same-named
    `X = tempfile.mkdtemp()` in another function is therefore not confused with
    it — that is a leak, but nothing removes its directory, so it is not this
    race and this gate does not speak to it.
    """
    sites: list[tuple[int, str, str]] = []
    tainted: dict[str, int] = {}  # name -> indent of the `with` that binds it
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.strip():
            here = _indent(line)
            tainted = {n: i for n, i in tainted.items() if here > i}

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
        if re.search(rf"\b{re.escape(handle)}\.close\(", text):
            continue
        sites.append((lineno, handle, base))
    return sites


def test_no_embedded_engine_outlives_its_temporary_directory():
    """No file in the cookbook opens an embedded engine on a `TemporaryDirectory`
    without closing it — the defect class, enumerated rather than remembered."""
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
    """The gate is not vacuous: the pre-fix shape is flagged, the fixed shape is
    not, and the close-in-a-finally form counts as closed."""
    bad = (
        'with tempfile.TemporaryDirectory() as d:\n'
        '    embedded = jammi.connect(f"file://{d}")\n'
        '    embedded.set_tenant("x")\n'
    )
    aliased = (
        'with tempfile.TemporaryDirectory() as tmp:\n'
        '    tmp_path = Path(tmp)\n'
        '    db = jammi.connect(f"file://{str(tmp_path)}")\n'
    )
    fixed = bad + '    embedded.close()\n'
    context_form = (
        'with tempfile.TemporaryDirectory() as d, jammi.connect(f"file://{d}") as db:\n'
        '    db.set_tenant("x")\n'
    )
    not_a_tempdir = 'db = jammi.connect(f"file://{ARTIFACT_DIR}")\n'
    # an emit script's `--target` override: the f-string is not the first token
    # after the open paren (real defect found in build_cdc_cache.py et al.).
    fallback_target = (
        'with tempfile.TemporaryDirectory() as catalog:\n'
        '    db = jammi.connect(args.target or f"file://{catalog}")\n'
    )
    fallback_target_fixed = fallback_target + '    db.close()\n'
    other_scope = (
        'def a():\n'
        '    with tempfile.TemporaryDirectory() as catalog:\n'
        '        db = jammi.connect(f"file://{catalog}")\n'
        '        db.close()\n'
        'def b():\n'
        '    catalog = tempfile.mkdtemp()\n'
        '    other = jammi.connect(f"file://{catalog}")\n'
    )

    assert [s[1] for s in _offending_sites(bad)] == ["embedded"]
    assert [s[1] for s in _offending_sites(aliased)] == ["db"]
    assert _offending_sites(fixed) == []
    assert _offending_sites(context_form) == []
    assert _offending_sites(not_a_tempdir) == []
    assert [s[1] for s in _offending_sites(fallback_target)] == ["db"]
    assert _offending_sites(fallback_target_fixed) == []
    # `b`'s `catalog` is a different, never-removed directory: not this race.
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
