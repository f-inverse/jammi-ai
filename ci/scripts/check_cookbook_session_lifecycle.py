#!/usr/bin/env python3
"""Assert every embedded engine opened under `cookbook/**`'s NON-pytest lanes
is closed before its `TemporaryDirectory` is removed (issue #539).

**Guarded property**: for every `with tempfile.TemporaryDirectory() as X:`
(or `tempfile.TemporaryDirectory` aliased through `tempfile.`, any spelling)
anywhere in cookbook build scripts, recipes, the quickstart, or an executed
`.qmd` chapter cell, every `jammi.connect(...)` call whose FIRST argument
mentions `X` (an f-string, a `BoolOp` like `args.target or f"file://{X}"`,
string concatenation, or `X.name`) is itself a `with`-item of the SAME `with`
statement or of a `with` nested inside that statement's body, and its context
manager IS the `jammi.connect(...)` call directly — never a wrapper around
it, an assignment, a subscript, an `.append(...)`, a walrus, a `return`, or
`contextlib.ExitStack().enter_context(...)`.

## Why this mechanism, not the registry observer (the issue's other candidate)

The issue names two candidates: (a) run these lanes under a harness that
installs `clients/python/jammi/_sessions.py`'s registry observer and fails
the lane BY LABEL, mirroring `cookbook/book/tests/conftest.py`'s pytest rail;
or (b) a static control-flow analysis. (a) is REJECTED for this unit: every
lane it would need to wrap (`build_*.py` cache scripts run at book-render
time, `quickstart.py`, a recipe `example.py`, and — critically — a `.qmd`
chapter's python cells, which quarto/jupyter execute as a persistent kernel
across cells, never as one importable module a harness could wrap end-to-end
without re-implementing quarto's own execution engine) has no single process
entry point this repo controls the way pytest's `conftest.py` controls test
collection; installing an `atexit`/`observe()` listener around a `.qmd`
render would report a leak only after the WHOLE chapter finishes, deep into
a nightly render no PR ever exercises hermetically, defeating the "RED on a
PR" requirement `ci.yml`'s guard matrix exists for. This gate is therefore
(b): the AST analysis, run over the real committed tree in every PR, no
render needed.

## Why AST and not the excised regex/indent gate (`3a10b546`, issue #536)

The prior static gate (`test_session_lifecycle_shape.py`, deleted on
`fix/536-embedded-close`) tracked taint by line INDENTATION and matched
`jammi.connect(...)` with a single-line regex. A third audit on that PR found
it unsound on: a `with`-item whose context manager is not the session itself
(a wrapper), an `ExitStack` built outside the `with` block, two
`TemporaryDirectory` items on one physical line, a >20-line parenthesized
`with (` header, and any DEDENTED line inside the block — all passing with
zero offenders while an ad hoc AST shadow oracle found the same zero
offenders were genuinely fixed. This module is that shadow oracle, committed
and CI-wired instead of thrown away: it walks the REAL block extent (a
`with` statement's own `.items` and `.body`, at any nesting depth, via
`ast.walk` — no line count, no indentation column ever enters the analysis),
so every one of those five unsound shapes is structurally impossible here:
- **wrapper with-item**: credit requires a with-item's `context_expr` to BE
  the `jammi.connect(...)` `Call` node itself (`_credited_connect_ids`,
  computed once as identity-set over the whole tree); a wrapper's own call
  node is a *different* node than the nested connect call inside its
  arguments, so the wrapper is never credited and the connect call nested in
  its argument is still walked and still flagged if tainted.
- **`ExitStack` outside the block**: `stack.enter_context(jammi.connect(...))`
  is an ordinary `Call` argument, not a with-item at all — never credited.
- **two `TemporaryDirectory` items on one physical line**: each with-item is
  its own AST node regardless of how many share a source line; both bind
  their own name into this statement's taint set.
- **>20-line parenthesized header**: `ast.parse` handles a parenthesized
  `with (A as a, B as b):` (3.10+ grammar) exactly like the un-parenthesized
  form — same `With` node, same `.items` list, no length cliff.
- **dedented line inside the block**: block membership is "is this node a
  descendant, in the real AST, of this `With` node's `.items`/`.body`" — a
  line's column offset in the source text never enters that question.

Exit-path soundness (the "finally-body control flow" in the issue's title):
crediting ONLY a with-item's own `context_expr` — never anything textually
inside a `finally:` clause — is what makes this sound rather than merely
narrower. Python's `with` statement calls `__exit__` on every exit path from
its body (a normal fall-through, a `return`, a `break`/`continue`, or an
exception unwinding past it) as a language guarantee, so a with-item connect
is closed before the block's `TemporaryDirectory` item unwinds NO MATTER
what control flow the body executes, including a `return` buried inside a
nested `try/finally`. A `close()` call written inside a `finally:` clause,
by contrast, is NOT distinguishable — by any static analysis short of a full
call-graph walk — from a close on an aliasing name, a close inside a nested
`def`, or a close guarded by an `if`; crediting it was exactly what the
excised gate did unsoundly. So this gate credits with-items only, same as
the excised gate's own final (E1) oracle, but reaches that oracle by
structure instead of by indentation text, which is what removes the five
unsound shapes above rather than merely renaming them.

## Scope

`cookbook/book/scripts/**`, `cookbook/quickstart/**`, `cookbook/recipes/**`
(`*.py`), and `cookbook/book/chapters/**/*.qmd` (every `python` fenced code
cell, concatenated per file so cross-cell state — a chapter executes as one
continuous kernel across its `.qmd`'s cells — cannot hide a connect the
"whole file" reach below would otherwise see, blank-line-padded so line
numbers in a finding still point at the real `.qmd` line).

`cookbook/book/tests/**` is EXCLUDED on purpose: that is the pytest lane,
already covered by `conftest.py::_no_leaked_sessions` (the registry rail),
which the issue itself says needs no static gate alongside it ("No static
shape gate exists for those lanes and none is needed.").

## Known limits (labelled, not silently absorbed)

- **Intra-procedural.** Taint does not cross a function call: a
  `TemporaryDirectory` passed as a parameter into a helper that derives the
  real catalog directory several calls deep (`build_recompute_cache.py`'s
  `emit` -> `run_cache`/`_fresh_chain` -> `tempfile.mkdtemp(dir=...)` chain)
  is invisible to this gate — found and fixed by hand on `fix/536-embedded-close`,
  not by any static gate, before or after this one.
- **`jammi.connect` only.** The taint sink this gate recognizes is the
  attribute call `jammi.connect(...)`; an aliased import
  (`from jammi import connect`) binding a bare `connect(...)` call is
  invisible to it, the same limit the excised gate's `_CONNECT` regex had.
  No cookbook `.py`/`.qmd` file does this today (swept: every
  `from jammi import ...` site in `cookbook/**` names `Session`/`Capability`,
  never `connect`) — a reviewer introducing one should not rely on this gate
  to catch it.
- **A held, never-removed directory is out of scope by construction**, same
  as before: `jammi.connect(f"file://{tempfile.mkdtemp(...)}")` opens a
  catalog nothing ever `shutil.rmtree`s, so there is no removal race for this
  gate to speak to (`cookbook/recipes/*/0N-*.py`'s shared `ARTIFACT_DIR`, the
  quickstart's persisted example directories, etc.).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
COOKBOOK = REPO_ROOT / "cookbook"

# The non-pytest lanes this gate owns (see module docstring's "Scope").
_PY_ROOTS = (
    COOKBOOK / "book" / "scripts",
    COOKBOOK / "quickstart",
    COOKBOOK / "recipes",
)
_QMD_ROOTS = (COOKBOOK / "book" / "chapters",)

_QMD_CELL_OPEN = re.compile(r"^```\{\s*python\b")
_QMD_CELL_CLOSE = re.compile(r"^```\s*$")


class Offense(NamedTuple):
    path: str
    lineno: int
    tainted_by: str
    snippet: str


def extract_qmd_python(text: str) -> str:
    """Every executed ```{python} cell's source, concatenated in file order,
    with every non-cell line (prose, other-language cells, fences
    themselves) replaced by a blank line — so line numbers in the returned
    text index the SAME lines as the original `.qmd` file, and `ast.parse`
    sees exactly what quarto executes: one continuous sequence of Python
    statements, cell boundaries and all (a chapter's cells share one kernel,
    so a name bound in an earlier cell is real in a later one; concatenation
    with blank filler reproduces that without needing to model quarto
    itself, because a well-formed `.qmd` cell is always a self-contained,
    independently parseable chunk of Python -- a `with` block cannot span a
    cell boundary, since Python's own parser must see it complete within the
    one unit quarto sends it).
    """
    out: list[str] = []
    in_cell = False
    for line in text.splitlines():
        if not in_cell:
            if _QMD_CELL_OPEN.match(line):
                in_cell = True
            out.append("")
            continue
        if _QMD_CELL_CLOSE.match(line):
            in_cell = False
            out.append("")
            continue
        out.append(line)
    return "\n".join(out)


def _is_temporarydirectory_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "TemporaryDirectory"
    if isinstance(func, ast.Attribute):
        return func.attr == "TemporaryDirectory"
    return False


def _is_jammi_connect_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "connect"
        and isinstance(func.value, ast.Name)
        and func.value.id == "jammi"
    )


def _connect_first_argument(call: ast.Call) -> ast.AST | None:
    """The `target` argument of a `jammi.connect(...)` call: positional if
    given, else the `target=` keyword (both are valid — `target` is the
    first, non-keyword-only parameter of `jammi.connect`)."""
    if call.args:
        return call.args[0]
    for kw in call.keywords:
        if kw.arg == "target":
            return kw.value
    return None


def _tainted_by(call: ast.Call, names: set[str]) -> set[str]:
    """The subset of `names` that appears as a bare `Name` anywhere in the
    connect call's first-argument expression subtree -- an f-string's
    `FormattedValue`, a `BoolOp` (`args.target or f"file://{d}"`), string
    concatenation (`"file://" + d`), and `d.name` (a `TemporaryDirectory`
    OBJECT's own attribute) all contain a plain `Name` node for `d`
    somewhere in that subtree, so one walk covers all four shapes the issue
    names."""
    arg = _connect_first_argument(call)
    if arg is None:
        return set()
    found = {n.id for n in ast.walk(arg) if isinstance(n, ast.Name)}
    return found & names


def _credited_connect_ids(tree: ast.AST) -> set[int]:
    """`id()` of every `jammi.connect(...)` `Call` node that is DIRECTLY a
    with-item's own `context_expr`, anywhere in `tree` -- the ONLY shape
    `Session.__exit__` is guaranteed to run for on every exit path. A
    wrapper (`SomeCM(jammi.connect(...))` as the item) has its OWN call node
    as the item's `context_expr`; the nested connect call is a different
    node and is never added here."""
    credited: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if _is_jammi_connect_call(item.context_expr):
                    credited.add(id(item.context_expr))
    return credited


def _tempdir_with_statements(tree: ast.AST) -> list[ast.With | ast.AsyncWith]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.With, ast.AsyncWith))
        and any(_is_temporarydirectory_call(item.context_expr) for item in node.items)
    ]


def find_offenses(tree: ast.AST, path_label: str) -> list[Offense]:
    """Every `jammi.connect(...)` call anywhere in `tree` that is tainted by
    a `TemporaryDirectory` with-item's bound name and is not itself a
    with-item whose context manager IS that connect call."""
    credited = _credited_connect_ids(tree)
    flagged: dict[int, tuple[ast.Call, set[str]]] = {}

    for with_stmt in _tempdir_with_statements(tree):
        names = {
            item.optional_vars.id
            for item in with_stmt.items
            if _is_temporarydirectory_call(item.context_expr)
            and isinstance(item.optional_vars, ast.Name)
        }
        if not names:
            continue
        # The block extent: this statement's OWN items (a sibling connect
        # item evaluates after the tempdir item binds its name -- see
        # module docstring) and its body, at any nesting depth -- found by
        # walking the `With`/`AsyncWith` node itself, which structurally
        # includes both `.items` and `.body`.
        for node in ast.walk(with_stmt):
            if not _is_jammi_connect_call(node):
                continue
            hit = _tainted_by(node, names)
            if not hit:
                continue
            existing = flagged.get(id(node))
            if existing is None:
                flagged[id(node)] = (node, set(hit))
            else:
                existing[1].update(hit)

    offenses: list[Offense] = []
    for call_id, (call, hit_names) in flagged.items():
        if call_id in credited:
            continue
        try:
            snippet = ast.unparse(call)
        except (ValueError, TypeError):  # pragma: no cover - defensive only
            snippet = "jammi.connect(...)"
        offenses.append(
            Offense(
                path=path_label,
                lineno=getattr(call, "lineno", -1),
                tainted_by=", ".join(sorted(hit_names)),
                snippet=snippet,
            )
        )
    offenses.sort(key=lambda o: o.lineno)
    return offenses


def _iter_targets() -> Iterable[Path]:
    for root in _PY_ROOTS:
        if root.exists():
            yield from sorted(root.rglob("*.py"))
    for root in _QMD_ROOTS:
        if root.exists():
            yield from sorted(root.rglob("*.qmd"))


def scan_tree() -> tuple[list[Offense], list[str]]:
    """Returns (offenses, parse_errors) over the real committed tree."""
    offenses: list[Offense] = []
    parse_errors: list[str] = []
    for path in _iter_targets():
        label = str(path.relative_to(REPO_ROOT))
        text = path.read_text(errors="replace")
        source = extract_qmd_python(text) if path.suffix == ".qmd" else text
        try:
            tree = ast.parse(source, filename=label)
        except SyntaxError as exc:
            parse_errors.append(f"{label}: {exc}")
            continue
        offenses.extend(find_offenses(tree, label))
    return offenses, parse_errors


def _format_offense(o: Offense) -> str:
    return (
        f"{o.path}:{o.lineno}: `{o.snippet}` is tainted by TemporaryDirectory "
        f"`{o.tainted_by}` and is not a with-item whose context manager is the "
        "session itself -- it is not guaranteed closed before that directory "
        "is removed"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Cookbook session-lifecycle AST gate: every embedded engine opened "
            "under cookbook/**'s non-pytest lanes on a TemporaryDirectory is "
            "closed before that directory is removed (issue #539)."
        )
    )
    ap.add_argument("--self-test", action="store_true", help="run the RED-proof self-tests and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    offenses, parse_errors = scan_tree()

    if parse_errors:
        print("::error::cookbook session-lifecycle gate: could not parse:")
        for e in parse_errors:
            print(f"    {e}")

    if offenses:
        print(
            "::error::an embedded engine outlives the TemporaryDirectory "
            "rmtree is about to remove (ENOTEMPTY: Errno 39 on Linux / 66 on "
            "macOS):"
        )
        for o in offenses:
            print(f"    {_format_offense(o)}")
        print(
            "\nClose the session before the directory is removed: put the "
            'connect ON the same `with` statement (`with TemporaryDirectory() '
            'as d, jammi.connect(f"file://{d}") as db:` -- `Session.__exit__` '
            "closes it, and with-items unwind in reverse order)."
        )

    if offenses or parse_errors:
        return 1

    n_scanned = sum(1 for _ in _iter_targets())
    print(
        f"cookbook session-lifecycle gate: clean -- {n_scanned} file(s) scanned "
        "under cookbook/book/scripts, cookbook/quickstart, cookbook/recipes, "
        "cookbook/book/chapters/**/*.qmd; no embedded engine outlives its "
        "TemporaryDirectory."
    )
    return 0


def _self_test() -> int:
    failures: list[str] = []
    total = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal total
        total += 1
        print(f"self-test[{name}]: " + ("ok" if cond else f"FAIL -- {detail}"))
        if not cond:
            failures.append(name)

    def offenses_for(src: str) -> list[Offense]:
        tree = ast.parse(src)
        return find_offenses(tree, "<fixture>")

    # The real tree is green (a gate that reds everywhere proves nothing
    # about the fixtures below).
    real_offenses, real_parse_errors = scan_tree()
    check(
        "real-tree-is-clean",
        not real_offenses and not real_parse_errors,
        f"offenses={real_offenses!r} parse_errors={real_parse_errors!r}",
    )

    # 1/6 positive control: a bare assignment on a tainted directory -> RED.
    assignment = (
        "import tempfile\n"
        "import jammi\n"
        "with tempfile.TemporaryDirectory() as d:\n"
        "    db = jammi.connect(f'file://{d}')\n"
        "    db.set_tenant('x')\n"
    )
    check("assignment-form-flagged", len(offenses_for(assignment)) == 1, str(offenses_for(assignment)))

    # 2/6 negative control: the genuine with-item shape -> GREEN.
    with_item = (
        "import tempfile\n"
        "import jammi\n"
        "with tempfile.TemporaryDirectory() as d, jammi.connect(f'file://{d}') as db:\n"
        "    db.set_tenant('x')\n"
    )
    check("with-item-shape-silent", offenses_for(with_item) == [], str(offenses_for(with_item)))

    # 3/6 negative control: a nested `with ... as handle:` inside the block
    # -> GREEN (closes before the outer tempdir unwinds).
    nested_with = (
        "import tempfile\n"
        "import jammi\n"
        "with tempfile.TemporaryDirectory() as d:\n"
        "    with jammi.connect(f'file://{d}') as handle:\n"
        "        handle.set_tenant('x')\n"
    )
    check("nested-with-silent", offenses_for(nested_with) == [], str(offenses_for(nested_with)))

    # 4/6 wrapper control: the with-item's context manager is NOT the
    # session itself -> RED (the exact shape the excised gate missed).
    wrapper = (
        "import tempfile\n"
        "import jammi\n"
        "class _Wrap:\n"
        "    def __init__(self, inner):\n"
        "        self.inner = inner\n"
        "    def __enter__(self):\n"
        "        return self.inner.__enter__()\n"
        "    def __exit__(self, *a):\n"
        "        return self.inner.__exit__(*a)\n"
        "with tempfile.TemporaryDirectory() as d, _Wrap(jammi.connect(f'file://{d}')) as w:\n"
        "    w.set_tenant('x')\n"
    )
    check("wrapper-with-item-flagged", len(offenses_for(wrapper)) == 1, str(offenses_for(wrapper)))

    # 5/6 an ExitStack built OUTSIDE the with-item mechanism entirely -> RED.
    exit_stack = (
        "import tempfile\n"
        "import contextlib\n"
        "import jammi\n"
        "with tempfile.TemporaryDirectory() as d:\n"
        "    stack = contextlib.ExitStack()\n"
        "    db = stack.enter_context(jammi.connect(f'file://{d}'))\n"
        "    db.set_tenant('x')\n"
        "    stack.close()\n"
    )
    check("exit-stack-flagged", len(offenses_for(exit_stack)) == 1, str(offenses_for(exit_stack)))

    # 6/6 finally-body control flow: a `finally:` close, with an early
    # `return` inside a nested `try` -- this gate credits nothing inside a
    # `finally:` (only a with-item is sound; see module docstring), so this
    # stays RED even though *some* exit paths do call `.close()`.
    finally_body = (
        "import tempfile\n"
        "import jammi\n"
        "def emit():\n"
        "    with tempfile.TemporaryDirectory() as d:\n"
        "        try:\n"
        "            db = jammi.connect(f'file://{d}')\n"
        "            try:\n"
        "                if db.count_rows('t') == 0:\n"
        "                    return None\n"
        "            finally:\n"
        "                pass\n"
        "        finally:\n"
        "            db.close()\n"
    )
    check("finally-body-control-flow-still-flagged", len(offenses_for(finally_body)) == 1, str(offenses_for(finally_body)))

    # Two TemporaryDirectory items on ONE physical `with` statement -- both
    # bind, and a connect tainted by EITHER (not itself a with-item) is RED.
    two_tempdirs_one_stmt = (
        "import tempfile\n"
        "import jammi\n"
        "with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:\n"
        "    db = jammi.connect(f'file://{a}/{b}')\n"
        "    db.set_tenant('x')\n"
    )
    check(
        "two-tempdirs-one-statement-flagged",
        len(offenses_for(two_tempdirs_one_stmt)) == 1,
        str(offenses_for(two_tempdirs_one_stmt)),
    )

    # A parenthesized multi-line `with (` header (3.10+ grammar) -- the
    # with-item shape stays GREEN even split across many lines.
    paren_header = (
        "import tempfile\n"
        "import jammi\n"
        "with (\n"
        "    tempfile.TemporaryDirectory(prefix='x_') as artifact_dir,\n"
        "    jammi.connect(f'file://{artifact_dir}') as db,\n"
        "):\n"
        "    db.set_tenant('x')\n"
    )
    check("parenthesized-header-with-item-silent", offenses_for(paren_header) == [], str(offenses_for(paren_header)))

    # A `--target` CLI override via `BoolOp` (`args.target or f"file://{d}"`)
    # -- with-item shape stays GREEN; bare assignment form stays RED.
    boolop_with_item = (
        "import tempfile\n"
        "import jammi\n"
        "with tempfile.TemporaryDirectory() as catalog, jammi.connect(args.target or f'file://{catalog}') as db:\n"
        "    db.set_tenant('x')\n"
    )
    check("boolop-fallback-with-item-silent", offenses_for(boolop_with_item) == [], str(offenses_for(boolop_with_item)))

    boolop_assignment = (
        "import tempfile\n"
        "import jammi\n"
        "with tempfile.TemporaryDirectory() as catalog:\n"
        "    db = jammi.connect(args.target or f'file://{catalog}')\n"
        "    db.set_tenant('x')\n"
    )
    check("boolop-fallback-assignment-flagged", len(offenses_for(boolop_assignment)) == 1, str(offenses_for(boolop_assignment)))

    # A directory that is NOT a TemporaryDirectory (a held, never-removed
    # catalog) -- out of scope by construction, stays GREEN.
    not_a_tempdir = "import jammi\ndb = jammi.connect(f'file://{ARTIFACT_DIR}')\n"
    check("held-directory-out-of-scope", offenses_for(not_a_tempdir) == [], str(offenses_for(not_a_tempdir)))

    # A `.qmd` cell extraction round-trip: only the `python`-fenced cell's
    # code is parsed, prose and non-python fences are blanked, and the
    # tainted-assignment shape inside a cell is still flagged, with the
    # `.qmd` file's own line number preserved.
    qmd_text = (
        "# A chapter\n"
        "\n"
        "Some prose that mentions `jammi.connect` but is not code.\n"
        "\n"
        "```{r}\n"
        "1 + 1\n"
        "```\n"
        "\n"
        "```{python}\n"
        "import tempfile\n"
        "import jammi\n"
        "with tempfile.TemporaryDirectory() as d:\n"
        "    db = jammi.connect(f'file://{d}')\n"
        "    db.set_tenant('x')\n"
        "```\n"
    )
    extracted = extract_qmd_python(qmd_text)
    qmd_offenses = find_offenses(ast.parse(extracted), "<fixture>.qmd")
    check(
        "qmd-cell-extraction-flags-and-keeps-lineno",
        len(qmd_offenses) == 1 and qmd_offenses[0].lineno == qmd_text.splitlines().index("    db = jammi.connect(f'file://{d}')") + 1,
        str(qmd_offenses),
    )

    total_str = f"{total - len(failures)}/{total}"
    if failures:
        print(f"self-test: FAIL ({len(failures)}/{total} failing): {failures}", file=sys.stderr)
        return 1
    print(f"self-test: all {total_str} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
