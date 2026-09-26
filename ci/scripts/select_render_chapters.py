#!/usr/bin/env python3
"""Select which cookbook/book chapters a diff must render (the FORWARD half
of the engine<->cookbook loop).

Every chapter runs its capability live and checks what it measured against a
frozen golden, so the render IS the check. Rendering every chapter on every
push is the nightly's job (`cookbook-render.yml`); this script returns the
subset a GIVEN diff could move, which the PR gate
(`.github/workflows/cookbook-book.yml`) renders at `small` scale on CPU.

Buckets:

  LIVE                An executed ```{python}``` cell does more than import:
                      it opens an engine, runs verbs, and asserts. A change
                      to the engine the book's wheel is built from, to the
                      book's own library, or to the fixtures it reads can
                      move it.

  LIVE_NEEDS_SERVER   A LIVE chapter that starts a `jammi-server` of its own
                      (the client's `LiveServer` harness) or connects to a
                      `grpc://` target. NEEDS_SERVER is a LANE CAPABILITY,
                      not a selection filter: such a chapter is selected by
                      exactly the LIVE rules, and `--needs-server` reports
                      whether the selected set asks the caller to build the
                      server. Excluding the chapter instead would let a
                      touched one merge having executed nowhere -- the
                      nightly renders the PRE-merge base.

  STATIC              No executed cell beyond imports: prose, links, a
                      reference page. Selected only when its own file is in
                      the diff.

A chapter is selected when:

  * its own `.qmd` is in the diff (any bucket);
  * the diff touches the engine (ENGINE_PREFIXES) or the book's inputs
    (BOOK_INPUT_PREFIXES: its library, its packaging, the fixtures) -- every
    LIVE chapter;
  * the diff touches a golden file, `goldens/<dataset>[.<scale>].json` --
    the LIVE chapters that check a `<dataset>.` metric.

Usage:
    python3 ci/scripts/select_render_chapters.py --diff <path-to-file-list>
    python3 ci/scripts/select_render_chapters.py --base <sha> --head <sha>
    python3 ci/scripts/select_render_chapters.py --classify   # dry-run table, no diff
    python3 ci/scripts/select_render_chapters.py --self-test
    python3 ci/scripts/select_render_chapters.py --base <sha> --head <sha> --needs-server

`--diff` reads a newline-separated list of repo-root-relative changed paths
(what a CI job's `git diff --name-only` produces) from a file, or `-` for
stdin. `--base`/`--head` run `git diff --name-only` in-process. Prints the
selected chapters' repo-root-relative paths, one per line, to stdout; the
classification table goes to stderr.

`--needs-server` prints, INSTEAD of the chapter list, exactly `true` or
`false` on one line: whether the set this same diff selects contains a
LIVE_NEEDS_SERVER chapter. A workflow reads it into a step output and builds
the server on exactly that condition.

Hermetic: no network. `--base`/`--head` shell out to `git diff` only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAPTERS_DIR = REPO_ROOT / "cookbook" / "book" / "chapters"

# Paths that feed the PR-wheel build (`.github/actions/setup-jammi-py`,
# mode: wheel).
ENGINE_PREFIXES = ("crates/", "packaging/native/", "clients/python/")
# What every chapter reads besides the engine: the book's library, its
# packaging, and the committed fixtures. The goldens live under the library
# but select narrowly, by dataset (GOLDEN_RE).
BOOK_INPUT_PREFIXES = (
    "cookbook/book/jammi_cookbook/",
    "cookbook/book/pyproject.toml",
    "cookbook/fixtures/",
    "tests/fixtures/",
)
GOLDEN_RE = re.compile(r"^cookbook/book/jammi_cookbook/goldens/([a-zA-Z0-9_]+)(?:\.[a-z]+)?\.json$")

# Executed-cell fence: quarto's python cell opener is exactly ```{python}
# (optionally with trailing whitespace); an option such as `#| eval: false`
# goes on its own line inside the cell, which _CELL_EVAL_FALSE_RE catches.
_CELL_OPEN_RE = re.compile(r"^```\{python\}\s*$")
_CELL_CLOSE_RE = re.compile(r"^```\s*$")
_CELL_EVAL_FALSE_RE = re.compile(r"^#\|\s*eval:\s*false\s*$")
# A line that executes nothing beyond binding a module: an import, a cell
# option or comment, or a blank.
_INERT_LINE_RE = re.compile(r"^\s*(?:$|#|import\s|from\s+\S+\s+import\s)")

# A `connect()` call whose target literal is `grpc://`.
GRPC_CONNECT_RE = re.compile(r"connect\(\s*f?[\"']grpc://")
# A chapter that starts its own `jammi-server` through the client's harness.
LIVE_SERVER_RE = re.compile(r"\bLiveServer\(")
# The goldens a chapter checks: `assert_close("<dataset>.…")` / `golden(…)`.
GOLDEN_CHECK_RE = re.compile(r"\b(?:golden|assert_close)\(\s*f?[\"']([a-zA-Z0-9_]+)\.")


@dataclass(frozen=True)
class Classification:
    path: Path
    bucket: str  # LIVE | LIVE_NEEDS_SERVER | STATIC
    datasets: frozenset[str] = field(default_factory=frozenset)

    @property
    def live(self) -> bool:
        return self.bucket != "STATIC"


def _executed_python_cells(text: str) -> list[str]:
    """Every ```{python}``` fenced cell body, skipping any cell whose first
    directive line is `#| eval: false` (never executed by quarto)."""
    cells: list[str] = []
    body: list[str] | None = None
    for line in text.splitlines():
        if body is None:
            if _CELL_OPEN_RE.match(line):
                body = []
        elif _CELL_CLOSE_RE.match(line):
            if not (body and _CELL_EVAL_FALSE_RE.match(body[0].strip())):
                cells.append("\n".join(body))
            body = None
        else:
            body.append(line)
    return cells


def classify_chapter(path: Path) -> Classification:
    executed = "\n".join(_executed_python_cells(path.read_text()))
    datasets = frozenset(m.group(1) for m in GOLDEN_CHECK_RE.finditer(executed))
    if all(_INERT_LINE_RE.match(line) for line in executed.splitlines()):
        return Classification(path, "STATIC", datasets)
    if LIVE_SERVER_RE.search(executed) or GRPC_CONNECT_RE.search(executed):
        return Classification(path, "LIVE_NEEDS_SERVER", datasets)
    return Classification(path, "LIVE", datasets)


def classify_all(chapters_dir: Path = CHAPTERS_DIR) -> list[Classification]:
    return [classify_chapter(p) for p in sorted(chapters_dir.rglob("*.qmd"))]


def select(
    changed_paths: list[str],
    *,
    chapters_dir: Path = CHAPTERS_DIR,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[Classification], set[Path]]:
    """Return (all classifications, selected chapter paths) for a diff."""
    changed = {p.strip().replace("\\", "/") for p in changed_paths if p.strip()}
    classifications = classify_all(chapters_dir)

    golden_datasets = {m.group(1) for p in changed if (m := GOLDEN_RE.match(p))}
    every_live = any(
        p.startswith(ENGINE_PREFIXES + BOOK_INPUT_PREFIXES) and not GOLDEN_RE.match(p)
        for p in changed
    )

    def rel(c: Classification) -> str:
        return c.path.relative_to(repo_root).as_posix()

    selected = {
        c.path
        for c in classifications
        if rel(c) in changed
        or (c.live and (every_live or c.datasets & golden_datasets))
    }
    return classifications, selected


def selection_needs_server(
    classifications: list[Classification], selected: set[Path]
) -> bool:
    """Does rendering `selected` require a running `jammi-server`?"""
    return any(c.bucket == "LIVE_NEEDS_SERVER" and c.path in selected for c in classifications)


# --------------------------------------------------------------------------
# Self-test -- against synthetic fixtures in a temp tree, never the real
# chapters: a self-test that reads the real book would stop proving anything
# the day the real book stops containing an edge case.
# --------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _chapter(*cells: str, prose: str = "") -> str:
    fenced = "".join(f"```{{python}}\n{c}\n```\n\n" for c in cells)
    return f"---\ntitle: t\n---\n\n{fenced}{prose}"


def _self_test() -> int:
    import contextlib
    import io
    import tempfile

    failures: list[str] = []
    total = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal total
        total += 1
        print(f"self-test[{name}]: {'ok' if cond else 'FAIL'}"
              + (f" -- {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        chapters = root / "cookbook" / "book" / "chapters"

        # A chapter that opens an engine and checks a golden is LIVE, and
        # names the dataset it checks.
        _write(chapters / "embed" / "embed.qmd", _chapter(
            "import jammi\nfrom jammi_cookbook import contracts",
            'db = jammi.connect(f"file://{tmp}")\n'
            'contracts.assert_close("widget.recall", db.sql("SELECT 1").num_rows)',
        ))
        # A LIVE chapter that checks another dataset's goldens.
        _write(chapters / "other" / "other.qmd", _chapter(
            'db = jammi.connect(f"file://{tmp}")\ncontracts.assert_close("gadget.n", 1)',
        ))
        # grpc:// and the LiveServer harness each need a server.
        _write(chapters / "remote" / "remote.qmd", _chapter(
            'remote = jammi.connect("grpc://127.0.0.1:8081")\nremote.list_models()',
        ))
        _write(chapters / "served" / "served.qmd", _chapter(
            "from jammi.testing import LiveServer",
            "with LiveServer(tmp) as server:\n    jammi.connect(server.endpoint).list_models()",
        ))
        # An import-only preamble and prose -- a live verb named only in a
        # non-executed fence, or in an `eval: false` cell, is not execution.
        _write(chapters / "prose" / "prose.qmd", _chapter(
            "# | echo: false\nimport jammi_cookbook",
            "#| eval: false\ndb.generate_embeddings(source='s')",
            prose="```\nadd_source(\"docs\") -> generate_embeddings(...)\n```\n",
        ))

        buckets = {c.path.parent.name: c for c in classify_all(chapters)}
        check("engine-opening-chapter-is-live", buckets["embed"].bucket == "LIVE",
              buckets["embed"].bucket)
        check("golden-datasets-are-extracted", buckets["embed"].datasets == {"widget"},
              str(buckets["embed"].datasets))
        check("grpc-chapter-needs-server", buckets["remote"].bucket == "LIVE_NEEDS_SERVER",
              buckets["remote"].bucket)
        check("harness-chapter-needs-server", buckets["served"].bucket == "LIVE_NEEDS_SERVER",
              buckets["served"].bucket)
        check("imports-and-prose-are-static", buckets["prose"].bucket == "STATIC",
              buckets["prose"].bucket)

        def selected(*paths: str) -> tuple[list[Classification], set[str]]:
            cls, sel = select(list(paths), chapters_dir=chapters, repo_root=root)
            return cls, {p.parent.name for p in sel}

        live = {"embed", "other", "remote", "served"}
        for trigger in ("crates/jammi-ai/src/lib.rs", "cookbook/book/jammi_cookbook/keystone.py",
                        "cookbook/fixtures/tiny_corpus.parquet"):
            _, sel = selected(trigger)
            check(f"{trigger}-selects-every-live-chapter", sel == live, str(sel))

        _, sel = selected("cookbook/book/jammi_cookbook/goldens/widget.small.json")
        check("a-golden-diff-selects-its-datasets-chapters", sel == {"embed"}, str(sel))
        _, sel = selected("cookbook/book/jammi_cookbook/goldens/gadget.json")
        check("a-scale-free-golden-diff-selects-its-datasets-chapters", sel == {"other"}, str(sel))

        cls, sel = selected("docs/guide/something.md")
        check("docs-only-diff-selects-nothing", sel == set(), str(sel))
        check("an-empty-selection-needs-no-server",
              selection_needs_server(cls, set()) is False)

        _, sel = selected("cookbook/book/chapters/prose/prose.qmd")
        check("a-self-touched-static-chapter-is-selected", sel == {"prose"}, str(sel))

        cls, sel = select(["cookbook/book/chapters/remote/remote.qmd"],
                          chapters_dir=chapters, repo_root=root)
        check("a-self-touched-needs-server-chapter-flags-the-server",
              selection_needs_server(cls, sel) is True)
        cls, sel = select(["cookbook/book/chapters/embed/embed.qmd"],
                          chapters_dir=chapters, repo_root=root)
        check("a-plain-live-selection-needs-no-server",
              selection_needs_server(cls, sel) is False)

        # The `--needs-server` CLI surface: a workflow captures its stdout
        # straight into a step output, so exactly one token goes to stdout
        # and the table to stderr.
        for touched, want in (("remote", "true\n"), ("embed", "false\n")):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = _cmd_needs_server([f"cookbook/book/chapters/{touched}/{touched}.qmd"],
                                       chapters_dir=chapters, repo_root=root)
            check(f"needs-server-cli-prints-a-bare-{want.strip()}",
                  rc == 0 and out.getvalue() == want and "# classification" in err.getvalue(),
                  f"exit {rc}, stdout={out.getvalue()!r}")

    if failures:
        print(f"self-test: FAIL ({len(failures)}/{total} failing): {failures}", file=sys.stderr)
        return 1
    print(f"self-test: all {total} checks passed")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _git_diff_names(base: str, head: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.splitlines()


def _table_lines(
    classifications: list[Classification], repo_root: Path = REPO_ROOT
) -> list[str]:
    """`repo_root` is a parameter only so `_self_test` can render the table
    for a synthetic tree; every production caller takes the default."""
    lines = []
    for c in classifications:
        rel = c.path.relative_to(repo_root).as_posix()
        ds = ",".join(sorted(c.datasets)) if c.datasets else "-"
        lines.append(f"{c.bucket:24s} {ds:30s} {rel}")
    return lines


def _cmd_classify() -> int:
    for line in _table_lines(classify_all()):
        print(line)
    return 0


def _cmd_needs_server(
    changed_paths: list[str],
    *,
    chapters_dir: Path = CHAPTERS_DIR,
    repo_root: Path = REPO_ROOT,
) -> int:
    """Print exactly `true`/`false`: does the set THIS diff selects need a
    running `jammi-server`? One machine-readable token on stdout, so a
    workflow can capture it straight into a step output; the classification
    table still goes to stderr, never mixed into the answer.

    The directory overrides exist for `_self_test`; `main()` takes the
    defaults."""
    classifications, selected = select(changed_paths, chapters_dir=chapters_dir,
                                       repo_root=repo_root)
    print("# classification", file=sys.stderr)
    for line in _table_lines(classifications, repo_root):
        print(f"#   {line}", file=sys.stderr)
    print("true" if selection_needs_server(classifications, selected) else "false")
    return 0


def _cmd_select(changed_paths: list[str]) -> int:
    classifications, selected = select(changed_paths)
    print("# classification", file=sys.stderr)
    for line in _table_lines(classifications):
        print(f"#   {line}", file=sys.stderr)
    if not selected:
        print("# no chapter needs rendering for this diff", file=sys.stderr)
        return 0
    for path in sorted(selected, key=lambda p: p.relative_to(REPO_ROOT).as_posix()):
        print(path.relative_to(REPO_ROOT).as_posix())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--diff", help="file of newline-separated changed paths, or '-' for stdin"
    )
    ap.add_argument("--base", help="base git ref (with --head)")
    ap.add_argument("--head", help="head git ref (with --base)")
    ap.add_argument(
        "--classify",
        action="store_true",
        help="print the full classification table and exit",
    )
    ap.add_argument(
        "--self-test", action="store_true", help="run the RED-proof self-tests and exit"
    )
    ap.add_argument(
        "--needs-server",
        action="store_true",
        help="print `true`/`false` (does the selected set need a running jammi-server?) "
        "instead of the chapter list",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    if args.classify:
        return _cmd_classify()

    run = _cmd_needs_server if args.needs_server else _cmd_select

    if args.base or args.head:
        if not (args.base and args.head):
            ap.error("--base and --head must be given together")
        changed = _git_diff_names(args.base, args.head)
        return run(changed)

    if args.diff:
        if args.diff == "-":
            changed = sys.stdin.read().splitlines()
        else:
            changed = Path(args.diff).read_text().splitlines()
        return run(changed)

    ap.error("one of --diff, --base/--head, --classify, or --self-test is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
