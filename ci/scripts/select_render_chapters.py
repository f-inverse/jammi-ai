#!/usr/bin/env python3
"""Select which cookbook/book chapters a diff must render (the FORWARD half
of the engine<->cookbook loop).

The lead-dispatched phase 6.5 (cookbook-emit) is discipline, not a mechanism —
it fires only when a human/agent remembers to dispatch it. This script is the
mechanized floor underneath it: given a diff, it classifies every chapter into
one of three buckets and returns the render set a PR-gate job can actually
render on CPU, in the ~1-chapter-today cost envelope the Book gate job
budgeted for (see `.github/workflows/cookbook-book.yml`'s own "no render"
rationale, which THIS script's caller narrows, not overrides: full nightly
render stays in `cookbook-render.yml`; this gate renders only the chapters a
GIVEN diff could actually move, cheaply, pre-merge).

Three buckets, in order of priority:

  LIVE_COMPUTE       An executed ```{python}``` cell calls one of the engine's
                      live-compute verbs (`generate_embeddings(`, `.fine_tune(`,
                      `.fine_tune_graph(`, `.embed(`) DIRECTLY -- the chapter's
                      own executed cell drives the engine, not a committed
                      cache. Rendering it re-executes that call against the PR
                      wheel; a behavior change shows up as a render failure
                      (an assertion inside the cell) or a nonzero `quarto
                      render` exit. ALWAYS rendered when the diff touches the
                      engine surface the wheel is built from.

                      A LIVE_COMPUTE chapter whose live cell ALSO opens a
                      `grpc://` target needs a running `jammi-server` to
                      render -- the Book gate job builds no server (see
                      `cookbook-book.yml`'s docstring; that is deliberate,
                      cost-bounded scope, preserved here). Such a chapter is
                      classified LIVE_COMPUTE_NEEDS_SERVER: real live-compute,
                      but out of THIS gate's reach, staying on the nightly
                      `cookbook-render.yml` leg. It is reported, never
                      silently dropped.

  CACHE_READ          No executed cell calls a live-compute verb, but the
                      chapter reads a committed artifact (`contracts.load_artifact(`
                      / `contracts.golden(` / `contracts.assert_close(`) keyed
                      `<dataset>.<name>`. Rendered only when the diff touches
                      that dataset's producer -- either the `build_<x>_cache.py`
                      script that emits `artifacts/<dataset>/` (mapped
                      mechanically off each script's own
                      `.../ "artifacts" / "<dataset>"` path expression, no
                      hand-maintained table) or a committed artifact file
                      under that same `artifacts/<dataset>/` directory.

  STATIC              Neither: prose, links, or a cell that executes real
                      engine calls but never a live-compute verb and never
                      reads a committed cache (e.g. `datasets.qmd`'s raw
                      source registration + row counts). Never selected by
                      this script; a docs-only diff renders nothing.

A chapter whose own `.qmd` file appears in the diff is always selected,
regardless of bucket -- the trivial "you touched it, prove it still renders"
case this script would be dishonest to omit.

Usage:
    python3 ci/scripts/select_render_chapters.py --diff <path-to-file-list>
    python3 ci/scripts/select_render_chapters.py --base <sha> --head <sha>
    python3 ci/scripts/select_render_chapters.py --classify   # dry-run table, no diff
    python3 ci/scripts/select_render_chapters.py --self-test

`--diff` reads a newline-separated list of repo-root-relative changed paths
(what a CI job's `git diff --name-only` produces) from a file, or `-` for
stdin. `--base`/`--head` run `git diff --name-only` in-process (requires a git
checkout with both revisions present). Prints the selected chapters'
repo-root-relative paths, one per line, to stdout; a job substitutes that list
into `quarto render`.

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
BOOK_ROOT = REPO_ROOT / "cookbook" / "book"
CHAPTERS_DIR = BOOK_ROOT / "chapters"
SCRIPTS_DIR = BOOK_ROOT / "scripts"

# Paths that feed the PR-wheel build (`.github/actions/setup-jammi-py`,
# mode: wheel) -- an engine-code diff touching any of these can move a
# LIVE_COMPUTE chapter's measured verdict.
ENGINE_PREFIXES = ("crates/", "packaging/native/", "clients/python/")

# Executed-cell fence: quarto's python cell opener is exactly ```{python}
# (optionally with trailing whitespace); a bare ``` or ```{python} with other
# attributes on the SAME line (e.g. ```{python} #| eval: false is invalid
# quarto syntax -- the option goes on its own `#|` line inside the cell, which
# _CELL_EVAL_FALSE_RE below catches) is not an executed python cell.
_CELL_OPEN_RE = re.compile(r"^```\{python\}\s*$")
_CELL_CLOSE_RE = re.compile(r"^```\s*$")
_CELL_EVAL_FALSE_RE = re.compile(r"^#\|\s*eval:\s*false\s*$")

LIVE_CALL_RE = re.compile(
    r"\.generate_embeddings\(|\.fine_tune_graph\(|\.fine_tune\(|\.embed\("
)
# A `connect()` call whose target literal is `grpc://` -- a live network
# target, not the `file://` embedded (source-registration-only) backend every
# other chapter opens.
GRPC_CONNECT_RE = re.compile(r"connect\(\s*f?[\"']grpc://")
CACHE_READ_RE = re.compile(
    r"\b(?:load_artifact|golden|assert_close)\(\s*f?[\"']([a-zA-Z0-9_]+)\."
)
# Script -> dataset: variable-name-agnostic (build_unified_client_cache.py
# assigns `_OUT`, every other script assigns `ARTIFACTS`) -- match the RHS
# path expression itself, not the LHS name, so a differently-named future
# script is still picked up.
SCRIPT_DATASET_RE = re.compile(r"[\"']artifacts[\"']\s*/\s*[\"']([a-zA-Z0-9_]+)[\"']")


@dataclass(frozen=True)
class Classification:
    path: Path  # repo-root-relative
    bucket: str  # LIVE_COMPUTE | LIVE_COMPUTE_NEEDS_SERVER | CACHE_READ | STATIC
    datasets: frozenset[str] = field(default_factory=frozenset)


def _executed_python_cells(text: str) -> list[str]:
    """Every ```{python}``` fenced cell body, skipping any cell whose first
    directive line is `#| eval: false` (never actually executed by quarto)."""
    lines = text.splitlines()
    cells: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if _CELL_OPEN_RE.match(lines[i]):
            body: list[str] = []
            i += 1
            while i < n and not _CELL_CLOSE_RE.match(lines[i]):
                body.append(lines[i])
                i += 1
            # skip the closing fence itself
            i += 1
            if body and _CELL_EVAL_FALSE_RE.match(body[0].strip()):
                continue
            cells.append("\n".join(body))
        else:
            i += 1
    return cells


def classify_chapter(path: Path) -> Classification:
    text = path.read_text()
    cells = _executed_python_cells(text)
    executed = "\n".join(cells)

    live = bool(LIVE_CALL_RE.search(executed))
    needs_server = live and bool(GRPC_CONNECT_RE.search(executed))
    datasets = frozenset(m.group(1) for m in CACHE_READ_RE.finditer(executed))

    if live and needs_server:
        return Classification(path, "LIVE_COMPUTE_NEEDS_SERVER", datasets)
    if live:
        return Classification(path, "LIVE_COMPUTE", datasets)
    if datasets:
        return Classification(path, "CACHE_READ", datasets)
    return Classification(path, "STATIC", datasets)


def all_chapters(chapters_dir: Path = CHAPTERS_DIR) -> list[Path]:
    return sorted(chapters_dir.rglob("*.qmd"))


def classify_all(chapters_dir: Path = CHAPTERS_DIR) -> list[Classification]:
    return [classify_chapter(p) for p in all_chapters(chapters_dir)]


def script_dataset_map(
    scripts_dir: Path = SCRIPTS_DIR, repo_root: Path = REPO_ROOT
) -> dict[str, set[str]]:
    """`build_<x>_cache.py` (`repo_root`-relative POSIX path) -> the set of
    `artifacts/<dataset>/` directories it writes to."""
    out: dict[str, set[str]] = {}
    for script in sorted(scripts_dir.glob("build_*_cache.py")):
        text = script.read_text()
        datasets = {m.group(1) for m in SCRIPT_DATASET_RE.finditer(text)}
        if datasets:
            try:
                rel = script.relative_to(repo_root).as_posix()
            except ValueError:
                rel = script.as_posix()
            out[rel] = datasets
    return out


def _norm(paths: list[str]) -> list[str]:
    return [p.strip().replace("\\", "/") for p in paths if p.strip()]


def select(
    changed_paths: list[str],
    *,
    chapters_dir: Path = CHAPTERS_DIR,
    scripts_dir: Path = SCRIPTS_DIR,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[Classification], set[Path]]:
    """Return (all classifications, selected chapter paths) for a diff."""
    changed = _norm(changed_paths)
    classifications = classify_all(chapters_dir)

    engine_touched = any(p.startswith(ENGINE_PREFIXES) for p in changed)

    scr_map = script_dataset_map(scripts_dir, repo_root)
    changed_datasets: set[str] = set()
    for p in changed:
        if p in scr_map:
            changed_datasets |= scr_map[p]
        # A committed artifact file itself moved (e.g. a regenerated
        # LFS-backed golden) -- artifacts/<dataset>/... under the book root.
        m = re.match(r"cookbook/book/artifacts/([a-zA-Z0-9_]+)/", p)
        if m:
            changed_datasets.add(m.group(1))

    # A chapter's own file is in the diff -- rendered no matter its bucket.
    try:
        book_rel_chapters_dir = chapters_dir.relative_to(repo_root).as_posix()
    except ValueError:
        book_rel_chapters_dir = None
    self_touched: set[str] = set()
    if book_rel_chapters_dir is not None:
        prefix = book_rel_chapters_dir + "/"
        self_touched = {p for p in changed if p.startswith(prefix) and p.endswith(".qmd")}

    selected: set[Path] = set()
    for c in classifications:
        try:
            rel = c.path.relative_to(repo_root).as_posix()
        except ValueError:
            rel = c.path.as_posix()
        # LIVE_COMPUTE_NEEDS_SERVER is never selected by this gate — not even
        # when the chapter's own .qmd is in the diff. This gate's runner has
        # no server harness (cookbook-render.yml's nightly does), so selecting
        # a needs-server chapter here guarantees a structural red regardless
        # of the chapter's correctness (campaign #443: a diff-touched
        # needs-server chapter was auto-selected and died on the missing
        # `jammi-server` binary). Reported, not silently rendered or dropped;
        # the nightly full render is the lane that executes it.
        if c.bucket == "LIVE_COMPUTE_NEEDS_SERVER":
            continue
        if rel in self_touched:
            selected.add(c.path)
            continue
        if c.bucket == "LIVE_COMPUTE" and engine_touched:
            selected.add(c.path)
            continue
        if c.bucket == "CACHE_READ" and (c.datasets & changed_datasets):
            selected.add(c.path)
            continue
        # STATIC is likewise never selected by this gate, on purpose --
        # reported, not silently rendered or dropped.
    return classifications, selected


# --------------------------------------------------------------------------
# Self-test -- RED-proves the misclassification shapes the design named.
# Runs against synthetic fixtures in a temp tree, never against the real
# chapters (a self-test that reads the real book would silently stop
# proving anything the day the real book stops containing an edge case).
# --------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _self_test() -> int:
    import tempfile

    failures: list[str] = []
    total = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal total
        total += 1
        status = "ok" if cond else "FAIL"
        print(f"self-test[{name}]: {status}" + (f" -- {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        chapters = root / "chapters"
        scripts = root / "scripts"

        # 1. A live call INSIDE an executed cell, alongside a load_artifact
        #    read -- must classify LIVE_COMPUTE, never CACHE_READ. This is
        #    the exact misclassification shape the design named: "a live
        #    call in a cache-read chapter must be caught".
        _write(
            chapters / "mixed" / "mixed.qmd",
            """---\ntitle: mixed\n---\n\n"""
            """```{python}\nfrom jammi_cookbook import contracts\nrecord = contracts.load_artifact("widget.record")\n```\n\n"""
            """```{python}\ndb = jammi.connect(f"file://{tmp}")\ntable = db.generate_embeddings(source="s", model="m", columns=["c"], key="id")\n```\n""",
        )
        c = classify_chapter(chapters / "mixed" / "mixed.qmd")
        check(
            "live-call-in-cache-read-chapter-caught",
            c.bucket == "LIVE_COMPUTE",
            f"got {c.bucket}, wanted LIVE_COMPUTE",
        )

        # 2. A live call mentioned only in a PROSE / non-executed fence
        #    (the real recompute.qmd shape: a plain ``` diagram naming
        #    `generate_embeddings(...)`) must NOT trigger LIVE_COMPUTE.
        _write(
            chapters / "prose" / "prose.qmd",
            """---\ntitle: prose\n---\n\n"""
            """```{python}\nfrom jammi_cookbook import contracts\nm = contracts.load_artifact("widget.matrix")\n```\n\n"""
            """```\nadd_source("docs") -> generate_embeddings(...) -> emb\n```\n""",
        )
        c = classify_chapter(chapters / "prose" / "prose.qmd")
        check(
            "prose-only-live-mention-not-triggered",
            c.bucket == "CACHE_READ" and c.datasets == frozenset({"widget"}),
            f"got bucket={c.bucket} datasets={c.datasets}",
        )

        # 3. A live call whose executed cell ALSO opens a grpc:// target
        #    must classify LIVE_COMPUTE_NEEDS_SERVER, not plain LIVE_COMPUTE
        #    -- this gate builds no server (cookbook-book.yml's own scope),
        #    so a server-needing chapter must never enter the PR-gate
        #    render set even though it IS live-compute.
        _write(
            chapters / "remote" / "remote.qmd",
            """---\ntitle: remote\n---\n\n"""
            """```{python}\nremote = jammi.connect("grpc://127.0.0.1:8081")\nremote.generate_embeddings(source="s", model="m", columns=["c"], key="id")\n```\n""",
        )
        c = classify_chapter(chapters / "remote" / "remote.qmd")
        check(
            "grpc-live-chapter-flagged-needs-server",
            c.bucket == "LIVE_COMPUTE_NEEDS_SERVER",
            f"got {c.bucket}",
        )

        # 4. A chapter with neither a live call nor a load_artifact read
        #    (raw source registration + counts, the real datasets.qmd
        #    shape) classifies STATIC.
        _write(
            chapters / "raw" / "raw.qmd",
            """---\ntitle: raw\n---\n\n"""
            """```{python}\ndb = jammi.connect(f"file://{tmp}")\nn = db.sql("SELECT COUNT(*) FROM x").to_pylist()\n```\n""",
        )
        c = classify_chapter(chapters / "raw" / "raw.qmd")
        check("no-cache-no-live-is-static", c.bucket == "STATIC", f"got {c.bucket}")

        # 5. Build-script -> dataset extraction is variable-name-agnostic
        #    (build_unified_client_cache.py assigns `_OUT`, not `ARTIFACTS`).
        _write(
            scripts / "build_widget_cache.py",
            'from pathlib import Path\n_OUT = Path(__file__).resolve().parent.parent / "artifacts" / "widget"\n',
        )
        smap = script_dataset_map(scripts, root)
        check(
            "script-dataset-map-variable-name-agnostic",
            any(v == {"widget"} for v in smap.values()),
            f"got {smap}",
        )

        # 6. End-to-end selection: an engine-code diff selects LIVE_COMPUTE
        #    (never LIVE_COMPUTE_NEEDS_SERVER), a cache-build-script diff
        #    selects only that dataset's CACHE_READ chapters, and a
        #    docs-only diff (no engine, no script) selects nothing.
        _, sel = select(["crates/jammi-ai/src/lib.rs"], chapters_dir=chapters, scripts_dir=scripts, repo_root=root)
        sel_rel = {p.relative_to(root).as_posix() for p in sel}
        check(
            "engine-diff-selects-live-compute-only",
            sel_rel == {"chapters/mixed/mixed.qmd"},
            f"got {sel_rel}",
        )

        _, sel = select(
            ["scripts/build_widget_cache.py"], chapters_dir=chapters, scripts_dir=scripts, repo_root=root
        )
        sel_rel = {p.relative_to(root).as_posix() for p in sel}
        check(
            "script-diff-selects-only-its-dataset-cache-read-chapters",
            sel_rel == {"chapters/prose/prose.qmd"},
            f"got {sel_rel}",
        )

        _, sel = select(
            ["docs/guide/something.md"], chapters_dir=chapters, scripts_dir=scripts, repo_root=root
        )
        check("docs-only-diff-selects-nothing", len(sel) == 0, f"got {sel}")

        # 7. A chapter's own file in the diff is always selected, even a
        #    STATIC one -- the "you touched it, prove it still renders"
        #    safety net.
        _, sel = select(
            ["chapters/raw/raw.qmd"], chapters_dir=chapters, scripts_dir=scripts, repo_root=root
        )
        sel_rel = {p.relative_to(root).as_posix() for p in sel}
        check("self-touched-chapter-always-selected", sel_rel == {"chapters/raw/raw.qmd"}, f"got {sel_rel}")

        # 8. A self-touched LIVE_COMPUTE_NEEDS_SERVER chapter must NOT be
        #    selected — this gate has no server harness, so selecting it
        #    guarantees a structural red independent of the chapter's own
        #    correctness (campaign #443: the diff-touched needs-server
        #    chapter was auto-selected and died on the missing
        #    `jammi-server` binary). The nightly full render owns it.
        _, sel = select(
            ["chapters/remote/remote.qmd"], chapters_dir=chapters, scripts_dir=scripts, repo_root=root
        )
        check(
            "self-touched-needs-server-chapter-never-selected",
            len(sel) == 0,
            f"got {sel}",
        )

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


def _table_lines(classifications: list[Classification]) -> list[str]:
    lines = []
    for c in classifications:
        rel = c.path.relative_to(REPO_ROOT).as_posix()
        ds = ",".join(sorted(c.datasets)) if c.datasets else "-"
        lines.append(f"{c.bucket:24s} {ds:30s} {rel}")
    return lines


def _cmd_classify() -> int:
    for line in _table_lines(classify_all()):
        print(line)
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
    ap.add_argument("--diff", help="file of newline-separated changed paths, or '-' for stdin")
    ap.add_argument("--base", help="base git ref (with --head)")
    ap.add_argument("--head", help="head git ref (with --base)")
    ap.add_argument("--classify", action="store_true", help="print the full classification table and exit")
    ap.add_argument("--self-test", action="store_true", help="run the RED-proof self-tests and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    if args.classify:
        return _cmd_classify()

    if args.base or args.head:
        if not (args.base and args.head):
            ap.error("--base and --head must be given together")
        changed = _git_diff_names(args.base, args.head)
        return _cmd_select(changed)

    if args.diff:
        if args.diff == "-":
            changed = sys.stdin.read().splitlines()
        else:
            changed = Path(args.diff).read_text().splitlines()
        return _cmd_select(changed)

    ap.error("one of --diff, --base/--head, --classify, or --self-test is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
