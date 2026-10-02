#!/usr/bin/env python3
"""Select which cookbook/book chapters a diff must render (the FORWARD half
of the engine<->cookbook loop).

Every chapter runs its capability live and checks what it measured against a
frozen golden, so the render IS the check. Rendering every chapter is the
nightly's job (`cookbook-render.yml`); this script returns the subset a GIVEN
diff could move, which `ci.yml`'s book jobs render at `small` scale on CPU,
spread over parallel slices (`--slice K/N`).

Buckets:

  LIVE      An executed ```{python}``` cell does more than import: it opens
            an engine, runs verbs, and asserts.

  STATIC    No executed cell beyond imports: prose, links, a reference page.
            Selected only when its own file is in the diff.

A chapter is selected when:

  * its own `.qmd` is in the diff (either bucket);
  * the diff changes what every live chapter runs on -- every LIVE chapter:
      - a build input of a package the book runs (`SHIPPED_PACKAGES`: the
        native engine, the server and the CLI) or of a workspace crate they
        depend on, normally or at build time (read from `cargo metadata
        --no-deps`). Every file in such a package's directory is a build
        input except its `tests/`, `benches/` and `examples/`; the same rule
        covers the Python packages the book installs (`PYTHON_PACKAGES`);
      - the workspace's build configuration, or the CI image the render
        runs in, which carries quarto and python (`WORKSPACE_INPUTS`);
      - the book's own library, packaging or fixtures (`BOOK_INPUT_PREFIXES`);
  * the diff touches a golden file, `goldens/<dataset>[.<scale>].json` --
    the LIVE chapters that check a `<dataset>.` metric.

Usage:
    python3 ci/scripts/select_render_chapters.py --diff <path-to-file-list>
    python3 ci/scripts/select_render_chapters.py --base <sha> --head <sha> [--slice K/N]
    python3 ci/scripts/select_render_chapters.py --classify   # dry-run table, no diff
    python3 ci/scripts/select_render_chapters.py --self-test

`--diff` reads a newline-separated list of repo-root-relative changed paths
(what a CI job's `git diff --name-only` produces) from a file, or `-` for
stdin. `--base`/`--head` run `git diff --name-only` in-process. Prints the
selected chapters' repo-root-relative paths, one per line, to stdout; the
classification table goes to stderr. `--slice K/N` prints only the K-th of N
interleaved slices of that list (1-based), so N jobs render it in parallel.

Hermetic: no network. `--base`/`--head` shell out to `git diff`, and the
crate closure to `cargo metadata --no-deps`, which reads manifests only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAPTERS_DIR = REPO_ROOT / "cookbook" / "book" / "chapters"

# The workspace packages whose builds the book runs: `jammi-python` is the
# native engine (`packaging/native`), and the server and the CLI are the
# binaries the chapters start and shell out to.
SHIPPED_PACKAGES = ("jammi-python", "jammi-server", "jammi-cli")
# The Python packages the book installs from the checkout: the native
# engine's packaging and the base client.
PYTHON_PACKAGES = ("packaging/native", "clients/python")
# A package directory's subtrees that build nothing the package ships.
NON_BUILD_DIRS = frozenset({"tests", "benches", "examples"})
# Workspace-wide build inputs: a trailing `/` is a directory, anything else
# one file.
WORKSPACE_INPUTS = (
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/",
    ".docker/",
)
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

# The goldens a chapter checks: `assert_close("<dataset>.…")` / `golden(…)`.
GOLDEN_CHECK_RE = re.compile(r"\b(?:golden|assert_close)\(\s*f?[\"']([a-zA-Z0-9_]+)\.")


@dataclass(frozen=True)
class Classification:
    path: Path
    bucket: str  # LIVE | STATIC
    datasets: frozenset[str] = field(default_factory=frozenset)

    @property
    def live(self) -> bool:
        return self.bucket == "LIVE"


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
    return Classification(path, "LIVE", datasets)


def classify_all(chapters_dir: Path = CHAPTERS_DIR) -> list[Classification]:
    return [classify_chapter(p) for p in sorted(chapters_dir.rglob("*.qmd"))]


def shipped_package_dirs(metadata: dict) -> list[str]:
    """The repo-relative directories of `SHIPPED_PACKAGES` and of every
    workspace crate they reach through normal or build dependencies, plus
    `PYTHON_PACKAGES`. `metadata` is `cargo metadata --no-deps` output: a
    workspace dependency carries the `path` it resolves to; dev-dependencies
    build only tests, so they are not followed. An optional dependency is
    followed whatever features enable it."""
    root = Path(metadata["workspace_root"])
    members = set(metadata["workspace_members"])
    by_name = {p["name"]: p for p in metadata["packages"] if p["id"] in members}
    missing = [name for name in SHIPPED_PACKAGES if name not in by_name]
    if missing:
        raise SystemExit(f"select_render_chapters: no workspace package named {missing}")

    seen: set[str] = set()
    stack = list(SHIPPED_PACKAGES)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(
            dep["name"]
            for dep in by_name[name]["dependencies"]
            if dep.get("path") and dep.get("kind") in (None, "build") and dep["name"] in by_name
        )
    crate_dirs = {
        Path(by_name[name]["manifest_path"]).parent.relative_to(root).as_posix() for name in seen
    }
    return sorted(crate_dirs | set(PYTHON_PACKAGES))


def _is_build_input(path: str, package_dirs: list[str]) -> bool:
    for directory in package_dirs:
        prefix = directory + "/"
        if path.startswith(prefix):
            return path[len(prefix):].split("/", 1)[0] not in NON_BUILD_DIRS
    return any(
        path.startswith(entry) if entry.endswith("/") else path == entry
        for entry in WORKSPACE_INPUTS
    )


def select(
    changed_paths: list[str],
    *,
    package_dirs: list[str],
    chapters_dir: Path = CHAPTERS_DIR,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[Classification], list[Path]]:
    """Return (all classifications, the selected chapter paths in path order)
    for a diff."""
    changed = {p.strip().replace("\\", "/") for p in changed_paths if p.strip()}
    classifications = classify_all(chapters_dir)

    golden_datasets = {m.group(1) for p in changed if (m := GOLDEN_RE.match(p))}
    every_live = any(
        _is_build_input(p, package_dirs)
        or (p.startswith(BOOK_INPUT_PREFIXES) and not GOLDEN_RE.match(p))
        for p in changed
    )

    def rel(c: Classification) -> str:
        return c.path.relative_to(repo_root).as_posix()

    selected = [
        c.path
        for c in classifications
        if rel(c) in changed or (c.live and (every_live or c.datasets & golden_datasets))
    ]
    return classifications, sorted(selected, key=lambda p: p.relative_to(repo_root).as_posix())


def take_slice(items: list[Path], slice_spec: str) -> list[Path]:
    """The K-th of N interleaved slices of `items` (`slice_spec` is `K/N`,
    1-based): items K, K+N, K+2N, ... Interleaving spreads neighbouring
    chapters, which tend to share fixtures and cost, across the slices."""
    match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", slice_spec)
    if not match or int(match.group(1)) > int(match.group(2)):
        raise SystemExit(f"select_render_chapters: --slice wants K/N with 1 <= K <= N, got {slice_spec!r}")
    k, n = int(match.group(1)), int(match.group(2))
    return items[k - 1 :: n]


def cargo_metadata(repo_root: Path = REPO_ROOT) -> dict:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


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


def _metadata(root: Path) -> dict:
    """A workspace shaped like the real one: the shipped packages reach
    `jammi-db` and `jammi-wire`; `jammi-test-utils` is a dev-dependency only;
    `jammi-bench` depends on the engine but nothing shipped depends on it."""

    def package(name: str, *deps: tuple[str, str | None]) -> dict:
        return {
            "name": name,
            "id": f"path+file://{root}/crates/{name}#0.1.0",
            "manifest_path": f"{root}/crates/{name}/Cargo.toml",
            "dependencies": [
                {"name": dep, "kind": kind, "path": f"{root}/crates/{dep}"} for dep, kind in deps
            ]
            + [{"name": "serde", "kind": None}],
        }

    packages = [
        package("jammi-python", ("jammi-ai", None), ("jammi-test-utils", "dev")),
        package("jammi-server", ("jammi-ai", None), ("jammi-wire", "build")),
        package("jammi-cli", ("jammi-db", None)),
        package("jammi-ai", ("jammi-db", None)),
        package("jammi-db", ("jammi-test-utils", "dev")),
        package("jammi-wire"),
        package("jammi-test-utils", ("jammi-db", None)),
        package("jammi-bench", ("jammi-ai", None)),
    ]
    return {
        "workspace_root": str(root),
        "workspace_members": [p["id"] for p in packages],
        "packages": packages,
    }


def _self_test() -> int:
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
        package_dirs = shipped_package_dirs(_metadata(root))

        check("the-closure-follows-normal-and-build-dependencies",
              package_dirs == ["clients/python", "crates/jammi-ai", "crates/jammi-cli",
                               "crates/jammi-db", "crates/jammi-python", "crates/jammi-server",
                               "crates/jammi-wire", "packaging/native"],
              str(package_dirs))

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
        # A chapter that talks to a server it starts is LIVE like any other.
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
        check("server-starting-chapter-is-live", buckets["served"].bucket == "LIVE",
              buckets["served"].bucket)
        check("imports-and-prose-are-static", buckets["prose"].bucket == "STATIC",
              buckets["prose"].bucket)

        def selected(*paths: str) -> set[str]:
            _, sel = select(list(paths), package_dirs=package_dirs,
                            chapters_dir=chapters, repo_root=root)
            return {p.parent.name for p in sel}

        live = {"embed", "other", "served"}
        for trigger in (
            "crates/jammi-ai/src/lib.rs",
            "crates/jammi-db/build.rs",
            "crates/jammi-wire/proto/jammi/v1/data.proto",
            "crates/jammi-server/Cargo.toml",
            "packaging/native/python/jammi_native/__init__.py",
            "clients/python/jammi/remote.py",
            "Cargo.lock",
            "rust-toolchain.toml",
            ".cargo/config.toml",
            ".docker/ci.Dockerfile",
            "cookbook/book/jammi_cookbook/keystone.py",
            "cookbook/fixtures/tiny_corpus.parquet",
        ):
            sel = selected(trigger)
            check(f"{trigger}-selects-every-live-chapter", sel == live, str(sel))

        for inert in (
            "crates/jammi-db/tests/it/broker_parity.rs",
            "crates/jammi-ai/benches/encode.rs",
            "crates/jammi-server/examples/serve.rs",
            "crates/jammi-test-utils/src/lib.rs",
            "crates/jammi-bench/src/lib.rs",
            "clients/python/tests/test_remote.py",
            "crates/jammi-db-extra/src/lib.rs",
            "Cargo.toml.orig",
            "docs/guide/something.md",
            ".github/workflows/ci.yml",
        ):
            sel = selected(inert)
            check(f"{inert}-selects-nothing", sel == set(), str(sel))

        sel = selected("cookbook/book/jammi_cookbook/goldens/widget.small.json")
        check("a-golden-diff-selects-its-datasets-chapters", sel == {"embed"}, str(sel))
        sel = selected("cookbook/book/jammi_cookbook/goldens/gadget.json")
        check("a-scale-free-golden-diff-selects-its-datasets-chapters", sel == {"other"}, str(sel))

        sel = selected("cookbook/book/chapters/prose/prose.qmd")
        check("a-self-touched-static-chapter-is-selected", sel == {"prose"}, str(sel))

        _, everything = select(["Cargo.lock"], package_dirs=package_dirs,
                               chapters_dir=chapters, repo_root=root)
        slices = [take_slice(everything, f"{k}/2") for k in (1, 2)]
        check("slices-partition-the-selection",
              sorted(p for s in slices for p in s) == sorted(everything)
              and not set(slices[0]) & set(slices[1]),
              str(slices))
        check("slices-interleave", slices[0] == everything[0::2], str(slices[0]))
        check("a-slice-beyond-the-selection-is-empty",
              take_slice(everything, f"{len(everything) + 1}/{len(everything) + 1}") == [])
        for bad in ("0/2", "3/2", "1", "a/b"):
            try:
                take_slice(everything, bad)
                check(f"slice-{bad}-is-refused", False, "accepted")
            except SystemExit:
                check(f"slice-{bad}-is-refused", True)

        try:
            shipped_package_dirs({**_metadata(root), "packages": []})
            check("a-missing-shipped-package-fails-loudly", False, "accepted")
        except SystemExit:
            check("a-missing-shipped-package-fails-loudly", True)

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
        lines.append(f"{c.bucket:8s} {ds:30s} {rel}")
    return lines


def _cmd_classify() -> int:
    for line in _table_lines(classify_all()):
        print(line)
    return 0


def _cmd_select(changed_paths: list[str], slice_spec: str | None) -> int:
    classifications, selected = select(changed_paths, package_dirs=shipped_package_dirs(cargo_metadata()))
    print("# classification", file=sys.stderr)
    for line in _table_lines(classifications):
        print(f"#   {line}", file=sys.stderr)
    if slice_spec is not None:
        selected = take_slice(selected, slice_spec)
    if not selected:
        print("# no chapter needs rendering for this diff", file=sys.stderr)
    for path in selected:
        print(path.relative_to(REPO_ROOT).as_posix())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--diff", help="file of newline-separated changed paths, or '-' for stdin"
    )
    ap.add_argument("--base", help="base git ref (with --head)")
    ap.add_argument("--head", help="head git ref (with --base)")
    ap.add_argument("--slice", help="print only the K-th of N interleaved slices (K/N)")
    ap.add_argument(
        "--classify",
        action="store_true",
        help="print the full classification table and exit",
    )
    ap.add_argument(
        "--self-test", action="store_true", help="run the RED-proof self-tests and exit"
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    if args.classify:
        return _cmd_classify()

    if args.base or args.head:
        if not (args.base and args.head):
            ap.error("--base and --head must be given together")
        return _cmd_select(_git_diff_names(args.base, args.head), args.slice)

    if args.diff:
        if args.diff == "-":
            changed = sys.stdin.read().splitlines()
        else:
            changed = Path(args.diff).read_text().splitlines()
        return _cmd_select(changed, args.slice)

    ap.error("one of --diff, --base/--head, --classify, or --self-test is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
