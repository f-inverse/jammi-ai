#!/usr/bin/env python3
"""The tests of the repository's own scripts: every `test_*.py` and `test_*.sh`
under `ci/`, `tests/compose/` and `crates/jammi-bench/reference/`, and every
script under `ci/scripts/` that answers `--self-test`. One runner executes
them, in CI's `guard` job and locally alike:

    python3 ci/scripts/run_script_tests.py                 # every test of the CI image's lane
    python3 ci/scripts/run_script_tests.py --base main     # only when the change can affect them
    python3 ci/scripts/run_script_tests.py --list
    python3 ci/scripts/run_script_tests.py --only test_gpu_prove_lane.sh
    python3 ci/scripts/run_script_tests.py --lane torch-host

These are tests of the machinery that ships and measures the product, not
properties of the tree: a property lives in `ci/guards.toml`, run by
`run_guards.py`. A script test is selected when the change touches the
scripts' own roots (below) or this runner; with no `--base` every test runs.

A test names what it needs of its host on its first lines, `# needs: a, b`
(names from `ci/needs.toml`, the table the guards share); the runner provides
each need of the selection before running, and a need still absent fails the
run naming it. A test that needs a host the CI image is not declares its lane
too, `# lane: <name>`; `--lane <name>` runs exactly those, and a run without
`--lane` runs the rest. The one lane today is `torch-host`: a checkout with a
cargo toolchain and the PyTorch reference venv (`TORCH_VENV`, default
`.venv-torch-ref`).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checks import (  # noqa: E402
    REPO_ROOT,
    CheckListError,
    Need,
    changed_paths,
    load_needs,
    matches_any,
    needs_of,
    provide_all,
    report,
    resolve_needs,
    run,
)

# Where the scripts and their tests live: a change under any of these can
# affect a script test.
ROOTS = ("ci/**", "tests/compose/**", "crates/jammi-bench/reference/**")
TEST_DIRS = ("ci/scripts", "ci/scripts/perf", "tests/compose", "crates/jammi-bench/reference")
SELF_TEST_DIRS = ("ci/scripts", "ci/scripts/perf")
SELF_PATHS = ("ci/scripts/run_script_tests.py", "ci/scripts/checks.py")
# A script answers `--self-test` when its source names the flag as code — an
# argument-parser literal, an `argv` membership test, a `case` pattern —
# never when prose merely mentions it.
SELF_TEST_TOKEN = re.compile(r"""["']--self-test["']|--self-test\)""")
LANE_LINE = re.compile(r"^#\s*lane:\s*(\S+)\s*$", re.M)
NEEDS_LINE = re.compile(r"^#\s*needs:\s*(.+?)\s*$", re.M)
LANES = {
    "torch-host": "a checkout with a cargo toolchain and the PyTorch reference venv: TORCH_VENV, default .venv-torch-ref",
}


@dataclass(frozen=True)
class ScriptTest:
    path: str
    command: str
    lane: str | None
    needs: tuple[str, ...]


def _head(source: str) -> str:
    return "\n".join(source.splitlines()[:12])


def lane_of(source: str) -> str | None:
    m = LANE_LINE.search(_head(source))
    return m.group(1) if m else None


def declared_needs(source: str) -> tuple[str, ...]:
    m = NEEDS_LINE.search(_head(source))
    return tuple(n.strip() for n in m.group(1).split(",")) if m else ()


def _test(repo_root: Path, path: Path, command: str, needs: dict[str, Need]) -> ScriptTest:
    rel = path.relative_to(repo_root).as_posix()
    source = path.read_text(errors="replace")
    lane = lane_of(source)
    if lane is not None and lane not in LANES:
        raise CheckListError(f"{rel}: unknown lane {lane!r} (lanes: {sorted(LANES)})")
    return ScriptTest(rel, command.format(rel=rel), lane, resolve_needs(declared_needs(source), needs, rel))


def _runner(path: Path) -> str:
    return "python3" if path.suffix == ".py" else "bash"


def discover(needs: dict[str, Need], repo_root: Path = REPO_ROOT) -> tuple[ScriptTest, ...]:
    """Every script test, in a fixed order: test files by path, then self-tests by path."""
    test_files = (
        path
        for d in TEST_DIRS
        for path in sorted((repo_root / d).glob("test_*"))
        if path.suffix in (".py", ".sh")
    )
    self_testing = (
        path
        for d in SELF_TEST_DIRS
        for path in sorted((repo_root / d).iterdir())
        if path.suffix in (".py", ".sh")
        and not path.name.startswith("test_")
        and SELF_TEST_TOKEN.search(path.read_text(errors="replace"))
    )
    return tuple(
        [_test(repo_root, p, f"{_runner(p)} {{rel}}", needs) for p in test_files]
        + [_test(repo_root, p, f"{_runner(p)} {{rel}} --self-test", needs) for p in self_testing]
    )


def select(tests: tuple[ScriptTest, ...], changed: frozenset[str] | None, lane: str | None) -> tuple[ScriptTest, ...]:
    """The tests of `lane` a change to `changed` can affect: all of them when the
    change is unknown or touches the scripts' roots or this runner."""
    in_lane = tuple(t for t in tests if t.lane == lane)
    if changed is None or any(p in changed for p in SELF_PATHS) or matches_any(changed, ROOTS):
        return in_lane
    return ()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", help="run only when the diff base..HEAD can affect the scripts")
    parser.add_argument("--lane", help="run the tests of this lane instead of the CI image's")
    parser.add_argument("--only", action="append", default=[], metavar="PATH", help="run the tests whose path ends with this (repeatable), whatever their lane")
    parser.add_argument("--list", action="store_true", help="print the selection and exit")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    if args.lane is not None and args.lane not in LANES:
        print(f"no such lane: {args.lane!r} (lanes: {sorted(LANES)})", file=sys.stderr)
        return 2
    try:
        needs = load_needs()
        tests = discover(needs)
    except CheckListError as err:
        print(err, file=sys.stderr)
        return 2
    if args.only:
        selected = tuple(t for t in tests if any(t.path.endswith(o) for o in args.only))
        if unmatched := [o for o in args.only if not any(t.path.endswith(o) for t in tests)]:
            print(f"no such script test: {unmatched}", file=sys.stderr)
            return 2
    else:
        selected = select(tests, changed_paths(args.base) if args.base else None, args.lane)
    if args.lane is not None:
        print(f"lane {args.lane}: {LANES[args.lane]}")
    print(f"{len(selected)} script tests selected of {len(tests)}")
    if args.list:
        for t in selected:
            print(t.command + (f"  (needs {', '.join(t.needs)})" if t.needs else ""))
        return 0
    if failures := provide_all(needs_of(tuple(t.needs for t in selected), needs)):
        print("\n".join(failures), file=sys.stderr)
        return 1
    outcomes = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for future in as_completed([pool.submit(run, t.command, t.command) for t in selected]):
            outcomes.append(future.result())
            print(report(outcomes[-1]), flush=True)
    failed = sorted(o.label for o in outcomes if not o.passed)
    print(f"\n{len(outcomes) - len(failed)} passed, {len(failed)} failed")
    for command in failed:
        print(f"  FAIL  {command}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
