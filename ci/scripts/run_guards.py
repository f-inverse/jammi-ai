#!/usr/bin/env python3
"""Run the repository's guards — the one entry point CI and a local run share.

A guard is a hermetic command that holds one property of the tree (no build,
no GPU, no network beyond what it declares). `ci/guards.toml` lists them; this
module selects the ones a change can affect, provides what they declare they
need, runs them, and reports.

    python3 ci/scripts/run_guards.py                  # every guard
    python3 ci/scripts/run_guards.py --base main      # those the diff can affect
    python3 ci/scripts/run_guards.py --only 'doc↔enum parity'
    python3 ci/scripts/run_guards.py --list --base main

Selection: a guard with no `when` always runs. A guard with `when = [scopes]`
runs when a changed path matches one of those scopes' globs. With no `--base`
(a push to main, or a local full run) nothing is known about the change, so
every guard runs; the same holds when the guard list or this runner changed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDS_FILE = Path("ci/guards.toml")
# A change to either re-runs every guard: the selection itself is what changed.
SELF_PATHS = (str(GUARDS_FILE), "ci/scripts/run_guards.py")


class GuardListError(ValueError):
    """`ci/guards.toml` does not describe a runnable guard list."""


@dataclass(frozen=True)
class Need:
    """Something a guard requires of its host: `probe` succeeds when it is
    present, `provide` makes it present where the host allows."""

    name: str
    probe: str
    provide: str


@dataclass(frozen=True)
class Guard:
    name: str
    run: str
    why: str
    when: tuple[str, ...]
    needs: tuple[str, ...]


@dataclass(frozen=True)
class GuardList:
    guards: tuple[Guard, ...]
    scopes: dict[str, tuple[str, ...]]
    needs: dict[str, Need]


@dataclass(frozen=True)
class Outcome:
    guard: Guard
    passed: bool
    seconds: float
    output: str


# ------------------------------------------------------------------ parsing


def _fields(table: dict, where: str, required: set[str], optional: set[str]) -> dict:
    missing = required - table.keys()
    unknown = table.keys() - required - optional
    if missing or unknown:
        raise GuardListError(
            f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}"
        )
    return table


def parse_guard_list(text: str) -> GuardList:
    """The typed guard list `text` describes, or a `GuardListError` naming the
    first thing wrong with it. Every cross-reference is resolved here, so
    nothing downstream meets an unknown scope or need."""
    doc = tomllib.loads(text)
    _fields(doc, "top level", {"guard"}, {"scope", "need"})
    scopes = {
        name: tuple(_fields(t, f"scope.{name}", {"paths"}, set())["paths"])
        for name, t in doc.get("scope", {}).items()
    }
    needs = {
        name: Need(name, **_fields(t, f"need.{name}", {"probe", "provide"}, set()))
        for name, t in doc.get("need", {}).items()
    }
    guards = tuple(
        Guard(
            name=t["name"],
            run=t["run"],
            why=t["why"],
            when=tuple(t.get("when", ())),
            needs=tuple(t.get("needs", ())),
        )
        for t in (
            _fields(t, f"guard #{i + 1}", {"name", "run", "why"}, {"when", "needs"})
            for i, t in enumerate(doc["guard"])
        )
    )
    names = [g.name for g in guards]
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise GuardListError(f"duplicate guard names: {dupes}")
    for g in guards:
        if bad := [s for s in g.when if s not in scopes]:
            raise GuardListError(f"guard {g.name!r}: unknown scope {bad}")
        if bad := [n for n in g.needs if n not in needs]:
            raise GuardListError(f"guard {g.name!r}: unknown need {bad}")
    for name, paths in scopes.items():
        for pattern in paths:
            glob_to_regex(pattern)  # a malformed glob is refused at load
        if not any(name in g.when for g in guards):
            raise GuardListError(f"scope {name!r} is used by no guard")
    for name in needs:
        if not any(name in g.needs for g in guards):
            raise GuardListError(f"need {name!r} is used by no guard")
    return GuardList(guards, scopes, needs)


# ---------------------------------------------------------------- selection


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """A path glob as a regex over repo-relative paths: `**` crosses
    directories, `*` and `?` stay inside one segment, all else is literal."""
    if not pattern or pattern.startswith(("/", "!")):
        raise GuardListError(f"unsupported path glob: {pattern!r}")
    out = []
    for token in re.findall(r"\*\*/|\*\*|\*|\?|[^*?]+", pattern):
        out.append(
            {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}.get(token)
            or re.escape(token)
        )
    return re.compile("".join(out))


def select(guard_list: GuardList, changed: frozenset[str] | None) -> tuple[Guard, ...]:
    """The guards a change to `changed` can affect; all of them when the
    change is unknown (`None`) or touches the selection itself."""
    if changed is None or any(p in changed for p in SELF_PATHS):
        return guard_list.guards
    regexes = {
        name: [glob_to_regex(p) for p in paths]
        for name, paths in guard_list.scopes.items()
    }
    live = {
        name
        for name, rs in regexes.items()
        if any(r.fullmatch(path) for r in rs for path in changed)
    }
    return tuple(g for g in guard_list.guards if not g.when or live & set(g.when))


def needs_of(guard_list: GuardList, selected: tuple[Guard, ...]) -> tuple[Need, ...]:
    """Each need the selected guards declare, once, in first-declared order."""
    ordered = dict.fromkeys(n for g in selected for n in g.needs)
    return tuple(guard_list.needs[n] for n in ordered)


# ---------------------------------------------------------------- execution


def _shell(command: str) -> subprocess.CompletedProcess[str]:
    # stdin from /dev/null: a guard that reads stdin must fail, never hang.
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", command],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )


def changed_paths(base: str) -> frozenset[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", base, "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    return frozenset(filter(None, out.split("\n")))


def provide(need: Need) -> str | None:
    """`None` once `need` is present; otherwise why it could not be provided."""
    if _shell(need.probe).returncode == 0:
        return None
    provided = _shell(need.provide)
    if provided.returncode == 0 and _shell(need.probe).returncode == 0:
        return None
    return f"need {need.name!r} is absent and `{need.provide}` did not provide it:\n{provided.stdout}"


def run_guard(guard: Guard) -> Outcome:
    started = time.monotonic()
    done = _shell(guard.run)
    return Outcome(guard, done.returncode == 0, time.monotonic() - started, done.stdout)


def report(outcome: Outcome) -> str:
    mark = "PASS" if outcome.passed else "FAIL"
    head = f"{mark}  {outcome.guard.name}  ({outcome.seconds:.1f}s)"
    if outcome.passed:
        return head
    return f"{head}\n  why: {outcome.guard.why}\n  run: {outcome.guard.run}\n{outcome.output.rstrip()}\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", help="run only the guards the diff base..HEAD can affect")
    parser.add_argument("--only", action="append", default=[], metavar="NAME", help="run this guard (repeatable)")
    parser.add_argument("--list", action="store_true", help="print the selection and exit")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)

    try:
        guard_list = parse_guard_list((REPO_ROOT / GUARDS_FILE).read_text())
    except (GuardListError, tomllib.TOMLDecodeError) as err:
        print(f"{GUARDS_FILE}: {err}", file=sys.stderr)
        return 2

    if args.only:
        known = {g.name: g for g in guard_list.guards}
        if unknown := [n for n in args.only if n not in known]:
            print(f"no such guard: {unknown}", file=sys.stderr)
            return 2
        selected = tuple(known[n] for n in args.only)
    else:
        selected = select(guard_list, changed_paths(args.base) if args.base else None)

    skipped = len(guard_list.guards) - len(selected)
    print(f"{len(selected)} guards selected, {skipped} out of this change's scope")
    if args.list:
        print("\n".join(g.name for g in selected))
        return 0

    if failures := list(filter(None, map(provide, needs_of(guard_list, selected)))):
        print("\n".join(failures), file=sys.stderr)
        return 1

    outcomes = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for future in as_completed([pool.submit(run_guard, g) for g in selected]):
            outcomes.append(future.result())
            print(report(outcomes[-1]), flush=True)

    failed = sorted(o.guard.name for o in outcomes if not o.passed)
    print(f"\n{len(outcomes) - len(failed)} passed, {len(failed)} failed, {skipped} skipped")
    for name in failed:
        print(f"  FAIL  {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
