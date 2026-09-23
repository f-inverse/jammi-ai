#!/usr/bin/env python3
"""The repository's guards (`ci/guards.toml`): hermetic commands, each holding
one property of the tree, run by one runner in CI's `guard` job and locally
alike — the CI image's lane a hermetic one (no build, no GPU, no network beyond
what a guard declares it needs).

    python3 ci/scripts/run_guards.py                  # every guard
    python3 ci/scripts/run_guards.py --base main      # those the diff can affect
    python3 ci/scripts/run_guards.py --only "engine dep-direction"
    python3 ci/scripts/run_guards.py --list --base main

Selection: a guard with no `when` always runs. A guard with `when = [scopes]`
runs when a changed path matches one of those scopes' globs. With no `--base`
(a push to main, or a local full run) nothing is known about the change, so
every guard runs; the same holds when the guard list, the need table or this
runner changed. A guard declares what it `needs` of its host (`ci/needs.toml`)
and the runner provides it before running; a need still absent fails the run
naming it. A test of a script is not a guard: `run_script_tests.py` runs those.
"""
from __future__ import annotations

import argparse
import os
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checks import (  # noqa: E402
    NEEDS_FILE,
    REPO_ROOT,
    CheckListError,
    Need,
    changed_paths,
    fields,
    glob_to_regex,
    load_needs,
    matches_any,
    needs_of,
    provide_all,
    report,
    resolve_needs,
    run,
)

GUARDS_FILE = Path("ci/guards.toml")
# A change to any of these re-runs every guard: the selection itself is what changed.
SELF_PATHS = (str(GUARDS_FILE), str(NEEDS_FILE), "ci/scripts/run_guards.py", "ci/scripts/checks.py")


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


def parse_guard_list(text: str, needs: dict[str, Need]) -> GuardList:
    """The typed guard list `text` describes, or a `CheckListError` naming the
    first thing wrong with it. Every cross-reference is resolved here, so
    nothing downstream meets an unknown scope or need."""
    doc = tomllib.loads(text)
    fields(doc, "top level", {"guard"}, {"scope"})
    scopes = {
        name: tuple(fields(t, f"scope.{name}", {"paths"}, set())["paths"])
        for name, t in doc.get("scope", {}).items()
    }
    guards = tuple(
        Guard(
            name=t["name"],
            run=t["run"],
            why=t["why"],
            when=tuple(t.get("when", ())),
            needs=resolve_needs(tuple(t.get("needs", ())), needs, f"guard {t['name']!r}"),
        )
        for t in (
            fields(t, f"guard #{i + 1}", {"name", "run", "why"}, {"when", "needs"})
            for i, t in enumerate(doc["guard"])
        )
    )
    names = [g.name for g in guards]
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise CheckListError(f"duplicate guard names: {dupes}")
    for g in guards:
        if bad := [s for s in g.when if s not in scopes]:
            raise CheckListError(f"guard {g.name!r}: unknown scope {bad}")
    for name, paths in scopes.items():
        for pattern in paths:
            glob_to_regex(pattern)  # a malformed glob is refused at load
        if not any(name in g.when for g in guards):
            raise CheckListError(f"scope {name!r} is used by no guard")
    return GuardList(guards, scopes, needs)


def select(guard_list: GuardList, changed: frozenset[str] | None) -> tuple[Guard, ...]:
    """The guards a change to `changed` can affect; all of them when the
    change is unknown (`None`) or touches the selection itself."""
    if changed is None or any(p in changed for p in SELF_PATHS):
        return guard_list.guards
    live = {name for name, globs in guard_list.scopes.items() if matches_any(changed, globs)}
    return tuple(g for g in guard_list.guards if not g.when or live & set(g.when))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", help="run only the guards the diff base..HEAD can affect")
    parser.add_argument("--only", action="append", default=[], metavar="NAME", help="run this guard (repeatable)")
    parser.add_argument("--list", action="store_true", help="print the selection and exit")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)

    try:
        guard_list = parse_guard_list((REPO_ROOT / GUARDS_FILE).read_text(), load_needs())
    except (CheckListError, tomllib.TOMLDecodeError) as err:
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

    if failures := provide_all(needs_of(tuple(g.needs for g in selected), guard_list.needs)):
        print("\n".join(failures), file=sys.stderr)
        return 1

    outcomes = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(run, g.name, g.run): g for g in selected}
        for future in as_completed(futures):
            guard = futures[future]
            outcomes.append(future.result())
            print(report(outcomes[-1], f"  why: {guard.why}\n  run: {guard.run}\n"), flush=True)

    failed = sorted(o.label for o in outcomes if not o.passed)
    print(f"\n{len(outcomes) - len(failed)} passed, {len(failed)} failed, {skipped} skipped")
    for name in failed:
        print(f"  FAIL  {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
