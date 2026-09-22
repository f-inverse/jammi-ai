"""What a check runner shares: the host-need table (`ci/needs.toml`) and its
provisioning, path globs over repo-relative paths, the changed paths of a
diff, and running one hermetic command to a reported outcome. `run_guards.py`
and `run_script_tests.py` both build on this and on nothing of each other's.
"""
from __future__ import annotations

import re
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NEEDS_FILE = Path("ci/needs.toml")


class CheckListError(ValueError):
    """A check list, or the need table, is not what its schema says."""


@dataclass(frozen=True)
class Need:
    """Something a check requires of its host: `probe` succeeds when it is
    present, `provide` makes it present where the host allows."""

    name: str
    probe: str
    provide: str


@dataclass(frozen=True)
class Outcome:
    label: str
    passed: bool
    seconds: float
    output: str


def fields(table: dict, where: str, required: set[str], optional: set[str]) -> dict:
    """`table` with exactly the keys its schema allows, or a `CheckListError`
    naming the first key that is missing or unknown."""
    if missing := sorted(required - table.keys()):
        raise CheckListError(f"{where}: missing {missing}")
    if unknown := sorted(table.keys() - required - optional):
        raise CheckListError(f"{where}: unknown {unknown}")
    return table


def parse_needs(text: str) -> dict[str, Need]:
    """The host-need table `text` describes, by name."""
    doc = tomllib.loads(text)
    fields(doc, "top level", {"need"}, set())
    return {
        name: Need(name, **fields(t, f"need.{name}", {"probe", "provide"}, set()))
        for name, t in doc["need"].items()
    }


def load_needs(repo_root: Path = REPO_ROOT) -> dict[str, Need]:
    return parse_needs((repo_root / NEEDS_FILE).read_text())


def resolve_needs(names: tuple[str, ...], needs: dict[str, Need], where: str) -> tuple[str, ...]:
    """`names`, each a known need, or a `CheckListError` naming the unknown ones."""
    if bad := [n for n in names if n not in needs]:
        raise CheckListError(f"{where}: unknown need {bad} (needs: {sorted(needs)})")
    return names


def needs_of(declared: tuple[tuple[str, ...], ...], needs: dict[str, Need]) -> tuple[Need, ...]:
    """Each need any of `declared` names, once, in first-declared order."""
    ordered = dict.fromkeys(n for names in declared for n in names)
    return tuple(needs[n] for n in ordered)


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """A path glob as a regex over repo-relative paths: `**` crosses
    directories, `*` and `?` stay inside one segment, all else is literal."""
    if not pattern or pattern.startswith(("/", "!")):
        raise CheckListError(f"unsupported path glob: {pattern!r}")
    out = []
    for token in re.findall(r"\*\*/|\*\*|\*|\?|[^*?]+", pattern):
        out.append(
            {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}.get(token)
            or re.escape(token)
        )
    return re.compile("".join(out))


def matches_any(paths: frozenset[str], globs: tuple[str, ...]) -> bool:
    regexes = [glob_to_regex(p) for p in globs]
    return any(r.fullmatch(path) for r in regexes for path in paths)


def changed_paths(base: str) -> frozenset[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", base, "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    return frozenset(filter(None, out.split("\n")))


def shell(command: str) -> subprocess.CompletedProcess[str]:
    """One command from the repository root, `-e -o pipefail`, stdin from
    /dev/null so a command that reads stdin fails rather than hangs."""
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", command],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )


def provide(need: Need) -> str | None:
    """`None` once `need` is present; otherwise why it could not be provided."""
    probed = shell(need.probe)
    if probed.returncode == 0:
        return None
    provided = shell(need.provide)
    if provided.returncode == 0 and shell(need.probe).returncode == 0:
        return None
    return (
        f"need {need.name!r} is absent and `{need.provide}` did not provide it:\n"
        f"{probed.stdout}{provided.stdout}"
    )


def provide_all(needs: tuple[Need, ...]) -> list[str]:
    """Why each need that could not be provided was not; empty when all are present."""
    return list(filter(None, map(provide, needs)))


def run(label: str, command: str) -> Outcome:
    started = time.monotonic()
    done = shell(command)
    return Outcome(label, done.returncode == 0, time.monotonic() - started, done.stdout)


def report(outcome: Outcome, detail: str = "") -> str:
    """One line for a pass; the line, `detail` and the output for a failure."""
    mark = "PASS" if outcome.passed else "FAIL"
    head = f"{mark}  {outcome.label}  ({outcome.seconds:.1f}s)"
    if outcome.passed:
        return head
    return f"{head}\n{detail}{outcome.output.rstrip()}\n"
