#!/usr/bin/env python3
"""Assert every hermetic gate script / python test suite this repo owns is
wired into some workflow.

A `check_*.py` / `check_*.sh` gate script that compiles and passes locally but
is never invoked from any `.github/workflows/*.yml` is dead weight that looks
like coverage but enforces nothing — the author wrote the gate, and stopped one
step short of making CI run it. This check is the completeness tripwire for
that class: it is deliberately mechanical (a name-appears-in-workflow-text
scan), not a semantic understanding of what each gate does, because the
property being enforced is purely "is this script's name mentioned by some
workflow file" — anything richer would be checking a different thing.

F6 (PR #372 audit round) WIDENED this beyond `check_*.py`/`check_*.sh` at the
TOP LEVEL of `ci/scripts/`: three `test_*.py` python `unittest` suites
(`ci/scripts/perf/test_ab_merge.py`, `ci/scripts/perf/test_compare_grad_oracle.py`,
`crates/jammi-bench/reference/test_torch_grad_oracle_names.py`, 49 assertions
combined) landed with ZERO workflow ever mentioning any of their names —
structurally INVISIBLE to the original `gate_scripts()` glob, which only
looked at `check_*.py`/`check_*.sh` sitting directly in `ci/scripts/`, never
recursing into a subdirectory (`ci/scripts/perf/`) and never looking outside
`ci/scripts/` at all (`crates/jammi-bench/reference/`). `python_test_suites()`
below closes that blind spot.

F7 (round-2 audit fix on PR #372, advisory iii) WIDENED it AGAIN: F6's own fix
still hand-picked exactly TWO roots (`ci/scripts/` and
`crates/jammi-bench/reference/`) via two separate `Path.rglob`/`Path.glob`
calls — a THIRD `test_*.py` suite landing under a different crate's own
`reference/` directory, or under the repo's top-level `tests/`, would have
reproduced the EXACT SAME blind spot F6 closed, just one directory over.
`python_test_suites()` below is now driven by `git ls-files` (TRACKED files
only — an untracked/generated `test_*.py` was never really "shipped", and
`git ls-files` is what CI's own checkout actually contains, so this matches
what a CI run can see) filtered against three PREFIX roots: `ci/`,
`crates/<any-crate>/reference/`, `tests/` — none of them hand-picking a
specific crate name, so a fourth crate's `reference/` directory (or a nested
`ci/scripts/**/test_*.py`, or a `tests/test_*.py`) is covered automatically,
never requiring a future PR to remember to widen this file again.

This round ALSO tightens the "is it wired" check itself: the OLD check
searched for a script's name ANYWHERE in a workflow file's raw text,
including inside a `#`-prefixed COMMENT line — so a workflow comment that
merely NAMES a script (this very file's own module doc, several lines above,
names `test_compare_grad_oracle.py` and `test_ab_merge.py` in prose one
workflow file over) would have been enough to satisfy the old check even if
no `run:`/matrix-`cmd:` step ever actually executed it. `workflow_run_text()`
below drops every comment-only line (anything whose stripped content starts
with `#`) before building the search corpus, so only lines that are actually
part of a step body (a literal `run:` line, a `run: |` block's indented
body, or — this repo's OWN indirection convention for a guard-script matrix,
see the `Guard` job in `ci.yml` — a matrix `cmd:` field later interpolated
into `run: ${{ matrix.cmd }}`) can satisfy the wiring requirement. A literal
"only `run:` lines, never `cmd:`" rule would have falsely reddened every
existing entry in that matrix (they are wired via `cmd:` + a single shared
`run: ${{ matrix.cmd }}` step, not a `run:` line naming the script directly)
— this stays a comment-vs-code distinction, not a `run:`-vs-`cmd:` one.

Also new this round: a COMMITTED allowlist (`ALLOWLIST_PATH` below) for a
script that is deliberately not (yet) wired into any workflow — e.g. a gate
still being staged, or a suite intentionally run only by hand. Empty by
default (every script this repo owns right now IS wired); a future PR adds a
line with a reason, never silently skips this check by deleting/renaming the
gate.

Self-inclusive: this script (`check_ci_guard_wiring.py`) is itself a
`check_*.py` script under `ci/scripts/`, so it is required to find its own
name in some workflow file just like every other gate.

Run: `python3 ci/scripts/check_ci_guard_wiring.py`
Hermetic: reads only files in the working tree via `git ls-files` (no
network, no build) — requires running inside a git checkout.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "ci" / "scripts"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Committed exceptions: one relative-path-from-repo-root per line, `#`
# comments and blank lines ignored. A script listed here is EXEMPT from the
# "must be wired into some workflow" requirement — but the exemption is a
# tracked, reviewable line in this file, never a silent absence.
ALLOWLIST_PATH = SCRIPTS_DIR / "ci_guard_wiring_allowlist.txt"

# The three PREFIX roots advisory (iii) names: `ci/` (recursive — a
# `test_*.py` suite legitimately nests under a feature subdirectory, e.g.
# `ci/scripts/perf/`), `crates/<any>/reference/` (a crate's own reference-
# implementation directory, not hand-picked to `jammi-bench` specifically),
# and the repo's top-level `tests/`. Deliberately NOT every directory
# anywhere in the tree literally named `tests/` (e.g. `clients/python/tests/`,
# `cookbook/book/tests/`, `crates/jammi-python/tests/` — those are pytest
# suites this repo already wires through their OWN crate-specific CI jobs by
# a different mechanism (a `pytest` invocation over a directory, not a
# per-file script-name mention), widening to them would be a different,
# larger change than this advisory asked for, and would risk false-reddening
# jobs that are demonstrably already running those suites today).
_CRATES_REFERENCE_RE = re.compile(r"^crates/[^/]+/reference/")


def _tracked_files() -> list[str]:
    """Every file `git` tracks in this checkout, repo-root-relative POSIX
    paths — what a CI run's own `actions/checkout` actually materializes,
    unlike a filesystem glob which would also pick up untracked/generated
    files a `.gitignore` was relying on this check never seeing.
    """
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.splitlines()


def gate_scripts() -> list[Path]:
    """Every TRACKED `check_*.py` / `check_*.sh` gate script anywhere under
    `ci/scripts/`, in a stable order.

    F7 (round-2 audit fix on PR #372, self-inflicted): the ORIGINAL version
    of this function globbed only the TOP LEVEL of `ci/scripts/`, on the
    claimed convention that "a `check_*.py` gate script is, by this repo's
    own convention, flat at `ci/scripts/`'s top level". That claim was true
    right up until THIS SAME round-2 fix added
    `ci/scripts/perf/check_citations.py` (advisory i) — a nested
    `check_*.py` gate script the top-level-only glob below would have been
    STRUCTURALLY BLIND to, the exact class of gap this module's own doc
    already names for `python_test_suites()` (F6/F7), just one function
    over. Rather than special-case ONE nested exception, this function now
    matches `python_test_suites()`'s own tracked-and-recursive shape (`git
    ls-files` under `ci/`, filtered to `check_*.py`/`check_*.sh` by name),
    so a FOURTH nested gate script needs no future PR to remember to widen
    this again either.
    """
    scripts: list[Path] = []
    for rel in _tracked_files():
        if not rel.startswith("ci/"):
            continue
        name = rel.rsplit("/", 1)[-1]
        if name.startswith("check_") and (name.endswith(".py") or name.endswith(".sh")):
            scripts.append(REPO_ROOT / rel)
    return sorted(scripts)


def python_test_suites() -> list[Path]:
    """Every TRACKED `test_*.py` python test suite under one of the three
    prefix roots this module's doc names — the general, root-driven
    replacement for F6's two hand-picked `Path.rglob`/`Path.glob` roots (see
    this module's own doc for why hand-picking a root reproduces the exact
    blind spot it was meant to close, one directory over).
    """
    suites: list[Path] = []
    for rel in _tracked_files():
        if not rel.endswith(".py"):
            continue
        name = rel.rsplit("/", 1)[-1]
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        if rel.startswith("ci/") or rel.startswith("tests/") or _CRATES_REFERENCE_RE.match(rel):
            suites.append(REPO_ROOT / rel)
    return sorted(suites)


def _allowlisted_names() -> set[str]:
    if not ALLOWLIST_PATH.exists():
        return set()
    names = set()
    for line in ALLOWLIST_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        names.add(Path(stripped).name)
    return names


def workflow_run_text() -> str:
    """The concatenated text of every workflow file, with every comment-only
    line (stripped content starting with `#`) DROPPED — so a script's name
    appearing only in prose (a comment explaining what the script does)
    cannot satisfy the wiring requirement; it must appear in an actual step
    body: a `run:` line, a `run: |` block's body, or a matrix `cmd:` field
    (this repo's own indirection convention for the `Guard` job — see this
    module's own doc). See `test_check_ci_guard_wiring.py`'s
    `test_workflow_run_text_drops_comment_only_lines` for the pinned
    reproduction: a script named ONLY in a `#` comment must not count as
    wired.
    """
    lines: list[str] = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        for line in path.read_text().splitlines():
            if line.strip().startswith("#"):
                continue
            lines.append(line)
    return "\n".join(lines)


def main() -> int:
    scripts = gate_scripts() + python_test_suites()
    if not scripts:
        print(
            "ci-guard-wiring: FAIL — no check_*.py/check_*.sh gate scripts or test_*.py suites found",
            file=sys.stderr,
        )
        return 1

    allowlisted = _allowlisted_names()
    corpus = workflow_run_text()
    unwired = [
        script for script in scripts if script.name not in corpus and script.name not in allowlisted
    ]

    if unwired:
        print("ci-guard-wiring: FAIL", file=sys.stderr)
        for script in unwired:
            print(
                f"  - {script.relative_to(REPO_ROOT)} is not referenced by any "
                ".github/workflows/*.yml file's run:/cmd: step body (comments do not count) — "
                f"wire it into a job, delete it, or add it to {ALLOWLIST_PATH.relative_to(REPO_ROOT)} "
                "with a reason.",
                file=sys.stderr,
            )
        return 1

    for script in scripts:
        tag = "ALLOWLISTED" if script.name in allowlisted and script.name not in corpus else "OK"
        print(f"ci-guard-wiring[{script.name}]: {tag}")
    print(f"ci-guard-wiring: all {len(scripts)} gate script(s)/test suite(s) are wired into a workflow or allowlisted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
