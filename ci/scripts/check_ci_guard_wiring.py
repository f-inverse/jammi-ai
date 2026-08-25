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
below closes that blind spot: every `test_*.py` anywhere under `ci/scripts/`
(recursive — a `test_*.py` suite legitimately nests under a feature
subdirectory the way `check_*.py` gate scripts, by convention, do not) PLUS
every `test_*.py` directly under `crates/jammi-bench/reference/` (that
directory's own test suites, which are not `ci/scripts/`-rooted at all) are
now covered by the SAME wiring check `gate_scripts()` already ran for
`check_*.py`/`check_*.sh` — the execution-provenance principle applied to
this checker's own blind spot: zero-execution is RED, not a skip, and that
must hold for a NEW test suite added anywhere this checker looks, not just
the directory the original author happened to think of.

Self-inclusive: this script (`check_ci_guard_wiring.py`) is itself a
`check_*.py` script under `ci/scripts/`, so it is required to find its own
name in some workflow file just like every other gate. There is no allowlist —
an unwired script is a bug in the PR that added it, not a permanent exception
to carve around.

Run: `python3 ci/scripts/check_ci_guard_wiring.py`
Hermetic: reads only files in the working tree; no network, no build.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "ci" / "scripts"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
BENCH_REFERENCE_DIR = REPO_ROOT / "crates" / "jammi-bench" / "reference"


def gate_scripts() -> list[Path]:
    """Every `check_*.py` / `check_*.sh` gate script at the TOP LEVEL of
    `ci/scripts/`, in a stable order. Deliberately NOT recursive: a
    `check_*.py` gate script is, by this repo's own convention, flat at
    `ci/scripts/`'s top level (see every existing one); `python_test_suites()`
    below is the recursive counterpart for `test_*.py` suites, which do
    legitimately nest.
    """
    return sorted(
        [*SCRIPTS_DIR.glob("check_*.py"), *SCRIPTS_DIR.glob("check_*.sh")]
    )


def python_test_suites() -> list[Path]:
    """Every `test_*.py` python test suite this repo owns outside a Cargo
    crate's own `#[cfg(test)]`/`tests/` — i.e. every standalone `unittest`
    suite a human has to remember to wire into a workflow by hand, the same
    way a `check_*.py` gate script has to be. See this module's own doc for
    the F6 reproduction (three such suites, 49 assertions, wired into zero
    workflows) this function's addition closes.

    Two roots, both RECURSIVE (`rglob`, unlike `gate_scripts()`'s top-level
    `glob`): every `test_*.py` under `ci/scripts/` (covers
    `ci/scripts/perf/test_ab_merge.py` and
    `ci/scripts/perf/test_compare_grad_oracle.py`, and any FUTURE
    `ci/scripts/**/test_*.py`), and every `test_*.py` directly under
    `crates/jammi-bench/reference/` (covers
    `test_torch_grad_oracle_names.py` — a `jammi-bench`-owned reference
    script's own suite, which sits outside `ci/scripts/` entirely, so a
    recursive glob rooted only at `ci/scripts/` would still have missed it).
    """
    perf_tests = sorted(SCRIPTS_DIR.rglob("test_*.py"))
    reference_tests = sorted(BENCH_REFERENCE_DIR.glob("test_*.py")) if BENCH_REFERENCE_DIR.is_dir() else []
    return perf_tests + reference_tests


def workflow_text() -> str:
    """The concatenated text of every workflow file — the corpus a script's name is searched in."""
    return "\n".join(
        path.read_text() for path in sorted(WORKFLOWS_DIR.glob("*.yml"))
    )


def main() -> int:
    scripts = gate_scripts() + python_test_suites()
    if not scripts:
        print(
            "ci-guard-wiring: FAIL — no check_*.py/check_*.sh gate scripts or test_*.py suites found",
            file=sys.stderr,
        )
        return 1

    corpus = workflow_text()
    unwired = [script for script in scripts if script.name not in corpus]

    if unwired:
        print("ci-guard-wiring: FAIL", file=sys.stderr)
        for script in unwired:
            print(
                f"  - {script.relative_to(REPO_ROOT)} is not referenced by any "
                ".github/workflows/*.yml file — wire it into a job or delete it.",
                file=sys.stderr,
            )
        return 1

    for script in scripts:
        print(f"ci-guard-wiring[{script.name}]: OK")
    print(f"ci-guard-wiring: all {len(scripts)} gate script(s)/test suite(s) are wired into a workflow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
