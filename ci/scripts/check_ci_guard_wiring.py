#!/usr/bin/env python3
"""Assert every hermetic gate script under `ci/scripts/` is wired into some workflow.

A `check_*.py` / `check_*.sh` gate script that compiles and passes locally but
is never invoked from any `.github/workflows/*.yml` is dead weight that looks
like coverage but enforces nothing — the author wrote the gate, and stopped one
step short of making CI run it. This check is the completeness tripwire for
that class: it is deliberately mechanical (a name-appears-in-workflow-text
scan), not a semantic understanding of what each gate does, because the
property being enforced is purely "is this script's name mentioned by some
workflow file" — anything richer would be checking a different thing.

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


def gate_scripts() -> list[Path]:
    """Every `check_*.py` / `check_*.sh` gate script, in a stable order."""
    return sorted(
        [*SCRIPTS_DIR.glob("check_*.py"), *SCRIPTS_DIR.glob("check_*.sh")]
    )


def workflow_text() -> str:
    """The concatenated text of every workflow file — the corpus a script's name is searched in."""
    return "\n".join(
        path.read_text() for path in sorted(WORKFLOWS_DIR.glob("*.yml"))
    )


def main() -> int:
    scripts = gate_scripts()
    if not scripts:
        print("ci-guard-wiring: FAIL — no check_*.py/check_*.sh scripts found under ci/scripts/", file=sys.stderr)
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
    print(f"ci-guard-wiring: all {len(scripts)} gate script(s) are wired into a workflow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
