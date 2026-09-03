#!/usr/bin/env python3
"""Assert every hermetic gate script / python test suite this repo owns is
wired into some workflow.

A `check_*.py` / `check_*.sh` gate script that compiles and passes locally but
is never invoked from any `.github/workflows/*.yml` or `*.yaml` file is dead
weight that looks like coverage but enforces nothing — the author wrote the
gate, and stopped one step short of making CI run it. This check is the
completeness tripwire for that class: it is deliberately mechanical (a
name-appears-in-workflow-text
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
`ci/scripts/` at all (`crates/jammi-bench/reference/`). `tracked_test_suites()`
below closes that blind spot.

F7 (round-2 audit fix on PR #372, advisory iii) WIDENED it AGAIN: F6's own fix
still hand-picked exactly TWO roots (`ci/scripts/` and
`crates/jammi-bench/reference/`) via two separate `Path.rglob`/`Path.glob`
calls — a THIRD `test_*.py` suite landing under a different crate's own
`reference/` directory, or under the repo's top-level `tests/`, would have
reproduced the EXACT SAME blind spot F6 closed, just one directory over.
`tracked_test_suites()` below is now driven by `git ls-files` (TRACKED files
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

## Second property: the Book gate's server provisioning is WIRED, not read

"Is this script's name mentioned by some workflow" is the right question for
a gate script, because a gate script's whole contract is "run me". It is the
WRONG question for a multi-step job whose correctness lives in the ORDER of
its steps and in which value feeds which input — the name-mention scan reads
green on such a job no matter how its steps are rearranged.

`.github/workflows/cookbook-book.yml`'s forward render gate is exactly that
shape. Three facts have to hold together or the job silently under-proves:

  1. the chapter SELECTOR step (`id: select`) runs BEFORE the wheel build
     (`uses: ./.github/actions/setup-jammi-py`). The selector emits
     `needs_server`, the LANE CAPABILITY the selected set asks for; a build
     that runs first cannot be told to produce a server, and the only repair
     left at that point is to trim the render set to what the job happens to
     have — the exact backwards direction the gate exists to refuse.
  2. `build-server:` is bound to `${{ steps.select.outputs.needs_server }}`,
     never to a literal. A hardcoded `false` silently drops every grpc://
     chapter's proof; a hardcoded `true` pays a server build on every diff
     and, worse, makes the flag stop tracking the selection at all.
  3. the render step puts `target/release` on `PATH`, which is where a built
     `jammi-server` lands. A server built and never reachable is a server
     that was not built.

All three were discipline only: true today, held by nothing, and each one is
a single-line edit away from being false with every existing check still
green. `book_provisioning_violations()` below pins them.

Deliberately a comment-STRIPPED line scan (`workflow_run_text`'s own
comment-vs-code rule, reused rather than reinvented), not a YAML parse. Two
reasons, and the first is not the convenience one: that workflow ALREADY
carries a comment reading "`target/release` goes on PATH exactly as
`cookbook-render.yml` does". A scan over raw text would be satisfied by that
sentence — it would assert that somebody once DESCRIBED the wiring, which is
precisely the "discipline only" state this check exists to end, dressed up as
a gate. Only a line that is part of a step body may satisfy a pin. The second
reason is that no gate in this repo imports PyYAML today, and a gate that
fails closed on a missing third-party import is a gate that fails closed for
the wrong reason.

Fail-closed on absence: if the workflow file is gone (renamed, deleted), that
is a violation, not a skip. A gate that silently passes when its subject
disappears is the fail-open shape this whole file exists to close.

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

# The workflow whose step ORDER and input BINDINGS this file's second property
# pins (see the module doc). Named once, so a rename is a one-line edit here
# rather than a silently-skipped check — `book_provisioning_violations()`
# treats a missing file as a violation, never as "nothing to do".
BOOK_WORKFLOW_NAME = "cookbook-book.yml"

# `build-server:` must take the SELECTOR's output, not a literal. Whitespace
# inside the `${{ }}` expression is normalized by the alternation on spaces
# because GitHub accepts `${{x}}` and `${{ x }}` alike; a literal `true`,
# `false`, or any other `steps.*.outputs.*` fails to match, which is the
# point.
_BUILD_SERVER_BINDING_RE = re.compile(
    r"^build-server:\s*\$\{\{\s*steps\.select\.outputs\.needs_server\s*\}\}\s*$"
)

# The render step's PATH export. Requires all three of: an `export PATH=`, the
# `target/release` directory, and a `$PATH` back-reference — so a line that
# REPLACES PATH rather than prepending to it, or that names some other
# directory, does not satisfy this.
_RELEASE_ON_PATH_RE = re.compile(r"export\s+PATH=.*target/release.*\$PATH")


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
    already names for `tracked_test_suites()` (F6/F7), just one function
    over. Rather than special-case ONE nested exception, this function now
    matches `tracked_test_suites()`'s own tracked-and-recursive shape (`git
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


def tracked_test_suites() -> list[Path]:
    """Every TRACKED `test_*.py` OR `test_*.sh` test suite under one of the
    three prefix roots this module's doc names — the general, root-driven
    replacement for F6's two hand-picked `Path.rglob`/`Path.glob` roots (see
    this module's own doc for why hand-picking a root reproduces the exact
    blind spot it was meant to close, one directory over).

    Widened to `.sh` (round-2 audit on PR #387): `ci/scripts/test_gpu_dev_lifecycle.sh`
    landed as a hermetic `test_*.sh` regression suite and was structurally
    INVISIBLE to this function's original `.py`-only filter — the exact F6/F7
    blind-spot shape, one extension over, catchable only because this suite
    happened to also be hand-wired into `ci.yml`'s Guard matrix already.
    Renamed from `python_test_suites` in the SAME round once it stopped being
    python-exclusive — a name that no longer describes what a function
    returns is exactly the kind of drift this module's own gate exists to
    catch in OTHER files; every call site (here and
    `test_check_ci_guard_wiring.py`'s own pinned suite) was updated with it.
    """
    suites: list[Path] = []
    for rel in _tracked_files():
        name = rel.rsplit("/", 1)[-1]
        if not name.startswith("test_"):
            continue
        if not (name.endswith(".py") or name.endswith(".sh")):
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
    # BLOCK B7 audit fix (same class as check_gpu_prove_once.py's own):
    # GitHub Actions runs BOTH `.yml` and `.yaml` workflow files -- a
    # `*.yml`-only glob is structurally blind to a script wired ONLY from a
    # `.yaml` workflow, exactly the F6 "invisible to the glob" class this
    # module's own doc already warns about for a different dimension
    # (directory depth). Glob both, deduplicated, sorted.
    lines: list[str] = []
    workflow_paths = sorted(set(WORKFLOWS_DIR.glob("*.yml")) | set(WORKFLOWS_DIR.glob("*.yaml")))
    for path in workflow_paths:
        for line in path.read_text().splitlines():
            if line.strip().startswith("#"):
                continue
            lines.append(line)
    return "\n".join(lines)


def _code_lines(path: Path) -> list[str]:
    """A workflow file's lines with every comment-only line dropped — the
    same comment-vs-code rule `workflow_run_text()` applies, so a sentence in
    a comment can never satisfy a wiring pin (see this module's own doc for
    the concrete sentence in `cookbook-book.yml` that would otherwise do it).
    """
    return [line for line in path.read_text().splitlines() if not line.strip().startswith("#")]


def book_provisioning_violations() -> list[str]:
    """The three step-order/binding facts `cookbook-book.yml`'s forward
    render gate needs — see this module's doc for what each one buys and how
    a one-line edit breaks it. Returns a (possibly empty) list of findings.
    """
    path = WORKFLOWS_DIR / BOOK_WORKFLOW_NAME
    if not path.exists():
        return [
            f"{BOOK_WORKFLOW_NAME} is missing from {WORKFLOWS_DIR.name}/ — the forward render "
            "gate's provisioning wiring cannot be asserted. If the workflow was renamed, update "
            "BOOK_WORKFLOW_NAME here in the same commit."
        ]

    lines = _code_lines(path)
    findings: list[str] = []

    select_idx = next((i for i, l in enumerate(lines) if l.strip() == "id: select"), None)
    setup_idx = next(
        (i for i, l in enumerate(lines) if l.strip() == "uses: ./.github/actions/setup-jammi-py"),
        None,
    )

    if select_idx is None:
        findings.append(
            f"{BOOK_WORKFLOW_NAME}: no step carries `id: select` — the chapter selector is what "
            "emits `needs_server`, and nothing downstream can read an output that is never named."
        )
    if setup_idx is None:
        findings.append(
            f"{BOOK_WORKFLOW_NAME}: no step `uses: ./.github/actions/setup-jammi-py` — the wheel "
            "build this job's server provisioning rides on is gone."
        )
    if select_idx is not None and setup_idx is not None and select_idx > setup_idx:
        findings.append(
            f"{BOOK_WORKFLOW_NAME}: the `id: select` step (line {select_idx + 1} of the "
            f"comment-stripped file) runs AFTER setup-jammi-py (line {setup_idx + 1}). The "
            "selector must run FIRST so the build can be told whether to produce a server — "
            "otherwise the only repair left is trimming the render set to what the job already "
            "has, which is the backwards direction this gate exists to refuse."
        )

    if not any(_BUILD_SERVER_BINDING_RE.match(l.strip()) for l in lines):
        findings.append(
            f"{BOOK_WORKFLOW_NAME}: no step body binds `build-server:` to "
            "`${{ steps.select.outputs.needs_server }}`. A literal `true`/`false` there stops "
            "tracking the selection: `false` silently drops every grpc:// chapter's proof, `true` "
            "pays a server build on every diff and hides the same drift."
        )

    if not any(_RELEASE_ON_PATH_RE.search(l) for l in lines):
        findings.append(
            f"{BOOK_WORKFLOW_NAME}: no step body exports `target/release` onto `PATH`. A "
            "`jammi-server` built there and never reachable is a server that was not built."
        )

    return findings


def main() -> int:
    scripts = gate_scripts() + tracked_test_suites()
    if not scripts:
        print(
            "ci-guard-wiring: FAIL — no check_*.py/check_*.sh gate scripts or test_*.py/test_*.sh suites found",
            file=sys.stderr,
        )
        return 1

    allowlisted = _allowlisted_names()
    corpus = workflow_run_text()
    unwired = [
        script for script in scripts if script.name not in corpus and script.name not in allowlisted
    ]

    book_findings = book_provisioning_violations()

    if unwired or book_findings:
        print("ci-guard-wiring: FAIL", file=sys.stderr)
        for script in unwired:
            print(
                f"  - {script.relative_to(REPO_ROOT)} is not referenced by any "
                ".github/workflows/*.yml or *.yaml file's run:/cmd: step body (comments do not count) — "
                f"wire it into a job, delete it, or add it to {ALLOWLIST_PATH.relative_to(REPO_ROOT)} "
                "with a reason.",
                file=sys.stderr,
            )
        for finding in book_findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1

    print(f"ci-guard-wiring[{BOOK_WORKFLOW_NAME} provisioning]: OK")

    for script in scripts:
        tag = "ALLOWLISTED" if script.name in allowlisted and script.name not in corpus else "OK"
        print(f"ci-guard-wiring[{script.name}]: {tag}")
    print(f"ci-guard-wiring: all {len(scripts)} gate script(s)/test suite(s) are wired into a workflow or allowlisted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
