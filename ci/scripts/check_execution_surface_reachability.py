#!/usr/bin/env python3
"""Execution-surface reachability gate — hermetic, static, no build, no GPU.

## The class this closes (esc-050 / esc-051, class_id `seed-tuple-unguarded`)

`ci/scripts/pod_seed_target.sh` runs a fixed set of CUDA/CUTLASS-toolchain-
gated `cargo` invocations (its own T1/T1b/T2/T3 tuples) on every fresh pod's
auto-seed — a leg reds the WHOLE seed the moment its own tuple regresses.
`ci/scripts/runpod_gpu_prove.sh` (invoked only by `gpu-prove.yml`, itself
`workflow_dispatch` / `pull_request: types: [labeled]` / nightly `schedule`
— NEVER a trigger that fires on every PR-to-main or push-to-main) carries a
byte-identical twin of several of those same tuples.

`check_ci_guard_wiring.py` (the gate this one supersedes-in-part for this
class) answers ONE question: does a script's NAME appear in SOME workflow's
run body? That question has no notion of `on:` triggers at all — a tuple
wired only into a dispatch/label/schedule-only workflow satisfies it while
NOTHING on the actual merge path ever runs it. That is exactly the esc-050 /
esc-051 escape shape: `pod_seed_target.sh:859`'s
`cargo clippy -p jammi-kernels --all-targets --features cuda -- -D warnings`
went red on a fresh pod's seed the SAME day #389 merged, because clippy's
only workflow-level twin (`runpod_gpu_prove.sh:56`, byte-identical) lives
behind `gpu-prove.yml`'s label/dispatch/schedule-only trigger — green
"wiring", dead on the path that gates merges.

## Rule 1 — reachability

Every REGISTERED execution-surface tuple (see Rule 2) must be reachable on
the merge path: its own exact `cargo <subcommand> ...` invocation text (see
`extract_tuple` — normalized by dropping a leading `run:`/`cmd:`/`- cmd:`
YAML-step prefix and any trailing ` || ...` shell fallback, never anything
else) must appear, CHARACTER FOR CHARACTER, as its own line inside a
workflow file whose `on:` block genuinely fires on the merge path: a `push`
whose `branches` includes `main` (or carries no `branches`/`tags` filter at
all — fires on every ref push, `main` included), or a `pull_request` whose
`types` is either unset or intersects the GitHub default PR lifecycle types
(`opened`/`synchronize`/`reopened`) AND whose `branches` is either unset or
includes `main`. `workflow_dispatch`, `schedule`, and a `pull_request` whose
`types` is some OTHER set entirely (`gpu-prove.yml`'s `[labeled]`) do NOT
count — parsed honestly from each workflow's own `on:` block (`parse_on_block`
below), never assumed from the workflow's file name or its job names.

Comment-only lines do not count (`ci.yml:217` NAMES
`` `cargo test -p jammi-ai --features cuda,live-gpu-tests gpu_capability`. ``
inside a `#` comment one line above a DIFFERENT, non-cuda compile-check step
— satisfying the OLD wiring gate's name-appears-anywhere scan while never
actually running the cuda-gated tuple. `_drop_comment_lines` strips every
line whose stripped content starts with `#` before any tuple is extracted
from a workflow file, matching `check_ci_guard_wiring.py`'s own
`workflow_run_text` precedent).

Exact-tuple match, NEVER substring (esc-051's own control, restated as
mechanism here): a workflow line reading
`cargo clippy -p jammi-kernels --all-targets --features cuda,flash-attn -- -D warnings`
must NOT satisfy the registered tuple
`cargo clippy -p jammi-kernels --all-targets --features cuda -- -D warnings`
— the two are extracted as two DIFFERENT strings and compared by set
membership, never by one containing the other as a substring.

## Rule 2 — registry completeness

The tuple registry is DERIVED, never hand-maintained: `discover_tuples`
walks every TRACKED file (`git ls-files`, matching `check_ci_guard_wiring.py`
and `check_doc_numbers_have_producers.py`'s own tracked-only precedent — a
CI checkout can only ever see what git tracks) under `ci/scripts/` —
recursively, so a future sibling script (e.g. a nested
`ci/scripts/pods/pod_seed_target_v2.sh`) cannot silently join the class
unregistered the way F6/F7 (`check_ci_guard_wiring.py`'s own module doc)
already had to fix once for a hand-picked, non-recursive glob — and extracts
every line-shaped `cargo (build|test|clippy|check|run) ...` invocation. A
tuple is REGISTERED (subject to Rule 1) only if it is GATED: its
`--features` argument's comma-split token set intersects
`GATED_FEATURE_TOKENS = {"cuda", "flash-attn"}` — the two features that pull
a real CUDA/CUTLASS toolchain (`dep:bindgen_cuda` / vendored CUTLASS,
`crates/jammi-kernels/build.rs`, `ci.yml`'s own "CANNOT be covered here"
comment on the hermetic runner) this repo's ordinary hermetic CI runners do
not carry. A default-feature invocation (e.g.
`cargo test -p jammi-kernels --no-run`, no `--features` at all) is not part
of THIS class — it needs no special hardware/toolchain and is already
exercised, non-exactly but functionally, by the ordinary workspace test job;
registering it here would be a different, broader gate than the one the
retrospective asked for.

`ci/scripts/` only, deliberately (documented, not silently narrow, the
SAME "never widen inside another rule's fix" discipline
`check_ci_guard_wiring.py`'s own module doc names for its two prefix roots):
every tuple `esc-050`/`esc-051` named lives there today
(`pod_seed_target.sh`, `runpod_gpu_prove.sh`). If the class is later found
occupying another root, that is a follow-up PR's job to widen this
constant, exactly as `check_ci_guard_wiring.py`'s `tracked_test_suites`
needed two follow-up rounds (F6, F7) to stop hand-picking roots.

Two paths under `ci/scripts/` are excluded from discovery
(`_DISCOVERY_EXCLUDED_RELPATHS`): this gate's OWN source file (its
`--self-test` fixtures are `cargo ...`-shaped string literals, not real
invocations) and its own allowlist file (whose rows are themselves
`cargo ...`-prefixed lines). Without this exclusion the gate would register
tuples out of its own fixture/waiver data and immediately flag them
UNREACHABLE against itself — a self-inflicted false positive, not a real
finding about the tree.

## Rule 3 — waiver rot

`EXECUTION_SURFACE_ALLOWLIST_PATH` carries one `<tuple text>\t<reason>` row
(TAB-separated, never ` | ` — several real tuples in this class pipe their
own output through `tee`, e.g. `... 2>&1 | tee "$L1"`, so a `|`-based
delimiter would collide with the tuple's OWN text; no cargo invocation in
this repo's `ci/scripts/` contains a literal tab) per registered-but-off-
merge-path tuple. A row's PREDICATE is mechanical and re-checked every run:
the row's tuple text must still be a member of the CURRENT registry (Rule
2's own discovery, re-run fresh every invocation) — a row naming a tuple
whose script was renamed, deleted, or whose exact command line changed is
FAILURE (rot), never a silent no-op skip. A row missing a non-empty reason
is also a failure.

## Honest residual — CUDA tuples force a written choice

Every tuple this class registers needs a REAL CUDA/CUTLASS toolchain to run
meaningfully; the only lane that has one (`gpu-prove.yml`, driving
`runpod_gpu_prove.sh` on a rented RunPod A100) is, by this repo's own design
(module doc, `gpu-prove.yml`), never a merge-path trigger — GPU minutes cost
money and A100 capacity is intermittent. That leaves exactly two honest
choices per tuple, never a silent third:

  (a) `gpu-prove.yml` is promoted to a REQUIRED merge-path check. This is a
      GitHub branch-protection ruleset setting, not committed workflow YAML
      — nothing in this checkout can mechanically prove or disprove it, so
      this gate can never credit it automatically. `GPU_PROVE_PROMOTED_TO_REQUIRED`
      below is the single named constant a human flips (with a comment
      explaining how the promotion was verified) the day that changes; until
      then it stays `False` and every gated tuple falls through to choice (b).
  (b) The tuple owns an explicit, reasoned row in
      `EXECUTION_SURFACE_ALLOWLIST_PATH`, subject to Rule 3's rot check.

There is no code path that lets a registered-but-unreachable tuple pass
silently: Rule 1 fails it unless (a) or (b) holds, and (b) is itself
re-verified (not merely present) every run.

Run: `python3 ci/scripts/check_execution_surface_reachability.py`
Self-test (RED mutants for every rule above, driven against an ephemeral
`git init`'d fixture repo, never this checkout):
`python3 ci/scripts/check_execution_surface_reachability.py --self-test`
Hermetic: reads the working tree (or a `--self-test` tempdir) and shells out
only to `git ls-files`; no network, no cargo, no GPU.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS_ROOT = "ci/scripts/"
WORKFLOWS_DIR_REL = ".github/workflows"

EXECUTION_SURFACE_ALLOWLIST_REL = "ci/scripts/execution_surface_reachability_allowlist.txt"
EXECUTION_SURFACE_ALLOWLIST_PATH = REPO_ROOT / EXECUTION_SURFACE_ALLOWLIST_REL

# This gate's OWN two files, excluded from `discover_all_tuples`'s walk of
# `ci/scripts/**` even though both live under it: this module's own source
# carries `cargo ...`-SHAPED string-literal fixtures for its `--self-test`
# (see `GATED_SCRIPT` and friends below), and the allowlist file's own rows
# are `<tuple text>\t<reason>` lines that themselves start with `cargo` —
# both would otherwise be swept into the "real" registry as if they were
# genuine execution-surface scripts, which they are not.
_DISCOVERY_EXCLUDED_RELPATHS = {
    "ci/scripts/check_execution_surface_reachability.py",
    EXECUTION_SURFACE_ALLOWLIST_REL,
}

# See "Honest residual" above — never flipped by this script itself.
GPU_PROVE_PROMOTED_TO_REQUIRED = False

GATED_FEATURE_TOKENS = {"cuda", "flash-attn"}
CARGO_SUBCOMMANDS = ("build", "test", "clippy", "check", "run")
DEFAULT_PR_LIFECYCLE_TYPES = {"opened", "synchronize", "reopened"}

_CARGO_RE = re.compile(r"^(cargo\s+(?:" + "|".join(CARGO_SUBCOMMANDS) + r")\b.*)$")
_YAML_STEP_PREFIX_RE = re.compile(r"^(?:-\s*)?(?:run|cmd):\s*(.*)$")
_FEATURES_RE = re.compile(r"--features[=\s]+(\S+)")


# --------------------------------------------------------------------------- #
# tuple extraction — shared by registry discovery (ci/scripts/**) and
# workflow-corpus extraction (.github/workflows/*.yml); the SAME function so
# a registered tuple and a workflow's own invocation are compared as
# identically-normalized strings, never two different normalizations that
# could silently agree or disagree by accident.
# --------------------------------------------------------------------------- #
def extract_tuple(raw_line: str) -> str | None:
    """Return the normalized `cargo ...` invocation a line contains, or None.

    Drops a leading `run:`/`cmd:`/`- cmd:` YAML-step prefix (so a single-line
    workflow step `run: cargo clippy ...` and a bare shell-script line
    `cargo clippy ...` normalize identically), then a trailing
    ` || <shell fallback>` (e.g. `|| exit 1`, `|| rc=$?`) and a trailing
    line-continuation backslash (a trailing "\\"; a genuine multi-line
    invocation still normalizes to its FIRST line only — a distinct, real
    tuple text, never stitched across lines) — NEVER anything else: a
    `-- -D warnings` or `-- --nocapture` `--` marker is legitimate
    cargo-argument syntax and must survive untouched, only ` || ` (shell-or,
    always space-delimited in this repo's scripts) and a bare trailing
    backslash are the boundaries.
    """
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    m_prefix = _YAML_STEP_PREFIX_RE.match(stripped)
    if m_prefix:
        stripped = m_prefix.group(1).strip()
    if not stripped or stripped in ("|", ">", "|-", ">-", "|+", ">+"):
        return None
    head = re.split(r"\s\|\|\s", stripped, maxsplit=1)[0].strip()
    if head.endswith("\\"):
        head = head[:-1].rstrip()
    m = _CARGO_RE.match(head)
    if not m:
        return None
    return m.group(1).strip()


def is_gated(tuple_text: str) -> bool:
    m = _FEATURES_RE.search(tuple_text)
    if not m:
        return False
    tokens = {t.strip() for t in m.group(1).split(",")}
    return bool(tokens & GATED_FEATURE_TOKENS)


def extract_all_tuples(text: str) -> set[str]:
    found: set[str] = set()
    for line in text.splitlines():
        t = extract_tuple(line)
        if t is not None:
            found.add(t)
    return found


# --------------------------------------------------------------------------- #
# registry (Rule 2)
# --------------------------------------------------------------------------- #
@dataclass
class TupleRecord:
    text: str
    origins: list[str] = field(default_factory=list)  # "path:lineno"


def _tracked_files(repo_root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.splitlines()


def discover_all_tuples(repo_root: Path) -> dict[str, TupleRecord]:
    """Every `cargo ...` invocation (gated or not) found under `ci/scripts/`
    in a TRACKED file, keyed by its normalized text. Rule 3's rot check reads
    this (not just the gated subset) so a row can also be judged stale if its
    tuple text still exists but is no longer gated (a feature was removed) —
    still "no longer a member of the registry" from Rule 1's point of view.
    """
    registry: dict[str, TupleRecord] = {}
    for rel in _tracked_files(repo_root):
        if not rel.startswith(SCRIPTS_ROOT):
            continue
        if rel in _DISCOVERY_EXCLUDED_RELPATHS:
            continue
        path = repo_root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            t = extract_tuple(line)
            if t is None:
                continue
            rec = registry.setdefault(t, TupleRecord(text=t))
            rec.origins.append(f"{rel}:{lineno}")
    return registry


def gated_tuples(registry: dict[str, TupleRecord]) -> dict[str, TupleRecord]:
    return {t: rec for t, rec in registry.items() if is_gated(t)}


# --------------------------------------------------------------------------- #
# `on:` trigger parsing (Rule 1) — a deliberately minimal, targeted reader:
# not a general YAML parser, only enough structure to answer "does this
# workflow's `on:` block fire on the merge path" for the shapes this repo's
# own workflows actually use (plain block-style `push`/`pull_request` with
# `branches:`/`types:`/`tags:` as either an inline `[a, b]` list or a `- x`
# block list). A workflow using flow-style `on: [push, pull_request]` (none
# does today) would need a widening, exactly the same documented-narrow
# discipline Rule 2's `ci/scripts/`-only scope above states.
# --------------------------------------------------------------------------- #
def _extract_on_block(text: str) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^on:\s*(#.*)?$", line):
            start = i
            break
    if start is None:
        return ""
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() == "" or line[:1] in (" ", "\t") or line.lstrip().startswith("#"):
            body.append(line)
            continue
        break
    return "\n".join(body)


def _first_indent(lines: list[str]) -> int | None:
    for line in lines:
        if line.strip() and not line.lstrip().startswith("#"):
            return len(line) - len(line.lstrip())
    return None


def _extract_list_field(body: str, field_name: str) -> list[str] | None:
    lines = body.splitlines()
    for i, line in enumerate(lines):
        m = re.match(rf"^\s*{re.escape(field_name)}:\s*(.*)$", line)
        if not m:
            continue
        rest = m.group(1).strip()
        if rest.startswith("["):
            inner = rest.strip("[]")
            items = [x.strip().strip("\"'") for x in inner.split(",") if x.strip()]
            return items
        if rest and not rest.startswith("#"):
            return [rest.strip("\"'")]
        items = []
        field_indent = len(line) - len(line.lstrip())
        for l2 in lines[i + 1 :]:
            if not l2.strip():
                continue
            l2_indent = len(l2) - len(l2.lstrip())
            if l2_indent <= field_indent:
                break
            m2 = re.match(r"^\s*-\s*(.+)$", l2)
            if not m2:
                break
            items.append(m2.group(1).strip().strip("\"'"))
        return items if items else None
    return None


def parse_on_block(text: str) -> dict[str, dict[str, list[str] | None]]:
    block = _extract_on_block(text)
    lines = block.splitlines()
    base_indent = _first_indent(lines)
    if base_indent is None:
        return {}
    entries: dict[str, list[str]] = {}
    current_key: str | None = None
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            if current_key is not None:
                entries[current_key].append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if indent == base_indent:
            m = re.match(r"^\s*([A-Za-z_]+):\s*(.*)$", line)
            if not m:
                current_key = None
                continue
            current_key = m.group(1)
            entries[current_key] = [line]
        elif indent > base_indent and current_key is not None:
            entries[current_key].append(line)
        else:
            current_key = None
    return {
        key: {
            "branches": _extract_list_field(body_text, "branches"),
            "types": _extract_list_field(body_text, "types"),
            "tags": _extract_list_field(body_text, "tags"),
        }
        for key, body_lines in entries.items()
        for body_text in ["\n".join(body_lines)]
    }


def is_merge_path(on_dict: dict[str, dict[str, list[str] | None]]) -> tuple[bool, str]:
    push = on_dict.get("push")
    if push is not None:
        branches = push.get("branches")
        tags = push.get("tags")
        if branches and "main" in branches:
            return True, "push: branches include main"
        if branches is None and tags is None:
            return True, "push: no branches/tags filter (fires on any ref push, incl. main)"
    pr = on_dict.get("pull_request")
    if pr is not None:
        types = pr.get("types")
        branches = pr.get("branches")
        types_ok = types is None or bool(set(types) & DEFAULT_PR_LIFECYCLE_TYPES)
        branches_ok = branches is None or "main" in branches
        if types_ok and branches_ok:
            return True, "pull_request: default/lifecycle PR types targeting main (or unrestricted)"
    return False, "no push-to-main and no non-label-only pull_request-to-main trigger"


def _drop_comment_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


def merge_path_workflow_tuples(repo_root: Path) -> tuple[set[str], list[str]]:
    """The union of every `cargo ...` invocation found in a run/cmd body of
    every workflow whose `on:` block genuinely fires on the merge path.
    Returns (reachable tuple texts, the list of merge-path workflow names
    for reporting)."""
    workflows_dir = repo_root / WORKFLOWS_DIR_REL
    reachable: set[str] = set()
    names: list[str] = []
    if not workflows_dir.is_dir():
        return reachable, names
    for path in sorted(workflows_dir.glob("*.yml")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        on_dict = parse_on_block(text)
        merge_path, _reason = is_merge_path(on_dict)
        if not merge_path:
            continue
        names.append(path.name)
        reachable |= extract_all_tuples(_drop_comment_lines(text))
    return reachable, names


# --------------------------------------------------------------------------- #
# allowlist (Rule 3)
# --------------------------------------------------------------------------- #
@dataclass
class AllowlistRow:
    tuple_text: str
    reason: str
    lineno: int


def parse_allowlist(path: Path) -> tuple[list[AllowlistRow], list[str]]:
    """Returns (rows, parse_failures). A malformed row (no TAB separator, or
    an empty reason) is a parse failure, never silently dropped. TAB, never
    `|` — see this function's own call site / the module doc's Rule 3
    section for why a `|`-based delimiter would collide with a real tuple's
    own piped shell text (`... | tee "$L1"`)."""
    if not path.exists():
        return [], []
    rows: list[AllowlistRow] = []
    failures: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        # `raw` is deliberately NOT `.strip()`-ed before the split below — a
        # bare `str.strip()` treats a tab as whitespace and would eat the
        # very separator this format depends on (e.g. a genuinely EMPTY
        # reason, `"...warnings\t"`, must survive as an empty-reason finding,
        # not silently collapse into a "no separator at all" one).
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t", 1)
        if len(parts) != 2:
            failures.append(
                f"{path.name}:{lineno}: malformed row (expected '<tuple text>\\t<reason>', TAB-separated): {raw!r}"
            )
            continue
        tuple_text, reason = parts[0].strip(), parts[1].strip()
        if not tuple_text or not reason:
            failures.append(
                f"{path.name}:{lineno}: row must carry a non-empty tuple text and a non-empty reason: {raw!r}"
            )
            continue
        rows.append(AllowlistRow(tuple_text=tuple_text, reason=reason, lineno=lineno))
    return rows, failures


# --------------------------------------------------------------------------- #
# gate driver
# --------------------------------------------------------------------------- #
def run_gate(repo_root: Path, allowlist_path: Path) -> tuple[list[str], list[str]]:
    """Returns (failures, info_lines)."""
    failures: list[str] = []
    info: list[str] = []

    registry = discover_all_tuples(repo_root)
    gated = gated_tuples(registry)
    reachable, merge_path_workflow_names = merge_path_workflow_tuples(repo_root)
    allow_rows, allow_parse_failures = parse_allowlist(allowlist_path)
    failures.extend(allow_parse_failures)

    allowlisted_texts = {row.tuple_text for row in allow_rows}

    info.append(
        f"{len(registry)} cargo invocation(s) discovered under {SCRIPTS_ROOT} "
        f"({len(gated)} gated on {sorted(GATED_FEATURE_TOKENS)}); "
        f"{len(merge_path_workflow_names)} merge-path workflow(s): {', '.join(merge_path_workflow_names) or '(none)'}"
    )

    # Rule 3 — waiver rot: every allowlist row's tuple must still be a
    # member of the CURRENT (re-discovered this run) registry.
    for row in allow_rows:
        if row.tuple_text not in registry:
            failures.append(
                f"{allowlist_path.name}:{row.lineno}: ROT — allowlisted tuple no longer found in the "
                f"registry (renamed/deleted script, or the exact command line changed): {row.tuple_text!r}"
            )

    # Rule 1 — reachability, subject to the honest residual (Rule "no third
    # silent state"): every gated tuple must be reachable OR allowlisted OR
    # (never true today, see GPU_PROVE_PROMOTED_TO_REQUIRED) mechanically
    # promoted.
    for text, rec in sorted(gated.items()):
        if text in reachable:
            continue
        if GPU_PROVE_PROMOTED_TO_REQUIRED:
            continue
        if text in allowlisted_texts:
            continue
        failures.append(
            "UNREACHABLE gated tuple, no allowlist row: "
            f"{text!r} (origin(s): {', '.join(rec.origins)}) — this cargo invocation needs a real "
            "CUDA/CUTLASS toolchain and is not invoked, byte-for-byte, by any workflow whose `on:` "
            "trigger fires on the merge path. Either wire an exact-matching invocation into a "
            f"merge-path workflow, or add a reasoned row to {allowlist_path.name}."
        )

    return failures, info


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()

    failures, info = run_gate(REPO_ROOT, EXECUTION_SURFACE_ALLOWLIST_PATH)
    for line in info:
        print(f"execution-surface-reachability: {line}")

    if failures:
        print("execution-surface-reachability: FAIL", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        print(f"\nexecution-surface-reachability: {len(failures)} finding(s).", file=sys.stderr)
        return 1

    print(
        "execution-surface-reachability: PASS — every gated execution-surface tuple is reachable on "
        "the merge path or carries a live allowlist row."
    )
    return 0


# --------------------------------------------------------------------------- #
# self-test — RED mutants for every rule, ephemeral `git init`'d fixtures,
# never the real checkout.
# --------------------------------------------------------------------------- #
def _write_repo(tmp: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp, check=True)


GATED_SCRIPT = """#!/usr/bin/env bash
# Documentation only, NOT a real invocation — must not be registered:
#   cargo clippy -p demo-doc-only --features cuda -- -D warnings
cargo clippy -p demo --all-targets --features cuda -- -D warnings || exit 1
cargo test -p demo --no-run || exit 1
"""

MERGE_PATH_WORKFLOW_REACHABLE = """name: fixture-ci
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

MERGE_PATH_WORKFLOW_UNREACHABLE = """name: fixture-ci
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: echo "no cargo cuda invocation here"
"""

MERGE_PATH_WORKFLOW_COMMENT_ONLY = """name: fixture-ci
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      # cargo clippy -p demo --all-targets --features cuda -- -D warnings
      - run: echo "the line above is a COMMENT, not a step body"
"""

MERGE_PATH_WORKFLOW_SUPERSET_ONLY = """name: fixture-ci
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda,flash-attn -- -D warnings
"""

LABEL_ONLY_WORKFLOW = """name: fixture-gpu-prove
on:
  workflow_dispatch:
  pull_request:
    types: [labeled]
  schedule:
    - cron: "0 0 * * *"
jobs:
  prove:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

PUSH_MAIN_WORKFLOW_REACHABLE = """name: fixture-push
on:
  push:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

TAG_ONLY_WORKFLOW = """name: fixture-release
on:
  push:
    tags: ["v*"]
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""


def _run_gate_in(tmp: Path, allowlist_text: str | None = None) -> tuple[list[str], list[str]]:
    allow_path = tmp / "ci" / "scripts" / "execution_surface_reachability_allowlist.txt"
    if allowlist_text is not None:
        allow_path.parent.mkdir(parents=True, exist_ok=True)
        allow_path.write_text(allowlist_text, encoding="utf-8")
    return run_gate(tmp, allow_path)


def self_test() -> int:
    failures: list[str] = []

    # --- Rule 2: registry completeness — a nested script under ci/scripts/
    # is discovered (recursion), and a doc-comment `cargo ...` line is NOT
    # registered as a real tuple. -------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_repo(
            tmp,
            {
                "ci/scripts/pods/pod_seed_target_v2.sh": GATED_SCRIPT,
                ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_UNREACHABLE,
            },
        )
        registry = discover_all_tuples(tmp)
        gated = gated_tuples(registry)
        if "cargo clippy -p demo --all-targets --features cuda -- -D warnings" not in gated:
            failures.append(
                "self-test FAILED: a gated tuple inside a NESTED ci/scripts/ subdirectory was not "
                f"discovered (Rule 2 recursion broken): {sorted(gated)}"
            )
        if "cargo clippy -p demo-doc-only --features cuda -- -D warnings" in registry:
            failures.append(
                "self-test FAILED: a `cargo ...` line living inside a `#` comment (documentation, "
                "never a real invocation) was registered as a real tuple"
            )
        if "cargo test -p demo --no-run" in gated:
            failures.append(
                "self-test FAILED: a non-gated (no --features cuda/flash-attn) invocation was "
                "classified as gated — Rule 2 must scope to GATED_FEATURE_TOKENS only"
            )

    # --- Rule 1: reachable via a genuine pull_request-to-main -> PASS ------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_repo(
            tmp,
            {
                "ci/scripts/pod_seed_target.sh": GATED_SCRIPT,
                ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_REACHABLE,
            },
        )
        got, _info = _run_gate_in(tmp)
        if got:
            failures.append(f"self-test FAILED: a genuinely reachable tuple was flagged: {got}")

    # --- Rule 1: reachable via a genuine push-to-main -> PASS --------------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_repo(
            tmp,
            {
                "ci/scripts/pod_seed_target.sh": GATED_SCRIPT,
                ".github/workflows/fixture-push.yml": PUSH_MAIN_WORKFLOW_REACHABLE,
            },
        )
        got, _info = _run_gate_in(tmp)
        if got:
            failures.append(f"self-test FAILED: a tuple reachable via push-to-main was flagged: {got}")

    # --- Rule 1: unreachable, no allowlist row -> FAIL (RED) ---------------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_repo(
            tmp,
            {
                "ci/scripts/pod_seed_target.sh": GATED_SCRIPT,
                ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_UNREACHABLE,
            },
        )
        got, _info = _run_gate_in(tmp)
        if not any("UNREACHABLE gated tuple" in g for g in got):
            failures.append(f"self-test FAILED: an unreachable, unallowlisted gated tuple not caught: {got}")

    # --- Rule 1: unreachable because the ONLY workflow mentioning it is
    # label/dispatch/schedule-only (gpu-prove.yml's own shape) -> FAIL ------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_repo(
            tmp,
            {
                "ci/scripts/pod_seed_target.sh": GATED_SCRIPT,
                ".github/workflows/fixture-gpu-prove.yml": LABEL_ONLY_WORKFLOW,
            },
        )
        got, _info = _run_gate_in(tmp)
        if not any("UNREACHABLE gated tuple" in g for g in got):
            failures.append(
                f"self-test FAILED: a tuple whose only workflow twin lives behind a label/dispatch/"
                f"schedule-only `on:` block was not flagged unreachable (the exact esc-050/051 "
                f"escape shape): {got}"
            )

    # --- Rule 1: a tag-push-only workflow does not count as push-to-main ---
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_repo(
            tmp,
            {
                "ci/scripts/pod_seed_target.sh": GATED_SCRIPT,
                ".github/workflows/fixture-release.yml": TAG_ONLY_WORKFLOW,
            },
        )
        got, _info = _run_gate_in(tmp)
        if not any("UNREACHABLE gated tuple" in g for g in got):
            failures.append(f"self-test FAILED: a tag-push-only (`push: tags:`) workflow was credited as merge-path: {got}")

    # --- Rule 1: a comment-only mention does not count ----------------------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_repo(
            tmp,
            {
                "ci/scripts/pod_seed_target.sh": GATED_SCRIPT,
                ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_COMMENT_ONLY,
            },
        )
        got, _info = _run_gate_in(tmp)
        if not any("UNREACHABLE gated tuple" in g for g in got):
            failures.append(f"self-test FAILED: a tuple named only in a `#` comment was credited as reachable: {got}")

    # --- Rule 1: exact-tuple match, never substring (esc-051's control) ----
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_repo(
            tmp,
            {
                "ci/scripts/pod_seed_target.sh": GATED_SCRIPT,
                ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_SUPERSET_ONLY,
            },
        )
        got, _info = _run_gate_in(tmp)
        if not any("UNREACHABLE gated tuple" in g for g in got):
            failures.append(
                "self-test FAILED: a `--features cuda,flash-attn` invocation satisfied a plain "
                f"`--features cuda` registered tuple (substring/superset match, esc-051's own bug "
                f"class) — must be an exact miss: {got}"
            )

    # --- Rule 1 + residual: unreachable, but allowlisted -> PASS -----------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        allow_text = (
            "cargo clippy -p demo --all-targets --features cuda -- -D warnings\t"
            "fixture: no merge-path CUDA toolchain lane exists; gpu-prove.yml is label/dispatch/"
            "schedule-only (GPU_PROVE_PROMOTED_TO_REQUIRED is False)\n"
        )
        _write_repo(
            tmp,
            {
                "ci/scripts/pod_seed_target.sh": GATED_SCRIPT,
                ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_UNREACHABLE,
            },
        )
        got, _info = _run_gate_in(tmp, allowlist_text=allow_text)
        if got:
            failures.append(f"self-test FAILED: a properly-allowlisted, reasoned, still-live tuple was flagged: {got}")

    # --- Rule 3: waiver rot — allowlist row cites a tuple that no longer
    # exists in the registry (renamed/deleted script) -> FAIL ---------------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        allow_text = "cargo clippy -p a-tuple-that-was-renamed --features cuda -- -D warnings\tstale reason\n"
        _write_repo(
            tmp,
            {
                "ci/scripts/pod_seed_target.sh": GATED_SCRIPT,
                ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_UNREACHABLE,
            },
        )
        got, _info = _run_gate_in(tmp, allowlist_text=allow_text)
        if not any("ROT" in g for g in got):
            failures.append(f"self-test FAILED: a rotted allowlist row (subject no longer in the registry) not caught: {got}")

    # --- Rule 3: an allowlist row missing a reason is a parse failure ------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        allow_text = "cargo clippy -p demo --all-targets --features cuda -- -D warnings\t\n"
        _write_repo(
            tmp,
            {
                "ci/scripts/pod_seed_target.sh": GATED_SCRIPT,
                ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_UNREACHABLE,
            },
        )
        got, _info = _run_gate_in(tmp, allowlist_text=allow_text)
        if not any("non-empty tuple text and a non-empty reason" in g for g in got):
            failures.append(f"self-test FAILED: an allowlist row with an empty reason not caught: {got}")

    # --- Rule 3: a malformed row (no TAB separator) is a parse failure ---
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        allow_text = "cargo clippy -p demo --features cuda -- -D warnings NO SEPARATOR HERE\n"
        _write_repo(
            tmp,
            {
                "ci/scripts/pod_seed_target.sh": GATED_SCRIPT,
                ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_UNREACHABLE,
            },
        )
        got, _info = _run_gate_in(tmp, allowlist_text=allow_text)
        if not any("malformed row" in g for g in got):
            failures.append(f"self-test FAILED: a malformed allowlist row (no TAB separator) not caught: {got}")

    # --- on: block parsing: workflow_dispatch/schedule alone is never
    # merge-path ------------------------------------------------------------
    dispatch_only = parse_on_block("on:\n  workflow_dispatch:\n  schedule:\n    - cron: \"0 0 * * *\"\n")
    ok, _reason = is_merge_path(dispatch_only)
    if ok:
        failures.append("self-test FAILED: workflow_dispatch/schedule alone was classified as merge-path")

    if failures:
        print("execution-surface-reachability self-test: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        "execution-surface-reachability self-test: OK — every rule bites: registry recursion + "
        "doc-comment exclusion + non-gated exclusion (Rule 2), reachable-via-pull_request-to-main, "
        "reachable-via-push-to-main, unreachable/unallowlisted, label-only-workflow unreachable "
        "(the exact esc-050/051 shape), tag-only-push not credited, comment-only mention not "
        "credited, exact-match-never-substring (esc-051's own control), allowlisted-and-live PASS, "
        "rotted allowlist row, empty-reason row, and a malformed row (Rule 3)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
