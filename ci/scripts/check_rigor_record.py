#!/usr/bin/env python3
"""check_rigor_record.py — R7b (esc-lead-gate-R7, v2/REFINE): a required
check that a lead cannot satisfy by under-reporting, because it never reads
anything the lead authors to decide WHETHER it applies.

Two design rounds killed the earlier shapes of this idea (see
`docs/plans/53-agentic-swarm/proposals/R7-committed-rigor-record.md` for the
full history, citations, and the measurements below). The one property that
survived every round: **arm on the DIFF's own shape, never on a count or a
flag the constrained party writes about itself.**

THE RULE. If `base...HEAD` (a three-dot range, never two-dot — see "Known,
undetectable residual" below) touches any path under `crates/**`, `ci/**`,
or `.github/workflows/**`, this check is ARMED for the unit on this branch
and requires, from COMMITTED files only:

  1. A rigor record at `docs/rigor/<unit_slug>.jsonl` — the hook's own row
     schema (`lead-gate-lib.py --export <slug>`), never hand-typed — exists
     and parses.
  2. That record carries at least one row with `agent_type ==
     "pressure-tester"` — the design pass `.claude/agents/lead.md:41`
     already requires before any non-local fix, now visible in a committed
     diff a human reviews at merge, not merely asserted in a transcript.
  3. This SAME diff (`base...HEAD`) adds or modifies at least one file
     under `docs/plans/*/proposals/**` or `docs/rigor/contracts/**` — the
     committed mechanism contract itself.
  4. That contract passes the four ANTI-VACUITY cost-floor checks below.
     Nothing mechanical judges a contract's QUALITY — this repo states
     that plainly rather than implying otherwise.

`unit_slug` is `slugify(<head branch name>)`, the SAME `slugify()`
`lead-gate-lib.py` already uses (loaded dynamically from
`.claude/hooks/lead-gate-lib.py` — never reimplemented, so the two never
drift) — the PR's own head ref, read from `GITHUB_HEAD_REF` in Actions or
`git rev-parse --abbrev-ref HEAD` otherwise. This is the SAME identity a
verifier's `unit_branch` field already resolves to everywhere else in this
system; a round of adversarial replay over 473 real hook decisions found
gated verifier re-dispatches bind to it at ~98% (cited in the proposal
doc), the most reliable identity this system has.

ARMED BY THE DIFF, SATISFIED BY DISCLOSURE, JUDGED BY THE HUMAN.
TAMPER-EVIDENT, NEVER TAMPER-PROOF — never described any other way.
`.jammi/gate-state/` (the source the rigor record is exported FROM)
carries no `permissions.deny` entry and the lead already writes into it by
design — a hand-appended row is possible and undetectable at the hook
layer (see `README.md`'s mechanical-vs-visible-only list, item 5). What
this check adds is not tamper-PREVENTION: it is that the record becomes a
COMMITTED file in a reviewed diff, so a human
merging the PR can see "this unit touched a mechanism; here is its design
pass; here is its round history" — or see that one of those three is
silently missing, which a gitignored state directory could never show
them at all.

ANCESTRY IS ADVISORY, NEVER A FAILURE (check 3, historical numbering).
Every recorded `head_sha` in the rigor record is checked for whether it
resolves and is an ancestor of HEAD, using the exact git shape
`_relay_rejection` already uses in `lead-gate-lib.py` — but ONLY reported,
never failed on. Measured, twice, independently: this session's own
documented workflow (amend an implementer's commit to add session
trailers) orphans every head_sha a verifier recorded before the amend —
`git merge-base --is-ancestor <pre-amend sha> HEAD` exits 1 on the
honest path, and in a transport clone (what `actions/checkout` produces)
the object is ABSENT entirely. `_relay_rejection`'s identical shape only
works because it runs in a local repository whose object store still
holds the orphan; a CI checkout has no such luxury.

KNOWN, STRUCTURAL NO-OP SHAPES (never an author-declared marker/trailer/
commit-message convention — every such predicate is a one-line opt-out):
  - not a `pull_request` event (a push to `main` with no PR context) —
    `GITHUB_EVENT_NAME` unset or not `pull_request`.
  - the PR's base is not the default branch — `swarm.yml`'s own
    `on: pull_request: branches: [main]` trigger already guarantees this
    whenever the workflow runs at all; checked again here defensively.
  - dependabot — `GITHUB_ACTOR == "dependabot[bot]"`.
  - a docs-only/typo diff — implied structurally: a diff touching none of
    `crates/**`/`ci/**`/`.github/workflows/**` never arms in the first
    place; no special case is needed.
  - a release-shaped diff — every changed file under `crates/**` is one of
    `Cargo.toml`/`Cargo.lock` and no `ci/**`/`.github/workflows/**` file is
    touched (a version-bump-only PR). Stated as a heuristic, not a proof:
    a release PR that ALSO edits a gate script is not exempted by this
    rule, and should not be.
  - a purely human-authored PR — none of the commits in `base..HEAD`
    carries this repo's own `Co-Authored-By: Claude` trailer convention.
    A single swarm-touched commit in the range is enough to arm; the
    swarm's own rigor process does not bind a human contributor who never
    touched the branch.
  - a revert — the PR's HEAD commit subject starts with `Revert "`, git's
    own `git revert` convention.

KNOWN, UNDETECTABLE RESIDUAL, STATED HONESTLY: a retargeted stacked branch
whose three-dot range still carries a parent unit's own commits (the base
this PR's history actually diverged from is stale relative to the PR's
declared `base_ref`) is NOT detectable from the diff alone — three-dot
ranges are the standard, correct GitHub PR semantics, and there is no
mechanical way to distinguish "this diff correctly represents the PR's own
work" from "this branch needs a rebase" without knowing the intended base.
Pin three-dot ranges explicitly (done); do not pretend this residual is
closed.

Modes:
  python3 ci/scripts/check_rigor_record.py                       # the check
  python3 ci/scripts/check_rigor_record.py --check-allowlist-only-shrinks
  python3 ci/scripts/check_rigor_record.py --check-r12-grandfather-only-shrinks
  python3 ci/scripts/check_rigor_record.py --check-required-commands-only-shrinks
  python3 ci/scripts/check_rigor_record.py --self-test
"""
from __future__ import annotations

import ast
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEAD_GATE_LIB = REPO_ROOT / ".claude" / "hooks" / "lead-gate-lib.py"
RIGOR_DIR = REPO_ROOT / "docs" / "rigor"

# --------------------------------------------------------------------------- #
# rust symbol index (ci/tools/symbol-index -- a real `syn` AST parse). This
# repo's convention (see check_plan_citations.py / check_no_consumer_names.py,
# its first two consumers) is a small, independently-maintained COPY of this
# one function per CI script that needs it, never a cross-script import — the
# tool's own module doc names this file as its third intended consumer
# (issue #557 item 2, the AST-derived required call-site set).
# --------------------------------------------------------------------------- #

SYMBOL_INDEX_CRATE = "symbol-index"


def build_symbol_index(roots: list[str], cwd: Path = REPO_ROOT) -> dict:
    """Runs the REAL `symbol-index` tool over `roots` and returns the parsed
    JSON index (`{"items": [...], "calls": [...], ...}`). Raises
    `RuntimeError` on a non-zero exit or unparseable stdout — never returns a
    partial/guessed index."""
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", SYMBOL_INDEX_CRATE, "--", *roots],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{SYMBOL_INDEX_CRATE} failed (rc={proc.returncode}) over {roots}:\n"
            f"{proc.stderr.strip()[-4000:]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{SYMBOL_INDEX_CRATE} did not emit valid JSON on stdout ({exc}); "
            f"stderr tail:\n{proc.stderr.strip()[-2000:]}"
        ) from exc
ALLOWLIST_PATH = REPO_ROOT / "ci" / "scripts" / "rigor_record_allowlist.txt"
# esc-lead-gate-R12 (M3'): in-flight units whose second-round BLOCK predates
# fix round 1's anticipation mechanism land without a
# `docs/rigor/<slug>.anticipation.jsonl` record. This is the SAME
# shrink-only-ratchet SHAPE as `ALLOWLIST_PATH`/`check_allowlist_only_
# shrinks` above (modelled on it directly, per swarm.yml's own two-step
# pairing) — a SEPARATE file and a SEPARATE check, never folded into the
# general allowlist, because the two exemptions arm on different diffs
# (this one is read only by `check_anticipation_witnesses`, never by
# `run_check`'s own top-level arming).
R12_GRANDFATHER_PATH = REPO_ROOT / "ci" / "scripts" / "rigor_record_r12_grandfather.txt"
# esc-lead-gate-R12 fix round 5 Z4: the OPPOSITE polarity from the two
# exemption lists above — this file names REQUIRED gate commands, so
# GROWING it (adding a line) tightens the swarm and ADDING/keeping every
# existing line is fine; REMOVING a line (a shrink in the count of
# required commands) WEAKENS item 8a and is exactly what this ratchet
# catches and fails, never allows silently.
REQUIRED_COMMANDS_PATH = REPO_ROOT / "ci" / "lead-gate-required-commands.txt"

ARMING_GLOBS = ("crates/*", "crates/**", "ci/*", "ci/**", ".github/workflows/*", ".github/workflows/**")
CONTRACT_GLOBS = ("docs/plans/*/proposals/*", "docs/plans/*/proposals/**", "docs/rigor/contracts/*", "docs/rigor/contracts/**")
# A file-extension-bearing path token followed by `:<line>[-<line>]` — the
# same grammar family `_probe_path`/`_PROBE_LINESPEC_RE` in lead-gate-lib.py
# already uses for citation-checkable evidence, reused rather than
# reinvented (never a length/word-count rule — Goodhart, and it is
# prose-reading by another name).
PATH_LINE_RE = re.compile(r"`?([A-Za-z0-9_./-]+\.[A-Za-z0-9]+):(\d+)(?:-(\d+))?`?")


class Result:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok(self) -> bool:
        return not self.failures


def _lib_module():
    spec = importlib.util.spec_from_file_location("lead_gate_lib_rigor", LEAD_GATE_LIB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _git(cwd: Path, *args: str) -> tuple[bool, str]:
    proc = subprocess.run(["git", "-C", str(cwd)] + list(args), capture_output=True, text=True)
    return proc.returncode == 0, (proc.stdout if proc.returncode == 0 else proc.stderr).strip()


def _display_path(p: Path) -> str:
    """`p` relative to REPO_ROOT when it is under it (the real invocation);
    otherwise the path as-is (a self-test fixture's own throwaway tree,
    which is never under REPO_ROOT)."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _matches_arming_glob(path: str) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in ARMING_GLOBS)


def _matches_contract_glob(path: str) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in CONTRACT_GLOBS)


def compute_diff_context(cwd: Path, base_ref: str, head_ref: str = "HEAD") -> tuple[bool, list[str], str]:
    """Fetches `base_ref` from `origin` and returns `(ok, changed_paths,
    range_spec)` for `origin/<base_ref>...<head_ref>` — a THREE-DOT range,
    never two-dot (the range this whole mechanism is pinned to)."""
    ok, _ = _git(cwd, "fetch", "--quiet", "origin", base_ref)
    if not ok:
        return False, [], ""
    ok, _ = _git(cwd, "rev-parse", "--verify", f"origin/{base_ref}")
    if not ok:
        return False, [], ""
    range_spec = f"origin/{base_ref}...{head_ref}"
    ok, out = _git(cwd, "diff", "--name-only", range_spec)
    if not ok:
        return False, [], range_spec
    return True, [p for p in out.splitlines() if p.strip()], range_spec


def commits_in_range(cwd: Path, range_spec: str) -> list[str]:
    ok, out = _git(cwd, "log", "--format=%H", range_spec.replace("...", ".."))
    return [c for c in out.splitlines() if c.strip()] if ok else []


def is_human_authored(cwd: Path, range_spec: str) -> bool:
    """True iff NO commit in the range carries this repo's own swarm
    co-authorship trailer — a single swarm-touched commit is enough to
    arm; a purely human PR that never touched the branch is not bound by
    the swarm's own rigor process."""
    shas = commits_in_range(cwd, range_spec)
    if not shas:
        return True
    for sha in shas:
        ok, msg = _git(cwd, "log", "-1", "--format=%B", sha)
        if ok and re.search(r"Co-Authored-By:\s*Claude", msg, re.IGNORECASE):
            return False
    return True


def is_revert(cwd: Path, head_ref: str = "HEAD") -> bool:
    ok, subject = _git(cwd, "log", "-1", "--format=%s", head_ref)
    return ok and subject.startswith('Revert "')


def is_release_shaped(changed: list[str]) -> bool:
    crate_touches = [p for p in changed if _matches_arming_glob(p) and p.startswith("crates/")]
    other_mechanism_touches = [
        p for p in changed
        if _matches_arming_glob(p) and not p.startswith("crates/")
    ]
    if other_mechanism_touches:
        return False
    if not crate_touches:
        return False
    return all(Path(p).name in ("Cargo.toml", "Cargo.lock") for p in crate_touches)


def _no_op_reason(cwd: Path, changed: list[str], range_spec: str) -> str | None:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    if event and event != "pull_request":
        return f"not a pull_request event ({event!r})"
    actor = os.environ.get("GITHUB_ACTOR", "")
    if actor == "dependabot[bot]":
        return "dependabot PR"
    if is_revert(cwd):
        return "revert PR (HEAD commit subject starts with Revert \")"
    if is_release_shaped(changed):
        return "release-shaped diff (only Cargo.toml/Cargo.lock under crates/**, no ci/**/.github/workflows/** touch)"
    if is_human_authored(cwd, range_spec):
        return "no commit in range carries the swarm's own Co-Authored-By: Claude trailer"
    return None


# --- Cost-floor (anti-vacuity) checks -----------------------------------

def _path_lines_at_head(cwd: Path, path: str) -> int | None:
    ok, out = _git(cwd, "show", f"HEAD:{path}")
    if not ok:
        return None
    return len(out.splitlines())


def check_path_line_citations(cwd: Path, contract_path: str, text: str, result: Result) -> None:
    seen = set()
    for m in PATH_LINE_RE.finditer(text):
        cited_path, line1, line2 = m.group(1), int(m.group(2)), m.group(3)
        key = (cited_path, line1, line2)
        if key in seen:
            continue
        seen.add(key)
        n = _path_lines_at_head(cwd, cited_path)
        if n is None:
            result.fail(f"{contract_path}: cites {cited_path}:{line1}"
                        f"{'-' + line2 if line2 else ''} but {cited_path} does not exist at HEAD")
            continue
        top = int(line2) if line2 else line1
        if top > n:
            result.fail(f"{contract_path}: cites {cited_path}:{top} but {cited_path} has only {n} line(s) at HEAD")


def check_introducing_commit_ancestor(cwd: Path, contract_path: str, result: Result) -> None:
    ok, out = _git(cwd, "log", "--follow", "--diff-filter=A", "--format=%H", "--", contract_path)
    if not ok or not out.strip():
        result.warn(f"{contract_path}: could not find an introducing (add) commit via --follow")
        return
    introducing = out.strip().splitlines()[-1]
    ok, _ = _git(cwd, "merge-base", "--is-ancestor", introducing, "HEAD")
    if not ok:
        result.fail(f"{contract_path}: introducing commit {introducing} is not an ancestor of HEAD")


def _normalize_for_dedup(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def check_not_near_identical(cwd: Path, contract_path: str, text: str, all_contracts: list[str], result: Result) -> None:
    my_hash = hashlib.sha256(_normalize_for_dedup(text).encode("utf-8")).hexdigest()
    for other in all_contracts:
        if other == contract_path:
            continue
        ok, other_text = _git(cwd, "show", f"HEAD:{other}")
        if not ok:
            continue
        other_hash = hashlib.sha256(_normalize_for_dedup(other_text).encode("utf-8")).hexdigest()
        if other_hash == my_hash:
            result.fail(f"{contract_path}: near-identical (normalized-hash match) to {other}"
                        " — a copy-paste contract is not a design pass")


def _r12_grandfathered_slugs() -> set[str]:
    if not R12_GRANDFATHER_PATH.exists():
        return set()
    return {s.strip() for s in R12_GRANDFATHER_PATH.read_text().splitlines()
            if s.strip() and not s.strip().startswith("#")}


def _r12_second_round_block_rows(mod, rows: list[dict]) -> list[dict]:
    """Every row in the EXPORTED unit record whose `agent_type` is one of
    the hook's own `VERIFIER_SECOND_ROUND_TYPES` and whose `verdict` is
    BLOCK-equivalent (`is_open`) — the SAME closed-world membership the
    hook itself uses for reader 1, loaded from the real module, never
    re-derived here."""
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("agent_type") not in mod.VERIFIER_SECOND_ROUND_TYPES:
            continue
        verdict = r.get("verdict")
        if isinstance(verdict, str) and mod.is_open(verdict):
            out.append(r)
    return out


def _r12_reject_foreign_anticipation_rows(path: str, rows: list[tuple[int, dict]],
                                           result: Result) -> list[tuple[int, dict]]:
    """esc-lead-gate-R12 fix round 6 Z12: PROPERTY — a committed
    `docs/rigor/<slug>.anticipation.jsonl` carries EXACTLY ONE row kind,
    `agent_type == "lead-anticipation"`. `cmd_export_anticipation` once
    also emitted the lead's OWN mutations/exclusions attestations
    (`agent_type == "lead-relay-attestation"`) into this SAME stdout
    stream the operator redirects into this SAME file; neither reader-3
    call site filtered by `agent_type`, so an attestation row (no
    `residual_risk`, no `gates`) could become the "governing" row `check_
    required_gates` selects (hiding a real `gates` object behind "no
    `gates` object"), or simply deny `check_anticipation_witnesses`
    outright ("no non-empty `residual_risk`"). The exporter's own
    attestation-row half was later reverted entirely; `mutations`/
    `exclusions` are now exported by a SEPARATE command
    (`--export-attestation`) into a SEPARATE committed stream
    (`docs/rigor/<slug>.attestation.jsonl`, read by `check_attestation_
    witnesses`, never this function) — never folded back into THIS
    stream. A `lead-relay-attestation` row found HERE is therefore always
    a FOREIGN row for THIS stream — a stale pre-revert export still
    committed, a row exported into the wrong file by hand, or a
    hand-edit. This function REFUSES it with a loud, NAMED FAIL — never
    silently ignores it (which would let a tampered/legacy row go
    undetected) and never silently SELECTS it (the earlier bug this
    replaces) — and returns only the rows that pass the filter for the
    caller's own downstream checks."""
    kept: list[tuple[int, dict]] = []
    for lineno, row in rows:
        agent_type = row.get("agent_type")
        if agent_type != "lead-anticipation":
            result.fail(
                f"{path}:{lineno}: row carries agent_type={agent_type!r}, not "
                "`lead-anticipation` -- a foreign row (e.g. a `lead-relay-attestation` row -- "
                "mutations/exclusions export into docs/rigor/<slug>.attestation.jsonl instead, "
                "never here) does not belong in the anticipation stream; it is REFUSED here, "
                "never ignored and never selected as governing (esc-lead-gate-R12 fix round 6 Z12)")
            continue
        kept.append((lineno, row))
    return kept


def check_anticipation_witnesses(cwd: Path, unit_slug: str, rows: list[dict], result: Result) -> None:
    """esc-lead-gate-R12 READER 3 (fix round 1, M3'): a committed-record
    SHAPE check, HARD-FAILED on shape alone. ARMED whenever the exported
    `docs/rigor/<unit_slug>.jsonl` carries ANY open second-round BLOCK row
    (`_r12_second_round_block_rows`) and `unit_slug` is NOT on the
    shrink-only `R12_GRANDFATHER_PATH` list — never on a ts cutover (M3'
    drops that; the grandfather list is the ONLY exemption, and it may only
    shrink, enforced by `check_r12_grandfather_only_shrinks`). When armed:
    `docs/rigor/<unit_slug>.anticipation.jsonl` (exported via
    `lead-gate-lib.py --export-anticipation`) must exist, every row must
    parse, and the UNION of every second-round BLOCK row's own
    `finding_locations` ∪ `class_enumeration`, reduced to files
    (`mod._key_to_file`), must be covered by SOME row's `attacks` keys —
    every command passing the hook's OWN denylist
    (`mod._r12_attack_command_denied`, loaded from the real
    `lead-gate-lib.py`, never reimplemented — `bash <path>` admitted only
    when TRACKED at the row's own `pre_fix_sha`). Re-EXECUTION against a
    REAL, independently checked-out `pre_fix_sha` is ADVISORY ONLY —
    reported per attack as reproduced / mismatched / not_re_executed, NEVER
    a failure (measured: BSD/GNU userland divergence between the lead's
    machine and CI's, and path divergence, both change a witness hash
    without the mechanism under attack having changed at all)."""
    mod = _lib_module()
    block_rows = _r12_second_round_block_rows(mod, rows)
    if not block_rows:
        return  # nothing to anticipate -- no open second-round BLOCK in this record

    path = f"docs/rigor/{unit_slug}.anticipation.jsonl"
    ok, text = _git(cwd, "show", f"HEAD:{path}")
    has_file = ok and bool(text.strip())

    if unit_slug in _r12_grandfathered_slugs():
        if not has_file:
            result.warn(f"{unit_slug!r} is on the shrink-only R12 grandfather list "
                        f"({_display_path(R12_GRANDFATHER_PATH)}) — no {path} carried; not required")
            return
    elif not has_file:
        result.fail(
            f"docs/rigor/{unit_slug}.jsonl carries {len(block_rows)} open second-round BLOCK "
            f"row(s) but no {path} — export one with `python3 .claude/hooks/lead-gate-lib.py "
            f"--export-anticipation {unit_slug} > {path}` and commit it (esc-lead-gate-R12 "
            "READER 3); an in-flight unit whose BLOCK predates fix round 1 is exempted only via "
            f"{_display_path(R12_GRANDFATHER_PATH)} (shrink-only, human-added)")
        return

    if not has_file:
        return

    required_files: set[str] = set()
    for row in block_rows:
        locs = {s for s in (row.get("finding_locations") or []) if isinstance(s, str)}
        enum = {s for s in (row.get("class_enumeration") or []) if isinstance(s, str)}
        for key in locs | enum:
            required_files.add(mod._key_to_file(key))

    art_rows: list[tuple[int, dict]] = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            result.fail(f"{path}:{i + 1}: not valid JSON ({exc})")
            continue
        if not isinstance(parsed, dict):
            result.fail(f"{path}:{i + 1}: not a JSON object")
            continue
        art_rows.append((i + 1, parsed))

    # esc-lead-gate-R12 fix round 6 Z12: PROPERTY -- this stream carries
    # exactly ONE row kind (`agent_type == "lead-anticipation"`). A
    # foreign row is REFUSED here, loudly and by name, never ignored and
    # never selected.
    art_rows = _r12_reject_foreign_anticipation_rows(path, art_rows, result)

    if not art_rows:
        result.fail(f"{path}: carries no parseable anticipation row")
        return

    covered_files: set[str] = set()
    for lineno, row in art_rows:
        attacks = row.get("attacks")
        pre_fix_sha = row.get("pre_fix_sha")
        if not isinstance(attacks, dict) or not attacks:
            result.fail(f"{path}:{lineno}: no non-empty `attacks` object")
            continue
        if not isinstance(pre_fix_sha, str) or not pre_fix_sha:
            result.fail(f"{path}:{lineno}: no `pre_fix_sha`")
        sha_resolvable = False
        if isinstance(pre_fix_sha, str) and pre_fix_sha:
            ok_sha, _ = _git(cwd, "rev-parse", "--verify", f"{pre_fix_sha}^{{commit}}")
            sha_resolvable = ok_sha
            if not ok_sha:
                result.warn(f"{path}:{lineno}: pre_fix_sha {pre_fix_sha} does not resolve in this "
                            "checkout (advisory — a shallow clone or an amend can cause this; "
                            "re-execution below is reported not_re_executed)")
        # Fix round 6 Z14: `entry` shape (an object, a non-empty `command`,
        # a valid 64-hex `hash`) is NEVER re-checked here — that is
        # EXACTLY the shared validator's own job (`_r12_anticipation_
        # rejection`, called once below over the whole `art_rows` union),
        # and a second, independently maintained copy of those three checks
        # is precisely how this file's own earlier drift went uncaught: a
        # hand-rolled duplicate that happens to agree with the shared
        # function is not a second examination, and neutering the SHARED
        # arm at that point killed nothing because THIS loop's own copy
        # had already produced the identical `result.fail` first, on every
        # input. `_r12_attack_command_denied` (the denylist re-check) is
        # the ONE thing this loop still does that the shared validator
        # does not — it needs a real, dict-shaped entry with a real string
        # `command` to run at all, so it silently skips a malformed entry
        # here (the shared call below is what NAMES the malformed shape).
        for file_key, entry in attacks.items():
            covered_files.add(file_key)
            if not isinstance(entry, dict):
                continue
            command = entry.get("command")
            if not (isinstance(command, str) and command.strip()):
                continue
            deny = mod._r12_attack_command_denied(
                command, str(cwd),
                require_tracked_at=pre_fix_sha if sha_resolvable else None,
                project_dir=str(cwd), deadline=time.monotonic() + 5.0)
            if deny is not None:
                result.fail(f"{path}:{lineno}: attacks[{file_key!r}] command is denied: {deny}")

    # Fix round 5 Z7: the omits-a-command arm (below, `required_files`
    # threaded through for real), `unit_branch`/`residual_risk` presence,
    # pair-reuse denial and the execution-class requirement all run
    # through the ONE shared validator reader 1 (the hook) calls — never
    # a second, independently maintained implementation (the OLD inline
    # `missing = required_files - covered_files` check this replaces was
    # exactly that: a duplicate that happened to agree with the shared
    # function, which is exactly how this file's own earlier gaps
    # (accepting an all-inspector record, a reused (command, hash) pair,
    # a row with no `residual_risk`/`unit_branch` at all) went uncaught).
    shape_why = mod._r12_anticipation_rejection([r for _, r in art_rows], [], check_attacks=True,
                                                 required_files=required_files)
    if shape_why is not None:
        result.fail(f"{path}: {shape_why}")

    # Re-execution against a REAL, independently checked-out `pre_fix_sha`
    # — ADVISORY ONLY, per M3': never a hard fail.
    for lineno, row in art_rows:
        pre_fix_sha = row.get("pre_fix_sha")
        attacks = row.get("attacks")
        if not (isinstance(pre_fix_sha, str) and pre_fix_sha and isinstance(attacks, dict)):
            continue
        ok_sha, _ = _git(cwd, "rev-parse", "--verify", f"{pre_fix_sha}^{{commit}}")
        if not ok_sha:
            result.warn(f"{path}:{lineno}: pre_fix_sha unreachable in this checkout — every attack "
                        "here is not_re_executed")
            continue
        with tempfile.TemporaryDirectory(prefix="r12-reader3-") as td:
            tmp_wt = Path(td) / "wt"
            ok_add, out_add = _git(cwd, "worktree", "add", "--detach", "-q", str(tmp_wt), pre_fix_sha)
            if not ok_add:
                result.warn(f"{path}:{lineno}: could not check out pre_fix_sha {pre_fix_sha} for "
                            f"advisory re-execution — {out_add} (not_re_executed)")
                continue
            try:
                for file_key, entry in sorted(attacks.items()):
                    if not isinstance(entry, dict):
                        continue
                    command = entry.get("command")
                    recorded_hash = entry.get("hash")
                    if not isinstance(command, str) or not command.strip():
                        continue
                    deny = mod._r12_attack_command_denied(
                        command, str(tmp_wt), require_tracked_at=pre_fix_sha,
                        project_dir=str(cwd), deadline=time.monotonic() + 5.0)
                    if deny is not None:
                        result.warn(f"{path}:{lineno}: attacks[{file_key!r}] not_re_executed "
                                    f"(denied in the detached checkout: {deny})")
                        continue
                    try:
                        proc = subprocess.run(["/bin/sh", "-c", command], cwd=str(tmp_wt),
                                               capture_output=True, text=True, timeout=120)
                    except subprocess.TimeoutExpired:
                        result.warn(f"{path}:{lineno}: attacks[{file_key!r}] not_re_executed "
                                    "(timed out)")
                        continue
                    stderr_lines = proc.stderr.splitlines()
                    stderr_line = stderr_lines[0] if stderr_lines else ""
                    actual_hash = mod._witness_hash(proc.returncode, proc.stdout, stderr_line)
                    if not (isinstance(recorded_hash, str) and actual_hash == recorded_hash):
                        result.warn(
                            f"{path}:{lineno}: attacks[{file_key!r}] mismatched at pre_fix_sha "
                            f"{pre_fix_sha} (advisory, never a failure — BSD/GNU userland and path "
                            "divergence between the lead's machine and CI's is a measured, known "
                            f"cause; recorded "
                            f"{recorded_hash[:12] if isinstance(recorded_hash, str) else recorded_hash}…, "
                            f"got {actual_hash[:12]}…)")
            finally:
                _git(cwd, "worktree", "remove", "--force", str(tmp_wt))


def _r12_reject_foreign_attestation_rows(path: str, rows: list[tuple[int, dict]],
                                          result: Result) -> list[tuple[int, dict]]:
    """The attestation stream's own Z12/Z18-shaped foreign-row refusal:
    every committed row's `kind` must be exactly `"lead-relay-
    attestation"` (`cmd_export_attestation`'s own stamp) — a row of any
    other shape (a hand-edit, a row copied from the anticipation stream)
    is REFUSED here, loudly and by name, never ignored and never selected
    as evidence."""
    kept: list[tuple[int, dict]] = []
    for lineno, row in rows:
        kind = row.get("kind")
        if kind != "lead-relay-attestation":
            result.fail(
                f"{path}:{lineno}: row carries kind={kind!r}, not `lead-relay-attestation` -- "
                "a foreign row does not belong in the attestation stream; it is REFUSED here, "
                "never ignored and never selected as evidence (issue #557 item 1)"
            )
            continue
        kept.append((lineno, row))
    return kept


def check_attestation_witnesses(cwd: Path, unit_slug: str, rows: list[dict],
                                 range_spec: str, result: Result) -> None:
    """issue #557 items 1-2, half 1 (the committed record) + half 2 (the
    CI-derived required set): `mutations`/`exclusions` (item 8b/8c) are
    LEAD-ATTESTED, visible to the hook at decision time from the
    (gitignored, ephemeral) relay artifact — never from anything this CI
    script can read. Reader 3 therefore never tries to see WHAT a relay
    attested; it RE-DERIVES, from the SAME two committed facts the hook's
    own arming conditions already key off (an open second-round BLOCK
    row's `finding_locations`/`class_enumeration`, and the fix's own
    `base...HEAD` diff), WHETHER an attestation was ever REQUIRED at all —
    then requires the committed `docs/rigor/<unit_slug>.attestation.jsonl`
    to carry real evidence for each requirement THAT RE-DERIVATION FINDS,
    never for a requirement it cannot see (the hook's own attack-budget-
    bounded per-relay scoping stays hook-side; this is a per-unit, whole-
    diff derivation, a stated widening, never a narrower re-check).

    Armed the SAME way `check_anticipation_witnesses` is (an open
    second-round BLOCK row, not on the shrink-only R12 grandfather list).
    `mutations` required iff the diff's own new-definition surfaces
    (`mod._parse_new_surfaces`, the SAME AST-derived enumeration item 8b
    itself uses) include one inside a file the open BLOCK(s)' own
    `finding_locations`/`class_enumeration` also name. `exclusions`
    required iff `mod._r12_new_test_surfaces` (the SAME subset item 8c
    uses) is non-empty. EITHER arming condition requires the committed
    file to exist and carry, across its rows, real shape-checked evidence
    for the arming(s) that fired — never a placeholder, never a file that
    exists but proves nothing (an empty stream, or a stream carrying only
    the OTHER field)."""
    mod = _lib_module()
    block_rows = _r12_second_round_block_rows(mod, rows)
    if not block_rows:
        return  # nothing to attest -- no open second-round BLOCK in this record

    by_file: set[str] = set()
    for row in block_rows:
        locs = {s for s in (row.get("finding_locations") or []) if isinstance(s, str)}
        enum = {s for s in (row.get("class_enumeration") or []) if isinstance(s, str)}
        for key in locs | enum:
            by_file.add(mod._key_to_file(key))

    ok_diff, diff_out = _git(cwd, "diff", "-U0", "--end-of-options", range_spec)
    new_surfaces = mod._parse_new_surfaces(diff_out) if ok_diff else {}
    new_surface_files = {mod._key_to_file(k) for k in new_surfaces}
    mutations_required = bool(by_file & new_surface_files)
    new_test_surfaces = mod._r12_new_test_surfaces(new_surfaces)
    exclusions_required = bool(new_test_surfaces)

    if not (mutations_required or exclusions_required):
        return  # armed by the BLOCK, but this diff never widens either 8b/8c set

    path = f"docs/rigor/{unit_slug}.attestation.jsonl"
    ok, text = _git(cwd, "show", f"HEAD:{path}")
    has_file = ok and bool(text.strip())

    if unit_slug in _r12_grandfathered_slugs():
        if not has_file:
            result.warn(f"{unit_slug!r} is on the shrink-only R12 grandfather list "
                        f"({_display_path(R12_GRANDFATHER_PATH)}) — no {path} carried; not required")
            return
    elif not has_file:
        requirement = "mutations" if mutations_required and not exclusions_required else (
            "exclusions" if exclusions_required and not mutations_required else "mutations and exclusions")
        result.fail(
            f"the fix's own diff ({requirement}) arms item 8b/8c's committed attestation "
            f"requirement but no {path} — export one with `python3 .claude/hooks/lead-gate-lib.py "
            f"--export-attestation {unit_slug} > {path}` and commit it (issue #557 items 1-2); an "
            f"in-flight unit whose BLOCK predates this mechanism is exempted only via "
            f"{_display_path(R12_GRANDFATHER_PATH)} (shrink-only, human-added)")
        return

    if not has_file:
        return

    art_rows: list[tuple[int, dict]] = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            result.fail(f"{path}:{i + 1}: not valid JSON ({exc})")
            continue
        if not isinstance(parsed, dict):
            result.fail(f"{path}:{i + 1}: not a JSON object")
            continue
        art_rows.append((i + 1, parsed))

    art_rows = _r12_reject_foreign_attestation_rows(path, art_rows, result)
    if not art_rows:
        result.fail(f"{path}: carries no parseable lead-relay-attestation row")
        return

    # Every row's own `mutations`/`exclusions` is judged through the SAME
    # shared shape functions the hook itself validated it against before
    # the relay was ever accepted (`mod._r12_mutations_array_rejection`/
    # `mod._r12_exclusions_shape_rejection`) — never a second,
    # independently maintained copy of those checks in THIS file (the
    # exact class the R12sweepast/RR31 detectors exist to catch). A row
    # that fails the shared check is reported (so a hand-edit-after-export
    # drift is visible) but does not by itself deny the whole check — only
    # the ABSENCE of any row that satisfies the requirement does.
    mutations_ok = False
    exclusions_ok = False
    for lineno, row in art_rows:
        if "mutations" in row:
            why = mod._r12_mutations_array_rejection(row["mutations"])
            if why is None:
                mutations_ok = True
            else:
                result.warn(f"{path}:{lineno}: mutations {why}")
        if "exclusions" in row:
            why = mod._r12_exclusions_shape_rejection(new_test_surfaces, row["exclusions"])
            if why is None:
                exclusions_ok = True
            else:
                result.warn(f"{path}:{lineno}: exclusions {why}")

    if mutations_required and not mutations_ok:
        result.fail(
            f"{path}: the fix's own diff adds a new definition in a file the open BLOCK's own "
            "finding_locations/class_enumeration also names (item 8b), but no committed row "
            "carries a shape-valid, non-empty `mutations` array"
        )
    if exclusions_required and not exclusions_ok:
        result.fail(
            f"{path}: the fix's own diff adds {len(new_test_surfaces)} new TEST definition(s) "
            f"(item 8c), e.g. {list(new_test_surfaces)[:3]}, but no committed row carries a "
            "shape-valid `exclusions` object covering all of them"
        )


_RR_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _rust_added_lines_by_file(cwd: Path, range_spec: str) -> dict[str, set[int]]:
    """`{path: {new_line_no, ...}}` for every ADDED (`+`) line in a `.rs`
    file under `git diff --unified=0 <range_spec>` — the SAME `+++
    b/<path>` / `@@ -a,b +c,d @@` tracking `check_no_consumer_names.py`'s
    `added_crate_lines_with_paths` already established, applied here to ANY
    `.rs` path (never scoped to `crates/` alone — a `mutations[].site` can
    legitimately name a file under `ci/tools/**` too) and returning a
    per-file LINE SET (never text) since `_r12_required_call_site_set`
    cross-references the symbol-index's OWN line numbers, not the diff's
    raw text a second time."""
    ok, diff_out = _git(cwd, "diff", "--unified=0", "--end-of-options", range_spec)
    if not ok:
        return {}
    out: dict[str, set[int]] = {}
    current_file: str | None = None
    new_line_no = 0
    for line in diff_out.splitlines():
        if line.startswith("+++ "):
            raw_path = line[len("+++ "):]
            if raw_path.startswith("b/"):
                raw_path = raw_path[2:]
            current_file = None if raw_path == "/dev/null" else raw_path
            continue
        if line.startswith("+++"):
            continue
        if line.startswith("@@ "):
            m = _RR_HUNK_HEADER_RE.match(line)
            if m:
                new_line_no = int(m.group(1))
            continue
        if line.startswith("+") and current_file is not None:
            if current_file.endswith(".rs"):
                out.setdefault(current_file, set()).add(new_line_no)
            new_line_no += 1
    return out


def _r12_required_call_site_set(cwd: Path, added: dict[str, set[int]]) -> tuple[set[str], dict]:
    """issue #557 item 2: the REQUIRED call-site set, derived from a REAL
    `syn` AST parse (`ci/tools/symbol-index`), never a regex reader and
    never a hand list — `{"path:line", ...}` covering:

      (a) every NEW non-test fn/method DEFINITION this diff's own added
          lines (`added`) introduce, and
      (b) every NEW call site this diff's own added lines add, whose
          callee's bare name matches a fn/method whose OWN pre-existing
          definition span this SAME diff also touches (`changed_fn_names`
          — a "changed fn", never a "new fn": (a) and (b) are disjoint
          categories of the ONE property, "this diff makes new or
          different code run").

    `index` (the raw symbol-index JSON, covering the FULL text of every
    scanned file, not just added lines) is returned alongside so a caller
    can also ask "does this path:line resolve to ANY real position at
    HEAD" (`_r12_site_resolves`, below) without a second `cargo run`.
    `roots` are the PARENT DIRECTORIES of `added`'s own files, de-
    duplicated — `symbol-index`'s own `walkdir` then covers every sibling
    file in the same directory too (a legitimate `mutations[].site` in an
    UNTOUCHED sibling file inside a touched directory still resolves), a
    stated, bounded widening, never the whole `crates/` tree."""
    if not added:
        return set(), {"items": [], "calls": []}
    roots = sorted({str((cwd / rel).parent) for rel in added})
    index = build_symbol_index(roots)

    def _rel(index_path: str) -> str | None:
        for rel in added:
            if index_path == rel or index_path.endswith("/" + rel):
                return rel
        return None

    fn_items = [it for it in index.get("items", [])
                if it.get("kind") == "fn" and not it.get("is_test")]
    new_defs: set[str] = set()
    changed_fn_names: set[str] = set()
    for it in fn_items:
        rel = _rel(it["path"])
        if rel is None:
            continue
        new_lines = added.get(rel, set())
        line, line_end = it["line"], it.get("line_end", it["line"])
        if line in new_lines:
            new_defs.add(f"{rel}:{line}")
        elif any(l in new_lines for l in range(line, line_end + 1)):
            changed_fn_names.add(it["name"])

    new_call_sites: set[str] = set()
    for c in index.get("calls", []):
        rel = _rel(c["path"])
        if rel is None or c.get("in_test"):
            continue
        new_lines = added.get(rel, set())
        if c["line"] in new_lines and c["callee"] in changed_fn_names:
            new_call_sites.add(f"{rel}:{c['line']}")

    return new_defs | new_call_sites, index


def _r12_site_resolves(site: str, index: dict) -> bool:
    """`True` iff `site` (a `mutations[].site` string, `"path:line"` per
    `_r12_mutations_array_rejection`'s own shape check) names a REAL
    position in `index` — either inside an item's own `[line, line_end]`
    span or exactly a call site's own `line` — matched by path SUFFIX
    (`index`'s own `path` is however `symbol-index` reported it,
    ABSOLUTE when `roots` were absolute; `site`'s own path is always the
    diff/repo-relative form `_r12_mutations_array_rejection` never
    normalizes). Never "the file exists" alone — that would accept ANY
    line number in a real file as if it named something; a phantom site
    is exactly a `path:line` this returns `False` for."""
    if ":" not in site:
        return False
    rel_path, _, line_txt = site.rpartition(":")
    if not line_txt.isdigit():
        return False
    line = int(line_txt)

    def _matches(index_path: str) -> bool:
        return index_path == rel_path or index_path.endswith("/" + rel_path)

    for it in index.get("items", []):
        if _matches(it["path"]) and it["line"] <= line <= it.get("line_end", it["line"]):
            return True
    for c in index.get("calls", []):
        if _matches(c["path"]) and c["line"] == line:
            return True
    return False


def check_required_call_site_set(cwd: Path, unit_slug: str, rows: list[dict],
                                  range_spec: str, result: Result) -> None:
    """issue #557 item 2 (Reader 3's AST-derived required call-site set):
    `mutations[].site` is LEAD-ATTESTED text a human cannot mechanically
    catch FABRICATION in merely by re-validating its JSON SHAPE
    (`_r12_mutations_array_rejection`, already run by
    `check_attestation_witnesses`, above — shape alone accepts
    `"site": "nowhere.rs:99999"` exactly as readily as a real one). This
    reader independently derives, from a REAL `syn` AST parse of the
    diff's own touched `.rs` files (`ci/tools/symbol-index`, never a
    regex reader and never a hand list), whether EACH committed
    `mutations[].site` resolves to a real definition or call site at
    HEAD — an unresolvable ("phantom") site is a hard FAIL, named by
    text, LAYERED ON TOP OF the shape check above, never a re-derivation
    of it.

    Armed only over rows whose `mutations` array ALREADY passes the
    shared shape check (never re-derives THAT arming a second time — the
    exact class RR31 exists to catch) and only over `.rs`-named sites (a
    Python `mutations[].site` stays covered by the shape check alone —
    this reader carries a Rust index only, stated, never silently
    widened to a language it cannot parse). A no-op whenever the diff
    touches no `.rs` file, or no committed row carries a shape-valid,
    non-empty `mutations` array at all."""
    mod = _lib_module()
    block_rows = _r12_second_round_block_rows(mod, rows)
    if not block_rows:
        return

    path = f"docs/rigor/{unit_slug}.attestation.jsonl"
    ok, text = _git(cwd, "show", f"HEAD:{path}")
    if not ok or not text.strip():
        return  # check_attestation_witnesses already reports a missing-when-required file

    art_rows: list[tuple[int, dict]] = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            art_rows.append((i + 1, parsed))
    quiet = Result()
    art_rows = _r12_reject_foreign_attestation_rows(path, art_rows, quiet)

    mutation_sites: list[tuple[int, str]] = []
    for lineno, row in art_rows:
        mutations = row.get("mutations")
        if mod._r12_mutations_array_rejection(mutations) is not None:
            continue  # not shape-valid -- check_attestation_witnesses already names this
        for entry in mutations:
            site = entry.get("site")
            if isinstance(site, str) and site.strip():
                mutation_sites.append((lineno, site))
    if not mutation_sites:
        return  # no shape-valid `mutations` row to examine -- nothing for this reader to add

    rust_sites = [(lineno, site) for lineno, site in mutation_sites
                  if site.rpartition(":")[0].endswith(".rs")]
    if not rust_sites:
        return  # every committed site names a non-Rust file

    added = _rust_added_lines_by_file(cwd, range_spec)
    if not added:
        return  # this diff touches no `.rs` file -- nothing to derive a required set from

    try:
        required_set, index = _r12_required_call_site_set(cwd, added)
    except RuntimeError as exc:
        result.warn(f"{path}: could not build the symbol-index required call-site set — {exc} "
                    "(advisory; the shape check above still applies)")
        return

    for lineno, site in rust_sites:
        if _r12_site_resolves(site, index):
            continue
        hint = sorted(required_set)[:3] or "(none — this diff adds no new Rust definition or " \
            "call site of a changed fn)"
        result.fail(
            f"{path}:{lineno}: mutations `site` {site!r} does not resolve to a real definition "
            f"or call site at HEAD (symbol-index, issue #557 item 2) — a phantom site; the "
            f"diff's own AST-derived required set names, e.g., {hint}"
        )


def _r12_select_governing_anticipation_row(mod, path: str, text: str) -> dict | None:
    """The SAME governing-row selection `check_required_gates` runs (head
    match, else the greatest normalized `ts` instant), reused here rather
    than re-derived a third time. Returns `None` — SILENTLY, never a
    SECOND `result.fail(...)` — on ANY ambiguity (no parseable rows, an
    all-foreign stream, unreliable ordering evidence, or a tie):
    `check_required_gates` already reports that EXACT ambiguity, over the
    SAME committed data, in the SAME `run_check` pass; a second report of
    the identical problem from a different reader would be noise, not a
    second finding."""
    raw_rows: list[tuple[int, dict]] = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            raw_rows.append((i + 1, parsed))
    if not raw_rows:
        return None
    quiet = Result()
    rows_with_lineno = _r12_reject_foreign_anticipation_rows(path, raw_rows, quiet)
    if not rows_with_lineno:
        return None

    ok_head, head_now_raw = _git(REPO_ROOT, "rev-parse", "HEAD")
    head_now = head_now_raw.strip() if ok_head else None

    def _row_head(r: dict) -> str | None:
        h = r.get("head_sha")
        return h if isinstance(h, str) and h else None

    def _row_instant(r: dict):
        return mod._r12_normalize_ts_instant(r.get("ts"))

    matching = [(ln, r) for ln, r in rows_with_lineno if head_now and _row_head(r) == head_now]
    pool = matching if matching else rows_with_lineno
    if len(pool) >= 2 and any(_row_instant(r) is None for _, r in pool):
        return None
    max_instant = max(_row_instant(r) for _, r in pool)
    tied = [(ln, r) for ln, r in pool if _row_instant(r) == max_instant]
    if len(tied) >= 2:
        return None
    return tied[0][1]


_R12_RESIDUAL_MARKER_RE = re.compile(r"#\s*R12-RESIDUAL:")
_BACKTICK_IDENT_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")


def _residual_marker_enclosing_functions(source: str) -> dict[int, str]:
    """`{marker_line_no: enclosing_function_name}` for every `#
    R12-RESIDUAL:` -marked line in `source` (a real `ast.parse` of
    `lead-gate-lib.py`'s own text, never a text scan for "def") — the
    SMALLEST enclosing `FunctionDef`/`AsyncFunctionDef` span, so a marker
    inside a nested closure attributes to the closure, not its outer
    function. A marker line outside every function span (module level) is
    simply absent from the returned map — this repo's own markers are
    always inside a function body, a checked-against-the-real-file scope,
    not an unverified assumption."""
    tree = ast.parse(source)
    lines = source.splitlines()
    marker_lines = {i + 1 for i, line in enumerate(lines) if _R12_RESIDUAL_MARKER_RE.search(line)}
    if not marker_lines:
        return {}
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            spans.append((node.lineno, end, node.name))
    spans.sort(key=lambda s: s[1] - s[0])  # smallest span first
    out: dict[int, str] = {}
    for ml in marker_lines:
        for start, end, name in spans:
            if start <= ml <= end:
                out[ml] = name
                break
    return out


def _all_function_names(source: str) -> set[str]:
    tree = ast.parse(source)
    return {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _r12_new_residual_marker_lines(cwd: Path, range_spec: str) -> set[int]:
    """Line numbers, in `.claude/hooks/lead-gate-lib.py`'s OWN file, that
    are BOTH a `# R12-RESIDUAL:` marker AND a line `base...HEAD` itself
    ADDS — never the whole file's accumulated history of markers (this
    repo's own `lead-gate-lib.py` carries many pre-existing, already-
    reviewed residuals no CURRENT unit's own `residual_risk` was ever
    meant to re-name). The SAME `-U0` diff-hunk line-number technique
    `check_attestation_witnesses` already uses, scoped to this one file."""
    ok, diff_out = _git(cwd, "diff", "-U0", "--end-of-options", range_spec,
                         "--", ".claude/hooks/lead-gate-lib.py")
    if not ok:
        return set()
    lines: set[int] = set()
    new_line_no = 0
    for line in diff_out.splitlines():
        if line.startswith("@@ "):
            m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                new_line_no = int(m.group(1))
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            if _R12_RESIDUAL_MARKER_RE.search(line[1:]):
                lines.add(new_line_no)
            new_line_no += 1
    return lines


def check_residual_risk_bidirectional(cwd: Path, unit_slug: str, range_spec: str, result: Result) -> None:
    """issue #570: an admission that a property is UNCOVERED is a
    committed, machine-readable record a reader ENFORCES — never prose
    alone. Bidirectional cross-reference between `lead-gate-lib.py`'s own
    `# R12-RESIDUAL` markers and the committed anticipation record's
    GOVERNING row's `residual_risk` field:

      FORWARD — every `# R12-RESIDUAL`-marked line THIS UNIT'S OWN DIFF
      ADDS (`_r12_new_residual_marker_lines` — never a pre-existing,
      already-reviewed marker from before this unit's own fix; the
      original closing-audit finding #570 rebuilds was about a NEWLY
      introduced residual, not the file's whole history) has its own
      enclosing function name (`_residual_marker_enclosing_functions`, a
      real `ast.parse`) appear as a literal substring in `residual_risk`.

      BACKWARD — every backtick-quoted identifier in `residual_risk` that
      is ALSO a real function name in `lead-gate-lib.py` must carry AT
      LEAST ONE `# R12-RESIDUAL` marker somewhere in its own span (checked
      against the WHOLE file, not diff-scoped — a residual_risk citing a
      function with NO marker anywhere is over-claimed regardless of
      when that function was last touched) — an over-claimed residual is
      refused the same way an under-claimed one is.

    Armed only when THIS UNIT'S OWN DIFF adds at least one new `#
    R12-RESIDUAL` marker AND a governing anticipation row is selectable
    (silently no-ops on ambiguity — see `_r12_select_governing_
    anticipation_row`'s own docstring for why that is never a SECOND
    report of the same ambiguity)."""
    ok, source = _git(cwd, "show", "HEAD:.claude/hooks/lead-gate-lib.py")
    if not ok:
        return  # nothing to check (file missing at HEAD is another reader's concern)
    new_marker_lines = _r12_new_residual_marker_lines(cwd, range_spec)
    if not new_marker_lines:
        return  # this unit's own diff adds no NEW residual marker -- nothing to require
    try:
        enclosing = _residual_marker_enclosing_functions(source)
    except SyntaxError:
        result.fail(".claude/hooks/lead-gate-lib.py does not parse at HEAD -- cannot verify the "
                    "# R12-RESIDUAL <-> residual_risk bidirectional property (issue #570)")
        return
    marked_functions = {name for line, name in enclosing.items() if line in new_marker_lines}
    if not marked_functions:
        return  # every new marker line fell outside every function span (module level) -- nothing to require

    path = f"docs/rigor/{unit_slug}.anticipation.jsonl"
    ok, text = _git(cwd, "show", f"HEAD:{path}")
    if not ok or not text.strip():
        return  # no anticipation record -- check_anticipation_witnesses owns that absence

    mod = _lib_module()
    governing = _r12_select_governing_anticipation_row(mod, path, text)
    if governing is None:
        return  # no unambiguous governing row -- check_required_gates already reports why

    residual_risk = governing.get("residual_risk")
    if not isinstance(residual_risk, str) or not residual_risk.strip():
        return  # check_anticipation_witnesses already requires a non-empty residual_risk

    missing_forward = sorted(fn for fn in marked_functions if fn not in residual_risk)
    if missing_forward:
        result.fail(
            f"{path}: the governing row's residual_risk does not name "
            f"{len(missing_forward)} `# R12-RESIDUAL`-marked function(s) in lead-gate-lib.py, "
            f"e.g. {missing_forward[:3]} (issue #570)"
        )

    # BACKWARD is checked against the WHOLE file's marker set (every
    # function with a `# R12-RESIDUAL` marker ANYWHERE, not just a new
    # one this unit's own diff adds) — a residual_risk citing an OLDER,
    # already-marked function is legitimate; only a citation naming a
    # function with NO marker at all, anywhere, is over-claimed.
    all_marked_functions = set(enclosing.values())
    all_function_names = _all_function_names(source)
    cited = set(_BACKTICK_IDENT_RE.findall(residual_risk))
    over_claimed = sorted(
        name for name in cited if name in all_function_names and name not in all_marked_functions
    )
    if over_claimed:
        result.fail(
            f"{path}: the governing row's residual_risk names {over_claimed[:3]} as a residual, "
            "but lead-gate-lib.py carries no `# R12-RESIDUAL` marker anywhere inside that "
            "function's own span (issue #570)"
        )


def _unit_allowlisted(unit_slug: str) -> bool:
    if not ALLOWLIST_PATH.exists():
        return False
    for line in ALLOWLIST_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == unit_slug:
            return True
    return False


def run_check(cwd: Path = REPO_ROOT) -> Result:
    result = Result()
    base_ref = os.environ.get("GITHUB_BASE_REF", "main")
    head_ref_env = os.environ.get("GITHUB_HEAD_REF")

    ok, changed, range_spec = compute_diff_context(cwd, base_ref)
    if not ok:
        result.warn("could not resolve the base ref / three-dot diff — treating as a no-op (no PR context)")
        return result

    reason = _no_op_reason(cwd, changed, range_spec)
    if reason is not None:
        print(f"check-rigor-record: no-op — {reason}")
        return result

    armed_paths = [p for p in changed if _matches_arming_glob(p)]
    if not armed_paths:
        print("check-rigor-record: not armed — diff touches none of crates/**, ci/**, .github/workflows/**")
        return result

    mod = _lib_module()
    if head_ref_env:
        head_branch = head_ref_env
    else:
        ok, head_branch = _git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
        if not ok:
            head_branch = "HEAD"
    unit_slug = mod.slugify(head_branch)

    if _unit_allowlisted(unit_slug):
        print(f"check-rigor-record: armed but {unit_slug!r} is on the shrink-only allowlist "
              f"({_display_path(ALLOWLIST_PATH)}) — no-op")
        return result

    print(f"check-rigor-record: ARMED — diff touches {len(armed_paths)} mechanism path(s), e.g. {armed_paths[:3]}")

    record_path = RIGOR_DIR / f"{unit_slug}.jsonl"
    ok, record_text = _git(cwd, "show", f"HEAD:docs/rigor/{unit_slug}.jsonl")
    if not ok:
        result.fail(f"no committed rigor record at docs/rigor/{unit_slug}.jsonl — export one with "
                    f"`python3 .claude/hooks/lead-gate-lib.py --export {unit_slug} "
                    f"> docs/rigor/{unit_slug}.jsonl` and commit it")
    else:
        rows = []
        for i, line in enumerate(record_text.splitlines()):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                result.fail(f"docs/rigor/{unit_slug}.jsonl:{i + 1}: not valid JSON ({exc})")
        pressure_rows = [r for r in rows if isinstance(r, dict) and r.get("agent_type") == "pressure-tester"]
        if not pressure_rows:
            result.fail(f"docs/rigor/{unit_slug}.jsonl carries no pressure-tester row — "
                        "design-before-mechanism (.claude/agents/lead.md:41) requires one before "
                        "a non-local fix; dispatch it and re-export the record")
        # Check 3 (ancestry) — ADVISORY ONLY, never fails the check.
        for r in rows:
            if not isinstance(r, dict):
                continue
            sha = r.get("head_sha")
            if not isinstance(sha, str) or not sha:
                continue
            ok_sha, _ = _git(cwd, "rev-parse", "--verify", f"{sha}^{{commit}}")
            if not ok_sha:
                result.warn(f"docs/rigor/{unit_slug}.jsonl: head_sha {sha} does not resolve in this "
                            "checkout (advisory — an amend or a shallow clone can cause this)")
                continue
            ok_anc, _ = _git(cwd, "merge-base", "--is-ancestor", sha, "HEAD")
            if not ok_anc:
                result.warn(f"docs/rigor/{unit_slug}.jsonl: head_sha {sha} is not an ancestor of HEAD "
                            "(advisory — the amend-after-verification workflow does this on the honest path)")

        # esc-lead-gate-R12 READER 3 (M3') — armed whenever the exported
        # record carries an open second-round BLOCK row not on the
        # shrink-only grandfather list; a shape-only hard fail, re-execution
        # ADVISORY.
        check_anticipation_witnesses(cwd, unit_slug, [r for r in rows if isinstance(r, dict)], result)

        # issue #557 items 1-2: the committed mutations/exclusions
        # attestation record — armed by the SAME open-BLOCK condition, but
        # additionally re-derives (never reads) whether item 8b/8c's own
        # requirement actually fired for THIS diff.
        check_attestation_witnesses(
            cwd, unit_slug, [r for r in rows if isinstance(r, dict)], range_spec, result)

        # issue #557 item 2: the CI-derived required call-site set (a real
        # `syn` AST parse via ci/tools/symbol-index, never a regex reader
        # or a hand list) — layered on top of check_attestation_witnesses'
        # own shape check; a phantom `mutations[].site` is a hard FAIL.
        check_required_call_site_set(
            cwd, unit_slug, [r for r in rows if isinstance(r, dict)], range_spec, result)

        # esc-lead-gate-R12 fix round 3 item 8a READER 3 — armed
        # UNCONDITIONALLY (fix round 5 Z4 made a missing/empty/all-comment
        # `ci/lead-gate-required-commands.txt` a hard FAIL here, never a
        # silent skip when it is absent); shape+value only, never
        # re-executed.
        check_required_gates(cwd, unit_slug, result)

        # issue #570: the bidirectional # R12-RESIDUAL <-> residual_risk
        # property — armed unconditionally whenever lead-gate-lib.py
        # carries at least one marker; silently no-ops when there is no
        # unambiguous governing anticipation row (check_required_gates
        # already reports that ambiguity).
        check_residual_risk_bidirectional(cwd, unit_slug, range_spec, result)

    contract_paths = [p for p in changed if _matches_contract_glob(p)]
    if not contract_paths:
        result.fail("diff arms (touches crates/**/ci/**/.github/workflows/**) but adds/modifies no "
                    "committed mechanism contract under docs/plans/*/proposals/** or docs/rigor/contracts/**")
    else:
        ok, all_contracts_out = _git(cwd, "ls-files", "docs/plans/*/proposals", "docs/rigor/contracts")
        all_contracts = [p for p in all_contracts_out.splitlines() if p.strip()] if ok else contract_paths
        for cp in contract_paths:
            ok, text = _git(cwd, "show", f"HEAD:{cp}")
            if not ok:
                result.fail(f"{cp}: named in the diff but does not read at HEAD (deleted?)")
                continue
            check_path_line_citations(cwd, cp, text, result)
            check_introducing_commit_ancestor(cwd, cp, result)
            check_not_near_identical(cwd, cp, text, all_contracts, result)

    return result


def check_required_gates(cwd: Path, unit_slug: str, result: Result) -> None:
    """esc-lead-gate-R12 fix round 3 item 8a, READER 3: when the repo
    commits `ci/lead-gate-required-commands.txt` (human-amend-only; this
    unit's own diff need not touch it), the GOVERNING row of the exported
    anticipation record must carry a `gates` object naming EVERY committed
    line VERBATIM, each with an integer `rc`, and every `rc` must be `0` —
    the COMMITTED record is expected to reflect the fix's own verified
    state, never a transient broken-tip snapshot recorded mid-round. Shape
    and value only — never re-executed (these are already CI jobs
    elsewhere in `.github/workflows/`).

    The governing row is selected ORDER-INDEPENDENTLY, never by
    `rows[-1]` (append position). `cmd_export_anticipation` dumps every
    `<slug>.anticipation.*.json` artifact still on disk,
    `sorted(sdir.iterdir())` — i.e. sorted by FILENAME, a tip sha, which
    is pseudorandom hex and carries no chronological meaning; an older
    round's artifact can sort AFTER a newer round's and land on the last
    line of the committed export. The exporter stamps `ts` (the artifact
    FILE's own mtime) and `head_sha` (the artifact's own `pre_fix_sha`) on
    every row it emits — so `_row_head`/`_row_instant` below select the
    row whose own `head_sha` matches this checkout's actual `HEAD` when
    one does (the case where the export was captured at the exact commit
    reader 3 is validating); when none does — the common case, since a
    pre-fix witness by construction predates the commit it is validated
    against — every row is eligible. Within whichever pool applies, the
    row naming the GREATEST normalized `ts` INSTANT governs, never the
    row nearest the end of the file — and when the pool holds >=2
    candidate rows and ANY of them lacks a reliably-orderable `ts`, OR two
    or more rows normalize to the IDENTICAL instant, this FAILS LOUDLY,
    naming the tied rows' own line numbers, rather than silently falling
    back to `rows[0]`/append order.

    `ts` is normalized to a single well-defined, always-AWARE comparable
    form via the shared `_r12_normalize_ts_instant` (`lead-gate-lib.py`,
    the SAME function `_r12_previous_relay_row`'s own ordering compare
    there uses) BEFORE any comparison, so a mixed pool — one row's `ts`
    with no UTC offset at all beside another's `...Z`/`...+00:00` — can
    never raise `TypeError: can't compare offset-naive and
    offset-aware datetimes`: a naive value normalizes to `None`, the same
    bucket a missing/malformed `ts` already occupies, refused as
    unparseable rather than compared. Because the compare is on the
    normalized INSTANT rather than the raw TEXT, two rows naming the SAME
    instant in different spellings (a trailing `Z` vs an explicit
    `+00:00` offset) correctly TIE, rather than a text-only compare's
    silent, incorrect ordering of the two.

    The gates SHAPE/VALUE check itself is the SAME shared
    `_r12_gates_shape_rejection` reader 1 and reader 2 call — never a
    second, independently hand-rolled implementation that can drift from
    theirs (this is exactly the bug three earlier readers of this file
    found: a hand-rolled loop here accepted `{"rc": "1"}`/`{"rc": true}`/
    non-dict `gates` entries that the shared function has always denied)."""
    ok_req, req_text = _git(cwd, "show", "HEAD:ci/lead-gate-required-commands.txt")
    if not ok_req:
        # Fix round 5 Z4: MISSING is a hard FAIL here, never a silent
        # return — a missing committed gate-command list must never
        # disarm item 8a with zero CI signal.
        result.fail("ci/lead-gate-required-commands.txt does not read at HEAD in this checkout — "
                    "item 8a's committed, human-amend-only gate-command list must be present "
                    "(esc-lead-gate-R12 fix round 5 Z4)")
        return
    required_commands: list[str] = []
    for line in req_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        command = stripped.split("  #", 1)[0].rstrip()
        if command:
            required_commands.append(command)
    if not required_commands:
        # Fix round 5 Z4: EMPTY/all-comment is likewise a hard FAIL.
        result.fail("ci/lead-gate-required-commands.txt exists but names no command line (empty "
                    "or all-comment) — item 8a's gate-command list must never silently disarm "
                    "(esc-lead-gate-R12 fix round 5 Z4)")
        return
    path = f"docs/rigor/{unit_slug}.anticipation.jsonl"
    ok, text = _git(cwd, "show", f"HEAD:{path}")
    if not ok or not text.strip():
        # No anticipation record for THIS unit at all -- structurally
        # nothing to check item 8a against; `check_anticipation_witnesses`
        # already fails the missing-record shape when IT is armed (an open
        # second-round BLOCK), which is the only case that requires one.
        return
    raw_rows: list[tuple[int, dict]] = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            raw_rows.append((i + 1, parsed))
    if not raw_rows:
        return

    # esc-lead-gate-R12 fix round 6 Z12: same property as `check_
    # anticipation_witnesses` — reject (never silently select) a foreign
    # row BEFORE it can become "the governing row", which is exactly how a
    # `lead-relay-attestation` row (no `gates` object at all) used to hide
    # the real one.
    rows_with_lineno = _r12_reject_foreign_anticipation_rows(path, raw_rows, result)
    if not rows_with_lineno:
        return

    ok_head, head_now_raw = _git(cwd, "rev-parse", "HEAD")
    head_now = head_now_raw.strip() if ok_head else None

    def _row_head(r: dict) -> str | None:
        h = r.get("head_sha")
        return h if isinstance(h, str) and h else None

    mod = _lib_module()

    def _row_instant(r: dict):
        """issue #557 item 3: `ts` normalized through the SAME shared
        function `_r12_previous_relay_row`'s own `<` compare in
        `lead-gate-lib.py` uses (`mod._r12_normalize_ts_instant`) — one
        fix covers both call sites identically. Always AWARE or `None`;
        never a naive `datetime`, so `max()`/`==` over this pool can never
        raise `TypeError: can't compare offset-naive and offset-aware
        datetimes` (the fix-round-6 crash an executed audit probe found).
        A naive `ts` is refused as unparseable, folded into the same
        `None` bucket a missing/malformed one already occupies below —
        never compared."""
        return mod._r12_normalize_ts_instant(r.get("ts"))

    matching = [(ln, r) for ln, r in rows_with_lineno if head_now and _row_head(r) == head_now]
    pool = matching if matching else rows_with_lineno

    if len(pool) >= 2 and any(_row_instant(r) is None for _, r in pool):
        pool_lines = ", ".join(str(ln) for ln, _ in pool)
        result.fail(
            f"{path}: {len(pool)} candidate anticipation row(s) (lines {pool_lines}) carry no "
            "reliable ordering evidence (at least one has no non-empty, parseable, "
            "timezone-aware `ts`) -- the GOVERNING row is AMBIGUOUS; re-export with "
            "`lead-gate-lib.py --export-anticipation` (which stamps `ts`/`head_sha` on every "
            "row it emits) and commit the result (esc-lead-gate-R12)"
        )
        return

    # issue #557 item 3: ties are on the NORMALIZED INSTANT, never the
    # `ts` TEXT — two rows naming the SAME instant in different text (a
    # trailing `Z` vs an explicit `+00:00` offset) correctly tie instead
    # of silently ordering by text. `max()` returning the FIRST maximal
    # element is still why a tie must be detected explicitly (a two-row
    # pool with an identical governing value, the `rc=0` row first, would
    # otherwise silently govern and shadow a genuinely `rc=1` sibling) —
    # a tie is exactly as AMBIGUOUS as a missing `ts`.
    max_instant = max(_row_instant(r) for _, r in pool)
    tied = [(ln, r) for ln, r in pool if _row_instant(r) == max_instant]
    if len(tied) >= 2:
        tied_lines = ", ".join(str(ln) for ln, _ in tied)
        tied_ts_texts = sorted({r.get("ts") for _, r in tied})
        result.fail(
            f"{path}: {len(tied)} candidate anticipation row(s) (lines {tied_lines}) share the "
            f"SAME greatest `ts` instant ({max_instant.isoformat()!r}, spelled as "
            f"{tied_ts_texts!r} across the tied rows) -- the GOVERNING row is AMBIGUOUS on a "
            "tie, exactly as it is when `ts` is missing entirely; re-export so each round's row "
            "carries a distinguishing `ts` (esc-lead-gate-R12, instant-aware tie-break)"
        )
        return
    governing = tied[0][1]

    gates_why = mod._r12_gates_shape_rejection(f"{path}: the governing row", governing.get("gates"),
                                                required_commands, judge_rc=True)
    if gates_why is not None:
        result.fail(gates_why)


def check_allowlist_only_shrinks(cwd: Path = REPO_ROOT) -> int:
    ok, _ = _git(cwd, "fetch", "--quiet", "origin", "main")
    if not ok:
        print("rigor-record-allowlist-only-shrinks: FAIL — git fetch origin main failed", file=sys.stderr)
        return 1
    ok, _ = _git(cwd, "rev-parse", "--verify", "origin/main")
    if not ok:
        print("rigor-record-allowlist-only-shrinks: FAIL — origin/main does not resolve", file=sys.stderr)
        return 1
    current = set()
    if ALLOWLIST_PATH.exists():
        for line in ALLOWLIST_PATH.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                current.add(s)
    rel = ALLOWLIST_PATH.relative_to(cwd).as_posix()
    ok, base_text = _git(cwd, "show", f"origin/main:{rel}")
    if not ok:
        print(f"rigor-record-allowlist-only-shrinks: OK (bootstrap) — origin/main has no {rel} yet; "
              f"this branch's {len(current)} entries establish the baseline.")
        return 0
    base = {s.strip() for s in base_text.splitlines() if s.strip() and not s.strip().startswith("#")}
    added = current - base
    if added:
        print("rigor-record-allowlist-only-shrinks: FAIL", file=sys.stderr)
        for e in sorted(added):
            print(f"  + {e}", file=sys.stderr)
        print("\nrigor-record-allowlist-only-shrinks: this branch adds a NEW exemption. The "
              "allowlist may only shrink — a genuinely new exemption is a human-reviewed decision, "
              "made on main directly, never an autonomous addition on a swarm branch.", file=sys.stderr)
        return 1
    print(f"rigor-record-allowlist-only-shrinks: OK — {len(current)} entries "
          f"({len(base) - len(current)} shrunk vs origin/main).")
    return 0


def check_r12_grandfather_only_shrinks(cwd: Path = REPO_ROOT) -> int:
    """esc-lead-gate-R12 (M3'): the SAME shrink-only ratchet as
    `check_allowlist_only_shrinks`, applied to `R12_GRANDFATHER_PATH`
    instead — a separate check because the two exemptions arm on
    different diffs and are read by different functions."""
    ok, _ = _git(cwd, "fetch", "--quiet", "origin", "main")
    if not ok:
        print("rigor-record-r12-grandfather-only-shrinks: FAIL — git fetch origin main failed", file=sys.stderr)
        return 1
    ok, _ = _git(cwd, "rev-parse", "--verify", "origin/main")
    if not ok:
        print("rigor-record-r12-grandfather-only-shrinks: FAIL — origin/main does not resolve", file=sys.stderr)
        return 1
    current = set()
    if R12_GRANDFATHER_PATH.exists():
        for line in R12_GRANDFATHER_PATH.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                current.add(s)
    rel = R12_GRANDFATHER_PATH.relative_to(cwd).as_posix()
    ok, base_text = _git(cwd, "show", f"origin/main:{rel}")
    if not ok:
        print(f"rigor-record-r12-grandfather-only-shrinks: OK (bootstrap) — origin/main has no {rel} "
              f"yet; this branch's {len(current)} entries establish the baseline.")
        return 0
    base = {s.strip() for s in base_text.splitlines() if s.strip() and not s.strip().startswith("#")}
    added = current - base
    if added:
        print("rigor-record-r12-grandfather-only-shrinks: FAIL", file=sys.stderr)
        for e in sorted(added):
            print(f"  + {e}", file=sys.stderr)
        print("\nrigor-record-r12-grandfather-only-shrinks: this branch adds a NEW exemption. The "
              "grandfather list may only shrink — a genuinely new exemption (an in-flight unit "
              "whose BLOCK predates fix round 1) is a human-reviewed decision, made on main "
              "directly, never an autonomous addition on a swarm branch.", file=sys.stderr)
        return 1
    print(f"rigor-record-r12-grandfather-only-shrinks: OK — {len(current)} entries "
          f"({len(base) - len(current)} shrunk vs origin/main).")
    return 0


def _parse_required_commands(text: str) -> set[str]:
    """The SAME parse `_r12_required_commands_or_deny` in lead-gate-lib.py
    applies — `#`-comment/blank lines skipped, a trailing `  # ...`
    annotation stripped — reduced to a SET of command strings (never the
    annotation, which a committer may re-measure/reword without that
    being a real removal)."""
    out: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        command = stripped.split("  #", 1)[0].rstrip()
        if command:
            out.add(command)
    return out


def check_required_commands_only_shrinks(cwd: Path = REPO_ROOT) -> int:
    """esc-lead-gate-R12 fix round 5 Z4(b): the OPPOSITE polarity from
    `check_allowlist_only_shrinks`/`check_r12_grandfather_only_shrinks` —
    `REQUIRED_COMMANDS_PATH` names REQUIRED gates, so this check's job is
    to CATCH a SHRINK (a committed command REMOVED, weakening item 8a)
    and FAIL it; adding a new required command, or leaving the set
    unchanged, always passes. BOOTSTRAP arm: `origin/main` carrying no
    such file yet (this PR is the one introducing it) establishes the
    baseline instead of failing."""
    ok, _ = _git(cwd, "fetch", "--quiet", "origin", "main")
    if not ok:
        print("rigor-record-required-commands-only-shrinks: FAIL — git fetch origin main failed", file=sys.stderr)
        return 1
    ok, _ = _git(cwd, "rev-parse", "--verify", "origin/main")
    if not ok:
        print("rigor-record-required-commands-only-shrinks: FAIL — origin/main does not resolve", file=sys.stderr)
        return 1
    current = _parse_required_commands(REQUIRED_COMMANDS_PATH.read_text()) if REQUIRED_COMMANDS_PATH.exists() else set()
    rel = REQUIRED_COMMANDS_PATH.relative_to(cwd).as_posix()
    ok, base_text = _git(cwd, "show", f"origin/main:{rel}")
    if not ok:
        print(f"rigor-record-required-commands-only-shrinks: OK (bootstrap) — origin/main has no "
              f"{rel} yet; this branch's {len(current)} command(s) establish the baseline.")
        return 0
    base = _parse_required_commands(base_text)
    removed = base - current
    if removed:
        print("rigor-record-required-commands-only-shrinks: FAIL", file=sys.stderr)
        for e in sorted(removed):
            print(f"  - {e}", file=sys.stderr)
        print("\nrigor-record-required-commands-only-shrinks: this branch REMOVES a committed "
              "required gate command. The list may only grow (or stay the same) — removing a "
              "required gate weakens item 8a and is a human-reviewed decision, made on main "
              "directly, never an autonomous removal on a swarm branch.", file=sys.stderr)
        return 1
    print(f"rigor-record-required-commands-only-shrinks: OK — {len(current)} command(s) "
          f"({len(current) - len(base)} added vs origin/main).")
    return 0


# ==========================================================================
# --self-test — hermetic fixtures, the check_lead_gate.py harness pattern:
# a real `origin` remote + a real feature-branch clone, run against the
# REAL run_check()/check_allowlist_only_shrinks(), never a reimplementation.
# ==========================================================================

class Failure(Exception):
    pass


def _assert(cond: bool, label: str, detail: str = "") -> None:
    if not cond:
        raise Failure(f"{label}: {detail}")


def _sh(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(cwd)] + list(args), capture_output=True, text=True)
    _assert(proc.returncode == 0, "git fixture setup", f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


_RR_BASELINE_REQUIRED_COMMAND = "python3 ci/scripts/check_swarm_bijection.py"

_FIXTURE_ENV = {
    "GIT_AUTHOR_NAME": "rigor-fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
    "GIT_COMMITTER_NAME": "rigor-fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
}


def _pr_repo(tmp: Path) -> tuple[Path, Path]:
    """A real `origin` repo (its own `main`, one seed commit) and a real
    clone (`work`) with `origin` wired up — `git fetch origin main` in
    `work` resolves `origin/main` exactly as a real CI checkout would.
    Copies THIS run's own `.claude/hooks/lead-gate-lib.py` and
    `ci/scripts/check_rigor_record.py` into `work` (never `.claude/hooks/`
    from the real repo directly — a throwaway copy, per this file's own
    module doc) so `_lib_module()`/self-invocation resolve inside the
    fixture, not the real tree."""
    origin = tmp / "origin"
    work = tmp / "work"
    origin.mkdir()
    env = dict(os.environ)
    env.update(_FIXTURE_ENV)
    subprocess.run(["git", "init", "-q", "-b", "main", str(origin)], check=True, env=env)
    subprocess.run(["git", "-C", str(origin), "config", "commit.gpgsign", "false"], check=True)
    # Fixture-only: `origin` is a NON-bare repo so its own working tree can
    # be read directly for setup; allow `work` to push updates to its
    # currently-checked-out `main` (git denies this by default) — never a
    # real-repo concern, since the real `origin` in CI is GitHub itself.
    subprocess.run(["git", "-C", str(origin), "config", "receive.denyCurrentBranch", "updateInstead"], check=True)
    (origin / "README.md").write_text("seed\n")
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-q", "-m", "seed"], check=True, env=env)
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True)
    subprocess.run(["git", "-C", str(work), "config", "commit.gpgsign", "false"], check=True)
    # Mirror the real tree's own module layout inside the fixture: this
    # script and lead-gate-lib.py at their real relative paths, so
    # REPO_ROOT/`_lib_module()` resolve inside `work`, never the real repo.
    (work / ".claude" / "hooks").mkdir(parents=True)
    (work / "ci" / "scripts").mkdir(parents=True)
    (work / "docs" / "rigor" / "contracts").mkdir(parents=True)
    (work / "docs" / "plans" / "99-fixture" / "proposals").mkdir(parents=True)
    lib_text = LEAD_GATE_LIB.read_text() if LEAD_GATE_LIB.exists() else _MINIMAL_SLUGIFY_STUB
    (work / ".claude" / "hooks" / "lead-gate-lib.py").write_text(lib_text)
    (work / "ci" / "scripts" / "check_rigor_record.py").write_text(Path(__file__).read_text())
    # Fix round 5 Z4: a missing/empty/all-comment required-commands file is
    # now a hard FAIL in `check_required_gates` whenever an anticipation
    # record exists at all -- every fixture gets a baseline, non-empty file
    # by default so unrelated fixtures never incidentally exercise item
    # 8a's own FAIL arm; `_rr_gates_commit` (RR14-17) overwrites this with
    # its own custom line, and RR12c/d/e/g/h's own anticipation rows carry
    # a matching `gates` entry.
    (work / "ci" / "lead-gate-required-commands.txt").write_text(
        f"{_RR_BASELINE_REQUIRED_COMMAND}  # measured ~0.0s\n")
    subprocess.run(["git", "-C", str(work), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(work), "commit", "-q", "-m", "scaffold"], check=True, env=env)
    subprocess.run(["git", "-C", str(work), "push", "-q", "origin", "HEAD:main"], check=True)
    subprocess.run(["git", "-C", str(work), "checkout", "-q", "-b", "feat/rr-fixture"], check=True)
    return origin, work


# A minimal, self-contained fallback if this fixture ever runs with no real
# lead-gate-lib.py reachable (never used against the real tree — only a
# defensive stub so the fixture harness itself cannot silently pass by
# accident when the real file is missing).
_MINIMAL_SLUGIFY_STUB = (
    "import re\n"
    "def slugify(branch):\n"
    "    return re.sub(r'[^A-Za-z0-9._-]', '_', branch.strip()) or 'UNBOUND'\n"
)


def _commit(work: Path, message: str, files: dict[str, str], swarm: bool = True) -> str:
    for rel, content in files.items():
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    subprocess.run(["git", "-C", str(work), "add", "-A"], check=True)
    full_msg = message + ("\n\nCo-Authored-By: Claude Fixture <noreply@anthropic.com>" if swarm else "")
    env = dict(os.environ)
    env.update(_FIXTURE_ENV)
    subprocess.run(["git", "-C", str(work), "commit", "-q", "-m", full_msg], check=True, env=env)
    return _sh(work, "rev-parse", "HEAD")


def _run_check_in(work: Path, env_overrides: dict | None = None) -> Result:
    env = {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_BASE_REF": "main", "GITHUB_HEAD_REF": "feat/rr-fixture"}
    if env_overrides:
        env.update(env_overrides)
    old = {}
    for k, v in env.items():
        old[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        global LEAD_GATE_LIB
        real_lib = LEAD_GATE_LIB
        LEAD_GATE_LIB = work / ".claude" / "hooks" / "lead-gate-lib.py"
        try:
            return run_check(work)
        finally:
            LEAD_GATE_LIB = real_lib
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_VALID_CONTRACT = "# A mechanism contract\n\nCites `docs/README-fixture.md:1` which exists.\n"


def fixture_rr1_not_armed_docs_only() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "docs: a docs-only change", {"docs/only.md": "hello\n"})
        r = _run_check_in(work)
        _assert(r.ok(), "RR1", f"a docs-only diff must not arm: {r.failures}")


def fixture_rr2_armed_no_record() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "ci: touch a gate script", {"ci/scripts/probe.py": "print('x')\n"})
        r = _run_check_in(work)
        _assert(not r.ok(), "RR2", "armed with no rigor record must FAIL")
        _assert(any("no committed rigor record" in f for f in r.failures), "RR2",
                f"reason must name the missing record: {r.failures}")


def fixture_rr3_record_no_pressure_row() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "adversarial-audit", "verdict": "PASS"})
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR3", "a record with no pressure-tester row must FAIL")
        _assert(any("no pressure-tester row" in f for f in r.failures), "RR3", f"{r.failures}")


def fixture_rr4_no_contract_file() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR4", "armed with a record+pressure row but no contract file must FAIL")
        _assert(any("no committed mechanism contract" in f for f in r.failures), "RR4", f"{r.failures}")


def fixture_rr5_full_disclosure_allows() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR5", f"full disclosure must ALLOW: {r.failures}")


def fixture_rr6_bad_citation_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        bad_contract = "# Contract\n\nCites `docs/README-fixture.md:99` which does not have 99 lines.\n"
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": bad_contract,
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR6", "a citation past the file's own line count must FAIL")
        _assert(any("has only" in f for f in r.failures), "RR6", f"{r.failures}")


def fixture_rr7_near_identical_contract_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        # Commit an EXISTING contract on main first (via a swarm commit on
        # the feature branch itself, at the SAME path a second file will
        # copy — near-identical after whitespace normalization).
        _commit(work, "docs: an existing contract", {
            "docs/rigor/contracts/other-unit.md": "# A mechanism contract\n\nCites `docs/README-fixture.md:1` which exists.\n",
        })
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": "  # A mechanism contract  \n\n\nCites `docs/README-fixture.md:1` which exists.\n",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR7", "a whitespace-only variant of an existing contract must FAIL as near-identical")
        _assert(any("near-identical" in f for f in r.failures), "RR7", f"{r.failures}")


def fixture_rr8_noop_shapes() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "ci: touch a gate script, human-authored", {"ci/scripts/probe.py": "print('x')\n"}, swarm=False)
        r = _run_check_in(work)
        _assert(r.ok(), "RR8a", f"a human-authored-only range must no-op: {r.failures}")

    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "chore: bump crate version", {"crates/foo/Cargo.toml": "[package]\nversion=\"0.2.0\"\n"})
        r = _run_check_in(work)
        _assert(r.ok(), "RR8b", f"a release-shaped (Cargo.toml-only) diff must no-op: {r.failures}")

    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "ci: touch a gate script", {"ci/scripts/probe.py": "print('x')\n"})
        r = _run_check_in(work, {"GITHUB_ACTOR": "dependabot[bot]"})
        _assert(r.ok(), "RR8c", f"a dependabot actor must no-op: {r.failures}")

    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "ci: touch a gate script", {"ci/scripts/probe.py": "print('x')\n"})
        _commit(work, 'Revert "ci: touch a gate script"', {"ci/scripts/probe.py": "orig\n"})
        r = _run_check_in(work)
        _assert(r.ok(), "RR8d", f"a revert HEAD commit must no-op: {r.failures}")

    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        r = _run_check_in(work, {"GITHUB_EVENT_NAME": "push"})
        _assert(r.ok(), "RR8e", f"a non-pull_request event must no-op: {r.failures}")


def fixture_rr9_ancestry_advisory_only() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED",
                           "head_sha": "cafef00d1234567890abcdef1234567890abcdef"})
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR9", f"an unresolvable head_sha must WARN, never FAIL: {r.failures}")
        _assert(any("does not resolve" in w for w in r.warnings), "RR9",
                f"the unresolvable sha must be reported as a warning: {r.warnings}")


def fixture_rr10_allowlisted_unit_noop() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        (work / "ci" / "scripts" / "rigor_record_allowlist.txt").write_text("# fixture allowlist\nfeat_rr-fixture\n")
        _commit(work, "ci: touch a gate script", {"ci/scripts/probe.py": "print('x')\n"})
        global ALLOWLIST_PATH
        real_allowlist = ALLOWLIST_PATH
        ALLOWLIST_PATH = work / "ci" / "scripts" / "rigor_record_allowlist.txt"
        try:
            r = _run_check_in(work)
        finally:
            ALLOWLIST_PATH = real_allowlist
        _assert(r.ok(), "RR10", f"an allowlisted unit must no-op even though armed: {r.failures}")


def fixture_rr11_allowlist_only_shrinks() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        allow_rel = "ci/scripts/rigor_record_allowlist.txt"
        _commit(work, "seed allowlist", {allow_rel: "unit-a\n"})
        _sh(work, "push", "-q", "origin", "HEAD:main")
        _sh(work, "fetch", "-q", "origin", "main")
        global ALLOWLIST_PATH
        real_allowlist = ALLOWLIST_PATH
        ALLOWLIST_PATH = work / allow_rel
        try:
            rc = check_allowlist_only_shrinks(work)
            _assert(rc == 0, "RR11a", "an unchanged allowlist must pass the shrink-only ratchet")
            (work / allow_rel).write_text("unit-a\nunit-b\n")
            rc = check_allowlist_only_shrinks(work)
            _assert(rc != 0, "RR11b", "adding a NEW entry (not yet on origin/main) must FAIL the ratchet")
        finally:
            ALLOWLIST_PATH = real_allowlist


def fixture_rr12a_no_open_block_row_is_a_noop() -> None:
    """esc-lead-gate-R12 READER 3 (M3'): a record with NO open second-round
    BLOCK row at all (only a pressure-tester row) never arms this check —
    the SAME full-disclosure record RR5 already proves ALLOWS stays
    ALLOWED, with no `docs/rigor/<slug>.anticipation.jsonl` carried
    either."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR12a", f"no open second-round BLOCK row must be a no-op (still ALLOW): {r.failures}")


def fixture_rr12b_open_block_no_anticipation_file_fails() -> None:
    """An open `adversarial-audit` BLOCK row is present, but NO
    `docs/rigor/<slug>.anticipation.jsonl` at all — HARD FAILS, naming the
    `--export-anticipation` remedy (never a retroactive pass just because
    the artifact happens to be absent)."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR12b", "an open second-round BLOCK row with no anticipation file must FAIL")
        _assert(any("--export-anticipation" in f for f in r.failures), "RR12b", f"{r.failures}")


def fixture_rr12c_shape_allows_despite_unresolvable_pre_fix_sha() -> None:
    """A correctly-shaped anticipation record (covers the required file,
    denylist-clean command) ALLOWS even though `pre_fix_sha` does not
    resolve in this checkout (a shallow-clone-shaped repro) — the HARD
    requirement is shape alone; re-execution is advisory and only WARNS
    when it cannot even attempt reproduction."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        mod = _lib_module()
        witness = mod._witness_hash(0, "ok\n", "")
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        anticipation_row = json.dumps({
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40, "agent_type": "lead-anticipation",
            "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": witness}},
            "residual_risk": "fixture residual",
            "gates": {_RR_BASELINE_REQUIRED_COMMAND: {"rc": 0}},
        })
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": anticipation_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR12c", f"a correctly-shaped record must ALLOW despite an unresolvable "
                                  f"pre_fix_sha: {r.failures}")
        _assert(any("does not resolve" in w for w in r.warnings), "RR12c",
                f"the unresolvable pre_fix_sha must still be visible as a warning: {r.warnings}")


def fixture_rr12d_real_reproducing_witness_allows() -> None:
    """The SAME shape as RR12c, but `pre_fix_sha` is a REAL commit this
    check independently checks out and the command REPRODUCES there —
    ALLOWS, with no mismatch warning for this attack."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        pre_fix_sha = _sh(work, "rev-parse", "HEAD")
        mod = _lib_module()
        witness = mod._witness_hash(0, "ok\n", "")
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        anticipation_row = json.dumps({
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": pre_fix_sha, "agent_type": "lead-anticipation",
            "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": witness}},
            "residual_risk": "fixture residual",
            "gates": {_RR_BASELINE_REQUIRED_COMMAND: {"rc": 0}},
        })
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": anticipation_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR12d", f"a real reproducing witness at pre_fix_sha must ALLOW: {r.failures}")
        _assert(not any("mismatched at pre_fix_sha" in w for w in r.warnings), "RR12d",
                f"a genuinely reproducing witness must not be reported mismatched: {r.warnings}")


def fixture_rr12e_real_nonreproducing_witness_warns_never_fails() -> None:
    """The SAME REAL `pre_fix_sha` checkout as RR12d, but the recorded
    `hash` does NOT match what re-executing the command actually produces
    there — WARNS naming the mismatch, but NEVER FAILS (M3': re-execution
    is advisory only, since a lead's machine and CI's userland can
    genuinely diverge on some commands without the mechanism itself having
    changed)."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        pre_fix_sha = _sh(work, "rev-parse", "HEAD")
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        anticipation_row = json.dumps({
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": pre_fix_sha, "agent_type": "lead-anticipation",
            "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "0" * 64}},
            "residual_risk": "fixture residual",
            "gates": {_RR_BASELINE_REQUIRED_COMMAND: {"rc": 0}},
        })
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": anticipation_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR12e", f"a non-reproducing witness must WARN, never FAIL: {r.failures}")
        _assert(any("mismatched at pre_fix_sha" in w for w in r.warnings), "RR12e",
                f"the mismatch must still be visible as a warning: {r.warnings}")


def fixture_rr12f_grandfathered_unit_no_file_allows() -> None:
    """An open second-round BLOCK row with NO anticipation file at all is a
    HARD FAIL (RR12b) — UNLESS the unit is on the shrink-only R12
    grandfather list, in which case it ALLOWS (advisory warning only)."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        (work / "ci" / "scripts" / "rigor_record_r12_grandfather.txt").write_text("# fixture\nfeat_rr-fixture\n")
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        global R12_GRANDFATHER_PATH
        real_path = R12_GRANDFATHER_PATH
        R12_GRANDFATHER_PATH = work / "ci" / "scripts" / "rigor_record_r12_grandfather.txt"
        try:
            r = _run_check_in(work)
        finally:
            R12_GRANDFATHER_PATH = real_path
        _assert(r.ok(), "RR12f", f"a grandfathered unit must ALLOW despite no anticipation file: {r.failures}")


def fixture_rr12g_tracked_bash_path_allows() -> None:
    """M3': `bash <path>` is admitted only for paths git-TRACKED AT
    `pre_fix_sha` — a script added IN that same commit is tracked there,
    and the shape check (plus a real, matching re-execution) ALLOWS."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        _commit(work, "add probe.sh, tracked here", {"probe.sh": "#!/bin/sh\nprintf ok\n"})
        pre_fix_sha = _sh(work, "rev-parse", "HEAD")
        mod = _lib_module()
        witness = mod._witness_hash(0, "ok", "")
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        anticipation_row = json.dumps({
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": pre_fix_sha, "agent_type": "lead-anticipation",
            "attacks": {"a.py": {"command": "bash probe.sh", "hash": witness}},
            "residual_risk": "fixture residual",
            "gates": {_RR_BASELINE_REQUIRED_COMMAND: {"rc": 0}},
        })
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": anticipation_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR12g", f"a bash path tracked at pre_fix_sha must ALLOW: {r.failures}")


def fixture_rr12h_untracked_bash_path_fails_shape() -> None:
    """M3': the SAME `bash probe.sh` command, but `pre_fix_sha` names a
    commit BEFORE `probe.sh` was ever added — untracked there — HARD
    FAILS the shape check, naming it; tracked-ness buys reviewability, not
    safety, but CI must never execute an untracked lead-written file from
    a detached checkout."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        pre_fix_sha = _sh(work, "rev-parse", "HEAD")  # BEFORE probe.sh exists
        _commit(work, "add probe.sh AFTER pre_fix_sha", {"probe.sh": "#!/bin/sh\nprintf ok\n"})
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        anticipation_row = json.dumps({
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": pre_fix_sha, "agent_type": "lead-anticipation",
            "attacks": {"a.py": {"command": "bash probe.sh", "hash": "a" * 64}},
            "residual_risk": "fixture residual",
            "gates": {_RR_BASELINE_REQUIRED_COMMAND: {"rc": 0}},
        })
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": anticipation_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR12h", "a bash path NOT tracked at pre_fix_sha must FAIL the shape check")
        _assert(any("is not git-TRACKED" in f for f in r.failures), "RR12h", f"{r.failures}")


def fixture_rr13_r12_grandfather_only_shrinks() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        gf_rel = "ci/scripts/rigor_record_r12_grandfather.txt"
        _commit(work, "seed r12 grandfather list", {gf_rel: "unit-a\n"})
        _sh(work, "push", "-q", "origin", "HEAD:main")
        _sh(work, "fetch", "-q", "origin", "main")
        global R12_GRANDFATHER_PATH
        real_path = R12_GRANDFATHER_PATH
        R12_GRANDFATHER_PATH = work / gf_rel
        try:
            rc = check_r12_grandfather_only_shrinks(work)
            _assert(rc == 0, "RR13a", "an unchanged grandfather list must pass the shrink-only ratchet")
            (work / gf_rel).write_text("unit-a\nunit-b\n")
            rc = check_r12_grandfather_only_shrinks(work)
            _assert(rc != 0, "RR13b", "adding a NEW entry (not yet on origin/main) must FAIL the ratchet")
        finally:
            R12_GRANDFATHER_PATH = real_path


def fixture_rr19_required_commands_only_shrinks() -> None:
    """fix round 5 Z4(b): the OPPOSITE polarity from RR13's own ratchet —
    `ci/lead-gate-required-commands.txt` names REQUIRED gates, so ADDING a
    line passes (tightening) and REMOVING a committed line FAILS
    (weakening item 8a with zero human review)."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        rc_rel = "ci/lead-gate-required-commands.txt"
        _commit(work, "seed required-commands file", {rc_rel: "python3 ci/scripts/probe_a.py  # measured ~0.1s\n"})
        _sh(work, "push", "-q", "origin", "HEAD:main")
        _sh(work, "fetch", "-q", "origin", "main")
        global REQUIRED_COMMANDS_PATH
        real_path = REQUIRED_COMMANDS_PATH
        REQUIRED_COMMANDS_PATH = work / rc_rel
        try:
            rc = check_required_commands_only_shrinks(work)
            _assert(rc == 0, "RR19a", "an unchanged required-commands file must pass the ratchet")
            (work / rc_rel).write_text(
                "python3 ci/scripts/probe_a.py  # measured ~0.1s\n"
                "python3 ci/scripts/probe_b.py  # measured ~0.1s\n")
            rc = check_required_commands_only_shrinks(work)
            _assert(rc == 0, "RR19b", "ADDING a new required command must PASS the ratchet (tightening)")
            (work / rc_rel).write_text("# nothing but comments\n")
            rc = check_required_commands_only_shrinks(work)
            _assert(rc != 0, "RR19c", "REMOVING a committed required command must FAIL the ratchet")
        finally:
            REQUIRED_COMMANDS_PATH = real_path


def _rr_gates_commit(work: Path, extra_files: dict[str, str], gates: dict | None) -> None:
    """Shared setup for RR14-RR16: a committed `ci/lead-gate-required-
    commands.txt` (one line) plus an anticipation record whose single row
    carries `gates` (or omits it, when `gates is None`)."""
    pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
    block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                             "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                             "class_enumeration": ["a.py:1"]})
    art: dict = {
        "unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40, "agent_type": "lead-anticipation",
        "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
        "residual_risk": "fixture residual",
    }
    if gates is not None:
        art["gates"] = gates
    files = {
        "ci/lead-gate-required-commands.txt": "python3 ci/scripts/probe.py  # measured ~0.1s\n",
        "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
        "docs/rigor/feat_rr-fixture.anticipation.jsonl": json.dumps(art) + "\n",
        "docs/README-fixture.md": "line one\n",
        "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
    }
    files.update(extra_files)
    _commit(work, "ci: touch a gate script", files)


def fixture_rr14_missing_gates_fails() -> None:
    """item 8a READER 3: `ci/lead-gate-required-commands.txt` is
    committed, but the exported anticipation record's row carries no
    `gates` object at all -- HARD FAIL."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_gates_commit(work, {}, gates=None)
        r = _run_check_in(work)
        _assert(not r.ok(), "RR14", "a missing `gates` object must FAIL when required-commands.txt is committed")
        _assert(any("carries no `gates` object" in f for f in r.failures), "RR14", f"{r.failures}")


def fixture_rr15_complete_gates_rc_zero_allows() -> None:
    """POSITIVE CONTROL — green at base by construction: guards the reader
    against over-refusal (a shape-complete, fully green governing `gates`
    row must ALLOW). The RED half is RR14/RR16. item 8a READER 3: the
    satisfiable case -- every committed line present with `rc == 0`
    ALLOWS."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_gates_commit(work, {}, gates={"python3 ci/scripts/probe.py": {"rc": 0}})
        r = _run_check_in(work)
        _assert(r.ok(), "RR15", f"a fully green gates object must ALLOW: {r.failures}")


def fixture_rr16_nonzero_rc_fails() -> None:
    """item 8a READER 3: a `gates` entry recording a non-zero `rc` in the
    COMMITTED record HARD FAILS -- the exported record is expected to
    reflect the fix's own verified state."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_gates_commit(work, {}, gates={"python3 ci/scripts/probe.py": {"rc": 1}})
        r = _run_check_in(work)
        _assert(not r.ok(), "RR16", "a non-zero committed rc must FAIL")
        _assert(any("recorded rc=1" in f for f in r.failures), "RR16", f"{r.failures}")


def _real_anticipation_export(work: Path, slug: str, artifacts: list[dict],
                               mtimes: list[float],
                               relay_artifacts: list[dict] | None = None) -> str:
    """Fix round 5 Z5: runs the REAL `lead-gate-lib.py --export-anticipation`
    entry point (a subprocess, `CLAUDE_PROJECT_DIR=work` — the exact
    command the lead runs) against real `.jammi/gate-state/<slug>.
    anticipation.<tip>.json` artifact files, one per `artifacts[i]`, whose
    mtime is set to `mtimes[i]` — proving the fixture exercises the
    EXPORTED shape (real `ts`/`head_sha` stamped from the artifact file's
    own mtime/`pre_fix_sha`), never a hand-typed simulation of it.

    `relay_artifacts` (each a full relay-shaped dict carrying `agent_type`/
    `block_ts`/`fix_head` and a non-empty `mutations`/`exclusions`) are
    ALSO written to real `.jammi/gate-state/<slug>.relay.<agent_type>.
    <block_ts>.json` files, at the SAME path shape `relay_artifact_path`
    itself uses (loaded from `work`'s own copy of the real module), before
    the real export runs — so a fixture can exercise the REAL exporter
    against a mixed-source disk state (an anticipation artifact AND a
    relay artifact carrying non-empty `mutations`/`exclusions`) rather
    than a hand-typed simulation of it. Round-6 stop rule (Z18): the
    exporter writes stdout ONLY — no second, attestation-shaped file is
    ever produced, regardless of what `relay_artifacts` carries; the
    caller asserts that absence itself."""
    sdir = work / ".jammi" / "gate-state"
    sdir.mkdir(parents=True, exist_ok=True)
    for art, mtime in zip(artifacts, mtimes):
        tip = art["pre_fix_sha"]
        p = sdir / f"{slug}.anticipation.{tip}.json"
        p.write_text(json.dumps(art))
        os.utime(p, (mtime, mtime))
    if relay_artifacts:
        mod = _lib_module()
        for relay in relay_artifacts:
            p = mod.relay_artifact_path(sdir, slug, relay.get("agent_type", "adversarial-audit"),
                                         relay.get("block_ts", "2026-01-01T00:00:00Z"))
            p.write_text(json.dumps(relay))
    env = dict(os.environ)
    env["CLAUDE_PROJECT_DIR"] = str(work)
    proc = subprocess.run(
        [sys.executable, str(work / ".claude" / "hooks" / "lead-gate-lib.py"),
         "--export-anticipation", slug],
        capture_output=True, text=True, env=env, timeout=10,
    )
    _assert(proc.returncode == 0, "real export setup", f"--export-anticipation failed: {proc.stderr}")
    return proc.stdout


def fixture_rr17_interleaved_export_selects_by_ts_not_position() -> None:
    """fix round 4 Z2 / fix round 5 Z5: `cmd_export_anticipation` sorts
    artifacts by FILENAME (a tip sha -- pseudorandom hex, no chronological
    meaning), so an OLDER round's row can land on the LAST line of the
    committed export while a NEWER, green row sits earlier in the file.
    Exercises the REAL exporter (`_real_anticipation_export`) against two
    real artifact files whose mtimes are set OLDER-first/NEWER-second in
    wall-clock time but whose FILENAMES sort in the OPPOSITE order (the
    older, broken `rc=1` artifact's tip sorts AFTER the newer, green
    `rc=0` artifact's tip) -- the governing row must be the one with the
    greatest `ts` (the real, exported mtime), never the one nearest the
    end of the file; the fix ALLOWS."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        sha0 = _sh(work, "rev-parse", "HEAD")
        sha1 = _commit(work, "seed an intermediate ancestor tip", {"docs/README-fixture2.md": "seed2\n"})
        # tip_new sorts BEFORE tip_old alphabetically ("0..." < "f...") --
        # the OPPOSITE of chronological order, exactly fix round 4 F1's
        # own bug shape (an older round's artifact filename can sort
        # AFTER a newer one).
        tip_new, tip_old = "0" * 40, "f" * 40
        art_new = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": tip_new,
                   "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
                   "residual_risk": "fixture residual",
                   "gates": {"python3 ci/scripts/probe.py": {"rc": 0}}}
        art_old = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": tip_old,
                   "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
                   "residual_risk": "fixture residual",
                   "gates": {"python3 ci/scripts/probe.py": {"rc": 1}}}
        now = time.time()
        exported = _real_anticipation_export(work, "feat_rr-fixture", [art_old, art_new],
                                              [now - 3600.0, now])
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        files = {
            "ci/lead-gate-required-commands.txt": "python3 ci/scripts/probe.py  # measured ~0.1s\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": exported,
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        }
        _commit(work, "ci: touch a gate script (interleaved REAL export, older head last)", files)
        r = _run_check_in(work)
        _assert(r.ok(), "RR17", f"the greatest-ts (real, exported) row (rc=0) must govern despite "
                                  f"the older-ts row (rc=1) sorting LAST by filename: {r.failures}")


def fixture_rr18_ambiguous_pool_without_ts_fails_loudly() -> None:
    """fix round 5 Z5: the PRE-fix-round-5 shape -- two candidate rows,
    NEITHER carrying `ts`/`head_sha` at all (what `cmd_export_anticipation`
    produced before it started stamping them) -- is now a LOUD FAIL naming
    the ambiguity, never a silent `rows[0]`/append-order pick. RED at
    daebd948 by the executed probe: the old code's `max(pool, key=_row_ts)`
    with an all-empty `_row_ts` silently returned the FIRST row for every
    such pool, regardless of which one was actually current."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        row_a = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40, "agent_type": "lead-anticipation",
                 "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
                 "residual_risk": "fixture residual",
                 "gates": {"python3 ci/scripts/probe.py": {"rc": 0}}}
        row_b = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": "f" * 40, "agent_type": "lead-anticipation",
                 "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
                 "residual_risk": "fixture residual",
                 "gates": {"python3 ci/scripts/probe.py": {"rc": 1}}}
        files = {
            "ci/lead-gate-required-commands.txt": "python3 ci/scripts/probe.py  # measured ~0.1s\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": json.dumps(row_a) + "\n" + json.dumps(row_b) + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        }
        _commit(work, "ci: touch a gate script (no ts on either row)", files)
        r = _run_check_in(work)
        _assert(not r.ok(), "RR18", "two ts-less candidate rows must FAIL loudly, never silently "
                                     "pick one")
        _assert(any("AMBIGUOUS" in f for f in r.failures), "RR18", f"{r.failures}")


def fixture_rr20_omits_a_required_file_fails() -> None:
    """fix round 5 Z7: the BLOCK's own `finding_locations` names TWO files
    (`a.py`, `b.py`), but the anticipation record's `attacks` covers only
    ONE -- the omits-a-command arm, now enforced through the SAME shared
    validator (`_r12_anticipation_rejection`) reader 1 calls, must FAIL
    naming the omission. RED at daebd948 by the executed probe: the OLD
    inline `missing = required_files - covered_files` check this replaced
    happened to catch this case too, which is exactly why the shared
    function's OWN identical arm went unexercised (an `if False:` mutation
    on it killed NOTHING before this fixture existed)."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1", "b.py:1"],
                                 "class_enumeration": ["a.py:1", "b.py:1"]})
        anticipation_row = json.dumps({
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40, "agent_type": "lead-anticipation",
            "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
            "residual_risk": "fixture residual",
        })
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": anticipation_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR20", "an anticipation record omitting a required file must FAIL")
        _assert(any("omits" in f and "b.py" in f for f in r.failures), "RR20", f"{r.failures}")


def _rr_anticipation_commit(work: Path, art: dict) -> None:
    """Shared setup for RR21-24: one open second-round BLOCK naming
    a single required file (`a.py`), and `art` written verbatim as the
    anticipation record's own (only) row — `agent_type` defaults to
    `lead-anticipation` (the real exporter's own stamp, esc-lead-gate-R12
    fix round 6 Z12) unless a caller deliberately overrides it."""
    pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
    block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                             "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                             "class_enumeration": ["a.py:1"]})
    art = dict(art)
    art.setdefault("agent_type", "lead-anticipation")
    _commit(work, "ci: touch a gate script", {
        "ci/scripts/probe.py": "print('x')\n",
        "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
        "docs/rigor/feat_rr-fixture.anticipation.jsonl": json.dumps(art) + "\n",
        "docs/README-fixture.md": "line one\n",
        "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
    })


def fixture_rr21_no_unit_branch_fails() -> None:
    """fix round 5 Z7: a row with NO `unit_branch` at all -- reader 3
    inherits this check from the SAME shared validator reader 1 already
    enforces (a case the audit's A1-A5 probe found reader 3 previously
    never checked at all)."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_anticipation_commit(work, {
            "pre_fix_sha": "0" * 40,
            "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
            "residual_risk": "fixture residual",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR21", "a row with no unit_branch must FAIL")
        _assert(any("unit_branch" in f for f in r.failures), "RR21", f"{r.failures}")


def fixture_rr22_no_residual_risk_fails() -> None:
    """fix round 5 Z7: a row with NO `residual_risk` at all."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_anticipation_commit(work, {
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40,
            "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR22", "a row with no residual_risk must FAIL")
        _assert(any("residual_risk" in f for f in r.failures), "RR22", f"{r.failures}")


def fixture_rr23_identical_pair_reused_fails() -> None:
    """fix round 5 Z7: TWO keys in the SAME row reusing the IDENTICAL
    (command, hash) pair -- a templated attack, never a per-site
    examination."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        same = {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}
        _rr_anticipation_commit(work, {
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40,
            "attacks": {"a.py": dict(same), "b.py": dict(same)},
            "residual_risk": "fixture residual",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR23", "a reused (command, hash) pair must FAIL")
        _assert(any("IDENTICAL" in f for f in r.failures), "RR23", f"{r.failures}")


def fixture_rr24_inspector_only_fails() -> None:
    """fix round 5 Z7: the record's ONLY attack is inspector-class
    (`cat`) -- reading a file is not attacking a mechanism."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_anticipation_commit(work, {
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40,
            "attacks": {"a.py": {"command": "cat a.py", "hash": "a" * 64}},
            "residual_risk": "fixture residual",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR24", "an inspector-only record must FAIL")
        _assert(any("inspector-class" in f for f in r.failures), "RR24", f"{r.failures}")


def fixture_rr27_shared_validator_entry_not_object_denies() -> None:
    """issue #569: the shared validator's OWN "is not an object" arm
    (`_r12_anticipation_rejection` in `lead-gate-lib.py`) -- reader 3 has
    no by-file pre-check at all (its own preceding loop explicitly SKIPS a
    non-dict `entry`, see this file's own `check_anticipation_witnesses`),
    so a malformed `attacks[key]` reaches the shared validator's own arm
    DIRECTLY, never `_r12_validate_and_run_entry`'s per-entry runner
    (reader 1's OWN second implementation of the identical three checks).
    Binds to the arm's OWN producer text (`anticipation-validator:`),
    asserting the OTHER producer's text (`anticipation artifact`) is
    ABSENT -- the two are satisfiable by the same substring `is not an
    object` alone, which is exactly what let this arm's own coverage go
    unexercised before."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_anticipation_commit(work, {
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40,
            "attacks": {"a.py": "not-an-object"},
            "residual_risk": "fixture residual",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR27", "a non-object attacks[a.py] entry must FAIL")
        joined = " | ".join(r.failures)
        _assert("anticipation-validator: attacks['a.py'] is not an object" in joined, "RR27", joined)
        _assert("anticipation artifact" not in joined, "RR27", joined)


def fixture_rr28_shared_validator_entry_no_command_denies() -> None:
    """issue #569: the shared validator's OWN "has no `command`" arm."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_anticipation_commit(work, {
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40,
            "attacks": {"a.py": {"hash": "a" * 64}},
            "residual_risk": "fixture residual",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR28", "an attacks[a.py] entry with no command must FAIL")
        joined = " | ".join(r.failures)
        _assert("anticipation-validator: attacks['a.py'] has no `command`" in joined, "RR28", joined)
        _assert("anticipation artifact" not in joined, "RR28", joined)


def fixture_rr29_shared_validator_entry_no_valid_hash_denies() -> None:
    """issue #569: the shared validator's OWN "has no valid `hash`" arm."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_anticipation_commit(work, {
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40,
            "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "not-hex"}},
            "residual_risk": "fixture residual",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR29", "an attacks[a.py] entry with an invalid hash must FAIL")
        joined = " | ".join(r.failures)
        _assert("anticipation-validator: attacks['a.py'] has no valid `hash`" in joined, "RR29", joined)
        _assert("anticipation artifact" not in joined, "RR29", joined)


def fixture_rr25_mixed_stream_via_real_export_selects_anticipation_only() -> None:
    """esc-lead-gate-R12 fix round 6 Z12, round-6 stop rule Z18: the
    production-shaped case — ONE real anticipation artifact AND ONE real
    relay artifact carrying non-empty `mutations`/`exclusions`, both on
    disk, exported through the REAL `--export-anticipation` entry point
    (`_real_anticipation_export`'s relay-artifact arm). RED at 3273f51b:
    the pre-fix exporter interleaved an attestation row into the SAME
    stdout stream this fixture redirects into
    `docs/rigor/<slug>.anticipation.jsonl` — its own missing
    `residual_risk` denied `check_anticipation_witnesses` ("no non-empty
    `residual_risk`"), and its greatest-`ts` position could govern
    `check_required_gates`, hiding the real `gates` object entirely ("the
    governing row carries no `gates` object"). GREEN after the fix: the
    exporter writes stdout ONLY (the `lead-anticipation` row), reader 3
    ALLOWS, and — Z8's export half REVERTED at the round-6 stop rule — the
    on-disk relay's own `mutations`/`exclusions` produce NO
    `docs/rigor/<slug>.attestation.jsonl` file at all; they stay
    hook-attested only, visible in the relay artifact readers 1 and 2
    already read."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        art = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40,
               "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
               "residual_risk": "fixture residual",
               "gates": {"python3 ci/scripts/probe.py": {"rc": 0}}}
        relay = {"agent_type": "adversarial-audit", "block_ts": "2026-01-01T00:01:00Z",
                 "unit_branch": "feat/rr-fixture", "fix_head": "f" * 40,
                 "mutations": [{"site": "bar.py:4", "command": "true", "rc_before": 0,
                                 "rc_after": 1, "marker_after": "test result: FAILED"}],
                 "exclusions": {"tests/test_z.py::test_new": "does not cover X"}}
        now = time.time()
        exported = _real_anticipation_export(work, "feat_rr-fixture", [art], [now], relay_artifacts=[relay])
        files = {
            "ci/lead-gate-required-commands.txt": "python3 ci/scripts/probe.py  # measured ~0.1s\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": exported,
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        }
        _commit(work, "ci: touch a gate script (mixed-source real export)", files)
        r = _run_check_in(work)
        _assert(r.ok(), "RR25", f"reader 3 must select the anticipation row only and ALLOW: {r.failures}")
        exported_rows = [json.loads(line) for line in exported.splitlines() if line.strip()]
        _assert(bool(exported_rows) and all(row.get("agent_type") == "lead-anticipation" for row in exported_rows),
                "RR25", f"the anticipation stdout stream must carry ONLY lead-anticipation rows: {exported_rows}")
        attestation_path = work / "docs" / "rigor" / "feat_rr-fixture.attestation.jsonl"
        _assert(not attestation_path.exists(), "RR25",
                f"the exporter must write NO attestation file at all (Z8's export half is "
                f"reverted): {attestation_path}")


def fixture_rr26_foreign_row_in_anticipation_stream_fails_loudly() -> None:
    """esc-lead-gate-R12 fix round 6 Z12: a HAND-PLANTED foreign row — the
    shape a stale pre-fix-round-6 export or a hand-edit could still commit
    — sitting in the anticipation stream ALONGSIDE a real anticipation
    row. Reader 3 must NAME it and FAIL, never silently ignore it (which
    would leave a tampered/legacy row undetected) and never silently
    SELECT it (the RED-time bug RR25 exercises via the real export path)."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        anticipation_row = json.dumps({
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40, "agent_type": "lead-anticipation",
            "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
            "residual_risk": "fixture residual",
            "gates": {"python3 ci/scripts/probe.py": {"rc": 0}},
            "ts": "2026-01-01T00:02:00Z", "head_sha": "d" * 40,
        })
        foreign_row = json.dumps({
            "agent_type": "lead-relay-attestation", "unit_branch": "feat/rr-fixture",
            "mutations": [{"site": "bar.py:4", "command": "true", "rc_before": 0,
                            "rc_after": 1, "marker_after": "test result: FAILED"}],
            "ts": "2026-01-01T00:03:00Z", "head_sha": "d" * 40,
        })
        _commit(work, "ci: touch a gate script (hand-planted foreign row)", {
            "ci/scripts/probe.py": "print('x')\n",
            "ci/lead-gate-required-commands.txt": "python3 ci/scripts/probe.py  # measured ~0.1s\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": anticipation_row + "\n" + foreign_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR26", "a foreign row in the anticipation stream must FAIL")
        _assert(any("lead-relay-attestation" in f and "not `lead-anticipation`" in f for f in r.failures),
                "RR26", f"{r.failures}")


def fixture_rr30_tied_ts_governing_row_fails_loudly() -> None:
    """TWO candidate rows normalize to the SAME greatest `ts` instant —
    `max(pool, key=_row_instant)` resolves a tie by APPEND POSITION (the
    FIRST maximal element), so an `rc=0` row listed first would silently
    govern and shadow a genuinely `rc=1` sibling recorded at the identical
    instant. A tie is exactly as AMBIGUOUS as a missing `ts` and must FAIL
    the same way, in BOTH orders."""
    pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
    block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                             "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                             "class_enumeration": ["a.py:1"]})
    row_ok = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40, "agent_type": "lead-anticipation",
              "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
              "residual_risk": "fixture residual", "ts": "2026-01-01T00:02:00Z", "head_sha": "0" * 40,
              "gates": {"python3 ci/scripts/probe.py": {"rc": 0}}}
    row_broken = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": "f" * 40, "agent_type": "lead-anticipation",
                  "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
                  "residual_risk": "fixture residual", "ts": "2026-01-01T00:02:00Z", "head_sha": "f" * 40,
                  "gates": {"python3 ci/scripts/probe.py": {"rc": 1}}}
    for order_name, ordered in (("ok-first", [row_ok, row_broken]), ("broken-first", [row_broken, row_ok])):
        with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
            _origin, work = _pr_repo(Path(td))
            _commit(work, f"ci: touch a gate script (tied ts, {order_name})", {
                "ci/scripts/probe.py": "print('x')\n",
                "ci/lead-gate-required-commands.txt": "python3 ci/scripts/probe.py  # measured ~0.1s\n",
                "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
                "docs/rigor/feat_rr-fixture.anticipation.jsonl":
                    "\n".join(json.dumps(r) for r in ordered) + "\n",
                "docs/README-fixture.md": "line one\n",
                "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
            })
            r = _run_check_in(work)
            _assert(not r.ok(), "RR30", f"a tied greatest-ts pool ({order_name}) must FAIL loudly")
            _assert(any("AMBIGUOUS" in f and "tie" in f for f in r.failures), "RR30", f"{order_name}: {r.failures}")


def fixture_rr32_mixed_naive_aware_ts_pool_fails_loudly_not_a_crash() -> None:
    """issue #557 item 3: reproduces the EXACT mixed naive/aware pool an
    executed audit probe found in an earlier normalization attempt — one
    candidate row's `ts` carries NO UTC offset at all
    (`2026-01-01T00:03:00`), the other's carries an explicit one
    (`2026-01-01T00:03:00Z`). Comparing those two as parsed `datetime`
    objects without normalizing them to one comparable form first raises
    `TypeError: can't compare offset-naive and offset-aware datetimes`
    inside `max()`, aborting the whole self-test run rather than the
    promised loud AMBIGUOUS FAIL. This fixture asserts the LOUD FAIL: it
    calls `_run_check_in` with NO surrounding `try`/`except` of its own,
    so a regression that reintroduces the crash surfaces to `self_test()`'s
    own harness as `FAIL (unexpected exception)` — DISTINCT from a normal
    `_assert`-driven `FAIL`, and the property this fixture actually pins.
    The naive `ts` normalizes to `None` (refused as unparseable, the same
    bucket a missing `ts` already occupies) and is never fed into a
    `datetime` comparison against its aware sibling at all."""
    pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
    block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                             "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                             "class_enumeration": ["a.py:1"]})
    row_naive = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40, "agent_type": "lead-anticipation",
                 "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
                 "residual_risk": "fixture residual", "ts": "2026-01-01T00:03:00", "head_sha": "0" * 40,
                 "gates": {"python3 ci/scripts/probe.py": {"rc": 0}}}
    row_aware = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": "f" * 40, "agent_type": "lead-anticipation",
                 "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
                 "residual_risk": "fixture residual", "ts": "2026-01-01T00:03:00Z", "head_sha": "f" * 40,
                 "gates": {"python3 ci/scripts/probe.py": {"rc": 1}}}
    for order_name, ordered in (("naive-first", [row_naive, row_aware]), ("aware-first", [row_aware, row_naive])):
        with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
            _origin, work = _pr_repo(Path(td))
            _commit(work, f"ci: touch a gate script (mixed naive/aware ts, {order_name})", {
                "ci/scripts/probe.py": "print('x')\n",
                "ci/lead-gate-required-commands.txt": "python3 ci/scripts/probe.py  # measured ~0.1s\n",
                "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
                "docs/rigor/feat_rr-fixture.anticipation.jsonl":
                    "\n".join(json.dumps(r) for r in ordered) + "\n",
                "docs/README-fixture.md": "line one\n",
                "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
            })
            r = _run_check_in(work)  # no try/except -- a crash here is a DIFFERENT self-test failure shape
            _assert(not r.ok(), "RR32", f"a mixed naive/aware ts pool ({order_name}) must FAIL loudly")
            _assert(any("AMBIGUOUS" in f for f in r.failures), "RR32", f"{order_name}: {r.failures}")


def fixture_rr33_same_instant_different_text_ts_ties() -> None:
    """issue #557 item 3: two rows name the SAME instant in DIFFERENT
    text — a trailing `Z` (`2026-01-01T00:03:00Z`) vs the equivalent
    explicit offset (`2026-01-01T00:03:00+00:00`). A TEXT-only compare
    treats these as UNEQUAL and lets whichever sorts later silently
    govern; the normalized-INSTANT compare correctly recognizes them as
    the SAME instant and ties, exactly like `RR30`'s identical-text case."""
    pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
    block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                             "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                             "class_enumeration": ["a.py:1"]})
    row_z = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40, "agent_type": "lead-anticipation",
             "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
             "residual_risk": "fixture residual", "ts": "2026-01-01T00:03:00Z", "head_sha": "0" * 40,
             "gates": {"python3 ci/scripts/probe.py": {"rc": 0}}}
    row_offset = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": "f" * 40, "agent_type": "lead-anticipation",
                  "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
                  "residual_risk": "fixture residual", "ts": "2026-01-01T00:03:00+00:00", "head_sha": "f" * 40,
                  "gates": {"python3 ci/scripts/probe.py": {"rc": 1}}}
    for order_name, ordered in (("z-first", [row_z, row_offset]), ("offset-first", [row_offset, row_z])):
        with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
            _origin, work = _pr_repo(Path(td))
            _commit(work, f"ci: touch a gate script (same instant, different text, {order_name})", {
                "ci/scripts/probe.py": "print('x')\n",
                "ci/lead-gate-required-commands.txt": "python3 ci/scripts/probe.py  # measured ~0.1s\n",
                "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
                "docs/rigor/feat_rr-fixture.anticipation.jsonl":
                    "\n".join(json.dumps(r) for r in ordered) + "\n",
                "docs/README-fixture.md": "line one\n",
                "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
            })
            r = _run_check_in(work)
            _assert(not r.ok(), "RR33", f"a same-instant/different-text ts pool ({order_name}) must tie and FAIL")
            _assert(any("AMBIGUOUS" in f and "tie" in f for f in r.failures), "RR33", f"{order_name}: {r.failures}")


# ==========================================================================
# fix round 7 Z16(a), narrowed at fix round 7 Z20: a STRUCTURAL fixture
# (modeled on check_lead_gate.py's `R12sweepast`) that scans, by AST
# against the REAL file, every top-level FunctionDef in
# check_rigor_record.py for an `if` whose body calls `result.fail(...)`
# and whose test is an entry-shape check (an isinstance(x, dict) test over
# an attacks/mutations/exclusions loop variable, a `command`-presence
# test, or a `hash` fullmatch test). This is a NAMED, NARROW detector of
# the ONE duplicate shape fix round 6 Z14 deleted from
# `check_anticipation_witnesses` (a shape test bound to its own
# `result.fail(...)` call) — it is NOT a proof that the shared validator
# (`_r12_anticipation_rejection` in lead-gate-lib.py) is the only
# implementation of these checks anywhere. An executed audit found two
# gaps this detector cannot see: (1) a wholesale copy of the shared
# validator's own function body, pasted into this file, RETURNS deny text
# rather than calling `.fail(...)` — the same shape as the real function
# it copies — and trips no arm here; (2) `lead-gate-lib.py`'s OWN
# `_r12_validate_and_run_entry` (~:2072-2079) already re-implements the
# identical three checks, in the OTHER file this detector never scans, and
# always has. Z14's "the shared validator is the only implementation"
# property is therefore recorded, not claimed proven: this unit's own
# committed docs/rigor/lead-gate-r12-anticipation.anticipation.jsonl names
# it in `residual_risk` (see the docs, and RR31's own docstring, below).
# ==========================================================================

def _rr_for_loop_vars_over_r12_dicts(fn: ast.FunctionDef) -> set[str]:
    """Loop target name(s) of every `for ... in <X>:` inside `fn` whose
    iterated expression's own identifiers/string constants mention
    `attacks`/`mutations`/`exclusions` (e.g. `for file_key, entry in
    attacks.items():` — `entry` qualifies)."""
    names: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.For):
            continue
        iter_idents: set[str] = set()
        for sub in ast.walk(node.iter):
            if isinstance(sub, ast.Name):
                iter_idents.add(sub.id)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                iter_idents.add(sub.value)
        if not any(kw in ident for ident in iter_idents for kw in ("attacks", "mutations", "exclusions")):
            continue
        target = node.target
        elts = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
        names.update(e.id for e in elts if isinstance(e, ast.Name))
    return names


def _rr_get_key_assigned_vars(fn: ast.FunctionDef, key: str) -> set[str]:
    """Name(s) assigned, anywhere in `fn`, from a bare `<expr>.get(<key>)`
    call (e.g. `command = entry.get("command")`)."""
    names: set[str] = set()
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        val = node.value
        if (isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute) and val.func.attr == "get"
                and val.args and isinstance(val.args[0], ast.Constant) and val.args[0].value == key):
            names.add(node.targets[0].id)
    return names


def _rr_entry_shape_duplicate_violations(source: str) -> list[str]:
    """Fix round 7 Z16(a), narrowed at Z20: `[]` iff no top-level
    FunctionDef in `source` contains an `if` whose test is an
    attacks/mutations/exclusions entry-shape check (isinstance-dict/
    `command`-presence/`hash`-fullmatch) AND whose body calls
    `result.fail(...)`. Each violation names the enclosing function, the
    source line, and which arm it duplicates. This is a detector of ONE
    duplicate SHAPE (a shape test bound to its own `.fail(...)` call) —
    it does not see a duplicate that RETURNS deny text instead of calling
    `.fail(...)` directly (a wholesale copy of the shared validator's own
    function body would return, not fail, and passes this detector), and
    it never scans lead-gate-lib.py at all."""
    tree = ast.parse(source)
    violations: list[str] = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        dict_loop_vars = _rr_for_loop_vars_over_r12_dicts(fn)
        command_vars = _rr_get_key_assigned_vars(fn, "command")
        hash_vars = _rr_get_key_assigned_vars(fn, "hash")
        for node in ast.walk(fn):
            if not isinstance(node, ast.If):
                continue
            calls_fail = any(
                isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "fail"
                for stmt in node.body for sub in ast.walk(stmt)
            )
            if not calls_fail:
                continue  # a residual SKIP-only guard is not this smell
            test = node.test
            inner = test.operand if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) else test
            test_names = {n.id for n in ast.walk(test) if isinstance(n, ast.Name)}
            if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "isinstance"
                    and len(inner.args) == 2 and isinstance(inner.args[0], ast.Name)
                    and inner.args[0].id in dict_loop_vars
                    and isinstance(inner.args[1], ast.Name) and inner.args[1].id == "dict"):
                violations.append(f"{fn.name}:{node.lineno}: isinstance({inner.args[0].id}, dict) guard over "
                                   "an attacks/mutations/exclusions loop variable ALSO calls result.fail(...)")
            if test_names & command_vars:
                violations.append(f"{fn.name}:{node.lineno}: a `command`-presence test "
                                   f"({sorted(test_names & command_vars)}) ALSO calls result.fail(...)")
            has_fullmatch = any(isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                                 and sub.func.attr == "fullmatch" for sub in ast.walk(test))
            if has_fullmatch and (test_names & hash_vars):
                violations.append(f"{fn.name}:{node.lineno}: a `hash` fullmatch test "
                                   f"({sorted(test_names & hash_vars)}) ALSO calls result.fail(...)")
    return violations


# The EXACT duplicate fix round 6 Z14 deleted from `check_anticipation_
# witnesses` — reinserted TEMPORARILY, in a synthetic copy of the real
# source only, to prove `_rr_entry_shape_duplicate_violations` actually
# catches its return.
_RR_Z14_DUPLICATE_ANCHOR = (
    "            if not isinstance(entry, dict):\n"
    "                continue\n"
    "            command = entry.get(\"command\")\n"
    "            if not (isinstance(command, str) and command.strip()):\n"
    "                continue\n"
)
_RR_Z14_DUPLICATE_REPLACEMENT = (
    "            if not isinstance(entry, dict):\n"
    "                result.fail(f\"{path}:{lineno}: attacks[{file_key!r}] is not an object\")\n"
    "                continue\n"
    "            command = entry.get(\"command\")\n"
    "            recorded_hash = entry.get(\"hash\")\n"
    "            if not isinstance(command, str) or not command.strip():\n"
    "                result.fail(f\"{path}:{lineno}: attacks[{file_key!r}] has no `command`\")\n"
    "                continue\n"
    "            if not (isinstance(recorded_hash, str) and re.fullmatch(r\"[0-9a-f]{64}\", recorded_hash)):\n"
    "                result.fail(f\"{path}:{lineno}: attacks[{file_key!r}] has no valid `hash`\")\n"
    "                continue\n"
)


def fixture_rr31_no_fail_reporting_entry_shape_duplicate() -> None:
    """fix round 7 Z16(a), narrowed at Z20: STRUCTURAL — asserts, by AST
    against the REAL file, that `_rr_entry_shape_duplicate_violations`
    finds NOTHING in `check_rigor_record.py`'s own current source. Its
    universe, stated honestly: top-level FunctionDefs of THIS file whose
    `if` body calls `result.fail(...)` over an entry-shape test — a
    detector of the ONE duplicate shape fix round 6 deleted (a shape test
    bound to its own `.fail(...)` call), never a proof that the shared
    validator (`_r12_anticipation_rejection` in lead-gate-lib.py) is the
    ONLY implementation of these checks. It does not see a duplicate that
    RETURNS deny text instead of calling `.fail(...)` (a wholesale copy of
    the shared validator's own body would), and it never scans
    lead-gate-lib.py — which already carries its own second
    implementation of these three checks in `_r12_validate_and_run_entry`
    (~:2072-2079); this unit's own committed
    docs/rigor/lead-gate-r12-anticipation.anticipation.jsonl names that gap
    in `residual_risk`, recorded rather than hidden. RED half executed
    against fix round 6 Z14's own deleted duplicate, reinserted into a
    SYNTHETIC copy of this file's real source (never the file on disk):
    confirms this detector is not merely vacuously empty by accident."""
    source = Path(__file__).read_text()
    violations = _rr_entry_shape_duplicate_violations(source)
    _assert(not violations, "RR31", f"a duplicate entry-shape reporter re-appeared: {violations}")

    mutated = source.replace(_RR_Z14_DUPLICATE_ANCHOR, _RR_Z14_DUPLICATE_REPLACEMENT, 1)
    _assert(mutated != source, "RR31 setup",
            "the fix round 6 Z14 duplicate-restoration anchor did not match the real file — "
            "check_anticipation_witnesses' own residual loop must have moved")
    mutated_violations = _rr_entry_shape_duplicate_violations(mutated)
    _assert(bool(mutated_violations), "RR31",
            "restoring fix round 6 Z14's deleted duplicate must make this detector non-empty — it did not")


# --------------------------------------------------------------------------- #
# issue #557 items 1-2: the committed mutations/exclusions attestation
# record + its required, CI-derived reader (`check_attestation_witnesses`).
# --------------------------------------------------------------------------- #


def _rr_attestation_block_setup(work: Path, attestation_jsonl: str | None) -> None:
    """Shared setup: one open second-round BLOCK naming `a.py`
    (`finding_locations`), a real NEW `def compute_thing()` committed
    INSIDE `a.py` (so the diff's own AST-derived new-definition surfaces,
    `mod._parse_new_surfaces`, land inside a `finding_locations` file --
    item 8b's own arming condition), and (when `attestation_jsonl` is not
    `None`) a committed `docs/rigor/feat_rr-fixture.attestation.jsonl`."""
    pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
    block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                             "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                             "class_enumeration": ["a.py:1"]})
    files = {
        "ci/scripts/probe.py": "print('x')\n",
        "a.py": "def compute_thing():\n    return 1\n",
        "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
        "docs/README-fixture.md": "line one\n",
        "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
    }
    if attestation_jsonl is not None:
        files["docs/rigor/feat_rr-fixture.attestation.jsonl"] = attestation_jsonl
    _commit(work, "ci: touch a gate script (new def in a.py, item 8b armed)", files)


def fixture_rr34_missing_attestation_when_armed_fails() -> None:
    """issue #557 items 1-2: the fix's own diff adds `a.py::compute_thing`,
    a NEW definition inside `a.py` -- the SAME file the open BLOCK's own
    `finding_locations` names -- so item 8b is armed by RE-DERIVATION
    alone (no relay artifact exists or is read here at all). No
    `docs/rigor/feat_rr-fixture.attestation.jsonl` is committed. Must FAIL
    naming the export command."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_attestation_block_setup(work, attestation_jsonl=None)
        r = _run_check_in(work)
        _assert(not r.ok(), "RR34", "a new definition inside a finding_locations file with no "
                                      "committed attestation must FAIL")
        _assert(any("attestation.jsonl" in f and "--export-attestation" in f for f in r.failures),
                "RR34", f"{r.failures}")


def fixture_rr35_attestation_file_with_no_valid_row_fails() -> None:
    """issue #557 items 1-2: a committed attestation file EXISTS, but its
    one row's `mutations` is an empty array (fails
    `_r12_mutations_array_rejection`'s own shape check) -- no row
    satisfies item 8b's requirement, so this must still FAIL, naming that
    no committed row carries a shape-valid array."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        bad_row = json.dumps({"kind": "lead-relay-attestation", "unit_branch": "feat/rr-fixture",
                               "agent_type": "adversarial-audit", "mutations": []})
        _rr_attestation_block_setup(work, attestation_jsonl=bad_row + "\n")
        r = _run_check_in(work)
        _assert(not r.ok(), "RR35", "an attestation file with no shape-valid row must FAIL")
        _assert(any("no committed row carries a shape-valid" in f for f in r.failures), "RR35", f"{r.failures}")


def fixture_rr36_foreign_attestation_row_refused() -> None:
    """issue #557 items 1-2: a committed attestation file carries a row
    whose `kind` is NOT `lead-relay-attestation` (a hand-edit, or a row
    copied from the anticipation stream) -- REFUSED by name, never
    silently ignored or selected as evidence."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        foreign_row = json.dumps({"kind": "lead-anticipation", "unit_branch": "feat/rr-fixture",
                                   "attacks": {}, "residual_risk": "x"})
        _rr_attestation_block_setup(work, attestation_jsonl=foreign_row + "\n")
        r = _run_check_in(work)
        _assert(not r.ok(), "RR36", "a foreign-kind row in the attestation stream must FAIL")
        _assert(any("not `lead-relay-attestation`" in f for f in r.failures), "RR36", f"{r.failures}")


def fixture_rr37_valid_attestation_row_satisfies() -> None:
    """issue #557 items 1-2, positive control: a committed attestation
    file carries ONE real, shape-valid `lead-relay-attestation` row with a
    non-empty `mutations` array -- item 8b's own requirement is satisfied;
    `check_attestation_witnesses` itself reports NO failure (other readers
    in the same run, e.g. the anticipation record, are not this fixture's
    concern and are asserted separately elsewhere)."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        good_row = json.dumps({
            "kind": "lead-relay-attestation", "unit_branch": "feat/rr-fixture",
            "agent_type": "adversarial-audit", "block_ts": "2026-01-01T00:01:00Z",
            "mutations": [{"site": "a.py:1", "command": "python3 -c \"print('ok')\"",
                            "rc_before": 0, "rc_after": 1, "marker_after": "test result: FAILED"}],
        })
        _rr_attestation_block_setup(work, attestation_jsonl=good_row + "\n")
        r = _run_check_in(work)
        joined_failures = " | ".join(r.failures)
        _assert("attestation" not in joined_failures, "RR37",
                f"a shape-valid attestation row must satisfy item 8b, got: {r.failures}")


# --------------------------------------------------------------------------- #
# issue #570: the bidirectional # R12-RESIDUAL <-> residual_risk property.
# --------------------------------------------------------------------------- #


def _rr_residual_setup(work: Path, extra_lib_marker: str | None, residual_risk: str) -> None:
    """Commits a real anticipation record with `residual_risk` set as
    given, plus (when `extra_lib_marker` is not `None`) a MUTATED copy of
    `.claude/hooks/lead-gate-lib.py` that injects `extra_lib_marker` as a
    NEW statement right inside the real `slugify` function -- a stable,
    always-present anchor every copy of the file carries -- so THIS
    unit's own diff (against `origin/main`'s UNMUTATED copy, from
    `_pr_repo`'s own scaffold commit) adds exactly one NEW `#
    R12-RESIDUAL` line, never the file's whole pre-existing history."""
    if extra_lib_marker is not None:
        lib_path = work / ".claude" / "hooks" / "lead-gate-lib.py"
        original = lib_path.read_text()
        anchor = "def slugify(branch: str) -> str:\n"
        if anchor not in original:  # pragma: no cover - guards fixture drift, not swarm behavior
            raise AssertionError("RR38/39/40 anchor drifted from the real lead-gate-lib.py's slugify()")
        mutated = original.replace(anchor, anchor + f"    pass  {extra_lib_marker}\n", 1)
        lib_path.write_text(mutated)
    pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
    block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                             "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                             "class_enumeration": ["a.py:1"]})
    anticipation_row = json.dumps({
        "unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40, "agent_type": "lead-anticipation",
        "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
        "residual_risk": residual_risk,
        "gates": {_RR_BASELINE_REQUIRED_COMMAND: {"rc": 0}},
    })
    _commit(work, "fix: touch a gate script + lead-gate-lib.py residual state", {
        "ci/scripts/probe.py": "print('x')\n",
        "a.py": "def compute_thing():\n    return 1\n",
        "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
        "docs/rigor/feat_rr-fixture.anticipation.jsonl": anticipation_row + "\n",
        "docs/README-fixture.md": "line one\n",
        "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
    })


def fixture_rr38_new_residual_marker_not_named_forward_fails() -> None:
    """issue #570 FORWARD: the fix's own diff adds a NEW `#
    R12-RESIDUAL:` marker inside `slugify` in lead-gate-lib.py, but the
    governing row's `residual_risk` never names `slugify`. Must FAIL."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_residual_setup(work, "# R12-RESIDUAL: fixture-injected residual for RR38",
                            residual_risk="unrelated residual text naming nothing real")
        r = _run_check_in(work)
        _assert(not r.ok(), "RR38", "a new residual marker not named in residual_risk must FAIL")
        _assert(any("does not name" in f and "slugify" in f for f in r.failures), "RR38", f"{r.failures}")


def fixture_rr39_over_claimed_residual_backward_fails() -> None:
    """issue #570 BACKWARD: `residual_risk` backtick-cites `state_dir` — a
    REAL function in lead-gate-lib.py that carries NO `# R12-RESIDUAL`
    marker anywhere — an over-claimed residual. Must FAIL by name, even
    though `residual_risk` ALSO correctly names `slugify` (the function
    the fixture's own new marker actually sits inside), satisfying
    FORWARD: BACKWARD's own over-claim check is independent of whether
    FORWARD passed."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        lib_text = (work / ".claude" / "hooks" / "lead-gate-lib.py").read_text()
        marker_lines = [l for l in lib_text.splitlines() if _R12_RESIDUAL_MARKER_RE.search(l)]
        _assert("def state_dir(" in lib_text and not any("state_dir" in l for l in marker_lines),
                "RR39 setup", "`state_dir` must be real and carry no marker in the unmutated file")
        _rr_residual_setup(
            work, "# R12-RESIDUAL: fixture-injected residual for RR39",
            residual_risk="the `slugify` gap is real; see `state_dir` for another untracked one",
        )
        r = _run_check_in(work)
        _assert(not r.ok(), "RR39", "an over-claimed residual (citing an unmarked real function) must FAIL")
        _assert(any("state_dir" in f and "no `# R12-RESIDUAL` marker" in f for f in r.failures),
                "RR39", f"{r.failures}")


def fixture_rr40_correctly_named_residual_satisfies() -> None:
    """issue #570, positive control: `residual_risk` names `slugify`,
    exactly the function the new marker sits inside -- both FORWARD and
    BACKWARD are satisfied; `check_residual_risk_bidirectional` itself
    reports no failure."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_residual_setup(work, "# R12-RESIDUAL: fixture-injected residual for RR40",
                            residual_risk="the `slugify` fixture-injected gap for RR40 is untracked")
        r = _run_check_in(work)
        joined = " | ".join(r.failures)
        _assert("issue #570" not in joined, "RR40",
                f"a correctly-named residual must satisfy #570, got: {r.failures}")


# --------------------------------------------------------------------------- #
# issue #557 item 2: the CI-derived required call-site set
# (`check_required_call_site_set`), a real `syn` AST parse via
# `ci/tools/symbol-index` -- never a regex reader, never a hand list.
# --------------------------------------------------------------------------- #

def _rr_call_site_setup(work: Path, rust_file_content: str, mutations_site: str) -> None:
    """Shared setup for RR41-43: an open second-round BLOCK naming
    `src/lib.rs` (`finding_locations`), a real committed `src/lib.rs` (so
    item 8b's own `mod._parse_new_surfaces` arming ALSO fires -- the SAME
    arming condition `_rr_attestation_block_setup` uses for `a.py`, a
    `.rs` file this time so `check_required_call_site_set` arms too), plus
    a `lead-relay-attestation` row whose ONE `mutations` entry's `site` is
    `mutations_site` -- the caller's own choice, real for a positive
    control, fabricated for the phantom-site fixture."""
    pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
    block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                             "verdict": "BLOCK", "finding_locations": ["src/lib.rs:1"],
                             "class_enumeration": ["src/lib.rs:1"]})
    good_row = json.dumps({
        "kind": "lead-relay-attestation", "unit_branch": "feat/rr-fixture",
        "agent_type": "adversarial-audit", "block_ts": "2026-01-01T00:01:00Z",
        "mutations": [{"site": mutations_site, "command": "python3 -c \"print('ok')\"",
                        "rc_before": 0, "rc_after": 1, "marker_after": "test result: FAILED"}],
    })
    files = {
        "ci/scripts/probe.py": "print('x')\n",
        "src/lib.rs": rust_file_content,
        "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
        "docs/rigor/feat_rr-fixture.attestation.jsonl": good_row + "\n",
        "docs/README-fixture.md": "line one\n",
        "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
    }
    _commit(work, "ci: touch a gate script (new .rs def, item 8b + #557 item 2 armed)", files)


def fixture_rr41_mutation_site_resolves_to_real_def_satisfies() -> None:
    """issue #557 item 2, positive control: `mutations[0].site` names the
    EXACT line of a real, newly-added `pub fn compute_new()` in
    `src/lib.rs` -- the symbol-index-derived required set's own `new_defs`
    entry. `check_required_call_site_set` must report no failure."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        rust = "pub fn compute_new() -> i32 {\n    1\n}\n"
        _rr_call_site_setup(work, rust, mutations_site="src/lib.rs:1")
        r = _run_check_in(work)
        joined = " | ".join(r.failures)
        _assert("phantom site" not in joined, "RR41",
                f"a mutations site naming a real new definition must resolve, got: {r.failures}")


def fixture_rr42_phantom_mutation_site_fails() -> None:
    """issue #557 item 2: `mutations[0].site` names `src/lib.rs:999` --
    the real committed file has only 3 lines. Must FAIL by name as a
    phantom site, never merely accepted because the FILE exists."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        rust = "pub fn compute_new() -> i32 {\n    1\n}\n"
        _rr_call_site_setup(work, rust, mutations_site="src/lib.rs:999")
        r = _run_check_in(work)
        _assert(not r.ok(), "RR42", "a mutations site naming a non-existent line must FAIL")
        _assert(any("phantom site" in f and "src/lib.rs:999" in f for f in r.failures),
                "RR42", f"{r.failures}")


def fixture_rr43_mutation_site_resolves_to_real_call_satisfies() -> None:
    """issue #557 item 2, positive control (the CALL-SITE half of the
    required set, not just definitions): `mutations[0].site` names the
    EXACT line of a real, newly-added call expression (`helper()` inside
    `compute_new`) -- resolved via the index's own `calls` list, not its
    `items` list. Must satisfy; proves `_r12_site_resolves` checks BOTH,
    never definitions alone."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        rust = "pub fn compute_new() -> i32 {\n    helper()\n}\n\npub fn helper() -> i32 {\n    1\n}\n"
        _rr_call_site_setup(work, rust, mutations_site="src/lib.rs:2")
        r = _run_check_in(work)
        joined = " | ".join(r.failures)
        _assert("phantom site" not in joined, "RR43",
                f"a mutations site naming a real call expression must resolve, got: {r.failures}")


RR_FIXTURES = [
    ("RR1", fixture_rr1_not_armed_docs_only),
    ("RR2", fixture_rr2_armed_no_record),
    ("RR3", fixture_rr3_record_no_pressure_row),
    ("RR4", fixture_rr4_no_contract_file),
    ("RR5", fixture_rr5_full_disclosure_allows),
    ("RR6", fixture_rr6_bad_citation_fails),
    ("RR7", fixture_rr7_near_identical_contract_fails),
    ("RR8", fixture_rr8_noop_shapes),
    ("RR9", fixture_rr9_ancestry_advisory_only),
    ("RR10", fixture_rr10_allowlisted_unit_noop),
    ("RR11", fixture_rr11_allowlist_only_shrinks),
    ("RR12a", fixture_rr12a_no_open_block_row_is_a_noop),
    ("RR12b", fixture_rr12b_open_block_no_anticipation_file_fails),
    ("RR12c", fixture_rr12c_shape_allows_despite_unresolvable_pre_fix_sha),
    ("RR12d", fixture_rr12d_real_reproducing_witness_allows),
    ("RR12e", fixture_rr12e_real_nonreproducing_witness_warns_never_fails),
    ("RR12f", fixture_rr12f_grandfathered_unit_no_file_allows),
    ("RR12g", fixture_rr12g_tracked_bash_path_allows),
    ("RR12h", fixture_rr12h_untracked_bash_path_fails_shape),
    ("RR13", fixture_rr13_r12_grandfather_only_shrinks),
    ("RR19", fixture_rr19_required_commands_only_shrinks),
    ("RR20", fixture_rr20_omits_a_required_file_fails),
    ("RR21", fixture_rr21_no_unit_branch_fails),
    ("RR22", fixture_rr22_no_residual_risk_fails),
    ("RR23", fixture_rr23_identical_pair_reused_fails),
    ("RR24", fixture_rr24_inspector_only_fails),
    ("RR27", fixture_rr27_shared_validator_entry_not_object_denies),
    ("RR28", fixture_rr28_shared_validator_entry_no_command_denies),
    ("RR29", fixture_rr29_shared_validator_entry_no_valid_hash_denies),
    ("RR14", fixture_rr14_missing_gates_fails),
    ("RR15", fixture_rr15_complete_gates_rc_zero_allows),
    ("RR16", fixture_rr16_nonzero_rc_fails),
    ("RR17", fixture_rr17_interleaved_export_selects_by_ts_not_position),
    ("RR18", fixture_rr18_ambiguous_pool_without_ts_fails_loudly),
    ("RR25", fixture_rr25_mixed_stream_via_real_export_selects_anticipation_only),
    ("RR26", fixture_rr26_foreign_row_in_anticipation_stream_fails_loudly),
    ("RR30", fixture_rr30_tied_ts_governing_row_fails_loudly),
    ("RR31", fixture_rr31_no_fail_reporting_entry_shape_duplicate),
    ("RR32", fixture_rr32_mixed_naive_aware_ts_pool_fails_loudly_not_a_crash),
    ("RR33", fixture_rr33_same_instant_different_text_ts_ties),
    ("RR34", fixture_rr34_missing_attestation_when_armed_fails),
    ("RR35", fixture_rr35_attestation_file_with_no_valid_row_fails),
    ("RR36", fixture_rr36_foreign_attestation_row_refused),
    ("RR37", fixture_rr37_valid_attestation_row_satisfies),
    ("RR38", fixture_rr38_new_residual_marker_not_named_forward_fails),
    ("RR39", fixture_rr39_over_claimed_residual_backward_fails),
    ("RR40", fixture_rr40_correctly_named_residual_satisfies),
    ("RR41", fixture_rr41_mutation_site_resolves_to_real_def_satisfies),
    ("RR42", fixture_rr42_phantom_mutation_site_fails),
    ("RR43", fixture_rr43_mutation_site_resolves_to_real_call_satisfies),
]


def self_test() -> int:
    failures: list[str] = []
    for name, fn in RR_FIXTURES:
        try:
            fn()
            print(f"check-rigor-record[{name}]: OK")
        except Failure as e:
            failures.append(f"{name}: {e}")
            print(f"check-rigor-record[{name}]: FAIL — {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: unexpected exception: {e!r}")
            print(f"check-rigor-record[{name}]: FAIL (unexpected exception) — {e!r}", file=sys.stderr)
    if failures:
        print("check-rigor-record: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"check-rigor-record: all {len(RR_FIXTURES)} self-test fixture(s) passed.")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    if "--check-allowlist-only-shrinks" in argv:
        return check_allowlist_only_shrinks()
    if "--check-r12-grandfather-only-shrinks" in argv:
        return check_r12_grandfather_only_shrinks()
    if "--check-required-commands-only-shrinks" in argv:
        return check_required_commands_only_shrinks()
    result = run_check()
    for w in result.warnings:
        print(f"check-rigor-record: WARNING (advisory) — {w}")
    if not result.ok():
        print("check-rigor-record: FAIL", file=sys.stderr)
        for f in result.failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("check-rigor-record: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
