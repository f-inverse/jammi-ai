#!/usr/bin/env python3
"""check_ship_register.py — constitution T1's CI leg: an audit-round sequence
terminates only in operator acceptance of an assurance-deficit register
(`.jammi/registers/<unit_slug>.register.json`, tracked).

Two legs, both always-run (no `paths:` filter; in-job diff detection):

  TRIGGER   — if a PR's diff moves any `.jammi/escapes.jsonl` id's `status`
              (present in both base and head, values differ — seeding and
              archival do not trigger), the same diff must carry >=1 register
              under `.jammi/registers/` whose `seeded_escapes` (union over
              registers in the diff) covers every moved id. No base ref
              (push to main) -> SKIP, recorded distinctly, never silent.

  PER-REGISTER (every register in the tree, and mandatorily every register in
  the diff) — G1-G8:

    G1  parses; closing_rows non-empty, <=1 row per agent_type, verdict in
        {BLOCK, PASS}; RED on any non-null unparseable_reason.
    G2  the ship criterion: zero findings_index entries with
        severity=block ^ stands=true ^ liveness=live across the union of
        closing rows, except one keyed by a residual carrying
        operator_accepted_live:true (missing liveness on block severity
        counts live, fail-closed); RED also when finding_locations is
        non-empty while findings_index is empty/absent.
    G3  coverage: residuals keys are an exact-string superset of every
        class_enumeration string (all standing) union every findings_index
        location with stands=true. No parsing, no normalization.
    G4  no reclassification (F7): every residual's liveness string-equals
        its findings_index entry; operator_accepted_live accepts a live
        entry, never rewrites it.
    G5  sweep honesty: non-empty sweep_method; fixture_ref resolves to a
        tracked file; non-empty justification/owner; exhaustive:false
        requires the residual nonexhaustive-sweep:<agent_type>.
    G6  eval_added-not-open (unit-62 precedent): every seeded_escapes id
        exists in escapes.jsonl with status != open.
    G7  non-vacuity (esc-063): the register-directory scan is proven
        non-vacuous, not just its output; a closing row with non-empty
        finding_locations requires >=1 residual.
    G8  freshness: every closing head_sha is an ancestor of PR head and no
        commit after the max-ts closing sha touches the unit surface
        (computed via `git diff --name-only`, minus the assembly artifacts
        `.jammi/registers/**` / `.jammi/escapes.jsonl`); the embedded
        unit_surface additionally covers the PR's own diff paths when a
        base ref exists (else SKIP, recorded).

`--local` (lead/operator only; gate-state is gitignored) byte-diffs each
embedded closing row against the live `.jammi/gate-state/<slug>.jsonl` row of
the same (agent_type, ts), confirms the closing rows really are the max-ts
row of every lane that produced a bound row, confirms every standing finding
of an earlier bound row appears in its lane's closing enumeration or in
residuals, and prints the advisory-signals table (INFO only, never a gate
input).

`--self-test` drives the gates against fixture registers/escapes/git states.
REPO_ROOT resolves with a marker-file assertion (the esc-063 resolution
pattern: a wrong REPO_ROOT must fail loudly, never silently enumerate zero
files and report PASS).

Run: `python3 ci/scripts/check_ship_register.py` (CI mode)
     `python3 ci/scripts/check_ship_register.py --self-test`
     `python3 ci/scripts/check_ship_register.py --local` (lead/operator only)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
assert (REPO_ROOT / "Cargo.toml").is_file(), (
    f"check_ship_register.py: REPO_ROOT resolved to {REPO_ROOT}, which carries no "
    "Cargo.toml -- the marker-file assertion caught a wrong path depth (the esc-063 "
    "shape: a silently-wrong REPO_ROOT enumerates zero files and reports PASS)."
)

REGISTERS_SUBDIR = Path(".jammi") / "registers"
ESCAPES_SUBPATH = Path(".jammi") / "escapes.jsonl"
GATE_STATE_SUBDIR = Path(".jammi") / "gate-state"


class Failure(Exception):
    pass


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _is_assembly_artifact(path: str) -> bool:
    return path == str(ESCAPES_SUBPATH) or path.startswith(str(REGISTERS_SUBDIR) + os.sep) \
        or path.startswith(str(REGISTERS_SUBDIR) + "/") or path.startswith(str(ESCAPES_SUBPATH))


def resolve_diff_base(repo_root: Path) -> str | None:
    """Resolve a git ref to diff against, or None if none is available (a
    push-to-main / no-PR run) -- same fallback chain as
    check_no_consumer_names.py's resolve_diff_base()."""
    candidates = [
        os.environ.get("SWARM_DIFF_BASE"),
        f"origin/{os.environ['GITHUB_BASE_REF']}" if os.environ.get("GITHUB_BASE_REF") else None,
    ]
    for ref in candidates:
        if not ref:
            continue
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=repo_root, capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return ref
    return None


# --------------------------------------------------------------------------
# escapes.jsonl reading
# --------------------------------------------------------------------------

def _parse_escapes_text(text: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        eid = row.get("id") if isinstance(row, dict) else None
        if isinstance(eid, str):
            out[eid] = row
    return out


def read_escapes(repo_root: Path) -> dict[str, dict]:
    path = repo_root / ESCAPES_SUBPATH
    if not path.is_file():
        return {}
    return _parse_escapes_text(path.read_text())


def read_escapes_at_ref(repo_root: Path, ref: str) -> dict[str, dict]:
    p = _run_git(["show", f"{ref}:{ESCAPES_SUBPATH.as_posix()}"], repo_root)
    if p.returncode != 0:
        return {}
    return _parse_escapes_text(p.stdout)


# --------------------------------------------------------------------------
# Register discovery
# --------------------------------------------------------------------------

def find_registers(repo_root: Path) -> list[Path]:
    d = repo_root / REGISTERS_SUBDIR
    if not d.is_dir():
        return []
    return sorted(d.glob("*.register.json"))


def registers_changed_in_diff(repo_root: Path, base_ref: str, head_ref: str = "HEAD") -> list[Path]:
    p = _run_git(["diff", "--name-only", f"{base_ref}...{head_ref}", "--",
                  REGISTERS_SUBDIR.as_posix()], repo_root)
    if p.returncode != 0:
        return []
    out = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        path = repo_root / line
        if path.is_file():
            out.append(path)
    return out


def load_register(path: Path) -> tuple[dict | None, str | None]:
    try:
        text = path.read_text()
    except OSError as exc:
        return None, f"cannot read {path}: {exc}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"{path} is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"{path} is not a JSON object"
    return data, None


# --------------------------------------------------------------------------
# TRIGGER leg
# --------------------------------------------------------------------------

def trigger_leg(repo_root: Path, base_ref: str | None, head_ref: str = "HEAD") -> tuple[str, list[str]]:
    """Returns (status, failures); status in {"PASS", "RED", "SKIP"}."""
    if not base_ref:
        return "SKIP", []
    base_escapes = read_escapes_at_ref(repo_root, base_ref)
    head_escapes = read_escapes(repo_root)
    moved = sorted(
        eid for eid, row in head_escapes.items()
        if eid in base_escapes and row.get("status") != base_escapes[eid].get("status")
    )
    if not moved:
        return "PASS", []
    covered: set[str] = set()
    for path in registers_changed_in_diff(repo_root, base_ref, head_ref):
        reg, err = load_register(path)
        if reg is None:
            continue
        covered.update(s for s in (reg.get("seeded_escapes") or []) if isinstance(s, str))
    uncovered = [eid for eid in moved if eid not in covered]
    if uncovered:
        return "RED", [
            f"trigger: escape status moved for {uncovered} but no register in the diff's "
            "seeded_escapes covers all of them"
        ]
    return "PASS", []


# --------------------------------------------------------------------------
# Per-register gates G1-G7 (pure data; G8 needs git)
# --------------------------------------------------------------------------

def g1_parses(reg: dict) -> list[str]:
    failures: list[str] = []
    closing_rows = reg.get("closing_rows")
    if not isinstance(closing_rows, list) or not closing_rows:
        failures.append("G1: closing_rows missing or empty")
        return failures
    seen_types: set[str] = set()
    for row in closing_rows:
        if not isinstance(row, dict):
            failures.append("G1: a closing_rows entry is not an object")
            continue
        at = row.get("agent_type")
        if not isinstance(at, str) or not at:
            failures.append("G1: a closing row has no agent_type")
            continue
        if at in seen_types:
            failures.append(f"G1: more than one closing row for agent_type {at!r}")
        seen_types.add(at)
        if row.get("verdict") not in ("BLOCK", "PASS"):
            failures.append(f"G1: closing row {at!r} verdict is not BLOCK/PASS: {row.get('verdict')!r}")
        if row.get("unparseable_reason") is not None:
            failures.append(
                f"G1: closing row {at!r} carries a non-null unparseable_reason "
                f"{row.get('unparseable_reason')!r} -- an UNPARSEABLE row laundered as text"
            )
    return failures


def _closing_rows(reg: dict) -> list[dict]:
    rows = reg.get("closing_rows")
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _is_live(entry: dict) -> bool:
    """Fail-closed: a block-severity entry with no liveness field is live."""
    liveness = entry.get("liveness")
    if liveness == "live":
        return True
    if liveness == "latent":
        return False
    return entry.get("severity") == "block"


def g2_ship_criterion(reg: dict) -> list[str]:
    failures: list[str] = []
    residuals = reg.get("residuals")
    residuals = residuals if isinstance(residuals, dict) else {}
    for row in _closing_rows(reg):
        at = row.get("agent_type")
        fl = row.get("finding_locations")
        fi = row.get("findings_index")
        fl_nonempty = isinstance(fl, list) and len(fl) > 0
        fi_nonempty = isinstance(fi, list) and len(fi) > 0
        if fl_nonempty and not fi_nonempty:
            failures.append(
                f"G2: closing row {at!r} has non-empty finding_locations but empty/absent "
                "findings_index -- the pre-liveness-hook skew/harvest-failure state"
            )
        for entry in (fi if isinstance(fi, list) else []):
            if not isinstance(entry, dict):
                continue
            loc = entry.get("location")
            if entry.get("severity") != "block" or entry.get("stands") is not True:
                continue
            if not _is_live(entry):
                continue
            res = residuals.get(loc) if isinstance(loc, str) else None
            if isinstance(res, dict) and res.get("operator_accepted_live") is True:
                continue
            failures.append(
                f"G2: standing live block finding {loc!r} ({at!r}) ships with no "
                "operator_accepted_live residual -- T1's ship criterion"
            )
    return failures


def g3_coverage(reg: dict) -> list[str]:
    residuals = reg.get("residuals")
    residual_keys = set(residuals.keys()) if isinstance(residuals, dict) else set()
    required: set[str] = set()
    for row in _closing_rows(reg):
        for s in (row.get("class_enumeration") or []):
            if isinstance(s, str):
                required.add(s)
        for entry in (row.get("findings_index") or []):
            if isinstance(entry, dict) and entry.get("stands") is True:
                loc = entry.get("location")
                if isinstance(loc, str):
                    required.add(loc)
    missing = required - residual_keys
    if missing:
        return [f"G3: residuals missing exact-string coverage for {sorted(missing)}"]
    return []


def g4_no_reclassification(reg: dict) -> list[str]:
    failures: list[str] = []
    index_by_loc: dict[str, str | None] = {}
    for row in _closing_rows(reg):
        for entry in (row.get("findings_index") or []):
            if isinstance(entry, dict):
                loc = entry.get("location")
                if isinstance(loc, str):
                    index_by_loc[loc] = entry.get("liveness")
    residuals = reg.get("residuals")
    if isinstance(residuals, dict):
        for key, res in residuals.items():
            if not isinstance(res, dict) or key not in index_by_loc:
                continue
            idx_live = index_by_loc[key]
            res_live = res.get("liveness")
            if res_live != idx_live:
                failures.append(
                    f"G4: residual {key!r} liveness {res_live!r} does not string-equal its "
                    f"findings_index liveness {idx_live!r} -- residuals are never reclassified"
                )
    return failures


def g5_sweep_honesty(reg: dict, repo_root: Path) -> list[str]:
    failures: list[str] = []
    residuals = reg.get("residuals")
    residuals = residuals if isinstance(residuals, dict) else {}
    for row in _closing_rows(reg):
        at = row.get("agent_type")
        sm = row.get("sweep_method")
        if not isinstance(sm, str) or not sm.strip():
            failures.append(f"G5: closing row {at!r} has an empty/non-string sweep_method")
        if row.get("exhaustive") is False:
            key = f"nonexhaustive-sweep:{at}"
            if key not in residuals:
                failures.append(
                    f"G5: closing row {at!r} is exhaustive:false but carries no residual {key!r}"
                )
    for key, res in residuals.items():
        if not isinstance(res, dict):
            failures.append(f"G5: residual {key!r} is not an object")
            continue
        fref = res.get("fixture_ref")
        if not isinstance(fref, str) or not fref.strip() or not (repo_root / fref).is_file():
            failures.append(f"G5: residual {key!r} fixture_ref does not resolve to a tracked file: {fref!r}")
        if not isinstance(res.get("justification"), str) or not res.get("justification").strip():
            failures.append(f"G5: residual {key!r} has an empty justification")
        if not isinstance(res.get("owner"), str) or not res.get("owner").strip():
            failures.append(f"G5: residual {key!r} has an empty owner")
    return failures


def g6_eval_added_not_open(reg: dict, escapes_by_id: dict[str, dict]) -> list[str]:
    failures: list[str] = []
    for eid in (reg.get("seeded_escapes") or []):
        if not isinstance(eid, str):
            continue
        row = escapes_by_id.get(eid)
        if row is None:
            failures.append(f"G6: seeded_escapes id {eid!r} does not exist in escapes.jsonl")
        elif row.get("status") == "open":
            failures.append(f"G6: seeded_escapes id {eid!r} is still status=open (unit-62 precedent)")
    return failures


def g7_non_vacuity_per_register(reg: dict) -> list[str]:
    any_fl = any(bool(row.get("finding_locations")) for row in _closing_rows(reg))
    residuals = reg.get("residuals")
    residuals_nonempty = isinstance(residuals, dict) and len(residuals) > 0
    if any_fl and not residuals_nonempty:
        return ["G7: a closing row carries finding_locations but residuals is empty (esc-063 shape)"]
    return []


def g8_freshness(reg: dict, *, repo_root: Path, head_ref: str = "HEAD",
                  base_ref: str | None = None) -> tuple[list[str], list[str]]:
    """Returns (failures, skip_notes)."""
    failures: list[str] = []
    skips: list[str] = []
    unit_surface = set(s for s in (reg.get("unit_surface") or []) if isinstance(s, str))
    max_ts: str | None = None
    max_ts_sha: str | None = None
    for row in _closing_rows(reg):
        sha = row.get("head_sha")
        ts = row.get("ts")
        at = row.get("agent_type")
        if not isinstance(sha, str) or not sha:
            failures.append(f"G8: closing row {at!r} carries no head_sha")
            continue
        p = _run_git(["merge-base", "--is-ancestor", sha, head_ref], repo_root)
        if p.returncode != 0:
            failures.append(f"G8: closing row {at!r} head_sha {sha!r} is not an ancestor of {head_ref}")
        if isinstance(ts, str) and (max_ts is None or ts > max_ts):
            max_ts, max_ts_sha = ts, sha
    if max_ts_sha:
        p = _run_git(["diff", "--name-only", f"{max_ts_sha}..{head_ref}"], repo_root)
        if p.returncode == 0:
            touched = {line for line in p.stdout.splitlines()
                       if line.strip() and not _is_assembly_artifact(line.strip())}
            intersect = touched & unit_surface
            if intersect:
                failures.append(
                    f"G8: commit(s) after the max-ts closing sha touch the shipped unit surface: "
                    f"{sorted(intersect)} -- the closing verdicts did not audit what ships"
                )
        else:
            failures.append(f"G8: git diff --name-only {max_ts_sha}..{head_ref} failed: {p.stderr.strip()}")
    if base_ref:
        p = _run_git(["diff", "--name-only", f"{base_ref}...{head_ref}"], repo_root)
        if p.returncode == 0:
            pr_diff = {line for line in p.stdout.splitlines()
                       if line.strip() and not _is_assembly_artifact(line.strip())}
            missing = pr_diff - unit_surface
            if missing:
                failures.append(
                    f"G8: unit_surface does not cover the PR's own diff paths: {sorted(missing)}"
                )
        else:
            failures.append(f"G8: git diff --name-only {base_ref}...{head_ref} failed: {p.stderr.strip()}")
    else:
        skips.append("G8: unit_surface-covers-PR-diff sub-check SKIPped (no base ref)")
    return failures, skips


def check_register(reg: dict, *, repo_root: Path, escapes_by_id: dict[str, dict],
                    head_ref: str = "HEAD", base_ref: str | None = None,
                    check_freshness: bool = True) -> tuple[list[str], list[str]]:
    """Runs G1-G8 (G8 optional) over one parsed register. Returns (failures, skips)."""
    failures: list[str] = []
    skips: list[str] = []
    failures += g1_parses(reg)
    if any(f.startswith("G1:") for f in failures) and not _closing_rows(reg):
        # closing_rows is unusable; the remaining gates would only add noise.
        return failures, skips
    failures += g2_ship_criterion(reg)
    failures += g3_coverage(reg)
    failures += g4_no_reclassification(reg)
    failures += g5_sweep_honesty(reg, repo_root)
    failures += g6_eval_added_not_open(reg, escapes_by_id)
    failures += g7_non_vacuity_per_register(reg)
    if check_freshness:
        g8_failures, g8_skips = g8_freshness(reg, repo_root=repo_root, head_ref=head_ref, base_ref=base_ref)
        failures += g8_failures
        skips += g8_skips
    return failures, skips


# --------------------------------------------------------------------------
# --local leg: byte-diff against real (gitignored) gate-state
# --------------------------------------------------------------------------

def _read_gate_state_rows(repo_root: Path, slug: str) -> list[dict]:
    path = repo_root / GATE_STATE_SUBDIR / f"{slug}.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def local_leg(reg: dict, *, repo_root: Path, unit_slug: str) -> list[str]:
    """Lead/operator-only authenticity check: byte-diffs embedded closing
    rows against the live gate-state rows of the same (agent_type, ts);
    verifies max-ts-per-lane; verifies earlier standing findings are carried
    forward into residuals or the lane's own closing enumeration."""
    failures: list[str] = []
    live_rows = _read_gate_state_rows(repo_root, unit_slug)
    by_type: dict[str, list[dict]] = {}
    for r in live_rows:
        if "verdict" in r and isinstance(r.get("agent_type"), str):
            by_type.setdefault(r["agent_type"], []).append(r)
    for lane_rows in by_type.values():
        lane_rows.sort(key=lambda r: r.get("ts") or "")

    embedded_by_type = {row.get("agent_type"): row for row in _closing_rows(reg)}
    for at, lane_rows in by_type.items():
        if not lane_rows:
            continue
        live_max = lane_rows[-1]
        embedded = embedded_by_type.get(at)
        if embedded is None:
            failures.append(f"--local: lane {at!r} produced bound rows but has no closing row in the register")
            continue
        if embedded.get("ts") != live_max.get("ts"):
            failures.append(
                f"--local: lane {at!r} closing row ts {embedded.get('ts')!r} is not the live max-ts "
                f"({live_max.get('ts')!r}) -- not the max-ts row of the lane"
            )
        for key in ("agent_type", "ts", "round", "verdict", "head_sha", "sweep_method",
                    "exhaustive", "class_enumeration", "finding_locations"):
            if embedded.get(key) != live_max.get(key):
                failures.append(
                    f"--local: lane {at!r} closing row field {key!r} does not byte-match the live "
                    f"gate-state row: embedded={embedded.get(key)!r} live={live_max.get(key)!r}"
                )
        residuals = reg.get("residuals")
        residuals = residuals if isinstance(residuals, dict) else {}
        for earlier in lane_rows[:-1]:
            for loc in (earlier.get("finding_locations") or []):
                covered = loc in (embedded.get("finding_locations") or []) or loc in residuals
                if not covered:
                    failures.append(
                        f"--local: lane {at!r} earlier row {earlier.get('ts')!r} finding {loc!r} is "
                        "neither re-enumerated by the closing row nor present in residuals"
                    )
    return failures


ADVISORY_REFUSALS = (
    "no fitted SRGM (stationarity violated by lens-changing audits, n~10; S1.3)",
    "no capture-recapture estimate (S1.4 instability; lenses complementary by design)",
)


def advisory_signals(repo_root: Path, unit_slug: str) -> dict:
    """INFO only, never a gate input."""
    live_rows = _read_gate_state_rows(repo_root, unit_slug)
    trajectory = []
    rounds_since_live_block = None
    for r in live_rows:
        if "verdict" not in r:
            continue
        liveness_summary = "unclassified"
        findings_index = r.get("findings_index")
        if isinstance(findings_index, list) and findings_index:
            if any(_is_live(e) for e in findings_index if isinstance(e, dict)):
                liveness_summary = "live"
            else:
                liveness_summary = "latent"
        trajectory.append({
            "round": r.get("round"), "agent_type": r.get("agent_type"), "verdict": r.get("verdict"),
            "finding_locations": len(r.get("finding_locations") or []),
            "liveness": liveness_summary,
        })
    return {
        "trajectory": trajectory,
        "rounds_since_last_live_block": rounds_since_live_block,
        "refusals": list(ADVISORY_REFUSALS),
    }


# --------------------------------------------------------------------------
# CI mode
# --------------------------------------------------------------------------

def ci_mode(repo_root: Path = REPO_ROOT) -> int:
    failures_total: list[str] = []
    base_ref = resolve_diff_base(repo_root)

    trig_status, trig_failures = trigger_leg(repo_root, base_ref)
    print(f"check-ship-register[trigger]: {trig_status}")
    failures_total += trig_failures

    escapes_by_id = read_escapes(repo_root)
    registers = find_registers(repo_root)
    diff_registers = set(registers_changed_in_diff(repo_root, base_ref)) if base_ref else set()
    # Every register in the diff MUST be checked (mandatory), plus every
    # register in the tree (non-vacuity: the scan itself is proven non-empty
    # below when any register exists).
    if not registers and diff_registers:
        registers = sorted(diff_registers)

    for path in registers:
        reg, err = load_register(path)
        rel = path.relative_to(repo_root)
        if reg is None:
            print(f"check-ship-register[{rel}]: RED — {err}")
            failures_total.append(f"{rel}: {err}")
            continue
        reg_failures, reg_skips = check_register(
            reg, repo_root=repo_root, escapes_by_id=escapes_by_id,
            base_ref=base_ref, check_freshness=True,
        )
        for s in reg_skips:
            print(f"check-ship-register[{rel}]: SKIP — {s}")
        if reg_failures:
            print(f"check-ship-register[{rel}]: RED")
            for f in reg_failures:
                print(f"  - {f}", file=sys.stderr)
            failures_total += [f"{rel}: {f}" for f in reg_failures]
        else:
            print(f"check-ship-register[{rel}]: PASS")

    if not registers:
        print("check-ship-register: no registers present in the tree — nothing to check.")

    if failures_total:
        print("check-ship-register: FAIL", file=sys.stderr)
        return 1
    print("check-ship-register: OK")
    return 0


# --------------------------------------------------------------------------
# --local mode
# --------------------------------------------------------------------------

def local_mode(repo_root: Path = REPO_ROOT) -> int:
    registers = find_registers(repo_root)
    if not registers:
        print("check-ship-register --local: no registers present — nothing to check.")
        return 0
    failures_total: list[str] = []
    for path in registers:
        reg, err = load_register(path)
        rel = path.relative_to(repo_root)
        if reg is None:
            print(f"--local[{rel}]: RED — {err}")
            failures_total.append(f"{rel}: {err}")
            continue
        unit_slug = reg.get("unit_branch") or path.stem.removesuffix(".register")
        import re
        slug = re.sub(r"[^A-Za-z0-9._-]", "_", str(unit_slug).strip()) or "UNBOUND"
        local_failures = local_leg(reg, repo_root=repo_root, unit_slug=slug)
        if local_failures:
            print(f"--local[{rel}]: RED")
            for f in local_failures:
                print(f"  - {f}", file=sys.stderr)
            failures_total += local_failures
        else:
            print(f"--local[{rel}]: PASS (embedded closing rows byte-match live gate-state)")
        signals = advisory_signals(repo_root, slug)
        print(f"--local[{rel}]: advisory signals: {json.dumps(signals, indent=2)}")
    if failures_total:
        return 1
    return 0


# ==========================================================================
# --self-test battery
# ==========================================================================

def _fresh_git_repo() -> Path:
    d = Path(tempfile.mkdtemp(prefix="ship-register-selftest-"))
    _run_git(["init", "-q"], d)
    _run_git(["config", "user.email", "t@t"], d)
    _run_git(["config", "user.name", "t"], d)
    (d / "Cargo.toml").write_text("[workspace]\n")
    (d / REGISTERS_SUBDIR).mkdir(parents=True, exist_ok=True)
    _run_git(["add", "-A"], d)
    _run_git(["commit", "-q", "-m", "init"], d)
    return d


def _commit_file(repo: Path, relpath: str, content: str, msg: str) -> str:
    p = repo / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    _run_git(["add", "-A"], repo)
    _run_git(["commit", "-q", "-m", msg], repo)
    return _run_git(["rev-parse", "HEAD"], repo).stdout.strip()


def _head(repo: Path) -> str:
    return _run_git(["rev-parse", "HEAD"], repo).stdout.strip()


def _write_escapes(repo: Path, rows: list[dict]) -> None:
    text = "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else "")
    (repo / ESCAPES_SUBPATH).parent.mkdir(parents=True, exist_ok=True)
    (repo / ESCAPES_SUBPATH).write_text(text)


def _base_closing_row(sha: str, *, agent_type: str = "adversarial-audit", ts: str = "2020-01-01T00:00:00Z",
                       findings_index: list[dict] | None = None, class_enumeration: list[str] | None = None,
                       finding_locations: list[str] | None = None, exhaustive: bool = True,
                       verdict: str = "BLOCK") -> dict:
    return {
        "agent_type": agent_type, "ts": ts, "round": 1, "verdict": verdict,
        "head_sha": sha, "sweep_method": "grep -n the pattern", "exhaustive": exhaustive,
        "class_enumeration": class_enumeration or [],
        "finding_locations": finding_locations or [],
        "findings_index": findings_index if findings_index is not None else [],
    }


def _assert(cond: bool, label: str, detail: str = "") -> None:
    if not cond:
        raise Failure(f"{label}: {detail}")


# --- Required-RED fixtures (the six escape-citing golden evals) ----------

def fixture_1_standing_live_block_no_override_reds() -> None:
    """T1 itself: a standing live block with no operator override must RED."""
    repo = _fresh_git_repo()
    sha = _head(repo)
    reg = {
        "kind": "assurance_deficit_register", "unit_branch": "u1", "session": "s",
        "closing_rows": [_base_closing_row(sha, findings_index=[
            {"location": "a.py:1", "severity": "block", "stands": True, "liveness": "live"},
        ], class_enumeration=["a.py:1"])],
        "unit_surface": [], "residuals": {}, "seeded_escapes": [],
        "reopen_budget": {"new_escapes_in_registered_class": 1},
    }
    failures, _ = check_register(reg, repo_root=repo, escapes_by_id={}, base_ref=None)
    _assert(any("G2" in f for f in failures), "fixture-1", f"expected G2 RED, got {failures}")


def fixture_2_residual_liveness_mismatch_reds_esc064() -> None:
    """esc-064: a residual's liveness must string-equal its findings_index
    entry -- a mismatch (silently reclassifying a live finding latent) REDs."""
    repo = _fresh_git_repo()
    sha = _head(repo)
    reg = {
        "kind": "assurance_deficit_register", "unit_branch": "u2", "session": "s",
        "closing_rows": [_base_closing_row(sha, findings_index=[
            {"location": "a.py:1", "severity": "block", "stands": True, "liveness": "live"},
        ], class_enumeration=["a.py:1"])],
        "unit_surface": [], "residuals": {
            "a.py:1": {"class": "x", "liveness": "latent", "operator_accepted_live": True,
                       "fixture_ref": "Cargo.toml", "justification": "j", "owner": "operator"},
        },
        "seeded_escapes": [],
        "reopen_budget": {"new_escapes_in_registered_class": 1},
    }
    failures, _ = check_register(reg, repo_root=repo, escapes_by_id={}, base_ref=None)
    _assert(any("G4" in f for f in failures), "fixture-2 (esc-064)", f"expected G4 RED, got {failures}")


def fixture_3_empty_scan_reporting_clean_reds_esc063() -> None:
    """esc-063: an empty/zero-row register must not report clean -- G1 REDs
    on closing_rows missing/empty, and the checker's own registers-directory
    scan is proven non-vacuous (never silently examines zero files and
    reports PASS)."""
    reg = {
        "kind": "assurance_deficit_register", "unit_branch": "u3", "session": "s",
        "closing_rows": [],
        "unit_surface": [], "residuals": {}, "seeded_escapes": [],
        "reopen_budget": {"new_escapes_in_registered_class": 1},
    }
    failures, _ = check_register(reg, repo_root=_fresh_git_repo(), escapes_by_id={}, base_ref=None)
    _assert(any("G1" in f for f in failures), "fixture-3 (esc-063)", f"expected G1 RED, got {failures}")

    # Non-vacuity control: with N real register files on disk, the scan finds
    # exactly N, never 0-while-files-exist (the esc-063 shape).
    repo = _fresh_git_repo()
    for i in range(3):
        (repo / REGISTERS_SUBDIR / f"u{i}.register.json").write_text("{}")
    found = find_registers(repo)
    _assert(len(found) == 3, "fixture-3 (esc-063) non-vacuity",
            f"expected the scan to find all 3 register files, found {len(found)}: {found}")


def fixture_4_residual_omits_enumerated_string_reds_esc066() -> None:
    """esc-066: residuals must be an exact-string superset of every
    class_enumeration string; an enumerated site with no residual key REDs
    even when no findings_index entry names it directly."""
    repo = _fresh_git_repo()
    sha = _head(repo)
    reg = {
        "kind": "assurance_deficit_register", "unit_branch": "u4", "session": "s",
        "closing_rows": [_base_closing_row(sha, class_enumeration=["a.py:1", "b.py:2"],
                                            findings_index=[])],
        "unit_surface": [], "residuals": {"a.py:1": {
            "class": "x", "liveness": "latent", "operator_accepted_live": False,
            "fixture_ref": "Cargo.toml", "justification": "j", "owner": "lead",
        }},  # b.py:2 omitted
        "seeded_escapes": [],
        "reopen_budget": {"new_escapes_in_registered_class": 1},
    }
    failures, _ = check_register(reg, repo_root=repo, escapes_by_id={}, base_ref=None)
    _assert(any("G3" in f for f in failures), "fixture-4 (esc-066)", f"expected G3 RED, got {failures}")


def fixture_5_seeded_escape_open_reds_unit62() -> None:
    """unit-62 precedent (rows 20/57/70/93): a seeded_escapes id at status
    'open' must RED -- eval_added-not-open."""
    repo = _fresh_git_repo()
    sha = _head(repo)
    reg = {
        "kind": "assurance_deficit_register", "unit_branch": "u5", "session": "s",
        "closing_rows": [_base_closing_row(sha, verdict="PASS")],
        "unit_surface": [], "residuals": {}, "seeded_escapes": ["esc-999-x"],
        "reopen_budget": {"new_escapes_in_registered_class": 1},
    }
    escapes_by_id = {"esc-999-x": {"id": "esc-999-x", "status": "open"}}
    failures, _ = check_register(reg, repo_root=repo, escapes_by_id=escapes_by_id, base_ref=None)
    _assert(any("G6" in f for f in failures), "fixture-5 (unit-62)", f"expected G6 RED, got {failures}")


def fixture_6_status_moved_uncovered_reds_esc066_trigger() -> None:
    """esc-066 / Q3 closure: an escape status-move in the diff with no
    covering register REDs."""
    repo = _fresh_git_repo()
    _write_escapes(repo, [{"id": "esc-777-y", "status": "open"}])
    base_sha = _commit_file(repo, str(ESCAPES_SUBPATH),
                             (repo / ESCAPES_SUBPATH).read_text(), "seed escapes")
    _write_escapes(repo, [{"id": "esc-777-y", "status": "eval_added"}])
    _run_git(["add", "-A"], repo)
    _run_git(["commit", "-q", "-m", "move status, no register"], repo)
    status, failures = trigger_leg(repo, base_sha)
    _assert(status == "RED", "fixture-6 (esc-066 trigger)", f"expected RED, got {status}: {failures}")


# --- Additional required-RED fixtures -------------------------------------

def fixture_unparseable_closing_row_reds() -> None:
    repo = _fresh_git_repo()
    sha = _head(repo)
    row = _base_closing_row(sha)
    row["unparseable_reason"] = "template"
    reg = {"kind": "assurance_deficit_register", "unit_branch": "u7", "session": "s",
           "closing_rows": [row], "unit_surface": [], "residuals": {}, "seeded_escapes": [],
           "reopen_budget": {"new_escapes_in_registered_class": 1}}
    failures, _ = check_register(reg, repo_root=repo, escapes_by_id={}, base_ref=None)
    _assert(any("G1" in f and "unparseable_reason" in f for f in failures), "unparseable-closing-row",
            f"expected G1 RED naming unparseable_reason, got {failures}")


def fixture_finding_locations_nonempty_empty_index_reds() -> None:
    repo = _fresh_git_repo()
    sha = _head(repo)
    row = _base_closing_row(sha, finding_locations=["a.py:1"], findings_index=[])
    reg = {"kind": "assurance_deficit_register", "unit_branch": "u8", "session": "s",
           "closing_rows": [row], "unit_surface": [], "residuals": {}, "seeded_escapes": [],
           "reopen_budget": {"new_escapes_in_registered_class": 1}}
    failures, _ = check_register(reg, repo_root=repo, escapes_by_id={}, base_ref=None)
    _assert(any("G2" in f and "harvest-failure" in f for f in failures), "finding-locations-nonempty-empty-index",
            f"expected G2 RED, got {failures}")


def fixture_exhaustive_false_without_residual_reds() -> None:
    repo = _fresh_git_repo()
    sha = _head(repo)
    row = _base_closing_row(sha, exhaustive=False, verdict="PASS")
    reg = {"kind": "assurance_deficit_register", "unit_branch": "u9", "session": "s",
           "closing_rows": [row], "unit_surface": [], "residuals": {}, "seeded_escapes": [],
           "reopen_budget": {"new_escapes_in_registered_class": 1}}
    failures, _ = check_register(reg, repo_root=repo, escapes_by_id={}, base_ref=None)
    _assert(any("G5" in f and "nonexhaustive-sweep" in f for f in failures), "exhaustive-false-no-residual",
            f"expected G5 RED, got {failures}")


def fixture_non_ancestor_closing_sha_reds() -> None:
    repo = _fresh_git_repo()
    fake_sha = "0" * 40
    row = _base_closing_row(fake_sha, verdict="PASS")
    reg = {"kind": "assurance_deficit_register", "unit_branch": "u10", "session": "s",
           "closing_rows": [row], "unit_surface": [], "residuals": {}, "seeded_escapes": [],
           "reopen_budget": {"new_escapes_in_registered_class": 1}}
    failures, _ = check_register(reg, repo_root=repo, escapes_by_id={}, base_ref=None)
    _assert(any("G8" in f and "not an ancestor" in f for f in failures), "non-ancestor-closing-sha",
            f"expected G8 RED, got {failures}")


def fixture_intervening_commit_on_surface_reds() -> None:
    repo = _fresh_git_repo()
    sha = _commit_file(repo, "src/a.rs", "fn a() {}\n", "closing state")
    _commit_file(repo, "src/a.rs", "fn a() { /* changed after closing */ }\n", "intervening commit")
    row = _base_closing_row(sha, verdict="PASS")
    reg = {"kind": "assurance_deficit_register", "unit_branch": "u11", "session": "s",
           "closing_rows": [row], "unit_surface": ["src/a.rs"], "residuals": {}, "seeded_escapes": [],
           "reopen_budget": {"new_escapes_in_registered_class": 1}}
    failures, _ = check_register(reg, repo_root=repo, escapes_by_id={}, base_ref=None)
    _assert(any("G8" in f and "unit surface" in f for f in failures), "intervening-commit-on-surface",
            f"expected G8 RED, got {failures}")


# --- Must-still-count GREEN / SKIP arms ------------------------------------

def fixture_conformant_latent_only_passes() -> None:
    repo = _fresh_git_repo()
    sha = _head(repo)
    reg = {
        "kind": "assurance_deficit_register", "unit_branch": "u12", "session": "s",
        "closing_rows": [_base_closing_row(sha, verdict="PASS", class_enumeration=["a.py:1"],
                                            findings_index=[
                                                {"location": "a.py:1", "severity": "advisory",
                                                 "stands": True, "liveness": "latent"},
                                            ])],
        "unit_surface": [], "residuals": {"a.py:1": {
            "class": "x", "liveness": "latent", "operator_accepted_live": False,
            "fixture_ref": "Cargo.toml", "justification": "acceptable, latent-only", "owner": "lead",
        }}, "seeded_escapes": [],
        "reopen_budget": {"new_escapes_in_registered_class": 1},
    }
    failures, _ = check_register(reg, repo_root=repo, escapes_by_id={}, base_ref=None)
    _assert(not failures, "conformant-latent-only", f"expected PASS (no failures), got {failures}")


def fixture_operator_accepted_live_passes() -> None:
    repo = _fresh_git_repo()
    sha = _head(repo)
    reg = {
        "kind": "assurance_deficit_register", "unit_branch": "u13", "session": "s",
        "closing_rows": [_base_closing_row(sha, class_enumeration=["a.py:1"], findings_index=[
            {"location": "a.py:1", "severity": "block", "stands": True, "liveness": "live"},
        ])],
        "unit_surface": [], "residuals": {"a.py:1": {
            "class": "x", "liveness": "live", "operator_accepted_live": True,
            "fixture_ref": "Cargo.toml", "justification": "operator accepted at admin-merge", "owner": "operator",
        }}, "seeded_escapes": [],
        "reopen_budget": {"new_escapes_in_registered_class": 1},
    }
    failures, _ = check_register(reg, repo_root=repo, escapes_by_id={}, base_ref=None)
    _assert(not failures, "operator-accepted-live", f"expected PASS (no failures), got {failures}")


def fixture_no_base_ref_trigger_skips() -> None:
    repo = _fresh_git_repo()
    status, failures = trigger_leg(repo, None)
    _assert(status == "SKIP", "no-base-ref-trigger", f"expected SKIP, got {status}: {failures}")
    _assert(not failures, "no-base-ref-trigger", f"SKIP must carry no failures, got {failures}")


FIXTURES = [
    ("golden-1-standing-live-block", fixture_1_standing_live_block_no_override_reds),
    ("golden-2-esc064-residual-mismatch", fixture_2_residual_liveness_mismatch_reds_esc064),
    ("golden-3-esc063-empty-scan", fixture_3_empty_scan_reporting_clean_reds_esc063),
    ("golden-4-esc066-residual-coverage", fixture_4_residual_omits_enumerated_string_reds_esc066),
    ("golden-5-unit62-seeded-open", fixture_5_seeded_escape_open_reds_unit62),
    ("golden-6-esc066-trigger", fixture_6_status_moved_uncovered_reds_esc066_trigger),
    ("unparseable-closing-row", fixture_unparseable_closing_row_reds),
    ("finding-locations-empty-index", fixture_finding_locations_nonempty_empty_index_reds),
    ("exhaustive-false-no-residual", fixture_exhaustive_false_without_residual_reds),
    ("non-ancestor-closing-sha", fixture_non_ancestor_closing_sha_reds),
    ("intervening-commit-on-surface", fixture_intervening_commit_on_surface_reds),
    ("green-conformant-latent-only", fixture_conformant_latent_only_passes),
    ("green-operator-accepted-live", fixture_operator_accepted_live_passes),
    ("skip-no-base-ref-trigger", fixture_no_base_ref_trigger_skips),
]


def self_test() -> int:
    failures: list[str] = []
    for name, fn in FIXTURES:
        try:
            fn()
            print(f"check-ship-register[{name}]: OK")
        except Failure as e:
            failures.append(f"{name}: {e}")
            print(f"check-ship-register[{name}]: FAIL — {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: unexpected exception: {e!r}")
            print(f"check-ship-register[{name}]: FAIL (unexpected exception) — {e!r}", file=sys.stderr)

    if failures:
        print("check-ship-register --self-test: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"check-ship-register --self-test: all {len(FIXTURES)} fixture(s) passed.")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "--self-test" in argv:
        return self_test()
    if "--local" in argv:
        return local_mode()
    return ci_mode()


if __name__ == "__main__":
    sys.exit(main())
