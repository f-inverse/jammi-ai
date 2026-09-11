#!/usr/bin/env python3
"""replay_relays.py — replays every recorded relay artifact under a
`.jammi/gate-state/` directory against `lead-gate-lib.py`'s own
`_relay_rejection` (R1 coverage, R2 proactivity, R3 probe-the-fix — whatever
rule set the `--lib` argument's own file on disk implements), printing a
table of `(unit, agent_type, verdict, block_ts, has_relay, decision,
reason)`.

Originally built for the lead-gate-R4 proposal (R4a/R4b/R4c — KILLED after
two design rounds; see `docs/plans/53-agentic-swarm/proposals/
R7-committed-rigor-record.md` for the full history). It is
mechanism-agnostic — pointed at the CURRENT (R1-R3) `lead-gate-lib.py` it
gives the BASELINE table (what the hook decides today); pointed at a
THROWAWAY copy with a patch applied, it gives the table a proposal's own
replay-evidence section can cite. `--rigor-record-ranges` (below) is the
esc-lead-gate-R7 extension: R7's own check is armed by DIFF SHAPE, not a
round count, so what needs replaying is "which real historical PRs would
have armed" against `check_rigor_record.py`'s own arming logic, over real
merge ranges — never a counter, which R4/R5/R6 all proved does not exist
in a form this hook's rows can support.

Read-only: never writes to `--state-dir`. The only git calls made are the
ones the loaded lib's own R3 arm already makes (`rev-parse --verify`,
`merge-base --is-ancestor`, `diff --name-only`) — all read-only plumbing,
no commit, no checkout, no ref write. `--project-dir` is threaded through
as `CLAUDE_PROJECT_DIR` exactly as the real hook expects it.

This is an operator/audit tool, not a CI check: `.jammi/gate-state/` is
gitignored, session-local state that is never present in a hermetic CI
checkout, so this script is never wired into `ci.yml`.

Usage:
  python3 replay_relays.py --state-dir <dir> --project-dir <dir> \\
      --lib <path/to/lead-gate-lib.py> [--only-open]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _load_lib(path: Path):
    spec = importlib.util.spec_from_file_location("lead_gate_lib_replay", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def replay(state_dir: Path, project_dir: Path, lib_path: Path, only_open: bool = False) -> list[tuple]:
    os.environ["CLAUDE_PROJECT_DIR"] = str(project_dir)
    mod = _load_lib(lib_path)

    out: list[tuple] = []
    for unit_file in sorted(state_dir.glob("*.jsonl")):
        if unit_file.name in ("bindings.jsonl",) or ".relay." in unit_file.name:
            continue
        unit_slug = unit_file.stem
        for row in mod.read_rows(unit_file):
            if "verdict" not in row:
                continue
            agent_type = row.get("agent_type") or ""
            block_ts = row.get("ts") or ""
            if not agent_type or not block_ts:
                continue
            verdict = row.get("verdict")
            if only_open and not mod.is_open(verdict):
                continue
            relay_path = mod.relay_artifact_path(state_dir, unit_slug, agent_type, block_ts)
            has_relay = relay_path.exists()
            reason = mod._relay_rejection(state_dir, unit_slug, row)
            decision = "ALLOW" if reason is None else "DENY"
            out.append((unit_slug, agent_type, verdict, block_ts, has_relay, decision, reason or ""))
    return out


def print_table(rows: list[tuple]) -> None:
    headers = ["unit", "agent_type", "verdict", "block_ts", "has_relay", "decision", "reason"]
    all_rows = rows + [tuple(headers)]
    widths = [min(max(len(str(r[i])) for r in all_rows), 40) for i in range(7)] if rows else [10] * 7
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        cells = [str(c) for c in r]
        cells[-1] = cells[-1][:200]
        print(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))
    print(f"\n{len(rows)} row(s) replayed; "
          f"{sum(1 for r in rows if r[5] == 'DENY')} DENY, "
          f"{sum(1 for r in rows if r[5] == 'ALLOW')} ALLOW.")


# --------------------------------------------------------------------------
# esc-lead-gate-R7 (v2): replay `check_rigor_record.py`'s own arming/no-op
# logic against REAL historical merge ranges — the counter this harness was
# originally built for (R4/R5/R6) is dead; R7's check is diff-shape-armed,
# so what it needs replayed is "which real PRs would have armed", never a
# round count. Read-only: every git call is `log`/`diff --name-only`
# plumbing against already-merged history, never a fetch, never a write.
# --------------------------------------------------------------------------

def _load_rigor_module(path: Path):
    spec = importlib.util.spec_from_file_location("check_rigor_record_replay", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def replay_rigor_record_ranges(project_dir: Path, rigor_script: Path, limit: int) -> list[tuple]:
    """For each of the last `limit` merge commits into the branch checked
    out at `project_dir` (each `parent1...parent2` IS a real, historical
    `base...head` PR range — no fetch needed, the merge commit already
    carries both parents), reports `(merge_sha7, subject, armed?,
    no_op_reason_or_none, n_armed_paths, has_contract_in_range,
    human_authored?)` using `check_rigor_record.py`'s OWN functions
    (`is_human_authored`, `is_revert`, `is_release_shaped`,
    `_matches_arming_glob`, `_matches_contract_glob`) — never a
    reimplementation. `GITHUB_ACTOR`/`GITHUB_EVENT_NAME`-gated no-op arms
    (dependabot, non-PR events) are not evaluated here — those never
    applied to historical git history in the first place; only the
    structurally-derivable no-op shapes (human-authored, revert,
    release-shaped) and the arming glob are replayed."""
    mod = _load_rigor_module(rigor_script)

    def _git(*args: str) -> tuple[bool, str]:
        return mod._git(project_dir, *args)

    ok, out = _git("log", "--merges", f"-{limit}", "--format=%H %P")
    if not ok:
        return []
    out_rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue  # not a two-parent merge
        merge_sha, p1, p2 = parts[0], parts[1], parts[2]
        ok, subject = _git("log", "-1", "--format=%s", merge_sha)
        subject = subject if ok else ""
        range_spec = f"{p1}...{p2}"
        ok, changed_out = _git("diff", "--name-only", range_spec)
        changed = [p for p in changed_out.splitlines() if p.strip()] if ok else []
        armed_paths = [p for p in changed if mod._matches_arming_glob(p)]
        human = mod.is_human_authored(project_dir, range_spec)
        revert = subject.startswith('Revert "')
        release = mod.is_release_shaped(changed)
        no_op = None
        if human:
            no_op = "human-authored (no Co-Authored-By: Claude commit in range)"
        elif revert:
            no_op = "revert"
        elif release:
            no_op = "release-shaped"
        elif not armed_paths:
            no_op = "not armed (no crates/**, ci/**, .github/workflows/** touch)"
        contract_paths = [p for p in changed if mod._matches_contract_glob(p)]
        out_rows.append((merge_sha[:7], subject[:60], no_op is None, no_op or "",
                          len(armed_paths), bool(contract_paths), human))
    return out_rows


def print_rigor_table(rows: list[tuple]) -> None:
    headers = ["merge", "subject", "armed", "no_op_reason", "n_armed_paths", "has_contract_in_range", "human_authored"]
    all_rows = rows + [tuple(headers)]
    widths = [min(max(len(str(r[i])) for r in all_rows), 60) for i in range(7)] if rows else [10] * 7
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        cells = [str(c) for c in r]
        print(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))
    armed = [r for r in rows if r[2]]
    print(f"\n{len(rows)} merge(s) replayed; {len(armed)} would ARM under R7b; "
          f"{sum(1 for r in armed if not r[5])} of those carry NO contract file in their own range "
          "(expected — no unit has authored one under this scheme yet).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir")
    ap.add_argument("--project-dir", required=True,
                     help="CLAUDE_PROJECT_DIR for the lib's own R3 git arm (read-only calls only)")
    ap.add_argument("--lib")
    ap.add_argument("--only-open", action="store_true",
                     help="only replay rows whose verdict is BLOCK/UNPARSEABLE (is_open)")
    ap.add_argument("--rigor-record-ranges", metavar="RIGOR_SCRIPT",
                     help="replay check_rigor_record.py's arming logic against the last "
                          "--limit real merge ranges instead of the relay table")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    if args.rigor_record_ranges:
        rows = replay_rigor_record_ranges(Path(args.project_dir), Path(args.rigor_record_ranges), args.limit)
        print_rigor_table(rows)
        return 0

    if not args.state_dir or not args.lib:
        sys.stderr.write("replay_relays.py: --state-dir and --lib are required unless "
                          "--rigor-record-ranges is given\n")
        return 2
    rows = replay(Path(args.state_dir), Path(args.project_dir), Path(args.lib), args.only_open)
    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
