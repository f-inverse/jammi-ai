#!/usr/bin/env python3
"""replay_relays.py — replays every recorded relay artifact under a
`.jammi/gate-state/` directory against `lead-gate-lib.py`'s own
`_relay_rejection` (R1 coverage, R2 proactivity, R3 probe-the-fix — whatever
rule set the `--lib` argument's own file on disk implements), printing a
table of `(unit, agent_type, verdict, block_ts, has_relay, decision,
reason)`.

Built for `docs/plans/53-agentic-swarm/proposals/R4-verifiable-sweep.md`
(the lead-gate-R4 proposal): the proposal claims R4a/R4b/R4c would have
caught three real session incidents; this script is how that claim is
checked against real behaviour instead of narrated. It is
mechanism-agnostic — pointed at the CURRENT (R1-R3) `lead-gate-lib.py` it
gives the BASELINE table (what the hook decides today); pointed at a
THROWAWAY copy with the R4 patch applied, it gives the table the proposal's
own replay evidence section cites.

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--project-dir", required=True,
                     help="CLAUDE_PROJECT_DIR for the lib's own R3 git arm (read-only calls only)")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--only-open", action="store_true",
                     help="only replay rows whose verdict is BLOCK/UNPARSEABLE (is_open)")
    args = ap.parse_args()

    rows = replay(Path(args.state_dir), Path(args.project_dir), Path(args.lib), args.only_open)
    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
