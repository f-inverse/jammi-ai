#!/usr/bin/env python3
"""esc-lead-gate-R12 fix round 2 item 11: derives the set of unit slugs
that currently carry an OPEN `VERIFIER_SECOND_ROUND_TYPES` BLOCK, from the
two sources that can ever carry one:

  - the committed, cross-branch `docs/rigor/*.jsonl` export corpus (the
    LATEST row, by `ts`, per (unit_branch, agent_type) determines whether
    that (unit, type) is currently open — the SAME `is_open()` predicate
    the hook itself uses on its own live ledger);
  - the live, gitignored `.jammi/gate-state/<slug>.jsonl` ledger (present
    only in a worktree that still carries open state locally).

Prints one unit slug per line, sorted — this is exactly the content
`ci/scripts/rigor_record_r12_grandfather.txt`'s BODY should carry after a
run at MERGE TIME on a checkout with full visibility into the swarm's
current state; this script's own stdout belongs in that commit's message
verbatim, never a hand-typed guess.

This script NEVER commits and, by default, never writes the grandfather
file either (the ratchet in `check_rigor_record.py`'s
`check_r12_grandfather_only_shrinks` is deliberately the only enforcement
point on `main`; this script only tells a human what the ledger currently
says). `--write` overwrites the file's BODY (every line after its leading
`#`-comment/blank header block) with the derived list — still requires a
human to review the diff and commit it. Per-slug reason comments
interleaved with the OLD body are NOT preserved across `--write`; if any
existed, re-add them by hand after inspecting the diff.

Run: `python3 ci/scripts/derive_r12_grandfather.py [--write]`
Self-test: `python3 ci/scripts/derive_r12_grandfather.py --self-test`

HONEST LIMIT: this script's OWN view is only as complete as the checkout
it runs in. `docs/rigor/*.jsonl` is authoritative once merged to `main`;
`.jammi/gate-state/` is per-worktree and gitignored, so a slug with open
state ONLY in a different, unmerged worktree is invisible here. Run this
at merge time on an up-to-date `main` checkout, not from an isolated
feature worktree.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEAD_GATE_LIB = REPO_ROOT / ".claude" / "hooks" / "lead-gate-lib.py"
GRANDFATHER_PATH = REPO_ROOT / "ci" / "scripts" / "rigor_record_r12_grandfather.txt"


def _lib():
    spec = importlib.util.spec_from_file_location("lead_gate_lib_derive", LEAD_GATE_LIB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _from_gate_state(mod, repo_root: Path) -> set[str]:
    sdir = repo_root / ".jammi" / "gate-state"
    slugs: set[str] = set()
    for unit_slug, atype, row, _idx in mod.all_open_blocks(sdir):
        if atype in mod.VERIFIER_SECOND_ROUND_TYPES:
            slugs.add(unit_slug)
    return slugs


def _from_rigor_docs(mod, repo_root: Path) -> set[str]:
    rigor_dir = repo_root / "docs" / "rigor"
    latest: dict[tuple[str, str], dict] = {}
    if rigor_dir.exists():
        for p in sorted(rigor_dir.glob("*.jsonl")):
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                atype = row.get("agent_type")
                unit_branch = row.get("unit_branch")
                ts = row.get("ts")
                if not (isinstance(atype, str) and isinstance(unit_branch, str) and isinstance(ts, str)):
                    continue
                key = (unit_branch, atype)
                if key not in latest or ts > latest[key].get("ts", ""):
                    latest[key] = row
    slugs: set[str] = set()
    for (unit_branch, atype), row in latest.items():
        if atype not in mod.VERIFIER_SECOND_ROUND_TYPES:
            continue
        # `docs/rigor/*.jsonl` rows carry the ALREADY-NORMALIZED `verdict`
        # field (PASS/BLOCK/UNPARSEABLE) as exported — re-running
        # `normalize_verdict()` on it would treat this pre-normalized
        # string as agent-authored RAW prose and re-match it against
        # `_pass_word_for(atype)` a second time, which is wrong whenever an
        # agent type's own pass word is not the literal string "PASS"
        # (e.g. acceptance-verifier's own pass word is not "PASS", so a
        # row already normalized to "PASS" would be double-normalized back
        # to "BLOCK"). Read `verdict` directly instead.
        verdict = row.get("verdict")
        if isinstance(verdict, str) and mod.is_open(verdict):
            slugs.add(mod.slugify(unit_branch))
    return slugs


def derive(repo_root: Path = REPO_ROOT) -> list[str]:
    mod = _lib()
    return sorted(_from_gate_state(mod, repo_root) | _from_rigor_docs(mod, repo_root))


def _self_test() -> int:
    """Two regression cases in a synthetic repo root (never the real
    `docs/rigor/`): (1) a unit whose LATEST second-round row is a real
    open BLOCK is derived; (2) a unit whose exported `verdict` is already
    normalized to "PASS" is NOT re-flagged as open — the double-
    normalization bug this script's own `_from_rigor_docs` fixed
    (`normalize_verdict()` re-matching an ALREADY-normalized "PASS"
    string against an agent type whose own pass word is not literally
    "PASS", e.g. acceptance-verifier, would wrongly flip it back to
    BLOCK)."""
    import tempfile
    failures = []
    with tempfile.TemporaryDirectory(prefix="r12-grandfather-selftest-") as d:
        root = Path(d)
        rigor = root / "docs" / "rigor"
        rigor.mkdir(parents=True)
        (rigor / "feat_open.jsonl").write_text(
            json.dumps({"agent_type": "adversarial-audit", "unit_branch": "feat/open",
                        "verdict": "BLOCK", "ts": "2026-01-01T00:00:00+00:00"}) + "\n"
        )
        (rigor / "feat_closed.jsonl").write_text(
            "\n".join([
                json.dumps({"agent_type": "acceptance-verifier", "unit_branch": "feat/closed",
                            "verdict": "BLOCK", "ts": "2026-01-01T00:00:00+00:00"}),
                json.dumps({"agent_type": "acceptance-verifier", "unit_branch": "feat/closed",
                            "verdict": "PASS", "ts": "2026-01-02T00:00:00+00:00"}),
            ]) + "\n"
        )
        got = derive(root)
        if got != ["feat_open"]:
            failures.append(f"expected exactly ['feat_open'], got {got!r}")
    if failures:
        for f in failures:
            print(f"derive_r12_grandfather[self-test]: FAIL — {f}", file=sys.stderr)
        return 1
    print("derive_r12_grandfather[self-test]: OK (2/2 cases)")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    slugs = derive()
    for s in slugs:
        print(s)
    if "--write" in sys.argv[1:]:
        header_lines: list[str] = []
        if GRANDFATHER_PATH.exists():
            for line in GRANDFATHER_PATH.read_text().splitlines():
                if line.strip().startswith("#") or not line.strip():
                    header_lines.append(line)
                else:
                    break
        body_lines = header_lines + slugs
        GRANDFATHER_PATH.write_text("\n".join(body_lines) + ("\n" if body_lines else ""))
        print(f"derive_r12_grandfather: wrote {len(slugs)} slug(s) to {GRANDFATHER_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
