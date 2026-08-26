#!/usr/bin/env python3
"""Two mechanical, grep-shaped static assertions over every tracked producer
script under `ci/scripts/perf/` — the class two round-N audit findings on
`perf/unification-p2` named: a dry-run-only test knob left live in a REAL
run is a bypass, not a fixture, and a cross-check added to two of four
producers that share the exact same hole is a partial fix, not a closed
class.

## (A) FAKE-knob inertness

Any tracked `.sh` under `ci/scripts/` that references an environment
variable whose name contains `FAKE` (the shape `stacked_sweep.sh`'s
`SWEEP_FAKE_BIN_SHA` set, contract C5.2 — a test-only injection knob for
exercising the provenance-mismatch refusal path without a GPU or a real
binary) must ALSO contain an explicit REFUSAL guard: a line that tests the
knob is set (`-n "${<VAR>...}"`), tests some `*DRY_RUN*` variable `!= "1"`,
and `exit`s — appearing BEFORE every other textual use of that variable in
the file. A knob referenced with no such guard, or referenced (even in a
comment) before its own guard, cannot be trusted to be inert in a real run.

## (B) Producer parity — every jammi-bench-invoking producer carries the
`$BIN provenance` cross-check

Unification contract C5.1: "Every shell/Python producer cross-checks `$BIN
provenance`'s build_sha against the sha it is about to stamp before writing
a GREEN leg." Mechanically: every tracked `.sh` under `ci/scripts/perf/`
whose text names a `jammi-bench` BINARY PATH (a variable assignment ending
`/jammi-bench` — the shape every producer's own `$BIN`/`$B`/`$JAMMI_BIN`
takes; never a source-tree reference like `crates/jammi-bench/...`) must
also contain BOTH `provenance` and `build_sha` somewhere in its text — the
two tokens the cross-check itself is built from
(`"$BIN" provenance` / `["build_sha"]`). This is the exact `grep -l
jammi-bench` / `grep -l provenance` methodology the audit that found the
gap used — reproduced here as a standing gate, not a one-time hand check,
so a FIFTH producer landing tomorrow cannot silently reopen the class.

Both are deliberately mechanical (name/pattern presence), not a semantic
understanding of the guard's control flow — the same "grep for the shape,
not the meaning" stance `check_ci_guard_wiring.py`'s own module doc states.

Run: `python3 ci/scripts/perf/check_producer_provenance_gates.py`
Self-test (RED cases for both (A) and (B), on throwaway fixture files):
`python3 ci/scripts/perf/check_producer_provenance_gates.py --self-test`
Hermetic: reads tracked files via `git ls-files` only (no network, no
build, no GPU).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PERF_DIR = REPO_ROOT / "ci" / "scripts" / "perf"

FAKE_VAR_RE = re.compile(r"\b([A-Z][A-Z0-9_]*FAKE[A-Z0-9_]*)\b")
DRY_RUN_VAR_RE = re.compile(r"[A-Z][A-Z0-9_]*DRY_RUN")
# `/jammi-bench` NOT immediately followed by another `/` — the BINARY path
# shape (`.../release/jammi-bench"`, `.../jammi-bench` end-of-token), never
# the CRATE source-tree shape (`crates/jammi-bench/reference/...`, which
# also contains the bare substring `/jammi-bench` but is followed by `/`).
BIN_ASSIGN_RE = re.compile(r"/jammi-bench(?!/)")


def _tracked_sh_under(repo_root: Path, prefix: str) -> list[Path]:
    proc = subprocess.run(["git", "ls-files", prefix], cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"`git ls-files {prefix}` failed: {proc.stderr.strip()}")
    return sorted(
        repo_root / rel
        for rel in proc.stdout.splitlines()
        if rel.endswith(".sh")
    )


def _is_comment_line(line: str) -> bool:
    return line.strip().startswith("#")


def check_fake_knob_inertness(path: Path) -> list[str]:
    """(A) — see module doc. Operates on the file's CODE lines only (comment
    lines are skipped when looking for "uses", so a knob merely NAMED in a
    module-doc comment above its real guard is not mistaken for an
    unguarded use — the same comment-vs-code distinction
    `check_ci_guard_wiring.py`'s `workflow_run_text()` already draws)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    fake_vars = sorted(set(FAKE_VAR_RE.findall(text)))
    findings: list[str] = []
    for var in fake_vars:
        code_use_idx = [i for i, line in enumerate(lines) if not _is_comment_line(line) and var in line]
        if not code_use_idx:
            continue  # only ever named in comments/docs — nothing live to guard
        guard_idx = [
            i
            for i in code_use_idx
            if DRY_RUN_VAR_RE.search(lines[i]) and "!=" in lines[i] and '"1"' in lines[i]
        ]
        if not guard_idx:
            findings.append(
                f"{path}: `{var}` is referenced in code but no refusal guard (a line combining "
                f"`{var}`, a *DRY_RUN* variable, and `!= \"1\"`) was found — inert-unless-dry-run "
                "is not provable"
            )
            continue
        first_guard = min(guard_idx)
        guard_window = lines[first_guard] + (lines[first_guard + 1] if first_guard + 1 < len(lines) else "")
        if "exit" not in guard_window:
            findings.append(
                f"{path}:{first_guard + 1}: `{var}`'s guard line does not `exit` — a guard that "
                "does not refuse is not a refusal"
            )
        earlier = [i for i in code_use_idx if i < first_guard]
        if earlier:
            findings.append(
                f"{path}: `{var}` used at line(s) {[i + 1 for i in earlier]} BEFORE its refusal "
                f"guard at line {first_guard + 1} — a use preceding the guard cannot be covered by it"
            )
    return findings


def check_producer_parity(path: Path) -> list[str]:
    """(B) — see module doc. Only applies to scripts that actually name a
    jammi-bench BINARY PATH (`.../jammi-bench`, never the source-tree
    `crates/jammi-bench/...`)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if not BIN_ASSIGN_RE.search(text):
        return []
    missing = [tok for tok in ("provenance", "build_sha") if tok not in text]
    if missing:
        return [
            f"{path}: names a jammi-bench binary path but is missing {missing} — every producer "
            "that runs a jammi-bench binary must cross-check `$BIN provenance`'s build_sha before "
            "writing a GREEN leg (unification contract C5.1)"
        ]
    return []


def run_gate(perf_dir: Path, repo_root: Path) -> list[str]:
    findings: list[str] = []
    for path in _tracked_sh_under(repo_root, "ci/scripts/"):
        findings += check_fake_knob_inertness(path)
    for path in _tracked_sh_under(repo_root, "ci/scripts/perf/"):
        findings += check_producer_parity(path)
    return findings


def self_test() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
        perf = repo / "ci" / "scripts" / "perf"
        perf.mkdir(parents=True)

        def commit_and_check(rel: str, text: str, check_fn, expect_hit: str | None):
            p = repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", rel], cwd=repo, check=True)
            got = check_fn(p)
            if expect_hit is None:
                if got:
                    failures.append(f"self-test FAILED: {rel} expected clean, got {got}")
            elif not any(expect_hit in g for g in got):
                failures.append(f"self-test FAILED: {rel} expected a finding containing {expect_hit!r}, got {got}")

        # (A) RED: a knob referenced with no guard at all.
        commit_and_check(
            "ci/scripts/perf/bad_no_guard.sh",
            '#!/usr/bin/env bash\nif [ -n "${SWEEP_FAKE_BIN_SHA:-}" ]; then BIN_PROV_SHA="$SWEEP_FAKE_BIN_SHA"; fi\n',
            check_fake_knob_inertness,
            "no refusal guard",
        )

        # (A) RED: a knob used BEFORE its own guard.
        commit_and_check(
            "ci/scripts/perf/bad_use_before_guard.sh",
            (
                '#!/usr/bin/env bash\n'
                'BIN_PROV_SHA="$SWEEP_FAKE_BIN_SHA"\n'
                'if [ -n "${SWEEP_FAKE_BIN_SHA:-}" ] && [ "$SWEEP_DRY_RUN" != "1" ]; then echo refuse; exit 2; fi\n'
            ),
            check_fake_knob_inertness,
            "BEFORE its refusal",
        )

        # (A) RED: a guard line that never exits (not a real refusal).
        commit_and_check(
            "ci/scripts/perf/bad_guard_no_exit.sh",
            '#!/usr/bin/env bash\nif [ -n "${SWEEP_FAKE_BIN_SHA:-}" ] && [ "$SWEEP_DRY_RUN" != "1" ]; then echo warn; fi\n',
            check_fake_knob_inertness,
            "does not `exit`",
        )

        # (A) GREEN control: the real stacked_sweep.sh shape — guard first,
        # dry-run-only use after.
        commit_and_check(
            "ci/scripts/perf/good_guard.sh",
            (
                '#!/usr/bin/env bash\n'
                'if [ -n "${SWEEP_FAKE_BIN_SHA:-}" ] && [ "$SWEEP_DRY_RUN" != "1" ]; then echo refuse >&2; exit 2; fi\n'
                'if [ "$SWEEP_DRY_RUN" = "1" ]; then\n'
                '  if [ -n "${SWEEP_FAKE_BIN_SHA:-}" ]; then BIN_PROV_SHA="$SWEEP_FAKE_BIN_SHA"; fi\n'
                'fi\n'
            ),
            check_fake_knob_inertness,
            None,
        )

        # (A) GREEN control: knob named ONLY in a comment (e.g. a module doc)
        # — never a live code use, so nothing to guard.
        commit_and_check(
            "ci/scripts/perf/good_comment_only.sh",
            '#!/usr/bin/env bash\n# SWEEP_FAKE_BIN_SHA is documented elsewhere; not read by this script.\necho hi\n',
            check_fake_knob_inertness,
            None,
        )

        # (B) RED: names a jammi-bench binary path, invokes it, but never
        # cross-checks provenance.
        commit_and_check(
            "ci/scripts/perf/bad_no_provenance.sh",
            '#!/usr/bin/env bash\nBIN="$TARGET_DIR/release/jammi-bench"\n"$BIN" finetune-step --batch 1\n',
            check_producer_parity,
            "missing",
        )

        # (B) GREEN control: names the binary AND carries both tokens.
        commit_and_check(
            "ci/scripts/perf/good_provenance.sh",
            (
                '#!/usr/bin/env bash\n'
                'BIN="$TARGET_DIR/release/jammi-bench"\n'
                'J="$("$BIN" provenance)"\n'
                'S="$(python3 -c \'import json,sys;print(json.load(sys.stdin)["build_sha"])\' <<<"$J")"\n'
            ),
            check_producer_parity,
            None,
        )

        # (B) GREEN control: a script that never names a jammi-bench BINARY
        # path (only the crate source tree) is out of scope entirely.
        commit_and_check(
            "ci/scripts/perf/good_out_of_scope.sh",
            '#!/usr/bin/env bash\n# see crates/jammi-bench/reference/torch_finetune_step.py\necho hi\n',
            check_producer_parity,
            None,
        )

    # End-to-end: the REAL tree, both checks, must be clean today.
    real_findings = run_gate(PERF_DIR, REPO_ROOT)
    if real_findings:
        failures.append(f"self-test FAILED: real tree is not clean: {real_findings}")

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("check-producer-provenance-gates self-test: FAIL", file=sys.stderr)
        return 1
    print(
        "check-producer-provenance-gates self-test: OK — (A) FAKE-knob inertness "
        "(no-guard / use-before-guard / guard-without-exit all RED; a real guard, a "
        "comment-only mention, both GREEN) and (B) producer parity (a jammi-bench-binary "
        "producer missing provenance/build_sha is RED; one carrying both, or one that never "
        "names a binary path at all, is GREEN) both bite on throwaway fixtures; the real "
        "tree is clean."
    )
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()

    findings = run_gate(PERF_DIR, REPO_ROOT)
    if findings:
        print("check-producer-provenance-gates: FAIL", file=sys.stderr)
        for msg in findings:
            print(f"  - {msg}", file=sys.stderr)
        print(f"\ncheck-producer-provenance-gates: {len(findings)} finding(s).", file=sys.stderr)
        return 1
    print(
        "check-producer-provenance-gates: PASS — every FAKE-shaped test knob under "
        "ci/scripts/ is inert unless *DRY_RUN=1, and every ci/scripts/perf/*.sh naming a "
        "jammi-bench binary path cross-checks its provenance build_sha."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
