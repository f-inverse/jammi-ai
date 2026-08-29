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

REPO_ROOT = Path(__file__).resolve().parents[3]
# Fail loud rather than scan-zero-silently: this file lives at
# `ci/scripts/perf/check_producer_provenance_gates.py`, three directories
# below the repo root, and every OTHER sibling script under `ci/scripts/
# perf/` already uses `parents[3]` for exactly this reason (a prior
# `parents[2]` resolved to `<repo>/ci` instead — `git ls-files <prefix>` run
# with THAT as `cwd` looks for `ci/scripts/**` under `<repo>/ci/ci/scripts/
# **`, which never exists, so both `run_gate` and this module's own
# `self_test`'s "the real tree is clean" end-to-end arm passed VACUOUSLY —
# zero files scanned, zero findings, reads as PASS). A silent scan-zero is
# worse than a loud crash: this assertion makes a future re-introduction of
# that mistake (or a refactor that moves this file another directory deep)
# fail on the very first line of `main()`/`self_test()` that touches
# `REPO_ROOT`, not silently downstream as an empty findings list that looks
# identical to "everything is fine".
assert (REPO_ROOT / "Cargo.toml").is_file(), (
    f"REPO_ROOT resolved to {REPO_ROOT}, which has no Cargo.toml -- "
    "parents[N] is wrong for this file's depth under the repo root"
)
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


_IF_FI_TOKEN_RE = re.compile(r"\b(if|fi)\b")
_GUARD_BLOCK_MAX_SCAN = 40


def _guard_block_lines(lines: list[str], start_idx: int) -> list[str]:
    """The physical lines of the `if [...]; then ... fi` block whose OWN
    condition line is `lines[start_idx]` — a `bash`-`if`/`fi` DEPTH walk
    (nested `if`s inside the guard both increment and later decrement the
    same counter, so a nested conditional inside the guard body does not
    prematurely end the scan), not a fixed line count.

    Round-N false positive this replaces: the ORIGINAL implementation
    concatenated only `lines[start_idx]` and `lines[start_idx + 1]` — two
    lines — before searching for `exit`. `stacked_sweep.sh`'s own real
    `SWEEP_FAKE_BIN_SHA` guard (contract C5.2) is a legitimate THREE
    physical-line shape: `if [ -n "$VAR" ] && [ "$DRY_RUN" != "1" ]; then`
    / `echo "::error::..." >&2` / `exit 2` — the `exit` sits on the guard's
    THIRD line, one line past what a 2-line window could ever see, so that
    guard read as a false-positive FINDING ("does not `exit`") even though
    it manifestly does. Bounded at `_GUARD_BLOCK_MAX_SCAN` lines so a
    malformed/never-closed `if` cannot make this loop unbounded — an
    unterminated block still returns everything up to the cap, which keeps
    the caller's `"exit" not in guard_window` check fail-closed (a
    guard whose `fi` never resolves within the cap is treated the same as
    one with no `exit` at all, never silently credited).
    """
    depth = 0
    end = start_idx
    limit = min(len(lines), start_idx + _GUARD_BLOCK_MAX_SCAN)
    for i in range(start_idx, limit):
        for tok in _IF_FI_TOKEN_RE.findall(lines[i]):
            depth += 1 if tok == "if" else -1
        end = i
        if depth <= 0:
            break
    return lines[start_idx : end + 1]


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
        guard_window = "\n".join(_guard_block_lines(lines, first_guard))
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


# CI incident (run 33230050451, main, "Guard (arch validation freshness
# self-test)"), same class here: `shutil.rmtree` during a `tempfile.
# TemporaryDirectory`'s teardown can hit `OSError: [Errno 39] Directory not
# empty: '.git'` — a race between tempdir cleanup and a background `git
# maintenance`/`gc --auto` process the scratch repo `self_test` builds below
# can spawn. `-c gc.auto=0 -c gc.autoDetach=false -c maintenance.auto=false`
# kills the background writer AT THE SOURCE.
_GIT_NO_BACKGROUND_MAINTENANCE = ("-c", "gc.auto=0", "-c", "gc.autoDetach=false", "-c", "maintenance.auto=false")


def _scratch_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *_GIT_NO_BACKGROUND_MAINTENANCE, *args], cwd=cwd, check=True)


def run_gate(perf_dir: Path, repo_root: Path) -> list[str]:
    findings: list[str] = []
    for path in _tracked_sh_under(repo_root, "ci/scripts/"):
        findings += check_fake_knob_inertness(path)
    for path in _tracked_sh_under(repo_root, "ci/scripts/perf/"):
        findings += check_producer_parity(path)
    return findings


def self_test() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        repo = Path(tmp)
        _scratch_git(["init", "-q"], repo)
        _scratch_git(["config", "user.email", "test@example.com"], repo)
        _scratch_git(["config", "user.name", "Test"], repo)
        perf = repo / "ci" / "scripts" / "perf"
        perf.mkdir(parents=True)

        def commit_and_check(rel: str, text: str, check_fn, expect_hit: str | None):
            p = repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            _scratch_git(["add", "-A"], repo)
            _scratch_git(["commit", "-q", "-m", rel], repo)
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

        # (A) RED, still: a THREE physical-line guard block that genuinely
        # never `exit`s (warns and falls through) — proves the widened
        # if/fi-depth window is not a rubber stamp: it reads the WHOLE
        # enclosing block, not "any 3+ lines", and still fires when that
        # block really has no `exit` anywhere in it.
        commit_and_check(
            "ci/scripts/perf/bad_guard_no_exit_multiline.sh",
            (
                '#!/usr/bin/env bash\n'
                'if [ -n "${SWEEP_FAKE_BIN_SHA:-}" ] && [ "$SWEEP_DRY_RUN" != "1" ]; then\n'
                '  echo "::error::refusing" >&2\n'
                'fi\n'
            ),
            check_fake_knob_inertness,
            "does not `exit`",
        )

        # (A) GREEN control (round-N false-positive fix): `stacked_sweep.sh`'s
        # OWN real guard shape — a THREE physical-line `if`/`echo`/`exit`/`fi`
        # block, `exit` on the guard's third line. The ORIGINAL 2-line window
        # (`lines[first_guard]` + `lines[first_guard + 1]`) never reached the
        # `exit` line at all and misreported this exact shape as "does not
        # `exit`" — see `_guard_block_lines`'s own docstring.
        commit_and_check(
            "ci/scripts/perf/good_guard_three_line.sh",
            (
                '#!/usr/bin/env bash\n'
                'if [ -n "${SWEEP_FAKE_BIN_SHA:-}" ] && [ "$SWEEP_DRY_RUN" != "1" ]; then\n'
                '  echo "::error::SWEEP_FAKE_BIN_SHA is set but SWEEP_DRY_RUN != 1" >&2\n'
                '  exit 2\n'
                'fi\n'
            ),
            check_fake_knob_inertness,
            None,
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

    # Non-vacuousness control (the actual bug this round fixes): a wrong
    # `REPO_ROOT` (previously `parents[2]`, resolving to `<repo>/ci` instead
    # of `<repo>`) makes `git ls-files ci/scripts/` run with the WRONG `cwd`
    # look for `<repo>/ci/ci/scripts/**`, which never exists — zero files,
    # zero findings, a PASS that enforced nothing. Assert BOTH tracked-file
    # scans this gate depends on see a REAL, nonzero count on the actual
    # repo tree, so a future regression of `REPO_ROOT` (or of the
    # `Cargo.toml` guard above being weakened/removed) cannot silently
    # revert to scanning nothing while still printing PASS.
    real_sh_under_scripts = _tracked_sh_under(REPO_ROOT, "ci/scripts/")
    real_sh_under_perf = _tracked_sh_under(REPO_ROOT, "ci/scripts/perf/")
    if not real_sh_under_scripts:
        failures.append(
            "self-test FAILED: `git ls-files ci/scripts/` under the real REPO_ROOT found ZERO "
            "`.sh` files -- the scan is vacuous (REPO_ROOT is almost certainly wrong)"
        )
    if not real_sh_under_perf:
        failures.append(
            "self-test FAILED: `git ls-files ci/scripts/perf/` under the real REPO_ROOT found "
            "ZERO `.sh` files -- the scan is vacuous (REPO_ROOT is almost certainly wrong)"
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
