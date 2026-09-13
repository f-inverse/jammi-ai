#!/usr/bin/env python3
"""check_oracle_gate.py — esc-lead-invariants-2: a required check that a
branch carrying a source-code diff has a phase-5 `oracle` PASS verdict
COMMITTED, and that the verdict is FRESH at the branch's own current head —
never one recorded against an older, since-changed state of the code.

WHY. On the last program, phase 5 (`oracle`, ARCHITECTURE §4/§5,
`.claude/agents/oracle.md`) — the one gate whose verdict is never
consensus-overridable — did not run until the very end of a 57-commit
branch, where it hard-blocked immediately. Nothing prevented that ordering;
the oracle's own verdict lived only in `.jammi/gate-state/` (gitignored,
lead-writable, exactly the class of ledger
`docs/plans/53-agentic-swarm/proposals/R7-committed-rigor-record.md` was
written to stop trusting for the SAME reason). This check moves "was the
oracle actually run, and against what is actually here" into a committed
artifact a human reviews at merge, the same move R7 makes for the
pressure-tester's design pass.

THE RULE. If `origin/<base>...HEAD` (three-dot, pinned — see
`_swarm_diff_shape.py`) touches any path under `crates/**` or `cookbook/**`
(the surfaces `oracle.md`'s own principle rubric reasons about — dep
direction, cookbook one-way, lockstep version, append-only migrations,
tenant isolation, embedded<->remote parity, per-variant safety oracles),
this check is ARMED for the unit on this branch and requires, from
COMMITTED files only:

  1. A record at `docs/rigor/<unit_slug>.oracle.jsonl` — the hook's own row
     schema (`lead-gate-lib.py --export-oracle <slug>`), never hand-typed —
     exists and parses.
  2. That record carries at least one row with `agent_type == "oracle"` and
     `verdict == "PASS"` (a `HARD_BLOCK` row is not a pass — its own
     `verdict_raw` says so; `oracle.md`'s hard-blocks are never
     overridable, so a record whose only rows are HARD_BLOCK correctly
     fails this check rather than being read as "an oracle ran").
  3. At least one such PASS row is FRESH: its own `head_sha` resolves
     locally AND `git diff <head_sash> HEAD` — over every path EXCEPT
     `docs/rigor/**` itself — is EMPTY. A verdict recorded at an older
     commit, with real code changes on top of it that the oracle never
     saw, is NOT fresh and does not satisfy this check — the whole point.

FRESHNESS IS EXACT-CONTENT, NOT ANCESTRY. `check_rigor_record.py`'s own
design (R7) checked `git merge-base --is-ancestor <head_sha> HEAD` for a
DIFFERENT field and made it advisory-only, because that check is measurably
broken under this repo's own amend-then-push workflow (an amend orphans the
pre-amend sha; a shallow/transport clone never has the object at all) and,
independently, ancestry is not even the right property here: an ancestor
commit can be arbitrarily stale, and "some ancestor of HEAD passed the
oracle" is compatible with HEAD itself having since drifted arbitrarily far
from what the oracle actually read. What this check needs is the STRONGER
and, it turns out, SIMPLER property: is the code identical to what the
oracle read. `git diff <sha> HEAD -- . ':(exclude)docs/rigor/**'` answers
that directly, by content, with no ancestry reasoning at all — a
verdict recorded on a orphaned sibling commit whose TREE happens to match
HEAD's (outside the record path) is, correctly, just as fresh as one on a
literal ancestor; a verdict recorded on a real ancestor whose tree has
since diverged is, correctly, stale. This also resolves the
export-changes-HEAD bootstrap for free: the lead commits code, dispatches
`oracle` against that commit, exports the PASS row (`head_sha` = that
commit), and commits the record as a FOLLOW-UP commit or in the same
commit — either way the diff between the recorded sha and the new HEAD
touches only `docs/rigor/**`, so freshness holds without any special-cased
"the record's own commit doesn't count" carve-out.

THIS CHECK FAILS CLOSED ON AN UNRESOLVABLE `head_sha` — the opposite of
`check_rigor_record.py`'s choice for its own (different, advisory) ancestry
check. That check made unresolvability advisory because ancestry there was
never the enforcement mechanism, only a disclosure. Here, resolvability
*plus* content-equality IS the entire freshness proof; an unresolvable sha
means the proof cannot be constructed at all — CONTRACT-RULES.md's own
R-I ("a state defined by missing evidence cannot be given a definite
consequence") cuts the other way when the state in question is the one the
check exists to prevent (a stale-but-unverifiable claim of freshness): the
consequence of "cannot be shown fresh" is BLOCK, not silent pass.

KNOWN, STRUCTURAL NO-OP SHAPES — see `_swarm_diff_shape.no_op_reason` for
the shared predicates (not a `pull_request` event, dependabot, a revert, a
release-shaped diff, a purely human-authored range) plus: a diff touching
neither `crates/**` nor `cookbook/**` is simply NOT ARMED (a branch with no
such unit in it is not blocked — there is nothing for `oracle` to have
reviewed).

THE RESIDUAL, STATED PLAINLY, NEVER IMPLIED CLOSED. This check proves a
SHAPE: a committed row claims `agent_type: "oracle"`, `verdict: "PASS"`,
and a `head_sha` whose tree matches HEAD outside `docs/rigor/**`. It does
NOT prove the `oracle` subagent actually ran, or ran honestly, or that its
`checks[]` array reflects a real gate execution — `.jammi/gate-state/`
carries no `permissions.deny` entry and the lead already writes into it by
design (the exact residual `check_rigor_record.py` already states for its
own committed record). Concretely, a lead CAN satisfy this check
dishonestly: hand-append a row `{"agent_type": "oracle", "verdict": "PASS",
"head_sha": "<current HEAD sha>"}` to `docs/rigor/<slug>.oracle.jsonl` and
commit it, without ever dispatching `oracle` at all — the freshness check
would find the tree diff empty (trivially: `head_sha` IS `HEAD`) and pass.
Making the tree-equality proof strong does not make the CLAIM the row
encodes any harder to fabricate; it only makes STALENESS impossible to
fabricate past. ARMED BY THE DIFF, SATISFIED BY DISCLOSURE, JUDGED BY THE
HUMAN — the same three-clause honesty R7 states for itself, repeated here
because it is still true here, not because it is decorative.

Modes:
  python3 ci/scripts/check_oracle_gate.py                       # the check
  python3 ci/scripts/check_oracle_gate.py --check-allowlist-only-shrinks
  python3 ci/scripts/check_oracle_gate.py --self-test
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _swarm_diff_shape as ds  # noqa: E402

REPO_ROOT = ds.REPO_ROOT
RIGOR_DIR = REPO_ROOT / "docs" / "rigor"
ALLOWLIST_PATH = REPO_ROOT / "ci" / "scripts" / "oracle_gate_allowlist.txt"
ARMING_GLOBS = ("crates/*", "crates/**", "cookbook/*", "cookbook/**")


class Result:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def ok(self) -> bool:
        return not self.failures


def _display(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _unit_allowlisted(slug: str, allowlist_path: Path) -> bool:
    if not allowlist_path.exists():
        return False
    for line in allowlist_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s == slug:
            return True
    return False


def _is_fresh(cwd: Path, sha: str) -> tuple[bool, str]:
    """`(fresh, reason)` — `reason` explains a non-fresh (or unresolvable)
    result; empty when fresh."""
    ok, _ = ds.git(cwd, "rev-parse", "--verify", f"{sha}^{{commit}}")
    if not ok:
        return False, f"head_sha {sha} does not resolve in this checkout"
    ok, out = ds.git(cwd, "diff", "--name-only", sha, "HEAD", "--", ".", ":(exclude)docs/rigor/**")
    if not ok:
        return False, f"could not diff head_sha {sha} against HEAD (unrelated histories?)"
    changed = [p for p in out.splitlines() if p.strip()]
    if changed:
        shown = ", ".join(changed[:5]) + (f" (+{len(changed) - 5} more)" if len(changed) > 5 else "")
        return False, f"head_sha {sha} is stale — {len(changed)} non-record file(s) changed since: {shown}"
    return True, ""


def run_check(cwd: Path = REPO_ROOT, allowlist_path: Path = ALLOWLIST_PATH) -> Result:
    result = Result()
    base_ref = os.environ.get("GITHUB_BASE_REF", "main")

    ok, changed, range_spec = ds.compute_diff_context(cwd, base_ref)
    if not ok:
        result.note("could not resolve the base ref / three-dot diff — treating as a no-op (no PR context)")
        return result

    reason = ds.no_op_reason(cwd, changed, range_spec, ARMING_GLOBS)
    if reason is not None:
        print(f"check-oracle-gate: no-op — {reason}")
        return result

    armed_paths = [p for p in changed if ds._matches(p, ARMING_GLOBS)]
    if not armed_paths:
        print("check-oracle-gate: not armed — diff touches neither crates/** nor cookbook/**")
        return result

    slug = ds.unit_slug(cwd)
    if _unit_allowlisted(slug, allowlist_path):
        print(f"check-oracle-gate: armed but {slug!r} is on the shrink-only allowlist "
              f"({_display(allowlist_path)}) — no-op")
        return result

    print(f"check-oracle-gate: ARMED — diff touches {len(armed_paths)} oracle-reviewed path(s), "
          f"e.g. {armed_paths[:3]}")

    ok, record_text = ds.git(cwd, "show", f"HEAD:docs/rigor/{slug}.oracle.jsonl")
    if not ok:
        result.fail(f"no committed oracle verdict record at docs/rigor/{slug}.oracle.jsonl — export one with "
                    f"`python3 .claude/hooks/lead-gate-lib.py --export-oracle {slug} "
                    f"> docs/rigor/{slug}.oracle.jsonl` and commit it")
        return result

    rows = []
    for i, line in enumerate(record_text.splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            result.fail(f"docs/rigor/{slug}.oracle.jsonl:{i + 1}: not valid JSON ({exc})")

    pass_rows = [r for r in rows if isinstance(r, dict)
                 and r.get("agent_type") == "oracle" and r.get("verdict") == "PASS"]
    if not pass_rows:
        seen = sorted({r.get("verdict_raw") or r.get("verdict") for r in rows
                       if isinstance(r, dict) and r.get("agent_type") == "oracle"})
        result.fail(f"docs/rigor/{slug}.oracle.jsonl carries no oracle PASS row "
                    f"(verdicts present: {seen or 'none'}) — a HARD_BLOCK is never overridable; "
                    "clear it, get a real PASS, and re-export")
        return result

    fresh_reasons = []
    for r in pass_rows:
        sha = r.get("head_sha")
        if not isinstance(sha, str) or not sha:
            fresh_reasons.append("a PASS row carries no head_sha at all")
            continue
        fresh, why = _is_fresh(cwd, sha)
        if fresh:
            print(f"check-oracle-gate: fresh oracle PASS found at head_sha {sha}")
            return result
        fresh_reasons.append(why)

    result.fail("docs/rigor/{}.oracle.jsonl has {} PASS row(s) but none is fresh at HEAD: {}".format(
        slug, len(pass_rows), "; ".join(fresh_reasons)))
    return result


def check_allowlist_only_shrinks(cwd: Path = REPO_ROOT, allowlist_path: Path = ALLOWLIST_PATH) -> int:
    ok, _ = ds.git(cwd, "fetch", "--quiet", "origin", "main")
    if not ok:
        print("oracle-gate-allowlist-only-shrinks: FAIL — git fetch origin main failed", file=sys.stderr)
        return 1
    ok, _ = ds.git(cwd, "rev-parse", "--verify", "origin/main")
    if not ok:
        print("oracle-gate-allowlist-only-shrinks: FAIL — origin/main does not resolve", file=sys.stderr)
        return 1
    current = set()
    if allowlist_path.exists():
        for line in allowlist_path.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                current.add(s)
    rel = allowlist_path.relative_to(cwd).as_posix()
    ok, base_text = ds.git(cwd, "show", f"origin/main:{rel}")
    if not ok:
        print(f"oracle-gate-allowlist-only-shrinks: OK (bootstrap) — origin/main has no {rel} yet; "
              f"this branch's {len(current)} entries establish the baseline.")
        return 0
    base = {s.strip() for s in base_text.splitlines() if s.strip() and not s.strip().startswith("#")}
    added = current - base
    if added:
        print("oracle-gate-allowlist-only-shrinks: FAIL", file=sys.stderr)
        for e in sorted(added):
            print(f"  + {e}", file=sys.stderr)
        print("\noracle-gate-allowlist-only-shrinks: this branch adds a NEW exemption. The allowlist "
              "may only shrink — a genuinely new exemption is a human decision made directly on main, "
              "never an autonomous addition on a swarm branch.", file=sys.stderr)
        return 1
    print(f"oracle-gate-allowlist-only-shrinks: OK — {len(current)} entries "
          f"({len(base) - len(current)} shrunk vs origin/main).")
    return 0


# ==========================================================================
# --self-test — hermetic fixtures: a real `origin` + a real feature-branch
# clone, run against the REAL run_check()/check_allowlist_only_shrinks(),
# never a reimplementation — the same harness pattern check_rigor_record.py
# already established.
# ==========================================================================

class Failure(Exception):
    pass


def _assert(cond: bool, label: str, detail: str = "") -> None:
    if not cond:
        raise Failure(f"{label}: {detail}")


_FIXTURE_ENV = {
    "GIT_AUTHOR_NAME": "oracle-fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
    "GIT_COMMITTER_NAME": "oracle-fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
}


def _sh(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(cwd)] + list(args), capture_output=True, text=True)
    _assert(proc.returncode == 0, "git fixture setup", f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


_MINIMAL_SLUGIFY_STUB = (
    "import re\n"
    "def slugify(branch):\n"
    "    return re.sub(r'[^A-Za-z0-9._-]', '_', branch.strip()) or 'UNBOUND'\n"
)


def _pr_repo(tmp: Path) -> tuple[Path, Path]:
    origin = tmp / "origin"
    work = tmp / "work"
    origin.mkdir()
    env = dict(os.environ)
    env.update(_FIXTURE_ENV)
    subprocess.run(["git", "init", "-q", "-b", "main", str(origin)], check=True, env=env)
    subprocess.run(["git", "-C", str(origin), "config", "commit.gpgsign", "false"], check=True)
    subprocess.run(["git", "-C", str(origin), "config", "receive.denyCurrentBranch", "updateInstead"], check=True)
    (origin / "README.md").write_text("seed\n")
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-q", "-m", "seed"], check=True, env=env)
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True)
    subprocess.run(["git", "-C", str(work), "config", "commit.gpgsign", "false"], check=True)
    (work / ".claude" / "hooks").mkdir(parents=True)
    (work / "ci" / "scripts").mkdir(parents=True)
    (work / "docs" / "rigor").mkdir(parents=True)
    lib_text = ds.LEAD_GATE_LIB.read_text() if ds.LEAD_GATE_LIB.exists() else _MINIMAL_SLUGIFY_STUB
    (work / ".claude" / "hooks" / "lead-gate-lib.py").write_text(lib_text)
    (work / "ci" / "scripts" / "_swarm_diff_shape.py").write_text(Path(__file__).parent.joinpath("_swarm_diff_shape.py").read_text())
    (work / "ci" / "scripts" / "check_oracle_gate.py").write_text(Path(__file__).read_text())
    subprocess.run(["git", "-C", str(work), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(work), "commit", "-q", "-m", "scaffold"], check=True, env=env)
    subprocess.run(["git", "-C", str(work), "push", "-q", "origin", "HEAD:main"], check=True)
    subprocess.run(["git", "-C", str(work), "checkout", "-q", "-b", "feat/og-fixture"], check=True)
    return origin, work


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


def _run_check_in(work: Path, env_overrides: dict | None = None, allowlist: Path | None = None) -> Result:
    env = {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_BASE_REF": "main", "GITHUB_HEAD_REF": "feat/og-fixture"}
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
        global ds
        real_lib = ds.LEAD_GATE_LIB
        ds.LEAD_GATE_LIB = work / ".claude" / "hooks" / "lead-gate-lib.py"
        try:
            return run_check(work, allowlist or (work / "ci" / "scripts" / "oracle_gate_allowlist.txt"))
        finally:
            ds.LEAD_GATE_LIB = real_lib
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def fixture_og1_not_armed_docs_only() -> None:
    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "docs: a docs-only change", {"docs/only.md": "hello\n"})
        r = _run_check_in(work)
        _assert(r.ok(), "OG1", f"a docs-only diff must not arm: {r.failures}")


def fixture_og2_armed_no_record() -> None:
    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "feat: touch a crate", {"crates/foo/src/lib.rs": "pub fn f() {}\n"})
        r = _run_check_in(work)
        _assert(not r.ok(), "OG2", "armed with no oracle record must FAIL")
        _assert(any("no committed oracle verdict record" in f for f in r.failures), "OG2", f"{r.failures}")


def fixture_og3_hard_block_only_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        head = _sh(work, "rev-parse", "HEAD")
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "oracle",
                           "verdict": "BLOCK", "verdict_raw": "HARD_BLOCK", "head_sha": head})
        _commit(work, "feat: touch a crate", {
            "crates/foo/src/lib.rs": "pub fn f() {}\n",
            "docs/rigor/feat_og-fixture.oracle.jsonl": row + "\n",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "OG3", "a record with only a HARD_BLOCK row must FAIL")
        _assert(any("no oracle PASS row" in f for f in r.failures), "OG3", f"{r.failures}")


def fixture_og4_stale_pass_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        first = _commit(work, "feat: touch a crate", {"crates/foo/src/lib.rs": "pub fn f() {}\n"})
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "oracle",
                           "verdict": "PASS", "head_sha": first})
        _commit(work, "chore: export the (now-stale) oracle record", {
            "docs/rigor/feat_og-fixture.oracle.jsonl": row + "\n",
        })
        # A REAL further crates/** change lands on top, never re-verified.
        _commit(work, "feat: change the crate AFTER the recorded oracle verdict", {
            "crates/foo/src/lib.rs": "pub fn f() { /* changed */ }\n",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "OG4", f"a PASS row recorded at an OLDER commit must not satisfy a later HEAD: {r.failures}")
        _assert(any("stale" in f for f in r.failures), "OG4", f"{r.failures}")


def fixture_og5_record_only_followup_is_fresh() -> None:
    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        code_head = _commit(work, "feat: touch a crate", {"crates/foo/src/lib.rs": "pub fn f() {}\n"})
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "oracle",
                           "verdict": "PASS", "head_sha": code_head})
        _commit(work, "chore: export the oracle record for the code above, unchanged since", {
            "docs/rigor/feat_og-fixture.oracle.jsonl": row + "\n",
        })
        r = _run_check_in(work)
        _assert(r.ok(), "OG5", f"a record-only follow-up commit (code unchanged) must stay FRESH: {r.failures}")


def fixture_og6_same_commit_head_sha_is_fresh() -> None:
    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        # Write both the code AND the record in the SAME commit, head_sha
        # pointing at that very commit (resolvable only once committed —
        # the fixture computes it after commit and rewrites the row via a
        # second, tiny follow-up, which is itself the OG5 shape; exercised
        # separately here as "head_sha == HEAD directly", the trivial case).
        head = _sh(work, "rev-parse", "HEAD")
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "oracle",
                           "verdict": "PASS", "head_sha": head})
        _commit(work, "feat: touch a crate + its own oracle record", {
            "crates/foo/src/lib.rs": "pub fn f() {}\n",
            "docs/rigor/feat_og-fixture.oracle.jsonl": row + "\n",
        })
        real_head = _sh(work, "rev-parse", "HEAD")
        row2 = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "oracle",
                            "verdict": "PASS", "head_sha": real_head})
        _commit(work, "chore: point the record at its own committed head", {
            "docs/rigor/feat_og-fixture.oracle.jsonl": row2 + "\n",
        })
        r = _run_check_in(work)
        _assert(r.ok(), "OG6", f"head_sha == HEAD directly must be fresh: {r.failures}")


def fixture_og7_unresolvable_sha_fails_closed() -> None:
    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "oracle", "verdict": "PASS",
                           "head_sha": "cafef00d1234567890abcdef1234567890abcdef"})
        _commit(work, "feat: touch a crate", {
            "crates/foo/src/lib.rs": "pub fn f() {}\n",
            "docs/rigor/feat_og-fixture.oracle.jsonl": row + "\n",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "OG7", "an unresolvable head_sha must FAIL CLOSED here (unlike R7's advisory ancestry check)")
        _assert(any("does not resolve" in f for f in r.failures), "OG7", f"{r.failures}")


def fixture_og8_noop_shapes() -> None:
    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "feat: touch a crate, human-authored", {"crates/foo/src/lib.rs": "pub fn f() {}\n"}, swarm=False)
        r = _run_check_in(work)
        _assert(r.ok(), "OG8a", f"a human-authored-only range must no-op: {r.failures}")

    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "chore: bump crate version", {"crates/foo/Cargo.toml": "[package]\nversion=\"0.2.0\"\n"})
        r = _run_check_in(work)
        _assert(r.ok(), "OG8b", f"a release-shaped (Cargo.toml-only) diff must no-op: {r.failures}")

    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "feat: touch a crate", {"crates/foo/src/lib.rs": "pub fn f() {}\n"})
        r = _run_check_in(work, {"GITHUB_ACTOR": "dependabot[bot]"})
        _assert(r.ok(), "OG8c", f"a dependabot actor must no-op: {r.failures}")

    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "feat: touch a crate", {"crates/foo/src/lib.rs": "pub fn f() {}\n"})
        _commit(work, 'Revert "feat: touch a crate"', {"crates/foo/src/lib.rs": "orig\n"})
        r = _run_check_in(work)
        _assert(r.ok(), "OG8d", f"a revert HEAD commit must no-op: {r.failures}")

    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        r = _run_check_in(work, {"GITHUB_EVENT_NAME": "push"})
        _assert(r.ok(), "OG8e", f"a non-pull_request event must no-op: {r.failures}")

    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "docs: a branch with no crates/cookbook unit in it at all", {"docs/x.md": "x\n"})
        r = _run_check_in(work)
        _assert(r.ok(), "OG8f", f"a branch touching no oracle-reviewed path must not arm: {r.failures}")


def fixture_og9_allowlisted_unit_noop() -> None:
    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        allow = work / "ci" / "scripts" / "oracle_gate_allowlist.txt"
        allow.write_text("# fixture allowlist\nfeat_og-fixture\n")
        _commit(work, "feat: touch a crate", {"crates/foo/src/lib.rs": "pub fn f() {}\n"})
        r = _run_check_in(work, allowlist=allow)
        _assert(r.ok(), "OG9", f"an allowlisted unit must no-op even though armed: {r.failures}")


def fixture_og10_allowlist_only_shrinks() -> None:
    with tempfile.TemporaryDirectory(prefix="og-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        allow_rel = "ci/scripts/oracle_gate_allowlist.txt"
        _commit(work, "seed allowlist", {allow_rel: "unit-a\n"})
        _sh(work, "push", "-q", "origin", "HEAD:main")
        _sh(work, "fetch", "-q", "origin", "main")
        allow_path = work / allow_rel
        rc = check_allowlist_only_shrinks(work, allow_path)
        _assert(rc == 0, "OG10a", "an unchanged allowlist must pass the shrink-only ratchet")
        allow_path.write_text("unit-a\nunit-b\n")
        rc = check_allowlist_only_shrinks(work, allow_path)
        _assert(rc != 0, "OG10b", "adding a NEW entry (not yet on origin/main) must FAIL the ratchet")


OG_FIXTURES = [
    ("OG1", fixture_og1_not_armed_docs_only),
    ("OG2", fixture_og2_armed_no_record),
    ("OG3", fixture_og3_hard_block_only_fails),
    ("OG4", fixture_og4_stale_pass_fails),
    ("OG5", fixture_og5_record_only_followup_is_fresh),
    ("OG6", fixture_og6_same_commit_head_sha_is_fresh),
    ("OG7", fixture_og7_unresolvable_sha_fails_closed),
    ("OG8", fixture_og8_noop_shapes),
    ("OG9", fixture_og9_allowlisted_unit_noop),
    ("OG10", fixture_og10_allowlist_only_shrinks),
]


def self_test() -> int:
    failures: list[str] = []
    for name, fn in OG_FIXTURES:
        try:
            fn()
            print(f"check-oracle-gate[{name}]: OK")
        except Failure as e:
            failures.append(f"{name}: {e}")
            print(f"check-oracle-gate[{name}]: FAIL — {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: unexpected exception: {e!r}")
            print(f"check-oracle-gate[{name}]: FAIL (unexpected exception) — {e!r}", file=sys.stderr)
    if failures:
        print("check-oracle-gate: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"check-oracle-gate: all {len(OG_FIXTURES)} self-test fixture(s) passed.")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    if "--check-allowlist-only-shrinks" in argv:
        return check_allowlist_only_shrinks()
    result = run_check()
    for n in result.notes:
        print(f"check-oracle-gate: NOTE — {n}")
    if not result.ok():
        print("check-oracle-gate: FAIL", file=sys.stderr)
        for f in result.failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("check-oracle-gate: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
