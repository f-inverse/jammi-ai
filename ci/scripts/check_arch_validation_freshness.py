#!/usr/bin/env python3
"""Arch-validation FRESHNESS gate — hermetic, static, no build, no GPU.

## The gap this closes

M3 made per-arch validation representable at all: `crates/jammi-kernels/
build.rs::VALIDATED_SMS` is the STRUCTURAL admitted set every flash-attn
fence reads (`crate::admission::flash_validated_arches`), narrower than the
merely-COMPILED `GENCODE_ARCHES`, and each entry's evidence is a committed
per-arch pod-parity artifact under `crates/jammi-kernels/artifacts/
cuda-runs/` (the four `m3-arch-set-*` files this gate reads).

Nothing, until this gate, re-demands that evidence when the validated
SURFACE changes. A future edit to the FA2 build (`build.rs`), the flash
kernels themselves (`src/flash/`), the vendored FlashAttention-2 sources
(`third_party/flash-attention/`), or the CUDA-side admission fence
(`src/admission.rs`) can land with every hermetic test still green — those
tests exercise CPU-only code paths and static pins, never the actual GPU
kernel behaviour the arch-set artifacts prove — while every existing
"VALIDATED" table cell and `VALIDATED_SMS` entry keeps proving a tree that
no longer exists. This is the esc-050/esc-051 shape (evidence present, dead
against current code) applied to the arch-validation surface specifically.

## Rule 1 — evidence exists

Every arch string in `build.rs::VALIDATED_SMS` (parsed from the LITERAL
array — `parse_validated_sms` below regex-extracts the quoted string
literals out of the actual `pub(crate) const VALIDATED_SMS: &[&str] = &[...]`
declaration; never a hand-copied list that could silently drift from the
real array, the same anti-drift discipline `check_cuda_run_artifacts.py`'s
own rule (i) uses for `crates/jammi-bench/src/report.rs`'s identity-field
consts) must have at least one committed artifact under
`crates/jammi-kernels/artifacts/cuda-runs/` (top-level `*.json` only — see
"Why top-level only" below) that:

  - identifies that arch (see "The discriminator" below),
  - carries `status == "GREEN"`, and
  - carries a `git_sha` that is a 40-hex ancestor of `HEAD`.

An arch with zero such artifacts is a hard FAIL naming the arch.

### The discriminator

Every artifact under `cuda-runs/` carries a free-text `box` field (a human-
written pod description, e.g. `"jammi-a100 (NVIDIA A100-SXM4-80GB, driver
570.172.08, compute_cap 8.0)"`). Reading the four `m3-arch-set-*` artifacts
that exist today shows exactly one robust, already-present convention: a
`compute_cap <major>.<minor>` substring. This gate's discriminator is
`arch_from_box`: regex `compute_cap` (a literal digit-dot-digit capture) on `box`, mapped to the
two-digit `sm_XX`-style string `VALIDATED_SMS` itself uses by
CONCATENATING the digits (`8.0` -> `"80"`, `8.6` -> `"86"`, `8.9` -> `"89"`,
`9.0` -> `"90"` — and, honestly, `10.0` would concatenate to `"100"`,
matching the `arch=compute_100,code=sm_100` convention `GENCODE_ARCHES`'s
own doc anticipates for a future arch, never zero-padded to a fixed width).
An artifact whose `box` carries no `compute_cap` substring (every artifact
that predates the M3 arch-set work — checked against the real tree: none of
them carry this substring) identifies NO arch and is silently excluded from
this gate's candidate pool — it was never claiming per-arch validation
evidence in the first place, so this is not a finding, just a non-match.

### Why top-level only

`cuda-runs/*-raw-runs/` subdirectories hold per-seed/per-leg raw dumps that
inherit no independent `box`/`status`/`git_sha` triple of their own (see
`check_cuda_run_artifacts.py`'s own module doc, defect 3) — they are never,
by themselves, "a committed artifact [that] identifies that arch"; the
top-level artifact they were folded into is. Recursing into them would only
ever produce non-matches (no `compute_cap`), so this gate globs
`cuda-runs/*.json` only, never `cuda-runs/**/*.json`.

## Rule 2 — freshness

For each `VALIDATED_SMS` entry with >=1 Rule-1-qualifying artifact, take the
NEWEST such artifact (by its own `date` field, ISO-8601 UTC; an unparsable/
missing `date` sorts as the oldest possible, so it never masks a genuinely
newer sibling — ties broken by filename, descending, for determinism) and
compute:

    git diff --name-only <that artifact's git_sha>..HEAD -- <flash surface>

The flash surface (`FLASH_SURFACE` below) is deliberately the WHOLE of:

  - `crates/jammi-kernels/build.rs` (the `-gencode` set and `VALIDATED_SMS`
    itself — `GENCODE_ARCHES`'s own doc: "adding a `-gencode` pair alone
    leaves that arch compiled-but-REFUSED... until its OWN entry lands
    here, which is the commit where that arch's per-arch pod parity
    artifact also lands" — a change here is exactly the class this gate
    exists to catch),
  - `crates/jammi-kernels/src/flash/` (the Rust-side flash kernel surface),
  - `crates/jammi-kernels/third_party/flash-attention/` (the vendored FA2
    sources) — INCLUDING `VENDORED.md` prose. A carve-out for "docs-only"
    changes under this directory was considered and rejected: `VENDORED.md`
    is not incidental prose here, it is the sha256-pinned file manifest
    (`## Files (sha256 of the vendored copy...)`) and the per-arch
    VALIDATED table this gate's own module doc points at as "the single
    source to cross-check" — a hand-edit to that table (e.g. someone
    manually flipping a cell to VALIDATED without a new pod run) is exactly
    the kind of surface drift this gate must not wave through. Default:
    include everything under this directory, and let a genuinely docs-only
    touch demand either a fresh artifact (cheap: nothing about the CUDA
    surface actually changed, so re-running the suite is a formality) or an
    explicit Rule-3 waiver.
  - `crates/jammi-kernels/src/admission.rs` (the CUDA-side `flash_validated_
    arches`/`check_arch` fence that actually reads `VALIDATED_SMS` at
    runtime).

`crates/jammi-encoders/src/modernbert.rs` (the encoder-side flash fence) is
DELIBERATELY EXCLUDED from the trigger surface. A whole-file trigger on
that file would fire on every unrelated encoder-side edit (it changes every
unit, per the M3 hand-off's own framing) with zero signal about the CUDA
surface. The honest resolution: `modernbert.rs`'s OWN correctness — that it
calls into `flash_validated_arches`/`check_arch` at all and degrades
correctly when an arch is not validated — is covered by this crate's own
hermetic pin tests (`admission.rs`'s `flash_validated_arches_env_var_is_a_
pinned_subset_of_compiled` and the `modernbert.rs` flash-arm fence tests
compiled into every CI run), not by re-demanding a GPU pod run on every
encoder edit. This gate's job is narrower and specific: re-demand evidence
when the COMPILED KERNEL SURFACE (what actually runs on the GPU) changes,
not every consumer of its Rust-level admission API. Flagged here per the
task brief's own request, for a human to confirm or override.

A stale entry (non-empty diff) is a hard FAIL unless Rule 3 waives it.

## Rule 3 — waiver

`ci/scripts/arch_validation_freshness_allowlist.txt` (same TAB-delimited,
`#`-comment, rot-checked shape as `execution_surface_reachability_
allowlist.txt`): one row per line, `<arch><TAB><reviewed_up_to_sha><TAB>
<reason>`. A row records "arch `<arch>` is deliberately accepted as stale;
its evidence was last reviewed as sufficient through `<reviewed_up_to_sha>`,
for `<reason>`" — the sha range it covers is `<the qualifying artifact's own
git_sha>..<reviewed_up_to_sha>`.

Rot, all hard FAILs:

  - `arch` not present in the CURRENT `VALIDATED_SMS` (an arch that was
    dropped, or never existed, has no meaning to waive staleness for).
  - `reviewed_up_to_sha` not a well-formed 40-hex sha, or not an ancestor of
    `HEAD` (a waiver cannot reach into the future).
  - `arch` is NOT currently Rule-2-stale at all — a "dead waiver": once a
    fresh artifact lands (or the surface reverts), the waiver row itself
    becomes inert prose and must be deleted, exactly like a reachability
    row for a tuple that "has become reachable on the merge path" in the
    sibling allowlist.
  - the waiver's range no longer matches HEAD: `git diff --name-only
    <reviewed_up_to_sha>..HEAD -- <flash surface>` is NON-empty — i.e. the
    surface moved again AFTER the point this waiver last reviewed, so the
    accepted-staleness claim no longer covers the current tree and must be
    re-reviewed (bump `reviewed_up_to_sha`, or drop the arch).

A row that is well-formed, names a currently-VALIDATED_SMS arch, is
genuinely Rule-2-stale, and whose range DOES still cover HEAD (the diff
above is empty) suppresses that arch's Rule 2 STALE finding.

## Fail-closed contract

Every finding is NAMED (which arch, which artifact, which rule) — never a
bare non-zero exit. A STALE finding names the arch, the artifact
(`git_sha` + relpath), the changed files, and the two resolutions: land a
fresh per-arch pod-parity artifact validating that arch against the current
surface, OR drop the arch from `VALIDATED_SMS` in `build.rs` until
re-validated.

Ancestry (`git merge-base --is-ancestor`) and the surface diff (`git diff
<sha>..HEAD`) both need REAL commit history: a shallow checkout
(`actions/checkout`'s default `fetch-depth: 1`) makes every `git_sha` read
back as a false non-ancestor, indistinguishable from a genuine one without
this guard — checked BEFORE any per-artifact work, one explicit failure
naming the shallow checkout, never N misleading findings (the same
discipline `check_cuda_run_artifacts.py`/`check_pod_build_timings.py`
already use).

## Expected result on the real repo, today

GREEN: all four `VALIDATED_SMS` entries (`80`/`86`/`89`/`90`) have a GREEN,
ancestor-`git_sha` artifact at `80a451aa0d5dbaa07a1f0594d94453fa3fe03a29`
(the four `2026-08-28-m3-arch-set-80a451a-*.json` files), and the flash
surface is UNCHANGED between that sha and the M2/M3 train tip
(`git diff --name-only 80a451aa..HEAD -- <flash surface>` is empty — verify
this yourself against the real checkout before trusting this note; it is a
statement about the tree at the time this gate was written, not a promise
this gate itself enforces staying true).

Expected future interaction: the concurrently-developed `feat/m2-memeff-op`
family lives on CPU-hermetic ops (`CustomOp3`) OUTSIDE the flash surface, so
it does not redden this gate on its own. If a LATER branch (e.g. a
"memeff part 2" wiring pass) touches `crates/jammi-kernels/src/flash/` or
`build.rs` to integrate memory-efficient attention with the flash path, this
gate goes RED for every `VALIDATED_SMS` arch the moment that branch merges,
by design — the train's own final pre-merge validation pass is expected to
refresh the four per-arch artifacts (or add a scoped Rule-3 waiver) as part
of landing that change, not to discover this gate's failure as a surprise.

Run: `python3 ci/scripts/check_arch_validation_freshness.py`
Self-test (RED cases for every rule above, on throwaway `git init`'d
fixture repos — never the real checkout):
`python3 ci/scripts/check_arch_validation_freshness.py --self-test`
Hermetic: reads the working tree (or an ephemeral tempdir git repo under
`--self-test`) and shells out only to `git`; no network, no cargo, no GPU.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_RS_PATH = REPO_ROOT / "crates" / "jammi-kernels" / "build.rs"
CUDA_RUNS_DIR = REPO_ROOT / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs"
ALLOWLIST_PATH = REPO_ROOT / "ci" / "scripts" / "arch_validation_freshness_allowlist.txt"

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
VALIDATED_SMS_RE = re.compile(r"const VALIDATED_SMS:\s*&\[&str\]\s*=\s*&\[(.*?)\];", re.DOTALL)
STR_LIT_RE = re.compile(r'"([^"]*)"')
COMPUTE_CAP_RE = re.compile(r"compute_cap\s+(\d+)\.(\d+)")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# The flash surface (module doc "Rule 2 — freshness" above): relative,
# forward-slash paths handed straight to `git diff -- <pathspec>...`. Each
# may be a file or a directory (git's own `--` pathspec matches either);
# deliberately EXCLUDES crates/jammi-encoders/src/modernbert.rs — see the
# module doc's own "Why top-level only" / modernbert.rs carve-out section.
FLASH_SURFACE: tuple[str, ...] = (
    "crates/jammi-kernels/build.rs",
    "crates/jammi-kernels/src/flash/",
    "crates/jammi-kernels/third_party/flash-attention/",
    "crates/jammi-kernels/src/admission.rs",
)

SHALLOW_CHECKOUT_MESSAGE = "shallow checkout — ancestry/diff cannot be evaluated; use fetch-depth: 0"
ANCESTOR_MESSAGE = (
    "is not an ancestor of HEAD — a green artifact whose sha is not an ancestor of the "
    "branch is evidence about a tree that no longer exists."
)


class ArtifactError(Exception):
    """Uncomputable input (parse failure, missing dir) — fails closed."""


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def is_shallow_repository(repo_root: Path) -> bool:
    proc = _run(["git", "rev-parse", "--is-shallow-repository"], repo_root)
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _is_ancestor(sha: str, repo_root: Path, target: str = "HEAD") -> bool:
    proc = _run(["git", "merge-base", "--is-ancestor", sha, target], repo_root)
    return proc.returncode == 0


def _changed_surface_files(sha: str, repo_root: Path, target: str = "HEAD") -> list[str]:
    proc = _run(["git", "diff", "--name-only", f"{sha}..{target}", "--", *FLASH_SURFACE], repo_root)
    if proc.returncode != 0:
        raise ArtifactError(f"`git diff --name-only {sha}..{target}` failed: {proc.stderr.strip()}")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _parse_date(value) -> datetime:
    """Best-effort ISO-8601 UTC parse for `date` — used ONLY to order
    candidates by recency (never for pass/fail correctness). An unparsable
    or missing value sorts as the oldest possible timestamp, so it can never
    masquerade as "the newest evidence" over a genuinely dated sibling.
    """
    if isinstance(value, str) and TS_RE.match(value):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return datetime.min


# --------------------------------------------------------------------------- #
# parse VALIDATED_SMS — the literal array, never a hand-copied list
# --------------------------------------------------------------------------- #
def parse_validated_sms(build_rs_path: Path) -> list[str]:
    if not build_rs_path.is_file():
        raise ArtifactError(f"build.rs not found: {build_rs_path}")
    text = build_rs_path.read_text(encoding="utf-8")
    m = VALIDATED_SMS_RE.search(text)
    if m is None:
        raise ArtifactError(
            f"no `const VALIDATED_SMS: &[&str] = &[...]` literal array found in {build_rs_path} — "
            "this gate reads the real array, never a hand-copied list, and cannot proceed without it"
        )
    archs = STR_LIT_RE.findall(m.group(1))
    if not archs:
        raise ArtifactError(f"VALIDATED_SMS array matched in {build_rs_path} but named zero archs")
    return archs


# --------------------------------------------------------------------------- #
# discriminator + artifact loading
# --------------------------------------------------------------------------- #
def arch_from_box(box) -> str | None:
    if not isinstance(box, str):
        return None
    m = COMPUTE_CAP_RE.search(box)
    if m is None:
        return None
    return f"{m.group(1)}{m.group(2)}"


def load_artifacts(cuda_runs_dir: Path) -> tuple[list[dict], list[str]]:
    """Returns (records, findings). Each record: {relpath, data, arch}. A
    file that cannot be read/parsed produces a named finding and is excluded
    from the candidate pool (never silently dropped)."""
    if not cuda_runs_dir.is_dir():
        raise ArtifactError(f"cuda-runs dir not found: {cuda_runs_dir}")

    records: list[dict] = []
    findings: list[str] = []
    for f in sorted(cuda_runs_dir.glob("*.json")):
        relpath = f.name
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            findings.append(f"{relpath}: could not read file: {e}")
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            findings.append(f"{relpath}: JSON parse error: {e}")
            continue
        if not isinstance(data, dict):
            findings.append(f"{relpath}: top-level JSON value is not an object")
            continue
        arch = arch_from_box(data.get("box"))
        records.append({"relpath": relpath, "data": data, "arch": arch})
    return records, findings


# --------------------------------------------------------------------------- #
# Rule 1 — evidence exists
# --------------------------------------------------------------------------- #
def check_rule1_evidence(
    validated_sms: list[str], records: list[dict], repo_root: Path
) -> tuple[dict[str, list[dict]], list[str]]:
    candidates_by_arch: dict[str, list[dict]] = {arch: [] for arch in validated_sms}
    findings: list[str] = []

    for rec in records:
        arch = rec["arch"]
        if arch is None or arch not in candidates_by_arch:
            continue  # does not identify a VALIDATED_SMS arch — not this gate's business
        data = rec["data"]
        status = data.get("status")
        sha = data.get("git_sha")
        if status != "GREEN":
            continue
        if not isinstance(sha, str) or not GIT_SHA_RE.match(sha):
            continue
        if not _is_ancestor(sha, repo_root):
            continue
        candidates_by_arch[arch].append(rec)

    for arch in validated_sms:
        if not candidates_by_arch[arch]:
            findings.append(
                f"arch {arch}: VALIDATED_SMS entry has NO qualifying evidence — no committed "
                f"artifact under {CUDA_RUNS_DIR.relative_to(REPO_ROOT)}/*.json identifies compute_cap "
                f"for arch {arch} with status GREEN and a git_sha that is an ancestor of HEAD. Land a "
                "per-arch pod-parity artifact, or drop this arch from VALIDATED_SMS in build.rs."
            )

    return candidates_by_arch, findings


def _newest(candidates: list[dict]) -> dict:
    return sorted(candidates, key=lambda r: (_parse_date(r["data"].get("date")), r["relpath"]), reverse=True)[0]


# --------------------------------------------------------------------------- #
# Rule 2 — freshness
# --------------------------------------------------------------------------- #
def check_rule2_freshness(
    candidates_by_arch: dict[str, list[dict]], repo_root: Path
) -> dict[str, tuple[dict, list[str]]]:
    """Returns {arch: (newest_record, changed_files)} for every arch that
    HAS qualifying evidence (Rule 1) — `changed_files` is empty iff fresh."""
    stale: dict[str, tuple[dict, list[str]]] = {}
    for arch, candidates in candidates_by_arch.items():
        if not candidates:
            continue  # Rule 1 already reported this arch
        newest = _newest(candidates)
        sha = newest["data"]["git_sha"]
        changed = _changed_surface_files(sha, repo_root)
        if changed:
            stale[arch] = (newest, changed)
    return stale


def _stale_message(arch: str, newest: dict, changed: list[str]) -> str:
    sha = newest["data"]["git_sha"]
    return (
        f"arch {arch}: VALIDATED_SMS entry is STALE — its newest qualifying artifact "
        f"`{newest['relpath']}` (git_sha {sha}) predates changes to the flash surface: "
        f"{', '.join(changed)}. Resolve by (a) landing a fresh per-arch pod-parity artifact "
        f"re-validating arch {arch} against the current surface, or (b) dropping \"{arch}\" from "
        "VALIDATED_SMS in build.rs until re-validated."
    )


# --------------------------------------------------------------------------- #
# Rule 3 — waiver
# --------------------------------------------------------------------------- #
def load_waivers(path: Path) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Returns ({arch: (reviewed_up_to_sha, reason)}, findings). A duplicate
    arch row is a finding (never last-one-wins)."""
    waivers: dict[str, tuple[str, str]] = {}
    findings: list[str] = []
    if not path.is_file():
        return waivers, findings
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 3:
            findings.append(
                f"{path.name}:{lineno}: malformed row (expected exactly 3 TAB-separated fields "
                f"<arch><TAB><reviewed_up_to_sha><TAB><reason>), got {len(parts)}: {raw!r}"
            )
            continue
        arch, reviewed_up_to_sha, reason = (p.strip() for p in parts)
        if not reason:
            findings.append(f"{path.name}:{lineno}: waiver for arch {arch!r} has an empty reason")
        if arch in waivers:
            findings.append(f"{path.name}:{lineno}: duplicate waiver row for arch {arch!r}")
            continue
        waivers[arch] = (reviewed_up_to_sha, reason)
    return waivers, findings


def check_rule3_waivers(
    waivers: dict[str, tuple[str, str]],
    validated_sms: list[str],
    stale: dict[str, tuple[dict, list[str]]],
    repo_root: Path,
) -> tuple[list[str], set[str]]:
    """Returns (findings, suppressed_archs) — `suppressed_archs` is the set
    of arches whose Rule 2 STALE finding is waived (well-formed, currently
    stale, and the waiver's own range still covers HEAD)."""
    findings: list[str] = []
    suppressed: set[str] = set()

    for arch, (reviewed_up_to_sha, reason) in waivers.items():
        if arch not in validated_sms:
            findings.append(
                f"{ALLOWLIST_PATH.name}: waiver for arch {arch!r} is ROT — that arch is not in the "
                "current VALIDATED_SMS; delete this row"
            )
            continue
        if not GIT_SHA_RE.match(reviewed_up_to_sha):
            findings.append(
                f"{ALLOWLIST_PATH.name}: waiver for arch {arch!r} has a malformed reviewed_up_to_sha "
                f"(must be 40 lowercase hex chars), got {reviewed_up_to_sha!r}"
            )
            continue
        if not _is_ancestor(reviewed_up_to_sha, repo_root):
            findings.append(
                f"{ALLOWLIST_PATH.name}: waiver for arch {arch!r} names reviewed_up_to_sha "
                f"{reviewed_up_to_sha} which {ANCESTOR_MESSAGE}"
            )
            continue
        if arch not in stale:
            findings.append(
                f"{ALLOWLIST_PATH.name}: waiver for arch {arch!r} is a DEAD WAIVER — that arch is "
                "not currently Rule-2-stale (a fresh artifact landed, or the surface reverted); "
                "delete this row"
            )
            continue
        changed_since_review = _changed_surface_files(reviewed_up_to_sha, repo_root)
        if changed_since_review:
            findings.append(
                f"{ALLOWLIST_PATH.name}: waiver for arch {arch!r} (reviewed_up_to_sha "
                f"{reviewed_up_to_sha}) no longer covers HEAD — the flash surface changed again "
                f"since that review: {', '.join(changed_since_review)}. Re-review and bump "
                "reviewed_up_to_sha, or drop the arch from VALIDATED_SMS."
            )
            continue
        suppressed.add(arch)

    return findings, suppressed


# --------------------------------------------------------------------------- #
# gate driver
# --------------------------------------------------------------------------- #
def run_gate(
    build_rs_path: Path,
    cuda_runs_dir: Path,
    allowlist_path: Path,
    repo_root: Path,
) -> list[str]:
    if is_shallow_repository(repo_root):
        raise ArtifactError(SHALLOW_CHECKOUT_MESSAGE)

    validated_sms = parse_validated_sms(build_rs_path)
    records, load_findings = load_artifacts(cuda_runs_dir)

    all_failures: list[str] = list(load_findings)

    candidates_by_arch, rule1_findings = check_rule1_evidence(validated_sms, records, repo_root)
    all_failures.extend(rule1_findings)

    stale = check_rule2_freshness(candidates_by_arch, repo_root)

    waivers, waiver_parse_findings = load_waivers(allowlist_path)
    all_failures.extend(waiver_parse_findings)
    rule3_findings, suppressed = check_rule3_waivers(waivers, validated_sms, stale, repo_root)
    all_failures.extend(rule3_findings)

    for arch, (newest, changed) in stale.items():
        if arch in suppressed:
            continue
        all_failures.append(_stale_message(arch, newest, changed))

    return all_failures


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()

    try:
        failures = run_gate(BUILD_RS_PATH, CUDA_RUNS_DIR, ALLOWLIST_PATH, REPO_ROOT)
    except ArtifactError as exc:
        print(f"arch-validation-freshness: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    if failures:
        print("arch-validation-freshness: FAIL", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        print(f"\narch-validation-freshness: {len(failures)} finding(s).", file=sys.stderr)
        return 1

    print(
        "arch-validation-freshness: PASS — every VALIDATED_SMS entry has GREEN, ancestor-sha "
        "evidence, and the flash surface is unchanged since each entry's newest qualifying artifact "
        "(or covered by a live, rot-free waiver)."
    )
    return 0


# --------------------------------------------------------------------------- #
# self-test — ephemeral `git init`'d fixture repos, never the real checkout,
# proving each rule above actually bites.
# --------------------------------------------------------------------------- #
BUILD_RS_TEMPLATE = """\
pub(crate) const GENCODE_ARCHES: &[&str] = &[
    "arch=compute_80,code=sm_80",
];

pub(crate) const VALIDATED_SMS: &[&str] = &[{archs}];
"""


def _write_fixture_tree(repo_root: Path, archs: list[str]) -> None:
    (repo_root / "crates" / "jammi-kernels").mkdir(parents=True, exist_ok=True)
    (repo_root / "crates" / "jammi-kernels" / "src" / "flash").mkdir(parents=True, exist_ok=True)
    (repo_root / "crates" / "jammi-kernels" / "third_party" / "flash-attention").mkdir(
        parents=True, exist_ok=True
    )
    (repo_root / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs").mkdir(parents=True, exist_ok=True)
    (repo_root / "crates" / "jammi-encoders" / "src").mkdir(parents=True, exist_ok=True)
    (repo_root / "ci" / "scripts").mkdir(parents=True, exist_ok=True)

    archs_literal = ", ".join(f'"{a}"' for a in archs)
    (repo_root / "crates" / "jammi-kernels" / "build.rs").write_text(
        BUILD_RS_TEMPLATE.format(archs=archs_literal), encoding="utf-8"
    )
    (repo_root / "crates" / "jammi-kernels" / "src" / "flash" / "mod.rs").write_text(
        "// flash kernel surface fixture\n", encoding="utf-8"
    )
    (repo_root / "crates" / "jammi-kernels" / "third_party" / "flash-attention" / "VENDORED.md").write_text(
        "# vendored fixture\n", encoding="utf-8"
    )
    (repo_root / "crates" / "jammi-kernels" / "src" / "admission.rs").write_text(
        "// admission fence fixture\n", encoding="utf-8"
    )
    (repo_root / "crates" / "jammi-encoders" / "src" / "modernbert.rs").write_text(
        "// encoder fence fixture\n", encoding="utf-8"
    )


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return _run(["git", *args], repo_root)


def _init_fixture_repo(tmp: Path, archs: list[str]) -> Path:
    repo_root = tmp
    _git(repo_root, "init", "-q", "-b", "main")
    _git(repo_root, "config", "user.email", "test@example.com")
    _git(repo_root, "config", "user.name", "Test")
    _write_fixture_tree(repo_root, archs)
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-q", "-m", "root")
    return repo_root


def _commit_all(repo_root: Path, msg: str) -> str:
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-q", "-m", msg)
    return _git(repo_root, "rev-parse", "HEAD").stdout.strip()


def _write_artifact(repo_root: Path, name: str, data: dict) -> None:
    (repo_root / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs" / name).write_text(
        json.dumps(data), encoding="utf-8"
    )


def _good_artifact(sha: str, compute_cap: str, status: str = "GREEN") -> dict:
    return {
        "schema_version": 1,
        "box": f"fixture-box (FIXTURE-GPU, driver 1.0, compute_cap {compute_cap})",
        "git_sha": sha,
        "date": "2026-08-28T13:30:00Z",
        "status": status,
    }


def _write_allowlist(repo_root: Path, rows: list[str]) -> None:
    path = repo_root / "ci" / "scripts" / "arch_validation_freshness_allowlist.txt"
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if not cond:
            failures.append(f"self-test FAILED: {name}{(' — ' + detail) if detail else ''}")

    def paths(repo_root: Path):
        return (
            repo_root / "crates" / "jammi-kernels" / "build.rs",
            repo_root / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs",
            repo_root / "ci" / "scripts" / "arch_validation_freshness_allowlist.txt",
        )

    # --- control: fresh single-arch fixture, artifact sha == HEAD ---------
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        _write_artifact(repo_root, "good.json", _good_artifact(head, "8.0"))
        head2 = _commit_all(repo_root, "add good artifact")
        # artifact sha predates the commit that added itself but not any
        # SURFACE file — still fresh, since only cuda-runs/ changed.
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check("control (fresh single-arch fixture) is clean", not got, f"{got}")

    # --- Rule 1: zero evidence at all (no artifact identifies the arch) ---
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check(
            "Rule 1: arch with zero committed artifacts",
            any("arch 80" in g and "NO qualifying evidence" in g for g in got),
            f"{got}",
        )

    # --- Rule 1: artifact exists but status != GREEN -----------------------
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        _write_artifact(repo_root, "yellow.json", _good_artifact(head, "8.0", status="YELLOW"))
        _commit_all(repo_root, "add non-green artifact")
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check(
            "Rule 1: artifact present but not GREEN is not evidence",
            any("arch 80" in g and "NO qualifying evidence" in g for g in got),
            f"{got}",
        )

    # --- Rule 1: artifact exists, GREEN, but git_sha is a REAL non-ancestor
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        _git(repo_root, "checkout", "-q", "-b", "side")
        (repo_root / "side.txt").write_text("side\n", encoding="utf-8")
        side_sha = _commit_all(repo_root, "side commit, never merged")
        _git(repo_root, "checkout", "-q", "main")
        _write_artifact(repo_root, "nonancestor.json", _good_artifact(side_sha, "8.0"))
        _commit_all(repo_root, "add non-ancestor-sha artifact")
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check(
            "Rule 1: a REAL non-ancestor git_sha is not evidence",
            any("arch 80" in g and "NO qualifying evidence" in g for g in got),
            f"{got}",
        )

    # --- Rule 1: an artifact for a DIFFERENT arch does not satisfy this one
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80", "86"])
        head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        _write_artifact(repo_root, "only80.json", _good_artifact(head, "8.0"))
        _commit_all(repo_root, "add sm80-only artifact")
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check(
            "Rule 1: sm86 has no evidence when only an sm80 artifact exists",
            any("arch 86" in g and "NO qualifying evidence" in g for g in got)
            and not any("arch 80" in g and "NO qualifying evidence" in g for g in got),
            f"{got}",
        )

    # --- Rule 2: STALE — a surface file changes AFTER the artifact's sha ---
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        _write_artifact(repo_root, "good.json", _good_artifact(head, "8.0"))
        _commit_all(repo_root, "add good artifact")
        (repo_root / "crates" / "jammi-kernels" / "src" / "flash" / "mod.rs").write_text(
            "// changed flash kernel surface\n", encoding="utf-8"
        )
        _commit_all(repo_root, "touch flash surface after artifact landed")
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check(
            "Rule 2: STALE finding names arch, artifact, changed files, both resolutions",
            any(
                "arch 80" in g
                and "STALE" in g
                and "good.json" in g
                and head in g
                and "src/flash/mod.rs" in g
                and "dropping" in g
                and "fresh per-arch pod-parity artifact" in g
                for g in got
            ),
            f"{got}",
        )

    # --- Rule 2 positive control: a non-surface change does NOT go stale ---
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        _write_artifact(repo_root, "good.json", _good_artifact(head, "8.0"))
        _commit_all(repo_root, "add good artifact")
        (repo_root / "crates" / "jammi-encoders" / "src" / "modernbert.rs").write_text(
            "// changed encoder fence, NOT flash surface\n", encoding="utf-8"
        )
        _commit_all(repo_root, "touch modernbert.rs only (excluded surface)")
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check(
            "Rule 2 control: a modernbert.rs-only change (excluded surface) stays fresh",
            not any("STALE" in g for g in got),
            f"{got}",
        )

    # --- Rule 3: valid waiver suppresses a genuine STALE finding -----------
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        _write_artifact(repo_root, "good.json", _good_artifact(head, "8.0"))
        _commit_all(repo_root, "add good artifact")
        (repo_root / "crates" / "jammi-kernels" / "src" / "flash" / "mod.rs").write_text(
            "// changed flash kernel surface\n", encoding="utf-8"
        )
        review_head = _commit_all(repo_root, "touch flash surface")
        _write_allowlist(repo_root, [f"80\t{review_head}\treviewed and accepted for a fixture reason"])
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check("Rule 3: a valid, current waiver suppresses the STALE finding", not got, f"{got}")

    # --- Rule 3: waiver's arch is not in VALIDATED_SMS (rot) ---------------
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        _write_allowlist(repo_root, [f"99\t{head}\tarch 99 was never in VALIDATED_SMS"])
        _commit_all(repo_root, "add rot waiver")
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check(
            "Rule 3: waiver for an arch not in VALIDATED_SMS is ROT",
            any("arch '99'" in g and "ROT" in g for g in got),
            f"{got}",
        )

    # --- Rule 3: dead waiver (arch not actually stale) ---------------------
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        _write_artifact(repo_root, "good.json", _good_artifact(head, "8.0"))
        artifact_sha = _commit_all(repo_root, "add good artifact")
        _write_allowlist(repo_root, [f"80\t{artifact_sha}\tnothing to waive here"])
        _commit_all(repo_root, "add dead waiver (nothing is stale)")
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check(
            "Rule 3: a waiver for an arch that is NOT stale is a DEAD WAIVER",
            any("arch '80'" in g and "DEAD WAIVER" in g for g in got),
            f"{got}",
        )

    # --- Rule 3: waiver range no longer covers HEAD (surface moved again) -
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        _write_artifact(repo_root, "good.json", _good_artifact(head, "8.0"))
        _commit_all(repo_root, "add good artifact")
        (repo_root / "crates" / "jammi-kernels" / "src" / "flash" / "mod.rs").write_text(
            "// first surface change\n", encoding="utf-8"
        )
        reviewed_sha = _commit_all(repo_root, "first surface change, reviewed here")
        _write_allowlist(repo_root, [f"80\t{reviewed_sha}\treviewed the first change only"])
        _commit_all(repo_root, "commit the waiver naming the first change")
        (repo_root / "crates" / "jammi-kernels" / "src" / "flash" / "mod.rs").write_text(
            "// second surface change, AFTER the waiver's own review point\n", encoding="utf-8"
        )
        _commit_all(repo_root, "second surface change, unreviewed")
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check(
            "Rule 3: a waiver whose range no longer matches HEAD is red, and STALE still fires",
            any("no longer covers HEAD" in g for g in got) and any("STALE" in g for g in got),
            f"{got}",
        )

    # --- Rule 3: malformed reviewed_up_to_sha (not 40-hex) -----------------
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        _write_artifact(repo_root, "good.json", _good_artifact(head, "8.0"))
        _commit_all(repo_root, "add good artifact")
        _write_allowlist(repo_root, ["80\tnot-a-real-sha\tmalformed sha fixture"])
        _commit_all(repo_root, "add malformed waiver")
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check(
            "Rule 3: a malformed reviewed_up_to_sha is caught",
            any("arch '80'" in g and "malformed reviewed_up_to_sha" in g for g in got),
            f"{got}",
        )

    # --- malformed allowlist row (wrong field count) ------------------------
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        _write_artifact(repo_root, "good.json", _good_artifact(head, "8.0"))
        _commit_all(repo_root, "add good artifact")
        _write_allowlist(repo_root, ["80\tmissing-the-reason-field"])
        _commit_all(repo_root, "add malformed-row waiver")
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check(
            "Rule 3: a malformed (wrong field count) allowlist row is caught",
            any("malformed row" in g for g in got),
            f"{got}",
        )

    # --- VALIDATED_SMS literal not found ------------------------------------
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        (repo_root / "crates" / "jammi-kernels" / "build.rs").write_text(
            "// no VALIDATED_SMS const here at all\n", encoding="utf-8"
        )
        _commit_all(repo_root, "remove VALIDATED_SMS")
        build_rs, cuda_runs, allowlist = paths(repo_root)
        try:
            run_gate(build_rs, cuda_runs, allowlist, repo_root)
            failures.append("self-test FAILED: missing VALIDATED_SMS literal did not raise ArtifactError")
        except ArtifactError as exc:
            check("VALIDATED_SMS literal missing raises a named ArtifactError", "VALIDATED_SMS" in str(exc))

    # --- unparsable artifact JSON is a named finding, never a crash --------
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        (
            repo_root / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs" / "bad.json"
        ).write_text("{not valid json", encoding="utf-8")
        _commit_all(repo_root, "add unparsable artifact json")
        build_rs, cuda_runs, allowlist = paths(repo_root)
        got = run_gate(build_rs, cuda_runs, allowlist, repo_root)
        check(
            "an unparsable artifact JSON produces a named finding, not a crash",
            any("bad.json" in g and "JSON parse error" in g for g in got),
            f"{got}",
        )

    # --- shallow-checkout guard ---------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        repo_root = _init_fixture_repo(Path(td), ["80"])
        head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        _write_artifact(repo_root, "good.json", _good_artifact(head, "8.0"))
        _commit_all(repo_root, "add good artifact")
        with tempfile.TemporaryDirectory() as td2:
            clone_dir = Path(td2) / "shallow-clone"
            clone_proc = _run(
                ["git", "clone", "-q", "--depth", "1", "file://" + str(repo_root), str(clone_dir)],
                Path(td2),
            )
            if clone_proc.returncode != 0:
                failures.append(
                    f"self-test FAILED: could not create a --depth 1 clone fixture: {clone_proc.stderr}"
                )
            else:
                build_rs, cuda_runs, allowlist = paths(clone_dir)
                try:
                    run_gate(build_rs, cuda_runs, allowlist, clone_dir)
                    failures.append("self-test FAILED: run_gate did not raise on a shallow checkout")
                except ArtifactError as exc:
                    if str(exc) != SHALLOW_CHECKOUT_MESSAGE:
                        failures.append(
                            f"self-test FAILED: shallow-checkout ArtifactError had the wrong message: {exc}"
                        )

    if failures:
        print("arch-validation-freshness self-test: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        "arch-validation-freshness self-test: OK — every rule bites: Rule 1 (zero evidence, "
        "non-GREEN, non-ancestor sha, wrong-arch evidence), Rule 2 (STALE on a real surface change, "
        "a positive control that an excluded-surface change stays fresh), Rule 3 (valid waiver "
        "suppression, rot for an unknown arch, a dead waiver, a range that no longer covers HEAD, a "
        "malformed sha, a malformed row), a missing VALIDATED_SMS literal (named ArtifactError), an "
        "unparsable artifact JSON (named finding, never a crash), and the shallow-checkout guard."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
