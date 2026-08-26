#!/usr/bin/env python3
"""CUDA-run-artifact schema + provenance gate — hermetic, static, no build, no GPU.

## The escape this closes (census, `.jammi/escapes.jsonl` row 215)

Nothing under `crates/jammi-kernels/artifacts/cuda-runs/` was read by any CI
script — the directory's own README claims "the file is the evidence, the
commit message is a pointer to it", but nothing ever checked the evidence was
shaped like evidence. Three concrete defects that absence hid:

  1. **4 of the 5 committed artifacts name a `git_sha` that is an ancestor of
     NO ref on this branch** (`5932520`, `e32ed90`, `eac63fd`, `80f02fb` — the
     branches that produced them were squash-merged, so the exact commit each
     artifact proved never survives in `HEAD`'s history). A green artifact
     whose sha is not an ancestor of the branch is evidence about the
     ORACLE, not the code (`docs/maintainer/cuda-kernel-guide.md` §4) — the
     artifact is proving a tree that no longer exists.
  2. **Five mutually incompatible top-level schemas coexist** across this
     repo's cuda-run artifacts (`git_sha` / `tip_sha` full / `tip_sha` sha7 /
     `deliverable_git_sha` / no build-ref field at all), so no script could
     have reconciled them even if one had tried.
  3. **Raw-run files under a `*-raw-runs/` subdirectory carry no build ref at
     all** — they are the per-leg raw `jammi-bench` JSON a parent artifact
     folds numbers from, and inherit nothing from that parent mechanically.

The fix is not "add a check for those four known-bad shas" (a grep for one
known-bad string is exactly the anti-pattern `check_doc_parity.py`'s own
docstring warns against, and `check_gpu_parity_matrix.py`'s docstring
restates for the device axis); it is a *schema* every artifact must satisfy,
checked by a script that recurses into `*-raw-runs/` too, so a newly
committed artifact with an unproven sha, an unresolvable producer, or a
silently-defaulted "no producer" cannot land unnoticed.

## The schema (documented for humans in `cuda-runs/README.md`)

Every `*.json` under `cuda-runs/` (including every `*-raw-runs/` leg file)
must carry:

  - `schema_version` (int)
  - `git_sha` (40-hex, lowercase) — OR, for a reviewed legacy artifact only,
    `git_sha_unresolved` (whatever short/malformed ref the pod session saw)
    paired with `producer.kind == "none"`.
  - `box` (string — the physical/pod identifier the run measured on)
  - `producer` — `{path, kind, invocation, gating}`:
      - `kind`: `"cargo-test"` (a single `#[test] fn`, invoked with
        `--exact <fn>`) | `"script"` (a tracked producer script) | `"none"`
        (reviewed legacy artifact — see `LEGACY_NONE_ALLOWLIST` below; a NEW
        file may never default to this).
      - `gating`: `"#[ignore]"` | `"env:<VAR>"` | `"required-features"` |
        `"none"` — how the named test/script stays out of a plain
        `cargo test`/CI run (mirrors `docs/maintainer/cuda-kernel-guide.md`
        §5's "no CI lane has a GPU" constraint).
  - `status` (string)
  - `merged_as` (40-hex, OPTIONAL) + `merged_via_pr` (int, OPTIONAL, required
    together with `merged_as`) — for a measured tip that was itself
    squash-merged (so `git_sha` is legitimately never an ancestor of
    anything again): the squash commit the SAME content landed on `main`
    as, and the PR that merged it. `git_sha` is kept VERBATIM (the tip that
    was actually measured); `merged_as` only ever supplements it, never
    replaces it, and is only valid alongside a resolved `git_sha` (never
    `git_sha_unresolved`).

## Fail-closed contract

  (a) Every required field is present and well-typed (including the
      `git_sha` XOR `git_sha_unresolved` split, the `git_sha_unresolved`
      ⇒ `producer.kind == "none"` consistency rule, and the
      `merged_as`/`merged_via_pr` pairing — `merged_as` requires
      `merged_via_pr` and a resolved `git_sha`, and vice versa).
  (b) `producer.path`, when non-null, exists on disk AND is `git
      ls-files`-tracked (an artifact cannot cite a producer CI's own
      checkout would not have).
  (c) `producer.kind == "cargo-test"` ⇒ the invocation names `--exact
      <fn>`, that `fn` is found by a static brace-balanced scan of
      `producer.path`, it is confirmed to sit under a `#[test]` attribute,
      and the STATED `gating` attribute genuinely appears there
      (`#[ignore]` immediately above the fn; the named env var — or the
      `cuda_device()` helper — inside the fn body; or a `required-features`
      key on the matching `[[test]]` section of the crate's `Cargo.toml`).
  (d) PASS if `git merge-base --is-ancestor <git_sha> HEAD`, OR — for a
      squash-merged tip — if `merged_as` is ALSO an ancestor of HEAD (with
      `merged_via_pr` present). Neither ancestor is a hard FAIL naming the
      guide's own sentence (never silently accepted as "was green once").
  (e) `cuda-runs/README.md`'s named producer script is itself
      `git ls-files`-tracked (the README cannot point at a file CI's
      checkout would not have either).
  (f) `producer.kind == "none"` is allowed ONLY for a path in the reviewed,
      in-script `LEGACY_NONE_ALLOWLIST` — a NEW artifact defaulting to
      `"none"` is a hard FAIL.
  (g) KO-3 (`docs/maintainer/cuda-kernel-guide.md` §3, an instance of §3.8):
      an OPTIONAL `oracle_separation: {healthy_max_offsample, bound,
      min_control}` block, attached to ANY leg anywhere in the artifact
      (found by recursing the whole document, not a fixed top-level key —
      see `check_oracle_separation`'s own doc), asserts
      `healthy_max_offsample < bound < min_control` when present. Absent
      entirely on every artifact committed before this rule existed, so no
      existing artifact reddens.

      NOTE: a SECOND, UNRELATED rule below (`check_v2_leg`/`find_v2_legs`,
      unification contract C6 — v2 leg identity) is ALSO commented "rule
      (g)" in this file. Both landed calling themselves "(g)" independently
      (KO-3 first; C6 second, without checking). Deliberately left
      un-renumbered rather than risk a mechanical rename introducing a typo
      across two unrelated mechanisms: the letter is read by NO gate,
      allowlist, or error message anywhere in this file or its self-test
      (`git grep` confirms) — it is comment/self-test-label prose only, so
      the collision is cosmetic, never a correctness hazard. If a future
      change makes some gate or doc actually READ the letter (e.g. an error
      message citing "rule (g)"), split them into distinct letters then.

Rule (d) needs REAL commit history to mean anything: `git merge-base
--is-ancestor` on a shallow checkout (`actions/checkout`'s default
`fetch-depth: 1`) reads back EVERY `git_sha` as a false non-ancestor —
indistinguishable from a genuine one without checking first. Before any
per-file work, `run_gate` calls `git rev-parse --is-shallow-repository` and,
if shallow, raises ONE explicit failure ("shallow checkout — ancestry cannot
be evaluated; use fetch-depth: 0") instead of N misleading per-file findings
that would look like real drift. `.github/workflows/ci.yml`'s `guard` job
gives ONLY this matrix leg `fetch-depth: 0` (a full clone; this repository's
`.git` is small — see the PR that added this check for the measured size —
negligible next to the Rust build jobs elsewhere in this workflow); every
other leg stays at the normal shallow default.

Run: `python3 ci/scripts/check_cuda_run_artifacts.py`
Self-test (RED cases for every rule above, on a throwaway `git init`'d
fixture repo — never the real checkout):
`python3 ci/scripts/check_cuda_run_artifacts.py --self-test`
Hermetic: reads the working tree (or an ephemeral tempdir git repo under
`--self-test`) and shells out only to `git`; no network, no cargo, no GPU.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA_RUNS_DIR = REPO_ROOT / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs"
README_PATH = CUDA_RUNS_DIR / "README.md"

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PRODUCER_KINDS = {"cargo-test", "script", "none"}
GATING_STATIC = {"#[ignore]", "required-features", "none"}
GATING_ENV_RE = re.compile(r"^env:[A-Za-z_][A-Za-z0-9_]*$")
README_PRODUCER_RE = re.compile(r"`([\w./-]*proof_artifact\.py)`")

# --------------------------------------------------------------------------- #
# LEGACY_NONE_ALLOWLIST — reviewed at gate-introduction time. Every artifact
# committed to `main` BEFORE this schema existed has no `producer` field to
# migrate honestly, so each is grandfathered here with a one-line reason. A
# NEW artifact is never added to this list; it must name a real producer.
# --------------------------------------------------------------------------- #
LEGACY_NONE_ALLOWLIST: dict[str, str] = {
    "2026-08-25-cast-w1-80f02fb-a100-sxm4.json": (
        "pre-schema artifact: the same-build forced-arm A/B was run by hand via "
        "jammi-bench finetune-step invocations on the lead's exclusive box, not a "
        "single #[test] fn or a tracked producer script."
    ),
    "2026-08-25-p1-5f29e3b-a100-sxm4.json": (
        "pre-schema artifact: produced by an earlier, untracked copy of "
        "proof_artifact.py, before this gate's PR tracked it at "
        "ci/scripts/perf/proof_artifact.py."
    ),
    "2026-08-25-p2-5932520-a100-sxm4.json": (
        "pre-schema artifact: same untracked-proof_artifact.py provenance as the "
        "p1 sibling artifact above."
    ),
    "2026-08-25-p3-e32ed90-a100-sxm4.json": (
        "pre-schema artifact: same untracked-proof_artifact.py provenance as the "
        "p1 sibling artifact above."
    ),
    "2026-08-25-p6a-eac63fd-a100-sxm4.json": (
        "pre-schema artifact: a flash_smoke execution-provenance dump captured by "
        "hand (16 test names from one binary run, not a single #[test] fn)."
    ),
    # Unification contract C8.3 (phase 2, B6): the two `crates/jammi-bench/
    # baselines/*.json` records `git mv`d under this directory. Neither has a
    # tracked producer script (both were hand-driven `jammi-bench finetune-step`
    # invocations run on the lead's own box — see each file's own `_comment`),
    # so `producer.kind == "none"` is the honest reading, same as the five
    # entries above. Growth of this list is no longer purely a human review
    # call: `check_none_allowlist_history` (rule (f)'s mechanical companion)
    # requires every entry's FIRST INTRODUCTION commit (`git log --follow
    # --diff-filter=A`) to be an ancestor of this gate's own introduction
    # (`c7fd1df`, GATE_INTRODUCTION_SHA below) — both of these predate it
    # (`3719bc8`, `5879c48`), so a genuinely NEW artifact can never satisfy the
    # condition and this list cannot grow again without a gate edit AND a
    # history it does not have.
    "2026-08-24-finetune-step-reference-d361515-a100-pcie.json": (
        "pre-schema baseline, moved from crates/jammi-bench/baselines/"
        "finetune_step_reference.json (unification contract C8): a same-box "
        "A/B reference run by hand via jammi-bench finetune-step invocations, "
        "not a single #[test] fn or a tracked producer script — see this "
        "file's own _comment."
    ),
    "2026-08-24-p1-softmax-fold-bf8e807-a100-sxm4.json": (
        "pre-schema baseline, moved from crates/jammi-bench/baselines/"
        "p1_softmax_scale_fold_ab.json (unification contract C8): a same-box "
        "A/B run by hand proving the P1 softmax-scale-fold change, not a "
        "single #[test] fn or a tracked producer script — see this file's "
        "own _comment."
    ),
}

# Unification contract C8.3: the commit that introduced THIS gate (schema +
# ancestry + producer-provenance for cuda-run artifacts, #379). Every
# `LEGACY_NONE_ALLOWLIST` entry's own first-introduction commit must be an
# ancestor of this one — see `check_none_allowlist_history`.
GATE_INTRODUCTION_SHA = "c7fd1df58b81761374431597d6de414a863f0f83"

ANCESTOR_MESSAGE = (
    "is not an ancestor of HEAD — a green artifact whose sha is not an ancestor "
    "of the branch is evidence about the ORACLE, not the code "
    "(docs/maintainer/cuda-kernel-guide.md §4)."
)


class ArtifactError(Exception):
    """Uncomputable input (parse failure, missing dir) — fails closed."""


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def git_ls_files(repo_root: Path) -> set[str]:
    proc = _run(["git", "ls-files"], repo_root)
    if proc.returncode != 0:
        raise ArtifactError(f"`git ls-files` failed in {repo_root}: {proc.stderr.strip()}")
    return set(proc.stdout.splitlines())


SHALLOW_CHECKOUT_MESSAGE = "shallow checkout — ancestry cannot be evaluated; use fetch-depth: 0"


def is_shallow_repository(repo_root: Path) -> bool:
    """`actions/checkout`'s default (`fetch-depth: 1`) hands `git merge-base
    --is-ancestor` a truncated object graph — every `git_sha` this gate has
    ever seen reads back as a false non-ancestor in that state, which is
    indistinguishable from a REAL non-ancestor without this check (the exact
    failure mode that made even `p1`'s genuinely-ancestor `git_sha` FAIL in
    CI: `.github/workflows/ci.yml`'s `guard` job checked out at the default
    depth). `git rev-parse --is-shallow-repository` is the one command that
    tells the two apart; a non-zero exit (e.g. run outside a git repo at
    all) is treated as "not shallow" here — `git_ls_files`/`_is_ancestor`
    will raise their own, more specific errors moments later if the
    checkout is unusable for some other reason.
    """
    proc = _run(["git", "rev-parse", "--is-shallow-repository"], repo_root)
    return proc.returncode == 0 and proc.stdout.strip() == "true"


# --------------------------------------------------------------------------- #
# rule (a) — schema/typing
# --------------------------------------------------------------------------- #
def check_schema_types(data: dict) -> list[str]:
    failures: list[str] = []

    sv = data.get("schema_version")
    if not isinstance(sv, int) or isinstance(sv, bool):
        failures.append(f"schema_version must be an int, got {sv!r}")

    box = data.get("box")
    if not isinstance(box, str) or not box.strip():
        failures.append(f"box must be a non-empty string, got {box!r}")

    status = data.get("status")
    if not isinstance(status, str) or not status.strip():
        failures.append(f"status must be a non-empty string, got {status!r}")

    has_sha = "git_sha" in data
    has_unresolved = "git_sha_unresolved" in data
    if has_sha and has_unresolved:
        failures.append("carries BOTH git_sha and git_sha_unresolved — pick one")
    elif not has_sha and not has_unresolved:
        failures.append("missing git_sha (and no git_sha_unresolved fallback)")
    elif has_sha:
        sha = data["git_sha"]
        if not isinstance(sha, str) or not GIT_SHA_RE.match(sha):
            failures.append(f"git_sha must be 40 lowercase hex chars, got {sha!r}")
    else:
        unresolved = data["git_sha_unresolved"]
        if not isinstance(unresolved, str) or not unresolved.strip():
            failures.append(f"git_sha_unresolved must be a non-empty string, got {unresolved!r}")

    # `merged_as` / `merged_via_pr` — the optional squash-landing pair. A
    # branch tip a measurement ran on can be squash-merged, so `git_sha`
    # itself is legitimately never an ancestor of any ref again; `merged_as`
    # names the squash commit the SAME content landed on `main` as, kept
    # alongside (never instead of) the measured `git_sha`.
    has_merged_as = "merged_as" in data
    has_merged_via_pr = "merged_via_pr" in data
    if has_merged_as:
        merged_as = data["merged_as"]
        if not isinstance(merged_as, str) or not GIT_SHA_RE.match(merged_as):
            failures.append(f"merged_as must be 40 lowercase hex chars, got {merged_as!r}")
        if not has_merged_via_pr:
            failures.append("merged_as is present but merged_via_pr is missing")
        if not has_sha:
            failures.append(
                "merged_as requires git_sha (the measured tip, kept verbatim) — not valid "
                "alongside git_sha_unresolved"
            )
    if has_merged_via_pr:
        merged_via_pr = data["merged_via_pr"]
        if not isinstance(merged_via_pr, int) or isinstance(merged_via_pr, bool):
            failures.append(f"merged_via_pr must be an int, got {merged_via_pr!r}")
        if not has_merged_as:
            failures.append("merged_via_pr is present but merged_as is missing")

    producer = data.get("producer")
    if not isinstance(producer, dict):
        failures.append(f"producer must be an object, got {producer!r}")
        producer = {}
    else:
        for key in ("path", "kind", "invocation", "gating"):
            if key not in producer:
                failures.append(f"producer missing required key `{key}`")
        kind = producer.get("kind")
        if kind is not None and kind not in PRODUCER_KINDS:
            failures.append(f"producer.kind must be one of {sorted(PRODUCER_KINDS)}, got {kind!r}")
        gating = producer.get("gating")
        if gating is not None and gating not in GATING_STATIC and not (
            isinstance(gating, str) and GATING_ENV_RE.match(gating)
        ):
            failures.append(
                f"producer.gating must be '#[ignore]' | 'required-features' | 'none' | 'env:<VAR>', got {gating!r}"
            )
        for key in ("path", "invocation"):
            val = producer.get(key)
            if val is not None and not isinstance(val, str):
                failures.append(f"producer.{key} must be a string or null, got {val!r}")

    if has_unresolved and isinstance(producer, dict) and producer.get("kind") not in (None, "none"):
        failures.append(
            "git_sha_unresolved requires producer.kind == 'none' (an unresolvable sha cannot be "
            "attributed to a real producer)"
        )

    return failures


# --------------------------------------------------------------------------- #
# rule (b) — producer.path exists and is tracked
# --------------------------------------------------------------------------- #
def check_producer_path(producer: dict, repo_root: Path, tracked: set[str]) -> list[str]:
    path = producer.get("path")
    if not isinstance(path, str) or not path:
        return []
    failures: list[str] = []
    if not (repo_root / path).is_file():
        failures.append(f"producer.path `{path}` does not exist on disk")
    if path not in tracked:
        failures.append(f"producer.path `{path}` is not `git ls-files`-tracked")
    return failures


# --------------------------------------------------------------------------- #
# rule (c) — cargo-test producer static verification
# --------------------------------------------------------------------------- #
EXACT_RE = re.compile(r"--exact\s+(\S+)")


def _extract_fn_body(source: str, fn_kw_start: int) -> str:
    brace_start = source.find("{", fn_kw_start)
    if brace_start == -1:
        return ""
    depth = 0
    for i in range(brace_start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[brace_start : i + 1]
    return source[brace_start:]


def _find_crate_root(repo_root: Path, rel_path: str) -> Path | None:
    p = (repo_root / rel_path).parent
    while True:
        if (p / "Cargo.toml").is_file():
            return p
        if p == repo_root or p.parent == p:
            return None
        p = p.parent


def _test_target_has_required_features(cargo_toml_text: str, test_stem: str) -> bool:
    blocks = re.split(r"(?m)^\[\[test\]\]\s*$", cargo_toml_text)[1:]
    for block in blocks:
        end = re.search(r"(?m)^\[", block)
        body = block[: end.start()] if end else block
        name_m = re.search(r'name\s*=\s*"([^"]+)"', body)
        if name_m and name_m.group(1) == test_stem:
            return "required-features" in body
    return False


def check_cargo_test_gating(data: dict, producer: dict, repo_root: Path) -> list[str]:
    path = producer.get("path")
    invocation = producer.get("invocation") or ""
    gating = producer.get("gating")
    if not isinstance(path, str) or not path:
        return ["producer.kind == 'cargo-test' requires a non-null producer.path"]
    if not (repo_root / path).is_file():
        return []  # rule (b) already reported the missing-file failure

    m = EXACT_RE.search(invocation)
    if not m:
        return [
            f"producer.kind == 'cargo-test' but invocation `{invocation}` lacks `--exact <fn_name>` "
            "— cannot statically verify which test this artifact proves ran"
        ]
    fn_full = m.group(1)
    fn_short = fn_full.rsplit("::", 1)[-1]

    source = (repo_root / path).read_text(encoding="utf-8", errors="replace")
    fn_re = re.compile(rf"\bfn\s+{re.escape(fn_short)}\s*\(")
    fn_m = fn_re.search(source)
    if not fn_m:
        return [f"named test fn `{fn_short}` not found by static scan in {path}"]

    fn_line_idx = source.count("\n", 0, fn_m.start())
    lines = source.splitlines()
    # Walk upward from the `fn` line while the line is blank, an attribute
    # (`#[...]`), or a doc/line comment — stop at the first line that is
    # none of those (e.g. the closing `}` of a PRECEDING, unrelated fn), so
    # a neighbour's `#[ignore]` a few lines up can never be mistaken for
    # this fn's own gating attribute.
    window_lines: list[str] = []
    i = fn_line_idx - 1
    while i >= 0:
        stripped = lines[i].strip()
        if stripped == "" or stripped.startswith("#[") or stripped.startswith("//"):
            window_lines.insert(0, lines[i])
            i -= 1
        else:
            break
    window_text = "\n".join(window_lines)

    failures: list[str] = []
    if "#[test]" not in window_text:
        failures.append(
            f"`{fn_short}` in {path} has no `#[test]` attribute in its contiguous "
            "attribute block — not confirmed to be a #[test] fn"
        )

    if gating == "#[ignore]":
        if "#[ignore]" not in window_text:
            failures.append(
                f"producer claims gating '#[ignore]' but `{fn_short}` in {path} has no "
                "#[ignore] attribute"
            )
    elif isinstance(gating, str) and gating.startswith("env:"):
        var = gating.split(":", 1)[1]
        body = _extract_fn_body(source, fn_m.start())
        if var not in body and "cuda_device(" not in body:
            failures.append(
                f"producer claims gating '{gating}' but neither `{var}` nor `cuda_device(` "
                f"appears in `{fn_short}`'s body in {path}"
            )
    elif gating == "required-features":
        crate_root = _find_crate_root(repo_root, path)
        if crate_root is None:
            failures.append(f"no Cargo.toml found above {path}; cannot verify required-features")
        else:
            test_stem = Path(path).stem
            cargo_toml = (crate_root / "Cargo.toml").read_text(encoding="utf-8", errors="replace")
            if not _test_target_has_required_features(cargo_toml, test_stem):
                failures.append(
                    f"producer claims gating 'required-features' but {crate_root / 'Cargo.toml'} "
                    f"has no `[[test]]` section named `{test_stem}` carrying `required-features`"
                )
    # gating == "none": nothing further to verify.

    return failures


# --------------------------------------------------------------------------- #
# rule (d) — ancestry
# --------------------------------------------------------------------------- #
def _is_ancestor(sha: str, repo_root: Path, target: str = "HEAD") -> bool:
    proc = _run(["git", "merge-base", "--is-ancestor", sha, target], repo_root)
    return proc.returncode == 0


def check_ancestry(data: dict, repo_root: Path) -> list[str]:
    """PASS if `git_sha` is an ancestor of HEAD, OR — for a branch tip that
    was squash-merged, so `git_sha` itself can never be an ancestor of
    anything again — if `merged_as` is an ancestor of HEAD and the artifact
    also carries `git_sha` (the measured tip, kept verbatim) plus
    `merged_via_pr`. `merged_as`'s own well-typedness (40-hex, paired with
    `merged_via_pr`, only valid alongside a resolved `git_sha`) is rule (a)'s
    job (`check_schema_types`); this function only re-checks GIT_SHA_RE here
    so a malformed `merged_as` cannot be handed to `git merge-base` as a
    literal ref expression.
    """
    sha = data.get("git_sha")
    if not sha:
        return []  # git_sha_unresolved artifacts have nothing resolvable to check
    if _is_ancestor(sha, repo_root):
        return []

    merged_as = data.get("merged_as")
    merged_via_pr = data.get("merged_via_pr")
    if isinstance(merged_as, str) and GIT_SHA_RE.match(merged_as) and merged_via_pr is not None:
        if _is_ancestor(merged_as, repo_root):
            return []
        return [
            f"git_sha {sha} {ANCESTOR_MESSAGE} merged_as {merged_as} (PR #{merged_via_pr}) is "
            "ALSO not an ancestor of HEAD — the squash-landing claim does not hold either."
        ]

    return [f"git_sha {sha} {ANCESTOR_MESSAGE}"]


# --------------------------------------------------------------------------- #
# rule (e) — README's named producer is tracked
# --------------------------------------------------------------------------- #
def check_readme_producer(readme_path: Path, repo_root: Path, tracked: set[str]) -> list[str]:
    if not readme_path.is_file():
        return [f"README not found: {readme_path}"]
    text = readme_path.read_text(encoding="utf-8", errors="replace")
    m = README_PRODUCER_RE.search(text)
    if not m:
        return [
            f"{readme_path}: does not name a `...proof_artifact.py` producer in backticks — "
            "the schema doc's own producer citation is missing"
        ]
    named = m.group(1)
    if named not in tracked:
        return [f"{readme_path}: names producer `{named}` which is not `git ls-files`-tracked"]
    return []


# --------------------------------------------------------------------------- #
# rule (f) — kind == "none" only for the reviewed allow-list
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# rule (g) — KO-3: `oracle_separation` (OPTIONAL per artifact leg)
# --------------------------------------------------------------------------- #
# `docs/maintainer/cuda-kernel-guide.md` §3's `KO-3` id (an instance of
# §3.8's "no absolute ULP floor" discipline): a bound is not evidence of
# real separation just because it PASSES today — an artifact MAY attach an
# `oracle_separation: {healthy_max_offsample, bound, min_control}` block to
# any leg (any nested object anywhere in the artifact JSON, not a fixed
# top-level key — this repo's artifacts already carry ad hoc named legs,
# e.g. `cuda_parity_adamw_legs`/`optimizer_phase_wall_time_ms`, so "per
# leg" means "wherever a leg object chooses to carry it", found by
# recursing the whole document rather than assuming one fixed shape) to
# DEMONSTRATE, numerically, that the chosen bound sits strictly between the
# healthiest off-sample measurement and the smallest value a real control/
# regression would produce: `healthy_max_offsample < bound < min_control`.
# OPTIONAL in v1 — absent entirely on every artifact committed before this
# rule existed, so no existing artifact reddens; where present, it is
# checked.
ORACLE_SEPARATION_KEY = "oracle_separation"
ORACLE_SEPARATION_FIELDS = ("healthy_max_offsample", "bound", "min_control")


def _walk_oracle_separation_blocks(data, path: str = "$"):
    """Yields (json_path, block_dict) for every dict anywhere in `data`
    (recursing through nested dicts and lists) that itself carries an
    `oracle_separation` key — the "per leg, wherever a leg carries it"
    search the module doc above describes.
    """
    if isinstance(data, dict):
        if ORACLE_SEPARATION_KEY in data:
            yield f"{path}.{ORACLE_SEPARATION_KEY}", data[ORACLE_SEPARATION_KEY]
        for key, value in data.items():
            yield from _walk_oracle_separation_blocks(value, f"{path}.{key}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            yield from _walk_oracle_separation_blocks(item, f"{path}[{i}]")


def check_oracle_separation(data: dict) -> list[str]:
    failures: list[str] = []
    for json_path, block in _walk_oracle_separation_blocks(data):
        if not isinstance(block, dict):
            failures.append(f"{json_path} must be an object, got {block!r}")
            continue
        missing = [f for f in ORACLE_SEPARATION_FIELDS if f not in block]
        if missing:
            failures.append(f"{json_path} missing required field(s): {', '.join(missing)}")
            continue
        values: dict[str, float] = {}
        bad_type = False
        for f in ORACLE_SEPARATION_FIELDS:
            v = block[f]
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                failures.append(f"{json_path}.{f} must be a number, got {v!r}")
                bad_type = True
            else:
                values[f] = float(v)
        if bad_type:
            continue
        healthy = values["healthy_max_offsample"]
        bound = values["bound"]
        min_control = values["min_control"]
        if not (healthy < bound < min_control):
            failures.append(
                f"{json_path}: healthy_max_offsample ({healthy}) < bound ({bound}) < "
                f"min_control ({min_control}) does not hold — the bound does not "
                "demonstrably separate healthy noise from a real control/regression"
            )
    return failures


def check_none_allowlist(data: dict, relpath: str, allowlist: dict[str, str]) -> list[str]:
    producer = data.get("producer")
    if isinstance(producer, dict) and producer.get("kind") == "none" and relpath not in allowlist:
        return [
            f"producer.kind == 'none' but `{relpath}` is not in the reviewed LEGACY_NONE_ALLOWLIST — "
            "a NEW artifact must name a real producer (cargo-test or script), never default to 'none'"
        ]
    return []


def _first_introduction_sha_for_path(path: Path, repo_root: Path) -> str | None:
    """The oldest commit that `git log --follow --diff-filter=A` reports for
    `path` — the commit that FIRST added this exact file path (following
    renames). `None` if `path` does not resolve to somewhere inside
    `repo_root`, or git found no such commit (the file was never added
    under this name, or the repo has no history for it)."""
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None
    proc = _run(["git", "log", "--follow", "--diff-filter=A", "--format=%H", "--", str(rel)], repo_root)
    if proc.returncode != 0:
        return None
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _first_introduction_sha(relpath: str, cuda_runs_dir: Path, repo_root: Path) -> str | None:
    """`_first_introduction_sha_for_path` for a `LEGACY_NONE_ALLOWLIST`
    entry, resolved relative to `cuda_runs_dir` rather than `repo_root`
    directly."""
    return _first_introduction_sha_for_path(cuda_runs_dir / relpath, repo_root)


def check_gate_introduction_sha_anchor(
    repo_root: Path = REPO_ROOT,
    gate_file: Path | None = None,
    gate_introduction_sha: str = GATE_INTRODUCTION_SHA,
) -> list[str]:
    """C8.3's own anchor, otherwise unverified: `GATE_INTRODUCTION_SHA` is a
    hand-typed constant that every `LEGACY_NONE_ALLOWLIST` entry's history
    check (`check_none_allowlist_history`) is pinned to — "this entry's
    first-introduction commit must be an ancestor of GATE_INTRODUCTION_SHA".
    If that constant is silently repointed FORWARD (to a commit strictly
    AFTER this gate's real introduction), that check quietly widens what
    "predates the gate" means and a genuinely NEW allowlist entry could
    start satisfying it. This function is the anchor: it asserts
    `GATE_INTRODUCTION_SHA` equals THIS GATE FILE's OWN first-introduction
    commit (`_first_introduction_sha_for_path`, the same `git log --follow
    --diff-filter=A` machinery `check_none_allowlist_history` already
    trusts) — never a second, independently-drifting source of truth.
    Editing `GATE_INTRODUCTION_SHA` at all is a `ci/scripts/check_cuda_run_
    artifacts.py` edit, i.e. SWARM_GATE_TOUCHED by construction, same as
    every other change to this file's own rules."""
    gate_file = gate_file if gate_file is not None else Path(__file__).resolve()
    actual = _first_introduction_sha_for_path(gate_file, repo_root)
    if actual is None:
        return [
            f"GATE_INTRODUCTION_SHA anchor: could not determine {gate_file}'s own first-introduction "
            "commit via `git log --follow --diff-filter=A` — cannot verify GATE_INTRODUCTION_SHA"
        ]
    if actual != gate_introduction_sha:
        return [
            f"GATE_INTRODUCTION_SHA = {gate_introduction_sha} does not match this gate file's own "
            f"first-introduction commit ({actual}) — every LEGACY_NONE_ALLOWLIST entry's history "
            "check is anchored to this constant; a value that is not the gate's own real "
            "introduction silently changes what 'predates the gate' means"
        ]
    return []


def check_none_allowlist_history(
    relpath: str,
    cuda_runs_dir: Path,
    repo_root: Path,
    gate_introduction_sha: str = GATE_INTRODUCTION_SHA,
) -> list[str]:
    """Unification contract C8.3 — rule (f)'s mechanical companion: an entry
    in `LEGACY_NONE_ALLOWLIST` is legitimate ONLY if the artifact it names
    was first committed (under this exact path, following renames) BEFORE
    this gate itself existed. A genuinely NEW artifact's first-introduction
    commit can never predate `gate_introduction_sha` (the gate did not exist
    yet when a truly pre-schema artifact was added, but it DOES exist by the
    time any new commit lands), so this list cannot grow again without a
    gate edit — which is SWARM_GATE_TOUCHED by construction — AND a history
    it does not have.
    """
    intro = _first_introduction_sha(relpath, cuda_runs_dir, repo_root)
    if intro is None:
        return [
            f"LEGACY_NONE_ALLOWLIST entry `{relpath}`: could not determine its first-introduction "
            f"commit via `git log --follow --diff-filter=A` — cannot verify it predates the gate "
            f"({gate_introduction_sha})"
        ]
    if not _is_ancestor(intro, repo_root, gate_introduction_sha):
        return [
            f"LEGACY_NONE_ALLOWLIST entry `{relpath}`: its first-introduction commit {intro} is NOT "
            f"an ancestor of this gate's own introduction ({gate_introduction_sha}) — a genuinely NEW "
            f"artifact can never satisfy this condition; LEGACY_NONE_ALLOWLIST cannot grow for it"
        ]
    return []


# --------------------------------------------------------------------------- #
# rule (g) — leg identity on self-declaring v2 legs (unification contract C6,
# phase 2). NOTE: this is a SEPARATE, unrelated mechanism from the OTHER
# "rule (g)" above (KO-3 `oracle_separation`) — see that rule's own note for
# why the collision is left as-is (cosmetic; the letter is read by no gate,
# allowlist, or error message).
#
# A v2 leg is ANY JSON object, anywhere in a `cuda-runs/**` tree,
# carrying `leg_schema_version >= 2` — a key with ZERO occurrences anywhere
# in this repo at the time this rule was written, so no pre-existing (v1)
# leg can satisfy it by accident. A leg WITHOUT the key is v1 and is
# validated only by rules (a)-(f) above, exactly as before this rule existed.
#
# The required identity TUPLE per (tier, producer_kind) is never hand-typed
# here: the jammi side is extracted by regex from `FinetuneStepTier::
# IDENTITY_FIELDS` + `REPORT_IDENTITY_FIELDS` in `crates/jammi-bench/src/
# report.rs` (the SAME const `ci/scripts/perf/test_identity_fields_subset.py`
# reads, contract C4.2); the torch side is IMPORTED directly from
# `crates/jammi-bench/reference/torch_finetune_step.py`'s own
# `TORCH_IDENTITY_FIELDS` / `TORCH_IDENTITY_FIELDS_NULL_MEANS` (contract
# C3.5) and `ci/scripts/perf/ab_merge.py`'s own `_TORCH_ARGS_LEVEL_FIELDS`
# (the field-placement map the existing jammi-vs-torch comparator already
# depends on) — never re-typed as a second, independently-drifting copy.
# --------------------------------------------------------------------------- #
LEG_SCHEMA_VERSION_KEY = "leg_schema_version"
RAW_RUNS_DIR_SUFFIX = "-raw-runs"

# CLOSED — exactly the 10 `*.json.raw` files committed by #380 (`6d07b20`)
# AFTER this gate existed (`c7fd1df`, #379), deliberately renamed to dodge
# the `*.json` glob rather than fabricate a `git_sha` (contract §1 Part D
# CV3; §2 A3 quotes the artifact's own `provenance_note`; pressure-v2 NF8/
# pin B). Every one is a bare `Report` dump ({engine_version, host,
# subcommand, tiers}) with no schema/provenance fields at all — they cannot
# be brought under rules (a)-(f), let alone rule (g), without inventing a
# sha the run never resolved. `--self-test` (`check_legacy_raw_nonjson_files_
# exist`) proves every listed relpath still exists; a deletion must shrink
# this list in the SAME commit, and growth is a gate edit (SWARM_GATE_TOUCHED
# by construction — the list is closed, not merely long).
LEGACY_RAW_NONJSON: dict[str, str] = {
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s128_disabled.r1.json.raw": (
        "pre-rule-(g) raw leg (a100b box, s128, disabled arm, replicate 1): bare Report "
        "dump committed by #380 before rule (g) existed; kept .json.raw per this parent "
        "artifact's own provenance_note (a100b_full_step_ab_reference) rather than "
        "fabricate a git_sha for a tip not resolvable against this worktree's ancestry."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s128_disabled.r2.json.raw": (
        "pre-rule-(g) raw leg (a100b box, s128, disabled arm, replicate 2): same "
        "provenance_note as the r1 sibling above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s128_fused.r1.json.raw": (
        "pre-rule-(g) raw leg (a100b box, s128, fused arm, replicate 1): same "
        "provenance_note as the disabled-arm siblings above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s128_fused.r2.json.raw": (
        "pre-rule-(g) raw leg (a100b box, s128, fused arm, replicate 2): same "
        "provenance_note as the disabled-arm siblings above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s512_disabled.r1.json.raw": (
        "pre-rule-(g) raw leg (a100b box, s512, disabled arm, replicate 1): same "
        "provenance_note as the s128 siblings above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s512_disabled.r2.json.raw": (
        "pre-rule-(g) raw leg (a100b box, s512, disabled arm, replicate 2): same "
        "provenance_note as the s128 siblings above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s512_fused.r1.json.raw": (
        "pre-rule-(g) raw leg (a100b box, s512, fused arm, replicate 1): same "
        "provenance_note as the s128 siblings above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s512_fused.r2.json.raw": (
        "pre-rule-(g) raw leg (a100b box, s512, fused arm, replicate 2): same "
        "provenance_note as the s128 siblings above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/r2/b8_s128_disabled.json.raw": (
        "pre-rule-(g) raw leg (r2 box, s128, disabled arm, single replicate): a separate "
        "confirmation session, same 'kept .json.raw rather than fabricate a git_sha' "
        "reasoning as the a100b/ siblings above — see this dir's own PROVENANCE.md."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/r2/b8_s128_fused.json.raw": (
        "pre-rule-(g) raw leg (r2 box, s128, fused arm, single replicate): same "
        "PROVENANCE.md reasoning as the r2/ sibling above."
    ),
}
_ABSENT = object()  # sentinel: key genuinely absent from the object (never confused with JSON null)

_JAMMI_REPORT_RS = REPO_ROOT / "crates" / "jammi-bench" / "src" / "report.rs"
_AB_MERGE_PY = REPO_ROOT / "ci" / "scripts" / "perf" / "ab_merge.py"
_TORCH_FINETUNE_STEP_PY = REPO_ROOT / "crates" / "jammi-bench" / "reference" / "torch_finetune_step.py"

_TIER_IDENTITY_FIELDS_BLOCK_RE = re.compile(
    r"pub const IDENTITY_FIELDS:\s*&'static \[\(&'static str,\s*"
    r"[\w:]*Nullable\)\]\s*=\s*&\[(.*?)\n    \];",
    re.DOTALL,
)
_REPORT_IDENTITY_FIELDS_BLOCK_RE = re.compile(
    r"pub const REPORT_IDENTITY_FIELDS:\s*&\[\(&str,\s*Nullable\)\]\s*=\s*&\[(.*?)\n\];",
    re.DOTALL,
)
# `("field_name", Nullable::NonNull)` or `("field_name",
# crate::report::Nullable::NullMeans("reason"))` (both single-line and the
# multi-line, one-item-per-line spelling `grad_oracle.rs` uses for long
# field names) — captures (name, NonNull|NullMeans, reason-or-empty).
_FIELD_ENTRY_RE = re.compile(
    r'\(\s*"([A-Za-z0-9_]+)"\s*,\s*(?:crate::report::)?Nullable::(NonNull|NullMeans)'
    r'(?:\(\s*"((?:[^"\\]|\\.)*)"\s*\))?\s*,?\s*\)',
    re.DOTALL,
)


def _extract_rust_identity_block(path: Path, block_re: re.Pattern) -> list[tuple[str, str, str | None]]:
    if not path.is_file():
        raise ArtifactError(f"{path} does not exist — cannot derive rule (g)'s jammi identity tuple")
    text = path.read_text(encoding="utf-8")
    m = block_re.search(text)
    if m is None:
        raise ArtifactError(f"no matching IDENTITY_FIELDS-shaped const block found in {path}")
    entries = [(name, kind, reason or None) for name, kind, reason in _FIELD_ENTRY_RE.findall(m.group(1))]
    if not entries:
        raise ArtifactError(f"IDENTITY_FIELDS-shaped block in {path} matched but named zero fields")
    return entries


def _load_module_from_path(module_name: str, path: Path):
    import importlib.util

    if not path.is_file():
        raise ArtifactError(f"{path} does not exist — cannot derive rule (g)'s torch identity tuple")
    perf_dir = str((REPO_ROOT / "ci" / "scripts" / "perf"))
    if perf_dir not in sys.path:
        sys.path.insert(0, perf_dir)  # ab_merge.py imports its sibling identity_fields.py
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ArtifactError(f"could not load {path} as a Python module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# torch_finetune_step.py's own report-assembly code (`run()`, its `report =
# {...}` literal) places every TORCH_IDENTITY_FIELDS entry under EXACTLY one
# of three top-level keys: `args` (the three `ab_merge.py::
# _TORCH_ARGS_LEVEL_FIELDS` entries, plus `adamw_foreach` — the K7-only
# addition sharing that same placement), `provenance` (the nine fields the
# `provenance()` function itself fills), and `finetune_step` for every other
# entry.
_TORCH_PROVENANCE_ROOT_FIELDS = frozenset(
    {
        "torch_version",
        "torch_cuda_version",
        "transformers_version",
        "peft_version",
        "python_version",
        "fast_path_globals",
        "device_name",
        "nvidia_driver_version",
        "git_rev",
    }
)

_IDENTITY_TUPLES_CACHE: dict[tuple[str, str], dict] | None = None


def build_identity_tuples() -> dict[tuple[str, str], dict]:
    """`{(tier, producer_kind): {"sha_root": ..., "sha_field": ..., "fields":
    [(name, root, "NonNull"|"NullMeans", reason_or_None), ...]}}` — computed
    once (module-level cache) by reading the Rust consts and importing the
    Python reference/comparator modules described above; never hand-typed.
    """
    global _IDENTITY_TUPLES_CACHE
    if _IDENTITY_TUPLES_CACHE is not None:
        return _IDENTITY_TUPLES_CACHE

    tier_entries = _extract_rust_identity_block(_JAMMI_REPORT_RS, _TIER_IDENTITY_FIELDS_BLOCK_RE)
    report_entries = _extract_rust_identity_block(_JAMMI_REPORT_RS, _REPORT_IDENTITY_FIELDS_BLOCK_RE)
    jammi_fields = [(name, "tier", kind, reason) for name, kind, reason in tier_entries]
    jammi_fields += [(name, "provenance", kind, reason) for name, kind, reason in report_entries]

    ab_merge = _load_module_from_path("_gate_ab_merge", _AB_MERGE_PY)
    torch_mod = _load_module_from_path("_gate_torch_finetune_step", _TORCH_FINETUNE_STEP_PY)
    torch_args_fields = set(ab_merge._TORCH_ARGS_LEVEL_FIELDS) | {"adamw_foreach"}
    null_means: dict = torch_mod.TORCH_IDENTITY_FIELDS_NULL_MEANS
    torch_fields = []
    for field in torch_mod.TORCH_IDENTITY_FIELDS:
        if field in torch_args_fields:
            root = "args"
        elif field in _TORCH_PROVENANCE_ROOT_FIELDS:
            root = "provenance"
        else:
            root = "finetune_step"
        if field in null_means:
            torch_fields.append((field, root, "NullMeans", null_means[field]))
        else:
            torch_fields.append((field, root, "NonNull", None))

    _IDENTITY_TUPLES_CACHE = {
        ("finetune_step", "jammi"): {"sha_root": "provenance", "sha_field": "build_sha", "fields": jammi_fields},
        ("finetune_step", "torch"): {"sha_root": "provenance", "sha_field": "git_rev", "fields": torch_fields},
    }
    return _IDENTITY_TUPLES_CACHE


def _leg_field_value(leg: dict, field: str, root: str):
    if root == "tier":
        tiers = leg.get("tiers")
        tier = tiers.get("finetune_step") if isinstance(tiers, dict) else None
        return tier.get(field, _ABSENT) if isinstance(tier, dict) else _ABSENT
    src = leg.get(root)
    return src.get(field, _ABSENT) if isinstance(src, dict) else _ABSENT


def _is_v2(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 2


def check_raw_leg_identity_fields(leg: dict, tuple_spec: dict, label: str) -> list[str]:
    failures: list[str] = []
    for field, root, kind, _reason in tuple_spec["fields"]:
        value = _leg_field_value(leg, field, root)
        if value is _ABSENT:
            failures.append(f"{label}: v2 raw leg missing identity field `{field}` (expected under `{root}`)")
        elif value is None and kind == "NonNull":
            failures.append(f"{label}: identity field `{field}` is declared NonNull but reads null")
        # kind == "NullMeans" and value is None: OK — a declared-nullable reading.
    return failures


def check_raw_leg_sha(leg: dict, tuple_spec: dict, parent_git_sha: str | None, label: str) -> list[str]:
    if parent_git_sha is None:
        return []  # nothing to cross-check the leg's own sha against
    sha_root, sha_field = tuple_spec["sha_root"], tuple_spec["sha_field"]
    value = _leg_field_value(leg, sha_field, sha_root)
    if value is _ABSENT:
        return [f"{label}: v2 raw leg missing `{sha_root}.{sha_field}` — cannot cross-check provenance"]
    if value is None:
        entry = next((e for e in tuple_spec["fields"] if e[0] == sha_field), None)
        if entry is not None and entry[2] == "NullMeans":
            return []  # legitimately nullable on this producer_kind (e.g. torch git_rev)
        return [f"{label}: `{sha_root}.{sha_field}` is null and not declared nullable"]
    if not isinstance(value, str) or not GIT_SHA_RE.match(value):
        return [
            f"{label}: `{sha_root}.{sha_field}` = {value!r} is not a resolved 40-hex sha (covers "
            f"'unknown', a '-dirty' suffix, or any other unresolved reading) — a GREEN v2 leg can "
            f"never carry an unresolved build identity"
        ]
    if value != parent_git_sha:
        return [
            f"{label}: `{sha_root}.{sha_field}` = {value} does not match the parent artifact's "
            f"git_sha {parent_git_sha} — this leg was not proven at the sha the artifact claims"
        ]
    return []


def check_v2_leg(
    leg: dict,
    label: str,
    parent_git_sha: str | None,
    raw_runs_dir: Path,
    cuda_runs_dir: Path,
) -> list[str]:
    identity = leg.get("identity")
    if not isinstance(identity, dict):
        return [f"{label}: v2 leg missing `identity` object"]
    tier = identity.get("tier")
    producer_kind = identity.get("producer_kind")
    leg_shape = identity.get("leg_shape")
    failures: list[str] = []
    if not isinstance(tier, str) or not tier:
        failures.append(f"{label}: identity.tier must be a non-empty string")
    if producer_kind not in ("jammi", "torch"):
        failures.append(f"{label}: identity.producer_kind must be 'jammi' or 'torch', got {producer_kind!r}")
    if leg_shape not in ("raw", "folded"):
        failures.append(f"{label}: identity.leg_shape must be 'raw' or 'folded', got {leg_shape!r}")
    if failures:
        return failures

    tuple_spec = build_identity_tuples().get((tier, producer_kind))
    if tuple_spec is None:
        return [f"{label}: no known identity tuple for (tier={tier!r}, producer_kind={producer_kind!r})"]

    if leg_shape == "folded":
        own_field_names = {f[0] for f in tuple_spec["fields"]}
        leaked = sorted(own_field_names & set(leg.keys()))
        if leaked:
            failures.append(f"{label}: folded leg carries identity field(s) of its own {leaked} — identity has ONE home (the raw leg)")
        file_name = leg.get("file")
        if not isinstance(file_name, str) or not file_name:
            return failures + [f"{label}: folded leg missing `file` naming its raw sibling"]
        raw_path = raw_runs_dir / file_name
        if not raw_path.is_file():
            try:
                shown_dir = raw_runs_dir.relative_to(cuda_runs_dir)
            except ValueError:
                shown_dir = raw_runs_dir
            return failures + [f"{label}: folded leg's file `{file_name}` does not exist under {shown_dir}"]
        try:
            raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return failures + [f"{label}: folded leg's raw sibling `{file_name}` failed to parse: {e}"]
        if not isinstance(raw_data, dict) or not _is_v2(raw_data.get(LEG_SCHEMA_VERSION_KEY)):
            return failures + [f"{label}: folded leg's raw sibling `{file_name}` does not carry leg_schema_version >= 2"]
        raw_label = f"{label} -> {file_name}"
        failures += check_v2_leg(raw_data, raw_label, parent_git_sha, raw_runs_dir, cuda_runs_dir)
        return failures

    # raw
    failures += check_raw_leg_identity_fields(leg, tuple_spec, label)
    failures += check_raw_leg_sha(leg, tuple_spec, parent_git_sha, label)
    return failures


def find_v2_legs(data, path_prefix: str = "") -> list[tuple[str, dict]]:
    """Recursively walks `data` for every dict carrying `leg_schema_version
    >= 2` — this is the WHOLE discriminator (contract C6.1): a v2 leg can be
    the top-level document itself, or nested anywhere inside it (a folded
    `shapes.<s>.legs.<leg>` record, a `bench_legs[i]` entry, an embedded
    `clip_on_flash_leg.record`, ...). Recursion continues INTO a matched leg
    too (harmless: no real leg nests a second `leg_schema_version` key)."""
    found: list[tuple[str, dict]] = []
    if isinstance(data, dict):
        if _is_v2(data.get(LEG_SCHEMA_VERSION_KEY)):
            found.append((path_prefix, data))
        for k, v in data.items():
            found += find_v2_legs(v, f"{path_prefix}/{k}" if path_prefix else k)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            found += find_v2_legs(v, f"{path_prefix}[{i}]")
    return found


def _raw_runs_dir_for(artifact_path: Path) -> Path:
    """A single NAMING GUESS (`<stem>-raw-runs`) — a fallback default only,
    used where no directory has actually been discovered to belong to
    `artifact_path` (so a nonexistent guessed path is harmless: `.is_dir()`/
    `.is_file()` on it reads False). This function is NEVER the authority
    for "does a raw-runs directory exist" or "which directories must be
    walked" — a committed directory can be named after a shorter "unit"
    prefix instead of the full stem (`2026-08-25-p6-b3-dense-raw-runs/`
    next to `2026-08-25-p6-b3-dense-b98f7e1-a100-sxm4.json`), so a rule that
    derives its ONE candidate path this way and stops cannot see a
    directory whose name diverges. `_find_raw_runs_dirs` below is that
    authority; `_artifact_for_raw_runs_dir` is the ownership lookup a v2
    rule needs on top of it."""
    return artifact_path.parent / (artifact_path.stem + RAW_RUNS_DIR_SUFFIX)


def _find_raw_runs_dirs(cuda_runs_dir: Path) -> list[Path]:
    """Every directory anywhere under `cuda_runs_dir` whose name ends with
    `-raw-runs` — discovered by NAME PATTERN across the whole tree (an
    `rglob`, not a `glob`, so a raw-runs directory nested more than one
    level down is still found), never by deriving one candidate sibling
    path from a particular artifact's stem. This is the ONE list every rule
    that reasons about raw legs — non-json-payload coverage, the v2-leg-
    required-under-a-v2-parent rule, the `--census` falsifier — walks; none
    of them may instead loop over top-level artifacts and guess a sibling
    path per artifact, because that guess has a known committed
    counterexample (see `_raw_runs_dir_for`'s docstring)."""
    return sorted(p for p in cuda_runs_dir.rglob("*" + RAW_RUNS_DIR_SUFFIX) if p.is_dir())


def _artifact_for_raw_runs_dir(raw_runs_dir: Path, cuda_runs_dir: Path) -> Path | None:
    """The top-level `*.json` artifact that OWNS `raw_runs_dir`, for the one
    thing that still needs ownership: does the owning artifact's
    `schema_version` require the legs under this directory to be v2. Prefers
    an EXACT stem match (`<stem>-raw-runs`, the common case: adamw, the
    fa2-vram-attrib artifact, the p6-stacked-sweep artifact all name their
    raw-runs directory after their own full stem); falls back to the unique
    top-level artifact whose stem starts with `<unit-stem>-` (the
    mismatched-name case: `2026-08-25-p6-b3-dense-raw-runs/` owned by
    `2026-08-25-p6-b3-dense-b98f7e1-a100-sxm4.json`, whose stem drops the
    raw-runs directory's own sha7/gpu suffix). Returns `None` on no match or
    an AMBIGUOUS match (more than one candidate) — a raw-runs directory
    whose ownership cannot be pinned down is still discovered and coverage-
    checked by `_find_raw_runs_dirs` callers; it is simply not yet tied to
    a v2 requirement, which fails toward MORE scrutiny elsewhere, never
    toward silently skipping the directory outright."""
    unit_stem = raw_runs_dir.name[: -len(RAW_RUNS_DIR_SUFFIX)]
    exact = cuda_runs_dir / f"{unit_stem}.json"
    if exact.is_file():
        return exact
    candidates = [
        p
        for p in cuda_runs_dir.glob("*.json")
        if p.stem == unit_stem or p.stem.startswith(unit_stem + "-")
    ]
    return candidates[0] if len(candidates) == 1 else None


def _containing_raw_runs_dir(f: Path, cuda_runs_dir: Path) -> Path | None:
    """The raw-runs directory `f` sits under, however many levels deep — the
    NEAREST ancestor of `f` whose name ends with `-raw-runs`, never `f`'s
    own immediate `.parent`. A leg nested in a per-box subdirectory (e.g.
    `<...>-raw-runs/a100c/leg.json`, the shape `stacked_sweep.sh`'s
    `stamp_leg()` actually writes — all 40 committed stacked legs and 8 of
    16 p6-b3-dense-a100b legs sit this way) belongs to the `-raw-runs`
    ancestor two levels up, not to `a100c/`; a rule keying on `f.parent`
    alone silently never reaches it."""
    for parent in f.parents:
        if parent == cuda_runs_dir:
            return None
        if parent.name.endswith(RAW_RUNS_DIR_SUFFIX):
            return parent
    return None


def find_objects_with_key(data, key: str, path_prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(data, dict):
        if key in data:
            found.append(path_prefix)
        for k, v in data.items():
            found += find_objects_with_key(v, key, f"{path_prefix}/{k}" if path_prefix else k)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            found += find_objects_with_key(v, key, f"{path_prefix}[{i}]")
    return found


def _covered_by(path: str, reached_paths: set[str]) -> bool:
    return any(path == r or path.startswith(r + "/") or path.startswith(r + "[") for r in reached_paths)


def census_unreached_measurement_objects(cuda_runs_dir: Path) -> list[str]:
    """Falsifier F4 (contract §6): for every top-level artifact whose OWN
    `schema_version >= 2`, every JSON object anywhere in its tree — the
    document itself and every `*.json` payload under its sibling
    `*-raw-runs/` directory — that carries `s_per_step_p50` must be
    REACHABLE as (or nested inside) a v2 leg rule (g) actually validated.
    Trivially empty today (zero `schema_version >= 2` top-level artifacts
    exist yet) — the falsifier is standing for the day one does. Reaches
    exactly the same raw-runs directories `check_raw_runs_nonjson_coverage`
    does: discovered by `_find_raw_runs_dirs`'s name pattern plus
    `_artifact_for_raw_runs_dir`'s ownership lookup, never a per-artifact
    `<stem>-raw-runs` guess."""
    unreached: list[str] = []
    dir_for_artifact: dict[Path, Path] = {}
    for raw_runs_dir in _find_raw_runs_dirs(cuda_runs_dir):
        owner = _artifact_for_raw_runs_dir(raw_runs_dir, cuda_runs_dir)
        if owner is not None:
            dir_for_artifact[owner] = raw_runs_dir
    for f in sorted(cuda_runs_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict) or not _is_v2(data.get("schema_version")):
            continue
        reached = {p for p, _ in find_v2_legs(data)}
        for measure_path in find_objects_with_key(data, "s_per_step_p50"):
            if not _covered_by(measure_path, reached):
                unreached.append(f"{f.name}#{measure_path}")
        raw_runs_dir = dir_for_artifact.get(f)
        if raw_runs_dir is not None and raw_runs_dir.is_dir():
            for rf in sorted(raw_runs_dir.rglob("*.json")):
                try:
                    rdata = json.loads(rf.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(rdata, dict) or _is_v2(rdata.get(LEG_SCHEMA_VERSION_KEY)):
                    continue  # a genuine v2 raw leg IS reached, by definition
                for measure_path in find_objects_with_key(rdata, "s_per_step_p50"):
                    rel = rf.relative_to(cuda_runs_dir).as_posix()
                    unreached.append(f"{rel}#{measure_path}")
    return unreached


def check_legacy_raw_nonjson_files_exist(cuda_runs_dir: Path) -> list[str]:
    """`--self-test`'s own shrink-only proof for `LEGACY_RAW_NONJSON`: every
    listed relpath must exist on disk. A deletion must shrink this list in
    the SAME commit — growth (a new entry) is a gate edit, i.e.
    SWARM_GATE_TOUCHED by construction (contract C6.2)."""
    return [
        f"LEGACY_RAW_NONJSON lists `{relpath}` but no such file exists under {cuda_runs_dir}"
        for relpath in LEGACY_RAW_NONJSON
        if not (cuda_runs_dir / relpath).is_file()
    ]


def check_raw_runs_nonjson_coverage(cuda_runs_dir: Path) -> list[str]:
    """Every non-`.json` payload (excluding `.md`/`.log`) under ANY `*-raw-
    runs/` directory, at ANY depth, must be in the closed `LEGACY_RAW_NONJSON`
    list — the `.json.raw` rename bypass (contract C6.2, pressure-v2 NF8) a
    NEW leg may never reuse. Walks `_find_raw_runs_dirs`'s full, name-
    pattern-based discovery — never a per-artifact `<stem>-raw-runs` guess,
    which has a committed counterexample that does not match any artifact's
    stem (`2026-08-25-p6-b3-dense-raw-runs/`) and would otherwise never be
    visited at all."""
    failures: list[str] = []
    for raw_runs_dir in _find_raw_runs_dirs(cuda_runs_dir):
        for p in sorted(raw_runs_dir.rglob("*")):
            if p.is_dir() or p.suffix in (".md", ".log", ".json"):
                continue
            relpath = p.relative_to(cuda_runs_dir).as_posix()
            if relpath not in LEGACY_RAW_NONJSON:
                failures.append(
                    f"{relpath}: non-`.json` payload under a `*-raw-runs/` directory is not in the "
                    f"closed LEGACY_RAW_NONJSON list — a NEW raw leg must be named `*.json`, never "
                    f"renamed to dodge the schema gate's glob"
                )
    return failures


def check_raw_runs_require_v2(
    data: dict, f: Path, cuda_runs_dir: Path, schema_v2_raw_runs_dirs: set[Path]
) -> list[str]:
    """Contract C6.2's last sentence: under a parent artifact with
    `schema_version >= 2`, every `*.json` payload under its `*-raw-runs/`
    sibling MUST carry `leg_schema_version >= 2` (else RED) — a v1-shaped
    raw leg silently coexisting under an already-v2 parent is exactly the
    kind of container drift rule (g) exists to catch. Keys on the raw-runs
    directory `f` actually sits under (`_containing_raw_runs_dir`, which
    walks ancestors to any depth), never on `f.parent` — a leg nested one
    level further down (`<...>-raw-runs/<box>/leg.json`, the shape every
    committed stacked-sweep and p6-b3-dense-a100b leg actually has) has
    `f.parent` equal to the per-box subdirectory, not to the raw-runs
    directory itself, and would otherwise silently escape this rule."""
    containing = _containing_raw_runs_dir(f, cuda_runs_dir)
    if containing is None or containing not in schema_v2_raw_runs_dirs:
        return []
    if _is_v2(data.get(LEG_SCHEMA_VERSION_KEY)):
        return []
    return [
        f"sits under a `*-raw-runs/` directory whose parent artifact is schema_version >= 2 — "
        f"this payload MUST carry leg_schema_version >= 2"
    ]


# --------------------------------------------------------------------------- #
# orchestration — pure over (data, relpath, repo_root, tracked, allowlist) so
# `--self-test` can drive it against a synthetic fixture repo.
# --------------------------------------------------------------------------- #
def validate_artifact(
    data: dict,
    relpath: str,
    repo_root: Path,
    tracked: set[str],
    allowlist: dict[str, str],
) -> list[str]:
    failures = check_schema_types(data)

    producer = data.get("producer") if isinstance(data.get("producer"), dict) else {}
    failures += check_producer_path(producer, repo_root, tracked)
    if producer.get("kind") == "cargo-test":
        failures += check_cargo_test_gating(data, producer, repo_root)
    failures += check_none_allowlist(data, relpath, allowlist)
    failures += check_ancestry(data, repo_root)
    failures += check_oracle_separation(data)
    return failures


def run_gate(
    cuda_runs_dir: Path,
    repo_root: Path,
    allowlist: dict[str, str],
    *,
    gate_introduction_sha: str = GATE_INTRODUCTION_SHA,
) -> list[str]:
    if not cuda_runs_dir.is_dir():
        raise ArtifactError(f"cuda-runs dir not found: {cuda_runs_dir}")

    # Checked BEFORE any per-file work: a shallow checkout makes every
    # single git_sha (even a genuine ancestor's) read back as a false
    # non-ancestor — one explicit, named failure here, never N misleading
    # per-file ancestry findings that look like real drift.
    if is_shallow_repository(repo_root):
        raise ArtifactError(SHALLOW_CHECKOUT_MESSAGE)

    tracked = git_ls_files(repo_root)
    files = sorted(cuda_runs_dir.rglob("*.json"))
    if not files:
        raise ArtifactError(f"no *.json artifacts found under {cuda_runs_dir}")

    # rule (g) precompute: EVERY `*-raw-runs/` directory in the tree
    # (`_find_raw_runs_dirs` — name-pattern discovery, not a per-artifact
    # guess), which top-level artifact owns each one, and which of those
    # owners is already schema_version >= 2 (C6.2's "MUST carry
    # leg_schema_version >= 2" mandate is conditional on that).
    raw_runs_dirs = _find_raw_runs_dirs(cuda_runs_dir)
    dir_for_artifact: dict[Path, Path] = {}
    for raw_runs_dir in raw_runs_dirs:
        owner = _artifact_for_raw_runs_dir(raw_runs_dir, cuda_runs_dir)
        if owner is not None:
            dir_for_artifact[owner] = raw_runs_dir
    schema_v2_raw_runs_dirs: set[Path] = set()
    for owner, owned_dir in dir_for_artifact.items():
        try:
            odata = json.loads(owner.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(odata, dict) and _is_v2(odata.get("schema_version")):
            schema_v2_raw_runs_dirs.add(owned_dir)

    all_failures: list[str] = []
    for f in files:
        relpath = f.relative_to(cuda_runs_dir).as_posix()
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            all_failures.append(f"{relpath}: JSON parse error: {e}")
            continue
        if not isinstance(data, dict):
            all_failures.append(f"{relpath}: top-level JSON value is not an object")
            continue
        failures = validate_artifact(data, relpath, repo_root, tracked, allowlist)
        all_failures.extend(f"{relpath}: {msg}" for msg in failures)

        # rule (g): every v2-leg-shaped object anywhere in THIS document
        # (the document itself, at "<root>", or nested inside it).
        parent_git_sha = data.get("git_sha") if isinstance(data.get("git_sha"), str) else None
        raw_runs_dir = dir_for_artifact.get(f, _raw_runs_dir_for(f))
        for subpath, leg in find_v2_legs(data):
            label = f"{relpath}#{subpath}" if subpath else relpath
            all_failures.extend(
                f"{msg}" for msg in check_v2_leg(leg, label, parent_git_sha, raw_runs_dir, cuda_runs_dir)
            )
        all_failures.extend(
            f"{relpath}: {msg}"
            for msg in check_raw_runs_require_v2(data, f, cuda_runs_dir, schema_v2_raw_runs_dirs)
        )

    all_failures.extend(check_raw_runs_nonjson_coverage(cuda_runs_dir))

    # rule (f)'s mechanical companion (C8.3): every LEGACY_NONE_ALLOWLIST
    # entry's own first-introduction commit must predate this gate.
    for relpath in allowlist:
        all_failures.extend(
            check_none_allowlist_history(relpath, cuda_runs_dir, repo_root, gate_introduction_sha)
        )

    all_failures.extend(
        f"README.md: {msg}"
        for msg in check_readme_producer(cuda_runs_dir / "README.md", repo_root, tracked)
    )
    return all_failures


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    if "--census" in sys.argv[1:]:
        return run_census()

    try:
        failures = run_gate(CUDA_RUNS_DIR, REPO_ROOT, LEGACY_NONE_ALLOWLIST)
    except ArtifactError as exc:
        print(f"cuda-run-artifacts: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    # LEGACY_RAW_NONJSON's own shrink-only closure — only meaningful against
    # the REAL checkout (the module-level dict names real repo relpaths), so
    # this runs here, never inside `run_gate` (which `--self-test` also
    # drives against synthetic fixture repos that do not carry these files).
    failures = failures + check_legacy_raw_nonjson_files_exist(CUDA_RUNS_DIR)

    # GATE_INTRODUCTION_SHA's own anchor — same reasoning: only meaningful
    # against THIS gate script's real, committed history, never a
    # `--self-test` fixture repo (which never contains this file at all).
    failures = failures + check_gate_introduction_sha_anchor()

    if failures:
        print("cuda-run-artifacts: FAIL", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        print(f"\ncuda-run-artifacts: {len(failures)} finding(s).", file=sys.stderr)
        return 1

    print(
        f"cuda-run-artifacts: PASS — every *.json under "
        f"{CUDA_RUNS_DIR.relative_to(REPO_ROOT)} satisfies the schema, ancestry, and "
        "producer-provenance contract."
    )
    return 0


def run_census() -> int:
    """`--census`: falsifier F4 — lists every `s_per_step_p50`-carrying JSON
    object under a `schema_version >= 2` top-level artifact that rule (g)
    did NOT reach. Must print 0 today (no such artifact exists yet); the
    check is standing for the day one does."""
    unreached = census_unreached_measurement_objects(CUDA_RUNS_DIR)
    if unreached:
        print("cuda-run-artifacts --census: FAIL — unreached measurement object(s):", file=sys.stderr)
        for u in unreached:
            print(f"  - {u}", file=sys.stderr)
        print(f"\ncuda-run-artifacts --census: {len(unreached)} unreached.", file=sys.stderr)
        return 1
    print("cuda-run-artifacts --census: PASS — 0 unreached measurement objects under a schema_version >= 2 parent.")
    return 0


# --------------------------------------------------------------------------- #
# self-test — an ephemeral `git init`'d fixture repo, never the real
# checkout, proving each rule (a)-(f) actually bites.
# --------------------------------------------------------------------------- #
def self_test() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        _run(["git", "init", "-q"], repo)
        _run(["git", "config", "user.email", "test@example.com"], repo)
        _run(["git", "config", "user.name", "Test"], repo)

        crate_dir = repo / "crates" / "fixture-crate"
        (crate_dir / "tests").mkdir(parents=True)
        (crate_dir / "tests" / "cuda_parity.rs").write_text(
            "#[test]\n"
            "#[ignore]\n"
            "fn some_gated_test() {\n"
            "    assert!(true);\n"
            "}\n"
            "\n"
            "#[test]\n"
            "fn env_gated_test() {\n"
            "    if std::env::var_os(\"JAMMI_REQUIRE_CUDA\").is_some() {\n"
            "        assert!(true);\n"
            "    }\n"
            "}\n"
        )
        (crate_dir / "Cargo.toml").write_text(
            '[package]\nname = "fixture-crate"\nversion = "0.0.0"\n\n'
            "[[test]]\n"
            'name = "cuda_parity"\n'
            'path = "tests/cuda_parity.rs"\n'
            'required-features = ["cuda"]\n'
        )

        cuda_runs = repo / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs"
        cuda_runs.mkdir(parents=True)
        (cuda_runs / "README.md").write_text("# CUDA run artifacts\n\nProduced by `ci/scripts/perf/proof_artifact.py`.\n")
        perf_dir = repo / "ci" / "scripts" / "perf"
        perf_dir.mkdir(parents=True)
        (perf_dir / "proof_artifact.py").write_text("# stub producer\n")

        _run(["git", "add", "-A"], repo)
        _run(["git", "commit", "-q", "-m", "root"], repo)
        root_sha = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()

        (repo / "unrelated.txt").write_text("x\n")
        _run(["git", "add", "-A"], repo)
        _run(["git", "commit", "-q", "-m", "second"], repo)

        tracked = git_ls_files(repo)
        allowlist = {"legacy-none.json": "synthetic legacy fixture"}

        def baseline() -> dict:
            return {
                "schema_version": 1,
                "git_sha": root_sha,
                "box": "a100-fixture",
                "producer": {
                    "path": "crates/fixture-crate/tests/cuda_parity.rs",
                    "kind": "cargo-test",
                    "invocation": "cargo test -p fixture-crate --test cuda_parity -- --exact some_gated_test",
                    "gating": "#[ignore]",
                },
                "status": "GREEN",
            }

        def expect_clean(data: dict, relpath: str, label: str) -> None:
            got = validate_artifact(data, relpath, repo, tracked, allowlist)
            if got:
                failures.append(f"self-test FAILED: {label} expected clean, got {got}")

        def expect_hit(data: dict, relpath: str, needle: str, label: str) -> None:
            got = validate_artifact(data, relpath, repo, tracked, allowlist)
            if not any(needle in g for g in got):
                failures.append(f"self-test FAILED: {label} expected a finding containing {needle!r}, got {got}")

        # GREEN controls -----------------------------------------------------
        expect_clean(baseline(), "control-ignore.json", "cargo-test + #[ignore] baseline")

        env_variant = baseline()
        env_variant["producer"] = dict(env_variant["producer"])
        env_variant["producer"]["invocation"] = (
            "cargo test -p fixture-crate --test cuda_parity -- --exact env_gated_test"
        )
        env_variant["producer"]["gating"] = "env:JAMMI_REQUIRE_CUDA"
        expect_clean(env_variant, "control-env.json", "cargo-test + env:VAR baseline")

        rf_variant = baseline()
        rf_variant["producer"] = dict(rf_variant["producer"])
        rf_variant["producer"]["gating"] = "required-features"
        expect_clean(rf_variant, "control-required-features.json", "cargo-test + required-features baseline")

        none_variant = {
            "schema_version": 1,
            "git_sha_unresolved": "abc1234",
            "box": "a100-fixture",
            "producer": {"path": None, "kind": "none", "invocation": None, "gating": "none"},
            "status": "GREEN",
        }
        expect_clean(none_variant, "legacy-none.json", "allow-listed producer.kind == none baseline")

        # rule (a) — schema/typing ------------------------------------------
        bad = baseline()
        del bad["schema_version"]
        expect_hit(bad, "x.json", "schema_version must be", "rule (a): missing schema_version")

        bad = baseline()
        del bad["git_sha"]
        expect_hit(bad, "x.json", "missing git_sha", "rule (a): missing git_sha and git_sha_unresolved")

        bad = baseline()
        bad["git_sha"] = "not-hex"
        expect_hit(bad, "x.json", "git_sha must be 40", "rule (a): malformed git_sha")

        bad = baseline()
        bad["git_sha_unresolved"] = "abc1234"
        expect_hit(bad, "x.json", "BOTH git_sha and git_sha_unresolved", "rule (a): both sha fields present")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["kind"] = "bogus"
        expect_hit(bad, "x.json", "producer.kind must be one of", "rule (a): bogus producer.kind")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["gating"] = "bogus"
        expect_hit(bad, "x.json", "producer.gating must be", "rule (a): bogus gating")

        bad = {
            "schema_version": 1,
            "git_sha_unresolved": "abc1234",
            "box": "a100-fixture",
            "producer": {"path": None, "kind": "cargo-test", "invocation": None, "gating": "none"},
            "status": "GREEN",
        }
        expect_hit(bad, "x.json", "requires producer.kind == 'none'", "rule (a): unresolved sha with non-none producer")

        # rule (b) — producer.path exists + tracked ---------------------------
        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["path"] = "crates/fixture-crate/tests/does_not_exist.rs"
        expect_hit(bad, "x.json", "does not exist on disk", "rule (b): nonexistent producer.path")

        untracked_dir = crate_dir / "tests"
        (untracked_dir / "untracked.rs").write_text("#[test]\n#[ignore]\nfn some_gated_test() {}\n")
        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["path"] = "crates/fixture-crate/tests/untracked.rs"
        expect_hit(bad, "x.json", "is not `git ls-files`-tracked", "rule (b): untracked producer.path")

        # rule (c) — cargo-test static verification ---------------------------
        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["invocation"] = "cargo test -p fixture-crate --test cuda_parity"
        expect_hit(bad, "x.json", "lacks `--exact", "rule (c): invocation without --exact")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["invocation"] = "cargo test -p fixture-crate --test cuda_parity -- --exact no_such_fn"
        expect_hit(bad, "x.json", "not found by static scan", "rule (c): fn not found")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["invocation"] = "cargo test -p fixture-crate --test cuda_parity -- --exact env_gated_test"
        # env_gated_test has NO #[ignore] attribute — claiming '#[ignore]' must fail.
        expect_hit(bad, "x.json", "has no", "rule (c): claimed #[ignore] absent")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["invocation"] = "cargo test -p fixture-crate --test cuda_parity -- --exact some_gated_test"
        bad["producer"]["gating"] = "env:SOME_OTHER_VAR"
        # some_gated_test's body never mentions SOME_OTHER_VAR or cuda_device(.
        expect_hit(bad, "x.json", "neither `SOME_OTHER_VAR`", "rule (c): claimed env var absent from body")

        no_rf_dir = repo / "crates" / "no-rf-crate" / "tests"
        no_rf_dir.mkdir(parents=True)
        (no_rf_dir / "cuda_parity.rs").write_text("#[test]\n#[ignore]\nfn some_gated_test() {}\n")
        (repo / "crates" / "no-rf-crate" / "Cargo.toml").write_text(
            '[package]\nname = "no-rf-crate"\nversion = "0.0.0"\n\n'
            "[[test]]\n"
            'name = "cuda_parity"\n'
            'path = "tests/cuda_parity.rs"\n'
        )
        _run(["git", "add", "-A"], repo)
        _run(["git", "commit", "-q", "-m", "third"], repo)
        tracked = git_ls_files(repo)
        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["path"] = "crates/no-rf-crate/tests/cuda_parity.rs"
        bad["producer"]["gating"] = "required-features"
        expect_hit(bad, "x.json", "has no `[[test]]` section", "rule (c): claimed required-features absent")

        # rule (d) — ancestry ---------------------------------------------------
        bad = baseline()
        bad["git_sha"] = "f" * 40
        expect_hit(bad, "x.json", "is not an ancestor of HEAD", "rule (d): non-ancestor git_sha")

        # GREEN control: a squash-merged tip whose OWN sha is not (and never
        # will be again) an ancestor of HEAD, but whose content landed on
        # HEAD via a real (here: the fixture repo's own) commit named by
        # merged_as + merged_via_pr.
        merged_ok = baseline()
        merged_ok["git_sha"] = "f" * 40
        merged_ok["merged_as"] = root_sha
        merged_ok["merged_via_pr"] = 363
        expect_clean(merged_ok, "control-merged-as.json", "non-ancestor git_sha rescued by an ancestor merged_as")

        # RED: merged_as ALSO not an ancestor of HEAD — the rescue must not
        # be granted just because the field is present and well-shaped.
        bad = baseline()
        bad["git_sha"] = "f" * 40
        bad["merged_as"] = "e" * 40
        bad["merged_via_pr"] = 999
        expect_hit(bad, "x.json", "is ALSO not an ancestor of HEAD", "rule (d): non-ancestor merged_as too")

        # RED: merged_as present but git_sha is NOT (only git_sha_unresolved)
        # — merged_as must never stand in for an actually-resolved git_sha.
        bad = {
            "schema_version": 1,
            "git_sha_unresolved": "abc1234",
            "box": "a100-fixture",
            "merged_as": root_sha,
            "merged_via_pr": 363,
            "producer": {"path": None, "kind": "none", "invocation": None, "gating": "none"},
            "status": "GREEN",
        }
        expect_hit(bad, "legacy-none.json", "merged_as requires git_sha", "rule (a): merged_as without a resolved git_sha")

        # RED: merged_as present without merged_via_pr, and vice versa.
        bad = baseline()
        bad["merged_as"] = root_sha
        expect_hit(bad, "x.json", "merged_as is present but merged_via_pr is missing", "rule (a): merged_as without merged_via_pr")

        bad = baseline()
        bad["merged_via_pr"] = 363
        expect_hit(bad, "x.json", "merged_via_pr is present but merged_as is missing", "rule (a): merged_via_pr without merged_as")

        # rule (e) — README's named producer is tracked -------------------------
        untracked_readme = repo / "untracked-readme-dir"
        untracked_readme.mkdir()
        (untracked_readme / "README.md").write_text(
            "Produced by `ci/scripts/perf/untracked/proof_artifact.py`.\n"
        )
        readme_failures = check_readme_producer(untracked_readme / "README.md", repo, tracked)
        if not any("not `git ls-files`-tracked" in f for f in readme_failures):
            failures.append(f"self-test FAILED: rule (e) untracked README producer not caught: {readme_failures}")

        missing_marker_readme = repo / "no-marker-dir"
        missing_marker_readme.mkdir()
        (missing_marker_readme / "README.md").write_text("No producer named here.\n")
        readme_failures2 = check_readme_producer(missing_marker_readme / "README.md", repo, tracked)
        if not any("does not name a" in f for f in readme_failures2):
            failures.append(f"self-test FAILED: rule (e) missing producer marker not caught: {readme_failures2}")

        # rule (f) — kind == none only for the allow-list ------------------------
        bad = {
            "schema_version": 1,
            "git_sha": root_sha,
            "box": "a100-fixture",
            "producer": {"path": None, "kind": "none", "invocation": None, "gating": "none"},
            "status": "GREEN",
        }
        expect_hit(bad, "brand-new-not-allowlisted.json", "not in the reviewed LEGACY_NONE_ALLOWLIST", "rule (f): new file defaulting to none")

        # rule (g) — KO-3 oracle_separation, optional per leg -------------------
        # GREEN: absent entirely — no existing (pre-rule) artifact reddens.
        expect_clean(baseline(), "control-no-separation-block.json", "rule (g): oracle_separation absent is clean")

        # GREEN: present, nested under an arbitrary leg name, with real
        # separation (healthy_max_offsample < bound < min_control).
        good_sep = baseline()
        good_sep["some_arbitrary_leg_name"] = {
            "oracle_separation": {
                "healthy_max_offsample": 0.01,
                "bound": 0.05,
                "min_control": 0.5,
            }
        }
        expect_clean(good_sep, "control-separation-ok.json", "rule (g): a genuinely-separated bound is clean")

        # RED: present but the bound does NOT sit strictly between the two —
        # no demonstrated separation.
        bad_sep = baseline()
        bad_sep["some_arbitrary_leg_name"] = {
            "oracle_separation": {
                "healthy_max_offsample": 0.05,
                "bound": 0.05,
                "min_control": 0.5,
            }
        }
        expect_hit(bad_sep, "x.json", "does not hold", "rule (g): bound not strictly separated is caught")

        # RED: missing a required field inside the block.
        bad_sep2 = baseline()
        bad_sep2["some_arbitrary_leg_name"] = {
            "oracle_separation": {"healthy_max_offsample": 0.01, "bound": 0.05}
        }
        expect_hit(bad_sep2, "x.json", "missing required field", "rule (g): incomplete oracle_separation block is caught")

    # rule (g) — v2 leg identity (contract C6) -----------------------------
    # A dedicated, isolated fixture repo per case (never the real checkout),
    # exercised through `run_gate` end-to-end so the discovery walk,
    # raw/folded dispatch, and sha cross-check all fire together, exactly as
    # they would on a committed artifact.
    def _rule_g_fixture(tmp_root: Path) -> tuple[Path, str]:
        _run(["git", "init", "-q"], tmp_root)
        _run(["git", "config", "user.email", "test@example.com"], tmp_root)
        _run(["git", "config", "user.name", "Test"], tmp_root)
        cr = tmp_root / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs"
        cr.mkdir(parents=True)
        (cr / "README.md").write_text("Produced by `ci/scripts/perf/proof_artifact.py`.\n")
        perf_dir = tmp_root / "ci" / "scripts" / "perf"
        perf_dir.mkdir(parents=True)
        (perf_dir / "proof_artifact.py").write_text("# stub producer\n")
        _run(["git", "add", "-A"], tmp_root)
        _run(["git", "commit", "-q", "-m", "root"], tmp_root)
        sha = _run(["git", "rev-parse", "HEAD"], tmp_root).stdout.strip()
        return cr, sha

    def _none_producer() -> dict:
        return {"path": None, "kind": "none", "invocation": None, "gating": "none"}

    def _synthetic_value_for(field: str):
        if field.endswith("_bytes") or field in ("seed", "batch", "seq", "lora_rank", "steps_measured"):
            return 1
        if field in ("lora_alpha", "lora_dropout", "margin"):
            return 0.1
        if field == "batched_forward":
            return True
        if field in ("target_modules", "kernels_disabled_requested", "kernels_disabled_fired", "build_features"):
            return []
        if field in ("fast_path_globals", "sdpa_backend_probe"):
            return {}
        return "x"

    def _full_leg_fixture(producer_kind: str, build_sha: str, outer_sha: str = "1" * 40) -> dict:
        """Every identity-tuple field for `producer_kind`, populated with a
        typed placeholder — derived from `build_identity_tuples()` itself
        (never a second, independently-drifting hand-typed field list), so
        this fixture builder tracks the Rust/Python source automatically.
        ALSO carries the base rules (a)-(f) schema (`schema_version`,
        `git_sha`, `box`, `producer`, `status`) — a v2 raw leg is written
        as its own standalone `*.json` document under `*-raw-runs/`, and
        the pre-existing rules (a)-(f) already apply to EVERY such file,
        regardless of rule (g). `outer_sha` (rules a-f's own `git_sha`) is
        deliberately a SEPARATE parameter from `build_sha` (rule (g)'s
        `provenance.build_sha`/`git_rev`) — a case exercising an invalid/
        mismatched `build_sha` must not also break the file's OWN base
        schema, or the two rules' findings become impossible to tell apart.
        """
        tuple_spec = build_identity_tuples()[("finetune_step", producer_kind)]
        doc: dict = {
            "identity": {"tier": "finetune_step", "producer_kind": producer_kind, "leg_shape": "raw"},
            LEG_SCHEMA_VERSION_KEY: 2,
            "schema_version": 2,
            "git_sha": outer_sha,
            "box": "a100-fixture",
            "producer": {
                "path": "ci/scripts/perf/proof_artifact.py", "kind": "script",
                "invocation": "python3 ci/scripts/perf/proof_artifact.py <out> <tag>", "gating": "none",
            },
            "status": "GREEN",
        }
        for field, root, _kind, _reason in tuple_spec["fields"]:
            value = _synthetic_value_for(field)
            if root == "tier":
                doc.setdefault("tiers", {}).setdefault("finetune_step", {})[field] = value
            else:
                doc.setdefault(root, {})[field] = value
        sha_root, sha_field = tuple_spec["sha_root"], tuple_spec["sha_field"]
        doc.setdefault(sha_root, {})[sha_field] = build_sha
        return doc

    # (iii) v2 raw leg missing a NonNull field, missing a NullMeans field,
    # and a NullMeans field present-null — checked directly against
    # `check_raw_leg_identity_fields`, the unit rule (g)'s presence/nullness
    # logic lives in.
    jammi_tuple = build_identity_tuples()[("finetune_step", "jammi")]
    torch_tuple = build_identity_tuples()[("finetune_step", "torch")]

    good_jammi_leg = _full_leg_fixture("jammi", "a" * 40)
    if check_raw_leg_identity_fields(good_jammi_leg, jammi_tuple, "x"):
        failures.append(f"self-test FAILED: rule (g) iii control: a fully-populated jammi leg fixture reported findings: {check_raw_leg_identity_fields(good_jammi_leg, jammi_tuple, 'x')}")

    missing_nonnull_leg = _full_leg_fixture("jammi", "a" * 40)
    del missing_nonnull_leg["tiers"]["finetune_step"]["seed"]
    got = check_raw_leg_identity_fields(missing_nonnull_leg, jammi_tuple, "x")
    if not any("missing identity field `seed`" in g for g in got):
        failures.append(f"self-test FAILED: rule (g) iii: missing NonNull field `seed` not caught: {got}")

    # torch_cuda_version is a real TORCH_IDENTITY_FIELDS_NULL_MEANS entry.
    good_torch_leg = _full_leg_fixture("torch", "b" * 40)
    if check_raw_leg_identity_fields(good_torch_leg, torch_tuple, "x"):
        failures.append(f"self-test FAILED: rule (g) iii control: a fully-populated torch leg fixture reported findings: {check_raw_leg_identity_fields(good_torch_leg, torch_tuple, 'x')}")

    missing_nullmeans_leg = _full_leg_fixture("torch", "b" * 40)
    del missing_nullmeans_leg["provenance"]["torch_cuda_version"]
    got = check_raw_leg_identity_fields(missing_nullmeans_leg, torch_tuple, "x")
    if not any("missing identity field `torch_cuda_version`" in g for g in got):
        failures.append(f"self-test FAILED: rule (g) iii: missing NullMeans field `torch_cuda_version` not caught: {got}")

    present_null_nullmeans_leg = _full_leg_fixture("torch", "b" * 40)
    present_null_nullmeans_leg["provenance"]["torch_cuda_version"] = None
    got = check_raw_leg_identity_fields(present_null_nullmeans_leg, torch_tuple, "x")
    if any("torch_cuda_version" in g for g in got):
        failures.append(f"self-test FAILED: rule (g) iii: a present-but-null NullMeans field must NOT be a finding: {got}")

    # (iv)/(v) sha cross-check: mismatch, and unknown/-dirty on an
    # otherwise-GREEN leg — exercised end to end via run_gate so the
    # parent-artifact git_sha comparison (not just field presence) fires.
    with tempfile.TemporaryDirectory() as tmp_iv:
        cr, root = _rule_g_fixture(Path(tmp_iv))
        parent = {
            "schema_version": 1, "git_sha": root, "box": "a100-fixture",
            "producer": {
                "path": "ci/scripts/perf/proof_artifact.py", "kind": "script",
                "invocation": "python3 ci/scripts/perf/proof_artifact.py <out> <tag>", "gating": "none",
            },
            "status": "GREEN",
        }
        (cr / "rg-iv-parent.json").write_text(json.dumps(parent))
        raw_dir = cr / "rg-iv-parent-raw-runs"
        raw_dir.mkdir()
        mismatched_leg = _full_leg_fixture("jammi", "f" * 40, outer_sha=root)  # build_sha != root
        (raw_dir / "leg.json").write_text(json.dumps(mismatched_leg))
        got = run_gate(cr, Path(tmp_iv), {})
        if not any("does not match the parent artifact's git_sha" in g for g in got):
            failures.append(f"self-test FAILED: rule (g) iv: build_sha/git_sha mismatch not caught: {got}")

    with tempfile.TemporaryDirectory() as tmp_v:
        cr, root = _rule_g_fixture(Path(tmp_v))
        parent = {
            "schema_version": 1, "git_sha": root, "box": "a100-fixture",
            "producer": {
                "path": "ci/scripts/perf/proof_artifact.py", "kind": "script",
                "invocation": "python3 ci/scripts/perf/proof_artifact.py <out> <tag>", "gating": "none",
            },
            "status": "GREEN",
        }
        (cr / "rg-v-parent.json").write_text(json.dumps(parent))
        raw_dir = cr / "rg-v-parent-raw-runs"
        raw_dir.mkdir()
        dirty_leg = _full_leg_fixture("jammi", root + "-dirty", outer_sha=root)
        (raw_dir / "leg.json").write_text(json.dumps(dirty_leg))
        got = run_gate(cr, Path(tmp_v), {})
        if not any("not a resolved 40-hex sha" in g for g in got):
            failures.append(f"self-test FAILED: rule (g) v: a '-dirty'-suffixed build_sha on a GREEN leg was not caught: {got}")

    # (vi) folded leg carrying identity fields of its own -> RED
    with tempfile.TemporaryDirectory() as tmp_vi:
        cr, root = _rule_g_fixture(Path(tmp_vi))
        raw_dir = cr / "rg-vi-parent-raw-runs"
        raw_dir.mkdir()
        raw_leg = _full_leg_fixture("jammi", root, outer_sha=root)
        (raw_dir / "leg.json").write_text(json.dumps(raw_leg))
        folded_leg_with_leak = {
            "identity": {"tier": "finetune_step", "producer_kind": "jammi", "leg_shape": "folded"},
            LEG_SCHEMA_VERSION_KEY: 2,
            "file": "leg.json",
            "seed": 42,  # LEAK: a folded leg must carry NO identity fields of its own
        }
        parent = {
            "schema_version": 1, "git_sha": root, "box": "a100-fixture",
            "producer": {
                "path": "ci/scripts/perf/proof_artifact.py", "kind": "script",
                "invocation": "python3 ci/scripts/perf/proof_artifact.py <out> <tag>", "gating": "none",
            },
            "status": "GREEN",
            "bench_legs": [folded_leg_with_leak],
        }
        (cr / "rg-vi-parent.json").write_text(json.dumps(parent))
        got = run_gate(cr, Path(tmp_vi), {})
        if not any("carries identity field(s) of its own" in g for g in got):
            failures.append(f"self-test FAILED: rule (g) vi: a folded leg leaking an identity field was not caught: {got}")

    # (ii)/(i) — a non-.json raw payload (e.g. a `.json.raw` rename) under a
    # `*-raw-runs/` dir that is NOT in the closed LEGACY_RAW_NONJSON list ->
    # RED, regardless of the parent's own schema_version or the payload's
    # own leg_schema_version (both contract acceptance bullets (i) and (ii)
    # land on this SAME finding under this gate's design: the rename bypass
    # is caught before content is ever inspected).
    with tempfile.TemporaryDirectory() as tmp_ii:
        cr, root = _rule_g_fixture(Path(tmp_ii))
        parent = {
            "schema_version": 2, "git_sha": root, "box": "a100-fixture",
            "producer": {
                "path": "ci/scripts/perf/proof_artifact.py", "kind": "script",
                "invocation": "python3 ci/scripts/perf/proof_artifact.py <out> <tag>", "gating": "none",
            },
            "status": "GREEN",
        }
        (cr / "rg-ii-parent.json").write_text(json.dumps(parent))
        raw_dir = cr / "rg-ii-parent-raw-runs"
        raw_dir.mkdir()
        (raw_dir / "leg.json.raw").write_text(json.dumps({"engine_version": "0.0.0"}))
        got = run_gate(cr, Path(tmp_ii), {})
        if not any("not in the closed LEGACY_RAW_NONJSON list" in g for g in got):
            failures.append(f"self-test FAILED: rule (g) i/ii: a non-allowlisted .json.raw payload was not caught: {got}")

    # (ix) discovery domain — a `*-raw-runs/` directory whose name does NOT
    # match `<any-artifact-stem>-raw-runs` (the committed counterexample:
    # `2026-08-25-p6-b3-dense-raw-runs/` sits next to
    # `2026-08-25-p6-b3-dense-b98f7e1-a100-sxm4.json`, a stem that drops the
    # sha7/gpu suffix) must still be discovered and coverage-checked. Two
    # legs here: one under the DIR ITSELF (flat) and one nested a level
    # further down (`.../subdir/leg.json.raw`, any depth) — both must be
    # caught; a rule that only derives ONE candidate sibling path per
    # artifact stem sees neither.
    with tempfile.TemporaryDirectory() as tmp_ix:
        cr, root = _rule_g_fixture(Path(tmp_ix))
        owner = {
            "schema_version": 1, "git_sha": root, "box": "a100-fixture",
            "producer": {
                "path": "ci/scripts/perf/proof_artifact.py", "kind": "script",
                "invocation": "python3 ci/scripts/perf/proof_artifact.py <out> <tag>", "gating": "none",
            },
            "status": "GREEN",
        }
        # The owning artifact's stem carries a suffix the raw-runs dir name
        # drops entirely — same shape as the real p6-b3-dense mismatch.
        (cr / "rg-ix-unit-deadbeef-a100-sxm4.json").write_text(json.dumps(owner))
        mismatched_dir = cr / "rg-ix-unit-raw-runs"  # NOT "<stem>-raw-runs"
        (mismatched_dir / "subdir").mkdir(parents=True)
        (mismatched_dir / "flat.json.raw").write_text(json.dumps({"engine_version": "0.0.0"}))
        (mismatched_dir / "subdir" / "nested.json.raw").write_text(json.dumps({"engine_version": "0.0.0"}))
        got = run_gate(cr, Path(tmp_ix), {})
        if not any("rg-ix-unit-raw-runs/flat.json.raw" in g and "not in the closed LEGACY_RAW_NONJSON list" in g for g in got):
            failures.append(f"self-test FAILED: rule (g) ix: a mismatched-name raw-runs dir's FLAT non-json payload was not caught: {got}")
        if not any("rg-ix-unit-raw-runs/subdir/nested.json.raw" in g and "not in the closed LEGACY_RAW_NONJSON list" in g for g in got):
            failures.append(f"self-test FAILED: rule (g) ix: a mismatched-name raw-runs dir's NESTED non-json payload was not caught: {got}")

    # (x) leg identity keys on the raw-runs dir a leg belongs to, not its
    # immediate parent: a v1 leg (no `leg_schema_version`) nested ONE level
    # below a schema_version >= 2 parent's raw-runs dir
    # (`<...>-raw-runs/<box>/leg.json`, the shape stacked_sweep.sh's
    # `stamp_leg()` actually writes for every committed leg) must still be
    # caught -> RED. A compliant v2 leg at the SAME depth must stay clean.
    with tempfile.TemporaryDirectory() as tmp_x:
        cr, root = _rule_g_fixture(Path(tmp_x))
        parent = {
            "schema_version": 2, "git_sha": root, "box": "a100-fixture",
            "producer": {
                "path": "ci/scripts/perf/proof_artifact.py", "kind": "script",
                "invocation": "python3 ci/scripts/perf/proof_artifact.py <out> <tag>", "gating": "none",
            },
            "status": "GREEN",
        }
        (cr / "rg-x-parent.json").write_text(json.dumps(parent))
        raw_dir = cr / "rg-x-parent-raw-runs"
        box_dir = raw_dir / "a100c"
        box_dir.mkdir(parents=True)
        v1_leg_nested = {
            "tiers": {"finetune_step": {"seed": 1}},
            "schema_version": 1, "git_sha": root, "box": "a100-fixture",
            "producer": {
                "path": "ci/scripts/perf/proof_artifact.py", "kind": "script",
                "invocation": "python3 ci/scripts/perf/proof_artifact.py <out> <tag>", "gating": "none",
            },
            "status": "GREEN",
        }
        (box_dir / "v1-leg.json").write_text(json.dumps(v1_leg_nested))
        compliant_v2_leg = _full_leg_fixture("jammi", root, outer_sha=root)
        (box_dir / "v2-leg.json").write_text(json.dumps(compliant_v2_leg))
        got = run_gate(cr, Path(tmp_x), {})
        if not any(
            "rg-x-parent-raw-runs/a100c/v1-leg.json" in g and "MUST carry leg_schema_version >= 2" in g
            for g in got
        ):
            failures.append(f"self-test FAILED: rule (g) x: a v1 leg nested under a per-box subdir of a v2 parent's raw-runs dir was not caught: {got}")
        if any("v2-leg.json" in g for g in got):
            failures.append(f"self-test FAILED: rule (g) x control: a compliant v2 leg nested under a per-box subdir was flagged: {got}")

    # (vii) a v1 leg (no `leg_schema_version` at all) under a schema_version:
    # 1 parent -> unchanged behaviour: rule (g) finds nothing, only rules
    # (a)-(f) apply, exactly as before this rule existed.
    with tempfile.TemporaryDirectory() as tmp_vii:
        cr, root = _rule_g_fixture(Path(tmp_vii))
        parent = {
            "schema_version": 1, "git_sha": root, "box": "a100-fixture",
            "producer": {
                "path": "ci/scripts/perf/proof_artifact.py", "kind": "script",
                "invocation": "python3 ci/scripts/perf/proof_artifact.py <out> <tag>", "gating": "none",
            },
            "status": "GREEN",
        }
        (cr / "rg-vii-parent.json").write_text(json.dumps(parent))
        raw_dir = cr / "rg-vii-parent-raw-runs"
        raw_dir.mkdir()
        v1_leg = {"tiers": {"finetune_step": {"seed": 1}}, "schema_version": 1, "git_sha": root, "box": "x", "producer": _none_producer(), "status": "GREEN"}
        (raw_dir / "leg.json").write_text(json.dumps(v1_leg))
        got = run_gate(cr, Path(tmp_vii), {"rg-vii-parent-raw-runs/leg.json": "v1 control, allowlisted for this fixture only"})
        if any("leg_schema_version" in g or "identity" in g for g in got):
            failures.append(f"self-test FAILED: rule (g) vii: a v1 leg under a v1 parent triggered rule (g) findings unexpectedly: {got}")

    # (viii) allowlisted `kind: none` relpath whose first-introduction commit
    # is AFTER the (fixture) gate-introduction sha -> RED. A GREEN control
    # (introduced BEFORE) must pass.
    with tempfile.TemporaryDirectory() as tmp_viii:
        cr, gate_sha = _rule_g_fixture(Path(tmp_viii))
        # `git log --follow`'s rename detection is CONTENT-similarity based
        # (needed for real `git mv`-tracked baselines, contract C8.2/C8.3).
        # A single-line, LOW-ENTROPY body (shared JSON key/value boilerplate,
        # or even a long run of one repeated padding character) can hash-
        # similar enough for git to treat the second file as a rename of the
        # first regardless of intent — confirmed empirically: single-line
        # bodies differing only by a repeated-character suffix (`"p"*400` vs
        # `"n"*400`) STILL cross-attributed under git 2.50. Multi-line,
        # varied-word bodies do not.
        (cr / "pre-existing.json").write_text(
            "line one alpha bravo charlie\nline two delta echo foxtrot\nline three golf hotel india\n"
        )
        _run(["git", "add", "-A"], Path(tmp_viii))
        _run(["git", "commit", "-q", "-m", "pre-existing artifact, then the gate lands"], Path(tmp_viii))
        gate_sha2 = _run(["git", "rev-parse", "HEAD"], Path(tmp_viii)).stdout.strip()

        (cr / "new-after-gate.json").write_text(
            "totally unrelated juliet kilo lima\nmike november oscar papa quebec\nromeo sierra tango uniform victor\n"
        )
        _run(["git", "add", "-A"], Path(tmp_viii))
        _run(["git", "commit", "-q", "-m", "a NEW artifact, introduced after the fixture gate sha"], Path(tmp_viii))

        got_control = check_none_allowlist_history(
            "pre-existing.json", cr, Path(tmp_viii), gate_introduction_sha=gate_sha2
        )
        if got_control:
            failures.append(f"self-test FAILED: rule (f) history GREEN control: a legitimately pre-gate artifact was flagged: {got_control}")

        got_red = check_none_allowlist_history(
            "new-after-gate.json", cr, Path(tmp_viii), gate_introduction_sha=gate_sha2
        )
        if not any("is NOT an ancestor of this gate's own introduction" in g for g in got_red):
            failures.append(f"self-test FAILED: rule (f) history: a post-gate-introduction allowlist entry was not caught: {got_red}")

    # LEGACY_RAW_NONJSON's own shrink-only closure: every listed relpath
    # must exist on disk under the REAL cuda-runs dir.
    real_missing = check_legacy_raw_nonjson_files_exist(CUDA_RUNS_DIR)
    if real_missing:
        failures.append(f"self-test FAILED: LEGACY_RAW_NONJSON lists a relpath that does not exist on disk: {real_missing}")

    # GATE_INTRODUCTION_SHA anchor — only meaningful against THIS gate
    # script's own REAL, committed history (a `--self-test` fixture repo
    # never contains this file). GREEN control: the real constant, checked
    # against this file's own real first-introduction commit, is clean.
    # RED (M-H repro): repointing the constant FORWARD to any commit that
    # is NOT this gate's own first-introduction commit must be caught —
    # e3d8cb7cb12d641e1a0bd64c4d1f663a052b9def is a real, later commit on
    # this repo's history, confirmed NOT an ancestor of the gate's actual
    # introduction.
    real_gate_file = Path(__file__).resolve()
    got_anchor_control = check_gate_introduction_sha_anchor(
        REPO_ROOT, real_gate_file, GATE_INTRODUCTION_SHA
    )
    if got_anchor_control:
        failures.append(f"self-test FAILED: GATE_INTRODUCTION_SHA anchor GREEN control: the real constant was flagged: {got_anchor_control}")

    got_anchor_red = check_gate_introduction_sha_anchor(
        REPO_ROOT, real_gate_file, "e3d8cb7cb12d641e1a0bd64c4d1f663a052b9def"
    )
    if not any("does not match this gate file's own first-introduction commit" in g for g in got_anchor_red):
        failures.append(f"self-test FAILED: GATE_INTRODUCTION_SHA anchor: a forward-repointed constant was not caught: {got_anchor_red}")

    # Shallow-checkout detection — a GENUINE `git clone --depth 1` (not a
    # simulated flag), proving `is_shallow_repository` tells a shallow
    # checkout apart from a normal one, and that `run_gate` raises ONE
    # explicit ArtifactError naming the exact remediation instead of
    # false-failing every artifact's ancestry check. This is the regression
    # a real CI run hit: `p1`'s TRUE-ancestor git_sha FAILED under the
    # default `actions/checkout` (fetch-depth 1) shallow clone.
    with tempfile.TemporaryDirectory() as shallow_src_dir, tempfile.TemporaryDirectory() as shallow_dst_dir:
        shallow_src = Path(shallow_src_dir)
        _run(["git", "init", "-q"], shallow_src)
        _run(["git", "config", "user.email", "test@example.com"], shallow_src)
        _run(["git", "config", "user.name", "Test"], shallow_src)
        cr = shallow_src / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs"
        cr.mkdir(parents=True)
        (cr / "README.md").write_text("Produced by `ci/scripts/perf/proof_artifact.py`.\n")
        (cr / "one.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "git_sha": "0" * 40,
                    "box": "x",
                    "producer": {"path": None, "kind": "none", "invocation": None, "gating": "none"},
                    "status": "GREEN",
                }
            )
        )
        _run(["git", "add", "-A"], shallow_src)
        _run(["git", "commit", "-q", "-m", "c1"], shallow_src)
        (shallow_src / "unrelated.txt").write_text("x\n")
        _run(["git", "add", "-A"], shallow_src)
        _run(["git", "commit", "-q", "-m", "c2"], shallow_src)

        if is_shallow_repository(shallow_src):
            failures.append("self-test FAILED: a normal (non-shallow, 2-commit) fixture repo was reported shallow")

        shallow_clone = Path(shallow_dst_dir) / "clone"
        # `--depth` is silently ignored for a plain local path ("warning:
        # --depth is ignored in local clones; use file:// instead.") — the
        # `file://` scheme is required to force git to actually honor it and
        # produce a genuinely shallow clone, not just a fast local hardlink
        # copy of the full history.
        clone_proc = _run(
            ["git", "clone", "-q", "--depth", "1", "file://" + str(shallow_src), str(shallow_clone)],
            shallow_src,
        )
        if clone_proc.returncode != 0:
            failures.append(f"self-test FAILED: could not create a --depth 1 clone fixture: {clone_proc.stderr}")
        else:
            if not is_shallow_repository(shallow_clone):
                failures.append("self-test FAILED: a genuine `git clone --depth 1` was NOT detected as shallow")

            shallow_allowlist = {"one.json": "synthetic shallow-checkout fixture"}
            try:
                run_gate(
                    shallow_clone / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs",
                    shallow_clone,
                    shallow_allowlist,
                )
                failures.append("self-test FAILED: run_gate did not raise on a shallow checkout")
            except ArtifactError as exc:
                if SHALLOW_CHECKOUT_MESSAGE not in str(exc):
                    failures.append(
                        f"self-test FAILED: shallow-checkout ArtifactError had the wrong message: {exc}"
                    )

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("cuda-run-artifacts self-test: FAIL", file=sys.stderr)
        return 1
    print(
        "cuda-run-artifacts self-test: OK — every rule (a) schema/typing (including the "
        "merged_as/merged_via_pr pairing), (b) producer.path existence+tracking, (c) cargo-test "
        "static gating verification (#[ignore]/env:VAR/required-features), (d) ancestry (both the "
        "plain git_sha path and the merged_as squash-landing rescue, and its own non-ancestor RED "
        "case), (e) README producer tracking, (f) the none-allowlist closure (including its C8.3 "
        "first-introduction-predates-the-gate history check), and the OPTIONAL KO-3 "
        "oracle_separation block (absent is clean; a genuinely-separated bound is clean; a bound "
        "that is not strictly between healthy_max_offsample and min_control, or an incomplete "
        "block, is caught) all bite on a throwaway fixture repo; GREEN controls "
        "(ignore/env/required-features/merged_as-rescue/oracle_separation) plus one allow-listed "
        "none control stay clean; v2-leg identity (missing NonNull/NullMeans fields, a "
        "present-null NullMeans pass, the build_sha/git_rev cross-check on both mismatch and "
        "unknown/-dirty, the folded-leg-must-carry-no-identity-of-its-own rule, the LEGACY_RAW_NONJSON "
        "closed-list rename-bypass rejection, and a v1 leg's unchanged no-op) all bite too; and a "
        "GENUINE `git clone --depth 1` fixture proves is_shallow_repository tells shallow from normal "
        "apart and run_gate raises the one explicit shallow-checkout ArtifactError instead of N false "
        "per-file ancestry findings."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
