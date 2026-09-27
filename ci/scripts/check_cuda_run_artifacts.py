#!/usr/bin/env python3
# needs: full-history
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
      - `gating`: `"feature:<name>"` | `"required-features"` | `"#[ignore]"` |
        `"env:<VAR>"` | `"none"` — how the named test/script stays out of a
        plain `cargo test`/CI run, as it was when the artifact was measured
        (a test needing a GPU is compiled only under `live-gpu-tests`; older
        records carry the `env:` form of their day).
  - `status` (string)
  - `merged_as` (40-hex, OPTIONAL) + `merged_via_pr` (int, OPTIONAL, required
    together with `merged_as`) — for a measured tip that was itself
    squash-merged (so `git_sha` is legitimately never an ancestor of
    anything again): the squash commit the SAME content landed on `main`
    as, and the PR that merged it. `git_sha` is kept VERBATIM (the tip that
    was actually measured); `merged_as` only ever supplements it, never
    replaces it, and is only valid alongside a resolved `git_sha` (never
    `git_sha_unresolved`).
  - `producer.source_sha256` (OPTIONAL — `{<repo-root-relative path>: <sha256
    hex>}`): a producer's own CONTENT identity, for a producer whose module
    doc names this the regeneration-provenance convention instead of a git
    commit sha (see rule (j)).
  - `producer.identity` (OPTIONAL — MANDATORY for a known convention-
    declaring producer, see rule (j)): the fixed marker string
    `"source_sha256+input_manifest"`, stamped by a producer whose module doc
    declares that BOTH `producer.source_sha256` (its own content identity)
    AND `producer.input_sha256` (its input files' content identity) are
    always present together — never one without the other.

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
  (g) separation in the artifact (`docs/maintainer/cuda-kernel-guide.md`
      §3, an instance of §3.8): an OPTIONAL `oracle_separation:
      {healthy_max_offsample, bound, min_control}` block, attached to ANY
      leg anywhere in the artifact (found by recursing the whole document,
      not a fixed top-level key — see `check_oracle_separation`'s own doc),
      asserts `healthy_max_offsample < bound < min_control` when present.
      An artifact without the block is not checked by this rule.
  (i) leg identity on self-declaring v2 legs: any JSON object anywhere in a
      `cuda-runs/**` tree carrying `leg_schema_version >= 2` must carry the complete identity tuple for
      its `(tier, producer_kind)` and a `provenance.build_sha` equal to its
      parent artifact's own `git_sha` — see `check_v2_leg`/`find_v2_legs`
      below.

      There is no rule (h). The letters are comment/self-test-label prose
      only — no gate, allowlist, or error message parses them.
  (j) `producer.source_sha256`, when present, is re-hashed HERE — every
      named repo-root-relative path is re-read from THIS gate's own HEAD
      (the real working tree, never a historical blob) and its sha256
      compared against the recorded value; a mismatch is a hard FAIL naming
      the path (`check_producer_source_sha256`). The producer's own CONTENT
      identity is the determinant, never a commit sha (an artifact cannot
      know, at render time, which future commit will contain it): editing
      the producer (or a file it reads a live constant from) and forgetting
      to regenerate the committed artifact is caught here, never silently
      accepted as still-valid provenance.
      `source_sha256`/`input_sha256` stay OPTIONAL for a producer that never
      opted into this convention — but a producer that DOES (stamped via
      `producer.identity == "source_sha256+input_manifest"`) must carry
      BOTH blocks together (a marker with only one is an incomplete
      identity claim, refused by name, never treated as "no identity"), and
      the marker becomes MANDATORY the moment any ONE of three independent
      anchors fires: the producer's own PATH is a known, reviewed
      convention-declarer (`SOURCE_IDENTITY_DECLARING_PRODUCER_PATHS`,
      mirroring `LEGACY_NONE_ALLOWLIST`'s shape); the artifact ALREADY
      carries a non-empty `producer.source_sha256` block (declaring the
      convention by its own content, regardless of what `producer.path`
      says); or the artifact's own committed FILENAME matches a known
      profile/frontend artifact family
      (`SOURCE_IDENTITY_DECLARING_FILENAME_RE`) — silently omitting the
      marker on a future regeneration, OR renaming `producer.path` away
      from the reviewed allowlist, would otherwise let that artifact
      quietly fall back to the unchecked "no identity" state this rule has
      always allowed for a producer that never opted in; the source_sha256/
      filename anchors close exactly that escape hatch
      (`check_producer_source_identity_marker`).

  (k) the `topology` ARTIFACT KIND (the GPU topology lane's evidence): an
      artifact declared `topology` by ANY of three independent anchors —
      `artifact_kind == "topology"`, a top-level `topology` block, or a
      committed filename matching `TOPOLOGY_ARTIFACT_FILENAME_RE` — names
      the lane's driver as its producer and carries every row of
      `TOPOLOGY_FIELD_REGISTRY`: the fleet's shape (`hosts` >= 2,
      `gpus_per_host` >= 2), the one-host cell's passed device and product
      tests, the one NCCL runtime the fleet ran, the fleet's runs under
      each transport (the coordinator's selected transport and attempt
      count; one distinct
      process per rank, one rank per GPU, spanning every host; the loss
      curve; the probe embeddings' digest), the single-rank reference, the
      trainer's registered ε, and the lane's `verdict`/`reasons`. On a
      `pass` the claims are cross-checked — each gang published on its first
      attempt, both transports' curves and digests equal, the gang's loss
      within ε of the single rank; a `fail`
      is admitted as measured with its reasons and a top-level `status`
      that is not GREEN.

Rule (d) needs REAL commit history to mean anything: `git merge-base
--is-ancestor` on a shallow checkout (`actions/checkout`'s default
`fetch-depth: 1`) reads back EVERY `git_sha` as a false non-ancestor —
indistinguishable from a genuine one without checking first. Before any
per-file work, `run_gate` calls `git rev-parse --is-shallow-repository` and,
if shallow, raises ONE explicit failure ("shallow checkout — ancestry cannot
be evaluated; use fetch-depth: 0") instead of N misleading per-file findings
that would look like real drift. `ci/guards.toml` declares this guard's `full-history` need, which the
runner provides by deepening a shallow checkout.

Run: `python3 ci/scripts/check_cuda_run_artifacts.py`
Self-test (RED cases for every rule above, on a throwaway `git init`'d
fixture repo — never the real checkout):
`python3 ci/scripts/check_cuda_run_artifacts.py --self-test`
Hermetic: reads the working tree (or an ephemeral tempdir git repo under
`--self-test`) and shells out only to `git`; no network, no cargo, no GPU.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ancestry  # noqa: E402 — the ONE ancestry rule every artifact gate shares

REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA_RUNS_DIR = REPO_ROOT / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs"
README_PATH = CUDA_RUNS_DIR / "README.md"

GIT_SHA_RE = ancestry.GIT_SHA_RE
PRODUCER_KINDS = {"cargo-test", "script", "none"}
GATING_STATIC = {"#[ignore]", "required-features", "none"}
GATING_ENV_RE = re.compile(r"^env:[A-Za-z_][A-Za-z0-9_]*$")
GATING_FEATURE_RE = re.compile(r"^feature:[a-z0-9][a-z0-9_-]*$")
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
        "jammi-bench finetune-step invocations on one exclusive box, not a "
        "single #[test] fn or a tracked producer script."
    ),
    "2026-08-25-p1-5f29e3b-a100-sxm4.json": (
        "pre-schema artifact: produced by an earlier, untracked copy of "
        "proof_artifact.py, before it was tracked at "
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
    # The two `crates/jammi-bench/baselines/*.json` records `git mv`d under
    # this directory. Neither has a tracked producer script (both were
    # hand-driven `jammi-bench finetune-step` invocations on one box — see
    # each file's own `_comment`), so `producer.kind == "none"` is the honest
    # reading, same as the five entries above. Growth of this list is not a
    # human review call alone: `check_none_allowlist_history` (rule (f)'s
    # mechanical companion)
    # requires every entry's FIRST INTRODUCTION commit (`git log --follow
    # --diff-filter=A`) to be an ancestor of this gate's own introduction
    # (`c7fd1df`, GATE_INTRODUCTION_SHA below) — both of these predate it
    # (`3719bc8`, `5879c48`), so a genuinely NEW artifact can never satisfy the
    # condition and this list cannot grow again without a gate edit AND a
    # history it does not have.
    "2026-08-24-finetune-step-reference-d361515-a100-pcie.json": (
        "pre-schema baseline, moved from crates/jammi-bench/baselines/"
        "finetune_step_reference.json: a same-box "
        "A/B reference run by hand via jammi-bench finetune-step invocations, "
        "not a single #[test] fn or a tracked producer script — see this "
        "file's own _comment."
    ),
    "2026-08-24-p1-softmax-fold-bf8e807-a100-sxm4.json": (
        "pre-schema baseline, moved from crates/jammi-bench/baselines/"
        "p1_softmax_scale_fold_ab.json: a same-box "
        "A/B run by hand proving the P1 softmax-scale-fold change, not a "
        "single #[test] fn or a tracked producer script — see this file's "
        "own _comment."
    ),
}

# The commit that introduced THIS gate (schema + ancestry +
# producer-provenance for cuda-run artifacts). Every
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


# `_run` IS `ancestry.run`: the shared git-plumbing helper, which suppresses
# gc/maintenance so a `shutil.rmtree` during a `tempfile.TemporaryDirectory`'s
# teardown never races a background `git maintenance`/`gc --auto` spawned by
# this file's own scratch-repo `git init`/`add`/`commit`/`clone` calls.
_run = ancestry.run


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
    indistinguishable from a REAL non-ancestor without this check.
    `git rev-parse --is-shallow-repository` is the one command that
    tells the two apart; a non-zero exit (e.g. run outside a git repo at
    all) is treated as "not shallow" here — `git_ls_files`/`_is_ancestor`
    will raise their own, more specific errors moments later if the
    checkout is unusable for some other reason. Delegates to `ancestry.
    is_shallow_repository` -- the identical check `check_pod_
    build_timings.py` uses.
    """
    return ancestry.is_shallow_repository(repo_root)


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
            isinstance(gating, str) and (GATING_ENV_RE.match(gating) or GATING_FEATURE_RE.match(gating))
        ):
            failures.append(
                f"producer.gating must be '#[ignore]' | 'required-features' | 'none' | 'env:<VAR>' | 'feature:<name>', got {gating!r}"
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
# A producer is LIVE (tracked at HEAD) or RETIRED (deleted, but in HEAD's
# history). An artifact records a run that happened; retiring the script that
# ran it does not unmake the run, so evidence is held to this repository's
# history, never to today's tree alone.
# --------------------------------------------------------------------------- #
def historical_sha256s(repo_root: Path, path: str) -> frozenset[str]:
    """The sha256 of every version of `path` committed in HEAD's history —
    empty when no commit ever tracked it."""
    commits = _run(["git", "log", "--format=%H", "--", path], repo_root).stdout.split()
    blobs = (
        subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repo_root, capture_output=True)
        for commit in commits
    )
    return frozenset(hashlib.sha256(blob.stdout).hexdigest() for blob in blobs if blob.returncode == 0)


# --------------------------------------------------------------------------- #
# rule (b) — producer.path is tracked, now or in this history
# --------------------------------------------------------------------------- #
def check_producer_path(producer: dict, repo_root: Path, tracked: set[str]) -> list[str]:
    path = producer.get("path")
    if not isinstance(path, str) or not path:
        return []
    if path in tracked:
        return []
    if (repo_root / path).is_file():
        return [f"producer.path `{path}` is not `git ls-files`-tracked"]
    if historical_sha256s(repo_root, path):
        return []
    return [f"producer.path `{path}` does not exist on disk, and no commit in this history ever tracked it"]


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

# The fixed marker a producer stamps to declare "regeneration provenance is
# CONTENT identity, not a git commit sha" (rule (j)). Never a free-form
# string — a single closed value, so a typo or a half-adopted convention
# reads as "wrong marker", never as a silently-accepted new spelling.
PRODUCER_SOURCE_IDENTITY_MARKER = "source_sha256+input_manifest"

# Producer paths whose OWN module doc declares the source-identity
# convention (`producer.identity == PRODUCER_SOURCE_IDENTITY_MARKER`) —
# reviewed, closed set, same shape as `LEGACY_NONE_ALLOWLIST` above. A
# producer added here MUST stamp the marker on every artifact it emits from
# now on; this is the enforcement half of that promise, not a suggestion.
SOURCE_IDENTITY_DECLARING_PRODUCER_PATHS: frozenset[str] = frozenset(
    {
        # `producer.path` on the committed source-identity artifacts
        # names the LEG DRIVER script, not the artifact-assembling
        # module (the schema's `producer.path` field is "the thing
        # that ran", which for a `kind: "script"` producer is the driver a
        # human invoked, not every module that driver's pipeline imports).
        "ci/scripts/perf/profile_421_legs.sh",
    }
)

# A SECOND, INDEPENDENT anchor for the same marker-mandatory arm
# (`check_producer_source_identity_marker`): `producer.path` is a
# self-declared, free-form string a later regeneration could rename with no
# other signal changing at all, so membership in
# `SOURCE_IDENTITY_DECLARING_PRODUCER_PATHS` alone is not enough to catch a
# renamed producer. An artifact's own committed FILENAME (never
# `producer.path`) is not something a producer's own code can rename without
# also renaming what CI actually sees on disk — every "tower-profile" family
# artifact (`-profile-<N>-towers-...`, for any N —
# `2026-09-07-profile-421-towers-...json` is one instance) and every
# "frontend" artifact (`-frontend-...`) is matched here by FAMILY TOKEN,
# never by one particular N: a later profile artifact with a different N
# must be caught without an edit here. The REQUIRED `-towers-` token (never
# a bare `-profile-\d+-`) is not optional:
# `2026-08-31-profile-356-closeout-...json` also matches `-profile-\d+-` but
# is NOT a tower-profile source-identity-declaring artifact at all — a bare
# `-profile-\d+-` would wrongly demand the marker on that unrelated
# artifact. Matched against the artifact's own BASENAME ONLY
# (`relpath.rsplit("/", 1)[-1]`, never the full `relpath`) — `relpath` can
# carry directory segments (a `*-raw-runs/` subdirectory name, say) that
# have nothing to do with THIS file's own identity, and matching the full
# path would let an unrelated ancestor directory's name decide whether THIS
# artifact is in the mandatory arm.
SOURCE_IDENTITY_DECLARING_FILENAME_RE = re.compile(r"-profile-\d+-towers-|-frontend-")


# --------------------------------------------------------------------------- #
# rule (j) — producer.source_sha256: content identity this history can show
# --------------------------------------------------------------------------- #
def check_producer_source_sha256(producer: dict, repo_root: Path) -> list[str]:
    """`producer.source_sha256` (OPTIONAL — carried by a producer whose
    regeneration provenance is content identity rather than a commit sha) is
    a `{<repo-root-relative path>: <sha256 hex>}` map naming every file whose
    BYTES could change a NUMBER that producer emitted. Each recorded hash
    must be the sha256 of some committed version of that path in this
    history: the artifact states what it was rendered from, and the
    repository can show it. A path that has since changed, or been deleted
    with its retired producer, leaves the record true; a hash no commit ever
    carried is refused BY NAME."""
    source_sha256 = producer.get("source_sha256")
    if source_sha256 is None:
        return []
    if not isinstance(source_sha256, dict) or not source_sha256:
        return [f"producer.source_sha256 must be a non-empty object, got {source_sha256!r}"]
    failures: list[str] = []
    for path, expected in sorted(source_sha256.items()):
        if not isinstance(path, str) or not path:
            failures.append(f"producer.source_sha256 has a non-string/empty path key {path!r}")
        elif not isinstance(expected, str) or not _SHA256_HEX_RE.match(expected):
            failures.append(f"producer.source_sha256[{path!r}] must be a 64-lowercase-hex sha256, got {expected!r}")
        elif expected not in historical_sha256s(repo_root, path):
            failures.append(
                f"producer.source_sha256[{path!r}] = {expected}: no committed version of `{path}` in this "
                "history has that sha256 — the artifact names bytes this repository cannot show"
            )
    return failures


def check_producer_input_sha256(producer: dict) -> list[str]:
    """`producer.input_sha256` (OPTIONAL — `{<input label>: <sha256 hex>}`,
    e.g. `{"merge_json": ..., "attribution_json": ..., "identity": ...}`):
    the sha256 of the producer's own INPUT files as
    given at render time. Unlike `source_sha256` (repo-root-relative PATHS,
    re-hashable against this gate's own HEAD), an input's label is not a
    path this gate can re-resolve on its own — the recorded value is
    shape-checked here (a non-empty object of non-empty-string-keyed,
    64-lowercase-hex values), never re-hashed against a file, which is the
    one difference from rule (j)'s own `check_producer_source_sha256`."""
    input_sha256 = producer.get("input_sha256")
    if input_sha256 is None:
        return []
    if not isinstance(input_sha256, dict) or not input_sha256:
        return [f"producer.input_sha256 must be a non-empty object, got {input_sha256!r}"]
    failures: list[str] = []
    for label, value in sorted(input_sha256.items()):
        if not isinstance(label, str) or not label:
            failures.append(f"producer.input_sha256 has a non-string/empty label key {label!r}")
            continue
        if not isinstance(value, str) or not _SHA256_HEX_RE.match(value):
            failures.append(f"producer.input_sha256[{label!r}] must be a 64-lowercase-hex sha256, got {value!r}")
    return failures


def check_producer_source_identity_marker(producer: dict, relpath: str) -> list[str]:
    """`producer.identity`, when present, must equal
    `PRODUCER_SOURCE_IDENTITY_MARKER` and license BOTH `source_sha256` and
    `input_sha256` to be present (a marker with only one of the two blocks
    is an incomplete identity claim — refused BY NAME, never quietly
    downgraded to "no identity was declared here"). Absent the marker,
    `source_sha256`/`input_sha256` stay entirely OPTIONAL — UNLESS one of
    THREE independent anchors fires: `producer.path` is itself one of the
    reviewed, closed `SOURCE_IDENTITY_DECLARING_PRODUCER_PATHS`; the
    artifact ALREADY carries a non-empty `producer.source_sha256` block (it
    has, by its own content, declared the convention regardless of what
    `producer.path` says today); or this artifact's own committed FILENAME
    (`relpath`, never `producer.path`) matches a known profile/frontend
    artifact family (`SOURCE_IDENTITY_DECLARING_FILENAME_RE`). A producer
    this repo already knows declares the convention omitting the marker on
    one of its own artifacts is exactly the silent-regression rule (j)'s own
    docstring warns about (a future edit that regenerates without
    re-stamping the marker, OR renames `producer.path` away from the
    reviewed allowlist, would otherwise fall back to the unchecked "no
    identity" state unnoticed — the source_sha256/filename anchors close
    exactly that escape hatch)."""
    identity = producer.get("identity")
    path = producer.get("path")
    if identity is None:
        reasons: list[str] = []
        if isinstance(path, str) and path in SOURCE_IDENTITY_DECLARING_PRODUCER_PATHS:
            reasons.append(
                f"producer.path `{path}` is a known source-identity-declaring producer "
                "(SOURCE_IDENTITY_DECLARING_PRODUCER_PATHS)"
            )
        if isinstance(producer.get("source_sha256"), dict) and producer.get("source_sha256"):
            reasons.append("this artifact already carries a non-empty producer.source_sha256 block")
        # `relpath` is a posix-style, repo-cuda-runs-relative path
        # (`Path.relative_to(...).as_posix()` at the call site) that can
        # carry directory segments (e.g. a `*-raw-runs/` subdirectory name)
        # with nothing to do with THIS file's own identity — matched
        # against the BASENAME only, never the full path.
        basename = relpath.rsplit("/", 1)[-1]
        if SOURCE_IDENTITY_DECLARING_FILENAME_RE.search(basename):
            reasons.append(
                f"this artifact's own filename `{basename}` matches a known profile/frontend artifact family "
                "(SOURCE_IDENTITY_DECLARING_FILENAME_RE)"
            )
        if reasons:
            return [
                "; ".join(reasons) + " — but this artifact carries no producer.identity marker: every artifact "
                "from a source-identity-declaring producer (by producer.path, by already carrying "
                "source_sha256, or by its own filename family) must stamp producer.identity = "
                f"{PRODUCER_SOURCE_IDENTITY_MARKER!r}"
            ]
        return []
    failures: list[str] = []
    if identity != PRODUCER_SOURCE_IDENTITY_MARKER:
        failures.append(
            f"producer.identity must be {PRODUCER_SOURCE_IDENTITY_MARKER!r} when present, got {identity!r}"
        )
    if not isinstance(producer.get("source_sha256"), dict) or not producer.get("source_sha256"):
        failures.append(
            f"producer.identity == {PRODUCER_SOURCE_IDENTITY_MARKER!r} but producer.source_sha256 is "
            "missing/empty — the marker declares BOTH blocks present, never just one"
        )
    if not isinstance(producer.get("input_sha256"), dict) or not producer.get("input_sha256"):
        failures.append(
            f"producer.identity == {PRODUCER_SOURCE_IDENTITY_MARKER!r} but producer.input_sha256 is "
            "missing/empty — the marker declares BOTH blocks present, never just one"
        )
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


def _manifest_at(repo_root: Path, rev: str, rel_path: str) -> tuple[str, str] | None:
    """The nearest `Cargo.toml` above `rel_path` at commit `rev`, as
    `(its path, its content)`."""
    parent = Path(rel_path).parent
    for directory in [parent, *parent.parents]:
        candidate = (directory / "Cargo.toml").as_posix()
        text = _file_at(repo_root, rev, candidate)
        if text is not None:
            return candidate, text
    return None


def _test_target_has_required_features(cargo_toml_text: str, test_stem: str) -> bool:
    return bool(_test_target_required_features(cargo_toml_text, test_stem))


def _features_table(cargo_toml_text: str) -> str:
    """The body of a manifest's `[features]` table."""
    m = re.search(r"(?ms)^\[features\]\s*$(.*?)(?=^\[|\Z)", cargo_toml_text)
    return m.group(1) if m else ""


def _test_target_required_features(cargo_toml_text: str, test_stem: str) -> list[str]:
    """The `required-features` of the `[[test]]` target named `test_stem`."""
    for block in re.split(r"(?m)^\[\[test\]\]\s*$", cargo_toml_text)[1:]:
        end = re.search(r"(?m)^\[", block)
        body = block[: end.start()] if end else block
        name_m = re.search(r'name\s*=\s*"([^"]+)"', body)
        if name_m and name_m.group(1) == test_stem:
            req = re.search(r"required-features\s*=\s*\[([^\]]*)\]", body)
            return re.findall(r'"([^"]+)"', req.group(1)) if req else []
    return []


def _file_at(repo_root: Path, rev: str, rel_path: str) -> str | None:
    """`rel_path`'s content at commit `rev`, or `None` when that commit does
    not hold it."""
    shown = subprocess.run(
        ["git", "show", f"{rev}:{rel_path}"], cwd=repo_root, capture_output=True
    )
    if shown.returncode != 0:
        return None
    return shown.stdout.decode("utf-8", errors="replace")


def _measured_tree(repo_root: Path, data: dict) -> str | None:
    """The commit holding the tree the artifact measured: its `git_sha` when this
    repository has that commit, else the `merged_as` commit that landed it."""
    for key in ("git_sha", "merged_as"):
        rev = data.get(key)
        if isinstance(rev, str) and rev:
            exists = subprocess.run(
                ["git", "cat-file", "-e", f"{rev}^{{commit}}"], cwd=repo_root, capture_output=True
            )
            if exists.returncode == 0:
                return rev
    return None


def check_cargo_test_gating(data: dict, producer: dict, repo_root: Path) -> list[str]:
    """The artifact's recorded gating is how its test was gated WHEN IT RAN, so
    it is checked against the test file (and its crate's manifest) at the
    artifact's own `git_sha`, never against today's tree: a later change to how
    tests are gated leaves every earlier record true."""
    path = producer.get("path")
    invocation = producer.get("invocation") or ""
    gating = producer.get("gating")
    if not isinstance(path, str) or not path:
        return ["producer.kind == 'cargo-test' requires a non-null producer.path"]
    if data.get("status") == "SUPERSEDED":
        return []  # a superseded record proves nothing; its replacement is checked
    rev = _measured_tree(repo_root, data)
    if rev is None:
        return []  # rule (a) owns a missing or unresolved sha

    m = EXACT_RE.search(invocation)
    if not m:
        return [
            f"producer.kind == 'cargo-test' but invocation `{invocation}` lacks `--exact <fn_name>` "
            "— cannot statically verify which test this artifact proves ran"
        ]
    fn_full = m.group(1)
    fn_short = fn_full.rsplit("::", 1)[-1]

    source = _file_at(repo_root, rev, path)
    if source is None:
        return [f"producer.path `{path}` does not exist at the artifact's git_sha {rev}"]
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
        manifest = _manifest_at(repo_root, rev, path)
        if manifest is None:
            failures.append(
                f"no Cargo.toml above {path} at {rev}; cannot verify required-features"
            )
        else:
            manifest_path, cargo_toml = manifest
            test_stem = Path(path).stem
            if not _test_target_has_required_features(cargo_toml, test_stem):
                failures.append(
                    f"producer claims gating 'required-features' but {manifest_path} at {rev} "
                    f"has no `[[test]]` section named `{test_stem}` carrying `required-features`"
                )
    elif isinstance(gating, str) and gating.startswith("feature:"):
        feature = gating.split(":", 1)[1]
        manifest = _manifest_at(repo_root, rev, path)
        declared = manifest is not None and re.search(
            rf'(?m)^{re.escape(feature)}\s*=', _features_table(manifest[1])
        )
        gates = f'feature = "{feature}"' in source or (
            manifest is not None
            and feature in _test_target_required_features(manifest[1], Path(path).stem)
        )
        if not (declared and gates):
            failures.append(
                f"producer claims gating '{gating}' but at {rev} the crate does not declare "
                f"`{feature}`, or neither {path} nor its `[[test]]` target is gated on it"
            )
    # gating == "none": nothing further to verify.

    return failures


# --------------------------------------------------------------------------- #
# rule (d) — ancestry (delegates entirely to the ONE shared rule in
# ancestry.py)
# --------------------------------------------------------------------------- #
_is_ancestor = ancestry.is_ancestor


def check_ancestry(data: dict, repo_root: Path) -> list[str]:
    """PASS if `git_sha` is an ancestor of HEAD, OR — for a branch tip that
    was rewritten (rebased or squash-merged) before its own landing, so
    `git_sha` itself can never be an ancestor of anything again — if
    `merged_as` is an ancestor of HEAD and the artifact also carries
    `git_sha` (the measured tip, kept verbatim) plus `merged_via_pr`.
    `merged_as`'s own well-typedness (40-hex, paired with `merged_via_pr`,
    only valid alongside a resolved `git_sha`) is rule (a)'s job
    (`check_schema_types`); `ancestry.check_ancestry` only re-checks
    GIT_SHA_RE on `merged_as` itself, so a malformed one cannot be handed
    to `git merge-base` as a literal ref expression."""
    sha = data.get("git_sha")
    if not sha:
        return []  # git_sha_unresolved artifacts have nothing resolvable to check
    return ancestry.check_ancestry(data, repo_root, ANCESTOR_MESSAGE)


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
# rule (g) — `oracle_separation` (OPTIONAL per artifact leg)
# --------------------------------------------------------------------------- #
# `docs/maintainer/cuda-kernel-guide.md` §3's separation-in-the-artifact
# rule (an instance of §3.8's "no absolute ULP floor" discipline): a bound is not evidence of
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
# OPTIONAL — absent on artifacts that do not carry the block; where
# present, it is checked.
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


# --------------------------------------------------------------------------- #
# rule (k) — the `topology` artifact kind (the GPU topology lane's evidence).
#
# The fine-tune gang on real GPUs at every topology the engine lays a job out
# as, through the product path, proves something no single-device artifact
# can, with a payload of its own: which one-host proofs passed, and a
# multi-host fleet run under each transport — where every rank ran (instance,
# label, machine), which transport the coordinator selected, the per-epoch
# loss, the probe embeddings' digest — plus the single-rank reference at the
# same global batch and the ε the gang's loss is read against.
#
# The claims are CROSS-FIELD and bind on a `pass`, because a `pass` is a
# CLAIM: the two transports' curves and embeddings are identical (the
# transport is invisible in the result), the gang's ranks span every host,
# one process each, and the gang's loss is within ε of the single rank's. A
# `fail` is REPRESENTABLE — admitted as measured, with its reasons and a
# top-level `status` that is not GREEN — because a failing run's own numbers
# are the evidence that has to survive into the repository.
#
# ε is the trainer's pre-registered W-vs-1 bound (`trainer.rs`'s
# `gather_exactness_w2_matches_w1_within_pre_registered_epsilon`, mirrored by
# `jammi-server`'s `gpu::topology` tests and `gpu_topology_assemble.py`): an
# artifact carries the value it was read against, and a value other than the
# registered one is refused — an ε chosen after seeing the delta it excuses is
# not a tolerance.
#
# LETTER: (k). As with every other letter here, it is comment/self-test-label
# prose only — no gate, allowlist, or error message parses it.
#
# WHY A REGISTRY, NOT AN INLINE LITERAL: a new required field lands as a
# registry ROW (the discipline `_TIER_SOURCE_REGISTRY` follows), so the field
# set, its validator and the REASON it is required sit in one table.
# --------------------------------------------------------------------------- #
TOPOLOGY_ARTIFACT_KIND = "topology"
ARTIFACT_KIND_KEY = "artifact_kind"
TOPOLOGY_BLOCK_KEY = "topology"
TOPOLOGY_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

# The lane's one driver is the sole writer of this kind.
TOPOLOGY_PRODUCER_PATH = "ci/scripts/runpod_gpu_topology.sh"

# Third anchor (rule (j)'s three-anchor shape): a committed artifact whose
# FILENAME declares the family cannot escape this rule by dropping its kind.
TOPOLOGY_ARTIFACT_FILENAME_RE = re.compile(r"(?:^|[-_])topology(?:[-_.]|$)")

# The multi-host cell proves a fleet of MORE than one host with MORE than one
# GPU each — fewer is a topology the one-host cell already covers.
TOPOLOGY_MIN_HOSTS = 2
TOPOLOGY_MIN_GPUS_PER_HOST = 2

# The trainer's pre-registered W-vs-1 loss bound (see the section comment).
TOPOLOGY_REGISTERED_EPSILON = 1e-4

# The transport each fleet run's coordinator must have selected: the run's
# key is the `[worker] collective` phase it ran under.
TOPOLOGY_RUN_TRANSPORTS = {"nccl": "Nccl", "inline": "Inline"}

TOPOLOGY_VERDICT_PASS = "pass"
TOPOLOGY_VERDICT_FAIL = "fail"
TOPOLOGY_VERDICTS = (TOPOLOGY_VERDICT_PASS, TOPOLOGY_VERDICT_FAIL)

# The one top-level `status` a `fail` may not carry (`status` is free text
# repo-wide; this refuses the one contradiction it can state).
TOPOLOGY_GREEN_STATUS = "GREEN"


def _is_real_number(value) -> bool:
    """A JSON number that is not a bool (`isinstance(True, int)` is True in
    Python) and not a NaN/Infinity (`json.load` accepts both by default)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and value not in (float("inf"), float("-inf"))


def _is_count(value, minimum: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _topology_check_shape(topo: dict, _data: dict) -> list[str]:
    failures: list[str] = []
    if not _is_count(topo.get("hosts"), TOPOLOGY_MIN_HOSTS):
        failures.append(
            f"`topology.hosts` must be an integer >= {TOPOLOGY_MIN_HOSTS} — the multi-host cell "
            f"spans machines, got {topo.get('hosts')!r}"
        )
    if not _is_count(topo.get("gpus_per_host"), TOPOLOGY_MIN_GPUS_PER_HOST):
        failures.append(
            f"`topology.gpus_per_host` must be an integer >= {TOPOLOGY_MIN_GPUS_PER_HOST}, got "
            f"{topo.get('gpus_per_host')!r}"
        )
    return failures


def _topology_check_one_host(topo: dict, _data: dict) -> list[str]:
    one_host = topo.get("one_host")
    if not isinstance(one_host, dict):
        return [f"`topology.one_host` must be an object, got {one_host!r}"]
    failures: list[str] = []
    for key in ("device_tests", "product_tests"):
        tests = one_host.get(key)
        if not isinstance(tests, list) or not tests or not all(
            isinstance(t, str) and t.strip() for t in tests
        ):
            failures.append(
                f"`topology.one_host.{key}` must be a non-empty list of the test names that passed, "
                f"got {tests!r}"
            )
    return failures


def _topology_check_nccl_version(topo: dict, _data: dict) -> list[str]:
    versions = topo.get("nccl_version")
    if not isinstance(versions, list) or len(versions) != 1 or not isinstance(versions[0], str):
        return [
            "`topology.nccl_version` must list exactly the one NCCL runtime every server reported, "
            f"got {versions!r}"
        ]
    return []


def _topology_check_run(name: str, run, world, hosts) -> list[str]:
    where = f"`topology.fleet.{name}`"
    if not isinstance(run, dict):
        return [f"{where} must be an object, got {run!r}"]
    failures: list[str] = []
    if run.get("transport") != TOPOLOGY_RUN_TRANSPORTS[name]:
        failures.append(
            f"{where}.transport must be {TOPOLOGY_RUN_TRANSPORTS[name]!r} — the transport the "
            f"coordinator selected under that phase, got {run.get('transport')!r}"
        )
    if not _is_count(run.get("attempts"), 1):
        failures.append(f"{where}.attempts must be an integer >= 1, got {run.get('attempts')!r}")
    ranks = run.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        return failures + [f"{where}.ranks must be a non-empty list, got {ranks!r}"]
    if _is_count(world, 1) and len(ranks) != world:
        failures.append(f"{where}.ranks carries {len(ranks)} rank(s) for a world of {world}")
    instances, machines = [], set()
    for i, rank in enumerate(ranks):
        if not isinstance(rank, dict) or not all(
            isinstance(rank.get(k), str) and rank[k].strip() for k in ("instance", "host")
        ):
            failures.append(f"{where}.ranks[{i}] must name its `instance` and `host`, got {rank!r}")
            continue
        instances.append(rank["instance"])
        machines.add(rank["host"])
    if len(set(instances)) != len(instances):
        failures.append(f"{where}.ranks repeats an instance — a fleet rank is one process")
    if _is_count(hosts, 1) and len(machines) != hosts:
        failures.append(f"{where}.ranks span {len(machines)} host(s), not the fleet's {hosts}")
    curve = run.get("loss_curve")
    if not isinstance(curve, list) or not curve or not all(_is_real_number(v) for v in curve):
        failures.append(f"{where}.loss_curve must be a non-empty list of finite losses, got {curve!r}")
    if not isinstance(run.get("embeddings_sha256"), str) or not TOPOLOGY_DIGEST_RE.match(
        run["embeddings_sha256"]
    ):
        failures.append(f"{where}.embeddings_sha256 must be a 64-lowercase-hex digest")
    return failures


def _topology_check_fleet(topo: dict, _data: dict) -> list[str]:
    fleet = topo.get("fleet")
    if not isinstance(fleet, dict):
        return [f"`topology.fleet` must be an object, got {fleet!r}"]
    failures: list[str] = []
    hosts, per_host, world = topo.get("hosts"), topo.get("gpus_per_host"), fleet.get("world_size")
    if _is_count(hosts, 1) and _is_count(per_host, 1) and world != hosts * per_host:
        failures.append(
            f"`topology.fleet.world_size` ({world!r}) must be `hosts` x `gpus_per_host` "
            f"({hosts * per_host}) — one rank per GPU of the fleet"
        )
    for name in TOPOLOGY_RUN_TRANSPORTS:
        failures.extend(_topology_check_run(name, fleet.get(name), world, hosts))
    reference = fleet.get("reference")
    curve = reference.get("loss_curve") if isinstance(reference, dict) else None
    if not isinstance(curve, list) or not curve or not all(_is_real_number(v) for v in curve):
        failures.append(
            f"`topology.fleet.reference.loss_curve` must be the single rank's per-epoch loss, got {curve!r}"
        )
    if fleet.get("epsilon") != TOPOLOGY_REGISTERED_EPSILON:
        failures.append(
            f"`topology.fleet.epsilon` must be the trainer's registered {TOPOLOGY_REGISTERED_EPSILON} "
            f"— an ε chosen after the measurement is not a tolerance, got {fleet.get('epsilon')!r}"
        )
    delta = fleet.get("max_loss_delta_vs_reference")
    if not _is_real_number(delta) or delta < 0:
        failures.append(
            f"`topology.fleet.max_loss_delta_vs_reference` must be a finite number >= 0, got {delta!r}"
        )
    return failures


def _topology_check_verdict(topo: dict, data: dict) -> list[str]:
    verdict, reasons = topo.get("verdict"), topo.get("reasons")
    if verdict not in TOPOLOGY_VERDICTS:
        return [f"`topology.verdict` must be exactly one of {list(TOPOLOGY_VERDICTS)}, got {verdict!r}"]
    if not isinstance(reasons, list) or not all(isinstance(r, str) and r.strip() for r in reasons):
        return [f"`topology.reasons` must be a list of non-empty strings, got {reasons!r}"]
    if verdict == TOPOLOGY_VERDICT_PASS and reasons:
        return [f"`topology.verdict` is {TOPOLOGY_VERDICT_PASS!r} but `topology.reasons` names failures"]
    if verdict == TOPOLOGY_VERDICT_FAIL:
        failures = []
        if not reasons:
            failures.append(f"`topology.verdict` is {TOPOLOGY_VERDICT_FAIL!r} with no `topology.reasons`")
        status = data.get("status")
        if isinstance(status, str) and status.strip().upper() == TOPOLOGY_GREEN_STATUS:
            failures.append(
                f"`topology.verdict` is {TOPOLOGY_VERDICT_FAIL!r} but the top-level `status` is "
                f"{TOPOLOGY_GREEN_STATUS}"
            )
        return failures
    return []


# key -> (validator, why the field is required).
TOPOLOGY_FIELD_REGISTRY: tuple = (
    ("hosts", _topology_check_shape, "the fleet's shape: how many machines, how many GPUs each"),
    ("gpus_per_host", _topology_check_shape, "the fleet's shape: how many machines, how many GPUs each"),
    ("one_host", _topology_check_one_host, "the one-host cell's passed tests, device and product"),
    ("nccl_version", _topology_check_nccl_version, "the NCCL runtime the fleet's communicators ran"),
    ("fleet", _topology_check_fleet, "the multi-host runs, the reference and ε"),
    ("verdict", _topology_check_verdict, "the lane's own call; a failing run is representable"),
    ("reasons", _topology_check_verdict, "what failed, empty on a pass"),
)


def _topology_check_claims(topo: dict) -> list[str]:
    """On a `pass`: the transports agree byte for byte and the gang is within
    ε of the single rank. Only read once the registry validated the parts."""
    fleet = topo["fleet"]
    nccl, inline = fleet["nccl"], fleet["inline"]
    failures: list[str] = [
        f"`topology.fleet.{name}` records a pass whose gang took {fleet[name]['attempts']} attempts "
        "— a retry that published hides a hang"
        for name in TOPOLOGY_RUN_TRANSPORTS
        if fleet[name]["attempts"] != 1
    ]
    if nccl["loss_curve"] != inline["loss_curve"]:
        failures.append("`topology.fleet` records a pass whose two transports' loss curves differ")
    if nccl["embeddings_sha256"] != inline["embeddings_sha256"]:
        failures.append("`topology.fleet` records a pass whose two transports' embeddings differ")
    if fleet["max_loss_delta_vs_reference"] > fleet["epsilon"]:
        failures.append(
            f"`topology.fleet.max_loss_delta_vs_reference` ({fleet['max_loss_delta_vs_reference']}) "
            f"exceeds ε ({fleet['epsilon']}) on a pass — record it as {TOPOLOGY_VERDICT_FAIL!r}"
        )
    return failures


def topology_anchors(data: dict, relpath: str) -> list[str]:
    """Which independent anchor(s) declare this a topology artifact — three,
    so dropping any ONE does not return it to the unchecked state."""
    anchors: list[str] = []
    if data.get(ARTIFACT_KIND_KEY) == TOPOLOGY_ARTIFACT_KIND:
        anchors.append(f"{ARTIFACT_KIND_KEY} == {TOPOLOGY_ARTIFACT_KIND!r}")
    if isinstance(data.get(TOPOLOGY_BLOCK_KEY), dict):
        anchors.append(f"a top-level `{TOPOLOGY_BLOCK_KEY}` block")
    if TOPOLOGY_ARTIFACT_FILENAME_RE.search(relpath.rsplit("/", 1)[-1]):
        anchors.append("the committed filename (TOPOLOGY_ARTIFACT_FILENAME_RE)")
    return anchors


def check_topology_artifact(data: dict, relpath: str, _repo_root: Path) -> list[str]:
    """Rule (k): an artifact declared `topology` by ANY anchor carries the
    whole registry payload, names the lane's driver as its producer, and —
    on a `pass` — satisfies the cross-field claims."""
    anchors = topology_anchors(data, relpath)
    if not anchors:
        return []
    if data.get(ARTIFACT_KIND_KEY) != TOPOLOGY_ARTIFACT_KIND:
        return [
            f"declared a topology artifact by {anchors[0]} but `{ARTIFACT_KIND_KEY}` is "
            f"{data.get(ARTIFACT_KIND_KEY)!r} — a topology artifact names its own kind"
        ]
    topo = data.get(TOPOLOGY_BLOCK_KEY)
    if not isinstance(topo, dict):
        return [
            f"`{ARTIFACT_KIND_KEY}` is {TOPOLOGY_ARTIFACT_KIND!r} but there is no `{TOPOLOGY_BLOCK_KEY}` "
            f"object carrying {', '.join(f'`{k}`' for k, _v, _w in TOPOLOGY_FIELD_REGISTRY)}"
        ]
    failures: list[str] = []
    producer = data.get("producer")
    if not isinstance(producer, dict) or producer.get("path") != TOPOLOGY_PRODUCER_PATH:
        failures.append(
            f"a topology artifact requires `producer.path` == {TOPOLOGY_PRODUCER_PATH!r} — the "
            "lane's driver is the sole writer of this kind"
        )
    missing = [(k, why) for k, _v, why in TOPOLOGY_FIELD_REGISTRY if k not in topo]
    failures.extend(f"`{TOPOLOGY_BLOCK_KEY}.{k}` is missing — {why}" for k, why in missing)
    if missing:
        return failures
    part_failures: list[str] = []
    for validator in dict.fromkeys(v for _k, v, _w in TOPOLOGY_FIELD_REGISTRY):
        part_failures.extend(validator(topo, data))
    failures.extend(part_failures)
    if not part_failures and topo["verdict"] == TOPOLOGY_VERDICT_PASS:
        failures.extend(_topology_check_claims(topo))
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
    """The anchor for `GATE_INTRODUCTION_SHA`, otherwise unverified: it is a
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
    Editing `GATE_INTRODUCTION_SHA` at all is an edit to this gate file,
    reviewed like every other change to this file's own rules.

    Checked BEFORE any `git log --follow` work, same discipline as
    `run_gate`'s own shallow guard: a shallow checkout (`actions/checkout`'s
    default `fetch-depth: 1`, or any `--depth 1` clone) makes `git log
    --follow --diff-filter=A` read back only the single fetched commit —
    on THIS gate file that commit is whatever HEAD happens to be, which is
    never `GATE_INTRODUCTION_SHA` (a much older commit), so an unguarded
    shallow call here would always misreport a false "does not match"
    finding, indistinguishable from a genuine repoint without checking
    first. `--self-test` calls this against the REAL checkout (never a
    fixture repo — this gate file itself is not tracked inside a `--self-
    test` fixture), so it needs this guard even though `run_gate`'s own
    (plain-mode) shallow check never reaches this function at all (it
    raises `ArtifactError` first)."""
    gate_file = gate_file if gate_file is not None else Path(__file__).resolve()
    if is_shallow_repository(repo_root):
        return [SHALLOW_CHECKOUT_MESSAGE]
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
    """Rule (f)'s mechanical companion: an entry
    in `LEGACY_NONE_ALLOWLIST` is legitimate ONLY if the artifact it names
    was first committed (under this exact path, following renames) BEFORE
    this gate itself existed. A genuinely NEW artifact's first-introduction
    commit can never predate `gate_introduction_sha` (the gate did not exist
    yet when a truly pre-schema artifact was added, but it DOES exist by the
    time any new commit lands), so this list cannot grow again without a
    gate edit AND a history it does not have.
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
# rule (i) — leg identity on self-declaring v2 legs. A SEPARATE, unrelated
# mechanism from rule (g) above (`oracle_separation`).
#
# A v2 leg is ANY JSON object, anywhere in a `cuda-runs/**` tree,
# carrying `leg_schema_version >= 2`; no v1 leg carries that key, so none
# can satisfy it by accident. A leg WITHOUT the key is v1 and is validated
# only by rules (a)-(f) above.
#
# The required identity TUPLE per (tier, producer_kind) is never hand-typed
# here: a workload's identity is declared ONCE, in the Rust payload type's
# `Payload::IDENTITY_FIELDS` (`crates/jammi-bench/src/report.rs`), and every
# producer's leg — the engine's own and another framework's alike — is held
# to that one declaration at its tier root. The engine's legs additionally
# carry `REPORT_IDENTITY_FIELDS` under `provenance`; another framework's leg
# carries its own `provenance.git_rev`, nullable.
# --------------------------------------------------------------------------- #
LEG_SCHEMA_VERSION_KEY = "leg_schema_version"
RAW_RUNS_DIR_SUFFIX = "-raw-runs"

# CLOSED — exactly the 10 `*.json.raw` files committed (`6d07b20`) AFTER
# this gate existed (`c7fd1df`), named `.json.raw` rather than fabricate a
# `git_sha` (the parent artifact's own `provenance_note` records why).
# Every one is a bare `Report` dump ({engine_version, host,
# subcommand, tiers}) with no schema/provenance fields at all — they cannot
# be brought under rules (a)-(f), let alone rule (i), without inventing a
# sha the run never resolved. `--self-test` (`check_legacy_raw_nonjson_files_
# exist`) proves every listed relpath still exists; a deletion must shrink
# this list in the SAME commit, and growth is a gate edit (the list is
# closed, not merely long).
LEGACY_RAW_NONJSON: dict[str, str] = {
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s128_disabled.r1.json.raw": (
        "pre-rule-(i) raw leg (a100b box, s128, disabled arm, replicate 1): bare Report "
        "dump committed before rule (i) existed; kept .json.raw per this parent "
        "artifact's own provenance_note (a100b_full_step_ab_reference) rather than "
        "fabricate a git_sha for a tip not resolvable against this worktree's ancestry."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s128_disabled.r2.json.raw": (
        "pre-rule-(i) raw leg (a100b box, s128, disabled arm, replicate 2): same "
        "provenance_note as the r1 sibling above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s128_fused.r1.json.raw": (
        "pre-rule-(i) raw leg (a100b box, s128, fused arm, replicate 1): same "
        "provenance_note as the disabled-arm siblings above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s128_fused.r2.json.raw": (
        "pre-rule-(i) raw leg (a100b box, s128, fused arm, replicate 2): same "
        "provenance_note as the disabled-arm siblings above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s512_disabled.r1.json.raw": (
        "pre-rule-(i) raw leg (a100b box, s512, disabled arm, replicate 1): same "
        "provenance_note as the s128 siblings above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s512_disabled.r2.json.raw": (
        "pre-rule-(i) raw leg (a100b box, s512, disabled arm, replicate 2): same "
        "provenance_note as the s128 siblings above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s512_fused.r1.json.raw": (
        "pre-rule-(i) raw leg (a100b box, s512, fused arm, replicate 1): same "
        "provenance_note as the s128 siblings above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s512_fused.r2.json.raw": (
        "pre-rule-(i) raw leg (a100b box, s512, fused arm, replicate 2): same "
        "provenance_note as the s128 siblings above."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/r2/b8_s128_disabled.json.raw": (
        "pre-rule-(i) raw leg (r2 box, s128, disabled arm, single replicate): a separate "
        "confirmation session, same 'kept .json.raw rather than fabricate a git_sha' "
        "reasoning as the a100b/ siblings above — see this dir's own PROVENANCE.md."
    ),
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs/r2/b8_s128_fused.json.raw": (
        "pre-rule-(i) raw leg (r2 box, s128, fused arm, single replicate): same "
        "PROVENANCE.md reasoning as the r2/ sibling above."
    ),
}
_ABSENT = object()  # sentinel: key genuinely absent from the object (never confused with JSON null)

_JAMMI_REPORT_RS = REPO_ROOT / "crates" / "jammi-bench" / "src" / "report.rs"

_TIER_IDENTITY_FIELDS_BLOCK_RE = re.compile(
    r"const IDENTITY_FIELDS:\s*&'static \[\(&'static str,\s*"
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


def _scoped_to_struct_impl(text: str, struct: str | None) -> str:
    """Narrows `text` to everything from a payload's own `impl Payload for
    <struct> {` marker onward — required because every payload in
    `report.rs` declares a const of the same name, so an unscoped search
    would return whichever block sits first. `struct=None` (the module-level
    `REPORT_IDENTITY_FIELDS`) leaves `text` unscoped.
    """
    if struct is None:
        return text
    anchor = f"impl Payload for {struct} {{"
    idx = text.find(anchor)
    if idx == -1:
        raise ArtifactError(f"no `{anchor}` block found — cannot scope extraction to struct {struct!r}")
    return text[idx:]


def _extract_rust_identity_block(
    path: Path, block_re: re.Pattern, struct: str | None = None
) -> list[tuple[str, str, str | None]]:
    """FIRST-MATCH extraction: `block_re.search()` on `text` (optionally
    scoped to `struct`'s own `impl` block via `_scoped_to_struct_impl` —
    see that function's own doc for why scoping is what makes "first match"
    a well-defined, per-struct answer rather than an accidental collision).
    Fails closed (`ArtifactError`) on a missing file, an unresolvable
    struct anchor, no matching const block, or a matched block naming zero
    fields — never a silent empty tuple.
    """
    if not path.is_file():
        raise ArtifactError(f"{path} does not exist — cannot derive rule (i)'s jammi identity tuple")
    text = path.read_text(encoding="utf-8")
    scoped = _scoped_to_struct_impl(text, struct)
    m = block_re.search(scoped)
    if m is None:
        where = f", scoped to `impl Payload for {struct} {{`" if struct else ""
        raise ArtifactError(f"no matching IDENTITY_FIELDS-shaped const block found in {path}{where}")
    entries = [(name, kind, reason or None) for name, kind, reason in _FIELD_ENTRY_RE.findall(m.group(1))]
    if not entries:
        raise ArtifactError(f"IDENTITY_FIELDS-shaped block in {path} matched but named zero fields")
    return entries




# --------------------------------------------------------------------------- #
# Every jammi tier rule (i) derives an identity tuple for is a row here — the
# payload struct whose `impl Payload for <struct>` block declares the tier's
# identity. A new tier lands as a new row, in the same commit as its payload.
# `torch_twin` marks the tiers another framework's script also produces:
# that producer's leg is held to the SAME identity fields at the same tier
# root, with its own nullable `provenance.git_rev` as its sha.
# --------------------------------------------------------------------------- #
_TIER_SOURCE_REGISTRY: dict[str, dict] = {
    "finetune_step": {"path": _JAMMI_REPORT_RS, "struct": "TrainStepPayload", "torch_twin": True},
    "finetune_run": {"path": _JAMMI_REPORT_RS, "struct": "TrainRunPayload", "torch_twin": False},
    "encode_step": {"path": _JAMMI_REPORT_RS, "struct": "EncodePayload", "torch_twin": False},
}

_IDENTITY_TUPLES_CACHE: dict[tuple[str, str], dict] | None = None


def build_identity_tuples() -> dict[tuple[str, str], dict]:
    """`{(tier, producer_kind): {"sha_root": ..., "sha_field": ..., "fields":
    [(name, root, "NonNull"|"NullMeans", reason_or_None), ...]}}` — computed
    once (module-level cache), every entry derived from `_TIER_SOURCE_REGISTRY`
    and the Rust declarations it names. Never hand-typed.
    """
    global _IDENTITY_TUPLES_CACHE
    if _IDENTITY_TUPLES_CACHE is not None:
        return _IDENTITY_TUPLES_CACHE

    tuples: dict[tuple[str, str], dict] = {}
    report_entries = _extract_rust_identity_block(_JAMMI_REPORT_RS, _REPORT_IDENTITY_FIELDS_BLOCK_RE)
    for tier, spec in _TIER_SOURCE_REGISTRY.items():
        tier_entries = _extract_rust_identity_block(spec["path"], _TIER_IDENTITY_FIELDS_BLOCK_RE, struct=spec["struct"])
        payload_fields = [(name, "tier", kind, reason) for name, kind, reason in tier_entries]
        jammi_fields = payload_fields + [(name, "provenance", kind, reason) for name, kind, reason in report_entries]
        tuples[(tier, "jammi")] = {"sha_root": "provenance", "sha_field": "build_sha", "fields": jammi_fields}
        if spec["torch_twin"]:
            torch_fields = payload_fields + [
                ("git_rev", "provenance", "NullMeans", "git unavailable (not on PATH, not a git worktree, or the subprocess timed out)"),
            ]
            tuples[(tier, "torch")] = {"sha_root": "provenance", "sha_field": "git_rev", "fields": torch_fields}

    _IDENTITY_TUPLES_CACHE = tuples
    return _IDENTITY_TUPLES_CACHE


def _leg_field_value(leg: dict, field: str, root: str, tier: str):
    """`root == "tier"` reads `leg["tiers"][tier][field]` — `tier` is the
    CALLER's own `identity.tier` reading, never a
    hardcoded tier name, so this same helper serves every registry row in
    `_TIER_SOURCE_REGISTRY` (`finetune_step`, `encode_step`, and whatever
    tier lands next) rather than only ever reading `tiers.finetune_step`
    regardless of which tier a leg actually declares."""
    if root == "tier":
        tiers = leg.get("tiers")
        tier_block = tiers.get(tier) if isinstance(tiers, dict) else None
        return tier_block.get(field, _ABSENT) if isinstance(tier_block, dict) else _ABSENT
    src = leg.get(root)
    return src.get(field, _ABSENT) if isinstance(src, dict) else _ABSENT


def _is_v2(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 2


def check_raw_leg_identity_fields(leg: dict, tuple_spec: dict, label: str, tier: str) -> list[str]:
    failures: list[str] = []
    for field, root, kind, _reason in tuple_spec["fields"]:
        value = _leg_field_value(leg, field, root, tier)
        if value is _ABSENT:
            failures.append(f"{label}: v2 raw leg missing identity field `{field}` (expected under `{root}`)")
        elif value is None and kind == "NonNull":
            failures.append(f"{label}: identity field `{field}` is declared NonNull but reads null")
        # kind == "NullMeans" and value is None: OK — a declared-nullable reading.
    return failures


def check_raw_leg_sha(leg: dict, tuple_spec: dict, parent_git_sha: str | None, label: str, tier: str) -> list[str]:
    if parent_git_sha is None:
        return []  # nothing to cross-check the leg's own sha against
    sha_root, sha_field = tuple_spec["sha_root"], tuple_spec["sha_field"]
    value = _leg_field_value(leg, sha_field, sha_root, tier)
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
    failures += check_raw_leg_identity_fields(leg, tuple_spec, label, tier)
    failures += check_raw_leg_sha(leg, tuple_spec, parent_git_sha, label, tier)
    return failures


def find_v2_legs(data, path_prefix: str = "") -> list[tuple[str, dict]]:
    """Recursively walks `data` for every dict carrying `leg_schema_version
    >= 2` — this is the WHOLE discriminator: a v2 leg can be
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
    required-under-a-v2-parent rule, the `--census` check — walks; none
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
    `<...>-raw-runs/a100c/leg.json`, the shape a per-box sweep's
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
    """`--census`'s check: for every top-level artifact whose OWN
    `schema_version >= 2`, every JSON object anywhere in its tree — the
    document itself and every `*.json` payload under its sibling
    `*-raw-runs/` directory — that carries `s_per_step_p50` must be
    REACHABLE as (or nested inside) a v2 leg rule (i) actually validated.
    Trivially empty today (zero `schema_version >= 2` top-level artifacts
    exist yet) — the check is standing for the day one does. Reaches
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
    the SAME commit — growth (a new entry) is a gate edit."""
    return [
        f"LEGACY_RAW_NONJSON lists `{relpath}` but no such file exists under {cuda_runs_dir}"
        for relpath in LEGACY_RAW_NONJSON
        if not (cuda_runs_dir / relpath).is_file()
    ]


def check_raw_runs_nonjson_coverage(cuda_runs_dir: Path) -> list[str]:
    """Every non-`.json` payload (excluding `.md`/`.log`) under ANY `*-raw-
    runs/` directory, at ANY depth, must be in the closed `LEGACY_RAW_NONJSON`
    list — the `.json.raw` rename bypass a NEW leg may never reuse. Walks `_find_raw_runs_dirs`'s full, name-
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
    """Under a parent artifact with
    `schema_version >= 2`, every `*.json` payload under its `*-raw-runs/`
    sibling MUST carry `leg_schema_version >= 2` (else RED) — a v1-shaped
    raw leg silently coexisting under an already-v2 parent is exactly the
    kind of container drift rule (i) exists to catch. Keys on the raw-runs
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
    failures += check_producer_source_sha256(producer, repo_root)
    failures += check_producer_input_sha256(producer)
    failures += check_producer_source_identity_marker(producer, relpath)
    failures += check_none_allowlist(data, relpath, allowlist)
    failures += check_ancestry(data, repo_root)
    failures += check_oracle_separation(data)
    failures += check_topology_artifact(data, relpath, repo_root)
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

    # rule (i) precompute: EVERY `*-raw-runs/` directory in the tree
    # (`_find_raw_runs_dirs` — name-pattern discovery, not a per-artifact
    # guess), which top-level artifact owns each one, and which of those
    # owners is already schema_version >= 2 (the "MUST carry
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

        # rule (i): every v2-leg-shaped object anywhere in THIS document
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

    # rule (f)'s mechanical companion: every LEGACY_NONE_ALLOWLIST
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
    """`--census`: lists every `s_per_step_p50`-carrying JSON
    object under a `schema_version >= 2` top-level artifact that rule (i)
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
RETIRED_PRODUCER = "ci/scripts/perf/retired_producer.sh"


def self_test() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
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
        # A tracked stand-in for the ONE known source-identity-declaring
        # producer path (`SOURCE_IDENTITY_DECLARING_PRODUCER_PATHS`) — rule
        # (j)'s own marker-mandatory arm needs `producer.path` to exist +
        # be tracked (rule (b)) so its own findings never contaminate the
        # marker-specific `expect_hit` needle below.
        (perf_dir / "profile_421_legs.sh").write_text("# stub source-identity-declaring producer\n")
        # Tracked stand-ins for the topology lane's driver (rule (k)'s
        # producer binding) and another paid driver (the binding's RED case),
        # so rule (b) — producer.path exists and is tracked — never
        # contaminates rule (k)'s own needles.
        (repo / "ci" / "scripts").mkdir(parents=True, exist_ok=True)
        (repo / "ci" / "scripts" / "runpod_gpu_topology.sh").write_text("# stub topology driver\n")
        (repo / "ci" / "scripts" / "runpod_gpu_prove.sh").write_text("# stub prove driver\n")

        # A producer committed at the root and deleted by the next commit:
        # retired, but in this history.
        (repo / RETIRED_PRODUCER).write_text("# stub retired producer\n")
        retired_hash = hashlib.sha256((repo / RETIRED_PRODUCER).read_bytes()).hexdigest()

        _run(["git", "add", "-A"], repo)
        _run(["git", "commit", "-q", "-m", "root"], repo)
        root_sha = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()

        (repo / RETIRED_PRODUCER).unlink()
        (repo / "unrelated.txt").write_text("x\n")
        _run(["git", "add", "-A"], repo)
        _run(["git", "commit", "-q", "-m", "second"], repo)
        # rule (k) needs TWO shas with a real ancestry relation between
        # them: an ε registered at `root_sha` was on record before the tree
        # `second_sha` names ever existed.
        second_sha = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()

        # rule (k)'s evidence-anchor arms need two MORE shas:
        #   `third_sha` (== HEAD) is a commit strictly AFTER `second_sha`, so
        #   an ε "registered" there is an ancestor of HEAD yet NOT an
        #   ancestor of the anchor;
        #   `orphan_sha` is a real commit object on an orphan branch, an
        #   ancestor of nothing in HEAD's history — the shape a measured tip
        #   takes once its landing commit rewrote it (the `merged_as` case).
        (repo / "unrelated2.txt").write_text("y\n")
        _run(["git", "add", "-A"], repo)
        _run(["git", "commit", "-q", "-m", "third"], repo)
        third_sha = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
        main_branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo).stdout.strip()
        _run(["git", "checkout", "-q", "--orphan", "sidebranch"], repo)
        (repo / "orphan.txt").write_text("z\n")
        _run(["git", "add", "-A"], repo)
        _run(["git", "commit", "-q", "-m", "orphan tip"], repo)
        orphan_sha = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
        _run(["git", "checkout", "-q", "-f", main_branch], repo)
        # The fixture is worthless if the checkout did not come back: every
        # anchor arm below is stated relative to HEAD == third_sha.
        if _run(["git", "rev-parse", "HEAD"], repo).stdout.strip() != third_sha:
            failures.append(
                "self-test FAILED: the fixture repo did not return to its main branch after the "
                "orphan commit — every rule (k) anchor case below is stated relative to HEAD"
            )

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

        ok = baseline()
        ok["producer"] = dict(ok["producer"])
        ok["producer"].update(path=RETIRED_PRODUCER, kind="script", gating="none")
        expect_clean(ok, "control-retired-producer.json", "rule (b): a producer deleted since, tracked in this history")

        untracked_dir = crate_dir / "tests"
        (untracked_dir / "untracked.rs").write_text("#[test]\n#[ignore]\nfn some_gated_test() {}\n")
        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["path"] = "crates/fixture-crate/tests/untracked.rs"
        expect_hit(bad, "x.json", "is not `git ls-files`-tracked", "rule (b): untracked producer.path")

        # rule (j) — producer.source_sha256 names bytes this history holds ------
        real_relpath = "crates/fixture-crate/tests/cuda_parity.rs"
        real_hash = hashlib.sha256((repo / real_relpath).read_bytes()).hexdigest()

        # Carrying `source_sha256` alone anchors the marker-mandatory arm
        # too, so this control ALSO stamps the marker + `input_sha256` —
        # a realistic fully-declared producer,
        # exercising `check_producer_source_sha256`'s own hash-match logic
        # without tripping the (separately self-tested, below) marker rule.
        ok = baseline()
        ok["producer"] = dict(ok["producer"])
        ok["producer"]["source_sha256"] = {real_relpath: real_hash}
        ok["producer"]["identity"] = "source_sha256+input_manifest"
        ok["producer"]["input_sha256"] = {"merge_json": "0" * 64}
        expect_clean(ok, "control-source-sha256.json", "rule (j): source_sha256 matching the real file's bytes")

        # No `source_sha256` key at all (the retired `tree_sha` convention,
        # or a producer kind that never carries this) is equally clean —
        # rule (j) is entirely OPTIONAL, never required.
        expect_clean(baseline(), "control-no-source-sha256.json", "rule (j): producer with no source_sha256 at all")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["source_sha256"] = {real_relpath: "0" * 64}
        expect_hit(bad, "x.json", "no committed version of", "rule (j): a sha256 no version of a real, existing path ever had")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["source_sha256"] = {"crates/fixture-crate/tests/does_not_exist.rs": real_hash}
        expect_hit(bad, "x.json", "no committed version of", "rule (j): source_sha256 names a path no commit ever tracked")

        retired = {"identity": "source_sha256+input_manifest", "input_sha256": {"merge_json": "0" * 64}}
        ok = baseline()
        ok["producer"] = {**ok["producer"], **retired, "source_sha256": {RETIRED_PRODUCER: retired_hash}}
        expect_clean(ok, "control-retired-source.json", "rule (j): a deleted source whose recorded bytes are in this history")

        bad = baseline()
        bad["producer"] = {**bad["producer"], **retired, "source_sha256": {RETIRED_PRODUCER: real_hash}}
        expect_hit(bad, "x.json", "no committed version of", "rule (j): a deleted source never committed with the recorded bytes")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["source_sha256"] = {real_relpath: "not-a-sha256"}
        expect_hit(bad, "x.json", "must be a 64-lowercase-hex sha256", "rule (j): malformed sha256 value")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["source_sha256"] = {}
        expect_hit(bad, "x.json", "must be a non-empty object", "rule (j): empty source_sha256 object")

        # rule (j) — producer.input_sha256 shape (never re-hashed, no path) ---
        ok = baseline()
        ok["producer"] = dict(ok["producer"])
        ok["producer"]["input_sha256"] = {"merge_json": "0" * 64}
        expect_clean(ok, "control-input-sha256.json", "rule (j): well-shaped input_sha256")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["input_sha256"] = {}
        expect_hit(bad, "x.json", "must be a non-empty object", "rule (j): empty input_sha256 object")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["input_sha256"] = {"merge_json": "not-a-sha256"}
        expect_hit(bad, "x.json", "must be a 64-lowercase-hex sha256", "rule (j): malformed input_sha256 value")

        # rule (j) — producer.identity marker: MANDATORY once stamped, and
        # MANDATORY for a known source-identity-declaring producer.path -----
        identity_source = {real_relpath: real_hash}
        identity_input = {"merge_json": "0" * 64}

        ok = baseline()
        ok["producer"] = dict(ok["producer"])
        ok["producer"]["identity"] = "source_sha256+input_manifest"
        ok["producer"]["source_sha256"] = identity_source
        ok["producer"]["input_sha256"] = identity_input
        expect_clean(ok, "control-identity-marker.json", "rule (j): marker + both blocks present")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["identity"] = "bogus-marker"
        bad["producer"]["source_sha256"] = identity_source
        bad["producer"]["input_sha256"] = identity_input
        expect_hit(bad, "x.json", "producer.identity must be", "rule (j): wrong marker string")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["identity"] = "source_sha256+input_manifest"
        bad["producer"]["input_sha256"] = identity_input
        expect_hit(bad, "x.json", "producer.source_sha256 is missing/empty", "rule (j): marker with no source_sha256")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["identity"] = "source_sha256+input_manifest"
        bad["producer"]["source_sha256"] = identity_source
        expect_hit(bad, "x.json", "producer.input_sha256 is missing/empty", "rule (j): marker with no input_sha256")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["path"] = "ci/scripts/perf/profile_421_legs.sh"
        # No `producer.identity` at all — this path IS a known
        # source-identity-declaring producer, so omitting the marker is
        # itself a hit, never silently treated as "this producer never
        # opted in".
        expect_hit(
            bad,
            "x.json",
            "known source-identity-declaring producer",
            "rule (j): known convention-declaring producer.path with no marker",
        )

        # rule (j) — the marker-mandatory arm's THREE INDEPENDENT anchors:
        # (1) `producer.path` already in
        # `SOURCE_IDENTITY_DECLARING_PRODUCER_PATHS` (tested above); (2) an
        # artifact that already carries a non-empty `producer.source_sha256`
        # block; (3) an artifact whose own committed FILENAME matches a
        # known profile/frontend artifact family
        # (`SOURCE_IDENTITY_DECLARING_FILENAME_RE`). Each of the three fires
        # the mandatory-marker arm on its own, independent of the other two
        # — a `producer.path` RENAME away from the reviewed allowlist can
        # never silently let an artifact that already declares the
        # convention by (2) or (3) escape the marker requirement.
        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        # `producer.path` is NOT in SOURCE_IDENTITY_DECLARING_PRODUCER_PATHS
        # (still the ordinary fixture-crate path) — only `source_sha256`
        # itself anchors this arm.
        bad["producer"]["source_sha256"] = identity_source
        expect_hit(
            bad,
            "x.json",
            "already carries a non-empty producer.source_sha256 block",
            "rule (j): source_sha256 alone anchors the marker-mandatory arm, independent of producer.path",
        )

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        # No source_sha256/input_sha256 at all, and `producer.path` is NOT
        # a known declaring path — only the artifact's own FILENAME anchors
        # this arm.
        expect_hit(
            bad,
            "2026-09-07-profile-421-towers-c1b0b0ba-a100-sxm4.json",
            "matches a known profile/frontend artifact family",
            "rule (j): -profile-<N>- filename family anchors the marker-mandatory arm (N=421), "
            "independent of producer.path",
        )

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        # A DIFFERENT N (never 421) proves the anchor is a FAMILY TOKEN, not
        # a hard-coded number.
        expect_hit(
            bad,
            "2027-01-01-profile-500-towers-deadbeef-a100-sxm4.json",
            "matches a known profile/frontend artifact family",
            "rule (j): -profile-<N>- filename family anchors the marker-mandatory arm for a DIFFERENT N (500, "
            "never hard-coded to 421), independent of producer.path",
        )

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        expect_hit(
            bad,
            "2026-09-07-frontend-towers-c1b0b0ba-a100-sxm4.json",
            "matches a known profile/frontend artifact family",
            "rule (j): -frontend- filename anchors the marker-mandatory arm, independent of producer.path",
        )

        # The filename anchor is matched against the BASENAME only — a
        # DIRECTORY segment that happens to carry a family token (here, a
        # `*-raw-runs/` sibling directory literally named
        # `...-profile-421-...`) must never anchor an artifact whose own
        # basename is an ordinary, non-family name.
        expect_clean(
            baseline(),
            "2026-09-07-profile-421-towers-c1b0b0ba-a100-sxm4-raw-runs/ordinary-payload.json",
            "rule (j): filename family anchor matches the BASENAME only, never an ancestor directory's own name",
        )

        # A GENUINE regression control, off the REAL committed
        # `2026-08-31-profile-356-closeout-...json` filename: it matches a
        # bare `-profile-\d+-` (a closeout artifact that does not declare
        # the source-identity convention this rule enforces) but must NOT
        # match `SOURCE_IDENTITY_DECLARING_FILENAME_RE`'s own required
        # `-towers-` token — a bare `-profile-\d+-` regex would wrongly
        # demand the marker on this unrelated artifact.
        expect_clean(
            baseline(),
            "2026-08-31-profile-356-closeout-7820d697-a100-sxm4.json",
            "rule (j): a -profile-<N>- filename WITHOUT -towers- (a closeout artifact) "
            "anchors nothing",
        )

        # A filename NOT in either family, with no source_sha256 and an
        # ordinary producer.path, stays clean — the anchors are additive,
        # never a blanket "every artifact now needs the marker".
        expect_clean(baseline(), "control-ordinary-filename.json", "rule (j): ordinary filename anchors nothing")

        ok = baseline()
        ok["producer"] = dict(ok["producer"])
        # `kind`/`gating` switched to a committed source-identity
        # artifact's own producer-block shape ("script"/"none") — `cargo-test`'s
        # own rule (c) static scan has nothing to do with this rule and
        # would otherwise spuriously fire against a non-Rust stub path.
        ok["producer"]["path"] = "ci/scripts/perf/profile_421_legs.sh"
        ok["producer"]["kind"] = "script"
        ok["producer"]["gating"] = "none"
        ok["producer"]["identity"] = "source_sha256+input_manifest"
        ok["producer"]["source_sha256"] = identity_source
        ok["producer"]["input_sha256"] = identity_input
        expect_clean(ok, "control-identity-marker-known-path.json", "rule (j): known producer.path WITH the marker")

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
        no_rf_sha = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
        bad = baseline()
        bad["git_sha"] = no_rf_sha
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["path"] = "crates/no-rf-crate/tests/cuda_parity.rs"
        bad["producer"]["gating"] = "required-features"
        expect_hit(bad, "x.json", "has no `[[test]]` section", "rule (c): claimed required-features absent")

        feat_dir = repo / "crates" / "feat-crate" / "tests"
        feat_dir.mkdir(parents=True)
        (feat_dir / "cuda_parity.rs").write_text(
            '#[cfg(feature = "live-gpu-tests")]\n#[test]\nfn some_gated_test() {}\n'
        )
        (repo / "crates" / "feat-crate" / "Cargo.toml").write_text(
            '[package]\nname = "feat-crate"\nversion = "0.0.0"\n\n'
            '[features]\nlive-gpu-tests = []\n'
        )
        _run(["git", "add", "-A"], repo)
        _run(["git", "commit", "-q", "-m", "feature-gated"], repo)
        tracked = git_ls_files(repo)
        feat_sha = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
        ok = baseline()
        ok["git_sha"] = feat_sha
        ok["producer"] = dict(ok["producer"])
        ok["producer"]["path"] = "crates/feat-crate/tests/cuda_parity.rs"
        ok["producer"]["gating"] = "feature:live-gpu-tests"
        expect_clean(ok, "feature-gated.json", "rule (c): feature gating declared and applied")
        bad = dict(ok, producer=dict(ok["producer"], gating="feature:live-metal-tests"))
        expect_hit(bad, "x.json", "does not declare `live-metal-tests`", "rule (c): claimed feature absent")

        # rule (c) reads the test as it was when the artifact ran: a later
        # change to how the test is gated leaves the earlier record true.
        (crate_dir / "tests" / "cuda_parity.rs").write_text(
            "#[test]\n#[ignore]\nfn some_gated_test() {\n    assert!(true);\n}\n\n"
            "#[test]\nfn env_gated_test() {\n    assert!(true);\n}\n"
        )
        _run(["git", "add", "-A"], repo)
        _run(["git", "commit", "-q", "-m", "regate"], repo)
        tracked = git_ls_files(repo)
        ok = baseline()
        ok["producer"] = dict(ok["producer"])
        ok["producer"]["invocation"] = "cargo test -p fixture-crate --test cuda_parity -- --exact env_gated_test"
        ok["producer"]["gating"] = "env:JAMMI_REQUIRE_CUDA"
        expect_clean(ok, "control-gating-at-its-own-commit.json", "rule (c): gating is read at the artifact's git_sha")

        # A superseded record proves nothing; its replacement is what is checked.
        ok = baseline()
        ok["status"] = "SUPERSEDED"
        ok["producer"] = dict(ok["producer"])
        ok["producer"]["path"] = "crates/no-rf-crate/tests/cuda_parity.rs"
        expect_clean(ok, "control-superseded.json", "rule (c): a superseded record is not held to its producer")

        # rule (k) — the `topology` artifact kind --------------------------------
        # One mutation per DETERMINANT: each required field removed, each part
        # malformed the way it is actually gettable wrong, each cross-field
        # claim broken on a pass (and admitted on a fail), the producer
        # binding, and each of the three anchors.
        def topology_baseline() -> dict:
            d = baseline()
            d["artifact_kind"] = "topology"
            d["producer"] = {
                "path": "ci/scripts/runpod_gpu_topology.sh",
                "kind": "script",
                "invocation": "bash ci/scripts/runpod_gpu_topology.sh",
                "gating": "feature:live-gpu-gang-tests",
            }

            def run(transport: str) -> dict:
                return {
                    "job_id": f"job-{transport}",
                    "status": "completed",
                    "transport": transport,
                    "attempts": 1,
                    "ranks": [
                        {"instance": f"i{r}-{transport}", "label": f"h{r // 2}-gpu{r % 2}", "host": f"pod{r // 2}"}
                        for r in range(4)
                    ],
                    "loss_curve": [0.9, 0.7, 0.6],
                    "embeddings_sha256": "a" * 64,
                }

            d["topology"] = {
                "hosts": 2,
                "gpus_per_host": 2,
                "nccl_version": ["2.23.4"],
                "one_host": {
                    "device_tests": ["gang_nccl::an_abort_ends_a_real_nccl_wait_on_a_rank_that_never_joins"],
                    "product_tests": ["gpu::topology::every_gpu_topology_and_transport_publishes_the_same_adapter"],
                },
                "fleet": {
                    "world_size": 4,
                    "nccl": run("Nccl"),
                    "inline": run("Inline"),
                    "reference": {"job_id": "job-ref", "loss_curve": [0.9, 0.7, 0.6]},
                    "max_loss_delta_vs_reference": 2e-5,
                    "epsilon": 1e-4,
                },
                "verdict": "pass",
                "reasons": [],
            }
            return d

        expect_clean(topology_baseline(), "2026-01-01-topology-a6000.json", "rule (k): complete topology artifact")
        expect_clean(baseline(), "control-single-device.json", "rule (k): a non-topology artifact is untouched")

        for field in ("hosts", "gpus_per_host", "one_host", "nccl_version", "fleet", "verdict", "reasons"):
            bad = topology_baseline()
            del bad["topology"][field]
            expect_hit(bad, "x.json", f"`topology.{field}` is missing", f"rule (k): missing topology.{field}")

        bad = topology_baseline()
        bad["topology"]["hosts"] = 1
        expect_hit(bad, "x.json", "`topology.hosts` must be an integer >= 2", "rule (k): one host")
        bad = topology_baseline()
        bad["topology"]["gpus_per_host"] = True
        expect_hit(bad, "x.json", "`topology.gpus_per_host` must be an integer >= 2", "rule (k): bool GPU count")
        bad = topology_baseline()
        bad["topology"]["one_host"]["product_tests"] = []
        expect_hit(bad, "x.json", "`topology.one_host.product_tests` must be a non-empty list", "rule (k): no product tests")
        bad = topology_baseline()
        bad["topology"]["nccl_version"] = ["2.23.4", "2.21.5"]
        expect_hit(bad, "x.json", "`topology.nccl_version` must list exactly the one", "rule (k): mixed NCCL runtimes")
        bad = topology_baseline()
        bad["topology"]["fleet"]["world_size"] = 2
        expect_hit(bad, "x.json", "must be `hosts` x `gpus_per_host`", "rule (k): world is not one rank per GPU")
        bad = topology_baseline()
        bad["topology"]["fleet"]["nccl"]["transport"] = "Inline"
        expect_hit(bad, "x.json", "`topology.fleet.nccl`.transport must be 'Nccl'", "rule (k): nccl run selected inline")
        bad = topology_baseline()
        for rank in bad["topology"]["fleet"]["inline"]["ranks"]:
            rank["host"] = "pod0"
        expect_hit(bad, "x.json", "ranks span 1 host(s)", "rule (k): a run that never left one host")
        bad = topology_baseline()
        bad["topology"]["fleet"]["nccl"]["ranks"][1]["instance"] = "i0-Nccl"
        expect_hit(bad, "x.json", "repeats an instance", "rule (k): two ranks in one process")
        bad = topology_baseline()
        del bad["topology"]["fleet"]["nccl"]["ranks"][2]
        expect_hit(bad, "x.json", "carries 3 rank(s) for a world of 4", "rule (k): a missing rank")
        bad = topology_baseline()
        bad["topology"]["fleet"]["inline"]["embeddings_sha256"] = "nope"
        expect_hit(bad, "x.json", "embeddings_sha256 must be a 64-lowercase-hex digest", "rule (k): malformed digest")
        bad = topology_baseline()
        bad["topology"]["fleet"]["reference"]["loss_curve"] = []
        expect_hit(bad, "x.json", "`topology.fleet.reference.loss_curve` must be", "rule (k): no reference curve")
        bad = topology_baseline()
        bad["topology"]["fleet"]["epsilon"] = 1e-3
        expect_hit(bad, "x.json", "must be the trainer's registered", "rule (k): a loosened ε")
        bad = topology_baseline()
        bad["topology"]["verdict"] = "PASS"
        expect_hit(bad, "x.json", "`topology.verdict` must be exactly one of", "rule (k): verdict in the wrong case")
        bad = topology_baseline()
        bad["topology"]["reasons"] = ["something failed"]
        expect_hit(bad, "x.json", "but `topology.reasons` names failures", "rule (k): a pass with reasons")

        # The cross-field claims, broken on a pass, admitted on a fail.
        for mutate, needle, label in (
            (lambda t: t["fleet"]["inline"].update(loss_curve=[0.9, 0.7, 0.61]), "loss curves differ", "curves"),
            (lambda t: t["fleet"]["inline"].update(embeddings_sha256="b" * 64), "embeddings differ", "embeddings"),
            (lambda t: t["fleet"].update(max_loss_delta_vs_reference=2e-4), "exceeds ε", "delta beyond ε"),
            (lambda t: t["fleet"]["nccl"].update(attempts=2), "took 2 attempts", "retried gang"),
        ):
            bad = topology_baseline()
            mutate(bad["topology"])
            expect_hit(bad, "x.json", needle, f"rule (k): a pass whose {label} contradict it")
            ok = topology_baseline()
            mutate(ok["topology"])
            ok["topology"]["verdict"] = "fail"
            ok["topology"]["reasons"] = [f"the {label} disagreed"]
            ok["status"] = "RED"
            expect_clean(ok, "control-topology-fail.json", f"rule (k): a fail records its {label} as measured")
        bad = topology_baseline()
        bad["topology"]["verdict"] = "fail"
        expect_hit(bad, "x.json", "with no `topology.reasons`", "rule (k): a fail with no reason")
        bad = topology_baseline()
        bad["topology"]["verdict"] = "fail"
        bad["topology"]["reasons"] = ["the nccl run timed out"]
        expect_hit(bad, "x.json", "but the top-level `status` is GREEN", "rule (k): a fail filed GREEN")

        bad = topology_baseline()
        bad["producer"] = dict(bad["producer"], path="ci/scripts/runpod_gpu_prove.sh")
        expect_hit(bad, "x.json", "requires `producer.path` == 'ci/scripts/runpod_gpu_topology.sh'", "rule (k): another producer")

        # Anchors: dropping any one does not return the artifact to the
        # unchecked state.
        bad = topology_baseline()
        del bad["artifact_kind"]
        expect_hit(bad, "control-block-anchor.json", "a top-level `topology` block", "rule (k): block anchor")
        bad = topology_baseline()
        del bad["artifact_kind"]
        del bad["topology"]
        expect_hit(bad, "2026-01-01-topology-a6000.json", "the committed filename", "rule (k): filename anchor")
        bad = baseline()
        bad["artifact_kind"] = "topology"
        expect_hit(bad, "control-kind-without-block.json", "but there is no `topology` object", "rule (k): kind with no block")

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

        # rule (g) — oracle_separation, optional per leg ------------------------
        # GREEN: absent entirely is clean.
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

    # rule (i) — v2 leg identity -------------------------------------------
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
        if field in ("target_modules", "row_lengths", "matryoshka_dims"):
            return []
        return "x"

    def _full_leg_fixture(producer_kind: str, build_sha: str, outer_sha: str = "1" * 40, tier_name: str = "finetune_step") -> dict:
        """Every identity-tuple field for `(tier_name, producer_kind)`,
        populated with a typed placeholder — derived from
        `build_identity_tuples()` itself (never a second, independently-
        drifting hand-typed field list), so this fixture builder tracks the
        Rust/Python source automatically. ALSO carries the base rules (a)-(f)
        schema (`schema_version`, `git_sha`, `box`, `producer`, `status`) —
        a v2 raw leg is written as its own standalone `*.json` document
        under `*-raw-runs/`, and the pre-existing rules (a)-(f) already
        apply to EVERY such file, regardless of rule (i). `outer_sha`
        (rules a-f's own `git_sha`) is deliberately a SEPARATE parameter
        from `build_sha` (rule (i)'s `provenance.build_sha`/`git_rev`) — a
        case exercising an invalid/mismatched `build_sha` must not also
        break the file's OWN base schema, or the two rules' findings become
        impossible to tell apart. `tier_name` defaults to `"finetune_step"`;
        the encode-step self-test rows below pass `tier_name="encode_step"`
        explicitly.
        """
        tuple_spec = build_identity_tuples()[(tier_name, producer_kind)]
        doc: dict = {
            "identity": {"tier": tier_name, "producer_kind": producer_kind, "leg_shape": "raw"},
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
        for field, root, kind, _reason in tuple_spec["fields"]:
            value = None if kind == "NullMeans" else _synthetic_value_for(field)
            if root == "tier":
                doc.setdefault("tiers", {}).setdefault(tier_name, {})[field] = value
            else:
                doc.setdefault(root, {})[field] = value
        sha_root, sha_field = tuple_spec["sha_root"], tuple_spec["sha_field"]
        doc.setdefault(sha_root, {})[sha_field] = build_sha
        return doc

    # (iii) v2 raw leg missing a NonNull field, missing a NullMeans field,
    # and a NullMeans field present-null — checked directly against
    # `check_raw_leg_identity_fields`, the unit rule (i)'s presence/nullness
    # logic lives in.
    jammi_tuple = build_identity_tuples()[("finetune_step", "jammi")]
    torch_tuple = build_identity_tuples()[("finetune_step", "torch")]

    good_jammi_leg = _full_leg_fixture("jammi", "a" * 40)
    if check_raw_leg_identity_fields(good_jammi_leg, jammi_tuple, "x", "finetune_step"):
        failures.append(f"self-test FAILED: rule (i) iii control: a fully-populated jammi leg fixture reported findings: {check_raw_leg_identity_fields(good_jammi_leg, jammi_tuple, 'x', 'finetune_step')}")

    missing_nonnull_leg = _full_leg_fixture("jammi", "a" * 40)
    del missing_nonnull_leg["tiers"]["finetune_step"]["seed"]
    got = check_raw_leg_identity_fields(missing_nonnull_leg, jammi_tuple, "x", "finetune_step")
    if not any("missing identity field `seed`" in g for g in got):
        failures.append(f"self-test FAILED: rule (i) iii: missing NonNull field `seed` not caught: {got}")

    # git_rev is the torch twin's one NullMeans provenance entry.
    good_torch_leg = _full_leg_fixture("torch", "b" * 40)
    if check_raw_leg_identity_fields(good_torch_leg, torch_tuple, "x", "finetune_step"):
        failures.append(f"self-test FAILED: rule (i) iii control: a fully-populated torch leg fixture reported findings: {check_raw_leg_identity_fields(good_torch_leg, torch_tuple, 'x', 'finetune_step')}")

    missing_nullmeans_leg = _full_leg_fixture("torch", "b" * 40)
    del missing_nullmeans_leg["tiers"]["finetune_step"]["max_grad_norm"]
    got = check_raw_leg_identity_fields(missing_nullmeans_leg, torch_tuple, "x", "finetune_step")
    if not any("missing identity field `max_grad_norm`" in g for g in got):
        failures.append(f"self-test FAILED: rule (i) iii: missing NullMeans field `max_grad_norm` not caught: {got}")

    present_null_nullmeans_leg = _full_leg_fixture("torch", "b" * 40)
    present_null_nullmeans_leg["tiers"]["finetune_step"]["max_grad_norm"] = None
    got = check_raw_leg_identity_fields(present_null_nullmeans_leg, torch_tuple, "x", "finetune_step")
    if any("max_grad_norm" in g for g in got):
        failures.append(f"self-test FAILED: rule (i) iii: a present-but-null NullMeans field must NOT be a finding: {got}")

    # The torch twin is held to the same payload identity as the engine's
    # own leg: the two tuples differ only in their sha field.
    jammi_names = {f[0] for f in jammi_tuple["fields"] if f[1] == "tier"}
    torch_names = {f[0] for f in torch_tuple["fields"] if f[1] == "tier"}
    if jammi_names != torch_names:
        failures.append(f"self-test FAILED: the torch twin's identity {sorted(torch_names)} differs from the payload's {sorted(jammi_names)}")

    # (iii-encode) the `_TIER_SOURCE_REGISTRY` `encode_step` row is
    # exercised the same way: its payload's identity plus the report-level
    # provenance, and no torch twin.
    encode_tuple = build_identity_tuples()[("encode_step", "jammi")]
    if ("encode_step", "torch") in build_identity_tuples():
        failures.append(
            "self-test FAILED: build_identity_tuples() carries an (encode_step, torch) entry — "
            "encode_step has no torch twin"
        )
    encode_field_names = {f[0] for f in encode_tuple["fields"]}
    if len(encode_field_names) != 22:  # 19 payload identity fields + 3 REPORT_IDENTITY_FIELDS
        failures.append(
            f"self-test FAILED: (encode_step, jammi) identity tuple has {len(encode_field_names)} "
            f"field(s), expected 22 (19 identity + 3 report provenance): {sorted(encode_field_names)}"
        )

    good_encode_leg = _full_leg_fixture("jammi", "c" * 40, tier_name="encode_step")
    got = check_raw_leg_identity_fields(good_encode_leg, encode_tuple, "x", "encode_step")
    if got:
        failures.append(f"self-test FAILED: rule (i) iii-encode control: a fully-populated encode_step leg fixture reported findings: {got}")

    missing_encode_nonnull_leg = _full_leg_fixture("jammi", "c" * 40, tier_name="encode_step")
    del missing_encode_nonnull_leg["tiers"]["encode_step"]["seed"]
    got = check_raw_leg_identity_fields(missing_encode_nonnull_leg, encode_tuple, "x", "encode_step")
    if not any("missing identity field `seed`" in g for g in got):
        failures.append(f"self-test FAILED: rule (i) iii-encode: missing NonNull field `seed` not caught: {got}")

    # `checkpoint_pooling_sha256` is the encode payload's one NullMeans entry.
    missing_encode_nullmeans_leg = _full_leg_fixture("jammi", "c" * 40, tier_name="encode_step")
    del missing_encode_nullmeans_leg["tiers"]["encode_step"]["checkpoint_pooling_sha256"]
    got = check_raw_leg_identity_fields(missing_encode_nullmeans_leg, encode_tuple, "x", "encode_step")
    if not any("missing identity field `checkpoint_pooling_sha256`" in g for g in got):
        failures.append(f"self-test FAILED: rule (i) iii-encode: missing NullMeans field `checkpoint_pooling_sha256` not caught: {got}")

    present_null_encode_leg = _full_leg_fixture("jammi", "c" * 40, tier_name="encode_step")
    present_null_encode_leg["tiers"]["encode_step"]["checkpoint_pooling_sha256"] = None
    got = check_raw_leg_identity_fields(present_null_encode_leg, encode_tuple, "x", "encode_step")
    if any("checkpoint_pooling_sha256" in g for g in got):
        failures.append(f"self-test FAILED: rule (i) iii-encode: a present-but-null NullMeans field must NOT be a finding: {got}")

    # (mis-mapped) a registry row naming a struct that does not exist in the
    # file must fail CLOSED (ArtifactError), never silently return an empty
    # tuple or fall through to some other struct's block.
    try:
        _extract_rust_identity_block(_JAMMI_REPORT_RS, _TIER_IDENTITY_FIELDS_BLOCK_RE, struct="NoSuchTierStruct")
        failures.append("self-test FAILED: _extract_rust_identity_block with a nonexistent struct name did not raise")
    except ArtifactError as exc:
        if "impl Payload for NoSuchTierStruct {" not in str(exc):
            failures.append(f"self-test FAILED: mis-mapped-struct ArtifactError had the wrong message: {exc}")

    # (iv)/(v) sha cross-check: mismatch, and unknown/-dirty on an
    # otherwise-GREEN leg — exercised end to end via run_gate so the
    # parent-artifact git_sha comparison (not just field presence) fires.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_iv:
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
            failures.append(f"self-test FAILED: rule (i) iv: build_sha/git_sha mismatch not caught: {got}")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_v:
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
            failures.append(f"self-test FAILED: rule (i) v: a '-dirty'-suffixed build_sha on a GREEN leg was not caught: {got}")

    # (vi) folded leg carrying identity fields of its own -> RED
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_vi:
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
            failures.append(f"self-test FAILED: rule (i) vi: a folded leg leaking an identity field was not caught: {got}")

    # (ii)/(i) — a non-.json raw payload (e.g. a `.json.raw` rename) under a
    # `*-raw-runs/` dir that is NOT in the closed LEGACY_RAW_NONJSON list ->
    # RED, regardless of the parent's own schema_version or the payload's
    # own leg_schema_version (both shapes land on this SAME finding: the
    # rename bypass is caught before content is ever inspected).
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_ii:
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
            failures.append(f"self-test FAILED: rule (i) i/ii: a non-allowlisted .json.raw payload was not caught: {got}")

    # (ix) discovery domain — a `*-raw-runs/` directory whose name does NOT
    # match `<any-artifact-stem>-raw-runs` (the committed counterexample:
    # `2026-08-25-p6-b3-dense-raw-runs/` sits next to
    # `2026-08-25-p6-b3-dense-b98f7e1-a100-sxm4.json`, a stem that drops the
    # sha7/gpu suffix) must still be discovered and coverage-checked. Two
    # legs here: one under the DIR ITSELF (flat) and one nested a level
    # further down (`.../subdir/leg.json.raw`, any depth) — both must be
    # caught; a rule that only derives ONE candidate sibling path per
    # artifact stem sees neither.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_ix:
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
            failures.append(f"self-test FAILED: rule (i) ix: a mismatched-name raw-runs dir's FLAT non-json payload was not caught: {got}")
        if not any("rg-ix-unit-raw-runs/subdir/nested.json.raw" in g and "not in the closed LEGACY_RAW_NONJSON list" in g for g in got):
            failures.append(f"self-test FAILED: rule (i) ix: a mismatched-name raw-runs dir's NESTED non-json payload was not caught: {got}")

    # (x) leg identity keys on the raw-runs dir a leg belongs to, not its
    # immediate parent: a v1 leg (no `leg_schema_version`) nested ONE level
    # below a schema_version >= 2 parent's raw-runs dir
    # (`<...>-raw-runs/<box>/leg.json`, the shape a per-box sweep's
    # `stamp_leg()` actually writes for every committed leg) must still be
    # caught -> RED. A compliant v2 leg at the SAME depth must stay clean.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_x:
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
            failures.append(f"self-test FAILED: rule (i) x: a v1 leg nested under a per-box subdir of a v2 parent's raw-runs dir was not caught: {got}")
        if any("v2-leg.json" in g for g in got):
            failures.append(f"self-test FAILED: rule (i) x control: a compliant v2 leg nested under a per-box subdir was flagged: {got}")

    # (vii) a v1 leg (no `leg_schema_version` at all) under a schema_version:
    # 1 parent -> unchanged behaviour: rule (i) finds nothing, only rules
    # (a)-(f) apply.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_vii:
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
            failures.append(f"self-test FAILED: rule (i) vii: a v1 leg under a v1 parent triggered rule (i) findings unexpectedly: {got}")

    # (viii) allowlisted `kind: none` relpath whose first-introduction commit
    # is AFTER the (fixture) gate-introduction sha -> RED. A GREEN control
    # (introduced BEFORE) must pass.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_viii:
        cr, gate_sha = _rule_g_fixture(Path(tmp_viii))
        # `git log --follow`'s rename detection is CONTENT-similarity based
        # (needed for real `git mv`-tracked baselines).
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
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as shallow_src_dir, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as shallow_dst_dir:
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

            # `check_gate_introduction_sha_anchor` gets the SAME shallow
            # guard, tested directly (never inferred from `run_gate`'s own
            # check, which this function does not go through under
            # `--self-test` — it is called standalone against the REAL
            # checkout, see below). `gate_file` need not resolve to
            # anything real here: the shallow check fires before it is
            # ever touched. Reproduces the exact false-positive a real CI
            # run hit: an unguarded call reported "does not match this
            # gate file's own first-introduction commit" under a shallow
            # clone instead of the named shallow-checkout reason.
            got_anchor_shallow = check_gate_introduction_sha_anchor(
                shallow_clone, shallow_clone / "check_cuda_run_artifacts.py", "0" * 40
            )
            if got_anchor_shallow != [SHALLOW_CHECKOUT_MESSAGE]:
                failures.append(
                    f"self-test FAILED: GATE_INTRODUCTION_SHA anchor under a shallow clone did not "
                    f"return the named shallow-checkout reason (got a false 'does not match' instead "
                    f"of refusing to evaluate): {got_anchor_shallow}"
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
        "case), (e) README producer tracking, (f) the none-allowlist closure (including its "
        "first-introduction-predates-the-gate history check), and the OPTIONAL "
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
        "per-file ancestry findings; and (j) producer.source_sha256 (OPTIONAL, matching a real file's own "
        "bytes is clean, absent entirely is clean, a wrong hash / nonexistent path / malformed hash / "
        "empty object are each caught by name) is re-hashed against THIS gate's own HEAD, never a "
        "historical blob; producer.input_sha256 (OPTIONAL, shape-checked but never re-hashed) and "
        "producer.identity (OPTIONAL in general, but MANDATORY once stamped — requiring BOTH blocks "
        "together — and MANDATORY, even with no marker stamped at all, the moment ANY ONE of its three "
        "INDEPENDENT anchors fires: a known SOURCE_IDENTITY_DECLARING_PRODUCER_PATHS producer.path, an "
        "artifact that already carries a non-empty producer.source_sha256 block, or an artifact whose own "
        "basename matches a known profile/frontend SOURCE_IDENTITY_DECLARING_FILENAME_RE family, matched "
        "against the basename only, never an ancestor directory's own name) round out rule (j). "
        "Rule (k)'s `topology` kind bites on every determinant of its registry — each field missing, a "
        "one-host fleet, a non-integer GPU count, an empty test list, mixed NCCL runtimes, a world that is "
        "not one rank per GPU, a run whose coordinator selected the other transport, ranks that never "
        "left one host / share a process / are missing, a malformed digest, no reference curve, a "
        "loosened ε, a mis-cased verdict and a pass with reasons — each cross-field claim (a retried gang included) is caught on a "
        "pass and admitted on a fail, a fail with no reason or filed GREEN is caught, another producer is "
        "refused, and each of its three anchors independently pulls an artifact into the rule."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
