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

  (k) the `gang` ARTIFACT KIND (the RunPod POD and CLUSTER legs' own
      evidence): an artifact declared `gang` by ANY of
      three independent anchors — `artifact_kind == "gang"`, a top-level
      `gang` block, or a committed filename matching
      `GANG_ARTIFACT_FILENAME_RE` — must FIRST carry `gang.leg`, exactly one
      of `"pod"`/`"cluster"` (checked before either registry below — a
      missing or unrecognized leg leaves nothing else checkable, since the
      two legs owe DIFFERENT payloads). `gang.leg == "pod"` carries every
      row of `GANG_POD_FIELD_REGISTRY`: `world` (>= 2), `collective`, one per-rank `device` for
      each of `world` ranks, the same-seed `digests` PAIR (exactly two), the
      measured `per_step_loss_delta`, the leg's own `verdict` (exactly
      `pass` or `fail`), and `epsilon` with its `value`, its `derivation`,
      and the `registered_sha` it was PRE-registered at — a commit that must
      be an ancestor of HEAD and a STRICT ancestor of the artifact's own
      EVIDENCE ANCHOR (an ε landing in the same commit as the tree it
      excuses is not pre-registered). The anchor is `git_sha` when that is
      an ancestor of HEAD, else `merged_as` when that is present, well-typed
      and an ancestor of HEAD, else the artifact FAILS naming both — the
      same order rule (d) itself applies, so a measured tip whose landing
      commit REWROTE it is still ordered against something real instead of
      skipping the ε check entirely, while a tip that IS in this history is
      ordered against itself and no landing commit stamped beside it can
      loosen that (`_gang_evidence_anchor` carries the full reasoning).

      `gang.leg == "cluster"` (the two-HOST RunPod-cluster leg,
      `ncclCommInitRank`, never `ncclCommInitAll`) carries every row of
      `GANG_CLUSTER_FIELD_REGISTRY` instead: `world`, `collective`, `hosts`
      (exactly 2), `ranks[]` (`rank`/`host`/`device`/`iface` per entry —
      never merely `device`, since the topology this leg proves spans
      SEPARATE hosts), `reduced_vector_digest` (a bit-exact digest, equal
      across both ranks on a `pass` — asserted by the DRIVER before
      assembly, never re-derived here; this is a bit-exact sum, never
      conflated with the pod leg's LoRA-shaped same-seed reproducibility
      PAIR, a different regime this rule does not claim anything about for
      the cluster leg), `verdict` (the SAME generic check as the pod leg),
      and `pod_count`/`gpu_count_per_pod`/`ttl_hours` — the shape and
      deadline the cluster was RENTED at, as measured from the create
      response. It carries NO `digests`/`per_step_loss_delta`/`epsilon` row
      at all: those name a training-loss reproducibility bound the cluster
      leg does not measure.

      A FAILING gang run is REPRESENTABLE: a `fail` is admitted with its deltas and digests as
      measured, and owes a non-empty `gang.reason` plus a top-level
      `status` that is not `GREEN`. A `pass` is a CLAIM, so on a `pass` the
      measured delta must be within ε (an artifact cannot record a run that
      blew its own tolerance as if it passed) and the same-seed digest PAIR
      must be EQUAL at `world == 2` — the one regime a spike measured
      byte-identical. Above that world the pair is recorded, not asserted:
      the NCCL pin set is untested at world >= 3, and a state defined by
      missing evidence gets no definite consequence. A new required field
      lands as a registry row (the same discipline `_TIER_SOURCE_REGISTRY`
      follows), never an inline literal in a checker.

      WHAT THE ANCHOR RULE ASKS OF A PRODUCER: register ε in its OWN
      commit, BEFORE the commit that measures with it, on the same branch.
      A landing that keeps the branch's commits (this repository's merge
      commits, with no pre-merge rebase) keeps that ε a strict ancestor of
      `git_sha` afterwards. A squash, or a rebase performed AFTER the
      measurement, rewrites both commits, and the artifact then fails this
      rule from the merge onwards — so do not rebase a branch after
      measuring on it; land it, or re-measure.

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
import ancestry  # noqa: E402 — the ONE ancestry rule, shared with check_pod_build_timings.py

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
# ancestry.py, imported by this gate AND check_pod_build_timings.py)
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
# rule (k) — the `gang` artifact kind (the gpu-gang pod leg's own evidence).
#
# A distributed fine-tune run on one 2-GPU pod proves something no
# single-device artifact can, and it proves it with a DIFFERENT payload: the
# topology it ran (`world`, the collective, the device each rank held), a
# digest PAIR from two same-seed runs, the measured per-step loss delta, and
# the ε that delta is read against. The ε is the part that rots silently, so
# it is the part this rule is hardest about: it must carry its own
# derivation AND the commit it was registered at, and that commit must
# already be history by the time the measured tree existed. An ε chosen
# after seeing the delta it excuses is not a tolerance, it is a rationalisation.
#
# "By the time the measured tree existed" is checked against the artifact's
# EVIDENCE ANCHOR, not unconditionally against `git_sha`: see
# `_gang_evidence_anchor`. Guarding the check behind "`git_sha` is an
# ancestor of HEAD" would skip it entirely for exactly the artifacts whose
# measured tip was rewritten on landing.
#
# LETTER: (k). As with every other letter here, it is comment/self-test-label
# prose only — no gate, allowlist, or error message parses it.
#
# WHY A REGISTRY, NOT AN INLINE LITERAL: this file's own module doc requires
# a new kind to land as registry ROWS (the same discipline `_TIER_SOURCE_
# REGISTRY` follows for the v2 identity tuples), so the field set, its
# validator, and the REASON each field is required all sit in one table a
# reader can enumerate — and the next required field is a row, never another
# `if` buried in a checker.
#
# A FAILING GANG RUN IS REPRESENTABLE. `gang.verdict` (`pass`/`fail`) is the
# leg's own call, and it is what every consequence in this rule is
# conditioned on: a `fail` is ADMITTED with its deltas and digests exactly
# as measured (with a `gang.reason` naming what failed, and a top-level
# `status` that is not GREEN), because a non-reproducible run's own numbers
# are precisely the evidence that has to survive into the repository. The
# tolerance and reproducibility assertions bind on `pass` only — and there
# they bind hard, because a `pass` is a CLAIM.
#
# WHAT THIS RULE ASSERTS ONLY CONDITIONALLY: digest EQUALITY. On a `pass`
# it is required at `world == 2` — the regime a spike measured
# byte-identical for candle 0.11's LoRA-shaped forward/backward/SGD across
# A100s, with no env pins — and merely RECORDED above that, because nothing
# has established byte-identity for a reduction whose NCCL pin set is
# untested at world >= 3. A state defined by missing
# evidence gets no definite consequence: the gate does not decide the
# higher-world case in either direction.
# --------------------------------------------------------------------------- #
# A `gang` artifact names WHICH RENTAL LEG produced it. `gang.leg` is a
# required, closed-set field, checked FIRST, before either registry below:
# `pod` is a single 2-GPU RunPod POD; `cluster` is the two-HOST RunPod CLUSTER leg's own shape (2 separate
# hosts over NCCL's `ncclCommInitRank`, never `ncclCommInitAll`) and owes a
# DIFFERENT payload: `hosts`, per-rank `host`/`iface` (never `digests`/
# `per_step_loss_delta`/`epsilon` -- the cluster leg proves a bit-exact
# reduced-vector match, not a training-loss reproducibility bound; the
# LoRA-shaped byte-identical regime was never measured across two separate
# HOSTS) plus the shape it was RENTED at (`pod_count`, `gpu_count_per_pod`,
# `ttl_hours`). Both legs share `world`/`collective`/`verdict` (the SAME
# generic `_gang_check_verdict` binds on either leg identically) and the
# three-anchor kind detection below.
#
# The cluster registry ALSO asserts, over
# `ranks[]`: every `host` is DISTINCT across ranks (two ranks on one host is
# not the two-host bootstrap this leg proves) and neither `host` nor `iface`
# is the driver's own `unknown` placeholder (the driver's assembler refuses
# to WRITE an artifact carrying one — see `_rpc_assemble_gang_artifact` — so
# a committed `unknown` means hand-editing around that refusal); each
# rank's OWN `reduced_vector_digest` is kept in its own row (never collapsed
# at assembly) and asserted equal across every rank on a `pass`; `hosts`
# must equal `pod_count` and `world` must equal `pod_count *
# gpu_count_per_pod` — both derived from the create response's own measured
# shape, never a second, independently duplicated literal. `gang.leg` is
# also bound to `producer.path` (`GANG_LEG_PRODUCER_PATH`): each leg's OWN
# renting driver is the sole writer of that leg's artifact, so a
# self-declared leg can never dodge the other leg's (differently-shaped)
# registry by pointing at a different driver.
GANG_ARTIFACT_KIND = "gang"
ARTIFACT_KIND_KEY = "artifact_kind"
GANG_BLOCK_KEY = "gang"
GANG_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

GANG_LEG_POD = "pod"
GANG_LEG_CLUSTER = "cluster"
GANG_LEGS = (GANG_LEG_POD, GANG_LEG_CLUSTER)

# A self-declared `gang.leg` cannot dodge the OTHER leg's registry by
# pointing `producer.path` at a different driver -- each leg's OWN renting
# driver is the sole writer of that leg's artifact, so the two are bound
# together here, never left as two independently-editable fields. A leg
# with NO entry here has no registered producer at all -- see
# `_gang_check_leg_producer_binding`'s own refusal for that shape (fail
# closed, never a silent pass).
GANG_LEG_PRODUCER_PATH = {
    GANG_LEG_POD: "ci/scripts/runpod_gpu_gang.sh",
    GANG_LEG_CLUSTER: "ci/scripts/runpod_gpu_cluster.sh",
}

# The cluster leg proves EXACTLY two hosts -- the plan's own shape; a
# different host count is a different, unproven regime, not a value this
# rule tolerates.
GANG_CLUSTER_HOSTS = 2

# Third anchor (the same three-anchor shape rule (j) uses): a committed
# artifact whose FILENAME declares the family cannot escape this rule by
# dropping its own `artifact_kind` key.
GANG_ARTIFACT_FILENAME_RE = re.compile(r"(?:^|[-_])gang(?:[-_.]|$)")

# The leg's own call. EXACT strings, closed set: a verdict spelled anything
# else (`PASS`, `ok`, `failed`) is a FAIL, never coerced — every conditional
# consequence below reads this field, so a value the gate does not
# understand must never silently take the lenient branch.
GANG_VERDICT_PASS = "pass"
GANG_VERDICT_FAIL = "fail"
GANG_VERDICTS = (GANG_VERDICT_PASS, GANG_VERDICT_FAIL)

# The one top-level `status` spelling a `fail` verdict may not carry. The
# corpus vocabulary is GREEN/RED/RECORD/SUPERSEDED, but `status` itself is
# free text repo-wide (several pre-schema artifacts carry their own
# phrasing) — so this rule refuses exactly the one contradiction it can
# state without inventing a vocabulary: a leg that says it failed cannot
# also be filed GREEN.
GANG_GREEN_STATUS = "GREEN"

# The world size at which a `pass` must show an EQUAL same-seed digest pair.
# Above it the pair is recorded, not asserted — see the section comment.
GANG_DIGEST_EQUALITY_WORLD = 2


def _is_real_number(value) -> bool:
    """A JSON number that is not a bool (`isinstance(True, int)` is True in
    Python) and not a NaN/Infinity (`json.load` accepts both by default)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and value not in (float("inf"), float("-inf"))


def _gang_check_world(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
    world = gang.get("world")
    if isinstance(world, bool) or not isinstance(world, int):
        return [f"`gang.world` must be an integer rank count, got {world!r}"]
    if world < 2:
        return [f"`gang.world` must be >= 2 — a one-rank run is not a gang, got {world}"]
    return []


def _gang_check_collective(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
    collective = gang.get("collective")
    if not isinstance(collective, str) or not collective.strip():
        return [
            f"`gang.collective` must be a non-empty string naming the collective the run used, "
            f"got {collective!r}"
        ]
    return []


def _gang_check_ranks(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
    ranks = gang.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        return [f"`gang.ranks` must be a non-empty list, one entry per rank, got {ranks!r}"]
    failures: list[str] = []
    world = gang.get("world")
    if isinstance(world, int) and not isinstance(world, bool) and len(ranks) != world:
        failures.append(
            f"`gang.ranks` carries {len(ranks)} per-rank `device` entr(y/ies) but `gang.world` is "
            f"{world} — every rank records the device it held, so the counts must be equal"
        )
    seen: list[int] = []
    for i, entry in enumerate(ranks):
        if not isinstance(entry, dict):
            failures.append(f"`gang.ranks[{i}]` must be an object with `rank` and `device`, got {entry!r}")
            continue
        rank = entry.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            failures.append(f"`gang.ranks[{i}].rank` must be a rank index >= 0, got {rank!r}")
        else:
            seen.append(rank)
        device = entry.get("device")
        if not isinstance(device, str) or not device.strip():
            failures.append(
                f"`gang.ranks[{i}].device` must be a non-empty string naming the device this rank "
                f"held, got {device!r}"
            )
    if len(set(seen)) != len(seen):
        failures.append(f"`gang.ranks` repeats a rank index ({sorted(seen)}) — each rank appears once")
    elif isinstance(world, int) and not isinstance(world, bool) and seen and sorted(seen) != list(range(world)):
        failures.append(
            f"`gang.ranks` covers rank indices {sorted(seen)}, not 0..{world - 1} — every rank of the "
            "gang records its own device"
        )
    return failures


def _gang_check_digests(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
    digests = gang.get("digests")
    if not isinstance(digests, list) or len(digests) != 2:
        return [
            "`gang.digests` must be the PAIR of same-seed runs the equal-topology reproducibility "
            f"oracle produces — exactly two entries, got {digests!r}"
        ]
    failures: list[str] = []
    seeds: list = []
    values: list[str] = []
    for i, entry in enumerate(digests):
        if not isinstance(entry, dict):
            failures.append(f"`gang.digests[{i}]` must be an object with `seed` and `digest`, got {entry!r}")
            continue
        seed = entry.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            failures.append(f"`gang.digests[{i}].seed` must be an integer seed, got {seed!r}")
        else:
            seeds.append(seed)
        digest = entry.get("digest")
        if not isinstance(digest, str) or not GANG_DIGEST_RE.match(digest):
            failures.append(
                f"`gang.digests[{i}].digest` must be a 64-lowercase-hex digest, got {digest!r}"
            )
        else:
            values.append(digest)
    if len(seeds) == 2 and seeds[0] != seeds[1]:
        failures.append(
            f"`gang.digests` records two DIFFERENT seeds ({seeds[0]} and {seeds[1]}) — the pair is two "
            "runs of the SAME seed; two seeds prove nothing about reproducibility"
        )
    # Equality, asserted CONDITIONALLY (see the section comment above): only
    # on a `pass` verdict, and only in the world size a spike actually
    # measured byte-identical. A `fail` records its pair as measured — that
    # is the whole point of admitting a failing run — and a `pass` at a
    # larger world records it too, because nothing here establishes what
    # byte-identity should mean for an untested NCCL pin set.
    world = gang.get("world")
    if (
        gang.get("verdict") == GANG_VERDICT_PASS
        and not isinstance(world, bool)
        and world == GANG_DIGEST_EQUALITY_WORLD
        and len(seeds) == 2
        and seeds[0] == seeds[1]
        and len(values) == 2
        and values[0] != values[1]
    ):
        failures.append(
            f"`gang.digests` records a `pass` at `gang.world` {GANG_DIGEST_EQUALITY_WORLD} whose two "
            f"same-seed runs produced DIFFERENT digests ({values[0]} vs {values[1]}) — a spike "
            "measured this regime byte-identical, so an unequal pair is a failed run: record it as "
            f"`gang.verdict` {GANG_VERDICT_FAIL!r} with its own `gang.reason`, never as a pass"
        )
    return failures


def _gang_check_delta_series(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
    deltas = gang.get("per_step_loss_delta")
    if not isinstance(deltas, list) or not deltas:
        return [
            "`gang.per_step_loss_delta` must be a non-empty list of the MEASURED per-step loss "
            f"deltas (one per global step), got {deltas!r}"
        ]
    bad = [(i, v) for i, v in enumerate(deltas) if not _is_real_number(v)]
    if bad:
        i, v = bad[0]
        return [f"`gang.per_step_loss_delta[{i}]` must be a finite number, got {v!r}"]
    return []


def _gang_check_verdict(gang: dict, data: dict, _repo_root: Path) -> list[str]:
    """The leg's own pass/fail call, and the two things a `fail` owes a
    reader. Every other conditional consequence in this rule (the ε
    tolerance read, the conditional digest-equality assertion) is gated on
    this field, so an unrecognised value is refused here rather than
    quietly routed down the lenient branch."""
    verdict = gang.get("verdict")
    if verdict not in GANG_VERDICTS:
        return [
            f"`gang.verdict` must be exactly one of {list(GANG_VERDICTS)} — the leg's own call about "
            "the run this artifact records; a run that failed is recorded AS failed, with its "
            f"numbers as measured, never omitted or filed as a pass, got {verdict!r}"
        ]
    if verdict != GANG_VERDICT_FAIL:
        return []
    failures: list[str] = []
    reason = gang.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        failures.append(
            f"`gang.verdict` is {GANG_VERDICT_FAIL!r} but `gang.reason` is missing or blank — a "
            "recorded failure states WHAT failed (which rank, which collective, which step), or it "
            f"is an unactionable number, got {reason!r}"
        )
    status = data.get("status")
    if isinstance(status, str) and status.strip().upper() == GANG_GREEN_STATUS:
        failures.append(
            f"`gang.verdict` is {GANG_VERDICT_FAIL!r} but the artifact's top-level `status` is "
            f"{status!r} — a leg that reports its own failure cannot also be filed "
            f"{GANG_GREEN_STATUS}; the two fields would contradict each other in the corpus"
        )
    return failures


def _gang_evidence_anchor(data: dict, repo_root: Path) -> tuple[str | None, str | None, list[str]]:
    """The commit in THIS history that a gang artifact's evidence is anchored
    at — the thing ε's pre-registration is ordered against.

    Returns `(anchor_field, anchor_sha, failures)`; `anchor_sha is None`
    means no anchor exists in this history and `failures` carries the one
    finding that names BOTH candidates.

    The order mirrors `check_ancestry` (rule (d)) — `git_sha` FIRST, then
    `merged_as` as the RESCUE, each held to `git merge-base --is-ancestor`
    against HEAD:

      * `git_sha` when 40-hex and an ancestor of HEAD — the ordinary shape,
        a tip that survived into this history verbatim. It is where the
        measurement actually happened, so it is the tree ε has to predate;
        a `merged_as` stamped beside it names a LATER commit, and reading
        the order against that later commit would admit an ε registered in
        the measured commit itself. `merged_as` is a rescue for a lost
        anchor, never a relaxation of a present one.
      * else `merged_as` when present, 40-hex, and an ancestor of HEAD. A
        measured branch tip that landed by a commit which REWROTE it (a
        squash, or a pre-merge rebase) has a `git_sha` that is an ancestor
        of nothing; `merged_as` is then the only commit whose content is in
        this history, so it is the only anchor an ordering claim can mean.
        Concretely: ε is here ordered against the LANDING commit
        (`merged_as`), never against the (now-unreachable) commit where the
        measurement itself ran — that commit has no content in this
        history for anything to be ordered against. On this repo's corpus
        that is not a corner case: of the 111 cuda-run artifacts carrying a
        `git_sha`, 7 need this arm.
      * else neither: a hard FAIL naming both, never a silent skip. Guarding
        BOTH ε arms behind `_is_ancestor(git_sha, HEAD)` would skip the
        whole ε pre-registration check for exactly the artifacts that need
        the `merged_as` rescue — including an ε registered in the landing
        commit itself.

    ANCESTRY, never "resolvable": whether a rewritten sha is READABLE here
    is a property of the clone (4 of those 7 exist only as loose objects in
    one operator's checkout, and not at all in a fresh CI clone), so
    `git cat-file`-style resolvability would make this gate's verdict
    depend on who ran it. Ancestry is the same question for every clone
    with real history — which `run_gate`'s shallow guard already insists on.

    `git_sha_unresolved` is NOT an exemption here: rule (a) lets a reviewed
    legacy artifact carry a short/malformed ref instead of a `git_sha`, and
    rule (d) has nothing resolvable to check for it — but an ε is a claim
    about ORDER IN THIS HISTORY, and an artifact with no anchor in this
    history cannot make that claim at all. Such an artifact lands as this
    finding, not as a silent pass.

    Git failures are NOT swallowed: `_run` (see its own definition) does not
    catch a missing/broken `git`, so a `FileNotFoundError`/`OSError`
    propagates out of `_is_ancestor` and the gate exits non-zero with the
    traceback. Fail-closed BY EXIT — never by a `False` that would read as
    "not an ancestor" and produce a misleading per-artifact finding.
    """
    git_sha = data.get("git_sha")
    merged_as = data.get("merged_as")
    if isinstance(git_sha, str) and GIT_SHA_RE.match(git_sha) and _is_ancestor(git_sha, repo_root):
        return "git_sha", git_sha, []
    if isinstance(merged_as, str) and GIT_SHA_RE.match(merged_as) and _is_ancestor(merged_as, repo_root):
        return "merged_as", merged_as, []
    return (
        None,
        None,
        [
            f"`gang.epsilon` has no commit in this history to be pre-registered AGAINST: neither "
            f"`git_sha` ({git_sha!r}) nor `merged_as` ({merged_as!r}) is an ancestor of HEAD, so the "
            "claim 'ε was on record before the measured tree' names no order this checkout can "
            "establish (a `git_sha_unresolved` artifact is not exempt — it has no anchor either)"
        ],
    )


def _gang_check_epsilon(gang: dict, data: dict, repo_root: Path) -> list[str]:
    epsilon = gang.get("epsilon")
    if not isinstance(epsilon, dict):
        return [
            "`gang.epsilon` must be an object carrying the PRE-REGISTERED tolerance: "
            f"`value`, `derivation`, `registered_sha`, got {epsilon!r}"
        ]
    failures: list[str] = []
    value = epsilon.get("value")
    if not _is_real_number(value) or value <= 0:
        failures.append(f"`gang.epsilon.value` must be a finite number > 0, got {value!r}")
    derivation = epsilon.get("derivation")
    if not isinstance(derivation, str) or not derivation.strip():
        failures.append(
            "`gang.epsilon.derivation` must state HOW this ε was arrived at (the spike that measured "
            "it, or the max delta over the same-seed baseline runs on this box) — a bare number is "
            f"not a tolerance, got {derivation!r}"
        )
    registered_sha = epsilon.get("registered_sha")
    if not isinstance(registered_sha, str) or not GIT_SHA_RE.match(registered_sha):
        failures.append(
            "`gang.epsilon.registered_sha` must be the 40-hex commit this ε was registered at, "
            f"got {registered_sha!r}"
        )
        return failures
    # PRE-registered means: already in history when the measured tree existed.
    # Checked in the strongest form this checkout can actually establish —
    # ancestry — never inferred from the artifact's own prose.
    if not _is_ancestor(registered_sha, repo_root):
        failures.append(
            f"`gang.epsilon.registered_sha` ({registered_sha}) is not an ancestor of HEAD — an ε whose "
            "own registration commit is not in this history was never pre-registered here"
        )
        return failures
    anchor_field, anchor, anchor_failures = _gang_evidence_anchor(data, repo_root)
    if anchor is None:
        failures.extend(anchor_failures)
        return failures
    if registered_sha == anchor:
        failures.append(
            f"`gang.epsilon.registered_sha` equals the artifact's own `{anchor_field}` ({anchor}) — the ε "
            "must be registered BEFORE the tree that was measured, never in the same commit"
        )
    elif not _is_ancestor(registered_sha, repo_root, target=anchor):
        failures.append(
            f"`gang.epsilon.registered_sha` ({registered_sha}) is not an ancestor of the measured "
            f"`{anchor_field}` ({anchor}) — this ε was not on record before the run it gates"
        )
    return failures


# (field key, validator, why this field is required). The registry IS the
# schema: a new required field lands as a row here, in the same unit that
# teaches the leg to emit it.
def _gang_check_leg(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
    leg = gang.get("leg")
    if leg not in GANG_LEGS:
        return [f"`gang.leg` must be exactly one of {list(GANG_LEGS)} — which rental leg produced this run, got {leg!r}"]
    return []


def _gang_check_hosts(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
    hosts = gang.get("hosts")
    if isinstance(hosts, bool) or not isinstance(hosts, int):
        return [f"`gang.hosts` must be an integer host count, got {hosts!r}"]
    if hosts != GANG_CLUSTER_HOSTS:
        return [
            f"`gang.hosts` must be exactly {GANG_CLUSTER_HOSTS} — this leg proves a "
            f"{GANG_CLUSTER_HOSTS}-host cluster only, got {hosts}"
        ]
    return []


GANG_UNKNOWN_SENTINELS = ("unknown", "")


def _gang_check_cluster_ranks(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
    """Beyond shape (one row per rank, non-empty fields), this asserts
    the properties that make a cluster artifact TRUST-WORTHY evidence of a
    real two-host run: `host` is DISTINCT across ranks (two ranks on one
    host is not the two-host bootstrap this leg exists to prove), `host`/
    `iface` are never the driver's own `unknown` placeholder (a committed
    `unknown` means the schema was hand-edited around the driver's own
    assembly-time refusal — see `_rpc_assemble_gang_artifact`), and — kept
    PER RANK, never collapsed into one shared value at assembly — each
    rank's own `reduced_vector_digest` is asserted equal across ranks on a
    `pass` verdict HERE, not merely trusted from an already-collapsed
    top-level field."""
    ranks = gang.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        return [f"`gang.ranks` must be a non-empty list, one entry per rank, got {ranks!r}"]
    failures: list[str] = []
    world = gang.get("world")
    verdict = gang.get("verdict")
    if isinstance(world, int) and not isinstance(world, bool) and len(ranks) != world:
        failures.append(
            f"`gang.ranks` carries {len(ranks)} entries but `gang.world` is {world} — every rank of "
            "the cluster gang records its own row"
        )
    seen: list[int] = []
    hosts_seen: list[str] = []
    digests_seen: list[object] = []
    for i, entry in enumerate(ranks):
        if not isinstance(entry, dict):
            failures.append(
                f"`gang.ranks[{i}]` must be an object with `rank`, `host`, `device`, `iface`, "
                f"`reduced_vector_digest`, got {entry!r}"
            )
            continue
        rank = entry.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            failures.append(f"`gang.ranks[{i}].rank` must be a rank index >= 0, got {rank!r}")
        else:
            seen.append(rank)
        for field in ("host", "device", "iface"):
            v = entry.get(field)
            if not isinstance(v, str) or not v.strip():
                failures.append(f"`gang.ranks[{i}].{field}` must be a non-empty string, got {v!r}")
            elif field in ("host", "iface") and v.strip().lower() in GANG_UNKNOWN_SENTINELS:
                failures.append(
                    f"`gang.ranks[{i}].{field}` is {v!r} — the driver's own assembler refuses to "
                    "assemble when a rank's host or iface is unresolved; a committed `unknown` means "
                    "this artifact was hand-edited around that refusal"
                )
        host_v = entry.get("host")
        if isinstance(host_v, str) and host_v.strip():
            hosts_seen.append(host_v.strip())
        digest_v = entry.get("reduced_vector_digest")
        if verdict == GANG_VERDICT_PASS:
            if not isinstance(digest_v, str) or not GANG_DIGEST_RE.match(digest_v):
                failures.append(
                    f"`gang.ranks[{i}].reduced_vector_digest` must be a 64-lowercase-hex digest on a "
                    f"{GANG_VERDICT_PASS!r} verdict — kept PER RANK, asserted equal across ranks below, "
                    f"got {digest_v!r}"
                )
        elif digest_v is not None and not (isinstance(digest_v, str) and GANG_DIGEST_RE.match(digest_v)):
            failures.append(
                f"`gang.ranks[{i}].reduced_vector_digest` must be null or a 64-lowercase-hex digest, "
                f"got {digest_v!r}"
            )
        digests_seen.append(digest_v)
    if len(set(seen)) != len(seen):
        failures.append(f"`gang.ranks` repeats a rank index ({sorted(seen)}) — each rank appears once")
    elif isinstance(world, int) and not isinstance(world, bool) and seen and sorted(seen) != list(range(world)):
        failures.append(
            f"`gang.ranks` covers rank indices {sorted(seen)}, not 0..{world - 1} — every rank of the "
            "cluster gang records its own row"
        )
    # Compared case-insensitively (and stripped) -- the SAME
    # normalization the assembler's own `_norm_host` applies before it ever
    # writes the artifact, so "Host-A" and "host-a" are never read as two
    # distinct hosts on either side of the producer/checker boundary.
    hosts_norm = [h.strip().casefold() for h in hosts_seen]
    if len(hosts_norm) >= 2 and len(set(hosts_norm)) != len(hosts_norm):
        failures.append(
            f"`gang.ranks[].host` repeats a host across ranks ({hosts_seen}, compared case-insensitively) "
            "— a two-host cluster gang with two ranks on the SAME host is not the two-host bootstrap this "
            "leg exists to prove"
        )
    if verdict == GANG_VERDICT_PASS and len(digests_seen) >= 2:
        valid = [d for d in digests_seen if isinstance(d, str) and GANG_DIGEST_RE.match(d)]
        if len(valid) == len(digests_seen) and len(set(valid)) != 1:
            failures.append(
                f"`gang.ranks[].reduced_vector_digest` disagree across ranks on a {GANG_VERDICT_PASS!r} "
                f"verdict ({digests_seen!r}) — a pass asserts the SAME reduced vector at every rank"
            )
    return failures


def _gang_check_cluster_shape(gang: dict) -> list[str]:
    """CLUSTER LEG ONLY cross-field check: `hosts` must equal
    `pod_count` (the cluster's own measured member count — the host count
    this leg proves is not an independent literal), and `world` must equal
    `pod_count * gpu_count_per_pod` (the rank count derives from the shape
    the create response measured, never a second, independently duplicated
    literal). Silent when an input is not yet the right type — the
    single-field registry checks already reported that."""
    failures: list[str] = []

    def _int(v: object) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    hosts, pod_count, gpu_count_per_pod, world = (
        gang.get("hosts"),
        gang.get("pod_count"),
        gang.get("gpu_count_per_pod"),
        gang.get("world"),
    )
    if _int(hosts) and _int(pod_count) and hosts != pod_count:
        failures.append(
            f"`gang.hosts` ({hosts}) does not equal `gang.pod_count` ({pod_count}) — the host count "
            "this leg proves is the cluster's own measured member count"
        )
    if _int(world) and _int(pod_count) and _int(gpu_count_per_pod) and world != pod_count * gpu_count_per_pod:
        failures.append(
            f"`gang.world` ({world}) does not equal `gang.pod_count` x `gang.gpu_count_per_pod` "
            f"({pod_count} x {gpu_count_per_pod} = {pod_count * gpu_count_per_pod}) — the rank count "
            "derives from the create response's own shape, never a second literal"
        )
    return failures


def _gang_check_leg_producer_binding(gang: dict, data: dict, _repo_root: Path) -> list[str]:
    """`gang.leg` names which rental driver produced this artifact —
    bound to `producer.path` so a self-declared leg cannot dodge the other
    leg's (stricter- or differently-shaped) registry by pointing at a
    different driver, or at no driver at all. A leg with NO
    `GANG_LEG_PRODUCER_PATH` entry (a THIRD leg added without a row) has no registered producer to bind against at ALL -- that is
    a REFUSAL, never a silent pass just because there is nothing to compare
    `producer.path` to; an artifact cannot claim a leg this tree has no
    producer for."""
    leg = gang.get("leg")
    if leg not in GANG_LEG_PRODUCER_PATH:
        return [
            f"`gang.leg` == {leg!r} has no registered producer on this tree ({sorted(GANG_LEG_PRODUCER_PATH)} "
            "are the only legs with a shipped renting driver) — refused; an artifact for this leg cannot be "
            "accepted until a driver is registered for it"
        ]
    expected = GANG_LEG_PRODUCER_PATH[leg]
    producer = data.get("producer")
    path = producer.get("path") if isinstance(producer, dict) else None
    if path != expected:
        return [
            f"`gang.leg` == {leg!r} requires `producer.path` == {expected!r} (that leg's own renting "
            f"driver — the sole writer of its artifact), got {path!r}"
        ]
    return []


def _gang_check_reduced_vector_digest(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
    digest = gang.get("reduced_vector_digest")
    verdict = gang.get("verdict")
    if verdict == GANG_VERDICT_PASS:
        if not isinstance(digest, str) or not GANG_DIGEST_RE.match(digest):
            return [
                f"`gang.reduced_vector_digest` must be a 64-lowercase-hex digest on a "
                f"{GANG_VERDICT_PASS!r} verdict (equal across both ranks — asserted BEFORE the driver "
                f"assembles this artifact), got {digest!r}"
            ]
        return []
    if digest is not None and not (isinstance(digest, str) and GANG_DIGEST_RE.match(digest)):
        return [f"`gang.reduced_vector_digest` must be null or a 64-lowercase-hex digest, got {digest!r}"]
    return []


def _gang_check_pod_count(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
    v = gang.get("pod_count")
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        return [f"`gang.pod_count` must be a positive integer (the create response's own member count), got {v!r}"]
    return []


def _gang_check_gpu_count_per_pod(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
    v = gang.get("gpu_count_per_pod")
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        return [
            f"`gang.gpu_count_per_pod` must be a positive integer (the create response's own shape), got {v!r}"
        ]
    return []


def _gang_check_ttl_hours(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
    v = gang.get("ttl_hours")
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        return [f"`gang.ttl_hours` must be a positive integer (the cluster's own deadline), got {v!r}"]
    return []


# The two-HOST bootstrap can be
# rented over TWO independent RunPod object types -- an INSTANT CLUSTER
# (`POST /v2/clusters`, near-zero capacity) or two ORDINARY pods joined by
# Global Networking (`POST /v2/pods` x2, the default -- ordinary pods
# provision reliably where clusters do not). `gang.leg` stays `"cluster"`
# either way (the two-HOST leg is the fact that matters to every OTHER
# reader of this registry); `gang.transport` is the sub-fact naming WHICH
# mechanism actually carried it, closed-set, required, so a reader can
# never mistake a Global-Networking run for an Instant-Cluster one.
GANG_TRANSPORT_INSTANT_CLUSTER = "instant-cluster"
GANG_TRANSPORT_GLOBAL_NETWORKING = "global-networking"
GANG_TRANSPORTS = (GANG_TRANSPORT_INSTANT_CLUSTER, GANG_TRANSPORT_GLOBAL_NETWORKING)


def _gang_check_transport(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
    v = gang.get("transport")
    if v not in GANG_TRANSPORTS:
        return [
            f"`gang.transport` must be exactly one of {list(GANG_TRANSPORTS)} -- which RENTAL MECHANISM "
            f"actually carried this two-host run, got {v!r}"
        ]
    return []


# The POD-leg registry.
GANG_POD_FIELD_REGISTRY: tuple[tuple[str, object, str], ...] = (
    (
        "world",
        _gang_check_world,
        "the rank count the run actually ran at — every other field is read against it",
    ),
    (
        "collective",
        _gang_check_collective,
        "which collective carried the reduction; the same topology over a different collective is a "
        "different run",
    ),
    (
        "ranks",
        _gang_check_ranks,
        "the device each rank held — one entry per rank, so a 'two-GPU' run that silently placed both "
        "ranks on one device cannot be recorded as a gang",
    ),
    (
        "digests",
        _gang_check_digests,
        "the same-seed digest PAIR the equal-topology reproducibility oracle produces. Equality is "
        f"ASSERTED on a {GANG_VERDICT_PASS!r} verdict at `world` {GANG_DIGEST_EQUALITY_WORLD} (the "
        "regime a spike measured byte-identical for candle 0.11's LoRA-shaped forward/backward/SGD "
        "on A100s with no env pins) and merely RECORDED above that world — the NCCL pin set is "
        "untested at world >= 3, and a state defined by missing evidence gets no "
        f"definite consequence. A {GANG_VERDICT_FAIL!r} verdict records its pair as measured",
    ),
    (
        "per_step_loss_delta",
        _gang_check_delta_series,
        "the MEASURED per-step loss delta — the quantity ε is about; an artifact carrying ε and no "
        "delta records a tolerance with nothing to tolerate",
    ),
    (
        "epsilon",
        _gang_check_epsilon,
        "the pre-registered tolerance, its derivation, and the commit it was registered at — a STRICT "
        "ancestor of this artifact's evidence anchor (`git_sha` when that is in this history, else "
        "`merged_as` as the rescue; neither in this history is a FAIL, never a skip — see "
        "`_gang_evidence_anchor`)",
    ),
    (
        "verdict",
        _gang_check_verdict,
        f"the leg's own call, exactly {list(GANG_VERDICTS)}. Without it a FAILING gang run is not "
        "representable at all: the tolerance and digest-equality assertions would refuse to record "
        f"the very numbers a non-reproducible run must leave behind. A {GANG_VERDICT_FAIL!r} carries "
        f"a non-empty `gang.reason` and a top-level `status` that is not {GANG_GREEN_STATUS}",
    ),
)

# The CLUSTER-leg registry: world/collective/verdict are shared with
# the pod leg (the identical `_gang_check_world`/`_gang_check_collective`/
# `_gang_check_verdict` validators — the SAME property binds on either leg);
# `hosts`/`ranks`(host+device+iface)/`reduced_vector_digest` are this leg's
# OWN topology and reproducibility claim; `pod_count`/`gpu_count_per_pod`/
# `ttl_hours` are the shape it was RENTED at, as measured from the create
# response. Deliberately NO `digests`/`per_step_loss_delta`/`epsilon` row —
# those are the pod leg's training-loss reproducibility bound, a regime
# this leg does not run at all (see the section comment above).
GANG_CLUSTER_FIELD_REGISTRY: tuple[tuple[str, object, str], ...] = (
    (
        "world",
        _gang_check_world,
        "the rank count the run actually ran at — every other field is read against it",
    ),
    (
        "collective",
        _gang_check_collective,
        "which collective carried the reduction; the same topology over a different collective is a "
        "different run",
    ),
    (
        "hosts",
        _gang_check_hosts,
        f"the host count this leg proves — exactly {GANG_CLUSTER_HOSTS}, the two-HOST bootstrap "
        "(`ncclCommInitRank`) this leg exists to exercise",
    ),
    (
        "ranks",
        _gang_check_cluster_ranks,
        "the host/device/iface each rank held — one entry per rank, so a run that silently collapsed "
        "both ranks onto one host cannot be recorded as a two-host gang",
    ),
    (
        "reduced_vector_digest",
        _gang_check_reduced_vector_digest,
        "the bit-exact reduced-vector digest, equal across both ranks on a pass (asserted by the "
        "driver BEFORE assembly — never re-derived here) — a bit-exact sum, never conflated with the "
        "pod leg's LoRA-shaped same-seed reproducibility PAIR",
    ),
    (
        "verdict",
        _gang_check_verdict,
        f"the leg's own call, exactly {list(GANG_VERDICTS)}. A {GANG_VERDICT_FAIL!r} carries a "
        f"non-empty `gang.reason` and a top-level `status` that is not {GANG_GREEN_STATUS}",
    ),
    (
        "pod_count",
        _gang_check_pod_count,
        "the cluster's own member count, as measured from the create response",
    ),
    (
        "gpu_count_per_pod",
        _gang_check_gpu_count_per_pod,
        "the cluster's own per-pod GPU count, as measured from the create response",
    ),
    (
        "ttl_hours",
        _gang_check_ttl_hours,
        "the cluster's own deadline, baked into its entrypoint at create time",
    ),
    (
        "transport",
        _gang_check_transport,
        f"which RENTAL MECHANISM carried this two-host run -- exactly one of {list(GANG_TRANSPORTS)} -- so "
        "a reader never mistakes a Global-Networking run for an Instant-Cluster one",
    ),
)


def gang_anchors(data: dict, relpath: str) -> list[str]:
    """Which independent anchor(s) declare this artifact a gang artifact.
    Three of them (rule (j)'s own shape), so dropping any ONE — renaming the
    kind key, folding the block away, renaming the file — does not silently
    return the artifact to the unchecked state."""
    anchors: list[str] = []
    if data.get(ARTIFACT_KIND_KEY) == GANG_ARTIFACT_KIND:
        anchors.append(f"{ARTIFACT_KIND_KEY} == {GANG_ARTIFACT_KIND!r}")
    if isinstance(data.get(GANG_BLOCK_KEY), dict):
        anchors.append(f"a top-level `{GANG_BLOCK_KEY}` block")
    if GANG_ARTIFACT_FILENAME_RE.search(relpath.rsplit("/", 1)[-1]):
        anchors.append("the committed filename (GANG_ARTIFACT_FILENAME_RE)")
    return anchors


def check_gang_artifact(data: dict, relpath: str, repo_root: Path) -> list[str]:
    """Rule (k): an artifact declared `gang` by ANY anchor must carry the
    complete leg-appropriate registry payload. `gang.leg` is checked FIRST —
    it decides whether `GANG_POD_FIELD_REGISTRY` or
    `GANG_CLUSTER_FIELD_REGISTRY` applies — and a missing or unrecognized
    leg stops the check there (nothing else is checkable without knowing
    which schema applies). An artifact declared by none of the anchors is
    not a gang artifact and is untouched by this rule."""
    anchors = gang_anchors(data, relpath)
    if not anchors:
        return []
    if data.get(ARTIFACT_KIND_KEY) != GANG_ARTIFACT_KIND:
        return [
            f"declared a gang artifact by {anchors[0]} but `{ARTIFACT_KIND_KEY}` is "
            f"{data.get(ARTIFACT_KIND_KEY)!r} — a gang artifact names its own kind"
        ]
    gang = data.get(GANG_BLOCK_KEY)
    if not isinstance(gang, dict):
        return [
            f"`{ARTIFACT_KIND_KEY}` is {GANG_ARTIFACT_KIND!r} but there is no `{GANG_BLOCK_KEY}` "
            f"object carrying `leg` plus the leg-appropriate rows (pod: "
            f"{', '.join(f'`{k}`' for k, _v, _w in GANG_POD_FIELD_REGISTRY)}; cluster: "
            f"{', '.join(f'`{k}`' for k, _v, _w in GANG_CLUSTER_FIELD_REGISTRY)})"
        ]
    failures: list[str] = []
    if "leg" not in gang:
        failures.append(
            f"`{GANG_BLOCK_KEY}.leg` is missing — every gang artifact names which rental leg produced "
            f"it, exactly one of {list(GANG_LEGS)}"
        )
    else:
        failures.extend(_gang_check_leg(gang, data, repo_root))
    leg = gang.get("leg")
    if leg not in GANG_LEGS:
        return failures
    failures.extend(_gang_check_leg_producer_binding(gang, data, repo_root))
    registry = GANG_POD_FIELD_REGISTRY if leg == GANG_LEG_POD else GANG_CLUSTER_FIELD_REGISTRY
    for key, validator, why in registry:
        if key not in gang:
            failures.append(f"`{GANG_BLOCK_KEY}.{key}` is missing — {why}")
            continue
        failures.extend(validator(gang, data, repo_root))
    if leg == GANG_LEG_CLUSTER:
        failures.extend(_gang_check_cluster_shape(gang))
        return failures
    if leg != GANG_LEG_POD:
        return failures
    # Cross-field (POD LEG ONLY): on a `pass`, the recorded delta is read
    # against the recorded ε. A committed artifact whose own numbers
    # contradict its own verdict is a finding here, not a thing a later
    # reader has to notice by hand. On a `fail` the deltas are admitted
    # exactly as measured — a run that blew its tolerance is what a `fail`
    # IS, and refusing to record it would leave the corpus with only the
    # runs that went well. The cluster leg carries no ε/delta row at all
    # (see GANG_CLUSTER_FIELD_REGISTRY's own comment), so this block never
    # runs for it.
    deltas = gang.get("per_step_loss_delta")
    epsilon = gang.get("epsilon")
    if (
        gang.get("verdict") == GANG_VERDICT_PASS
        and isinstance(deltas, list)
        and deltas
        and all(_is_real_number(v) for v in deltas)
        and isinstance(epsilon, dict)
        and _is_real_number(epsilon.get("value"))
        # Deliberately redundant with `_gang_check_epsilon`'s own `value > 0`
        # guard (that validator runs unconditionally, above, in the
        # GANG_POD_FIELD_REGISTRY loop): this cross-field block runs
        # regardless of whether that validator already appended a failure,
        # so without this guard a malformed epsilon (0, negative, or
        # non-numeric) would ALSO produce a nonsensical "worst step exceeds
        # epsilon.value" finding piled on top of the primary
        # `gang.epsilon.value must be...` one. Pinned here, not dropped: the
        # two checks read the same field for two different questions (is
        # epsilon well-formed vs. does the measured delta respect it) and
        # this block must not run its own comparison against a value the
        # other validator already rejected.
        and epsilon["value"] > 0
    ):
        worst = max(abs(v) for v in deltas)
        # Strict `>`, deliberately: `worst == epsilon.value` is INSIDE the
        # tolerance (inclusive), never a boundary failure — ε is "within",
        # not "strictly less than". A fixture at exactly that boundary stays
        # green (see the self-test's own boundary case).
        if worst > epsilon["value"]:
            failures.append(
                f"`{GANG_BLOCK_KEY}.per_step_loss_delta`'s worst step ({worst}) exceeds "
                f"`{GANG_BLOCK_KEY}.epsilon.value` ({epsilon['value']}) — this artifact records a run "
                f"that failed its own pre-registered tolerance as a {GANG_VERDICT_PASS!r}; record it "
                f"as {GANG_VERDICT_FAIL!r} with its own `{GANG_BLOCK_KEY}.reason`"
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
# here: the jammi side is extracted by regex from `FinetuneStepTier::
# IDENTITY_FIELDS` + `REPORT_IDENTITY_FIELDS` in `crates/jammi-bench/src/
# report.rs` (the SAME const `ci/scripts/perf/test_identity_fields_subset.py`
# reads); the torch side is IMPORTED directly from
# `crates/jammi-bench/reference/torch_finetune_step.py`'s own
# `TORCH_IDENTITY_FIELDS` / `TORCH_IDENTITY_FIELDS_NULL_MEANS` and `ci/scripts/perf/ab_merge.py`'s own `_TORCH_ARGS_LEVEL_FIELDS`
# (the field-placement map the existing jammi-vs-torch comparator already
# depends on) — never re-typed as a second, independently-drifting copy.
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
# `EncodeStepTier`'s own disjoint provenance const — same
# shape as `_TIER_IDENTITY_FIELDS_BLOCK_RE`, different const name. Unlike
# `FinetuneStepTier`, `EncodeStepTier` never folds its provenance fields
# into `IDENTITY_FIELDS`; they live here instead, at the SAME `tiers.
# encode_step` root as identity (never the Report-level `provenance` block
# `_REPORT_IDENTITY_FIELDS_BLOCK_RE` reads) — see `_TIER_SOURCE_REGISTRY`'s
# `encode_step` row below.
_PROVENANCE_FIELDS_BLOCK_RE = re.compile(
    r"pub const PROVENANCE_FIELDS:\s*&'static \[\(&'static str,\s*"
    r"[\w:]*Nullable\)\]\s*=\s*&\[(.*?)\n    \];",
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
    """Narrows `text` to everything from a given struct's OWN `impl
    <struct> {` marker onward — REQUIRED the moment more than one struct in
    the SAME file declares a const of the same name (`report.rs`
    carries both `FinetuneStepTier::IDENTITY_FIELDS` and `EncodeStepTier::
    IDENTITY_FIELDS`): an unscoped `block_re.search(text)` would always
    find whichever struct's block sits FIRST in the file, silently
    returning the wrong struct's fields for every other one. `struct=None`
    (the Report-level `REPORT_IDENTITY_FIELDS`/`_REPORT_IDENTITY_FIELDS_
    BLOCK_RE` case — that const is declared exactly once, module-level, not
    inside any `impl <Struct> { .. }` block) leaves `text` unscoped, same
    behaviour as before this function existed.
    """
    if struct is None:
        return text
    anchor = f"impl {struct} {{"
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
        where = f", scoped to `impl {struct} {{`" if struct else ""
        raise ArtifactError(f"no matching IDENTITY_FIELDS-shaped const block found in {path}{where}")
    entries = [(name, kind, reason or None) for name, kind, reason in _FIELD_ENTRY_RE.findall(m.group(1))]
    if not entries:
        raise ArtifactError(f"IDENTITY_FIELDS-shaped block in {path} matched but named zero fields")
    return entries


def _load_module_from_path(module_name: str, path: Path):
    import importlib.util

    if not path.is_file():
        raise ArtifactError(f"{path} does not exist — cannot derive rule (i)'s torch identity tuple")
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

# --------------------------------------------------------------------------- #
# HARD-MAPPED file→tier registry: every jammi tier
# rule (i) knows how to derive an identity tuple for is a ROW here, never a
# hand-typed literal inline inside `build_identity_tuples()` — a NEW tier
# (the next bench tier after `encode_step`) lands as a new registry row, in
# the SAME commit that adds its own bench-side `IDENTITY_FIELDS` const, so
# this file's registry and `report.rs`'s own tier structs move together.
# Each row: `struct` (the `impl <struct> { .. }` block
# `_TIER_IDENTITY_FIELDS_BLOCK_RE` is scoped into for this tier's identity
# fields — see `_scoped_to_struct_impl`), `provenance_block_re` (which
# const carries this tier's provenance-but-not-identity fields) +
# `provenance_struct` (`None` when that const is Report-level / module-wide,
# never struct-scoped) + `provenance_root` (WHERE those provenance fields
# live in the JSON leg itself — `"tier"` when they sit alongside identity
# under `tiers.<tier>`, `"provenance"` when they sit at the Report-level
# top-level `provenance` block instead).
# --------------------------------------------------------------------------- #
_TIER_SOURCE_REGISTRY: dict[str, dict] = {
    "finetune_step": {
        "path": _JAMMI_REPORT_RS,
        "struct": "FinetuneStepTier",
        # FinetuneStepTier's own completeness convention (contrast
        # EncodeStepTier below): provenance/dispatch facts are folded in
        # via the SEPARATE, Report-level `REPORT_IDENTITY_FIELDS` const —
        # a strict superset shape, not a disjoint one.
        "provenance_block_re": _REPORT_IDENTITY_FIELDS_BLOCK_RE,
        "provenance_struct": None,
        "provenance_root": "provenance",
    },
    "encode_step": {
        "path": _JAMMI_REPORT_RS,
        "struct": "EncodeStepTier",
        # By design this tier's provenance is its OWN struct-scoped const, DISJOINT from
        # IDENTITY_FIELDS, at the SAME `tiers.encode_step` root as identity
        # (never the Report-level `provenance` block).
        "provenance_block_re": _PROVENANCE_FIELDS_BLOCK_RE,
        "provenance_struct": "EncodeStepTier",
        "provenance_root": "tier",
    },
}

_IDENTITY_TUPLES_CACHE: dict[tuple[str, str], dict] | None = None


def build_identity_tuples() -> dict[tuple[str, str], dict]:
    """`{(tier, producer_kind): {"sha_root": ..., "sha_field": ..., "fields":
    [(name, root, "NonNull"|"NullMeans", reason_or_None), ...]}}` — computed
    once (module-level cache). Every `("<tier>", "jammi")` entry is derived
    uniformly from `_TIER_SOURCE_REGISTRY` (first-match `block_re`,
    struct-scoped per row — see `_extract_rust_identity_block`); the torch
    side (only `finetune_step` has a row — no committed artifact carries a
    torch `encode_step` leg)
    is imported directly from
    `torch_finetune_step.py`'s own `TORCH_IDENTITY_FIELDS`. Never hand-typed.
    """
    global _IDENTITY_TUPLES_CACHE
    if _IDENTITY_TUPLES_CACHE is not None:
        return _IDENTITY_TUPLES_CACHE

    tuples: dict[tuple[str, str], dict] = {}
    for tier, spec in _TIER_SOURCE_REGISTRY.items():
        tier_entries = _extract_rust_identity_block(spec["path"], _TIER_IDENTITY_FIELDS_BLOCK_RE, struct=spec["struct"])
        jammi_fields = [(name, "tier", kind, reason) for name, kind, reason in tier_entries]
        provenance_entries = _extract_rust_identity_block(
            spec["path"], spec["provenance_block_re"], struct=spec["provenance_struct"]
        )
        jammi_fields += [(name, spec["provenance_root"], kind, reason) for name, kind, reason in provenance_entries]
        tuples[(tier, "jammi")] = {"sha_root": "provenance", "sha_field": "build_sha", "fields": jammi_fields}

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
    tuples[("finetune_step", "torch")] = {"sha_root": "provenance", "sha_field": "git_rev", "fields": torch_fields}

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
    failures += check_gang_artifact(data, relpath, repo_root)
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
        # Tracked stand-ins for the two gang-leg renting drivers, so
        # `_gang_check_leg_producer_binding`'s own path=="the leg's own
        # driver" assertion has a real, `git ls-files`-tracked file to bind
        # against (rule (b) — producer.path exists and is tracked).
        (repo / "ci" / "scripts").mkdir(parents=True, exist_ok=True)
        (repo / "ci" / "scripts" / "runpod_gpu_gang.sh").write_text("# stub pod-leg gang driver\n")
        (repo / "ci" / "scripts" / "runpod_gpu_cluster.sh").write_text("# stub cluster-leg gang driver\n")

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

        # rule (k) — the `gang` artifact kind -----------------------------------
        # One mutation per DETERMINANT of the kind: each required field
        # removed, then each malformed in the way that field is actually
        # gettable wrong, plus the two anchors that must not let an artifact
        # slip back into the unchecked state, plus the delta-vs-ε
        # cross-field. Every needle below names the field, so the finding a
        # producer reads tells it which part of its own payload is wrong.
        def gang_baseline() -> dict:
            d = baseline()
            # A gang artifact measures a tree; its ε was registered EARLIER
            # (root_sha), which is what makes it pre-registered.
            d["git_sha"] = second_sha
            d["artifact_kind"] = "gang"
            # Producer bound to the pod leg's OWN renting driver — a
            # self-declared `leg` cannot point at a different (or no)
            # driver script.
            d["producer"] = {
                "path": "ci/scripts/runpod_gpu_gang.sh",
                "kind": "script",
                "invocation": "bash ci/scripts/runpod_gpu_gang.sh",
                "gating": "none",
            }
            d["gang"] = {
                "leg": "pod",
                "world": 2,
                "collective": "nccl",
                "ranks": [
                    {"rank": 0, "device": "cuda:0 NVIDIA A100-SXM4-80GB"},
                    {"rank": 1, "device": "cuda:1 NVIDIA A100-SXM4-80GB"},
                ],
                "digests": [
                    {"seed": 7, "digest": "a" * 64},
                    {"seed": 7, "digest": "a" * 64},
                ],
                "per_step_loss_delta": [0.0, 1.0e-7, 2.0e-7],
                "epsilon": {
                    "value": 1.0e-6,
                    "derivation": (
                        "max |per-step loss delta| over three same-seed baseline runs on this box, "
                        "registered before the first gating run"
                    ),
                    "registered_sha": root_sha,
                },
                "verdict": "pass",
            }
            return d

        def gang_cluster_baseline() -> dict:
            d = baseline()
            d["git_sha"] = second_sha
            d["artifact_kind"] = "gang"
            # Producer bound to the cluster leg's OWN renting driver.
            d["producer"] = {
                "path": "ci/scripts/runpod_gpu_cluster.sh",
                "kind": "script",
                "invocation": "bash ci/scripts/runpod_gpu_cluster.sh",
                "gating": "none",
            }
            d["gang"] = {
                "leg": "cluster",
                "world": 2,
                "collective": "nccl",
                "hosts": 2,
                "ranks": [
                    {
                        "rank": 0,
                        "host": "runpod-member-0",
                        "device": "cuda:0",
                        "iface": "ens1",
                        "reduced_vector_digest": "b" * 64,
                    },
                    {
                        "rank": 1,
                        "host": "runpod-member-1",
                        "device": "cuda:0",
                        "iface": "ens1",
                        "reduced_vector_digest": "b" * 64,
                    },
                ],
                "reduced_vector_digest": "b" * 64,
                "verdict": "pass",
                "pod_count": 2,
                "gpu_count_per_pod": 1,
                "ttl_hours": 1,
                "transport": "instant-cluster",
            }
            return d

        expect_clean(gang_baseline(), "2026-01-01-gang-2xa100.json", "rule (k): complete pod-leg gang artifact")
        expect_clean(
            gang_cluster_baseline(),
            "2026-01-01-gang-cluster-2x1.json",
            "rule (k): complete cluster-leg gang artifact",
        )

        # A non-gang artifact is untouched by this rule (every artifact
        # committed before the kind existed stays green).
        expect_clean(
            baseline(), "control-single-device.json", "rule (k): non-gang artifact is not gang-checked"
        )

        # `leg` itself: missing, and each unrecognized/legacy value.
        bad = gang_baseline()
        del bad["gang"]["leg"]
        expect_hit(bad, "x.json", "`gang.leg` is missing", "rule (k): missing gang.leg")
        bad = gang_baseline()
        bad["gang"]["leg"] = "solo"
        expect_hit(bad, "x.json", "`gang.leg` must be exactly one of", "rule (k): leg outside the closed set")

        for field in ("world", "collective", "ranks", "digests", "per_step_loss_delta", "epsilon", "verdict"):
            bad = gang_baseline()
            del bad["gang"][field]
            expect_hit(bad, "x.json", f"`gang.{field}` is missing", f"rule (k): pod leg missing gang.{field}")

        # A pod artifact lacking epsilon fails (covered by the loop above;
        # re-stated standalone as the canonical example).
        bad = gang_baseline()
        del bad["gang"]["epsilon"]
        expect_hit(bad, "x.json", "`gang.epsilon` is missing", "rule (k): a pod artifact lacking epsilon still fails")

        # Cluster leg: missing rows, one per registry entry (a cluster
        # artifact lacking `hosts` fails by name, and so does every sibling
        # row).
        for field in (
            "world",
            "collective",
            "hosts",
            "ranks",
            "reduced_vector_digest",
            "verdict",
            "pod_count",
            "gpu_count_per_pod",
            "ttl_hours",
            "transport",
        ):
            bad = gang_cluster_baseline()
            del bad["gang"][field]
            expect_hit(
                bad, "x.json", f"`gang.{field}` is missing", f"rule (k): cluster leg missing gang.{field}"
            )

        # `transport` -- closed set, both spellings accepted, anything
        # else refused.
        for good_transport in ("instant-cluster", "global-networking"):
            ok = gang_cluster_baseline()
            ok["gang"]["transport"] = good_transport
            expect_clean(ok, "x.json", f"rule (k): transport={good_transport!r} is accepted")
        bad = gang_cluster_baseline()
        bad["gang"]["transport"] = "carrier-pigeon"
        expect_hit(bad, "x.json", "`gang.transport` must be exactly one of", "rule (k): transport outside the closed set")
        bad = gang_cluster_baseline()
        bad["gang"]["transport"] = "cluster"
        expect_hit(
            bad,
            "x.json",
            "`gang.transport` must be exactly one of",
            "rule (k): transport must not be confused with `gang.leg`'s own value",
        )

        bad = gang_cluster_baseline()
        bad["gang"]["hosts"] = 3
        expect_hit(bad, "x.json", "must be exactly 2", "rule (k): cluster hosts != 2")

        bad = gang_cluster_baseline()
        bad["gang"]["ranks"] = [
            {"rank": 0, "host": "h0", "device": "cuda:0", "iface": "ens1"},
        ]
        expect_hit(
            bad, "x.json", "gang.ranks` carries 1 entries but", "rule (k): cluster ranks count != world"
        )

        bad = gang_cluster_baseline()
        del bad["gang"]["ranks"][0]["iface"]
        expect_hit(bad, "x.json", "`gang.ranks[0].iface` must be a non-empty string", "rule (k): cluster rank missing iface")

        bad = gang_cluster_baseline()
        bad["gang"]["ranks"][1]["rank"] = 0
        expect_hit(bad, "x.json", "repeats a rank index", "rule (k): cluster ranks repeat an index")

        bad = gang_cluster_baseline()
        bad["gang"]["reduced_vector_digest"] = None
        expect_hit(
            bad,
            "x.json",
            "must be a 64-lowercase-hex digest on a 'pass' verdict",
            "rule (k): cluster pass with no reduced_vector_digest",
        )

        ok = gang_cluster_baseline()
        ok["gang"]["verdict"] = "fail"
        ok["gang"]["reason"] = "rank 1 disagreed with rank 0's reduced vector"
        ok["gang"]["reduced_vector_digest"] = None
        ok["status"] = "RED"
        expect_clean(ok, "control-cluster-fail-no-digest.json", "rule (k): a cluster fail may record no digest at all")

        bad = gang_cluster_baseline()
        bad["gang"]["pod_count"] = 0
        expect_hit(bad, "x.json", "`gang.pod_count` must be a positive integer", "rule (k): cluster pod_count 0")

        bad = gang_cluster_baseline()
        bad["gang"]["gpu_count_per_pod"] = -1
        expect_hit(
            bad, "x.json", "`gang.gpu_count_per_pod` must be a positive integer", "rule (k): cluster gpu_count_per_pod negative"
        )

        bad = gang_cluster_baseline()
        bad["gang"]["ttl_hours"] = 0
        expect_hit(bad, "x.json", "`gang.ttl_hours` must be a positive integer", "rule (k): cluster ttl_hours 0")

        # The cluster leg carries NO ε/delta cross-field check at all -- a
        # cluster artifact with those keys simply present (never required,
        # never validated) does not trip the pod-only cross-field block.
        odd = gang_cluster_baseline()
        odd["gang"]["per_step_loss_delta"] = [999.0]
        odd["gang"]["epsilon"] = {"value": 1e-9}
        expect_clean(
            odd,
            "control-cluster-extra-pod-fields-ignored.json",
            "rule (k): cluster leg ignores stray pod-only fields rather than cross-checking them",
        )

        # the cluster registry's cross-rank properties ------------------------
        # (i) two ranks on one host FAILS (not the two-host bootstrap this
        # leg exists to prove).
        bad = gang_cluster_baseline()
        bad["gang"]["ranks"][1]["host"] = bad["gang"]["ranks"][0]["host"]
        expect_hit(
            bad,
            "x.json",
            "repeats a host across ranks",
            "rule (k): two ranks on the same host FAILS",
        )

        # (i-b) two ranks on the SAME host differing only in CASE also FAILS
        # -- compared case-insensitively (and stripped), the same
        # normalization the assembler's own `_norm_host` applies before it
        # ever writes an artifact.
        bad = gang_cluster_baseline()
        bad["gang"]["ranks"][0]["host"] = "Host-A"
        bad["gang"]["ranks"][1]["host"] = "host-a"
        expect_hit(
            bad,
            "x.json",
            "repeats a host across ranks",
            "rule (k): two ranks on the same host differing only in case FAILS",
        )

        # (ii) an `unknown` host or iface FAILS -- the driver's own
        # assembler refuses to write one; a committed `unknown` means the
        # artifact was hand-edited around that refusal.
        bad = gang_cluster_baseline()
        bad["gang"]["ranks"][0]["host"] = "unknown"
        expect_hit(
            bad,
            "x.json",
            "`gang.ranks[0].host` is 'unknown'",
            "rule (k): an `unknown` host FAILS",
        )
        bad = gang_cluster_baseline()
        bad["gang"]["ranks"][1]["iface"] = "UNKNOWN"
        expect_hit(
            bad,
            "x.json",
            "`gang.ranks[1].iface` is 'UNKNOWN'",
            "rule (k): an `unknown` iface FAILS case-insensitively",
        )

        # (iii) a per-rank digest mismatch on a `pass` FAILS -- kept PER
        # RANK, never trusted from an already-collapsed top-level value.
        bad = gang_cluster_baseline()
        bad["gang"]["ranks"][1]["reduced_vector_digest"] = "c" * 64
        expect_hit(
            bad,
            "x.json",
            "reduced_vector_digest` disagree across ranks",
            "rule (k): per-rank digest mismatch on pass FAILS",
        )
        bad = gang_cluster_baseline()
        del bad["gang"]["ranks"][0]["reduced_vector_digest"]
        expect_hit(
            bad,
            "x.json",
            "`gang.ranks[0].reduced_vector_digest` must be a 64-lowercase-hex digest",
            "rule (k): a pass with a missing per-rank digest FAILS",
        )

        # (iv) hosts != pod_count, world != pod_count*gpu_count_per_pod.
        bad = gang_cluster_baseline()
        bad["gang"]["pod_count"] = 3
        expect_hit(
            bad,
            "x.json",
            "does not equal `gang.pod_count`",
            "rule (k): gang.hosts != gang.pod_count FAILS",
        )
        bad = gang_cluster_baseline()
        bad["gang"]["gpu_count_per_pod"] = 2
        expect_hit(
            bad,
            "x.json",
            "does not equal `gang.pod_count` x `gang.gpu_count_per_pod`",
            "rule (k): gang.world != pod_count x gpu_count_per_pod FAILS",
        )

        # (v) leg/producer mismatch FAILS -- a self-declared leg cannot
        # dodge the other leg's registry by pointing at a different driver.
        bad = gang_cluster_baseline()
        bad["producer"] = dict(gang_baseline()["producer"])  # the POD leg's own driver
        expect_hit(
            bad,
            "x.json",
            "requires `producer.path` == 'ci/scripts/runpod_gpu_cluster.sh'",
            "rule (k): gang.leg == cluster with the pod leg's producer.path FAILS",
        )
        bad = gang_baseline()
        bad["producer"] = dict(gang_cluster_baseline()["producer"])  # the CLUSTER leg's own driver
        expect_hit(
            bad,
            "x.json",
            "requires `producer.path` == 'ci/scripts/runpod_gpu_gang.sh'",
            "rule (k): gang.leg == pod with the cluster leg's producer.path FAILS",
        )

        bad = gang_baseline()
        bad["gang"]["world"] = 1
        expect_hit(bad, "x.json", "`gang.world` must be >= 2", "rule (k): world 1 is not a gang")

        bad = gang_baseline()
        bad["gang"]["world"] = "2"
        expect_hit(bad, "x.json", "`gang.world` must be an integer", "rule (k): world as a string")

        bad = gang_baseline()
        bad["gang"]["collective"] = "   "
        expect_hit(bad, "x.json", "`gang.collective` must be a non-empty string", "rule (k): blank collective")

        bad = gang_baseline()
        bad["gang"]["ranks"] = [{"rank": 0, "device": "cuda:0"}]
        expect_hit(bad, "x.json", "but `gang.world` is 2", "rule (k): one device for a world of two")

        bad = gang_baseline()
        bad["gang"]["ranks"] = [{"rank": 0, "device": "cuda:0"}, {"rank": 1, "device": ""}]
        expect_hit(bad, "x.json", "`gang.ranks[1].device` must be a non-empty string", "rule (k): empty device")

        bad = gang_baseline()
        bad["gang"]["ranks"] = [{"rank": 0, "device": "cuda:0"}, {"rank": 0, "device": "cuda:0"}]
        expect_hit(bad, "x.json", "repeats a rank index", "rule (k): both entries claim rank 0")

        # The remaining `_gang_check_ranks` arms, one fixture each. Each was
        # written and left undriven: neutralising the arm kept `--self-test`
        # green, and three of them would then have ADMITTED the malformed
        # payload into the corpus.
        bad = gang_baseline()
        bad["gang"]["ranks"] = []
        expect_hit(bad, "x.json", "`gang.ranks` must be a non-empty list", "rule (k): an empty rank list")

        bad = gang_baseline()
        bad["gang"]["ranks"] = [{"rank": 0, "device": "cuda:0"}, "cuda:1 NVIDIA A100-SXM4-80GB"]
        expect_hit(bad, "x.json", "`gang.ranks[1]` must be an object", "rule (k): a rank entry that is not an object")

        bad = gang_baseline()
        bad["gang"]["ranks"] = [{"rank": 0, "device": "cuda:0"}, {"rank": "zero", "device": "cuda:1"}]
        expect_hit(
            bad,
            "x.json",
            "`gang.ranks[1].rank` must be a rank index >= 0",
            "rule (k): a rank index spelled as a string",
        )

        bad = gang_baseline()
        bad["gang"]["ranks"] = [{"rank": 0, "device": "cuda:0"}, {"rank": -1, "device": "cuda:1"}]
        expect_hit(
            bad,
            "x.json",
            "`gang.ranks[1].rank` must be a rank index >= 0",
            "rule (k): a negative rank index",
        )

        # Right count, right types, no repeat — and still not the gang's own
        # ranks: 0 recorded no device. Only the coverage arm catches this.
        bad = gang_baseline()
        bad["gang"]["ranks"] = [{"rank": 1, "device": "cuda:0"}, {"rank": 2, "device": "cuda:1"}]
        expect_hit(
            bad,
            "x.json",
            "covers rank indices [1, 2], not 0..1",
            "rule (k): two well-formed ranks that are not 0..world-1",
        )

        bad = gang_baseline()
        bad["gang"]["digests"] = [{"seed": 7, "digest": "a" * 64}]
        expect_hit(bad, "x.json", "exactly two entries", "rule (k): a single digest is not a pair")

        bad = gang_baseline()
        bad["gang"]["digests"] = [
            {"seed": 7, "digest": "a" * 64},
            {"seed": 8, "digest": "a" * 64},
        ]
        expect_hit(bad, "x.json", "two DIFFERENT seeds", "rule (k): the pair must share one seed")

        bad = gang_baseline()
        bad["gang"]["digests"] = [{"seed": 7, "digest": "nope"}, {"seed": 7, "digest": "a" * 64}]
        expect_hit(bad, "x.json", "`gang.digests[0].digest` must be a 64-lowercase-hex", "rule (k): malformed digest")

        bad = gang_baseline()
        bad["gang"]["digests"] = ["a" * 64, {"seed": 7, "digest": "a" * 64}]
        expect_hit(
            bad,
            "x.json",
            "`gang.digests[0]` must be an object",
            "rule (k): a digest entry that is not an object",
        )

        # A non-integer seed is not just a type finding: with `seeds` left
        # empty, BOTH the same-seed arm and the pass@world-2 equality arm
        # skip, so an unreadable seed would otherwise buy an artifact its way
        # out of the digest oracle entirely.
        bad = gang_baseline()
        bad["gang"]["digests"] = [{"seed": "a", "digest": "a" * 64}, {"seed": 7, "digest": "b" * 64}]
        expect_hit(
            bad,
            "x.json",
            "`gang.digests[0].seed` must be an integer seed",
            "rule (k): a seed that is not an integer",
        )

        bad = gang_baseline()
        bad["gang"]["per_step_loss_delta"] = []
        expect_hit(bad, "x.json", "`gang.per_step_loss_delta` must be a non-empty list", "rule (k): empty delta series")

        bad = gang_baseline()
        bad["gang"]["per_step_loss_delta"] = [0.0, "1e-7"]
        expect_hit(bad, "x.json", "`gang.per_step_loss_delta[1]` must be a finite number", "rule (k): non-numeric delta")

        bad = gang_baseline()
        bad["gang"]["epsilon"] = 1.0e-6
        expect_hit(bad, "x.json", "`gang.epsilon` must be an object", "rule (k): ε as a bare number")

        bad = gang_baseline()
        bad["gang"]["epsilon"] = dict(bad["gang"]["epsilon"], value=0)
        expect_hit(bad, "x.json", "`gang.epsilon.value` must be a finite number > 0", "rule (k): ε of zero")

        bad = gang_baseline()
        bad["gang"]["epsilon"] = dict(bad["gang"]["epsilon"], derivation="")
        expect_hit(bad, "x.json", "`gang.epsilon.derivation` must state HOW", "rule (k): ε with no derivation")

        bad = gang_baseline()
        bad["gang"]["epsilon"] = dict(bad["gang"]["epsilon"], registered_sha="abc1234")
        expect_hit(bad, "x.json", "`gang.epsilon.registered_sha` must be the 40-hex", "rule (k): short registration sha")

        bad = gang_baseline()
        bad["gang"]["epsilon"] = dict(bad["gang"]["epsilon"], registered_sha="b" * 40)
        expect_hit(bad, "x.json", "is not an ancestor of HEAD", "rule (k): ε registered at an unknown commit")

        bad = gang_baseline()
        bad["gang"]["epsilon"] = dict(bad["gang"]["epsilon"], registered_sha=second_sha)
        expect_hit(bad, "x.json", "equals the artifact's own `git_sha`", "rule (k): ε registered in the measured commit")

        # The EVIDENCE ANCHOR (`_gang_evidence_anchor`). One mutation per
        # arm, over both anchor shapes plus the no-anchor case. Guarding the
        # ε ordering arms behind "`git_sha` is an ancestor of HEAD" would
        # SILENTLY ADMIT every case in the `merged_as` block below.

        # (1) merged_as shape: the measured tip was rewritten on landing, so
        # `git_sha` is an ancestor of nothing and `merged_as` is the anchor.
        def merged_gang() -> dict:
            d = gang_baseline()
            d["git_sha"] = orphan_sha
            d["merged_as"] = third_sha
            d["merged_via_pr"] = 4242
            return d

        ok = merged_gang()
        ok["gang"]["epsilon"] = dict(ok["gang"]["epsilon"], registered_sha=second_sha)
        expect_clean(ok, "control-merged-anchor.json", "rule (k): ε strictly before the merged_as anchor")

        bad = merged_gang()
        bad["gang"]["epsilon"] = dict(bad["gang"]["epsilon"], registered_sha=third_sha)
        expect_hit(
            bad,
            "x.json",
            "equals the artifact's own `merged_as`",
            "rule (k): ε registered in the landing commit itself (merged_as anchor)",
        )

        bad = merged_gang()
        bad["gang"]["epsilon"] = dict(bad["gang"]["epsilon"], registered_sha=orphan_sha)
        expect_hit(
            bad,
            "x.json",
            "is not an ancestor of HEAD",
            "rule (k): ε registered on the rewritten tip, which is in no history here",
        )

        # (2) ancestor shape: an ε registered AFTER the anchor. `third_sha`
        # is an ancestor of HEAD (so the HEAD arm passes) but is NOT an
        # ancestor of `second_sha`, which is the artifact's own anchor.
        bad = gang_baseline()
        bad["gang"]["epsilon"] = dict(bad["gang"]["epsilon"], registered_sha=third_sha)
        expect_hit(
            bad,
            "x.json",
            "is not an ancestor of the measured `git_sha`",
            "rule (k): ε registered at a commit AFTER the measured tree",
        )

        # (2b) BOTH anchors are in this history: the measured `git_sha`
        # survived into HEAD's history AND a `merged_as` was stamped beside
        # it. The evidence is the tree `git_sha` names — that is where the
        # measurement happened — so the anchor is `git_sha`, and the later
        # landing commit must not be allowed to relax the ordering. Trying
        # `merged_as` FIRST would retarget ε's constraint onto the landing
        # commit and ADMIT an ε registered in the very commit it measures.
        def both_anchors_gang() -> dict:
            d = gang_baseline()  # git_sha == second_sha, itself an ancestor of HEAD
            d["merged_as"] = third_sha
            d["merged_via_pr"] = 4243
            return d

        bad = both_anchors_gang()
        bad["gang"]["epsilon"] = dict(bad["gang"]["epsilon"], registered_sha=second_sha)
        expect_hit(
            bad,
            "x.json",
            "equals the artifact's own `git_sha`",
            "rule (k): ε registered in the measured commit, `merged_as` also in this history",
        )

        bad = both_anchors_gang()
        bad["gang"]["epsilon"] = dict(bad["gang"]["epsilon"], registered_sha=third_sha)
        expect_hit(
            bad,
            "x.json",
            "is not an ancestor of the measured `git_sha`",
            "rule (k): ε registered AFTER the measured commit, `merged_as` also in this history",
        )

        ok = both_anchors_gang()
        ok["gang"]["epsilon"] = dict(ok["gang"]["epsilon"], registered_sha=root_sha)
        expect_clean(
            ok,
            "control-both-anchors.json",
            "rule (k): ε strictly before the measured `git_sha`, `merged_as` also in this history",
        )

        # (3) no anchor at all — FAILS naming BOTH candidates, never skipped.
        bad = gang_baseline()
        bad["git_sha"] = orphan_sha
        expect_hit(
            bad,
            "x.json",
            "has no commit in this history to be pre-registered AGAINST",
            "rule (k): neither git_sha nor merged_as is an ancestor of HEAD",
        )

        # `git_sha_unresolved` is not an exemption for ε: rule (d) has
        # nothing resolvable to check for such an artifact, but an ε is a
        # claim about ORDER IN THIS HISTORY and there is no anchor to order
        # it against.
        bad = gang_baseline()
        del bad["git_sha"]
        bad["git_sha_unresolved"] = "abc1234"
        bad["producer"] = {"path": None, "kind": "none", "invocation": None, "gating": "none"}
        allowlist["legacy-gang.json"] = "synthetic legacy gang fixture"
        expect_hit(
            bad,
            "legacy-gang.json",
            "has no commit in this history to be pre-registered AGAINST",
            "rule (k): git_sha_unresolved is not an ε exemption",
        )
        del allowlist["legacy-gang.json"]

        bad = gang_baseline()
        bad["gang"]["per_step_loss_delta"] = [0.0, 1.0e-5]
        expect_hit(bad, "x.json", "exceeds", "rule (k): a delta outside the pre-registered ε")

        # The boundary: `worst == epsilon.value` exactly. ε is an INCLUSIVE
        # tolerance (a run measured AT its pre-registered ceiling is not a
        # violation of it), so this must stay green -- the strict `>` in
        # `check_gang_artifact`'s cross-field block is deliberate, not an
        # off-by-one to fix.
        ok = gang_baseline()
        ok["gang"]["per_step_loss_delta"] = [0.0, 1.0e-6]
        expect_clean(ok, "control-gang-epsilon-boundary.json", "rule (k): worst == epsilon.value is inside the tolerance")

        # `gang.verdict`. One mutation per DETERMINANT: the value
        # itself, each thing a `fail` owes, and each consequence that binds
        # on `pass` ONLY (so the same payload that FAILS as a pass must be
        # ADMITTED as a fail — the property is "a failing run is
        # representable", which an assertion-only gate silently refused).
        bad = gang_baseline()
        bad["gang"]["verdict"] = "PASS"
        expect_hit(bad, "x.json", "`gang.verdict` must be exactly one of", "rule (k): verdict in the wrong case")

        bad = gang_baseline()
        bad["gang"]["verdict"] = "unknown"
        expect_hit(bad, "x.json", "`gang.verdict` must be exactly one of", "rule (k): verdict outside the closed set")

        # A `pass` at world 2 with an unequal same-seed pair: the spike's
        # measured byte-identical regime, so this is a failed run recorded as a pass.
        bad = gang_baseline()
        bad["gang"]["digests"] = [
            {"seed": 7, "digest": "a" * 64},
            {"seed": 7, "digest": "c" * 64},
        ]
        expect_hit(
            bad,
            "x.json",
            "same-seed runs produced DIFFERENT digests",
            "rule (k): a pass at world 2 whose digest pair disagrees",
        )

        # The SAME unequal pair above world 2 is RECORDED, not asserted —
        # nothing has measured byte-identity for an untested NCCL pin set.
        ok = gang_baseline()
        ok["gang"]["world"] = 4
        ok["gang"]["ranks"] = [
            {"rank": i, "device": f"cuda:{i} NVIDIA A100-SXM4-80GB"} for i in range(4)
        ]
        ok["gang"]["digests"] = [
            {"seed": 7, "digest": "a" * 64},
            {"seed": 7, "digest": "c" * 64},
        ]
        expect_clean(ok, "control-gang-world4.json", "rule (k): an unequal pair above world 2 is recorded, not asserted")

        bad = gang_baseline()
        bad["gang"]["verdict"] = "fail"
        bad["status"] = "RED"
        expect_hit(bad, "x.json", "`gang.reason` is missing or blank", "rule (k): a fail with no reason")

        bad = gang_baseline()
        bad["gang"]["verdict"] = "fail"
        bad["gang"]["reason"] = "rank 1's all-reduce diverged at global step 3"
        expect_hit(
            bad,
            "x.json",
            "cannot also be filed GREEN",
            "rule (k): a fail filed as a GREEN artifact",
        )

        # The two admissions: the exact payloads that FAIL as a `pass` are
        # ADMITTED as a `fail`, deltas and digests as measured.
        ok = gang_baseline()
        ok["gang"]["verdict"] = "fail"
        ok["gang"]["reason"] = "worst per-step delta 1e-5 exceeded the pre-registered 1e-6"
        ok["gang"]["per_step_loss_delta"] = [0.0, 1.0e-5]
        ok["status"] = "RED"
        expect_clean(ok, "control-gang-fail-delta.json", "rule (k): a fail records a delta outside ε as measured")

        ok = gang_baseline()
        ok["gang"]["verdict"] = "fail"
        ok["gang"]["reason"] = "the same-seed pair did not reproduce"
        ok["gang"]["digests"] = [
            {"seed": 7, "digest": "a" * 64},
            {"seed": 7, "digest": "c" * 64},
        ]
        ok["status"] = "RED"
        expect_clean(ok, "control-gang-fail-digests.json", "rule (k): a fail records an unequal digest pair as measured")

        # Anchors: dropping the `artifact_kind` key does NOT return a gang
        # artifact to the unchecked state — the `gang` block itself, and the
        # committed filename, each independently declare the kind.
        bad = gang_baseline()
        del bad["artifact_kind"]
        expect_hit(bad, "control-block-anchor.json", "a top-level `gang` block", "rule (k): block anchor")

        bad = gang_baseline()
        del bad["artifact_kind"]
        del bad["gang"]
        expect_hit(bad, "2026-01-01-gang-2xa100.json", "the committed filename", "rule (k): filename anchor")

        # The other side of the same door: an artifact that DECLARES the kind
        # and carries no `gang` block at all owes the whole registry, not a
        # pass for having nothing to check.
        bad = baseline()
        bad["artifact_kind"] = "gang"
        expect_hit(
            bad,
            "control-kind-without-block.json",
            "but there is no `gang` object",
            "rule (k): artifact_kind gang with no gang block",
        )

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
        if field in ("target_modules", "kernels_disabled_requested", "kernels_disabled_fired", "build_features"):
            return []
        if field in ("fast_path_globals", "sdpa_backend_probe"):
            return {}
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
        for field, root, _kind, _reason in tuple_spec["fields"]:
            value = _synthetic_value_for(field)
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

    # torch_cuda_version is a real TORCH_IDENTITY_FIELDS_NULL_MEANS entry.
    good_torch_leg = _full_leg_fixture("torch", "b" * 40)
    if check_raw_leg_identity_fields(good_torch_leg, torch_tuple, "x", "finetune_step"):
        failures.append(f"self-test FAILED: rule (i) iii control: a fully-populated torch leg fixture reported findings: {check_raw_leg_identity_fields(good_torch_leg, torch_tuple, 'x', 'finetune_step')}")

    missing_nullmeans_leg = _full_leg_fixture("torch", "b" * 40)
    del missing_nullmeans_leg["provenance"]["torch_cuda_version"]
    got = check_raw_leg_identity_fields(missing_nullmeans_leg, torch_tuple, "x", "finetune_step")
    if not any("missing identity field `torch_cuda_version`" in g for g in got):
        failures.append(f"self-test FAILED: rule (i) iii: missing NullMeans field `torch_cuda_version` not caught: {got}")

    present_null_nullmeans_leg = _full_leg_fixture("torch", "b" * 40)
    present_null_nullmeans_leg["provenance"]["torch_cuda_version"] = None
    got = check_raw_leg_identity_fields(present_null_nullmeans_leg, torch_tuple, "x", "finetune_step")
    if any("torch_cuda_version" in g for g in got):
        failures.append(f"self-test FAILED: rule (i) iii: a present-but-null NullMeans field must NOT be a finding: {got}")

    # (iii-encode) the `_TIER_SOURCE_REGISTRY`
    # `encode_step` row is exercised the SAME way the finetune_step rows
    # above are — a struct-scoped extraction (`EncodeStepTier::
    # IDENTITY_FIELDS` + its OWN disjoint `::PROVENANCE_FIELDS`, both folded
    # under root="tier" — never the finetune_step superset shape), proving
    # the registry's per-row `provenance_root` actually threads through.
    encode_tuple = build_identity_tuples()[("encode_step", "jammi")]
    if ("encode_step", "torch") in build_identity_tuples():
        failures.append(
            "self-test FAILED: build_identity_tuples() carries an (encode_step, torch) entry — "
            "this registry declares no torch row for encode_step; a registry mistake "
            "here would silently accept a torch-shaped encode leg rule (i) should reject"
        )
    encode_field_names = {f[0] for f in encode_tuple["fields"]}
    if len(encode_field_names) != 30:  # 19 IDENTITY_FIELDS + 11 disjoint PROVENANCE_FIELDS
        failures.append(
            f"self-test FAILED: (encode_step, jammi) identity tuple has {len(encode_field_names)} "
            f"field(s), expected 30 (19 identity + 11 disjoint provenance): {sorted(encode_field_names)}"
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

    # `partitions` is a `NullMeans` entry of `EncodeStepTier::PROVENANCE_FIELDS`
    # (the direct rung builds no plan) — same missing/present-null pair the
    # torch `torch_cuda_version` checks above exercise, proving the
    # disjoint-provenance row's NullMeans field reads correctly too.
    missing_encode_nullmeans_leg = _full_leg_fixture("jammi", "c" * 40, tier_name="encode_step")
    del missing_encode_nullmeans_leg["tiers"]["encode_step"]["partitions"]
    got = check_raw_leg_identity_fields(missing_encode_nullmeans_leg, encode_tuple, "x", "encode_step")
    if not any("missing identity field `partitions`" in g for g in got):
        failures.append(f"self-test FAILED: rule (i) iii-encode: missing NullMeans field `partitions` not caught: {got}")

    present_null_encode_leg = _full_leg_fixture("jammi", "c" * 40, tier_name="encode_step")
    present_null_encode_leg["tiers"]["encode_step"]["partitions"] = None
    got = check_raw_leg_identity_fields(present_null_encode_leg, encode_tuple, "x", "encode_step")
    if any("partitions" in g for g in got):
        failures.append(f"self-test FAILED: rule (i) iii-encode: a present-but-null NullMeans field must NOT be a finding: {got}")

    # (mis-mapped) a registry row naming a struct that does not exist in the
    # file must fail CLOSED (ArtifactError), never silently return an empty
    # tuple or fall through to some other struct's block.
    try:
        _extract_rust_identity_block(_JAMMI_REPORT_RS, _TIER_IDENTITY_FIELDS_BLOCK_RE, struct="NoSuchTierStruct")
        failures.append("self-test FAILED: _extract_rust_identity_block with a nonexistent struct name did not raise")
    except ArtifactError as exc:
        if "impl NoSuchTierStruct {" not in str(exc):
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
    # (`<...>-raw-runs/<box>/leg.json`, the shape stacked_sweep.sh's
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
        "Rule (k)'s `gang` kind bites on every determinant of its own registry — each GANG_POD_FIELD_"
        "REGISTRY field missing, world < 2, a device count that disagrees with world, a repeated rank, "
        "an empty rank list, a rank entry that is not an object, a rank index that is a string or "
        "negative, two well-formed ranks that are not 0..world-1, a digest entry that is not an "
        "object, a seed that is not an integer (which would otherwise skip BOTH the same-seed and "
        "the digest-equality arms), a declared `artifact_kind` with no `gang` block at all, "
        "a digest pair that is not exactly two same-seed entries, a malformed digest, an empty or "
        "non-numeric delta series, an ε that is zero / has no derivation / names a short or unknown "
        "registration commit / was registered in the very commit it measures, and a measured delta "
        "outside its own pre-registered ε — and each of its three anchors (artifact_kind, the gang "
        "block, the committed filename) independently pulls an artifact into the rule, while a "
        "non-gang artifact stays untouched. ε's EVIDENCE ANCHOR is driven arm by arm: an artifact "
        "whose measured tip was rewritten on landing is ordered against `merged_as` (clean when ε "
        "precedes it, caught when ε IS it, caught when ε is the rewritten tip), an artifact carrying "
        "BOTH a `git_sha` in this history and a `merged_as` is ordered against the `git_sha` — the "
        "tree it measured — so the later landing commit cannot relax it (caught when ε is the "
        "measured commit, caught when ε is after it, clean when ε precedes it), an ε registered at a "
        "commit that is an ancestor of HEAD but AFTER the measured tree is caught, and an artifact "
        "with neither anchor in this history — `git_sha_unresolved` included — is caught by a "
        "finding naming BOTH candidates, never skipped. `gang.verdict` closes the same "
        "loop from the other side: the wrong case and an unknown string are refused, a `fail` with "
        "no reason and a `fail` filed GREEN are caught, a `pass` at world 2 whose same-seed pair "
        "disagrees is caught while the same pair above world 2 is recorded rather than asserted, "
        "and the exact delta/digest payloads that FAIL as a pass are ADMITTED as a fail. "
        "`gang.leg` discriminates the pod registry above from the CLUSTER leg's own "
        "(hosts == 2, per-rank host/device/iface, a bit-exact reduced_vector_digest, pod_count/"
        "gpu_count_per_pod/ttl_hours, no ε/delta row at all) -- missing/unrecognized leg, every "
        "missing cluster row, a wrong host count, a malformed rank, and a pass with no digest are "
        "each caught by name, and a cluster artifact ignores stray pod-only fields rather than "
        "cross-checking them."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
