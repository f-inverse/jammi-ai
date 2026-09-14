#!/usr/bin/env python3
"""GPU-prove-once guard (esc-084, issue #454; #454 follow-up, operator
direction 2026-09-03: every release publisher, not only the CUDA lanes) —
hermetic, static, no build, no GPU. Reads workflow YAML through
`check_execution_surface_reachability.py`'s shared PyYAML-backed loader
(a declared prerequisite of this gate, installed by `.docker/ci.Dockerfile`).

**Guarded property**: a release commit is proven ONCE per shipped arch, and
EVERY release-publishing workflow — CUDA and non-CUDA alike (crates.io, npm,
every PyPI dist, the server image, the release binaries) — gates its
promotion on that SAME recorded verdict, all-or-nothing: a tag push
publishes NOTHING until the commit's prove is green. The prove lane itself
is a manual dev run, never in the critical path of any automated workflow --
see `gpu-prove.yml`'s own header for the canonical statement of why; a
publisher gates on the SUMMARY of a prove execution already on record for
the commit it promotes.

Six positive-and-negative rules (F7, ask-6 — every rule has a fixture that
must PASS as well as fixtures that must FAIL, never a grep for one known-bad
string):

  P1 (exactly-once producer + the never-in-the-critical-path doctrine):
     exactly one workflow's comment-stripped step body invokes
     `ci/scripts/runpod_gpu_prove.sh`, and it is `gpu-prove.yml`; that
     workflow's own `on:` block carries neither `push:` nor `workflow_call:`
     (an unreadable — quoted or flow-style — `on:` block is itself a FAIL,
     never a silent skip); no OTHER workflow `uses:` it, local
     (`./.github/workflows/gpu-prove.yml`) or cross-repo
     (`<owner>/<repo>/.github/workflows/gpu-prove.yml@<ref>`) form.

  P2 (no renting reusable): no workflow `uses:` a `_gpu-prove-gate.yml`
     (any path), and that file must not exist in the tree at all.

  P3 (PROMOTION_TABLE, every row reconciled): every reviewed row in
     `PROMOTION_TABLE` (workflow, promoting job, gate job/kind, tag family)
     is structurally sound. A `"direct"` row's gate job must `uses:
     ./.github/workflows/_gpu-proof-required.yml`; a `"chained"` row's gate
     job must itself be some OTHER row's promoting job in the SAME workflow
     (e.g. `crates.yml`'s `github-release` chains off `publish`, which is
     itself a `"direct"` row); a `"none"` row is a reviewed, deliberately
     UNGATED promotion (e.g. `image.yml`/`image-cuda.yml`'s CI base-image
     rebuild on every merge to `main`, or `server-image.yml`'s manual
     `:latest` refresh on a `workflow_dispatch` against `main` -- neither is
     ever a release-tag promotion) and must structurally prove it
     can NEVER fire on a release tag ref: its job `if:` must be a PURE
     top-level conjunction containing the EXACT conjunct
     `github.ref_type != 'tag'` (F3 audit fix -- a `refs/tags/`-substring-
     absence check used to pass an `if:` that merely never MENTIONED a tag
     pattern, which a `workflow_dispatch` on a tag ref trivially satisfies
     without ever excluding one; a missing `if:` at all on a `"none"` row is
     the same failure). For `"direct"`/`"chained"` rows: the promoting job's
     `needs:` lists the gate job, and the promoting job's `if:` — or, when
     the row names a `step_name`, that ONE step's `if:` (npm.yml's `publish`
     job also runs build+test unconditionally, so the gate lives on its
     "Publish" step, not the job) — is a PURE top-level conjunction
     (parenthesis- and quote-aware structural scan, never a substring check)
     containing the exact conjunct `needs.<gate>.result == 'success'`, the
     exact conjunct `startsWith(github.ref, 'refs/tags/<family>')` for the
     row's own `tag_family` (F7 audit fix -- the tag guard itself used to be
     unpinned; a `"direct"` row's own gate job `if:` is held to the SAME
     exact conjunct, since a gate job usable off a tag ref for the wrong
     family, or off no tag ref restriction at all, would let the verdict be
     consulted -- and satisfied -- outside the release-tag path it exists
     to gate), and NO depth-0 `||` anywhere. When the row names a
     `step_name`, every OTHER step in that same job must NOT itself invoke a
     publishing primitive (F4 audit fix -- a step-gated row only pinned the
     NAMED step's `if:`; a second, ungated publishing step in the same job
     used to sail through unseen). A missing workflow file, job, or step
     named by a row is a FAIL, never a silent skip. `ci/release-feature-
     manifest.json`'s own CUDA lane set is reconciled against the table as a
     SUBSET check: every manifest CUDA lane's promoting job must have a
     table row (the reverse direction — a table row naming no manifest
     lane — is expected and fine, since most rows promote a non-CUDA,
     non-manifest surface).

  P4 (consumer/producer name agreement): `gpu_prove_verdict.py`'s
     `JOB_NAME_TEMPLATE` matches `gpu-prove.yml`'s own matrix job `name:`
     line, and the required-arch set the consumer derives
     (`check_gpu_parity_matrix.py`'s `GENCODE_ARCHES` parser) equals the
     workflow's own matrix `arch:` list — never a hand-typed list on either
     side.

  P5 (the reusable actually consults the verdict, BLOCK B8 audit fix,
     hardened by F6): P3 only checks a gate job's `uses:` line, so gutting
     `_gpu-proof-required.yml` to `run: echo ok` would otherwise leave P1-P4
     green with no real promotion gate behind it. `_gpu-proof-required.yml`
     must exist, its `on:` block must be `workflow_call`-only, and its
     comment-stripped step body must invoke `python3 ci/scripts/
     gpu_prove_verdict.py` with `--sha` bound to the commit being promoted
     (`github.sha`/`$GITHUB_SHA`) — a literal sha or a tag name FAILS — and
     (F6 audit fix) `--repo` bound to `github.repository`/
     `$GITHUB_REPOSITORY` — a literal/foreign repo would key the verdict
     lookup at the wrong repo — with any `--workflow` override forbidden
     from naming anything other than `gpu-prove.yml` itself (a pointed-
     elsewhere consumer could read a DIFFERENT, unrelated workflow's runs as
     if they proved this one).

  P6 (DISCOVERY: an unlisted publishing job FAILS by name, F1+F2 hardened).
     P3 only reconciles the rows already IN `PROMOTION_TABLE` — the
     disclosed limit this module used to carry was that a brand-new
     promoting job could be invisible to it. P6 closes that: EVERY workflow
     file is scanned (F2 audit fix — no `push:`/`tags:` trigger filtering at
     all; a publishing primitive anywhere in the tree must be in the table,
     regardless of what triggers the file; an unreadable `on:`/`jobs:` block
     is itself a FAIL, never a silent skip, same fail-loud doctrine as P1).
     Every job whose comment-stripped body matches a `PRIMITIVE_PATTERNS`
     entry (F1 audit fix — a regex list over comment-stripped step bodies
     and `uses:` lines, whitespace-tolerant, never five literal marker
     strings: `cargo publish`, `npm publish`, `twine upload`, `maturin
     upload`, `docker push`, `gh release create`/`upload`, `pypa/gh-action-
     pypi-publish`, `softprops/action-gh-release`, `docker/build-push-
     action` — any `push:` value that is not literally `false`/`"false"`,
     including an unquoted `true`, `'true'`, or any `${{ }}` expression —
     `./.github/actions/docker-publish` and its cross-repo form under the
     SAME push rule, `./.github/actions/release-upload` and its cross-repo
     form, `ci/scripts/publish_crates.sh`, `docker buildx imagetools create`
     (never bare `imagetools` -- `imagetools inspect` is a read-only
     assertion, not a promotion) must be listed as SOME row's
     `(workflow, promoting_job)` in `PROMOTION_TABLE`. RECURSIVE: a job that
     merely `uses:` a LOCAL reusable workflow (job-level `uses: ./.github/
     workflows/<X>.yml`) whose OWN jobs match a primitive is itself a
     promoting job too — e.g. `_ci-base-image.yml`'s `build-and-push` job
     pushes to GHCR, so `image.yml`/`image-cuda.yml`'s `build` jobs (which
     each `uses:` it) are discovered and tabled (`gate_kind="none"`, proven
     structurally unreachable from a tag ref per F3's own rule — never
     reachable via `workflow_dispatch` on a tag ref, since neither image is
     part of any release). A workflow whose OWN `on:` block is
     `workflow_call`-only (the same "never independently starts" doctrine
     P1/P5 hold `gpu-prove.yml`/`_gpu-proof-required.yml` to) is skipped by
     the DIRECT scan — it is inert without a caller and is reached only via
     that caller's recursive check above, never double-tabled against
     itself. An unlisted match FAILS, naming the workflow and job — it can
     never again silently promote ungated.

  P7 (EVERY paid pod lane, not only the prove one). P1's three sub-rules
     — exactly one invoker, no `push:`/`workflow_call:` in that invoker's
     own `on:` block, nothing `uses:` it — restated over the reviewed
     `PAID_POD_LANE_TABLE` registry (driver script -> its one workflow):
     the prove lane, the gang lane (1 pod x 2 GPU), the perf-A/B lane and
     the how-well lane. The doctrine was never about one script's name; it
     is about a leg that RENTS HARDWARE, and a lane added without this rule
     would have reproduced P1's own escape shape one file over. A row's
     workflow missing from the tree, an unreadable `on:` block, and zero
     invokers are each a FAIL, never a silent skip.

     IDENTITY IS THE REPO-RELATIVE PATH, EVERYWHERE in this rule.
     `PAID_POD_LANE_TABLE`'s keys are `ci/scripts/...` paths, never
     basenames — keying (or re-keying, via `rsplit("/")`) on the basename
     would fold `ci/scripts/perf/<x>.sh` and `ci/scripts/<x>.sh` into one
     bucket and silently drop whichever one a lookup didn't pick (a renting
     `ci/scripts/perf/runpod_gpu_evil.sh` beside a flat, reaping
     `ci/scripts/runpod_gpu_evil.sh` produced zero findings and zero notes
     under the earlier basename-keyed revision of this rule). No
     `rsplit("/")` appears anywhere in this file — a self-test asserts it.

     P7's SUBJECT SET IS DERIVED, and the table is a COMPLETENESS
     ASSERTION over it. The deploy-capable function set is computed from
     `ci/scripts/runpod_lib.sh` — the transitive callers of
     `_rp_deploy_payload` — and a RENTING DRIVER is any tracked
     `ci/scripts/**` file whose comment-stripped text calls one of them.

     Each derived driver is either a table row (held to the three
     sub-rules) or must satisfy a machine predicate — WHOLE-FILE SCOPE, no
     verb parsing: it is a RENTER, and needs a table row, the moment its
     repo-relative path appears ANYWHERE in a comment-stripped workflow
     file that also carries `RUNPOD_API_KEY` at ANY scope (top-level
     `env:`, job `env:`, step `env:`, `with:` — the capability, not its
     spelling site). An earlier revision cleared an invocation whose first
     argument was a literal non-renting verb (e.g. `reap`); that clearance
     is GONE — no invocation-site parsing decides clearance any more, only
     the driver's path and the file's secret. This over-approximates in the
     fail-closed direction ON PURPOSE: a `paths:` filter entry mentioning a
     derived driver inside a secret-holding workflow demands a row exactly
     like a real invocation would, even though a filter entry alone cannot
     run anything. That is the price of the rule, paid deliberately rather
     than trying to parse what a `paths:` mention, a `run:` line, and every
     other place a path can appear in YAML actually DO.

     A derived driver whose repo-relative path is mentioned in NO workflow
     file at all (comment-stripped, whole file) is a printed NOTE, not a
     finding: nothing in the committed text says it can rent, and nothing
     says it cannot. That NOTE fires on ABSENCE, never on "not invoked" —
     a driver mentioned in a secret-FREE workflow only (say, a `paths:`
     entry with no `RUNPOD_API_KEY` anywhere in that file) is mentioned
     somewhere, so the absence rule does not apply, and that workflow
     cannot make it rent either, so there is nothing to flag: neither a
     finding nor a NOTE, and that silence is this rule's own stated outcome
     for that shape, not an oversight.

     A derived driver in neither state (a row, or provably unable to rent
     anywhere), and a table row whose driver has left the derived set, are
     both FAILs by name. `check_p7_paid_pod_lanes` takes the script map and
     the library text as parameters (the same shape `load_workflow_texts`
     gives the workflow scan), so its own suite injects a fifth renting
     driver, or a new deploy wrapper, without touching this tree.

Mechanism: comment-stripped line/regex scanning for `if:`/`needs:`/step
shapes over job-body text spans, where those spans come from the ONE real
YAML parse `check_execution_surface_reachability.py` exposes
(`job_source_spans`/`jobs_or_fail`) -- never a second, independently-
drifting `jobs:` header regex. Every check function takes an explicit
`workflows_dir`/`manifest_path` so
`test_check_gpu_prove_once.py` can drive them against synthetic fixture
trees, including a fixture reproducing the PRE-FIX shape (esc-084: three
publishers `uses:` a renting reusable).

`PROMOTION_TABLE` is still a hand-REVIEWED table, not derived from the
workflow tree — P3 checks each row's own internal structure, and P6 (above)
is what now catches a row that was never added at all, closing the "new
promoting job is invisible" gap the previous revision of this module
disclosed as an open limit.

Run: `python3 ci/scripts/check_gpu_prove_once.py`
Self-test: `python3 ci/scripts/test_check_gpu_prove_once.py`
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
MANIFEST_PATH = REPO_ROOT / "ci" / "release-feature-manifest.json"

sys.path.insert(0, str(REPO_ROOT / "ci" / "scripts"))
import check_gpu_parity_matrix as gpu_parity_matrix  # noqa: E402
import gpu_prove_verdict  # noqa: E402
import check_execution_surface_reachability as exec_mod  # noqa: E402
from check_execution_surface_reachability import (  # noqa: E402
    WorkflowLoadError,
    job_source_spans,
    jobs_or_fail,
    read_top_level_on_block,
    read_top_level_on_block_from_path,
)

# Repo-relative PATH, not a basename (P7's identity discipline applies here
# too): every real invocation site spells the full `ci/scripts/...` path
# (`bash ci/scripts/runpod_gpu_prove.sh`), so matching on the path -- rather
# than the basename -- also closes a same-basename-different-directory
# collision (`ci/scripts/perf/runpod_gpu_prove.sh` would not falsely satisfy
# this identity) and a same-basename PREFIX false-positive
# (`other_runpod_gpu_prove.sh` no longer contains this constant as a
# substring).
PROVE_SCRIPT = "ci/scripts/runpod_gpu_prove.sh"
PROVE_PRODUCER_WORKFLOW = "gpu-prove.yml"
GATE_WORKFLOW = "_gpu-prove-gate.yml"  # the DELETED renting reusable -- must stay gone.
PROOF_REQUIRED_WORKFLOW = "_gpu-proof-required.yml"


@dataclass(frozen=True)
class PromotionRow:
    """One reviewed row of `PROMOTION_TABLE`.

    `gate_kind`:
      - `"direct"`: `gate_job` is a job in THIS workflow that itself
        `uses: _gpu-proof-required.yml` — the promoting job's `needs:`/
        `if:` conjunct names it directly.
      - `"chained"`: `gate_job` is ANOTHER row's `promoting_job` in the
        SAME workflow (already itself gated, directly or chained) — e.g.
        `crates.yml`'s `github-release` chains off `publish`.
      - `"none"`: a reviewed, deliberately UNGATED promotion (e.g.
        `image.yml`/`image-cuda.yml`'s CI base-image rebuild on a merge to
        `main` — never a release tag promotion). `gate_job` is `None`; P3
        (F3 audit fix) instead asserts the promoting job's `if:` is a PURE
        top-level conjunction carrying the EXACT conjunct
        `github.ref_type != 'tag'`, so it can structurally never fire on a
        release tag ref (a substring-absence check on `refs/tags/` used to
        pass an `if:` with no ref restriction at all).

    `step_name`: `None` for the common case (the gate conjunct lives on the
    promoting JOB's own `if:`). When set, the gate conjunct instead lives on
    that ONE named step's `if:` — npm.yml's `publish` job runs build+test
    unconditionally (so a branch dispatch keeps its build-and-test dry run),
    with the actual publish gated at the step level.

    `tag_family` (F7 audit fix): the exact `refs/tags/<family>` prefix this
    row's release tag uses (`"v"` for the default/`crates.io`/`npm`/server-
    image/binaries surface, `"py-v"` for the lockstep Python dist surface).
    For `"direct"`/`"chained"` rows, P3 requires the promoting job's (or
    gated step's) `if:` — and, for `"direct"` rows, the gate job's own
    `if:` too — to carry the exact top-level conjunct
    `startsWith(github.ref, 'refs/tags/<family>')`. Not consulted for
    `"none"` rows, which instead prove unreachability via `github.ref_type
    != 'tag'` (see the module doc's P3 entry).
    """

    workflow: str
    promoting_job: str
    gate_job: str | None
    gate_kind: str  # "direct" | "chained" | "none"
    step_name: str | None = None
    tag_family: str = "v"


# P3's reviewed table -- WHICH job promotes WHAT, and how it is gated, is a
# design decision, never recoverable from the YAML alone, so this is
# hand-maintained. P6 (module doc above) is the DISCOVERY backstop: a
# publishing job with no row here fails by name instead of going unnoticed.
PROMOTION_TABLE: dict[str, PromotionRow] = {
    # ---- CUDA lanes (also the ci/release-feature-manifest.json CUDA lanes) ----
    "cu12-image": PromotionRow("server-image.yml", "build-and-push-cu12", "gpu-proof", "direct"),
    "cu12-tarball": PromotionRow("release-binaries.yml", "server-cu12-promote", "gpu-proof", "direct"),
    "cu12-wheel": PromotionRow("pypi-server-cuda.yml", "publish", "gpu-proof", "direct", tag_family="py-v"),
    # ---- server-image.yml's other arms ----
    "cpu-image-tag": PromotionRow("server-image.yml", "build-and-push", "gpu-proof", "direct"),
    "cpu-image-main": PromotionRow(
        "server-image.yml", "build-and-push-main", None, "none"
    ),  # manual :latest refresh via workflow_dispatch on main (F8 audit fix: server-image.yml carries
    # no push: branches: trigger, so this never fires on a mere merge) -- never a release tag promotion.
    # ---- server-image.yml's two-arch CPU merge jobs (S1/T8): `docker buildx
    # imagetools create` merges the two per-arch immutable sources into the
    # real tags -- itself a promotion, distinct from the per-arch legs above,
    # which push only their own `sha-<sha>-<arch>` tag (never a real tag).
    "cpu-image-merge-tag": PromotionRow(
        "server-image.yml", "merge-cpu-tag", "build-and-push", "chained"
    ),  # chained off build-and-push (itself direct-gated by gpu-proof) -- same tag-family conjunct.
    "cpu-image-merge-main": PromotionRow(
        "server-image.yml", "merge-cpu-main", None, "none"
    ),  # deliberately UNGATED, same as cpu-image-main above -- never a release tag promotion.
    "cpu-image-selfcontained": PromotionRow(
        "server-image.yml", "build-and-push-selfcontained", None, "none"
    ),  # manual dispatch-only opt-in image (Cloudflare Containers) -- pre-existing behavior, out of this unit's scope.
    # ---- image.yml / image-cuda.yml: CI base images (F1 audit fix -- P6's
    # recursion discovers these two `build` jobs BECAUSE they `uses:` the
    # LOCAL reusable `_ci-base-image.yml`, whose own `build-and-push` job
    # pushes to GHCR). Both are reviewed UNGATED rows: they publish the
    # toolchain LAYER the release lanes build inside, on a merge to `main`,
    # never a release tag -- and (F3) each `build` job's own `if:` carries
    # the exact `github.ref_type != 'tag'` conjunct so a workflow_dispatch
    # on a tag ref can never reach them either.
    "ci-image-cpu": PromotionRow("image.yml", "build", None, "none"),
    "ci-image-cuda": PromotionRow("image-cuda.yml", "build", None, "none"),
    # ---- release-binaries.yml's remaining lanes ----
    "cli-binaries": PromotionRow("release-binaries.yml", "promote-binaries", "gpu-proof", "direct"),
    "server-cpu-tarball": PromotionRow("release-binaries.yml", "server-cpu-promote", "gpu-proof", "direct"),
    # ---- crates.io ----
    "crates-publish": PromotionRow("crates.yml", "publish", "gpu-proof", "direct"),
    "crates-github-release": PromotionRow("crates.yml", "github-release", "publish", "chained"),
    # ---- npm ----
    "npm-publish": PromotionRow("npm.yml", "publish", "gpu-proof", "direct", step_name="Publish"),
    # ---- PyPI (lockstep "py-v*" tag family) ----
    "native-wheel": PromotionRow("pypi.yml", "publish", "gpu-proof", "direct", tag_family="py-v"),
    "client-wheel": PromotionRow("pypi-client.yml", "publish", "gpu-proof", "direct", tag_family="py-v"),
    "server-cpu-wheel": PromotionRow("pypi-server.yml", "publish", "gpu-proof", "direct", tag_family="py-v"),
}

_USES_LOCAL_RE = re.compile(r"uses:\s*\./\.github/workflows/([A-Za-z0-9_.-]+)")
_USES_CROSS_REPO_RE = re.compile(r"uses:\s*[\w.-]+/[\w.-]+/\.github/workflows/([A-Za-z0-9_.-]+)@")


# --------------------------------------------------------------------------- #
# Line-level helpers (comment-vs-code, the same rule check_ci_guard_wiring.py
# and check_execution_surface_reachability.py both already apply).
# --------------------------------------------------------------------------- #
def drop_comment_lines(text: str) -> str:
    return "\n".join("" if line.strip().startswith("#") else line for line in text.splitlines())


def load_workflow_texts(workflows_dir: Path) -> dict[str, str]:
    # BLOCK B7 audit fix: GitHub Actions runs BOTH `.yml` and `.yaml`
    # workflow files -- a `*.yml`-only glob is blind to a second producer, a
    # renting reusable, or a `uses:` reference hiding under the `.yaml`
    # spelling. Glob both, deduplicated, sorted for deterministic iteration
    # (the same discipline `check_execution_surface_reachability.py`
    # already applies to its own workflow scan).
    if not workflows_dir.is_dir():
        return {}
    paths = sorted(set(workflows_dir.glob("*.yml")) | set(workflows_dir.glob("*.yaml")))
    return {p.name: p.read_text(encoding="utf-8") for p in paths}


def _workflow_name_variants(name: str) -> list[str]:
    """GitHub treats `.yml` and `.yaml` as the same workflow-file family;
    `LANE_TABLE` and `GATE_WORKFLOW` are hand-maintained with a canonical
    `.yml` spelling, so a lookup against the actually-discovered
    `workflow_texts` (BLOCK B7) must try both spellings rather than assume
    the file on disk matches the constant's own extension literally."""
    if name.endswith(".yml"):
        return [name, name[: -len(".yml")] + ".yaml"]
    if name.endswith(".yaml"):
        return [name, name[: -len(".yaml")] + ".yml"]
    return [name]


def resolve_workflow(workflow_texts: dict[str, str], name: str) -> str | None:
    """The discovered key in `workflow_texts` matching `name` under either
    the `.yml` or `.yaml` spelling, or `None` if neither is present."""
    for variant in _workflow_name_variants(name):
        if variant in workflow_texts:
            return variant
    return None


# --------------------------------------------------------------------------- #
# jobs: block spans -- `jobs_or_fail`/`job_source_spans` (imported above)
# are the ONE reader, shared with `check_execution_surface_reachability.py`
# itself: derived from the REAL parsed document (`yaml.compose`'s own node
# marks), never a second, independently-drifting regex header match. An
# unreadable/unparseable `jobs:` block (quoted, flow-style, a genuine YAML
# syntax error anywhere in the file, ...) is a named FAIL, never a silent
# "this file has zero jobs" -- the same doctrine `read_top_level_on_block`
# already holds `on:` to.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# `if:` expression reconstitution + top-level conjunction scanner (P3's
# structural, parenthesis- and quote-aware rule -- never a substring check).
# --------------------------------------------------------------------------- #
_BLOCK_SCALAR_HEADS = {">", ">-", "|", "|-"}


def reconstruct_if_expr(
    lines: list[str], job_start: int, job_end: int, indent: int = 4
) -> tuple[str | None, str | None]:
    """(expr, error). `expr` is `None` with `error` `None` when the range
    carries no `if:` at all (not itself an error). `error` is set (expr
    `None`) when a found `if:` cannot be read in full -- an unterminated
    block scalar, an empty inline value with no recognized block form.

    `indent` is the exact column an `if:` key must sit at to count --
    4 for a JOB-level `if:` (2-space `jobs:` + 2-space job name), or the
    step's own key column for a STEP-level `if:` (`find_step_key_indent`
    below). This is never a `\\s*`-style loose match: an `if:` at the wrong
    depth (e.g. inside a nested `with:` block, or on a DIFFERENT step) must
    never be mistaken for this range's own condition."""
    if_key_re = re.compile(r"^" + " " * indent + r"if:(.*)$")
    for i in range(job_start, job_end):
        line = lines[i]
        if line.strip().startswith("#"):
            continue
        m = if_key_re.match(line)
        if not m:
            continue
        rest = m.group(1).strip()
        if rest in _BLOCK_SCALAR_HEADS:
            body: list[str] = []
            j = i + 1
            while j < job_end:
                bl = lines[j]
                if bl.strip() == "":
                    j += 1
                    continue
                bl_indent = len(bl) - len(bl.lstrip(" "))
                if bl_indent <= indent:
                    break
                if not bl.strip().startswith("#"):
                    body.append(bl.strip())
                j += 1
            if not body:
                return None, f"if: (line {i + 1}): block scalar `{rest}` has no body -- unterminated block"
            expr = " ".join(body)
        elif rest == "":
            return None, f"if: (line {i + 1}): empty inline value, not a recognized block scalar"
        else:
            expr = rest
        if expr.startswith("${{") and expr.endswith("}}"):
            expr = expr[3:-2].strip()
        if len(expr) >= 2 and expr[0] == expr[-1] and expr[0] in ("'", '"'):
            expr = expr[1:-1]
        return expr, None
    return None, None


def split_top_level(expr: str) -> tuple[list[str], bool]:
    """Quote-aware (single-quoted GitHub strings, `''` escapes), paren-
    depth-aware split on top-level `&&`/`||`. Returns (tokens, balanced) --
    tokens alternate operand/operator/operand/...; `balanced` is False
    (FAIL LOUD, never silently accepted) if paren depth never returns to
    zero or a string is left open at end of input."""
    depth = 0
    in_str = False
    i = 0
    n = len(expr)
    tokens: list[str] = []
    buf: list[str] = []
    while i < n:
        c = expr[i]
        if in_str:
            if c == "'":
                if i + 1 < n and expr[i + 1] == "'":
                    buf.append("''")
                    i += 2
                    continue
                in_str = False
                buf.append(c)
                i += 1
                continue
            buf.append(c)
            i += 1
            continue
        if c == "'":
            in_str = True
            buf.append(c)
            i += 1
            continue
        if c == "(":
            depth += 1
            buf.append(c)
            i += 1
            continue
        if c == ")":
            depth -= 1
            buf.append(c)
            i += 1
            continue
        if depth == 0 and expr[i : i + 2] == "&&":
            tokens.append("".join(buf).strip())
            buf = []
            tokens.append("&&")
            i += 2
            continue
        if depth == 0 and expr[i : i + 2] == "||":
            tokens.append("".join(buf).strip())
            buf = []
            tokens.append("||")
            i += 2
            continue
        buf.append(c)
        i += 1
    tokens.append("".join(buf).strip())
    return tokens, (depth == 0 and not in_str)


def normalize_conjunct(c: str) -> str:
    c = re.sub(r"\s+", " ", c.strip())
    c = re.sub(r"\s*==\s*", " == ", c)
    c = re.sub(r"\s*!=\s*", " != ", c)
    return c


def gate_conjunct(gate_job: str) -> str:
    return f"needs.{gate_job}.result == 'success'"


def tag_guard_conjunct(family: str) -> str:
    """F7 audit fix: the exact top-level conjunct a `"direct"`/`"chained"`
    row's promoting job/step `if:` (and, for `"direct"` rows, the gate
    job's own `if:`) must carry -- pinned by `family` (`PromotionRow.
    tag_family`), never a hand-typed literal at each call site."""
    return f"startsWith(github.ref, 'refs/tags/{family}')"


NONE_ROW_REF_TYPE_CONJUNCT = "github.ref_type != 'tag'"


def check_top_level_conjunct_present(expr: str, want: str, what: str) -> list[str]:
    """Shared structural rule behind P3's `needs.<gate>.result == 'success'`
    check, F3's `github.ref_type != 'tag'` check, and F7's
    `startsWith(github.ref, 'refs/tags/<family>')` check: `expr` must be a
    parenthesis-/quote-aware PURE top-level conjunction (no depth-0 `||`)
    containing `want` as one of its (whitespace/`==`/`!=`-normalized)
    conjuncts. `what` names the missing/violated conjunct in the finding for
    the caller to prefix with its own row/location context."""
    tokens, balanced = split_top_level(expr)
    if not balanced:
        return [f"if: `{expr}` has unbalanced parens/an unterminated string -- refusing to analyze a truncated expression"]
    operators = [t for t in tokens if t in ("&&", "||")]
    conjuncts = [t for t in tokens if t not in ("&&", "||")]
    if "||" in operators:
        return [f"if: `{expr}` carries a depth-0 `||` -- {what} must be a PURE conjunction"]
    normalized = [normalize_conjunct(c) for c in conjuncts]
    want_normalized = normalize_conjunct(want)
    if want_normalized not in normalized:
        return [f"if: `{expr}` has no top-level conjunct equal to `{want}`"]
    return []


def check_promoting_if(expr: str, gate_job: str, tag_family: str | None = None) -> list[str]:
    """The pure-top-level-conjunction rule for one promoting job's (or gated
    step's) already-reconstituted `if:` expression: contains the exact
    `needs.<gate>.result == 'success'` conjunct and, when `tag_family` is
    given (F7 audit fix), the exact `startsWith(github.ref,
    'refs/tags/<family>')` conjunct too, with NO depth-0 `||` anywhere.
    Returns a (possibly empty) findings list -- never raises on a
    malformed-but-parseable expression (a genuinely unreadable one is
    handled by the caller via `reconstruct_if_expr`'s own error return)."""
    tokens, balanced = split_top_level(expr)
    if not balanced:
        return [f"if: `{expr}` has unbalanced parens/an unterminated string -- refusing to analyze a truncated expression"]
    operators = [t for t in tokens if t in ("&&", "||")]
    conjuncts = [t for t in tokens if t not in ("&&", "||")]
    if "||" in operators:
        return [f"if: `{expr}` carries a depth-0 `||` -- a promoting job's condition must be a PURE conjunction"]
    normalized = [normalize_conjunct(c) for c in conjuncts]
    findings: list[str] = []
    want_gate = gate_conjunct(gate_job)
    if want_gate not in normalized:
        findings.append(f"if: `{expr}` has no top-level conjunct equal to `{want_gate}`")
    if tag_family is not None:
        want_tag = tag_guard_conjunct(tag_family)
        if normalize_conjunct(want_tag) not in normalized:
            findings.append(f"if: `{expr}` has no top-level conjunct equal to `{want_tag}` (F7 tag guard)")
    return findings


# --------------------------------------------------------------------------- #
# on: block reader (P1's fail-loud-on-unreadable rule). SHARED: `read_top_
# level_on_block`/`read_top_level_on_block_from_path` live in
# `check_execution_surface_reachability.py` (imported above) -- the ONE
# reader `check_p7_paid_pod_lanes` (push/workflow_call absence), P5, P6, and
# `test_gpu_gang_lane.sh`'s G7 (schedule absence, via this module's own
# `--read-on-block` CLI below) all read the `on:` block through, never a
# second, independently-drifting copy.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# P1 + P2
# --------------------------------------------------------------------------- #
def check_p1_p2(workflow_texts: dict[str, str]) -> list[str]:
    findings: list[str] = []

    producers = sorted(
        name for name, text in workflow_texts.items() if PROVE_SCRIPT in drop_comment_lines(text)
    )
    if not producers:
        findings.append(f"P1: zero workflows invoke {PROVE_SCRIPT} -- the prove lane was deleted everywhere")
    else:
        if PROVE_PRODUCER_WORKFLOW not in producers:
            findings.append(
                f"P1: {PROVE_SCRIPT} is invoked by {producers}, none of which is {PROVE_PRODUCER_WORKFLOW}"
            )
        extra = [p for p in producers if p != PROVE_PRODUCER_WORKFLOW]
        if extra:
            findings.append(
                f"P1: {PROVE_SCRIPT} is invoked by more than one workflow ({producers}) -- "
                f"only {PROVE_PRODUCER_WORKFLOW} may; extra site(s): {extra}"
            )

    prove_text = workflow_texts.get(PROVE_PRODUCER_WORKFLOW)
    if prove_text is None:
        findings.append(f"P1: {PROVE_PRODUCER_WORKFLOW} is missing from the workflow tree")
    else:
        keys, err = read_top_level_on_block(prove_text)
        if err is not None:
            findings.append(f"P1: {PROVE_PRODUCER_WORKFLOW}: {err}")
        else:
            bad_triggers = [k for k in (keys or []) if k in ("push", "workflow_call")]
            if bad_triggers:
                findings.append(
                    f"P1: {PROVE_PRODUCER_WORKFLOW}'s on: block carries {bad_triggers} -- the RULE this "
                    f"gate holds {PROVE_PRODUCER_WORKFLOW} to: the prove workflow carries no push:/"
                    "workflow_call: trigger and no workflow uses: it"
                )

    # BLOCK B7 audit fix: a `uses:` (or a bare GATE_WORKFLOW reference) can
    # name either the `.yml` or the `.yaml` spelling of the target file --
    # both must be caught, never just the constant's own literal extension.
    prove_producer_variants = set(_workflow_name_variants(PROVE_PRODUCER_WORKFLOW))
    gate_workflow_variants = _workflow_name_variants(GATE_WORKFLOW)

    # F2 audit fix (issue #454 round-2): the skip below used to exempt EVERY
    # workflow whose file NAME matched a producer-name spelling
    # (`gpu-prove.yml`/`gpu-prove.yaml`) from the `uses:` scan -- so a
    # sibling file literally named `gpu-prove.yaml` that itself `uses:
    # ./.github/workflows/gpu-prove.yml` passed with zero findings, because
    # ITS OWN name matched the skip set even though it is not the resolved
    # producer. The skip must exempt only the resolved producer file that
    # actually invokes `runpod_gpu_prove.sh` (computed above as
    # `producers`), never both name spellings unconditionally. The real
    # `gpu-prove.yml`'s only `uses:` is `actions/checkout@v4`, so it never
    # self-matches these patterns and needs no skip at all in practice.
    resolved_producer = producers[0] if len(producers) == 1 and producers[0] == PROVE_PRODUCER_WORKFLOW else None

    for name, text in workflow_texts.items():
        if resolved_producer is not None and name == resolved_producer:
            continue
        stripped = drop_comment_lines(text)
        for m in _USES_LOCAL_RE.finditer(stripped):
            if m.group(1) in prove_producer_variants:
                findings.append(f"P1: {name} `uses:` {m.group(1)} -- nothing may call the prove lane")
        for m in _USES_CROSS_REPO_RE.finditer(stripped):
            if m.group(1) in prove_producer_variants:
                findings.append(
                    f"P1: {name} `uses:` a cross-repo reference to {m.group(1)} -- "
                    "nothing may call the prove lane"
                )
        for gate_variant in gate_workflow_variants:
            if gate_variant in stripped:
                findings.append(f"P2: {name} references the deleted renting reusable {gate_variant}")

    return findings


# --------------------------------------------------------------------------- #
# P7 (every PAID POD LANE, not only the prove lane)
# --------------------------------------------------------------------------- #
# P1 states the doctrine for ONE script by name. The doctrine is not about
# that script: it is about a leg that RENTS HARDWARE. Every such leg costs
# money per run, depends on intermittent third-party capacity, and must
# therefore be started deliberately — never by a push, never by another
# workflow calling it, and never from more than one place (two invokers
# means two rentals for one commit, and no single lane owning the verdict).
#
# `PAID_POD_LANE_TABLE` is the reviewed registry of those legs: driver
# script -> the ONE workflow allowed to invoke it. Each row is held to
# exactly P1's three sub-rules:
#
#   (1) exactly one workflow's comment-stripped body invokes the driver, and
#       it is the row's workflow (zero invokers is a FAIL too — a paid lane
#       wired nowhere is a lane that silently stopped running);
#   (2) that workflow's own `on:` block carries neither `push:` nor
#       `workflow_call:` (an unreadable — quoted or flow-style — `on:` block
#       is itself a FAIL, never a silent skip);
#   (3) no OTHER workflow `uses:` it, local or cross-repo form.
#
# The prove lane is a row here as well as P1's subject: P1 additionally
# pins the exactly-once PRODUCER identity that the release verdict depends
# on (and reports its own findings in its own words), while this rule is
# the class the prove lane is one member of. A new paid lane lands as a
# row, in the same commit as its driver and its workflow.
# Keys are repo-relative PATHS, never basenames -- see the module doc's P7
# "IDENTITY IS THE REPO-RELATIVE PATH" paragraph.
PAID_POD_LANE_TABLE: dict[str, str] = {
    # The release-gating proof lane (also P1's own subject).
    "ci/scripts/runpod_gpu_prove.sh": "gpu-prove.yml",
    # The distributed-training gang leg: 1 pod x 2 GPU — the priciest row
    # here per run, and the only one that rents more than one device.
    "ci/scripts/runpod_gpu_gang.sh": "gpu-gang.yml",
    # The within-run GPU perf A/B (two resident clones on one pod).
    "ci/scripts/runpod_gpu_perf_ab.sh": "gpu-perf-ab.yml",
    # The how-well A/B campaign driver.
    "ci/scripts/runpod_gpu_howwell.sh": "gpu-howwell.yml",
    # gpu-dev.sh IS deploy-capable (it can `up` a pod as well as `reap`
    # one), and its one real invoker, gpu-reap.yml, carries RUNPOD_API_KEY
    # at step scope to authenticate the reap call. The whole-file-scope
    # predicate below has no verb parsing to clear the `reap` invocation
    # with, so this row is the price of that rule, not a sign gpu-dev.sh
    # actually rents from this site today -- see `test_gpu_dev_lifecycle.sh`
    # for the lifecycle-safety assertions on what `reap` itself may do.
    "ci/scripts/gpu-dev.sh": "gpu-reap.yml",
}

# --------------------------------------------------------------------------- #
# P7's SUBJECT SET is DERIVED, and the table above is a COMPLETENESS
# ASSERTION over it — never the other way round.
#
# A hand-maintained table of paid lanes has the failure mode its own subject
# matter warns about: the next renting driver is added, nobody remembers the
# table, and the doctrine silently does not apply to it. That is the exact
# escape shape P7 was written to close one file over, reproduced inside P7.
#
# So the set of DEPLOY-CAPABLE functions is computed from `runpod_lib.sh`
# itself — the transitive callers of `_rp_deploy_payload`, the one function
# that builds a pod-creation payload — and a RENTING DRIVER is any tracked
# `ci/scripts/**` file whose comment-stripped text calls a member of that
# closure. Each derived driver must then be EITHER a table row (held to the
# three sub-rules above) OR must satisfy a machine predicate showing that no
# workflow can make it rent (below). A table row whose driver has left the
# derived set is reported as ROT.
#
# WHAT THE DERIVATION DELIBERATELY DOES NOT DO:
#
#   * It does not follow `source`. Every `source` target in this class is
#     variable-interpolated (`source "$DIR/runpod_lib.sh"`), so a transitive
#     "this file sources a file that can deploy" clause is not decidable by
#     a static scan — and a clause that pretended otherwise would be a
#     guess. A driver is judged on the calls IT makes.
#   * It does not parse bash. Function bodies are split at top-level
#     `name() {` starts, and a body runs to the NEXT such start — so
#     top-level code sitting between two functions is attributed to the
#     preceding one. That over-approximates: it can only ADD members to the
#     closure, never drop one, which is the fail-closed direction.
#   * A word-boundary occurrence of a closure member's name in
#     comment-stripped text counts as a call. That over-approximates too,
#     again in the fail-closed direction: `test_check_gpu_prove_once.py`
#     spells the members out in its own fixture strings and is therefore
#     derived as a driver itself. Nothing is exempted for being ours — such
#     a file is cleared by the same machine predicate as any other (the
#     workflow whose guard job runs it carries no `RUNPOD_API_KEY` at any
#     scope), which is exactly the outcome an exemption list would have
#     hidden.
#
# RESIDUALS, disclosed rather than assumed away. All three are
# UNDER-approximations — the direction that can miss a renter — so they are
# named, not argued away:
#
#   * VARIABLE INVOCATION PATH: a workflow step that invokes a driver through
#     a variable path (`bash "$SCRIPT"`) is invisible to the whole-file
#     mention scan below, exactly as it is to every other line-shaped rule in
#     this file. `git grep -nE 'bash +"?\$' -- .github/workflows` returns
#     nothing on this tree, which covers that one spelling only; the scan
#     reports what it can see, and a driver whose path is mentioned in NO
#     workflow file is reported as a NOTE by `_check_derived_driver_cannot_rent`
#     — printed, never a failure and never a clear.
#   * COMPOSED CALL NAME: `_mentions` matches a closure member's name as a
#     word in the driver's comment-stripped text, so a call whose FUNCTION
#     NAME is assembled at run time — `p=rp_deploy_; s=arch; "${p}${s}" a100`
#     — spells no member and the file is not derived as a driver at all. Bash
#     name composition is not decidable by a static scan; the same limit
#     applies to `derive_deploy_closure`'s own caller scan inside
#     `runpod_lib.sh`. What was actually checked, and what it found:
#     `git grep -nE '"\$\{[A-Za-z_][A-Za-z0-9_]*\}\$\{' -- ci/scripts` returns
#     the composed-parameter form in VALUE position — string accumulators in
#     `pod_push_stamp.sh`/`pod_seed_target.sh`/`test_pod_substrate.sh` — AND
#     its own self-match: the very example quoted two sentences up
#     (`"${p}${s}"`) is itself comment text inside THIS file, so the same
#     grep also reports a hit here. It is disclosed rather than filtered out,
#     because filtering "our own doc comment" out by hand is exactly the kind
#     of judgment call a static scan cannot make either — none of the matches
#     on this tree sit in COMMAND position. That is an enumeration of one
#     spelling on one day, not a proof: a driver written this way is missed,
#     which is why the residual is written down here instead of a claim that
#     the derivation is complete.
#   * COMPOSITE ACTIONS: `load_workflow_texts` globs `.github/workflows/*.yml`
#     and `*.yaml` only. `.github/actions/**` composite actions are invisible
#     to it, and therefore to every rule in this file, not only P7's
#     whole-file mention scan — a composite action is never itself an entry
#     in `workflow_texts`. It does NOT get folded into the fallback's
#     whole-file scan (that would require a second glob and a second
#     "carries the secret" question this module does not yet ask). On this
#     tree: `git grep -l RUNPOD_API_KEY -- .github/actions` and
#     `git grep -l 'ci/scripts/runpod' -- .github/actions` both return
#     nothing, so no derived driver's path and no secret sits inside a
#     composite action today — but a future one would be exactly as invisible
#     here as it already is to P1-P6's own workflow-only scans.
# --------------------------------------------------------------------------- #
SCRIPTS_ROOT = "ci/scripts/"
RUNPOD_LIB_REL = "ci/scripts/runpod_lib.sh"
DEPLOY_PAYLOAD_FN = "_rp_deploy_payload"

# The secret that turns a script that CAN deploy into a step that WILL: with
# no `RUNPOD_API_KEY` anywhere in the invoking WORKFLOW, `rp_init` refuses
# before any pod is created. The secret's presence is the capability — its
# scope in the file (top-level env, job env, step env, with:) is not, and
# neither is what verb, if any, a step hands the driver: the fallback rule
# below has no verb parsing left to read one with.
RUNPOD_SECRET = "RUNPOD_API_KEY"

_BASH_FN_DEF_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{", re.MULTILINE)


def _bash_function_bodies(text: str) -> dict[str, str]:
    """`{function name: its (over-approximated) body}` for every top-level
    `name() {` definition in a comment-stripped bash file. See the section
    comment for why the body boundary is the next definition."""
    stripped = drop_comment_lines(text)
    starts = list(_BASH_FN_DEF_RE.finditer(stripped))
    bodies: dict[str, str] = {}
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(stripped)
        bodies[m.group("name")] = bodies.get(m.group("name"), "") + stripped[m.start() : end]
    return bodies


def _mentions(text: str, name: str) -> bool:
    return re.search(r"\b" + re.escape(name) + r"\b", text) is not None


def derive_deploy_closure(lib_text: str) -> tuple[frozenset[str], list[str]]:
    """The deploy-capable function set: the TRANSITIVE CALLERS of
    `_rp_deploy_payload` inside `runpod_lib.sh`. Returns
    `(closure, findings)`; a non-empty `findings` means the closure could
    not be computed and P7 has no subject set — a FAIL, never a skip."""
    bodies = _bash_function_bodies(lib_text)
    if DEPLOY_PAYLOAD_FN not in bodies:
        return frozenset(), [
            f"P7: cannot derive the deploy closure — no `{DEPLOY_PAYLOAD_FN}() {{` definition in "
            f"{RUNPOD_LIB_REL}. P7's subject set is computed from that function's transitive "
            "callers; with no seed there is no set, and every renting driver would go unchecked"
        ]
    members: set[str] = set()
    changed = True
    while changed:
        changed = False
        targets = {DEPLOY_PAYLOAD_FN} | members
        for name, body in bodies.items():
            if name == DEPLOY_PAYLOAD_FN or name in members:
                continue
            if any(_mentions(body, t) for t in targets):
                members.add(name)
                changed = True
    if not members:
        return frozenset(), [
            f"P7: the deploy closure is EMPTY — nothing in {RUNPOD_LIB_REL} calls "
            f"`{DEPLOY_PAYLOAD_FN}`. Either the payload builder was renamed (rename it here too) or "
            "the deploy path moved; an empty closure would silently exempt every renting driver"
        ]
    return frozenset(members), []


def derive_renting_drivers(
    script_texts: dict[str, str], closure: frozenset[str]
) -> dict[str, list[str]]:
    """`{repo-relative script path: the closure members it calls}` for every
    tracked `ci/scripts/**` file except `runpod_lib.sh` itself (which DEFINES
    the closure — its own definitions and internal calls are the seam, not a
    lane)."""
    drivers: dict[str, list[str]] = {}
    for rel, text in sorted(script_texts.items()):
        if rel == RUNPOD_LIB_REL or not rel.startswith(SCRIPTS_ROOT):
            continue
        stripped = drop_comment_lines(text)
        called = sorted(name for name in closure if _mentions(stripped, name))
        if called:
            drivers[rel] = called
    return drivers


def load_script_texts(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """Every TRACKED `ci/scripts/**` file's text, keyed by repo-relative
    path (`git ls-files`, the same enumeration `check_execution_surface_
    reachability.py` and `check_ci_guard_wiring.py` already use — a script
    CI's own checkout would not have is not a lane)."""
    out = subprocess.run(
        ["git", "ls-files", "ci/scripts"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    texts: dict[str, str] = {}
    for rel in out.stdout.splitlines():
        if not rel.startswith(SCRIPTS_ROOT):
            continue
        path = repo_root / rel
        try:
            texts[rel] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return texts


def _check_derived_driver_cannot_rent(
    script_rel: str, workflow_texts: dict[str, str], notes: list[str] | None = None
) -> list[str]:
    """The machine predicate a derived driver that is NOT a
    `PAID_POD_LANE_TABLE` row must satisfy: its repo-relative path is
    mentioned in NO workflow file whose comment-stripped text also carries
    `RUNPOD_SECRET` at any scope. WHOLE-FILE SCOPE, no verb parsing: there is
    no job/step attribution and no first-argument reading here at all — a
    prior revision cleared an invocation whose literal first argument was a
    non-renting verb (`reap`) and read the secret at workflow scope; both the
    verb clearance AND the job-body scoping it was computed over are GONE.
    The only two questions this predicate asks are "does the driver's path
    appear in this file" and "does the secret appear in this file", each
    over the SAME comment-stripped whole-file text.

    That is an over-approximation in the FAIL-CLOSED direction, and
    deliberately so: a `paths:` filter entry naming a derived driver inside a
    secret-holding workflow now demands a row exactly as a real invocation
    would, even though the filter entry alone runs nothing. Parsing what
    kind of YAML construct a path sits inside — a `run:` command, a `paths:`
    filter, a `with:` value — is exactly the kind of case analysis a
    verb-parsing predicate already tried and lost fixtures over; the
    remaining rule asks only whether the two strings share a file. The
    remedy is unchanged — give the driver a PAID_POD_LANE_TABLE row, or drop
    the secret from that workflow (there is no verb to hand it any more).

    The text is still comment-stripped, so a commented-out or documented
    secret, or a commented-out mention of the driver's path, is text, not a
    capability and not a mention.

    THE NOTE CHANNEL FIRES ON ABSENCE. A derived driver whose path is
    mentioned in NO workflow file at all lands in `notes` (printed, never a
    failure): nothing in the committed text establishes that it can rent,
    and nothing establishes that it cannot — a state defined by missing
    evidence gets no definite consequence, and a silent clear would have
    made this file's own prose false. A driver mentioned in at least one
    workflow, none of which carries the secret, produces NEITHER a finding
    NOR a note: it is not absent (so the note does not fire), and nothing
    that mentions it can make it rent (so there is nothing to condemn). That
    silence is this rule's own stated outcome for that shape, not a gap —
    see the module doc's P7 section and the test named for exactly this
    shape."""
    findings: list[str] = []
    mentioned_anywhere = False
    for name in sorted(workflow_texts):
        stripped = drop_comment_lines(workflow_texts[name])
        if script_rel not in stripped:
            continue
        mentioned_anywhere = True
        if RUNPOD_SECRET not in stripped:
            continue
        findings.append(
            f"P7: {script_rel} calls runpod_lib.sh's deploy closure and its repo-relative path "
            f"is mentioned in {name}, whose comment-stripped text also carries {RUNPOD_SECRET} "
            "somewhere (top-level env, job env, step env, with:, or a paths: filter — no verb "
            "parsing clears a mention under this rule) — that workflow can RENT with this "
            "driver. A deploy-capable driver whose path is mentioned in a secret-holding "
            "workflow is a paid pod lane: give it a PAID_POD_LANE_TABLE row (one workflow, no "
            "push:/workflow_call: trigger, nothing uses: it), or drop the secret from that "
            "workflow"
        )
    if not mentioned_anywhere and notes is not None:
        notes.append(
            f"P7 NOTE: {script_rel} calls runpod_lib.sh's deploy closure, but its repo-relative "
            "path is mentioned in no workflow file in this tree (comment-stripped, whole file) — "
            "so nothing here establishes that it can rent, and nothing establishes that it "
            "cannot. Reported, not judged: it is not a failure, and it is not cleared either"
        )
    return findings


def check_p7_paid_pod_lanes(
    workflow_texts: dict[str, str],
    script_texts: dict[str, str] | None = None,
    lib_text: str | None = None,
    notes: list[str] | None = None,
) -> list[str]:
    findings: list[str] = []

    # --- the DERIVED subject set (see the section comment above) ---------- #
    if script_texts is None:
        script_texts = load_script_texts()
    if lib_text is None:
        lib_text = script_texts.get(RUNPOD_LIB_REL)
    if lib_text is None:
        findings.append(
            f"P7: {RUNPOD_LIB_REL} is not in the scanned ci/scripts set — the deploy closure P7's "
            "subject set is derived from cannot be computed, so no renting driver can be checked"
        )
        closure: frozenset[str] = frozenset()
        derived: dict[str, list[str]] = {}
    else:
        closure, closure_findings = derive_deploy_closure(lib_text)
        findings += closure_findings
        derived = derive_renting_drivers(script_texts, closure) if closure else {}

    if closure:
        # Identity is the repo-relative PATH here too: compare `derived`'s
        # own keys against `PAID_POD_LANE_TABLE`'s own keys directly, never
        # via a basename re-keying (`rel.rsplit("/", 1)[-1]`) that would fold
        # `ci/scripts/perf/<x>.sh` and `ci/scripts/<x>.sh` into one bucket —
        # see the module doc's P7 "IDENTITY IS THE REPO-RELATIVE PATH"
        # paragraph. No `rsplit("/")` anywhere in this file.
        table_paths = set(PAID_POD_LANE_TABLE)
        # Rot: a row whose driver no longer calls the deploy closure (it was
        # rewritten, or the closure moved) is a row asserting a fact that
        # stopped being true.
        for rel in sorted(table_paths - set(derived)):
            findings.append(
                f"P7: PAID_POD_LANE_TABLE row `{rel}` names a driver that does NOT call "
                f"runpod_lib.sh's deploy closure ({sorted(closure)}) — the row asserts a paid pod "
                "lane that no longer exists; delete the row, or restore the call"
            )
        # Completeness: a derived driver that is not a row must be provably
        # unable to rent from any workflow.
        for rel in sorted(derived):
            if rel in table_paths:
                continue
            findings += _check_derived_driver_cannot_rent(rel, workflow_texts, notes)

    for script, workflow in sorted(PAID_POD_LANE_TABLE.items()):
        producers = sorted(
            name for name, text in workflow_texts.items() if script in drop_comment_lines(text)
        )
        resolved_workflow = resolve_workflow(workflow_texts, workflow)

        if not producers:
            findings.append(
                f"P7: zero workflows invoke {script} — the paid pod lane it drives is wired nowhere, "
                "so nothing runs it and nothing can be proven by it"
            )
        else:
            # Two DISTINCT states, each with its own message. They used to
            # share one ("more than one workflow") that was simply false for
            # the commonest shape — a single invoker which is the WRONG one —
            # and a finding that misdescribes what it found sends the reader
            # looking for a second site that does not exist.
            if resolved_workflow is None:
                findings.append(
                    f"P7: {script} is invoked by {producers}, but its PAID_POD_LANE_TABLE row names "
                    f"{workflow}, which does not resolve to a workflow in this tree — the row's "
                    "'exactly one invoker' claim cannot be checked against anything"
                )
            elif resolved_workflow not in producers:
                findings.append(
                    f"P7: {script} is invoked by {producers}, none of which is {workflow} — a paid pod "
                    f"lane's driver belongs to the one workflow its row names, so either {workflow} "
                    f"lost the invocation or the row now names the wrong workflow"
                )
            extra = [p for p in producers if p != resolved_workflow]
            if extra and len(producers) > 1:
                findings.append(
                    f"P7: {script} is invoked by more than one workflow ({producers}) — only {workflow} "
                    f"may rent for this lane; extra site(s): {extra}"
                )

        if resolved_workflow is None:
            findings.append(f"P7: {workflow} is missing from the workflow tree")
            continue

        keys, err = read_top_level_on_block(workflow_texts[resolved_workflow])
        if err is not None:
            findings.append(f"P7: {resolved_workflow}: {err}")
        else:
            bad_triggers = [k for k in (keys or []) if k in ("push", "workflow_call")]
            if bad_triggers:
                findings.append(
                    f"P7: {resolved_workflow}'s on: block carries {bad_triggers} — a leg that RENTS "
                    "hardware is never started by a push and is never callable by another workflow "
                    "(it fires on a label, a schedule, or a manual dispatch only)"
                )

        workflow_variants = set(_workflow_name_variants(workflow))
        for name, text in workflow_texts.items():
            if name == resolved_workflow:
                continue
            stripped = drop_comment_lines(text)
            for m in _USES_LOCAL_RE.finditer(stripped):
                if m.group(1) in workflow_variants:
                    findings.append(
                        f"P7: {name} `uses:` {m.group(1)} — nothing may call a paid pod lane"
                    )
            for m in _USES_CROSS_REPO_RE.finditer(stripped):
                if m.group(1) in workflow_variants:
                    findings.append(
                        f"P7: {name} `uses:` a cross-repo reference to {m.group(1)} — nothing may call "
                        "a paid pod lane"
                    )
    return findings


def check_gate_file_absent(workflows_dir: Path) -> list[str]:
    # BLOCK B7 audit fix: check both spellings -- a resurrected
    # `_gpu-prove-gate.yaml` is exactly as real to GitHub as the `.yml` form.
    findings: list[str] = []
    for variant in _workflow_name_variants(GATE_WORKFLOW):
        if (workflows_dir / variant).exists():
            findings.append(f"P2: {variant} still exists -- the renting reusable must be deleted, not merely unused")
    return findings


# --------------------------------------------------------------------------- #
# P3 (PROMOTION_TABLE reconciliation) + P6 (discovery)
# --------------------------------------------------------------------------- #
def cuda_lanes(manifest: dict) -> set[str]:
    return {
        lane
        for lane, spec in manifest.get("lanes", {}).items()
        if "cuda" in spec.get("cargo_features", [])
    }


def _parse_needs_names(job_body: str) -> list[str]:
    """The job-body's `needs:` list, single-line (`needs: a` / `needs: [a,
    b]`) or multi-line (`needs:` then `- a` / `- b` items). `job_body` must
    already be comment-stripped."""
    needs_names: list[str] = []
    # Advisory A7 fix: `[ \t]*`, never `\s*`, right after `needs:` -- `\s`
    # matches `\n` too, so `\s*` would swallow the newline AND the next
    # line's leading whitespace when `needs:` carries no inline value,
    # landing the cursor on the multi-line list's FIRST `- item` and
    # letting `(.*)$` capture `- gpu-proof` as a bogus single literal
    # "needs name" (dash and all) instead of falling through to the
    # multi-line-list branch below.
    needs_m = re.search(r"^[ \t]*needs:[ \t]*(.*)$", job_body, re.MULTILINE)
    if needs_m and needs_m.group(1).strip():
        rest = needs_m.group(1).strip()
        if rest.startswith("["):
            needs_names = [x.strip() for x in rest.strip("[]").split(",") if x.strip()]
        else:
            needs_names = [rest]
    elif needs_m:
        job_lines = job_body.splitlines()
        for idx, l2 in enumerate(job_lines):
            if re.match(r"^\s*needs:\s*$", l2):
                for l3 in job_lines[idx + 1 :]:
                    m3 = re.match(r"^\s*-\s*([A-Za-z0-9_.-]+)\s*$", l3)
                    if m3:
                        needs_names.append(m3.group(1))
                    else:
                        break
                break
    return needs_names


def _step_key_column(lines: list[str], s: int, e: int) -> int | None:
    """The indentation column every top-level key of the step spanning
    [s, e) sits at -- the same computation `_parse_step_keys` makes
    internally, exposed standalone so a step-scoped `if:` can be
    reconstituted at the RIGHT depth (see `find_step_if_by_name`)."""
    dash_line = lines[s]
    bullet_col = len(dash_line) - len(dash_line.lstrip(" "))
    after_dash = dash_line[bullet_col + 1 :]
    after_dash_stripped = after_dash.lstrip(" ")
    if after_dash_stripped.strip() != "":
        return bullet_col + 1 + (len(after_dash) - len(after_dash_stripped))
    for i in range(s + 1, e):
        text = lines[i]
        if text.strip() == "" or text.strip().startswith("#"):
            continue
        return len(text) - len(text.lstrip(" "))
    return None


def find_step_if_by_name(
    lines: list[str], job_start: int, job_end: int, step_name: str
) -> tuple[str | None, str | None, bool]:
    """(expr, error, found). Locates the ONE step directly under this job's
    `steps:` whose `name:` value equals `step_name` exactly, and
    reconstitutes ITS OWN `if:` at that step's own key column -- never a
    job-level `if:` or a different step's. `found` is `False` when no step
    with that name exists in this job at all (its own P3 finding, distinct
    from an unreadable `if:`)."""
    for s, e in _find_step_ranges(lines, job_start, job_end):
        keys = _parse_step_keys(lines, s, e)
        name = _step_display_name(keys)
        if name != step_name:
            continue
        key_col = _step_key_column(lines, s, e)
        if key_col is None:
            return None, f"step `{step_name}`: could not determine its key column", True
        expr, err = reconstruct_if_expr(lines, s, e, indent=key_col)
        return expr, err, True
    return None, None, False


def _step_display_name(keys: dict[str, str]) -> str:
    name = keys.get("name", "").strip()
    if len(name) >= 2 and name[0] == name[-1] and name[0] in ("'", '"'):
        name = name[1:-1]
    return name


def _other_publishing_steps(
    lines: list[str], job_start: int, job_end: int, gated_step_name: str
) -> list[tuple[str, str]]:
    """F4 audit fix: `[(step_name, matched_primitive), ...]` for every step
    in this job OTHER than `gated_step_name` whose own (comment-stripped)
    body itself invokes a publishing primitive -- a step-gated row (P3)
    only ever pinned the NAMED step's `if:`; a second, ungated publishing
    step in the SAME job used to sail through unseen."""
    out: list[tuple[str, str]] = []
    for s, e in _find_step_ranges(lines, job_start, job_end):
        keys = _parse_step_keys(lines, s, e)
        name = _step_display_name(keys)
        if name == gated_step_name:
            continue
        step_body = drop_comment_lines("\n".join(lines[s:e]))
        primitive = job_invokes_publish_primitive(step_body)
        if primitive is not None:
            out.append((name or "<unnamed step>", primitive))
    return out


def check_promotion_table(workflow_texts: dict[str, str], manifest: dict) -> list[str]:
    findings: list[str] = []

    # Subset check (both directions collapsed to one, per the module doc):
    # every manifest CUDA lane must have SOME row promoting it; the reverse
    # (a table row naming no manifest lane) is expected -- most rows promote
    # a non-CUDA, non-manifest surface (crates.io, npm, the CPU wheel/image).
    manifest_lanes = cuda_lanes(manifest)
    table_lanes = set(PROMOTION_TABLE)
    for lane in sorted(manifest_lanes - table_lanes):
        findings.append(f"P3: manifest CUDA lane `{lane}` has no PROMOTION_TABLE row (no promotion chain wired)")

    # promoting_job -> [row keys] per workflow, needed by "chained" rows to
    # confirm their gate_job is itself some OTHER row's promoting job in the
    # SAME workflow.
    promoting_jobs_by_workflow: dict[str, set[str]] = {}
    for row in PROMOTION_TABLE.values():
        promoting_jobs_by_workflow.setdefault(row.workflow, set()).add(row.promoting_job)

    for key in sorted(PROMOTION_TABLE):
        row = PROMOTION_TABLE[key]
        # BLOCK B7 audit fix (carried over from LANE_TABLE): the table's
        # workflow name is a canonical `.yml` literal; resolve either
        # spelling against what was actually discovered on disk.
        resolved_name = resolve_workflow(workflow_texts, row.workflow)
        text = workflow_texts.get(resolved_name) if resolved_name is not None else None
        if text is None:
            findings.append(f"P3: row `{key}`: workflow file {row.workflow} is missing")
            continue
        jobs, jobs_err = jobs_or_fail(text)
        if jobs_err is not None:
            findings.append(f"P3: row `{key}`: {row.workflow}: {jobs_err}")
            continue
        lines = text.splitlines()

        promo_range = jobs.get(row.promoting_job)
        if promo_range is None:
            findings.append(f"P3: row `{key}`: {row.workflow} has no job `{row.promoting_job}`")
            continue
        promo_body = drop_comment_lines("\n".join(lines[promo_range[0] : promo_range[1]]))

        if row.gate_kind == "none":
            # F3 audit fix: a reviewed, deliberately UNGATED promotion must
            # structurally PROVE it can never fire on a release tag ref --
            # its job-level `if:` must be a PURE top-level conjunction
            # carrying the EXACT conjunct `github.ref_type != 'tag'`. A
            # substring-absence check ("no `refs/tags/` mentioned") used to
            # pass an `if:` with no ref restriction at all -- exactly the
            # shape a `workflow_dispatch` on a tag ref can reach.
            expr, err = reconstruct_if_expr(lines, promo_range[0], promo_range[1])
            if err is not None:
                findings.append(f"P3: row `{key}`: {row.workflow}'s promoting job `{row.promoting_job}`: {err}")
            elif expr is None:
                findings.append(
                    f"P3: row `{key}`: {row.workflow}'s promoting job `{row.promoting_job}` is marked "
                    "gate_kind='none' (deliberately ungated) but carries no `if:` at all -- it must "
                    f"structurally exclude a tag ref via the exact conjunct `{NONE_ROW_REF_TYPE_CONJUNCT}`"
                )
            else:
                findings.extend(
                    f"P3: row `{key}`: {row.workflow}'s promoting job `{row.promoting_job}` is marked "
                    f"gate_kind='none' (deliberately ungated): {f}"
                    for f in check_top_level_conjunct_present(
                        expr, NONE_ROW_REF_TYPE_CONJUNCT, "a gate_kind='none' row's condition"
                    )
                )
            continue

        assert row.gate_job is not None  # gate_kind in {"direct", "chained"} always carries a gate_job.
        gate_job = row.gate_job

        if row.gate_kind == "direct":
            gate_range = jobs.get(gate_job)
            if gate_range is None:
                findings.append(f"P3: row `{key}`: {row.workflow} has no gate job `{gate_job}`")
            else:
                gate_body = drop_comment_lines("\n".join(lines[gate_range[0] : gate_range[1]]))
                if f"uses: ./.github/workflows/{PROOF_REQUIRED_WORKFLOW}" not in gate_body:
                    findings.append(
                        f"P3: row `{key}`: {row.workflow}'s gate job `{gate_job}` does not "
                        f"`uses: ./.github/workflows/{PROOF_REQUIRED_WORKFLOW}`"
                    )
                # F7 audit fix: the gate job's OWN `if:` must also carry the
                # row's exact tag-family conjunct -- a gate job reachable
                # off no tag restriction (or the wrong family) would let the
                # verdict be consulted, and satisfied, outside the
                # release-tag path this row exists to gate.
                gate_expr, gate_err = reconstruct_if_expr(lines, gate_range[0], gate_range[1])
                if gate_err is not None:
                    findings.append(f"P3: row `{key}`: {row.workflow}'s gate job `{gate_job}`: {gate_err}")
                elif gate_expr is None:
                    findings.append(
                        f"P3: row `{key}`: {row.workflow}'s gate job `{gate_job}` has no `if:` at all -- "
                        f"F7 tag guard: must carry `{tag_guard_conjunct(row.tag_family)}`"
                    )
                else:
                    findings.extend(
                        f"P3: row `{key}`: {row.workflow}'s gate job `{gate_job}`: {f} (F7 tag guard)"
                        for f in check_top_level_conjunct_present(
                            gate_expr, tag_guard_conjunct(row.tag_family), "a gate job's condition"
                        )
                    )
        else:  # "chained"
            if gate_job not in (promoting_jobs_by_workflow.get(row.workflow, set()) - {row.promoting_job}):
                findings.append(
                    f"P3: row `{key}`: gate_job `{gate_job}` (gate_kind='chained') is not some OTHER "
                    f"row's promoting_job in {row.workflow}"
                )

        needs_names = _parse_needs_names(promo_body)
        if gate_job not in needs_names:
            findings.append(
                f"P3: row `{key}`: {row.workflow}'s promoting job `{row.promoting_job}` does not `needs:` "
                f"`{gate_job}` (found needs={needs_names})"
            )

        if row.step_name is None:
            expr, err = reconstruct_if_expr(lines, promo_range[0], promo_range[1])
            where = f"promoting job `{row.promoting_job}`"
        else:
            expr, err, step_found = find_step_if_by_name(lines, promo_range[0], promo_range[1], row.step_name)
            where = f"promoting job `{row.promoting_job}`'s step `{row.step_name}`"
            if not step_found:
                findings.append(f"P3: row `{key}`: {row.workflow}'s {where} does not exist")
                continue
            # F4 audit fix: a step-gated row only pins the NAMED step's
            # `if:` -- a second, ungated step in the SAME job that itself
            # invokes a publishing primitive used to sail through unseen.
            findings.extend(
                f"P3: row `{key}`: {row.workflow}'s promoting job `{row.promoting_job}` has a SECOND "
                f"step (`{other_name}`) that also invokes a publishing primitive ({primitive}) but is "
                f"not the gated step `{row.step_name}`"
                for other_name, primitive in _other_publishing_steps(
                    lines, promo_range[0], promo_range[1], row.step_name
                )
            )

        if err is not None:
            findings.append(f"P3: row `{key}`: {row.workflow}'s {where}: {err}")
        elif expr is None:
            findings.append(f"P3: row `{key}`: {row.workflow}'s {where} has no `if:` at all")
        else:
            findings.extend(
                f"P3: row `{key}`: {row.workflow}'s {where}: {f}"
                for f in check_promoting_if(expr, gate_job, tag_family=row.tag_family)
            )

    return findings


# Publishing-primitive detection (P6's discovery rule, F1 audit fix: a
# regex list over comment-stripped step bodies and `uses:` lines,
# whitespace-tolerant, never five literal marker strings -- a `cargo
# publish` invocation with two spaces, or a brand-new `twine upload` step,
# used to be entirely invisible).
#
# Simple substring-shaped primitives: any occurrence anywhere in a
# comment-stripped job body is itself a promotion, unconditionally.
_SIMPLE_PRIMITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cargo publish", re.compile(r"cargo\s+publish")),
    ("npm publish", re.compile(r"npm\s+publish")),
    ("twine upload", re.compile(r"twine\s+upload")),
    ("maturin upload", re.compile(r"maturin\s+upload")),
    ("docker push", re.compile(r"docker\s+push")),
    ("gh release create/upload", re.compile(r"gh\s+release\s+(create|upload)")),
    ("pypa/gh-action-pypi-publish", re.compile(r"pypa/gh-action-pypi-publish")),
    ("softprops/action-gh-release", re.compile(r"softprops/action-gh-release")),
    ("ci/scripts/publish_crates.sh", re.compile(r"ci/scripts/publish_crates\.sh")),
    # `docker buildx imagetools create` merges per-arch immutable sources into
    # one multi-arch index under a REAL tag -- itself a promotion, distinct
    # from `imagetools inspect` (read-only, used to assert the merged index's
    # platform set). The pattern is anchored on the FULL three-word primitive
    # (`create`, never bare `imagetools`) so an inspect-only job is never
    # mistaken for one.
    ("docker buildx imagetools create", re.compile(r"docker\s+buildx\s+imagetools\s+create\b")),
)

# `push:`-conditional primitives: matching the marker is not enough on its
# own -- these are ALSO used for build-only verification (`push: "false"`),
# so the marker only counts when the SAME job body also carries a `push:`
# value that is not literally `false`/`"false"`/`'false'` (an unquoted
# `true`, a quoted `'true'`, or any `${{ }}` expression all count, since any
# of these MAY resolve to a push at runtime -- only a literal false
# structurally never can).
_DOCKER_BUILD_PUSH_ACTION_RE = re.compile(r"docker/build-push-action(?:@|\b)")
_LOCAL_DOCKER_PUBLISH_RE = re.compile(r"uses:\s*\./\.github/actions/docker-publish\b")
_CROSS_REPO_DOCKER_PUBLISH_RE = re.compile(r"uses:\s*[\w.-]+/[\w.-]+/\.github/actions/docker-publish@")
_PUSH_VALUE_RE = re.compile(r"^[ \t]*push:[ \t]*(.+?)[ \t]*$", re.MULTILINE)

# `release-upload`: unconditional (unlike docker-publish, this action has no
# `push: "false"`-shaped build-only mode -- every call is a real upload).
_LOCAL_RELEASE_UPLOAD_RE = re.compile(r"uses:\s*\./\.github/actions/release-upload\b")
_CROSS_REPO_RELEASE_UPLOAD_RE = re.compile(r"uses:\s*[\w.-]+/[\w.-]+/\.github/actions/release-upload@")


def _push_value_is_promoting(job_body: str) -> bool:
    for m in _PUSH_VALUE_RE.finditer(job_body):
        val = m.group(1).strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if val.strip().lower() == "false":
            continue
        return True
    return False


def job_invokes_publish_primitive(job_body: str) -> str | None:
    """`job_body` must already be comment-stripped. Returns the matched
    primitive's display name, or `None`. A job whose body invokes ANY
    listed primitive is a "promotion job" for P6's purposes -- it must be
    listed in `PROMOTION_TABLE` (any row, any gate_kind) or this gate fails
    by name. DIRECT match only -- see `job_invokes_publish_primitive_
    recursive` for the "delegates to a local reusable that itself pushes"
    case."""
    for label, pattern in _SIMPLE_PRIMITIVE_PATTERNS:
        if pattern.search(job_body):
            return label
    if _DOCKER_BUILD_PUSH_ACTION_RE.search(job_body) and _push_value_is_promoting(job_body):
        return "docker/build-push-action (push != false)"
    if (
        _LOCAL_DOCKER_PUBLISH_RE.search(job_body) or _CROSS_REPO_DOCKER_PUBLISH_RE.search(job_body)
    ) and _push_value_is_promoting(job_body):
        return "./.github/actions/docker-publish (push != false)"
    if _LOCAL_RELEASE_UPLOAD_RE.search(job_body) or _CROSS_REPO_RELEASE_UPLOAD_RE.search(job_body):
        return "./.github/actions/release-upload"
    return None


def _local_reusable_workflow_targets(job_body: str) -> list[str]:
    """Job-level `uses: ./.github/workflows/<X>.yml` targets referenced
    directly in this job's body (never a step-level action `uses:`, which
    `job_invokes_publish_primitive` already covers by pattern)."""
    return [m.group(1) for m in _USES_LOCAL_RE.finditer(job_body)]


def _workflow_job_bodies(text: str) -> dict[str, str]:
    """{job_id: body} for every job under `text`'s own top-level `jobs:`.
    W10 audit fix: a target this cannot parse/compose is NEVER swallowed
    into `{}` here -- `job_source_spans`'s own `WorkflowLoadError` is left
    to propagate to the caller (`job_invokes_publish_primitive_recursive`),
    which names the RESOLVED PATH of the reusable this body came from before
    re-raising; `check_p6_discovery` is the one place that turns it into a
    FINDING. A workflow reached ONLY through a caller's `uses:` (a
    `workflow_call`-only reusable, P6's own top-level loop skips scanning it
    directly by design) would otherwise have no other path to examination at
    all -- swallowing its load error here made it invisible everywhere."""
    stripped = drop_comment_lines(text)
    lines = stripped.splitlines()
    jobs = job_source_spans(stripped)
    return {name: "\n".join(lines[s:e]) for name, (s, e) in jobs.items()}


def job_invokes_publish_primitive_recursive(
    job_body: str,
    workflow_texts: dict[str, str],
    _visited: frozenset[str] = frozenset(),
) -> str | None:
    """F1 audit fix (RECURSIVE discovery): a direct match first; if none,
    and this job's body itself `uses:` a LOCAL reusable workflow (job-level
    `uses: ./.github/workflows/<X>.yml` -- e.g. `image.yml`'s `build` job
    calling `_ci-base-image.yml`), recurse into THAT workflow's own jobs.
    `_ci-base-image.yml` itself pushes to GHCR (`docker/build-push-action`,
    `push: true`); a job that merely delegates to it is still a promoting
    job for P6's purposes. `_visited` guards a workflow-`uses:`-cycle from
    recursing forever (never expected in this repo's tree, but a guard, not
    an assumption).

    W10 audit fix: a reusable this delegates into but cannot load/compose
    (its own `jobs:` is flow-style, a job value that does not occupy its
    own line span, an anchor/alias/tag, a syntax error, ...) raises
    `WorkflowLoadError` -- RE-RAISED here with the resolved reusable's own
    path folded into the message, never caught and treated as "no
    primitive found". `check_p6_discovery` is the one place that turns this
    into a named finding; every OTHER caller of this function must let it
    propagate too, for the same reason `on_err`/`jobs_err` are never
    discarded at the top level.

    W11 audit fix: the sibling-job loop used to `return` the instant ONE
    sibling job yielded a primitive -- so whenever a LATER sibling job's own
    `uses:` reached a reusable this reader cannot examine, that reusable's
    `_workflow_job_bodies` call was never even made, its `WorkflowLoadError`
    never fired, and the whole chain stayed permanently invisible to P6
    (`image.yml`'s `build` -> a fan-out reusable with a `hop-good` job that
    finds a primitive and a `hop-bad` job whose own target is flow-style:
    0 findings with `hop-good` first, 1 with `hop-bad` first -- the verdict
    depended on job order, which the property below forbids). The fix:
    EVERY sibling reachable through a given `uses:` target -- and every
    target a job body itself names -- is visited before this function ever
    returns; a refusal anywhere in that whole reachable set is remembered
    and re-raised, never masked by another sibling's find, with the
    resolved target folded onto the FRONT of the message so a multi-hop
    chain reads as the true edge sequence (`caller's job -> mid -> bad`),
    never a single flattened edge naming only the last hop."""
    direct = job_invokes_publish_primitive(job_body)
    if direct is not None:
        return direct
    found_result: str | None = None
    first_refusal: WorkflowLoadError | None = None
    for target in _local_reusable_workflow_targets(job_body):
        resolved = resolve_workflow(workflow_texts, target)
        if resolved is None or resolved in _visited:
            continue
        target_text = workflow_texts[resolved]
        try:
            sub_bodies = _workflow_job_bodies(target_text)
        except WorkflowLoadError as exc:
            if first_refusal is None:
                first_refusal = WorkflowLoadError(
                    f"uses local reusable workflow {resolved!r}, whose jobs: cannot be examined: {exc}"
                )
                first_refusal.__cause__ = exc
            continue
        # Collect refusals AND finds over the whole sibling set before any
        # short-circuit -- a sibling that already found a primitive must
        # never stop this loop from reaching a LATER sibling's own
        # unexaminable reusable.
        for sub_body in sub_bodies.values():
            try:
                found = job_invokes_publish_primitive_recursive(
                    sub_body, workflow_texts, _visited=_visited | {resolved}
                )
            except WorkflowLoadError as exc:
                if first_refusal is None:
                    first_refusal = WorkflowLoadError(
                        f"uses local reusable workflow {resolved!r}, which reaches an "
                        f"unexaminable reusable: {exc}"
                    )
                    first_refusal.__cause__ = exc
                continue
            if found is not None and found_result is None:
                found_result = f"{found} (via {resolved})"
    if first_refusal is not None:
        raise first_refusal
    return found_result


def check_p6_discovery(workflow_texts: dict[str, str]) -> list[str]:
    findings: list[str] = []
    listed = {(row.workflow, row.promoting_job) for row in PROMOTION_TABLE.values()}
    # Every table row's workflow may be discovered under either the `.yml`
    # or `.yaml` spelling actually on disk (BLOCK B7 discipline) -- widen
    # the listed set to both spellings so a row naming the canonical `.yml`
    # form still matches a `.yaml` file discovered on disk.
    listed_resolved: set[tuple[str, str]] = set()
    for workflow, job in listed:
        resolved = resolve_workflow(workflow_texts, workflow)
        listed_resolved.add((resolved if resolved is not None else workflow, job))

    for name, text in sorted(workflow_texts.items()):
        # F2 audit fix: NO trigger filtering at all -- every workflow file
        # is in scope; a publishing primitive anywhere must be in the
        # table, regardless of what triggers the file. A workflow whose OWN
        # `on:` block is `workflow_call`-only (the same "never independently
        # starts" doctrine P1/P5 hold `gpu-prove.yml`/`_gpu-proof-
        # required.yml` to) is inert without a caller and is reached only
        # via the recursive check below on that caller's job -- it is
        # skipped here so it is never double-tabled against itself. An
        # unreadable `on:`/`jobs:` block is a FAIL LOUD, never a silent
        # skip (same doctrine P1 already holds `gpu-prove.yml`'s `on:` to).
        on_keys, on_err = read_top_level_on_block(text)
        if on_err is not None:
            findings.append(f"P6: {name}: {on_err}")
            continue
        if on_keys == ["workflow_call"]:
            continue
        jobs, jobs_err = jobs_or_fail(text)
        if jobs_err is not None:
            findings.append(f"P6: {name}: {jobs_err}")
            continue
        assert jobs is not None
        lines = text.splitlines()
        for job_name, (start, end) in jobs.items():
            body = drop_comment_lines("\n".join(lines[start:end]))
            # W10 audit fix: a reusable this job's `uses:` reaches but that
            # cannot be loaded/composed is a named FINDING here -- never a
            # silent "no primitive found". This is the ONLY place in the
            # tree that would otherwise ever examine a `workflow_call`-only
            # reusable (the loop above skips scanning it as its own
            # top-level entry, by design); swallowing the load error inside
            # the recursive helper made such a reusable invisible to P6
            # entirely, regardless of what its own `jobs:` actually did.
            try:
                primitive = job_invokes_publish_primitive_recursive(body, workflow_texts)
            except WorkflowLoadError as exc:
                findings.append(f"P6: {name}'s job `{job_name}` {exc}")
                continue
            if primitive is None:
                continue
            if (name, job_name) not in listed_resolved:
                findings.append(
                    f"P6: {name}'s job `{job_name}` invokes a publishing primitive ({primitive}) but is "
                    "not listed in PROMOTION_TABLE -- an unlisted promoting job is invisible to the "
                    "gpu-prove-once guarantee; add a reviewed row for it"
                )

    return findings


# --------------------------------------------------------------------------- #
# P4
# --------------------------------------------------------------------------- #
def check_p4(workflow_texts: dict[str, str], shipped_arches: set[str]) -> list[str]:
    findings: list[str] = []
    text = workflow_texts.get(PROVE_PRODUCER_WORKFLOW)
    if text is None:
        return [f"P4: {PROVE_PRODUCER_WORKFLOW} is missing -- cannot verify job-name/arch agreement"]

    name_lines = [l for l in text.splitlines() if re.match(r"^\s*name:\s*GPU prove on RunPod", l)]
    want_name = "name: " + gpu_prove_verdict.JOB_NAME_TEMPLATE.format(arch="${{ matrix.arch }}")
    if not any(l.strip() == want_name for l in name_lines):
        findings.append(
            f"P4: {PROVE_PRODUCER_WORKFLOW}'s matrix job `name:` does not match "
            f"gpu_prove_verdict.JOB_NAME_TEMPLATE (want `{want_name}`, found {name_lines})"
        )

    arch_m = re.search(r"arch:\s*\[([^\]]*)\]", text)
    if arch_m is None:
        findings.append(f"P4: {PROVE_PRODUCER_WORKFLOW} has no `arch: [...]` matrix list")
    else:
        workflow_arches = {a.strip() for a in arch_m.group(1).split(",") if a.strip()}
        if workflow_arches != shipped_arches:
            findings.append(
                f"P4: {PROVE_PRODUCER_WORKFLOW}'s matrix arch list {sorted(workflow_arches)} != "
                f"shipped GENCODE_ARCHES {sorted(shipped_arches)}"
            )
    return findings


# --------------------------------------------------------------------------- #
# P5 (BLOCK B8 audit fix, round-2 F1 hardening): the reusable actually
# CONSULTS the verdict, as an un-bypassable step -- never a whole-file
# substring check.
# --------------------------------------------------------------------------- #
_SHA_ARG_RE = re.compile(r'--sha\s+(?:"(?P<q>[^"]*)"|(?P<u>\$\{\{[^}]*\}\}|\S+))')
_REPO_ARG_RE = re.compile(r'--repo\s+(?:"(?P<q>[^"]*)"|(?P<u>\$\{\{[^}]*\}\}|\S+))')
_WORKFLOW_ARG_RE = re.compile(r'--workflow\s+(?:"(?P<q>[^"]*)"|(?P<u>\$\{\{[^}]*\}\}|\S+))')
_GPU_PROVE_VERDICT_INVOCATION = "python3 ci/scripts/gpu_prove_verdict.py"
_CONTROL_OPERATOR_RE = re.compile(r"\|\||;|&&")


def _sha_arg_is_commit_bound(value: str) -> bool:
    """`True` only for the two shapes that key the verdict by the exact
    commit a caller promotes: the bash env var `$GITHUB_SHA`, or the GitHub
    expression `${{ github.sha }}` (any internal whitespace). A literal sha
    or a tag name (`v1.2.3`, `${{ github.ref_name }}`, ...) is REFUSED --
    proof surface == shipped surface (esc-081/esc-084) means the verdict
    lookup itself must be bound to the identity being promoted, never a
    sibling ref."""
    value = value.strip()
    if value == "$GITHUB_SHA":
        return True
    if value.startswith("${{") and value.endswith("}}"):
        return value[3:-2].strip() == "github.sha"
    return False


def _repo_arg_is_bound(value: str) -> bool:
    """F6 audit fix: `True` only for the two shapes that key the verdict
    lookup at THIS repo: `$GITHUB_REPOSITORY`, or `${{ github.repository
    }}`. A literal/foreign `owner/repo` would read a DIFFERENT repository's
    runs as if they proved this commit."""
    value = value.strip()
    if value == "$GITHUB_REPOSITORY":
        return True
    if value.startswith("${{") and value.endswith("}}"):
        return value[3:-2].strip() == "github.repository"
    return False


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value


def _find_step_ranges(lines: list[str], job_start: int, job_end: int) -> list[tuple[int, int]]:
    """Raw 0-based (start, end_exclusive) line ranges for each `- ` list
    item directly under this job's `steps:` key. Bullet column is taken
    from the FIRST bullet found after `steps:`; a dedent below that column
    (or a non-bullet line at that column) ends the list."""
    steps_line = None
    for i in range(job_start, job_end):
        if re.match(r"^\s*steps:\s*(#.*)?$", lines[i]):
            steps_line = i
            break
    if steps_line is None:
        return []
    bullet_col: int | None = None
    starts: list[int] = []
    end_of_list = job_end
    for i in range(steps_line + 1, job_end):
        line = lines[i]
        if line.strip() == "" or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.lstrip(" ")
        if bullet_col is None:
            bullet_col = indent
        if indent < bullet_col:
            end_of_list = i
            break
        if indent == bullet_col:
            if not stripped.startswith("-"):
                end_of_list = i
                break
            starts.append(i)
    if not starts:
        return []
    ranges: list[tuple[int, int]] = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else end_of_list
        ranges.append((s, e))
    return ranges


def _parse_step_keys(lines: list[str], s: int, e: int) -> dict[str, str]:
    """{key: value_text} of every top-level key directly inside this one
    step (the `- ` list item spanning [s, e)), whether the first key sits
    inline on the dash line (`- name: Foo`) or the dash is bare and the
    first key follows on its own line. A block-scalar value (`>`/`>-`/
    `|`/`|-`) is expanded to its full joined body (newline-separated, each
    physical line's own indentation stripped) so a multi-line `run:` is
    inspected whole, not just its `run: |` header line."""
    dash_line = lines[s]
    bullet_col = len(dash_line) - len(dash_line.lstrip(" "))
    after_dash = dash_line[bullet_col + 1 :]
    after_dash_stripped = after_dash.lstrip(" ")
    scan: list[tuple[int, str]] = []
    if after_dash_stripped.strip() != "":
        scan.append((s, after_dash_stripped))
    for i in range(s + 1, e):
        scan.append((i, lines[i]))

    # Key column: either the inline dash-line key's own column, or the
    # first subsequent line's indentation (bare-dash form).
    if after_dash_stripped.strip() != "":
        key_col = bullet_col + 1 + (len(after_dash) - len(after_dash_stripped))
    else:
        key_col = None
        for li, text in scan:
            if text.strip() == "" or text.strip().startswith("#"):
                continue
            key_col = len(text) - len(text.lstrip(" "))
            break
        if key_col is None:
            return {}

    results: dict[str, str] = {}
    idx = 0
    n = len(scan)
    while idx < n:
        li, text = scan[idx]
        is_inline_dash = li == s and after_dash_stripped.strip() != ""
        if text.strip() == "" or text.strip().startswith("#"):
            idx += 1
            continue
        if is_inline_dash:
            content = text
        else:
            indent = len(text) - len(text.lstrip(" "))
            if indent != key_col:
                idx += 1
                continue
            content = text.strip()
        m = re.match(r"^([A-Za-z0-9_.-]+):\s*(.*)$", content)
        if not m:
            idx += 1
            continue
        key, val = m.group(1), m.group(2).strip()
        if val in _BLOCK_SCALAR_HEADS:
            body: list[str] = []
            j = idx + 1
            while j < n:
                bli, btext = scan[j]
                if btext.strip() == "":
                    j += 1
                    continue
                bindent = len(btext) - len(btext.lstrip(" "))
                if bindent <= key_col:
                    break
                if not btext.strip().startswith("#"):
                    body.append(btext.strip())
                j += 1
            results[key] = "\n".join(body)
            idx = j
            continue
        results[key] = val
        idx += 1
    return results


def _join_shell_continuations(text: str) -> list[str]:
    """Logical (backslash-continuation-joined) lines of a shell `run:`
    body -- each physical line ending in a trailing `\\` is folded onto the
    next, so a multi-line invocation's arguments become one line to scan
    for a trailing control operator or the LAST `--sha`."""
    logical: list[str] = []
    buf = ""
    for line in text.splitlines():
        piece = line.strip()
        buf = f"{buf} {piece}".strip() if buf else piece
        if buf.endswith("\\"):
            buf = buf[:-1].rstrip()
            continue
        logical.append(buf)
        buf = ""
    if buf:
        logical.append(buf)
    return logical


def check_p5(workflow_texts: dict[str, str]) -> list[str]:
    """P3 only checks a gate job's `uses:` line -- gutting
    `_gpu-proof-required.yml` to `run: echo ok` would leave P1-P4 green
    while no promotion is actually conditioned on a real verdict lookup.
    P5 asserts the reusable ITSELF: it must exist, its `on:` block must be
    `workflow_call`-only (the same never-independently-starts doctrine P1
    holds the producer to), and it must contain a real STEP -- not a
    `name:`/`env:` mention, not a quoted echo string -- whose `run:` body
    invokes `python3 ci/scripts/gpu_prove_verdict.py` as an actual shell
    command, with no `||`/`;`/`&&` after the invocation on its
    continuation-joined logical line (a trailing `|| true` or a `--sha`
    that never runs would fail open), no `continue-on-error:`/`if:` on
    either that step or its job (either would let the verdict check be
    skipped or silenced), and whose LAST `--sha` argument (argparse's own
    last-wins semantics, never a first-match regex) is bound to the commit
    being promoted (`github.sha`/`$GITHUB_SHA`) -- never a literal sha or a
    tag name."""
    findings: list[str] = []
    resolved = resolve_workflow(workflow_texts, PROOF_REQUIRED_WORKFLOW)
    if resolved is None:
        return [f"P5: {PROOF_REQUIRED_WORKFLOW} is missing from the workflow tree"]
    text = workflow_texts[resolved]

    keys, err = read_top_level_on_block(text)
    if err is not None:
        findings.append(f"P5: {resolved}: {err}")
    elif keys != ["workflow_call"]:
        findings.append(
            f"P5: {resolved}'s on: block must be `workflow_call`-only (found {keys}) -- the reusable "
            "must never independently start anything"
        )

    stripped_text = drop_comment_lines(text)
    lines = stripped_text.splitlines()
    jobs, jobs_err = jobs_or_fail(stripped_text)
    if jobs_err is not None:
        findings.append(f"P5: {resolved}: {jobs_err}")
        return findings

    valid_found = False
    for job_start, job_end in jobs.values():
        job_if_present = any(
            re.match(r"^    if:", lines[i]) for i in range(job_start, job_end)
        )
        job_coe_present = any(
            re.match(r"^    continue-on-error:", lines[i]) for i in range(job_start, job_end)
        )
        for step_start, step_end in _find_step_ranges(lines, job_start, job_end):
            step_keys = _parse_step_keys(lines, step_start, step_end)
            run_text = step_keys.get("run")
            if run_text is None or _GPU_PROVE_VERDICT_INVOCATION not in run_text:
                continue
            logical_lines = _join_shell_continuations(run_text)
            invocation_line = next(
                (ll for ll in logical_lines if ll.startswith(_GPU_PROVE_VERDICT_INVOCATION)), None
            )
            if invocation_line is None:
                # The invocation text is present in this step's `run:` body
                # (e.g. inside a quoted `echo '...'`) but is not itself the
                # command that runs -- not a real invocation site.
                continue
            remainder = invocation_line[len(_GPU_PROVE_VERDICT_INVOCATION) :]
            if _CONTROL_OPERATOR_RE.search(remainder):
                findings.append(
                    f"P5: {resolved} invokes gpu_prove_verdict.py but a shell control operator "
                    f"(`||`/`;`/`&&`) follows it on its logical line (`{invocation_line}`) -- the "
                    "verdict check could fail open"
                )
                continue
            if "if" in step_keys or "continue-on-error" in step_keys or job_if_present or job_coe_present:
                findings.append(
                    f"P5: {resolved}'s step invoking gpu_prove_verdict.py (or its job) carries "
                    "`if:`/`continue-on-error:` -- the verdict check could be skipped or its "
                    "failure silenced"
                )
                continue
            sha_matches = list(_SHA_ARG_RE.finditer(invocation_line))
            if not sha_matches:
                findings.append(f"P5: {resolved} invokes gpu_prove_verdict.py with no --sha argument at all")
                continue
            # LAST occurrence wins, matching argparse's own last-flag-wins
            # semantics -- never the first match a naive regex would find.
            sha_m = sha_matches[-1]
            raw = sha_m.group("q") if sha_m.group("q") is not None else sha_m.group("u")
            if not _sha_arg_is_commit_bound(raw):
                findings.append(
                    f"P5: {resolved}'s --sha argument is `{raw}`, not bound to `github.sha`/`$GITHUB_SHA` -- "
                    "a literal sha or a tag name would key the verdict by the wrong identity"
                )
                continue
            # F6 audit fix: --repo must be pinned to THIS repo too -- a
            # literal/foreign owner/repo would read a different
            # repository's runs as if they proved this commit.
            repo_matches = list(_REPO_ARG_RE.finditer(invocation_line))
            if not repo_matches:
                findings.append(f"P5: {resolved} invokes gpu_prove_verdict.py with no --repo argument at all")
                continue
            repo_m = repo_matches[-1]
            repo_raw = repo_m.group("q") if repo_m.group("q") is not None else repo_m.group("u")
            if not _repo_arg_is_bound(repo_raw):
                findings.append(
                    f"P5: {resolved}'s --repo argument is `{repo_raw}`, not bound to `github.repository`/"
                    "`$GITHUB_REPOSITORY` -- a literal/foreign repo would key the verdict lookup at the "
                    "wrong repo"
                )
                continue
            # F6 audit fix: a --workflow override, if present at all, may
            # never name anything other than gpu-prove.yml itself -- a
            # pointed-elsewhere consumer could read a DIFFERENT, unrelated
            # workflow's runs as if they proved this one.
            workflow_matches = list(_WORKFLOW_ARG_RE.finditer(invocation_line))
            if workflow_matches:
                wf_m = workflow_matches[-1]
                wf_raw = wf_m.group("q") if wf_m.group("q") is not None else wf_m.group("u")
                wf_val = _unquote(wf_raw)
                if wf_val != gpu_prove_verdict.DEFAULT_WORKFLOW:
                    findings.append(
                        f"P5: {resolved} overrides --workflow to `{wf_val}` -- only "
                        f"`{gpu_prove_verdict.DEFAULT_WORKFLOW}` may ever be consulted"
                    )
                    continue
            valid_found = True

    if not valid_found and not findings:
        findings.append(
            f"P5: {resolved} does not invoke {_GPU_PROVE_VERDICT_INVOCATION} as a real step's `run:` "
            "command (a mention in `name:`/`env:`/a quoted echo string does not count)"
        )
    return findings


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_gate(
    workflows_dir: Path = WORKFLOWS_DIR,
    manifest_path: Path = MANIFEST_PATH,
    script_texts: dict[str, str] | None = None,
    notes: list[str] | None = None,
) -> list[str]:
    """`notes` collects the OBSERVATIONS this gate makes that are not
    verdicts (today: a derived renting driver no visible workflow step
    invokes). They are printed by `main`; they never change the exit code,
    so a note can never be read as a pass or as a failure."""
    workflow_texts = load_workflow_texts(workflows_dir)
    if not workflow_texts:
        return ["no workflow files found -- cannot verify the gpu-prove-once property"]
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    findings: list[str] = []
    findings += check_p1_p2(workflow_texts)
    findings += check_p7_paid_pod_lanes(workflow_texts, script_texts, notes=notes)
    findings += check_gate_file_absent(workflows_dir)
    findings += check_promotion_table(workflow_texts, manifest)
    findings += check_p4(workflow_texts, gpu_parity_matrix.load_shipped_cuda_silicon())
    findings += check_p5(workflow_texts)
    findings += check_p6_discovery(workflow_texts)
    return findings


def _cli_read_on_block(path: Path) -> int:
    """`--read-on-block <path>` CLI form of the shared `on:` block reader
    (X1): prints each top-level trigger key on its own line and exits 0, or
    prints the reader's own "cannot read"/"cannot examine" message to
    stderr and exits 1 -- an unreadable path is the same FAIL, never a
    silent "no key". `test_gpu_gang_lane.sh`'s G7 shells out to this exact
    CLI so the bash lane suite and this gate's own P7 arm read the `on:`
    block through one function, never two independently-drifting regexes."""
    keys, err = read_top_level_on_block_from_path(path)
    if err is not None:
        print(err, file=sys.stderr)
        return 1
    for k in keys or []:
        print(k)
    return 0


def main() -> int:
    # A missing PyYAML install is a GATE PREREQUISITE failure, never a
    # finding and never a pass -- checked FIRST, before EITHER CLI form
    # runs (and before any `--self-test`-shaped dispatch this script might
    # grow), via the ONE predicate `check_execution_surface_reachability.py`
    # exports for every importing gate. Returns a distinct code (3, never
    # 2 -- `--read-on-block`'s own usage-error arm already returns that)
    # so it is never mistaken for a normal usage error either.
    prereq_rc = exec_mod.require_pyyaml_or_exit("gpu-prove-once", exit_code=3)
    if prereq_rc is not None:
        return prereq_rc
    argv = sys.argv[1:]
    if argv and argv[0] == "--read-on-block":
        if len(argv) != 2:
            print("usage: check_gpu_prove_once.py --read-on-block <path>", file=sys.stderr)
            return 2
        return _cli_read_on_block(Path(argv[1]))
    notes: list[str] = []
    findings = run_gate(notes=notes)
    for n in notes:
        print(f"  * {n}")
    if findings:
        print("gpu-prove-once: FAIL", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("gpu-prove-once: OK -- exactly one prove producer, no renting reusable, every release "
          "publisher's promotion gates on the shared verdict (all-or-nothing, not only the CUDA "
          "lanes), consumer/producer names agree, the reusable actually consults the verdict keyed "
          "by the promoted commit, no publishing job in the tree is unlisted, every paid pod "
          "lane in PAID_POD_LANE_TABLE (keyed by repo-relative path, never a basename) has exactly "
          "one invoker whose on: block carries no push:/workflow_call: trigger and which nothing "
          "uses:, and every renting driver DERIVED from runpod_lib.sh's own deploy closure is "
          "either such a row or has its path mentioned in no workflow file whose comment-stripped "
          "text carries the secret at any scope (whole-file scope, no verb parsing clears an "
          "invocation) -- a driver mentioned in no workflow at all reported above as a NOTE rather "
          "than cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
