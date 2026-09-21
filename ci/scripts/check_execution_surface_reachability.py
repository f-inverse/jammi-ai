#!/usr/bin/env python3
"""Execution-surface reachability gate — hermetic, static, no build, no GPU.

## The failure this closes: a seed tuple unguarded on the merge path

`ci/scripts/pod_seed_target.sh` runs a fixed set of CUDA/CUTLASS-toolchain-
gated `cargo` invocations (its own T1/T1b/T2/T3 tuples) on every fresh pod's
auto-seed — a leg reds the WHOLE seed the moment its own tuple regresses.
`ci/scripts/runpod_gpu_prove.sh` (invoked from exactly ONE place,
`gpu-prove.yml` — `workflow_dispatch` / `pull_request: types: [labeled]` /
nightly `schedule`, NEVER a trigger that fires on every PR-to-main or
push-to-main, and never `push:`/`workflow_call:`-able either
(`check_gpu_prove_once.py`'s P1 rule pins this by name); every CUDA release
lane consumes its already-recorded verdict instead of invoking it a second
time — see the allowlist's own notes) carries a byte-identical twin of
several of those same tuples.

"Does this invocation appear in SOME workflow's run body?" is the wrong
question: it has no notion of `on:` triggers at all — a tuple
wired only into a dispatch/label/schedule-only workflow satisfies it while
NOTHING on the actual merge path ever runs it. Example: `pod_seed_target.sh`'s
`cargo clippy -p jammi-kernels --all-targets --features cuda -- -D warnings`
can go red on a fresh pod's seed while its only workflow-level twin lives
behind `gpu-prove.yml`'s label/dispatch/schedule-only trigger — green
"wiring", dead on the path that gates merges. The prove lane carries no
clippy invocation (it never needs a GPU to run clippy); `ci.yml`'s hermetic
`Clippy jammi-kernels --features flash-attn --all-targets` step (no GPU, no
`gpu-prove.yml` dependency) is the merge-path clippy coverage — see
`check_lint_surface_closure.py`'s own module doc.

## Rule 1 — reachability

Every REGISTERED execution-surface tuple (see Rule 2) must be reachable on
the merge path: its own `cargo <subcommand> ...` invocation, in the SAME
NORMALIZED form `extract_tuples_from_line` produces on both sides (an
env-var-assignment/wrapper-prefix chain is stripped identically on BOTH
sides; a trailing `||`-tail is stripped on the REGISTRY side unconditionally
but only credited-through on the WORKFLOW side when it matches a KNOWN
fail-loud shape — see `_FAIL_LOUD_FALLBACK_RE` below, otherwise the whole
segment is refused rather than stripped — NOT literal source-byte
equality; see that function's own docstring for the honest consequence: a
workflow-side twin wrapped in a DIFFERENT env prefix than the registered
script still credits, because the underlying invocation genuinely is the
same command once normalized), must
appear inside a JOB+STEP of a workflow whose `on:` block genuinely fires on
the merge path AND whose path filter (if any) is capable of matching the
tuple's own origin path AND whose job/step is not conditioned off (see the
three sub-sections below). "Reachable" is a conjunction of three
independent honesty checks — a tuple satisfying only one or two is still
UNREACHABLE:

### 1a — trigger honesty

A `push` whose `branches` includes `main` (or carries no `branches`/
`branches-ignore`/`tags`/`tags-ignore` filter at all — fires on every ref
push, `main` included), or whose `branches-ignore` does NOT list `main`; or
a `pull_request` whose `types` is either unset or intersects the GitHub
default PR lifecycle types (`opened`/`synchronize`/`reopened`) AND whose
`branches` is either unset or includes `main` (or whose `branches-ignore`
does not list `main`). `workflow_dispatch`, `schedule`, `workflow_call`
(reachable only via a caller's OWN `uses:`, not evaluated transitively —
see the residual note below), a `push` scoped ONLY by `tags`/`tags-ignore`
(no `branches` key at all — GitHub's own documented semantics: such a
trigger does not fire for ordinary branch pushes), and a `pull_request`
whose `types` is some OTHER set entirely (`gpu-prove.yml`'s `[labeled]`) do
NOT count — parsed honestly from each workflow's own `on:` block
(`parse_on_block` / `merge_path_lanes` below), never assumed from the
workflow's file name or its job names.

### 1b — path-filter capability

A merge-path trigger's OWN `paths:`/`paths-ignore:` filter (per-trigger,
not per-workflow — `docs.yml` carries DIFFERENT `paths:` lists under its
`push:` and `pull_request:` blocks) must be CAPABLE of matching the
specific tuple's origin path (its `ci/scripts/**` source file) before that
trigger credits anything: a workflow whose `on:` block otherwise fires on
the merge path but whose `paths:` allowlist can never match a change under
`ci/scripts/**` (eight such workflows exist: `docs.yml`, `image.yml`,
`image-cuda.yml`, `dep-dag.yml` — whose one `ci/scripts/` entry is the
single literal file `ci/scripts/gen_dep_dag.py`, never a glob covering the
whole directory — `devcontainer-image.yml`, `pypi-server.yml`,
`pypi-server-cuda.yml`, `server-image.yml`) would never actually RUN in
response to an edit of `pod_seed_target.sh`/`runpod_gpu_prove.sh`, so
crediting it as reachability is illusory regardless of what text happens to
sit in its run bodies. `_glob_to_regex` translates a GitHub Actions path
glob (`**`, `*`, `?`, literal segments) to a regex; `_lane_admits_any_origin`
requires at least one of the tuple's own recorded origins to match.

Every pattern in every lane of every scanned merge-path workflow is
validated EAGERLY, in `scan_workflows` itself, unconditionally:
`_lane_admits_path`'s own `any()` short-circuits at the
first matching pattern, so a per-tuple-triggered validation would never
even LOOK at an unsupported pattern sitting AFTER an earlier one that
already matched (`paths: ["**", "!ci/scripts/**"]` would credit with zero
findings — the exact over-broad-admit `_glob_to_regex`'s own docstring
names — since `**` alone satisfies the `any()` and the loop never reaches
the `!`-prefixed entry); and a merge-path workflow whose bad pattern sits
in a lane NO gated tuple ever routes a reachability check through would
otherwise never be validated at all. `_validate_lane_patterns` closes both:
it compiles every pattern of every lane the moment `scan_workflows`
discovers it, independent of match order and independent of any tuple.

### 1c — job/step conditional honesty

Extraction is restricted to `run:`/`cmd:` bodies of jobs and steps that are
NOT conditioned off the merge path: a job or step carrying ANY `if:` key is
excluded WHOLESALE (this gate cannot evaluate arbitrary GitHub Actions
expression syntax, so ANY condition — even one that looks merge-path-safe
on inspection — is treated fail-closed as potentially non-merge-path,
matching this class's own "or refuse credit" discipline), and a job or step
carrying `continue-on-error: true` (or any continue-on-error EXPRESSION
other than this repo's own documented `${{ matrix.continue_on_error ==
'true' }}` per-leg indirection, see below) is excluded too — a failure
there provably does not gate anything. `ci.yml`'s `test-live` job
(`if: github.ref == 'refs/heads/main'` + `continue-on-error: true`,
explicitly excluded from `ci-summary`'s own required set by name) is
exactly this shape: a whole-file text scan would credit its run body even
though nothing there can ever fail a merge.

This repo's own `Guard` job matrix indirection (`cmd: <script>` fields
under `strategy: matrix: include:`, interpolated into a single shared
`run: ${{ matrix.cmd }}` step) is honored structurally, and CONJOINED with
the interpolating step's own blocked-state (the matrix `include:` legs are
credited ONLY if some step in the SAME job both interpolates
`${{ matrix.cmd }}` verbatim AND is itself unblocked; a step-level
`if:`/`continue-on-error: true` on that step, or the total ABSENCE of any
interpolating step, excludes every leg — otherwise a leg could be credited
even when nothing in the job would ever actually execute its `cmd:` text):
a step
whose body IS that literal interpolation expression pulls its candidate
text from each matrix `include:` leg instead, and a leg's own
`continue_on_error: "true"` field excludes THAT leg only (this repo's own
per-leg soft-fail convention, e.g. the "doc numbers have producers" leg) —
a sibling leg without that field still credits normally.

A WORKFLOW-side ` || <fallback>` tail is credited ONLY if it matches a
KNOWN FAIL-LOUD shape — `_FAIL_LOUD_FALLBACK_RE`: a literal nonzero `exit
N`/`return N`, or `exit $?`/`return $?` re-propagating the ALREADY-nonzero
code we are guaranteed to be holding inside the `||` branch. EVERYTHING
ELSE is refused (the polarity is deliberately an ALLOWLIST of known-safe
shapes, never a denylist of known-unsafe ones: a denylist of
`true`/`:`/`exit 0`/`return 0` would silently credit an UNENUMERATED
zero-exit tail — `|| echo "..."`, `|| /bin/true`, `|| test 1 = 1`,
`|| { echo oops; exit 0; }` — even though bash gives every one of those the
SAME zero exit status a bare `|| true` does, semantically equivalent to
`continue-on-error: true`).
`;`-chaining is NOT part of this class on the WORKFLOW side specifically
(the only side this refusal applies to) — `_split_on_semicolons` already
isolates each `;`-separated statement into its own segment before this
check ever runs, and GitHub Actions' own DEFAULT shell for a `run:` step
is `bash --noprofile --norc -eo pipefail {0}` (errexit ON unless a
workflow overrides `shell:`), so a failing LEFT command aborts the whole
step before a later `;`-joined statement's own exit status could matter —
`cargo ...; something_that_succeeds` is not a swallow the way an `||`
tail is, THERE. This does NOT hold for `ci/scripts/**`'s own operator-run
shell scripts, most of which set `-uo pipefail` WITHOUT `-e` (confirmed by
inspection) — but registry-side discovery never applies this refusal at
all, so the distinction is moot for that side. The invocation before a
refused `||` tail is never credited as reachable, even though the SAME
text is still a legitimate REGISTRY tuple when the identical pattern shows
up in one of THIS class's own operator-run scripts (registry discovery
does not refuse it — the subject existing is what matters there, not
whether it swallows its own failure).

A bare `name=$?` capture (this repo's own `runpod_gpu_prove.sh`
convention: capture now, `exit "$rc"` in a LATER statement) is
DELIBERATELY NOT in `_FAIL_LOUD_FALLBACK_RE`: a plain shell assignment ALWAYS
exits 0 regardless of the value it captures, so `cmd || rc=$?` gives the
COMPOUND statement itself a zero exit status — under GitHub Actions' own
default `bash -eo pipefail`, a merge-path step whose entire body is
`run: <tuple> || rc=$?` reads as a SUCCESS even when `<tuple>` genuinely
failed, exactly the swallow class this rule exists to refuse. This gate
sees only the single `||`-tail segment in isolation (no cross-segment
control-flow analysis — a LATER statement checking `$rc` and exiting
nonzero is invisible to a line-shaped check), so crediting `name=$?` can
never be justified from the registry-side convention alone: that
convention is a MATCHED PAIR (capture, then a later `exit "$rc"`), and
nothing here verifies the second half exists at all — precisely the
spelling a maintainer would reach for first to silence an UNREACHABLE
finding, since it is copied verbatim from a script this gate already
cites approvingly.

`_FAIL_LOUD_FALLBACK_RE`'s enumerated set is provably status-propagating —
every member re-exits/re-returns a value that is GUARANTEED nonzero at the
point it runs (a literal `N > 0`, or `$?` read before anything else could
change it) — not merely "probably fine, empirically fail-closed"; that is
the bar `name=$?` fails. A genuinely status-propagating shape this set does
not name is (safely) REFUSED rather than credited, the fail-closed
direction; any addition needs the SAME per-member proof, checked
standalone, never inherited from whatever multi-statement convention it
was copied out of.

Two related conditional-honesty gaps are DISCLOSED, not modeled (both
would need meaningfully more machinery than a hermetic text-shape gate
should carry, and neither is exercised by any real workflow in this
repo): (1) `needs:` SKIP
PROPAGATION — a job B with no `if:`/`continue-on-error:` of its own, whose
`needs: [A]` names a job A that IS `if:`-blocked, is credited by THIS gate
as unblocked, even though GitHub's own default `needs:` semantics would
likely skip B too (a skipped upstream job does not satisfy the implicit
`success()` check a plain `needs:` carries, unless B declares its own
`if: always()`/`if: ${{ !cancelled() }}`-shaped override) — this gate does
not resolve the job-dependency graph at all; (2) a `run: |` block-scalar
step that internally does `set +e` followed by an unconditional `exit 0`
(or similar) makes EVERY cargo invocation inside that SAME step soft-
failing, with no per-line marker (`if:`, `continue-on-error:`, or a
per-invocation `|| true`) this gate's line-shaped checks could ever see —
this gate does not parse a `run: |` block's own internal control flow.
Both are known limits, not silent gaps: either shape appearing in a real
workflow requires widening this section.

### exact-tuple match, never substring

A workflow line reading
`cargo clippy -p jammi-kernels --all-targets --features cuda,flash-attn -- -D warnings`
must NOT satisfy the registered tuple
`cargo clippy -p jammi-kernels --all-targets --features cuda -- -D warnings`
— the two are extracted as two DIFFERENT strings and compared by set
membership, never by one containing the other as a substring.

Comment-only lines never count in either direction (registry OR reachable
corpus): a cuda-gated tuple named inside a `#` comment one line above a
DIFFERENT, non-cuda compile-check step would satisfy a
name-appears-anywhere scan while never actually running the cuda-gated
tuple. `_drop_comment_lines` blanks every
line whose stripped content starts with `#` (never REMOVES it — removing
would shift every subsequent line's number out from under this gate's own
origin bookkeeping) before any tuple is extracted.

### disclosed narrowness, not silently assumed

`workflow_call`-only workflows (reusable workflows with no direct trigger
of their own, e.g. `_gpu-proof-required.yml`, `_pypi-server.yml`) are never
evaluated TRANSITIVELY through a caller's `uses:` — a cargo invocation
living inside a reusable workflow's own body would not be credited even if
its caller is genuinely merge-path, because nothing in this class lives
there (`_pypi-server.yml`'s own `cargo build ... --features
${{ inputs.cargo_features }}` is a parameterized value that could never
character-match a literal registered tuple anyway). A gated tuple inside a
reusable workflow would require real call-graph resolution here. Workflow
discovery globs BOTH `*.yml` AND `*.yaml` (GitHub Actions accepts either
extension).

## Rule 2 — registry completeness

The tuple registry is DERIVED, never hand-maintained: `discover_all_tuples`
walks every TRACKED file (`git ls-files` — a CI checkout can only ever see
what git tracks) under `ci/scripts/` — recursively, so a future sibling
script (e.g. a nested `ci/scripts/pods/pod_seed_target_v2.sh`) cannot
silently join the class unregistered — and extracts
every line-shaped `cargo (build|test|clippy|check|run) ...` invocation,
after (a) blanking full-line comments, (b) joining a physical line ending
in a bare trailing backslash with its continuation (so a `--features`
argument that lands on the SECOND line of a wrapped invocation is still
visible to gating), (c) quote-aware splitting on `;` (so a compound
`echo ...; cargo build ...; echo ...` shell statement is examined
per-statement, not rejected outright because the LINE doesn't start with
`cargo`), and (d) stripping a leading chain of shell env-var assignments
(`FOO=bar `, `FOO="$BAR" `) and this repo's own known wrapper functions
(`run_cmd`) that legitimately precede a real invocation in these scripts.
`_looks_like_real_invocation` then requires the token immediately after the
subcommand to be either absent (a bare `cargo build`) or flag/variable-
shaped (`-`/`$`-prefixed) — the discriminator that keeps a PROSE sentence
merely starting with the words "cargo build" (a docstring explaining WHY a
suite needs a real cargo toolchain, not a script line that runs one) from
being registered as if it were code. This is a heuristic, not a parser: a
hypothetical positional-argument cargo invocation (none exists in this
class) would be misjudged as prose, and a bash/Python STRING-LITERAL
assignment is excluded only because its own syntax (`NAME=value` needs no
space before `=` in bash; `name = value` needs one in Python, and a
quoted-string assignment with nothing trailing it fails the "assignment
must be followed by more content on the same physical line" shape this
gate's env-prefix stripper requires) happens not to overlap with a bare
`cargo ...` line start — not a general string-literal-aware scan.

A tuple is REGISTERED (subject to Rule 1) only if it is GATED: the UNION of
every `--features`/`-F` argument's comma-split token set (a line may carry
more than one `--features` flag) intersects `GATED_FEATURE_TOKENS =
{"cuda", "flash-attn"}`, OR any token is itself a NAMESPACED flash-attn
forward (`<crate>/flash-attn`, e.g. `jammi-encoders/flash-attn` — the same
feature-forwarding shape `ci.yml`'s own flash-attn-closure guard polices,
just spelled with an explicit crate prefix rather than the bare token) —
the two features that pull a real CUDA/CUTLASS toolchain
(`dep:bindgen_cuda` / vendored CUTLASS, `crates/jammi-kernels/build.rs`,
`ci.yml`'s own "CANNOT be covered here" comment on the hermetic runner)
this repo's ordinary hermetic CI runners do not carry. A default-feature
invocation (e.g. `cargo test -p jammi-kernels --no-run`, no `--features` at
all) is not part of THIS class — it needs no special hardware/toolchain and
is already exercised, non-exactly but functionally, by the ordinary
workspace test job; registering it here would make this a different,
broader gate.

`ci/scripts/` only, deliberately: every gated tuple lives there
(`pod_seed_target.sh`, `runpod_gpu_prove.sh`, ...). A gated tuple under
another root requires widening this constant.

Two paths under `ci/scripts/` are excluded from discovery
(`_DISCOVERY_EXCLUDED_RELPATHS`): this gate's OWN source file (its
`--self-test` fixtures are `cargo ...`-shaped string literals, not real
invocations — `_looks_like_real_invocation`'s prose discriminator does NOT
help here, since a copied-verbatim real command line IS syntactically
indistinguishable from a real one) and its own allowlist file (whose rows
are themselves `cargo ...`-prefixed lines). Without this exclusion the gate
would register tuples out of its own fixture/waiver data and immediately
flag them UNREACHABLE against itself — a self-inflicted false positive,
never a real finding about the tree. `ci/scripts/check_execution_surface_reachability.py`'s
own entry is ALSO derived from `Path(__file__)` at runtime (belt-and-braces
— correct even if this file is ever renamed and someone forgets to update
the hand-written string), not solely the hand-written path constant.

Other files under `ci/scripts/` that mention `cargo ...` text for
TESTING/ASSERTING purposes rather than running it (`test_pod_substrate.sh`'s
own T3-lockstep fixture variable `X_T3_EXPECTED='cargo clippy ...'`, or a
Python `old = "cargo clippy ...\n"` string inside an embedded heredoc) are
NOT hand-excluded the way this gate's own two files are — they are excluded
BY CONSTRUCTION of `_looks_like_real_invocation`'s bash/Python assignment-
shape discrimination above, a narrower and more honest mechanism than a
growing hand-list would be, and (per this section's own residual note) not
airtight in general.

## Rule 3 — waiver rot (and its mirror: dead waivers)

`EXECUTION_SURFACE_ALLOWLIST_PATH` carries one `<tuple text>\t<reason>` row
(TAB-separated, never ` | ` — several real tuples in this class pipe their
own output through `tee`, e.g. `... 2>&1 | tee "$L1"`, so a `|`-based
delimiter would collide with the tuple's OWN text; no cargo invocation in
this repo's `ci/scripts/` contains a literal tab) per registered-but-off-
merge-path tuple.

ROT: a row's PREDICATE is mechanical and re-checked every run — the row's
tuple text must still be a member of the CURRENT (full, gated-or-not)
registry. `discover_all_tuples` returns the FULL registry, not just the
gated subset, because rot is about the SUBJECT's continued EXISTENCE
(has the exact command line been renamed, deleted, or edited away?), which
is orthogonal to whether it is currently gated. A row naming a tuple that
no longer exists anywhere in the registry is FAILURE (rot), never a silent
no-op skip. A row missing a non-empty reason is also a failure.

DEAD WAIVER (the mirror direction, checked independently — NOT the same
condition as rot, and NOT derived from `discover_all_tuples`'s scope): a
row whose tuple text is still gated AND has become REACHABLE on the merge
path (someone wired an exact-matching invocation into a qualifying
job/step) is not rot — its subject is very much alive — but the waiver
itself is now unnecessary. Flagged separately so the allowlist cannot
silently accumulate rows nobody needs anymore.

## Honest residual — CUDA tuples force a written choice

Every tuple this class registers needs a REAL CUDA/CUTLASS toolchain to run
meaningfully; the only lane that has one (`gpu-prove.yml`, the single
`runpod_gpu_prove.sh` producer — see the allowlist's own notes) is, by this
repo's own design, never a merge-path trigger for this class's own scripts,
and never in the critical path of an automated workflow at all — see that
workflow's own header for the canonical statement of why; every CUDA
release lane instead consumes its already-recorded verdict via
`_gpu-proof-required.yml`. That leaves exactly two honest choices per
tuple, never a silent third:

  (a) `gpu-prove.yml` is promoted to a REQUIRED merge-path check.
      This is a GitHub branch-protection ruleset setting, not committed
      workflow YAML — nothing in this checkout can mechanically prove or
      disprove it, so this gate can never credit it automatically.
      `GPU_PROVE_PROMOTED_TO_REQUIRED` below is the single named constant a
      human flips (with a comment explaining how the promotion was
      verified) the day that changes; until then it stays `False` and
      every gated tuple falls through to choice (b).
  (b) The tuple owns an explicit, reasoned row in
      `EXECUTION_SURFACE_ALLOWLIST_PATH`, subject to Rule 3's rot (and
      dead-waiver) checks — and that reason must be TRUE at HEAD, verified
      by re-derivable content (grep-confirmable claims), never a bare line
      number, which rots the moment any PR (including THIS gate's own)
      touches an unrelated part of the cited file.

There is no code path that lets a registered-but-unreachable tuple pass
silently: Rule 1 fails it unless (a) or (b) holds, and (b) is itself
re-verified (not merely present) every run.

Run: `python3 ci/scripts/check_execution_surface_reachability.py`
Self-test (RED mutants for every rule above, driven against an ephemeral
`git init`'d fixture repo, never this checkout):
`python3 ci/scripts/check_execution_surface_reachability.py --self-test`
Hermetic: reads the working tree (or a `--self-test` tempdir) and shells out
only to `git ls-files`; no network, no cargo, no GPU.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# YAML loading. PyYAML is a declared prerequisite of this gate (installed by
# `.docker/ci.Dockerfile` and, on any workflow job that runs this gate
# outside that container, by an explicit `pip install` step) -- guarded here
# so a missing install is ONE distinct, named message at every entry point
# below, never a Python traceback and never silently treated as a finding or
# a pass.
# --------------------------------------------------------------------------- #
try:
    import yaml
except ImportError as _yaml_import_exc:  # pragma: no cover -- exercised via simulated import below
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR: Exception | None = _yaml_import_exc
else:
    _YAML_IMPORT_ERROR = None

_MISSING_PYYAML_MESSAGE = (
    "gate prerequisite missing: PyYAML ({}) -- installed by .docker/ci.Dockerfile"
)

# Set to "1" in a spawned self-test subprocess's own environment so that
# subprocess never spawns a further subprocess of its own -- bounds the
# self-test's real-subprocess PyYAML-dispatch-order fixture to exactly one
# level of nesting regardless of what the code under test does.
_SELFTEST_SUBPROCESS_GUARD_ENV = "JAMMI_ESR_SELFTEST_SUBPROCESS_GUARD"


class WorkflowLoadError(Exception):
    """Raised by `load_workflow_text`/`load_workflow_from_path`/
    `job_source_spans` on anything this reader cannot examine: a missing
    PyYAML install, a read/decode error, unparseable YAML, a document whose
    top level is not a mapping, or a document carrying both a boolean-
    resolved `on` key (YAML 1.1 resolves the bare word `on`/`On`/`ON`/
    `true`/`yes`/... to the boolean `True`) and a literal `"on"` string key.
    Every reader in this module -- the `on:` trigger set, `jobs:`, `steps:`
    -- traces back to ONE parse of the document; there is no second,
    independently-drifting reader of the same text."""


if yaml is not None:

    class _NoDuplicateKeysSafeLoader(yaml.SafeLoader):
        """YAML 1.1 `safe_load` semantics, refusing a document that reuses a
        mapping key at the SAME level -- PyYAML's own default constructor
        silently keeps only the LAST value and never says so, which would
        let a re-added top-level `on:` (or `jobs:`) block shadow an earlier
        one with no signal at all."""

    def _no_duplicate_keys_construct_mapping(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False):
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None, f"expected a mapping node, got {node.id}", node.start_mark
            )
        mapping: dict = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key: {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _NoDuplicateKeysSafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys_construct_mapping
    )
else:  # pragma: no cover -- exercised via simulated import below
    _NoDuplicateKeysSafeLoader = None  # type: ignore[assignment]


def _assert_no_github_incompatible_yaml(text: str) -> None:
    """GitHub Actions' own workflow-file parser does not accept YAML
    anchors, aliases (including merge keys, which are always defined via an
    alias), or explicit tags (`!!str`, a custom `!foo`, ...) -- resolving
    one of these here would let this reader parse a document GitHub itself
    refuses to run at all. Walks the raw PARSER EVENT stream (`yaml.parse`,
    never a resolved node or constructed doc) because an event's `.tag` is
    `None` for every implicitly-typed scalar/mapping/sequence -- including a
    QUOTED scalar -- and is only ever set to a real tag string when the
    source text carried an EXPLICIT tag; a composed `Node`'s own `.tag`
    cannot make that distinction; the two paths would resolve `!!str push`
    and a bare `push` to the SAME `tag:yaml.org,2002:str`. Raises
    `WorkflowLoadError`, naming the exact construct, the moment one is
    seen -- before either `load_workflow_text` or `job_source_spans` trusts
    any parse of this text."""
    if yaml is None:
        raise WorkflowLoadError(_MISSING_PYYAML_MESSAGE.format(_YAML_IMPORT_ERROR))
    try:
        events = yaml.parse(text, Loader=yaml.SafeLoader)
        for event in events:
            if isinstance(event, yaml.AliasEvent):
                raise WorkflowLoadError(
                    "YAML anchors/aliases/tags are not accepted by GitHub Actions -- "
                    f"cannot examine: alias *{event.anchor}"
                )
            anchor = getattr(event, "anchor", None)
            if anchor:
                raise WorkflowLoadError(
                    "YAML anchors/aliases/tags are not accepted by GitHub Actions -- "
                    f"cannot examine: anchor &{anchor}"
                )
            tag = getattr(event, "tag", None)
            if tag is not None:
                raise WorkflowLoadError(
                    "YAML anchors/aliases/tags are not accepted by GitHub Actions -- "
                    f"cannot examine: explicit tag {tag}"
                )
    except yaml.YAMLError as exc:
        raise WorkflowLoadError(f"cannot parse YAML: {exc}") from exc


def load_workflow_text(text: str) -> dict:
    """The workflow document, fully parsed (`yaml.load` under
    `_NoDuplicateKeysSafeLoader`) and validated as a top-level mapping with
    at most one spelling of the `on:` trigger key. Raises `WorkflowLoadError`
    -- never returns a partial or empty dict standing in for a refusal --
    on: a missing PyYAML install; any YAML syntax error (tab indentation, an
    inconsistent dedent, an undefined alias, ...); an anchor, alias
    (including a merge key), or explicit tag anywhere in the document
    (GitHub's own parser rejects all three -- see
    `_assert_no_github_incompatible_yaml`); a document whose top level is
    not a mapping; a duplicate mapping key at any level; or a document
    carrying both a boolean-resolved `on` key and a literal `"on"` string
    key (ambiguous -- GitHub's own YAML 1.1 boolean-resolution gotcha, the
    reason some workflow authors quote `on:` in the first place)."""
    if yaml is None:
        raise WorkflowLoadError(_MISSING_PYYAML_MESSAGE.format(_YAML_IMPORT_ERROR))
    _assert_no_github_incompatible_yaml(text)
    try:
        doc = yaml.load(text, Loader=_NoDuplicateKeysSafeLoader)
    except yaml.YAMLError as exc:
        raise WorkflowLoadError(f"cannot parse YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise WorkflowLoadError(
            f"workflow document's top level is not a mapping (got {type(doc).__name__}) -- cannot examine"
        )
    if True in doc and "on" in doc:
        raise WorkflowLoadError(
            "on: block: the document carries both a boolean-resolved `on` key (YAML 1.1 resolves "
            "on/On/ON/true/yes/... to True) and a literal \"on\" string key -- ambiguous, cannot examine"
        )
    return doc


def load_workflow_from_path(path: Path) -> dict:
    """`load_workflow_text`, reading `path` first -- every read error
    (missing, a directory, permission denied, not valid UTF-8) is the SAME
    named `WorkflowLoadError`, for every euid, never a silent `{}`."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise WorkflowLoadError(f"cannot read file {path}: {exc}") from exc
    return load_workflow_text(text)


def _assert_no_duplicate_keys_in_node_tree(node) -> None:
    """The composed-node equivalent of `_no_duplicate_keys_construct_mapping`
    -- needed because `yaml.compose` never invokes a loader's constructors
    (duplicate-key refusal at CONSTRUCTION time, which `load_workflow_text`
    gets for free from `_NoDuplicateKeysSafeLoader`, does not cover a
    caller that only composes, like `job_source_spans`). Walks EVERY
    mapping node in the tree, at every nesting level, so a duplicate job id
    nested under `jobs:` is caught exactly like a duplicate top-level
    `jobs:` key -- one guard, not two independently-drifting ones."""
    if isinstance(node, yaml.MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            key = key_node.value if isinstance(key_node, yaml.ScalarNode) else repr(key_node.value)
            if key in seen:
                raise WorkflowLoadError(f"cannot parse YAML: found duplicate key: {key!r}")
            seen.add(key)
            _assert_no_duplicate_keys_in_node_tree(value_node)
    elif isinstance(node, yaml.SequenceNode):
        for child in node.value:
            _assert_no_duplicate_keys_in_node_tree(child)


def job_source_spans(text: str) -> dict[str, tuple[int, int]]:
    """{job_id: (start_line, end_line_exclusive)} of RAW 0-based line
    indices for each job directly under the workflow's own top-level
    `jobs:` mapping -- derived from the REAL parsed document structure
    (`yaml.compose`'s own node marks), never a second, independently-
    drifting regex header match. A job's body runs from its own header
    line through the line before the next job's header, or EOF. Returns
    `{}` when there is no top-level `jobs:` mapping at all (the caller
    decides whether that is itself a refusal -- see `jobs_or_fail`).
    Raises `WorkflowLoadError`, self-sufficiently -- never relying on a
    caller having gone through `load_workflow_text` first -- on: a missing
    PyYAML install; an anchor/alias/tag anywhere in the text
    (`_assert_no_github_incompatible_yaml`); a YAML syntax error; a
    duplicate mapping key at ANY level, including two top-level `jobs:`
    blocks and two job ids inside the same `jobs:` mapping
    (`_assert_no_duplicate_keys_in_node_tree`, composed-node based, since
    `yaml.compose` never runs a loader's constructors); and a flow-style
    `jobs:` mapping, or any job entry whose own node does not occupy at
    least one full line of its own -- both shapes make "the line span
    between this job's header and the next" meaningless, and a real
    GitHub Actions workflow's `jobs:` block is always block-style in this
    repo (never silently degraded to a zero-length or wrong-job span)."""
    if yaml is None:
        raise WorkflowLoadError(_MISSING_PYYAML_MESSAGE.format(_YAML_IMPORT_ERROR))
    _assert_no_github_incompatible_yaml(text)
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise WorkflowLoadError(f"cannot parse YAML: {exc}") from exc
    if root is None or not isinstance(root, yaml.MappingNode):
        return {}
    _assert_no_duplicate_keys_in_node_tree(root)
    jobs_node = None
    for key_node, value_node in root.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == "jobs":
            jobs_node = value_node
            break
    if jobs_node is None or not isinstance(jobs_node, yaml.MappingNode):
        return {}
    if jobs_node.flow_style:
        raise WorkflowLoadError("jobs: is flow-style -- cannot examine")
    total_lines = len(text.splitlines())
    starts: list[tuple[str, int]] = []
    for key_node, value_node in jobs_node.value:
        job_id = str(key_node.value)
        start = key_node.start_mark.line
        if value_node.end_mark.line <= start:
            # `jobs:` ITSELF is block-style here (the top-level flow-style
            # shape is the SEPARATE refusal above) -- what this line
            # actually found is a single
            # job's own VALUE not occupying a line span of its own: either
            # an inline flow mapping/sequence written entirely on the
            # key's own line (`a: { runs-on: u }`), or a job id with no
            # body at all (`b:` followed by nothing). Both make "the line
            # span between this job's header and the next" meaningless the
            # same way the top-level flow-style shape does, but neither IS
            # a flow-style `jobs:` mapping, so the message names the job's
            # own value, not the `jobs:` construct.
            raise WorkflowLoadError(
                f"job {job_id!r} does not occupy its own line span in a block-style jobs: "
                "mapping -- cannot examine (its value is either empty or written entirely "
                "on the key's own line)"
            )
        starts.append((job_id, start))
    spans: dict[str, tuple[int, int]] = {}
    for idx, (job_id, start) in enumerate(starts):
        end = starts[idx + 1][1] if idx + 1 < len(starts) else total_lines
        spans[job_id] = (start, end)
    return spans


def jobs_or_fail(text: str) -> tuple[dict[str, tuple[int, int]] | None, str | None]:
    """(job_source_spans, error). The SAME fail-loud doctrine
    `read_top_level_on_block` holds `on:` to, now for `jobs:`:
    `job_source_spans` is self-sufficient (it raises `WorkflowLoadError` on
    every unparseable/flow-style/duplicate-key shape itself -- see its own
    docstring); this wrapper adds only the "truly zero job entries" case on
    top, an unparseable document is a named FAIL, never a silent `{}`
    standing in for "this file has zero jobs"; a `jobs:` block recognized
    as truly empty is refused the same way -- a real GitHub Actions
    workflow always has at least one job."""
    try:
        jobs = job_source_spans(text)
    except WorkflowLoadError as exc:
        return None, str(exc)
    if not jobs:
        return None, "no top-level jobs: block found (or it carries zero job entries)"
    return jobs, None


REPO_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS_ROOT = "ci/scripts/"
WORKFLOWS_DIR_REL = ".github/workflows"

EXECUTION_SURFACE_ALLOWLIST_REL = "ci/scripts/execution_surface_reachability_allowlist.txt"
EXECUTION_SURFACE_ALLOWLIST_PATH = REPO_ROOT / EXECUTION_SURFACE_ALLOWLIST_REL

# This gate's OWN two files, excluded from `discover_all_tuples`'s walk of
# `ci/scripts/**` even though both live under it — see the module doc's
# Rule 2 section. `Path(__file__).name` makes the FIRST entry self-deriving
# (correct even under a rename this hand-written string forgets to track);
# the allowlist path is already a single named constant, not a growing
# hand-list.
_DISCOVERY_EXCLUDED_RELPATHS = {
    f"{SCRIPTS_ROOT}{Path(__file__).name}",
    EXECUTION_SURFACE_ALLOWLIST_REL,
}

# See "Honest residual" above — never flipped by this script itself.
GPU_PROVE_PROMOTED_TO_REQUIRED = False

GATED_FEATURE_TOKENS = {"cuda", "flash-attn"}
CARGO_SUBCOMMANDS = ("build", "test", "clippy", "check", "run")

# Boundary is a lookahead (whitespace or end-of-string), NOT `\b`: `\b` only
# checks word-vs-non-word, so `cargo build/run ...` (a real PROSE sentence
# found in this repo, not a `/`-joined invocation) would satisfy `\b`
# between "d" and "/" even though there is no actual whitespace there —
# `(?=\s|$)` requires the subcommand to be a genuine standalone token.
_CARGO_HEAD_RE = re.compile(r"^cargo\s+(?:" + "|".join(CARGO_SUBCOMMANDS) + r")(?=\s|$)")
_YAML_STEP_PREFIX_RE = re.compile(r"^(?:-\s*)?(?:run|cmd):\s*(.*)$")
_FEATURES_RE = re.compile(r"(?:--features|-F)[=\s]+(\S+)")

# A bash env-var assignment (`FOO=bar `, `FOO="$BAR" `, `FOO='' `) that
# legitimately precedes a real invocation in this repo's scripts — matched
# ONLY when followed by more content on the SAME line (an assignment with
# nothing trailing it, e.g. `X_T3_EXPECTED='cargo clippy ...'` sitting
# alone as its own statement, must NOT be treated as an env-prefixed
# invocation — see the module doc's Rule 2 residual note).
_ENV_ASSIGNMENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=(?:"[^"]*"|\'[^\']*\'|\S*)\s+')
# This repo's own known wrapper function that legitimately precedes a real
# cargo invocation (the perf producers' own `run_cmd()`
# provenance-logging wrapper). A hand-list, same class of "documented, not
# silently narrow" scoping Rule 2's `ci/scripts/`-only root already states —
# a new wrapper function requires widening this tuple.
_KNOWN_WRAPPER_PREFIXES = ("run_cmd",)

# An ALLOWLIST of KNOWN fail-loud `|| <fallback>` tail shapes, never a
# denylist of known-swallowing ones — a denylist (`true|:|exit 0|return 0`)
# silently credits
# every UNENUMERATED zero-exit shape (`|| echo "..."`, `|| /bin/true`,
# `|| test 1 = 1`, `|| { echo oops; exit 0; }` all give bash the SAME zero
# exit status `|| true` does). Matches ONLY a literal nonzero `exit N`/
# `return N`, or `exit $?`/`return $?` (re-propagating the exit code we
# are GUARANTEED to be holding nonzero, since we are inside the `||`
# branch precisely because the left-hand command failed) — every member
# here is PROVABLY status-propagating at the point it runs.
#
# A bare `name=$?` capture (this repo's own `runpod_gpu_prove.sh`
# convention — capture now, `exit "$rc"` in a LATER statement) is NOT in
# this set. A plain shell assignment ALWAYS exits 0 regardless of the value
# it captures, so `cmd || rc=$?` gives the COMPOUND statement a zero exit
# status under GitHub Actions' own default `bash -eo pipefail` — a
# merge-path step whose whole body is `run: <tuple> || rc=$?` would credit
# as reachable even though `<tuple>` genuinely failed: the exact swallow
# class this rule exists to refuse. This gate sees only the
# single `||`-tail segment (no cross-segment control-flow analysis — a
# LATER statement checking `$rc` is invisible to a line-shaped check), so
# the registry-side convention (a MATCHED PAIR: capture, then a later
# `exit "$rc"`) can never justify crediting the capture half alone. See
# `extract_tuples_from_line`'s `refuse_swallowing_fallback` parameter and
# the module doc's own honest-residual note on widening this set safely.
_FAIL_LOUD_FALLBACK_RE = re.compile(
    r"^(?:"
    r"exit\s+[1-9][0-9]*"
    r"|exit\s+\$\?"
    r"|return\s+[1-9][0-9]*"
    r"|return\s+\$\?"
    r")\s*(?:#.*)?$"
)


# --------------------------------------------------------------------------- #
# tuple extraction — shared by registry discovery (ci/scripts/**) and
# workflow job/step-body extraction (.github/workflows/*.yml[a]); the SAME
# pipeline so a registered tuple and a workflow's own invocation are
# compared as identically-normalized strings, never two different
# normalizations that could silently agree or disagree by accident.
# --------------------------------------------------------------------------- #
_INVOCATION_FIRST_TOKEN_PREFIXES = ("-", "$", '"$', "'$")


def _looks_like_real_invocation(head: str) -> bool:
    """See the module doc's Rule 2 section: distinguishes a genuine
    `cargo <subcommand> <flags...>` invocation from PROSE that merely
    starts with the same words. Every real invocation in this repo's own
    scripts/workflows has either NOTHING after the subcommand (a bare
    `cargo build`) or a flag/variable-shaped first token: `-`-prefixed (a
    flag), `$`-prefixed (an unquoted shell-variable-held flag, e.g.
    `cargo build $FLASH_BUILD_FLAG -p ...`), or `"$`/`'$`-prefixed (a
    QUOTED variable/array expansion, e.g. `check_client_deps.sh`'s own
    `cargo build "${packages[@]}" ...` — without this form the
    "suspicious unregistered line" report shows it as a genuine invocation
    mis-judged as prose). Prose's first token is an
    ordinary English word. A heuristic, not a parser — disclosed in the
    module doc, not silently assumed airtight.
    """
    m = _CARGO_HEAD_RE.match(head)
    if not m:
        return False
    rest = head[m.end() :].strip()
    if not rest:
        return True
    first_token = rest.split(None, 1)[0]
    return first_token.startswith(_INVOCATION_FIRST_TOKEN_PREFIXES)


def _strip_env_and_wrapper_prefixes(segment: str) -> str:
    changed = True
    guard = 0
    while changed and guard < 8:
        changed = False
        guard += 1
        m = _ENV_ASSIGNMENT_RE.match(segment)
        if m:
            segment = segment[m.end() :]
            changed = True
            continue
        for w in _KNOWN_WRAPPER_PREFIXES:
            if segment == w:
                segment = ""
                changed = True
                break
            if segment.startswith(w + " "):
                segment = segment[len(w) :].lstrip()
                changed = True
                break
    return segment


def _split_on_semicolons(line: str) -> list[str]:
    """Quote-aware `;` split — a compound shell statement
    (`echo "..."; cargo build ...; echo "..."`) is examined per-statement.
    Tracks single/double-quote state so a `;` INSIDE a quoted string is
    never treated as a statement boundary. Does not handle backslash-
    escaped quotes within a string (`\\"`) — no real line in this class
    needs that, and it is a strictly more permissive miss (fewer splits,
    never a false EXTRA one) than the naive `str.split(";")` this replaced.
    """
    segments: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    for c in line:
        if c == "'" and not in_double:
            in_single = not in_single
            current.append(c)
        elif c == '"' and not in_single:
            in_double = not in_double
            current.append(c)
        elif c == ";" and not in_single and not in_double:
            segments.append("".join(current))
            current = []
        else:
            current.append(c)
    segments.append("".join(current))
    return [s.strip() for s in segments]


def _extract_from_segment(segment: str, refuse_swallowing_fallback: bool) -> str | None:
    segment = segment.strip()
    if not segment:
        return None
    segment = _strip_env_and_wrapper_prefixes(segment)
    parts = re.split(r"\s\|\|\s", segment, maxsplit=1)
    head = parts[0].strip()
    if refuse_swallowing_fallback and len(parts) == 2:
        # A WORKFLOW-side `||` tail is credited ONLY if it matches the KNOWN
        # fail-loud allowlist (`_FAIL_LOUD_FALLBACK_RE`) — EVERY other tail
        # is refused, including unenumerated zero-exit shapes a denylist
        # would have missed (`|| echo "..."`, `|| /bin/true`, etc. all give
        # bash the same zero exit status `|| true` does — semantically
        # equivalent to `continue-on-error: true`, which `_step_is_blocked`
        # already refuses). Registry (`ci/scripts/**`) discovery does NOT
        # pass this flag: the SUBJECT tuple existing is what matters there,
        # not whether its own script happens to swallow its own failure.
        if not _FAIL_LOUD_FALLBACK_RE.match(parts[1].strip()):
            return None
    if head.endswith("\\"):
        head = head[:-1].rstrip()
    if not _looks_like_real_invocation(head):
        return None
    return head


def extract_tuples_from_line(raw_line: str, refuse_swallowing_fallback: bool = False) -> list[str]:
    """Every normalized `cargo ...` invocation a (continuation-joined,
    comment-free) logical line contains — zero, one, or more (a `;`-chained
    compound statement can carry more than one). Drops a leading
    `run:`/`cmd:`/`- cmd:` YAML-step prefix first (so a single-line
    workflow step `run: cargo clippy ...` and a bare shell-script line
    `cargo clippy ...` normalize identically), then quote-aware-splits on
    `;`, then per segment: strips a leading env-assignment/wrapper-prefix
    chain, drops a trailing ` || <shell fallback>` (e.g. `|| exit 1` — when
    `refuse_swallowing_fallback` is UNSET, as it is on the registry side,
    ANY ` || <fallback>` tail is dropped unconditionally, since the SUBJECT
    tuple existing is what matters there; when `refuse_swallowing_fallback`
    IS set, as it is on the workflow-corpus side, ONLY a fallback matching
    the KNOWN fail-loud allowlist `_FAIL_LOUD_FALLBACK_RE` (e.g. `|| exit 1`
    — but NOT `|| rc=$?`, a plain assignment that always exits 0 regardless
    of the value it captures, see that constant's own module-doc note) is
    dropped this way; every OTHER fallback tail refuses the whole segment
    instead, see `_extract_from_segment`) and a trailing line-continuation
    backslash (belt-and-braces — `_join_line_continuations` already
    stitches genuine continuations before this function ever sees the
    line; this only fires on a stray unterminated trailing backslash, e.g.
    the last physical line of a file), and finally requires
    `_looks_like_real_invocation`. A `-- -D warnings` or `-- --nocapture`
    `--` marker is legitimate cargo-argument syntax and must survive
    untouched — only ` || `, a bare trailing backslash, and (quote-aware)
    `;` are ever treated as boundaries.

    NORMALIZED-FORM matching, not byte-for-byte: this function's output —
    used identically on both the registry side (`discover_all_tuples`) and
    the workflow-corpus side (`_extract_tuples_from_text`) — already
    discards an env-var-assignment/wrapper-prefix chain on BOTH sides, and
    a `|| ...` tail ASYMMETRICALLY (any tail on the registry side, only a
    known fail-loud one on the workflow side — see above) before
    comparison. A workflow-side invocation wrapped in a DIFFERENT env
    prefix than the registered script (e.g. a hypothetical
    `CI=1 cargo clippy ...` twin of a registered `cargo clippy ...` tuple)
    would therefore still credit — intentional (the underlying invocation
    IS the same), but a real consequence of comparing NORMALIZED strings,
    not literal source bytes; see the module doc's Rule 1 section.
    """
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
        return []
    m_prefix = _YAML_STEP_PREFIX_RE.match(stripped)
    if m_prefix:
        stripped = m_prefix.group(1).strip()
    if not stripped or stripped in ("|", ">", "|-", ">-", "|+", ">+"):
        return []
    found: list[str] = []
    for segment in _split_on_semicolons(stripped):
        t = _extract_from_segment(segment, refuse_swallowing_fallback)
        if t is not None:
            found.append(t)
    return found


def is_gated(tuple_text: str) -> bool:
    tokens: set[str] = set()
    for m in _FEATURES_RE.finditer(tuple_text):
        # `.strip("\"'")` on the WHOLE captured value:
        # `--features="cuda"`/`--features='cuda,flash-attn'`
        # (a quoted `=`-form, `\S+` in `_FEATURES_RE` happily captures the
        # surrounding quote characters too) would otherwise leave a
        # feature TOKEN that never equals the bare `"cuda"`/`GATED_FEATURE_
        # TOKENS` entries — an evasion of the gate.
        value = m.group(1).strip("\"'")
        tokens.update(t.strip() for t in value.split(","))
    if not tokens:
        return False
    if tokens & GATED_FEATURE_TOKENS:
        return True
    # Namespaced flash-attn forward (`<crate>/flash-attn`) — the same
    # feature-forwarding shape `check_flash_attn_closure.py` polices,
    # spelled with an explicit crate prefix instead of the bare token.
    return any(t == "flash-attn" or t.endswith("/flash-attn") for t in tokens)


def _drop_comment_lines(text: str) -> str:
    """Blank (never REMOVE) a full-line comment — removing would shift
    every subsequent physical line number out from under this gate's own
    origin bookkeeping and `_join_line_continuations`'s starting-lineno
    tracking."""
    return "\n".join("" if line.strip().startswith("#") else line for line in text.splitlines())


def _join_line_continuations(text: str) -> list[tuple[int, str]]:
    """Return `(starting_lineno, logical_line)` pairs — a physical line
    ending in a bare trailing backslash is joined with the following
    physical line(s) into ONE logical line (the `cargo build` token on the
    first physical line, its `--features` argument on the second), so a
    `--features` argument
    landing on a continuation line is visible to gating.
    `starting_lineno` is the 1-indexed physical line the logical line
    STARTS on, used for origin reporting.
    """
    physical = text.splitlines()
    out: list[tuple[int, str]] = []
    i = 0
    n = len(physical)
    while i < n:
        start_lineno = i + 1
        parts = [physical[i]]
        while parts[-1].rstrip().endswith("\\") and i + 1 < n:
            stripped_line = parts[-1].rstrip()
            parts[-1] = stripped_line[:-1].rstrip()
            i += 1
            parts.append(physical[i])
        out.append((start_lineno, " ".join(p.strip() for p in parts)))
        i += 1
    return out


def _extract_tuples_from_text(text: str) -> set[str]:
    """The reachable-corpus (WORKFLOW-side) extraction pipeline: blank
    comments, join continuations, extract per logical line —
    `refuse_swallowing_fallback=True`: a workflow
    line whose `||` tail is not a KNOWN fail-loud shape
    (`_FAIL_LOUD_FALLBACK_RE`) must never be credited as if it genuinely
    gates a merge. Origins are not tracked here (only a set of tuple
    texts) — callers that need origins use `discover_all_tuples`'s own
    line-numbered walk instead."""
    found: set[str] = set()
    for _lineno, logical_line in _join_line_continuations(_drop_comment_lines(text)):
        found.update(extract_tuples_from_line(logical_line, refuse_swallowing_fallback=True))
    return found


# --------------------------------------------------------------------------- #
# registry (Rule 2)
# --------------------------------------------------------------------------- #
@dataclass
class TupleRecord:
    text: str
    origins: list[str] = field(default_factory=list)  # "path:lineno"


def _tracked_files(repo_root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.splitlines()


def discover_all_tuples(repo_root: Path) -> dict[str, TupleRecord]:
    """Every `cargo ...` invocation (gated or not) found under `ci/scripts/`
    in a TRACKED file, keyed by its normalized text — the FULL registry,
    not just the gated subset. Rule 3's rot check reads this full registry
    because rot is about the SUBJECT's continued EXISTENCE (was the exact
    command line renamed/deleted/edited away?), which is orthogonal to
    whether it is currently gated; the SEPARATE dead-waiver check (see
    `run_gate`) is the opposite direction — a still-gated, still-existing
    tuple that has become reachable — and is not derived from this
    function's scope at all.
    """
    registry: dict[str, TupleRecord] = {}
    for rel in _tracked_files(repo_root):
        if not rel.startswith(SCRIPTS_ROOT):
            continue
        if rel in _DISCOVERY_EXCLUDED_RELPATHS:
            continue
        path = repo_root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, logical_line in _join_line_continuations(_drop_comment_lines(text)):
            for t in extract_tuples_from_line(logical_line):
                rec = registry.setdefault(t, TupleRecord(text=t))
                rec.origins.append(f"{rel}:{lineno}")
    return registry


def gated_tuples(registry: dict[str, TupleRecord]) -> dict[str, TupleRecord]:
    return {t: rec for t, rec in registry.items() if is_gated(t)}


def _is_near_miss(head: str) -> bool:
    """A line whose subcommand boundary matches `cargo <subcommand>` as a
    standalone token (the same boundary `_looks_like_real_invocation`
    itself checks) but whose first following token is neither absent nor
    flag/variable-shaped — the exact PROSE shape Rule 2's own discovery
    intentionally drops. A bare `cargo build` (nothing following) is NOT a
    near-miss — `_looks_like_real_invocation` accepts that shape outright.
    """
    m = _CARGO_HEAD_RE.match(head)
    if not m:
        return False
    rest = head[m.end() :].strip()
    if not rest:
        return False
    first_token = rest.split(None, 1)[0]
    return not first_token.startswith(_INVOCATION_FIRST_TOKEN_PREFIXES)


def discover_suspicious_lines(repo_root: Path) -> list[str]:
    """Every line under `ci/scripts/**` that
    is cargo-subcommand-shaped but gets DROPPED by
    `_looks_like_real_invocation`'s prose discriminator — reported (never
    a FAILURE; this is intentional-drop pinning, not a defect) so a human
    reviewing this gate's own output can sanity-check that what got
    dropped really is prose (a docstring sentence like "cargo build and a
    real torch install...") and not a real invocation this heuristic
    mis-judged. Walks the SAME tracked-file set `discover_all_tuples` does
    (excluding this gate's own two files, same reason)."""
    found: list[str] = []
    for rel in _tracked_files(repo_root):
        if not rel.startswith(SCRIPTS_ROOT) or rel in _DISCOVERY_EXCLUDED_RELPATHS:
            continue
        path = repo_root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, logical_line in _join_line_continuations(_drop_comment_lines(text)):
            stripped = logical_line.strip()
            if not stripped:
                continue
            for segment in _split_on_semicolons(stripped):
                segment = _strip_env_and_wrapper_prefixes(segment.strip())
                head = re.split(r"\s\|\|\s", segment, maxsplit=1)[0].strip()
                if head.endswith("\\"):
                    head = head[:-1].rstrip()
                if _is_near_miss(head):
                    found.append(f"{rel}:{lineno}: {head!r}")
    return found


# --------------------------------------------------------------------------- #
# THE `on:`-block trigger-key reader. This is the ONE place in the whole
# `ci/scripts` tree that reads an `on:` block down to its top-level trigger
# KEYS (never the fuller `parse_on_block_or_fail` field shape below, which
# stays a separate, Rule-1-specific reader over the SAME parsed document) —
# `check_gpu_prove_once.py`'s P1, P5, P6, P7 and its `--read-on-block` CLI
# (which `test_gpu_gang_lane.sh`'s G7 shells out to) all import this
# function, never a second, independently-drifting copy. Every value this
# module derives from a workflow's `on:` block -- the trigger set here, and
# the per-trigger `branches`/`paths`/... fields `parse_on_block_or_fail`
# reads -- comes from ONE parse of the document (`load_workflow_text`), so
# the two readers can no longer disagree with each other the way two
# independent regexes could.
# --------------------------------------------------------------------------- #
def _trigger_value(doc: dict) -> tuple[object, str | None]:
    """(on_value, error). Locates the workflow's own `on:` value, handling
    YAML 1.1's boolean resolution of a bare `on`/`On`/`ON`/`true`/`yes`/...
    key to `True` -- `load_workflow_text` already refuses a document
    carrying both spellings, so this only needs to pick whichever one is
    present."""
    if True in doc:
        return doc[True], None
    if "on" in doc:
        return doc["on"], None
    return None, "no top-level on: block found"


def _normalize_trigger_set(on_value: object) -> tuple[list[str] | None, str | None]:
    """(trigger_keys, error). A bare/quoted scalar is ONE trigger key; a
    list is used as-is (every element must be a string, and it must be
    non-empty); a mapping's own keys are the trigger set (a non-string key
    -- impossible for a real GitHub event name -- is refused); `null`/any
    other YAML type is a loud refusal, never `[]` standing in for "no
    triggers"."""
    if on_value is None:
        return None, "on: block is empty (a null value) -- cannot examine"
    if isinstance(on_value, str):
        return [on_value], None
    if isinstance(on_value, list):
        if not on_value:
            return None, "on: block is an empty list -- cannot examine"
        if not all(isinstance(x, str) for x in on_value):
            return None, f"on: block is a list containing a non-string element -- cannot examine: {on_value!r}"
        return list(on_value), None
    if isinstance(on_value, dict):
        if not on_value:
            return None, "on: block is an empty mapping -- cannot examine"
        keys: list[str] = []
        for k in on_value:
            if not isinstance(k, str):
                return None, f"on: block has a non-string trigger key -- cannot examine: {k!r}"
            keys.append(k)
        return keys, None
    return None, f"on: block has an unrecognized shape -- cannot examine: {on_value!r}"


def read_top_level_on_block(text: str) -> tuple[list[str] | None, str | None]:
    """(trigger_keys, error). `error` is non-None on every shape this
    reader cannot examine (a missing PyYAML install, unparseable YAML, a
    duplicate key, an ambiguous `on`/`"on"` collision, a `null`/empty/
    wrongly-typed `on:` value, an anchor/alias/tag anywhere in the
    document); `trigger_keys` is a definite, exhaustive list only when
    `error` is None — never a partial list silently missing a key this
    reader could not parse. A folded/literal block scalar, a flow
    mapping/sequence, `on :` (whitespace before the colon), a quoted
    `"on":` key, a BOM, and CRLF line endings are all real, valid YAML and
    are read exactly as GitHub itself would read them. An anchor, alias
    (including a merge key, which is always defined via an alias), or an
    explicit YAML tag is a DIFFERENT case: this is real, valid YAML too,
    but GitHub Actions' own workflow parser does not accept any of the
    three in a workflow file, so this reader refuses them outright
    (naming the exact construct) rather than silently resolve a document
    GitHub itself would refuse to run."""
    try:
        doc = load_workflow_text(text)
    except WorkflowLoadError as exc:
        return None, str(exc)
    on_value, err = _trigger_value(doc)
    if err is not None:
        return None, err
    return _normalize_trigger_set(on_value)


def read_top_level_on_block_from_path(path: Path) -> tuple[list[str] | None, str | None]:
    """(trigger_keys, error). Wraps `read_top_level_on_block` with the
    FILE-level fail-loud rule: every read error on the workflow file --
    missing, a directory, permission denied, not valid UTF-8 -- is a named
    FAIL, never `([], None)` / "no key", for EVERY euid. A missing path or
    a directory path raises `FileNotFoundError`/`IsADirectoryError` (both
    `OSError`) regardless of the caller's privilege, so those two cases
    hold even for a root caller; a permission-denied path is bypassed by
    root's own DAC override and can only be exercised as a caller that is
    genuinely not root -- callers of this function must not assume the
    permission-denied arm ran under every euid, only that it is the same
    named FAIL when it does. A file that is not valid UTF-8 is the SAME
    named FAIL (`UnicodeDecodeError` is not an `OSError` subclass -- caught
    separately so a non-UTF-8 workflow file does not raise past this
    function as an uncaught traceback)."""
    try:
        doc = load_workflow_from_path(path)
    except WorkflowLoadError as exc:
        return None, str(exc)
    on_value, err = _trigger_value(doc)
    if err is not None:
        return None, err
    return _normalize_trigger_set(on_value)


# --------------------------------------------------------------------------- #
# `on:` trigger parsing (Rule 1a/1b) -- the per-trigger field shape
# (`branches`/`branches-ignore`/`types`/`tags`/`tags-ignore`/`paths`/
# `paths-ignore`), read directly off the SAME parsed `on:` value
# `read_top_level_on_block` reads its trigger keys from.
# --------------------------------------------------------------------------- #
_ON_BLOCK_FIELDS = ("branches", "branches-ignore", "types", "tags", "tags-ignore", "paths", "paths-ignore")


def _normalize_str_list_field(value: object, field_name: str) -> tuple[list[str] | None, str | None]:
    """(normalized, error). A missing field is `None` (no filter); a bare
    string is ONE item; a list must carry only strings; anything else is a
    loud refusal, never a silently-empty or partially-read list."""
    if value is None:
        return None, None
    if isinstance(value, str):
        return [value], None
    if isinstance(value, list):
        if not all(isinstance(x, str) for x in value):
            return None, f"{field_name}: list contains a non-string element -- cannot examine: {value!r}"
        return list(value), None
    return None, f"{field_name}: unrecognized shape (expected a string or a list of strings) -- cannot examine: {value!r}"


def _on_block_dict(doc: dict) -> tuple[dict[str, object] | None, str | None]:
    """(per-trigger raw child value, error) -- the `on:` value's own
    mapping form, unresolved into fields yet. A bare/quoted scalar or list-
    shaped `on:` normalizes to `{trigger: None}` per trigger (no fields to
    read, matching GitHub's own "no filter" semantics for the short
    forms)."""
    on_value, err = _trigger_value(doc)
    if err is not None:
        return None, err
    if on_value is None:
        return None, "on: block is empty (a null value) -- cannot examine"
    if isinstance(on_value, str):
        return {on_value: None}, None
    if isinstance(on_value, list):
        if not on_value or not all(isinstance(x, str) for x in on_value):
            return None, f"on: block is a list with a non-string or empty element set -- cannot examine: {on_value!r}"
        return {k: None for k in on_value}, None
    if isinstance(on_value, dict):
        if not on_value:
            return None, "on: block is an empty mapping -- cannot examine"
        out: dict[str, object] = {}
        for k, v in on_value.items():
            if not isinstance(k, str):
                return None, f"on: block has a non-string trigger key -- cannot examine: {k!r}"
            out[k] = v
        return out, None
    return None, f"on: block has an unrecognized shape -- cannot examine: {on_value!r}"


def parse_on_block_or_fail(text: str) -> tuple[dict[str, dict[str, list[str] | None]] | None, str | None]:
    """(on_dict, error). `on_dict[trigger] = {field: [...] | None}` for
    every field in `_ON_BLOCK_FIELDS` -- the Rule-1b shape
    `merge_path_lanes` reads. `error` is set (and `on_dict` is `None`) on
    anything this reader cannot examine -- never a partial or empty dict
    standing in for a refusal."""
    try:
        doc = load_workflow_text(text)
    except WorkflowLoadError as exc:
        return None, str(exc)
    return _parse_on_block_from_doc(doc)


def _parse_on_block_from_doc(doc: dict) -> tuple[dict[str, dict[str, list[str] | None]] | None, str | None]:
    raw, err = _on_block_dict(doc)
    if err is not None:
        return None, err
    out: dict[str, dict[str, list[str] | None]] = {}
    for trigger, child in raw.items():
        # A trigger's own child value is a mapping ONLY for the triggers
        # this reader's fields (branches/paths/...) actually apply to
        # (`push:`/`pull_request:`/...); `schedule:` is a LIST of `{cron:
        # ...}` entries, a bare `workflow_dispatch:`/`workflow_call:` child
        # is often `None`, and other shapes are possible too -- none of
        # them carry a branches/paths filter, so they normalize to "no
        # filter on any field" rather than a refusal (this reader is never
        # asked to validate a trigger's OWN schema, only to read the
        # filter fields Rule 1b needs off whichever triggers have them).
        child_map = child if isinstance(child, dict) else {}
        fields: dict[str, list[str] | None] = {}
        for field in _ON_BLOCK_FIELDS:
            normalized, ferr = _normalize_str_list_field(child_map.get(field), f"on.{trigger}.{field}")
            if ferr is not None:
                return None, ferr
            fields[field] = normalized
        out[trigger] = fields
    return out, None


def parse_on_block(text: str) -> dict[str, dict[str, list[str] | None]]:
    """Backward-compatible view over `parse_on_block_or_fail`, for the one
    external consumer that predates the error-aware form
    (`check_lint_surface_closure.py`, which only ever needs
    `.get("pull_request")` off a workflow it already trusts): returns `{}`
    on anything this reader cannot examine, exactly as a workflow with no
    qualifying trigger would. A caller that needs to DISTINGUISH "no
    qualifying trigger" from "could not be examined" uses
    `parse_on_block_or_fail` directly -- `scan_workflows` does, since
    silently treating a refusal as "not merge-path" is exactly the
    fail-open shape this rewrite closes."""
    on_dict, err = parse_on_block_or_fail(text)
    return on_dict if err is None else {}


def _push_admits_main(push: dict[str, list[str] | None]) -> bool:
    branches = push.get("branches")
    branches_ignore = push.get("branches-ignore")
    tags = push.get("tags")
    tags_ignore = push.get("tags-ignore")
    if branches is not None:
        return "main" in branches
    if branches_ignore is not None:
        return "main" not in branches_ignore
    if tags is not None or tags_ignore is not None:
        # A push trigger scoped ONLY by tags/tags-ignore (no branches key
        # at all) does not fire for ordinary branch pushes at all — GitHub's
        # own documented semantics; `release-binaries.yml`/`crates.yml`/
        # `npm.yml`'s own shape.
        return False
    return True  # no branches/branches-ignore/tags/tags-ignore filter at all


def _pr_admits_main(pr: dict[str, list[str] | None]) -> bool:
    """... `types:` credits a host only when `synchronize`
    is present (or `types:` is ABSENT entirely, which GitHub defaults to
    `[opened, synchronize, reopened]`) -- a `types: [opened]`-only host
    re-runs once, on the PR's initial creation, but NEVER AGAIN on a later
    push to that PR's branch (the `synchronize` event); crediting it as
    reachable promises per-commit coverage this trigger shape cannot
    deliver. Merge-blocking rescues most real cases (a required check
    missing on the NEW head still blocks the merge button), but that is a
    DIFFERENT mechanism than this rule's own promise, and is not assumed
    here. See https://github.com/f-inverse/jammi-ai/issues/533."""
    types = pr.get("types")
    types_ok = types is None or "synchronize" in types
    if not types_ok:
        return False
    branches = pr.get("branches")
    branches_ignore = pr.get("branches-ignore")
    if branches is not None:
        return "main" in branches
    if branches_ignore is not None:
        return "main" not in branches_ignore
    return True


@dataclass
class PathLane:
    paths: list[str] | None
    paths_ignore: list[str] | None


def merge_path_lanes(on_dict: dict[str, dict[str, list[str] | None]]) -> list[PathLane]:
    """Every qualifying (Rule 1a) trigger on this workflow, each carrying
    its OWN `paths:`/`paths-ignore:` filter (Rule 1b reads these
    per-lane — `push:` and `pull_request:` can and do carry DIFFERENT
    `paths:` lists in this repo, e.g. `docs.yml`)."""
    lanes: list[PathLane] = []
    push = on_dict.get("push")
    if push is not None and _push_admits_main(push):
        lanes.append(PathLane(paths=push.get("paths"), paths_ignore=push.get("paths-ignore")))
    pr = on_dict.get("pull_request")
    if pr is not None and _pr_admits_main(pr):
        lanes.append(PathLane(paths=pr.get("paths"), paths_ignore=pr.get("paths-ignore")))
    return lanes


def is_merge_path(on_dict: dict[str, dict[str, list[str] | None]]) -> tuple[bool, str]:
    """Thin convenience wrapper over `merge_path_lanes` — a workflow "is
    merge-path" (Rule 1a only, ignoring Rule 1b path-capability, which is
    evaluated per-tuple-origin, not per-workflow) iff it has at least one
    qualifying lane."""
    lanes = merge_path_lanes(on_dict)
    if lanes:
        return True, "at least one qualifying push-to-main/pull_request-to-main trigger"
    return False, "no push-to-main and no non-label-only pull_request-to-main trigger"


class UnsupportedPathPatternError(Exception):
    """Raised by `_glob_to_regex` on a `paths:`/`paths-ignore:` pattern this
    gate does not evaluate: a leading `!` or a `{a,b}` brace-expansion
    group. Refused rather than silently computed past — guessing wrong
    here could fail OPEN (credit a lane GitHub's own order-evaluated
    negation would actually exclude for a specific origin)."""


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a GitHub Actions path glob (`**`, `**/`, `*`, `?`, literal
    segments) into a regex. Not a general glob engine, and NOT a full
    implementation of GitHub's own path-filter language: GitHub Actions DOES
    support a leading `!` as an ORDER-EVALUATED negation within a
    `paths:`/`paths-ignore:` list (a later `!pattern` excludes paths a
    PRIOR positive pattern matched — GitHub's own filter-pattern cheat
    sheet documents this) and `{a,b}` brace-expansion groups; NEITHER is
    implemented here. Raises `UnsupportedPathPatternError` on either rather
    than silently mis-translating: a `!`-prefixed pattern naively passed
    through this function's literal-character branch would become a
    NEVER-MATCHING regex, discarding the negation entirely — if that
    pattern was meant to NARROW an otherwise-broad `paths:` allowlist, the
    silent result is an OVER-broad admit (this gate crediting a lane that
    GitHub's own real semantics would have excluded for a specific origin)
    — the exact fail-open direction this gate exists to prevent."""
    if pattern.startswith("!") or "{" in pattern:
        raise UnsupportedPathPatternError(
            f"paths:/paths-ignore: pattern {pattern!r} uses syntax this gate does not evaluate "
            "(a leading `!` order-evaluated negation, or a `{...}` brace-expansion group) — refused, "
            "never silently computed past"
        )
    out = ["^"]
    i, n = 0, len(pattern)
    while i < n:
        if pattern[i : i + 3] == "**/":
            out.append("(?:.*/)?")
            i += 3
        elif pattern[i : i + 2] == "**":
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    out.append("$")
    return re.compile("".join(out))


def _lane_admits_path(lane: PathLane, path: str) -> bool:
    if lane.paths is not None:
        if not any(_glob_to_regex(p).match(path) for p in lane.paths):
            return False
    if lane.paths_ignore is not None:
        if any(_glob_to_regex(p).match(path) for p in lane.paths_ignore):
            return False
    return True


def _lane_admits_any_origin(lane: PathLane, origins: list[str]) -> bool:
    for origin in origins:
        path = origin.rsplit(":", 1)[0] if ":" in origin else origin
        if _lane_admits_path(lane, path):
            return True
    return False


# --------------------------------------------------------------------------- #
# `jobs:` / step-body extraction (Rule 1c)
# --------------------------------------------------------------------------- #
# Matches this repo's OWN exact spelling only (`run: ${{ matrix.cmd }}`,
# unquoted, single-line) — a quoted (`run: "${{ matrix.cmd }}"`) or
# block-scalar (`run: |` / `run: >` wrapping the interpolation) spelling
# would fail this match and be treated as an ordinary (non-matrix-
# indirection) step body instead, which fails CLOSED (the matrix `include:`
# legs are then never credited at all, per `has_unblocked_matrix_cmd_step`
# below) rather than crediting anything wrongly — disclosed, not modeled;
# no real workflow in this repo uses either spelling today.
_MATRIX_CMD_INTERP_RE = re.compile(r"^\$\{\{\s*matrix\.cmd\s*\}\}$")
_MATRIX_CONTINUE_ON_ERROR_EXPR = "${{ matrix.continue_on_error == 'true' }}"

# Rule 1c's own YAML-level honesty (`_job_is_blocked`/`_step_is_
# blocked` already exclude a job/step carrying an `if:`) says nothing
# about SHELL-level control flow INSIDE a `run:` body's own text -- a line
# extracted from `if [ "${{ github.event_name }}" = "push" ]; then ... fi`
# is still credited even though the shell `if` may never take that branch
# on a PR run. `&&`-guarded compound heads already fail closed
# (`_looks_like_real_invocation` rejects the compound head outright); this
# closes the SAME class for `if`/`case` block-shaped control flow: a `run:`
# body containing either keyword, as its own token (never a substring
# inside an identifier), is excluded WHOLESALE from tuple extraction --
# fail-closed, the same "exclude rather than guess" doctrine `_job_is_
# blocked` already holds a YAML-level `if:` to.
_SHELL_CONTROL_FLOW_KEYWORD_RE = re.compile(r"(?<![\w.-])(?:if|case)(?![\w.-])")


def _job_is_blocked(job: dict) -> bool:
    """Fail-closed: a job carrying ANY `if:` (this gate cannot evaluate
    arbitrary GH Actions expressions) or a `continue-on-error:` key at all
    is excluded wholesale, regardless of value — `ci.yml`'s own `test-live`
    job (`if: github.ref == 'refs/heads/main'` + `continue-on-error: true`,
    excluded from `ci-summary`'s own required set by name) is exactly this
    shape."""
    return "if" in job or "continue-on-error" in job


def _step_is_blocked(step: dict) -> bool:
    if "if" in step:
        return True
    if "continue-on-error" not in step:
        return False
    coe = step["continue-on-error"]
    if isinstance(coe, str) and coe.strip() == _MATRIX_CONTINUE_ON_ERROR_EXPR:
        return False  # handled per-leg via the matrix `continue_on_error` field
    return True  # a literal `true`/`false`, or any OTHER value/expression — fail-closed


def _job_tuples(job: object) -> set[str]:
    """`job` is this job id's own parsed value under the workflow's
    `jobs:` mapping -- a genuine dict, read straight off the SAME parsed
    document `load_workflow_text` produces (never a second, text-sliced
    re-parse of it)."""
    if not isinstance(job, dict) or _job_is_blocked(job):
        return set()
    found: set[str] = set()
    # The matrix `include:` legs below are only a real
    # execution surface if SOME step in THIS job both interpolates
    # `${{ matrix.cmd }}` AND survives `_step_is_blocked` itself — a step-
    # level `if:`/`continue-on-error: true` on the interpolating step (or
    # its total ABSENCE — no step interpolates the matrix at all) means the
    # legs below never actually run, so they must not be credited. This
    # flag conjoins the step loop with the matrix-leg loop.
    has_unblocked_matrix_cmd_step = False

    steps = job.get("steps")
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict) or _step_is_blocked(step):
            continue
        for key in ("run", "cmd"):
            body_text = step.get(key)
            if not isinstance(body_text, str):
                continue
            if _MATRIX_CMD_INTERP_RE.match(body_text.strip()):
                has_unblocked_matrix_cmd_step = True
                continue  # handled via the matrix include: legs below
            if _SHELL_CONTROL_FLOW_KEYWORD_RE.search(body_text):
                continue  # an if:/case-wrapped run: body may never execute the tuple on this trigger
            found |= _extract_tuples_from_text(body_text)

    if has_unblocked_matrix_cmd_step:
        strategy = job.get("strategy")
        matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
        include = matrix.get("include") if isinstance(matrix, dict) else None
        for leg in include if isinstance(include, list) else []:
            if not isinstance(leg, dict):
                continue
            leg_coe = leg.get("continue_on_error")
            if leg_coe is True or (isinstance(leg_coe, str) and leg_coe.strip("\"'") == "true"):
                continue
            cmd_body = leg.get("cmd")
            if isinstance(cmd_body, str) and not _SHELL_CONTROL_FLOW_KEYWORD_RE.search(cmd_body):
                found |= _extract_tuples_from_text(cmd_body)

    return found


def _workflow_job_tuples(doc: dict) -> set[str]:
    jobs = doc.get("jobs")
    found: set[str] = set()
    for job in jobs.values() if isinstance(jobs, dict) else []:
        found |= _job_tuples(job)
    return found


# --------------------------------------------------------------------------- #
# workflow scan (Rules 1a + 1b + 1c together)
# --------------------------------------------------------------------------- #
@dataclass
class WorkflowScan:
    name: str
    lanes: list[PathLane]
    tuples: set[str]


def _validate_lane_patterns(workflow_name: str, lane: PathLane) -> list[str]:
    """Compile EVERY pattern in `lane.paths` AND `lane.paths_ignore`,
    unconditionally — never short-circuited by an earlier pattern already
    matching (`_lane_admits_path`'s own `any()` stops at the first match,
    so `paths: ["**", "!ci/scripts/**"]` would never even LOOK at the
    second, unsupported pattern under a per-tuple-triggered validation) and
    never gated on whether any tuple happens to route through this workflow
    at all (a merge-path workflow carrying a bad pattern but no gated tuple
    must still be validated). Returns one named finding per invalid
    pattern, empty if all patterns in this lane are supported."""
    findings: list[str] = []
    for patterns in (lane.paths, lane.paths_ignore):
        if patterns is None:
            continue
        for p in patterns:
            try:
                _glob_to_regex(p)
            except UnsupportedPathPatternError as exc:
                findings.append(f"{workflow_name}: {exc}")
    return findings


def scan_workflows(repo_root: Path) -> tuple[list[WorkflowScan], list[str]]:
    """Every workflow with at least one qualifying (Rule 1a) merge-path
    trigger, carrying its own lanes (Rule 1b's per-lane paths filters) and
    its job/step-scoped (Rule 1c) tuple corpus. Globs BOTH `*.yml` and
    `*.yaml` (see the module doc's disclosed narrowness note). Returns
    `(scans, findings)` — `findings` carries BOTH every unsupported
    `paths:`/`paths-ignore:` pattern finding, validated eagerly here for
    every lane of every scanned merge-path workflow,
    independent of match order within a lane and independent of whether
    any gated tuple ever routes a reachability check through this
    workflow at all, AND every workflow file this reader could not
    examine at all (a read error, an unparseable document, an ambiguous
    `on:`/`"on"` collision, ...) — never a silent `continue` standing in
    for "this file has no qualifying trigger": a workflow this reader
    cannot even parse might be the one genuinely-reachable merge-path
    file for some gated tuple, and silently skipping it would read that
    tuple as unreachable for the wrong reason, or (worse) never surface
    that the file itself needs fixing."""
    workflows_dir = repo_root / WORKFLOWS_DIR_REL
    if not workflows_dir.is_dir():
        return [], []
    paths = sorted(set(workflows_dir.glob("*.yml")) | set(workflows_dir.glob("*.yaml")))
    scans: list[WorkflowScan] = []
    findings: list[str] = []
    for path in paths:
        try:
            doc = load_workflow_from_path(path)
        except WorkflowLoadError as exc:
            findings.append(f"{path.name}: {exc}")
            continue
        on_dict, err = _parse_on_block_from_doc(doc)
        if err is not None:
            findings.append(f"{path.name}: {err}")
            continue
        lanes = merge_path_lanes(on_dict)
        if not lanes:
            continue
        for lane in lanes:
            findings.extend(_validate_lane_patterns(path.name, lane))
        scans.append(WorkflowScan(name=path.name, lanes=lanes, tuples=_workflow_job_tuples(doc)))
    return scans, findings


def is_tuple_reachable(tuple_text: str, origins: list[str], scans: list[WorkflowScan]) -> bool:
    """Belt-and-braces: `scan_workflows` already validates every lane's
    patterns EAGERLY before this function is ever
    called, so `UnsupportedPathPatternError` is not expected to fire here
    — but a lane carrying an invalid pattern is still left in place (never
    silently pruned), so a defensive catch treats such a lane as NOT
    admitting (fail-closed) rather than crashing the whole tuple's
    reachability computation, and keeps checking any OTHER lane/scan that
    might still legitimately admit the tuple. Never adds a SECOND,
    differently-formatted finding for the same bad pattern —
    `scan_workflows`'s own eager pass already reported it once."""
    for scan in scans:
        if tuple_text not in scan.tuples:
            continue
        for lane in scan.lanes:
            try:
                if _lane_admits_any_origin(lane, origins):
                    return True
            except UnsupportedPathPatternError:
                continue
    return False


# --------------------------------------------------------------------------- #
# allowlist (Rule 3)
# --------------------------------------------------------------------------- #
@dataclass
class AllowlistRow:
    tuple_text: str
    reason: str
    lineno: int


def parse_allowlist(path: Path) -> tuple[list[AllowlistRow], list[str]]:
    """Returns (rows, parse_failures). A malformed row (no TAB separator, or
    an empty reason) is a parse failure, never silently dropped. TAB, never
    `|` — see the module doc's Rule 3 section for why a `|`-based delimiter
    would collide with a real tuple's own piped shell text
    (`... | tee "$L1"`)."""
    if not path.exists():
        return [], []
    rows: list[AllowlistRow] = []
    failures: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        # `raw` is deliberately NOT `.strip()`-ed before the split below — a
        # bare `str.strip()` treats a tab as whitespace and would eat the
        # very separator this format depends on (e.g. a genuinely EMPTY
        # reason, `"...warnings\t"`, must survive as an empty-reason finding,
        # not silently collapse into a "no separator at all" one).
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t", 1)
        if len(parts) != 2:
            failures.append(
                f"{path.name}:{lineno}: malformed row (expected '<tuple text>\\t<reason>', TAB-separated): {raw!r}"
            )
            continue
        tuple_text, reason = parts[0].strip(), parts[1].strip()
        if not tuple_text or not reason:
            failures.append(
                f"{path.name}:{lineno}: row must carry a non-empty tuple text and a non-empty reason: {raw!r}"
            )
            continue
        rows.append(AllowlistRow(tuple_text=tuple_text, reason=reason, lineno=lineno))
    return rows, failures


# --------------------------------------------------------------------------- #
# gate driver
# --------------------------------------------------------------------------- #
def run_gate(repo_root: Path, allowlist_path: Path) -> tuple[list[str], list[str]]:
    """Returns (failures, info_lines)."""
    failures: list[str] = []
    info: list[str] = []

    registry = discover_all_tuples(repo_root)
    gated = gated_tuples(registry)
    scans, path_pattern_findings = scan_workflows(repo_root)
    allow_rows, allow_parse_failures = parse_allowlist(allowlist_path)
    failures.extend(allow_parse_failures)

    allowlisted_texts = {row.tuple_text for row in allow_rows}

    # An unsupported paths:/paths-ignore: pattern
    # (leading `!`, or `{a,b}`) is validated EAGERLY by `scan_workflows`
    # itself — every pattern of every lane of every merge-path workflow,
    # unconditionally, never short-circuited by an earlier pattern already
    # matching and never gated on whether any gated tuple happens to route
    # through that workflow at all. A tuple that routes through a lane carrying a bad
    # pattern falls through to Rule 1's normal "needs an allowlist row"
    # path (`is_tuple_reachable`'s own defensive catch, fail-closed) AND
    # the pattern itself is named here exactly once.
    failures.extend(sorted(set(path_pattern_findings)))
    reachable_gated = {t for t in gated if is_tuple_reachable(t, gated[t].origins, scans)}

    info.append(
        f"{len(registry)} cargo invocation(s) discovered under {SCRIPTS_ROOT} "
        f"({len(gated)} gated on {sorted(GATED_FEATURE_TOKENS)}); "
        f"{len(scans)} merge-path workflow(s): {', '.join(s.name for s in scans) or '(none)'}"
    )

    # Report (never fail on) every
    # cargo-subcommand-shaped line the prose discriminator dropped, so a
    # human can spot-check the intentional drop rather than trust it blind.
    suspicious = discover_suspicious_lines(repo_root)
    if suspicious:
        info.append(
            f"{len(suspicious)} suspicious unregistered line(s) (cargo-subcommand-shaped but dropped "
            f"as prose — review): " + "; ".join(suspicious)
        )

    # Rule 3 — waiver rot: every allowlist row's tuple must still be a
    # member of the CURRENT (re-discovered this run) full registry.
    for row in allow_rows:
        if row.tuple_text not in registry:
            failures.append(
                f"{allowlist_path.name}:{row.lineno}: ROT — allowlisted tuple no longer found in the "
                f"registry (renamed/deleted script, or the exact command line changed): {row.tuple_text!r}"
            )
            continue
        # Rule 3 mirror — dead waiver: still gated, still exists, but has
        # become reachable. Independent of the rot check above (a row can
        # be rotted OR dead-waived, never both — its subject either exists
        # or it doesn't).
        if row.tuple_text in gated and row.tuple_text in reachable_gated:
            failures.append(
                f"{allowlist_path.name}:{row.lineno}: DEAD WAIVER — this tuple is now reachable on the "
                f"merge path; remove the row (an allowlist that never shrinks stops meaning anything): "
                f"{row.tuple_text!r}"
            )

    # Rule 1 — reachability, subject to the honest residual (no third
    # silent state): every gated tuple must be reachable OR allowlisted OR
    # (never true today, see GPU_PROVE_PROMOTED_TO_REQUIRED) mechanically
    # promoted.
    for text, rec in sorted(gated.items()):
        if text in reachable_gated:
            continue
        if GPU_PROVE_PROMOTED_TO_REQUIRED:
            continue
        if text in allowlisted_texts:
            continue
        failures.append(
            "UNREACHABLE gated tuple, no allowlist row: "
            f"{text!r} (origin(s): {', '.join(rec.origins)}) — this cargo invocation needs a real "
            "CUDA/CUTLASS toolchain and no eligible job/step of a workflow whose `on:` trigger and "
            "path filter fire on the merge path for this origin carries the SAME invocation in its own "
            "normalized form (see extract_tuples_from_line's own docstring for what 'normalized' "
            "discards). Do NOT fix this by hand-copying this exact text into a workflow step — a "
            "second, independently-maintained literal copy is the twin-drift shape this gate "
            "exists to catch. The two honest routes (see this gate's own module doc, 'Honest "
            "residual'): invoke the SAME script this tuple already lives in from an eligible merge-"
            "path job/step, promote the CUDA/CUTLASS toolchain lane that already runs it to a required "
            "merge-path check (GPU_PROVE_PROMOTED_TO_REQUIRED), or add a reasoned row to "
            f"{allowlist_path.name}."
        )

    return failures, info


def require_pyyaml_or_exit(prefix: str, exit_code: int = 2) -> int | None:
    """`None` when PyYAML is importable; otherwise prints ONE distinct,
    named "gate prerequisite missing: PyYAML" line under `prefix` --
    never a finding, never a pass, never a Python traceback -- and returns
    `exit_code`. Every gate built on this module's shared loader
    (`check_lint_surface_closure.py`, `check_gpu_prove_once.py`, and this
    module's own `main`) calls this FIRST, before dispatching `--self-test`
    or ANY other argv-driven behavior: a missing install reads as ONE
    prerequisite failure, never as however many "cannot examine" findings
    a reader produces when the underlying parser is unavailable -- the
    property this predicate exists to hold, not a count of any one
    battery's own case list."""
    if yaml is None:
        print(f"{prefix}: {_MISSING_PYYAML_MESSAGE.format(_YAML_IMPORT_ERROR)}", file=sys.stderr)
        return exit_code
    return None


def _pyyaml_prerequisite_rc() -> int | None:
    """This module's own prerequisite check, prefixed for its own gate
    name. A thin, name-stable wrapper over `require_pyyaml_or_exit` kept so
    the self-test can exercise this exact decision, under a simulated
    missing import, without re-entering `main`'s own `--self-test`
    dispatch (which would recurse into `self_test()` again)."""
    return require_pyyaml_or_exit("execution-surface-reachability")


def main() -> int:
    prereq_rc = _pyyaml_prerequisite_rc()
    if prereq_rc is not None:
        return prereq_rc

    if "--self-test" in sys.argv[1:]:
        return self_test()

    failures, info = run_gate(REPO_ROOT, EXECUTION_SURFACE_ALLOWLIST_PATH)
    for line in info:
        print(f"execution-surface-reachability: {line}")

    if failures:
        print("execution-surface-reachability: FAIL", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        print(f"\nexecution-surface-reachability: {len(failures)} finding(s).", file=sys.stderr)
        return 1

    print(
        "execution-surface-reachability: PASS — every gated execution-surface tuple is reachable on "
        "the merge path or carries a live, non-dead allowlist row."
    )
    return 0


# --------------------------------------------------------------------------- #
# self-test — RED mutants for every rule, ephemeral `git init`'d fixtures,
# never the real checkout.
# --------------------------------------------------------------------------- #
# CI incident (run 33230050451, main, "Guard (arch validation freshness
# self-test)"): `shutil.rmtree` during a `tempfile.TemporaryDirectory`'s
# teardown hit `OSError: [Errno 39] Directory not empty: '.git'` — a race
# between tempdir cleanup and a background `git maintenance`/`gc --auto`
# process a scratch repo's own `git init`/`add` calls can spawn. Same
# exposure here: `_write_repo` below builds one such scratch repo per
# self-test fixture. `-c gc.auto=0 -c gc.autoDetach=false
# -c maintenance.auto=false` kills the background writer AT THE SOURCE.
_GIT_NO_BACKGROUND_MAINTENANCE = ("-c", "gc.auto=0", "-c", "gc.autoDetach=false", "-c", "maintenance.auto=false")


def _scratch_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *_GIT_NO_BACKGROUND_MAINTENANCE, *args], cwd=cwd, check=True)


def _write_repo(tmp: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    _scratch_git(["init", "-q", "-b", "main"], tmp)
    _scratch_git(["config", "user.email", "test@example.com"], tmp)
    _scratch_git(["config", "user.name", "Test"], tmp)
    _scratch_git(["add", "-A"], tmp)


GATED_TUPLE_TEXT = "cargo clippy -p demo --all-targets --features cuda -- -D warnings"

GATED_SCRIPT = """#!/usr/bin/env bash
# Documentation only, NOT a real invocation — must not be registered:
#   cargo clippy -p demo-doc-only --features cuda -- -D warnings
cargo clippy -p demo --all-targets --features cuda -- -D warnings || exit 1
cargo test -p demo --no-run || exit 1
"""

MERGE_PATH_WORKFLOW_REACHABLE = """name: fixture-ci
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

MERGE_PATH_WORKFLOW_UNREACHABLE = """name: fixture-ci
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: echo "no cargo cuda invocation here"
"""

MERGE_PATH_WORKFLOW_COMMENT_ONLY = """name: fixture-ci
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      # cargo clippy -p demo --all-targets --features cuda -- -D warnings
      - run: echo "the line above is a COMMENT, not a step body"
"""

MERGE_PATH_WORKFLOW_SUPERSET_ONLY = """name: fixture-ci
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda,flash-attn -- -D warnings
"""

LABEL_ONLY_WORKFLOW = """name: fixture-gpu-prove
on:
  workflow_dispatch:
  pull_request:
    types: [labeled]
  schedule:
    - cron: "0 0 * * *"
jobs:
  prove:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

PUSH_MAIN_WORKFLOW_REACHABLE = """name: fixture-push
on:
  push:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

TAG_ONLY_WORKFLOW = """name: fixture-release
on:
  push:
    tags: ["v*"]
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

BRANCHES_IGNORE_MAIN_WORKFLOW = """name: fixture-branches-ignore
on:
  push:
    branches-ignore: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

BRANCHES_IGNORE_OTHER_WORKFLOW = """name: fixture-branches-ignore-other
on:
  push:
    branches-ignore: [some-other-branch]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

PATHS_FILTER_MISS_WORKFLOW = """name: fixture-paths-miss
on:
  pull_request:
    branches: [main]
    paths:
      - "docs/guide/**"
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

PATHS_FILTER_ADMIT_WORKFLOW = """name: fixture-paths-admit
on:
  pull_request:
    branches: [main]
    paths:
      - "ci/scripts/**"
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

JOB_IF_WORKFLOW = """name: fixture-job-if
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    continue-on-error: true
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

STEP_IF_WORKFLOW = """name: fixture-step-if
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - if: github.event_name == 'schedule'
        run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

STEP_CONTINUE_ON_ERROR_WORKFLOW = """name: fixture-step-coe
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - continue-on-error: true
        run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

# Fixture: the clippy line lives inside a SHELL `if` block (never a
# YAML-level `if:`, which `_job_is_blocked`/`_step_is_blocked` already
# exclude) -- on a `pull_request` run, `github.event_name` is `pull_
# request`, never `push`, so this shell `if` never takes the clippy
# branch at all; crediting it as reachable is false.
SHELL_IF_WRAPPED_WORKFLOW = """name: fixture-shell-if
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: |
          if [ "${{ github.event_name }}" = "push" ]; then
            cargo clippy -p demo --all-targets --features cuda -- -D warnings
          fi
"""

# Fixture: `types: [opened]` alone re-runs on the PR's initial
# creation but never again on a later push to that PR's branch
# (`synchronize`) -- crediting per-commit coverage from this shape is
# false.
PR_TYPES_OPENED_ONLY_WORKFLOW = """name: fixture-pr-types-opened-only
on:
  pull_request:
    types: [opened]
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

# Positive control: `types: [synchronize]` (explicit, not the implicit
# default) still credits -- the rule requires `synchronize` present, not
# `types:` absent specifically.
PR_TYPES_SYNCHRONIZE_WORKFLOW = """name: fixture-pr-types-synchronize
on:
  pull_request:
    types: [synchronize]
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

# Negative control: a `case` shell keyword must be blocked the same
# way `if` is.
SHELL_CASE_WRAPPED_WORKFLOW = """name: fixture-shell-case
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: |
          case "${{ github.event_name }}" in
            push) cargo clippy -p demo --all-targets --features cuda -- -D warnings ;;
          esac
"""

MATRIX_CMD_WORKFLOW = """name: fixture-matrix-cmd
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        include:
          - name: harmless
            cmd: echo hi
          - name: the-real-one
            cmd: cargo clippy -p demo --all-targets --features cuda -- -D warnings
    steps:
      - run: ${{ matrix.cmd }}
"""

MATRIX_CMD_SOFT_FAIL_WORKFLOW = """name: fixture-matrix-cmd-soft
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        include:
          - name: soft-leg
            cmd: cargo clippy -p demo --all-targets --features cuda -- -D warnings
            continue_on_error: "true"
    steps:
      - continue-on-error: ${{ matrix.continue_on_error == 'true' }}
        run: ${{ matrix.cmd }}
"""

ENV_PREFIXED_SCRIPT = """#!/usr/bin/env bash
CARGO_TARGET_DIR=$SOME_DIR cargo build $SOME_FLAG -p demo --features cuda,jammi-encoders/flash-attn 2>&1 | tail -n 3
"""

WRAPPER_PREFIXED_SCRIPT = """#!/usr/bin/env bash
run_cmd() { "$@"; }
run_cmd cargo build --release -p demo --features cuda --manifest-path "$X/Cargo.toml" \\
  || { echo "::error::failed"; exit 1; }
"""

SEMICOLON_CHAIN_SCRIPT = """#!/usr/bin/env bash
echo "=== build ==="; cargo build --release -p demo --features cuda,jammi-encoders/flash-attn 2>&1 | tail -n 1; echo "done"
"""

CONTINUATION_FEATURES_SCRIPT = """#!/usr/bin/env bash
CARGO_TARGET_DIR="$X" run_cmd cargo build --release -p demo \\
  --features cuda,jammi-encoders/flash-attn --manifest-path "$X/Cargo.toml" || {
  echo "::error::failed"
  exit 1
}
"""

PROSE_SCRIPT = """#!/usr/bin/env bash
# a bash comment naming an unrelated thing
X_EXPECTED='cargo clippy -p demo --all-targets --features cuda -- -D warnings'
echo "you need a real cargo build and a real torch install to run this"
cargo build and a real torch install are both required to run this suite
"""

# --- matrix cmd: legs must be gated by the SAME
# blocked-state as the step that actually interpolates ${{ matrix.cmd }} —
# never independently of it. -------------------------------------------
MATRIX_CMD_STEP_IF_WORKFLOW = """name: fixture-matrix-cmd-step-if
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        include:
          - name: the-real-one
            cmd: cargo clippy -p demo --all-targets --features cuda -- -D warnings
    steps:
      - if: github.event_name == 'schedule'
        run: ${{ matrix.cmd }}
"""

MATRIX_CMD_NO_INTERPOLATING_STEP_WORKFLOW = """name: fixture-matrix-cmd-no-step
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        include:
          - name: the-real-one
            cmd: cargo clippy -p demo --all-targets --features cuda -- -D warnings
    steps:
      - run: echo "no step here ever runs matrix.cmd"
"""

MATRIX_CMD_STEP_CONTINUE_ON_ERROR_WORKFLOW = """name: fixture-matrix-cmd-step-coe
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        include:
          - name: the-real-one
            cmd: cargo clippy -p demo --all-targets --features cuda -- -D warnings
    steps:
      - continue-on-error: true
        run: ${{ matrix.cmd }}
"""

# --- a WORKFLOW-side `|| true`-style tail swallows the real exit
# status and must not credit -- the registry side (a script carrying the
# identical `|| true`) is unaffected. --------------------------------
SWALLOWING_FALLBACK_WORKFLOW = """name: fixture-swallow
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings || true
"""

SWALLOWING_FALLBACK_SCRIPT = """#!/usr/bin/env bash
cargo clippy -p demo --all-targets --features cuda -- -D warnings || true
"""

# --- every UNENUMERATED zero-exit tail below gives bash the SAME zero exit
# status `|| true` does and must be refused too (a denylist of
# true/:/exit 0/return 0 would miss them; the polarity is an ALLOWLIST of
# known-safe shapes).
SWALLOW_ECHO_WORKFLOW = """name: fixture-swallow-echo
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings || echo "oops"
"""

SWALLOW_BIN_TRUE_WORKFLOW = """name: fixture-swallow-bin-true
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings || /bin/true
"""

SWALLOW_TEST_WORKFLOW = """name: fixture-swallow-test
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings || test 1 = 1
"""

SWALLOW_BRACE_EXIT0_WORKFLOW = """name: fixture-swallow-brace
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings || { echo oops; exit 0; }
"""

# --- positive control: a genuinely fail-loud tail must still credit.
FAIL_LOUD_EXIT_N_WORKFLOW = """name: fixture-fail-loud-exit-n
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings || exit 2
"""

# --- RED mutant: a bare `|| rc=$?` capture, with NO later statement in
# sight, must be REFUSED -- a plain assignment always exits 0, so this
# compound statement's own exit status is 0 under GitHub Actions' default
# `bash -eo pipefail` regardless of whether `<tuple>` failed. It is the
# exact spelling of runpod_gpu_prove.sh's own capture-then-exit convention
# -- which only works as a MATCHED PAIR this gate cannot see the second
# half of.
RC_CAPTURE_ALONE_SWALLOWS_WORKFLOW = """name: fixture-rc-capture-alone-swallows
on:
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings || rc=$?
"""

# --- an unsupported paths:/paths-ignore: pattern (leading `!`, or
# `{a,b}`) must be REFUSED, never silently computed past. ----------------
UNSUPPORTED_NEGATION_PATH_WORKFLOW = """name: fixture-bang-path
on:
  pull_request:
    branches: [main]
    paths:
      - "!ci/scripts/**"
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

UNSUPPORTED_BRACE_PATH_WORKFLOW = """name: fixture-brace-path
on:
  pull_request:
    branches: [main]
    paths:
      - "ci/scripts/{a,b}/**"
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

# --- a `!`-pattern sitting AFTER an earlier pattern
# that already matches must STILL be validated -- `any()`'s own short-
# circuit must never hide it. --------------------------------------------
BANG_AFTER_MATCH_PATH_WORKFLOW = """name: fixture-bang-after-match
on:
  pull_request:
    branches: [main]
    paths:
      - "**"
      - "!ci/scripts/**"
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

# --- a `!`-pattern in a merge-path workflow that carries NO gated
# tuple at all must STILL be validated -- tuple-routing must never gate
# whether a bad pattern gets reported. -----------------------------------
BANG_PATH_NO_GATED_TUPLE_WORKFLOW = """name: fixture-bang-no-tuple
on:
  pull_request:
    branches: [main]
    paths:
      - "!ci/scripts/**"
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: echo "nothing cargo-shaped lives in this workflow at all"
"""

# --- a quoted `"on":` key must parse identically to a bare `on:`. ---
QUOTED_ON_KEY_WORKFLOW = """name: fixture-quoted-on
"on":
  pull_request:
    branches: [main]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

# --- paths-ignore/tags-ignore RED mutants (both arms of each) -----------
PATHS_IGNORE_EXCLUDES_WORKFLOW = """name: fixture-paths-ignore-excludes
on:
  pull_request:
    branches: [main]
    paths-ignore:
      - "ci/scripts/**"
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

PATHS_IGNORE_ADMITS_WORKFLOW = """name: fixture-paths-ignore-admits
on:
  pull_request:
    branches: [main]
    paths-ignore:
      - "docs/**"
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""

TAGS_IGNORE_ALONE_WORKFLOW = """name: fixture-tags-ignore-alone
on:
  push:
    tags-ignore: ["v*"]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -p demo --all-targets --features cuda -- -D warnings
"""


def _run_gate_in(tmp: Path, allowlist_text: str | None = None) -> tuple[list[str], list[str]]:
    allow_path = tmp / "ci" / "scripts" / "execution_surface_reachability_allowlist.txt"
    if allowlist_text is not None:
        allow_path.parent.mkdir(parents=True, exist_ok=True)
        allow_path.write_text(allowlist_text, encoding="utf-8")
    return run_gate(tmp, allow_path)


def self_test() -> int:  # noqa: C901 - a flat sequence of independent RED-mutant legs
    failures: list[str] = []

    def check(label: str, tmp_files: dict[str, str], allowlist_text: str | None, expect_fail_substr: str | None) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            tmp = Path(td)
            _write_repo(tmp, tmp_files)
            got, _info = _run_gate_in(tmp, allowlist_text=allowlist_text)
            if expect_fail_substr is None:
                if got:
                    failures.append(f"self-test FAILED ({label}): expected PASS, got findings: {got}")
            else:
                if not any(expect_fail_substr in g for g in got):
                    failures.append(f"self-test FAILED ({label}): expected a finding containing {expect_fail_substr!r}, got: {got}")

    # --- Rule 2: registry completeness — nested-dir recursion, doc-comment
    # exclusion, non-gated exclusion. -----------------------------------
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        _write_repo(
            tmp,
            {
                "ci/scripts/pods/pod_seed_target_v2.sh": GATED_SCRIPT,
                ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_UNREACHABLE,
            },
        )
        registry = discover_all_tuples(tmp)
        gated = gated_tuples(registry)
        if GATED_TUPLE_TEXT not in gated:
            failures.append(f"self-test FAILED: a gated tuple inside a NESTED ci/scripts/ subdirectory was not discovered: {sorted(gated)}")
        if "cargo clippy -p demo-doc-only --features cuda -- -D warnings" in registry:
            failures.append("self-test FAILED: a `cargo ...` line living inside a `#` comment was registered as a real tuple")
        if "cargo test -p demo --no-run" in gated:
            failures.append("self-test FAILED: a non-gated invocation was classified as gated")

    # --- Rule 1a: reachable via a genuine pull_request-to-main / push-to-main -> PASS
    check("pull_request-to-main reachable", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_REACHABLE}, None, None)
    check("push-to-main reachable", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-push.yml": PUSH_MAIN_WORKFLOW_REACHABLE}, None, None)

    # --- Rule 1: unreachable, no allowlist row -> FAIL ----------------------
    check("unreachable unallowlisted", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_UNREACHABLE}, None, "UNREACHABLE gated tuple")

    # --- Rule 1a: label-only workflow -> FAIL -------------------------------
    check("label-only workflow", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-gpu-prove.yml": LABEL_ONLY_WORKFLOW}, None, "UNREACHABLE gated tuple")

    # --- Rule 1a: tag-only push -> FAIL -------------------------------------
    check("tag-only push", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-release.yml": TAG_ONLY_WORKFLOW}, None, "UNREACHABLE gated tuple")

    # --- Rule 1a: branches-ignore: [main] -> FAIL (does NOT admit main) ----
    check("branches-ignore excludes main", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-bi.yml": BRANCHES_IGNORE_MAIN_WORKFLOW}, None, "UNREACHABLE gated tuple")

    # --- Rule 1a: branches-ignore: [other] -> PASS (main not excluded) -----
    check("branches-ignore admits main via omission", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-bio.yml": BRANCHES_IGNORE_OTHER_WORKFLOW}, None, None)

    # --- Rule 1b: paths filter cannot match ci/scripts/** -> FAIL ----------
    check("paths filter misses ci/scripts", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-pm.yml": PATHS_FILTER_MISS_WORKFLOW}, None, "UNREACHABLE gated tuple")

    # --- Rule 1b: paths filter DOES admit ci/scripts/** -> PASS ------------
    check("paths filter admits ci/scripts", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-pa.yml": PATHS_FILTER_ADMIT_WORKFLOW}, None, None)

    # --- Rule 1c: job-level if:/continue-on-error: -> FAIL -----------------
    check("job-level if + continue-on-error excludes", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-ji.yml": JOB_IF_WORKFLOW}, None, "UNREACHABLE gated tuple")

    # --- Rule 1c: step-level if: -> FAIL ------------------------------------
    check("step-level if excludes", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-si.yml": STEP_IF_WORKFLOW}, None, "UNREACHABLE gated tuple")

    # --- Rule 1c: step-level continue-on-error: true -> FAIL ---------------
    check("step-level continue-on-error excludes", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-sc.yml": STEP_CONTINUE_ON_ERROR_WORKFLOW}, None, "UNREACHABLE gated tuple")

    # --- a shell `if`/`case` block wrapping the clippy line is
    # excluded WHOLESALE from tuple extraction, never credited on mere
    # YAML-level presence. ---------------------------------------------------
    check("shell if-wrapped run: body excludes", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-shell-if.yml": SHELL_IF_WRAPPED_WORKFLOW}, None, "UNREACHABLE gated tuple")
    check("shell case-wrapped run: body excludes", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-shell-case.yml": SHELL_CASE_WRAPPED_WORKFLOW}, None, "UNREACHABLE gated tuple")

    # --- `pull_request: types: [opened]` alone never re-runs on
    # a later push to the PR (synchronize) -- must not be credited; an
    # EXPLICIT `types: [synchronize]` still credits (positive control). ----
    check("pull_request types: [opened] alone excludes", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-pr-types-opened.yml": PR_TYPES_OPENED_ONLY_WORKFLOW}, None, "UNREACHABLE gated tuple")
    check("pull_request types: [synchronize] explicit still credits", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-pr-types-sync.yml": PR_TYPES_SYNCHRONIZE_WORKFLOW}, None, None)

    # --- Rule 1c: matrix cmd: indirection is honored (positive control) ----
    check("matrix cmd indirection credits a real leg", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-mc.yml": MATRIX_CMD_WORKFLOW}, None, None)

    # --- Rule 1c: a matrix leg's own continue_on_error: "true" excludes
    # JUST that leg (never credited even though the matrix job itself runs) -
    check("matrix leg continue_on_error excludes that leg", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-mcs.yml": MATRIX_CMD_SOFT_FAIL_WORKFLOW}, None, "UNREACHABLE gated tuple")

    # --- matrix legs must be gated by
    # the SAME blocked-state as the interpolating step, never independently
    # of it -- three RED mutants proving the conjoined check. ---------------
    check("matrix legs excluded when the interpolating step has if:", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-mcsi.yml": MATRIX_CMD_STEP_IF_WORKFLOW}, None, "UNREACHABLE gated tuple")
    check("matrix legs excluded when NO step interpolates matrix.cmd at all", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-mcns.yml": MATRIX_CMD_NO_INTERPOLATING_STEP_WORKFLOW}, None, "UNREACHABLE gated tuple")
    check("matrix legs excluded when the interpolating step has continue-on-error: true", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-mcsc.yml": MATRIX_CMD_STEP_CONTINUE_ON_ERROR_WORKFLOW}, None, "UNREACHABLE gated tuple")

    # --- a workflow-side `|| true`-style tail swallows the real exit
    # status and must not credit; the registry side is unaffected. ---------
    check("workflow-side `|| true` swallow does not credit", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-swallow.yml": SWALLOWING_FALLBACK_WORKFLOW}, None, "UNREACHABLE gated tuple")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        _write_repo(tmp, {"ci/scripts/swallow.sh": SWALLOWING_FALLBACK_SCRIPT})
        gated = gated_tuples(discover_all_tuples(tmp))
        if GATED_TUPLE_TEXT not in gated:
            failures.append(f"self-test FAILED: a `|| true`-tailed REGISTRY-side tuple was not discovered/gated (the swallow refusal must be workflow-side only): {sorted(gated)}")

    # --- the ALLOWLIST polarity refuses every
    # UNENUMERATED zero-exit tail a denylist would have missed. -------------
    check("`|| echo \"...\"` swallow does not credit", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-swallow-echo.yml": SWALLOW_ECHO_WORKFLOW}, None, "UNREACHABLE gated tuple")
    check("`|| /bin/true` swallow does not credit", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-swallow-bin-true.yml": SWALLOW_BIN_TRUE_WORKFLOW}, None, "UNREACHABLE gated tuple")
    check("`|| test 1 = 1` swallow does not credit", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-swallow-test.yml": SWALLOW_TEST_WORKFLOW}, None, "UNREACHABLE gated tuple")
    check("`|| { echo oops; exit 0; }` swallow does not credit", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-swallow-brace.yml": SWALLOW_BRACE_EXIT0_WORKFLOW}, None, "UNREACHABLE gated tuple")
    check("`|| exit 2` (fail-loud, nonzero) still credits", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-fail-loud-exit-n.yml": FAIL_LOUD_EXIT_N_WORKFLOW}, None, None)
    check(
        "`|| rc=$?` ALONE (no later `exit \"$rc\"` in sight) swallows and must refuse",
        {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-rc-capture-alone.yml": RC_CAPTURE_ALONE_SWALLOWS_WORKFLOW},
        None,
        "UNREACHABLE gated tuple",
    )

    # --- an unsupported paths:/paths-ignore: pattern (leading `!`, or
    # `{a,b}`) is REFUSED with a named finding, never silently computed
    # past -- the affected tuple stays UNREACHABLE either way. --------------
    check("`!`-negation path pattern is refused, not silently computed", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-bang.yml": UNSUPPORTED_NEGATION_PATH_WORKFLOW}, None, "uses syntax this gate does not evaluate")
    check("`{a,b}`-brace path pattern is refused, not silently computed", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-brace.yml": UNSUPPORTED_BRACE_PATH_WORKFLOW}, None, "uses syntax this gate does not evaluate")

    # --- eager, order-independent, tuple-independent
    # pattern validation in scan_workflows itself. ---------------------------
    check(
        "a `!`-pattern AFTER an earlier matching pattern is still validated (any() must not short-circuit past it)",
        {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-bang-after-match.yml": BANG_AFTER_MATCH_PATH_WORKFLOW},
        None,
        "uses syntax this gate does not evaluate",
    )
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        _write_repo(
            tmp,
            {
                "ci/scripts/pod_seed_target.sh": GATED_SCRIPT,
                ".github/workflows/fixture-bang-no-tuple.yml": BANG_PATH_NO_GATED_TUPLE_WORKFLOW,
            },
        )
        got, _info = _run_gate_in(tmp)
        if not any("uses syntax this gate does not evaluate" in g for g in got):
            failures.append(
                "self-test FAILED (a bad paths: pattern in a merge-path workflow with NO gated tuple "
                f"at all must still be validated eagerly): {got}"
            )

    # --- a quoted `"on":` key parses identically to a bare `on:`. ----------
    check("quoted \"on\": key parses like bare on:", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-quoted-on.yml": QUOTED_ON_KEY_WORKFLOW}, None, None)

    # --- paths-ignore/tags-ignore RED mutants (both arms of each). ----------
    check("paths-ignore excluding ci/scripts refuses credit", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-pie.yml": PATHS_IGNORE_EXCLUDES_WORKFLOW}, None, "UNREACHABLE gated tuple")
    check("paths-ignore NOT excluding ci/scripts admits it", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-pia.yml": PATHS_IGNORE_ADMITS_WORKFLOW}, None, None)
    check("tags-ignore alone (no branches key) is not merge-path", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-tia.yml": TAGS_IGNORE_ALONE_WORKFLOW}, None, "UNREACHABLE gated tuple")

    # --- Rule 1: comment-only mention does not count ------------------------
    check("comment-only mention", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-co.yml": MERGE_PATH_WORKFLOW_COMMENT_ONLY}, None, "UNREACHABLE gated tuple")

    # --- Rule 1: exact-tuple match, never substring -------------------------
    check("exact match never substring", {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-ss.yml": MERGE_PATH_WORKFLOW_SUPERSET_ONLY}, None, "UNREACHABLE gated tuple")

    # --- Rule 1 + residual: unreachable but allowlisted -> PASS ------------
    check(
        "allowlisted unreachable tuple passes",
        {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_UNREACHABLE},
        f"{GATED_TUPLE_TEXT}\tfixture: no merge-path CUDA toolchain lane exists\n",
        None,
    )

    # --- Rule 3: waiver rot -------------------------------------------------
    check(
        "rotted allowlist row",
        {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_UNREACHABLE},
        "cargo clippy -p a-tuple-that-was-renamed --features cuda -- -D warnings\tstale reason\n",
        "ROT",
    )

    # --- Rule 3 mirror: dead waiver — allowlisted tuple has become
    # reachable -> FAIL, naming DEAD WAIVER, never a silent pass ------------
    check(
        "dead waiver flagged",
        {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_REACHABLE},
        f"{GATED_TUPLE_TEXT}\tstale: this used to be unreachable but a fix landed\n",
        "DEAD WAIVER",
    )

    # --- Rule 3: an allowlist row missing a reason is a parse failure ------
    check(
        "empty reason row",
        {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_UNREACHABLE},
        f"{GATED_TUPLE_TEXT}\t\n",
        "non-empty tuple text and a non-empty reason",
    )

    # --- Rule 3: a malformed row (no TAB separator) -------------------------
    check(
        "malformed row",
        {"ci/scripts/pod_seed_target.sh": GATED_SCRIPT, ".github/workflows/fixture-ci.yml": MERGE_PATH_WORKFLOW_UNREACHABLE},
        "cargo clippy -p demo --features cuda -- -D warnings NO SEPARATOR HERE\n",
        "malformed row",
    )

    # --- Rule 2: env-var-prefixed invocation is discovered + gated ---------
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        _write_repo(tmp, {"ci/scripts/env_prefixed.sh": ENV_PREFIXED_SCRIPT})
        gated = gated_tuples(discover_all_tuples(tmp))
        expect = "cargo build $SOME_FLAG -p demo --features cuda,jammi-encoders/flash-attn 2>&1 | tail -n 3"
        if expect not in gated:
            failures.append(f"self-test FAILED: env-var-prefixed invocation not discovered/gated: {sorted(gated)}")

    # --- Rule 2: wrapper-prefixed (run_cmd) + backslash continuation -------
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        _write_repo(tmp, {"ci/scripts/wrapper_prefixed.sh": WRAPPER_PREFIXED_SCRIPT})
        gated = gated_tuples(discover_all_tuples(tmp))
        expect = 'cargo build --release -p demo --features cuda --manifest-path "$X/Cargo.toml"'
        if expect not in gated:
            failures.append(f"self-test FAILED: run_cmd-wrapped invocation not discovered/gated: {sorted(gated)}")

    # --- Rule 2: `;`-chained invocation is discovered ----------------------
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        _write_repo(tmp, {"ci/scripts/semicolon_chain.sh": SEMICOLON_CHAIN_SCRIPT})
        gated = gated_tuples(discover_all_tuples(tmp))
        expect = "cargo build --release -p demo --features cuda,jammi-encoders/flash-attn 2>&1 | tail -n 1"
        if expect not in gated:
            failures.append(f"self-test FAILED: semicolon-chained invocation not discovered/gated: {sorted(gated)}")

    # --- Rule 2: --features landing on a CONTINUATION line is still
    # visible to gating (env-prefixed + wrapper-prefixed + continuation) ----
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        _write_repo(tmp, {"ci/scripts/continuation_features.sh": CONTINUATION_FEATURES_SCRIPT})
        registry = discover_all_tuples(tmp)
        gated = gated_tuples(registry)
        expect = 'cargo build --release -p demo --features cuda,jammi-encoders/flash-attn --manifest-path "$X/Cargo.toml"'
        if expect not in gated:
            failures.append(f"self-test FAILED: continuation-line --features not visible to gating: {sorted(gated)}")
        else:
            origin = registry[expect].origins[0]
            # Line 1 of CONTINUATION_FEATURES_SCRIPT is the shebang; the
            # continuation-joined invocation itself STARTS on line 2 — the
            # property under test is that the origin is the invocation's
            # OWN first physical line, not line 3 (where the joined
            # `--features` argument physically sits).
            if not origin.endswith(":2"):
                failures.append(f"self-test FAILED: continuation-joined tuple's origin lineno should be its own FIRST physical line (2), got {origin!r}")

    # --- Rule 2: is_gated unions multiple --features/-F occurrences,
    # and recognizes a bare namespaced <crate>/flash-attn with NO "cuda"
    # token anywhere -----------------------------------------------------
    if not is_gated("cargo build -p x --features jammi-encoders/flash-attn"):
        failures.append("self-test FAILED: a bare namespaced <crate>/flash-attn with no 'cuda' token was not classified as gated")
    if not is_gated("cargo build -p x -F cuda"):
        failures.append("self-test FAILED: the -F short-flag form was not recognized as --features")
    if not is_gated('cargo build -p x --features live-gpu-tests --features cuda'):
        failures.append("self-test FAILED: a SECOND --features occurrence was not unioned into gating")
    if is_gated("cargo build -p x --features live-gpu-tests,not-flash-attn-at-all"):
        failures.append("self-test FAILED: a non-gated feature set was misclassified as gated")
    # A quoted `--features="cuda"` spelling (the
    # `--features=` equals-form with a quoted value) must not evade gating.
    if not is_gated('cargo build -p x --features="cuda"'):
        failures.append("self-test FAILED: a quoted --features=\"cuda\" spelling evaded gating")
    if not is_gated("cargo build -p x --features='cuda,flash-attn'"):
        failures.append("self-test FAILED: a single-quoted --features='cuda,flash-attn' spelling evaded gating")

    # --- Rule 2: prose that merely STARTS with "cargo build" is never
    # registered, even without a comment/assignment context -----------------
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        _write_repo(tmp, {"ci/scripts/prose.sh": PROSE_SCRIPT})
        registry = discover_all_tuples(tmp)
        if any("cargo clippy" in t for t in registry):
            failures.append(f"self-test FAILED: a quoted bash-variable assignment ('cargo clippy ...') was registered as a real tuple: {sorted(registry)}")
        if any(t.startswith("cargo build and") or t.startswith("cargo build a real") for t in registry):
            failures.append(f"self-test FAILED: a prose sentence starting with 'cargo build' was registered as a real tuple: {sorted(registry)}")
        # The SAME dropped prose line must be
        # visible via discover_suspicious_lines — pinning the intentional
        # drop, not just its absence from the real registry.
        suspicious = discover_suspicious_lines(tmp)
        if not any("cargo build and a real torch install" in s for s in suspicious):
            failures.append(f"self-test FAILED: discover_suspicious_lines did not report the dropped bare-line prose sentence: {suspicious}")

    # --- on: block parsing: workflow_dispatch/schedule alone is never
    # merge-path; a workflow_call-only `on:` block is never merge-path ------
    dispatch_only = parse_on_block("on:\n  workflow_dispatch:\n  schedule:\n    - cron: \"0 0 * * *\"\n")
    ok, _reason = is_merge_path(dispatch_only)
    if ok:
        failures.append("self-test FAILED: workflow_dispatch/schedule alone was classified as merge-path")
    call_only = parse_on_block("on:\n  workflow_call:\n    inputs:\n      git_ref:\n        required: true\n")
    ok, _reason = is_merge_path(call_only)
    if ok:
        failures.append("self-test FAILED: a workflow_call-only on: block was classified as merge-path")

    # A quoted `"pull_request":` trigger is recognized as a qualifying
    # merge-path lane exactly like the bare form -- the parsed `on:` value's
    # own keys are read directly, so a quoted `"push":`/`"pull_request":`
    # trigger is never silently invisible.
    quoted_pr = parse_on_block('on:\n  "pull_request":\n    branches: [main]\n')
    ok, _reason = is_merge_path(quoted_pr)
    if not ok:
        failures.append(f'self-test FAILED: a quoted "pull_request": trigger is not classified as merge-path: {_reason}')

    # --- `read_top_level_on_block`/`parse_on_block`: the shared `on:`-block
    # readers `check_gpu_prove_once.py`'s P1, P5, P6, P7 and its
    # `--read-on-block` CLI all import (and `check_lint_surface_closure.py`,
    # via `parse_on_block`). Both derive from ONE parse of the document
    # (`load_workflow_text`), so they can no longer disagree with each
    # other -- asserted below as an explicit AGREEMENT check over the whole
    # battery, not just each reader's own expectation. Real, valid YAML
    # (folded/literal block scalars, flow mappings/sequences, a BOM, CRLF,
    # `on :` whitespace, a quoted `"on":` key) is read exactly as GitHub
    # itself would read it -- never refused merely for being an
    # unusual-looking spelling; a genuinely unparseable or ambiguous
    # document (a YAML syntax error, a duplicate key, a null/empty/
    # wrongly-typed `on:` value, an `on`/`"on"` collision) is a loud
    # refusal, and so -- a DIFFERENT case, real valid YAML that GitHub's
    # own parser still does not accept in a workflow file -- is any
    # anchor, alias (including a merge key), or explicit tag. ------------
    def _want_ok(label: str, text: str, want_keys: list[str]) -> None:
        keys, err = read_top_level_on_block(text)
        if err is not None or keys != want_keys:
            failures.append(f"self-test FAILED (on: reader, {label}): expected {want_keys}, got keys={keys} err={err!r}")

    def _want_refused(label: str, text: str, want_substr: str) -> None:
        keys, err = read_top_level_on_block(text)
        if keys is not None or err is None or want_substr not in err:
            failures.append(f"self-test FAILED (on: reader, {label}): expected a refusal naming {want_substr!r}, got keys={keys} err={err!r}")

    # A 30-shape GitHub-semantics differential (each shape's own oracle is
    # `yaml.safe_load` -- the same library this reader is built on -- so
    # this battery pins AGREEMENT with real
    # YAML semantics, never a hand-guessed expectation).
    _want_ok("inline bare", "on: push\njobs: {}\n", ["push"])
    _want_ok("inline double-quoted", 'on: "push"\njobs: {}\n', ["push"])
    _want_ok("inline single-quoted", "on: 'push'\njobs: {}\n", ["push"])
    _want_ok("inline with trailing comment", "on: push # hi\njobs: {}\n", ["push"])
    _want_ok("folded scalar >-", "name: x\non: >-\n  push\njobs:\n  a:\n    runs-on: u\n", ["push"])
    _want_ok("literal block scalar |", "name: x\non: |\n  push\njobs:\n  a:\n    runs-on: u\n", ["push\n"])
    _want_ok("flow sequence", "on: [push, pull_request]\njobs: {}\n", ["push", "pull_request"])
    _want_ok("flow mapping", "on: {push: {branches: [main]}}\njobs: {}\n", ["push"])
    _want_ok("block sequence", "on:\n  - push\n  - pull_request\njobs: {}\n", ["push", "pull_request"])
    _want_ok("quoted top-level on (double)", '"on":\n  push:\njobs: {}\n', ["push"])
    _want_ok("quoted top-level on (single)", "'on':\n  push:\njobs: {}\n", ["push"])
    _want_refused(
        "merge key with an undefined alias",
        "on:\n  <<: *base\n  push:\njobs: {}\n",
        "not accepted by GitHub Actions",
    )
    _want_ok("comment then a child key", "on:\n  # a comment\n  push:\n    branches: [main]\njobs: {}\n", ["push"])
    _want_refused("two on: blocks (duplicate top-level key)", "on:\n  workflow_dispatch:\non:\n  push:\njobs: {}\n", "duplicate key")
    _want_refused(
        "child value carries an anchor",
        "on:\n  push: &p\n    branches: [main]\njobs: {}\n",
        "not accepted by GitHub Actions",
    )
    _want_ok("leading doc-start marker + comment", "# lead comment\n---\non:\n  push:\njobs: {}\n", ["push"])
    # `True:` resolves to the SAME YAML 1.1 boolean key `on:` does -- a
    # workflow author who forgot to quote `on:` and typed `true:` instead
    # is reading the exact key GitHub's own parser reads too.
    _want_ok("True: (same boolean key as on:)", "True:\n  push:\njobs: {}\n", ["push"])
    _want_ok("on : (whitespace before the colon)", "on :\n  push:\njobs: {}\n", ["push"])
    _want_ok("BOM + CRLF", "\ufeffon:\r\n  push:\r\njobs: {}\r\n", ["push"])
    _want_ok("CRLF only", "on:\r\n  push:\r\n    branches: [main]\r\njobs: {}\r\n", ["push"])
    _want_refused("empty on: (null value)", "on:\njobs: {}\n", "empty")
    _want_refused(
        "anchor on the whole on: value + a live schedule child",
        'on: &trig\n  schedule:\n    - cron: "0 0 * * *"\njobs: {}\n',
        "not accepted by GitHub Actions",
    )
    _want_refused(
        "YAML tag forcing a string",
        "on: !!str push\njobs: {}\n",
        "not accepted by GitHub Actions",
    )
    _want_ok("quoted child key", 'on:\n  "pull_request":\n    branches: [main]\njobs: {}\n', ["pull_request"])
    _want_refused("tab indentation", "on:\n\tpush:\njobs: {}\n", "cannot parse YAML")
    _want_refused("inconsistently dedented sibling", "on:\n    push:\n  pull_request:\njobs: {}\n", "cannot parse YAML")
    _want_ok("trailing space inside a quoted child key", 'on:\n  "push ":\njobs: {}\n', ["push "])
    _want_ok("4-space child re-dedented to 2 for the next sibling", "on:\n  push:\n    branches:\n      - main\n  pull_request:\njobs: {}\n", ["push", "pull_request"])
    _want_ok(
        "a fake on:/push: pair embedded inside a run: block scalar is never confused with the real one",
        'on:\n  push:\njobs:\n  a:\n    steps:\n      - run: |\n          cat > w.yml <<EOF\n          "on": push\n          EOF\n',
        ["push"],
    )
    _want_refused("on: ~ (null value)", "on: ~\njobs: {}\n", "empty")
    _want_ok("trailing whitespace after the on: colon", "on:  \n  push:\njobs: {}\n", ["push"])

    # `read_top_level_on_block` and `parse_on_block`/`is_merge_path` can no
    # longer disagree: both derive from the SAME `load_workflow_text` parse.
    # Asserted directly over a representative slice of the battery above
    # (the shapes that legitimately carry a `push`/`pull_request` trigger).
    for label, text in (
        ("inline bare", "on: push\njobs: {}\n"),
        ("folded scalar", "on: >-\n  push\njobs: {}\n"),
        ("flow sequence", "on: [push, pull_request]\njobs: {}\n"),
        ("quoted top-level on", '"on":\n  push:\n    branches: [main]\njobs: {}\n'),
        ("block sequence", "on:\n  - push\n  - pull_request\njobs: {}\n"),
    ):
        keys, err = read_top_level_on_block(text)
        on_dict, perr = parse_on_block_or_fail(text)
        if err is None and perr is None:
            if sorted(keys or []) != sorted((on_dict or {}).keys()):
                failures.append(
                    f"self-test FAILED (reader agreement, {label}): read_top_level_on_block={keys} "
                    f"but parse_on_block_or_fail keys={sorted((on_dict or {}).keys())}"
                )
        elif (err is None) != (perr is None):
            failures.append(
                f"self-test FAILED (reader agreement, {label}): one reader refused and the other did "
                f"not -- on_err={err!r} parse_err={perr!r}"
            )

    # --- `job_source_spans`/`jobs_or_fail`: a flow-style `jobs:` mapping
    # must be refused, not collapsed to zero-length or wrong-job spans (a
    # `jobs: {sneaky: {...}, tail: {...}}` shape would leave `sneaky`'s own
    # body empty and credit `tail` with `sneaky`'s text); a duplicate job
    # id, or two top-level `jobs:` blocks, would silently shadow the earlier
    # entry. All three are a named, loud refusal, never a partial or
    # wrong-attributed result. ---------------------------------------------
    def _want_jobs_refused(label: str, text: str, want_substr: str) -> None:
        jobs, err = jobs_or_fail(text)
        if jobs is not None or err is None or want_substr not in err:
            failures.append(
                f"self-test FAILED (jobs: reader, {label}): expected a refusal naming {want_substr!r}, "
                f"got jobs={jobs} err={err!r}"
            )

    _want_jobs_refused(
        "flow-style jobs: mapping",
        "on:\n  push:\njobs: { a: { runs-on: u }, b: { runs-on: u } }\n",
        "flow-style",
    )
    _want_jobs_refused(
        "duplicate job id under jobs:",
        "on:\n  push:\njobs:\n  x:\n    runs-on: u\n  x:\n    runs-on: u\n",
        "duplicate key",
    )
    _want_jobs_refused(
        "two top-level jobs: blocks",
        "on:\n  push:\njobs:\n  x:\n    runs-on: u\n\njobs:\n  y:\n    runs-on: u\n",
        "duplicate key",
    )
    # `jobs:` is BLOCK-style in both of these (unlike the "flow-style jobs:
    # mapping" case above) -- what makes each one unexaminable is a single
    # JOB's own value not occupying its own line span. The refusal must
    # name that construct, never "flow-style" (a wrong diagnosis).
    _want_jobs_refused(
        "block jobs: with one job's value inline on its own key's line",
        "on:\n  push:\njobs:\n  a: { runs-on: u, steps: [ { run: 'npm publish' } ] }\n",
        "does not occupy its own line span in a block-style jobs: mapping",
    )
    _want_jobs_refused(
        "block jobs: with a later empty-bodied job id",
        "on:\n  push:\njobs:\n  a:\n    runs-on: u\n    steps:\n      - run: npm publish\n  b:\n",
        "does not occupy its own line span in a block-style jobs: mapping",
    )
    for _label, _text in (
        (
            "block jobs: with one job's value inline on its own key's line",
            "on:\n  push:\njobs:\n  a: { runs-on: u, steps: [ { run: 'npm publish' } ] }\n",
        ),
        (
            "block jobs: with a later empty-bodied job id",
            "on:\n  push:\njobs:\n  a:\n    runs-on: u\n    steps:\n      - run: npm publish\n  b:\n",
        ),
    ):
        _jobs, _err = jobs_or_fail(_text)
        if _err is not None and "flow-style" in _err:
            failures.append(
                f"self-test FAILED (jobs: reader, {_label}): refusal wrongly names 'flow-style' for a "
                f"block-style jobs: mapping, got err={_err!r}"
            )
    # The P6-shape end-to-end case (a flow-style `jobs:` mapping whose real
    # publisher job is not the last entry must never read as `[]`/
    # no-finding while crediting a DIFFERENT job with the missing job's own
    # text) is covered in `test_check_gpu_prove_once.py`'s own suite, which
    # owns `check_p6_discovery` -- not re-derived here. The "reusable
    # this delegates into but cannot examine is a finding, never `{}`"
    # property is ALSO owned there (`check_p6_discovery` is the only
    # consumer of the recursive helper this fixes).

    # --- gate prerequisite: a missing PyYAML install is ONE distinct
    # message, never a finding and never a pass -- simulated by
    # monkeypatching this module's own `yaml` name to `None` (never an
    # actual uninstall). `main()` checks this BEFORE calling `run_gate`, so
    # simulating it here never touches the real repo tree. -----------------
    _saved_yaml, _saved_err = globals()["yaml"], globals()["_YAML_IMPORT_ERROR"]
    globals()["yaml"] = None
    globals()["_YAML_IMPORT_ERROR"] = ImportError("simulated for self-test")
    try:
        rc = _pyyaml_prerequisite_rc()
        if rc != 2:
            failures.append(
                f"self-test FAILED: main() under a simulated missing PyYAML should exit 2 (a gate "
                f"prerequisite failure, never a finding and never a pass), got {rc}"
            )
        try:
            load_workflow_text("on: push\njobs: {}\n")
            failures.append("self-test FAILED: load_workflow_text did not raise under a simulated missing PyYAML")
        except WorkflowLoadError as exc:
            if "gate prerequisite missing: PyYAML" not in str(exc):
                failures.append(f"self-test FAILED: missing-PyYAML message is not the distinct prerequisite message: {exc}")
    finally:
        globals()["yaml"] = _saved_yaml
        globals()["_YAML_IMPORT_ERROR"] = _saved_err

    # --- the CLI entry point, in a REAL subprocess (never in-process): a
    # missing PyYAML must be caught BEFORE `--self-test` is ever dispatched
    # -- the previous ordering ran the whole ~30-case battery first, and
    # every case that needs a real parse failed with the prerequisite
    # message standing in for its own expected value, printed as ~20
    # separate "self-test FAILED" findings instead of the one prerequisite
    # line. A stub `yaml` package shadowing the real one on `PYTHONPATH`
    # (never an actual uninstall, and never touching this process's own
    # already-imported `yaml`) is the only way to exercise `main`'s actual
    # argv dispatch order end to end. Guarded by `_SELFTEST_SUBPROCESS_GUARD_ENV`
    # so this never recurses: under the EXACT mutation this fixture exists to
    # catch (the prerequisite check moved after the `--self-test` dispatch),
    # the spawned child would otherwise reach `self_test()` and hit this same
    # block again, spawning a grandchild, and so on without bound -- a
    # correctness-testing fixture must never itself become a fork bomb on the
    # mutant it is designed to kill. ---------------------------------------
    if os.environ.get(_SELFTEST_SUBPROCESS_GUARD_ENV) != "1":
        with tempfile.TemporaryDirectory() as td:
            stub_dir = Path(td) / "yaml"
            stub_dir.mkdir()
            (stub_dir / "__init__.py").write_text(
                "raise ImportError('stubbed out for self-test')\n", encoding="utf-8"
            )
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--self-test"],
                cwd=str(REPO_ROOT),
                env={**os.environ, "PYTHONPATH": td, _SELFTEST_SUBPROCESS_GUARD_ENV: "1"},
                capture_output=True,
                text=True,
                timeout=30,
            )
            combined = proc.stdout + proc.stderr
            if proc.returncode != 2:
                failures.append(
                    f"self-test FAILED (subprocess, missing PyYAML): expected exit 2, got "
                    f"{proc.returncode}; output: {combined!r}"
                )
            if "gate prerequisite missing: PyYAML" not in combined:
                failures.append(
                    f"self-test FAILED (subprocess, missing PyYAML): the distinct prerequisite "
                    f"message never appeared; output: {combined!r}"
                )
            if "self-test FAILED" in combined:
                failures.append(
                    f"self-test FAILED (subprocess, missing PyYAML): `--self-test` ran its battery "
                    f"instead of being short-circuited by the prerequisite check first; output: "
                    f"{combined!r}"
                )

    # --- docs.yml's own shape: a comment line NESTED inside a block `paths:`
    # list must not truncate the list (docs.yml carries this shape) --------
    on_dict, on_err = parse_on_block_or_fail(
        'on:\n  push:\n    branches: [main]\n    paths:\n      - "docs/guide/**"\n      # a comment mid-list\n      - "cookbook/recipes/**"\njobs: {}\n'
    )
    with_comment = (on_dict or {}).get("push", {}).get("paths") if on_err is None else None
    if with_comment != ["docs/guide/**", "cookbook/recipes/**"]:
        failures.append(f"self-test FAILED: a comment line nested inside a block `paths:` list truncated it: {with_comment} (err={on_err!r})")

    if failures:
        print("execution-surface-reachability self-test: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        "execution-surface-reachability self-test: OK — every rule bites: registry recursion + "
        "doc-comment exclusion + non-gated exclusion (Rule 2), pull_request/push-to-main reachable, "
        "unreachable/unallowlisted, label-only workflow, tag-only push, branches-ignore (both "
        "directions), paths-filter capability (both directions, incl. paths-ignore admit/exclude and "
        "tags-ignore-alone), job-level if:/continue-on-error:, step-level if:, step-level "
        "continue-on-error:, matrix cmd: indirection conjoined with its own interpolating step's "
        "blocked-state (a credited leg, a soft-failed leg, step-if, no-interpolating-step, and "
        "step-continue-on-error), a workflow-side `||`-tail credited ONLY against a "
        "known fail-loud allowlist (`|| true`, the unenumerated-swallow mutants — "
        "`|| echo`, `|| /bin/true`, `|| test`, `|| { ...; exit 0; }` — and `|| rc=$?` ALONE "
        "(a plain assignment always exits 0) — "
        "all refused, `|| exit N`/`|| exit $?`/`|| return N` still credited), an unsupported "
        "`!`/`{...}` paths pattern refused with a named "
        "finding EAGERLY (order-independent AND tuple-independent — a bang-after-a-matching-pattern "
        "and a bad-pattern workflow carrying no gated tuple at all), a quoted "
        "`\"on\":` key parsing like bare `on:`, comment-only mention, exact-match-never-substring, "
        "allowlisted-and-live PASS, rotted allowlist row, dead-waiver row, empty-reason row, malformed "
        "row (Rule 3), env-prefixed/wrapper-prefixed/semicolon-chained/continuation-spanning discovery, "
        "multi-feature/-F/namespaced-flash-attn gating (incl. quoted --features=\"cuda\"), prose "
        "exclusion (both a comment/assignment-wrapped and a bare cargo-subcommand-shaped sentence, the "
        "latter also pinned via discover_suspicious_lines), workflow_call-only exclusion, a "
        "paths: list surviving a nested comment line, a shell if:/case-wrapped run: body excluded "
        "wholesale, and pull_request types: [opened] alone excluded while an explicit "
        "types: [synchronize] still credits."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
