#!/usr/bin/env python3
"""Lint-surface closure gate (esc-059, class `enforcement-surface-gap`).

## The class this closes

`cargo metadata` enumerates every `[[test]]`/`[[bin]]`/`[[example]]`/
`[[bench]]` target a workspace crate declares with a non-empty
`required-features` list — the mechanical definition of "a target that only
exists/compiles under some feature combination". Before this gate,
`jammi-kernels`'s `cuda_parity` test target (`required-features = ["cuda"]`)
compiled under nobody's `cargo clippy -D warnings` on the merge path: the
one merge-path job that DID compile CUDA-feature code
(`flash-attn-compile` in `ci.yml`) ran `cargo check`, not `clippy`, and
never passed `--all-targets`/`--tests`, so it never even reached
`cuda_parity.rs`; the one place that DID run the right `clippy` invocation —
`runpod_gpu_prove.sh`'s own byte-identical twin — was wired only behind
`gpu-prove.yml`'s `workflow_dispatch` / `pull_request: types: [labeled]` /
nightly `schedule` triggers — never a trigger that fires on every
PR-to-main. That twin has since been REMOVED from `runpod_gpu_prove.sh`
entirely (esc-081: the prove lane never needed a GPU to run clippy); the
merge-path coverage this gate's own module doc describes below (`ci.yml`'s
hermetic `Clippy jammi-kernels --features flash-attn --all-targets` step) is
now the ONLY place this exact invocation runs, never a second copy. Four
`clippy::doc_lazy_continuation` lints landed in `cuda_parity.rs` (M2 audit
round 6, commit b0c16192) and sat on `main` — every required merge-path
check green — until they broke all four pods on the very next fresh-seed
run (esc-059's own observable).

This gate makes that class structurally impossible to reintroduce silently:
it is not a check for those four specific lints, or even for
`clippy::doc_lazy_continuation` as a lint id — it is a *closure* property
("every feature-gated target has SOME merge-path clippy lane covering it"),
so ANY lint clippy can catch, on ANY feature-gated target, is covered by
construction the day a new required lane is added, and REDs the day one
stops covering what it used to.

## Method (hermetic: `cargo metadata --no-deps`, no network beyond what a
warm local registry cache already provides, no build)

1. `feature_gated_targets` walks every workspace package's targets and
   keeps the ones with a non-empty `required-features` list — the ONLY
   metadata-visible notion of "this target needs a feature to even exist".
   A crate's `[features]` map is walked (implication only: `flash-attn =
   ["cuda"]` means enabling `flash-attn` also enables `cuda`; a
   `dep:name`/`name/feat`/`name?/feat` spec is a FORWARD to another
   package or an optional-dep activation, not a same-package feature name,
   and is intentionally not followed here — this gate only needs to know
   which of THIS package's own named features are active, not whether an
   external crate's feature flips).
2. This repo's own `check_execution_surface_reachability.py` already
   implements (and self-tests, at length) the honest "is this workflow
   genuinely on the merge path" question — `on:` trigger honesty (a
   `pull_request: types: [labeled]`-only trigger, like `gpu-prove.yml`,
   does NOT count), `if:`/`continue-on-error:` job/step-conditional
   honesty, and this repo's own `Guard`-matrix `cmd:` indirection. Rather
   than re-derive a second, subtly-different notion of "merge path" (the
   DRY invariant this swarm holds itself to), this gate imports that
   module's `scan_workflows` and reuses its already-vetted
   `WorkflowScan.tuples` corpus verbatim as the set of command lines that
   can be credited at all — together with the scan those lines came from,
   because WHICH workflow hosts a lane decides whether it runs on the PR
   that would break the lint (see the registry half below). Rule 1b —
   path-filter capability, asked with the crate's own sources as the
   origin — is applied by the REGISTRY half only
   (`required_lane_is_present`). The closure half deliberately does not
   apply it, and that is a disclosed narrowness, not a silent one: a
   `required-features` target is credited from Rule 1a/1c trigger honesty
   alone, so a lane hosted in a workflow whose `paths:` never admit that
   target's crate would still close a closure gap here. On today's tree
   the two halves agree — every `cargo clippy ... -D warnings` line in
   the merge-path corpus is hosted by `ci.yml`, whose `pull_request:
   branches: [main]` trigger carries no `paths:` filter at all. That
   premise is about the CORPUS, never about the repo: `crates.yml:56`
   carries a `cargo clippy --workspace --all-targets -- -D warnings` line
   too, and it is absent from the corpus because that workflow is
   `push:`-tags/`workflow_dispatch`-only and so never merge-path at all.
3. Every `cargo clippy ... -D warnings` (or `--deny warnings`) line in that
   corpus is parsed (`parse_clippy_lane`) into its crate scope
   (`-p`/`--workspace`/`--exclude`), its feature selection
   (`--features`/`--all-features`/`--no-default-features`, resolved
   through step 1's same-package implication walk), and its target
   selection (`--all-targets`/`--tests`/`--test NAME`/`--bins`/`--bin
   NAME`/`--examples`/`--benches`/`--lib`, or cargo's own default of
   "lib + bins only" when no selection flag is given at all).
4. `target_is_covered` asks, for every target from step 1: does ANY lane
   from step 3 (a) scope to this target's crate, (b) activate a feature
   set that is a SUPERSET of the target's `required-features`, and (c)
   select this target's OWN kind (a `test` target needs `--all-targets`,
   `--tests`, or an exact `--test <name>` match — cargo's default
   selection never reaches a test target at all)? A target with no
   covering lane is a FINDING; any finding fails the gate.

## Second property: the committed required-lane registry

Step 1's derivation is silent about a crate that declares no
`required-features` target at all. `jammi-ai` is exactly that crate: its
`cuda` arm lives in `src/` items and in `#[cfg(test)]` fns, not behind a
`[[test]]` target with `required-features = ["cuda"]`, so the closure half
above credits its `cargo clippy -p jammi-ai --features cuda --tests --
-D warnings` lane for nothing and would stay green if that lane were
deleted. Deleting it is a real, one-line regression: nothing else on the
merge path lints that arm, and nothing else compiles its `cfg(test)` half.

`ci/scripts/lint_surface_required_lanes.txt` closes that half. Each row
(`<crate> <feature-set> <target-selection>`) is an obligation in the other
direction: a lane matching it MUST be present in the SAME merge-path corpus
step 2/3 already build, or the gate FAILs naming the row. A feature may be
NEGATED (`!flash-attn`), because "some lane activates at least these
features" cannot express the reason a second, narrower lane exists: the
`jammi-encoders` `cuda`-only lane is the only one that compiles the
`#[cfg(all(feature = "cuda", not(feature = "flash-attn")))]` meta-test, and a
plain superset row for it would be satisfied by the `cuda,flash-attn` lane
and would therefore stay green when the lane it names is deleted. The registry is
matched against parsed lanes, never against step names or raw workflow
text, so renaming a step, or moving it between jobs of a workflow that
still runs on every PR touching the row's crate, is free — while dropping
it, narrowing it, de-required-path-ing it, or moving it into a workflow
that does NOT run on such a PR is a FAIL. A missing or
empty registry file is a FAIL, not a skip — the fail-closed rule that keeps
deleting the registry from being the way around it.

That last clause is a determinant of its own, and it is the one a
trigger-only notion of "merge path" misses. A workflow can qualify under
Rule 1a and still never see the PR that breaks the lint: `image-cuda.yml`
is `push:`-to-main-only (no `pull_request` trigger at all), and
`pypi-server-cuda.yml`'s `pull_request` trigger is filtered to
`packaging/server-cu12/**`, `crates/jammi-server/**` and the two wheel
workflow files. A `-p jammi-ai --features cuda --tests` lane moved into
either would satisfy the `jammi-ai cuda tests` row while a PR editing only
`crates/jammi-ai/**` ran neither, which is the fail-open this row exists to
prevent. So a row is credited only by a lane whose HOSTING workflow carries
a `pull_request`-to-main trigger admitting that crate's own sources —
`_lane_admits_any_origin` (the exec-surface module's own Rule 1b predicate,
reused) over `crates/<crate>/**`, the crate's directory derived from
`cargo metadata`'s manifest path rather than hard-coded. An unfiltered PR
trigger (`ci.yml`'s) admits it degenerately; a `push:`-only host has no
qualifying trigger to ask and is never credited.

`--self-test` proves the checker actually discriminates, against the REAL
(vetted) workflow corpus: a synthetic target requiring a feature no real
lane ever passes (`definitely-uncovered-xyz`) must come back UNCOVERED, and
a synthetic target shaped exactly like a real, currently-covered one
(`jammi-kernels` / `cuda`) must come back COVERED — so a checker that
always reports everything covered (or everything uncovered) cannot pass
either half. For the registry half it runs the mutation the registry
exists for: with every `-p jammi-ai` lane filtered out of the real corpus
(the shape of deleting that step from `ci.yml`), the real
`jammi-ai cuda tests` row must read UNSATISFIED, and against the unfiltered
corpus every real row must read SATISFIED. It runs the host-workflow
mutations end-to-end as well, through the real workflow parser: a copy of
`.github/workflows/` with that step's `run:` line MOVED out of `ci.yml`
into a synthetic job in `image-cuda.yml` (push-to-main + `paths:`), and the
same move into `pypi-server-cuda.yml` (a `pull_request` whose `paths:` do
not list `crates/jammi-ai/**`), must each read UNSATISFIED, while the
same copy with nothing moved reads SATISFIED — so the mutation's verdict
cannot be an artefact of copying the workflows.

Run: `python3 ci/scripts/check_lint_surface_closure.py [--self-test]`
Exit 0 = every feature-gated target has a covering merge-path clippy lane
AND every registry row is matched by one; 1 = at least one gap, an
unmatched/missing registry row (or a broken self-test control); 2 =
usage/metadata error, including a malformed registry row.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXEC_SURFACE_MODULE_PATH = REPO_ROOT / "ci" / "scripts" / "check_execution_surface_reachability.py"
REQUIRED_LANES_PATH = REPO_ROOT / "ci" / "scripts" / "lint_surface_required_lanes.txt"


def _load_exec_surface_module():
    """Dynamic load (this repo's own `check_lead_gate.py` precedent) — the
    module lives in `ci/scripts/`, not an importable package, and its
    filename is not a directly-importable name from this script's own
    working directory in every invocation shape (`python3 ci/scripts/x.py`
    run from the repo root, `python3 x.py` run from inside `ci/scripts/`,
    ...)."""
    mod_name = "check_execution_surface_reachability"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, EXEC_SURFACE_MODULE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {EXEC_SURFACE_MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec_module: the target module defines `@dataclass`
    # classes whose machinery looks itself up via `sys.modules[cls.__module__]`
    # while the module body is still executing -- registering only AFTER
    # `exec_module` returns is too late and raises `AttributeError` on
    # `None.__dict__` (reproduced on Python 3.14).
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# step 1 — feature-gated targets + same-package feature implication
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GatedTarget:
    crate: str
    target: str
    kind: str  # "test" | "bin" | "example" | "bench" (cargo's target.kind[0])
    required_features: tuple[str, ...]


def load_metadata(repo_root: Path = REPO_ROOT) -> dict:
    try:
        out = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as e:  # pragma: no cover
        print(f"ERROR: cargo metadata failed: {e}", file=sys.stderr)
        sys.exit(2)
    return json.loads(out)


def feature_gated_targets(metadata: dict) -> list[GatedTarget]:
    found: list[GatedTarget] = []
    for pkg in metadata.get("packages", []):
        for t in pkg.get("targets", []):
            rf = tuple(t.get("required-features") or [])
            if not rf:
                continue
            kinds = t.get("kind") or ["lib"]
            found.append(GatedTarget(pkg["name"], t["name"], kinds[0], rf))
    return sorted(found, key=lambda g: (g.crate, g.target))


def package_feature_maps(metadata: dict) -> dict[str, dict[str, list[str]]]:
    return {pkg["name"]: pkg.get("features", {}) for pkg in metadata.get("packages", [])}


def resolve_active_features(
    feature_map: dict[str, list[str]], requested: set[str], use_default: bool
) -> set[str]:
    """Same-package feature implication closure only (`flash-attn` ->
    `cuda`): a `dep:name` or `name/feat`/`name?/feat` spec forwards to a
    DIFFERENT package (or activates an optional dep) and is intentionally
    not followed — this gate never needs to know whether some OTHER
    crate's feature is active, only which of THIS crate's own named
    `[features]` keys are reachable, since `required-features` is always
    scoped to the same package as the target."""
    active: set[str] = set()

    def resolve(feat: str) -> None:
        if feat in active:
            return
        active.add(feat)
        for spec in feature_map.get(feat, []):
            if spec.startswith("dep:") or "/" in spec:
                continue
            resolve(spec)

    if use_default:
        resolve("default")
    for f in requested:
        resolve(f)
    active.discard("default")
    return active


# --------------------------------------------------------------------------- #
# step 3 — parse a normalized `cargo clippy ...` tuple into a coverage lane
# --------------------------------------------------------------------------- #
_P_RE = re.compile(r"(?:^|\s)-p\s+(\S+)")
_FEATURES_RE = re.compile(r"(?:--features|-F)[=\s]+(\S+)")
_DENY_WARNINGS_RE = re.compile(r"(?:-D\s*warnings|--deny[=\s]+warnings)")


@dataclass(frozen=True)
class PrTriggerLane:
    """One `pull_request`-to-main trigger's own `paths:`/`paths-ignore:`
    filter, in a hashable form. The exec-surface module's own `PathLane`
    is a mutable dataclass and cannot ride on a frozen `ClippyLane`; this
    carries the same two fields and is converted back to a real `PathLane`
    at the moment its Rule-1b predicate is asked, so the admission logic
    itself is never re-implemented here."""

    paths: tuple[str, ...] | None
    paths_ignore: tuple[str, ...] | None


@dataclass(frozen=True)
class LaneOrigin:
    """The workflow that hosts a parsed lane, with the `pull_request`-to-main
    triggers it fires on. `pr_lanes == ()` means the hosting workflow has NO
    qualifying `pull_request`-to-main trigger at all (`image-cuda.yml`'s
    `push:`-to-main-only shape) — such a host can never be credited for a
    registry row, whatever its `paths:` say."""

    workflow: str
    pr_lanes: tuple[PrTriggerLane, ...]


# The origin a hand-written lane in the self-test carries when the host is
# not what that assertion is about: a workflow whose PR-to-main trigger
# admits everything, i.e. `ci.yml`'s own shape. Never a default — a lane
# with NO origin is not credited for a registry row at all (fail-closed),
# so a corpus that forgot to record its origins REDs rather than passes.
UNFILTERED_PR_ORIGIN = LaneOrigin(workflow="<synthetic-unfiltered>", pr_lanes=(PrTriggerLane(None, None),))


@dataclass(frozen=True)
class ClippyLane:
    raw: str
    crates: frozenset[str] | None  # None == --workspace/--all (every member)
    excluded: frozenset[str]
    features: frozenset[str]
    all_features: bool
    no_default_features: bool
    all_targets: bool
    tests: bool
    explicit_tests: frozenset[str]
    bins: bool
    explicit_bins: frozenset[str]
    examples: bool
    benches: bool
    lib: bool
    origin: LaneOrigin | None = None


def _collect_valued(tokens: list[str], flag: str) -> set[str]:
    out: set[str] = set()
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            out.add(tokens[i + 1])
    return out


def parse_clippy_lane(raw: str, origin: LaneOrigin | None = None) -> ClippyLane | None:
    if not re.match(r"^cargo\s+clippy(?=\s|$)", raw):
        return None
    if not _DENY_WARNINGS_RE.search(raw):
        return None  # not a `-D warnings` lane -- does not close the esc-059 class
    tokens = raw.split()
    crates = set(_P_RE.findall(raw))
    workspace = "--workspace" in tokens or "--all" in tokens
    excluded: set[str] = set()
    if "--exclude" in tokens:
        i = tokens.index("--exclude")
        j = i + 1
        while j < len(tokens) and not tokens[j].startswith("-"):
            excluded.add(tokens[j])
            j += 1
    features: set[str] = set()
    for m in _FEATURES_RE.finditer(raw):
        features |= {f for f in m.group(1).split(",") if f}
    return ClippyLane(
        raw=raw,
        crates=None if workspace else frozenset(crates),
        excluded=frozenset(excluded),
        features=frozenset(features),
        all_features="--all-features" in tokens,
        no_default_features="--no-default-features" in tokens,
        all_targets="--all-targets" in tokens,
        tests="--tests" in tokens,
        explicit_tests=frozenset(_collect_valued(tokens, "--test")),
        bins="--bins" in tokens,
        explicit_bins=frozenset(_collect_valued(tokens, "--bin")),
        examples="--examples" in tokens,
        benches="--benches" in tokens,
        lib="--lib" in tokens,
        origin=origin,
    )


def workflow_pr_lanes(exec_mod, workflow_path: Path) -> tuple[PrTriggerLane, ...]:
    """The `pull_request`-to-main trigger lanes of ONE workflow file, read
    with the exec-surface module's own `parse_on_block`/`_pr_admits_main`
    (never a second `on:` parser). Empty when the workflow has no
    qualifying `pull_request` trigger — a `push:`-to-main-only workflow,
    or one whose `pull_request` is `types: [labeled]`-only. GitHub allows
    at most one `pull_request` key per workflow, so this returns 0 or 1
    lane; the tuple shape keeps the caller from caring."""
    try:
        text = workflow_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):  # pragma: no cover -- scanned it once already
        return ()
    pr = exec_mod.parse_on_block(text).get("pull_request")
    if pr is None or not exec_mod._pr_admits_main(pr):
        return ()
    paths = pr.get("paths")
    paths_ignore = pr.get("paths-ignore")
    return (
        PrTriggerLane(
            paths=tuple(paths) if paths is not None else None,
            paths_ignore=tuple(paths_ignore) if paths_ignore is not None else None,
        ),
    )


def crate_source_origins(metadata: dict, repo_root: Path = REPO_ROOT) -> dict[str, tuple[str, ...]]:
    """`crate name -> the path globs a PR touching that crate's own sources
    would carry`, derived from `cargo metadata`'s manifest paths (so a crate
    that does not live under `crates/` is still asked about correctly, and
    no directory layout is hard-coded). A package whose manifest is outside
    the repo root — a path dependency vendored elsewhere — gets no origin
    and is therefore never credited, which is the fail-closed side."""
    origins: dict[str, tuple[str, ...]] = {}
    for pkg in metadata.get("packages", []):
        manifest = Path(pkg.get("manifest_path", ""))
        try:
            rel = manifest.parent.resolve().relative_to(repo_root.resolve())
        except ValueError:
            continue
        rel_str = rel.as_posix()
        origins[pkg["name"]] = (f"{rel_str}/**",) if rel_str != "." else ("**",)
    return origins


def lanes_from_workflows(exec_mod, repo_root: Path = REPO_ROOT) -> list[ClippyLane]:
    """The union of every `cargo clippy -D warnings` line reachable from a
    genuinely merge-path-triggered job/step, across every workflow —
    `check_execution_surface_reachability.scan_workflows`'s own Rule 1a +
    1c honesty (trigger + if:/continue-on-error: conditioning), reused
    rather than re-derived. Each lane keeps the scan it came from (its
    hosting workflow and that workflow's `pull_request`-to-main trigger
    lanes): the registry half asks Rule 1b of the HOST, so discarding the
    scan here would make "moved into a workflow no PR touching this crate
    ever runs" indistinguishable from "still on `ci.yml`"."""
    scans, _pattern_findings = exec_mod.scan_workflows(repo_root)
    workflows_dir = repo_root / exec_mod.WORKFLOWS_DIR_REL
    lanes: list[ClippyLane] = []
    for scan in scans:
        origin = LaneOrigin(
            workflow=scan.name, pr_lanes=workflow_pr_lanes(exec_mod, workflows_dir / scan.name)
        )
        for tuple_text in scan.tuples:
            lane = parse_clippy_lane(tuple_text, origin=origin)
            if lane is not None:
                lanes.append(lane)
    return lanes


# --------------------------------------------------------------------------- #
# step 4 — coverage
# --------------------------------------------------------------------------- #
def _lane_covers_crate(lane: ClippyLane, crate: str) -> bool:
    if lane.crates is None:
        return crate not in lane.excluded
    return crate in lane.crates


def _lane_covers_features(
    lane: ClippyLane, crate: str, feature_maps: dict[str, dict[str, list[str]]], required: tuple[str, ...]
) -> bool:
    if lane.all_features:
        return True
    active = resolve_active_features(
        feature_maps.get(crate, {}), set(lane.features), use_default=not lane.no_default_features
    )
    return set(required).issubset(active)


def _lane_has_no_explicit_target_selection(lane: ClippyLane) -> bool:
    """True when the lane passes no target-selection flag at all, so cargo's
    own default selection ("lib + bins") applies."""
    return not (
        lane.all_targets
        or lane.tests
        or lane.bins
        or lane.examples
        or lane.benches
        or lane.lib
        or lane.explicit_tests
        or lane.explicit_bins
    )


def _lane_covers_target_kind(lane: ClippyLane, target: GatedTarget) -> bool:
    if lane.all_targets:
        return True
    if target.kind == "test":
        return lane.tests or target.target in lane.explicit_tests
    if target.kind == "bin":
        # cargo's own default target selection (no selection flag at all)
        # is "lib + bins" -- a bin target is covered even with zero flags.
        return (
            lane.bins
            or target.target in lane.explicit_bins
            or _lane_has_no_explicit_target_selection(lane)
        )
    if target.kind == "example":
        return lane.examples
    if target.kind == "bench":
        return lane.benches
    return True  # lib/cdylib/etc: cargo always compiles the lib target


def target_is_covered(
    lanes: list[ClippyLane], feature_maps: dict[str, dict[str, list[str]]], target: GatedTarget
) -> bool:
    for lane in lanes:
        if not _lane_covers_crate(lane, target.crate):
            continue
        if not _lane_covers_features(lane, target.crate, feature_maps, target.required_features):
            continue
        if not _lane_covers_target_kind(lane, target):
            continue
        return True
    return False


def find_gaps(
    targets: list[GatedTarget], lanes: list[ClippyLane], feature_maps: dict[str, dict[str, list[str]]]
) -> list[GatedTarget]:
    return [t for t in targets if not target_is_covered(lanes, feature_maps, t)]


# --------------------------------------------------------------------------- #
# step 5 -- the committed required-lane registry (the other direction)
# --------------------------------------------------------------------------- #
SELECTIONS = ("all-targets", "tests", "bins", "lib", "examples", "benches")


class RegistryError(Exception):
    """A malformed row in `lint_surface_required_lanes.txt`."""


@dataclass(frozen=True)
class RequiredLane:
    crate: str
    features: tuple[str, ...]  # features the lane must activate
    forbidden: tuple[str, ...]  # features the lane must NOT activate (`!feat`)
    selection: str
    lineno: int

    def __str__(self) -> str:  # the exact text a FAIL names
        feats = ",".join(list(self.features) + [f"!{f}" for f in self.forbidden])
        return f"{self.crate} {feats or '-'} {self.selection}"

    def features_field(self) -> str:
        return str(self).split(" ")[1]


def parse_required_lanes(text: str) -> list[RequiredLane]:
    rows: list[RequiredLane] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 3:
            raise RegistryError(
                f"{REQUIRED_LANES_PATH.name}:{lineno}: expected 3 whitespace-separated fields "
                f"(<crate> <feature-set> <target-selection>), got {len(fields)}: {stripped!r}"
            )
        crate, feature_field, selection = fields
        if selection not in SELECTIONS:
            raise RegistryError(
                f"{REQUIRED_LANES_PATH.name}:{lineno}: unknown target-selection {selection!r}; "
                f"expected one of {', '.join(SELECTIONS)}"
            )
        raw_features = () if feature_field == "-" else tuple(f for f in feature_field.split(",") if f)
        features = tuple(f for f in raw_features if not f.startswith("!"))
        forbidden = tuple(f[1:] for f in raw_features if f.startswith("!"))
        if any(not f for f in forbidden):
            raise RegistryError(
                f"{REQUIRED_LANES_PATH.name}:{lineno}: a bare `!` is not a feature name"
            )
        rows.append(
            RequiredLane(
                crate=crate,
                features=features,
                forbidden=forbidden,
                selection=selection,
                lineno=lineno,
            )
        )
    return rows


def load_required_lanes(path: Path = REQUIRED_LANES_PATH) -> list[RequiredLane]:
    """Fail-closed on absence: a missing registry raises rather than reading
    as "no obligations", so deleting the file is not the way around it."""
    if not path.is_file():
        raise RegistryError(
            f"required-lane registry {path} is missing -- it is the committed half of this "
            "gate; restore it (an empty file is also a FAIL, not a skip)"
        )
    return parse_required_lanes(path.read_text(encoding="utf-8"))


def _lane_selects(lane: ClippyLane, selection: str) -> bool:
    if lane.all_targets:
        return True
    if selection == "all-targets":
        return False
    if selection == "tests":
        return lane.tests
    if selection == "examples":
        return lane.examples
    if selection == "benches":
        return lane.benches
    # `lib` and `bins` are both in cargo's own default selection, so a lane
    # passing no selection flag at all compiles them.
    if selection == "lib":
        return lane.lib or _lane_has_no_explicit_target_selection(lane)
    if selection == "bins":
        return lane.bins or _lane_has_no_explicit_target_selection(lane)
    raise RegistryError(f"unknown target-selection {selection!r}")  # pragma: no cover


def lane_active_features(
    lane: ClippyLane, crate: str, feature_maps: dict[str, dict[str, list[str]]]
) -> set[str]:
    """The crate features a lane actually activates. `--all-features` is
    every feature the crate declares — spelled out rather than short-circuited
    to "matches anything", because a registry row can FORBID a feature and
    `--all-features` is precisely the lane that activates it."""
    fmap = feature_maps.get(crate, {})
    if lane.all_features:
        return {f for f in fmap if f != "default"}
    return resolve_active_features(fmap, set(lane.features), use_default=not lane.no_default_features)


def lane_runs_on_pr_touching_crate(
    exec_mod, lane: ClippyLane, crate: str, crate_dirs: dict[str, tuple[str, ...]]
) -> bool:
    """Rule 1b, asked of the lane's HOSTING workflow with the row's crate as
    the origin: would a PR that edits `crates/<crate>/**` actually run this
    workflow? A lane with no recorded origin, a host with no
    `pull_request`-to-main trigger, and a host whose `paths:` exclude the
    crate all answer NO — the three fail-closed arms. An unsupported
    `paths:` pattern is treated as not admitting (the same fail-closed
    convention `is_tuple_reachable` uses for it), never as a crash.

    The origin is the crate's WHOLE directory, so a host filtered to a
    proper subset of it (`crates/<crate>/src/**`) reads as not admitting.
    That is the conservative answer and the intended one: a PR editing
    `crates/<crate>/Cargo.toml` alone would not run such a host, so the
    lane it carries is no merge-path guarantee for the crate."""
    if lane.origin is None:
        return False
    origins = list(crate_dirs.get(crate, (f"crates/{crate}/**",)))
    for pr_lane in lane.origin.pr_lanes:
        path_lane = exec_mod.PathLane(
            paths=list(pr_lane.paths) if pr_lane.paths is not None else None,
            paths_ignore=list(pr_lane.paths_ignore) if pr_lane.paths_ignore is not None else None,
        )
        try:
            if exec_mod._lane_admits_any_origin(path_lane, origins):
                return True
        except exec_mod.UnsupportedPathPatternError:
            continue
    return False


def _lane_matches_row_except_host(
    lane: ClippyLane, feature_maps: dict[str, dict[str, list[str]]], req: RequiredLane
) -> bool:
    """Every axis of a row EXCEPT the hosting workflow's own reachability —
    factored out so a FAIL can name the near miss ("the lane exists, in a
    workflow that never runs on a PR touching this crate") instead of the
    much less useful "no lane matches"."""
    if not _lane_covers_crate(lane, req.crate):
        return False
    active = lane_active_features(lane, req.crate, feature_maps)
    if not set(req.features).issubset(active):
        return False
    if active & set(req.forbidden):
        return False
    return _lane_selects(lane, req.selection)


def required_lane_is_present(
    lanes: list[ClippyLane],
    feature_maps: dict[str, dict[str, list[str]]],
    req: RequiredLane,
    exec_mod,
    crate_dirs: dict[str, tuple[str, ...]],
) -> bool:
    for lane in lanes:
        if not _lane_matches_row_except_host(lane, feature_maps, req):
            continue
        if not lane_runs_on_pr_touching_crate(exec_mod, lane, req.crate, crate_dirs):
            continue
        return True
    return False


def hosts_that_do_not_run_on_the_crate(
    lanes: list[ClippyLane],
    feature_maps: dict[str, dict[str, list[str]]],
    req: RequiredLane,
    exec_mod,
    crate_dirs: dict[str, tuple[str, ...]],
) -> list[str]:
    """The workflows carrying a lane that matches this row on every other
    axis but is hosted where a PR touching the crate never runs."""
    return sorted(
        {
            lane.origin.workflow if lane.origin is not None else "<no recorded host workflow>"
            for lane in lanes
            if _lane_matches_row_except_host(lane, feature_maps, req)
            and not lane_runs_on_pr_touching_crate(exec_mod, lane, req.crate, crate_dirs)
        }
    )


def find_missing_required_lanes(
    required: list[RequiredLane],
    lanes: list[ClippyLane],
    feature_maps: dict[str, dict[str, list[str]]],
    exec_mod,
    crate_dirs: dict[str, tuple[str, ...]],
) -> list[RequiredLane]:
    return [
        r for r in required if not required_lane_is_present(lanes, feature_maps, r, exec_mod, crate_dirs)
    ]


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
#: The step the host-workflow controls move around. Written once, asserted to
#: exist exactly once in `ci.yml` before any mutation is judged — a control
#: that silently moved nothing would "prove" the row goes UNSATISFIED for the
#: wrong reason.
_MOVED_STEP_TUPLE = "cargo clippy -p jammi-ai --features cuda --tests -- -D warnings"
_SYNTHETIC_JOB = """
  moved-clippy-lane:
    runs-on: ubuntu-latest
    steps:
      - name: Clippy jammi-ai --features cuda (moved by the self-test)
        run: {tuple_text}
"""


def _corpus_with_step_moved(exec_mod, moved_to: str | None, remove_from_ci: bool) -> list[ClippyLane]:
    """A lane corpus built from a COPY of `.github/workflows/` in which the
    `-p jammi-ai --features cuda --tests` step is removed from `ci.yml`
    (`remove_from_ci`) and re-added as a synthetic job in `moved_to`. The
    mutation runs through the REAL workflow parser and the real scan, so it
    exercises the same path the gate takes on the real tree; `moved_to=None,
    remove_from_ci=False` is the copy-only control that proves a verdict is
    not an artefact of copying."""
    src = REPO_ROOT / exec_mod.WORKFLOWS_DIR_REL
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dst = root / exec_mod.WORKFLOWS_DIR_REL
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        if remove_from_ci:
            ci = dst / "ci.yml"
            text = ci.read_text(encoding="utf-8")
            hosting = [ln for ln in text.splitlines() if ln.strip() == f"run: {_MOVED_STEP_TUPLE}"]
            assert len(hosting) == 1, (
                "self-test FAILED: expected exactly one `run: "
                f"{_MOVED_STEP_TUPLE}` line in ci.yml, found {len(hosting)} -- the host-workflow "
                "controls below would prove nothing"
            )
            ci.write_text(text.replace(hosting[0] + "\n", "", 1), encoding="utf-8")
        if moved_to is not None:
            host = dst / moved_to
            assert host.is_file(), f"self-test FAILED: no such workflow to move the step into: {moved_to}"
            host.write_text(
                host.read_text(encoding="utf-8").rstrip("\n")
                + "\n"
                + _SYNTHETIC_JOB.format(tuple_text=_MOVED_STEP_TUPLE),
                encoding="utf-8",
            )
        return lanes_from_workflows(exec_mod, root)


def self_test() -> int:
    exec_mod = _load_exec_surface_module()
    lanes = lanes_from_workflows(exec_mod, REPO_ROOT)
    assert lanes, "self-test: no `cargo clippy -D warnings` lane found on the merge path at all"
    metadata = load_metadata(REPO_ROOT)
    feature_maps = package_feature_maps(metadata)
    crate_dirs = crate_source_origins(metadata, REPO_ROOT)
    assert crate_dirs.get("jammi-ai") == ("crates/jammi-ai/**",), (
        "self-test FAILED: `jammi-ai`'s source origin derived from cargo metadata is "
        f"{crate_dirs.get('jammi-ai')!r}; the host-workflow controls below are written against "
        "`crates/jammi-ai/**` and would prove nothing against a different one"
    )

    # Positive control: a target shaped exactly like the real,
    # currently-covered `jammi-kernels::cuda_parity` (required-features =
    # ["cuda"], a `test` target) must be reported COVERED — proves the
    # checker can find a real match, not just reject everything.
    covered = GatedTarget(crate="jammi-kernels", target="cuda_parity", kind="test", required_features=("cuda",))
    assert target_is_covered(lanes, feature_maps, covered), (
        "self-test FAILED (false negative): jammi-kernels/cuda_parity-shaped target reported "
        "UNCOVERED against the real merge-path lane corpus -- either the corpus regressed or "
        "the checker itself is broken"
    )

    # Negative control (the actual esc-059 shape): a target requiring a
    # feature no real lane ever passes must be reported UNCOVERED.
    uncovered = GatedTarget(
        crate="jammi-kernels",
        target="synthetic_uncovered_target",
        kind="test",
        required_features=("definitely-uncovered-xyz",),
    )
    assert not target_is_covered(lanes, feature_maps, uncovered), (
        "self-test FAILED (false positive): a target requiring a feature no real lane ever "
        "activates was reported COVERED -- the checker is vacuous"
    )

    # Same idea, but the gap is in TARGET SELECTION, not features: a crate
    # whose only covering lane never passes --all-targets/--tests for a
    # test-kind target must still read as uncovered even though the
    # feature set matches exactly.
    lane_no_tests = parse_clippy_lane("cargo clippy -p jammi-kernels --features cuda -- -D warnings")
    assert lane_no_tests is not None
    uncovered_by_target_selection = GatedTarget(
        crate="jammi-kernels", target="synthetic_test_only", kind="test", required_features=("cuda",)
    )
    assert not target_is_covered(
        [lane_no_tests], feature_maps, uncovered_by_target_selection
    ), "self-test FAILED: a lane with no --all-targets/--tests must not cover a test-kind target"
    covering_lane = parse_clippy_lane(
        "cargo clippy -p jammi-kernels --all-targets --features cuda -- -D warnings"
    )
    assert covering_lane is not None
    assert target_is_covered(
        [covering_lane], feature_maps, uncovered_by_target_selection
    ), "self-test FAILED: adding --all-targets to the same lane must flip it to covered"

    # A non-`-D warnings` clippy invocation (or a bare `cargo check`) must
    # never be parsed into a lane at all -- it does not close the lint
    # class this gate guards.
    assert parse_clippy_lane("cargo check -p jammi-kernels --features cuda --all-targets") is None
    assert parse_clippy_lane("cargo clippy -p jammi-kernels --features cuda --all-targets") is None

    # ----- the registry half -------------------------------------------- #
    required = load_required_lanes()
    assert required, (
        "self-test FAILED: the required-lane registry parsed to zero rows -- an empty registry "
        "is a fail-closed condition, not a skip"
    )

    # Positive control: every committed row is matched by the real corpus.
    for req in required:
        assert required_lane_is_present(lanes, feature_maps, req, exec_mod, crate_dirs), (
            f"self-test FAILED: committed required lane `{req}` "
            f"({REQUIRED_LANES_PATH.name}:{req.lineno}) is matched by no merge-path lane"
        )

    # Negative control, the mutation this registry exists for: filter every
    # `-p jammi-ai` lane out of the REAL corpus -- the in-process shape of
    # deleting `ci.yml`'s `Clippy jammi-ai --features cuda` step -- and the
    # committed `jammi-ai cuda tests` row must go UNSATISFIED.
    jammi_ai_rows = [r for r in required if r.crate == "jammi-ai"]
    assert jammi_ai_rows, (
        "self-test FAILED: the registry no longer carries a `jammi-ai` row, so this control "
        "proves nothing -- restore the row or rewrite this control against the crate that "
        "replaced it"
    )
    corpus_without_jammi_ai = [
        lane for lane in lanes if not (lane.crates is not None and "jammi-ai" in lane.crates)
    ]
    assert len(corpus_without_jammi_ai) < len(lanes), (
        "self-test FAILED: filtering `-p jammi-ai` removed no lane at all, so the mutation "
        "control is vacuous"
    )
    for req in jammi_ai_rows:
        assert not required_lane_is_present(
            corpus_without_jammi_ai, feature_maps, req, exec_mod, crate_dirs
        ), (
            f"self-test FAILED (vacuous registry): required lane `{req}` still reads SATISFIED "
            "after every `-p jammi-ai` lane was removed from the corpus"
        )

    # The HOST-workflow mutations, end-to-end through the real workflow
    # parser: a trigger-only notion of "merge path" reads all three of these
    # as SATISFIED. The first is the copy-only control -- it must stay
    # SATISFIED, or the two verdicts below would be artefacts of copying
    # `.github/workflows/` rather than of where the step now lives.
    jammi_ai_row = jammi_ai_rows[0]
    assert required_lane_is_present(
        _corpus_with_step_moved(exec_mod, None, False), feature_maps, jammi_ai_row, exec_mod, crate_dirs
    ), (
        f"self-test FAILED: `{jammi_ai_row}` reads UNSATISFIED against an UNMUTATED copy of "
        ".github/workflows -- the mutation controls below would prove nothing"
    )
    for host, why in (
        ("image-cuda.yml", "a `push:`-to-main-only workflow (no `pull_request` trigger at all)"),
        (
            "pypi-server-cuda.yml",
            "a workflow whose `pull_request` `paths:` do not list `crates/jammi-ai/**`",
        ),
    ):
        moved = _corpus_with_step_moved(exec_mod, host, True)
        assert not required_lane_is_present(
            moved, feature_maps, jammi_ai_row, exec_mod, crate_dirs
        ), (
            f"self-test FAILED (merge-path blindness): `{jammi_ai_row}` reads SATISFIED with its "
            f"only lane moved into {host} -- {why}, so a PR touching only crates/jammi-ai/** "
            "would run no lane at all"
        )
        assert hosts_that_do_not_run_on_the_crate(
            moved, feature_maps, jammi_ai_row, exec_mod, crate_dirs
        ) == [host], (
            f"self-test FAILED: the FAIL for `{jammi_ai_row}` must name {host} as the host that "
            "never runs on a PR touching the crate"
        )

    # ... and the same predicate must still SAY YES where the host genuinely
    # does run on the crate: `pypi-server-cuda.yml`'s `paths:` DO list
    # `crates/jammi-server/**`. Without this arm, "not ci.yml" would pass for
    # the rule.
    pypi_origin = LaneOrigin(
        workflow="pypi-server-cuda.yml",
        pr_lanes=workflow_pr_lanes(
            exec_mod, REPO_ROOT / exec_mod.WORKFLOWS_DIR_REL / "pypi-server-cuda.yml"
        ),
    )
    assert len(pypi_origin.pr_lanes) == 1 and pypi_origin.pr_lanes[0].paths, (
        "self-test FAILED: pypi-server-cuda.yml no longer carries a paths-filtered "
        "`pull_request` trigger, so the control above tests something else now"
    )
    image_cuda_origin = LaneOrigin(
        workflow="image-cuda.yml",
        pr_lanes=workflow_pr_lanes(
            exec_mod, REPO_ROOT / exec_mod.WORKFLOWS_DIR_REL / "image-cuda.yml"
        ),
    )
    assert image_cuda_origin.pr_lanes == (), (
        "self-test FAILED: image-cuda.yml now has a `pull_request`-to-main trigger, so the "
        "push-only control above tests something else now"
    )
    host_probe = parse_clippy_lane(
        "cargo clippy -p jammi-server --tests -- -D warnings", origin=pypi_origin
    )
    assert host_probe is not None
    assert not lane_runs_on_pr_touching_crate(exec_mod, host_probe, "jammi-ai", crate_dirs), (
        "self-test FAILED: pypi-server-cuda.yml's paths were read as admitting crates/jammi-ai/**"
    )
    assert lane_runs_on_pr_touching_crate(exec_mod, host_probe, "jammi-server", crate_dirs), (
        "self-test FAILED (over-strict): pypi-server-cuda.yml's `paths:` DO list "
        "`crates/jammi-server/**`, so a lane hosted there must be credited for a jammi-server row"
    )

    # Negative control on the feature axis: a row naming a feature no lane
    # ever activates must go unsatisfied.
    assert not required_lane_is_present(
        lanes,
        feature_maps,
        RequiredLane("jammi-ai", ("definitely-uncovered-xyz",), (), "tests", 0),
        exec_mod,
        crate_dirs,
    ), "self-test FAILED: a row requiring a feature no real lane activates read SATISFIED"

    # Negative control on the target-selection axis: same crate, same
    # features, but the only lane lacks `--tests`.
    lane_no_tests = parse_clippy_lane(
        "cargo clippy -p jammi-ai --features cuda -- -D warnings", origin=UNFILTERED_PR_ORIGIN
    )
    assert lane_no_tests is not None
    assert not required_lane_is_present(
        [lane_no_tests],
        feature_maps,
        RequiredLane("jammi-ai", ("cuda",), (), "tests", 0),
        exec_mod,
        crate_dirs,
    ), "self-test FAILED: a lane without --tests/--all-targets satisfied a `tests` row"
    lane_with_tests = parse_clippy_lane(
        "cargo clippy -p jammi-ai --features cuda --tests -- -D warnings",
        origin=UNFILTERED_PR_ORIGIN,
    )
    assert lane_with_tests is not None
    assert required_lane_is_present(
        [lane_with_tests],
        feature_maps,
        RequiredLane("jammi-ai", ("cuda",), (), "tests", 0),
        exec_mod,
        crate_dirs,
    ), "self-test FAILED: adding --tests to the same lane must flip the row to satisfied"

    # Negative control on the HOST axis, with everything else held
    # equal: the same lane, hosted in a workflow with no
    # `pull_request`-to-main trigger at all, satisfies nothing.
    lane_pushed_only = parse_clippy_lane(
        "cargo clippy -p jammi-ai --features cuda --tests -- -D warnings",
        origin=LaneOrigin(workflow="<synthetic-push-only>", pr_lanes=()),
    )
    assert lane_pushed_only is not None
    assert not required_lane_is_present(
        [lane_pushed_only],
        feature_maps,
        RequiredLane("jammi-ai", ("cuda",), (), "tests", 0),
        exec_mod,
        crate_dirs,
    ), "self-test FAILED: a lane whose host never runs on a PR satisfied a row"
    lane_no_origin = parse_clippy_lane(
        "cargo clippy -p jammi-ai --features cuda --tests -- -D warnings"
    )
    assert lane_no_origin is not None
    assert not required_lane_is_present(
        [lane_no_origin],
        feature_maps,
        RequiredLane("jammi-ai", ("cuda",), (), "tests", 0),
        exec_mod,
        crate_dirs,
    ), "self-test FAILED: a lane with no recorded host origin satisfied a row"

    # Negation control: `cuda,!flash-attn` must reject the very lane a plain
    # superset row would have accepted -- the reason the second
    # `jammi-encoders` row is not redundant with the first.
    flash_lane = parse_clippy_lane(
        "cargo clippy -p jammi-encoders --features cuda,flash-attn --tests -- -D warnings",
        origin=UNFILTERED_PR_ORIGIN,
    )
    cuda_only_lane = parse_clippy_lane(
        "cargo clippy -p jammi-encoders --features cuda --tests -- -D warnings",
        origin=UNFILTERED_PR_ORIGIN,
    )
    assert flash_lane is not None and cuda_only_lane is not None
    not_flash_row = RequiredLane("jammi-encoders", ("cuda",), ("flash-attn",), "tests", 0)
    assert not required_lane_is_present(
        [flash_lane], feature_maps, not_flash_row, exec_mod, crate_dirs
    ), "self-test FAILED: a `cuda,flash-attn` lane satisfied a row that FORBIDS flash-attn"
    assert required_lane_is_present(
        [cuda_only_lane], feature_maps, not_flash_row, exec_mod, crate_dirs
    ), "self-test FAILED: the cuda-only lane must satisfy the `cuda,!flash-attn` row"
    all_features_lane = parse_clippy_lane(
        "cargo clippy -p jammi-encoders --all-features --tests -- -D warnings",
        origin=UNFILTERED_PR_ORIGIN,
    )
    assert all_features_lane is not None
    assert not required_lane_is_present(
        [all_features_lane], feature_maps, not_flash_row, exec_mod, crate_dirs
    ), (
        "self-test FAILED: --all-features activates flash-attn, so it cannot satisfy a row "
        "that forbids it"
    )

    # A lane that does not deny warnings is not a lane at all, so it cannot
    # satisfy a row (the registry rides on `parse_clippy_lane`'s own filter).
    assert parse_clippy_lane("cargo clippy -p jammi-ai --features cuda --tests") is None

    # Registry parsing: comments and blanks are skipped, `-` is the empty
    # feature set, and a malformed row raises rather than being ignored.
    parsed = parse_required_lanes("# comment\n\njammi-ai cuda tests\njammi-ai - lib  # trailing\n")
    assert [str(p) for p in parsed] == ["jammi-ai cuda tests", "jammi-ai - lib"], parsed
    for bad in ("jammi-ai cuda", "jammi-ai cuda tests extra", "jammi-ai cuda --tests"):
        try:
            parse_required_lanes(bad)
        except RegistryError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"self-test FAILED: malformed registry row {bad!r} was accepted")

    print("self-test: ok")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        try:
            return self_test()
        except AssertionError as e:
            print(f"lint-surface-closure self-test: FAIL: {e}", file=sys.stderr)
            return 1

    exec_mod = _load_exec_surface_module()
    metadata = load_metadata(REPO_ROOT)
    targets = feature_gated_targets(metadata)
    feature_maps = package_feature_maps(metadata)
    lanes = lanes_from_workflows(exec_mod, REPO_ROOT)
    try:
        required = load_required_lanes()
    except RegistryError as e:
        print(f"lint-surface-closure: FAIL -- {e}", file=sys.stderr)
        return 2

    if not lanes:
        print(
            "lint-surface-closure: FAIL -- zero `cargo clippy ... -D warnings` lanes found on "
            "the merge path at all; either the corpus regressed or every clippy lane was moved "
            "off the required path",
            file=sys.stderr,
        )
        return 1

    if not required:
        print(
            f"lint-surface-closure: FAIL -- the required-lane registry "
            f"({REQUIRED_LANES_PATH.relative_to(REPO_ROOT)}) carries zero rows; an empty "
            "registry asserts nothing and is a fail-closed condition, not a skip",
            file=sys.stderr,
        )
        return 1

    gaps = find_gaps(targets, lanes, feature_maps)
    for t in targets:
        tag = "GAP" if t in gaps else "OK"
        print(
            f"lint-surface-closure[{t.crate}::{t.target}] required-features="
            f"{list(t.required_features)}: {tag}"
        )

    crate_dirs = crate_source_origins(metadata, REPO_ROOT)
    missing = find_missing_required_lanes(required, lanes, feature_maps, exec_mod, crate_dirs)
    for r in required:
        tag = "MISSING" if r in missing else "OK"
        print(f"lint-surface-closure[required-lane {r}]: {tag}")

    # Both halves are reported before returning: a run that stops at the
    # first finding hides the second, and a reader fixing one would then
    # discover the other only on the next CI round.
    if missing or gaps:
        print("lint-surface-closure: FAIL", file=sys.stderr)
    if missing:
        for r in missing:
            print(
                f"  - required lane `{r}` "
                f"({REQUIRED_LANES_PATH.relative_to(REPO_ROOT)}:{r.lineno}) is matched by no "
                "merge-path `cargo clippy ... -D warnings` invocation. Some workflow must run "
                f"`cargo clippy -p {r.crate}"
                + (f" --features {','.join(r.features)}" if r.features else "")
                + f" --{r.selection} -- -D warnings`"
                + (
                    f" (and it must NOT activate {', '.join(r.forbidden)})"
                    if r.forbidden
                    else ""
                )
                + " from a job/step that fires on every PR to main; restore or widen that "
                "lane, or remove this row deliberately if the lint surface is genuinely "
                "being narrowed.",
                file=sys.stderr,
            )
            near_misses = hosts_that_do_not_run_on_the_crate(
                lanes, feature_maps, r, exec_mod, crate_dirs
            )
            if near_misses:
                print(
                    f"    a matching lane DOES exist, in {', '.join(near_misses)} -- but that "
                    f"workflow has no `pull_request`-to-main trigger admitting "
                    f"{', '.join(crate_dirs.get(r.crate, (f'crates/{r.crate}/**',)))}, so a PR "
                    "touching only that crate never runs it. Host the lane in a workflow that "
                    "does (`ci.yml`), or widen that workflow's `paths:`.",
                    file=sys.stderr,
                )
    if gaps:
        for t in gaps:
            print(
                f"  - {t.crate}::{t.target} ({t.kind}, required-features={list(t.required_features)}) "
                "is compiled under no merge-path `cargo clippy -D warnings` lane -- a lint "
                "regression there can land on main and only surface on the next GPU pod run "
                "(the exact esc-059 shape). Add or widen a lane in .github/workflows/ci.yml.",
                file=sys.stderr,
            )
    if missing or gaps:
        return 1

    print(
        f"lint-surface-closure: all {len(targets)} feature-gated target(s) are covered, and "
        f"all {len(required)} committed required lane(s) are present."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
