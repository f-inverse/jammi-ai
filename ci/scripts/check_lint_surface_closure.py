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
`cuda_parity.rs`; the one place that DID run the right `clippy` invocation
(`runpod_gpu_prove.sh:56`) is wired only behind `gpu-prove.yml`'s
`workflow_dispatch` / `pull_request: types: [labeled]` / nightly `schedule`
triggers — never a trigger that fires on every PR-to-main. Four
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
   can be credited at all. (Deliberately NOT reusing its Rule-1b
   path-filter-capability check: that rule answers "would editing THIS
   ci/scripts/** file re-trigger the workflow", a question with no
   equivalent for a compiled crate target: `ci.yml`, today's only
   workflow carrying any `cargo clippy` line, carries no `paths:` filter
   at all, so 1b is vacuously satisfied for every crate-source path and
   adding it here would only reduplicate machinery for zero behavioural
   difference — a disclosed narrowness, not a silent one, see
   `target_is_covered`'s own doc.)
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

`--self-test` proves the checker actually discriminates, against the REAL
(vetted) workflow corpus: a synthetic target requiring a feature no real
lane ever passes (`definitely-uncovered-xyz`) must come back UNCOVERED, and
a synthetic target shaped exactly like a real, currently-covered one
(`jammi-kernels` / `cuda`) must come back COVERED — so a checker that
always reports everything covered (or everything uncovered) cannot pass
either half.

Run: `python3 ci/scripts/check_lint_surface_closure.py [--self-test]`
Exit 0 = every feature-gated target has a covering merge-path clippy lane;
1 = at least one gap (or a broken self-test control); 2 = usage/metadata
error.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXEC_SURFACE_MODULE_PATH = REPO_ROOT / "ci" / "scripts" / "check_execution_surface_reachability.py"


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


def _collect_valued(tokens: list[str], flag: str) -> set[str]:
    out: set[str] = set()
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            out.add(tokens[i + 1])
    return out


def parse_clippy_lane(raw: str) -> ClippyLane | None:
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
    )


def lanes_from_workflows(exec_mod, repo_root: Path = REPO_ROOT) -> list[ClippyLane]:
    """The union of every `cargo clippy -D warnings` line reachable from a
    genuinely merge-path-triggered job/step, across every workflow —
    `check_execution_surface_reachability.scan_workflows`'s own Rule 1a +
    1c honesty (trigger + if:/continue-on-error: conditioning), reused
    rather than re-derived. See this module's own docstring for why Rule
    1b (path-filter capability) is deliberately not applied here."""
    scans, _pattern_findings = exec_mod.scan_workflows(repo_root)
    lanes: list[ClippyLane] = []
    for scan in scans:
        for tuple_text in scan.tuples:
            lane = parse_clippy_lane(tuple_text)
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


def _lane_covers_target_kind(lane: ClippyLane, target: GatedTarget) -> bool:
    if lane.all_targets:
        return True
    if target.kind == "test":
        return lane.tests or target.target in lane.explicit_tests
    if target.kind == "bin":
        no_explicit_selection = not (
            lane.tests
            or lane.bins
            or lane.examples
            or lane.benches
            or lane.lib
            or lane.explicit_tests
            or lane.explicit_bins
        )
        # cargo's own default target selection (no selection flag at all)
        # is "lib + bins" -- a bin target is covered even with zero flags.
        return lane.bins or target.target in lane.explicit_bins or no_explicit_selection
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
# self-test
# --------------------------------------------------------------------------- #
def self_test() -> int:
    exec_mod = _load_exec_surface_module()
    lanes = lanes_from_workflows(exec_mod, REPO_ROOT)
    assert lanes, "self-test: no `cargo clippy -D warnings` lane found on the merge path at all"
    feature_maps = package_feature_maps(load_metadata(REPO_ROOT))

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

    if not lanes:
        print(
            "lint-surface-closure: FAIL -- zero `cargo clippy ... -D warnings` lanes found on "
            "the merge path at all; either the corpus regressed or every clippy lane was moved "
            "off the required path",
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

    if gaps:
        print("lint-surface-closure: FAIL", file=sys.stderr)
        for t in gaps:
            print(
                f"  - {t.crate}::{t.target} ({t.kind}, required-features={list(t.required_features)}) "
                "is compiled under no merge-path `cargo clippy -D warnings` lane -- a lint "
                "regression there can land on main and only surface on the next GPU pod run "
                "(the exact esc-059 shape). Add or widen a lane in .github/workflows/ci.yml.",
                file=sys.stderr,
            )
        return 1

    print(f"lint-surface-closure: all {len(targets)} feature-gated target(s) are covered.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
