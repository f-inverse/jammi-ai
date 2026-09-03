#!/usr/bin/env python3
"""Assert flash-attn is reachable exactly where it is DECLARED.

**Guarded property**: `jammi-kernels/flash-attn` is reachable exactly where a
lane or a workspace member DECLARES it, via the exact declared 1:1 chain, and
NEVER from `cuda`/`default` alone. `jammi-kernels`'s `flash-attn` feature
builds the vendored FlashAttention-2 kernels (CUTLASS submodule + a minute of
`nvcc` + a static archive). It DEPENDS on `cuda` but must never be IMPLIED by
it. Cargo has no "never enable this feature" primitive, so this script walks
the workspace's feature graph the way the resolver does and proves the
property mechanically.

Method (hermetic: `cargo metadata --no-deps`, no network, no build):

  1. Load every workspace package's `[features]` table and dependency list.
  2. Read `ci/release-feature-manifest.json`'s `lanes` object — the single
     source of truth for every CUDA release lane's exact cargo feature list
     (cu12 tarball, cu12 wheel, cu12 image today; any future lane the
     manifest gains is picked up automatically). For EVERY lane, assert its
     declared `capabilities.flash_compiled` matches whether its
     `cargo_features` selection actually reaches
     `jammi-kernels/flash-attn` — a `true` lane that fails to reach it (a
     broken or renamed forwarding chain) and a `false` lane that DOES reach
     it (an undeclared leak) both FAIL. FAILS on a missing/unreadable
     manifest, a missing/renamed `lanes` key, or an empty lane list — this is
     the PR-time drift enforcement for every release lane (this gate runs on
     every PR via ci.yml), covering `release-binaries.yml`'s own missing
     `pull_request` trigger.
  3. Lane-independent invariants on `jammi-server` (`ROOT`) directly, held
     regardless of what the manifest says: a plain `cuda` selection and the
     bare `default` selection never reach `jammi-kernels/flash-attn`.
  4. Positive control (broken-walk vacuity guard): `ROOT`'s plain `cuda`
     selection MUST reach `jammi-kernels/cuda` — proving the walk actually
     traverses the server -> ai -> encoders/lora -> kernels chain rather than
     trivially seeing an empty set.
  5. `ROOT`'s own `--all-features` selection uses the SAME member-exemption
     idiom `ALL_FEATURES_FLASH_EXEMPT` gives every other workspace member
     (see step 6): accepted iff `jammi-server`'s own `flash-attn` feature
     spec is EXACTLY `["jammi-ai/flash-attn"]` (a verified 1:1 passthrough)
     AND its `default`/`cuda` real selections (checked via the same
     `_check_exempt_member_real_lanes` helper the member loop uses,
     including `default = ["train"]`) stay flash-free.
  6. Beyond `ROOT`: for EVERY OTHER workspace member (a leak through
     `jammi-bench` or `jammi-python`, both of which reach `jammi-ai` ->
     ... -> `jammi-kernels` transitively, would not be visible from
     `jammi-server`'s closure alone), check THAT member's own
     `--all-features` selection. `jammi-kernels` itself is excluded (asking
     "does building jammi-kernels with all its own features enable
     jammi-kernels/flash-attn" is vacuously true and not the property this
     script guards). A member's leak under `--all-features` is accepted ONLY
     if `ALL_FEATURES_FLASH_EXEMPT` names that member with the member's
     OWN `flash-attn` feature spec matching EXACTLY — and even then, its own
     `default`/`cuda` selections must independently stay flash-free
     (`_check_exempt_member_real_lanes`).
  7. Propagation uses the resolver-v2 rules for workspace members: a plain
     `feat` enables it on the same package; `dep:name` activates an
     optional dependency (with its declared `features` and, unless
     `default-features = false`, `default`); `name/feat` activates `name`
     and enables `feat` on it; `name?/feat` enables `feat` on `name` only if
     `name` is already active. Non-optional normal/build dependencies of an
     active package are active. Dev dependencies are ignored (they are not
     part of a binary's closure). Non-workspace packages are opaque (they
     cannot enable a workspace member's feature).

`--self-test` runs the walker over synthetic metadata: a leaked edge, a weak-
dep edge, a clean graph, the manifest-derived per-lane checks (including a
synthetic `flash_compiled: false` lane so the negative-polarity branch is
provably alive), the member exemptions (`jammi-ai`, `jammi-bench`,
`jammi-encoders`), the `ROOT` `--all-features` exemption, and a name-only-
passthrough control (a declared `flash-attn` whose closure fails to actually
enable `jammi-kernels/flash-attn` FAILS).

Run: `python3 ci/scripts/check_flash_attn_closure.py`
Exit 0 = property holds; 1 = a leak/mismatch (or a broken control); 2 = usage/metadata/manifest error.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prove_surface  # noqa: E402
from check_execution_surface_reachability import (  # noqa: E402
    _drop_comment_lines,
    _join_line_continuations,
    discover_all_tuples,
    extract_tuples_from_line,
    is_gated,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "ci" / "release-feature-manifest.json"

# esc-081 scope map (ENUMERATED, never open-ended): every cuda-bearing
# (`is_gated`) tuple discovered anywhere under `ci/scripts/**` must live in
# exactly one of these two sets, or the gate FAILS it as unlisted.
PROVE_SCOPE = frozenset({"ci/scripts/runpod_gpu_prove.sh"})

_PERF_PRODUCER_REASON = "perf producer, not a proof lane"
EXEMPT_SCOPE: dict[str, str] = {
    "ci/scripts/pod_seed_target.sh": (
        "seed cache lane: T1 precedes CUTLASS provisioning, T1/T1b main-only "
        "split — a dedicated tuple-lockstep follow-up is tracked separately, "
        "not this gate's job to resolve"
    ),
    "ci/scripts/perf/finetune_ab.sh": _PERF_PRODUCER_REASON,
    "ci/scripts/perf/finetune_run_ab.sh": _PERF_PRODUCER_REASON,
    "ci/scripts/perf/encode_ab.sh": _PERF_PRODUCER_REASON,
    "ci/scripts/perf/gpu_inference_ab.sh": _PERF_PRODUCER_REASON,
    "ci/scripts/perf/fa2_ab.sh": _PERF_PRODUCER_REASON,
    "ci/scripts/perf/clip_artifact_producer.sh": _PERF_PRODUCER_REASON,
    "ci/scripts/perf/pod_build_timings.sh": _PERF_PRODUCER_REASON,
    "ci/scripts/perf/stacked_sweep.sh": _PERF_PRODUCER_REASON,
}

_PROVE_TUPLE_RE = re.compile(r'^echo\s+"PROVE_TUPLE crate=(\S+) kind=(\S+) features=(.*)"\s*$')

ROOT = "jammi-server"
TARGET_PKG = "jammi-kernels"
FORBIDDEN_FEATURE = "flash-attn"
# Positive control: this feature MUST be reachable from ROOT's plain `cuda`
# selection.
CONTROL_FEATURE = "cuda"

# `ROOT`'s own by-name `flash-attn` passthrough — the ONE spec its
# `--all-features` selection is exempted for (step 5 above).
ROOT_ALL_FEATURES_EXEMPT_SPEC = [f"jammi-ai/{FORBIDDEN_FEATURE}"]

# Workspace members permitted to reach TARGET_PKG/FORBIDDEN_FEATURE under
# their OWN `--all-features` selection ONLY (P6 Stage B, `jammi-encoders`'s
# `crate::modernbert` flash-cascade admission needs a declared forwarding
# path for `flash_attention_varlen`/`CuSeqlens` — a `#[cfg(feature =
# "flash-attn")]` call site, never a bare `cfg!()` runtime check around a
# `jammi_kernels::flash` type reference, which would fail to compile with
# the feature off; see the call site's own doc). `jammi-ai` forwards to both
# `jammi-encoders/flash-attn` and `jammi-kernels/flash-attn` directly (the
# same "explicit direct entry, not only transitive" precedent its `cuda`
# feature already carries for `jammi-kernels/cuda`). `jammi-bench` needs its
# own entry because the reporter-scenario producer (`jammi-bench
# finetune-run`) must be able to compile FA2 too. The value is the EXACT
# feature-spec list the member's own `flash-attn` entry must equal for the
# exemption to apply — a verified 1:1 passthrough, not "any leak from this
# crate is fine": if a future edit adds a second spec (e.g. also pulling in
# a heavier dependency), the exemption stops matching and this script goes
# back to FAILing on it.
#
# `--all-features` is a synthetic "build everything" selection no release
# lane uses; an opt-in feature `--all-features` could never reach would be
# untestable dead weight, so exempting it here does not weaken the
# property this script actually guards. That property — every REAL lane
# (the manifest's declared lanes) AND any plain `cuda`/`default` build of the
# exempted member stay CUTLASS-free unless the lane/member DECLARES
# `flash-attn` — is enforced separately, per exempted member, by re-running
# ITS OWN `default` and `cuda` selections (if it declares them) and still
# FAILing if either reaches `FORBIDDEN_FEATURE`.
ALL_FEATURES_FLASH_EXEMPT: dict[str, list[str]] = {
    "jammi-encoders": [f"{TARGET_PKG}/{FORBIDDEN_FEATURE}"],
    "jammi-ai": ["cuda", f"jammi-encoders/{FORBIDDEN_FEATURE}", f"{TARGET_PKG}/{FORBIDDEN_FEATURE}"],
    "jammi-bench": [f"jammi-ai/{FORBIDDEN_FEATURE}"],
}


def load_metadata() -> dict:
    try:
        out = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as e:  # pragma: no cover
        print(f"ERROR: cargo metadata failed: {e}", file=sys.stderr)
        sys.exit(2)
    return json.loads(out)


def load_manifest_lanes() -> dict[str, dict]:
    """Read and validate `ci/release-feature-manifest.json`'s `lanes`
    object. FAILS (exit 2) on a missing/unreadable file, a missing/renamed
    `lanes` key, an empty lane list, or a lane missing a required field —
    the manifest-read closure assertion for esc-074 must be unable to pass
    vacuously on a broken or absent manifest."""
    try:
        raw = MANIFEST_PATH.read_text()
    except OSError as e:
        print(f"ERROR: cannot read {MANIFEST_PATH}: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"ERROR: {MANIFEST_PATH} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)
    lanes = manifest.get("lanes")
    if not lanes:
        print(
            f"ERROR: {MANIFEST_PATH} has no (or an empty) `lanes` key — "
            f"nothing to guard (was it renamed?)",
            file=sys.stderr,
        )
        sys.exit(2)
    for lane_name, lane in lanes.items():
        for key in ("package", "cargo_features", "capabilities"):
            if key not in lane:
                print(
                    f"ERROR: lane `{lane_name}` in {MANIFEST_PATH} is missing "
                    f"required key `{key}`",
                    file=sys.stderr,
                )
                sys.exit(2)
        if "flash_compiled" not in lane["capabilities"]:
            print(
                f"ERROR: lane `{lane_name}` in {MANIFEST_PATH} is missing "
                f"required key `capabilities.flash_compiled`",
                file=sys.stderr,
            )
            sys.exit(2)
    return lanes


class Graph:
    """Workspace packages: name -> {features: {f: [specs]}, deps: [dep]}."""

    def __init__(self, metadata: dict):
        self.pkgs: dict[str, dict] = {}
        for p in metadata["packages"]:
            deps = []
            for d in p.get("dependencies", []):
                if d.get("kind") == "dev":
                    continue  # not part of a binary's closure
                deps.append(
                    {
                        # The name feature specs refer to is the rename if any.
                        "key": d.get("rename") or d["name"],
                        "pkg": d["name"],
                        "optional": bool(d.get("optional", False)),
                        "default": bool(d.get("uses_default_features", True)),
                        "features": list(d.get("features", [])),
                    }
                )
            self.pkgs[p["name"]] = {"features": p.get("features", {}), "deps": deps}

    def closure(self, root: str, root_features: list[str]) -> dict[str, set[str]]:
        enabled: dict[str, set[str]] = {n: set() for n in self.pkgs}
        active: set[str] = set()
        active_optional: dict[str, set[str]] = {n: set() for n in self.pkgs}
        work: list[tuple[str, str]] = []

        def activate(pkg: str, features: list[str], with_default: bool) -> None:
            if pkg not in self.pkgs:
                return  # opaque external crate
            if pkg not in active:
                active.add(pkg)
                for d in self.pkgs[pkg]["deps"]:
                    if not d["optional"]:
                        activate(d["pkg"], d["features"], d["default"])
            if with_default and "default" in self.pkgs[pkg]["features"]:
                work.append((pkg, "default"))
            for f in features:
                work.append((pkg, f))

        def dep_by_key(pkg: str, key: str) -> dict | None:
            for d in self.pkgs[pkg]["deps"]:
                if d["key"] == key:
                    return d
            return None

        activate(root, root_features, with_default=False)
        # Two-phase fixpoint. Phase 1 drains `work` to exhaustion; phase 2
        # promotes every deferred weak edge (`dep?/feat`) whose dep became
        # active during that drain, and the pair repeats until a drain
        # promotes nothing. The promotion scan MUST live outside the item
        # loop: several item paths end in `continue` (an unknown package, an
        # already-enabled feature, and — the one that bites — an implicit
        # optional-dependency feature, which ACTIVATES an optional dep and
        # then continues), so a scan at the bottom of the loop body is
        # skipped exactly on the paths that make a deferred edge fireable
        # and the walk silently drops it, reporting a leak-free closure for
        # a graph that leaks. Terminates: each promotion removes an entry
        # from `deferred`, which only ever grows while a feature is being
        # enabled for the first time.
        while True:
            while work:
                pkg, feat = work.pop()
                if pkg not in self.pkgs or feat in enabled[pkg]:
                    continue
                specs = self.pkgs[pkg]["features"].get(feat)
                if specs is None:
                    # `feat` names an implicit optional-dependency feature
                    # (`dep = { optional = true }` without `dep:` syntax).
                    d = dep_by_key(pkg, feat)
                    if d is not None and d["optional"]:
                        enabled[pkg].add(feat)
                        active_optional[pkg].add(d["key"])
                        activate(d["pkg"], d["features"], d["default"])
                    continue
                enabled[pkg].add(feat)
                for spec in specs:
                    if spec.startswith("dep:"):
                        d = dep_by_key(pkg, spec[4:])
                        if d is not None:
                            active_optional[pkg].add(d["key"])
                            activate(d["pkg"], d["features"], d["default"])
                    elif "/" in spec:
                        key, sub = spec.split("/", 1)
                        weak = key.endswith("?")
                        key = key.rstrip("?")
                        d = dep_by_key(pkg, key)
                        if d is None:
                            continue
                        if d["optional"] and not weak:
                            active_optional[pkg].add(d["key"])
                            activate(d["pkg"], d["features"], d["default"])
                        if (not d["optional"]) or (d["key"] in active_optional[pkg]):
                            work.append((d["pkg"], sub))
                        elif weak:
                            # Deferred: if the dep is activated later the
                            # feature must follow (phase 2 below).
                            deferred.append((pkg, d, sub))
                    else:
                        work.append((pkg, spec))
            # Phase 2: weak edges whose dep became active during the drain.
            promoted = False
            still: list[tuple[str, dict, str]] = []
            for (p2, d2, sub2) in deferred:
                if d2["key"] in active_optional[p2]:
                    work.append((d2["pkg"], sub2))
                    promoted = True
                else:
                    still.append((p2, d2, sub2))
            deferred[:] = still
            if not promoted:
                break
        return enabled

    def all_features(self, pkg: str) -> list[str]:
        feats = set(self.pkgs[pkg]["features"].keys())
        for d in self.pkgs[pkg]["deps"]:
            if d["optional"]:
                feats.add(d["key"])
        return sorted(feats)


deferred: list[tuple[str, dict, str]] = []


def _reaches_flash(graph: Graph, pkg: str, feats: list[str]) -> tuple[bool, list[str]]:
    deferred.clear()
    enabled = graph.closure(pkg, feats)
    kernels = sorted(enabled.get(TARGET_PKG, set()))
    return FORBIDDEN_FEATURE in kernels, kernels


def _check_manifest_lanes(graph: Graph, lanes: dict[str, dict], verbose: bool) -> int:
    """Per-lane, manifest-derived assertions (step 2 of the module doc)."""
    rc = 0
    for lane_name, lane in lanes.items():
        pkg = lane["package"]
        feats = lane["cargo_features"]
        want_flash = bool(lane["capabilities"]["flash_compiled"])
        if pkg not in graph.pkgs:
            if verbose:
                print(
                    f"FAIL: lane `{lane_name}` names package `{pkg}`, which is "
                    f"not a workspace member",
                    file=sys.stderr,
                )
            rc = 1
            continue
        reached, kernels = _reaches_flash(graph, pkg, feats)
        if verbose:
            print(
                f"lane `{lane_name}` ({pkg} [{','.join(feats)}]) -> "
                f"{TARGET_PKG} features: {kernels} (flash_compiled declared={want_flash})"
            )
        if reached != want_flash:
            if verbose:
                if want_flash:
                    print(
                        f"FAIL: lane `{lane_name}` declares capabilities.flash_compiled=true "
                        f"but its cargo_features do NOT reach {TARGET_PKG}/{FORBIDDEN_FEATURE} "
                        f"— a broken or renamed forwarding chain (name-only passthrough)",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"FAIL: lane `{lane_name}` declares capabilities.flash_compiled=false "
                        f"but its cargo_features DO reach {TARGET_PKG}/{FORBIDDEN_FEATURE} "
                        f"— an undeclared leak",
                        file=sys.stderr,
                    )
            rc = 1
    return rc


def _check_exempt_member_real_lanes(graph: Graph, pkg: str, verbose: bool) -> int:
    """The real property an `--all-features` exemption above must not
    weaken: an exempted member's OWN `default` and `cuda` selections (the
    selections a real build lane actually uses) must still exclude
    `FORBIDDEN_FEATURE`."""
    rc = 0
    for label in ("default", "cuda"):
        if label not in graph.pkgs[pkg]["features"]:
            continue
        reached, kernels = _reaches_flash(graph, pkg, [label])
        if verbose:
            print(f"{pkg} [{label}] -> {TARGET_PKG} features: {kernels}")
        if reached:
            if verbose:
                print(
                    f"FAIL: {pkg} [{label}] reaches {TARGET_PKG}/{FORBIDDEN_FEATURE} — the "
                    f"exemption only covers --all-features, not a real build lane",
                    file=sys.stderr,
                )
            rc = 1
    return rc


def verdict(graph: Graph, lanes: dict[str, dict], verbose: bool = True) -> int:
    rc = 0

    # (2) Per-lane, manifest-derived assertions — the PR-time drift
    # enforcement for every declared CUDA release lane.
    rc |= _check_manifest_lanes(graph, lanes, verbose)

    # (3) Lane-independent invariants: `cuda` alone and bare `default` never
    # reach jammi-kernels/flash-attn, regardless of what any lane declares.
    lane_independent = {
        "default": ["default"] if "default" in graph.pkgs[ROOT]["features"] else [],
        "cuda": [CONTROL_FEATURE],
    }
    for label, feats in lane_independent.items():
        if not feats:
            continue
        reached, kernels = _reaches_flash(graph, ROOT, feats)
        if verbose:
            print(f"{ROOT} [{label}] -> {TARGET_PKG} features: {kernels}")
        if reached:
            if verbose:
                print(
                    f"FAIL: {ROOT} [{label}] reaches {TARGET_PKG}/{FORBIDDEN_FEATURE} — "
                    f"`{label}` must never imply flash-attn",
                    file=sys.stderr,
                )
            rc = 1
        # (4) Positive control (broken-walk vacuity guard).
        if label == "cuda" and CONTROL_FEATURE not in kernels:
            if verbose:
                print(
                    f"FAIL (positive control): {ROOT} [{label}] does not reach "
                    f"{TARGET_PKG}/{CONTROL_FEATURE}; the walk is not traversing the "
                    f"consumer -> kernels chain, so a 'clean' verdict would be vacuous",
                    file=sys.stderr,
                )
            rc = 1

    # (5) ROOT's own `--all-features`, using the SAME exemption idiom the
    # member loop below uses (never dead config for ROOT).
    reached, kernels = _reaches_flash(graph, ROOT, graph.all_features(ROOT))
    if verbose:
        print(f"{ROOT} [--all-features] -> {TARGET_PKG} features: {kernels}")
    if reached:
        own_spec = graph.pkgs[ROOT]["features"].get(FORBIDDEN_FEATURE)
        if own_spec == ROOT_ALL_FEATURES_EXEMPT_SPEC:
            if verbose:
                print(
                    f"EXEMPT: {ROOT} [--all-features] reaches {TARGET_PKG}/{FORBIDDEN_FEATURE} "
                    f"only via its own by-name `{FORBIDDEN_FEATURE} = {own_spec}` passthrough "
                    f"— checking {ROOT}'s own default/cuda selections instead"
                )
            rc |= _check_exempt_member_real_lanes(graph, ROOT, verbose)
        else:
            if verbose:
                print(
                    f"FAIL: {ROOT} [--all-features] reaches {TARGET_PKG}/{FORBIDDEN_FEATURE} "
                    f"and its own `{FORBIDDEN_FEATURE}` spec ({own_spec}) does not match the "
                    f"exempted 1:1 passthrough {ROOT_ALL_FEATURES_EXEMPT_SPEC}",
                    file=sys.stderr,
                )
            rc = 1

    # (6) Widen beyond ROOT: every OTHER workspace member's OWN
    # --all-features selection (jammi-kernels itself excluded — see the
    # module docstring).
    for pkg in sorted(graph.pkgs):
        if pkg in (ROOT, TARGET_PKG):
            continue
        reached, kernels = _reaches_flash(graph, pkg, graph.all_features(pkg))
        if verbose:
            print(f"{pkg} [--all-features] -> {TARGET_PKG} features: {kernels}")
        if reached:
            own_spec = graph.pkgs[pkg]["features"].get(FORBIDDEN_FEATURE)
            exempt_spec = ALL_FEATURES_FLASH_EXEMPT.get(pkg)
            if exempt_spec is not None and own_spec == exempt_spec:
                if verbose:
                    print(
                        f"EXEMPT: {pkg} [--all-features] reaches {TARGET_PKG}/{FORBIDDEN_FEATURE} "
                        f"only via its own by-name `{FORBIDDEN_FEATURE} = {own_spec}` passthrough "
                        f"— checking {pkg}'s own default/cuda selections instead"
                    )
                rc |= _check_exempt_member_real_lanes(graph, pkg, verbose)
                continue
            if verbose:
                print(
                    f"FAIL: {pkg} [--all-features] reaches {TARGET_PKG}/{FORBIDDEN_FEATURE} — "
                    f"a workspace member other than {ROOT} would compile the vendored "
                    f"FlashAttention-2 kernels",
                    file=sys.stderr,
                )
            rc = 1
    return rc


# --------------------------------------------------------------------------- #
# esc-081: proof surface == shipped surface.
#
# The manifest's `prove_lane.crates.<c>.kinds` DECLARES the exact
# `(crate, kind)` pairs `ci/scripts/runpod_gpu_prove.sh` must invoke; this
# section asserts SET EQUALITY between that declaration and the script's own
# `PROVE_TUPLE crate=<c> kind=<k> features=<literal>`-echoed invocations, and
# separately enforces the closed cuda-tuple scope map (PROVE_SCOPE /
# EXEMPT_SCOPE) over EVERY gated tuple `discover_all_tuples` finds anywhere
# under `ci/scripts/**`.
# --------------------------------------------------------------------------- #


def _scan_prove_tuple_pairs(path: Path) -> list[dict]:
    """Every `(PROVE_TUPLE echo, cargo invocation)` pairing in `path`: walks
    logical lines (the SAME comment-stripped, continuation-joined pipeline
    `discover_all_tuples` uses, so "what counts as an invocation line" can
    never silently diverge between the two), remembering the LAST-seen
    `PROVE_TUPLE` echo and pairing it with the very NEXT cargo invocation
    line. A cargo invocation with no unconsumed preceding echo pairs with
    `crate=None` (an UNLISTED pair)."""
    text = path.read_text(encoding="utf-8")
    pending: tuple[str, str, str] | None = None
    out: list[dict] = []
    for lineno, logical in _join_line_continuations(_drop_comment_lines(text)):
        stripped = logical.strip()
        m = _PROVE_TUPLE_RE.match(stripped)
        if m:
            pending = (m.group(1), m.group(2), m.group(3))
            continue
        for t in extract_tuples_from_line(logical):
            actual_m = re.search(r"(?:--features|-F)[=\s]+(\S+)", t)
            actual_features = actual_m.group(1).strip("\"'") if actual_m else ""
            out.append(
                {
                    "crate": pending[0] if pending else None,
                    "kind": pending[1] if pending else None,
                    "echoed_features": pending[2] if pending else None,
                    "actual_features": actual_features,
                    "tuple": t,
                    "lineno": lineno,
                }
            )
            pending = None
    return out


def check_prove_surface(manifest: dict, repo_root: Path = REPO_ROOT, verbose: bool = True) -> int:
    """Returns 0 (property holds) or 1 (a scope-map or set-equality
    violation)."""
    rc = 0
    registry = discover_all_tuples(repo_root)

    # --- scope map: every GATED tuple's every origin must be PROVE_SCOPE or
    # EXEMPT_SCOPE; an exempt entry with no gated tuple left is DEAD.
    exempt_hit: dict[str, bool] = {p: False for p in EXEMPT_SCOPE}
    for tuple_text, rec in registry.items():
        if not is_gated(tuple_text):
            continue
        for origin in rec.origins:
            origin_path = origin.rsplit(":", 1)[0]
            if origin_path in PROVE_SCOPE:
                continue
            if origin_path in EXEMPT_SCOPE:
                exempt_hit[origin_path] = True
                continue
            if verbose:
                print(
                    f"FAIL: {origin} carries a cuda-bearing tuple (`{tuple_text}`) that is "
                    f"neither in PROVE_SCOPE nor EXEMPT_SCOPE — exempt by decision, never by "
                    f"silence",
                    file=sys.stderr,
                )
            rc = 1
    for path_str, hit in exempt_hit.items():
        if not hit:
            if verbose:
                print(
                    f"FAIL: EXEMPT_SCOPE entry `{path_str}` no longer carries any cuda-bearing "
                    f"tuple — dead exempt entry, delete the row",
                    file=sys.stderr,
                )
            rc = 1

    # --- set equality: declared (crate, kind) pairs vs. the prove script's
    # own PROVE_TUPLE-paired invocations.
    declared = prove_surface.declared_pairs(manifest)
    if not declared:
        if verbose:
            print("FAIL: manifest has no `prove_lane.crates` pairs to prove", file=sys.stderr)
        return 1

    seen_pairs: dict[tuple[str, str], str] = {}
    for prove_file in sorted(PROVE_SCOPE):
        fpath = repo_root / prove_file
        if not fpath.is_file():
            if verbose:
                print(f"FAIL: PROVE_SCOPE names `{prove_file}`, which does not exist", file=sys.stderr)
            rc = 1
            continue
        for pair in _scan_prove_tuple_pairs(fpath):
            crate, kind = pair["crate"], pair["kind"]
            if crate is None:
                # Every invocation found in a PROVE_SCOPE file is a FINDING
                # when un-echoed — NOT only a gated (cuda/flash-attn) one.
                # An un-gated bare invocation (e.g. a second, unlisted
                # `cargo test -p jammi-kernels` with no `--features`) is
                # itself an EXTRA `default`-kind invocation of a
                # prove-scope crate that this script's set-equality rule
                # would otherwise never see (it is not "gated", so a
                # gated-only check silently waved it through) — the proof
                # surface can drift by count even when every DECLARED pair
                # still resolves once.
                if verbose:
                    print(
                        f"FAIL: {prove_file}:{pair['lineno']}: invocation `{pair['tuple']}` "
                        f"has no preceding PROVE_TUPLE echo — an unlisted invocation",
                        file=sys.stderr,
                    )
                rc = 1
                continue
            if pair["echoed_features"] != pair["actual_features"]:
                if verbose:
                    print(
                        f"FAIL: {prove_file}:{pair['lineno']}: PROVE_TUPLE echo says "
                        f"features={pair['echoed_features']!r} but the invocation's own "
                        f"--features is {pair['actual_features']!r} — echo/tuple disagree",
                        file=sys.stderr,
                    )
                rc = 1
            try:
                expected_feats = prove_surface.expected(crate, kind, manifest, repo_root)
            except ValueError as e:
                if verbose:
                    print(f"FAIL: {prove_file}:{pair['lineno']}: {e}", file=sys.stderr)
                rc = 1
                continue
            expected_text = prove_surface.feature_text(expected_feats)
            if pair["actual_features"] != expected_text:
                if verbose:
                    print(
                        f"FAIL: {prove_file}:{pair['lineno']}: ({crate}, {kind}) carries "
                        f"features={pair['actual_features']!r} but the manifest-declared "
                        f"expected surface is {expected_text!r}",
                        file=sys.stderr,
                    )
                rc = 1
            key = (crate, kind)
            if key in seen_pairs and seen_pairs[key] != pair["actual_features"]:
                if verbose:
                    print(
                        f"FAIL: {prove_file}:{pair['lineno']}: ({crate}, {kind}) appears with "
                        f"a DIFFERENT literal ({pair['actual_features']!r}) than an earlier "
                        f"invocation of the SAME pair ({seen_pairs[key]!r})",
                        file=sys.stderr,
                    )
                rc = 1
            seen_pairs[key] = pair["actual_features"]

    missing = declared - set(seen_pairs)
    extra = set(seen_pairs) - declared
    if missing:
        if verbose:
            print(f"FAIL: declared prove_lane pair(s) never invoked: {sorted(missing)}", file=sys.stderr)
        rc = 1
    if extra:
        if verbose:
            print(f"FAIL: invoked (crate, kind) pair(s) the manifest never declared: {sorted(extra)}", file=sys.stderr)
        rc = 1
    return rc


def _synthetic(ai_cuda: list[str], kernels_extra: dict | None = None) -> dict:
    kernels_features = {
        "default": [],
        "cuda": ["dep:bindgen_cuda"],
        "flash-attn": ["cuda"],
    }
    if kernels_extra:
        kernels_features.update(kernels_extra)
    return {
        "packages": [
            {
                "name": "jammi-server",
                "features": {"default": ["train"], "train": [], "cuda": ["jammi-ai/cuda"],
                             "jetstream-broker": [], "storage-cloud": []},
                "dependencies": [
                    {"name": "jammi-ai", "optional": False, "uses_default_features": True, "features": []},
                ],
            },
            {
                "name": "jammi-ai",
                "features": {"default": [], "cuda": ai_cuda},
                "dependencies": [
                    {"name": "jammi-kernels", "optional": False, "uses_default_features": True, "features": []},
                    {"name": "jammi-dev-only", "kind": "dev", "optional": False, "features": ["flash-attn"]},
                ],
            },
            {
                "name": "jammi-kernels",
                "features": kernels_features,
                "dependencies": [
                    {"name": "bindgen_cuda", "optional": True, "uses_default_features": True, "features": []},
                ],
            },
        ]
    }


def _default_lanes() -> dict[str, dict]:
    """A single clean synthetic lane matching the clean synthetic graph —
    used by self-test cases that don't care about lane checking itself."""
    return {
        "cu12-tarball": {
            "package": "jammi-server",
            "cargo_features": ["cuda"],
            "capabilities": {"flash_compiled": False},
        }
    }


def self_test() -> int:
    # Clean graph: cuda reaches kernels/cuda, never flash-attn.
    g = Graph(_synthetic(["jammi-kernels/cuda"]))
    assert verdict(g, _default_lanes(), verbose=False) == 0, "clean graph must pass"
    # Leaked edge in the middle crate.
    g = Graph(_synthetic(["jammi-kernels/cuda", "jammi-kernels/flash-attn"]))
    assert verdict(g, _default_lanes(), verbose=False) == 1, "leaked edge must fail"
    # Leak through the kernels crate's own `cuda` feature.
    g = Graph(_synthetic(["jammi-kernels/cuda"], {"cuda": ["dep:bindgen_cuda", "flash-attn"]}))
    assert verdict(g, _default_lanes(), verbose=False) == 1, "self-implication must fail"
    # Broken chain: the positive control must trip.
    g = Graph(_synthetic([]))
    assert verdict(g, _default_lanes(), verbose=False) == 1, "unreached control must fail"
    # Weak dep edge `dep?/feat` does not activate an inactive optional dep —
    # checked against ROOT's OWN selections directly (`closure`, not
    # `verdict`): `verdict` ALSO runs the widened per-member loop below, and
    # jammi-ai's OWN `--all-features` DOES activate "extra" (the pseudo-
    # feature every optional dependency gets) as part of that selection,
    # which correctly fires the weak edge — that is the NEXT assertion.
    meta = _synthetic(["jammi-kernels/cuda"])
    meta["packages"][1]["features"]["cuda"] = ["jammi-kernels/cuda", "extra?/flash-attn"]
    meta["packages"][1]["dependencies"].append(
        {"name": "jammi-kernels", "rename": "extra", "optional": True, "uses_default_features": True, "features": []}
    )
    g = Graph(meta)
    deferred.clear()
    enabled = g.closure(ROOT, [CONTROL_FEATURE])
    assert FORBIDDEN_FEATURE not in enabled.get(TARGET_PKG, set()), (
        "weak edge on an inactive optional dep must not leak from ROOT's own cuda selection"
    )
    # The per-member loop catches what ROOT's own selections cannot see:
    # `jammi-ai --all-features` activates "extra" directly,
    # which makes `extra?/flash-attn` fire — a genuine leak under
    # `cargo build -p jammi-ai --all-features` that a jammi-server-only
    # closure walk would miss entirely.
    assert verdict(g, _default_lanes(), verbose=False) == 1, (
        "the widened per-member --all-features loop must catch the leak "
        "jammi-server's own selections cannot see"
    )

    # Order-dependence control for the weak-edge fixpoint: a deferred
    # `dep?/feat` edge whose optional dep is activated by the LAST work item,
    # on a path that ends in `continue` (an implicit optional-dependency
    # feature: `extra` named bare, with no `[features]` entry of its own).
    # `jammi-ai`'s spec order puts `extra` on the BOTTOM of the LIFO work
    # stack, so it is popped after every other item, and no package in this
    # graph declares `default`, so nothing is queued behind it. Cargo
    # activates `extra` here, so `extra?/flash-attn` MUST fire and ROOT's
    # plain `cuda` selection genuinely reaches jammi-kernels/flash-attn.
    # A walker that re-scans deferred edges only at the bottom of the item
    # loop never re-scans after that final `continue` and reports a
    # flash-free closure — an order-dependent FALSE PASS on a real leak.
    meta = _synthetic(["jammi-kernels/cuda"])
    del meta["packages"][1]["features"]["default"]
    del meta["packages"][2]["features"]["default"]
    meta["packages"][1]["features"]["cuda"] = [
        "extra", "jammi-kernels/cuda", "extra?/flash-attn",
    ]
    meta["packages"][1]["dependencies"].append(
        {"name": "jammi-kernels", "rename": "extra", "optional": True,
         "uses_default_features": False, "features": []}
    )
    g = Graph(meta)
    deferred.clear()
    enabled = g.closure(ROOT, [CONTROL_FEATURE])
    assert FORBIDDEN_FEATURE in enabled.get(TARGET_PKG, set()), (
        "a weak edge whose optional dep is activated by the final work item "
        "(an implicit optional-dependency feature, a `continue` path) must "
        "still fire — the deferred re-scan is a fixpoint outside the item "
        "loop, not a per-item scan the `continue` paths skip"
    )
    assert not deferred, "every deferred weak edge must be resolved on exit"

    # --- Manifest-derived per-lane checks, including the synthetic
    # `flash_compiled: false` lane the negative-polarity branch needs to be
    # provably alive (all three REAL lanes today are `true`; without this
    # case the `false` branch would have zero instances). These target
    # `jammi-ai` directly (not `jammi-server`/ROOT) with its OWN declared
    # `flash-attn` feature so the lane check is exercised independently of
    # ROOT's `cuda`-alone invariant (a different, always-on check) — the
    # spec mirrors the real `jammi-ai` Cargo.toml entry exactly so it also
    # matches the module-level `ALL_FEATURES_FLASH_EXEMPT["jammi-ai"]`
    # value, keeping the rest of verdict()'s checks clean for these cases.
    meta = _synthetic(["jammi-kernels/cuda"])
    meta["packages"][1]["features"]["flash-attn"] = [
        "cuda", "jammi-encoders/flash-attn", f"{TARGET_PKG}/{FORBIDDEN_FEATURE}",
    ]
    g = Graph(meta)
    lanes_true_pass = {
        "cu12-tarball": {
            "package": "jammi-ai",
            "cargo_features": ["flash-attn"],
            "capabilities": {"flash_compiled": True},
        }
    }
    assert verdict(g, lanes_true_pass, verbose=False) == 0, (
        "a lane declaring flash_compiled=true whose cargo_features genuinely "
        "reach jammi-kernels/flash-attn must pass"
    )
    lanes_false_fail = {
        "cu12-tarball": {
            "package": "jammi-ai",
            "cargo_features": ["flash-attn"],
            "capabilities": {"flash_compiled": False},
        }
    }
    assert verdict(g, lanes_false_fail, verbose=False) == 1, (
        "a lane declaring flash_compiled=false whose cargo_features reach "
        "jammi-kernels/flash-attn anyway (an undeclared leak) must FAIL — "
        "the synthetic false-lane negative-polarity case"
    )
    # Name-only passthrough: the lane DECLARES flash_compiled=true but the
    # closure fails to actually enable jammi-kernels/flash-attn (a broken
    # forwarding chain, e.g. a rename that silently stopped propagating —
    # `flash-attn` only forwards `cuda`, never `jammi-kernels/flash-attn`).
    meta_broken = _synthetic(["jammi-kernels/cuda"])
    meta_broken["packages"][1]["features"]["flash-attn"] = ["cuda"]
    g_broken = Graph(meta_broken)
    lanes_true_but_broken = {
        "cu12-tarball": {
            "package": "jammi-ai",
            "cargo_features": ["flash-attn"],
            "capabilities": {"flash_compiled": True},
        }
    }
    assert verdict(g_broken, lanes_true_but_broken, verbose=False) == 1, (
        "a lane declaring flash_compiled=true whose closure does NOT reach "
        "jammi-kernels/flash-attn (name-only passthrough) must FAIL"
    )
    # Empty/missing lanes must be treated as an error by the caller
    # (load_manifest_lanes), not silently accepted by verdict(); verdict()
    # itself just iterates whatever dict it's given, so this is asserted at
    # the load_manifest_lanes() level below instead.
    assert load_manifest_lanes_rejects_empty(), "empty `lanes` must be rejected"

    # The ALL_FEATURES_FLASH_EXEMPT mechanism (P6 Stage B): a member that
    # declares its OWN by-name `flash-attn` passthrough must pass under
    # `--all-features` (the exemption fires) but still FAIL if the SAME
    # feature leaks through a real lane (`cuda`/`default`), and must NOT
    # be exempted if its `flash-attn` spec doesn't match exactly.
    saved_exempt = dict(ALL_FEATURES_FLASH_EXEMPT)
    try:
        ALL_FEATURES_FLASH_EXEMPT.clear()
        ALL_FEATURES_FLASH_EXEMPT["jammi-ai"] = [f"{TARGET_PKG}/{FORBIDDEN_FEATURE}"]

        # Exempt member, clean cuda/default: --all-features leaks only via
        # the declared passthrough -> overall verdict must be clean.
        meta = _synthetic(["jammi-kernels/cuda"])
        meta["packages"][1]["features"]["flash-attn"] = [f"{TARGET_PKG}/{FORBIDDEN_FEATURE}"]
        g = Graph(meta)
        assert verdict(g, _default_lanes(), verbose=False) == 0, (
            "a member with a verified 1:1 flash-attn passthrough must pass "
            "once exempted, provided its own cuda/default selections stay clean"
        )

        # Same exempt member, but `cuda` ALSO reaches flash-attn directly —
        # the exemption must not launder a leak through a REAL build lane.
        meta = _synthetic(["jammi-kernels/cuda", f"{TARGET_PKG}/{FORBIDDEN_FEATURE}"])
        meta["packages"][1]["features"]["flash-attn"] = [f"{TARGET_PKG}/{FORBIDDEN_FEATURE}"]
        g = Graph(meta)
        assert verdict(g, _default_lanes(), verbose=False) == 1, (
            "the exemption covers --all-features only; a leak via the member's "
            "own cuda selection must still fail"
        )

        # Same exempt member, but its `flash-attn` feature pulls something
        # EXTRA beyond the declared 1:1 passthrough — the spec no longer
        # matches ALL_FEATURES_FLASH_EXEMPT's value, so it must NOT be
        # silently exempted (a broader leak must read as an ordinary FAIL).
        meta = _synthetic(["jammi-kernels/cuda"])
        meta["packages"][1]["features"]["flash-attn"] = [
            f"{TARGET_PKG}/{FORBIDDEN_FEATURE}",
            "some-other-feature",
        ]
        g = Graph(meta)
        assert verdict(g, _default_lanes(), verbose=False) == 1, (
            "a flash-attn spec that does not match the exemption exactly "
            "must not be silently exempted"
        )
    finally:
        ALL_FEATURES_FLASH_EXEMPT.clear()
        ALL_FEATURES_FLASH_EXEMPT.update(saved_exempt)

    # --- ROOT's own --all-features exemption (step 5): jammi-server's
    # `flash-attn = ["jammi-ai/flash-attn"]` passthrough must be accepted,
    # but only when ROOT's own default/cuda selections stay flash-free, and
    # only when the spec matches EXACTLY. `jammi-ai`'s own `flash-attn`
    # spec is set to match `ALL_FEATURES_FLASH_EXEMPT["jammi-ai"]` exactly
    # (mirroring the real Cargo.toml) so the widened per-member loop's own
    # check of `jammi-ai [--all-features]` is independently clean and these
    # cases isolate ROOT-level behavior only. ---
    ai_flash_spec = ["cuda", "jammi-encoders/flash-attn", f"{TARGET_PKG}/{FORBIDDEN_FEATURE}"]
    meta = _synthetic(["jammi-kernels/cuda"])
    meta["packages"][0]["features"]["flash-attn"] = ["jammi-ai/flash-attn"]
    meta["packages"][1]["features"]["flash-attn"] = ai_flash_spec
    g = Graph(meta)
    assert verdict(g, _default_lanes(), verbose=False) == 0, (
        "ROOT's own verified 1:1 flash-attn passthrough must be exempted under "
        "--all-features, provided ROOT's own default/cuda selections stay clean"
    )
    # Same shape, but ROOT's `cuda` selection ALSO reaches flash-attn
    # directly (e.g. someone folded the passthrough into `cuda` by
    # mistake) — must still FAIL.
    meta = _synthetic(["jammi-kernels/cuda"])
    meta["packages"][0]["features"]["cuda"] = ["jammi-ai/cuda", "jammi-ai/flash-attn"]
    meta["packages"][0]["features"]["flash-attn"] = ["jammi-ai/flash-attn"]
    meta["packages"][1]["features"]["flash-attn"] = ai_flash_spec
    g = Graph(meta)
    assert verdict(g, _default_lanes(), verbose=False) == 1, (
        "ROOT's --all-features exemption must not launder a leak through "
        "ROOT's own real cuda selection"
    )
    # ROOT's flash-attn spec doesn't match the exempted 1:1 form exactly —
    # must not be silently exempted.
    meta = _synthetic(["jammi-kernels/cuda"])
    meta["packages"][0]["features"]["flash-attn"] = ["jammi-ai/flash-attn", "some-other-feature"]
    meta["packages"][1]["features"]["flash-attn"] = ai_flash_spec
    g = Graph(meta)
    assert verdict(g, _default_lanes(), verbose=False) == 1, (
        "a ROOT flash-attn spec that does not match the exemption exactly "
        "must not be silently exempted"
    )

    _self_test_prove_surface()

    print("self-test: ok")
    return 0


# --------------------------------------------------------------------------- #
# esc-081 self-test (F5): a `git init`'d ephemeral fixture repo -- NEVER this
# checkout -- carrying the real jammi-{server,ai,bench,kernels} `[features]`
# tables (so `prove_surface.declared()` reads real shapes) plus a minimal
# `runpod_gpu_prove.sh` twin covering the six declared pairs, and every
# EXEMPT_SCOPE path (each with one cuda-bearing tuple, so a clean fixture
# never trips the dead-exempt-entry rule).
# --------------------------------------------------------------------------- #

_FIXTURE_CRATE_FEATURES = {
    "jammi-server": ["cuda", "flash-attn", "jetstream-broker", "storage-cloud", "live-gpu-tests", "train"],
    "jammi-ai": ["cuda", "flash-attn", "live-gpu-tests"],
    "jammi-bench": ["cuda", "flash-attn"],
    "jammi-kernels": ["cuda", "flash-attn", "default"],
}

_FIXTURE_MANIFEST = {
    "lanes": {
        "cu12-tarball": {
            "cargo_features": ["cuda", "flash-attn", "jetstream-broker", "storage-cloud"],
        }
    },
    "server_only_cargo_features": {"features": ["jetstream-broker", "storage-cloud"]},
    "prove_lane": {
        "crates": {
            "jammi-server": {"kinds": ["release", "test"], "prove_only": ["live-gpu-tests"]},
            "jammi-ai": {"kinds": ["test"], "prove_only": ["live-gpu-tests"]},
            "jammi-bench": {"kinds": ["release"], "prove_only": []},
            "jammi-kernels": {"kinds": ["default", "test"], "prove_only": []},
        }
    },
}


def _fixture_good_prove_script() -> str:
    lines = [
        "#!/usr/bin/env bash",
        'echo "PROVE_TUPLE crate=jammi-server kind=release features=cuda,flash-attn,jetstream-broker,storage-cloud"',
        "cargo build --release -p jammi-server --bin jammi-server --features cuda,flash-attn,jetstream-broker,storage-cloud",
        'echo "PROVE_TUPLE crate=jammi-ai kind=test features=cuda,flash-attn,live-gpu-tests"',
        "cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --test gpu_capability --no-run",
        'echo "PROVE_TUPLE crate=jammi-server kind=test features=cuda,flash-attn,jetstream-broker,live-gpu-tests,storage-cloud"',
        "cargo test -p jammi-server --features cuda,flash-attn,jetstream-broker,live-gpu-tests,storage-cloud --test it grpc_embedding_gpu -- --nocapture",
        'echo "PROVE_TUPLE crate=jammi-ai kind=test features=cuda,flash-attn,live-gpu-tests"',
        "cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --test gpu_capability -- --nocapture --skip capability_surface",
        'echo "PROVE_TUPLE crate=jammi-kernels kind=default features="',
        "cargo test -p jammi-kernels -- --nocapture",
        'echo "PROVE_TUPLE crate=jammi-kernels kind=test features=cuda,flash-attn"',
        "cargo test -p jammi-kernels --features cuda,flash-attn -- --nocapture",
        'echo "PROVE_TUPLE crate=jammi-bench kind=release features=cuda,flash-attn"',
        "cargo run -p jammi-bench --release --features cuda,flash-attn -- gpu-inference-scale",
    ]
    return "\n".join(lines) + "\n"


def _write_prove_surface_fixture(root: Path, script_body: str, manifest: dict | None = None) -> None:
    manifest = manifest if manifest is not None else json.loads(json.dumps(_FIXTURE_MANIFEST))
    (root / "ci" / "scripts" / "perf").mkdir(parents=True, exist_ok=True)
    (root / "ci" / "release-feature-manifest.json").write_text(json.dumps(manifest))
    (root / "ci" / "scripts" / "runpod_gpu_prove.sh").write_text(script_body)
    (root / "ci" / "scripts" / "pod_seed_target.sh").write_text(
        "#!/usr/bin/env bash\ncargo build --release -p jammi-bench --features cuda\n"
    )
    for perf_name in (
        "finetune_ab.sh",
        "finetune_run_ab.sh",
        "encode_ab.sh",
        "gpu_inference_ab.sh",
        "fa2_ab.sh",
        "clip_artifact_producer.sh",
        "pod_build_timings.sh",
        "stacked_sweep.sh",
    ):
        (root / "ci" / "scripts" / "perf" / perf_name).write_text(
            "#!/usr/bin/env bash\ncargo build --release -p jammi-bench --features cuda\n"
        )
    for crate, feats in _FIXTURE_CRATE_FEATURES.items():
        crate_dir = root / "crates" / crate
        crate_dir.mkdir(parents=True, exist_ok=True)
        feat_lines = "\n".join(f'{f} = []' for f in feats)
        (crate_dir / "Cargo.toml").write_text(f"[package]\nname = \"{crate}\"\nversion = \"0.1.0\"\n\n[features]\n{feat_lines}\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)


def _run_prove_surface_fixture(script_body: str, manifest: dict | None = None) -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_prove_surface_fixture(root, script_body, manifest)
        m = manifest if manifest is not None else _FIXTURE_MANIFEST
        return check_prove_surface(m, root, verbose=False)


def _self_test_prove_surface() -> None:
    good = _fixture_good_prove_script()

    assert _run_prove_surface_fixture(good) == 0, "a well-formed fixture must be green"

    # Lane minus flash-attn changes the verdict: the manifest's own lane no
    # longer carries flash-attn, so every pair's expected surface shrinks --
    # the UNCHANGED script now disagrees with the manifest.
    m = json.loads(json.dumps(_FIXTURE_MANIFEST))
    m["lanes"]["cu12-tarball"]["cargo_features"] = ["cuda", "jetstream-broker", "storage-cloud"]
    assert _run_prove_surface_fixture(good, m) == 1, "lane minus flash-attn must change the verdict"

    # Reverted literal: jammi-ai test invocation (both echo AND actual
    # --features, kept mutually consistent) reverted to the pre-esc-081
    # `cuda,live-gpu-tests` shape -- disagrees with the manifest-declared
    # expected surface (cuda,flash-attn,live-gpu-tests).
    reverted = good.replace(
        'echo "PROVE_TUPLE crate=jammi-ai kind=test features=cuda,flash-attn,live-gpu-tests"\n'
        "cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --test gpu_capability --no-run",
        'echo "PROVE_TUPLE crate=jammi-ai kind=test features=cuda,live-gpu-tests"\n'
        "cargo test -p jammi-ai --features cuda,live-gpu-tests --test gpu_capability --no-run",
    )
    assert _run_prove_surface_fixture(reverted) == 1, "a reverted literal must FAIL"

    # Echo/tuple disagree: the echo claims one literal, the invocation
    # itself carries another.
    disagree = good.replace(
        'echo "PROVE_TUPLE crate=jammi-kernels kind=test features=cuda,flash-attn"\n'
        "cargo test -p jammi-kernels --features cuda,flash-attn -- --nocapture",
        'echo "PROVE_TUPLE crate=jammi-kernels kind=test features=cuda"\n'
        "cargo test -p jammi-kernels --features cuda,flash-attn -- --nocapture",
    )
    assert _run_prove_surface_fixture(disagree) == 1, "echo/tuple disagreement must FAIL"

    # New bare cargo line: an extra cuda-bearing invocation with no
    # preceding PROVE_TUPLE echo at all -- an unlisted pair.
    unlisted = good + "cargo test -p jammi-ai --features cuda -- --nocapture\n"
    assert _run_prove_surface_fixture(unlisted) == 1, "a new bare unechoed cuda invocation must FAIL"

    # An un-echoed, UN-GATED bare invocation naming a prove-scope crate is
    # ALSO a FINDING -- a second, unlisted `cargo test -p jammi-kernels`
    # (no --features, so `is_gated` alone would never flag it) is an extra
    # `default`-kind invocation this script's own set-equality rule cannot
    # otherwise see.
    unlisted_ungated = good + "cargo test -p jammi-kernels -- --nocapture\n"
    assert _run_prove_surface_fixture(unlisted_ungated) == 1, (
        "an un-echoed, un-gated cargo invocation naming a prove-scope crate must FAIL too"
    )

    # Emptied manifest: prove_lane.crates has nothing to prove.
    m2 = json.loads(json.dumps(_FIXTURE_MANIFEST))
    m2["prove_lane"]["crates"] = {}
    assert _run_prove_surface_fixture(good, m2) == 1, "an emptied prove_lane.crates must FAIL"

    # Missing declared pair: drop the kernels-default invocation entirely --
    # the manifest still declares it.
    missing_pair = good.replace(
        'echo "PROVE_TUPLE crate=jammi-kernels kind=default features="\n'
        "cargo test -p jammi-kernels -- --nocapture\n",
        "",
    )
    assert _run_prove_surface_fixture(missing_pair) == 1, "a declared pair the script never invokes must FAIL"

    # Unlisted cuda-bearing script: a NEW ci/scripts/foo.sh (outside both
    # PROVE_SCOPE and EXEMPT_SCOPE) carries a gated tuple.
    with_stray_script = good  # base script unchanged; extra file added below

    def _with_extra_file(rel: str, body: str):
        def _augmented(root: Path, script_body: str, manifest):
            _write_prove_surface_fixture(root, script_body, manifest)
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)
            subprocess.run(["git", "add", "-A"], cwd=root, check=True)

        return _augmented

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _with_extra_file("ci/scripts/foo.sh", "#!/usr/bin/env bash\ncargo test -p jammi-ai --features cuda\n")(
            root, with_stray_script, None
        )
        rc = check_prove_surface(_FIXTURE_MANIFEST, root, verbose=False)
        assert rc == 1, "a cuda tuple in an unlisted ci/scripts/foo.sh must FAIL"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _with_extra_file(
            "ci/scripts/perf/bar.sh", "#!/usr/bin/env bash\ncargo test -p jammi-ai --features cuda\n"
        )(root, with_stray_script, None)
        rc = check_prove_surface(_FIXTURE_MANIFEST, root, verbose=False)
        assert rc == 1, "a cuda tuple in an unlisted ci/scripts/perf/bar.sh must FAIL (never silently exempt)"

    # Dead exempt entry: pod_seed_target.sh loses its own cuda tuple.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_prove_surface_fixture(root, good, None)
        (root / "ci" / "scripts" / "pod_seed_target.sh").write_text("#!/usr/bin/env bash\necho hi\n")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        rc = check_prove_surface(_FIXTURE_MANIFEST, root, verbose=False)
        assert rc == 1, "a dead EXEMPT_SCOPE entry (no cuda tuple left) must FAIL"


def load_manifest_lanes_rejects_empty() -> bool:
    """Exercise `load_manifest_lanes`'s fail-closed behavior on a missing/
    empty `lanes` key without touching the real manifest file on disk."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        bad_manifest = Path(td) / "release-feature-manifest.json"
        bad_manifest.write_text(json.dumps({"lanes": {}}))
        global MANIFEST_PATH
        saved = MANIFEST_PATH
        MANIFEST_PATH = bad_manifest
        try:
            load_manifest_lanes()
            return False  # should have exited
        except SystemExit as e:
            return e.code == 2
        finally:
            MANIFEST_PATH = saved


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    graph = Graph(load_metadata())
    for needed in (ROOT, TARGET_PKG):
        if needed not in graph.pkgs:
            print(f"ERROR: workspace has no package `{needed}`", file=sys.stderr)
            return 2
    if FORBIDDEN_FEATURE not in graph.pkgs[TARGET_PKG]["features"]:
        print(
            f"ERROR: {TARGET_PKG} declares no `{FORBIDDEN_FEATURE}` feature — "
            f"nothing to guard (update this script if the feature was renamed)",
            file=sys.stderr,
        )
        return 2
    lanes = load_manifest_lanes()
    rc = verdict(graph, lanes)
    full_manifest = prove_surface.load_manifest(MANIFEST_PATH)
    rc |= check_prove_surface(full_manifest, REPO_ROOT)
    print("check_flash_attn_closure: " + ("PASS" if rc == 0 else "FAIL"))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
