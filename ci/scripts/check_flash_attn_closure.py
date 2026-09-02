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
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "ci" / "release-feature-manifest.json"

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

    print("self-test: ok")
    return 0


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
    print("check_flash_attn_closure: " + ("PASS" if rc == 0 else "FAIL"))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
