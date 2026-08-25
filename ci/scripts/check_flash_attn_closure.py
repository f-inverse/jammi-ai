#!/usr/bin/env python3
"""Assert no workspace member's feature closure reaches
`jammi-kernels/flash-attn` except through the feature itself.

`jammi-kernels`'s `flash-attn` feature builds the vendored FlashAttention-2
kernels (CUTLASS submodule + a minute of `nvcc` + a static archive). It
DEPENDS on `cuda` but must never be IMPLIED by it: `.github/workflows/
release-binaries.yml` builds `jammi-server --features cuda,...` and that lane
must stay CUTLASS-free. Cargo has no "never enable this feature" primitive,
so this script walks the workspace's feature graph the way the resolver
does and proves the property mechanically.

Method (hermetic: `cargo metadata --no-deps`, no network, no build):

  1. Load every workspace package's `[features]` table and dependency list.
  2. For `jammi-server` specifically (`ROOT`) — the package
     `release-binaries.yml` actually builds — check three selections:
     `default`, the release lane's `cuda,jetstream-broker,storage-cloud`,
     and `jammi-server --all-features`.
  3. WIDEN beyond `jammi-server`: for EVERY OTHER workspace member (a leak
     through `jammi-bench` or `jammi-python`, both of which reach
     `jammi-ai` → ... → `jammi-kernels` transitively, would not be visible
     from `jammi-server`'s closure alone), check THAT member's own
     `--all-features` selection. `jammi-kernels` itself is excluded from
     this loop — asking "does building jammi-kernels with all its own
     features enable jammi-kernels/flash-attn" is vacuously true and not
     the property this script guards (a crate is not its own consumer).
  4. Propagation uses the resolver-v2 rules for workspace members: a plain
     `feat` enables it on the same package; `dep:name` activates an
     optional dependency (with its declared `features` and, unless
     `default-features = false`, `default`); `name/feat` activates `name`
     and enables `feat` on it; `name?/feat` enables `feat` on `name` only if
     `name` is already active. Non-optional normal/build dependencies of an
     active package are active. Dev dependencies are ignored (they are not
     part of a binary's closure). Non-workspace packages are opaque (they
     cannot enable a workspace member's feature).
  5. FAIL if `flash-attn` is enabled on `jammi-kernels` under any selection
     in steps 2 or 3. Also FAIL the positive control if `jammi-server`'s
     cuda-lane selection does NOT enable `jammi-kernels/cuda` — that proves
     the walk actually traverses the server → ai → encoders/lora → kernels
     chain rather than trivially seeing an empty set.

`--self-test` runs the walker over synthetic metadata: a leaked edge
(`ai: cuda = ["jammi-kernels/flash-attn"]`), a weak-dep edge, and a clean
graph, asserting each verdict.

Run: `python3 ci/scripts/check_flash_attn_closure.py`
Exit 0 = property holds; 1 = a leak (or a broken control); 2 = usage/metadata error.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ROOT = "jammi-server"
TARGET_PKG = "jammi-kernels"
FORBIDDEN_FEATURE = "flash-attn"
# The lane that must stay CUTLASS-free (release-binaries.yml's cuda build).
CUDA_LANE = ["cuda", "jetstream-broker", "storage-cloud"]
# Positive control: this feature MUST be reachable from the cuda lane.
CONTROL_FEATURE = "cuda"


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
                        # feature must follow. Re-queue via a sentinel scan
                        # at the end (simple fixpoint below).
                        deferred.append((pkg, d, sub))
                else:
                    work.append((pkg, spec))
            # Weak edges whose dep became active since they were seen.
            still: list[tuple[str, dict, str]] = []
            for (p2, d2, sub2) in deferred:
                if d2["key"] in active_optional[p2]:
                    work.append((d2["pkg"], sub2))
                else:
                    still.append((p2, d2, sub2))
            deferred[:] = still
        return enabled

    def all_features(self, pkg: str) -> list[str]:
        feats = set(self.pkgs[pkg]["features"].keys())
        for d in self.pkgs[pkg]["deps"]:
            if d["optional"]:
                feats.add(d["key"])
        return sorted(feats)


deferred: list[tuple[str, dict, str]] = []


def verdict(graph: Graph, verbose: bool = True) -> int:
    rc = 0
    selections = {
        "default": ["default"] if "default" in graph.pkgs[ROOT]["features"] else [],
        "cuda lane (" + ",".join(CUDA_LANE) + ")": CUDA_LANE,
        "--all-features": graph.all_features(ROOT),
    }
    for label, feats in selections.items():
        deferred.clear()
        enabled = graph.closure(ROOT, feats)
        kernels = sorted(enabled.get(TARGET_PKG, set()))
        if verbose:
            print(f"{ROOT} [{label}] -> {TARGET_PKG} features: {kernels}")
        if FORBIDDEN_FEATURE in kernels:
            if verbose:
                print(
                    f"FAIL: {ROOT} [{label}] reaches {TARGET_PKG}/{FORBIDDEN_FEATURE} — "
                    f"the release lane would compile the vendored FlashAttention-2 kernels",
                    file=sys.stderr,
                )
            rc = 1
        if label.startswith("cuda lane") and CONTROL_FEATURE not in kernels:
            if verbose:
                print(
                    f"FAIL (positive control): {ROOT} [{label}] does not reach "
                    f"{TARGET_PKG}/{CONTROL_FEATURE}; the walk is not traversing the "
                    f"consumer -> kernels chain, so a 'clean' verdict would be vacuous",
                    file=sys.stderr,
                )
            rc = 1

    # Widen beyond ROOT: every OTHER workspace member's OWN --all-features
    # selection (jammi-kernels itself excluded — see the module docstring).
    for pkg in sorted(graph.pkgs):
        if pkg in (ROOT, TARGET_PKG):
            continue
        deferred.clear()
        enabled = graph.closure(pkg, graph.all_features(pkg))
        kernels = sorted(enabled.get(TARGET_PKG, set()))
        if verbose:
            print(f"{pkg} [--all-features] -> {TARGET_PKG} features: {kernels}")
        if FORBIDDEN_FEATURE in kernels:
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


def self_test() -> int:
    # Clean graph: cuda reaches kernels/cuda, never flash-attn.
    g = Graph(_synthetic(["jammi-kernels/cuda"]))
    assert verdict(g, verbose=False) == 0, "clean graph must pass"
    # Leaked edge in the middle crate.
    g = Graph(_synthetic(["jammi-kernels/cuda", "jammi-kernels/flash-attn"]))
    assert verdict(g, verbose=False) == 1, "leaked edge must fail"
    # Leak through the kernels crate's own `cuda` feature.
    g = Graph(_synthetic(["jammi-kernels/cuda"], {"cuda": ["dep:bindgen_cuda", "flash-attn"]}))
    assert verdict(g, verbose=False) == 1, "self-implication must fail"
    # Broken chain: the positive control must trip.
    g = Graph(_synthetic([]))
    assert verdict(g, verbose=False) == 1, "unreached control must fail"
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
    enabled = g.closure(ROOT, CUDA_LANE)
    assert FORBIDDEN_FEATURE not in enabled.get(TARGET_PKG, set()), (
        "weak edge on an inactive optional dep must not leak from ROOT's own cuda-lane selection"
    )
    # The WIDENED per-member loop (F6) catches what ROOT's own selections
    # cannot see: `jammi-ai --all-features` activates "extra" directly,
    # which makes `extra?/flash-attn` fire — a genuine leak under
    # `cargo build -p jammi-ai --all-features` that a jammi-server-only
    # closure walk would miss entirely.
    assert verdict(g, verbose=False) == 1, (
        "the widened per-member --all-features loop must catch the leak "
        "jammi-server's own selections cannot see"
    )
    print("self-test: ok")
    return 0


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
    rc = verdict(graph)
    print("check_flash_attn_closure: " + ("PASS" if rc == 0 else "FAIL"))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
