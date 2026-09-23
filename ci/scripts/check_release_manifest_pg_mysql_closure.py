#!/usr/bin/env python3
# needs: cargo-registry
"""Assert no release lane's feature selection reaches jammi-db's postgres/
mysql features.

**Guarded property**: `jammi-db`'s `postgres`/`mysql` features pull
`datafusion-table-providers`'s federation drivers, which link a native TLS
stack (`native-tls` -> OpenSSL) unconditionally at every published version.
Building either feature requires OpenSSL's development headers on the build
host. This repo's release images and CI runner image deliberately omit that
package (the runtime image is distroless Debian 12, which ships only the 3
series; the CI/build image resolves to the 1.1 series) — a build produced
under the current images with these features enabled would link successfully
and then fail to LOAD at runtime, a mismatch no build-time gate can see (see
`crates/jammi-db/README.md`'s own "Build requirements" section).
No release lane enables these features, and this gate keeps that a
mechanically-checked property: it FAILS the moment any `ci/release-feature-
manifest.json` lane's `cargo_features` selection reaches either feature,
which is exactly the day the OpenSSL/runtime-image mismatch above would
start mattering for real.

Method (hermetic: `cargo metadata --no-deps`, no network, no build) —
mirrors `check_flash_attn_closure.py`'s own shape and REUSES its `Graph`
feature-closure walker (never a second, independently-drifting graph
implementation of the identical resolver semantics): for EVERY manifest
lane, walk its declared `package`/`cargo_features` selection through the
real workspace feature graph and assert the resulting `jammi-db` feature
set contains neither `postgres` nor `mysql`. FAILS on a missing/unreadable
manifest, a missing/renamed `lanes` key, or an empty lane list (the same
manifest-read closure discipline `check_flash_attn_closure.py` already
holds) — this cannot pass vacuously on a broken or absent manifest.

Run: `python3 ci/scripts/check_release_manifest_pg_mysql_closure.py`
Self-test: `python3 ci/scripts/check_release_manifest_pg_mysql_closure.py --self-test`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_flash_attn_closure as flash  # noqa: E402 — reuses Graph/load_metadata/load_manifest_lanes

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_PKG = "jammi-db"
FORBIDDEN_FEATURES = ("postgres", "mysql")


def reaches_forbidden(graph: "flash.Graph", pkg: str, feats: list[str]) -> tuple[set[str], list[str]]:
    """(forbidden_features_reached, jammi-db's own full enabled-feature
    list) for walking `pkg`'s `feats` selection through `graph`. Clears
    `flash.deferred` first — the SAME module-global the imported `Graph.
    closure`'s weak-edge (`dep?/feat`) fixed point reads/mutates, and the
    SAME reset `check_flash_attn_closure._reaches_flash` performs before
    every call of its own; skipping it would leak state between calls,
    since `closure()` is called once per manifest lane here."""
    flash.deferred.clear()
    enabled = graph.closure(pkg, feats)
    db_features = sorted(enabled.get(TARGET_PKG, set()))
    return {f for f in FORBIDDEN_FEATURES if f in db_features}, db_features


def verdict(graph: "flash.Graph", lanes: dict[str, dict], verbose: bool = True) -> int:
    rc = 0
    for lane_name, lane in lanes.items():
        pkg = lane["package"]
        feats = lane["cargo_features"]
        if pkg not in graph.pkgs:
            if verbose:
                print(
                    f"FAIL: lane `{lane_name}` names package `{pkg}`, which is not a "
                    "workspace member",
                    file=sys.stderr,
                )
            rc = 1
            continue
        reached, db_features = reaches_forbidden(graph, pkg, feats)
        if verbose:
            print(
                f"lane `{lane_name}` ({pkg} [{','.join(feats)}]) -> {TARGET_PKG} features: "
                f"{db_features}"
            )
        if reached:
            if verbose:
                print(
                    f"FAIL: lane `{lane_name}` reaches {TARGET_PKG}/{sorted(reached)} — this "
                    "lane's build would require OpenSSL development headers this repo's "
                    "release/CI images deliberately omit, and the runtime image cannot load "
                    "the native TLS stack these features link (see crates/jammi-db/README.md); "
                    "either this is a deliberate new release surface that needs OpenSSL parity "
                    "between the build and runtime images, or the lane's own cargo_features "
                    "leaked this feature unintentionally",
                    file=sys.stderr,
                )
            rc = 1
    if rc == 0 and verbose:
        print(
            f"OK: no manifest lane's cargo_features selection reaches {TARGET_PKG}/"
            f"{{{','.join(FORBIDDEN_FEATURES)}}}."
        )
    return rc


def _synthetic_metadata(db_deps_on_forbidden: bool) -> dict:
    """A minimal two-package synthetic workspace: `root` depends on
    `jammi-db` (optionally forwarding `postgres`), used by the self-test
    so it never depends on this repo's own real Cargo.toml shape drifting
    underneath it."""
    jammi_db_pkg = {
        "name": "jammi-db",
        "features": {"postgres": [], "mysql": [], "default": []},
        "dependencies": [],
    }
    root_features: dict[str, list[str]] = {"default": [], "cuda": []}
    if db_deps_on_forbidden:
        root_features["with-postgres"] = ["jammi-db/postgres"]
    root_pkg = {
        "name": "root",
        "features": root_features,
        "dependencies": [
            {"name": "jammi-db", "kind": None, "optional": False, "uses_default_features": True, "features": []}
        ],
    }
    return {"packages": [root_pkg, jammi_db_pkg]}


def self_test() -> int:
    failures: list[str] = []

    def check(label: str, cond: bool, detail: object = "") -> None:
        if not cond:
            failures.append(f"{label}: {detail}")

    # Positive control: a lane selecting a feature that forwards to
    # jammi-db/postgres must be caught.
    graph = flash.Graph(_synthetic_metadata(db_deps_on_forbidden=True))
    lanes = {"leaky": {"package": "root", "cargo_features": ["with-postgres"], "capabilities": {"flash_compiled": False}}}
    rc = verdict(graph, lanes, verbose=False)
    check("leaking lane is caught", rc == 1, rc)

    # Negative control: the SAME graph, a lane that never selects the
    # forwarding feature, must pass clean.
    lanes_clean = {"cuda-only": {"package": "root", "cargo_features": ["cuda"], "capabilities": {"flash_compiled": False}}}
    rc = verdict(graph, lanes_clean, verbose=False)
    check("non-leaking lane on the SAME graph passes clean", rc == 0, rc)

    # A graph where jammi-db's postgres/mysql are simply never reachable
    # from root at all (no forwarding feature exists) must also pass.
    graph_no_forward = flash.Graph(_synthetic_metadata(db_deps_on_forbidden=False))
    rc = verdict(graph_no_forward, lanes_clean, verbose=False)
    check("no forwarding path anywhere passes clean", rc == 0, rc)

    # A lane naming a package that is not a workspace member is a named
    # FAIL, never a silent skip.
    lanes_bad_pkg = {"ghost": {"package": "does-not-exist", "cargo_features": [], "capabilities": {"flash_compiled": False}}}
    rc = verdict(graph, lanes_bad_pkg, verbose=False)
    check("unknown package lane is a named fail", rc == 1, rc)

    # Real-tree control: the REAL manifest, walked through the REAL
    # workspace graph, must currently pass clean — proves the synthetic
    # fixtures above match production, not a toy shape.
    real_metadata = flash.load_metadata()
    real_graph = flash.Graph(real_metadata)
    real_lanes = flash.load_manifest_lanes()
    rc = verdict(real_graph, real_lanes, verbose=False)
    check("the real manifest currently reaches neither feature", rc == 0, rc)

    if failures:
        print("release-manifest-pg-mysql-closure self-test: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        "release-manifest-pg-mysql-closure self-test: OK — a lane reaching jammi-db/postgres "
        "(or mysql) is caught, a non-leaking lane on the identical graph and a graph with no "
        "forwarding path at all both pass clean, an unknown-package lane is a named fail, and "
        "the real manifest against the real workspace graph currently reaches neither feature."
    )
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    metadata = flash.load_metadata()
    graph = flash.Graph(metadata)
    lanes = flash.load_manifest_lanes()
    return verdict(graph, lanes, verbose=True)


if __name__ == "__main__":
    sys.exit(main())
