#!/usr/bin/env python3
# needs: cargo-registry
"""Assert jammi-db's database source features link their TLS and compression
statically.

**Guarded property**: `jammi-db`'s `postgres` and `mysql` features pull
`datafusion-table-providers`'s federation drivers, which depend on
`native-tls` (OpenSSL on Linux) unconditionally, and `mysql_async`'s
compression links zlib through `libz-sys`. Every published artifact enables
both features (`ci/release-feature-manifest.json`), and a manylinux artifact
may carry no `DT_NEEDED libssl` / `libcrypto` / `libz` — the platform tag
does not promise them, and the build and runtime images ship different
OpenSSL series. So each feature must turn on `native-tls`'s `vendored`
feature (OpenSSL built from source by `openssl-src` and linked statically),
and `mysql` must turn on `libz-sys`'s `static` feature. This gate reads
jammi-db's own declarations from `cargo metadata`, so it holds for every
lane that enables the features, present or future. The release workflows'
link-set checks (`readelf -d` on the built artifacts) are the runtime proof;
this is the declaration a regression would break first.

Method (hermetic: `cargo metadata --no-deps`, no network, no build): for
each database feature, the feature must activate the optional dependency
(`dep:<name>`) and that dependency must be declared with the static feature.
FAILS on a missing package, feature or dependency — never passes vacuously.

Run: `python3 ci/scripts/check_database_drivers_static.py`
Self-test: `python3 ci/scripts/check_database_drivers_static.py --self-test`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_flash_attn_closure import load_metadata  # noqa: E402 — the one workspace-metadata read

PACKAGE = "jammi-db"

# feature -> [(dependency, the feature on it that links it statically)]
REQUIRED: dict[str, list[tuple[str, str]]] = {
    "postgres": [("native-tls", "vendored")],
    "mysql": [("native-tls", "vendored"), ("libz-sys", "static")],
}


def load_package(metadata: dict, name: str = PACKAGE) -> dict:
    for pkg in metadata["packages"]:
        if pkg["name"] == name:
            return pkg
    raise SystemExit(f"FAIL: package `{name}` is not a workspace member")


def violations(pkg: dict) -> list[str]:
    """Every way `pkg`'s database features fall short of static linkage."""
    found = []
    deps = {d.get("rename") or d["name"]: d for d in pkg["dependencies"]}
    for feature, needed in REQUIRED.items():
        members = pkg["features"].get(feature)
        if members is None:
            found.append(f"`{PACKAGE}` declares no `{feature}` feature")
            continue
        for dep, static in needed:
            if f"dep:{dep}" not in members:
                found.append(f"feature `{feature}` does not activate `dep:{dep}`")
            declared = deps.get(dep)
            if declared is None:
                found.append(f"`{PACKAGE}` does not depend on `{dep}`")
            elif static not in declared.get("features", []):
                found.append(f"`{dep}` is declared without its `{static}` feature")
    return found


def verdict(pkg: dict, verbose: bool = True) -> int:
    found = violations(pkg)
    for v in found:
        if verbose:
            print(f"FAIL: {v}", file=sys.stderr)
    if not found and verbose:
        print(
            "OK: jammi-db's postgres and mysql features vendor OpenSSL "
            "(native-tls/vendored) and link zlib statically (libz-sys/static)."
        )
    return 1 if found else 0


def _synthetic(postgres: list[str], mysql: list[str], tls: list[str], zlib: list[str]) -> dict:
    return {
        "name": PACKAGE,
        "features": {"postgres": postgres, "mysql": mysql},
        "dependencies": [
            {"name": "native-tls", "features": tls},
            {"name": "libz-sys", "features": zlib},
        ],
    }


def self_test() -> int:
    failures: list[str] = []

    def check(label: str, cond: bool, detail: object = "") -> None:
        if not cond:
            failures.append(f"{label}: {detail}")

    good = _synthetic(
        ["dep:native-tls"], ["dep:native-tls", "dep:libz-sys"], ["vendored"], ["static"]
    )
    check("a fully static declaration passes", verdict(good, verbose=False) == 0, violations(good))
    no_vendor = _synthetic(
        ["dep:native-tls"], ["dep:native-tls", "dep:libz-sys"], [], ["static"]
    )
    check("native-tls without vendored is caught", verdict(no_vendor, verbose=False) == 1)
    no_static = _synthetic(
        ["dep:native-tls"], ["dep:native-tls", "dep:libz-sys"], ["vendored"], []
    )
    check("libz-sys without static is caught", verdict(no_static, verbose=False) == 1)
    not_activated = _synthetic([], ["dep:native-tls"], ["vendored"], ["static"])
    check(
        "a feature not activating its static dependency is caught",
        len(violations(not_activated)) == 2,
        violations(not_activated),
    )
    missing = {"name": PACKAGE, "features": {}, "dependencies": []}
    check("missing features are a named fail", verdict(missing, verbose=False) == 1)

    real = load_package(load_metadata())
    check("the real jammi-db declaration passes", verdict(real, verbose=False) == 0, violations(real))

    if failures:
        print("database-drivers-static self-test: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        "database-drivers-static self-test: OK — a fully static declaration passes; a missing "
        "`vendored`, a missing `static`, an unactivated dependency and missing features are "
        "each caught; the real jammi-db declaration passes."
    )
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    return verdict(load_package(load_metadata()))


if __name__ == "__main__":
    sys.exit(main())
