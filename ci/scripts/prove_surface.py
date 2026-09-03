#!/usr/bin/env python3
"""Shared prove-lane feature-surface canonicalization (esc-081).

**Guarded property**: proof surface == shipped surface. `ci/release-feature-
manifest.json`'s `prove_lane.crates.<c>.kinds` DECLARES, outside
`runpod_gpu_prove.sh`, exactly which `(crate, kind)` cargo invocations that
script must carry; this module is the ONE place both the script's own
tripwire, `check_flash_attn_closure.py`'s set-equality rule,
`check_release_manifest.py`'s validation, and `ci/scripts/perf/
gpu_prove_timings.py`'s freshness fingerprint compute what a pair's expected
feature list IS — one canonicalization, imported everywhere, so the producer
and the gates can never independently drift.

`kind` is one of:
  * `release` -- a `cargo build --release`/`cargo run --release` invocation.
  * `test`    -- a `cargo test` invocation carrying a `--features` tuple
                 (with or without `--no-run`; the TUPLE, not the test-name
                 filter, is what defines this kind).
  * `default` -- a `cargo test` invocation with NO `--features` flag at all.
                 Canonicalizes to an EMPTY feature list.

`expected(crate, kind) = (lane ∩ declared(crate)) ∪ (prove_only(crate) iff
kind == "test")`, where `lane` is this file's own `lanes.cu12-tarball.
cargo_features` (the single source of truth for the shipped CUDA release
surface) and `declared(crate)` is `crates/<crate>/Cargo.toml`'s own
`[features]` table keys, read via stdlib `tomllib` -- the FIRST use of
`tomllib` in this tree (the standing convention elsewhere is a regex over
Cargo.toml; a regex's failure direction is a silently NARROWER `declared()`,
which would under-report a crate's real feature surface and let a genuine
lane feature slip past the manifest-declared check). Residual, disclosed:
a `[features]` table's keys miss a crate's IMPLICIT optional-dependency
features (`dep = { optional = true }` with no matching `[features]` entry)
-- fail-closed today, since such a feature would read as "not declared",
never silently "declared".

No `cargo metadata` here (or in any of this module's importers in the
toolchain-free guard matrix) -- `tomllib` over each crate's own Cargo.toml
is hermetic and needs no toolchain at all.

Run: `python3 ci/scripts/prove_surface.py --self-test`
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError as e:  # pragma: no cover - guard-matrix runners are 3.11+
    print(
        f"ERROR: ci/scripts/prove_surface.py requires the stdlib `tomllib` "
        f"module (Python 3.11+): {e}",
        file=sys.stderr,
    )
    raise SystemExit(2) from e

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "ci" / "release-feature-manifest.json"
LANE_NAME = "cu12-tarball"

KIND_RELEASE = "release"
KIND_TEST = "test"
KIND_DEFAULT = "default"
KINDS = (KIND_RELEASE, KIND_TEST, KIND_DEFAULT)

# ONE marker grammar, one parser PER LANGUAGE (esc-080/esc-082/esc-083,
# BLOCK 3 audit fix): `PROVE_GROUP_RC name=<n> rc=<v>`, exactly as
# `runpod_lib.sh`'s own `rp_parse_prove_marker` (the bash-side twin, shared
# by `rp_run_remote_watched` and `runpod_gpu_prove.sh`'s `rp_prove_verdict`)
# parses it. This is the SINGLE Python-side source of truth --
# `ci/scripts/perf/gpu_prove_timings.py` imports this constant rather than
# compiling its own copy, so the two languages' grammars cannot silently
# drift apart. A cross-parser fixture in `test_gpu_prove_lane.sh` feeds the
# identical marker text to both the bash function and this regex and
# asserts identical (name, rc) extraction.
PROVE_GROUP_RC_RE = re.compile(r"PROVE_GROUP_RC name=(?P<name>\S+) rc=(?P<rc>-?\d+)")


def declared(crate: str, repo_root: Path = REPO_ROOT) -> set[str]:
    """`crates/<crate>/Cargo.toml`'s own `[features]` table keys."""
    path = repo_root / "crates" / crate / "Cargo.toml"
    with path.open("rb") as f:
        data = tomllib.load(f)
    return set(data.get("features", {}).keys())


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text())


def lane_features(manifest: dict, lane: str = LANE_NAME) -> set[str]:
    return set(manifest["lanes"][lane]["cargo_features"])


def prove_lane_crates(manifest: dict) -> dict:
    return manifest.get("prove_lane", {}).get("crates", {})


def declared_pairs(manifest: dict) -> set[tuple[str, str]]:
    """Every `(crate, kind)` pair the manifest DECLARES as required."""
    out: set[tuple[str, str]] = set()
    for crate, spec in prove_lane_crates(manifest).items():
        for kind in spec.get("kinds", []):
            out.add((crate, kind))
    return out


def expected(
    crate: str,
    kind: str,
    manifest: dict | None = None,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """The sorted, deduplicated feature list a `(crate, kind)` pair's cargo
    invocation must carry -- the SAME list used to build the literal
    `--features` text in `runpod_gpu_prove.sh` and to compute `expected_id`
    below, so the two can never independently drift."""
    if kind not in KINDS:
        raise ValueError(f"unknown prove-lane kind {kind!r} (want one of {KINDS})")
    if kind == KIND_DEFAULT:
        return []
    manifest = manifest if manifest is not None else load_manifest()
    lane = lane_features(manifest)
    base = lane & declared(crate, repo_root)
    if kind == KIND_TEST:
        prove_only = set(prove_lane_crates(manifest).get(crate, {}).get("prove_only", []))
        base = base | prove_only
    return sorted(base)


def feature_text(features: list[str]) -> str:
    """The literal `--features` argument text (empty string for `default`,
    meaning: no `--features` flag at all)."""
    return ",".join(features)


def expected_id(surface: dict[str, dict[str, list[str]]]) -> str:
    """One canonicalization over `{crate: {kind: [features...]}}` -- used by
    the producer (`ci/scripts/perf/gpu_prove_timings.py`) to fingerprint
    which surface a run actually proved, and by `check_gpu_prove_timings.py`
    (R5) to demand a fresh artifact whenever the surface moves. Sorted keys
    and sorted feature lists at every level -- key/list ORDER is never
    significant to the identity of a proof surface."""
    canon = {
        crate: {kind: sorted(feats) for kind, feats in kinds.items()}
        for crate, kinds in surface.items()
    }
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def current_expected_id(manifest: dict | None = None, repo_root: Path = REPO_ROOT) -> str:
    """`expected_id` over every pair the manifest CURRENTLY declares --
    moves only when a lane feature is added/removed, a `prove_only` entry
    changes, or a crate stops/starts declaring a lane feature."""
    manifest = manifest if manifest is not None else load_manifest()
    surface: dict[str, dict[str, list[str]]] = {}
    for crate, kind in sorted(declared_pairs(manifest)):
        surface.setdefault(crate, {})[kind] = expected(crate, kind, manifest, repo_root)
    return expected_id(surface)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _self_test() -> int:
    failures: list[str] = []
    total = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal total
        total += 1
        print(f"self-test[{name}]: " + ("ok" if cond else f"FAIL -- {detail}"))
        if not cond:
            failures.append(name)

    manifest = load_manifest()

    # Anchor self-test: jammi-server's real Cargo.toml must declare (at
    # least) the four lane features -- a broken tomllib read or a stripped
    # Cargo.toml would silently narrow `declared()` and this would trip.
    d = declared("jammi-server")
    check(
        "anchor-jammi-server-declares-lane",
        {"cuda", "flash-attn", "jetstream-broker", "storage-cloud"} <= d,
        f"declared(jammi-server)={sorted(d)}",
    )

    # jammi-server: release == the full lane; test == the full lane plus
    # prove_only (jammi-server declares every lane feature).
    check(
        "jammi-server-release",
        expected("jammi-server", "release", manifest) == ["cuda", "flash-attn", "jetstream-broker", "storage-cloud"],
        f"{expected('jammi-server', 'release', manifest)}",
    )
    check(
        "jammi-server-test",
        expected("jammi-server", "test", manifest)
        == ["cuda", "flash-attn", "jetstream-broker", "live-gpu-tests", "storage-cloud"],
        f"{expected('jammi-server', 'test', manifest)}",
    )

    # jammi-ai: does not declare jetstream-broker/storage-cloud, so its
    # lane-intersection is exactly {cuda, flash-attn}; test adds prove_only.
    check(
        "jammi-ai-test",
        expected("jammi-ai", "test", manifest) == ["cuda", "flash-attn", "live-gpu-tests"],
        f"{expected('jammi-ai', 'test', manifest)}",
    )

    # jammi-bench: declares only cuda/flash-attn; release == both, no
    # prove_only (release never adds prove_only, even if declared non-empty).
    check(
        "jammi-bench-release",
        expected("jammi-bench", "release", manifest) == ["cuda", "flash-attn"],
        f"{expected('jammi-bench', 'release', manifest)}",
    )

    # jammi-kernels: test widens to cuda,flash-attn (both declared and in
    # the lane); default canonicalizes to [] regardless of manifest content.
    check(
        "jammi-kernels-test",
        expected("jammi-kernels", "test", manifest) == ["cuda", "flash-attn"],
        f"{expected('jammi-kernels', 'test', manifest)}",
    )
    check(
        "jammi-kernels-default",
        expected("jammi-kernels", "default", manifest) == [],
        f"{expected('jammi-kernels', 'default', manifest)}",
    )

    # feature_text renders the canonical comma-joined literal.
    check(
        "feature-text",
        feature_text(["cuda", "flash-attn"]) == "cuda,flash-attn" and feature_text([]) == "",
        "",
    )

    # expected_id is order-insensitive at every level and changes when the
    # surface changes (the property R5 in check_gpu_prove_timings.py needs).
    a = expected_id({"jammi-ai": {"test": ["cuda", "flash-attn"]}})
    b = expected_id({"jammi-ai": {"test": ["flash-attn", "cuda"]}})
    check("expected-id-order-insensitive", a == b, f"{a} != {b}")
    c = expected_id({"jammi-ai": {"test": ["cuda"]}})
    check("expected-id-changes-with-surface", a != c, f"{a} == {c}")

    # An unknown kind is refused rather than silently treated as `default`.
    try:
        expected("jammi-ai", "bogus", manifest)
        check("unknown-kind-rejected", False, "did not raise")
    except ValueError:
        check("unknown-kind-rejected", True)

    # declared_pairs() reflects the manifest's own crates block exactly.
    pairs = declared_pairs(manifest)
    check(
        "declared-pairs-cover-manifest",
        ("jammi-server", "release") in pairs
        and ("jammi-server", "test") in pairs
        and ("jammi-ai", "test") in pairs
        and ("jammi-bench", "release") in pairs
        and ("jammi-kernels", "default") in pairs
        and ("jammi-kernels", "test") in pairs,
        f"{sorted(pairs)}",
    )

    if failures:
        print(f"self-test: FAIL ({len(failures)}/{total} failing): {failures}", file=sys.stderr)
        return 1
    print(f"self-test: all {total} checks passed")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return _self_test()
    manifest = load_manifest()
    for crate, kind in sorted(declared_pairs(manifest)):
        feats = expected(crate, kind, manifest)
        print(f"{crate} {kind}: {feature_text(feats) or '<default>'}")
    print(f"expected_id={current_expected_id(manifest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
