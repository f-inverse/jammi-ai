#!/usr/bin/env python3
"""Assert `ci/release-feature-manifest.json` is internally consistent.

**Guarded property**: the manifest's three CUDA release lanes describe ONE
shipped capability surface, and every op it names is classified by exactly
ONE proof mechanism.

The manifest is duplicated three ways on purpose (one block per lane, so a
lane that genuinely diverges can say so), which makes silent divergence the
failure mode: a reviewer editing one lane's `capabilities` and missing the
other two ships a manifest that claims three different surfaces for three
builds of the same feature list. And because a category IS a proof mechanism
(`fused_op_admission` = an observed dispatch-registry delta;
`internal_subkernels` = an admitted parent's observed delta — see the
manifest's own `_schema_doc`), an op named in two categories is a
contradiction: it cannot both have and not have its own admission site.

The set of categories is CLOSED at those two, and each must be non-empty.
Both halves of that are load-bearing. A kernel with no `admit()` site and no
admitted parent is provable only as "it compiled", which is not a capability
a consumer can act on; a category holding exactly that is an empty proof
slot, and an empty proof slot is where the next unwired kernel gets filed
instead of being wired or deleted. So a third, op-bearing capability key is a
FINDING here (not a new mechanism this gate silently tolerates), and a
category that has drained to `[]`/`{}` is a FINDING too — the manifest must
either name ops under a mechanism or not carry the mechanism at all.

Checks (hermetic: reads the manifest and the tracked Rust sources, no
network, no build, no toolchain):

  1. The file parses; `lanes` exists and is non-empty; every lane carries a
     `capabilities` object with both category keys present, each non-empty,
     and NO capability key outside the closed set (the two categories plus
     the non-op build facts `cuda_compiled`/`flash_compiled`/`flash_dtypes`).
     Fail-closed: a missing/renamed/extra key is a FINDING, never a silently
     skipped check.
  2. Every lane's `capabilities` block is IDENTICAL to every other lane's,
     compared as canonical JSON (key order included, so the file stays
     reviewable as three literally-equal blocks).
  3. Every op named across the two categories appears in EXACTLY ONE of
     them, and at most once within its own category.
  4. Every `internal_subkernels` entry carries a `parent` and a
     `launch_site`; the `launch_site` must RESOLVE to a file that exists
     (a citation that cannot rot into a dangling path), and the `parent`
     must resolve either to a `fused_op_admission` op or to a real
     dispatch-registry key literal passed to `counters_for(` /
     `cascade_counters_for(` somewhere in `crates/*/src` (so a renamed
     registry key breaks this file instead of leaving an orphan claim).

What this does NOT check: whether a lane's declared capability is TRUE on a
real device. That is `capability_surface.rs`'s job (a live registry-delta
probe) and `check_flash_attn_closure.py`'s (the feature-graph closure). This
gate is the internal-consistency floor underneath both.

Usage:
    python3 ci/scripts/check_release_manifest.py
    python3 ci/scripts/check_release_manifest.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
assert (REPO_ROOT / "Cargo.toml").is_file(), (
    f"REPO_ROOT resolved to {REPO_ROOT}, which has no Cargo.toml — this file "
    "sits at ci/scripts/<name>.py (parents[2] == repo root); a future move "
    "must update this constant, never fail silently downstream."
)

MANIFEST_PATH = REPO_ROOT / "ci" / "release-feature-manifest.json"

# The two proof mechanisms, in the manifest's own order. A LIST category
# names ops directly; the OBJECT category is keyed by op.
LIST_CATEGORIES = ("fused_op_admission",)
OBJECT_CATEGORIES = ("internal_subkernels",)
CATEGORIES = ("fused_op_admission", "internal_subkernels")

# The capability keys that are NOT proof mechanisms: build facts about the
# lane that name no op. Together with CATEGORIES these close the set of keys
# a `capabilities` block may carry — anything else is a finding, so a third
# op-bearing category cannot be introduced by a manifest-only edit (see this
# module's doc).
NON_CATEGORY_CAPABILITY_KEYS = frozenset(
    {"cuda_compiled", "flash_compiled", "flash_dtypes"}
)

# `counters_for("<key>")` / `cascade_counters_for("<key>")` — the only two
# ways a dispatch-registry key is named in this workspace.
_REGISTRY_KEY_RE = re.compile(r"(?:cascade_)?counters_for\(\s*\"([a-z0-9_]+)\"")


def registry_key_literals(repo_root: Path) -> set[str]:
    """Every dispatch-registry key literal named in `crates/*/src`."""
    keys: set[str] = set()
    for src in sorted(repo_root.glob("crates/*/src/**/*.rs")):
        try:
            text = src.read_text()
        except OSError:  # pragma: no cover - unreadable source is not this gate's finding
            continue
        keys.update(m.group(1) for m in _REGISTRY_KEY_RE.finditer(text))
    return keys


def _category_ops(caps: dict, lane_name: str, problems: list[str]) -> dict[str, list[str]]:
    """Per-category op names for one lane, appending a finding for any
    missing or wrongly-shaped category."""
    ops: dict[str, list[str]] = {}
    for cat in LIST_CATEGORIES:
        if cat not in caps:
            problems.append(f"lane `{lane_name}`: capabilities has no `{cat}` key")
            continue
        value = caps[cat]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            problems.append(f"lane `{lane_name}`: `{cat}` must be a list of op-name strings")
            continue
        ops[cat] = list(value)
    for cat in OBJECT_CATEGORIES:
        if cat not in caps:
            problems.append(f"lane `{lane_name}`: capabilities has no `{cat}` key")
            continue
        value = caps[cat]
        if not isinstance(value, dict):
            problems.append(f"lane `{lane_name}`: `{cat}` must be an object keyed by op name")
            continue
        ops[cat] = list(value.keys())
    # A category that is present but names no op is an EMPTY PROOF SLOT: it
    # asserts a mechanism the lane exercises for nothing, and it is where the
    # next unwired kernel gets parked instead of being wired or deleted. The
    # manifest must drop the mechanism rather than carry it empty.
    for cat, named in ops.items():
        if not named:
            problems.append(
                f"lane `{lane_name}`: `{cat}` is present but EMPTY — a proof mechanism "
                f"that names no op is an empty slot the next unwired kernel gets filed "
                f"into; name an op under it or remove the category"
            )
    # The category set is CLOSED. An unrecognized capability key is either a
    # new proof mechanism (which needs this gate and `capability_surface.rs`
    # to define how it is proven, in the same unit) or a resurrected
    # compiled-only bucket. Either way it is a finding, never tolerated.
    for key in caps:
        if key in CATEGORIES or key in NON_CATEGORY_CAPABILITY_KEYS:
            continue
        problems.append(
            f"lane `{lane_name}`: unknown capability key `{key}` — the proof mechanisms "
            f"are exactly {list(CATEGORIES)} and the non-op build facts are exactly "
            f"{sorted(NON_CATEGORY_CAPABILITY_KEYS)}; a key naming ops under any other "
            f"mechanism has no defined proof, so it cannot be added by a manifest-only edit"
        )
    return ops


def check_manifest(manifest: dict, repo_root: Path = REPO_ROOT) -> list[str]:
    """Return every finding (empty == green)."""
    problems: list[str] = []

    lanes = manifest.get("lanes")
    if not isinstance(lanes, dict) or not lanes:
        return ["manifest has no (or an empty) `lanes` object — nothing to check"]

    # (2) every lane's capability block is identical.
    canonical: dict[str, str] = {}
    for lane_name, lane in lanes.items():
        caps = lane.get("capabilities") if isinstance(lane, dict) else None
        if not isinstance(caps, dict):
            problems.append(f"lane `{lane_name}`: no `capabilities` object")
            continue
        canonical[lane_name] = json.dumps(caps, indent=2, sort_keys=False)
    if len(set(canonical.values())) > 1:
        reference = next(iter(canonical))
        for lane_name, blob in canonical.items():
            if blob != canonical[reference]:
                problems.append(
                    f"lane `{lane_name}`'s capabilities block differs from lane "
                    f"`{reference}`'s — the three lanes build one feature list and "
                    f"must declare one capability surface (edit all lanes in one unit)"
                )

    known_registry_keys = registry_key_literals(repo_root)

    for lane_name, lane in lanes.items():
        caps = lane.get("capabilities") if isinstance(lane, dict) else None
        if not isinstance(caps, dict):
            continue

        # (1)/(3) category shape, then exactly-one-category membership.
        ops = _category_ops(caps, lane_name, problems)
        seen: dict[str, str] = {}
        for cat in CATEGORIES:
            for op in ops.get(cat, []):
                if op in seen and seen[op] == cat:
                    problems.append(f"lane `{lane_name}`: op `{op}` listed twice in `{cat}`")
                elif op in seen:
                    problems.append(
                        f"lane `{lane_name}`: op `{op}` appears in both `{seen[op]}` and "
                        f"`{cat}` — the categories are DIFFERENT proof mechanisms, so an "
                        f"op belongs to exactly one"
                    )
                else:
                    seen[op] = cat

        # (4) internal sub-kernel citations resolve.
        subkernels = caps.get("internal_subkernels")
        if isinstance(subkernels, dict):
            admitted = set(ops.get("fused_op_admission", []))
            for op, entry in subkernels.items():
                if not isinstance(entry, dict):
                    problems.append(
                        f"lane `{lane_name}`: internal sub-kernel `{op}` must be an object "
                        f"with `parent` and `launch_site`"
                    )
                    continue
                parent = entry.get("parent")
                launch_site = entry.get("launch_site")
                if not isinstance(parent, str) or not parent:
                    problems.append(
                        f"lane `{lane_name}`: internal sub-kernel `{op}` has no `parent` — "
                        f"its whole proof is the parent's observed dispatch"
                    )
                elif parent not in admitted and parent not in known_registry_keys:
                    problems.append(
                        f"lane `{lane_name}`: internal sub-kernel `{op}`'s parent "
                        f"`{parent}` is neither a `fused_op_admission` op nor a dispatch-"
                        f"registry key named in crates/*/src — an unprovable parent"
                    )
                if not isinstance(launch_site, str) or not launch_site:
                    problems.append(
                        f"lane `{lane_name}`: internal sub-kernel `{op}` has no `launch_site`"
                    )
                elif not (repo_root / launch_site).is_file():
                    problems.append(
                        f"lane `{lane_name}`: internal sub-kernel `{op}`'s launch_site "
                        f"`{launch_site}` does not exist"
                    )

    return problems


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    try:
        raw = path.read_text()
    except OSError as e:
        print(f"ERROR: cannot read {path}: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"ERROR: {path} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)


# --------------------------------------------------------------------------
# Self-test — RED-proves each finding class against synthetic manifests.
# --------------------------------------------------------------------------


def _fixture_manifest() -> dict:
    caps = {
        "cuda_compiled": True,
        "flash_compiled": True,
        "fused_op_admission": ["layer_norm", "low_rank_residual_linear"],
        "internal_subkernels": {
            "scaled_cast_add": {
                "parent": "low_rank_residual_linear",
                "launch_site": "Cargo.toml",
            }
        },
    }
    return {
        "lanes": {
            "a": {"capabilities": json.loads(json.dumps(caps))},
            "b": {"capabilities": json.loads(json.dumps(caps))},
        }
    }


def _self_test() -> int:
    failures: list[str] = []
    total = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal total
        total += 1
        print(f"self-test[{name}]: " + ("ok" if cond else f"FAIL -- {detail}"))
        if not cond:
            failures.append(name)

    # A well-formed fixture is GREEN (a gate that reds on everything proves
    # nothing about the cases below).
    m = _fixture_manifest()
    check("well-formed-manifest-is-green", check_manifest(m, REPO_ROOT) == [], f"{check_manifest(m, REPO_ROOT)}")

    # 1. A lane whose capability block diverges from its siblings.
    m = _fixture_manifest()
    m["lanes"]["b"]["capabilities"]["fused_op_admission"] = [
        "layer_norm",
        "low_rank_residual_linear",
        "sneaked_in",
    ]
    probs = check_manifest(m, REPO_ROOT)
    check("divergent-lane-capability-block-caught", any("differs from lane" in p for p in probs), f"{probs}")

    # 2. An op in two categories at once — the two mechanisms contradict.
    #    Here an ADMITTED op is also claimed as a parentless sub-kernel.
    m = _fixture_manifest()
    for lane in m["lanes"].values():
        lane["capabilities"]["internal_subkernels"]["layer_norm"] = {
            "parent": "low_rank_residual_linear",
            "launch_site": "Cargo.toml",
        }
    probs = check_manifest(m, REPO_ROOT)
    check("op-in-two-categories-caught", any("appears in both" in p for p in probs), f"{probs}")

    # 2b. The other direction of the same overlap (a sub-kernel that ALSO
    #     claims its own admission site).
    m = _fixture_manifest()
    for lane in m["lanes"].values():
        lane["capabilities"]["fused_op_admission"] = [
            "layer_norm",
            "low_rank_residual_linear",
            "scaled_cast_add",
        ]
    probs = check_manifest(m, REPO_ROOT)
    check("subkernel-also-claiming-admission-caught", any("appears in both" in p for p in probs), f"{probs}")

    # 3. A duplicate inside one category.
    m = _fixture_manifest()
    for lane in m["lanes"].values():
        lane["capabilities"]["fused_op_admission"] = ["layer_norm", "layer_norm", "low_rank_residual_linear"]
    probs = check_manifest(m, REPO_ROOT)
    check("duplicate-within-a-category-caught", any("listed twice" in p for p in probs), f"{probs}")

    # 4. A launch_site that does not resolve.
    m = _fixture_manifest()
    for lane in m["lanes"].values():
        lane["capabilities"]["internal_subkernels"]["scaled_cast_add"]["launch_site"] = "crates/gone/src/nope.rs"
    probs = check_manifest(m, REPO_ROOT)
    check("dangling-launch-site-caught", any("does not exist" in p for p in probs), f"{probs}")

    # 5. A parent that is neither an admitted op nor a real registry key.
    m = _fixture_manifest()
    for lane in m["lanes"].values():
        lane["capabilities"]["internal_subkernels"]["scaled_cast_add"]["parent"] = "not_a_real_parent"
    probs = check_manifest(m, REPO_ROOT)
    check("unresolvable-parent-caught", any("unprovable parent" in p for p in probs), f"{probs}")

    # 5b. A parent that is ONLY a registry key (never a manifest op name) is
    #     accepted — this is the real `attention_block_flash` shape, so the
    #     check above must not be satisfied by "is in fused_op_admission".
    m = _fixture_manifest()
    for lane in m["lanes"].values():
        lane["capabilities"]["internal_subkernels"]["scaled_cast_add"]["parent"] = "attention_block_flash"
    probs = check_manifest(m, REPO_ROOT)
    check("registry-key-only-parent-accepted", probs == [], f"{probs}")

    # 6. A missing category key fails closed rather than skipping silently.
    m = _fixture_manifest()
    for lane in m["lanes"].values():
        del lane["capabilities"]["internal_subkernels"]
    probs = check_manifest(m, REPO_ROOT)
    check("missing-category-key-caught", any("has no `internal_subkernels` key" in p for p in probs), f"{probs}")

    # 6b. A category that has DRAINED to empty is caught — the shape this
    #     manifest would have had if the last compiled-only op were deleted
    #     while its category stayed. Both the list and the object category.
    m = _fixture_manifest()
    for lane in m["lanes"].values():
        # Re-point the sub-kernel's parent at a real REGISTRY KEY first, so
        # draining the admission list does not ALSO orphan the parent
        # citation — the property under test is the emptiness alone.
        lane["capabilities"]["internal_subkernels"]["scaled_cast_add"]["parent"] = (
            "attention_block_flash"
        )
        lane["capabilities"]["fused_op_admission"] = []
    probs = check_manifest(m, REPO_ROOT)
    check(
        "empty-list-category-caught",
        any("`fused_op_admission` is present but EMPTY" in p for p in probs),
        f"{probs}",
    )

    m = _fixture_manifest()
    for lane in m["lanes"].values():
        lane["capabilities"]["internal_subkernels"] = {}
    probs = check_manifest(m, REPO_ROOT)
    check(
        "empty-object-category-caught",
        any("`internal_subkernels` is present but EMPTY" in p for p in probs),
        f"{probs}",
    )

    # 6c. A THIRD, op-bearing capability category is caught — a proof
    #     mechanism nothing defines cannot be introduced by a manifest-only
    #     edit, whatever it is named.
    for extra_key, extra_value in (
        ("fused_kernels_compiled", ["some_kernel"]),
        ("some_new_mechanism", {"some_kernel": {"parent": "low_rank_residual_linear"}}),
    ):
        m = _fixture_manifest()
        for lane in m["lanes"].values():
            lane["capabilities"][extra_key] = json.loads(json.dumps(extra_value))
        probs = check_manifest(m, REPO_ROOT)
        check(
            f"unknown-op-bearing-category-caught[{extra_key}]",
            any(f"unknown capability key `{extra_key}`" in p for p in probs),
            f"{probs}",
        )

    # 6d. The non-category build facts are NOT mistaken for a category (a
    #     closed-set check that reds on `flash_dtypes` would be unusable).
    m = _fixture_manifest()
    for lane in m["lanes"].values():
        lane["capabilities"]["flash_dtypes"] = ["bf16", "f16"]
    probs = check_manifest(m, REPO_ROOT)
    check("non-category-build-facts-accepted", probs == [], f"{probs}")

    # 7. An empty/absent `lanes` object can never pass vacuously.
    probs = check_manifest({"lanes": {}}, REPO_ROOT)
    check("empty-lanes-object-caught", probs != [], f"{probs}")

    # 8. The registry-key scan is non-vacuous against the real tree.
    keys = registry_key_literals(REPO_ROOT)
    check(
        "registry-key-scan-non-vacuous",
        "attention_block_flash" in keys and "lora_linear_fused" in keys,
        f"got {sorted(keys)[:8]}...",
    )

    if failures:
        print(f"self-test: FAIL ({len(failures)}/{total} failing): {failures}", file=sys.stderr)
        return 1
    print(f"self-test: all {total} checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check ci/release-feature-manifest.json's internal consistency.")
    ap.add_argument("--self-test", action="store_true", help="run the RED-proof self-tests and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    problems = check_manifest(load_manifest())
    if problems:
        for p in problems:
            print(f"ERROR: {MANIFEST_PATH.relative_to(REPO_ROOT)}: {p}", file=sys.stderr)
        print(f"release-manifest: FAIL ({len(problems)} finding(s))", file=sys.stderr)
        return 1
    print("release-manifest: lanes agree, every op is in exactly one category, citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
