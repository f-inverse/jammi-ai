#!/usr/bin/env python3
"""Where a build runs: the one map from an architecture to its native hosted
runner, and the job matrices the reusable build workflows derive from it.

    python3 ci/scripts/build_matrix.py images --platforms linux/amd64,linux/arm64 [--cross variant=a,b]
    python3 ci/scripts/build_matrix.py cli --targets "x86_64-unknown-linux-gnu ..." --container IMAGE
    python3 ci/scripts/build_matrix.py wheels --platforms "linux-x86_64 macos-arm64 ..."
    python3 ci/scripts/build_matrix.py server --arch x86_64

Each prints `KEY=VALUE` lines for `$GITHUB_OUTPUT`, the values JSON a
`strategy.matrix` reads through `fromJson`. A workflow never spells a runner
label: an architecture's runner is decided here, once, and every native build
leg follows it. A Linux build runs in the tree's CI image (`ci_image.py`) on
the runner native to its architecture, never under emulation; a macOS build
runs on the Apple-silicon runner, which also cross-compiles the x86_64 wheel
with the same SDK.
"""

from __future__ import annotations

import argparse
import json
import sys

# A Linux architecture, in each spelling a tool uses for it, and the hosted
# runner native to it.
LINUX_RUNNERS = {
    "amd64": "ubuntu-latest",
    "arm64": "ubuntu-24.04-arm",
}
ARCH_BY_PLATFORM = {"linux/amd64": "amd64", "linux/arm64": "arm64"}
ARCH_BY_UNAME = {"x86_64": "amd64", "aarch64": "arm64"}
# Every macOS build runs here, natively on arm64 and cross-compiled to x86_64.
MACOS_RUNNER = "macos-14"


def runner(arch: str) -> str:
    """The hosted runner native to a Linux architecture (`amd64`/`arm64`,
    or its `uname -m` spelling)."""
    key = ARCH_BY_UNAME.get(arch, arch)
    if key not in LINUX_RUNNERS:
        raise ValueError(f"no runner for architecture {arch!r} (one of {sorted(LINUX_RUNNERS)} or {sorted(ARCH_BY_UNAME)})")
    return LINUX_RUNNERS[key]


def images(platforms: list[str], cross: dict[str, list[str]] | None = None) -> dict:
    """`{"include": [{platform, runner, arch, ...}]}` for a multi-arch image
    build: one row per platform, times every value of each `cross` key (an
    image built in several variants gets a row per variant and platform)."""
    rows: list[dict] = []
    for platform in platforms:
        if platform not in ARCH_BY_PLATFORM:
            raise ValueError(f"unsupported platform {platform!r} (one of {sorted(ARCH_BY_PLATFORM)})")
        arch = ARCH_BY_PLATFORM[platform]
        rows.append({"platform": platform, "runner": runner(arch), "arch": arch})
    for key, values in (cross or {}).items():
        if not values:
            raise ValueError(f"an empty value list for {key!r}: nothing to build")
        rows = [{**row, key: value} for value in values for row in rows]
    return {"include": rows}


def cli(targets: list[str], container: str) -> dict:
    """`{"include": [{target, runner, container}]}` for the CLI's target triples:
    the Linux targets build in the CI image, the Apple one on the macOS runner."""
    include = []
    for target in targets:
        if target == "x86_64-unknown-linux-gnu":
            include.append({"target": target, "runner": runner("amd64"), "container": container})
        elif target == "aarch64-unknown-linux-gnu":
            include.append({"target": target, "runner": runner("arm64"), "container": container})
        elif target == "aarch64-apple-darwin":
            include.append({"target": target, "runner": MACOS_RUNNER, "container": ""})
        else:
            raise ValueError(f"unsupported target {target!r}")
    return {"include": include}


def wheels(platforms: list[str]) -> dict[str, list[dict]]:
    """The native wheel legs: `linux` rows `{arch, runner}` (built in the CI
    image, tagged by the container) and `macos` rows `{target, artifact,
    assert_arch}` (built on the macOS runner, the Mach-O machine asserted)."""
    linux: list[dict] = []
    macos: list[dict] = []
    for platform in platforms:
        if platform == "linux-x86_64":
            linux.append({"arch": "x86_64", "runner": runner("amd64")})
        elif platform == "linux-aarch64":
            linux.append({"arch": "aarch64", "runner": runner("arm64")})
        elif platform == "macos-arm64":
            macos.append({"target": "aarch64-apple-darwin", "artifact": "wheels-native-macos-arm64", "assert_arch": "aarch64"})
        elif platform == "macos-x86_64":
            macos.append({"target": "x86_64-apple-darwin", "artifact": "wheels-native-macos-x86_64", "assert_arch": "x86_64"})
        else:
            raise ValueError(f"unsupported platform {platform!r}")
    return {"linux": linux, "macos": macos}


def _words(value: str) -> list[str]:
    words = [w for w in value.replace(",", " ").split() if w]
    if not words:
        raise ValueError("an empty list: nothing to build")
    return words


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("images")
    p.add_argument("--platforms", required=True)
    p.add_argument("--cross", action="append", default=[], metavar="KEY=V1,V2", help="a further axis, every row per value")
    p = sub.add_parser("cli")
    p.add_argument("--targets", required=True)
    p.add_argument("--container", required=True)
    sub.add_parser("wheels").add_argument("--platforms", required=True)
    sub.add_parser("server").add_argument("--arch", required=True)
    args = ap.parse_args(argv)
    try:
        if args.command == "images":
            cross = {}
            for spec in args.cross:
                key, sep, values = spec.partition("=")
                if not sep or not key:
                    raise ValueError(f"--cross takes KEY=V1,V2, got {spec!r}")
                cross[key] = _words(values)
            out = {"matrix": images(_words(args.platforms), cross)}
        elif args.command == "cli":
            out = {"matrix": cli(_words(args.targets), args.container)}
        elif args.command == "wheels":
            out = wheels(_words(args.platforms))
        else:
            out = {"runner": runner(args.arch)}
    except ValueError as e:
        print(f"::error::build_matrix: {e}", file=sys.stderr)
        return 1
    for key, value in out.items():
        print(f"{key}={value if isinstance(value, str) else json.dumps(value, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
