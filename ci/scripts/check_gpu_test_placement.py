#!/usr/bin/env python3
"""Every GPU test is where the GPU lane selects it, and nowhere else.

The GPU lane (`ci/scripts/runpod_gpu_prove.sh`) runs only tests that need a
GPU: a test target that is GPU-only as a whole (`required-features` includes
`live-gpu-tests` or `live-flash-oracle-tests`), or the `gpu` module a CPU
target keeps its GPU tests in, selected by `gpu::`. Every CPU test runs on the
hosted CI runners. So in a CPU target, the GPU features may gate exactly one
thing: a `mod gpu` declaration. Anything else they gate there is a GPU test the
lane never selects, or a CPU test that drags a GPU host into running it.

    python3 ci/scripts/check_gpu_test_placement.py [--self-test]
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

GPU_FEATURES = ("live-gpu-tests", "live-flash-oracle-tests")
GATE = re.compile(r'#\[cfg\((?:all\()?feature = "(%s)"' % "|".join(GPU_FEATURES))


def gpu_only_targets(crate_dir: Path) -> set[Path]:
    """Source files of the crate's test targets that require a GPU feature."""
    manifest = tomllib.loads((crate_dir / "Cargo.toml").read_text())
    return {
        crate_dir / t.get("path", f"tests/{t['name']}.rs")
        for t in manifest.get("test", [])
        if any(f in t.get("required-features", []) for f in GPU_FEATURES)
    }


def _skip_literal(s: str, i: int) -> int | None:
    """Index after the string, char, raw-string or comment starting at s[i], if one does."""
    if s.startswith("//", i):
        return s.find("\n", i) if "\n" in s[i:] else len(s)
    if s.startswith("/*", i):
        return s.index("*/", i) + 2
    raw = re.match(r'r(#*)"', s[i:i + 8])
    if raw and (i == 0 or not (s[i - 1].isalnum() or s[i - 1] == "_")):
        close = '"' + raw.group(1)
        return s.index(close, i + raw.end()) + len(close)
    if s[i] == '"':
        j = i + 1
        while s[j] != '"':
            j += 2 if s[j] == "\\" else 1
        return j + 1
    char = re.match(r"'(\\.[^']*|[^\\'])'", s[i:i + 12])
    return i + char.end() if char else None


def gpu_module_spans(source: str) -> list[tuple[int, int]]:
    """(start, end) offsets of every `mod gpu { ... }` body in `source`."""
    spans = []
    for m in re.finditer(r"\bmod gpu \{", source):
        depth, j = 0, m.end() - 1
        while True:
            k = _skip_literal(source, j)
            if k is not None:
                j = k
                continue
            depth += {"{": 1, "}": -1}.get(source[j], 0)
            j += 1
            if depth == 0:
                break
        spans.append((m.end(), j))
    return spans


def misplaced(source: str) -> list[int]:
    """Line numbers of GPU-feature gates in `source` that gate anything but `mod gpu`
    and are not already inside a `mod gpu` body."""
    lines = source.splitlines()
    starts, offset = [], 0
    for line in lines:
        starts.append(offset)
        offset += len(line) + 1
    spans = gpu_module_spans(source)
    out = []
    for n, line in enumerate(lines):
        if not GATE.search(line.strip()) or line.lstrip().startswith("//"):
            continue
        if any(a <= starts[n] < b for a, b in spans):
            continue
        following = next((l.strip() for l in lines[n + 1:] if l.strip() and not l.strip().startswith(("//", "#["))), "")
        if not re.match(r"(pub(\([^)]*\))? )?mod gpu\b", following):
            out.append(n + 1)
    return out


def problems(root: Path) -> list[str]:
    found = []
    for crate_dir in sorted(p.parent for p in root.glob("crates/*/Cargo.toml")):
        exempt = gpu_only_targets(crate_dir)
        for rs in sorted(crate_dir.rglob("*.rs")):
            if rs in exempt or "target" in rs.parts or "third_party" in rs.parts:
                continue
            for n in misplaced(rs.read_text()):
                found.append(f"{rs.relative_to(root)}:{n}: a GPU feature gates something other than `mod gpu`")
    return found


def self_test() -> int:
    ok = '#[cfg(feature = "live-gpu-tests")]\nmod gpu {\n    #[test]\n    fn t() {}\n}\n'
    bad = '#[cfg(feature = "live-gpu-tests")]\n#[test]\nfn t() {}\n'
    flash = '    #[cfg(all(feature = "live-gpu-tests", feature = "flash-attn"))]\n    #[test]\n    fn t() {}\n'
    assert misplaced(ok) == [], misplaced(ok)
    assert misplaced(bad) == [1], misplaced(bad)
    assert misplaced(flash) == [1], misplaced(flash)
    assert misplaced("// #[cfg(feature = \"live-gpu-tests\")]\nfn t() {}\n") == []
    nested = ok.replace("    #[test]", '    #[cfg(feature = "live-flash-oracle-tests")]\n    #[test]')
    assert misplaced(nested) == [], misplaced(nested)
    print("check_gpu_test_placement: self-test passed")
    return 0


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    found = problems(Path("."))
    for p in found:
        print(f"FAIL {p}")
    if not found:
        print("check_gpu_test_placement: every GPU test is in a GPU-only target or a `gpu` module")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
