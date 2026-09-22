#!/usr/bin/env python3
"""Every distributed test runs in exactly one leg of `distributed.yml`.

The distributed suites (`crates/<crate>/tests/distributed/`) need Postgres,
an S3 store and a server fleet, so they run only in `distributed.yml`, one leg per
matrix row, each leg naming its tests. A test no leg names is compiled and never
run; a test two legs name runs twice under different expectations. This reads
the matrix and the test functions and fails on either.

    python3 ci/scripts/check_distributed_legs.py [--self-test]
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import yaml

WORKFLOW = Path(".github/workflows/distributed.yml")
TEST_FN = re.compile(r"#\[(?:tokio::)?test\b[^\]]*\]\s*(?:#\[[^\]]*\]\s*)*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)")


def tests_in(source: str) -> set[str]:
    """The test function names a Rust source defines."""
    return set(TEST_FN.findall(source))


def leg_tests(workflow: dict) -> dict[str, Counter]:
    """Per crate, how many legs name each test."""
    named: dict[str, Counter] = {}
    for leg in workflow["jobs"]["distributed"]["strategy"]["matrix"]["include"]:
        named.setdefault(leg["crate"], Counter()).update(leg["tests"].split())
    return named


def problems(named: dict[str, Counter], defined: dict[str, set[str]]) -> list[str]:
    out = []
    for crate in sorted(set(named) | set(defined)):
        legs, fns = named.get(crate, Counter()), defined.get(crate, set())
        out += [f"{crate}: `{t}` is in no distributed.yml leg" for t in sorted(fns - set(legs))]
        out += [f"{crate}: leg filter `{t}` names no test" for t in sorted(set(legs) - fns)]
        out += [f"{crate}: `{t}` is in {n} legs" for t, n in sorted(legs.items()) if n > 1]
    return out


def defined_tests(root: Path) -> dict[str, set[str]]:
    """Per crate with a `tests/distributed/` suite, the tests it defines."""
    return {
        suite.parent.parent.name: set().union(*(tests_in(p.read_text()) for p in suite.glob("*.rs")))
        for suite in root.glob("crates/*/tests/distributed")
    }


def self_test() -> int:
    src = "#[tokio::test]\n#[serial]\nasync fn a() {}\n#[test]\nfn b() {}\nfn helper() {}\n"
    assert tests_in(src) == {"a", "b"}, tests_in(src)
    assert problems({"c": Counter(["a", "b"])}, {"c": {"a", "b"}}) == []
    assert problems({"c": Counter(["a"])}, {"c": {"a", "b"}}) == ["c: `b` is in no distributed.yml leg"]
    assert problems({"c": Counter(["a", "b", "b"])}, {"c": {"a", "b"}}) == ["c: `b` is in 2 legs"]
    assert problems({"c": Counter(["a", "z"])}, {"c": {"a"}}) == ["c: leg filter `z` names no test"]
    print("check_distributed_legs: self-test passed")
    return 0


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    named = leg_tests(yaml.safe_load(WORKFLOW.read_text()))
    found = problems(named, defined_tests(Path(".")))
    for p in found:
        print(f"FAIL {p}")
    if not found:
        print(f"check_distributed_legs: {sum(len(c) for c in named.values())} distributed tests, each in one leg")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
