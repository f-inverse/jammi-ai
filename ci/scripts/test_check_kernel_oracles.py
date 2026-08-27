#!/usr/bin/env python3
"""Hermetic `unittest` suite for `check_kernel_oracles.py` — drives the real
pure functions (`find_fns`, `check_ko7`, `check_ko2`, `check_ko5`,
`reconcile_ops`, `shipped_ops_from_sources`, marker parsing) against
in-memory synthetic Rust source strings, never the real checkout — mirrors
`test_check_ci_guard_wiring.py`'s "drive the real entry points against
throwaway fixtures" shape for this repo's `test_*.py` gate-suite convention.

Run: `python3 ci/scripts/test_check_kernel_oracles.py`
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_kernel_oracles as ko  # noqa: E402

# The frozen tokenizer-invariant fixture corpus (esc: kernel-oracle-golden-
# fixture) — see `_load_tokenizer_fixture_corpus` and
# `TestTokenizerCommentStringInvariant` below for why this exists
# separately from `ko.scan_files()`'s LIVE tree scan.
_TOKENIZER_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "ci" / "fixtures" / "kernel-oracle-tokenizer"


def _load_tokenizer_fixture_corpus() -> dict[str, str]:
    """Every `.rs` file under the committed, FROZEN tokenizer fixture
    corpus (`ci/fixtures/kernel-oracle-tokenizer/`), keyed by its path
    relative to that directory (stable regardless of where the repo is
    checked out). Raises if the corpus directory is missing or empty —
    an uncomputable golden must fail closed, never silently pass on zero
    files.
    """
    if not _TOKENIZER_FIXTURE_DIR.is_dir():
        raise FileNotFoundError(f"tokenizer fixture corpus not found: {_TOKENIZER_FIXTURE_DIR}")
    texts: dict[str, str] = {}
    for path in sorted(_TOKENIZER_FIXTURE_DIR.rglob("*.rs")):
        rel = str(path.relative_to(_TOKENIZER_FIXTURE_DIR))
        texts[rel] = path.read_text(encoding="utf-8")
    if not texts:
        raise FileNotFoundError(f"tokenizer fixture corpus is empty: {_TOKENIZER_FIXTURE_DIR}")
    return texts


def _strip_rust_golden_hash(texts: dict[str, str]) -> str:
    """SHA256 over every `label`'s `ko._strip_rust` output, sorted by
    label, `\\0`-separated — the ONE golden-hashing recipe shared by the
    fixture-corpus golden test and `--regenerate-tokenizer-golden` below,
    so the two can never compute the hash two different ways.
    """
    h = hashlib.sha256()
    for label in sorted(texts):
        h.update(label.encode())
        h.update(b"\0")
        h.update(ko._strip_rust(texts[label]).encode())
        h.update(b"\0")
    return h.hexdigest()


def _regenerate_tokenizer_golden() -> int:
    """Recompute `_STRIP_RUST_GOLDEN_HASH`/`_STRIP_RUST_GOLDEN_FILE_COUNT`
    over the committed fixture corpus and rewrite this file's own two
    constants in place. Run: `python3 ci/scripts/test_check_kernel_oracles.py
    --regenerate-tokenizer-golden`, then inspect the resulting diff and
    commit it DELIBERATELY (a reviewed PR diff) — never to silence a
    failure. Refuses under CI (`CI` env var set): a golden that can
    silently regenerate itself in CI can never fail, defeating the point
    of pinning it at all.
    """
    if os.environ.get("CI"):
        print(
            "test_check_kernel_oracles.py: refusing --regenerate-tokenizer-golden under CI "
            "— run this locally and commit the resulting diff",
            file=sys.stderr,
        )
        return 2
    texts = _load_tokenizer_fixture_corpus()
    new_hash = _strip_rust_golden_hash(texts)
    new_count = len(texts)
    self_path = Path(__file__).resolve()
    src = self_path.read_text(encoding="utf-8")
    src, n_hash = re.subn(
        r'_STRIP_RUST_GOLDEN_HASH = "[0-9a-f]+"',
        f'_STRIP_RUST_GOLDEN_HASH = "{new_hash}"',
        src,
    )
    src, n_count = re.subn(
        r"_STRIP_RUST_GOLDEN_FILE_COUNT = \d+",
        f"_STRIP_RUST_GOLDEN_FILE_COUNT = {new_count}",
        src,
    )
    if n_hash != 1 or n_count != 1:
        print(
            f"test_check_kernel_oracles.py: expected exactly one occurrence each of the golden "
            f"constants, found {n_hash} hash / {n_count} count occurrence(s) — refusing to write",
            file=sys.stderr,
        )
        return 2
    self_path.write_text(src, encoding="utf-8")
    print(f"wrote _STRIP_RUST_GOLDEN_HASH={new_hash!r} _STRIP_RUST_GOLDEN_FILE_COUNT={new_count}")
    return 0


def _helpers_for(source_map: dict[str, str], entries: list[tuple[str, str]]) -> set[str]:
    """Round-4: helpers are DECLARED (a registry), never DISCOVERED — the
    fixture-local equivalent of `ci/kernel-oracle-helpers.txt`. `entries`
    is `[(file_label, fn_name), ...]`; returns the VERIFIED name set (a
    name whose shape check fails is silently absent, matching
    `verify_helper_registry`'s real contract — call `verify_helper_
    registry` directly when a test needs to see the failure reasons too).
    """
    names, _failures = ko.verify_helper_registry(entries, source_map)
    return names


class TestFnDiscovery(unittest.TestCase):
    def test_finds_test_fn_and_plain_fn(self) -> None:
        src = """
fn helper() -> Option<i32> { Some(1) }

#[test]
fn a_test() {
    let x = helper();
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        names = {f.name: f.is_test for f in fns}
        self.assertEqual(names, {"helper": False, "a_test": True})

    def test_tokio_test_attribute_counts_as_test(self) -> None:
        src = """
#[tokio::test]
async fn an_async_test() {
    return;
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        self.assertTrue(fns[0].is_test)


class TestHelperRegistry(unittest.TestCase):
    """Round 4 (lead probe): helpers are DECLARED, never DISCOVERED — a
    fn is never a require-gate just because its OWN body happens to look
    like one; only a NAME in `ci/kernel-oracle-helpers.txt` gates
    anything, and even then only after the registry's own shape check
    passes.
    """

    def test_unregistered_conforming_fn_never_gates(self) -> None:
        src = """
fn real_helper() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("no device");
    }
    None
}
"""
        # NOT in the registry at all — must never verify as a helper,
        # however conforming its own shape is.
        names, failures = ko.verify_helper_registry([], {"fixture.rs": src})
        self.assertEqual(names, set())
        self.assertEqual(failures, [])

    def test_registered_conforming_if_form_verifies(self) -> None:
        src = """
fn real_helper() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("no device");
    }
    None
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "real_helper")], {"fixture.rs": src})
        self.assertEqual(names, {("fixture.rs", "real_helper")})
        self.assertEqual(failures, [])

    def test_registered_fn_with_env_read_but_no_panic_is_a_registry_fail(self) -> None:
        src = """
fn decoy_only_env() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        None
    } else {
        None
    }
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "decoy_only_env")], {"fixture.rs": src})
        self.assertEqual(names, set())
        self.assertEqual(len(failures), 1)
        self.assertIn("decoy_only_env", failures[0])

    def test_registered_fn_with_panic_but_no_env_read_is_a_registry_fail(self) -> None:
        src = """
fn decoy_only_panic() -> Option<i32> {
    panic!("always");
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "decoy_only_panic")], {"fixture.rs": src})
        self.assertEqual(names, set())
        self.assertEqual(len(failures), 1)

    def test_registered_fn_not_found_in_file_is_a_registry_fail(self) -> None:
        names, failures = ko.verify_helper_registry([("fixture.rs", "nonexistent")], {"fixture.rs": "fn other() {}\n"})
        self.assertEqual(names, set())
        self.assertIn("fn not found in file", failures[0])

    def test_registered_file_not_scanned_is_a_registry_fail(self) -> None:
        names, failures = ko.verify_helper_registry([("missing.rs", "cuda_device")], {"fixture.rs": "fn f() {}\n"})
        self.assertEqual(names, set())
        self.assertIn("file not found among scanned files", failures[0])

    def test_match_form_env_read_verifies(self) -> None:
        # round-4 audit N3: a conforming gate written as `match` (not
        # `if`) on the env var — a round-3 REGRESSION versus round 2,
        # closed by accepting BOTH shapes explicitly.
        src = """
fn cuda_device() -> Option<i32> {
    match std::env::var_os("JAMMI_REQUIRE_CUDA") {
        Some(_) => panic!("required"),
        None => None,
    }
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "cuda_device")], {"fixture.rs": src})
        self.assertEqual(names, {("fixture.rs", "cuda_device")})
        self.assertEqual(failures, [])

    def test_unreachable_macro_counts_as_the_panic_arm(self) -> None:
        src = """
fn helper() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        unreachable!("no device");
    }
    None
}
"""
        names, _f = ko.verify_helper_registry([("fixture.rs", "helper")], {"fixture.rs": src})
        self.assertEqual(names, {("fixture.rs", "helper")})

    def test_option_env_macro_is_rejected_compile_time_only(self) -> None:
        # round-4 audit advisory: `option_env!` is resolved AT COMPILE
        # TIME by the compiler that BUILT this test binary — a
        # `JAMMI_REQUIRE_*` gate is a RUNTIME enforcement switch (the pod
        # lane exports it before `cargo test` runs, long after the binary
        # was already compiled), so a helper gated only this way would
        # silently never observe it. Runtime `std::env::var_os`/`var`
        # reads only — this is a registry FAIL, not a conforming shape.
        src = """
fn helper() -> Option<i32> {
    if option_env!("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("required");
    }
    None
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "helper")], {"fixture.rs": src})
        self.assertEqual(names, set())
        self.assertEqual(len(failures), 1)
        self.assertIn("COMPILE-TIME", failures[0])

    def test_env_var_argument_not_starting_with_jammi_require_is_a_registry_fail(self) -> None:
        src = """
fn helper() -> Option<i32> {
    if std::env::var_os("SOME_OTHER_VAR").is_some() {
        panic!("x");
    }
    None
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "helper")], {"fixture.rs": src})
        self.assertEqual(names, set())
        self.assertEqual(len(failures), 1)

    def test_expect_in_addition_to_panic_is_a_registry_fail(self) -> None:
        # round-4 F1: an UNRELATED .expect( inside the if-block, alongside
        # a real panic!, still isn't the canonical "EXACTLY one panic!
        # statement" shape.
        src = """
fn helper() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        let cfg = load_config().expect("config missing");
        panic!("required: {cfg}");
    }
    None
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "helper")], {"fixture.rs": src})
        self.assertEqual(names, set())
        self.assertEqual(len(failures), 1)

    def test_panic_in_an_uncalled_closure_is_a_registry_fail(self) -> None:
        # round-4 F2: a panic! sitting inside a closure that is never
        # invoked is not "the branch's own statement" at all.
        src = """
fn helper() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        let hard_fail = || panic!("required");
        let _unused = hard_fail;
    }
    None
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "helper")], {"fixture.rs": src})
        self.assertEqual(names, set())
        self.assertEqual(len(failures), 1)

    def test_shadowed_env_read_via_use_as_is_a_registry_fail(self) -> None:
        # round-4 F6: `use std::env::var as getenv;` then `getenv(...)` —
        # not the literal call syntax the shape check requires.
        src = """
use std::env::var as getenv;
fn helper() -> Option<i32> {
    if getenv("JAMMI_REQUIRE_CUDA").is_ok() { panic!("required"); }
    None
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "helper")], {"fixture.rs": src})
        self.assertEqual(names, set())
        self.assertEqual(len(failures), 1)

    def test_discarded_option_env_result_is_a_registry_fail(self) -> None:
        # the env-read is never the CONDITION of an if/match at all — its
        # Option result is simply discarded — so IF_ENV_READ_RE/MATCH_ENV_
        # READ_RE structurally cannot match this shape.
        src = """
fn launder() -> Option<u32> {
    let _ = option_env!("JAMMI_REQUIRE_CUDA");
    let d = make().expect("device");
    Some(d)
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "launder")], {"fixture.rs": src})
        self.assertEqual(names, set())
        self.assertEqual(len(failures), 1)

    def test_real_registry_file_parses_and_every_seeded_helper_verifies(self) -> None:
        # Round-4 audit advisory: pinning an EXACT entry count/set makes
        # this test a required-edit tax on every legitimate new helper
        # registration — the actual invariant this test protects is "every
        # entry that is IN the file loads, resolves, and shape-verifies",
        # not "the registry has exactly N entries." >= this seed set, each
        # individually shape-ok.
        entries = ko.load_helper_registry()
        seed = {
            ("crates/jammi-kernels/tests/cuda_parity.rs", "cuda_device"),
            ("crates/jammi-kernels/tests/flash_smoke.rs", "cuda_device"),
            ("crates/jammi-encoders/src/modernbert.rs", "growth_oracle_cuda_device"),
        }
        self.assertTrue(seed.issubset(set(entries)), f"seed entries missing from {entries}")
        source_texts = {rel: (ko.REPO_ROOT / rel).read_text() for rel, _name in entries}
        names, failures = ko.verify_helper_registry(entries, source_texts)
        self.assertEqual(failures, [])
        self.assertEqual(names, set(entries))


class TestKo7(unittest.TestCase):
    def _run(self, src: str) -> list[ko.UngatedSkip]:
        file_label = "fixture.rs"
        fns = ko.find_fns(src, file_label)
        # "cuda_device" is registered by default (every fixture below that
        # defines a real one names it that); "looks_like_a_gate"/
        # "ordinary_helper" are deliberately NOT registered — round-4:
        # a name absent from the registry never gates, however conforming.
        helpers = _helpers_for({file_label: src}, [(file_label, "cuda_device")])
        return ko.check_ko7(fns, helpers, {file_label: src})

    def test_gated_skip_is_clean(self) -> None:
        src = """
fn cuda_device() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("no device");
    }
    None
}

#[test]
fn a_test() {
    let Some(dev) = cuda_device() else {
        return;
    };
    let _ = dev;
}
"""
        self.assertEqual(self._run(src), [])

    def test_gated_skip_brace_tail_form_is_clean(self) -> None:
        src = """
fn cuda_device() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("no device");
    }
    None
}

#[test]
fn a_test() {
    let Some(dev) = cuda_device() else { return };
    let _ = dev;
}
"""
        self.assertEqual(self._run(src), [])

    def test_ungated_skip_is_flagged(self) -> None:
        src = """
fn cuda_device() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("no device");
    }
    None
}

#[test]
fn a_test() {
    if some_unrelated_condition() {
        return;
    }
}
"""
        findings = self._run(src)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].fn_name, "a_test")

    def test_skip_gated_by_a_non_registered_decoy_is_still_flagged(self) -> None:
        # a fn that merely LOOKS like a gate (returns Option, named
        # plausibly) but never reads a JAMMI_REQUIRE_* var or panics must
        # NOT be treated as a registered helper — the return stays ungated.
        src = """
fn looks_like_a_gate() -> Option<i32> {
    None
}

#[test]
fn a_test() {
    let Some(dev) = looks_like_a_gate() else {
        return;
    };
    let _ = dev;
}
"""
        findings = self._run(src)
        self.assertEqual(len(findings), 1)

    def test_skip_in_a_non_test_fn_is_not_scanned(self) -> None:
        src = """
fn ordinary_helper() {
    if true {
        return;
    }
}
"""
        self.assertEqual(self._run(src), [])

    def test_multiple_gated_skips_in_one_fn_all_clean(self) -> None:
        src = """
fn cuda_device() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("no device");
    }
    None
}

#[test]
fn a_test() {
    let Some(dev1) = cuda_device() else {
        return;
    };
    let Some(dev2) = cuda_device() else {
        return;
    };
    let _ = (dev1, dev2);
}
"""
        self.assertEqual(self._run(src), [])

    def test_helper_call_after_the_skip_does_not_count(self) -> None:
        # a registered helper exists, but the ONLY call to it in this fn
        # comes textually AFTER the return — must still be flagged (this is
        # the literal "dominated ... before it" requirement).
        src = """
fn cuda_device() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("no device");
    }
    None
}

#[test]
fn a_test() {
    if some_condition() {
        return;
    }
    let _ = cuda_device();
}
"""
        findings = self._run(src)
        self.assertEqual(len(findings), 1)


class TestMarkerParsing(unittest.TestCase):
    def test_parses_a_well_formed_marker(self) -> None:
        line = (
            "//! oracle-cell: op=layer_norm_fused leg=fwd_parity dtype=bf16 "
            "bounds=BF16_REL_TOL,BF16_ABS_FLOOR control=layer_norm_bf16_parity "
            "derived-on=seed1,seed2 asserted-on=seed3"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        self.assertEqual(m.op, "layer_norm_fused")
        self.assertEqual(m.leg, "fwd_parity")
        self.assertEqual(m.dtype, "bf16")
        self.assertEqual(m.bounds, ("BF16_REL_TOL", "BF16_ABS_FLOOR"))
        self.assertEqual(m.control, "layer_norm_bf16_parity")
        self.assertEqual(m.derived_on, ("seed1", "seed2"))
        self.assertEqual(m.asserted_on, ("seed3",))

    def test_none_seed_lists_parse_as_none(self) -> None:
        line = (
            "//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL "
            "control=none:documentation_only derived-on=none asserted-on=none"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        self.assertTrue(m.is_control_none)
        self.assertIsNone(m.derived_on)
        self.assertIsNone(m.asserted_on)

    def test_missing_field_raises(self) -> None:
        line = "//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL control=none:x"
        with self.assertRaises(ko.OracleError):
            ko.parse_marker_line(line, "fixture.rs", 1)

    def test_bad_dtype_raises(self) -> None:
        line = (
            "//! oracle-cell: op=rope_fused leg=fwd dtype=fp8 bounds=TOL "
            "control=none:x derived-on=none asserted-on=none"
        )
        with self.assertRaises(ko.OracleError):
            ko.parse_marker_line(line, "fixture.rs", 1)

    def test_marker_forged_inside_a_raw_string_does_not_parse(self) -> None:
        """Round-7 audit advisory (a): `parse_markers` scans `_strip_
        strings_only`'s view, not raw text — a `//! oracle-cell:
        ... control=none:fake` FORGED inside a raw string's CONTENT (never
        a real doc comment at all) must not be able to declare a real op
        DECLARED_UNCONTROLLED via a fabricated `control=none` opt-out.
        """
        src = (
            'const S: &str = r#"//! oracle-cell: op=rope_fused leg=fwd '
            "dtype=f32 bounds=TOL control=none:fake derived-on=none "
            'asserted-on=none"#;\n'
        )
        markers = ko.parse_markers(src, "fixture.rs")
        self.assertEqual(markers, [])

    def test_real_marker_still_parses_through_strip_strings_only(self) -> None:
        src = (
            "//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL "
            "control=none:documentation_only derived-on=none asserted-on=none\n"
        )
        markers = ko.parse_markers(src, "fixture.rs")
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0].op, "rope_fused")


class TestKo2(unittest.TestCase):
    def test_control_covering_every_bound_is_clean(self) -> None:
        src = """
//! oracle-cell: op=layer_norm_fused leg=fwd dtype=bf16 bounds=REL_TOL,ABS_FLOOR control=my_control derived-on=none asserted-on=none

const REL_TOL: f32 = 0.01;
const ABS_FLOOR: f32 = 0.01;

#[test]
fn my_control() {
    assert!(diff <= REL_TOL * scale + ABS_FLOOR);
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        markers = ko.parse_markers(src, "fixture.rs")
        findings = ko.check_ko2(markers, fns)
        self.assertEqual(findings, [])

    def test_control_missing_a_bound_is_flagged(self) -> None:
        src = """
//! oracle-cell: op=layer_norm_fused leg=fwd dtype=bf16 bounds=REL_TOL,ABS_FLOOR control=my_control derived-on=none asserted-on=none

#[test]
fn my_control() {
    assert!(diff <= REL_TOL * scale);
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        markers = ko.parse_markers(src, "fixture.rs")
        findings = ko.check_ko2(markers, fns)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].missing_bounds, ("ABS_FLOOR",))

    def test_control_none_is_skipped(self) -> None:
        src = """
//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL control=none:documentation_only derived-on=none asserted-on=none
"""
        fns = ko.find_fns(src, "fixture.rs")
        markers = ko.parse_markers(src, "fixture.rs")
        self.assertEqual(ko.check_ko2(markers, fns), [])

    def test_control_fn_not_found_is_flagged(self) -> None:
        src = """
//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL control=fn_does_not_exist derived-on=none asserted-on=none
"""
        fns = ko.find_fns(src, "fixture.rs")
        markers = ko.parse_markers(src, "fixture.rs")
        findings = ko.check_ko2(markers, fns)
        self.assertEqual(len(findings), 1)
        self.assertTrue(findings[0].control_not_found)


class TestKo5(unittest.TestCase):
    def test_disjoint_seeds_is_clean(self) -> None:
        line = (
            "//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=seed1,seed2 asserted-on=seed3,seed4"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        self.assertEqual(ko.check_ko5([m]), [])

    def test_overlapping_seeds_is_flagged(self) -> None:
        line = (
            "//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=seed1,seed2 asserted-on=seed2,seed3"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        self.assertEqual(ko.check_ko5([m]), [m])

    def test_either_side_none_is_vacuously_clean(self) -> None:
        line = (
            "//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=none asserted-on=seed2,seed3"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        self.assertEqual(ko.check_ko5([m]), [])


class TestReconciliation(unittest.TestCase):
    def test_all_pending_when_nothing_covered_or_excluded(self) -> None:
        shipped = {"a", "b", "c"}
        pending, failures = ko.reconcile_ops(shipped, {}, {}, {})
        self.assertEqual(failures, [])
        self.assertEqual(set(pending), shipped)

    def test_covered_op_is_not_pending(self) -> None:
        shipped = {"a", "b"}
        line = (
            "//! oracle-cell: op=a leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=none asserted-on=none"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        pending, failures = ko.reconcile_ops(shipped, {"a": [m]}, {}, {})
        self.assertEqual(failures, [])
        self.assertEqual(set(pending), {"b"})

    def test_unknown_op_in_covered_is_a_failure(self) -> None:
        shipped = {"a"}
        line = (
            "//! oracle-cell: op=ghost_op leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=none asserted-on=none"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        _pending, failures = ko.reconcile_ops(shipped, {"ghost_op": [m]}, {}, {})
        self.assertTrue(any("unknown op `ghost_op`" in f for f in failures))

    def test_double_claimed_op_is_a_failure(self) -> None:
        shipped = {"a"}
        line = (
            "//! oracle-cell: op=a leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=none asserted-on=none"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        _pending, failures = ko.reconcile_ops(shipped, {"a": [m]}, {"a": "reviewed exclusion"}, {})
        self.assertTrue(any("claimed both COVERED and STRUCTURALLY_EXCLUDED" in f for f in failures))


class TestShippedOpsScan(unittest.TestCase):
    def test_finds_counters_for_call_sites(self) -> None:
        sources = {
            "crates/jammi-encoders/src/modernbert.rs": 'LazyLock::new(|| counters_for("rope_fused"));',
            "crates/jammi-lora/src/lora_linear.rs": 'counters_for("lora_linear_fused")',
        }
        self.assertEqual(ko.shipped_ops_from_sources(sources), {"rope_fused", "lora_linear_fused"})

    def test_excludes_the_admission_definition_file(self) -> None:
        sources = {
            ko.ADMISSION_DEFINITION_FILE: 'counters_for("registry_test_op_a")',
            "crates/jammi-lora/src/lora_linear.rs": 'counters_for("lora_linear_fused")',
        }
        self.assertEqual(ko.shipped_ops_from_sources(sources), {"lora_linear_fused"})


class TestRunGateEndToEnd(unittest.TestCase):
    def test_clean_tree_passes_with_all_pending(self) -> None:
        sources = {
            "crates/jammi-kernels/tests/cuda_parity.rs": """
fn cuda_device() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("no device");
    }
    None
}

#[test]
fn a_test() {
    let Some(dev) = cuda_device() else {
        return;
    };
    let _ = dev;
}
""",
        }
        shipped = {"lora_linear_fused"}
        registry = [("crates/jammi-kernels/tests/cuda_parity.rs", "cuda_device")]
        ko7, ko2, ko5, covered, declared_uncontrolled, pending, recon_failures, registry_failures = ko.run_gate(
            sources, shipped, registry
        )
        self.assertEqual(ko7, [])
        self.assertEqual(ko2, [])
        self.assertEqual(ko5, [])
        self.assertEqual(covered, {})
        self.assertEqual(set(pending), shipped)
        self.assertEqual(recon_failures, [])
        self.assertEqual(registry_failures, [])

    def test_ungated_skip_surfaces_end_to_end(self) -> None:
        sources = {
            "crates/jammi-kernels/tests/cuda_parity.rs": """
#[test]
fn a_test() {
    return;
}
""",
        }
        ko7, _ko2, _ko5, _covered, _declared_uncontrolled, _pending, _recon, _regf = ko.run_gate(
            sources, set(), []
        )
        self.assertEqual(len(ko7), 1)


# --------------------------------------------------------------------------- #
# round-2 (adversarial-audit) fixes — items 1, 2, 3, 6, 8
# --------------------------------------------------------------------------- #
class TestCommentStringLaundering(unittest.TestCase):
    """Item 1: a helper name/env-read/panic mentioned only in a COMMENT must
    not register a helper; a return-skip-shaped substring inside a comment
    or string must not be mistaken for a real skip.
    """

    def test_helper_name_only_in_a_comment_is_ungated(self) -> None:
        src = """
fn cuda_device() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("no device");
    }
    None
}

#[test]
fn a_test() {
    // cuda_device() is called somewhere, trust me
    if some_condition() {
        return;
    }
}
"""
        file_label = "fixture.rs"
        fns = ko.find_fns(src, file_label)
        # cuda_device IS registered and DOES verify — proving the gap is
        # specifically that a COMMENT mentioning it is not a real CALL,
        # not that the helper itself fails to register.
        helpers = _helpers_for({file_label: src}, [(file_label, "cuda_device")])
        self.assertEqual(helpers, {(file_label, "cuda_device")})
        findings = ko.check_ko7(fns, helpers, {file_label: src})
        self.assertEqual(len(findings), 1)

    def test_env_and_panic_only_in_a_comment_is_a_registry_fail(self) -> None:
        src = """
// fn fake_helper() { std::env::var_os("JAMMI_REQUIRE_CUDA"); panic!("x"); }
fn fake_helper() -> Option<i32> {
    None
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "fake_helper")], {"fixture.rs": src})
        self.assertEqual(names, set())
        self.assertEqual(len(failures), 1)

    def test_env_and_panic_only_in_a_block_comment_is_a_registry_fail(self) -> None:
        src = """
/* std::env::var_os("JAMMI_REQUIRE_CUDA"); panic!("x"); */
fn fake_helper() -> Option<i32> {
    None
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "fake_helper")], {"fixture.rs": src})
        self.assertEqual(names, set())
        self.assertEqual(len(failures), 1)

    def test_return_skip_shaped_text_inside_a_string_is_not_a_real_skip(self) -> None:
        src = """
#[test]
fn a_test() {
    let message = "if you see this: return;";
    assert!(message.len() > 0);
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        findings = ko.check_ko7(fns, set(), {"fixture.rs": src})
        self.assertEqual(findings, [])

    def test_env_read_still_registers_when_the_call_and_panic_are_real_code(self) -> None:
        # the exact real-world shape (tests/cuda_parity.rs's own cuda_device)
        # — the string ARGUMENT to the env-read call is real data, not
        # something the stripper may blank, or registration becomes
        # structurally impossible.
        src = """
fn cuda_device() -> Option<i32> {
    match acquire() {
        Ok(d) => Some(d),
        Err(e) => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                panic!(
                    "cuda_parity: JAMMI_REQUIRE_CUDA is set but no CUDA device \\
                     could be acquired: {e}"
                );
            }
            None
        }
    }
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "cuda_device")], {"fixture.rs": src})
        self.assertEqual(names, {("fixture.rs", "cuda_device")})
        self.assertEqual(failures, [])

    def test_panic_shaped_text_inside_an_unrelated_string_is_a_registry_fail(self) -> None:
        # a fn that reads the right env var but whose ONLY "panic!(" text
        # is inside an unrelated string (never real code) must fail the
        # shape check.
        src = """
fn decoy() -> Option<i32> {
    let doc = "calling this would panic!(\\"boom\\") in theory";
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        return None;
    }
    let _ = doc;
    None
}
"""
        names, failures = ko.verify_helper_registry([("fixture.rs", "decoy")], {"fixture.rs": src})
        self.assertEqual(names, set())
        self.assertEqual(len(failures), 1)


class TestKo7PerSkipWindowing(unittest.TestCase):
    """Item 6: an earlier helper call gates only the skip(s) downstream of
    it, not every later skip unconditionally — the audit's fixture D shape
    (a gated device check followed by an unrelated, ungated FLASH_COMPILED-
    style return further down the same fn).
    """

    HELPER = """
fn cuda_device() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("no device");
    }
    None
}
"""

    def _run(self, body: str) -> list:
        src = self.HELPER + body
        fns = ko.find_fns(src, "fixture.rs")
        helpers = _helpers_for({"fixture.rs": src}, [("fixture.rs", "cuda_device")])
        return ko.check_ko7(fns, helpers, {"fixture.rs": src})

    def test_gated_check_followed_by_an_unrelated_ungated_return_reds(self) -> None:
        body = """
#[test]
fn a_test() {
    let Some(dev) = cuda_device() else {
        return;
    };
    let _ = dev;
    if !FLASH_COMPILED {
        return;
    }
    do_the_real_check();
}
"""
        findings = self._run(body)
        self.assertEqual(len(findings), 1)

    def test_two_independently_gated_skips_both_pass(self) -> None:
        body = """
#[test]
fn a_test() {
    let Some(dev1) = cuda_device() else {
        return;
    };
    let _ = dev1;
    let Some(dev2) = cuda_device() else {
        return;
    };
    let _ = dev2;
}
"""
        self.assertEqual(self._run(body), [])

    def test_return_ok_is_recognized_as_a_skip_shape(self) -> None:
        body = """
#[test]
fn a_test() -> Result<(), String> {
    if some_condition() {
        return Ok(());
    }
    Ok(())
}
"""
        findings = self._run(body)
        self.assertEqual(len(findings), 1)

    def test_return_err_is_recognized_as_a_skip_shape(self) -> None:
        body = """
#[test]
fn a_test() -> Result<(), String> {
    if some_condition() {
        return Err("skipped".to_string());
    }
    Ok(())
}
"""
        findings = self._run(body)
        self.assertEqual(len(findings), 1)

    def test_gated_return_ok_passes(self) -> None:
        body = """
#[test]
fn a_test() -> Result<(), String> {
    let Some(dev) = cuda_device() else {
        return Ok(());
    };
    let _ = dev;
    Ok(())
}
"""
        self.assertEqual(self._run(body), [])


class TestKo2FileScopedAssertionContext(unittest.TestCase):
    """Item 4: control fn resolved ONLY in the marker's own file; a bound
    must be used inside an assert!-family call or a comparison, directly or
    via one level of same-file helper indirection.
    """

    def test_comment_only_mention_of_the_bound_reds(self) -> None:
        src = """
//! oracle-cell: op=layer_norm_fused leg=fwd dtype=bf16 bounds=REL_TOL control=my_control derived-on=none asserted-on=none

const REL_TOL: f32 = 0.01;

#[test]
fn my_control() {
    // this checks REL_TOL somehow
    assert!(true);
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        markers = ko.parse_markers(src, "fixture.rs")
        findings = ko.check_ko2(markers, fns)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].missing_bounds, ("REL_TOL",))

    def test_declared_only_const_with_no_usage_reds(self) -> None:
        src = """
//! oracle-cell: op=layer_norm_fused leg=fwd dtype=bf16 bounds=REL_TOL control=my_control derived-on=none asserted-on=none

const REL_TOL: f32 = 0.01;

#[test]
fn my_control() {
    let _unused = REL_TOL;
    assert!(true);
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        markers = ko.parse_markers(src, "fixture.rs")
        findings = ko.check_ko2(markers, fns)
        self.assertEqual(len(findings), 1)

    def test_assertion_in_a_same_file_helper_called_by_the_control_passes(self) -> None:
        src = """
//! oracle-cell: op=layer_norm_fused leg=fwd dtype=bf16 bounds=REL_TOL control=my_control derived-on=none asserted-on=none

const REL_TOL: f32 = 0.01;

fn helper_assert(diff: f32) {
    assert!(diff <= REL_TOL);
}

#[test]
fn my_control() {
    helper_assert(0.001);
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        markers = ko.parse_markers(src, "fixture.rs")
        findings = ko.check_ko2(markers, fns)
        self.assertEqual(findings, [])

    def test_same_named_fn_in_another_file_is_not_consulted(self) -> None:
        marker_src = """
//! oracle-cell: op=layer_norm_fused leg=fwd dtype=bf16 bounds=REL_TOL control=my_control derived-on=none asserted-on=none

const REL_TOL: f32 = 0.01;
"""
        # `my_control` does not even exist in this file — a same-named fn
        # in `other.rs` (below) must NOT be consulted to satisfy it.
        other_src = """
const REL_TOL: f32 = 0.01;

#[test]
fn my_control() {
    assert!(0.001 <= REL_TOL);
}
"""
        fns = ko.find_fns(marker_src, "fixture.rs") + ko.find_fns(other_src, "other.rs")
        markers = ko.parse_markers(marker_src, "fixture.rs")
        findings = ko.check_ko2(markers, fns)
        self.assertEqual(len(findings), 1)
        self.assertTrue(findings[0].control_not_found)

    def test_direct_comparison_without_an_assert_macro_still_passes(self) -> None:
        src = """
//! oracle-cell: op=layer_norm_fused leg=fwd dtype=bf16 bounds=REL_TOL,ABS_FLOOR control=within_bound derived-on=none asserted-on=none

const REL_TOL: f64 = 0.01;
const ABS_FLOOR: f64 = 0.05;

fn within_bound(diff: f64, magnitude: f64) -> bool {
    diff <= ABS_FLOOR || (diff - ABS_FLOOR) / magnitude <= REL_TOL
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        markers = ko.parse_markers(src, "fixture.rs")
        findings = ko.check_ko2(markers, fns)
        self.assertEqual(findings, [])


class TestDeclaredUncontrolled(unittest.TestCase):
    """Item 3: a `control=none:<reason>` marker never moves an op into
    COVERED — it becomes DECLARED_UNCONTROLLED, a fourth reconciliation
    category; PENDING excludes it too.
    """

    def test_none_only_marker_is_declared_uncontrolled_not_covered(self) -> None:
        src = """
//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL control=none:documentation_only derived-on=none asserted-on=none
"""
        shipped = {"rope_fused", "other_op"}
        ko7, ko2, ko5, covered, declared_uncontrolled, pending, recon_failures, _regf = ko.run_gate(
            {"fixture.rs": src}, shipped, []
        )
        self.assertEqual(covered, {})
        self.assertIn("rope_fused", declared_uncontrolled)
        self.assertNotIn("rope_fused", pending)
        self.assertIn("other_op", pending)
        self.assertEqual(recon_failures, [])

    def test_a_controlled_and_a_none_marker_for_the_same_op_is_covered(self) -> None:
        src = """
//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL control=my_control derived-on=none asserted-on=none
//! oracle-cell: op=rope_fused leg=bwd dtype=f32 bounds=TOL control=none:no_backward_oracle_yet derived-on=none asserted-on=none

const TOL: f32 = 0.01;

#[test]
fn my_control() {
    assert!(0.001 <= TOL);
}
"""
        shipped = {"rope_fused"}
        ko7, ko2, ko5, covered, declared_uncontrolled, pending, recon_failures, _regf = ko.run_gate(
            {"fixture.rs": src}, shipped, []
        )
        self.assertIn("rope_fused", covered)
        self.assertNotIn("rope_fused", declared_uncontrolled)

    def test_reconcile_ops_reports_four_disjoint_categories(self) -> None:
        shipped = {"a", "b", "c", "d"}
        line = (
            "//! oracle-cell: op=a leg=fwd dtype=f32 bounds=TOL control=real_control "
            "derived-on=none asserted-on=none"
        )
        covered_marker = ko.parse_marker_line(line, "fixture.rs", 1)
        none_line = (
            "//! oracle-cell: op=b leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=none asserted-on=none"
        )
        none_marker = ko.parse_marker_line(none_line, "fixture.rs", 2)
        pending, failures = ko.reconcile_ops(
            shipped,
            {"a": [covered_marker]},
            {"c": "reviewed exclusion"},
            {"b": [none_marker]},
        )
        self.assertEqual(failures, [])
        self.assertEqual(set(pending), {"d"})


class TestKo5SeedNormalization(unittest.TestCase):
    """Item 8: seed tokens normalized to integers (42 == 042 == 0x2a) before
    the disjointness check.
    """

    def test_decimal_and_zero_padded_decimal_are_the_same_seed(self) -> None:
        line = (
            "//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=42 asserted-on=042"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        self.assertEqual(ko.check_ko5([m]), [m])

    def test_hex_and_decimal_are_the_same_seed(self) -> None:
        line = (
            "//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=0x2a asserted-on=42"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        self.assertEqual(ko.check_ko5([m]), [m])

    def test_distinct_integer_seeds_stay_disjoint(self) -> None:
        line = (
            "//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=42 asserted-on=43"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        self.assertEqual(ko.check_ko5([m]), [])

    def test_non_numeric_seed_tokens_still_compare_as_strings(self) -> None:
        line = (
            "//! oracle-cell: op=rope_fused leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=seedA asserted-on=seedA"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        self.assertEqual(ko.check_ko5([m]), [m])


# --------------------------------------------------------------------------- #
# round-3 (scoped re-audit of 8041c09) fixes — the ONE-stripper redesign
# --------------------------------------------------------------------------- #
HELPER_SRC = (
    'fn cuda_device() -> Option<u32> { if std::env::var("JAMMI_REQUIRE_CUDA")'
    '.is_ok() { panic!("x"); } None }\n'
)


class TestStripperCharAndNestedComments(unittest.TestCase):
    """The live instance: a char literal containing a double quote
    (`feature_table.rs:45`'s `.trim_matches('"')`) must not desync the
    stripper's quote-tracking for the rest of the file.
    """

    def test_char_literal_containing_a_double_quote_does_not_desync(self) -> None:
        src = HELPER_SRC + """
#[test]
fn t() {
    let q = '"';
    if q == 'x' { return; }
    assert!(false);
}
"""
        # must not raise, and must find the ungated skip past the char literal.
        fns = ko.find_fns(src, "f.rs")
        helpers = _helpers_for({"f.rs": src}, [("f.rs", "cuda_device")])
        findings = ko.check_ko7(fns, helpers, {"f.rs": src})
        self.assertEqual(len(findings), 1)

    def test_char_literal_with_escaped_quote_and_backslash_do_not_desync(self) -> None:
        src = """
fn t() {
    let a = '\\'';
    let b = '\\\\';
    let c = 'x';
    assert!(a != b && b != c);
}
"""
        # must not raise the desync tripwire.
        ko.check_fn_desync(src, "f.rs")

    def test_lifetime_is_left_alone_not_consumed_as_a_char_literal(self) -> None:
        src = """
fn t<'a>(x: &'a str) -> &'a str {
    x
}
"""
        ko.check_fn_desync(src, "f.rs")
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual([f.name for f in fns], ["t"])

    def test_nested_block_comment_hides_a_fake_helper_and_fn(self) -> None:
        src = """
/* outer /* inner */
fn fake_gate() -> Option<u32> { if std::env::var("JAMMI_REQUIRE_CUDA").is_ok() { panic!("x"); } None }
*/
#[test]
fn t() { let Some(d) = fake_gate() else { return; }; assert!(d>0); }
"""
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual([f.name for f in fns], ["t"])
        # "fake_gate" is entirely hidden inside a nested block comment --
        # find_fns never discovers it as a real fn at all -- but even if it
        # somehow were registered, item 1 means nothing but registry
        # membership can ever gate anything; confirm both.
        self.assertEqual(_helpers_for({"f.rs": src}, [("f.rs", "fake_gate")]), set())
        helpers = _helpers_for({"f.rs": src}, [])
        self.assertEqual(helpers, set())
        findings = ko.check_ko7(fns, helpers, {"f.rs": src})
        self.assertEqual(len(findings), 1)


class TestFnBodyBraceCountingOnStrippedText(unittest.TestCase):
    def test_unbalanced_braces_inside_a_string_do_not_truncate_the_fn_body(self) -> None:
        src = HELPER_SRC + """
#[test]
fn t() {
    println!("brace }} }} literal");
    if 1 > 0 { return; }
    assert!(false);
}
fn tail() { }
"""
        fns = ko.find_fns(src, "f.rs")
        names = [f.name for f in fns]
        self.assertIn("t", names)
        self.assertIn("tail", names)
        t_fn = [f for f in fns if f.name == "t"][0]
        self.assertIn("assert!(false)", t_fn.body)
        helpers = _helpers_for({"f.rs": src}, [("f.rs", "cuda_device")])
        findings = ko.check_ko7(fns, helpers, {"f.rs": src})
        self.assertEqual(len(findings), 1)


class TestFnKeywordDesyncCheck(unittest.TestCase):
    def test_clean_file_does_not_raise(self) -> None:
        ko.check_fn_desync(HELPER_SRC + "#[test]\nfn t() { assert!(true); }\n", "f.rs")

    def test_fn_shaped_text_inside_a_raw_string_raises(self) -> None:
        src = (
            'const SRC: &str = r#"fn rawgate() -> Option<u32> '
            '{ if std::env::var("JAMMI_REQUIRE_CUDA").is_ok() { panic!("x"); } None }"#;\n'
            "#[test]\n"
            "fn t() { let Some(d) = rawgate() else { return; }; assert!(d>0); }\n"
        )
        with self.assertRaises(ko.OracleError):
            ko.check_fn_desync(src, "f.rs")

    def test_fn_shaped_text_inside_a_plain_string_raises(self) -> None:
        src = 'fn t() { let s = "fn phantom() {}"; assert!(s.len() > 0); }\n'
        with self.assertRaises(ko.OracleError):
            ko.check_fn_desync(src, "f.rs")


class TestFnDesyncMarkerEscapeHatch(unittest.TestCase):
    """Round-3b (lead probe): a fail-closed desync check with NO review/
    escape mechanism is a check no conforming file can ever satisfy — the
    live instance: crates/jammi-kernels/tests/stateful_op_discipline.rs's
    own grep-discipline fixture string. Mirrors this repo's own
    `no-producer: <reason>` opt-out idiom.
    """

    def test_marker_on_the_same_line_satisfies(self) -> None:
        src = (
            'fn t() {\n'
            '    let s = "fn phantom() {}"; // kernel-oracles: fn-in-literal reviewed: grep fixture\n'
            "    assert!(s.len() > 0);\n"
            "}\n"
        )
        ko.check_fn_desync(src, "f.rs")  # must not raise

    def test_marker_on_the_line_above_satisfies(self) -> None:
        src = (
            "fn t() {\n"
            "    // kernel-oracles: fn-in-literal reviewed: grep fixture, not a real fn\n"
            '    let s = "fn phantom() {}";\n'
            "    assert!(s.len() > 0);\n"
            "}\n"
        )
        ko.check_fn_desync(src, "f.rs")  # must not raise

    def test_unmarked_desync_still_fails(self) -> None:
        src = 'fn t() {\n    let s = "fn phantom() {}";\n    assert!(s.len() > 0);\n}\n'
        with self.assertRaises(ko.OracleError):
            ko.check_fn_desync(src, "f.rs")

    def test_stale_marker_with_no_nearby_desync_fails(self) -> None:
        src = (
            "fn t() {\n"
            "    // kernel-oracles: fn-in-literal reviewed: nothing to review here anymore\n"
            "    let s = 1 + 1;\n"
            "    assert!(s == 2);\n"
            "}\n"
        )
        with self.assertRaises(ko.OracleError):
            ko.check_fn_desync(src, "f.rs")

    def test_raw_string_with_hash_levels_containing_fn_plus_marker_satisfies(self) -> None:
        src = (
            "// kernel-oracles: fn-in-literal reviewed: raw-string fixture text, not a real fn\n"
            'const SRC: &str = r##"fn rawgate() -> Option<u32> { None }"##;\n'
            "fn t() { assert!(SRC.len() > 0); }\n"
        )
        ko.check_fn_desync(src, "f.rs")  # must not raise

    def test_reproduces_the_stateful_op_discipline_shape(self) -> None:
        # the exact live instance the audit found (paraphrased, not copied).
        src = (
            "fn t() {\n"
            "    // kernel-oracles: fn-in-literal reviewed: grep-discipline fixture "
            "string, not a real fn declaration\n"
            '    let attention_block_text = "pub(crate) fn foo() { qkv.apply_op3(rope_pack, mask, op) }";\n'
            "    assert!(!attention_block_text.is_empty());\n"
            "}\n"
        )
        ko.check_fn_desync(src, "f.rs")  # must not raise


class TestKo2GenericExclusionDirectAdjacency(unittest.TestCase):
    def _covered(self, ctl_body: str, bounds: str = "TOL") -> bool:
        src = (
            f"//! oracle-cell: op=rope_fused leg=l dtype=f32 bounds={bounds} "
            f"control=ctl derived-on=1 asserted-on=2\n{ctl_body}"
        )
        fns = ko.find_fns(src, "f.rs")
        markers = ko.parse_markers(src, "f.rs")
        return ko.check_ko2(markers, fns) == []

    def test_direct_comparison_covers(self) -> None:
        self.assertTrue(self._covered('fn ctl() { if x.max(y) < TOL { panic!("bad"); } }'))

    def test_bare_declaration_misses(self) -> None:
        self.assertFalse(self._covered("fn ctl() { let x = TOL; use_it(x); }"))

    def test_turbofish_generic_mention_misses(self) -> None:
        self.assertFalse(self._covered("fn ctl() { let v = foo::<f32>(TOL); consume(v); }"))

    def test_type_ascription_generic_misses(self) -> None:
        self.assertFalse(self._covered("fn ctl() { let v: Vec<f32> = vec![TOL]; consume(v); }"))

    def test_bound_feeding_an_unrelated_later_comparison_misses(self) -> None:
        # TOL is an ARGUMENT to compute(), not itself an operand of the `>`
        # comparison — must not count as coverage.
        self.assertFalse(self._covered("fn ctl() { let f = compute(TOL) as u32 > 0; }"))


# --------------------------------------------------------------------------- #
# round-4 (lead probe of the class) — attribute-to-item ASSOCIATION (N1/item
# 2), adopted directly from scratchpad/audit-r3/rs/G1-G7 (session-local audit
# fixtures, untracked; paraphrased, not copied — the tests below are the
# tracked record).
# --------------------------------------------------------------------------- #
class TestAttributeAssociation(unittest.TestCase):
    HELPER = (
        'fn cuda_device() -> Option<u32> { if std::env::var_os("JAMMI_REQUIRE_CUDA")'
        '.is_some() { panic!("required"); } None }\n'
    )

    def _test_fn_names(self, src: str) -> list[str]:
        fns = ko.find_fns(src, "f.rs")
        return [f.name for f in fns if f.is_test]

    def test_g1_test_attr_same_line_as_fn(self) -> None:
        src = self.HELPER + """
#[test] fn parity_bf16() {
    let Some(d) = raw_device_or_none() else { return; };
    assert!(d.is_cuda());
}
"""
        self.assertEqual(self._test_fn_names(src), ["parity_bf16"])

    def test_g2_doc_comment_between_attr_and_fn(self) -> None:
        src = self.HELPER + """
#[test]
/// Parity oracle for bf16.
fn parity_bf16() {
    let Some(d) = raw_device_or_none() else { return; };
    assert!(d.is_cuda());
}
"""
        self.assertEqual(self._test_fn_names(src), ["parity_bf16"])

    def test_g3_line_comment_between_attr_and_fn(self) -> None:
        src = self.HELPER + """
#[test]
// no GPU in this lane
fn parity_bf16() {
    let Some(d) = raw_device_or_none() else { return; };
    assert!(d.is_cuda());
}
"""
        self.assertEqual(self._test_fn_names(src), ["parity_bf16"])

    def test_g4_blank_line_between_attr_and_fn(self) -> None:
        src = self.HELPER + """
#[test]

fn parity_bf16() {
    let Some(d) = raw_device_or_none() else { return; };
    assert!(d.is_cuda());
}
"""
        self.assertEqual(self._test_fn_names(src), ["parity_bf16"])

    def test_g5_multiline_attr_above_test_attr(self) -> None:
        src = self.HELPER + """
#[test]
#[cfg_attr(
    not(feature = "cuda"),
    ignore
)]
fn parity_bf16() {
    let Some(d) = raw_device_or_none() else { return; };
    assert!(d.is_cuda());
}
"""
        self.assertEqual(self._test_fn_names(src), ["parity_bf16"])

    def test_g6_control_sanity_plain_test(self) -> None:
        src = self.HELPER + """
#[test]
fn parity_bf16() {
    let Some(d) = raw_device_or_none() else { return; };
    assert!(d.is_cuda());
}
"""
        self.assertEqual(self._test_fn_names(src), ["parity_bf16"])

    def test_g7_desync_balanced_out_still_desyncs_per_line(self) -> None:
        # a phantom fn hidden in a raw string could, under a WHOLE-FILE
        # desync count, cancel out against a real fn the SAME check missed
        # elsewhere — the round-3b PER-LINE design closes this by
        # construction (each line's own count is checked independently,
        # so two unrelated lines can never cancel each other's desync).
        src = self.HELPER + """
const DOC: &str = r##"fn phantom_a() {}"##;
fn marker() { let s = "/*"; let _ = s; }
#[test]
fn parity_bf16() {
    let Some(d) = raw_device_or_none() else { return; };
    assert!(d.is_cuda());
}
"""
        with self.assertRaises(ko.OracleError):
            ko.find_fns(src, "f.rs")

    def test_totality_still_holds_across_all_g_fixtures(self) -> None:
        for name, src in [
            ("g1", self.HELPER + '#[test] fn t() { assert!(true); }\n'),
            ("g2", self.HELPER + '#[test]\n/// doc\nfn t() { assert!(true); }\n'),
            ("g6", self.HELPER + '#[test]\nfn t() { assert!(true); }\n'),
        ]:
            fns = ko.find_fns(src, "f.rs")  # must not raise (totality holds)
            self.assertEqual(sum(1 for f in fns if f.is_test), 1, name)

    def test_async_pub_fn_modifiers_do_not_break_association(self) -> None:
        src = """
#[tokio::test]
pub async fn an_async_test() {
    assert!(true);
}
"""
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual([f.name for f in fns if f.is_test], ["an_async_test"])

    def test_cfg_test_attribute_is_not_mistaken_for_a_test_attribute(self) -> None:
        # the live false-positive this round's own audit tooling
        # independently surfaced: crates/jammi-encoders/src/attention.rs's
        # `in_proj_weight`, gated `#[cfg(test)]` (test-BUILD-only code, any
        # kind of fn) — not itself a `#[test]` function.
        src = """
#[cfg(test)]
pub(crate) fn in_proj_weight(&self) -> &Tensor {
    self.in_proj.weight()
}
"""
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual([f.name for f in fns if f.is_test], [])


class TestReturnSkipProcessExit(unittest.TestCase):
    """Round-4 audit F15: `std::process::exit(` is a skip idiom RETURN_SKIP_RE
    must also recognize — a #[test] fn that skips by terminating the
    process instead of returning is just as invisible to KO-7 without it.
    """

    def test_process_exit_is_a_recognized_skip(self) -> None:
        src = (
            'fn cuda_device() -> Option<u32> { if std::env::var_os("JAMMI_REQUIRE_CUDA")'
            '.is_some() { panic!("required"); } None }\n'
            "#[test]\n"
            "fn t() {\n"
            "    let Some(d) = raw_device_or_none() else {\n"
            '        eprintln!("skipping");\n'
            "        std::process::exit(0);\n"
            "    };\n"
            "    assert!(d.is_cuda());\n"
            "}\n"
        )
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual(len(ko.RETURN_SKIP_RE.findall(fns[-1].body_stripped)), 1)
        helpers = _helpers_for({"f.rs": src}, [("f.rs", "cuda_device")])
        findings = ko.check_ko7(fns, helpers, {"f.rs": src})
        # raw_device_or_none is unregistered -- the process::exit skip is
        # correctly found and correctly stays ungated.
        self.assertEqual(len(findings), 1)


class TestConformanceFixtures(unittest.TestCase):
    """Round-4 audit item 5: the remaining scratchpad/audit-r3/rs/F*.rs
    fixtures not already exercised above (session-local audit fixtures,
    untracked; paraphrased, not copied — these tests are the tracked
    record), adopted with the audit's stated expected outcome.
    """

    def test_f3b_bare_assert_form_is_not_a_conforming_shape_stays_ungated(self) -> None:
        # only `if`/`match` on the env-read are the canonical shapes; a
        # bare `assert!(env_read.is_none(), ...)` is not one of them.
        src = """
fn cuda_device() -> Option<u32> {
    match acquire() {
        Ok(d) => Some(d),
        Err(e) => {
            assert!(
                std::env::var_os("JAMMI_REQUIRE_CUDA").is_none(),
                "JAMMI_REQUIRE_CUDA set but no device: {e}"
            );
            None
        }
    }
}
#[test]
fn parity_bf16() {
    let Some(d) = cuda_device() else { return; };
    assert!(d.is_cuda());
}
"""
        names, failures = ko.verify_helper_registry([("f.rs", "cuda_device")], {"f.rs": src})
        self.assertEqual(names, set())
        self.assertEqual(len(failures), 1)

    def test_f4_helper_call_in_a_dead_branch_is_a_documented_lexical_limitation(self) -> None:
        # a lexical scanner cannot establish REACHABILITY — a registered
        # helper's call sitting in an `if false { }` dead branch still
        # counts as "called before the skip" textually. Not fixed by the
        # registry (registration answers "is this name reviewed", not
        # "is this call path live") — pinned here as a KNOWN, out-of-scope
        # class limitation (same class as KO-2's H9), not silently
        # dropped.
        src = """
fn cuda_device() -> Option<u32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() { panic!("required"); }
    Device::new_cuda(0).ok()
}
#[test]
fn parity_bf16() {
    if false { let _ = cuda_device(); }
    let dev = raw_device_or_none();
    let Some(d) = dev else { return; };
    assert!(d.is_cuda());
}
"""
        helpers = _helpers_for({"f.rs": src}, [("f.rs", "cuda_device")])
        fns = ko.find_fns(src, "f.rs")
        findings = ko.check_ko7(fns, helpers, {"f.rs": src})
        self.assertEqual(findings, [])  # documented gap, not a regression

    def test_f7_raw_string_with_hashes_containing_a_real_helper_name_desyncs(self) -> None:
        src = (
            'fn cuda_device() -> Option<u32> { if std::env::var_os("JAMMI_REQUIRE_CUDA")'
            '.is_some() { panic!("required"); } None }\n'
            'const DOC: &str = r##"fn cuda_device() { if std::env::var("JAMMI_REQUIRE_CUDA")'
            '.is_some() { panic!("x"); } }"##;\n'
            "#[test]\n"
            "fn t() { let Some(d) = raw_device_or_none() else { return; }; assert!(d.is_cuda()); }\n"
        )
        with self.assertRaises(ko.OracleError):
            ko.find_fns(src, "f.rs")

    def test_f8_doc_comment_carrying_a_rustdoc_code_example_does_not_launder_a_skip(self) -> None:
        src = self.HELPER = (
            'fn cuda_device() -> Option<u32> { if std::env::var_os("JAMMI_REQUIRE_CUDA")'
            '.is_some() { panic!("required"); } None }\n'
            "//! Example:\n"
            "//! ```\n"
            "//! let Some(d) = cuda_device() else { return; };\n"
            "//! ```\n"
            "#[test]\n"
            "fn parity_bf16() {\n"
            "    let Some(d) = raw_device_or_none() else { return; };\n"
            "    assert!(d.is_cuda());\n"
            "}\n"
        )
        helpers = _helpers_for({"f.rs": src}, [("f.rs", "cuda_device")])
        self.assertEqual(helpers, {("f.rs", "cuda_device")})
        fns = ko.find_fns(src, "f.rs")
        findings = ko.check_ko7(fns, helpers, {"f.rs": src})
        self.assertEqual(len(findings), 1)

    def test_f9_nested_block_comment_with_quotes_and_braces_does_not_corrupt_scanning(self) -> None:
        src = (
            'fn cuda_device() -> Option<u32> { if std::env::var_os("JAMMI_REQUIRE_CUDA")'
            '.is_some() { panic!("required"); } None }\n'
            '/* outer /* inner with a quote " and a brace } */ still inside */\n'
            "#[test]\n"
            "fn parity_bf16() {\n"
            "    let Some(d) = raw_device_or_none() else { return; };\n"
            "    assert!(d.is_cuda());\n"
            "}\n"
        )
        helpers = _helpers_for({"f.rs": src}, [("f.rs", "cuda_device")])
        fns = ko.find_fns(src, "f.rs")
        findings = ko.check_ko7(fns, helpers, {"f.rs": src})
        self.assertEqual(len(findings), 1)

    def test_f10_byte_char_quote_and_lifetime_do_not_corrupt_scanning(self) -> None:
        src = (
            'fn cuda_device() -> Option<u32> { if std::env::var_os("JAMMI_REQUIRE_CUDA")'
            '.is_some() { panic!("required"); } None }\n'
            "fn parse<'a>(s: &'a str) -> &'a str {\n"
            "    let q = b'\"';\n"
            "    let r = b'\\'';\n"
            "    let _ = (q, r);\n"
            "    s.trim_matches('\"')\n"
            "}\n"
            "#[test]\n"
            "fn parity_bf16() {\n"
            "    let Some(d) = raw_device_or_none() else { return; };\n"
            "    assert!(d.is_cuda());\n"
            "}\n"
        )
        fns = ko.find_fns(src, "f.rs")  # must not raise the desync tripwire
        self.assertEqual({f.name for f in fns}, {"cuda_device", "parse", "parity_bf16"})

    def test_f11_unicode_escape_quote_char_does_not_corrupt_scanning(self) -> None:
        src = (
            'fn cuda_device() -> Option<u32> { if std::env::var_os("JAMMI_REQUIRE_CUDA")'
            '.is_some() { panic!("required"); } None }\n'
            "fn strip(s: &str) -> String {\n"
            '    s.replace(\'\\u{22}\', "")\n'
            "}\n"
            "#[test]\n"
            "fn parity_bf16() {\n"
            "    let Some(d) = raw_device_or_none() else { return; };\n"
            "    assert!(d.is_cuda());\n"
            "}\n"
        )
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual({f.name for f in fns}, {"cuda_device", "strip", "parity_bf16"})

    def test_f12_apostrophe_in_string_prose_does_not_corrupt_scanning(self) -> None:
        src = (
            'fn cuda_device() -> Option<u32> { if std::env::var_os("JAMMI_REQUIRE_CUDA")'
            '.is_some() { panic!("required"); } None }\n'
            "#[test]\n"
            "fn parity_bf16() {\n"
            '    eprintln!("the device isn\'t available");\n'
            "    let Some(d) = raw_device_or_none() else { return; };\n"
            "    assert!(d.is_cuda());\n"
            "}\n"
        )
        fns = ko.find_fns(src, "f.rs")  # must not raise
        self.assertEqual({f.name for f in fns}, {"cuda_device", "parity_bf16"})

    def test_f13_loop_label_is_not_confused_with_a_char_literal(self) -> None:
        src = (
            'fn cuda_device() -> Option<u32> { if std::env::var_os("JAMMI_REQUIRE_CUDA")'
            '.is_some() { panic!("required"); } None }\n'
            "#[test]\n"
            "fn parity_bf16() {\n"
            "    'outer: for i in 0..4 {\n"
            "        if i == 2 { break 'outer; }\n"
            "    }\n"
            "    let Some(d) = raw_device_or_none() else { return; };\n"
            "    assert!(d.is_cuda());\n"
            "}\n"
        )
        fns = ko.find_fns(src, "f.rs")  # must not raise
        self.assertEqual({f.name for f in fns}, {"cuda_device", "parity_bf16"})


# --------------------------------------------------------------------------- #
# round-4 (scoped re-audit of 07308fc) — CLASS A + advisory fixes, adopting
# every audit fixture (b1_registry.py, b2_totality.py, b34.py, fp.py,
# tot_fp.py, a_prior.py) as a permanent regression with the audit's own
# stated expected outcome.
# --------------------------------------------------------------------------- #
class TestFileScopedHelperGating(unittest.TestCase):
    """Item 1: `verify_helper_registry`/`check_ko7` are (file, fn)-scoped —
    a flat NAME set let an unregistered same-named fn in a DIFFERENT file
    "borrow" another file's review."""

    def test_b1j_a_different_files_own_unregistered_same_named_fn_does_not_gate(self) -> None:
        real = """
fn cuda_device() -> Option<u8> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() { panic!("req"); }
    Some(0)
}
"""
        a_src = real + """
#[test]
fn t_real() { let d = cuda_device(); let Some(_d) = d else { return; }; }
"""
        b_src = """
fn cuda_device() -> Option<u8> { None }
#[test]
fn t_vacuous() {
    let d = cuda_device();
    let Some(_d) = d else { return; };
}
"""
        sources = {"a.rs": a_src, "b.rs": b_src}
        verified, failures = ko.verify_helper_registry([("a.rs", "cuda_device")], sources)
        self.assertEqual(failures, [])
        fns = ko.find_fns(a_src, "a.rs") + ko.find_fns(b_src, "b.rs")
        findings = ko.check_ko7(fns, verified, sources)
        by_fn = {f.fn_name for f in findings}
        self.assertNotIn("t_real", by_fn)
        self.assertIn("t_vacuous", by_fn)

    def test_b1c_two_same_named_fns_in_one_file_is_a_registry_fail_not_a_pick(self) -> None:
        src = """
mod a {
    fn cuda_device() -> Option<u8> { Some(0) }
}
mod b {
    fn cuda_device() -> Option<u8> {
        if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() { panic!("req"); }
        Some(0)
    }
}
#[test]
fn t_mods() {
    let d = a::cuda_device();
    let Some(_d) = d else { return; };
}
"""
        verified, failures = ko.verify_helper_registry([("a.rs", "cuda_device")], {"a.rs": src})
        self.assertEqual(verified, set())
        self.assertEqual(len(failures), 1)
        self.assertIn("2 fns named", failures[0])
        fns = ko.find_fns(src, "a.rs")
        findings = ko.check_ko7(fns, verified, {"a.rs": src})
        self.assertEqual(len(findings), 1)

    def test_b1k_nested_fn_definition_never_called_does_not_gate(self) -> None:
        real = """
fn cuda_device() -> Option<u8> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() { panic!("req"); }
    Some(0)
}
"""
        src = real + """
#[test]
fn t_defonly() {
    fn cuda_device() -> Option<u8> { None }
    let x: Option<u8> = None;
    let Some(_d) = x else { return; };
}
"""
        # The nested fn IS a second same-named fn in this one file, so the
        # registry entry itself correctly fails too (item 1) — both
        # signals point the same direction (RED), not a contradiction.
        verified, failures = ko.verify_helper_registry([("a.rs", "cuda_device")], {"a.rs": src})
        self.assertTrue(failures)
        fns = ko.find_fns(src, "a.rs")
        findings = ko.check_ko7(fns, verified, {"a.rs": src})
        self.assertEqual(len(findings), 1)

    def test_qualified_and_method_calls_are_not_bare_calls(self) -> None:
        self.assertFalse(ko._is_bare_call("Other::cuda_device(", len("Other::")))
        self.assertFalse(ko._is_bare_call("self.cuda_device(", len("self.")))
        self.assertFalse(ko._is_bare_call("fn cuda_device(", len("fn ")))
        self.assertTrue(ko._is_bare_call("let d = cuda_device(", len("let d = ")))


class TestTotalityMacroMetavarAndRawIdent(unittest.TestCase):
    """Item 3: `FN_HEAD_RE` accepts `fn $name(` (macro_rules! generator
    template) and `fn r#match(` (raw ident) — both previously desynced
    totality with no remedy."""

    def test_macro_rules_test_generator_template_is_computable(self) -> None:
        src = """
macro_rules! gen_case {
    ($name:ident, $v:expr) => {
        #[test]
        fn $name() {
            assert_eq!($v, $v);
        }
    };
}
gen_case!(case_a, 1);
gen_case!(case_b, 2);
"""
        fns = ko.find_fns(src, "f.rs")  # must not raise
        self.assertEqual([f.name for f in fns if f.is_test], ["$name"])

    def test_raw_ident_test_fn_is_computable(self) -> None:
        src = "#[test]\nfn r#match() { assert!(true); }\n"
        fns = ko.find_fns(src, "f.rs")  # must not raise
        self.assertEqual([f.name for f in fns if f.is_test], ["r#match"])

    def test_marker_escape_hatch_resolves_a_genuine_mismatch(self) -> None:
        src = (
            "#[test]\n#[test]\nfn t() { assert!(true); }\n"
            "// kernel-oracles: test-attr reviewed: two stacked #[test]-shaped "
            "attributes before one fn, fixture-reviewed\n"
        )
        fns = ko.find_fns(src, "f.rs")  # must not raise (1 marker == delta of 1)
        self.assertEqual([f.name for f in fns if f.is_test], ["t"])

    def test_stale_marker_when_totality_already_balances_fails(self) -> None:
        src = (
            "#[test]\nfn t() { assert!(true); }\n"
            "// kernel-oracles: test-attr reviewed: nothing to review\n"
        )
        with self.assertRaises(ko.OracleError):
            ko.find_fns(src, "f.rs")

    def test_wrong_marker_count_still_fails(self) -> None:
        src = "#[test]\n#[test]\nfn t() { assert!(true); }\n"  # delta=1, 0 markers
        with self.assertRaises(ko.OracleError):
            ko.find_fns(src, "f.rs")


class TestNestedScopeSkipExclusion(unittest.TestCase):
    """Advisory: a `return`/`process::exit` inside a closure or a nested
    `fn` item does not skip the TEST itself."""

    def test_for_each_closure_return_is_not_a_test_skip(self) -> None:
        src = (
            "#[test]\nfn t() { v.iter().for_each(|x| { "
            "if *x == 0 { return; } total += x; }); }\n"
        )
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual(ko.check_ko7(fns, set(), {"f.rs": src}), [])

    def test_closure_returning_ok_is_not_a_test_skip(self) -> None:
        src = (
            "#[test]\nfn t() { let f = || -> Result<(), u8> { "
            "if flag { return Ok(()); } Err(1) }; assert!(f().is_ok()); }\n"
        )
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual(ko.check_ko7(fns, set(), {"f.rs": src}), [])

    def test_nested_fn_item_return_is_not_a_test_skip(self) -> None:
        src = (
            "#[test]\nfn t() { fn helper(x: u8) { "
            'if x == 0 { return; } println!("{x}"); } helper(1); }\n'
        )
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual(ko.check_ko7(fns, set(), {"f.rs": src}), [])

    def test_top_level_return_inside_an_if_inside_the_test_body_still_skips(self) -> None:
        src = "#[test]\nfn t() { let x: Option<u8> = None; let Some(_d) = x else { return; }; }\n"
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual(len(ko.check_ko7(fns, set(), {"f.rs": src})), 1)


class TestProcessExitSpellings(unittest.TestCase):
    """Advisory: bare `process::exit(` (no `std::` prefix) is always
    detected; a bare, unqualified `exit(` is detected only when the file
    imports it via `use std::process::exit;` (or a `{.., exit, ..}` list
    import) — an unqualified `exit(` with no import is at least as likely
    a locally-defined/shadowed name."""

    def test_bare_process_colon_colon_exit_is_detected(self) -> None:
        src = "#[test]\nfn t() { if cond { process::exit(1); } }\n"
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual(len(ko.check_ko7(fns, set(), {"f.rs": src})), 1)

    def test_bare_exit_without_import_is_not_detected(self) -> None:
        src = "#[test]\nfn t() { if cond { exit(1); } }\n"
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual(ko.check_ko7(fns, set(), {"f.rs": src}), [])

    def test_bare_exit_with_import_is_detected(self) -> None:
        src = (
            "use std::process::exit;\n"
            "#[test]\nfn t() { if cond { exit(1); } }\n"
        )
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual(len(ko.check_ko7(fns, set(), {"f.rs": src})), 1)

    def test_bare_exit_with_list_import_is_detected(self) -> None:
        src = (
            "use std::process::{self, exit};\n"
            "#[test]\nfn t() { if cond { exit(1); } }\n"
        )
        fns = ko.find_fns(src, "f.rs")
        self.assertEqual(len(ko.check_ko7(fns, set(), {"f.rs": src})), 1)


class TestHelperShapeGuardsAndConjuncts(unittest.TestCase):
    """Advisory: a match-arm GUARD, or any trailing/leading condition
    beyond the bare env-read `.is_some()`/`.is_ok()`, is rejected — a
    conjunct like `&& false` or an inverted `.is_none()` previously still
    verified even though the panic could never (or never should) fire."""

    def test_always_false_conjunct_is_rejected(self) -> None:
        src = 'fn h() { if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() && false { panic!("x"); } }'
        ok, _reason = ko.helper_shape_ok(ko._strip_rust(src))
        self.assertFalse(ok)

    def test_inverted_is_none_condition_is_rejected(self) -> None:
        src = 'fn h() { if std::env::var_os("JAMMI_REQUIRE_CUDA").is_none() { panic!("x"); } }'
        ok, _reason = ko.helper_shape_ok(ko._strip_rust(src))
        self.assertFalse(ok)

    def test_plain_is_some_if_form_still_verifies(self) -> None:
        src = 'fn h() { if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() { panic!("x"); } }'
        ok, _reason = ko.helper_shape_ok(ko._strip_rust(src))
        self.assertTrue(ok)

    def test_guarded_match_arm_is_rejected(self) -> None:
        src = (
            'fn h() { match std::env::var("JAMMI_REQUIRE_CUDA") { '
            'Ok(v) if v == "impossible" => panic!("x"), _ => {} } }'
        )
        ok, _reason = ko.helper_shape_ok(ko._strip_rust(src))
        self.assertFalse(ok)

    def test_unguarded_match_arm_still_verifies(self) -> None:
        src = (
            'fn h() { match std::env::var_os("JAMMI_REQUIRE_CUDA") { '
            "Some(_) => panic!(\"x\"), None => {} } }"
        )
        ok, _reason = ko.helper_shape_ok(ko._strip_rust(src))
        self.assertTrue(ok)


class TestRegistryFailureLineNumbersAndDuplicates(unittest.TestCase):
    """Advisory: a REGISTRY FAIL names `file:LINE`; an exact duplicate
    registry line is flagged, not silently deduplicated."""

    def test_shape_fail_message_names_the_line(self) -> None:
        src = "\n\nfn helper() -> Option<i32> { None }\n"
        _names, failures = ko.verify_helper_registry([("fixture.rs", "helper")], {"fixture.rs": src})
        self.assertEqual(len(failures), 1)
        self.assertIn("fixture.rs:3::helper:", failures[0])

    def test_duplicate_ambiguous_fail_message_names_both_lines(self) -> None:
        src = "fn helper() {}\nfn helper() {}\n"
        _names, failures = ko.verify_helper_registry([("fixture.rs", "helper")], {"fixture.rs": src})
        self.assertEqual(len(failures), 1)
        self.assertIn("lines 1, 2", failures[0])

    def test_duplicate_registry_line_raises(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("a.rs::helper\na.rs::helper\n")
            path = Path(f.name)
        try:
            with self.assertRaises(ko.OracleError):
                ko.load_helper_registry(path)
        finally:
            path.unlink()


class TestRecursiveScanRoots(unittest.TestCase):
    """Advisory: the two scan roots are walked RECURSIVELY (not widened —
    `crates/jammi-encoders/src/context/*.rs` is a real subdirectory of the
    already-in-scope `crates/jammi-encoders/src/` that a flat glob
    silently missed)."""

    def test_context_subdirectory_is_scanned_on_the_real_tree(self) -> None:
        texts = ko.scan_files()
        self.assertIn("crates/jammi-encoders/src/context/attention.rs", texts)


# --------------------------------------------------------------------------- #
# round-5 (scoped re-audit of c9b85cd) — NF-1/NF-2 (class A) + advisories,
# adopting every audit fixture (partB_bcefg.py, partB_ad.py, partB_misc.py)
# as a permanent regression with the audit's own stated expected outcome.
# --------------------------------------------------------------------------- #
class TestReturnCommaSkipShape(unittest.TestCase):
    """NF-1: a match-arm `return` tail expression, terminated by the arm's
    own `,` rather than `;`/`}`, was invisible to RETURN_SKIP_RE — `match
    cuda_device() { Some(d) => d, None => return, }` reported 0 skips and
    PASSed vacuously."""

    def test_bc5_match_arm_return_comma_is_detected(self) -> None:
        src = (
            "fn cuda_device() -> Option<u8> { Some(0) }\n"
            "#[test]\nfn t() {\n"
            "  let _d = match cuda_device() { Some(d) => d, None => return, };\n"
            "  assert!(true);\n}\n"
        )
        fns = ko.find_fns(src, "a.rs")
        self.assertEqual(len(ko.check_ko7(fns, set(), {"a.rs": src})), 1)

    def test_bd2_match_arm_return_comma_via_full_gate(self) -> None:
        src = "#[test]\nfn t() { let x=1; match x { 0 => return, _ => {} } assert!(true); }\n"
        ko7, *_rest = ko.run_gate({"a.rs": src}, set(), [])
        self.assertEqual(len(ko7), 1)

    def test_return_skip_re_still_matches_semicolon_and_brace_tail(self) -> None:
        self.assertEqual(ko.RETURN_SKIP_RE.findall("return;"), ["return;"])
        self.assertEqual(ko.RETURN_SKIP_RE.findall("None => return }"), ["return }"])
        self.assertEqual(ko.RETURN_SKIP_RE.findall("None => return, }"), ["return,"])


class TestAsyncBlockReturnCountsFailClosed(unittest.TestCase):
    """Round-6 audit reversal of round-5's A1 fix: an async block's
    `return` is lexically INDISTINGUISHABLE between a genuinely DROPPED
    future (over-report is safe) and a future that IS the test body,
    driven synchronously via `block_on`/`rt.block_on(async { .. return;
    .. })` (excluding it there is a real detection-power regression — a
    live oracle vacuously passes). Choosing fail-CLOSED: async-block
    returns always count, at the enclosing fn's own depth."""

    HELPER = (
        'fn cuda_device() -> Option<u8> {\n'
        '  if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() { panic!("req"); }\n'
        "  Some(0)\n}\n"
    )

    def test_block_on_async_return_must_count(self) -> None:
        src = self.HELPER + (
            "#[test]\nfn t() {\n"
            "  futures::executor::block_on(async {\n"
            "    let Some(_d) = cuda_device() else { return; };\n"
            "    assert!(true);\n  });\n}\n"
        )
        fns = ko.find_fns(src, "a.rs")
        self.assertEqual(len(ko.check_ko7(fns, set(), {"a.rs": src})), 1)

    def test_rt_block_on_async_move_return_must_count(self) -> None:
        src = self.HELPER + (
            "#[test]\nfn t() {\n"
            "  let rt = tokio::runtime::Runtime::new().unwrap();\n"
            "  rt.block_on(async move {\n"
            "    let Some(_d) = cuda_device() else { return; };\n"
            "    assert!(true);\n  });\n}\n"
        )
        fns = ko.find_fns(src, "a.rs")
        self.assertEqual(len(ko.check_ko7(fns, set(), {"a.rs": src})), 1)

    def test_block_on_async_return_no_helper_must_count(self) -> None:
        src = (
            "#[test]\nfn t() {\n"
            "  futures::executor::block_on(async {\n"
            "    if 1 > 2 { return; }\n    assert!(true);\n  });\n}\n"
        )
        fns = ko.find_fns(src, "a.rs")
        self.assertEqual(len(ko.check_ko7(fns, set(), {"a.rs": src})), 1)

    def test_dropped_future_return_counts_too_documented_over_report(self) -> None:
        # `async move { return; }; drop(f);` — the future is dropped,
        # never polled to completion, so this `return` never really skips
        # the test. Fail-closed still counts it (over-report, not a bug):
        # a human can see the false RED and gate it or restructure.
        src = "#[test]\nfn t() { let f = async move { return; }; drop(f); }\n"
        fns = ko.find_fns(src, "a.rs")
        self.assertEqual(len(ko.check_ko7(fns, set(), {"a.rs": src})), 1)

    def test_bare_async_return_counts(self) -> None:
        src = "#[test]\nfn t() { let f = async { return; }; drop(f); }\n"
        fns = ko.find_fns(src, "a.rs")
        self.assertEqual(len(ko.check_ko7(fns, set(), {"a.rs": src})), 1)

    def test_block_bodied_closure_return_still_excluded(self) -> None:
        # The ONE exclusion round-6 keeps: a `|params| { .. }` closure.
        src = "#[test]\nfn t() { let f = |x: u8| { if x>0 { return; } }; f(1); }\n"
        fns = ko.find_fns(src, "a.rs")
        self.assertEqual(ko.check_ko7(fns, set(), {"a.rs": src}), [])

    def test_expression_bodied_closure_return_over_reports_documented(self) -> None:
        # Pre-existing, unrelated to the async reversal: no `{` right
        # after `|params|`, so this isn't a "block-bodied closure" and its
        # `return` (which really does exit only the closure) still counts.
        src = "#[test]\nfn t() { let f = |x: u8| match x { _ => return, }; f(1); }\n"
        fns = ko.find_fns(src, "a.rs")
        self.assertEqual(len(ko.check_ko7(fns, set(), {"a.rs": src})), 1)


class TestMarkerNotInsideAStringLiteral(unittest.TestCase):
    """A3: a reviewed marker is matched against `_strip_strings_only`'s
    view (comments/code verbatim, string CONTENT blanked) — a marker-
    shaped SUBSTRING sitting inside some OTHER string literal on the line
    cannot masquerade as a real reviewed comment and suppress a genuine
    desync/totality mismatch."""

    def test_marker_shaped_text_inside_a_string_does_not_suppress_a_real_desync(self) -> None:
        src = (
            '#[test]\nfn t() {\n'
            '  let a = "pub fn foo() {}"; let b = "// kernel-oracles: fn-in-literal reviewed: fake";\n'
            "  assert!(a.len()+b.len()>0);\n}\n"
        )
        with self.assertRaises(ko.OracleError):
            ko.find_fns(src, "a.rs")

    def test_a_real_trailing_comment_marker_still_resolves(self) -> None:
        src = (
            '#[test]\nfn t() { let s = "pub fn foo() {}"; '
            "// kernel-oracles: fn-in-literal reviewed: fixture\n"
            " assert!(s.len()>0); }\n"
        )
        fns = ko.find_fns(src, "a.rs")  # must not raise
        self.assertEqual([f.name for f in fns], ["t"])

    def test_strip_strings_only_blanks_string_content_leaves_comments_verbatim(self) -> None:
        src = 'fn a() {}\n// a real comment\nlet s = "blank me";\n'
        out = ko._strip_strings_only(src)
        self.assertIn("// a real comment", out)
        self.assertIn("fn a() {}", out)
        self.assertNotIn("blank me", out)


class TestAliasedHelperCallFailsClosed(unittest.TestCase):
    """A2 (documented, not a bug): `let g = cuda_device; g();` — a
    same-file ALIAS binding, then calling the alias — does not gate,
    because the call site is textually `g(`, not `cuda_device(`. Fails
    CLOSED (reads as ungated), never a false green."""

    def test_alias_then_call_does_not_gate(self) -> None:
        real = (
            'fn cuda_device() -> Option<u8> {\n'
            '  if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() { panic!("req"); }\n'
            "  Some(0)\n}\n"
        )
        src = real + (
            "#[test]\nfn t() { let g = cuda_device; let d = g(); "
            "let Some(_x) = d else { return; }; }\n"
        )
        verified, failures = ko.verify_helper_registry([("a.rs", "cuda_device")], {"a.rs": src})
        self.assertEqual(failures, [])
        fns = ko.find_fns(src, "a.rs")
        self.assertEqual(len(ko.check_ko7(fns, verified, {"a.rs": src})), 1)


# --------------------------------------------------------------------------- #
# round-6 (narrow re-audit of d32e345) — the two class-A interaction findings.
# --------------------------------------------------------------------------- #
class TestTokenizerCommentStringInvariant(unittest.TestCase):
    """Item 2: `_strip_strings_only` used to be a hand-copied scan with NO
    comment state — a `"` sitting inside a real `//` comment's PROSE
    flipped its string-open/close parity for everything after it, which
    could resurrect a forged marker-shaped string as if it were a real
    comment. `_tokenize_rust` is the ONE scanner both `_strip_rust` and
    `_strip_strings_only` now render from, so their comment/string
    BOUNDARIES cannot diverge."""

    def test_forged_marker_resurrected_by_an_odd_quote_comment_still_raises(self) -> None:
        src = (
            "#[test]\nfn t() {\n"
            '  // a comment with an unmatched " quote\n'
            '  let a = "pub fn foo() {}"; let b = "// kernel-oracles: fn-in-literal reviewed: fake";\n'
            "  assert!(a.len()+b.len()>0);\n}\n"
        )
        with self.assertRaises(ko.OracleError):
            ko.find_fns(src, "a.rs")

    def test_string_content_stays_blanked_in_strip_strings_only_independently_verified(self) -> None:
        """Round-7 audit item (c): the PRIOR version of this test iterated
        `_tokenize_rust`'s OWN token classification and checked
        `_render_tokens` obeyed it — tautological (the auditor's M1
        mutant: the plain-string branch in `_tokenize_rust` emitting
        `_TOK_COMMENT` instead of `_TOK_STRING` changes what tokens EXIST,
        so a loop driven by that same classification never visits the
        broken span and still passes). This version identifies STRING
        positions via TWO sources that never touch `_tokenize_rust`:
        `_strip_rust` (blanks comments AND strings) and `_strip_comments_
        only_independent` (a deliberately SEPARATE, simpler scanner with
        NO string/char state at all, blanking ONLY comments) — a position
        blanked by the first but NOT the second is, independently, inside
        a STRING (or char literal). M1 fails this: a plain string wrongly
        tagged `_TOK_COMMENT` is rendered VERBATIM by `_strip_strings_
        only` (comments are kept there), leaking its content through at
        exactly these positions.
        """
        texts = ko.scan_files()
        leaks = []
        for label, src in texts.items():
            rust_stripped = ko._strip_rust(src)
            comments_only = ko._strip_comments_only_independent(src)
            keep = ko._strip_strings_only(src)
            n = min(len(rust_stripped), len(comments_only), len(keep), len(src))
            for i in range(n):
                c = src[i]
                if c in " \t\n":
                    continue
                is_string_pos = rust_stripped[i] != c and comments_only[i] == c
                if is_string_pos and keep[i] == c:
                    leaks.append((label, src.count("\n", 0, i) + 1, c))
        self.assertEqual(leaks[:5], [], f"{len(leaks)} string-content position(s) leaked verbatim")

    # Round-7 audit item (b): the prior version of this test only checked
    # length/newline-count preservation, which a WRONG (but still
    # length-preserving) blanking bug would pass trivially. A real golden:
    # SHA256 over every scanned file's own `_strip_rust` output (sorted by
    # label, `\0`-separated). This used to run over `ko.scan_files()` — the
    # LIVE `crates/jammi-kernels/tests/` + `crates/jammi-encoders/src/`
    # tree — which pinned the TREE'S CONTENTS, not the tokenizer's
    # behaviour: adding an oracle test file (#398, 37 != 36) or editing one
    # doc comment under either scanned root (#396, `modernbert.rs`'s
    # citation line) moved the hash for a reason that had nothing to do
    # with `_strip_rust` itself, reddening every unrelated PR that touched
    # those roots. The golden now runs over the COMMITTED, FROZEN fixture
    # corpus at `ci/fixtures/kernel-oracle-tokenizer/` (every comment/
    # string form the tokenizer handles: block/line/doc comments, nested
    # block comments, raw/byte strings, strings containing `//`/`/*`,
    # `#[cfg(...)]` attribute strings, char/byte-char literals including
    # quote characters, and a lifetime that is deliberately NOT a char
    # literal) instead — a change to `_strip_rust`'s ACTUAL blanked/kept
    # content still moves this hash (that power is unchanged), but the
    # live tree's file count/contents no longer can. Update
    # `_STRIP_RUST_GOLDEN_HASH`/`_STRIP_RUST_GOLDEN_FILE_COUNT`
    # DELIBERATELY (a reviewed PR diff) — via `python3 ci/scripts/
    # test_check_kernel_oracles.py --regenerate-tokenizer-golden`, which
    # refuses under CI — when `_strip_rust`'s real behavior on the corpus
    # is meant to change, or the corpus itself gains/loses a fixture file;
    # never to silence an unrelated failure.
    _STRIP_RUST_GOLDEN_HASH = "e7272dfd6939f9aede59c14db2be1706d98c6d36e9c3cc4a4f8e4b2a69fd6d6d"
    _STRIP_RUST_GOLDEN_FILE_COUNT = 3

    # The live-tree file count observed on `main` at the time this floor
    # was set — NOT a pin. `test_real_tree_scan_meets_floor_and_every_
    # file_tokenizes_without_error` below only asserts `>=` this, so it
    # grows freely as files are added and never reddens on that alone.
    _REAL_TREE_MIN_FILE_COUNT = 36

    def test_strip_rust_output_matches_the_golden_hash_on_fixture_corpus(self) -> None:
        texts = _load_tokenizer_fixture_corpus()
        self.assertEqual(len(texts), self._STRIP_RUST_GOLDEN_FILE_COUNT)
        self.assertEqual(_strip_rust_golden_hash(texts), self._STRIP_RUST_GOLDEN_HASH)

    def test_real_tree_scan_meets_floor_and_every_file_tokenizes_without_error(self) -> None:
        """Companion, NON-golden assertion (kept alongside the fixture-
        corpus golden above, never merged into it): the live-tree scan
        keeps being OBSERVED — at least `_REAL_TREE_MIN_FILE_COUNT` files,
        and every one of them tokenizes without raising and reconstructs
        EXACTLY byte-for-byte from `_tokenize_rust`'s own token spans (the
        tokenizer covers the whole source contiguously, with no gap or
        overlap) — without PINNING the live tree's byte content. Adding an
        oracle test file only grows the observed count (never reddens
        this); an unrelated doc-comment edit changes no assertion here at
        all, since nothing about this test depends on any file's exact
        text.
        """
        texts = ko.scan_files()
        self.assertGreaterEqual(len(texts), self._REAL_TREE_MIN_FILE_COUNT)
        for label, src in texts.items():
            tokens = ko._tokenize_rust(src)
            reconstructed = "".join(src[start:end] for _kind, start, end, _cs, _ce in tokens)
            self.assertEqual(reconstructed, src, f"{label}: tokenizer did not cover the source contiguously")

    def test_tokenize_rust_distinguishes_comment_from_string(self) -> None:
        src = 'fn a() {}\n// a real comment\nlet s = "blank me";\n'
        kinds = {t[0] for t in ko._tokenize_rust(src)}
        self.assertIn(ko._TOK_COMMENT, kinds)
        self.assertIn(ko._TOK_STRING, kinds)

    def test_strip_strings_only_blanks_string_content_leaves_comments_verbatim(self) -> None:
        src = 'fn a() {}\n// a real comment\nlet s = "blank me";\n'
        out = ko._strip_strings_only(src)
        self.assertIn("// a real comment", out)
        self.assertIn("fn a() {}", out)
        self.assertNotIn("blank me", out)


if __name__ == "__main__":
    if "--regenerate-tokenizer-golden" in sys.argv:
        sys.exit(_regenerate_tokenizer_golden())
    unittest.main()
