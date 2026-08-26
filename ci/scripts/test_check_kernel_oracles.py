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

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_kernel_oracles as ko  # noqa: E402


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
        self.assertEqual(names, {"real_helper"})
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
        self.assertEqual(names, {"cuda_device"})
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
        self.assertEqual(names, {"helper"})

    def test_option_env_macro_counts_as_the_env_read(self) -> None:
        src = """
fn helper() -> Option<i32> {
    if option_env!("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("required");
    }
    None
}
"""
        names, _f = ko.verify_helper_registry([("fixture.rs", "helper")], {"fixture.rs": src})
        self.assertEqual(names, {"helper"})

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
        entries = ko.load_helper_registry()
        self.assertEqual(
            set(entries),
            {
                ("crates/jammi-kernels/tests/cuda_parity.rs", "cuda_device"),
                ("crates/jammi-kernels/tests/flash_smoke.rs", "cuda_device"),
                ("crates/jammi-encoders/src/modernbert.rs", "growth_oracle_cuda_device"),
            },
        )
        source_texts = {rel: (ko.REPO_ROOT / rel).read_text() for rel, _name in entries}
        names, failures = ko.verify_helper_registry(entries, source_texts)
        self.assertEqual(failures, [])
        self.assertEqual(names, {"cuda_device", "growth_oracle_cuda_device"})


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
        self.assertEqual(helpers, {"cuda_device"})
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
        self.assertEqual(names, {"cuda_device"})
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
# 2), adopted directly from scratchpad/audit-r3/rs/G1-G7 (paraphrased, not
# copied).
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
    fixtures not already exercised above (paraphrased, not copied),
    adopted with the audit's stated expected outcome.
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
        self.assertEqual(helpers, {"cuda_device"})
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


if __name__ == "__main__":
    unittest.main()
