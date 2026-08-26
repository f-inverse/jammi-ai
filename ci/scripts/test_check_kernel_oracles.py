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


class TestRequireGateHelperDiscovery(unittest.TestCase):
    def test_helper_needs_both_env_var_and_panic(self) -> None:
        src = """
fn real_helper() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("no device");
    }
    None
}

fn decoy_only_env() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        None
    } else {
        None
    }
}

fn decoy_only_panic() -> Option<i32> {
    panic!("always");
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        helpers = ko.find_require_gate_helpers(fns)
        self.assertEqual(helpers, {"real_helper"})


class TestKo7(unittest.TestCase):
    def _run(self, src: str) -> list[ko.UngatedSkip]:
        file_label = "fixture.rs"
        fns = ko.find_fns(src, file_label)
        helpers = ko.find_require_gate_helpers(fns)
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
        pending, failures = ko.reconcile_ops(shipped, {}, {})
        self.assertEqual(failures, [])
        self.assertEqual(set(pending), shipped)

    def test_covered_op_is_not_pending(self) -> None:
        shipped = {"a", "b"}
        line = (
            "//! oracle-cell: op=a leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=none asserted-on=none"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        pending, failures = ko.reconcile_ops(shipped, {"a": [m]}, {})
        self.assertEqual(failures, [])
        self.assertEqual(set(pending), {"b"})

    def test_unknown_op_in_covered_is_a_failure(self) -> None:
        shipped = {"a"}
        line = (
            "//! oracle-cell: op=ghost_op leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=none asserted-on=none"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        _pending, failures = ko.reconcile_ops(shipped, {"ghost_op": [m]}, {})
        self.assertTrue(any("unknown op `ghost_op`" in f for f in failures))

    def test_double_claimed_op_is_a_failure(self) -> None:
        shipped = {"a"}
        line = (
            "//! oracle-cell: op=a leg=fwd dtype=f32 bounds=TOL control=none:x "
            "derived-on=none asserted-on=none"
        )
        m = ko.parse_marker_line(line, "fixture.rs", 1)
        _pending, failures = ko.reconcile_ops(shipped, {"a": [m]}, {"a": "reviewed exclusion"})
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
        ko7, ko2, ko5, covered, pending, recon_failures = ko.run_gate(sources, shipped)
        self.assertEqual(ko7, [])
        self.assertEqual(ko2, [])
        self.assertEqual(ko5, [])
        self.assertEqual(covered, {})
        self.assertEqual(set(pending), shipped)
        self.assertEqual(recon_failures, [])

    def test_ungated_skip_surfaces_end_to_end(self) -> None:
        sources = {
            "crates/jammi-kernels/tests/cuda_parity.rs": """
#[test]
fn a_test() {
    return;
}
""",
        }
        ko7, _ko2, _ko5, _covered, _pending, _recon = ko.run_gate(sources, set())
        self.assertEqual(len(ko7), 1)


if __name__ == "__main__":
    unittest.main()
