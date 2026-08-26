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
        ko7, ko2, ko5, covered, declared_uncontrolled, pending, recon_failures = ko.run_gate(sources, shipped)
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
        ko7, _ko2, _ko5, _covered, _declared_uncontrolled, _pending, _recon = ko.run_gate(sources, set())
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
        helpers = ko.find_require_gate_helpers(fns)
        findings = ko.check_ko7(fns, helpers, {file_label: src})
        self.assertEqual(len(findings), 1)

    def test_env_and_panic_only_in_a_comment_does_not_register(self) -> None:
        src = """
// fn fake_helper() { std::env::var_os("JAMMI_REQUIRE_CUDA"); panic!("x"); }
fn fake_helper() -> Option<i32> {
    None
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        self.assertEqual(ko.find_require_gate_helpers(fns), set())

    def test_env_and_panic_only_in_a_block_comment_does_not_register(self) -> None:
        src = """
/* std::env::var_os("JAMMI_REQUIRE_CUDA"); panic!("x"); */
fn fake_helper() -> Option<i32> {
    None
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        self.assertEqual(ko.find_require_gate_helpers(fns), set())

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
        fns = ko.find_fns(src, "fixture.rs")
        self.assertEqual(ko.find_require_gate_helpers(fns), {"cuda_device"})

    def test_panic_shaped_text_inside_an_unrelated_string_does_not_register(self) -> None:
        # a fn that reads the right env var but whose ONLY "panic!(" text
        # is inside an unrelated string (never real code) must NOT
        # register — the panic-reachability half is checked on
        # comments-AND-strings-stripped text specifically for this.
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
        fns = ko.find_fns(src, "fixture.rs")
        self.assertEqual(ko.find_require_gate_helpers(fns), set())


class TestHelperRegistrationMechanism(unittest.TestCase):
    """Item 2: registration requires BOTH a real env-read call (var/var_os/
    option_env!) whose argument starts with JAMMI_REQUIRE_, AND a reachable
    panic!/unreachable!/.expect( — both arms tested independently.
    """

    def test_env_read_without_any_panic_does_not_register(self) -> None:
        src = """
fn just_checks() -> bool {
    std::env::var_os("JAMMI_REQUIRE_CUDA").is_some()
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        self.assertEqual(ko.find_require_gate_helpers(fns), set())

    def test_panic_without_any_env_read_does_not_register(self) -> None:
        src = """
fn always_panics() -> Option<i32> {
    panic!("no device ever");
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        self.assertEqual(ko.find_require_gate_helpers(fns), set())

    def test_env_var_argument_not_starting_with_jammi_require_does_not_register(self) -> None:
        src = """
fn other_var() -> Option<i32> {
    if std::env::var_os("SOME_OTHER_VAR").is_some() {
        panic!("x");
    }
    None
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        self.assertEqual(ko.find_require_gate_helpers(fns), set())

    def test_unreachable_macro_counts_as_the_panic_arm(self) -> None:
        src = """
fn helper() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        unreachable!("no device");
    }
    None
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        self.assertEqual(ko.find_require_gate_helpers(fns), {"helper"})

    def test_dot_expect_counts_as_the_panic_arm(self) -> None:
        src = """
fn helper() -> Option<i32> {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        let d: Option<i32> = None;
        d.clone().expect("no device");
    }
    None
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        self.assertEqual(ko.find_require_gate_helpers(fns), {"helper"})

    def test_option_env_macro_counts_as_the_env_read(self) -> None:
        src = """
fn helper() -> Option<i32> {
    if option_env!("JAMMI_REQUIRE_CUDA").is_some() {
        panic!("no device");
    }
    None
}
"""
        fns = ko.find_fns(src, "fixture.rs")
        self.assertEqual(ko.find_require_gate_helpers(fns), {"helper"})


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
        helpers = ko.find_require_gate_helpers(fns)
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
        ko7, ko2, ko5, covered, declared_uncontrolled, pending, recon_failures = ko.run_gate(
            {"fixture.rs": src}, shipped
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
        ko7, ko2, ko5, covered, declared_uncontrolled, pending, recon_failures = ko.run_gate(
            {"fixture.rs": src}, shipped
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


if __name__ == "__main__":
    unittest.main()
