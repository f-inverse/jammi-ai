//! esc-046 (GH#374) leg (1) BLINDNESS control, from
//! `.jammi/escapes.jsonl`'s `esc-046-lora-epilogue-rounds-delta-before-add`
//! row: the same-build forced-arm A/B (`JAMMI_KERNELS_DISABLE=
//! lora_linear_fused`, contract K-aux) is structurally BLIND to a rounding
//! defect that BOTH the fused (`jammi_kernels::ops::LowRankResidualLinear`,
//! whose epilogue reuses `ScaledCastAdd`) and eager
//! (`LoraLinear`'s own `eager_epilogue`, `lora_linear.rs:78-91`) arms carry
//! identically — forcing the fused kernel off and reading the eager
//! fallback instead proves NOTHING about which ORDER either arm rounds in,
//! because both arms always agreed (bit-identical) whether or not the bug
//! was present. esc-046's control clause (1) requires this leg to read
//! GREEN (bit-identical) both PRE-fix (the historical, buggy state) and
//! POST-fix (today's state, both arms fixed together) — a RED reading here
//! would mean the fix only touched one of the two arms, exactly the
//! regression this leg exists to catch.
//!
//! ## Two independent defects this revision closes (both found the same way:
//! ## reverting only one arm and checking the leg actually reddens)
//!
//! **(A) Zero dispatch — the ORIGINAL file never reached the fused kernel
//! at all.** `jammi_kernels::admission::disabled_ops()` memoizes
//! `JAMMI_KERNELS_DISABLE` into a **process-wide `OnceLock`, read once ever**
//! (`admission.rs`'s own doc: "This crate does not test the env-var
//! plumbing itself via `std::env::set_var` inside `cargo test`: the
//! `OnceLock` is initialized by whichever test's thread reads it FIRST in
//! the shared test binary... `crates/jammi-bench/tests/` proves the real
//! env-var path end to end by spawning the compiled `jammi-bench` binary as
//! a fresh child PROCESS instead, where a fresh `OnceLock` is guaranteed").
//! The ORIGINAL version of this file did exactly the thing that doc warns
//! against: `std::env::set_var("JAMMI_KERNELS_DISABLE", "lora_linear_fused")`
//! around the EAGER arm's `forward` call, `remove_var` after — but the
//! EAGER arm's `forward` was ALSO the first call into `admit()` in the
//! process, so it permanently initialized `disabled_ops()` to
//! `{"lora_linear_fused"}`. The "fused" arm's LATER `forward` call read the
//! SAME cached `OnceLock` — still `{"lora_linear_fused"}`, `remove_var`
//! notwithstanding — and ALSO dispatched eager.
//! [`lora_linear_fused_dispatch_snapshot`] proves it: measured
//! `{fused: 0, eager: 2}` on `jammi-a100` (PCIe, 2026-08-25) — the "fused"
//! arm never once reached `LowRankResidualLinear`/`ScaledCastAdd::cuda_fwd`,
//! so comparing it to the forced-eager arm was comparing eager to eager, a
//! tautology, independent of any fixture amplitude (kernel guide §3.5:
//! "Zero dispatch is RED, never green"). Fixed the same way
//! `crates/jammi-bench/tests/finetune_step_kernel_disable.rs` does — by
//! spawning a FRESH CHILD PROCESS per arm, guaranteeing a fresh `OnceLock`
//! each time — but `jammi-lora` has no `[[bin]]` target to point
//! `env!("CARGO_BIN_EXE_…")` at, so [`run_arm_in_subprocess`] instead
//! re-invokes THIS test binary itself (`std::env::current_exe()`),
//! `--exact print_one_arm_output_bf16_cuda --nocapture`, once per arm, and
//! parses that helper test's own printed output back. The orchestrator
//! test now asserts the dispatch snapshot directly: the fused subprocess
//! must show `fused == 1, eager == 0`.
//!
//! **(B) Non-discriminating fixture amplitude** (see the "Discrimination"
//! section below) — found and fixed independently of (A), but ALSO
//! necessary: even with dispatch now genuinely split across two processes,
//! a fixture whose `base_out`/`lora_out` land where round-then-add and
//! add-then-round always agree would still read GREEN on a one-arm-only
//! fix.
//!
//! ## Discrimination
//!
//! The `wide_fixture` amplitudes this file originally shipped with
//! (`w_scale=300`, `x_scale=3`, `a_scale=b_scale=5`) are NON-discriminating:
//! this fixture's values are a smooth deterministic RAMP (not noise) fed
//! through a 64-wide/16-rank GEMM reduction — the ramp's terms don't
//! partially cancel the way independent noise would, so the reduction
//! blows `base_out`'s amplitude up far past ModernBERT's own production
//! range (tens of thousands, not `-6688`), landing in the SAME
//! under-discriminating regime `scaled_cast_add_peft_rounding.rs`'s own
//! module doc reports for `|base|~6688` ALONE (delta's contribution falls
//! below one bf16 ULP almost everywhere, so both rounding orders round it
//! away identically regardless of order). Scaled DOWN
//! (`w_scale=5`/`x_scale=2`/`a_scale=b_scale=0.5`, below) so the GEMM
//! reduction lands `base_out` in a realistic amplitude band instead, and
//! self-checked below (`assert_fixture_discriminates`) against the SAME
//! PEFT-ordered / mis-ordered reference `lora_linear.rs`'s own
//! `eager_epilogue_tests` uses, computed from the fixture's REAL
//! `base_out`/`lora_out` tensors (a genuine forward pass through `Linear`,
//! never a hand-rederived GEMM) — never asserted as a fixed constant, so a
//! future amplitude change that silently degenerates back to
//! non-discriminating is caught here rather than only at the half-fix legs.
//!
//! ## Process isolation
//!
//! `JAMMI_KERNELS_DISABLE` mutates process environment and is read into a
//! process-wide `OnceLock` (see (A) above) — this file's own two arms MUST
//! run in separate processes, and [`print_one_arm_output_bf16_cuda`] (the
//! subprocess entry point) must never run concurrently with the orchestrator
//! test reading the SAME env var in-process. Both live in this one file
//! (cargo's own unit of test-binary granularity keeps them isolated from
//! every OTHER file's tests) but now ALSO isolated from EACH OTHER via the
//! subprocess boundary in (A).
//!
//! CUDA-gated, not CPU: `LowRankResidualLinear`'s CPU arm cannot even ADMIT
//! a `BF16` backbone (candle-core 0.11's CPU matmul has no `BF16` impl —
//! see that op's own "CPU `BF16` matmul" module-doc section) — and `BF16`
//! is exactly the dtype esc-046's rounding-order bug requires to be
//! observable at all (an `F32` epilogue has no rounding anywhere, any
//! ordering is bit-identical trivially). This leg is therefore only
//! meaningful on CUDA.

#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use jammi_lora::{lora_linear_fused_dispatch_snapshot, LoraInitMode, LoraLinear};
use std::process::Command;

const IN_FEATURES: usize = 64;
const OUT_FEATURES: usize = 128;
const RANK: usize = 16;
const ALPHA: f64 = 32.0; // scaling = alpha/rank = 2.0, computed internally by LoraLinear::new.

/// Deterministic, non-degenerate `f32`-then-cast fixture spanning a real
/// amplitude range (not a toy small-integer one — esc-046's own bug is
/// amplitude-dependent: one bf16 ULP is `1.0` at `|base|~100`, `32` at
/// ModernBERT-large's own layer-18 residual magnitude, `-6688`), built the
/// same way every time so building it twice — including across a process
/// boundary, since this is a PURE function of hardcoded constants, no OS
/// randomness anywhere — yields bit-identical tensors.
fn wide_fixture(n: usize, phase: i64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i as i64 * 7 + phase * 13).rem_euclid(2000) - 1000;
            (v as f32 / 1000.0) * scale
        })
        .collect()
}

/// `w` (base weight, `BF16`) and `x` (input, `BF16`) — shared by both arms.
/// Amplitudes chosen to keep the 64-wide GEMM reduction's `base_out` in a
/// realistic band rather than the tens-of-thousands the ORIGINAL
/// `w_scale=300`/`x_scale=3` amplitudes produced (see this file's module
/// doc's "Discrimination" section) — verified non-vacuous below, not
/// assumed.
fn fixture_tensors(device: &Device) -> (Tensor, Tensor) {
    let w_v = wide_fixture(OUT_FEATURES * IN_FEATURES, 2, 5.0);
    let w = Tensor::from_slice(&w_v, (OUT_FEATURES, IN_FEATURES), device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let x_v = wide_fixture(2 * 5 * IN_FEATURES, 1, 2.0);
    let x = Tensor::from_slice(&x_v, (2, 5, IN_FEATURES), device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    (w, x)
}

/// Builds one `LoraLinear` over `w`, with the same deterministic
/// `lora_a`/`lora_b` fixture every call (public fields, per
/// `fused_epilogue.rs`'s established precedent) — so two calls, even across
/// a process boundary, produce bit-identical adapters.
fn build_lora(device: &Device, w: &Tensor, bias: bool) -> LoraLinear {
    let varmap = VarMap::new();
    // `lora_a`/`lora_b` must be `F32` (this op's own domain — see
    // `lora_linear_admission_predicate`'s `lora_ab_dtype_f32` check); only
    // the BASE weight (passed in directly as an already-built `Linear`
    // below, never allocated through `vb`) is `BF16`.
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let base_bias = bias.then(|| Tensor::zeros((OUT_FEATURES,), DType::BF16, device).unwrap());
    let base = Linear::new(w.clone(), base_bias);
    let mut lora = LoraLinear::new(
        base,
        RANK,
        ALPHA,
        false,
        LoraInitMode::Gaussian,
        None,
        11,
        &varmap,
        &vb,
    )
    .unwrap();
    let a_v = wide_fixture(RANK * IN_FEATURES, 3, 0.5);
    let b_v = wide_fixture(OUT_FEATURES * RANK, 4, 0.5);
    lora.lora_a = Tensor::from_slice(&a_v, (RANK, IN_FEATURES), device).unwrap();
    lora.lora_b = Tensor::from_slice(&b_v, (OUT_FEATURES, RANK), device).unwrap();
    lora
}

/// Rounds every element through ONE real `BF16` round-trip via
/// `Tensor::to_dtype` on `Device::Cpu` (never a hand-rolled bf16 cast, and
/// deliberately NOT the CUDA device the arms under test use — this is a
/// host-side reference computation, independent of whichever CUDA kernel
/// is being reverted). The returned `f32`s are each an EXACT widening of a
/// real `bf16` bit pattern, so plain `f32` `==` on two values that both
/// went through this function is bit-exact comparison.
fn round_bf16_batch_cpu(values: &[f32]) -> Vec<f32> {
    Tensor::from_slice(values, values.len(), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// PEFT-ordered reference (correct): `base` (already bf16-exact) widens to
/// f32 losslessly, adds the f32-scaled delta, rounds to bf16 ONCE. Mirrors
/// `lora_linear.rs`'s `eager_epilogue_tests::peft_ordered` exactly.
fn peft_ordered(base: &[f32], delta: &[f32], scaling: f32) -> Vec<f32> {
    let sum: Vec<f32> = base
        .iter()
        .zip(delta)
        .map(|(&b, &d)| b + d * scaling)
        .collect();
    round_bf16_batch_cpu(&sum)
}

/// The pre-esc-046 mis-ordered formula (round the scaled delta to bf16
/// FIRST, then add and round the sum again) — kept ONLY to prove the
/// fixture discriminates the two orderings, never asserted as correct.
/// Mirrors `lora_linear.rs`'s `eager_epilogue_tests::mis_ordered` exactly.
fn mis_ordered(base: &[f32], delta: &[f32], scaling: f32) -> Vec<f32> {
    let scaled_raw: Vec<f32> = delta.iter().map(|&d| d * scaling).collect();
    let scaled_rounded = round_bf16_batch_cpu(&scaled_raw);
    let sum: Vec<f32> = base
        .iter()
        .zip(scaled_rounded.iter())
        .map(|(&b, &s)| b + s)
        .collect();
    round_bf16_batch_cpu(&sum)
}

/// Non-vacuous discrimination floor — measured at 147/1280 (11.5%) with the
/// amplitudes above on `jammi-a100`, PCIe (2026-08-25); `>= 20` leaves
/// headroom for a different cuBLAS/toolchain build while refusing a
/// fixture that has degenerated to "the two orderings always agree" —
/// exactly what the ORIGINAL `wide_fixture` amplitudes silently did (see
/// this file's module doc).
const MIN_DISCRIMINATING: usize = 20;

/// Asserts the fixture's REAL `base_out`/`lora_out` (a genuine forward pass
/// through `Linear`, computed on `device` — never a hand-rederived GEMM)
/// separate the PEFT-ordered formula from the pre-esc-046 mis-ordered one
/// on at least [`MIN_DISCRIMINATING`] elements — the precondition that
/// makes the half-fix legs meaningful at all.
fn assert_fixture_discriminates(device: &Device, w: &Tensor, x: &Tensor, lora: &LoraLinear) {
    let scaling = (ALPHA / RANK as f64) as f32;
    let base_lin = Linear::new(w.clone(), None);
    let base_out = base_lin.forward(x).unwrap();
    let base_flat: Vec<f32> = base_out
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();

    let x_f32 = x.to_dtype(DType::F32).unwrap();
    let a_lin = Linear::new(lora.lora_a.clone(), None);
    let after_a = a_lin.forward(&x_f32).unwrap();
    let b_lin = Linear::new(lora.lora_b.clone(), None);
    let lora_out = b_lin.forward(&after_a).unwrap();
    let lora_flat: Vec<f32> = lora_out.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(base_flat.len(), lora_flat.len());
    for (i, (&b, &d)) in base_flat.iter().zip(lora_flat.iter()).enumerate() {
        assert!(
            b.is_finite() && d.is_finite(),
            "index {i}: a non-finite fixture value slipped through (base={b} delta={d})"
        );
    }

    let peft = peft_ordered(&base_flat, &lora_flat, scaling);
    let buggy = mis_ordered(&base_flat, &lora_flat, scaling);
    let discriminating = (0..base_flat.len())
        .filter(|&i| peft[i] != buggy[i])
        .count();
    assert!(
        discriminating >= MIN_DISCRIMINATING,
        "fixture is not discriminating: only {discriminating}/{} elements separate the \
         PEFT-ordered formula from the pre-esc-046 mis-ordered one on this fixture's REAL \
         base_out/lora_out — the half-fix legs would read GREEN regardless of which arm (if \
         either) is broken; strengthen the amplitude before trusting this file's core claim. \
         (device={:?})",
        base_flat.len(),
        device,
    );
}

/// Subprocess entry point (defect (A)'s fix) — NOT meant to be inspected
/// directly by a normal `cargo test` run (though it is a harmless, correct
/// `#[test]` on its own: it just builds the fixture, forwards once, and
/// prints). The ORCHESTRATOR test below re-invokes THIS test binary via
/// [`run_arm_in_subprocess`], `--exact print_one_arm_output_bf16_cuda
/// --nocapture`, with `JAMMI_KERNELS_DISABLE` either set or unset in the
/// CHILD's environment — guaranteeing `jammi_kernels::admission`'s
/// process-wide `OnceLock` is read fresh, in a process that has made
/// exactly one `admit()` call, every time. Prints each output element as
/// `ELEM <index> <hex_bits>`, one per line — the raw `f32` bit pattern
/// (`BF16` widened losslessly, per this file's other functions' own note)
/// so the parent can reconstruct the array bit-exactly from captured
/// stdout without any lossy text-to-float round trip.
#[test]
fn print_one_arm_output_bf16_cuda() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("esc046 blindness leg: skipping — no CUDA device available");
        return;
    };
    let (w, x) = fixture_tensors(&device);
    let mut lora = build_lora(&device, &w, false);
    lora.set_training(true);
    let out = lora.forward(&x).unwrap();
    let flat: Vec<f32> = out
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    let snap = lora_linear_fused_dispatch_snapshot();
    println!("DISPATCH fused={} eager={}", snap.fused, snap.eager);
    for (i, v) in flat.iter().enumerate() {
        println!("ELEM {i} {:08x}", v.to_bits());
    }
}

/// One arm's `(output, dispatch_snapshot)`, read back from
/// [`print_one_arm_output_bf16_cuda`]'s captured stdout in a fresh child
/// process. `disable` is `Some("lora_linear_fused")` for the forced-eager
/// arm, `None` for the arm that should genuinely dispatch fused.
fn run_arm_in_subprocess(disable: Option<&str>) -> (Vec<f32>, (u64, u64)) {
    let exe = std::env::current_exe().expect("current_exe must resolve for a compiled test binary");
    let mut cmd = Command::new(&exe);
    cmd.args(["print_one_arm_output_bf16_cuda", "--exact", "--nocapture"]);
    match disable {
        Some(v) => {
            cmd.env("JAMMI_KERNELS_DISABLE", v);
        }
        None => {
            cmd.env_remove("JAMMI_KERNELS_DISABLE");
        }
    }
    let output = cmd
        .output()
        .expect("failed to spawn this test binary as a subprocess");
    assert!(
        output.status.success(),
        "subprocess (disable={disable:?}) exited non-zero: status={:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut dispatch: Option<(u64, u64)> = None;
    let mut elems: Vec<Option<f32>> = Vec::new();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("DISPATCH fused=") {
            let mut parts = rest.split(" eager=");
            let fused: u64 = parts.next().unwrap().parse().unwrap();
            let eager: u64 = parts.next().unwrap().parse().unwrap();
            dispatch = Some((fused, eager));
        } else if let Some(rest) = line.strip_prefix("ELEM ") {
            let mut parts = rest.split_whitespace();
            let idx: usize = parts.next().unwrap().parse().unwrap();
            let bits = u32::from_str_radix(parts.next().unwrap(), 16).unwrap();
            if elems.len() <= idx {
                elems.resize(idx + 1, None);
            }
            elems[idx] = Some(f32::from_bits(bits));
        }
    }
    let dispatch = dispatch.unwrap_or_else(|| {
        panic!("subprocess (disable={disable:?}) never printed a DISPATCH line — stdout={stdout}")
    });
    let out: Vec<f32> = elems
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            v.unwrap_or_else(|| {
                panic!("subprocess (disable={disable:?}) output missing ELEM {i} — truncated?")
            })
        })
        .collect();
    (out, dispatch)
}

#[test]
fn fused_vs_eager_forced_arm_ab_is_bit_identical_both_pre_and_post_fix_bf16_cuda() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("esc046 blindness leg: skipping — no CUDA device available");
        return;
    };

    // Non-vacuity precondition (this file's own module doc): the fixture's
    // REAL base_out/lora_out must actually separate the PEFT-ordered
    // formula from the pre-esc-046 mis-ordered one before the half-fix
    // legs (and this leg's own positive claim) mean anything. Computed
    // in-process — this is plain `Linear::forward`, never `admit()`, so it
    // does not touch the `OnceLock` defect (A)'s fix is about.
    let (w, x) = fixture_tensors(&device);
    let probe_lora = build_lora(&device, &w, false);
    assert_fixture_discriminates(&device, &w, &x, &probe_lora);

    // Each arm runs in its OWN fresh child process (defect (A)'s fix) —
    // `jammi_kernels::admission::disabled_ops()`'s process-wide `OnceLock`
    // is guaranteed uninitialized in each child, so `JAMMI_KERNELS_DISABLE`
    // actually takes effect (or doesn't) exactly once, cleanly, per arm.
    let (eager_v, eager_dispatch) = run_arm_in_subprocess(Some("lora_linear_fused"));
    let (fused_v, fused_dispatch) = run_arm_in_subprocess(None);

    // Zero-dispatch is RED, never green (kernel guide §3.5) — this is
    // exactly the assertion that was MISSING when defect (A) let this file
    // pass while never once reaching the fused kernel.
    assert_eq!(
        eager_dispatch,
        (0, 1),
        "forced-eager subprocess must show fused=0, eager=1 — got {eager_dispatch:?}; \
         JAMMI_KERNELS_DISABLE=lora_linear_fused did not take effect in the child"
    );
    assert_eq!(
        fused_dispatch,
        (1, 0),
        "the un-forced subprocess must show fused=1, eager=0 — got {fused_dispatch:?}; the \
         fused kernel never dispatched, so this leg's fused-vs-eager comparison below would be \
         eager-vs-eager, a tautology (kernel guide §3.5, and this file's own module doc's \
         defect (A))"
    );

    // Widened to `F32` for readback ONLY (never introduces its own
    // rounding: `BF16` is exactly the top 16 bits of `F32`, so this
    // widening is injective — two DISTINCT `BF16` bit patterns always
    // widen to two DISTINCT `F32` values, making an `F32` equality
    // comparison here exactly equivalent to comparing raw `BF16` bit
    // patterns, without needing the `half` crate as a direct dependency).
    // Both `eager_v`/`fused_v` are ALREADY exact `f32` widenings of the
    // child's own `BF16` output — see `print_one_arm_output_bf16_cuda`'s
    // doc.
    assert_eq!(fused_v.len(), eager_v.len(), "output length mismatch");
    // Finiteness-affirmative (clause 4) before the bit-identity compare.
    for (i, (&f, &e)) in fused_v.iter().zip(eager_v.iter()).enumerate() {
        assert!(
            f.is_finite() && e.is_finite(),
            "index {i}: a non-finite value slipped through (fused={f}, eager={e})"
        );
    }
    let mismatches: Vec<usize> = fused_v
        .iter()
        .zip(eager_v.iter())
        .enumerate()
        .filter(|(_, (f, e))| f.to_bits() != e.to_bits())
        .map(|(i, _)| i)
        .collect();
    assert!(
        mismatches.is_empty(),
        "esc-046 leg (1) BLINDNESS violated: the fused arm (`LowRankResidualLinear`) and the \
         forced-eager arm (`JAMMI_KERNELS_DISABLE=lora_linear_fused` -> `eager_epilogue`) \
         disagree on {}/{} elements — they must round IDENTICALLY (both arms were fixed \
         together in the same esc-046 change; a fix that touched only one arm reads RED here, \
         not GREEN). First mismatch at index {}: fused={} eager={}",
        mismatches.len(),
        fused_v.len(),
        mismatches[0],
        fused_v[mismatches[0]],
        eager_v[mismatches[0]],
    );
}
