//! Metal-gated dropout-position semantics oracle for `LoraLinear::forward`
//! (esc-070 conjunct 5).
//!
//! `tests/fused_epilogue.rs`'s dropout-position/resume oracles
//! (`resume_reproduces_the_uninterrupted_dropout_stream` and friends) all
//! hardcode `Device::Cpu` — nothing in this crate's own test suite ever
//! proved `LoraLinear::dropout_position`/`restore_dropout_position` on a
//! REAL Metal device. That gap was not merely "untested on one more
//! device": before issue #433's fix, `jammi_kernels::ops::DropoutFused` had
//! no `metal_fwd` at all, and candle-core's default `CustomOp1::metal_fwd`
//! is a typed `Err`, not a fallback — so EVERY `LoraLinear::forward` call
//! with dropout configured, on a real Metal device, in training mode,
//! FAILED outright. A CPU-only test suite could never have caught a
//! Metal-specific regression here (or proved the fix): this file is the
//! landing proof that a Metal training forward now actually SUCCEEDS, and
//! that the position/resume/no-dropout semantics `fused_epilogue.rs` pins
//! on CPU hold on real Metal hardware too.
//!
//! ## Why every training forward here takes the EAGER arm
//!
//! `lora_linear_admission_predicate`'s own device gate
//! (`jammi_kernels::admission::device_is_supported`) is CPU/CUDA-only —
//! `d.is_cpu() || (cfg!(feature = "cuda") && d.is_cuda())` — Metal never
//! satisfies it. So a Metal training forward structurally ALWAYS falls
//! back to `LoraLinear::forward_composed` (the eager `[base, dropout,
//! A-matmul, B-matmul, epilogue]` composition), regardless of whether the
//! base carries a bias — unlike `fused_epilogue.rs`'s CPU oracles, this
//! file does not need a bias-free-vs-zero-bias fixture pair to select an
//! arm: on Metal there is only one arm to reach, and it is exactly the arm
//! that dispatches `ops::DropoutFused` via `apply1`, exercising
//! `DropoutFused::metal_fwd` for real on every successful call below.
//!
//! `DropoutMasks::next_key` (the host-side `AtomicU64` counter backing
//! `dropout_position`/`restore_dropout_position`) is reserved once per
//! `LoraLinear::forward` call, before the fused/eager decision — so a
//! Metal forward that reaches the dropout op at all, and returns `Ok`,
//! proves the counter advanced for a call that ACTUALLY dispatched
//! `DropoutFused::metal_fwd`, not merely one that reserved a key and then
//! errored out before ever reaching the op (which is exactly what
//! happened, unconditionally, on the pre-#433 Metal build this file
//! guards against).
//!
//! Compiles and links ONLY with the `metal` feature (`required-features =
//! ["metal"]` in `Cargo.toml`; a plain `cargo test -p jammi-lora` never
//! even builds this file) — mirrors
//! `crates/jammi-kernels/tests/metal_parity.rs`'s own gating convention
//! (the crate that owns `DropoutFused` itself). At runtime, a machine that
//! compiled with the feature but has no physical Metal device (or is not
//! on macOS) is treated as "skip", not "fail" — `Device::new_metal(0)`
//! erroring, OR PANICKING, is the signal — UNLESS `JAMMI_REQUIRE_METAL` is
//! set, in which case a device-acquisition failure PANICS instead of
//! returning. `metal_device_or_skip` below is copied, verbatim in shape
//! (including the `catch_unwind` wrapper for the same real-hardware
//! probe-time panic mode — see `metal_parity.rs`'s own module doc for the
//! `objc2`/`residency_set.rs` citation), from `metal_parity.rs`'s helper of
//! the same name; this file registers its own copy in
//! `ci/kernel-oracle-helpers.txt` (KO-7 gating is `(file, fn)`-scoped, so a
//! same-named helper in one file never gates another file's skips).

#![cfg(feature = "metal")]

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap};
use jammi_lora::{LoraInitMode, LoraLinear};

/// A panic payload folded to a human-readable string — see
/// `metal_parity.rs`'s identically-named helper for why only `&str`/
/// `String` downcasts are needed.
fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Acquire a Metal device, or `None` to skip — unless `JAMMI_REQUIRE_METAL`
/// is set, in which case a failure PANICS. See `metal_parity.rs`'s
/// identically-shaped `metal_device_or_skip` for the full rationale (this
/// file registers its OWN copy in `ci/kernel-oracle-helpers.txt`, per that
/// registry's `(file, fn)`-scoping).
fn metal_device_or_skip() -> Option<Device> {
    let outcome: Result<Device, String> = match std::panic::catch_unwind(|| Device::new_metal(0)) {
        Ok(Ok(d)) => Ok(d),
        Ok(Err(e)) => Err(e.to_string()),
        Err(payload) => Err(format!(
            "Device::new_metal(0) panicked: {}",
            panic_payload_to_string(payload.as_ref())
        )),
    };
    match outcome {
        Ok(d) => Some(d),
        Err(msg) => {
            if std::env::var_os("JAMMI_REQUIRE_METAL").is_some() {
                panic!(
                    "metal_dropout_position: JAMMI_REQUIRE_METAL is set but no Metal device is \
                     available: {msg}"
                );
            }
            eprintln!("metal_dropout_position: skipping — no Metal device available: {msg}");
            None
        }
    }
}

/// Deterministic, non-degenerate base weight — same construction every
/// call, mirroring `fused_epilogue.rs`'s own `build_base` (family L: a
/// generic synthetic fixture, no external generator).
fn build_base(in_features: usize, out_features: usize, device: &Device) -> Linear {
    let mut row = Vec::with_capacity(in_features * out_features);
    for i in 0..out_features {
        for j in 0..in_features {
            row.push(((i * 7 + j * 3) as f32).sin());
        }
    }
    let w = Tensor::from_vec(row, (out_features, in_features), device).unwrap();
    Linear::new(w, None)
}

fn ones_input(device: &Device) -> Tensor {
    Tensor::ones((2, 5, 8), DType::F32, device).unwrap()
}

/// (a) Each successful Metal training forward advances `dropout_position`
/// by exactly `+1` — position `0` before any forward, position `i` after
/// the `i`-th successful forward, for `N` forwards in a row.
#[test]
fn metal_successful_train_forwards_advance_dropout_position_by_exactly_one_each() {
    let Some(device) = metal_device_or_skip() else {
        return;
    };
    const N: u64 = 5;

    let base = build_base(8, 16, &device);
    let x = ones_input(&device);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let lora = LoraLinear::new(
        base,
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        Some(0.3),
        123,
        &varmap,
        &vb,
    )
    .unwrap();

    assert_eq!(
        lora.dropout_position().unwrap(),
        Some(0),
        "a freshly-constructed layer with dropout configured must start at position 0"
    );

    for i in 1..=N {
        let out = lora
            .forward(&x)
            .expect("a Metal training forward with dropout configured must SUCCEED (issue #433)");
        assert!(
            out.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite()),
            "forward {i}: output contains a non-finite value"
        );
        assert_eq!(
            lora.dropout_position().unwrap(),
            Some(i),
            "after {i} successful Metal training forward(s), dropout_position must read exactly {i}"
        );
    }
}

/// (b) `restore_dropout_position(pos)` then re-forwarding reproduces the
/// earlier Metal forward's output BIT-IDENTICALLY — the production-path
/// resume invariant (esc-033), proved on real Metal hardware. Mirrors
/// `fused_epilogue.rs`'s `resume_reproduces_the_uninterrupted_dropout_stream`,
/// specialized to Metal (where there is only the eager arm to prove — see
/// this file's module doc).
#[test]
fn metal_restore_dropout_position_reproduces_an_earlier_forward_bit_identically() {
    let Some(device) = metal_device_or_skip() else {
        return;
    };
    const N: usize = 6;
    const K: u64 = 2;

    let build = |seed: u64, varmap: &VarMap, vb: &VarBuilder| -> LoraLinear {
        let base = build_base(8, 16, &device);
        LoraLinear::new(
            base,
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            Some(0.3),
            seed,
            varmap,
            vb,
        )
        .unwrap()
    };
    let x = ones_input(&device);

    // Uninterrupted reference: N forwards, every output recorded.
    let ref_varmap = VarMap::new();
    let ref_vb = VarBuilder::from_varmap(&ref_varmap, DType::F32, &device);
    let reference = build(321, &ref_varmap, &ref_vb);
    let mut ref_outputs = Vec::with_capacity(N);
    for _ in 0..N {
        ref_outputs.push(
            reference
                .forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
    }

    // The "crashed" run: a separate instance, only the first K forwards.
    let interrupted_varmap = VarMap::new();
    let interrupted_vb = VarBuilder::from_varmap(&interrupted_varmap, DType::F32, &device);
    let interrupted = build(321, &interrupted_varmap, &interrupted_vb);
    for _ in 0..K {
        interrupted.forward(&x).unwrap();
    }
    let pos = interrupted.dropout_position().unwrap().unwrap();
    assert_eq!(
        pos, K,
        "dropout_position must count PRODUCTION Metal forwards"
    );

    // The resumed run: a FRESH instance restored to that position,
    // continuing for every remaining forward.
    let resumed_varmap = VarMap::new();
    let resumed_vb = VarBuilder::from_varmap(&resumed_varmap, DType::F32, &device);
    let resumed = build(321, &resumed_varmap, &resumed_vb);
    resumed.restore_dropout_position(pos).unwrap();
    let mut resumed_outputs = Vec::with_capacity(N - K as usize);
    for _ in 0..(N - K as usize) {
        resumed_outputs.push(
            resumed
                .forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
    }

    for i in 0..(N - K as usize) {
        assert_eq!(
            resumed_outputs[i],
            ref_outputs[K as usize + i],
            "post-restore Metal production forward {i} diverged from the uninterrupted run"
        );
    }
}

/// The negative control proving (b)'s oracle has teeth: restoring to
/// `K + 1` instead of `K` must NOT reproduce the uninterrupted run's
/// continuation on Metal either — mirrors `fused_epilogue.rs`'s
/// `fused_arm_production_path_would_catch_an_off_by_one_resume_position`.
#[test]
fn metal_would_catch_an_off_by_one_resume_position() {
    let Some(device) = metal_device_or_skip() else {
        return;
    };
    const N: usize = 5;
    const K: u64 = 2;

    let build = |seed: u64, varmap: &VarMap, vb: &VarBuilder| -> LoraLinear {
        let base = build_base(8, 16, &device);
        LoraLinear::new(
            base,
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            Some(0.3),
            seed,
            varmap,
            vb,
        )
        .unwrap()
    };
    let x = ones_input(&device);

    let ref_varmap = VarMap::new();
    let ref_vb = VarBuilder::from_varmap(&ref_varmap, DType::F32, &device);
    let reference = build(654, &ref_varmap, &ref_vb);
    let mut ref_outputs = Vec::with_capacity(N);
    for _ in 0..N {
        ref_outputs.push(
            reference
                .forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
    }

    let off_by_one_varmap = VarMap::new();
    let off_by_one_vb = VarBuilder::from_varmap(&off_by_one_varmap, DType::F32, &device);
    let off_by_one = build(654, &off_by_one_varmap, &off_by_one_vb);
    off_by_one.restore_dropout_position(K + 1).unwrap(); // the injected bug: should be K.
    let wrong_output = off_by_one
        .forward(&x)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_ne!(
        wrong_output, ref_outputs[K as usize],
        "an off-by-one resume position must NOT reproduce the correct continuation on Metal — \
         if it did, the positive oracle above would be vacuous"
    );
}

/// (c) A layer with no dropout mask source reports `dropout_position() ==
/// None` on Metal — both for the "no dropout configured, training" shape
/// (`dropout: None`) and for the `from_loaded` eval-serving shape
/// (`training: false`, `dropout: None`), across successful Metal forwards.
#[test]
fn metal_layer_with_no_dropout_configured_reports_position_none() {
    let Some(device) = metal_device_or_skip() else {
        return;
    };
    let x = ones_input(&device);

    // Training-mode, no dropout configured at all.
    let base = build_base(8, 16, &device);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let lora = LoraLinear::new(
        base,
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        None,
        7,
        &varmap,
        &vb,
    )
    .unwrap();
    assert_eq!(lora.dropout_position().unwrap(), None);
    lora.forward(&x)
        .expect("a Metal forward with no dropout configured must succeed");
    assert_eq!(
        lora.dropout_position().unwrap(),
        None,
        "a layer with no dropout mask source must keep reporting None after a successful \
         Metal forward"
    );

    // Eval/serving shape: `from_loaded` — `training: false`, `dropout: None`.
    let loaded_base = build_base(8, 16, &device);
    let lora_a = Tensor::zeros((4, 8), DType::F32, &device).unwrap();
    let lora_b = Tensor::zeros((16, 4), DType::F32, &device).unwrap();
    let loaded = LoraLinear::from_loaded(loaded_base, lora_a, lora_b, 8.0, false).unwrap();
    assert_eq!(loaded.dropout_position().unwrap(), None);
    loaded
        .forward(&x)
        .expect("a Metal eval-mode forward must succeed");
    assert_eq!(
        loaded.dropout_position().unwrap(),
        None,
        "an eval-mode (from_loaded) layer must keep reporting None after a successful Metal \
         forward"
    );
}
