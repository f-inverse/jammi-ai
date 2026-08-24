//! Shared `#[cfg(test)]`-only test fixtures used by more than one encoder
//! module's `tests` mod. Crate-private (`mod test_support;` in `lib.rs`,
//! `#[cfg(test)]`-gated): this is not part of the crate's public API, only a
//! way to avoid duplicating a test helper verbatim in every tower's own
//! `tests` mod.

use candle_core::{Device, Tensor};
use candle_nn::VarMap;

/// Deterministic (non-RNG) fill: every variable gets values from a fixed LCG
/// walk over a stably-ordered (sorted-by-key) variable list, so two
/// independent test functions that each build a fresh `VarMap` land on
/// bit-identical weights — required for a training/eval defect-shape pair
/// that must observe the SAME fixture from two separate `#[test]`
/// functions. (Was duplicated verbatim between `clip_text.rs`'s and
/// `htsat_audio.rs`'s `tests` mods; this is the one copy both now call.)
///
/// Domain-validity edge case this fill must honor for towers that load a
/// `BatchNorm` (HTSAT): `running_var` is a VARIANCE, not an unconstrained
/// parameter — it must stay non-negative (`forward` computes
/// `1/sqrt(running_var + eps)`), unlike every other weight/bias this
/// fixture fills. A naive symmetric `[-0.1, 0.1)` fill can and does land
/// `running_var` on a negative value, producing `sqrt(negative) = NaN` that
/// silently propagates through the entire forward pass (and then backward,
/// since NaN arithmetic is closed under every op here) — the fixture would
/// otherwise manufacture a domain violation, not exercise one. Keys ending
/// in `running_var` are instead filled from a strictly-positive `[0.5,
/// 0.7)` range; every other key uses the original symmetric `[-0.1, 0.1)`
/// range (a no-op narrowing for towers, like `ClipText`, with no
/// `running_var` key at all).
pub(crate) fn deterministic_fill_varmap(varmap: &VarMap, device: &Device) {
    let mut state: u32 = 1;
    let data = varmap.data().lock().unwrap();
    let mut entries: Vec<_> = data.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (key, var) in entries {
        let shape = var.shape().clone();
        let n = shape.elem_count();
        let is_variance = key.ends_with("running_var");
        let values: Vec<f32> = (0..n)
            .map(|_| {
                // glibc-style LCG; deterministic, no external RNG/seed state.
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let unit = (state >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
                if is_variance {
                    0.5 + unit * 0.2 // [0.5, 0.7) — strictly positive
                } else {
                    (unit - 0.5) * 0.2 // [-0.1, 0.1)
                }
            })
            .collect();
        var.set(&Tensor::from_vec(values, shape, device).unwrap())
            .unwrap();
    }
}
