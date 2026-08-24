//! Shared `#[cfg(test)]`-only test fixtures used by more than one encoder
//! module's `tests` mod. Crate-private (`mod test_support;` in `lib.rs`,
//! `#[cfg(test)]`-gated): this is not part of the crate's public API, only a
//! way to avoid duplicating a test helper verbatim in every tower's own
//! `tests` mod.

use candle_core::backprop::GradStore;
use candle_core::{Device, Tensor, Var};
use candle_nn::VarMap;

/// Deterministic (non-RNG) fill: every variable gets values from a fixed LCG
/// walk over a stably-ordered (sorted-by-key) variable list, so two
/// independent test functions that each build a fresh `VarMap` land on
/// bit-identical weights — required for a training/eval defect-shape pair
/// that must observe the SAME fixture from two separate `#[test]`
/// functions. The one copy `clip_text.rs` and `htsat_audio.rs` both call.
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

/// Locate the [`Var`] whose VarMap key ends with `suffix` (VarBuilder keys
/// are dot-joined paths, e.g. `transformer.resblocks.0.attn.in_proj_weight`).
/// The one copy `clip_text.rs` and `htsat_audio.rs` both call.
pub(crate) fn find_var(varmap: &VarMap, suffix: &str) -> Var {
    let data = varmap.data().lock().unwrap();
    data.iter()
        .find(|(k, _)| k.ends_with(suffix))
        .unwrap_or_else(|| {
            let keys: Vec<_> = data.keys().collect();
            panic!("no var with suffix '{suffix}' in varmap; keys: {keys:?}")
        })
        .1
        .clone()
}

/// Non-uniform per-channel weights so `loss = (out * weights).sum()` cannot
/// accidentally cancel a broken (e.g. sign-flipped or transposed) gradient
/// reduction the way a uniform-weight sum could. The one copy `clip_text.rs`
/// and `htsat_audio.rs` both call.
pub(crate) fn nonuniform_loss(out: &Tensor, channels: usize, device: &Device) -> Tensor {
    let weights: Vec<f32> = (0..channels).map(|i| 1.0 + i as f32 * 0.37).collect();
    let weights = Tensor::from_vec(weights, channels, device).unwrap();
    out.broadcast_mul(&weights).unwrap().sum_all().unwrap()
}

/// A "grad must be present and non-degenerate" positive control, done right:
/// asserts finiteness FIRST and separately from nonzero-ness. `norm > 0.0`
/// alone is not a sufficient nonzero check on its own — the hole it leaves
/// open is `+inf`, not `NaN`: `NaN > 0.0` is `false`, so `assert!(norm >
/// 0.0)` already panics (correctly fails) on a NaN-poisoned gradient, but
/// `+inf > 0.0` is `true`, so an exploded, non-finite gradient would
/// silently satisfy a bare `> 0.0` check and pass through as if it were a
/// legitimate positive control. Asserting `is_finite()` first closes that
/// hole and gives a distinct failure message for each of the two ways a
/// "must be Some and non-degenerate" control can go wrong. The one helper
/// `clip_text.rs`, `open_clip_vision.rs`, and `htsat_audio.rs` all call.
pub(crate) fn assert_finite_nonzero(norm: f32, what: &str) {
    assert!(
        norm.is_finite(),
        "{what} grad norm must be finite, got {norm}"
    );
    assert!(norm > 0.0, "{what} grad norm must be nonzero, got {norm}");
}

/// BLANKET "reaches every parameter" oracle: iterates every [`Var`] in
/// `varmap` (stably sorted by key, matching [`deterministic_fill_varmap`]'s
/// fold order) and asserts each one — except a key CONTAINING one of the
/// `exclude_patterns` substrings — has a `Some`, finite, nonzero gradient in
/// `grads`. This replaces a hand-picked subset assertion (which only proves
/// the parameters someone thought to name are reachable, silently blind to
/// every OTHER parameter) with a measurement over the actual VarMap
/// contents. `exclude_patterns` exists for two disjoint reasons only, and
/// keep the reason documented at the call site, not silently widened: (1) a
/// non-differentiable-by-construction buffer (e.g. `BatchNorm`'s
/// `running_mean`/`running_var`, which candle's `BatchNorm::forward_t`
/// never routes through an autodiff op), or (2) a weight structurally
/// outside the forward composition the specific test under it builds (a
/// substring match, not `ends_with`, so a path PREFIX like
/// `"audio_projection"` excludes every key under that subtree). The one
/// helper `clip_text.rs`, `open_clip_vision.rs`, and `htsat_audio.rs` all
/// call.
pub(crate) fn assert_every_var_has_gradient(
    varmap: &VarMap,
    grads: &GradStore,
    exclude_patterns: &[&str],
) {
    let data = varmap.data().lock().unwrap();
    let mut entries: Vec<_> = data.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (key, var) in entries {
        if exclude_patterns.iter().any(|pat| key.contains(pat)) {
            continue;
        }
        let grad = grads
            .get(var.as_tensor())
            .unwrap_or_else(|| panic!("{key}: grad must be Some under training=true"));
        let norm = grad
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        assert_finite_nonzero(norm, key);
    }
}

/// The complementary BLANKET severance oracle for `training=false`: every
/// `Var` in `varmap` not matched by `exclude_patterns` (the same substring
/// exclusion mechanism [`assert_every_var_has_gradient`] uses — see its doc)
/// must have NO gradient entry at all in `grads` — `None`, not merely a
/// small or zero one — matching each tower's measured eval-mode truncation
/// (a `LayerNorm`/`softmax_last_dim` site upstream of every trainable
/// parameter with `BackpropOp::none()`). `exclude_patterns` here is for a
/// DIFFERENT reason than the training=true helper's list: a weight applied
/// to the truncated LayerNorm's OUTPUT by a plain differentiable op (e.g. a
/// final projection matmul) still receives its own gradient regardless of
/// the truncation upstream of its input — matmul backward for one operand
/// only needs the OTHER operand's already-computed forward value, not a
/// walk through it — so such a weight is NOT severed even though everything
/// upstream of the truncation is. The one helper `clip_text.rs`,
/// `open_clip_vision.rs`, and `htsat_audio.rs` all call.
pub(crate) fn assert_every_var_grad_is_none(
    varmap: &VarMap,
    grads: &GradStore,
    exclude_patterns: &[&str],
) {
    let data = varmap.data().lock().unwrap();
    let mut entries: Vec<_> = data.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (key, var) in entries {
        if exclude_patterns.iter().any(|pat| key.contains(pat)) {
            continue;
        }
        assert!(
            grads.get(var.as_tensor()).is_none(),
            "{key}: grad must be None under training=false, not merely small or zero"
        );
    }
}
