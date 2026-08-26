//! A Jammi-owned AdamW with serializable optimizer state.
//!
//! The update is numerically identical to `candle_nn::AdamW` — decoupled weight
//! decay, bias-corrected first/second moments — but this implementation exposes
//! its per-parameter moment buffers and global step counter through [`AdamW::state`]
//! / [`AdamW::load_state`]. candle's `AdamW` keeps those fields private with no
//! accessor, so a training run cannot be checkpointed and resumed *mid-flight*
//! through it: a resume that restarts Adam's moments at zero takes a different
//! trajectory than the uninterrupted run. Owning the optimizer is what makes a
//! faithful resume-after-crash possible (the trajectory state travels with the
//! checkpoint), so the right shape is to own it rather than wrap a type that
//! cannot surface its own state.

use std::sync::LazyLock;

use candle_core::backprop::GradStore;
use candle_core::{DType, Result, Tensor, Var};
use jammi_kernels::admission::{
    admission_mode, admit, counters_for, device_is_supported, DispatchCounters, DispatchOutcome,
};
use jammi_kernels::ops::{adamw_step_fused_t, AdamWParams};

/// Hyperparameters are candle's — reused verbatim so the update matches and a
/// caller configures one struct, not two.
pub use candle_nn::ParamsAdamW;

/// Per-op fused/eager dispatch counts for the fused multi-tensor AdamW step,
/// read from `jammi_kernels::admission`'s op-keyed registry — the same
/// `counters_for` pattern `jammi-encoders::layer_norm` uses (its own
/// `LN_DISPATCH_COUNTERS`), not a hand-declared static, since this crate has
/// no existing per-op statics of its own to keep byte-compatible.
static ADAMW_DISPATCH_COUNTERS: LazyLock<&'static DispatchCounters> =
    LazyLock::new(|| counters_for("adamw_step_fused"));

/// The fused kernel's domain, checked once per `Var` per step (family D):
/// `theta`/`first_moment`/`second_moment`/`grad` all live on a device
/// [`device_is_supported`] accepts, share [`DType::F32`] (the op's only
/// implemented dtype — see `adamw_step.rs`'s module doc), are ALL mutually
/// contiguous (the CUDA arm refuses a non-contiguous read/write outright,
/// and even on CPU a non-contiguous view is refused here rather than routed
/// through the slower strided-walk path silently), and share one shape (no
/// broadcasting — a shape mismatch here would otherwise be a silent
/// misread). Returns the aggregate predicate and the name of whichever
/// check is the first to fail (or `"domain_ok"`).
fn fused_admission_predicate(
    theta: &Tensor,
    first_moment: &Tensor,
    second_moment: &Tensor,
    grad: &Tensor,
) -> (bool, &'static str) {
    if !device_is_supported(theta.device()) {
        return (false, "device_is_cpu_or_cuda");
    }
    if !theta.device().same_device(first_moment.device())
        || !theta.device().same_device(second_moment.device())
        || !theta.device().same_device(grad.device())
    {
        return (false, "theta_moments_grad_share_one_device");
    }
    if theta.dtype() != DType::F32
        || first_moment.dtype() != DType::F32
        || second_moment.dtype() != DType::F32
        || grad.dtype() != DType::F32
    {
        return (false, "dtype_f32");
    }
    if !theta.is_contiguous()
        || !first_moment.is_contiguous()
        || !second_moment.is_contiguous()
        || !grad.is_contiguous()
    {
        return (false, "theta_moments_grad_all_contiguous");
    }
    if theta.dims() != first_moment.dims()
        || theta.dims() != second_moment.dims()
        || theta.dims() != grad.dims()
    {
        return (false, "theta_moments_grad_share_one_shape");
    }
    (true, "domain_ok")
}

/// One trainable parameter and its Adam moment buffers (first/second moment),
/// held together so a step updates all three in lockstep.
struct AdamVar {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

/// One step's derived scalars (`step_t`-dependent bias-correction terms
/// included) — the shared bookkeeping [`AdamW::advance_step_scales`]
/// computes once, so [`AdamW::step`] and the `#[cfg(test)]`-only
/// [`AdamW::step_forced`] cannot silently diverge on how `t`,
/// `scale_m`/`scale_v`, or `lr_lambda` are derived. `Copy` (all fields are
/// plain `f64`): [`step_eager_one`] takes this by value instead of six
/// separate scalar parameters, which is also what collapses that function's
/// argument count under `clippy::too_many_arguments`' default threshold.
#[derive(Clone, Copy)]
struct StepScales {
    beta1: f64,
    beta2: f64,
    scale_m: f64,
    scale_v: f64,
    lr: f64,
    lr_lambda: f64,
}

/// AdamW (decoupled weight decay) whose optimizer state — per-parameter first
/// and second moments plus the global step counter — is readable via
/// [`AdamW::state`] and restorable via [`AdamW::load_state`]. That state is the
/// full Adam trajectory, the thing a resume must carry to converge identically.
pub struct AdamW {
    vars: Vec<AdamVar>,
    step_t: usize,
    params: ParamsAdamW,
}

impl AdamW {
    /// Construct over the trainable variables, zero-initializing each one's
    /// moment buffers. Non-float variables are skipped (they carry no gradient),
    /// matching candle's filter, so the retained order is the float subset of
    /// `vars` in their original order — the order [`AdamW::state`] reports in.
    pub fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(AdamVar {
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            step_t: 0,
            params,
        })
    }

    /// The current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    /// Set the learning rate — the per-step lever the LR schedule drives.
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    /// Increments `step_t` and derives this step's bias-correction/decay
    /// scalars from it — factored out of [`Self::step`] so the
    /// `#[cfg(test)]`-only [`Self::step_forced`] (the dispatch's RED-oracle
    /// forcing mechanism — see `dispatch_arms`'s module doc) computes the
    /// exact same `step_t`/`scale_m`/`scale_v` bookkeeping as production
    /// `step`, rather than a second, independently-maintained copy that
    /// could silently drift.
    fn advance_step_scales(&mut self) -> StepScales {
        self.step_t += 1;
        let lr = self.params.lr;
        let lr_lambda = lr * self.params.weight_decay;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        StepScales {
            beta1,
            beta2,
            scale_m,
            scale_v,
            lr,
            lr_lambda,
        }
    }

    /// TEST-ONLY: identical to [`Self::step`] except it bypasses
    /// `fused_admission_predicate`/`admit` entirely and unconditionally
    /// runs one named arm for every `Var` this step. This is the RED-oracle
    /// "force" mechanism `dispatch_arms` uses in place of
    /// `JAMMI_KERNELS_DISABLE=<op key>` (K-aux, `feat/kernels-admission-
    /// disable` @ e602d7a — merged onto this branch's `main` base, so the
    /// env-var switch itself IS available here; `step_forced` predates that
    /// merge and is kept as the dispatch's oracle mechanism unchanged rather
    /// than reshaped into an env-var-driven test — see `dispatch_arms`'s
    /// module doc for the scope-amendment note). `#[cfg(test)]`: not part of
    /// the production API surface, compiled only for `cargo test`, so it can
    /// never be reached in a real training run regardless of admission mode.
    #[cfg(test)]
    fn step_forced(&mut self, grads: &GradStore, force_fused: bool) -> Result<()> {
        let scales = self.advance_step_scales();
        let t = self.step_t;
        let fused_params = AdamWParams {
            beta1: scales.beta1,
            beta2: scales.beta2,
            lr: scales.lr,
            weight_decay: self.params.weight_decay,
            eps: self.params.eps,
        };
        for entry in self.vars.iter() {
            let theta = &entry.var;
            let m = &entry.first_moment;
            let v = &entry.second_moment;
            if let Some(g) = grads.get(theta) {
                if force_fused {
                    step_fused_one(theta, m, v, g, t, fused_params)?;
                } else {
                    step_eager_one(theta, m, v, g, scales, self.params.eps)?;
                }
            }
        }
        Ok(())
    }

    /// Take one AdamW step over `grads`. Identical to `candle_nn::AdamW::step`:
    /// EMA the moments, bias-correct, apply decoupled weight decay, then the
    /// bias-corrected adaptive update. Per `Var`, dispatches through
    /// [`jammi_kernels::admission::admit`] to either the fused, in-place,
    /// zero-`Var::set` kernel (`step_fused_one`) or today's exact eager
    /// candle-op chain (`step_eager_one`) — see `adamw_step.rs`'s module
    /// doc and `scratchpad/design-multi-tensor-adamw.md` for why the fused
    /// arm is expected to admit on every real (freshly-allocated, F32,
    /// contiguous) LoRA `Var`. A `Var` with no `GradStore` entry this step
    /// is skipped exactly as before — the guard is unchanged either way.
    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        let scales = self.advance_step_scales();
        let t = self.step_t;
        let fused_params = AdamWParams {
            beta1: scales.beta1,
            beta2: scales.beta2,
            lr: scales.lr,
            weight_decay: self.params.weight_decay,
            eps: self.params.eps,
        };
        for entry in self.vars.iter() {
            let theta = &entry.var;
            let m = &entry.first_moment;
            let v = &entry.second_moment;
            if let Some(g) = grads.get(theta) {
                let (holds, predicate) =
                    fused_admission_predicate(theta.as_tensor(), m.as_tensor(), v.as_tensor(), g);
                // `admit` returns `jammi_kernels::error::Result`
                // (`KernelError`), not `candle_core::Result` — the two error
                // types are both foreign to this crate, so no blanket `?`
                // conversion exists (unlike `jammi-encoders::EncoderError`,
                // which owns a `#[from] KernelError` variant). `Error::wrap`
                // is candle's own sanctioned way to lift an arbitrary
                // `Display`-able error into its error type without losing
                // the message (only `StrictModeFallback` can appear here —
                // `Fallback` mode, the default, never errors).
                let outcome = admit(
                    admission_mode(),
                    "adamw_step_fused",
                    predicate,
                    holds,
                    *ADAMW_DISPATCH_COUNTERS,
                )
                .map_err(candle_core::Error::wrap)?;
                match outcome {
                    // `adamw_step_fused_t`'s own `validate_step_domain` runs
                    // BEFORE any `InplaceOpN` mutation (see that function's
                    // module doc) — so an `Err` here (which propagates
                    // straight out of `step` via `?`, never falling back to
                    // the eager arm for this `Var`) is guaranteed to leave
                    // `theta`/`first_moment`/`second_moment` byte-for-byte
                    // untouched, not partially advanced. In normal operation
                    // this predicate already re-checks device/dtype/
                    // contiguity/shape before reaching here, so the only NEW
                    // failure mode the kernel's own validation can surface
                    // that this predicate does not already gate is aliasing
                    // (`theta`/`first_moment`/`second_moment`/`grad` sharing
                    // storage) — never true for `AdamW`'s own `Var`s, which
                    // `AdamW::new` always allocates as three DISTINCT
                    // `Var::zeros`/caller-owned tensors, and `grads.get`
                    // returns a `GradStore` entry from a distinct backward
                    // pass — see `dispatch_arms::an_aliased_var_is_refused_
                    // and_leaves_state_untouched` for the oracle.
                    DispatchOutcome::Fused => step_fused_one(theta, m, v, g, t, fused_params)?,
                    DispatchOutcome::Eager => {
                        step_eager_one(theta, m, v, g, scales, self.params.eps)?
                    }
                }
            }
        }
        Ok(())
    }

    /// How many steps have been taken — the `t` the bias correction depends on,
    /// which a resume must restore so the first post-resume step corrects identically.
    pub fn step_t(&self) -> usize {
        self.step_t
    }

    /// Snapshot the optimizer state: each parameter's `(first_moment,
    /// second_moment)` in construction order, plus the step counter. The order
    /// matches the float `vars` passed to [`AdamW::new`], so a caller holding the
    /// same parameter ordering (its named trainable weights) can key the moments
    /// by name for serialization.
    ///
    /// The order is whatever `vars` the optimizer was built from — and a
    /// `VarMap`'s `all_vars()` iterates a `HashMap`, whose order is *not* stable
    /// across processes. A caller serializing by name must therefore capture the
    /// names from the **same** `vars` slice it passed to [`AdamW::new`], at
    /// snapshot time; it cannot re-derive the index→name map from a fresh
    /// `all_vars()` on resume.
    ///
    /// The moment tensors are **deep-copied** ([`Tensor::copy`]): candle's
    /// `Tensor::clone` shares storage and `Var::set` writes in place, so a shallow
    /// snapshot would be silently overwritten by the next [`AdamW::step`]. The
    /// snapshot is independent of continued training.
    pub fn state(&self) -> Result<(Vec<(Tensor, Tensor)>, usize)> {
        let moments = self
            .vars
            .iter()
            .map(|e| {
                Ok((
                    e.first_moment.as_tensor().copy()?,
                    e.second_moment.as_tensor().copy()?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((moments, self.step_t))
    }

    /// Restore optimizer state captured by [`AdamW::state`]. `moments` must hold
    /// one `(first, second)` pair per parameter, in the order [`AdamW::state`]
    /// reports and matching each parameter's shape; the step counter is restored
    /// too. A resumed run then continues the exact Adam trajectory rather than
    /// restarting its moments at zero.
    pub fn load_state(&mut self, moments: &[(Tensor, Tensor)], step_t: usize) -> Result<()> {
        if moments.len() != self.vars.len() {
            candle_core::bail!(
                "AdamW::load_state: {} moment pairs for {} parameters",
                moments.len(),
                self.vars.len()
            );
        }
        for (entry, (first, second)) in self.vars.iter().zip(moments) {
            entry.first_moment.set(first)?;
            entry.second_moment.set(second)?;
        }
        self.step_t = step_t;
        Ok(())
    }
}

/// The eager arm: today's exact candle-op chain, byte-for-byte unchanged
/// from the pre-fusion `AdamW::step` body (moved out to a free function so
/// [`AdamW::step`] can pick it per-`Var` via `admit`, and so a test can call
/// it directly to force the eager arm — see the `dispatch_arms` test module
/// below for why this is the RED-oracle's "force" mechanism on this branch).
/// `scales` bundles `beta1`/`beta2`/`scale_m`/`scale_v`/`lr`/`lr_lambda` —
/// see [`StepScales`]'s own doc for why this collapses the argument count
/// under `clippy::too_many_arguments`' threshold without changing the
/// arithmetic below at all.
fn step_eager_one(
    theta: &Var,
    m: &Var,
    v: &Var,
    g: &Tensor,
    scales: StepScales,
    eps: f64,
) -> Result<()> {
    let StepScales {
        beta1,
        beta2,
        scale_m,
        scale_v,
        lr,
        lr_lambda,
    } = scales;
    let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
    let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
    let m_hat = (&next_m * scale_m)?;
    let v_hat = (&next_v * scale_v)?;
    let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
    let adjusted_grad = (m_hat / (v_hat.sqrt()? + eps)?)?;
    let next_theta = (next_theta - (adjusted_grad * lr)?)?;
    m.set(&next_m)?;
    v.set(&next_v)?;
    theta.set(&next_theta)?;
    Ok(())
}

/// The fused arm: one [`adamw_step_fused_t`] call per `Var` — three
/// `InplaceOp2`/`InplaceOp3` kernel launches, zero `Var::set`/memcpy (see
/// `adamw_step.rs`'s module doc). `theta`/`m`/`v` are mutated in place
/// through candle's own `Arc<RwLock<Storage>>` write guard (via
/// `Var::as_tensor`, which derefs to the SAME storage the `Var` owns — not a
/// copy), so no `.set()` call is needed or made here. `t` is the 1-indexed
/// step counter (`AdamW::step_t` after `advance_step_scales` has already
/// incremented it) — `scale_m`/`scale_v` are derived from `t` INSIDE
/// `adamw_step_fused_t` itself now, not passed in, closing the bias-
/// correction footgun the now-`#[deprecated]` `adamw_step_fused` (a
/// caller-supplied `scale_m`/`scale_v`, computable from a stale or wrong
/// `t`) carried — see that function's own doc.
fn step_fused_one(
    theta: &Var,
    m: &Var,
    v: &Var,
    g: &Tensor,
    t: usize,
    params: AdamWParams,
) -> Result<()> {
    adamw_step_fused_t(
        theta.as_tensor(),
        m.as_tensor(),
        v.as_tensor(),
        g,
        t,
        params,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// A tiny convex problem: one parameter `w` pulled toward a target by an
    /// MSE gradient, so a `step` produces a deterministic, checkable update.
    fn setup() -> (Var, AdamW) {
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::zeros((4,), DType::F32, &dev).unwrap()).unwrap();
        let opt = AdamW::new(
            vec![w.clone()],
            ParamsAdamW {
                lr: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        (w, opt)
    }

    fn grad_toward(w: &Var, target: f64) -> GradStore {
        // A real backward over `sum((w - target)^2)` (grad = 2·(w − target)), so
        // the step is exercised through candle's autograd exactly as training is.
        let diff = (w.as_tensor() - target).unwrap();
        let loss = diff.sqr().unwrap().sum_all().unwrap();
        loss.backward().unwrap()
    }

    #[test]
    fn state_round_trip_resumes_the_exact_trajectory() {
        // Run a few steps on A; snapshot A's state mid-run; build a fresh B,
        // load the snapshot, and assert B's parameter + next step match A's —
        // i.e. the snapshot captured the full trajectory (moments + step_t), not
        // just the weights. This is the resume invariant in miniature.
        let (w_a, mut opt_a) = setup();
        for _ in 0..3 {
            let g = grad_toward(&w_a, 5.0);
            opt_a.step(&g).unwrap();
        }
        let (moments, step_t) = opt_a.state().unwrap();
        assert_eq!(step_t, 3);
        let w_after_3: Vec<f32> = w_a.as_tensor().to_vec1().unwrap();

        // Continue A two more steps → the reference trajectory.
        for _ in 0..2 {
            let g = grad_toward(&w_a, 5.0);
            opt_a.step(&g).unwrap();
        }
        let w_a_final: Vec<f32> = w_a.as_tensor().to_vec1().unwrap();

        // B starts from A's epoch-3 weights, loads A's epoch-3 optimizer state,
        // and takes the same two steps. Without the loaded moments/step_t it
        // would diverge (zero moments + step_t=1 bias correction).
        let dev = Device::Cpu;
        let w_b =
            Var::from_tensor(&Tensor::from_vec(w_after_3.clone(), (4,), &dev).unwrap()).unwrap();
        let mut opt_b = AdamW::new(
            vec![w_b.clone()],
            ParamsAdamW {
                lr: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        opt_b.load_state(&moments, step_t).unwrap();
        assert_eq!(opt_b.step_t(), 3);
        for _ in 0..2 {
            let g = grad_toward(&w_b, 5.0);
            opt_b.step(&g).unwrap();
        }
        let w_b_final: Vec<f32> = w_b.as_tensor().to_vec1().unwrap();

        for (a, b) in w_a_final.iter().zip(&w_b_final) {
            assert!(
                (a - b).abs() < 1e-6,
                "resumed trajectory diverged: {a} vs {b}"
            );
        }
    }

    #[test]
    fn load_state_rejects_a_mismatched_parameter_count() {
        let (_w, mut opt) = setup();
        let err = opt.load_state(&[], 1).unwrap_err().to_string();
        assert!(err.contains("moment pairs"), "got: {err}");
    }

    /// MUT-triage: nothing above ever calls `learning_rate()` and checks its
    /// return, so a mutant hardcoding `0.0`/`1.0`/`-1.0` survives; nothing
    /// checks `set_learning_rate` actually mutates `self.params.lr` (a
    /// mutant replacing its body with `()` survives too, since every OTHER
    /// test only observes lr's effect indirectly through a step). Read the
    /// getter back after both `new` and `set_learning_rate` to pin both.
    #[test]
    fn learning_rate_getter_reflects_construction_and_set_learning_rate() {
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::zeros((2,), DType::F32, &dev).unwrap()).unwrap();
        let opt = AdamW::new(
            vec![w],
            ParamsAdamW {
                lr: 0.0123,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(opt.learning_rate(), 0.0123);
        let mut opt = opt;
        opt.set_learning_rate(0.0456);
        assert_eq!(opt.learning_rate(), 0.0456);
    }

    /// MUT-triage: `advance_step_scales`'s bias-correction division
    /// (`scale_m = 1f64 / (1f64 - beta1.powi(t))`) is SHARED by both the
    /// fused and eager arms — a mutant that breaks it (`/` -> `*`) breaks
    /// BOTH arms IDENTICALLY, so `dispatch_arms`'s fused-vs-eager bit-
    /// identity oracle CANNOT see it (this is exactly family C/F's "one
    /// root cause, second home": a shared-bookkeeping bug is invisible to a
    /// cross-arm comparison alone). This test is an INDEPENDENT oracle —
    /// the expected `next_theta` is computed here with its OWN
    /// from-scratch formula (not by calling `advance_step_scales` or
    /// `step_eager_one`/`step_fused_one`), reproducing the SAME per-op `f32`
    /// rounding sequence candle's eager chain (and, bit-identically per
    /// `adamw_step.rs`'s own proven claim, the fused kernel) performs — a
    /// single f64 computation rounded once at the end would NOT bit-match
    /// (each op rounds separately in the real code), so this is written as
    /// the same sequence of individually-rounded `f32` ops, just typed out
    /// fresh here rather than by calling production code — and compared
    /// against the real `AdamW::step`'s output to the exact bit pattern. On
    /// CPU with F32/contiguous/matching-shape inputs the predicate admits
    /// the FUSED arm (this test exercises whichever arm `step` actually
    /// picks, matching production behaviour), so this is simultaneously an
    /// absolute (not merely cross-arm) correctness oracle for both arms
    /// through the code path a caller actually takes.
    #[test]
    fn one_step_matches_an_independently_hand_computed_adam_update() {
        let dev = Device::Cpu;
        let theta0: f32 = 1.0;
        let g0: f32 = 0.25;
        let beta1 = 0.9_f64;
        let beta2 = 0.999_f64;
        let lr = 0.01_f64;
        let weight_decay = 0.02_f64;
        let eps = 1e-8_f64;

        let w = Var::from_tensor(&Tensor::from_vec(vec![theta0], (1,), &dev).unwrap()).unwrap();
        let mut opt = AdamW::new(
            vec![w.clone()],
            ParamsAdamW {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            },
        )
        .unwrap();
        let mut grads = GradStore::default();
        grads.insert(
            w.as_tensor(),
            Tensor::from_vec(vec![g0], (1,), &dev).unwrap(),
        );
        opt.step(&grads).unwrap();

        // Independent formula, same per-op f32 rounding sequence as the
        // production code (m0 = v0 = 0, t = 1), host scalars cast to f32
        // via `as f32` exactly as `WithDType::from_f64` does:
        let beta1_f32 = beta1 as f32;
        let beta2_f32 = beta2 as f32;
        let one_minus_beta1 = (1.0 - beta1) as f32;
        let one_minus_beta2 = (1.0 - beta2) as f32;
        let m1 = beta1_f32 * 0.0f32 + one_minus_beta1 * g0; // = one_minus_beta1 * g0
        let g_sq = g0 * g0;
        let v1 = beta2_f32 * 0.0f32 + one_minus_beta2 * g_sq;
        let scale_m = (1.0 / (1.0 - beta1.powi(1))) as f32;
        let scale_v = (1.0 / (1.0 - beta2.powi(1))) as f32;
        let m_hat = m1 * scale_m;
        let v_hat = v1 * scale_v;
        let eps_f32 = eps as f32;
        let denom = v_hat.sqrt() + eps_f32;
        let adjusted_grad = m_hat / denom;
        let one_minus_lr_lambda = (1.0 - lr * weight_decay) as f32;
        let lr_f32 = lr as f32;
        let expected_theta = theta0 * one_minus_lr_lambda - adjusted_grad * lr_f32;

        let got: Vec<f32> = w.as_tensor().to_vec1().unwrap();
        assert_eq!(
            got[0].to_bits(),
            expected_theta.to_bits(),
            "got={} expected={}",
            got[0],
            expected_theta
        );
    }
}

/// MUT-triage (cargo-mutants, direct unit tests on `fused_admission_predicate`
/// itself — every `dispatch_arms` test below only exercises it INDIRECTLY
/// through `AdamW::step` with uniformly-valid inputs, so the predicate's own
/// return value — both the bool AND the specific `&'static str` reason — was
/// never independently pinned. Closes: `(true, "")`/`(true, "xyzzy")`
/// whole-function-replace mutants (undetectable unless a test asserts the
/// EXACT string, not just that dispatch went to the fused arm) and `||` ->
/// `&&` swaps in the dtype/contiguity/shape OR-chains (undetectable unless a
/// test supplies a MISMATCH on each individual clause in turn, not just an
/// all-valid or all-invalid input).
#[cfg(test)]
mod admission_predicate {
    use super::*;
    use candle_core::{DType, Device};

    fn valid_quad() -> (Tensor, Tensor, Tensor, Tensor) {
        let dev = Device::Cpu;
        let theta = Tensor::zeros((2, 3), DType::F32, &dev).unwrap();
        let m = Tensor::zeros((2, 3), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((2, 3), DType::F32, &dev).unwrap();
        let g = Tensor::zeros((2, 3), DType::F32, &dev).unwrap();
        (theta, m, v, g)
    }

    #[test]
    fn all_valid_returns_true_domain_ok_exactly() {
        let (theta, m, v, g) = valid_quad();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (true, "domain_ok")
        );
    }

    #[test]
    fn theta_dtype_mismatch_is_refused_with_the_exact_reason() {
        let (_, m, v, g) = valid_quad();
        let theta = Tensor::zeros((2, 3), DType::F64, &Device::Cpu).unwrap();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "dtype_f32")
        );
    }

    #[test]
    fn first_moment_dtype_mismatch_is_refused_with_the_exact_reason() {
        let (theta, _, v, g) = valid_quad();
        let m = Tensor::zeros((2, 3), DType::F64, &Device::Cpu).unwrap();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "dtype_f32")
        );
    }

    #[test]
    fn second_moment_dtype_mismatch_is_refused_with_the_exact_reason() {
        let (theta, m, _, g) = valid_quad();
        let v = Tensor::zeros((2, 3), DType::F64, &Device::Cpu).unwrap();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "dtype_f32")
        );
    }

    #[test]
    fn grad_dtype_mismatch_is_refused_with_the_exact_reason() {
        let (theta, m, v, _) = valid_quad();
        let g = Tensor::zeros((2, 3), DType::F64, &Device::Cpu).unwrap();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "dtype_f32")
        );
    }

    /// A `.t()` view is non-contiguous — the cheapest way to construct a
    /// domain-invalid-but-otherwise-well-formed tensor on CPU alone.
    fn transposed_noncontiguous(dev: &Device) -> Tensor {
        let base = Tensor::zeros((3, 2), DType::F32, dev).unwrap();
        let view = base.t().unwrap();
        assert!(!view.is_contiguous());
        view
    }

    #[test]
    fn theta_noncontiguous_is_refused_with_the_exact_reason() {
        let dev = Device::Cpu;
        let theta = transposed_noncontiguous(&dev);
        let (_, m, v, g) = valid_quad();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "theta_moments_grad_all_contiguous")
        );
    }

    #[test]
    fn first_moment_noncontiguous_is_refused_with_the_exact_reason() {
        let dev = Device::Cpu;
        let m = transposed_noncontiguous(&dev);
        let (theta, _, v, g) = valid_quad();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "theta_moments_grad_all_contiguous")
        );
    }

    #[test]
    fn second_moment_noncontiguous_is_refused_with_the_exact_reason() {
        let dev = Device::Cpu;
        let v = transposed_noncontiguous(&dev);
        let (theta, m, _, g) = valid_quad();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "theta_moments_grad_all_contiguous")
        );
    }

    #[test]
    fn grad_noncontiguous_is_refused_with_the_exact_reason() {
        let dev = Device::Cpu;
        let g = transposed_noncontiguous(&dev);
        let (theta, m, v, _) = valid_quad();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "theta_moments_grad_all_contiguous")
        );
    }

    #[test]
    fn first_moment_shape_mismatch_is_refused_with_the_exact_reason() {
        let (theta, _, v, g) = valid_quad();
        let m = Tensor::zeros((3, 2), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "theta_moments_grad_share_one_shape")
        );
    }

    #[test]
    fn second_moment_shape_mismatch_is_refused_with_the_exact_reason() {
        let (theta, m, _, g) = valid_quad();
        let v = Tensor::zeros((3, 2), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "theta_moments_grad_share_one_shape")
        );
    }

    #[test]
    fn grad_shape_mismatch_is_refused_with_the_exact_reason() {
        let (theta, m, v, _) = valid_quad();
        let g = Tensor::zeros((3, 2), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "theta_moments_grad_share_one_shape")
        );
    }

    /// MUT-triage: the device-agreement OR-chain (`fused_admission_predicate`,
    /// `theta.device().same_device(..)` × 3) is NOT constructible on a
    /// CPU-only build — `Device::Cpu` is a singleton-like variant
    /// (`same_device(Cpu, Cpu)` is always `true`; candle-core-0.11.0's
    /// `Device::same_device`, `device.rs:294-301`), so no two CPU tensors
    /// can ever disagree on device, and this build has no `metal` feature
    /// to construct a third device kind. A REAL mismatch needs a second
    /// device kind (CUDA), so this is `#[cfg(feature = "cuda")]`-only —
    /// exercised on the CUDA-enabled pod leg of this dispatch's acceptance
    /// gate, not the default hermetic `cargo test -p jammi-ai` lane. This is
    /// a disclosed, not silent, gap in the CPU-only mutation sweep (see the
    /// hand-off's mutation-triage table) for exactly this reason.
    #[test]
    #[cfg(feature = "cuda")]
    fn grad_on_a_different_device_is_refused_with_the_exact_reason() {
        let dev_cpu = Device::Cpu;
        let dev_cuda = Device::new_cuda(0).expect("a CUDA device must be available on this leg");
        let (theta, m, v, _) = valid_quad();
        let g = Tensor::zeros((2, 3), DType::F32, &dev_cuda).unwrap();
        let _ = &dev_cpu;
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "theta_moments_grad_share_one_device")
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn second_moment_on_a_different_device_is_refused_with_the_exact_reason() {
        let dev_cuda = Device::new_cuda(0).expect("a CUDA device must be available on this leg");
        let (theta, m, _, g) = valid_quad();
        let v = Tensor::zeros((2, 3), DType::F32, &dev_cuda).unwrap();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "theta_moments_grad_share_one_device")
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn first_moment_on_a_different_device_is_refused_with_the_exact_reason() {
        let dev_cuda = Device::new_cuda(0).expect("a CUDA device must be available on this leg");
        let (theta, _, v, g) = valid_quad();
        let m = Tensor::zeros((2, 3), DType::F32, &dev_cuda).unwrap();
        assert_eq!(
            fused_admission_predicate(&theta, &m, &v, &g),
            (false, "theta_moments_grad_share_one_device")
        );
    }
}

/// RED-oracle: the fused (`InplaceOp2`/`InplaceOp3`) arm of `AdamW::step`
/// must be bit-identical to the eager candle-op chain it replaces, at
/// production LoRA shapes, over multiple consecutive steps.
///
/// **Forcing mechanism, and why it differs from the dispatch's literal
/// `JAMMI_KERNELS_DISABLE=<op key>` clause (scope amendment):** K-aux
/// (`feat/kernels-admission-disable` @ e602d7a) IS merged onto this branch's
/// `main` base — `crate::admission` does carry the env-var disable switch
/// here (`jammi_kernels::admission::admit` honours `JAMMI_KERNELS_DISABLE`;
/// see `crates/jammi-kernels/src/admission.rs`'s own module doc). This
/// commit (`feat(ai): wire AdamW::step to the fused multi-tensor AdamW
/// kernel`) was authored against an earlier base that predated K-aux and is
/// carried onto this branch VERBATIM (byte-identical to
/// `perf/multi-tensor-adamw`'s own commit, per the lead's cherry-pick
/// authorization), so [`AdamW::step_forced`]
/// (`#[cfg(test)]`-only, not part of the production API) remains this RED
/// oracle's forcing mechanism rather than being reshaped into an
/// env-var-driven test: it bypasses `fused_admission_predicate`/`admit`
/// entirely and calls [`step_fused_one`]/[`step_eager_one`] directly,
/// sharing the exact same `step_t`/bias-correction bookkeeping as
/// production `step` ([`AdamW::advance_step_scales`]) so the two arms
/// differ ONLY in which update function runs — the same "same-build
/// forced-arm A/B" shape as every other fused op's oracle in this
/// workspace. The gate's own pod leg exercises the SAME production
/// dispatch through the real `JAMMI_KERNELS_DISABLE=adamw_step_fused`
/// env-var switch end-to-end (`crates/jammi-bench`'s `finetune-step`
/// tier), so the env-var path this doc once claimed was unavailable is
/// independently proven live, not merely asserted.
#[cfg(test)]
mod dispatch_arms {
    use super::*;
    use candle_core::Device;

    /// Fixed-seed SplitMix64 (family J: no unseeded RNG). A local
    /// `#[cfg(test)]`-only copy of the well-known SplitMix64 generator
    /// `crates/jammi-lora/src/seeded.rs` already uses for LoRA seed draws;
    /// that copy is `pub(crate)` to `jammi-lora` and unreachable from here,
    /// so this file carries its own minimal copy rather than depend on a
    /// visibility change to a crate this one does not otherwise need.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// A uniform `f32` in `[-scale, scale)` — "production amplitude"
        /// is set by the caller's `scale` (LoRA-init-realistic: ~0.02 for
        /// weights, ~1e-3 for gradients).
        fn next_f32(&mut self, scale: f32) -> f32 {
            let bits = (self.next_u64() >> 40) as u32; // top 24 bits.
            let unit = bits as f32 / (1u32 << 24) as f32; // [0, 1).
            (unit * 2.0 - 1.0) * scale
        }
    }

    /// The four production LoRA shapes this dispatch names, derived from
    /// `crates/jammi-lora/src/lora_linear.rs`'s `lora_a`
    /// (`(rank, in_features)`) / `lora_b` (`(out_features, rank)`)
    /// allocations at `rank = 16`: `[16,1024]` is a `Wqkv`/`Wo`/`Wi`
    /// `lora_a`; `[3072,16]` is `Wi`'s `lora_b` (`out_features = 3072`);
    /// `[1024,16]` is `Wo`'s `lora_b`; `[5248,16]` is the fused-QKV
    /// `lora_b`.
    const PROD_SHAPES: [(usize, usize); 4] = [(16, 1024), (3072, 16), (1024, 16), (5248, 16)];

    fn filled(seed: u64, shape: (usize, usize), scale: f32, dev: &Device) -> Tensor {
        let mut rng = SplitMix64(seed);
        let n = shape.0 * shape.1;
        let data: Vec<f32> = (0..n).map(|_| rng.next_f32(scale)).collect();
        Tensor::from_vec(data, shape, dev).unwrap()
    }

    /// Builds one `AdamW` over [`PROD_SHAPES`] plus the caller's kept clone
    /// of each `theta` `Var` (so the test can read `theta` back after a
    /// step without going through `AdamW::state`, which reports only the
    /// moments — matching the pattern the existing `state_round_trip...`
    /// test in `super::tests` already uses via `Var::clone`'s shared
    /// storage).
    fn setup_production(params: ParamsAdamW) -> (Vec<Var>, AdamW) {
        let dev = Device::Cpu;
        let thetas: Vec<Var> = PROD_SHAPES
            .iter()
            .enumerate()
            .map(|(i, &shape)| {
                Var::from_tensor(&filled(1000 + i as u64, shape, 0.02, &dev)).unwrap()
            })
            .collect();
        let kept = thetas.clone();
        let opt = AdamW::new(thetas, params).unwrap();
        (kept, opt)
    }

    /// Builds a `GradStore` keyed by `thetas` (one grad per `Var`, at
    /// production amplitude, seeded independently of the theta draw), with
    /// `skip_index` (if `Some`) omitted entirely — exercising the
    /// `if let Some(g) = grads.get(theta)` skip guard exactly as a real
    /// step where one parameter received no gradient this batch would.
    fn grads_for(thetas: &[Var], step: u64, skip_index: Option<usize>) -> GradStore {
        let dev = Device::Cpu;
        let mut store = GradStore::default();
        for (i, theta) in thetas.iter().enumerate() {
            if Some(i) == skip_index {
                continue;
            }
            let shape = (theta.dims()[0], theta.dims()[1]);
            let g = filled(9_000_000 + step * 100 + i as u64, shape, 1e-3, &dev);
            store.insert(theta.as_tensor(), g);
        }
        store
    }

    /// Non-panicking bit-identity check, `Vec<f32>` level — the shared
    /// primitive both [`assert_bit_identical_tensors`] (the oracle every
    /// PASS test in this module trusts) and
    /// `negative_control_a_one_ulp_perturbation_is_detected` (which needs a
    /// boolean result, not a panic, to prove the oracle catches a deliberate
    /// mismatch) call.
    fn tensors_bit_identical(a: &Tensor, b: &Tensor) -> bool {
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn assert_bit_identical_tensors(a: &Tensor, b: &Tensor, ctx: &str) {
        assert!(tensors_bit_identical(a, b), "{ctx}: bit mismatch");
    }

    /// THE RED-oracle: ≥3 consecutive `AdamW::step`s over the 4
    /// production-shaped `Var`s, lr changing every step, weight_decay
    /// nonzero, one `Var` without a grad on the middle step — fused vs
    /// eager arm, forced via [`AdamW::step_forced`], must be bit-identical
    /// in theta (read back through the kept `Var` clones), `state()`
    /// (moments), and `step_t`.
    #[test]
    fn fused_and_eager_arms_match_bit_for_bit_over_three_production_steps() {
        let params = ParamsAdamW {
            lr: 1e-3,
            weight_decay: 0.01,
            ..Default::default()
        };
        let (fused_thetas, mut fused_opt) = setup_production(params.clone());
        let (eager_thetas, mut eager_opt) = setup_production(params);

        // Same initial theta values on both sides — `setup_production` draws
        // from the SAME seeds each call, so this is itself an oracle: assert
        // it before trusting anything downstream.
        for (f, e) in fused_thetas.iter().zip(&eager_thetas) {
            assert_bit_identical_tensors(f.as_tensor(), e.as_tensor(), "initial theta");
        }

        let lrs = [1e-3, 5e-4, 2e-3];
        // Middle step: the `Var` at `PROD_SHAPES[3]` (0-indexed, the fused-QKV
        // `lora_b`, `[5248,16]` — see `PROD_SHAPES`'s own doc) gets no grad.
        let skip = [None, Some(3), None];
        for (step, (&lr, &skip_idx)) in lrs.iter().zip(&skip).enumerate() {
            fused_opt.set_learning_rate(lr);
            eager_opt.set_learning_rate(lr);
            let g_fused = grads_for(&fused_thetas, step as u64, skip_idx);
            let g_eager = grads_for(&eager_thetas, step as u64, skip_idx);
            fused_opt.step_forced(&g_fused, true).unwrap();
            eager_opt.step_forced(&g_eager, false).unwrap();
        }

        assert_eq!(fused_opt.step_t(), 3);
        assert_eq!(eager_opt.step_t(), 3);

        // Finiteness affirmative first (family F), then bit identity, then
        // the ‖Δθ‖ > 0 movement signal (skip the untouched Var — it never
        // received a grad and must stay AT its initial value, not move).
        for (i, (f, e)) in fused_thetas.iter().zip(&eager_thetas).enumerate() {
            let fv: Vec<f32> = f.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                fv.iter().all(|x| x.is_finite()),
                "var {i}: fused theta not finite"
            );
            assert_bit_identical_tensors(f.as_tensor(), e.as_tensor(), &format!("theta[{i}]"));
        }
        let delta: f32 = {
            let fv: Vec<f32> = fused_thetas[0]
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let initial = filled(1000, PROD_SHAPES[0], 0.02, &Device::Cpu)
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            fv.iter().zip(&initial).map(|(a, b)| (a - b).abs()).sum()
        };
        assert!(delta > 0.0, "var 0 theta did not move over 3 steps");

        let (fused_moments, fused_step_t) = fused_opt.state().unwrap();
        let (eager_moments, eager_step_t) = eager_opt.state().unwrap();
        assert_eq!(fused_step_t, eager_step_t);
        for (i, ((fm, fv2), (em, ev2))) in fused_moments.iter().zip(&eager_moments).enumerate() {
            assert_bit_identical_tensors(fm, em, &format!("first_moment[{i}]"));
            assert_bit_identical_tensors(fv2, ev2, &format!("second_moment[{i}]"));
        }
    }

    /// Same oracle with `weight_decay = 0.0` — the dispatch's semantics
    /// clause explicitly names both the zero and nonzero decoupled-decay
    /// paths.
    #[test]
    fn fused_and_eager_arms_match_bit_for_bit_with_zero_weight_decay() {
        let params = ParamsAdamW {
            lr: 5e-4,
            weight_decay: 0.0,
            ..Default::default()
        };
        let (fused_thetas, mut fused_opt) = setup_production(params.clone());
        let (eager_thetas, mut eager_opt) = setup_production(params);
        for step in 0..3u64 {
            let g_fused = grads_for(&fused_thetas, step, None);
            let g_eager = grads_for(&eager_thetas, step, None);
            fused_opt.step_forced(&g_fused, true).unwrap();
            eager_opt.step_forced(&g_eager, false).unwrap();
        }
        for (i, (f, e)) in fused_thetas.iter().zip(&eager_thetas).enumerate() {
            assert_bit_identical_tensors(f.as_tensor(), e.as_tensor(), &format!("theta[{i}]"));
        }
    }

    /// NaN-grad propagation parity: `optimizer.rs`'s off-cadence
    /// `refuse_nonfinite_norm` is what normally keeps a NaN gradient from
    /// reaching `AdamW::step` at all — this test is specifically the case
    /// where that guard is off-cadence (`optimizer.rs`, f0cad22), so a NaN
    /// DOES reach the op. The fused kernel must propagate it identically to
    /// the eager chain (neither "fixes" it nor diverges from it) — both
    /// arms are plain IEEE-754 f32 arithmetic with no NaN-guarding branch,
    /// so this is a parity claim, not a "NaN is handled" claim.
    #[test]
    fn nan_gradient_propagates_bit_identically_between_arms() {
        let dev = Device::Cpu;
        let shape = (4usize, 4usize);
        let theta_f = Var::from_tensor(&filled(42, shape, 0.02, &dev)).unwrap();
        let theta_e = Var::from_tensor(&filled(42, shape, 0.02, &dev)).unwrap();
        let params = ParamsAdamW {
            lr: 1e-3,
            weight_decay: 0.01,
            ..Default::default()
        };
        let mut fused_opt = AdamW::new(vec![theta_f.clone()], params.clone()).unwrap();
        let mut eager_opt = AdamW::new(vec![theta_e.clone()], params).unwrap();

        let mut g_data = vec![0.01f32; 16];
        g_data[3] = f32::NAN;
        g_data[9] = f32::NAN;
        let g = Tensor::from_vec(g_data, shape, &dev).unwrap();

        let mut gs_f = GradStore::default();
        gs_f.insert(theta_f.as_tensor(), g.clone());
        let mut gs_e = GradStore::default();
        gs_e.insert(theta_e.as_tensor(), g);

        fused_opt.step_forced(&gs_f, true).unwrap();
        eager_opt.step_forced(&gs_e, false).unwrap();

        let fv: Vec<f32> = theta_f
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let ev: Vec<f32> = theta_e
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, (f, e)) in fv.iter().zip(&ev).enumerate() {
            assert_eq!(
                f.is_nan(),
                e.is_nan(),
                "element {i}: NaN-ness diverged, fused={f} eager={e}"
            );
            if !f.is_nan() {
                assert_eq!(f.to_bits(), e.to_bits(), "element {i}: non-NaN mismatch");
            }
        }
        // At least the two grad-NaN positions (3, 9) propagated to theta,
        // AND further poisoned every element downstream of the shared
        // moment-update reduction is NOT expected here (this op is
        // strictly elementwise — no cross-element reduction), so exactly
        // the elements whose OWN gradient was NaN should be NaN in theta.
        assert!(fv[3].is_nan() && fv[9].is_nan(), "NaN did not propagate");
        assert!(
            fv.iter()
                .enumerate()
                .filter(|(i, _)| *i != 3 && *i != 9)
                .all(|(_, x)| x.is_finite()),
            "NaN leaked into an element whose own gradient was finite: {fv:?}"
        );
    }

    /// Dispatch counters must be NONZERO after real (unforced) steps on
    /// production Vars — zero dispatch is RED (family F). Uses a snapshot
    /// DELTA across this call (not an absolute count), since
    /// `ADAMW_DISPATCH_COUNTERS` is process-global and shared with every
    /// other test in this binary running concurrently.
    #[test]
    fn real_steps_on_production_vars_record_nonzero_fused_dispatches() {
        let params = ParamsAdamW {
            lr: 1e-3,
            ..Default::default()
        };
        let (_thetas, mut opt) = setup_production(params);
        let before = ADAMW_DISPATCH_COUNTERS.snapshot();
        for step in 0..3u64 {
            let g = grads_for(&_thetas, step, None);
            opt.step(&g).unwrap(); // the REAL dispatch path — not step_forced.
        }
        let after = ADAMW_DISPATCH_COUNTERS.snapshot();
        // CPU F32 contiguous same-shape production Vars satisfy
        // `fused_admission_predicate` unconditionally, and the default
        // admission mode is `Fallback` (not `Strict`) — so every one of the
        // 4 Vars × 3 steps = 12 calls this loop makes is expected to admit
        // Fused, and NONE Eager (a Fused-vs-Eager split here would itself
        // be a signal the predicate is wrong for a production shape).
        assert!(
            after.fused - before.fused >= 12,
            "expected >= 12 fused dispatches this run, got delta {} (before={before:?} after={after:?})",
            after.fused - before.fused
        );
        assert_eq!(
            after.eager, before.eager,
            "a production-shaped CPU F32 contiguous Var should never take the eager arm"
        );
    }

    /// `adamw_step_fused_t`'s own `validate_step_domain` refuses aliasing
    /// (`theta`/`first_moment`/`second_moment`/`grad` sharing storage) with
    /// a typed error BEFORE any `InplaceOpN` mutation runs — this test
    /// deliberately constructs the pathological case (the gradient handed
    /// back for `theta` is `theta.as_tensor().clone()` — a cheap `Arc`
    /// clone SHARING `theta`'s own storage, so `theta` and `grad` alias
    /// each other; never true in real training, where a `GradStore` entry
    /// comes from a distinct backward pass over a distinct tensor, but
    /// exercised here directly to prove `step`'s `Err` path leaves state
    /// untouched, not partially advanced) and asserts: (1) `step` returns
    /// `Err` naming the aliasing refusal; (2) `theta`/`first_moment`/
    /// `second_moment` are BIT-IDENTICAL to their pre-call values afterward
    /// (family F: measured against the actual pre-call snapshot, not
    /// assumed).
    #[test]
    fn an_aliased_var_is_refused_and_leaves_state_untouched() {
        let dev = Device::Cpu;
        let theta = Var::from_tensor(&filled(7, (4, 4), 0.02, &dev)).unwrap();
        let params = ParamsAdamW {
            lr: 1e-3,
            ..Default::default()
        };
        let mut opt = AdamW::new(vec![theta.clone()], params).unwrap();

        // Snapshot theta/m/v BEFORE the call (via `state()`/the kept `Var`).
        let theta_before: Vec<f32> = theta.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
        let (moments_before, step_t_before) = opt.state().unwrap();

        // `theta.as_tensor().clone()` is a cheap `Arc` clone SHARING
        // `theta`'s own storage — `grads` is keyed by `theta`'s id (so
        // `step` reaches the fused admission/dispatch at all), and the
        // VALUE handed back aliases `theta` itself, which
        // `validate_step_domain`'s aliasing guard refuses.
        let mut grads = GradStore::default();
        grads.insert(theta.as_tensor(), theta.as_tensor().clone());

        let err = opt.step(&grads).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("alias") || msg.contains("cannot set"),
            "expected an aliasing refusal, got: {msg}"
        );

        let theta_after: Vec<f32> = theta.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            theta_before, theta_after,
            "theta must be untouched after a refused (aliased) step"
        );
        let (moments_after, step_t_after) = opt.state().unwrap();
        // `advance_step_scales` increments `self.step_t` UNCONDITIONALLY,
        // before the per-Var loop that hits the aliasing error runs (see
        // `AdamW::step`'s body) — this is the one piece of `AdamW`'s state
        // NOT protected by `validate_step_domain`'s upfront-validation
        // guarantee, since it lives in `jammi-ai`, not inside the kernel
        // call itself. Documented explicitly here (family F: measure, don't
        // assume) rather than silently asserting the stronger "nothing
        // changed" claim, which would be false.
        assert_eq!(
            step_t_after,
            step_t_before + 1,
            "step_t advances even on a refused per-Var step (known, documented — see this \
             test's doc)"
        );
        for (i, ((mb, vb), (ma, va))) in moments_before.iter().zip(&moments_after).enumerate() {
            let mb: Vec<f32> = mb.flatten_all().unwrap().to_vec1().unwrap();
            let ma: Vec<f32> = ma.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(
                mb, ma,
                "first_moment[{i}] must be untouched after a refused step"
            );
            let vb: Vec<f32> = vb.flatten_all().unwrap().to_vec1().unwrap();
            let va: Vec<f32> = va.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(
                vb, va,
                "second_moment[{i}] must be untouched after a refused step"
            );
        }
    }

    /// RED CONTROL (family F): a one-ULP scalar change to a fused-arm
    /// result must be DETECTED by the same `to_bits()` equality this
    /// module's other tests rely on — proving the bit-identity oracle has
    /// power, at the ai-core trainer level (distinct from, and in addition
    /// to, `jammi-kernels::ops::adamw_step`'s own kernel-level negative
    /// control).
    #[test]
    fn negative_control_a_one_ulp_perturbation_is_detected() {
        let params = ParamsAdamW {
            lr: 1e-3,
            weight_decay: 0.01,
            ..Default::default()
        };
        let (fused_thetas, mut fused_opt) = setup_production(params.clone());
        let (eager_thetas, mut eager_opt) = setup_production(params);
        let g_fused = grads_for(&fused_thetas, 0, None);
        let g_eager = grads_for(&eager_thetas, 0, None);
        fused_opt.step_forced(&g_fused, true).unwrap();
        eager_opt.step_forced(&g_eager, false).unwrap();

        // Sanity: unperturbed, the two arms agree (this module's main
        // oracle already proves this more thoroughly; re-affirmed here so
        // the negative control below is contrasted against a KNOWN-PASSING
        // baseline, not assumed).
        assert_bit_identical_tensors(
            fused_thetas[0].as_tensor(),
            eager_thetas[0].as_tensor(),
            "control baseline",
        );

        // Perturb ONE element of the fused theta by exactly one ULP and
        // confirm the SAME boolean primitive `assert_bit_identical_tensors`
        // is built on (`tensors_bit_identical`) now reports `false`
        // (`candle_core::Tensor` is not `UnwindSafe` — it holds an
        // `Arc<Box<dyn CustomOp>>` inside its backprop graph — so this uses
        // a boolean-returning check rather than `catch_unwind`, which
        // `rustc` refuses to instantiate over a `&Tensor` closure).
        let mut fv: Vec<f32> = fused_thetas[0]
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let original_bits = fv[0].to_bits();
        fv[0] = f32::from_bits(original_bits ^ 1); // exactly one ULP.
        let perturbed = Tensor::from_vec(fv, PROD_SHAPES[0], &Device::Cpu).unwrap();

        assert!(
            !tensors_bit_identical(&perturbed, eager_thetas[0].as_tensor()),
            "a one-ULP perturbation must be caught by the bit-identity oracle, \
             but the comparison passed anyway — the oracle has no power"
        );
    }
}
