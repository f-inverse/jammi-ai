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

use candle_core::backprop::GradStore;
use candle_core::{Result, Tensor, Var};

/// Hyperparameters are candle's — reused verbatim so the update matches and a
/// caller configures one struct, not two.
pub use candle_nn::ParamsAdamW;

/// One trainable parameter and its Adam moment buffers (first/second moment),
/// held together so a step updates all three in lockstep.
struct AdamVar {
    var: Var,
    first_moment: Var,
    second_moment: Var,
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

    /// Take one AdamW step over `grads`. Identical to `candle_nn::AdamW::step`:
    /// EMA the moments, bias-correct, apply decoupled weight decay, then the
    /// bias-corrected adaptive update.
    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lr_lambda = lr * self.params.weight_decay;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for entry in self.vars.iter() {
            let theta = &entry.var;
            let m = &entry.first_moment;
            let v = &entry.second_moment;
            if let Some(g) = grads.get(theta) {
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
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
}
