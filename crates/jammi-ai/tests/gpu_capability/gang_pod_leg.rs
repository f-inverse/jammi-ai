//! The pod-leg artifact producer for `fine_tune::collective::nccl` (plan
//! #500, U4b's pod-leg acceptance, contract §2c; plan row B8). This module
//! is introduced across two commits, deliberately in this order:
//!
//! 1. **This commit** registers [`GANG_POD_LEG_EPSILON`] and its derivation,
//!    alone — before any test reads it and before the pod run this ε gates
//!    ever measures anything. `ci/scripts/check_cuda_run_artifacts.py`'s
//!    rule (k) requires a `gang` artifact's `epsilon.registered_sha` to be a
//!    STRICT ancestor of the artifact's own measured tree (`git_sha`): an ε
//!    chosen after seeing the delta it excuses is not a tolerance, it is a
//!    rationalisation. Registering it in its own commit, ahead of the test
//!    and the run, is what makes "pre-registered" a checkable claim rather
//!    than a prose assertion.
//! 2. The next commit adds [`GANG_POD_LEG_EPSILON_REGISTERED_SHA`] — this
//!    commit's own sha, learned only once it exists — plus the rest of this
//!    module (the `#[test]`, the artifact writer, the hermetic
//!    synthetic-artifact oracle) and wires it into `main.rs`'s module list.
//!
//! ## Derivation
//!
//! [`GANG_POD_LEG_EPSILON`] is **2e-4**: double the CPU-hermetic floor
//! already established and tested in this crate
//! (`gather_exactness_w2_matches_w1_within_pre_registered_epsilon`'s
//! `GATHER_EXACTNESS_EPSILON = 1e-4`, `crates/jammi-ai/src/fine_tune/
//! trainer.rs` — itself `batch_bucket.rs`'s own bucket-padding-variance
//! `TOLERANCE`), to admit exactly the ONE further fp32 reduction
//! reassociation a real GPU pod run adds beyond what that CPU oracle
//! already measures: cuBLAS's own forward/backward accumulation order, plus
//! NCCL's ring/tree `ncclAllReduce` summation order, neither of which is
//! bound to match candle's CPU rank-ordered fold the hermetic oracle
//! exercises.
//!
//! That additional term is bounded analytically, not merely asserted: a
//! fp32 reassociation error over `n` summed terms is bounded by
//! `(n - 1) * eps_f32 * max|term|`, `eps_f32 = 2^-23 ≈ 1.19e-7`. This
//! fixture's global batch (world 2 × per-rank batch 2) sums at most 4
//! terms, so the bound is `≈ 3 * 1.19e-7 * O(1) ≈ 3.6e-7` — three orders of
//! magnitude under the 2e-4 registered here, leaving ample headroom while
//! staying more than three orders of magnitude BELOW the divergence a real
//! bug produces: `gather_exactness_w2_matches_w1_within_pre_registered_
//! epsilon`'s own EXECUTED red-proof (the gather replaced with a `.clone()`
//! that never runs it) measured a gathered-vs-reference loss divergence of
//! `0.37497652` vs `1.0657526` — an O(1) mistake, not a reassociation-noise
//! one. 2e-4 is therefore discriminative: it tolerates the extra GPU fold
//! this leg adds while remaining far below any gradient-routing hazard.
///
/// The pre-registered ε for the pod-leg's W=2×B vs W=1×2B per-epoch
/// training-loss reproducibility bound. See the module doc's "Derivation".
pub(crate) const GANG_POD_LEG_EPSILON: f32 = 2.0e-4;

/// [`GANG_POD_LEG_EPSILON`]'s own derivation, restated as the exact
/// `gang.epsilon.derivation` string the committed artifact carries — kept
/// as one constant so the module doc above and the artifact's own field
/// can never drift apart from each other.
pub(crate) const GANG_POD_LEG_EPSILON_DERIVATION: &str = "2e-4 = 2x the CPU-hermetic gather-exactness floor (1e-4, gather_exactness_w2_matches_w1_within_pre_registered_epsilon's GATHER_EXACTNESS_EPSILON in crates/jammi-ai/src/fine_tune/trainer.rs), admitting exactly one further fp32 reduction reassociation a real GPU pod run adds beyond that CPU oracle: cuBLAS's own forward/backward accumulation order plus NCCL's ring/tree ncclAllReduce summation order. Analytically bounded: an (n-1)*eps_f32*max|term| reassociation-error bound (eps_f32 = 2^-23 ~= 1.19e-7, n <= 4 terms this fixture's global batch sums) is ~= 3.6e-7, three orders of magnitude under 2e-4, while staying more than three orders of magnitude below the O(1) divergence a real gradient-routing bug produces (the same oracle's own executed red-proof measured 0.37497652 vs 1.0657526).";
