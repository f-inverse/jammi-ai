# Unit 63 H5 — kernel-mutant RED column: mutant M1

CONTRACT 63 (PLAN v2 delta 7) requires a kernel-mutant RED column whose
sensitivity claim is precisely bounded: **"detects a regression >= mutant
M"**, never "detects movement" and never "detects hitting some unspecified
floor." This directory pins that mutant — its exact patch, its base sha, its
sha256, and the honest, narrow claim the campaign is allowed to make from it.

## What M1 is

**File:** [`M1.patch`](./M1.patch) — a unified diff against
`4257cde6d51184475b3e798f5d7e9c3885a763ca` (this repo's `main`, the campaign's
recorded base sha) touching exactly one file,
`crates/jammi-kernels/src/ops/adamw_step.rs`, inside
`adamw_step_fused_t` — the production fused-AdamW entry point
`jammi-ai::fine_tune::adamw` calls on the campaign's fused arm (confirmed by
grep: `crates/jammi-ai/src/fine_tune/adamw.rs:411` calls
`adamw_step_fused_t`, not the deprecated `adamw_step_fused` caller-scale
shape).

**sha256(M1.patch):**
`68d2023b936fe613c75e85a49ad4c6de01fb27442ffb967db00107fbc012d926`

**Shape chosen: off-by-one bias correction (`t+1` instead of `t`).**

```diff
-    let scale_m = 1f64 / (1f64 - params.beta1.powi(t_i32));
-    let scale_v = 1f64 / (1f64 - params.beta2.powi(t_i32));
+    let t_i32_biased = t_i32 + 1;
+    let scale_m = 1f64 / (1f64 - params.beta1.powi(t_i32_biased));
+    let scale_v = 1f64 / (1f64 - params.beta2.powi(t_i32_biased));
```

### Why this shape, not the other two candidates in the prompt

- **Constant-scale lr inflation (e.g. ×1.05 on the final update)** was
  rejected as the primary pick: it is a real defect class, but it is also
  *linear and global* — its effect on held-out loss is essentially "train at
  a different lr," which a learning-rate-sweep control could confound with
  a legitimate hyperparameter difference rather than a kernel defect. It
  remains a good SECOND mutant candidate (M2, not authored here) if the gate
  needs a magnitude-dial family.
- **Flash-attention softmax-scale-off-by-a-small-factor** was rejected for
  M1 because it is not CPU-reachable with a clean twin in this crate the way
  the fused-AdamW arithmetic is (attention's CPU fallback path is a
  different code shape than its CUDA kernel, so a CPU demonstration would
  not exercise the same defect the campaign's CUDA legs would run) — the
  task explicitly prefers a shape with a CPU-verifiable demonstration.
- **Off-by-one bias correction** is the strongest fit for all three
  requirements the task lists:
  - **Deterministic**: pure function of `t`, `beta1`, `beta2` — no RNG, no
    hardware-dependent FMA contraction, reproducible on CPU and CUDA alike.
  - **Direction-predictable**: `beta1, beta2 in (0, 1)`, so
    `beta^(t+1) < beta^t` for every `t >= 1`, so
    `1 - beta^(t+1) > 1 - beta^t`, so the biased `scale_m`/`scale_v` are
    **strictly smaller in magnitude** than the correct ones at every step.
    Smaller `scale_m`/`scale_v` under-corrects the EMA's cold-start bias,
    which shrinks `adjusted_grad = m_hat / (sqrt(v_hat) + eps)` — the
    parameter update is silently DEFLATED, not inflated, at every step (the
    effect is largest at small `t` and asymptotically vanishes as
    `t -> infinity`, since both `beta^t` and `beta^(t+1)` -> 0 either way).
    This is the textbook "wrong but still converging" regression: training
    proceeds, loss still goes down, nothing crashes or produces NaN/Inf —
    it just converges to a WORSE point than the correct kernel would, most
    detectably in the early-training regime where bias correction matters
    most. That is exactly the class a naive "did it crash / does the loss
    curve look monotonic" check would MISS and only a held-out-loss sign
    test against a control arm would catch.
  - **Realistic**: an off-by-one on a step counter feeding a bias-correction
    exponent is a classic, plausible silent bug (pre- vs post-increment
    counter, a resume path that doesn't restore `t` correctly, a caller
    passing `steps_completed` where the callee expects `steps_completed + 1`
    convention) — not a contrived pathology.
  - **Arch-independent**: the perturbation is in the shared scalar-derivation
    code (`t_i32_biased`, computed once in Rust before either the CPU or
    CUDA `InplaceOpN::*_fwd` arm runs), so it affects both A100 legs and any
    CPU legs identically — no CUDA-only PTX-intrinsic surgery needed.

### What M1 does NOT touch (by design)

- `InplaceOp2::name()` / `InplaceOp3::name()` (the strings that feed every
  typed error's `op` field) — unchanged, so error messages/error-variant
  identity are unaffected.
- `validate_step_domain` (dtype/shape/device/aliasing/contiguity/injectivity
  checks) — unchanged, so the mutant leg's ADMISSION behavior is identical
  to the unmutated fused arm. A shape/dtype/device mismatch is refused
  exactly the same way on a mutant leg as on a clean leg.
- Any dispatch counter, `DispatchCounters` field, or the `admission` module
  — the mutant leg's premise fields (which kernel ran, whether it was
  admitted, how many times it was invoked) read IDENTICAL to a clean fused
  leg. **This is the point**: the mutant leg must look premise-clean so that
  only the DECISION statistic (held-out loss, via the sign test) — not a
  cheaper proxy like "did the counters look right" — is what catches the
  regression. A gate that only checked premise fields would pass every
  mutant leg; the campaign's actual claim is that its DECISION statistic
  does not.
- `AdamMomentUpdate`/`AdamThetaUpdate`'s per-element arithmetic (the
  bit-identity-pinned rounding chain documented at the top of
  `adamw_step.rs`) — unchanged; only the SCALAR fed into that arithmetic
  (`scale_m`/`scale_v`) is wrong, not the arithmetic itself. This keeps M1 a
  single-fact perturbation, not a shotgun of unrelated changes.

## Expected direction and rough magnitude

- **Direction**: held-out loss on the mutant column is expected to be
  WORSE (higher) than the campaign's `alloff` (or clean fused) column,
  at matched step count — a one-sided degradation, not a two-sided
  "different" result. The sign test's null is "mutant column is no worse
  than alloff"; M1 is constructed so that null is expected to be REJECTED
  (RED).
- **Rough magnitude** (from the CPU demonstration below, at the campaign's
  own `beta1=0.9, beta2=0.999`): the per-step L2 perturbation in `theta`
  starts around `5e-4` at `t=1` (lr `1e-3`, so a ~50% relative perturbation
  on the FIRST step's update magnitude) and grows toward roughly `1.2e-3`
  by `t=5` before the growth rate slows as `t` increases and the bias
  correction terms for both the correct and mutant formula converge toward
  the same asymptote. The single-step perturbation is on the same ORDER as
  the learning rate itself in early steps — large enough to be a measurable
  training-dynamics regression, small enough that the run does not diverge
  or produce non-finite values (this is a "wrong but still converging"
  mutant, not a crash).

## On-pod procedure (mutant legs only — never merged)

1. On the pod, clone/checkout at the recorded base sha
   `4257cde6d51184475b3e798f5d7e9c3885a763ca` into a **scratch worktree**,
   separate from the campaign's production checkout:
   ```sh
   git worktree add /workspace/scratch-mutant-m1 4257cde6d51184475b3e798f5d7e9c3885a763ca --detach
   cd /workspace/scratch-mutant-m1
   ```
2. Verify the patch applies cleanly and record its hash before touching
   anything:
   ```sh
   sha256sum docs/plans/63-how-well/mutants/M1.patch   # must equal the README's recorded hash
   git apply --check docs/plans/63-how-well/mutants/M1.patch && echo CLEAN
   git apply docs/plans/63-how-well/mutants/M1.patch
   ```
3. Build with the campaign's exact feature list (same CUDA/bf16 features as
   the fused arm — do not add or drop features relative to the campaign's
   recorded build invocation):
   ```sh
   cargo build --release -p jammi-kernels --features cuda
   cargo build --release -p jammi-ai --features <campaign's exact feature list>
   ```
4. Run N mutant fused legs through the **same `run_leg` vector** the
   campaign's fused arm uses (same shapes, same seeds, same step count, same
   `attention_block_flash` + `adamw_step_fused` dispatch wiring) — the ONLY
   difference between a mutant leg and a clean fused leg is that this
   scratch build's `jammi-kernels` binary was compiled from the patched
   source.
5. Record, per mutant leg, in the column artifact:
   - `base_sha = 4257cde6d51184475b3e798f5d7e9c3885a763ca`
   - `patch_sha256 = 68d2023b936fe613c75e85a49ad4c6de01fb27442ffb967db00107fbc012d926`
   - `mutant_id = M1`
   - the same held-out-loss / step-count / dispatch-counter fields the
     clean fused legs record, so the mutant column is diff-able against the
     `alloff`/fused columns field-for-field.
6. **The mutant column artifact is honestly labeled as a MUTANT column and
   is never mixed into the A/B (fused vs alloff) comparison set.** It is a
   THIRD column, used only for the sign-test-power claim below — not
   folded into, averaged with, or substituted for either arm of the actual
   A/B campaign.
7. Tear down the scratch worktree and its build artifacts after the legs
   complete; do not leave the patched binary or scratch checkout on the pod
   past the mutant-leg run. The patch is committed to this repo as a FILE
   only (`M1.patch`) — it must never land as tree state on any branch that
   builds the production `jammi-kernels` binary.

## Pass criterion

The kernel-mutant RED column's claim is exactly: **a sign test over the
mutant (M1) column's held-out loss vs. the campaign's `alloff` legs' held-out
loss (or, equivalently, vs. the clean `fused` column) reads RED
(degradation) at the campaign's pre-registered significance threshold.**

This is the FULL claim and no more:
- It is a claim about DETECTING a regression of AT LEAST M1's magnitude —
  not a claim that the gate detects every possible regression, and not a
  claim about the gate's behavior on a smaller perturbation than M1.
- It is not a claim about "movement" (any observed difference, however
  small or in whatever direction) and not a claim about "hitting a floor"
  (some absolute loss threshold) — only the specific one-sided degradation
  direction M1 is constructed to produce.
- If the sign test does NOT read RED on the M1 column, that is itself a
  finding about the gate's sensitivity floor (the gate does not reliably
  detect a regression of M1's magnitude) and must be reported as such, not
  suppressed or retried with a larger, undocumented mutant.

## CPU-verifiable demonstration (not committed to the production suite)

A temporary, non-committed unit test was added locally
(`crates/jammi-kernels/src/ops/adamw_step.rs`, function
`mutant_m1_demonstration_diverges_from_the_correct_oracle`) that runs the
M1-patched `adamw_step_fused_t` against the file's own correct
`eager_step` oracle (unpatched formula, true `t`) for 5 consecutive steps
on a fixed 4-element input (`beta1=0.9, beta2=0.999, lr=1e-3,
weight_decay=0.01, eps=1e-8`). Output (`cargo test -p jammi-kernels --lib
mutant_m1_demonstration -- --nocapture`, patch applied, CPU only):

```
step=1 l2_divergence=5.117431e-4
step=2 l2_divergence=7.9484284e-4
step=3 l2_divergence=9.754748e-4
step=4 l2_divergence=1.0981262e-3
step=5 l2_divergence=1.1838377e-3
```

With the same patch applied, the file's own pre-existing bit-identity
oracle tests (which pin the CORRECT formula, not M1's) go RED as expected:
`cargo test -p jammi-kernels --lib adamw_step` reports **6 of 28 tests
failing** under M1 (`one_step_matches_the_eager_chain_bit_for_bit`,
`three_consecutive_steps_with_a_changing_lr_match_bit_for_bit`,
`zero_weight_decay_matches_bit_for_bit`, `single_element_matches_bit_for_bit`,
`large_eps_relative_to_v_hat_matches_bit_for_bit`,
`non_contiguous_grad_view_is_still_correct_on_cpu`) — confirming M1 is a
real, detectable numeric perturbation on this crate's own oracle, not a
no-op. `git checkout -- crates/jammi-kernels/src/ops/adamw_step.rs` restores
the clean tree afterward; neither the patch nor the temporary test is part
of any commit on this branch.

## Files

- `M1.patch` — the committed unified diff (patch-file-only; never applied to
  tree state on this branch).
- `README.md` — this file.
