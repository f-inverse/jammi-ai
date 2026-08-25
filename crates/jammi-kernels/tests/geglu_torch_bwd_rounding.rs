//! esc-045 (GH#374 round 2) op-level RED reproducer — `GegluFused`'s BF16
//! backward must differentiate the FORWARD'S OWN materialized,
//! bf16-rounded activation for `d_up`, never a hypothetical unrounded
//! f32 one. This is a self-consistency oracle (fwd vs. bwd of the SAME
//! kernel), not a fused-vs-eager comparison — see `geglu_oracles.rs`'s
//! module doc for why fused-vs-candle-eager is a separate, weaker leg on
//! this specific op (candle-eager is not the reference being matched
//! here; see `ops/geglu.rs`'s module doc's "esc-045 fix" section, and
//! ATen's own `mul_tensor_backward` / `GeluBackwardCUDAKernelImpl`, cited
//! there, for the reference this test IS anchored to).
//!
//! ## Why "differentiate the forward's own output" is the right property
//!
//! `GegluFused`'s BF16 forward computes `out = round2(round1(gelu(gate)) *
//! up)` — `round1` genuinely materializes a bf16-rounded activation
//! (matching the real two-op upstream reference: `gelu` then `mul` are
//! separate ops, and the separate `gelu` op's OUTPUT is what gets fed
//! into `mul`, at bf16 precision). Any backward that differentiates
//! w.r.t. `up` using the UNROUNDED f32 activation is differentiating a
//! DIFFERENT function than the one the forward actually computed — a
//! violation of "the gradient must be of the function that ran," not a
//! disclosable precision tradeoff.
//!
//! ## The fixture
//!
//! `FIXTURE` below is 214 real `(gate, up, dy)` triples read directly off
//! ModernBERT-large layer 18's `Wi` output during a real b8 s128 LoRA
//! r=16 training step (seed 42, ZerosB init, all fused kernels admitted —
//! esc-045's own defect-reproducing configuration), spanning this layer's
//! actual amplitude (`max|gate|` = 27.5, `max|up|` = 20.5 — production
//! amplitude per the kernel guide's §3.4, not a toy range) plus the
//! near-zero-gate boundary and the three outlier columns (812/833/2539)
//! esc-045's round-1 investigation flagged. Extracted on the pod
//! investigation box via `geglu_recheck.py`/`extract_fixture.py`
//! (pod-only, not checked in — this file is the checked-in, hermetic
//! result), each value printed at its own minimal round-trip precision.
//!
//! ## Non-vacuous control
//!
//! Before asserting anything about the kernel, this test proves the
//! fixture itself actually stresses the rounding boundary: it recomputes,
//! PURELY from the crate's own real forward dispatch (bf16 arm AND f32
//! arm — never a hand-rederived gelu formula), what `d_up` would be under
//! the CORRECT rounding placement and under the bug this test catches,
//! and requires a meaningful fraction of the 214 real rows to disagree
//! between the two. On this fixture, 57/214 (26.6%) disagree — this
//! oracle is not comparing two things that were always going to be equal
//! (the kernel guide's §3.7 "controls are non-vacuous" clause, and
//! `AGENTS.md`'s standing "non-vacuous negative control" clause).
//!
//! ## `d_gate` (the other half of `dwi_out`)
//!
//! [`bf16_bwd_d_gate_gets_the_atens_two_kernel_rounding_not_one`] closes
//! the other half of `geglu_bwd_row_bf16` (`ops/geglu.rs`'s `d_gate` line)
//! that the `d_up` test above does not touch — asserting only `d_up` from
//! a kernel that computes BOTH outputs in the same launch is a tautology
//! risk: a fix (or a regression) that touches only the `d_gate` line reads
//! GREEN on this file's `d_up`-only assertion regardless of what `d_gate`
//! does. Unlike `d_up`, this reference CANNOT be read off the crate's own
//! forward dispatch (the forward never materializes `gelu_erf'(gate)`), so
//! it is reimplemented independently here, straight from ATen's own two
//! kernels the module doc's "BF16 backward rounding" section cites:
//!
//! - `mul_tensor_backward` (`torch/csrc/autograd/FunctionsManual.cpp`):
//!   `grad * other`, same-dtype — `d_act = round(dy * up)`, a REAL
//!   bf16-rounded intermediate tensor, not kept in float across the two
//!   backward kernels.
//! - `GeluBackwardCUDAKernelImpl` (`ActivationGeluKernel.cu`, erf mode):
//!   `dy * (cdf(x) + x*pdf(x))` computed in float (`opmath_t`) with ONE
//!   cast on store, where this kernel's `dy` argument is the ALREADY
//!   bf16-rounded `d_act` from the step above — so
//!   `d_gate = round(round(dy*up) * (cdf(gate) + gate*pdf(gate)))`.
//!
//! `cdf`/`pdf` are recomputed here from `libm::erff` directly (the same
//! primitive `ops/geglu.rs`'s `GELU_ALPHA_F32`/`GELU_BETA_F32` wrap, per
//! its own doc comment citing ATen's `kAlpha`/`kBeta`), never by calling
//! the crate's private `gelu_erf_and_grad_f32` — that function is not
//! reachable from this integration test (it is `fn`, not `pub(crate)`),
//! and even if it were, reusing the implementation under test as its own
//! reference would make the assertion tautological, exactly the defect
//! this section exists to close.

use candle_core::{Device, Tensor, Var};
use half::bf16;
use jammi_kernels::ops::{apply1, GegluFused, GeluVariant};

fn fused(wi_out: &Tensor) -> candle_core::Result<Tensor> {
    apply1(wi_out, GegluFused::new(GeluVariant::Erf))
}

const FIXTURE: &[(f32, f32, f32)] = &[
    (27.5f32, 0.71875f32, 6.817281e-7f32),
    (5.75f32, -20.5f32, 5.7742e-8f32),
    (-15.4375f32, -11.0f32, 1.847744e-6f32),
    (-3.828125f32, -6.40625f32, -2.682209e-6f32),
    (2.1875f32, 1.0546875f32, -1.2479722e-7f32),
    (20.5f32, 2.1875f32, 1.2740493e-6f32),
    (22.625f32, 1.3515625f32, 2.413988e-6f32),
    (-0.66796875f32, -3.21875f32, 6.966293e-7f32),
    (7.65625f32, 10.375f32, 3.1664968e-8f32),
    (-1.6298145e-7f32, -0.265625f32, -1.1641532e-8f32),
    (6.519258e-7f32, 0.71875f32, -1.6018748e-7f32),
    (-9.834766e-7f32, 1.9296875f32, 2.1536835e-8f32),
    (-1.5795231e-6f32, -1.4140625f32, -1.0186341e-8f32),
    (-2.2500753e-6f32, 0.63671875f32, 3.5613775e-6f32),
    (-0.5390625f32, 0.69140625f32, -4.1909516e-7f32),
    (0.79296875f32, 0.765625f32, -8.789357e-9f32),
    (0.58203125f32, -0.15429688f32, 2.682209e-7f32),
    (-0.60546875f32, -0.106933594f32, 5.098991e-8f32),
    (0.0066223145f32, 0.00982666f32, -1.5366822e-8f32),
    (-0.23242188f32, 1.0390625f32, -2.750312e-9f32),
    (-0.95703125f32, -0.27929688f32, -5.8673322e-8f32),
    (-0.19921875f32, 0.51171875f32, -1.0768417e-8f32),
    (0.21484375f32, 1.203125f32, 3.0500814e-8f32),
    (0.890625f32, -0.5703125f32, 2.4400651e-7f32),
    (0.25390625f32, -2.578125f32, 5.378388e-8f32),
    (-0.83203125f32, -1.7421875f32, -2.3981556e-8f32),
    (-0.17773438f32, -1.0390625f32, -2.0489097e-7f32),
    (0.38085938f32, -0.8046875f32, -1.6516424e-9f32),
    (-1.0625f32, -0.18261719f32, 2.188608e-8f32),
    (0.6875f32, -0.21972656f32, -1.6763806e-8f32),
    (0.83984375f32, -0.0015487671f32, 9.895302e-9f32),
    (1.4140625f32, -0.17480469f32, 1.2591481e-6f32),
    (1.28125f32, -0.73046875f32, -4.5693014e-9f32),
    (-3.21875f32, -0.5078125f32, -2.3841858e-6f32),
    (0.012023926f32, -0.3984375f32, -5.9977174e-7f32),
    (1.3359375f32, 0.5859375f32, -1.6856939e-7f32),
    (1.1015625f32, -0.71875f32, -8.009374e-7f32),
    (0.8125f32, 0.3359375f32, -6.2864274e-8f32),
    (-0.19726563f32, 0.03881836f32, 3.4779077e-9f32),
    (0.3671875f32, -1.9921875f32, -3.5390258e-8f32),
    (0.65234375f32, -0.0234375f32, 1.542503e-9f32),
    (0.08251953f32, -0.46289063f32, 8.614734e-8f32),
    (-0.12890625f32, -1.3515625f32, 8.754432e-8f32),
    (0.83203125f32, -0.5078125f32, -2.3543835e-6f32),
    (0.36132813f32, -0.96484375f32, 5.5879354e-8f32),
    (0.71875f32, 1.3359375f32, 4.3958426e-7f32),
    (0.053466797f32, 0.52734375f32, -3.259629e-8f32),
    (-0.034423828f32, -0.58203125f32, -1.3783574e-7f32),
    (0.5078125f32, 0.7265625f32, 1.2572855e-7f32),
    (0.46484375f32, -0.86328125f32, 7.1013346e-9f32),
    (0.3984375f32, -0.3671875f32, -1.4144462e-8f32),
    (0.84375f32, 0.828125f32, 8.195639e-8f32),
    (0.51953125f32, 0.4921875f32, 1.46683306e-8f32),
    (-0.38085938f32, 0.34960938f32, 4.5867637e-8f32),
    (-1.125f32, 1.3046875f32, -2.1769665e-8f32),
    (-0.7109375f32, 0.75f32, -1.0011718e-8f32),
    (0.37304688f32, -0.81640625f32, 6.426126e-8f32),
    (0.048095703f32, -0.37109375f32, -1.1995435e-6f32),
    (0.14355469f32, -1.0859375f32, -6.511982e-10f32),
    (-0.56640625f32, -0.15527344f32, 2.4214387e-7f32),
    (0.049560547f32, 0.060791016f32, 7.4133277e-7f32),
    (0.59375f32, -0.14453125f32, -1.4603138e-6f32),
    (0.9375f32, -0.62109375f32, 2.575689e-9f32),
    (0.59765625f32, 1.4765625f32, 2.514571e-7f32),
    (0.7578125f32, 0.49023438f32, -1.2863893e-8f32),
    (0.98828125f32, 0.5078125f32, 8.731149e-9f32),
    (0.5078125f32, 0.92578125f32, -4.9127266e-8f32),
    (-0.30273438f32, -0.08642578f32, -1.5795231e-6f32),
    (0.11279297f32, 0.16503906f32, 1.4156103e-7f32),
    (0.05493164f32, -0.095703125f32, -1.4610123e-8f32),
    (-0.2265625f32, 0.26953125f32, 7.003546e-7f32),
    (0.14941406f32, -0.14746094f32, 6.6682696e-7f32),
    (0.09667969f32, -1.8671875f32, -1.967419e-8f32),
    (-0.46679688f32, -1.171875f32, -4.8894435e-8f32),
    (0.43945313f32, 0.60546875f32, 2.2351742e-7f32),
    (-0.41796875f32, -1.0546875f32, -4.02797e-8f32),
    (1.0546875f32, 0.98046875f32, -6.891787e-8f32),
    (0.022460938f32, 0.14453125f32, -2.0489097e-8f32),
    (-0.09326172f32, 0.359375f32, -2.0712614e-6f32),
    (1.015625f32, 0.115722656f32, 4.6857167e-9f32),
    (2.03125f32, -1.25f32, 2.1245796e-9f32),
    (2.3125f32, -0.43554688f32, 2.104789e-7f32),
    (0.04321289f32, 0.3046875f32, 9.3504786e-7f32),
    (-0.546875f32, 0.69921875f32, -1.1117663e-8f32),
    (-0.390625f32, 0.78515625f32, 2.1536835e-8f32),
    (-0.60546875f32, 0.041503906f32, -3.799796e-7f32),
    (-0.6875f32, 1.65625f32, -1.7462298e-8f32),
    (1.1015625f32, -0.953125f32, 1.0523945e-7f32),
    (0.59765625f32, -0.5703125f32, 7.376075e-7f32),
    (0.65234375f32, -0.78125f32, -1.0652002e-8f32),
    (0.5625f32, -0.47265625f32, -2.2555469e-9f32),
    (-0.10449219f32, 0.16601563f32, 2.5844201e-8f32),
    (-0.34179688f32, -0.51171875f32, 2.6309863e-8f32),
    (-0.328125f32, 0.7265625f32, -5.5646524e-8f32),
    (-1.4140625f32, -0.067871094f32, 9.685755e-7f32),
    (-0.030517578f32, 1.125f32, 8.940697e-7f32),
    (0.47460938f32, -0.71484375f32, -9.5926225e-8f32),
    (-0.072753906f32, -0.05493164f32, -5.180482e-9f32),
    (1.6796875f32, 0.50390625f32, -3.0966476e-8f32),
    (0.3515625f32, -1.234375f32, 3.4924597e-9f32),
    (-1.3671875f32, 0.4375f32, -5.5134296e-7f32),
    (-0.65234375f32, -0.5625f32, -8.789357e-9f32),
    (0.42773438f32, 0.609375f32, 5.448237e-8f32),
    (-0.5625f32, 0.87890625f32, 2.4959445e-7f32),
    (0.12792969f32, 0.23828125f32, 3.0413503e-9f32),
    (-1.53125f32, -0.9453125f32, -1.7811544e-8f32),
    (0.25195313f32, -0.83203125f32, 6.845221e-8f32),
    (-0.076171875f32, 0.15234375f32, -1.0803342e-6f32),
    (0.66796875f32, 1.59375f32, 1.1059456e-8f32),
    (-0.109375f32, 0.69140625f32, -6.984919e-8f32),
    (0.22265625f32, -0.46679688f32, 1.2591481e-6f32),
    (-1.0703125f32, 1.234375f32, -1.385808e-6f32),
    (0.17480469f32, 0.39453125f32, 1.36788e-9f32),
    (-0.90234375f32, 1.9921875f32, -1.7881393e-7f32),
    (0.30273438f32, 0.95703125f32, 7.171184e-8f32),
    (0.34960938f32, 1.203125f32, -1.3876706e-7f32),
    (-0.703125f32, 0.32226563f32, -1.21071935e-8f32),
    (-0.05493164f32, 0.11767578f32, 1.616776e-6f32),
    (0.7421875f32, -0.8984375f32, 4.7730282e-8f32),
    (0.63671875f32, 0.81640625f32, 1.3038516e-8f32),
    (-0.25195313f32, 0.94140625f32, 5.098991e-8f32),
    (-0.09375f32, -0.60546875f32, -2.2798777e-6f32),
    (0.06933594f32, -0.018798828f32, -8.707866e-8f32),
    (0.12207031f32, 0.036621094f32, -1.0244548e-7f32),
    (0.5625f32, -1.609375f32, -8.265488e-9f32),
    (0.22949219f32, -1.0859375f32, -9.720679e-9f32),
    (1.5703125f32, -0.34960938f32, -4.0927262e-10f32),
    (0.7265625f32, 0.061279297f32, 2.644956e-7f32),
    (0.8671875f32, -0.19140625f32, -1.9324943e-8f32),
    (0.70703125f32, -0.25585938f32, -5.7742e-7f32),
    (1.0703125f32, -1.2265625f32, 1.5599653e-8f32),
    (1.015625f32, 0.06689453f32, 8.791685e-7f32),
    (-0.24316406f32, -2.421875f32, 2.5890768e-7f32),
    (0.44335938f32, 0.234375f32, -2.7823262e-8f32),
    (-0.30664063f32, 0.6796875f32, 8.335337e-8f32),
    (-0.140625f32, -0.033447266f32, 5.9138983e-8f32),
    (0.8203125f32, -0.2265625f32, 1.1399388e-6f32),
    (0.70703125f32, 0.546875f32, 1.0803342e-6f32),
    (-0.1328125f32, 0.87890625f32, 8.195639e-8f32),
    (0.51953125f32, -0.6640625f32, 6.984919e-8f32),
    (0.22167969f32, 0.44726563f32, -1.6391277e-6f32),
    (-0.13183594f32, -0.63671875f32, 3.5576522e-7f32),
    (-0.41015625f32, 0.9296875f32, 8.403731e-10f32),
    (0.29296875f32, 0.41796875f32, -2.9685907e-9f32),
    (-0.49609375f32, 0.49414063f32, 3.4458935e-7f32),
    (0.30078125f32, -0.76171875f32, 5.711627e-10f32),
    (1.734375f32, 0.22460938f32, 1.00499165e-10f32),
    (-0.17675781f32, 0.890625f32, -1.0244548e-7f32),
    (-0.23730469f32, -1.09375f32, -1.5133992e-8f32),
    (0.78125f32, -0.26953125f32, 1.322478e-7f32),
    (-1.8359375f32, -0.89453125f32, 1.5720725e-6f32),
    (-0.98828125f32, -0.5078125f32, 5.378388e-8f32),
    (-0.20605469f32, 0.4296875f32, 6.7055225e-7f32),
    (-0.43554688f32, -1.578125f32, -1.2922101e-8f32),
    (0.73046875f32, -0.85546875f32, 3.7020072e-8f32),
    (-0.012878418f32, 0.828125f32, -6.891787e-8f32),
    (0.62890625f32, 0.47460938f32, 2.7794158e-9f32),
    (0.40820313f32, 0.015563965f32, -3.9115548e-8f32),
    (-0.29882813f32, 0.890625f32, 6.7055225e-7f32),
    (-0.08984375f32, -0.19140625f32, 2.5331974e-7f32),
    (1.234375f32, -0.73046875f32, 6.519258e-8f32),
    (-0.71875f32, 0.8984375f32, 1.2223609e-8f32),
    (1.28125f32, 0.46875f32, -2.1187589e-8f32),
    (-0.28515625f32, -0.7734375f32, 5.5879354e-7f32),
    (-0.6328125f32, 0.54296875f32, 2.9453076e-8f32),
    (-0.63671875f32, -0.890625f32, -3.1106174e-7f32),
    (0.82421875f32, -0.26953125f32, 1.7043203e-7f32),
    (1.0625f32, 0.03173828f32, -1.3387762e-8f32),
    (0.76953125f32, 0.26367188f32, 4.4703484e-8f32),
    (-0.76953125f32, -1.109375f32, 4.377216e-7f32),
    (-0.28515625f32, -0.1796875f32, 3.958121e-8f32),
    (0.8046875f32, 1.0078125f32, 5.296897e-9f32),
    (-0.32226563f32, -1.5234375f32, -1.8626451e-6f32),
    (0.40429688f32, -0.203125f32, 1.0803342e-6f32),
    (0.98828125f32, -1.453125f32, -1.1292286e-8f32),
    (0.16894531f32, 1.5859375f32, -4.1443855e-8f32),
    (0.765625f32, 0.76171875f32, -6.0070306e-8f32),
    (0.09277344f32, -0.15332031f32, -3.678724e-8f32),
    (-0.1796875f32, 0.421875f32, 7.748604e-7f32),
    (0.55078125f32, -0.027709961f32, -3.632158e-8f32),
    (0.72265625f32, 0.6171875f32, 4.307367e-8f32),
    (0.26757813f32, -1.0859375f32, -3.7718564e-8f32),
    (-0.359375f32, -0.57421875f32, -1.3969839e-7f32),
    (0.58203125f32, 0.765625f32, 7.0314854e-8f32),
    (-1.171875f32, 0.8984375f32, 7.8976154e-7f32),
    (-0.07763672f32, 0.640625f32, 1.1250377e-6f32),
    (0.21191406f32, -0.38867188f32, 6.072223e-7f32),
    (0.98046875f32, -1.5390625f32, 1.0617077e-7f32),
    (-0.6953125f32, 0.5234375f32, 1.2456439e-8f32),
    (-0.59375f32, -1.1171875f32, -3.562309e-8f32),
    (0.01373291f32, -0.34960938f32, -2.5890768e-7f32),
    (-0.67578125f32, 0.36914063f32, 3.632158e-8f32),
    (1.1484375f32, -0.33789063f32, 1.4621764e-7f32),
    (-0.32617188f32, 0.6640625f32, -3.306195e-8f32),
    (0.06738281f32, -0.07421875f32, -1.3038516e-7f32),
    (1.1796875f32, 0.3515625f32, 6.170012e-9f32),
    (-0.08984375f32, -0.125f32, -5.848706e-7f32),
    (0.609375f32, -0.42578125f32, -2.4680048e-8f32),
    (0.32617188f32, 1.21875f32, 1.322478e-7f32),
    (0.064453125f32, -0.15722656f32, 1.15484e-7f32),
    (-0.36328125f32, 0.0065307617f32, -2.130866e-6f32),
    (0.6953125f32, -0.16894531f32, -1.4975667e-6f32),
    (0.41992188f32, 0.875f32, -2.4735928e-6f32),
    (-0.36523438f32, 0.24609375f32, 2.1769665e-8f32),
    (0.56640625f32, -0.3125f32, -6.4319465e-9f32),
    (-0.33007813f32, -0.37695313f32, -1.1466909e-8f32),
    (1.9609375f32, -1.328125f32, -2.142042e-8f32),
    (-0.96875f32, 1.3828125f32, 2.1792948e-7f32),
    (-0.032714844f32, -1.046875f32, -2.9057264e-7f32),
    (0.83984375f32, 1.4296875f32, -2.7567148e-7f32),
    (-0.58984375f32, -1.3515625f32, 6.3795596e-8f32),
    (-2.609375f32, 1.1875f32, 2.6226044e-6f32),
    (-0.29492188f32, -0.0074157715f32, -1.169974e-8f32),
    (0.37109375f32, -1.015625f32, 1.1399388e-6f32),
];

/// Non-vacuous floor for the fixture-discrimination control — measured at
/// 57/214 today; `>= 40` leaves headroom for a different libm/erf build
/// while still refusing a fixture that has degenerated to "always
/// agrees".
const MIN_DISCRIMINATING_ROWS: usize = 40;

/// One captured mismatch: `(row index, the FIXTURE triple, kernel's real
/// `d_up`, the torch-consistent reference, the pre-fix buggy formula)`.
type Mismatch = (usize, (f32, f32, f32), bf16, bf16, bf16);

#[test]
fn bf16_bwd_d_up_differentiates_the_forwards_own_rounded_activation_not_an_unrounded_one() {
    let device = Device::Cpu;
    let rows = FIXTURE.len();

    // Ground truth #1: the FORWARD's own bf16-rounded activation for each
    // `gate`, read back by pinning `up` to EXACTLY `bf16::from_f32(1.0)`
    // — bf16's `Mul` by unit is idempotent, so `out = round2(act_bf16 *
    // 1.0) == act_bf16` exactly. This is the REAL production forward's
    // ROUND-1 artifact, read through the real `CustomOp1::cpu_fwd` entry
    // point — never a re-derived gelu formula.
    let mut wi_probe_bf16 = Vec::with_capacity(rows * 2);
    for &(g, _, _) in FIXTURE {
        wi_probe_bf16.push(bf16::from_f32(g));
        wi_probe_bf16.push(bf16::ONE);
    }
    let act_bf16_actual: Vec<bf16> =
        fused(&Tensor::from_slice(&wi_probe_bf16, (rows, 2), &device).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

    // Ground truth #2: the F32 forward's EXACT (unrounded) activation —
    // the F32 arm has no intermediate rounding point at all (one dtype
    // throughout, see `ops/geglu.rs`'s module doc), so this is the value
    // the PRE-fix bf16 backward effectively used for `d_up`.
    let mut wi_probe_f32 = Vec::with_capacity(rows * 2);
    for &(g, _, _) in FIXTURE {
        wi_probe_f32.push(g);
        wi_probe_f32.push(1.0f32);
    }
    let act_f32_unrounded: Vec<f32> =
        fused(&Tensor::from_slice(&wi_probe_f32, (rows, 2), &device).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

    // The REAL bwd dispatch — `Var` + `backward()`, through `GegluFused`'s
    // own `CustomOp1::bwd` (candle's real autodiff entry point, the same
    // one every production call site drives) — on the fixture's real
    // `(gate, up)` pairs, seeded by the fixture's REAL, non-uniform `dy`
    // (never `backward()`'s default `ones_like` seed — a uniform seed
    // makes `d_up` trivially proportional to a constant and would mask
    // exactly this class of bug, per `geglu_oracles.rs`'s own note).
    let mut wi = Vec::with_capacity(rows * 2);
    let mut dy = Vec::with_capacity(rows);
    for &(g, u, d) in FIXTURE {
        wi.push(bf16::from_f32(g));
        wi.push(bf16::from_f32(u));
        dy.push(bf16::from_f32(d));
    }
    let wi_var = Var::from_tensor(&Tensor::from_slice(&wi, (rows, 2), &device).unwrap()).unwrap();
    let out = fused(wi_var.as_tensor()).unwrap();
    let dy_t = Tensor::from_slice(&dy, (rows, 1), &device).unwrap();
    let loss = (&out * &dy_t).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let dwi: Vec<bf16> = grads
        .get(wi_var.as_tensor())
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let mut discriminating_rows = 0usize;
    let mut mismatches: Vec<Mismatch> = Vec::new();
    for i in 0..rows {
        let (_, _, d) = FIXTURE[i];
        let dy_bf16 = bf16::from_f32(d);
        let d_up_correct = bf16::from_f32(dy_bf16.to_f32() * act_bf16_actual[i].to_f32());
        let d_up_old_buggy = bf16::from_f32(dy_bf16.to_f32() * act_f32_unrounded[i]);
        if d_up_correct != d_up_old_buggy {
            discriminating_rows += 1;
        }
        let d_up_kernel = dwi[i * 2 + 1];
        // Affirmative, never `!(x > bound)` (guide §3.7) — a NaN/Inf must
        // fail this assertion outright, not read as "not disproven".
        assert!(
            d_up_kernel.to_f32().is_finite() && d_up_correct.to_f32().is_finite(),
            "row {i}: a non-finite value slipped through (kernel={d_up_kernel:?}, \
             reference={d_up_correct:?}) — FIXTURE={:?}",
            FIXTURE[i]
        );
        if d_up_kernel != d_up_correct {
            mismatches.push((i, FIXTURE[i], d_up_kernel, d_up_correct, d_up_old_buggy));
        }
    }

    assert!(
        discriminating_rows >= MIN_DISCRIMINATING_ROWS,
        "fixture is not discriminating: only {discriminating_rows}/{rows} rows separate the \
         torch-consistent formula from the pre-fix buggy one — this fixture would read GREEN \
         on a broken build regardless of the kernel (kernel guide §3.4/§3.7); strengthen it \
         before trusting this oracle"
    );

    assert!(
        mismatches.is_empty(),
        "GegluFused's bf16 bwd `d_up` does NOT differentiate the FORWARD's own rounded \
         activation on {}/{rows} real ModernBERT-large layer-18 rows (esc-045/GH#374). First \
         mismatch: idx={} (gate,up,dy)={:?} kernel_d_up={:?} torch-consistent d_up={:?} \
         pre-fix-buggy d_up={:?}. Reverting `ops/geglu.rs`'s `geglu_bwd_row_bf16` to `dy * \
         gelu_val` (the unrounded activation) reproduces every one of these mismatches — this \
         is the exact regression this test exists to catch; see `ops/geglu.rs`'s \"esc-045 \
         fix\" doc block.",
        mismatches.len(),
        mismatches[0].0,
        mismatches[0].1,
        mismatches[0].2,
        mismatches[0].3,
        mismatches[0].4,
    );
}

/// `1/sqrt(2)` — ATen's `kAlpha`, reimplemented independently of
/// `ops/geglu.rs`'s private `GELU_ALPHA_F32` (see the module doc's
/// "`d_gate`" section for why this must NOT call the crate's own helper).
const GELU_ALPHA_F32_REF: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// `1/sqrt(2*pi)` — ATen's `kBeta`, independently reimplemented (see above).
const GELU_BETA_F32_REF: f32 =
    std::f32::consts::FRAC_2_SQRT_PI * std::f32::consts::FRAC_1_SQRT_2 * 0.5;

/// `gelu_erf'(x) = Phi(x) + x*phi(x)` — ATen's `ActivationGeluKernel.cu`
/// erf-mode `gelu_backward`'s `cdf + x*pdf`, computed straight from
/// `libm::erff` (never through `jammi_kernels::ops::geglu`'s private
/// `gelu_erf_and_grad_f32`, which this integration test cannot even name).
fn gelu_erf_deriv_f32_reference(x: f32) -> f32 {
    let cdf = (libm::erff(x * GELU_ALPHA_F32_REF) + 1.0) * 0.5;
    let pdf = GELU_BETA_F32_REF * (-0.5 * x * x).exp();
    cdf + x * pdf
}

/// Non-vacuous floor for `d_gate`'s fixture-discrimination control —
/// measured at 64/214 today; `>= 40` mirrors [`MIN_DISCRIMINATING_ROWS`]'s
/// rationale (headroom for a different libm/erf build without degenerating
/// to "always agrees").
const MIN_DISCRIMINATING_ROWS_D_GATE: usize = 40;

/// One captured `d_gate` mismatch: `(row index, the FIXTURE triple,
/// kernel's real `d_gate`, the ATen-two-kernel-rounding reference, the
/// pre-fix single-rounding buggy value)`.
type MismatchDGate = (usize, (f32, f32, f32), bf16, bf16, bf16);

/// Closes the `d_gate` half `bf16_bwd_d_up_differentiates_...` above does
/// not touch (see the module doc's "`d_gate`" section) — `GegluFused`'s
/// bf16 backward must round `dy*up` to bf16 BEFORE multiplying by
/// `gelu_erf'(gate)`, matching `mul_tensor_backward`'s real bf16-rounded
/// output feeding `GeluBackwardCUDAKernelImpl` as ITS `dy` argument — never
/// a single-rounding `dy * up * gelu_erf'(gate)` computed and rounded once.
#[test]
fn bf16_bwd_d_gate_gets_the_atens_two_kernel_rounding_not_one() {
    let device = Device::Cpu;
    let rows = FIXTURE.len();

    // The REAL bwd dispatch — same construction as the `d_up` test above:
    // `Var` + `backward()` through `GegluFused`'s own `CustomOp1::bwd`, on
    // the fixture's real `(gate, up)` pairs seeded by the fixture's real,
    // non-uniform `dy`.
    let mut wi = Vec::with_capacity(rows * 2);
    let mut dy = Vec::with_capacity(rows);
    for &(g, u, d) in FIXTURE {
        wi.push(bf16::from_f32(g));
        wi.push(bf16::from_f32(u));
        dy.push(bf16::from_f32(d));
    }
    let wi_var = Var::from_tensor(&Tensor::from_slice(&wi, (rows, 2), &device).unwrap()).unwrap();
    let out = fused(wi_var.as_tensor()).unwrap();
    let dy_t = Tensor::from_slice(&dy, (rows, 1), &device).unwrap();
    let loss = (&out * &dy_t).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let dwi: Vec<bf16> = grads
        .get(wi_var.as_tensor())
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let mut discriminating_rows = 0usize;
    let mut mismatches: Vec<MismatchDGate> = Vec::new();
    for (i, &(g, u, d)) in FIXTURE.iter().enumerate() {
        let gate_bf16 = bf16::from_f32(g);
        let up_bf16 = bf16::from_f32(u);
        let dy_bf16 = bf16::from_f32(d);
        let deriv = gelu_erf_deriv_f32_reference(gate_bf16.to_f32());

        // ATen-consistent: `d_act` rounds to bf16 (a real intermediate
        // tensor `mul`'s backward materializes) BEFORE multiplying by the
        // gelu derivative.
        let d_act_correct_bf16 = bf16::from_f32(dy_bf16.to_f32() * up_bf16.to_f32());
        let d_gate_correct = bf16::from_f32(d_act_correct_bf16.to_f32() * deriv);

        // Pre-fix buggy: single f32 accumulation, rounded once at the end —
        // this crate's OLD (esc-045-round-2-buggy) formula, reproduced
        // exactly by reverting `ops/geglu.rs`'s `geglu_bwd_row_bf16`
        // `d_gate` line to `dy * up * gelu_deriv`.
        let d_gate_old_buggy = bf16::from_f32(dy_bf16.to_f32() * up_bf16.to_f32() * deriv);

        if d_gate_correct != d_gate_old_buggy {
            discriminating_rows += 1;
        }

        let d_gate_kernel = dwi[i * 2];
        // Affirmative, never `!(x > bound)` (guide §3.7) — a NaN/Inf must
        // fail this assertion outright, not read as "not disproven".
        assert!(
            d_gate_kernel.to_f32().is_finite() && d_gate_correct.to_f32().is_finite(),
            "row {i}: a non-finite value slipped through (kernel={d_gate_kernel:?}, \
             reference={d_gate_correct:?}) — FIXTURE={:?}",
            FIXTURE[i]
        );
        if d_gate_kernel != d_gate_correct {
            mismatches.push((
                i,
                FIXTURE[i],
                d_gate_kernel,
                d_gate_correct,
                d_gate_old_buggy,
            ));
        }
    }

    assert!(
        discriminating_rows >= MIN_DISCRIMINATING_ROWS_D_GATE,
        "fixture is not discriminating for d_gate: only {discriminating_rows}/{rows} rows \
         separate the ATen-two-kernel-rounding formula from the pre-fix single-rounding one — \
         this fixture would read GREEN on a broken build regardless of the kernel (kernel guide \
         §3.4/§3.7); strengthen it before trusting this oracle"
    );

    assert!(
        mismatches.is_empty(),
        "GegluFused's bf16 bwd `d_gate` does NOT match ATen's two-kernel rounding \
         (`mul_tensor_backward` then `GeluBackwardCUDAKernelImpl`) on {}/{rows} real \
         ModernBERT-large layer-18 rows (esc-045/GH#374). First mismatch: idx={} \
         (gate,up,dy)={:?} kernel_d_gate={:?} ATen-consistent d_gate={:?} \
         pre-fix-single-rounding d_gate={:?}. Reverting `ops/geglu.rs`'s `geglu_bwd_row_bf16`'s \
         `d_gate` line to `bf16::from_f32(dyi * up * gelu_deriv)` (single rounding) reproduces \
         this class of mismatch — this is the exact regression this test exists to catch; see \
         `ops/geglu.rs`'s \"esc-045 fix\" doc block.",
        mismatches.len(),
        mismatches[0].0,
        mismatches[0].1,
        mismatches[0].2,
        mismatches[0].3,
        mismatches[0].4,
    );
}
