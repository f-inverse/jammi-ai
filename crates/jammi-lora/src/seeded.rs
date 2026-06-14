//! Seeded, candle-RNG-free initialisation and dropout for the LoRA path.
//!
//! candle's CPU `rand_uniform` / `randn` draw from the process-global
//! `rand::rng()` (its `set_seed` is a no-op on CPU), so `Init::Kaiming` /
//! `Init::Randn` and `candle_nn::ops::dropout` are *unseedable* — two runs of
//! the same fine-tune produce different adapters. This module owns the draws
//! instead: a small self-contained SplitMix64 PRNG fills host buffers that are
//! then registered as trainable `Var`s, and a per-layer dropout stream draws a
//! Bernoulli mask. Nothing here touches a global RNG, so a fine-tune is a pure
//! function of `(seed, source rows, config)`.
//!
//! **Cross-process determinism.** Every draw stream is keyed by
//! `(seed, fully-qualified parameter name)` via [`seed_for_param`], never by
//! `VarMap`/`HashMap` iteration order. So which order the layers happen to be
//! constructed or iterated in is irrelevant: the `projection.lora_a` tensor is
//! byte-identical regardless of how many other layers exist or when they were
//! built. This is the same FNV-1a-then-SplitMix idiom the engine already uses
//! for its seeded graph walks.

/// A small, fast, self-contained PRNG (SplitMix64) so seeded init and dropout
/// reproduce byte-identically from a seed without pulling a `rand` dependency
/// into the LoRA primitive. Identical algorithm to the engine's graph-walk
/// PRNG, duplicated here only to keep `jammi-lora` dependency-free.
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform `f32` in `[0, 1)` from the top 24 mantissa bits (exact).
    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32)
    }

    /// A standard-normal `f32` via Box–Muller (the cosine variate). One
    /// transcendental per draw; fine for the small LoRA matrices and keeps the
    /// stream advance count = one `next_u64` pair per value, deterministic.
    pub(crate) fn next_standard_normal(&mut self) -> f32 {
        // u1 in (0, 1] to avoid ln(0); u2 in [0, 1).
        let u1 = 1.0 - self.next_f32();
        let u2 = self.next_f32();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        r * (std::f32::consts::TAU * u2).cos()
    }
}

/// Derive a deterministic per-parameter draw seed from the base run seed and the
/// fully-qualified parameter name (e.g. `"projection.lora_a"`). FNV-1a over the
/// name bytes mixed with the base seed and run through one SplitMix64 round, so
/// the stream is stable across processes and independent of construction /
/// `VarMap` iteration order. Never drawn from entropy.
pub(crate) fn seed_for_param(base_seed: u64, name: &str) -> u64 {
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    SplitMix64::new(hash ^ base_seed).next_u64()
}

/// Fill a Kaiming-uniform host buffer of length `rows * cols` for a weight with
/// `fan_in` inputs. Matches candle's `Init::Kaiming { Uniform, FanIn, Linear }`
/// distribution: `U(-bound, bound)` with `bound = sqrt(3 / fan_in)` (gain 1 for
/// the linear non-linearity). The draw is from the seeded `rng`.
pub(crate) fn kaiming_uniform_fill(rng: &mut SplitMix64, len: usize, fan_in: usize) -> Vec<f32> {
    let bound = (3.0_f32 / fan_in as f32).sqrt();
    (0..len)
        .map(|_| {
            // map [0,1) -> [-bound, bound)
            (rng.next_f32() * 2.0 - 1.0) * bound
        })
        .collect()
}

/// Fill a `Normal(0, stdev)` host buffer of length `len` from the seeded `rng`.
/// Matches candle's `Init::Randn { mean: 0, stdev }`.
pub(crate) fn gaussian_fill(rng: &mut SplitMix64, len: usize, stdev: f32) -> Vec<f32> {
    (0..len)
        .map(|_| rng.next_standard_normal() * stdev)
        .collect()
}

/// A run-owned, advancing dropout stream. Seeded per LoRA layer from
/// `(seed, "{param_prefix}.dropout")`, it draws an inverted-dropout Bernoulli
/// mask each time the LoRA path runs in training mode. Because forwards happen
/// in a deterministic order (ordered rows → fixed batching), an advancing
/// per-layer stream is equivalent to keying the mask by `(step, microbatch)`:
/// the k-th training forward through this layer always consumes the k-th block
/// of draws. Validation forwards skip dropout (`set_training(false)`), so they
/// never perturb the stream.
pub(crate) struct DropoutStream {
    rng: SplitMix64,
}

impl DropoutStream {
    pub(crate) fn new(seed: u64, layer_name: &str) -> Self {
        Self {
            rng: SplitMix64::new(seed_for_param(seed, &format!("{layer_name}.dropout"))),
        }
    }

    /// Draw an inverted-dropout mask of length `len`: each element is `0.0` with
    /// probability `p`, else `1/(1-p)` so the expected value is preserved (same
    /// scaling candle's `dropout` applies). Advances the stream by `len` draws.
    pub(crate) fn draw_mask(&mut self, len: usize, p: f32) -> Vec<f32> {
        let keep = 1.0 - p;
        let scale = 1.0 / keep;
        (0..len)
            .map(|_| if self.rng.next_f32() < p { 0.0 } else { scale })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_for_param_is_name_keyed_not_order_keyed() {
        // Same (seed, name) -> same stream seed, regardless of call order.
        let a = seed_for_param(42, "projection.lora_a");
        let b = seed_for_param(42, "projection.lora_a");
        assert_eq!(a, b);
        // Different names diverge; different base seeds diverge.
        assert_ne!(
            seed_for_param(42, "projection.lora_a"),
            seed_for_param(42, "projection.lora_b")
        );
        assert_ne!(
            seed_for_param(42, "projection.lora_a"),
            seed_for_param(43, "projection.lora_a")
        );
    }

    #[test]
    fn kaiming_uniform_respects_bound_and_is_deterministic() {
        let bound = (3.0_f32 / 8.0).sqrt();
        let mut r1 = SplitMix64::new(seed_for_param(7, "x"));
        let mut r2 = SplitMix64::new(seed_for_param(7, "x"));
        let v1 = kaiming_uniform_fill(&mut r1, 1000, 8);
        let v2 = kaiming_uniform_fill(&mut r2, 1000, 8);
        assert_eq!(v1, v2, "same seed -> byte-identical fill");
        for x in &v1 {
            assert!(x.abs() <= bound + 1e-6, "{x} exceeds kaiming bound {bound}");
        }
    }

    #[test]
    fn gaussian_fill_is_deterministic_and_centred() {
        let mut r1 = SplitMix64::new(seed_for_param(9, "g"));
        let mut r2 = SplitMix64::new(seed_for_param(9, "g"));
        let v1 = gaussian_fill(&mut r1, 10_000, 0.02);
        let v2 = gaussian_fill(&mut r2, 10_000, 0.02);
        assert_eq!(v1, v2);
        let mean: f32 = v1.iter().sum::<f32>() / v1.len() as f32;
        assert!(mean.abs() < 0.005, "mean {mean} not near 0");
    }

    #[test]
    fn dropout_stream_is_deterministic_and_scaled() {
        let mut s1 = DropoutStream::new(11, "projection");
        let mut s2 = DropoutStream::new(11, "projection");
        let m1 = s1.draw_mask(10_000, 0.3);
        let m2 = s2.draw_mask(10_000, 0.3);
        assert_eq!(m1, m2);
        let scale = 1.0 / 0.7;
        for x in &m1 {
            assert!(*x == 0.0 || (*x - scale).abs() < 1e-6);
        }
        // Roughly p of the mask is dropped.
        let dropped = m1.iter().filter(|x| **x == 0.0).count() as f32 / m1.len() as f32;
        assert!((dropped - 0.3).abs() < 0.03, "dropped fraction {dropped}");
    }
}
