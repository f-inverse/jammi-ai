//! Seeded, candle-RNG-free initialisation and dropout for the LoRA path.
//!
//! candle's CPU `rand_uniform` / `randn` draw from the process-global
//! `rand::rng()` (its `set_seed` is a no-op on CPU), so `Init::Kaiming` /
//! `Init::Randn` and `candle_nn::ops::dropout` are *unseedable* — two runs of
//! the same fine-tune produce different adapters. This module owns the draws
//! instead: a small self-contained SplitMix64 PRNG fills host buffers that are
//! then registered as trainable `Var`s, and a per-layer counter-keyed source
//! ([`DropoutMasks`]) draws each Bernoulli mask on the device that will consume
//! it. Nothing here touches an unseeded global RNG, so a fine-tune is a pure
//! function of `(seed, source rows, config)` on a given device class.
//!
//! **Cross-process determinism.** Every draw stream is keyed by
//! `(seed, fully-qualified parameter name)` via [`seed_for_param`], never by
//! `VarMap`/`HashMap` iteration order. So which order the layers happen to be
//! constructed or iterated in is irrelevant: the `projection.lora_a` tensor is
//! byte-identical regardless of how many other layers exist or when they were
//! built. This is the same FNV-1a-then-SplitMix idiom the engine already uses
//! for its seeded graph walks.

use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{Device, Tensor};

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

/// A LoRA layer's dropout mask source: a *counter-keyed* draw, not an advancing
/// stream.
///
/// The mask for a layer's k-th training forward is a pure function of
/// `(run seed, fully-qualified layer name, k)`. Two properties follow, and both
/// were absent from the advancing-stream design this replaces:
///
/// * **Restore is O(1).** The old stream had no closed-form skip, so restoring a
///   persisted position replayed that many draws one at a time from the origin.
///   Its comment justified the cost on the grounds that "the LoRA matrices are
///   small" — true of the *init* draws it was modelled on, but the dropout
///   position advances by one draw per *activation element*, which is
///   `batch * seq * in_features` per forward. After a single epoch of a large
///   encoder that is on the order of 1e11 draws per layer, so resume did not
///   merely slow down, it failed to finish. Restoring a counter is an
///   assignment.
///
/// * **The mask is built on the device that will consume it.** The old stream
///   generated a host `Vec<f32>` the size of the whole activation and copied it
///   across per LoRA site per forward, so a GPU run paid a single-threaded host
///   RNG and a PCIe transfer for every layer of every forward, and pinned each
///   mask in device memory until backward. Measured on an A100 with
///   ModernBERT-large at batch 8 / seq 128, removing that cost took the step
///   from 2.07 s to 0.72 s and freed 16.7 GB.
///
/// Determinism is per device class, which is the contract the surrounding code
/// already lives under (`jammi-numerics` documents single-architecture
/// determinism as the guarantee and cross-architecture reproducibility as an
/// explicit non-goal). A CPU run is byte-reproducible against a CPU run, a CUDA
/// run against a CUDA run; the two draw different masks from the same seed, as
/// they do in every mainstream framework.
pub(crate) struct DropoutMasks {
    /// `(seed, "{layer_name}.dropout")` — the layer's key, from which every
    /// mask is derived together with the forward counter.
    origin_seed: u64,
    /// Training forwards taken through this layer. The mask key, and the whole
    /// resume state.
    counter: AtomicU64,
}

impl DropoutMasks {
    pub(crate) fn new(seed: u64, layer_name: &str) -> Self {
        Self {
            origin_seed: seed_for_param(seed, &format!("{layer_name}.dropout")),
            counter: AtomicU64::new(0),
        }
    }

    /// Forwards taken so far — the unit of resume state.
    pub(crate) fn position(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }

    /// Restore the forward counter. O(1): the mask is a function of the counter,
    /// so there is no stream to replay.
    pub(crate) fn restore_position(&self, position: u64) {
        self.counter.store(position, Ordering::Relaxed);
    }

    /// Build the next inverted-dropout mask for `x`, on `x`'s own device, and
    /// advance the counter.
    ///
    /// Inverted dropout: a kept element is scaled by `1/(1-p)` so the expected
    /// value is preserved, matching what candle's own `dropout` applies.
    pub(crate) fn next_mask(&self, x: &Tensor, p: f32) -> Result<Tensor, candle_core::Error> {
        let k = self.counter.fetch_add(1, Ordering::Relaxed);
        let key = mix_counter(self.origin_seed, k);
        let scale = 1.0 / (1.0 - p) as f64;

        match x.device() {
            Device::Cpu => {
                // candle's CPU RNG cannot be seeded (`set_seed` errors), which is
                // why this crate carries its own generator at all. Building the
                // buffer host-side is free here: it is already the device the
                // tensor lives on, so there is no transfer.
                let mut rng = SplitMix64::new(key);
                let vals: Vec<f32> = (0..x.elem_count())
                    .map(|_| {
                        if rng.next_f32() < p {
                            0.0
                        } else {
                            scale as f32
                        }
                    })
                    .collect();
                Tensor::from_vec(vals, x.shape(), x.device())?.to_dtype(x.dtype())
            }
            device => {
                // Seed the device generator from the same key, then draw on the
                // device. `rand_uniform` is F32/F64 only, so the comparison runs
                // in F32 and the result casts to the activation's dtype.
                device.set_seed(key)?;
                let u = Tensor::rand(0f32, 1f32, x.shape(), device)?;
                let keep = u.ge(p)?.to_dtype(candle_core::DType::F32)?;
                (keep * scale)?.to_dtype(x.dtype())
            }
        }
    }
}

/// Fold a forward counter into a layer's origin seed.
///
/// SplitMix64's finalizer over `origin ^ (counter * golden-ratio odd constant)`:
/// consecutive counters land far apart, so successive forwards of one layer are
/// uncorrelated, and two layers never collide because their origins already
/// differ by [`seed_for_param`].
fn mix_counter(origin_seed: u64, counter: u64) -> u64 {
    let mut z = origin_seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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

    fn mask_values(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn masks_are_deterministic_and_inverted_scaled() {
        let d = Device::Cpu;
        let x = Tensor::zeros((100, 100), candle_core::DType::F32, &d).unwrap();
        let a = DropoutMasks::new(11, "projection");
        let b = DropoutMasks::new(11, "projection");
        let m1 = mask_values(&a.next_mask(&x, 0.3).unwrap());
        let m2 = mask_values(&b.next_mask(&x, 0.3).unwrap());
        assert_eq!(
            m1, m2,
            "same (seed, layer, counter) must give the same mask"
        );

        let scale = 1.0f32 / 0.7;
        for v in &m1 {
            assert!(
                *v == 0.0 || (*v - scale).abs() < 1e-6,
                "unexpected value {v}"
            );
        }
        let dropped = m1.iter().filter(|v| **v == 0.0).count() as f32 / m1.len() as f32;
        assert!((dropped - 0.3).abs() < 0.03, "dropped fraction {dropped}");
    }

    #[test]
    fn successive_forwards_draw_different_masks() {
        let d = Device::Cpu;
        let x = Tensor::zeros((64, 64), candle_core::DType::F32, &d).unwrap();
        let m = DropoutMasks::new(3, "projection");
        let first = mask_values(&m.next_mask(&x, 0.3).unwrap());
        let second = mask_values(&m.next_mask(&x, 0.3).unwrap());
        assert_ne!(
            first, second,
            "a counter-keyed mask must advance; an unchanging mask is not dropout"
        );
    }

    #[test]
    fn two_layers_do_not_share_a_mask() {
        let d = Device::Cpu;
        let x = Tensor::zeros((64, 64), candle_core::DType::F32, &d).unwrap();
        let a = DropoutMasks::new(5, "layer.0.Wqkv");
        let b = DropoutMasks::new(5, "layer.1.Wqkv");
        assert_ne!(
            mask_values(&a.next_mask(&x, 0.3).unwrap()),
            mask_values(&b.next_mask(&x, 0.3).unwrap()),
            "masks are keyed by layer name; two layers must not correlate"
        );
    }

    /// The resume invariant, and the reason this is counter-keyed rather than an
    /// advancing stream: restoring is an assignment, so it is O(1) at any
    /// position. The old design replayed `position` draws one at a time, and
    /// position advanced by one per activation *element* — order 1e11 per layer
    /// after one epoch of a large encoder.
    #[test]
    fn restore_position_reproduces_the_uninterrupted_masks() {
        let d = Device::Cpu;
        let x = Tensor::zeros((32, 32), candle_core::DType::F32, &d).unwrap();
        let reference = DropoutMasks::new(7, "projection");
        let _ = reference.next_mask(&x, 0.2).unwrap();
        let _ = reference.next_mask(&x, 0.2).unwrap();
        let pos = reference.position();
        assert_eq!(pos, 2, "position counts forwards, not draws");
        let third = mask_values(&reference.next_mask(&x, 0.2).unwrap());

        let resumed = DropoutMasks::new(7, "projection");
        resumed.restore_position(pos);
        assert_eq!(resumed.position(), pos);
        assert_eq!(
            third,
            mask_values(&resumed.next_mask(&x, 0.2).unwrap()),
            "a restored layer must draw the mask the uninterrupted run would have"
        );
    }

    /// Restoring must not depend on how far in the run the position is. A
    /// replay-based restore would take time proportional to it; this asserts the
    /// behaviour is identical at a position no replay could reach.
    #[test]
    fn restore_is_position_independent() {
        let d = Device::Cpu;
        let x = Tensor::zeros((8, 8), candle_core::DType::F32, &d).unwrap();
        let far = 5_000_000_000u64;
        let a = DropoutMasks::new(9, "projection");
        a.restore_position(far);
        let b = DropoutMasks::new(9, "projection");
        b.restore_position(far);
        assert_eq!(
            mask_values(&a.next_mask(&x, 0.25).unwrap()),
            mask_values(&b.next_mask(&x, 0.25).unwrap())
        );
        assert_eq!(a.position(), far + 1);
    }
}
