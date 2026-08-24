//! Seeded, candle-RNG-free initialisation and dropout for the LoRA path.
//!
//! candle's CPU `rand_uniform` / `randn` draw from the process-global
//! `rand::rng()` (its `set_seed` is a no-op on CPU), so `Init::Kaiming` /
//! `Init::Randn` and `candle_nn::ops::dropout` are *unseedable* — two runs of
//! the same fine-tune produce different adapters. This module owns the draws
//! instead: a small self-contained SplitMix64 PRNG fills host buffers that are
//! then registered as trainable `Var`s, and a per-layer [`DropoutMasks`]
//! source draws each training forward's Bernoulli mask device-side, in-kernel
//! (via `jammi_kernels::ops::DropoutFused`, a counter-based Philox draw —
//! never candle's unseedable global RNG, and never a host-materialized mask).
//! Nothing here touches a global RNG, so a fine-tune is a pure function of
//! `(seed, source rows, config)`.
//!
//! **Cross-process determinism.** Every seeded-init draw stream is keyed by
//! `(seed, fully-qualified parameter name)` via [`seed_for_param`], never by
//! `VarMap`/`HashMap` iteration order. So which order the layers happen to be
//! constructed or iterated in is irrelevant: the `projection.lora_a` tensor is
//! byte-identical regardless of how many other layers exist or when they were
//! built. This is the same FNV-1a-then-SplitMix idiom the engine already uses
//! for its seeded graph walks. [`DropoutMasks`] instead keys its Philox draw
//! on `(run seed, layer_id, forward_idx, element_index)` directly — see its
//! own doc for why the layer identity flows through a hashed `layer_id`
//! rather than being folded into the seed itself.

use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::Tensor;
use jammi_kernels::ops::{DropoutFused, DropoutKey};

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

/// A LoRA layer's dropout mask source: COUNTER-KEYED, not an advancing
/// stream. Adopts the shape preserved from `wip/device-side-dropout`
/// (that branch's `DropoutMasks`/atomic-counter design was worth keeping;
/// its `Device::set_seed` draw mechanism was NOT — see this crate's
/// `jammi_kernels::ops::dropout` module doc for the full rejected-mechanism
/// writeup) with the mechanism this commit actually ships: every draw runs
/// through `jammi_kernels::ops::DropoutFused`, a counter-based Philox
/// `CustomOp1` — never a host-materialized mask, never a per-device RNG.
///
/// The mask for a layer's k-th training forward is a pure function of
/// `(run seed, layer_id, k, element_index)`. Two properties follow, and
/// both were absent from the advancing-stream design this replaces:
///
/// * **Restore is O(1).** The old stream had no closed-form skip, so
///   restoring a persisted position replayed that many draws one at a
///   time from the origin. The position advanced by ONE DRAW PER
///   ACTIVATION ELEMENT (`batch * seq * in_features` per forward), so
///   after a single epoch of a large encoder that is on the order of 1e11
///   draws per layer — resume did not merely slow down, it failed to
///   finish (esc-033). Restoring a counter is an assignment.
/// * **The mask is never materialized at all**, on either device — closing
///   the wip branch's finding #2 (candle's `Binary::Mul` backward retains a
///   full-size gradient FOR a host-built mask tensor, since the mask is not
///   itself a graph leaf `sorted_nodes` walks). `DropoutFused` is one
///   `CustomOp1` node forward and one more node backward, with no third
///   (mask) tensor ever created.
///
/// `layer_id` is a pure hash of the layer's fully-qualified name (via
/// [`layer_id_for_name`]) — NOT mixed with the run seed the way
/// [`seed_for_param`] mixes a seeded-init stream's origin. The Philox
/// `key` (unchanging across a layer's whole life) IS the raw run seed;
/// layer identity flows entirely through the Philox `counter`'s
/// `layer_id` slot instead — see `jammi_kernels::philox`'s module doc for
/// the exact `(seed, layer_id, forward_idx, element_index)` mapping this
/// depends on.
pub(crate) struct DropoutMasks {
    /// The RUN's own seed — unchanged across every layer (see the struct
    /// doc for why layer identity is NOT folded in here).
    run_seed: u64,
    /// A hash of this layer's fully-qualified name — the Philox counter's
    /// layer-identifying slot.
    layer_id: u32,
    /// Training forwards taken through this layer so far. The mask key,
    /// and the whole resume state. An `AtomicU64` (not a `Mutex`) so a
    /// `LoraLinear` held behind `&self` (as `Module`-style `forward`
    /// requires) can still advance it without a lock — mirrors the wip
    /// branch's "atomic counter replacing the per-layer `Mutex`" choice.
    counter: AtomicU64,
}

impl DropoutMasks {
    pub(crate) fn new(seed: u64, layer_name: &str) -> Self {
        Self {
            run_seed: seed,
            layer_id: layer_id_for_name(layer_name),
            counter: AtomicU64::new(0),
        }
    }

    /// Forwards taken so far — the unit of resume state for this layer's
    /// dropout (a FORWARD COUNT, not a draw count — the unit this commit
    /// changes `ResumeState::dropout_positions` to, see
    /// `jammi-ai/src/fine_tune/resume.rs`).
    pub(crate) fn position(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }

    /// Restore the forward counter. O(1): the mask is a pure function of
    /// the counter, so there is nothing to replay.
    pub(crate) fn restore_position(&self, position: u64) {
        self.counter.store(position, Ordering::Relaxed);
    }

    /// Reserve the NEXT training forward's Philox key WITHOUT drawing
    /// anything: advances the forward counter by exactly ONE (the same
    /// atomic `fetch_add` [`Self::apply`] itself performs) and returns the
    /// `(seed, layer_id, forward_idx, p)` tuple as a
    /// [`jammi_kernels::ops::DropoutKey`].
    ///
    /// This exists so a call site can decide FUSED-vs-EAGER-FALLBACK
    /// *after* reserving the key and have BOTH arms consume the identical
    /// key — critically, the fallback arm must NOT call [`Self::apply`]
    /// (which would reserve a SECOND, different `forward_idx` for the same
    /// logical forward, breaking esc-033's O(1) resume invariant: two
    /// arms of the SAME forward must advance the counter by exactly one
    /// between them, not one each). `p` must already be validated to
    /// `[0.0, 1.0)` by the caller (`LoraLinear::new`); the eventual
    /// `DropoutFused::new` construction re-validates independently
    /// regardless (family D).
    pub(crate) fn next_key(&self, p: f32) -> Result<DropoutKey, candle_core::Error> {
        let k = self.counter.fetch_add(1, Ordering::Relaxed);
        let forward_idx: u32 = k.try_into().map_err(|_| {
            candle_core::Error::Msg(format!(
                "dropout: forward counter {k} for layer_id {} exceeds u32::MAX — the Philox \
                 counter mapping reserves exactly 32 bits for the forward index \
                 (jammi_kernels::philox)",
                self.layer_id
            ))
        })?;
        Ok(DropoutKey {
            seed: self.run_seed,
            layer_id: self.layer_id,
            forward_idx,
            p,
        })
    }

    /// Apply this layer's dropout to `x` for the NEXT training forward
    /// (advancing the counter by one, via [`Self::next_key`]) and return
    /// the result directly — the mask is never materialized as a separate
    /// tensor at any point.
    ///
    /// `LoraLinear::forward` no longer calls this directly (it calls
    /// [`Self::next_key`] itself, so
    /// both the fused and eager-fallback arms can consume the SAME
    /// reserved key — see `lora_linear::LoraLinear::forward`'s doc). Kept
    /// (not deleted) as the `#[cfg(test)]` module's own oracle surface
    /// below: `apply`'s determinism/resume tests (`masks_are_deterministic
    /// _and_inverted_scaled`, `restore_position_reproduces_the
    /// _uninterrupted_masks`, the esc-033 anti-relaxation clause, …) are
    /// this type's actual correctness proof and are simplest to state
    /// against the one-call `apply` shape rather than every test
    /// re-inlining `next_key` + `DropoutFused::new` + `apply1`.
    #[allow(
        dead_code,
        reason = "exercised extensively by this module's own #[cfg(test)] oracles; \
                  no longer reachable from the plain (non-test) library build now that \
                  LoraLinear::forward calls next_key directly"
    )]
    pub(crate) fn apply(&self, x: &Tensor, p: f32) -> Result<Tensor, candle_core::Error> {
        let key = self.next_key(p)?;
        let op = DropoutFused::new(key.seed, key.layer_id, key.forward_idx, key.p)?;
        jammi_kernels::ops::apply1(x, op)
    }
}

/// Hash a layer's fully-qualified name (e.g. `"layer.3.attn.Wqkv"`) to a
/// `u32` `layer_id` — the Philox counter's layer-identifying slot. Pure
/// function of the NAME ALONE (no run seed mixed in — see
/// [`DropoutMasks`]'s doc for why layer identity and run identity are kept
/// on separate Philox slots rather than combined into one hashed origin
/// the way seeded-init's [`seed_for_param`] does). FNV-1a over the name
/// bytes (matching [`seed_for_param`]'s own hash), then one SplitMix64
/// finalizer round folded down to 32 bits by XORing its two halves — a
/// well-mixed avalanche, not merely a truncation, so nearby names are not
/// likely to collide.
pub(crate) fn layer_id_for_name(name: &str) -> u32 {
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    let mut z = hash;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 32) as u32) ^ (z as u32)
}

/// Audit advisory (post-4aa1303 round): `layer_id` is the ONLY Philox
/// counter slot derived from a hash rather than carried verbatim — the
/// one place two DISTINCT sites CAN collide, and a collision means two
/// sites share every mask silently (no error, correlated dropout, forever
/// — the same `layer_id` at the same `forward_idx` draws the identical
/// Philox stream). All 112 ModernBERT-large site names were confirmed
/// collision-free by direct enumeration (see
/// `tests::real_modernbert_large_names_are_collision_free`), but that is
/// a property of a CONFIG-DEPENDENT input, not a proof — this function is
/// the structural guard: given every name that will construct a
/// `DropoutMasks` for one run (the run's "DropoutMasks set"), it
/// verifies the induced `layer_id` set has the SAME cardinality as the
/// name set — i.e. the hash is injective over this particular input — and
/// refuses with a typed error NAMING both colliding sites otherwise,
/// rather than silently proceeding into correlated dropout.
///
/// Intended call site: once, at training-loop construction, over every
/// name that will end up with a live `DropoutMasks` (the same names
/// [`crate::LoraLinear::dropout_position`]'s callers already collect via
/// `dropout_positions()`'s keys, stripped of their `.dropout` suffix —
/// exactly what was passed to `DropoutMasks::new` for that layer). NOT
/// called per-forward or per-layer-construction: this is a one-time,
/// whole-run structural check, not a hot-path cost.
pub fn assert_no_layer_id_collisions<'a, I>(names: I) -> Result<(), crate::error::LoraError>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut seen: std::collections::HashMap<u32, &str> = std::collections::HashMap::new();
    for name in names {
        let id = layer_id_for_name(name);
        match seen.get(&id) {
            Some(&prev) if prev != name => {
                return Err(crate::error::LoraError::Config(format!(
                    "layer_id collision: '{prev}' and '{name}' both hash to layer_id {id} — \
                     their dropout masks would silently correlate (identical draws at every \
                     forward, forever) if left unrefused; rename one site, or extend \
                     layer_id_for_name's mixing"
                )));
            }
            _ => {
                seen.insert(id, name);
            }
        }
    }
    Ok(())
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
    fn layer_id_for_name_is_a_pure_function_of_the_name_alone() {
        assert_eq!(
            layer_id_for_name("layer.0.Wqkv"),
            layer_id_for_name("layer.0.Wqkv")
        );
        assert_ne!(
            layer_id_for_name("layer.0.Wqkv"),
            layer_id_for_name("layer.1.Wqkv")
        );
    }

    /// The real 112-site ModernBERT-large naming scheme (28 layers x
    /// `{attn.Wqkv, attn.Wo, mlp.Wi, mlp.Wo}`, `"layer.{n}.{site}"` —
    /// matching `jammi_encoders::modernbert`'s own `collect_dropout_
    /// position` call sites) — the audit's config-dependent claim, pinned
    /// as a real regression oracle rather than left as prose.
    #[test]
    fn real_modernbert_large_names_are_collision_free() {
        let mut names = Vec::new();
        for layer in 0..28 {
            for site in ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"] {
                names.push(format!("layer.{layer}.{site}"));
            }
        }
        assert_eq!(
            names.len(),
            112,
            "the fixture itself must be the real site count"
        );
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        assert!(
            assert_no_layer_id_collisions(refs).is_ok(),
            "the real 112-name ModernBERT-large set must be collision-free"
        );
    }

    /// The audit advisory itself: `layer_id` is a 32-bit hash, so a
    /// collision IS structurally possible — engineer one by brute force
    /// (a fixed base name + an increasing suffix counter; the birthday
    /// bound over a 32-bit output puts the expected first collision
    /// around sqrt(2^32) ~= 65536 tries, so a 500k cap is generous) and
    /// assert [`assert_no_layer_id_collisions`] refuses it with a typed
    /// error naming BOTH colliding sites — not merely detecting *a*
    /// collision, but the one this specific run would have silently hit.
    #[test]
    fn engineered_collision_is_a_typed_refusal_naming_both_sites() {
        let mut seen: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
        let mut collision: Option<(String, String)> = None;
        for i in 0..500_000u64 {
            let name = format!("layer.0.attn.Wqkv#{i}");
            let id = layer_id_for_name(&name);
            if let Some(prev) = seen.get(&id) {
                collision = Some((prev.clone(), name));
                break;
            }
            seen.insert(id, name);
        }
        let (a, b) = collision.expect(
            "a 32-bit hash must produce a collision well within 500k tries \
             (birthday bound ~65536) — if this fails, layer_id_for_name's output \
             range grew and this fixture needs a larger search cap, not removal",
        );
        assert_ne!(a, b, "the engineered pair must be two DISTINCT names");
        assert_eq!(
            layer_id_for_name(&a),
            layer_id_for_name(&b),
            "the engineered pair must actually share a layer_id"
        );

        let err = assert_no_layer_id_collisions([a.as_str(), b.as_str()])
            .expect_err("two distinct names sharing a layer_id must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains(&a) && msg.contains(&b),
            "the typed error must name BOTH colliding sites: {msg}"
        );
    }

    /// The SAME name appearing twice (e.g. a caller passing a
    /// deduplicated-but-still-repeated list) is NOT a collision — only
    /// two DISTINCT names sharing a `layer_id` are.
    #[test]
    fn a_repeated_identical_name_is_not_a_false_positive_collision() {
        assert!(assert_no_layer_id_collisions(["layer.0.attn.Wqkv", "layer.0.attn.Wqkv"]).is_ok());
    }

    #[test]
    fn masks_are_deterministic_and_inverted_scaled() {
        let d = candle_core::Device::Cpu;
        let x = Tensor::ones((100, 100), candle_core::DType::F32, &d).unwrap();
        let a = DropoutMasks::new(11, "projection");
        let b = DropoutMasks::new(11, "projection");
        let m1 = mask_values(&a.apply(&x, 0.3).unwrap());
        let m2 = mask_values(&b.apply(&x, 0.3).unwrap());
        assert_eq!(
            m1, m2,
            "same (seed, layer, counter) must draw the same mask"
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
        let d = candle_core::Device::Cpu;
        let x = Tensor::ones((64, 64), candle_core::DType::F32, &d).unwrap();
        let m = DropoutMasks::new(3, "projection");
        let first = mask_values(&m.apply(&x, 0.3).unwrap());
        let second = mask_values(&m.apply(&x, 0.3).unwrap());
        assert_ne!(
            first, second,
            "a counter-keyed mask must advance; an unchanging mask is not dropout"
        );
    }

    #[test]
    fn two_layers_do_not_share_a_mask() {
        let d = candle_core::Device::Cpu;
        let x = Tensor::ones((64, 64), candle_core::DType::F32, &d).unwrap();
        let a = DropoutMasks::new(5, "layer.0.Wqkv");
        let b = DropoutMasks::new(5, "layer.1.Wqkv");
        assert_ne!(
            mask_values(&a.apply(&x, 0.3).unwrap()),
            mask_values(&b.apply(&x, 0.3).unwrap()),
            "masks are keyed by layer_id; two layers must not correlate"
        );
    }

    /// The resume invariant, and the reason this is counter-keyed rather
    /// than an advancing stream: restoring is an assignment, so it is O(1)
    /// at any position. The old design replayed `position` draws one at a
    /// time, and position advanced by one per activation ELEMENT — order
    /// 1e11 per layer after one epoch of a large encoder (esc-033).
    #[test]
    fn restore_position_reproduces_the_uninterrupted_masks() {
        let d = candle_core::Device::Cpu;
        let x = Tensor::ones((32, 32), candle_core::DType::F32, &d).unwrap();
        let reference = DropoutMasks::new(7, "projection");
        let _ = reference.apply(&x, 0.2).unwrap();
        let _ = reference.apply(&x, 0.2).unwrap();
        let pos = reference.position();
        assert_eq!(pos, 2, "position counts forwards, not draws");
        let third = mask_values(&reference.apply(&x, 0.2).unwrap());

        let resumed = DropoutMasks::new(7, "projection");
        resumed.restore_position(pos);
        assert_eq!(resumed.position(), pos);
        assert_eq!(
            third,
            mask_values(&resumed.apply(&x, 0.2).unwrap()),
            "a restored layer must draw the mask the uninterrupted run would have"
        );
    }

    /// esc-033's ANTI-RELAXATION CLAUSE (the promotion-gating oracle): O(1)
    /// restore must not be "bought" by resuming onto a DIFFERENT stream —
    /// the failure a counter-based redesign can still have is an off-by-
    /// one in `forward_idx`. The check above (one draw after restore)
    /// cannot see that class of bug if the off-by-one happens to
    /// self-correct after one step; this asserts the WHOLE post-restore
    /// stream — every forward from the restored position through the end
    /// of the run — is byte-identical to the uninterrupted run's, not just
    /// the immediate next one.
    #[test]
    fn post_restore_stream_is_byte_identical_to_the_uninterrupted_run_across_many_forwards() {
        let d = candle_core::Device::Cpu;
        let x = Tensor::ones((16, 16), candle_core::DType::F32, &d).unwrap();
        const N: u64 = 12;
        const K: u64 = 5;

        // Uninterrupted reference: N forwards, every output recorded.
        let reference = DropoutMasks::new(123, "projection");
        let mut ref_outputs = Vec::with_capacity(N as usize);
        for _ in 0..N {
            ref_outputs.push(mask_values(&reference.apply(&x, 0.3).unwrap()));
        }

        // The "crashed" run: a SEPARATE instance that only ran the first K
        // forwards before the persisted checkpoint.
        let interrupted = DropoutMasks::new(123, "projection");
        for _ in 0..K {
            interrupted.apply(&x, 0.3).unwrap();
        }
        let pos = interrupted.position();
        assert_eq!(pos, K);

        // The resumed run: a FRESH instance restored to that position,
        // continuing for every remaining forward.
        let resumed = DropoutMasks::new(123, "projection");
        resumed.restore_position(pos);
        let mut resumed_outputs = Vec::with_capacity((N - K) as usize);
        for _ in 0..(N - K) {
            resumed_outputs.push(mask_values(&resumed.apply(&x, 0.3).unwrap()));
        }

        for i in 0..(N - K) {
            assert_eq!(
                resumed_outputs[i as usize],
                ref_outputs[(K + i) as usize],
                "post-restore forward {i} (uninterrupted-run forward {}) diverged — \
                 O(1) restore must reproduce the identical stream, not merely a \
                 plausible-looking one",
                K + i
            );
        }
    }

    /// The negative control proving the oracle above has teeth: restoring
    /// to `K + 1` instead of `K` (the EXACT off-by-one esc-033's anti-
    /// relaxation clause names) must NOT reproduce the uninterrupted run's
    /// continuation. If it did, the positive test above would be
    /// vacuously insensitive to this failure mode.
    #[test]
    fn post_restore_stream_would_catch_an_off_by_one_forward_idx() {
        let d = candle_core::Device::Cpu;
        let x = Tensor::ones((16, 16), candle_core::DType::F32, &d).unwrap();
        const N: u64 = 8;
        const K: u64 = 3;

        let reference = DropoutMasks::new(77, "projection");
        let mut ref_outputs = Vec::with_capacity(N as usize);
        for _ in 0..N {
            ref_outputs.push(mask_values(&reference.apply(&x, 0.3).unwrap()));
        }

        let off_by_one = DropoutMasks::new(77, "projection");
        off_by_one.restore_position(K + 1); // the injected bug: should be K
        let wrong_output = mask_values(&off_by_one.apply(&x, 0.3).unwrap());

        assert_ne!(
            wrong_output, ref_outputs[K as usize],
            "an off-by-one restore position must NOT reproduce the correct \
             continuation — if it did, the positive oracle above would be vacuous"
        );
    }

    /// Restoring must not depend on how far into the run the position is —
    /// a replay-based restore would take time proportional to it; this
    /// asserts the behaviour is identical at a position no replay could
    /// reach in any reasonable time.
    #[test]
    fn restore_is_position_independent() {
        let d = candle_core::Device::Cpu;
        let x = Tensor::ones((8, 8), candle_core::DType::F32, &d).unwrap();
        // Within `u32::MAX` (the Philox counter's forward-index ceiling —
        // see `forward_counter_overflow_is_a_typed_refusal_not_a_silent_wrap`
        // below) but far beyond anything a replay loop could reach in test
        // time, which is the property this test asserts.
        let far = 2_000_000_000u64;
        let a = DropoutMasks::new(9, "projection");
        a.restore_position(far);
        let b = DropoutMasks::new(9, "projection");
        b.restore_position(far);
        assert_eq!(
            mask_values(&a.apply(&x, 0.25).unwrap()),
            mask_values(&b.apply(&x, 0.25).unwrap())
        );
        assert_eq!(a.position(), far + 1);
    }

    /// A forward counter that would overflow the Philox counter's 32-bit
    /// slot is a typed refusal, not a silent wraparound into a REUSED
    /// (and therefore wrongly correlated) counter value.
    #[test]
    fn forward_counter_overflow_is_a_typed_refusal_not_a_silent_wrap() {
        let d = candle_core::Device::Cpu;
        let x = Tensor::ones((4, 4), candle_core::DType::F32, &d).unwrap();
        let m = DropoutMasks::new(1, "projection");
        m.restore_position(u32::MAX as u64 + 1);
        assert!(
            m.apply(&x, 0.2).is_err(),
            "a forward index beyond u32::MAX must be refused, not silently wrapped"
        );
    }
}
