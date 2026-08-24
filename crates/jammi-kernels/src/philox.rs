//! Philox4x32-10, a counter-based PRNG: `(counter, key) -> 4 words of
//! output`, with NO internal state carried between calls — the same
//! `(counter, key)` pair always produces the same output, on any device,
//! forever. This is the property [`ops::DropoutFused`](crate::ops::DropoutFused)
//! is built on: a dropout mask element becomes a PURE FUNCTION of
//! `(seed, layer, forward#, element index)` rather than a position in an
//! advancing stream, so restoring it is an assignment (O(1)), not a replay.
//!
//! ## Provenance (cite in every consumer's doc, per the C7 contract)
//!
//! Ported from Random123 (D. E. Shaw Research,
//! <https://github.com/DEShawResearch/random123>, BSD-3-Clause — see that
//! repository's `LICENSE`), specifically `include/Random123/philox.h`'s
//! `philox4x32` round function and key schedule. This is a from-scratch
//! Rust re-derivation of the published algorithm and constants, NOT a
//! vendored/transliterated copy of the C header, and NOT curand's headers
//! (NVIDIA's curand ships its own Philox implementation under an EULA that
//! forbids the kind of open reproduction this crate needs — this port
//! never reads or reuses curand source in any form; only the freely
//! published Random123 constants and algorithm, and the BSD-3 kat_vectors
//! test file below, are used).
//!
//! The constants below are quoted directly from `philox.h`:
//! `PHILOX_M4x32_0`, `PHILOX_M4x32_1`, `PHILOX_W32_0`, `PHILOX_W32_1`; the
//! round function is `_philox4x32round` (one Threefry-style "quarter
//! round": two 32x32->64 multiplies, each split into hi/lo, XORed into the
//! opposite counter word together with a key word); rounds 2..=10 are
//! preceded by `_philox4x32bumpkey` (`key[0] += W0; key[1] += W1`) — the
//! FIRST round runs with the UNBUMPED key (this ordering is load-bearing:
//! bumping before round 1 as well would not match the published KAT
//! vectors below, which is exactly the failure mode the shared test
//! vector suite this module doc promises exists to catch).
//!
//! ## The shared test-vector suite (the crate's proof obligation)
//!
//! Random123 publishes known-answer test (KAT) vectors for `philox4x32-10`
//! in `tests/kat_vectors` (fetched 2026-08-24 from the `main` branch of
//! the repository above). `tests::kat_vector_all_zero`,
//! `tests::kat_vector_all_ones`, and `tests::kat_vector_mixed` below
//! assert this Rust implementation reproduces all three published vectors
//! EXACTLY. The CUDA device function in `cuda/dropout.cu` re-implements
//! this SAME algorithm in CUDA C (necessarily a second, textually
//! independent implementation — there is no shared source between a
//! `.rs` file and a `.cu` file compiled by nvcc); `tests/cuda_parity.rs`'s
//! `philox_kat_vectors_match_on_cuda` (strict-gated, `cuda` feature only)
//! runs the SAME three vectors through a minimal CUDA test kernel and
//! asserts bit-identical `u32` output against this module — that pair of
//! tests is the actual proof the two implementations compute the same
//! function, not merely "both compile".

/// `PHILOX_M4x32_0` (`philox.h`) — the first round's multiplier.
pub const PHILOX_M0: u32 = 0xD251_1F53;
/// `PHILOX_M4x32_1` (`philox.h`) — the second round's multiplier.
pub const PHILOX_M1: u32 = 0xCD9E_8D57;
/// `PHILOX_W32_0` (`philox.h`) — the Weyl increment added to `key[0]`
/// between rounds.
pub const PHILOX_W0: u32 = 0x9E37_79B9;
/// `PHILOX_W32_1` (`philox.h`) — the Weyl increment added to `key[1]`
/// between rounds.
pub const PHILOX_W1: u32 = 0xBB67_AE85;
/// The number of rounds this crate uses — `philox4x32-10`, Random123's own
/// documented "conservative" round count (`PHILOX4x32_DEFAULT_ROUNDS`),
/// not the faster-but-less-conservative 7-round variant `philox.h` also
/// publishes KAT vectors for.
pub const PHILOX_ROUNDS: u32 = 10;

/// `a*b` split into `(hi, lo)` 32-bit halves of the full 64-bit product —
/// `mulhilo32` in `philox.h`. CUDA's device function computes the
/// identical split via `__umulhi(a,b)` (hi) and the ordinary `a*b`
/// truncating multiply (lo); both are exact, so there is no rounding
/// question here, unlike the FINAL inverted-dropout scale multiply
/// (`ops::dropout`'s module doc) which does need explicitly-rounded
/// intrinsics.
#[inline]
fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
    let product = (a as u64) * (b as u64);
    ((product >> 32) as u32, product as u32)
}

/// One `_philox4x32round`: `(hi0,lo0) = mulhilo32(M0, ctr[0])`,
/// `(hi1,lo1) = mulhilo32(M1, ctr[2])`, output
/// `{hi1^ctr[1]^key[0], lo1, hi0^ctr[3]^key[1], lo0}` — quoted verbatim
/// from `philox.h`'s `_philox4x32round_tpl` macro body.
#[inline]
fn round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo32(PHILOX_M0, ctr[0]);
    let (hi1, lo1) = mulhilo32(PHILOX_M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// `_philox4x32bumpkey`: `key[0] += W0; key[1] += W1` (wrapping — `philox.h`'s
/// C `+=` on `uint32_t` is itself defined-wraparound arithmetic).
#[inline]
fn bumpkey(key: [u32; 2]) -> [u32; 2] {
    [
        key[0].wrapping_add(PHILOX_W0),
        key[1].wrapping_add(PHILOX_W1),
    ]
}

/// `philox4x32-10(counter, key) -> 4 words of output`.
///
/// Round 1 runs with the UNBUMPED key; `key` is bumped BEFORE each of
/// rounds 2..=10 (9 bumps total for 10 rounds) — see `philox.h`'s
/// `philoxNxW_R` macro, quoted in this module's doc, for why this specific
/// off-by-one ordering (bump-then-round, starting from round 2) is not a
/// free choice: getting it wrong (e.g. bumping before round 1 too) changes
/// every output word and fails the KAT vectors below.
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut ctr = round(counter, key);
    let mut k = key;
    for _ in 1..PHILOX_ROUNDS {
        k = bumpkey(k);
        ctr = round(ctr, k);
    }
    ctr
}

/// The counter mapping [`ops::DropoutFused`](crate::ops::DropoutFused) and
/// its CUDA counterpart both key their draw on: `key = (seed lo, seed
/// hi)` (the Philox `key`, unchanged across every layer/forward/element —
/// carrying only the RUN's identity), `counter = (layer_id, forward_idx,
/// element_index lo, element_index hi)` (the Philox `counter`, unique per
/// draw — carrying the LAYER's identity, WHICH training forward this is,
/// and WHICH element of the activation this is). Every element's draw is
/// therefore a pure function of `(seed, layer, forward#, index)`, with no
/// dependence on evaluation order, thread scheduling, or prior draws —
/// this is what makes O(1) restore possible (closing esc-033) and what
/// makes the mask never need to be materialized (closing wip finding #2).
///
/// Only the FIRST of Philox's 4 output words is used as "the draw" for the
/// KEEP/DROP decision (see `ops::dropout`'s module doc for why: one full
/// Philox invocation per ELEMENT, discarding 3 of the 4 output words, is a
/// deliberate simplicity-over-throughput choice — this kernel is memory-
/// bound, not compute-bound, so the discarded work costs nothing measured,
/// and it keeps the counter mapping a literal 1:1 statement of `(seed,
/// layer, forward#, index)` rather than a 4-ELEMENTS-per-counter packing
/// scheme that would need its own, separate correctness argument for the
/// other 3 words).
#[inline]
pub fn philox_draw(seed: u64, layer_id: u32, forward_idx: u32, element_index: u64) -> u32 {
    let key = [seed as u32, (seed >> 32) as u32];
    let counter = [
        layer_id,
        forward_idx,
        element_index as u32,
        (element_index >> 32) as u32,
    ];
    philox4x32_10(counter, key)[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random123 `tests/kat_vectors`, `philox4x32 10` line 1 (fetched
    /// 2026-08-24 from `DEShawResearch/random123@main`):
    /// `philox4x32 10 00000000 00000000 00000000 00000000 00000000 00000000
    /// 6627e8d5 e169c58d bc57ac4c 9b00dbd8` — fields are `ctr[4] key[2]`
    /// then the expected `result[4]`, all hex.
    #[test]
    fn kat_vector_all_zero() {
        let ctr = [0u32, 0, 0, 0];
        let key = [0u32, 0];
        let expected = [0x6627e8d5u32, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8];
        assert_eq!(philox4x32_10(ctr, key), expected);
    }

    /// `tests/kat_vectors`, `philox4x32 10` line 2 — all-`0xffffffff` input.
    #[test]
    fn kat_vector_all_ones() {
        let ctr = [0xffffffffu32; 4];
        let key = [0xffffffffu32, 0xffffffff];
        let expected = [0x408f276du32, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd];
        assert_eq!(philox4x32_10(ctr, key), expected);
    }

    /// `tests/kat_vectors`, `philox4x32 10` line 3 — the "mixed" vector
    /// (the first digits of pi/e in hex, Random123's own fixture).
    #[test]
    fn kat_vector_mixed() {
        let ctr = [0x243f6a88u32, 0x85a308d3, 0x13198a2e, 0x03707344];
        let key = [0xa4093822u32, 0x299f31d0];
        let expected = [0xd16cfe09u32, 0x94fdcceb, 0x5001e420, 0x24126ea1];
        assert_eq!(philox4x32_10(ctr, key), expected);
    }

    #[test]
    fn philox_draw_is_a_pure_function_of_its_four_inputs() {
        let a = philox_draw(42, 3, 7, 100);
        let b = philox_draw(42, 3, 7, 100);
        assert_eq!(
            a, b,
            "identical (seed, layer, forward, index) must draw identically"
        );
    }

    #[test]
    fn different_forward_idx_draws_a_different_value() {
        let a = philox_draw(42, 3, 7, 100);
        let b = philox_draw(42, 3, 8, 100);
        assert_ne!(
            a, b,
            "a different forward index must (almost certainly) draw differently"
        );
    }

    #[test]
    fn different_element_index_draws_a_different_value() {
        let a = philox_draw(42, 3, 7, 100);
        let b = philox_draw(42, 3, 7, 101);
        assert_ne!(a, b);
    }

    #[test]
    fn different_layer_id_draws_a_different_value() {
        let a = philox_draw(42, 3, 7, 100);
        let b = philox_draw(42, 4, 7, 100);
        assert_ne!(a, b);
    }

    #[test]
    fn different_seed_draws_a_different_value() {
        let a = philox_draw(42, 3, 7, 100);
        let b = philox_draw(43, 3, 7, 100);
        assert_ne!(a, b);
    }
}
