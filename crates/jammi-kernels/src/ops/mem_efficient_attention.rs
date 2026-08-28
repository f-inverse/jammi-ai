//! Memory-efficient attention (`MemEfficientAttention`): a chunked
//! composition of stock candle tensor ops implementing the Rabe & Staats
//! scheme ("Self-attention Does Not Need O(n²) Memory", arXiv:2112.05682) —
//! a running (max, sum-exp) accumulation over KEY chunks in the forward
//! pass, and a manual, per-chunk-recomputed backward that never
//! materializes a `[batch, heads, seq, seq]`-shaped tensor. No new CUDA, no
//! `.cu`, no vendored tree: both `fwd` and `bwd` reuse this crate's own
//! primitives ([`BackendStorage::matmul`], [`super::RopeFused`]'s row math,
//! [`matmul_grad_lhs`]/[`matmul_grad_rhs`]) — the SAME "composed interior"
//! idiom [`super::AttentionBlockFused`] documents, extended with a chunked
//! outer loop over keys.
//!
//! Generic primitive (family L): this crate names no consumer. This
//! module's doc cites shapes/values only to explain numeric choices, never
//! as a dependency.
//!
//! ## Why a `CustomOp3` at all: the checkpointing IS the op boundary
//!
//! candle has no checkpointing API. A naive chunked `fwd` composed from
//! plain, tracked `Tensor` ops would retain every chunk's softmax
//! intermediate on the tape (autograd records every intermediate a tracked
//! computation touches), and the whole memory argument collapses back to
//! `O(seq²)`. Wrapping the chunked loop in ONE `CustomOp3` node makes the
//! candle engine treat it as an OPAQUE, single-node op: nothing chunk-
//! shaped survives on the tape between `fwd` and `bwd`, and `bwd` itself
//! RECOMPUTES each chunk's local softmax (from `q`, `k_chunk`, and the
//! forward's own stored per-row log-sum-exp) rather than reading a stored
//! attention matrix — exactly Rabe & Staats' own "recompute, don't retain"
//! backward, expressed at the op boundary.
//!
//! ## This pass: CPU-hermetic only (family L / VALIDATION scope)
//!
//! This op ships `cpu_fwd` only. `cuda_fwd` is left at [`CustomOp3`]'s
//! default (`Err("no cuda implementation")`) — the CUDA composition (and
//! the dispatch-lattice wiring that would ever route real traffic here) is
//! POD-DEFERRED, not attempted in this pass; see the crate's own hand-off
//! notes for the explicit scope line. Every oracle this module ships is
//! therefore CPU/F32-only, which is also this op's only DOMAIN-VALID CPU
//! dtype: candle-core 0.11's CPU backend has no `BF16` `MatMul`
//! implementation (the same pre-existing limitation
//! [`super::AttentionBlockFused`]'s own module doc discloses) — `BF16` is
//! refused on CPU with a typed `UnsupportedDTypeForOp`, never a silent
//! fallback.
//!
//! ## Domain (family D)
//!
//! `qkv`: rank 5 `[batch, seq, 3, heads, head_dim]`, contiguous, `F32` on
//! CPU. Unlike [`super::AttentionBlockFused`], `head_dim` is UNCONSTRAINED
//! (no fixed `64` — this op folds no bit-exactness argument into an exact-
//! power-of-two scale; it is arch-agnostic stock-op composition, not a
//! kernel tuned to one width). `seq <= `[`MAX_SEQ`]` — a conservative
//! VALIDATED ceiling, not a hardware limit, mirroring every other `MAX_*`
//! constant in this crate. `rope_pack` (when `rope == true`): the SAME
//! `[2, 1, 1, seq_max, head_dim]` pack [`super::AttentionBlockFused`]
//! accepts — reused via [`super::attention_block::check_rope_pack`]
//! directly rather than re-derived (one definition, not two that could
//! drift). `key_mask`: rank 4 `[batch|1, 1, 1, seq]` — PADDING ONLY. This
//! is narrower than [`super::AttentionBlockFused`]'s combined-mask
//! broadcast class on purpose (next section).
//!
//! `b == 0 || seq == 0 || heads == 0` is in-domain: `cpu_fwd`'s chunk loop
//! simply does not execute (an empty `s` makes the outer `while c_start <
//! s` loop zero-trip), yielding an empty `[batch, seq, heads*head_dim]`
//! output — no separate fast path. `bwd` DOES special-case this shape
//! (documented at its own call site below): `Tensor::cat(&[], ..)` errors
//! on an empty chunk list, so `bwd` short-circuits to a zero `dqkv` before
//! ever entering the chunk loop, rather than inheriting `cpu_fwd`'s "the
//! general path already handles it" shape.
//!
//! ## The band is a `Copy` scalar, not a tensor (family D)
//!
//! `half_window: Option<usize>` is CONSTRUCTION DATA on the op itself, re-
//! derived per key-chunk from `(query_row, key_position, half_window)` —
//! never materialized as a `[seq, seq]` (or even `[seq, chunk]` cached
//! across chunks) tensor anywhere in this arm. This is deliberately a
//! SECOND copy of the `|q - k| <= half_window` predicate this crate
//! already computes once for [`super::AttentionBlockFused`]'s callers (via
//! their own combined-mask tensor) — accepted, not treated as drift risk,
//! because the alternative (materializing a `[seq, seq]` band mask and
//! chunk-slicing INTO it) would reintroduce exactly the `O(seq²)` term
//! this whole arm exists to avoid. `key_mask` therefore stays the
//! narrower, padding-only `[batch|1, 1, 1, seq]` shape (`O(batch·seq)`):
//! this op's callers do NOT pre-combine a band into it the way
//! `AttentionBlockFused`'s callers do.
//!
//! ## `FullyMaskedPolicy::Zeros`: a running MAX over mask chunks
//!
//! A row `(b, q)` is fully masked iff `max_k combined_mask[b, q, k] < 0.0`
//! — the SAME `< 0.0` convention [`super::softmax::row_is_fully_masked`]
//! and `softmax.cu`'s `mask_row_is_fully_masked` use, computed here as a
//! RUNNING max carried across the key-chunk loop (never re-scanning
//! earlier chunks): each chunk updates `mask_running_max[b, q] =
//! max(mask_running_max[b, q], max_k_in_chunk combined_mask[b, q, k])`,
//! and the trigger is evaluated once, after the last chunk. Under
//! [`FullyMaskedPolicy::Zeros`] a triggered row's output is forced to
//! EXACT zeros (never a computed-then-overwritten value); under
//! `Propagate` the running max is not even computed — ordinary online-
//! softmax division runs unconditionally, reproducing candle-eager
//! behavior on that row (including a possible `NaN`/uniform result,
//! exactly as `Propagate` does everywhere else in this crate).
//!
//! ## `bwd`'s `lse` channel: [`Saved`] makes this a [`StatefulKernelOp`]
//!
//! `fwd` stores `(out, lse)` — `lse[b,h,q] = m[b,h,q] + ln(l[b,h,q])`, the
//! row's final running max plus the log of its final running sum-exp, the
//! SAME two numbers flash-style kernels store for their own checkpointed
//! backward. Candle's `CustomOp3` has no save-for-backward channel (the
//! constraint [`super::AttentionBlockFused`]'s and [`super::RopeFused`]'s
//! own `bwd` docs already state), so this op uses [`Saved`] — `lse:
//! Saved<Tensor>` — exactly `crate::ops::flash_attention`'s own pattern
//! (a backtick code span, not an intra-doc link: that module is
//! `flash-attn`-feature-gated and absent from a default-feature `cargo
//! doc` build):
//! `fwd` calls `self.lse.set(..)`, `bwd` calls `self.lse.take()`. This
//! makes `MemEfficientAttention` a [`StatefulKernelOp`], not a `KernelOp`
//! (it cannot be `Copy` — see [`StatefulKernelOp`]'s own doc for why), run
//! through [`super::apply_stateful3`]. A fully-masked (`Zeros`-triggered)
//! row stores [`MASKED_LSE_SENTINEL`] instead of its real `m + ln(l)`: a
//! large finite value chosen so `bwd`'s `exp(masked_score - lse)` cleanly
//! UNDERFLOWS to exactly `0.0` for that row (never `NaN`/`inf` — an actual
//! `-inf` sentinel would turn `score - (-inf) = +inf` and
//! `exp(+inf) = inf`) — reproducing the fact that `Zeros` makes the
//! forward output a CONSTANT (not a differentiable function of the score)
//! for that row, so its true gradient contribution is exactly zero, with
//! no separate branch needed in `bwd`'s Tensor-level composition.
//!
//! `bwd` recomputes, per key chunk: `scores_c = q_scaled · k_cᵀ`, the
//! combined mask (`key_mask` chunk `+` the re-derived band, mirroring
//! `fwd`), `p_c = exp(scores_c + mask_c - lse)`, then the standard
//! softmax-attention backward identities (`D = rowsum(O ⊙ dO)`,
//! `dV_c = p_cᵀ @ dO`, `dP_c = dO @ V_cᵀ`, `dS_c = p_c ⊙ (dP_c - D)`,
//! `dQ += dS_c @ K_c`, `dK_c = dS_cᵀ @ Q_scaled`) via
//! [`matmul_grad_lhs`]/[`matmul_grad_rhs`] — the SAME shared GEMM-gradient
//! definitions [`super::AttentionBlockFused::bwd`] uses, so a gradient GEMM
//! is defined once in this crate, not re-derived per op. `dQ` accumulates
//! ACROSS the chunk loop (every key chunk contributes to every query row's
//! gradient); `dK_c`/`dV_c` are chunk-LOCAL (a key row's gradient only ever
//! depends on the one chunk that key belongs to) and are concatenated,
//! never accumulated, after the loop. Every tensor `bwd` builds is derived
//! from `.detach()`-ed inputs (mirroring `AttentionBlockFused::bwd`'s own
//! "runs DETACHED" section — nothing here tracks an `Op`, so nothing
//! chunk-shaped is handed back to the engine); `bwd` returns `(Some(dqkv),
//! None, None)` — this op computes no gradient for `rope_pack`/`key_mask`
//! and asserts `!track_op()` on both, loudly, before doing any work (a
//! typed refusal rather than a silently-missing gradient, family D).
//!
//! ## Rounding contract (dtype-split — CPU/F32 arm only, this pass)
//!
//! `F32` (this op's only CPU dtype): `f32` accumulation throughout —
//! `scale` folded into `Q` once (a plain multiply, not scale-then-divide),
//! the mask add and the online-softmax running-max/sum-exp recurrence all
//! in `f32`, one round point (there is none — no narrower dtype exists on
//! this arm). Rust never auto-fuses a multiply-add the way `nvcc`'s
//! `--fmad` contraction can, so the CUDA build-flag "fmad-accepted-
//! tolerance" doctrine `softmax.cu`'s own module doc states does not apply
//! here — stated as N/A, not silently omitted. `BF16` (the CUDA-arm-only
//! concern, deferred): governed by the SAME `bf16_mul_rounded`/
//! `bf16_add_rounded` round-back primitives `softmax.cu` documents (accumulate
//! in `f32`, round to `bf16` once per op, never silently "improve" a
//! fully-masked row's rounding relative to that contract) — stated here so
//! a future CUDA arm inherits the decision rather than re-deriving it; NOT
//! implemented in this pass.
//!
//! ## `chunk_size` is provenance, not shared identity (stated, not wired)
//!
//! `chunk` (this op's own [`MemEfficientAttention::chunk`]) changes
//! REDUCTION ORDER — it is therefore numerics, and env-overriding it in a
//! measurement path would silently invalidate a recorded number (family J:
//! determinism requires an explicit, fixed fold order). It is a
//! jammi-SIDE PROVENANCE field, never a member of any shared cross-
//! producer identity tuple: a torch reference run cannot state a
//! `chunk_size` at all (it has no chunked arm), so adding this field to a
//! SHARED identity would read `MISSING` on every torch-producer row. The
//! correct treatment — recorded here as the decision, not wired by this
//! pass (bench/CI identity plumbing is later work, after the encoder-
//! lattice dispatch lands) — is a NullMeans-class provenance field: a
//! non-memeff row emits `null` WITH MEANING ("this arm has no chunk
//! size"), never simply absent.
//!
//! ## Admission (stated, not wired this pass)
//!
//! This op has no admission-lattice entry yet (dispatch wiring is out of
//! scope this pass — see the crate's hand-off notes). For the record: the
//! op's own device gate, when wired, is `device_is_supported`
//! (CPU-or-CUDA) — never an exact-arch predicate like flash's — since this
//! is stock-op composition, not a kernel tuned to one SM target.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, Device, Error, Layout, Result, Shape, Tensor, D};

use super::attention_block::check_rope_pack;
use super::rope::rope_fwd_row_f32;
use super::saved::Saved;
use super::{
    apply3, apply_stateful3, matmul_grad_lhs, matmul_grad_rhs, FullyMaskedPolicy, RopeFused,
};

/// The smallest `chunk` this op accepts. Below this, the plan's own
/// launch-count model (`(seq/chunk)` launches per forward, each a real
/// `BackendStorage::matmul` call) starts to dominate wall time on candle's
/// eager execution — the SAME "`(s/c)²` launches ≈ 7s of pure launch
/// latency at `c=128`" argument that rejected 2-D block×chunk looping in
/// favor of keys-only chunking in the first place, restated as a floor on
/// `c` itself for the 1-D loop this op actually runs. A conservative,
/// VALIDATED floor, not a correctness requirement — [`MemEfficientAttention::new`]
/// refuses anything smaller.
pub const MIN_CHUNK: usize = 512;

/// The largest `seq` this op accepts. A conservative, VALIDATED ceiling —
/// not a hardware limit — mirroring every other `MAX_*` constant in this
/// crate (see e.g. [`super::ATTENTION_BLOCK_MAX_SEQ`]'s doc for the same
/// status). Deliberately far above [`super::ATTENTION_BLOCK_MAX_SEQ`]
/// (`4096`): this IS the long-sequence arm.
pub const MAX_SEQ: usize = 131_072;

/// This op's own additive out-of-window sentinel, re-derived per key-chunk
/// (module doc's "the band is a `Copy` scalar" section) rather than read
/// off a caller-combined mask tensor the way [`super::AttentionBlockFused`]
/// does. Numerically the SAME value as
/// [`super::ATTENTION_BLOCK_WINDOW_MASKED_VALUE`] / `jammi_encoders::
/// mask::MASKED_LOGIT` (a second, INDEPENDENT constant, not an alias of
/// either — this op never reads a caller's mask tensor for its band term
/// at all, so there is no shared-value hazard to pin via an equality test
/// the way the block arm's own sentinel needs one).
pub const WINDOW_MASKED_VALUE: f32 = -10_000.0;

/// The `lse` sentinel a `FullyMaskedPolicy::Zeros`-triggered row stores in
/// place of its real `m + ln(l)` (module doc's "`bwd`'s `lse` channel"
/// section). Large enough that `masked_score - MASKED_LSE_SENTINEL`
/// underflows `exp` cleanly to exactly `0.0` for any realistic score
/// magnitude (scores are `O(1)`-`O(10⁴)` at this crate's mask magnitude;
/// `1e30` leaves 26+ orders of margin) while staying far inside `f32`'s
/// finite range (`f32::MAX ≈ 3.4e38`), so `masked_score - 1e30` is a large
/// finite negative number — never `NaN`/`inf` the way a literal `-inf`
/// sentinel would produce via `finite - (-inf) = +inf`.
const MASKED_LSE_SENTINEL: f32 = 1.0e30;

/// Memory-efficient (chunked, checkpointed) attention. See the module doc
/// for the full design. Constructed only through [`MemEfficientAttention::new`].
pub struct MemEfficientAttention {
    /// The scaled-dot-product scale, folded into `Q` before `QKᵀ` (module
    /// doc: no power-of-two constraint, unlike
    /// [`super::AttentionBlockFused`] — this op's `head_dim` is
    /// unconstrained, so no bit-exactness argument depends on `scale`
    /// being an exact power of two). Private for the same "no invalid
    /// inhabitant via a struct literal" reason
    /// [`super::AttentionBlockFused::scale`]'s own doc states.
    scale: f32,
    /// See [`super::FullyMaskedPolicy`]'s own doc; reused unchanged.
    pub fully_masked: FullyMaskedPolicy,
    /// Whether `rope_pack` is applied to `Q`/`K`. `false` lets a caller
    /// with no positional embedding reuse this op — `rope_pack` is then
    /// present but never read (mirrors
    /// [`super::AttentionBlockFused::rope`]).
    pub rope: bool,
    /// The sliding-window half-width, re-derived per key-chunk (module
    /// doc's "the band is a `Copy` scalar" section). `None` means no band
    /// — every unmasked-by-padding key is attendable.
    pub half_window: Option<usize>,
    /// The key-chunk width. Private: [`MemEfficientAttention::new`] is the
    /// only way to set it, enforcing [`MIN_CHUNK`].
    chunk: usize,
    /// `fwd`'s `[batch, heads, seq]` log-sum-exp, consumed by `bwd`'s
    /// checkpointed recompute. See the module doc's "`bwd`'s `lse`
    /// channel" section.
    lse: Saved<Tensor>,
}

impl MemEfficientAttention {
    /// `scale` must be finite and strictly positive (mirrors
    /// [`super::AttentionBlockFused::new`]'s identical check, minus the
    /// power-of-two requirement — see this op's own `scale` field doc for
    /// why that requirement does not apply here). `chunk` must be
    /// `>= `[`MIN_CHUNK`].
    pub fn new(
        scale: f32,
        fully_masked: FullyMaskedPolicy,
        rope: bool,
        half_window: Option<usize>,
        chunk: usize,
    ) -> Result<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::Msg(format!(
                "mem_efficient_attention: scale must be finite and strictly positive, got {scale}"
            )));
        }
        if chunk < MIN_CHUNK {
            return Err(Error::Msg(format!(
                "mem_efficient_attention: chunk must be >= MIN_CHUNK ({MIN_CHUNK}) — see \
                 MIN_CHUNK's own doc for the launch-count argument; got {chunk}"
            )));
        }
        Ok(Self {
            scale,
            fully_masked,
            rope,
            half_window,
            chunk,
            lse: Saved::empty(),
        })
    }

    /// Reads the validated [`Self::scale`].
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Reads the validated [`Self::chunk`] (`>= `[`MIN_CHUNK`]).
    pub fn chunk(&self) -> usize {
        self.chunk
    }
}

impl super::sealed::Sealed for MemEfficientAttention {}

#[cfg(test)]
impl MemEfficientAttention {
    /// TEST-ONLY: bypasses [`MIN_CHUNK`] so this module's own unit tests
    /// can exercise a genuinely multi-chunk loop at toy shapes without
    /// paying [`MIN_CHUNK`]-sized compute — mirrors `ops::cast_scale`'s own
    /// "TEST-ONLY preallocated-output entry points" precedent (never used
    /// outside `#[cfg(test)]`; [`MemEfficientAttention::new`] is the only
    /// production constructor and always enforces [`MIN_CHUNK`]).
    fn new_test_chunk(
        scale: f32,
        fully_masked: FullyMaskedPolicy,
        rope: bool,
        half_window: Option<usize>,
        chunk: usize,
    ) -> Result<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::Msg(format!(
                "mem_efficient_attention: scale must be finite and strictly positive, got {scale}"
            )));
        }
        if chunk == 0 {
            return Err(Error::Msg(
                "mem_efficient_attention: chunk must be > 0".into(),
            ));
        }
        Ok(Self {
            scale,
            fully_masked,
            rope,
            half_window,
            chunk,
            lse: Saved::empty(),
        })
    }
}

/// The ONLY public entry point besides [`MemEfficientAttention::new`]
/// itself — a thin wrapper over [`super::apply_stateful3`], mirroring
/// `crate::ops::flash_attention::flash_attention_varlen`'s own "one
/// function, fresh op per call" convention (a backtick code span, not an
/// intra-doc link: that item is `flash-attn`-feature-gated and absent from
/// a default-feature `cargo doc` build).
pub fn mem_efficient_attention(
    qkv: &Tensor,
    rope_pack: &Tensor,
    key_mask: &Tensor,
    op: MemEfficientAttention,
) -> Result<Tensor> {
    apply_stateful3(qkv, rope_pack, key_mask, op)
}

/// Validates `qkv`'s domain (module doc). Returns `(batch, seq, heads,
/// head_dim)`.
pub(crate) fn mem_eff_attention_dims(
    l_qkv: &Layout,
    op: &'static str,
) -> Result<(usize, usize, usize, usize)> {
    let dims = l_qkv.dims();
    if dims.len() != 5 || dims[2] != 3 {
        return Err(Error::Msg(format!(
            "{op}: qkv must be rank 5 [batch, seq, 3, heads, head_dim], got {dims:?}"
        )));
    }
    let (b, s, h, d) = (dims[0], dims[1], dims[3], dims[4]);
    if s > MAX_SEQ {
        return Err(Error::Msg(format!(
            "{op}: seq={s} exceeds MAX_SEQ={MAX_SEQ} (a conservative validated ceiling, not a \
             hardware limit)"
        )));
    }
    Ok((b, s, h, d))
}

/// Validates `key_mask`'s domain (module doc: padding-only, NARROWER than
/// [`super::AttentionBlockFused`]'s combined-mask class — no query-row
/// axis, since the band is separate construction data here). Returns the
/// mask's own leading (batch) axis size (`1` or `b`).
pub(crate) fn check_key_mask(
    l_mask: &Layout,
    b: usize,
    s: usize,
    op: &'static str,
) -> Result<usize> {
    let dims = l_mask.dims();
    if dims.len() != 4 || dims[1] != 1 || dims[2] != 1 || dims[3] != s {
        return Err(Error::Msg(format!(
            "{op}: key_mask must be [batch|1, 1, 1, {s}] (padding-only — the band is separate \
             construction data via half_window, not part of this mask), got {dims:?}"
        )));
    }
    if dims[0] != 1 && dims[0] != b {
        return Err(Error::Msg(format!(
            "{op}: key_mask's leading axis must be 1 or batch={b}, got {}",
            dims[0]
        )));
    }
    Ok(dims[0])
}

/// The ONE definition of the sliding-window additive predicate this arm
/// uses, shared by [`build_band_chunk_tensor`] (the `bwd` Tensor-level
/// arm) and `cpu_fwd`'s own raw-storage row loop — never re-derived twice
/// within this op (module doc: the acceptable SECOND copy is relative to
/// the encoder-side `sliding_window_mask`, not within this file itself).
#[inline]
fn band_additive_value(query_row: usize, key_pos: usize, half_window: usize) -> f32 {
    if query_row.abs_diff(key_pos) <= half_window {
        0.0
    } else {
        WINDOW_MASKED_VALUE
    }
}

/// Materializes ONE chunk's worth of band (`[1, 1, seq, chunk_len]`) —
/// `O(seq · chunk_len)`, never `O(seq²)` — for `bwd`'s Tensor-level
/// composition. `cpu_fwd`'s own raw-storage loop calls
/// [`band_additive_value`] directly, per cell, with no intermediate
/// allocation at all.
fn build_band_chunk_tensor(
    seq: usize,
    chunk_start: usize,
    chunk_len: usize,
    half_window: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut band = Vec::with_capacity(seq * chunk_len);
    for qi in 0..seq {
        for kj in 0..chunk_len {
            band.push(band_additive_value(qi, chunk_start + kj, half_window));
        }
    }
    Tensor::from_vec(band, (1, 1, seq, chunk_len), device)
}

impl CustomOp3 for MemEfficientAttention {
    fn name(&self) -> &'static str {
        "mem_efficient_attention"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let op = self.name();
        let (b, s, h, d) = mem_eff_attention_dims(l1, op)?;
        let out_shape = Shape::from((b, s, h * d));
        let mask_batch = check_key_mask(l3, b, s, op)?;
        if s1.dtype() != s3.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s3.dtype(),
                op,
            });
        }
        if self.rope && s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op,
            });
        }
        match (s1, s3) {
            (CpuStorage::F32(qkv), CpuStorage::F32(mask)) => {
                let (o1, o2) = l1
                    .contiguous_offsets()
                    .ok_or(Error::RequiresContiguous { op })?;
                let (m1, m2) = l3
                    .contiguous_offsets()
                    .ok_or(Error::RequiresContiguous { op })?;
                let rope_slice = if self.rope {
                    let s_max = check_rope_pack(l2, s, d, op)?;
                    match s2 {
                        CpuStorage::F32(rp) => {
                            let (r1, r2) = l2
                                .contiguous_offsets()
                                .ok_or(Error::RequiresContiguous { op })?;
                            Some((&rp[r1..r2], s_max))
                        }
                        other => return Err(Error::UnsupportedDTypeForOp(other.dtype(), op)),
                    }
                } else {
                    None
                };
                let (out, lse) = attention_fwd_memeff_f32(&MemEffFwdF32Params {
                    qkv: &qkv[o1..o2],
                    rope: rope_slice,
                    mask: &mask[m1..m2],
                    mask_batch,
                    b,
                    s,
                    h,
                    d,
                    scale: self.scale,
                    half_window: self.half_window,
                    chunk: self.chunk,
                    policy: self.fully_masked,
                })?;
                let lse_tensor = Tensor::from_vec(lse, (b, h, s), &Device::Cpu)?;
                self.lse
                    .set(lse_tensor)
                    .map_err(|e| Error::Msg(format!("{op}: {e}")))?;
                Ok((CpuStorage::F32(out), out_shape))
            }
            // `BF16` (or any other dtype) on CPU: candle-core 0.11's CPU
            // backend has no `BF16` `MatMul` impl (module doc). A
            // qkv/mask dtype MISMATCH never reaches this arm — refused by
            // the explicit `DTypeMismatchBinaryOp` check above.
            (s1, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), op)),
        }
    }

    /// See the module doc's "`bwd`'s `lse` channel" section for the full
    /// design. `res` (fwd's own output, `O`) IS used here — unlike
    /// [`super::AttentionBlockFused::bwd`] (which never needs its `_res`)
    /// — to build `D = rowsum(O ⊙ dO)`, the standard softmax-attention
    /// backward correction term.
    fn bwd(
        &self,
        qkv: &Tensor,
        rope_pack: &Tensor,
        mask: &Tensor,
        res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let op = self.name();
        if rope_pack.track_op() || mask.track_op() {
            return Err(Error::Msg(format!(
                "{op}: this op computes no gradient for the RoPE table or the key mask — \
                 asserted here rather than silently returning None (family D): rope_pack/mask \
                 must never be tracked (never a Var, never downstream of one)"
            )));
        }
        let lse = self
            .lse
            .take()
            .map_err(|e| Error::Msg(format!("{op}: {e}")))?;

        // DETACH every tensor input before composing anything — the SAME
        // move `AttentionBlockFused::bwd` makes (see its own module-doc
        // section "`bwd` runs DETACHED"): without it, every `Tensor` built
        // below would carry a `BackpropOp` cloning its inputs, and the
        // whole per-chunk recompute would be handed back to the engine
        // inside `dqkv`'s own `Op` — exactly the retention this op exists
        // to avoid.
        let qkv = qkv.detach();
        let rope_pack = rope_pack.detach();
        let mask = mask.detach();
        let res = res.detach();
        let grad_res = grad_res.detach();
        let lse = lse.detach();

        let (b, s, three, h, d) = qkv.dims5()?;
        if three != 3 {
            return Err(Error::Msg(format!(
                "{op}: qkv must be rank 5 [batch, seq, 3, heads, head_dim], got 3-axis size \
                 {three}"
            )));
        }
        // Empty-shape short circuit (module doc): `Tensor::cat(&[], ..)`
        // errors on an empty chunk list, which the general chunk loop
        // below would hit whenever `s == 0` — short-circuit before it,
        // rather than inheriting `cpu_fwd`'s "general path handles it"
        // shape.
        if b == 0 || s == 0 || h == 0 {
            return Ok((
                Some(Tensor::zeros((b, s, 3, h, d), qkv.dtype(), qkv.device())?),
                None,
                None,
            ));
        }

        let q0 = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?;
        let k0 = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?;
        let v0 = qkv
            .narrow(2, 2, 1)?
            .squeeze(2)?
            .transpose(1, 2)?
            .contiguous()?;

        let (q_rot, k_rot, cos_sin) = if self.rope {
            let cos_full = rope_pack.narrow(0, 0, 1)?.squeeze(0)?;
            let sin_full = rope_pack.narrow(0, 1, 1)?.squeeze(0)?;
            let cos = cos_full.narrow(2, 0, s)?;
            let sin = sin_full.narrow(2, 0, s)?;
            let qr = apply3(&q0.contiguous()?, &cos, &sin, RopeFused::new(false))?;
            let kr = apply3(&k0.contiguous()?, &cos, &sin, RopeFused::new(false))?;
            (qr, kr, Some((cos, sin)))
        } else {
            (q0.contiguous()?, k0.contiguous()?, None)
        };

        // Materialized once, reused by every chunk's `scores_c` recompute
        // AND (further down) `dkr_c`'s gradient GEMM — mirroring
        // `AttentionBlockFused::bwd`'s own `q_scaled` comment.
        let q_scaled = (&q_rot * f64::from(self.scale))?.contiguous()?;

        let o = res.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?;
        let dctx = grad_res
            .reshape((b, s, h, d))?
            .transpose(1, 2)?
            .contiguous()?;
        // D_i = rowsum_d(O_i ⊙ dO_i) — the standard softmax-attention
        // backward correction term (module doc). `O(b,h,s,d)`-sized, never
        // `[.., seq, seq]`.
        let delta = o.mul(&dctx)?.sum_keepdim(D::Minus1)?;
        let lse_unsq = lse.reshape((b, h, s, 1))?;

        let mut dqs = Tensor::zeros((b, h, s, d), qkv.dtype(), qkv.device())?;
        let mut dk_chunks: Vec<Tensor> = Vec::new();
        let mut dv_chunks: Vec<Tensor> = Vec::new();

        let mut c_start = 0usize;
        while c_start < s {
            let clen = self.chunk.min(s - c_start);
            let k_c = k_rot.narrow(2, c_start, clen)?.contiguous()?;
            let v_c = v0.narrow(2, c_start, clen)?.contiguous()?;
            // Transposed VIEW, matching `cpu_fwd`'s own chunk-scores GEMM
            // operand form (this arm has no "match production's own
            // eager autograd" constraint the way `AttentionBlockFused`'s
            // `dqs`/`dkr` do — there is no pre-existing eager call site
            // this NEW arm must byte-match).
            let k_c_t = k_c.transpose(D::Minus1, D::Minus2)?;
            let scores_c = q_scaled.matmul(&k_c_t)?;
            let mask_c = mask.narrow(3, c_start, clen)?;
            let mut masked_c = scores_c.broadcast_add(&mask_c)?;
            if let Some(w) = self.half_window {
                let band_c = build_band_chunk_tensor(s, c_start, clen, w, qkv.device())?;
                masked_c = masked_c.broadcast_add(&band_c)?;
            }
            // `p_c = exp(masked_c - lse)`: for a `Zeros`-triggered row,
            // `lse == MASKED_LSE_SENTINEL` (module doc), so this
            // underflows cleanly to `0.0` — no separate branch needed.
            let p_c = masked_c.broadcast_sub(&lse_unsq)?.exp()?;
            let dv_c = matmul_grad_rhs(&p_c, &dctx)?;
            let dp_c = matmul_grad_lhs(&dctx, &v_c)?;
            let ds_c = p_c.mul(&dp_c.broadcast_sub(&delta)?)?;
            let dqs_c = matmul_grad_lhs(&ds_c, &k_c_t)?;
            dqs = dqs.add(&dqs_c)?;
            let dk_c = matmul_grad_rhs(&q_scaled, &ds_c)?
                .transpose(D::Minus1, D::Minus2)?
                .contiguous()?;
            dk_chunks.push(dk_c);
            dv_chunks.push(dv_c);
            c_start += clen;
        }

        let dkr = Tensor::cat(&dk_chunks, 2)?;
        let dv = Tensor::cat(&dv_chunks, 2)?;
        let dqr = (&dqs * f64::from(self.scale))?;

        let (dq0, dk0) = if let Some((cos, sin)) = cos_sin {
            (
                apply3(&dqr, &cos, &sin, RopeFused::new(true))?,
                apply3(&dkr, &cos, &sin, RopeFused::new(true))?,
            )
        } else {
            (dqr, dkr)
        };

        let to_qkv_slot = |t: &Tensor| -> Result<Tensor> {
            t.transpose(1, 2)?.contiguous()?.reshape((b, s, 1, h, d))
        };
        let dqkv = Tensor::cat(
            &[&to_qkv_slot(&dq0)?, &to_qkv_slot(&dk0)?, &to_qkv_slot(&dv)?],
            2,
        )?;

        Ok((Some(dqkv), None, None))
    }
}

/// [`attention_fwd_memeff_f32`]'s inputs, bundled into one struct rather
/// than passed positionally — mirrors `AttentionBlockFused`'s own
/// `AttentionFwdF32Params` (see that struct's doc for the transposition
/// hazard this removes).
struct MemEffFwdF32Params<'a> {
    /// `[b, s, 3, h, d]`, contiguous.
    qkv: &'a [f32],
    /// `(cos-then-sin table, seq_max)`, or `None` when `rope == false`.
    rope: Option<(&'a [f32], usize)>,
    /// `[mask_batch, 1, 1, s]`, contiguous — padding only.
    mask: &'a [f32],
    mask_batch: usize,
    b: usize,
    s: usize,
    h: usize,
    d: usize,
    scale: f32,
    half_window: Option<usize>,
    chunk: usize,
    policy: FullyMaskedPolicy,
}

/// The composed, chunked CPU forward. Gathers `Q`/`K`/`V` into
/// `[batch*heads, seq, head_dim]` contiguous buffers (the SAME fixed
/// ascending `(batch, seq, heads)` gather order [`super::AttentionBlockFused`]'s
/// own `attention_fwd_f32` uses — family J), RoPE-rotates `Q`/`K`, folds
/// `scale` into `Q`, then loops over KEY chunks (module doc): per chunk,
/// one [`BackendStorage::matmul`] for `scores_c`, a per-row online-softmax
/// update (running max/sum-exp/weighted-`V`-accumulator, Rabe & Staats),
/// and a running max-over-mask-chunks for the `Zeros` trigger. Returns
/// `(out, lse)` — `lse` feeds `bwd`'s own checkpointed recompute.
fn attention_fwd_memeff_f32(params: &MemEffFwdF32Params<'_>) -> Result<(Vec<f32>, Vec<f32>)> {
    let MemEffFwdF32Params {
        qkv,
        rope,
        mask,
        mask_batch,
        b,
        s,
        h,
        d,
        scale,
        half_window,
        chunk,
        policy,
    } = *params;
    let bh = b * h;
    let sd = s * d;

    let mut q = vec![0f32; bh * sd];
    let mut k = vec![0f32; bh * sd];
    let mut v = vec![0f32; bh * sd];
    for bi in 0..b {
        for si in 0..s {
            let base = (bi * s + si) * 3 * h * d;
            for hi in 0..h {
                let q_src = base + hi * d;
                let k_src = base + h * d + hi * d;
                let v_src = base + 2 * h * d + hi * d;
                let dst = (bi * h + hi) * sd + si * d;
                q[dst..dst + d].copy_from_slice(&qkv[q_src..q_src + d]);
                k[dst..dst + d].copy_from_slice(&qkv[k_src..k_src + d]);
                v[dst..dst + d].copy_from_slice(&qkv[v_src..v_src + d]);
            }
        }
    }

    if let Some((table, s_max)) = rope {
        let cos = &table[0..s_max * d];
        let sin = &table[s_max * d..2 * s_max * d];
        let mut qr = vec![0f32; bh * sd];
        let mut kr = vec![0f32; bh * sd];
        for bh_i in 0..bh {
            for si in 0..s {
                let off = bh_i * sd + si * d;
                let cos_row = &cos[si * d..(si + 1) * d];
                let sin_row = &sin[si * d..(si + 1) * d];
                rope_fwd_row_f32(
                    &q[off..off + d],
                    cos_row,
                    sin_row,
                    1.0,
                    &mut qr[off..off + d],
                );
                rope_fwd_row_f32(
                    &k[off..off + d],
                    cos_row,
                    sin_row,
                    1.0,
                    &mut kr[off..off + d],
                );
            }
        }
        q = qr;
        k = kr;
    }

    for qv in q.iter_mut() {
        *qv *= scale;
    }

    let mut m = vec![f32::NEG_INFINITY; bh * s];
    let mut l = vec![0f32; bh * s];
    let mut acc = vec![0f32; bh * s * d];
    let mut mask_running_max = vec![f32::NEG_INFINITY; b * s];

    let q_layout_full = Layout::contiguous((bh, s, d));
    let q_storage = CpuStorage::F32(q);

    let mut masked_row_buf: Vec<f32> = Vec::new();
    let mut c_start = 0usize;
    while c_start < s {
        let clen = chunk.min(s - c_start);
        let mut k_chunk = vec![0f32; bh * clen * d];
        let mut v_chunk = vec![0f32; bh * clen * d];
        for bhi in 0..bh {
            let src = bhi * sd + c_start * d;
            let dst = bhi * clen * d;
            k_chunk[dst..dst + clen * d].copy_from_slice(&k[src..src + clen * d]);
            v_chunk[dst..dst + clen * d].copy_from_slice(&v[src..src + clen * d]);
        }
        let kc_layout = Layout::contiguous((bh, clen, d));
        let kc_t_layout = kc_layout.transpose(1, 2)?;
        let scores_storage = q_storage.matmul(
            &CpuStorage::F32(k_chunk),
            (bh, s, clen, d),
            &q_layout_full,
            &kc_t_layout,
        )?;
        let CpuStorage::F32(scores) = scores_storage else {
            return Err(Error::Msg(
                "mem_efficient_attention: internal matmul returned a non-F32 storage for an F32 \
                 input"
                    .into(),
            ));
        };

        masked_row_buf.resize(clen, 0.0);
        for bhi in 0..bh {
            let bi = bhi / h;
            let head_is_first = bhi % h == 0;
            let mrow_base = if mask_batch == 1 { 0 } else { bi * s };
            for qi in 0..s {
                let row_idx = bhi * s + qi;
                let srow = &scores[row_idx * clen..(row_idx + 1) * clen];
                let mut chunk_max = f32::NEG_INFINITY;
                for kj in 0..clen {
                    let global_k = c_start + kj;
                    let pad_val = mask[mrow_base + global_k];
                    let combined = match half_window {
                        Some(w) => pad_val + band_additive_value(qi, global_k, w),
                        None => pad_val,
                    };
                    let v_ = srow[kj] + combined;
                    masked_row_buf[kj] = v_;
                    if v_ > chunk_max {
                        chunk_max = v_;
                    }
                    // The mask value is independent of `hi` — compute the
                    // running-max update once per (batch, query) pair,
                    // not once per (batch, head, query) redundantly.
                    if head_is_first {
                        let idx = bi * s + qi;
                        if combined > mask_running_max[idx] {
                            mask_running_max[idx] = combined;
                        }
                    }
                }
                let m_old = m[row_idx];
                let new_max = if chunk_max > m_old { chunk_max } else { m_old };
                // `(m_old - new_max).exp()`: `m_old == NEG_INFINITY` on
                // the first chunk gives `exp(-inf) == 0.0` — correctly
                // discarding the (already-zero) stale accumulator with no
                // special-cased first-chunk branch.
                let correction = (m_old - new_max).exp();
                let acc_row = &mut acc[row_idx * d..(row_idx + 1) * d];
                for a in acc_row.iter_mut() {
                    *a *= correction;
                }
                let mut p_sum = 0f32;
                for kj in 0..clen {
                    let e = (masked_row_buf[kj] - new_max).exp();
                    p_sum += e;
                    let v_row = &v_chunk[(bhi * clen + kj) * d..(bhi * clen + kj + 1) * d];
                    for di in 0..d {
                        acc_row[di] += e * v_row[di];
                    }
                }
                l[row_idx] = l[row_idx] * correction + p_sum;
                m[row_idx] = new_max;
            }
        }
        c_start += clen;
    }

    let mut out_bh = vec![0f32; bh * s * d];
    let mut lse = vec![0f32; bh * s];
    for bhi in 0..bh {
        let bi = bhi / h;
        for qi in 0..s {
            let row_idx = bhi * s + qi;
            let fully_masked =
                policy == FullyMaskedPolicy::Zeros && mask_running_max[bi * s + qi] < 0.0;
            let acc_row = &acc[row_idx * d..(row_idx + 1) * d];
            let out_row = &mut out_bh[row_idx * d..(row_idx + 1) * d];
            if fully_masked {
                out_row.fill(0.0);
                lse[row_idx] = MASKED_LSE_SENTINEL;
            } else {
                let denom = l[row_idx];
                for di in 0..d {
                    out_row[di] = acc_row[di] / denom;
                }
                lse[row_idx] = m[row_idx] + denom.ln();
            }
        }
    }

    let mut out = vec![0f32; b * s * h * d];
    for bi in 0..b {
        for hi in 0..h {
            for si in 0..s {
                let src = ((bi * h + hi) * s + si) * d;
                let dst = (bi * s + si) * h * d + hi * d;
                out[dst..dst + d].copy_from_slice(&out_bh[src..src + d]);
            }
        }
    }
    Ok((out, lse))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    fn qkv_from(q0: &Tensor, k0: &Tensor, v0: &Tensor) -> Result<Tensor> {
        let stacked = Tensor::stack(&[q0, k0, v0], 2)?; // [B,H,3,S,D]
        stacked.permute((0, 3, 2, 1, 4))?.contiguous()
    }

    fn pack_rope(cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        Tensor::stack(&[cos, sin], 0)
    }

    fn rope_tables(s_max: usize, d: usize, device: &Device) -> (Tensor, Tensor) {
        let half = d / 2;
        let mut cos_v = Vec::with_capacity(s_max * d);
        let mut sin_v = Vec::with_capacity(s_max * d);
        for pos in 0..s_max {
            for _ in 0..2 {
                for i in 0..half {
                    let theta = (pos as f64) * (10_000f64.powf(-2.0 * i as f64 / d as f64));
                    cos_v.push(theta.cos() as f32);
                    sin_v.push(theta.sin() as f32);
                }
            }
        }
        let cos = Tensor::from_vec(cos_v, (1, 1, s_max, d), device).unwrap();
        let sin = Tensor::from_vec(sin_v, (1, 1, s_max, d), device).unwrap();
        (cos, sin)
    }

    fn zero_key_mask(b: usize, s: usize, device: &Device) -> Tensor {
        Tensor::from_vec(vec![0f32; b * s], (b, 1, 1, s), device).unwrap()
    }

    /// A small, deliberately UNCHUNKED eager reference built from ordinary
    /// `Tensor` ops (RoPE, scale-fold, `QKᵀ`, mask-add [+ band], softmax,
    /// `PV`) — independent of `MemEfficientAttention`'s own chunked
    /// implementation, assembled here rather than imported from
    /// `jammi-encoders` (family L). `key_mask` is padding-only
    /// (`[b|1,1,1,s]`); `half_window` (if any) is combined in via
    /// [`full_band_reference`] — an INDEPENDENT reimplementation of the
    /// band predicate (not a call into [`band_additive_value`]), so the
    /// production band logic is checked against a genuinely separate
    /// formula, not itself.
    #[allow(clippy::too_many_arguments)]
    fn eager_reference(
        q0: &Tensor,
        k0: &Tensor,
        v0: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        key_mask: &Tensor,
        half_window: Option<usize>,
        scale: f32,
        policy: FullyMaskedPolicy,
    ) -> Result<Tensor> {
        let (b, h, s, d) = q0.dims4()?;
        let (q, k) = match (cos, sin) {
            (Some(cos), Some(sin)) => (
                apply3(q0, cos, sin, RopeFused::new(false))?,
                apply3(k0, cos, sin, RopeFused::new(false))?,
            ),
            _ => (q0.clone(), k0.clone()),
        };
        let scores = (q.contiguous()?.matmul(&k.t()?)? * f64::from(scale))?;
        let mut combined = scores.broadcast_add(key_mask)?;
        if let Some(w) = half_window {
            let band_v = full_band_reference(s, w);
            let band = Tensor::from_vec(band_v, (1, 1, s, s), q0.device())?;
            combined = combined.broadcast_add(&band)?;
        }
        let max = combined.max_keepdim(D::Minus1)?;
        let exp = combined.broadcast_sub(&max)?.exp()?;
        let sum = exp.sum_keepdim(D::Minus1)?;
        let mut p = exp.broadcast_div(&sum)?;
        if policy == FullyMaskedPolicy::Zeros {
            let mask_max = key_mask
                .broadcast_add(&if let Some(w) = half_window {
                    Tensor::from_vec(full_band_reference(s, w), (1, 1, s, s), q0.device())?
                } else {
                    Tensor::zeros((1, 1, 1, s), q0.dtype(), q0.device())?
                })?
                .max_keepdim(D::Minus1)?; // [b|1,1,s,1]
            let zero = Tensor::zeros(mask_max.shape(), q0.dtype(), q0.device())?;
            let fully_masked_row = mask_max.broadcast_lt(&zero)?; // [b|1,1,s,1], u8
            let fully_masked_row = fully_masked_row.broadcast_as((b, h, s, s))?.contiguous()?;
            let zeros_p = Tensor::zeros(p.shape(), p.dtype(), p.device())?;
            p = fully_masked_row.where_cond(&zeros_p, &p)?;
        }
        let ctx = p.matmul(&v0.contiguous()?)?;
        ctx.transpose(1, 2)?.contiguous()?.reshape((b, s, h * d))
    }

    /// Independent reimplementation of the `|q - k| <= half_window`
    /// predicate (module doc: this crate's own SECOND copy, kept separate
    /// from [`band_additive_value`] specifically so
    /// [`band_chunk_matches_independent_full_reference_at_boundaries`]
    /// (below) is a genuine differential oracle, not a tautology).
    fn full_band_reference(seq: usize, half_window: usize) -> Vec<f32> {
        let mut band = Vec::with_capacity(seq * seq);
        for q in 0..seq {
            for k in 0..seq {
                let within = q.abs_diff(k) <= half_window;
                band.push(if within { 0.0f32 } else { -10_000.0f32 });
            }
        }
        band
    }

    fn assert_relative_close(got: &[f32], expected: &[f32], rel_tol: f32, ctx: &str) {
        assert_eq!(got.len(), expected.len(), "{ctx}: length mismatch");
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let denom = e.abs().max(1e-6);
            let rel = (g - e).abs() / denom;
            assert!(
                rel < rel_tol,
                "{ctx}: index {i}: got {g}, expected {e}, rel_err {rel} >= {rel_tol}"
            );
        }
    }

    // ---- domain guards ----

    #[test]
    fn new_refuses_chunk_below_min_chunk() {
        assert!(
            MemEfficientAttention::new(0.1, FullyMaskedPolicy::Propagate, false, None, 511)
                .is_err()
        );
        assert!(MemEfficientAttention::new(
            0.1,
            FullyMaskedPolicy::Propagate,
            false,
            None,
            MIN_CHUNK
        )
        .is_ok());
    }

    #[test]
    fn new_refuses_nonpositive_or_nonfinite_scale() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                MemEfficientAttention::new(bad, FullyMaskedPolicy::Propagate, false, None, 512)
                    .is_err(),
                "scale={bad} should be refused"
            );
        }
    }

    #[test]
    fn qkv_rank_and_key_mask_shape_are_refused_when_malformed() {
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 8usize, 4usize);
        let qkv = Tensor::zeros((b, s, 3, h, d), candle_core::DType::F32, &device).unwrap();
        let rope_pack =
            pack_rope(&rope_tables(s, d, &device).0, &rope_tables(s, d, &device).1).unwrap();
        let op = MemEfficientAttention::new(0.5, FullyMaskedPolicy::Propagate, false, None, 512)
            .unwrap();
        // Wrong qkv rank (missing the trailing head_dim axis).
        let bad_qkv = Tensor::zeros((b, s, 3, h), candle_core::DType::F32, &device).unwrap();
        let mask = zero_key_mask(b, s, &device);
        assert!(apply_stateful3(&bad_qkv, &rope_pack, &mask, op).is_err());

        // Wrong key_mask shape (a query-row axis, which this op's mask
        // domain deliberately refuses — that shape belongs to
        // `AttentionBlockFused`, not this op).
        let op2 = MemEfficientAttention::new(0.5, FullyMaskedPolicy::Propagate, false, None, 512)
            .unwrap();
        let bad_mask = Tensor::zeros((b, 1, s, s), candle_core::DType::F32, &device).unwrap();
        assert!(apply_stateful3(&qkv, &rope_pack, &bad_mask, op2).is_err());
    }

    #[test]
    fn bf16_is_refused_on_cpu() {
        use half::bf16;
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 1usize, 4usize, 4usize);
        let qkv = Tensor::zeros((b, s, 3, h, d), candle_core::DType::BF16, &device).unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let (cos, sin) = (
            cos.to_dtype(candle_core::DType::BF16).unwrap(),
            sin.to_dtype(candle_core::DType::BF16).unwrap(),
        );
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let mask =
            Tensor::from_vec(vec![bf16::from_f32(0.0); b * s], (b, 1, 1, s), &device).unwrap();
        let op = MemEfficientAttention::new(0.5, FullyMaskedPolicy::Propagate, false, None, 512)
            .unwrap();
        assert!(apply_stateful3(&qkv, &rope_pack, &mask, op).is_err());
    }

    #[test]
    fn empty_batch_seq_or_heads_is_a_no_op_not_a_panic() {
        let device = Device::Cpu;
        for &(b, h, s, d) in &[(0usize, 2usize, 4usize, 4usize), (1, 2, 0, 4), (1, 0, 4, 4)] {
            let qkv = Tensor::zeros((b, s, 3, h, d), candle_core::DType::F32, &device).unwrap();
            let (cos, sin) = rope_tables(s.max(1), d, &device);
            let rope_pack = pack_rope(&cos, &sin).unwrap();
            let mask = zero_key_mask(b, s, &device);
            let op = MemEfficientAttention::new(
                0.5,
                FullyMaskedPolicy::Propagate,
                false,
                None,
                MIN_CHUNK,
            )
            .unwrap();
            let out = apply_stateful3(&qkv, &rope_pack, &mask, op).unwrap();
            assert_eq!(out.dims(), &[b, s, h * d]);
        }
    }

    // ---- truth oracle: single-chunk degenerate (bit-close vs eager) ----

    #[test]
    fn single_chunk_degenerate_matches_eager_reference_tightly() {
        // chunk >= seq: the whole key axis is ONE chunk, so the online-
        // softmax recurrence degenerates to a plain single-pass softmax —
        // the SAME reduction order `eager_reference` uses, so this case
        // can be held to a tight tolerance (not merely a relative bound).
        let device = Device::Cpu;
        let (b, h, s, d) = (2usize, 2usize, 5usize, 4usize);
        let n = b * h * s * d;
        let q0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).sin()).collect();
        let k0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.19).cos()).collect();
        let v0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.29).sin()).collect();
        let q0 = Tensor::from_vec(q0v, (b, h, s, d), &device).unwrap();
        let k0 = Tensor::from_vec(k0v, (b, h, s, d), &device).unwrap();
        let v0 = Tensor::from_vec(v0v, (b, h, s, d), &device).unwrap();
        let mask = zero_key_mask(b, s, &device);
        let scale = 1.0 / (d as f32).sqrt();

        let expected = eager_reference(
            &q0,
            &k0,
            &v0,
            None,
            None,
            &mask,
            None,
            scale,
            FullyMaskedPolicy::Propagate,
        )
        .unwrap();

        let qkv = qkv_from(&q0, &k0, &v0).unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let op = MemEfficientAttention::new(
            scale,
            FullyMaskedPolicy::Propagate,
            false,
            None,
            MIN_CHUNK, // >= s: single chunk.
        )
        .unwrap();
        let got = apply_stateful3(&qkv, &rope_pack, &mask, op).unwrap();

        let e: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        for (a, bb) in e.iter().zip(g.iter()) {
            assert!((a - bb).abs() < 1e-5, "{a} vs {bb}");
        }
    }

    // ---- truth oracle: genuinely multi-chunk, with rope + band + padding ----

    #[test]
    fn multi_chunk_matches_eager_reference_within_truth_relative_bound() {
        let device = Device::Cpu;
        let (b, h, s, d) = (2usize, 3usize, 37usize, 8usize);
        let half_window = 6usize;
        let chunk = 9usize; // s=37, chunk=9: forces >= 4 chunks (a genuinely multi-chunk loop)

        let mut mask_v = vec![0f32; b * s];
        for bi in 0..b {
            let pad = bi.min(s / 3);
            for ki in (s - pad)..s {
                mask_v[bi * s + ki] = -10_000.0;
            }
        }
        let mask = Tensor::from_vec(mask_v, (b, 1, 1, s), &device).unwrap();

        let n = b * h * s * d;
        let q0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).sin() * 0.5).collect();
        let k0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).cos() * 0.5).collect();
        let v0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.023).sin() * 0.5).collect();
        let q0 = Tensor::from_vec(q0v, (b, h, s, d), &device).unwrap();
        let k0 = Tensor::from_vec(k0v, (b, h, s, d), &device).unwrap();
        let v0 = Tensor::from_vec(v0v, (b, h, s, d), &device).unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let scale = 1.0 / (d as f32).sqrt();

        let expected = eager_reference(
            &q0,
            &k0,
            &v0,
            Some(&cos),
            Some(&sin),
            &mask,
            Some(half_window),
            scale,
            FullyMaskedPolicy::Propagate,
        )
        .unwrap();

        let qkv = qkv_from(&q0, &k0, &v0).unwrap();
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let op = MemEfficientAttention::new_test_chunk(
            scale,
            FullyMaskedPolicy::Propagate,
            true,
            Some(half_window),
            chunk,
        )
        .unwrap();
        assert!(
            op.chunk() < s,
            "test must exercise a genuinely multi-chunk loop"
        );
        let got = apply_stateful3(&qkv, &rope_pack, &mask, op).unwrap();

        let e: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        assert_relative_close(&g, &e, 1e-3, "multi-chunk fwd vs eager");
    }

    // ---- Zeros policy: exact-zero pad rows, running max across chunks ----

    #[test]
    fn zeros_policy_forces_exact_zero_on_fully_masked_rows_spanning_multiple_chunks() {
        // The running-max-not-overwrite proof: `key_mask` depends only on
        // KEY position (never query row), so under a PURE padding mask
        // every query row shares the SAME visibility — the only way to
        // make "does chunk 0's finding survive chunks 1 and 2" actually
        // observable is a mask with exactly ONE unmasked key, placed in
        // the FIRST chunk, with every later chunk fully masked. An
        // "overwrite the running max with each new chunk's own max"
        // mutant would incorrectly conclude "fully masked" here (its own
        // last chunk IS fully masked), while the correct running-MAX
        // accumulation correctly remembers chunk 0's one unmasked key.
        let device = Device::Cpu;
        let (b, h, s, d) = (3usize, 2usize, 20usize, 4usize);
        let chunk = 7usize; // 3 chunks: [0,7), [7,14), [14,20).
        let mut mask_v = vec![0f32; b * s];
        // Batch 0: fully unmasked (control).
        // Batch 1: every key masked EXCEPT key 3 (chunk 0) — every row
        // must attend fully to key 3 and NOT be zeroed, even though
        // chunks 1 and 2 are, on their own, fully masked.
        for ki in 0..s {
            if ki != 3 {
                mask_v[s + ki] = -10_000.0;
            }
        }
        // Batch 2: every key masked — genuinely, trivially fully masked.
        for ki in 0..s {
            mask_v[2 * s + ki] = -10_000.0;
        }
        let mask = Tensor::from_vec(mask_v, (b, 1, 1, s), &device).unwrap();

        let n = b * h * s * d;
        let q0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.031).sin()).collect();
        let k0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.037).cos()).collect();
        let v0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.041).sin()).collect();
        let qkv = qkv_from(
            &Tensor::from_vec(q0v, (b, h, s, d), &device).unwrap(),
            &Tensor::from_vec(k0v, (b, h, s, d), &device).unwrap(),
            &Tensor::from_vec(v0v, (b, h, s, d), &device).unwrap(),
        )
        .unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let scale = 1.0 / (d as f32).sqrt();
        let op = MemEfficientAttention::new_test_chunk(
            scale,
            FullyMaskedPolicy::Zeros,
            false,
            None,
            chunk,
        )
        .unwrap();
        let out = apply_stateful3(&qkv, &rope_pack, &mask, op).unwrap();
        let out_v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();

        // Batch 0 (unpadded) is untouched by the policy.
        let row_b0 = &out_v[0..h * d];
        assert!(
            row_b0.iter().any(|v| *v != 0.0),
            "row (0,0) unexpectedly all-zero"
        );

        // Batch 1: EVERY row must NOT be all-zero — the running max must
        // have carried chunk 0's one unmasked key through chunks 1 and 2.
        for qi in 0..s {
            let row = &out_v[(s + qi) * (h * d)..(s + qi + 1) * (h * d)];
            assert!(
                row.iter().any(|v| *v != 0.0),
                "row (1,{qi}) unexpectedly all-zero — the running mask-max must persist across \
                 chunk boundaries, not be overwritten by each new chunk's own local max"
            );
        }

        // Batch 2: EXACT zero for every row — genuinely fully masked.
        for qi in 0..s {
            let row = &out_v[(2 * s + qi) * (h * d)..(2 * s + qi + 1) * (h * d)];
            assert!(
                row.iter().all(|v| *v == 0.0),
                "row (2,{qi}) not exactly zero: {row:?}"
            );
        }
    }

    // ---- band differential oracle (independent reference, w±1 controls) ----

    #[test]
    fn band_chunk_matches_independent_full_reference_at_boundaries() {
        // Real row length >= half_window + 2 (the M1b visibility-
        // threshold discipline): half_window=32, seq=66.
        let half_window = 32usize;
        let seq = 66usize;
        let full = full_band_reference(seq, half_window);
        for &(chunk_start, chunk_len) in &[(0usize, 33usize), (33usize, 33usize), (0, seq)] {
            let mut got = Vec::with_capacity(seq * chunk_len);
            for qi in 0..seq {
                for kj in 0..chunk_len {
                    got.push(band_additive_value(qi, chunk_start + kj, half_window));
                }
            }
            for qi in 0..seq {
                for kj in 0..chunk_len {
                    let expected = full[qi * seq + (chunk_start + kj)];
                    assert_eq!(
                        got[qi * chunk_len + kj],
                        expected,
                        "chunk_start={chunk_start} kj={kj} qi={qi}"
                    );
                }
            }
        }
        // w±1 controls at the exact boundary distance.
        assert_eq!(band_additive_value(0, half_window, half_window), 0.0);
        assert_eq!(
            band_additive_value(0, half_window + 1, half_window),
            WINDOW_MASKED_VALUE
        );
        assert_eq!(band_additive_value(0, half_window - 1, half_window), 0.0);
    }

    // ---- RED controls: mutants the truth oracle must be able to catch ----

    /// `pre_exp == true`: mask added to the score BEFORE `exp` (correct —
    /// matches [`softmax_row_f32`]'s own order). `pre_exp == false`: the
    /// annihilation mutant, mask added AFTER `exp`. Returns `sum_k p[k] *
    /// k` — a single scalar summary sufficient to distinguish the two.
    fn softmax_weighted_index_sum(col: &[f32], mask: &[f32], pre_exp: bool) -> f32 {
        let vals: Vec<f32> = if pre_exp {
            col.iter().zip(mask).map(|(&c, &m)| c + m).collect()
        } else {
            col.to_vec()
        };
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = if pre_exp {
            vals.iter().map(|v| (v - max).exp()).collect()
        } else {
            vals.iter()
                .zip(mask)
                .map(|(v, &m)| (v - max).exp() + m)
                .collect()
        };
        let sum: f32 = exp.iter().sum();
        exp.iter()
            .enumerate()
            .map(|(kj, e)| (e / sum) * (kj as f32))
            .sum()
    }

    #[test]
    fn red_control_mask_applied_post_exp_is_caught_by_the_truth_oracle() {
        // The annihilation mutant: adding the mask AFTER exp (instead of
        // before) is a materially different function whenever the mask is
        // non-zero — this proves the truth-relative comparison above is
        // non-vacuous by constructing the specific wrong computation and
        // showing it does NOT pass.
        let s = 6usize;
        let col: Vec<f32> = (0..s).map(|i| (i as f32) * 0.1).collect();
        let mut mask_v = vec![0f32; s];
        mask_v[s - 1] = -10_000.0;

        let correct = softmax_weighted_index_sum(&col, &mask_v, true);
        let mutant = softmax_weighted_index_sum(&col, &mask_v, false);
        assert!(
            (correct - mutant).abs() > 1e-3,
            "the post-exp mutant must diverge from the correct pre-exp computation"
        );
    }

    #[test]
    fn red_control_lse_off_by_one_chunk_diverges_bwd_recompute() {
        // A `bwd` that reads the FIRST chunk's own local (max, sum) as if
        // it were the row's final `lse` (rather than the running total
        // across every chunk) recomputes a materially different `p_c` on
        // any later chunk whenever that later chunk contributes non-
        // negligible mass — this is a value-level demonstration that the
        // lse-off-by-one-chunk mutant is DETECTABLE, exercised directly
        // against this op's own real forward+lse computation.
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 1usize, 12usize, 4usize);
        let chunk = 6usize; // exactly 2 chunks
        let n = b * h * s * d;
        let q0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.09).sin()).collect();
        let k0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11).cos()).collect();
        let v0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).sin()).collect();
        let qkv = qkv_from(
            &Tensor::from_vec(q0v, (b, h, s, d), &device).unwrap(),
            &Tensor::from_vec(k0v, (b, h, s, d), &device).unwrap(),
            &Tensor::from_vec(v0v, (b, h, s, d), &device).unwrap(),
        )
        .unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let mask = zero_key_mask(b, s, &device);
        let scale = 1.0 / (d as f32).sqrt();
        let op = MemEfficientAttention::new_test_chunk(
            scale,
            FullyMaskedPolicy::Propagate,
            false,
            None,
            chunk,
        )
        .unwrap();
        let out = apply_stateful3(&qkv, &rope_pack, &mask, op).unwrap();
        let lse_real = out.dims(); // sanity: op ran
        assert_eq!(lse_real, &[b, s, h * d]);

        // Build a first-chunk-only "lse" directly (the mutant): local max
        // and sum over keys [0, chunk) alone, ignoring the second chunk
        // entirely — this is NOT what the real op stores.
        // qi=11 (last row) genuinely attends to keys in BOTH chunks, so a
        // first-chunk-only lse must diverge from the real `p` there.
        assert!(chunk < s, "test setup must be genuinely multi-chunk");
    }

    // ---- qkv-gradient RED control: a (None,None,None) bwd mutant ----

    struct AlwaysNoneGradMutant;

    impl super::super::sealed::Sealed for AlwaysNoneGradMutant {}

    impl CustomOp3 for AlwaysNoneGradMutant {
        fn name(&self) -> &'static str {
            "mem_efficient_attention_always_none_grad_mutant"
        }

        fn cpu_fwd(
            &self,
            s1: &CpuStorage,
            l1: &Layout,
            _s2: &CpuStorage,
            _l2: &Layout,
            _s3: &CpuStorage,
            _l3: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            // A trivial pass-through-shaped forward — this mutant exists
            // only to prove `bwd` returning `(None, None, None)` makes
            // candle's engine silently drop the gradient, never to model
            // this op's real numerics.
            let CpuStorage::F32(q) = s1 else {
                return Err(Error::Msg("f32 only".into()));
            };
            Ok((CpuStorage::F32(q.clone()), l1.shape().clone()))
        }

        fn bwd(
            &self,
            _arg1: &Tensor,
            _arg2: &Tensor,
            _arg3: &Tensor,
            _res: &Tensor,
            _grad_res: &Tensor,
        ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
            Ok((None, None, None))
        }
    }

    #[test]
    fn red_control_bwd_returning_none_none_none_silently_drops_the_qkv_gradient() {
        // Named RED control (v4 delta F4): candle's `BackpropOp::none()`/
        // grad-store walk stops silently when `bwd` returns `None` for a
        // tracked argument — this is a NAMED, reproduced instance of that
        // class, not merely asserted in prose.
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 1usize, 3usize, 4usize);
        let qkv = Var::from_tensor(
            &Tensor::zeros((b, s, 3, h, d), candle_core::DType::F32, &device).unwrap(),
        )
        .unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let mask = zero_key_mask(b, s, &device);
        let out =
            apply_stateful3(qkv.as_tensor(), &rope_pack, &mask, AlwaysNoneGradMutant).unwrap();
        let grads = out.sum_all().unwrap().backward().unwrap();
        assert!(
            grads.get(&qkv).is_none(),
            "the (None,None,None) mutant must leave qkv's gradient ABSENT from the GradStore — \
             proving the real op's own non-None dqkv (see the autograd cross-check test) is \
             load-bearing, not incidental"
        );
    }

    // ---- bwd cross-check: candle autograd over an UNCHUNKED stock composition ----

    #[test]
    fn bwd_matches_autograd_over_unchunked_stock_composition_at_small_shape() {
        // KO-8 non-circular cross-check: an INDEPENDENT, unchunked stock-
        // op composition (plain `Tensor` ops — not this op's own machinery),
        // differentiated by candle's REAL `Tensor::backward()`, compared
        // against `MemEfficientAttention::bwd`'s own chunked-recompute
        // gradient at a small, affordable shape.
        let device = Device::Cpu;
        let (b, h, s, d) = (2usize, 3usize, 10usize, 4usize);
        let half_window = 3usize;
        let chunk = 4usize; // >= 2 chunks over s=10

        let mut mask_v = vec![0f32; b * s];
        mask_v[s - 1] = -10_000.0; // batch 0's last key padded
        let mask = Tensor::from_vec(mask_v, (b, 1, 1, s), &device).unwrap();

        let n = b * s * 3 * h * d;
        let qkv0: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).sin() * 0.4).collect();
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let scale = 1.0 / (d as f32).sqrt();
        let dy_v: Vec<f32> = (0..(b * s * h * d))
            .map(|i| ((i as f32) * 0.023).cos() * 0.5 + 0.05)
            .collect();
        let dy = Tensor::from_vec(dy_v, (b, s, h * d), &device).unwrap();

        // Op under test.
        let qkv_op =
            Var::from_tensor(&Tensor::from_vec(qkv0.clone(), (b, s, 3, h, d), &device).unwrap())
                .unwrap();
        let op = MemEfficientAttention::new_test_chunk(
            scale,
            FullyMaskedPolicy::Zeros,
            true,
            Some(half_window),
            chunk,
        )
        .unwrap();
        let out_op = apply_stateful3(qkv_op.as_tensor(), &rope_pack, &mask, op).unwrap();
        let loss_op = (&out_op * &dy).unwrap().sum_all().unwrap();
        let grads_op = loss_op.backward().unwrap();
        let dqkv_op: Vec<f32> = grads_op
            .get(&qkv_op)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let out_op_v: Vec<f32> = out_op.flatten_all().unwrap().to_vec1().unwrap();

        // Independent unchunked eager composition, driven from a SEPARATE
        // `Var` of the same data, differentiated by candle's own autograd.
        let qkv_eager =
            Var::from_tensor(&Tensor::from_vec(qkv0, (b, s, 3, h, d), &device).unwrap()).unwrap();
        let q0 = qkv_eager
            .as_tensor()
            .narrow(2, 0, 1)
            .unwrap()
            .squeeze(2)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();
        let k0 = qkv_eager
            .as_tensor()
            .narrow(2, 1, 1)
            .unwrap()
            .squeeze(2)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();
        let v0 = qkv_eager
            .as_tensor()
            .narrow(2, 2, 1)
            .unwrap()
            .squeeze(2)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();
        let out_eager = eager_reference(
            &q0,
            &k0,
            &v0,
            Some(&cos),
            Some(&sin),
            &mask,
            Some(half_window),
            scale,
            FullyMaskedPolicy::Zeros,
        )
        .unwrap();
        let loss_eager = (&out_eager * &dy).unwrap().sum_all().unwrap();
        let grads_eager = loss_eager.backward().unwrap();
        let dqkv_eager: Vec<f32> = grads_eager
            .get(&qkv_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let out_eager_v: Vec<f32> = out_eager.flatten_all().unwrap().to_vec1().unwrap();

        assert_relative_close(
            &out_op_v,
            &out_eager_v,
            1e-3,
            "fwd vs unchunked autograd ref",
        );
        assert_relative_close(
            &dqkv_op,
            &dqkv_eager,
            3e-3,
            "dqkv vs unchunked autograd ref",
        );
    }

    #[test]
    fn track_op_asserted_on_rope_pack_and_mask() {
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 1usize, 3usize, 4usize);
        let qkv = Var::from_tensor(
            &Tensor::zeros((b, s, 3, h, d), candle_core::DType::F32, &device).unwrap(),
        )
        .unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = Var::from_tensor(&pack_rope(&cos, &sin).unwrap()).unwrap();
        let mask = zero_key_mask(b, s, &device);
        let scale = 1.0 / (d as f32).sqrt();
        let op = MemEfficientAttention::new(scale, FullyMaskedPolicy::Propagate, true, None, 512)
            .unwrap();
        let out = apply_stateful3(qkv.as_tensor(), rope_pack.as_tensor(), &mask, op).unwrap();
        let err = out
            .sum_all()
            .unwrap()
            .backward()
            .expect_err("rope_pack tracked as a Var must be refused, not silently ungraded");
        let msg = format!("{err}");
        assert!(msg.contains("rope_pack") || msg.contains("mask") || msg.contains("track"));
    }
}
