//! FlashAttention-2 varlen training op — TWO crate-private ops composing
//! [`crate::flash`]'s FFI boundary, reachable ONLY through the public
//! [`flash_attention_varlen`]. Feature `flash-attn`.
//!
//! # Two op types, one seam
//!
//! [`FlashVarlenAttention`] (`CustomOp1`, forward) takes exactly one
//! differentiable `Tensor` — the packed `qkv` — matching the vendored
//! FFI's own contract 1:1 (`crate::flash`'s module doc: ONE packed `qkv`
//! buffer in, ONE packed `d_qkv` buffer out — `flash_attn_varlen_qkvpacked_func`'s
//! upstream layout, not the separate-q/k/v `flash_attn_varlen_func`).
//! Compare [`crate::ops::AttentionBlockFused`] (this crate's OTHER
//! packed-`qkv` attention op): its `CustomOp3` arity is `(qkv, rope_pack,
//! mask)` — `qkv` is still ONE argument there too, split into Q/K/V
//! INTERNALLY (`crate::cuda::attention_block::slot_view`); this op has no
//! `rope_pack` (RoPE is the CALLER's job, see "Domain" below) or `mask`
//! tensor (`cu_seqlens` + `window` carry the same information, but as a
//! typed handle and a config — not `Tensor`s), so it has no third or
//! second argument to be a `CustomOp3` OVER.
//!
//! Its BACKWARD is a different shape: [`FlashVarlenBwdHelper`] (`CustomOp3`,
//! over `qkv`/`o`/`d_o` — the three tensors candle's `CustomOp1::bwd`
//! already hands the outer op as `(arg, res, grad_res)`) constructs the
//! returned `d_qkv` `Tensor` through candle's own `apply_op3` machinery —
//! the same "recompute via composing an INNER op" idiom
//! `LayerNormFused::bwd`'s internal `CustomOp3` dx/dgamma helpers already
//! use, rather than hand-assembling a `Tensor` from raw `Storage` (which
//! would sidestep candle's own bookkeeping for no reason). The forward's
//! `lse` — needed by the vendored backward, with no channel in
//! `CustomOp1::bwd`'s signature (`res` is `o`, bf16, a DIFFERENT
//! shape/dtype) — moves from [`FlashVarlenAttention`]'s own [`Saved`] slot
//! to [`FlashVarlenBwdHelper`]'s OWN, separate `Saved` slot at
//! construction (`take()` then `set()`, both typed, see [`crate::ops::saved`]'s
//! module doc): a `Saved<T>` SPANS two op types here, never a raw field
//! smuggled between them outside that channel.
//!
//! # Why `StatefulKernelOp`, not `KernelOp`, for BOTH op types
//!
//! Both hold a `Saved` field (interior-mutable), so neither can be `Copy`
//! — see [`crate::ops::StatefulKernelOp`]'s own doc for why `Clone` is
//! refused too, and why that makes hoisting either op into a long-lived
//! field a COMPILE ERROR.
//!
//! # Domain
//!
//! - CUDA, compute capability EXACTLY the build's own (`crate::flash::
//!   check_arch`, reached via every `crate::flash::flash_varlen_*`
//!   call) — no CPU arm on either op type (`cpu_fwd` refuses
//!   unconditionally; `crate::ops::AttentionBlockFused` is the CPU/eager
//!   arm).
//! - `qkv`: bf16, contiguous, rank 4, shape `[total_q, 3, H, 64]`
//!   (`crate::flash::HEAD_DIM`).
//! - `cu_seqlens`/`cfg`: validated by `crate::flash` itself (every
//!   sequence length `>= 1`, `softmax_scale` finite `> 0`, window radius
//!   fits `i32`, `total_q` in `qkv` matches `cu_seqlens`' own `total_q`
//!   — checked here, since the FFI has no way to know `qkv`'s shape
//!   disagreed with the geometry it was handed).
//! - RoPE: NOT applied here. The caller (Stage B2's encoder) rotates Q/K
//!   and packs `qkv` BEFORE calling [`flash_attention_varlen`] — see the
//!   P6 Stage B contract §3.6.
//! - `deterministic`: `cfg.deterministic` is whatever the caller passes;
//!   Stage B2 pins it `true` at the call site (this op itself has no
//!   opinion — it is a generic primitive, family L).

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::CudaSlice;
use candle_core::{
    CpuStorage, CudaStorage, CustomOp1, CustomOp3, DType, Device, Error, Layout, Result, Shape,
    Tensor,
};
use half::bf16;

use crate::flash::{self, CuSeqlens, VarlenConfig, HEAD_DIM};
use crate::ops::rope_positions::PositionArm;
use crate::ops::saved::Saved;

const OP_NAME: &str = "flash_attention_varlen";
const BWD_OP_NAME: &str = "flash_attention_varlen_bwd";

fn flash_err(e: flash::FlashError) -> Error {
    Error::Msg(format!("{OP_NAME}: {e}"))
}

fn saved_err(who: &'static str, e: crate::ops::saved::SavedError) -> Error {
    Error::Msg(format!("{who}: {e}"))
}

/// `qkv`'s expected packed shape: rank 4, `[total_q, 3, H, HEAD_DIM]`,
/// bf16. Returns `(total_q, num_heads)`. Shared by both ops' `cuda_fwd`.
fn check_qkv_domain(dims: &[usize], dtype: DType) -> Result<(usize, usize)> {
    if dims.len() != 4 || dims[1] != 3 || dims[3] != HEAD_DIM {
        return Err(Error::Msg(format!(
            "{OP_NAME}: qkv must be rank-4 [total_q, 3, H, {HEAD_DIM}], got {dims:?}"
        )));
    }
    if dtype != DType::BF16 {
        return Err(Error::UnsupportedDTypeForOp(dtype, OP_NAME));
    }
    Ok((dims[0], dims[2]))
}

/// `o`/`d_o`'s expected shape: rank 3, `[total_q, H, HEAD_DIM]`, bf16 —
/// the forward's own output shape (`crate::flash`'s module doc table).
fn check_o_domain(dims: &[usize], dtype: DType, total_q: usize, num_heads: usize) -> Result<()> {
    if dims.len() != 3 || dims[0] != total_q || dims[1] != num_heads || dims[2] != HEAD_DIM {
        return Err(Error::Msg(format!(
            "{OP_NAME}: expected o/grad_o shape [{total_q}, {num_heads}, {HEAD_DIM}], got {dims:?}"
        )));
    }
    if dtype != DType::BF16 {
        return Err(Error::UnsupportedDTypeForOp(dtype, OP_NAME));
    }
    Ok(())
}

/// One `cu_seqlens.geometry(num_heads)`, cross-checked against the
/// `Tensor`'s own `total_q` — the FFI trusts `CuSeqlens`' geometry
/// entirely (by design, see `crate::flash`'s module doc: it never reads
/// device memory to verify), so THIS is the one place a caller's `qkv`
/// and `cu_seqlens` disagreeing on `total_q` gets caught before a launch.
fn geometry_for(
    cu_seqlens: &CuSeqlens,
    num_heads: usize,
    total_q: usize,
) -> Result<flash::VarlenGeometry> {
    let geom = cu_seqlens.geometry(num_heads).map_err(flash_err)?;
    if geom.total_q() != total_q {
        return Err(Error::Msg(format!(
            "{OP_NAME}: qkv total_q={total_q} disagrees with cu_seqlens' total_q={} — both must \
             be derived from the SAME host lengths",
            geom.total_q()
        )));
    }
    Ok(geom)
}

/// The forward op. Crate-private BY CONSTRUCTION: no `pub` path to this
/// type exists outside this module (`flash_attention_varlen` is the only
/// constructor, and it never returns the op itself — only the `Tensor`
/// `apply_stateful1` produces).
pub(crate) struct FlashVarlenAttention {
    cu_seqlens: CuSeqlens,
    num_heads: usize,
    cfg: VarlenConfig,
    lse: Saved<CudaSlice<f32>>,
}

impl super::sealed::Sealed for FlashVarlenAttention {}

impl CustomOp1 for FlashVarlenAttention {
    fn name(&self) -> &'static str {
        OP_NAME
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        Err(Error::Msg(format!(
            "{OP_NAME}: no CPU arm — this op requires CUDA at EXACTLY the compute capability \
             this crate's cubin was built for (crate::flash::check_arch); \
             crate::ops::AttentionBlockFused is the CPU/eager arm"
        )))
    }

    fn cuda_fwd(&self, s1: &CudaStorage, l1: &Layout) -> Result<(CudaStorage, Shape)> {
        let (total_q, num_heads) = check_qkv_domain(l1.dims(), s1.dtype())?;
        if num_heads != self.num_heads {
            return Err(Error::Msg(format!(
                "{OP_NAME}: qkv has {num_heads} heads but this op was constructed for {} — the \
                 caller must build cu_seqlens' geometry with the SAME head count qkv carries",
                self.num_heads
            )));
        }
        let device = s1.device().clone();
        let geom = geometry_for(&self.cu_seqlens, num_heads, total_q)?;

        let (x1, x2) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: OP_NAME })?;
        let qkv_view = s1.as_cuda_slice::<bf16>()?.slice(x1..x2);

        // SAFETY: uninitialised outputs `flash_varlen_fwd_into` fully
        // overwrites — the identical allocation `flash::flash_varlen_fwd`
        // (its owning convenience wrapper) makes for the same reason.
        let mut o = unsafe { device.alloc::<bf16>(geom.o_len()) }?;
        let mut lse = unsafe { device.alloc::<f32>(geom.lse_len()) }?;
        flash::flash_varlen_fwd_into(
            &device,
            qkv_view,
            &self.cu_seqlens,
            o.as_view_mut(),
            lse.as_view_mut(),
            num_heads,
            &self.cfg,
        )
        .map_err(flash_err)?;

        // Stash `lse` for `bwd` — see the module doc's "Two op types, one
        // seam" section. `set()` cannot legitimately fail here: this
        // instance's `cuda_fwd` runs at most once (fresh instance per
        // `apply_stateful1` call — see `StatefulKernelOp`'s own doc) — an
        // `Err` here would mean candle called `cuda_fwd` twice on ONE
        // instance, a candle-internal contract violation this op cannot
        // recover from silently, so it is propagated rather than
        // `.expect()`-panicked.
        self.lse.set(lse).map_err(|e| saved_err(OP_NAME, e))?;

        let out_shape = Shape::from((total_q, num_heads, HEAD_DIM));
        Ok((CudaStorage::wrap_cuda_slice(o, device), out_shape))
    }

    fn bwd(&self, arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let lse = self.lse.take().map_err(|e| saved_err(OP_NAME, e))?;
        let helper = FlashVarlenBwdHelper {
            cu_seqlens: self.cu_seqlens.try_duplicate().map_err(flash_err)?,
            num_heads: self.num_heads,
            cfg: self.cfg,
            lse: Saved::empty(),
        };
        helper.lse.set(lse).map_err(|e| saved_err(BWD_OP_NAME, e))?;
        // DETACHED (mirrors `crate::ops::attention_block`'s identical "bwd
        // runs DETACHED" precedent): `arg` is a tracked graph node in
        // production (downstream of the `Wqkv` LoRA `Var`s); without this,
        // `apply_stateful3` would attach a NEW `BackpropOp` cloning `arg`/
        // `res`/`grad_res` as its own inputs, growing the graph pointlessly
        // — this call's whole purpose is to PRODUCE a gradient value, not
        // to be differentiated through a second time.
        let arg_d = arg.detach();
        let res_d = res.detach();
        let grad_d = grad_res.detach();
        let d_qkv = super::apply_stateful3(&arg_d, &res_d, &grad_d, helper)?;
        Ok(Some(d_qkv))
    }
}

/// The backward helper. See the module doc's "Two op types, one seam"
/// section. Crate-private, constructed ONLY inside [`FlashVarlenAttention::bwd`]
/// (above) — never independently, never by any caller of
/// [`flash_attention_varlen`].
struct FlashVarlenBwdHelper {
    cu_seqlens: CuSeqlens,
    num_heads: usize,
    cfg: VarlenConfig,
    lse: Saved<CudaSlice<f32>>,
}

impl super::sealed::Sealed for FlashVarlenBwdHelper {}

impl CustomOp3 for FlashVarlenBwdHelper {
    fn name(&self) -> &'static str {
        BWD_OP_NAME
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(Error::Msg(format!(
            "{BWD_OP_NAME}: no CPU arm — see {OP_NAME}'s own doc"
        )))
    }

    #[allow(clippy::too_many_arguments)]
    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
        s3: &CudaStorage,
        l3: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        // s1/l1 = qkv, s2/l2 = o, s3/l3 = d_o — exactly the three tensors
        // `FlashVarlenAttention::bwd` passed to `apply_stateful3`.
        let (total_q, num_heads) = check_qkv_domain(l1.dims(), s1.dtype())?;
        if num_heads != self.num_heads {
            return Err(Error::Msg(format!(
                "{BWD_OP_NAME}: qkv has {num_heads} heads but this op was constructed for {}",
                self.num_heads
            )));
        }
        check_o_domain(l2.dims(), s2.dtype(), total_q, num_heads)?;
        check_o_domain(l3.dims(), s3.dtype(), total_q, num_heads)?;
        let device = s1.device().clone();
        let geom = geometry_for(&self.cu_seqlens, num_heads, total_q)?;

        let (qx1, qx2) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: BWD_OP_NAME })?;
        let qkv_view = s1.as_cuda_slice::<bf16>()?.slice(qx1..qx2);
        let (ox1, ox2) = l2
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: BWD_OP_NAME })?;
        let o_view = s2.as_cuda_slice::<bf16>()?.slice(ox1..ox2);
        let (gx1, gx2) = l3
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: BWD_OP_NAME })?;
        let do_view = s3.as_cuda_slice::<bf16>()?.slice(gx1..gx2);

        let lse = self.lse.take().map_err(|e| saved_err(BWD_OP_NAME, e))?;

        let mut scratch =
            flash::BwdScratch::alloc(&device, &geom, self.cfg.deterministic).map_err(flash_err)?;
        // SAFETY: uninitialised output `flash_varlen_bwd_into` fully
        // overwrites (dq via `convert_dQ` over every row block, dk/dv over
        // every key block) — the identical allocation `flash::flash_varlen_bwd`
        // makes.
        let mut d_qkv = unsafe { device.alloc::<bf16>(geom.qkv_len()) }?;
        flash::flash_varlen_bwd_into(
            &device,
            &self.cu_seqlens,
            num_heads,
            flash::BwdBuffers {
                qkv: qkv_view,
                o: o_view,
                lse: lse.as_view(),
                d_o: do_view,
                d_qkv: d_qkv.as_view_mut(),
                softmax_d: scratch.softmax_d.as_view_mut(),
                dq_accum: scratch.dq_accum.as_view_mut(),
                dq_accum_splits: scratch.splits,
            },
            &self.cfg,
        )
        .map_err(flash_err)?;

        let out_shape = Shape::from((total_q, 3, num_heads, HEAD_DIM));
        Ok((CudaStorage::wrap_cuda_slice(d_qkv, device), out_shape))
    }

    // Second-order gradient (differentiating THROUGH `d_qkv`, this op's own
    // output) is NOT routed to candle's default `CustomOp3::bwd`
    // (`Error::BackwardNotSupported`) — an earlier version of this comment
    // claimed that, and `10b1f3b`'s audit found it FALSE (advisory finding,
    // confirmed correct by the auditor; pinned by a test below). The real
    // mechanism: `FlashVarlenAttention::bwd` (above) calls `arg.detach()` /
    // `res.detach()` / `grad_res.detach()` before constructing `d_qkv`
    // through `apply_stateful3` — by design, so backpropagating through
    // `FlashVarlenAttention` does not grow the graph with a second
    // differentiable node for no reason (see that call site's own comment).
    // A DETACHED input means candle's autograd never records a
    // `BackpropOp` edge from `d_qkv` back to `qkv`/`o`/`d_o` at all, so a
    // caller who tries to differentiate a second time through `d_qkv`
    // reaches `candle_core::Tensor::backward`'s "no gradient recorded for
    // this leaf" path, not this `bwd` method — `grads.get(qkv)` on that
    // second pass returns `None`, an ABSENT gradient, silently (candle does
    // not error on a detached/untracked leaf; it simply omits it from the
    // returned `GradStore`). This method is never called for a
    // second-order pass through `FlashVarlenAttention`'s own output for
    // that reason, and is therefore correctly left unimplemented (candle's
    // default IS reached, but only if something ELSE calls `.backward()`
    // directly on a `Tensor` produced by `apply_stateful3` on THIS type,
    // which no code path in this crate does) — but the second-order
    // CALLER's experience is "no gradient", not "a typed error", and that
    // is the fact worth stating plainly rather than the wrong one.
    // See `flash_op_oracles.rs`'s
    // `second_order_backward_through_flash_attention_varlen_output_is_a_silent_absent_gradient_not_an_error`
    // for the pinned behaviour.
}

/// The ONLY public entry point. Constructs a FRESH `FlashVarlenAttention`
/// per call (hoisting one across calls is unrepresentable outside this
/// module — see [`crate::ops::StatefulKernelOp`]'s doc) and runs it through
/// [`crate::ops::apply_stateful1`].
///
/// `cu_seqlens`/`cfg` are borrowed: this function takes its OWN owned copy
/// of `cu_seqlens`' (tiny — `batch + 1` `i32`s) device array via
/// `CuSeqlens::try_duplicate` (crate-private) so the op struct — which
/// must be `'static` — never borrows from the caller.
pub fn flash_attention_varlen(
    qkv: &Tensor,
    cu_seqlens: &CuSeqlens,
    cfg: &VarlenConfig,
) -> Result<Tensor> {
    let (_total_q, num_heads) = check_qkv_domain(qkv.dims(), qkv.dtype())?;
    let cu_seqlens = cu_seqlens.try_duplicate().map_err(flash_err)?;
    let op = FlashVarlenAttention {
        cu_seqlens,
        num_heads,
        cfg: *cfg,
        lse: Saved::empty(),
    };
    super::apply_stateful1(qkv, op)
}

const FUSED_ROPE_OP_NAME: &str = "flash_attention_varlen_fused_rope";
const FUSED_ROPE_BWD_OP_NAME: &str = "flash_attention_varlen_fused_rope_bwd";

/// Fused-RoPE flash-forward op — composes [`crate::ops::rope_positions::
/// RopePositionsFused`]'s row math with [`FlashVarlenAttention`]'s own FFI
/// calls INSIDE one `CustomOp3` node, over `(qkv, cos, sin)` — the
/// PRE-rotation packed buffer plus the RoPE table — rather than the two
/// separately-tracked ops [`flash_attention_varlen`] is built from
/// (`RopePositionsFused` producing a rotated `qkv` `Tensor`, THEN
/// `flash_attention_varlen` consuming it). See the module doc's "Two op
/// types, one seam" section for that two-op shape's own rationale — this
/// type exists ONLY to close a real, MEASURED VRAM regression the two-op
/// shape carries: candle's `BackpropOp::new3` (`op.rs`) clones EVERY
/// tracked argument into a result's `Op` unconditionally, so the rotated
/// `qkv` `RopePositionsFused` produces becomes a SECOND `[total, 3, H, 64]`
/// bf16 buffer (`crate::flash::HEAD_DIM`-sized, ≈24 MiB at ModernBERT-
/// large's `b=8, s=512, h=16` production shape) that candle's tape then
/// retains via `FlashVarlenAttention`'s OWN `Op::CustomOp1` node — ON TOP
/// of the pre-rotation `qkv` the `Wqkv` linear's own backward already
/// needs kept alive regardless of which attention arm runs.
/// [`crate::ops::AttentionBlockFused`] (the block arm) never pays this: it
/// fuses RoPE INSIDE its own `CustomOp3`, so its `bwd` RECOMPUTES the
/// rotation from the SAME `qkv` its own `Op` already retains (that op's
/// module doc, "`bwd`: ordinary `Tensor` composition") rather than needing
/// a second retained buffer. This type applies the IDENTICAL "recompute in
/// a DETACHED `bwd`" idiom to the flash arm: `bwd` reconstructs the
/// rotated buffer from the (already-alive) pre-rotation `qkv` plus
/// `cos`/`sin` (mirroring [`RopePositionsFused`]'s own row math, reused
/// directly, never re-derived) instead of candle's tape holding it for the
/// whole backward pass.
///
/// Measured: a per-layer VRAM attribution probe (`jammi-encoders`,
/// `modernbert::tests::flash_vs_block_per_layer_vram_attribution_probe_cuda`)
/// found the two-op composition's per-layer forward retention averaging
/// ≈28 MiB MORE than the block arm's own per-layer retention at this
/// shape (`b=8, s=512, h=16, head_dim=64`, bf16) — this type is the fix
/// that measurement motivated; see the P6 flash-vram-attribution round's
/// artifact for the before/after `peak_vram_bytes` numbers.
///
/// # Correctness: bit-identical to the two-op composition
///
/// Forward calls the EXACT SAME underlying functions the two-op
/// composition calls, in the SAME order (`crate::cuda::rope_positions::
/// cuda_fwd` then `flash::flash_varlen_fwd_into`) — no new numeric
/// derivation. Backward recomputes the rotation via [`super::apply3`] with
/// [`RopePositionsFused`] (the SAME kernel [`RopePositionsFused::bwd`]
/// itself calls for its own `dx`) and reuses [`FlashVarlenBwdHelper`]
/// UNCHANGED for the flash-side gradient — so the VALUES this op produces
/// are bit-identical to [`flash_attention_varlen`] called on an
/// already-rotated `qkv`, fwd and bwd; only the RETENTION differs. See
/// `tests::fused_rope_matches_two_op_composition_bit_identical_fwd_and_bwd_cuda`
/// (this module) for the pinned proof.
///
/// # Domain
///
/// Identical to [`FlashVarlenAttention`]'s own (`qkv`: bf16, contiguous,
/// `[total_q, 3, H, 64]`) PLUS `cos`/`sin`: same convention
/// [`RopePositionsFused`] accepts (`[period, 64]`, `period == seq` or
/// `period == 1`) — checked by `rope_positions_dims` internally, on top of
/// this op's own [`check_qkv_domain`] (the `HEAD_DIM == 64` / bf16 / rank
/// checks `RopePositionsFused` alone does not make, since IT accepts any
/// even `head_dim` — see that op's own module doc).
pub(crate) struct FlashVarlenAttentionFusedRope {
    /// Dense sequence length — [`RopePositionsFused::seq`]'s own field,
    /// same convention (`position = token % seq`).
    seq: usize,
    cu_seqlens: CuSeqlens,
    num_heads: usize,
    /// The config `cuda_fwd` ALWAYS launches with.
    cfg: VarlenConfig,
    /// TEST-ONLY: when `Some`, `bwd` launches [`FlashVarlenBwdHelper`] with
    /// THIS config instead of `cfg` — forward is UNAFFECTED either way, it
    /// always reads `cfg`. `None` for every production call site
    /// ([`flash_attention_varlen_with_rope`], the only public constructor
    /// besides the test-only one below), so production behaviour is
    /// completely unchanged by this field's existence: `bwd` then reads
    /// `cfg` exactly as it did before this field was added. See
    /// [`flash_attention_varlen_with_rope_test_only_bwd_window_override`]'s
    /// own doc for why a mismatched fwd/bwd config needs a real op-level
    /// seam rather than a crafted input tensor.
    bwd_cfg_override: Option<VarlenConfig>,
    /// Which [`super::RopePositionsFused`] domain `seq` means — `Dense`
    /// (`position = token % seq`, [`flash_attention_varlen_with_rope`]'s
    /// arm) or `Ragged` (`seq` holds the gathered table's own row-total,
    /// `position` degenerates to the row index,
    /// [`flash_attention_varlen_with_rope_ragged`]'s arm — M1a). BOTH
    /// `cuda_fwd` and `bwd`'s recompute read this field so the rotation
    /// `bwd` recomputes is provably the SAME one `fwd` applied — see
    /// [`super::rope_positions`]'s module doc, "The ragged arm" section.
    arm: PositionArm,
    lse: Saved<CudaSlice<f32>>,
}

impl super::sealed::Sealed for FlashVarlenAttentionFusedRope {}

impl CustomOp3 for FlashVarlenAttentionFusedRope {
    fn name(&self) -> &'static str {
        FUSED_ROPE_OP_NAME
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(Error::Msg(format!(
            "{FUSED_ROPE_OP_NAME}: no CPU arm — see {OP_NAME}'s own doc"
        )))
    }

    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
        s3: &CudaStorage,
        l3: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (total_q, num_heads) = check_qkv_domain(l1.dims(), s1.dtype())?;
        if num_heads != self.num_heads {
            return Err(Error::Msg(format!(
                "{FUSED_ROPE_OP_NAME}: qkv has {num_heads} heads but this op was constructed \
                 for {} — the caller must build cu_seqlens' geometry with the SAME head count \
                 qkv carries",
                self.num_heads
            )));
        }
        let device = s1.device().clone();
        let geom = geometry_for(&self.cu_seqlens, num_heads, total_q)?;

        // Rotate Q/K exactly as the two-op composition's `RopePositionsFused`
        // apply would — the SAME kernel launcher, called directly at the
        // storage level rather than through a tracked `Tensor` op (see this
        // type's own doc for why: candle's tape would otherwise retain the
        // rotated buffer for the whole backward pass). `rot_storage` is a
        // FRESH, contiguous, offset-0 allocation (`crate::cuda::
        // rope_positions::cuda_fwd`'s own contract, same as every other CUDA
        // kernel launcher in this crate) — `Layout::contiguous` reconstructs
        // exactly that, never a layout re-derived from anything else.
        let (rot_storage, rot_shape) = crate::cuda::rope_positions::cuda_fwd(
            self.seq, false, self.arm, s1, l1, s2, l2, s3, l3,
        )?;
        let rot_layout = Layout::contiguous(rot_shape);
        let (rx1, rx2) = rot_layout
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous {
                op: FUSED_ROPE_OP_NAME,
            })?;
        let qkv_view = rot_storage.as_cuda_slice::<bf16>()?.slice(rx1..rx2);

        // SAFETY: uninitialised outputs `flash_varlen_fwd_into` fully
        // overwrites — identical to `FlashVarlenAttention::cuda_fwd`'s own
        // allocation for the same reason.
        let mut o = unsafe { device.alloc::<bf16>(geom.o_len()) }?;
        let mut lse = unsafe { device.alloc::<f32>(geom.lse_len()) }?;
        flash::flash_varlen_fwd_into(
            &device,
            qkv_view,
            &self.cu_seqlens,
            o.as_view_mut(),
            lse.as_view_mut(),
            num_heads,
            &self.cfg,
        )
        .map_err(flash_err)?;

        // `rot_storage` (the rotated scratch buffer) is dropped HERE, at the
        // end of this function — nothing outside this call retains it: `o`
        // (the real output) is an independent allocation this function
        // built separately, and `lse` (below) is `f32`-sized, not
        // `[total,3,H,64]`-sized. This is the whole point of this type.
        self.lse
            .set(lse)
            .map_err(|e| saved_err(FUSED_ROPE_OP_NAME, e))?;

        let out_shape = Shape::from((total_q, num_heads, HEAD_DIM));
        Ok((CudaStorage::wrap_cuda_slice(o, device), out_shape))
    }

    /// Recomputes the rotation from the (already-alive) `qkv`/`cos`/`sin`
    /// rather than reading a saved rotated buffer — see this type's own
    /// doc. DETACHED throughout (mirrors `AttentionBlockFused::bwd_core`'s
    /// "`bwd` runs DETACHED" section): every composed `Tensor` here is a
    /// LOCAL binding this function drops on return, so nothing
    /// `[total, 3, H, 64]`-shaped survives past this call — in particular,
    /// `qkv_rot` (recomputed below) is NEVER retained by any `Op` node,
    /// unlike the two-op composition's own `qkv_rot`.
    fn bwd(
        &self,
        qkv: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        if cos.is_variable() || sin.is_variable() {
            return Err(Error::Msg(format!(
                "{FUSED_ROPE_OP_NAME}: cos/sin gradient is not implemented — every call site \
                 this op ships behind treats cos/sin as non-Var leaf tables, mirroring \
                 RopePositionsFused's own identical refusal (see that op's `bwd`)"
            )));
        }
        let lse = self
            .lse
            .take()
            .map_err(|e| saved_err(FUSED_ROPE_OP_NAME, e))?;

        let qkv_d = qkv.detach();
        let cos_d = cos.detach();
        let sin_d = sin.detach();
        let res_d = res.detach();
        let grad_d = grad_res.detach();

        // Recompute — the SAME `RopePositionsFused` kernel the two-op
        // composition's forward calls, on the SAME (now-detached) inputs.
        let qkv_rot = super::apply3(
            &qkv_d,
            &cos_d,
            &sin_d,
            super::RopePositionsFused::new_with_arm(self.seq, false, self.arm),
        )?;

        // The flash-side gradient: UNCHANGED `FlashVarlenBwdHelper`, the
        // SAME type/logic `FlashVarlenAttention::bwd` uses, constructed
        // identically (this op's own stashed `lse` moved into the helper's
        // OWN `Saved` slot, exactly that method's own pattern) — EXCEPT the
        // config: `bwd_cfg_override`, when set, overrides `cfg` here ONLY
        // (see that field's own doc; every production call site leaves it
        // `None`, so `unwrap_or(self.cfg)` is a no-op there).
        let helper = FlashVarlenBwdHelper {
            cu_seqlens: self.cu_seqlens.try_duplicate().map_err(flash_err)?,
            num_heads: self.num_heads,
            cfg: self.bwd_cfg_override.unwrap_or(self.cfg),
            lse: Saved::empty(),
        };
        helper
            .lse
            .set(lse)
            .map_err(|e| saved_err(FUSED_ROPE_BWD_OP_NAME, e))?;
        let d_qkv_rot = super::apply_stateful3(&qkv_rot, &res_d, &grad_d, helper)?;

        // Un-rotate — `RopePositionsFused`'s own `bwd` mechanism
        // (`negate_sin` flipped), reused directly rather than re-derived.
        let dqkv = super::apply3(
            &d_qkv_rot,
            &cos_d,
            &sin_d,
            super::RopePositionsFused::new_with_arm(self.seq, true, self.arm),
        )?;
        Ok((Some(dqkv), None, None))
    }
}

/// Same contract as [`flash_attention_varlen`], but fuses the RoPE
/// rotation of `qkv` INSIDE this op's own `CustomOp3` node instead of
/// requiring the caller to rotate first and hand over an already-rotated
/// buffer — see this module's `FlashVarlenAttentionFusedRope` (crate-private,
/// hence a code span here rather than a doc link) for why this exists and
/// what it costs less than the two-op composition. `qkv` here is the
/// PRE-rotation packed buffer `[total, 3, H, 64]`; `cos`/`sin` are
/// [`super::RopePositionsFused`]'s own table convention (`[period, 64]`).
pub fn flash_attention_varlen_with_rope(
    qkv: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    seq: usize,
    cu_seqlens: &CuSeqlens,
    cfg: &VarlenConfig,
) -> Result<Tensor> {
    let (_total_q, num_heads) = check_qkv_domain(qkv.dims(), qkv.dtype())?;
    // `RopePositionsFused`'s own dense-only scope (`ops::rope_positions`'s
    // module doc): `position = token % seq` is only valid when `cu_seqlens`
    // is dense/uniform at exactly `seq` — `rope_positions_dims` alone
    // cannot see that (it only checks `total % seq == 0`, which a
    // non-uniform batch summing to `batch*seq` also satisfies), so this op
    // — the ONLY caller that pairs `cu_seqlens` with a `seq` — checks it
    // here instead.
    cu_seqlens.check_dense_uniform(seq).map_err(flash_err)?;
    let cu_seqlens = cu_seqlens.try_duplicate().map_err(flash_err)?;
    let op = FlashVarlenAttentionFusedRope {
        seq,
        cu_seqlens,
        num_heads,
        cfg: *cfg,
        bwd_cfg_override: None,
        arm: PositionArm::Dense,
        lse: Saved::empty(),
    };
    super::apply_stateful3(qkv, cos, sin, op)
}

/// Ragged counterpart of [`flash_attention_varlen_with_rope`] (M1a —
/// varlen positions): `qkv`'s `total_q` rows are the CONCATENATION of
/// `lengths.len()` variable-length segments (no padding), never a padded
/// `[batch, seq]` grid. `cu_seqlens` (the FFI's own varlen geometry) and
/// the per-row rotation table are BOTH derived from this ONE `lengths`
/// slice — see [`crate::ops::rope_positions`]'s module doc, "The ragged
/// arm" section, for why that makes a table/segmentation mismatch (the
/// two-op composition's own hazard this type's doc names) structurally
/// unconstructible: there is exactly one source of truth for both.
///
/// `cos_base`/`sin_base`: the SAME base-table convention
/// [`super::RopePositionsFused::new`]'s dense arm accepts (`[period_base,
/// d]`, any leading dims of size 1) — gathered INTERNALLY, via
/// [`crate::ops::rope_positions::gather_ragged_tables`], into the `[total,
/// d]` tables this op's own `CustomOp3` node is actually built over —
/// exactly what [`crate::ops::rope_positions::rope_positions_fused_ragged`]
/// (the CPU/eager entry) does, so the two entries share ONE gathering
/// implementation rather than two.
///
/// The bwd recompute reuses the SAME gathered `cos_r`/`sin_r`: candle
/// hands `bwd` the exact `cos`/`sin` tensors THIS function passed to
/// `apply_stateful3` (`FlashVarlenAttentionFusedRope::bwd`'s `cos`/`sin`
/// parameters), so the rotation `bwd` recomputes is provably the SAME one
/// `fwd` applied — never re-derived from `lengths` a second time.
pub fn flash_attention_varlen_with_rope_ragged(
    qkv: &Tensor,
    cos_base: &Tensor,
    sin_base: &Tensor,
    lengths: &[usize],
    cfg: &VarlenConfig,
) -> Result<Tensor> {
    let (_total_q, num_heads) = check_qkv_domain(qkv.dims(), qkv.dtype())?;
    let Device::Cuda(device) = qkv.device() else {
        return Err(Error::Msg(format!(
            "{FUSED_ROPE_OP_NAME}: ragged entry requires a CUDA qkv tensor -- this op has no \
             CPU arm, see {OP_NAME}'s own doc"
        )));
    };
    let cu_seqlens = CuSeqlens::from_lengths(lengths, device).map_err(flash_err)?;
    let (total, cos_r, sin_r) =
        crate::ops::rope_positions::gather_ragged_tables(cos_base, sin_base, lengths)?;
    let op = FlashVarlenAttentionFusedRope {
        seq: total,
        cu_seqlens,
        num_heads,
        cfg: *cfg,
        bwd_cfg_override: None,
        arm: PositionArm::Ragged,
        lse: Saved::empty(),
    };
    super::apply_stateful3(qkv, &cos_r, &sin_r, op)
}

/// TEST-SUPPORT ONLY — never wired into [`flash_attention_varlen_with_rope`]
/// or any production/admission/dispatch path (production always calls the
/// function above, which leaves `bwd_cfg_override` `None`). Identical to
/// [`flash_attention_varlen_with_rope`] except the FORWARD launch uses
/// `fwd_cfg` and the BACKWARD launch uses a DIFFERENT config, `bwd_cfg` —
/// [`FlashVarlenAttentionFusedRope`]'s own `bwd_cfg_override` field, which
/// production construction never sets. This exists because a
/// backward-only defect in the window radius (or any other `VarlenConfig`
/// field) cannot be modelled by crafting a different INPUT tensor the way
/// [`crate::flash`]'s FFI-level RED controls elsewhere model a K-unrotated
/// defect (`tests/cuda_parity.rs`'s `k_unrotated` control) — the window is
/// a per-call CONFIG parameter each of `flash_varlen_fwd_into`/
/// `flash_varlen_bwd_into` takes independently, not something a tensor
/// value can encode — so the only honest way to exercise "forward correct,
/// backward wrong" is a real seam letting the two calls disagree. See
/// `tests/cuda_parity.rs`'s
/// `flash_upstream_acceptance_form_red_control_bwd_only_window_dropped_cuda`
/// (this crate's own op-level acceptance-form suite) for the RED control
/// built on top of this function, and
/// `jammi-encoders`' `modernbert::flash_oracle_wqkv_lora_b`'s own doc
/// for why that encoder-level oracle's gradient leg cannot see this defect
/// class on its own (it sits on a global, unwindowed layer) and defers to
/// this op-level control instead.
pub fn flash_attention_varlen_with_rope_test_only_bwd_window_override(
    qkv: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    seq: usize,
    cu_seqlens: &CuSeqlens,
    fwd_cfg: &VarlenConfig,
    bwd_cfg: &VarlenConfig,
) -> Result<Tensor> {
    let (_total_q, num_heads) = check_qkv_domain(qkv.dims(), qkv.dtype())?;
    // Same dense/uniform requirement as `flash_attention_varlen_with_rope`
    // — see that function's own call site for why.
    cu_seqlens.check_dense_uniform(seq).map_err(flash_err)?;
    let cu_seqlens = cu_seqlens.try_duplicate().map_err(flash_err)?;
    let op = FlashVarlenAttentionFusedRope {
        seq,
        cu_seqlens,
        num_heads,
        cfg: *fwd_cfg,
        bwd_cfg_override: Some(*bwd_cfg),
        arm: PositionArm::Dense,
        lse: Saved::empty(),
    };
    super::apply_stateful3(qkv, cos, sin, op)
}

/// Structural (white-box) proof that [`flash_attention_varlen`] does not
/// silently pin `cfg.deterministic` — see the module doc's "Domain"
/// section: "`cfg.deterministic` is whatever the caller passes ... this op
/// itself has no opinion". `10b1f3b`'s audit asked for a test deciding and
/// documenting this ("a caller passing `deterministic: false` gets it
/// PINNED to true by the op, or the op refuses"): the decision is NEITHER
/// — a generic primitive (family L) has no opinion of its own; Stage B2's
/// encoder is the ONLY place that pins `true`, at ITS call site, not here.
/// Mirrors `flash_attention_varlen`'s own construction exactly and reads
/// the field back — CUDA is needed only to build a real `CuSeqlens`, not to
/// launch anything.
#[cfg(test)]
mod deterministic_passthrough {
    use super::*;

    #[test]
    fn cfg_deterministic_flows_through_construction_unmodified() {
        let Ok(cuda) = candle_core::Device::new_cuda(0) else {
            eprintln!(
                "cfg_deterministic_flows_through_construction_unmodified: skipping — no CUDA \
                 device available"
            );
            return;
        };
        let dev = cuda.as_cuda_device().unwrap().clone();
        for det in [true, false] {
            let cu_seqlens = crate::flash::CuSeqlens::from_lengths(&[8usize], &dev).unwrap();
            let cfg = crate::flash::VarlenConfig {
                softmax_scale: 0.125,
                window: None,
                deterministic: det,
            };
            // The SAME construction `flash_attention_varlen` performs.
            let op = FlashVarlenAttention {
                cu_seqlens,
                num_heads: 1,
                cfg,
                lse: Saved::empty(),
            };
            assert_eq!(
                op.cfg.deterministic, det,
                "the op must store exactly what the caller passed — never overriding it"
            );
        }
    }
}

/// [`FlashVarlenAttentionFusedRope`]'s own oracle (module doc's
/// "Correctness: bit-identical to the two-op composition" section): the
/// fused op and the two-op composition it replaces must produce the
/// IDENTICAL forward output and the IDENTICAL gradient wrt `qkv`, from the
/// SAME inputs — only the RETENTION differs, never the VALUE. CUDA-only
/// (both paths are CUDA-only); skips (does not fail) without a device.
#[cfg(test)]
mod fused_rope_matches_two_op_composition {
    use super::*;
    use candle_core::{DType, Var};

    /// A real (non-trivial-angle) rotary table in `[1, 1, period, hidden]`
    /// form — the same shape [`ModernBertAttention::forward_flash_dense_attention`]
    /// hands this op (`RotaryEmbedding::cached_tables`'s own convention),
    /// cast to `bf16` since both paths under test require `qkv`/`cos`/`sin`
    /// to share one dtype.
    fn rope_table(period: usize, hidden: usize, device: &candle_core::Device) -> (Tensor, Tensor) {
        let half = hidden / 2;
        let mut cos = vec![0f32; period * hidden];
        let mut sin = vec![0f32; period * hidden];
        for pos in 0..period {
            for i in 0..half {
                let theta = 10_000f64.powf(-2.0 * (i as f64) / (hidden as f64));
                let angle = pos as f64 * theta;
                let (s, c) = angle.sin_cos();
                cos[pos * hidden + i] = c as f32;
                cos[pos * hidden + i + half] = c as f32;
                sin[pos * hidden + i] = s as f32;
                sin[pos * hidden + i + half] = s as f32;
            }
        }
        let cos = Tensor::from_vec(cos, (1, 1, period, hidden), device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let sin = Tensor::from_vec(sin, (1, 1, period, hidden), device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        (cos, sin)
    }

    #[test]
    fn fused_rope_matches_two_op_composition_bit_identical_fwd_and_bwd_cuda() {
        let Ok(cuda) = candle_core::Device::new_cuda(0) else {
            eprintln!(
                "fused_rope_matches_two_op_composition_bit_identical_fwd_and_bwd_cuda: skipping \
                 — no CUDA device available"
            );
            return;
        };
        let dev = cuda.as_cuda_device().unwrap().clone();
        let (batch, seq, h, d) = (2usize, 8usize, 2usize, HEAD_DIM);
        let total = batch * seq;
        let n = total * 3 * h * d;
        // Non-trivial, non-symmetric, distinct-per-element data.
        let xv: Vec<f32> = (0..n).map(|k| (k as f32 * 0.037).sin() * 3.0).collect();
        let qkv_bf16 = Tensor::from_vec(xv, (total, 3, h, d), &cuda)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let qkv_var = Var::from_tensor(&qkv_bf16).unwrap();
        let (cos, sin) = rope_table(seq, d, &cuda);
        let lengths = vec![seq; batch];
        let cu_seqlens_a = CuSeqlens::from_lengths(&lengths, &dev).unwrap();
        let cu_seqlens_b = CuSeqlens::from_lengths(&lengths, &dev).unwrap();
        let cfg = VarlenConfig {
            softmax_scale: 1.0 / (d as f32).sqrt(),
            window: None,
            deterministic: true,
        };

        // Path A: the pre-fix two-op composition — rotate via a TRACKED
        // `RopePositionsFused` apply, then `flash_attention_varlen` on the
        // already-rotated buffer.
        let qkv_rot_a = crate::ops::apply3(
            qkv_var.as_tensor(),
            &cos,
            &sin,
            crate::ops::RopePositionsFused::new(seq, false),
        )
        .unwrap();
        let o_a = flash_attention_varlen(&qkv_rot_a, &cu_seqlens_a, &cfg).unwrap();

        // Path B: this round's fix.
        let o_b = flash_attention_varlen_with_rope(
            qkv_var.as_tensor(),
            &cos,
            &sin,
            seq,
            &cu_seqlens_b,
            &cfg,
        )
        .unwrap();

        let to_f32_vec = |t: &Tensor| -> Vec<f32> {
            t.flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let o_a_v = to_f32_vec(&o_a);
        let o_b_v = to_f32_vec(&o_b);
        assert!(
            o_a_v.iter().chain(o_b_v.iter()).all(|v| v.is_finite()),
            "forward outputs must be finite before comparing"
        );
        assert_eq!(
            o_a_v, o_b_v,
            "flash_attention_varlen_with_rope's forward output must be bit-identical to the \
             two-op composition it replaces"
        );

        let loss_a = o_a
            .to_dtype(DType::F32)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let loss_b = o_b
            .to_dtype(DType::F32)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let grads_a = loss_a.backward().unwrap();
        let grads_b = loss_b.backward().unwrap();
        let dqkv_a = grads_a
            .get(qkv_var.as_tensor())
            .expect("path A: qkv must have a gradient");
        let dqkv_b = grads_b
            .get(qkv_var.as_tensor())
            .expect("path B: qkv must have a gradient");
        let dqkv_a_v = to_f32_vec(dqkv_a);
        let dqkv_b_v = to_f32_vec(dqkv_b);
        assert!(
            dqkv_a_v
                .iter()
                .chain(dqkv_b_v.iter())
                .all(|v| v.is_finite()),
            "gradients must be finite before comparing"
        );
        assert_eq!(
            dqkv_a_v, dqkv_b_v,
            "flash_attention_varlen_with_rope's dqkv must be bit-identical to the two-op \
             composition's own dqkv"
        );
    }
}

// M1a — varlen positions: the CUDA-gated RETENTION oracle, flash-level
// DENSE INVARIANCE oracle, and no-CPU-arm guard for
// [`flash_attention_varlen_with_rope_ragged`] moved to
// `tests/cuda_parity.rs` (`fused_rope_ragged_matches_two_op_composition_
// bit_identical_fwd_and_bwd_cuda` and its two siblings) so their CUDA
// skip is MECHANICALLY enforced by `check_kernel_oracles.py`'s KO-7 scan
// (`crates/jammi-kernels/tests/**`, not `crates/jammi-kernels/src/**`)
// via that file's already-registered `cuda_device` helper, rather than
// voluntarily mirrored here with no ci/kernel-oracle-helpers.txt entry to
// enforce it. The CPU-hermetic TRUTH oracle, `gather_ragged_tables`
// assertions, GUARDS, and op-level DENSE INVARIANCE for the ragged arm
// itself remain in `crate::ops::rope_positions`'s own `#[cfg(test)] mod
// tests` (see that module's own doc, "The ragged arm" section).

#[cfg(test)]
mod tests {
    //! Pure cells (no CUDA device needed) — `check_qkv_domain`/
    //! `check_o_domain` are OR-chains of independent predicates
    //! (family K's "a lattice per predicate" standing clause): each
    //! disjunct gets its OWN cell, violated ALONE (every other field
    //! correct), so a `||` -> `&&` mutation (which only fails when
    //! MULTIPLE disjuncts are simultaneously true) cannot survive —
    //! these are exactly the two cells `cargo mutants` found MISSED
    //! before this block existed (only integration-level, CUDA-requiring
    //! tests exercised these functions previously, none violating a
    //! single disjunct in isolation).
    use super::*;

    const OK_DIMS: [usize; 4] = [21, 3, 4, HEAD_DIM];

    #[test]
    fn check_qkv_domain_accepts_the_canonical_shape() {
        let (total_q, num_heads) = check_qkv_domain(&OK_DIMS, DType::BF16).unwrap();
        assert_eq!(total_q, 21);
        assert_eq!(num_heads, 4);
    }

    #[test]
    fn check_qkv_domain_refuses_wrong_rank_alone() {
        // Rank 3 (missing the `3` axis) — dims[1]/dims[3] positions shift,
        // but the point of THIS cell is "rank wrong, nothing else checked
        // yet" (a rank-5 case establishes rank-wrongness independent of
        // what dims[1]/dims[3] would even mean).
        let dims = [21usize, 3, 4, HEAD_DIM, 1];
        let e = check_qkv_domain(&dims, DType::BF16).unwrap_err();
        assert!(matches!(e, Error::Msg(_)), "{e}");
    }

    #[test]
    fn check_qkv_domain_refuses_wrong_middle_axis_alone() {
        // Rank correct (4), dims[3] correct (HEAD_DIM) — ONLY dims[1] != 3
        // is violated. This is exactly the cell a `||` -> `&&` mutation
        // survives on (both other disjuncts are false here).
        let dims = [21usize, 5, 4, HEAD_DIM];
        assert_eq!(dims.len(), 4);
        assert_eq!(dims[3], HEAD_DIM);
        let e = check_qkv_domain(&dims, DType::BF16).unwrap_err();
        assert!(matches!(e, Error::Msg(_)), "{e}");
    }

    #[test]
    fn check_qkv_domain_refuses_wrong_head_dim_alone() {
        // Rank correct (4), dims[1] correct (3) — ONLY dims[3] != HEAD_DIM
        // is violated — the SECOND cell the mutation run found missed.
        let dims = [21usize, 3, 4, HEAD_DIM + 1];
        assert_eq!(dims.len(), 4);
        assert_eq!(dims[1], 3);
        let e = check_qkv_domain(&dims, DType::BF16).unwrap_err();
        assert!(matches!(e, Error::Msg(_)), "{e}");
    }

    #[test]
    fn check_qkv_domain_refuses_non_bf16_with_shape_otherwise_correct() {
        let e = check_qkv_domain(&OK_DIMS, DType::F32).unwrap_err();
        assert!(
            matches!(e, Error::UnsupportedDTypeForOp(DType::F32, OP_NAME)),
            "{e}"
        );
    }

    const OK_O_DIMS: [usize; 3] = [21, 4, HEAD_DIM];

    #[test]
    fn check_o_domain_accepts_the_canonical_shape() {
        check_o_domain(&OK_O_DIMS, DType::BF16, 21, 4).unwrap();
    }

    #[test]
    fn check_o_domain_refuses_wrong_rank_alone() {
        let dims = [21usize, 4, HEAD_DIM, 1];
        let e = check_o_domain(&dims, DType::BF16, 21, 4).unwrap_err();
        assert!(matches!(e, Error::Msg(_)), "{e}");
    }

    #[test]
    fn check_o_domain_refuses_wrong_total_q_alone() {
        let dims = [20usize, 4, HEAD_DIM]; // total_q off by one, rest correct
        let e = check_o_domain(&dims, DType::BF16, 21, 4).unwrap_err();
        assert!(matches!(e, Error::Msg(_)), "{e}");
    }

    #[test]
    fn check_o_domain_refuses_wrong_num_heads_alone() {
        let dims = [21usize, 5, HEAD_DIM]; // num_heads off by one, rest correct
        let e = check_o_domain(&dims, DType::BF16, 21, 4).unwrap_err();
        assert!(matches!(e, Error::Msg(_)), "{e}");
    }

    #[test]
    fn check_o_domain_refuses_wrong_head_dim_alone() {
        let dims = [21usize, 4, HEAD_DIM + 1];
        let e = check_o_domain(&dims, DType::BF16, 21, 4).unwrap_err();
        assert!(matches!(e, Error::Msg(_)), "{e}");
    }

    #[test]
    fn check_o_domain_refuses_non_bf16_with_shape_otherwise_correct() {
        let e = check_o_domain(&OK_O_DIMS, DType::F32, 21, 4).unwrap_err();
        assert!(
            matches!(e, Error::UnsupportedDTypeForOp(DType::F32, OP_NAME)),
            "{e}"
        );
    }
}
