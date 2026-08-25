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
    CpuStorage, CudaStorage, CustomOp1, CustomOp3, DType, Error, Layout, Result, Shape, Tensor,
};
use half::bf16;

use crate::flash::{self, CuSeqlens, VarlenConfig, HEAD_DIM};
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

    // `bwd` (differentiating THROUGH a gradient computation — second-order
    // grad) is intentionally unimplemented: candle's default
    // `CustomOp3::bwd` returns `Error::BackwardNotSupported`, which is the
    // honest answer here (out of scope, see the module doc's "Domain").
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
