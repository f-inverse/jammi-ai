//! CUDA forward for [`crate::ops::AttentionBlockFused`] — the composed
//! interior the op contract's Tier 0 build order describes: NO new `.cu`
//! kernel. Every step below is either `BackendStorage::copy_strided_src`/
//! `affine`/`matmul`/`binary_impl`/`to_dtype` (candle's OWN generic storage
//! kernels — the same ones `Tensor::contiguous()`/`Tensor::matmul`/
//! `Tensor + Tensor`/`Tensor::to_dtype` already issue) or a DIRECT call
//! into [`crate::ops::RopeFused`]'s / [`crate::ops::SoftmaxLastDimFused`]'s
//! OWN `cuda_fwd` — the same "call the existing op's `cuda_fwd` directly"
//! reuse idiom `LowRankResidualLinear`'s CUDA glue documents for
//! `DropoutFused`/`ScaledCastAdd`.
//!
//! ## The permute problem this file's `slot_view`/`gather_bhsd` solve
//!
//! `qkv` is `[batch, seq, 3, heads, head_dim]`; `Q`/`K`/`V` are the three
//! `narrow`s along axis 2. A `[batch, heads]`-batched `matmul` needs a
//! UNIFORM per-`(b,h)`-step stride (cuBLAS strided-batched GEMM), which a
//! `[batch, seq, 3, heads, head_dim]`-shaped source cannot provide directly
//! — `(b, h)` is not affine in a single skip value over that layout (the
//! `3` axis sits between the batch axes and `(seq, head_dim)`). `gather_bhsd`
//! materializes each of `Q`/`K`/`V` into its OWN transient, genuinely
//! contiguous `[batch, heads, seq, head_dim]` scratch buffer via
//! `BackendStorage::copy_strided_src` — candle's own generic strided-copy
//! kernel (the one `Tensor::contiguous()` already calls internally),
//! invoked here at the STORAGE level so it creates no autograd node (this
//! whole function runs inside ONE `CustomOp3::cuda_fwd`).
//!
//! ## The window band is the caller's job, not this file's
//!
//! [`crate::ops::attention_block`]'s module doc states this op has no
//! `window`/`half_window` construction data at all: the caller pre-combines
//! its padding mask with any sliding-window band into ONE additive value
//! per `(batch, query, key)` BEFORE this op ever runs (how it builds or
//! caches that combination is the caller's business, not a premise of
//! this file), so `s3`/`l3` here are already the final additive mask —
//! [`crate::ops::SoftmaxLastDimFused`]'s `cuda_fwd` is handed `s3`/`l3`
//! DIRECTLY, with no scratch-mask build, no host-to-device band upload,
//! and no `broadcast_as`/`binary_impl::<Add>` combination step in this
//! file at all: `SoftmaxLastDimFused::cuda_fwd` already implements the
//! general `[batch|1, 1, seq|1, seq]`-broadcasts-onto-`[batch, heads, seq,
//! seq]` class this op's own `mask` domain now matches exactly (see
//! `crate::ops::attention_block::check_mask`'s doc) — reusing that
//! broadcast logic rather than re-deriving it here.
//!
//! This file compiles only under the `cuda` feature and mirrors
//! `crate::cuda::rope`/`crate::cuda::softmax`/`LowRankResidualLinear`'s
//! CUDA glue idioms; `tests/cuda_parity.rs`'s `attention_block_*` legs are
//! its landing proof.

use candle_core::backend::BackendStorage;
use candle_core::{CudaStorage, CustomOp2, CustomOp3, DType, Error, Layout, Result, Shape};

use crate::ops::attention_block::{
    attention_dims, check_mask, check_rope_pack, AttentionBlockFused,
};
use crate::ops::{RopeFused, SoftmaxLastDimFused};

type CudaDevice = candle_core::CudaDevice;

fn alloc_scratch(device: &CudaDevice, dtype: DType, n: usize) -> Result<CudaStorage> {
    match dtype {
        DType::F32 => {
            let s = unsafe { device.alloc::<f32>(n) }?;
            Ok(CudaStorage::wrap_cuda_slice(s, device.clone()))
        }
        DType::BF16 => {
            let s = unsafe { device.alloc::<half::bf16>(n) }?;
            Ok(CudaStorage::wrap_cuda_slice(s, device.clone()))
        }
        dtype => Err(Error::UnsupportedDTypeForOp(dtype, "attention_block_fused")),
    }
}

/// `qkv`'s `[batch, seq, 3, heads, head_dim]` layout, viewed as
/// `[batch, heads, seq, head_dim]` for slot `slot` (`0`=Q, `1`=K, `2`=V) —
/// a `narrow` (axis 2, size 1) with the size-1 axis dropped by hand (no
/// `Layout::squeeze` exists; `Layout::new` rebuilds the shape/stride pair
/// directly, the same low-level move `Layout::transpose`/`narrow`
/// themselves make) followed by a `transpose(1, 2)` VIEW — no data moved
/// yet, see [`gather_bhsd`] for the materializing step.
fn slot_view(l_qkv: &Layout, slot: usize) -> Result<Layout> {
    let narrowed = l_qkv.narrow(2, slot, 1)?;
    let dims = narrowed.dims();
    let stride = narrowed.stride();
    let squeezed = Layout::new(
        Shape::from((dims[0], dims[1], dims[3], dims[4])),
        vec![stride[0], stride[1], stride[3], stride[4]],
        narrowed.start_offset(),
    );
    squeezed.transpose(1, 2)
}

fn gather_bhsd(
    qkv: &CudaStorage,
    view: &Layout,
    dtype: DType,
    device: &CudaDevice,
) -> Result<(CudaStorage, Layout)> {
    let n = view.shape().elem_count();
    let mut dst = alloc_scratch(device, dtype, n)?;
    qkv.copy_strided_src(&mut dst, 0, view)?;
    Ok((dst, Layout::contiguous(view.shape().clone())))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cuda_fwd(
    op: &AttentionBlockFused,
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
    s3: &CudaStorage,
    l3: &Layout,
) -> Result<(CudaStorage, Shape)> {
    let name = op.name();
    let (b, s, h, d) = attention_dims(l1, name)?;
    let device = s1.device().clone();
    let out_shape = Shape::from((b, s, h * d));

    // `check_mask`/the dtype checks/`qkv`'s (and, under RoPE, `rope_pack`'s)
    // contiguity are validated UNCONDITIONALLY, before the zero-extent fast
    // path below — matching `cpu_fwd`'s documented domain exactly:
    // `ops::attention_block::AttentionBlockFused::cpu_fwd`'s own comment
    // states there is "No empty fast path on this arm" precisely because
    // its general path (mask/dtype/contiguity checks, then the F32
    // compute) runs even when `b`/`s`/`h` is 0, so a malformed mask or a
    // non-contiguous `qkv`/`rope_pack` is refused on CPU regardless of
    // whether the tensor is also empty. The CUDA arm keeps its own early
    // return below purely to avoid handing cuBLAS a zero-extent GEMM — but
    // only AFTER establishing the exact same admission this op gives on
    // CPU, not before it.
    //
    // `qkv` must be contiguous — the SAME domain the CPU arm's `cpu_fwd`
    // requires (`l1.contiguous_offsets()`). `slot_view`/`gather_bhsd`
    // could structurally tolerate an arbitrarily strided `qkv` (they read
    // through `copy_strided_src`), but requiring contiguity here anyway
    // keeps this op's PUBLIC domain contract (module doc: "qkv: ...
    // contiguous") identical across devices — a caller whose `qkv` happens
    // to satisfy CUDA's looser internal tolerance but fails CPU's explicit
    // check would otherwise see device-dependent admission.
    l1.contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: name })?;
    // `check_mask` validates `mask`'s domain (module doc); the broadcast
    // itself is handled structurally by `SoftmaxLastDimFused::cuda_fwd`
    // below, which is handed `s3`/`l3` directly — no local unravel of
    // `mask_b`/`mask_q` is needed in this file at all.
    let (_mask_b, _mask_q) = check_mask(l3, b, s, name)?;
    if s1.dtype() != s3.dtype() {
        return Err(Error::DTypeMismatchBinaryOp {
            lhs: s1.dtype(),
            rhs: s3.dtype(),
            op: name,
        });
    }
    if op.rope && s1.dtype() != s2.dtype() {
        return Err(Error::DTypeMismatchBinaryOp {
            lhs: s1.dtype(),
            rhs: s2.dtype(),
            op: name,
        });
    }
    if !matches!(s1.dtype(), DType::F32 | DType::BF16) {
        return Err(Error::UnsupportedDTypeForOp(s1.dtype(), name));
    }
    if op.rope {
        // `cos_l`/`sin_l` (built below, only once this fn is past the
        // zero-extent fast path) derive `sin`'s start offset by ADDING
        // `s_max * d` to `l2`'s own start offset — sound ONLY if `l2` is
        // itself contiguous from that offset. Validated here, unconditionally,
        // for the SAME reason `l1`'s own contiguity is checked above rather
        // than only inside the non-empty path: `cpu_fwd`'s own rope_pack
        // validation (`check_rope_pack` + `l2.contiguous_offsets()`) runs
        // inside its F32 match, which itself runs unconditionally (no
        // empty fast path there either).
        check_rope_pack(l2, s, d, name)?;
        l2.contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: name })?;
    }

    if b == 0 || s == 0 || h == 0 {
        let out = alloc_scratch(&device, s1.dtype(), 0)?;
        return Ok((out, out_shape));
    }

    let q_view = slot_view(l1, 0)?;
    let k_view = slot_view(l1, 1)?;
    let v_view = slot_view(l1, 2)?;
    let (q_storage, q_l) = gather_bhsd(s1, &q_view, s1.dtype(), &device)?;
    let (k_storage, k_l) = gather_bhsd(s1, &k_view, s1.dtype(), &device)?;
    let (v_storage, _v_l) = gather_bhsd(s1, &v_view, s1.dtype(), &device)?;

    let (q_rot, q_rot_l, k_rot) = if op.rope {
        let s_max = check_rope_pack(l2, s, d, name)?;
        // `cos_l`/`sin_l` below derive `sin`'s start offset by ADDING
        // `s_max * d` to `l2`'s own start offset — sound ONLY if `l2` is
        // itself contiguous from that offset (a narrowed/strided
        // `rope_pack` would make this arithmetic land on the WRONG
        // elements, silently, rather than erroring — the same
        // "missing-offset" class MAINTAINER-GUIDE's CUDA-glue rules call
        // out). Checked here, before that derivation, rather than assumed.
        l2.contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: name })?;
        let cos_l =
            Layout::contiguous_with_offset((1, 1, s_max, d), l2.start_offset()).narrow(2, 0, s)?;
        let sin_l = Layout::contiguous_with_offset((1, 1, s_max, d), l2.start_offset() + s_max * d)
            .narrow(2, 0, s)?;
        let rope_op = RopeFused::new(false);
        let (qo, qshape) = CustomOp3::cuda_fwd(&rope_op, &q_storage, &q_l, s2, &cos_l, s2, &sin_l)?;
        let (ko, _kshape) =
            CustomOp3::cuda_fwd(&rope_op, &k_storage, &k_l, s2, &cos_l, s2, &sin_l)?;
        (qo, Layout::contiguous(qshape), ko)
    } else {
        (q_storage, q_l, k_storage)
    };

    let q_scaled = q_rot.affine(&q_rot_l, f64::from(op.scale()), 0.0)?;

    let bh = b * h;
    let flat_l = Layout::contiguous((bh, s, d));
    let k_t_l = flat_l.transpose(1, 2)?;
    let scores_storage = q_scaled.matmul(&k_rot, (bh, s, s, d), &flat_l, &k_t_l)?;
    let scores_l = Layout::contiguous((b, h, s, s));

    // `s3`/`l3` are ALREADY the caller's combined padding-plus-band mask
    // (module doc's "The window band is the caller's job" section) — no
    // scratch band to build or broadcast here; `SoftmaxLastDimFused::
    // cuda_fwd` implements the `[batch|1, 1, seq|1, seq]`-onto-`[batch,
    // heads, seq, seq]` broadcast class directly.
    let softmax_op = SoftmaxLastDimFused::new(op.fully_masked);
    let (p_storage, p_shape) =
        CustomOp2::cuda_fwd(&softmax_op, &scores_storage, &scores_l, s3, l3)?;
    let p_flat_l = Layout::contiguous((bh, s, s));
    let v_flat_l = Layout::contiguous((bh, s, d));
    debug_assert_eq!(p_shape, *scores_l.shape());
    let ctx_storage = p_storage.matmul(&v_storage, (bh, s, d, s), &p_flat_l, &v_flat_l)?;

    // Scatter [batch, heads, seq, head_dim] -> [batch, seq, heads*head_dim]
    // via the same `copy_strided_src` materializing step `gather_bhsd`
    // uses, applied to the OTHER direction of the same transpose.
    let ctx_bhsd_l = Layout::contiguous((b, h, s, d));
    let ctx_view = ctx_bhsd_l.transpose(1, 2)?;
    let mut out_storage = alloc_scratch(&device, s1.dtype(), b * s * h * d)?;
    ctx_storage.copy_strided_src(&mut out_storage, 0, &ctx_view)?;

    Ok((out_storage, out_shape))
}
