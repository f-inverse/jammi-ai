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
//! ## The window band: materialized as a scratch mask (module doc's
//! disclosed Tier-0 choice)
//!
//! `[crate::ops::attention_block]`'s module doc states the Tier-0 choice
//! between extending the softmax kernel's mask-broadcast signature and
//! materializing the band into a scratch mask; this file takes the LATTER
//! (matching the CPU arm): [`build_band_storage`] uploads a HOST-computed
//! `[1, 1, seq, seq]` band (`clone_htod` — a real, disclosed host-to-device
//! transfer of `seq^2` floats, pure position math with no data dependency,
//! done once per forward call; not hidden here) and combines it with the
//! padding mask via `BackendStorage::binary_impl::<candle_core::op::Add>`
//! over `Layout::broadcast_as`-expanded views — the SAME additive
//! combination, in the SAME order (band-plus-padding, summed once, then fed
//! to softmax), the CPU arm and the current ModernBERT training arm both use.
//!
//! ## UNVERIFIED ON HARDWARE at Tier 0
//!
//! This file compiles only under the `cuda` feature; this development
//! environment has no CUDA toolchain, so it has never itself run on a GPU.
//! It mirrors `crate::cuda::rope`/`crate::cuda::softmax`/
//! `LowRankResidualLinear`'s CUDA glue idioms as closely as possible, and
//! the `tests/cuda_parity.rs` legs this commit adds are its landing proof
//! — they await the pod session that owns CUDA hardware access.

use candle_core::backend::BackendStorage;
use candle_core::{CudaStorage, CustomOp2, CustomOp3, DType, Error, Layout, Result, Shape};

use crate::ops::attention_block::{
    attention_dims, check_mask, check_rope_pack, check_window, AttentionBlockFused,
    WINDOW_MASKED_VALUE,
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

/// See the module doc's "The window band" section.
fn build_band_storage(
    s: usize,
    half_window: usize,
    dtype: DType,
    device: &CudaDevice,
) -> Result<(CudaStorage, Layout)> {
    let mut band = vec![0f32; s * s];
    for qi in 0..s {
        for ki in 0..s {
            if qi.abs_diff(ki) > half_window {
                band[qi * s + ki] = WINDOW_MASKED_VALUE;
            }
        }
    }
    let l = Layout::contiguous((1, 1, s, s));
    let slice = device.clone_htod(&band)?;
    let storage = CudaStorage::wrap_cuda_slice(slice, device.clone());
    if dtype == DType::F32 {
        Ok((storage, l))
    } else {
        let cast = storage.to_dtype(&l, dtype)?;
        Ok((cast, l))
    }
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
    if b == 0 || s == 0 || h == 0 {
        let out = alloc_scratch(&device, s1.dtype(), 0)?;
        return Ok((out, out_shape));
    }
    let mask_b = check_mask(l3, b, s, name)?;
    let _ = mask_b; // broadcast handled structurally by `Layout::broadcast_as` below
    let half_window = check_window(op.window, s, name)?;
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

    let q_view = slot_view(l1, 0)?;
    let k_view = slot_view(l1, 1)?;
    let v_view = slot_view(l1, 2)?;
    let (q_storage, q_l) = gather_bhsd(s1, &q_view, s1.dtype(), &device)?;
    let (k_storage, k_l) = gather_bhsd(s1, &k_view, s1.dtype(), &device)?;
    let (v_storage, _v_l) = gather_bhsd(s1, &v_view, s1.dtype(), &device)?;

    let (q_rot, q_rot_l, k_rot) = if op.rope {
        let s_max = check_rope_pack(l2, s, d, name)?;
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

    let q_scaled = q_rot.affine(&q_rot_l, f64::from(op.scale), 0.0)?;

    let bh = b * h;
    let flat_l = Layout::contiguous((bh, s, d));
    let k_t_l = flat_l.transpose(1, 2)?;
    let scores_storage = q_scaled.matmul(&k_rot, (bh, s, s, d), &flat_l, &k_t_l)?;
    let scores_l = Layout::contiguous((b, h, s, s));

    let (combined_mask_storage, combined_mask_l) = match half_window {
        Some(hw) => {
            let (band_storage, band_l) = build_band_storage(s, hw, s1.dtype(), &device)?;
            let mask_target = (b, 1usize, s, s);
            let mask_bc_l = l3.broadcast_as(mask_target)?;
            let band_bc_l = band_l.broadcast_as(mask_target)?;
            let combined =
                s3.binary_impl::<candle_core::op::Add>(&band_storage, &mask_bc_l, &band_bc_l)?;
            (combined, Layout::contiguous(mask_target))
        }
        None => (s3.try_clone(l3)?, l3.clone()),
    };

    let softmax_op = SoftmaxLastDimFused::new(op.fully_masked);
    let (p_storage, p_shape) = CustomOp2::cuda_fwd(
        &softmax_op,
        &scores_storage,
        &scores_l,
        &combined_mask_storage,
        &combined_mask_l,
    )?;
    let p_flat_l = Layout::contiguous((bh, s, s));
    let v_flat_l = Layout::contiguous((bh, s, d));
    let _ = p_shape;
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
