//! CUDA forward for [`crate::ops::LowRankResidualLinear`] — the SAME six-step
//! sequence `ops::low_rank_residual_linear`'s CPU `cpu_fwd` runs (see that module's
//! doc for the full rounding enumeration), issued at the storage level
//! with NO new `.cu` kernel: `BackendStorage::matmul`/`to_dtype` (cuBLAS
//! and candle's own `cast_*` kernels, already compiled into every candle
//! build) for the three GEMMs and the casts, and
//! [`crate::ops::DropoutFused`]/[`crate::ops::ScaledCastAdd`]'s OWN
//! `cuda_fwd` methods called directly for dropout and the epilogue —
//! exactly the same launchers `crate::cuda::dropout`/
//! `crate::cuda::scaled_cast_add` already ship, reused rather than
//! duplicated. `ab`'s row-packed `[in + out, rank]` layout (see
//! `ops::low_rank_residual_linear`'s module doc) means the `A^T`/`B` slices below are
//! zero-copy `Layout::narrow(0, ..)` views, not materialized buffers —
//! this file no longer issues its own `to_dtype` gather-copy for them (an
//! EARLIER, column-packed layout required one; it failed cuBLAS's
//! `gemm_config` admissibility check on-device — `MatMulNonContiguous` —
//! which is exactly why the pack layout changed, not merely a style
//! preference).

use candle_core::backend::BackendStorage;
use candle_core::{
    CudaStorage, CustomOp1, CustomOp2, CustomOp3, DType, Error, Layout, Result, Shape,
};

use crate::ops::low_rank_residual_linear::materialize_contiguous_if_needed;
use crate::ops::{DropoutFused, LowRankResidualLinear, ScaledCastAdd};

#[allow(clippy::too_many_arguments)]
pub(crate) fn cuda_fwd(
    op: &LowRankResidualLinear,
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
    s3: &CudaStorage,
    l3: &Layout,
) -> Result<(CudaStorage, Shape)> {
    let m = op.flatten_x(l1)?;
    op.check_w_and_ab(l2, l3.dims(), s3.dtype())?;
    if s1.dtype() != s2.dtype() {
        return Err(Error::DTypeMismatchBinaryOp {
            lhs: s1.dtype(),
            rhs: s2.dtype(),
            op: op.name(),
        });
    }
    if !matches!(s1.dtype(), DType::F32 | DType::BF16) {
        return Err(Error::UnsupportedDTypeForOp(s1.dtype(), op.name()));
    }
    for (l, what) in [(l2, "w"), (l3, "ab")] {
        if l.contiguous_offsets().is_none() {
            return Err(Error::RequiresContiguous {
                op: match what {
                    "w" => "low_rank_residual_linear(w)",
                    _ => "low_rank_residual_linear(ab)",
                },
            });
        }
    }

    // `x` may be a non-contiguous (e.g. transposed/narrowed) view;
    // materialize it at THIS op's storage level rather than refusing —
    // see `ops::low_rank_residual_linear::materialize_contiguous_if_needed`'s doc for
    // why only `x` (never `w`/`ab`) gets this treatment.
    let x_owned = materialize_contiguous_if_needed(s1, l1)?;
    let (s1, l1): (&CudaStorage, Layout) = match &x_owned {
        Some((owned, contig_l)) => (owned, contig_l.clone()),
        None => (s1, l1.clone()),
    };
    let l1 = &l1;

    let inf = op.in_features;
    let outf = op.out_features;
    let r = op.rank;

    // Step 1: base = x @ w^T — the same (b=1, m, out, in) config and
    // transposed-weight layout `candle_nn::Linear::forward` issues, so
    // cuBLAS picks the identical kernel.
    let x2d_l = Layout::contiguous_with_offset((m, inf), l1.start_offset());
    let w_t_l = l2.transpose(0, 1)?;
    let base_storage = s1.matmul(s2, (1, m, outf, inf), &x2d_l, &w_t_l)?;
    let base_l = Layout::contiguous((m, outf));

    // Step 2: x32 = to_dtype(x, F32).
    let x32_storage = s1.to_dtype(&x2d_l, DType::F32)?;
    let x32_l = Layout::contiguous((m, inf));

    // Step 3: xd = dropout(x32), via DropoutFused's OWN cuda_fwd.
    let (xd_storage, xd_l) = match &op.dropout {
        Some(key) => {
            let dropout_op = DropoutFused::new(key.seed, key.layer_id, key.forward_idx, key.p)?;
            let (s, shape) = CustomOp1::cuda_fwd(&dropout_op, &x32_storage, &x32_l)?;
            (s, Layout::contiguous(shape))
        }
        None => (x32_storage, x32_l),
    };

    // `ab`'s row-packed layout: the first `inf` rows ARE `A^T` (`[in,
    // rank]`) and the remaining `outf` rows ARE `B` (`[out, rank]`) — both
    // dim-0 narrows of a contiguous matrix, hence themselves contiguous
    // with NO copy (see `ops::low_rank_residual_linear`'s module doc's "packed-`ab`
    // GEMM eligibility problem" section for the `gemm_config`
    // admissibility argument this relies on).
    let a_t_l = l3.narrow(0, 0, inf)?;
    let b_l = l3.narrow(0, inf, outf)?;
    let b_t_l = b_l.transpose(0, 1)?;

    // Step 4: h = xd @ A^T — `a_t_l` used directly, zero-copy.
    let h_storage = xd_storage.matmul(s3, (1, m, r, inf), &xd_l, &a_t_l)?;
    let h_l = Layout::contiguous((m, r));

    // Step 5: delta = h @ B^T — `b_t_l` used directly, zero-copy.
    let delta_storage = h_storage.matmul(s3, (1, m, outf, r), &h_l, &b_t_l)?;
    let delta_l = Layout::contiguous((m, outf));

    // Step 6: out = base + cast(delta * scale), via ScaledCastAdd's OWN
    // cuda_fwd.
    let epilogue = ScaledCastAdd::new(f64::from(op.scale));
    let (out_storage, _flat_shape) =
        CustomOp2::cuda_fwd(&epilogue, &base_storage, &base_l, &delta_storage, &delta_l)?;

    Ok((out_storage, op.output_shape(l1)))
}
