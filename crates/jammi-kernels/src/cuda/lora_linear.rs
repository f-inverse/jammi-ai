//! CUDA forward for [`crate::ops::LoraLinearFused`] — the SAME six-step
//! sequence `ops::lora_linear`'s CPU `cpu_fwd` runs (see that module's
//! doc for the full rounding enumeration), issued at the storage level
//! with NO new `.cu` kernel: `BackendStorage::matmul`/`to_dtype` (cuBLAS
//! and candle's own `cast_*` kernels, already compiled into every candle
//! build) for the three GEMMs and the two casts, and
//! [`crate::ops::DropoutFused`]/[`crate::ops::ScaledCastAdd`]'s OWN
//! `cuda_fwd` methods called directly for dropout and the epilogue —
//! exactly the same launchers `crate::cuda::dropout`/
//! `crate::cuda::scaled_cast_add` already ship, reused rather than
//! duplicated.

use candle_core::backend::BackendStorage;
use candle_core::{
    CudaStorage, CustomOp1, CustomOp2, CustomOp3, DType, Error, Layout, Result, Shape,
};

use crate::ops::{DropoutFused, LoraLinearFused, ScaledCastAdd};

#[allow(clippy::too_many_arguments)]
pub(crate) fn cuda_fwd(
    op: &LoraLinearFused,
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
    for (l, what) in [(l1, "x"), (l2, "w"), (l3, "ab")] {
        if l.contiguous_offsets().is_none() {
            return Err(Error::RequiresContiguous {
                op: match what {
                    "x" => "lora_linear_fused(x)",
                    "w" => "lora_linear_fused(w)",
                    _ => "lora_linear_fused(ab)",
                },
            });
        }
    }

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

    // A/B^T materialized into their own dense buffers before either GEMM
    // touches them — see `ops::lora_linear`'s module doc: a column-slice
    // of the wider packed `ab` buffer fails cuBLAS's `gemm_config`
    // eligibility check (neither its row- nor column-contiguous branch
    // matches a padded row stride), so this cast-to-the-same-dtype gather
    // copy (bit-exact — no numeric operation, just a layout-aware read)
    // is required, not merely convenient.
    let a_view = l3.narrow(1, 0, inf)?;
    let bt_view = l3.narrow(1, inf, outf)?;
    let a_storage = s3.to_dtype(&a_view, DType::F32)?;
    let bt_storage = s3.to_dtype(&bt_view, DType::F32)?;
    let a_l = Layout::contiguous((r, inf));
    let bt_l = Layout::contiguous((r, outf));

    // Step 4: h = xd @ A^T.
    let a_t_l = a_l.transpose(0, 1)?;
    let h_storage = xd_storage.matmul(&a_storage, (1, m, r, inf), &xd_l, &a_t_l)?;
    let h_l = Layout::contiguous((m, r));

    // Step 5: delta = h @ B^T.
    let delta_storage = h_storage.matmul(&bt_storage, (1, m, outf, r), &h_l, &bt_l)?;
    let delta_l = Layout::contiguous((m, outf));

    // Step 6: out = base + cast(delta * scale), via ScaledCastAdd's OWN
    // cuda_fwd.
    let epilogue = ScaledCastAdd::new(f64::from(op.scale));
    let (out_storage, _flat_shape) =
        CustomOp2::cuda_fwd(&epilogue, &base_storage, &base_l, &delta_storage, &delta_l)?;

    Ok((out_storage, op.output_shape(l1)))
}
