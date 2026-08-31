use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
use candle_core::{CudaStorage, DType, Error, Layout, Result};

/// The kernel module name PTX functions are loaded under (see
/// `cuda::axpy`'s identical doc on `MODULE_NAME`) — arbitrary, but stable
/// and unique to this op.
const MODULE_NAME: &str = "jammi_kernels_adamw_step";

/// Domain check shared by both kernels here: F32 only (the optimizer is
/// not gated by esc-045's BF16 boundary — see `ops::adamw_step`'s module
/// doc) and every buffer contiguous. Returns each buffer's
/// `[start, end)` element range in its OWN base storage (`contiguous_
/// offsets`, candle's own idiom for a `narrow`'d-but-contiguous view with a
/// nonzero start offset — see `cuda::axpy::cuda_fwd`'s doc on why
/// `is_contiguous()` alone is not sufficient).
fn require_contiguous_f32(
    op: &'static str,
    storages: &[(&CudaStorage, &Layout)],
) -> Result<Vec<(usize, usize)>> {
    let mut ranges = Vec::with_capacity(storages.len());
    for (s, l) in storages {
        if s.dtype() != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(s.dtype(), op));
        }
        let range = l
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op })?;
        ranges.push(range);
    }
    Ok(ranges)
}

/// `m[i] = beta*m[i] + (1-beta)*(g[i] or g[i]^2)`, in place. Mirrors
/// `ops::AdamMomentUpdate::cpu_fwd`'s CPU arm exactly (see that type's doc
/// for the operation order this matches).
pub(crate) fn moment_update_cuda_fwd(
    beta: f64,
    square_grad: bool,
    s1: &mut CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
) -> Result<()> {
    const OP: &str = "adamw_moment_update";
    if l1.dims() != l2.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l2.shape().clone(),
            op: OP,
        });
    }
    let n = l1.shape().elem_count();
    // Dtype AND contiguity are validated UNCONDITIONALLY, before the
    // `n == 0` fast path below -- so an empty, non-contiguous buffer is
    // refused the SAME way a non-empty one would be, matching this op's
    // own documented CUDA domain (`require_contiguous_f32`'s doc: "every
    // buffer contiguous") rather than admitting it silently through the
    // empty fast path (`cpu_fwd` itself never requires contiguity here --
    // it walks `StridedOffsets` -- so this is the CUDA arm's OWN
    // self-consistency, not a match to a `cpu_fwd` requirement). `ranges`
    // is unused by the `n == 0` branch itself -- computed here only so the
    // domain check runs in the same place for both branches.
    let ranges = require_contiguous_f32(OP, &[(&*s1, l1), (s2, l2)])?;
    if n == 0 {
        // Match the CPU arm's documented no-op contract
        // (`ops::adamw_step::tests::empty_tensor_is_a_no_op_not_an_error`):
        // `LaunchConfig::for_num_elems(0)` yields an illegal `(0,1,1)` grid,
        // so this returns before ever building one.
        return Ok(());
    }
    super::check_elem_count_fits_u32(OP, n)?;
    let (o1, o2) = ranges[0];
    let (o1_g, o2_g) = ranges[1];

    let device = s2.device().clone();
    let g = s2.as_cuda_slice::<f32>()?.slice(o1_g..o2_g);
    let mut m = s1.as_cuda_slice_mut::<f32>()?.slice_mut(o1..o2);

    let cfg = super::elemwise_launch_config(n as u32);
    let func = device.get_or_load_custom_func(
        "adamw_moment_update_f32",
        MODULE_NAME,
        super::PTX_ADAMW_STEP,
    )?;
    let beta_f32 = beta as f32;
    let one_minus_beta_f32 = (1.0 - beta) as f32;
    let square_grad_i32: i32 = if square_grad { 1 } else { 0 };
    let mut builder = func.builder();
    builder.arg(&mut m);
    builder.arg(&g);
    builder.arg(&beta_f32);
    builder.arg(&one_minus_beta_f32);
    builder.arg(&square_grad_i32);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
    Ok(())
}

/// TEST-ONLY negative control: launches `adamw_moment_update_f32_fma_
/// contracted_red_control` (see the `.cu` file's doc on that kernel) —
/// deliberately the WRONG, single-rounding-FMA expression, sharing every
/// domain check `moment_update_cuda_fwd` uses so the ONLY difference
/// between the two is which kernel function gets launched. The sole
/// caller is `ops::adamw_step::AdamMomentUpdateFmaContractedRedControl`.
pub(crate) fn moment_update_fma_contracted_red_control_cuda_fwd(
    beta: f64,
    square_grad: bool,
    s1: &mut CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
) -> Result<()> {
    const OP: &str = "adamw_moment_update_fma_contracted_red_control";
    if l1.dims() != l2.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l2.shape().clone(),
            op: OP,
        });
    }
    let n = l1.shape().elem_count();
    // Dtype AND contiguity are validated UNCONDITIONALLY, before the
    // `n == 0` fast path -- see `moment_update_cuda_fwd`'s identical
    // comment above.
    let ranges = require_contiguous_f32(OP, &[(&*s1, l1), (s2, l2)])?;
    if n == 0 {
        return Ok(());
    }
    super::check_elem_count_fits_u32(OP, n)?;
    let (o1, o2) = ranges[0];
    let (o1_g, o2_g) = ranges[1];

    let device = s2.device().clone();
    let g = s2.as_cuda_slice::<f32>()?.slice(o1_g..o2_g);
    let mut m = s1.as_cuda_slice_mut::<f32>()?.slice_mut(o1..o2);

    let cfg = super::elemwise_launch_config(n as u32);
    let func = device.get_or_load_custom_func(
        "adamw_moment_update_f32_fma_contracted_red_control",
        MODULE_NAME,
        super::PTX_ADAMW_STEP,
    )?;
    let beta_f32 = beta as f32;
    let one_minus_beta_f32 = (1.0 - beta) as f32;
    let square_grad_i32: i32 = if square_grad { 1 } else { 0 };
    let mut builder = func.builder();
    builder.arg(&mut m);
    builder.arg(&g);
    builder.arg(&beta_f32);
    builder.arg(&one_minus_beta_f32);
    builder.arg(&square_grad_i32);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
    Ok(())
}

/// `theta[i] = theta[i]*one_minus_lr_lambda - lr*m_hat[i]/(sqrt(v_hat[i])+eps)`,
/// in place. Mirrors `ops::AdamThetaUpdate::cpu_fwd`'s CPU arm exactly.
/// `theta`/`m`/`v` bundle each tensor's `(storage, layout)` pair — the same
/// grouping [`require_contiguous_f32`] already takes as a slice — which is
/// what collapses this function's argument count under
/// `clippy::too_many_arguments`' threshold without changing the domain
/// checks or kernel launch below at all.
pub(crate) fn theta_update_cuda_fwd(
    op: crate::ops::AdamThetaUpdate,
    theta: (&mut CudaStorage, &Layout),
    m: (&CudaStorage, &Layout),
    v: (&CudaStorage, &Layout),
) -> Result<()> {
    const OP: &str = "adamw_theta_update";
    let (s1, l1) = theta;
    let (s2, l2) = m;
    let (s3, l3) = v;
    if l1.dims() != l2.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l2.shape().clone(),
            op: OP,
        });
    }
    if l1.dims() != l3.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l3.shape().clone(),
            op: OP,
        });
    }
    let n = l1.shape().elem_count();
    // Dtype AND contiguity are validated UNCONDITIONALLY, before the
    // `n == 0` fast path -- see `moment_update_cuda_fwd`'s identical
    // comment above.
    let ranges = require_contiguous_f32(OP, &[(&*s1, l1), (s2, l2), (s3, l3)])?;
    if n == 0 {
        return Ok(());
    }
    super::check_elem_count_fits_u32(OP, n)?;
    let (o1, o2) = ranges[0];
    let (o1_m, o2_m) = ranges[1];
    let (o1_v, o2_v) = ranges[2];

    let device = s2.device().clone();
    let m = s2.as_cuda_slice::<f32>()?.slice(o1_m..o2_m);
    let v = s3.as_cuda_slice::<f32>()?.slice(o1_v..o2_v);
    let mut theta = s1.as_cuda_slice_mut::<f32>()?.slice_mut(o1..o2);

    let cfg = super::elemwise_launch_config(n as u32);
    let func = device.get_or_load_custom_func(
        "adamw_theta_update_f32",
        MODULE_NAME,
        super::PTX_ADAMW_STEP,
    )?;
    let one_minus_lr_lambda_f32 = op.one_minus_lr_lambda as f32;
    let scale_m_f32 = op.scale_m as f32;
    let scale_v_f32 = op.scale_v as f32;
    let eps_f32 = op.eps as f32;
    let lr_f32 = op.lr as f32;
    let mut builder = func.builder();
    builder.arg(&mut theta);
    builder.arg(&m);
    builder.arg(&v);
    builder.arg(&one_minus_lr_lambda_f32);
    builder.arg(&scale_m_f32);
    builder.arg(&scale_v_f32);
    builder.arg(&eps_f32);
    builder.arg(&lr_f32);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
    Ok(())
}
