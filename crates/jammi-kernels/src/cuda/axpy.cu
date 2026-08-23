// axpy.cu — y' = alpha * x + y, elementwise. Compiled to PTX only when the
// `cuda` feature is active (see ../../build.rs); the pinned build flags
// (sm_80 baseline, no -use_fast_math) live there, not here — this file is
// architecture-agnostic CUDA C++.
//
// Domain: contiguous, identically-shaped, identically-dtyped inputs only.
// The Rust glue (../cuda.rs) checks shape/dtype/contiguity before a launch;
// this kernel assumes a flat linear index and does not itself re-validate.
#include <cuda_bf16.h>
#include <cstddef>

extern "C" __global__ void axpy_f32(
    const float alpha,
    const float *x,
    const float *y,
    float *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = alpha * x[i] + y[i];
    }
}

// bf16 accumulates in f32 (one round to bf16 on the way out) — the same
// accumulation semantics as the CPU BF16 arm (crate::ops::axpy), a
// deliberate precision choice documented there, not an accident of the
// native bf16 ALU's lower precision.
extern "C" __global__ void axpy_bf16(
    const float alpha,
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *y,
    __nv_bfloat16 *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float xv = __bfloat162float(x[i]);
        float yv = __bfloat162float(y[i]);
        out[i] = __float2bfloat16(alpha * xv + yv);
    }
}
