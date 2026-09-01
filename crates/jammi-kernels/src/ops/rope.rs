//! Fused RoPE (rotary position embedding) rotate-half forward + backward.
//!
//! `out = x*cos + rotate_half(x)*sin`, `rotate_half(x) = cat(-x[..,half:],
//! x[..,:half])` — reproduced bit-for-bit from the EXISTING eager
//! composition (`RotaryEmbedding::apply` in `jammi-encoders`' ModernBERT
//! port) as a single elementwise `CustomOp3`, replacing that ~12-op chain
//! (two `narrow`s, a `neg`, a `Tensor::cat` copy, two broadcast muls, an
//! add) with one kernel launch per Q/K application.
//!
//! ## The table convention this op relies on
//!
//! `cos`/`sin` are `[period, hidden]` tables (`hidden` == `head_dim`; any
//! leading dimensions of size 1 are also accepted — only the LAST
//! dimension and the total element count matter, see [`rope_dims`]) whose
//! `hidden`-wide row is built by computing each of the `hidden/2` angles
//! ONCE and writing it TWICE — at column `i` and again at column `i +
//! hidden/2` (`RotaryEmbedding::new`, `jammi-encoders/src/modernbert.rs`)
//! — so `cos[.., i] == cos[.., i + half]`, and likewise for `sin`. This op
//! indexes `cos`/`sin` at the SAME column as `x` (no half-splitting on the
//! table side) — it only computes the intended rotation BECAUSE the table
//! is redundant in exactly this way. A caller supplying a non-redundant
//! (true `[period, half]`) table would get a silently wrong rotation —
//! this is a documented, load-bearing premise of this op's domain, not a
//! simplification to remove later.
//!
//! ## Broadcast without a broadcast: `period`
//!
//! `x` is `[<any leading dims>, hidden]`; flattening those leading dims
//! against `hidden` gives `total_rows = x.elem_count() / hidden` "rows".
//! `cos`/`sin` cover only `period = cos.elem_count() / hidden` of those
//! rows (ModernBERT's actual shapes: `x` is `[batch, heads, seq,
//! head_dim]`, `cos`/`sin` cover just the `seq` axis and repeat across
//! `batch`/`heads`) — row `r`'s table row is `r % period`, a compile-time
//! candle broadcast (`x.broadcast_mul`) reproduced by hand here because a
//! raw-pointer `CustomOp` has no access to candle's own broadcasting
//! machinery.
//!
//! `total_rows % period == 0` is NECESSARY but NOT SUFFICIENT to make
//! `row % period` walk the right axis — e.g. `x = [3, 2, hidden]` against
//! a `period = 3` table has `total_rows = 6`, a clean multiple of `3`,
//! but row-major flattening makes `row % period` walk `x`'s size-2 axis,
//! not its size-3 one, silently reading the wrong table row for half the
//! rows. [`rope_dims`] therefore enforces the SUFFICIENT condition: the
//! axis immediately before `hidden` (`x.dims()[rank-2]`) must equal
//! `period` exactly — a real domain restriction (a caller with a
//! genuinely irregular row/table pairing is refused, not silently
//! misindexed), enforced on every call, both devices.
//!
//! This is also WHY `RotaryEmbedding::apply_training` in `jammi-encoders`
//! pays a real `x.contiguous()` copy on its hot path: Q/K reach RoPE as a
//! `transpose(1, 2)` VIEW whose axis order this op's `%period` model
//! cannot address without materializing it into the row-major layout the
//! model assumes. A future generalization — replacing the single
//! `period` derived from `cos`/`sin`'s own shape with an explicit
//! `rows_per_table_row: usize` construction parameter decoupled from
//! `cos`/`sin`'s literal shape — could in principle let the op walk a
//! transposed layout's strides directly and remove that copy; not done
//! here (this commit's scope is the rotate-half math, not a strided
//! index walk on top of it — see `layout_walk.rs`'s `Axpy`-only precedent
//! for why that is a deliberate, separate design decision).
//!
//! ## `bwd`: RoPE with the sign of `sin` flipped
//!
//! Pairing column `j < half` with column `j + half`, this op's forward is
//! the 2x2 rotation `[out_j; out_{j+half}] = [[cos, -sin], [sin, cos]] *
//! [x_j; x_{j+half}]` (using `cos`/`sin`'s column-`j` value for both, per
//! the table redundancy above). Its Jacobian is that same orthogonal
//! matrix, so `dx = J^T * dy` uses the TRANSPOSE — the rotation by
//! `-theta`, `[[cos, sin], [-sin, cos]]`. Working the algebra through this
//! op's actual `out = x*cos + rotate_half(x)*sin` formulation (rather than
//! the paired 2x2 form) shows that applying THIS SAME forward computation
//! to `dy` with `sin` negated computes exactly `dx` — no permutation of
//! `dy` needed. [`RopeFused::negate_sin`] exists so `bwd` can reuse this
//! one `KernelOp` for that (the flash-attn `conjugate=True` /
//! TransformerEngine `-shared_mem_sin` precedent cited in the
//! fused-kernels plan), rather than a second kernel — mirroring how
//! `LayerNormFused::bwd` dispatches into internal helper `KernelOp`s
//! (`jammi_kernels::ops::layer_norm`'s module doc) instead of composing
//! ordinary `Tensor` ops.
//!
//! `cos`/`sin` are true external constant tables in every call site this
//! crate ships (`Tensor::from_vec` at construction, never wrapped in
//! `Var`) — `dcos`/`dsin` are therefore `None` in practice. `bwd` checks
//! `cos.track_op()`/`sin.track_op()` directly (no extra construction-data
//! flag needed — unlike `dgamma_needed`, `bwd` already receives `cos`/
//! `sin` themselves as arguments, so there is nothing to freeze ahead of
//! time) and computes a REAL gradient — via ordinary `Tensor` composition,
//! not a further fused kernel, since this path is provably dead in every
//! call site today (the same "correctness over micro-optimization" choice
//! `Axpy::bwd` documents) — rather than hardcoding `None` forever and
//! risking the exact silent-missing-gradient landmine `LayerNormFused`'s
//! doc warns a hardcoded `dgamma_needed = false` would have been.
//! `track_op()`, NOT `is_variable()` alone: an earlier version of this
//! check used `is_variable()`, reasoning that `cos`/`sin` are "genuine
//! leaf construction data with no upstream op, never an intermediate on a
//! path to a `Var`" — sound for every call site's LITERAL `cos`/`sin`
//! handle, but NOT for a caller who derives its own table from a `Var`
//! through an intermediate op (e.g. `crate::ops::rope_positions`'s ragged
//! arm gathers a per-row table via `Tensor::index_select` before ever
//! calling a fused op) and hands the RESULT here: that result has
//! `is_variable() == false` (it is not itself a `Var`) but `track_op() ==
//! true` (it carries an `Op`, and — since candle's own `sorted_nodes`
//! walk, `backprop.rs`, recurses through `Op::IndexSelect`'s left operand
//! — that `Op` chain DOES reach a `Var`), so the old `is_variable()`-only
//! check silently returned `None` for a slot candle's own backward walk
//! still expected an entry for, panicking downstream at `backprop.rs:175`
//! ("grad not populated") instead of computing the real gradient this
//! `bwd` is fully able to produce. `rope_grad_table` (below) never reads
//! `table`'s VALUES — only its `elem_count()`/`shape()` — so widening the
//! gate to `track_op()` costs nothing and is correct regardless of
//! whether `arg2`/`arg3` is itself a `Var` or a tracked intermediate on a
//! path to one; this is the SAME predicate-hole class
//! `low_rank_residual_linear.rs`'s and `jammi-lora`'s `frozen_weight_gate`
//! already fixed for their own `w`/`weight` slots.
//!
//! ## Domain (family D)
//!
//! `x`, `cos`, `sin` must be fully contiguous (`contiguous_offsets()`,
//! same idiom as `LayerNormFused` and for the same reason: this op's
//! `period`-broadcast row grouping needs a well-defined flat linear
//! index, which a raw-pointer kernel cannot recover from an arbitrary
//! strided view). `hidden` (`head_dim`) must be even (rotate-half splits
//! it into two equal halves) and is enforced identically on CPU and CUDA.
//! `hidden == 0` degenerates to an empty output (nothing to rotate), the
//! same "zero-length last dim implies zero elements" argument
//! `LayerNormFused`'s `hidden == 0` case documents. CPU supports F32 and
//! BF16 (RoPE's real training dtypes; matches `LayerNormFused`'s CPU
//! domain deliberately, for the same reason: the profiled workload never
//! needs F64 here), plus F16 as a disclosed, temporary CPU-only
//! oracle-reference arm (`rope_fwd_f16` below — no CUDA F16 dispatch arm
//! exists yet, so admission is not widened to F16 until one does; see
//! `docs/maintainer/cuda-kernel-guide.md`'s per-op f16 reference-regime
//! table).

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, Error, Layout, Result, Shape, Tensor, D};
use half::{bf16, f16};

/// The largest `hidden` (`head_dim`) the CUDA kernel accepts.
///
/// A conservative, VALIDATED ceiling — NOT a hardware constraint, exactly
/// the same status `layer_norm::MAX_HIDDEN` documents: the CUDA kernel
/// below is a plain grid-stride elementwise pass with no per-row shared
/// memory at all (unlike LayerNorm's block-per-row reduction), so nothing
/// about the kernel's own resource usage caps `hidden`. The refusal above
/// this constant exists because oracle coverage stops here (ModernBERT-
/// large's `head_dim = 64` with 4x headroom over the profiled workload),
/// not because a larger `head_dim` would compute the wrong answer.
/// Enforced only on the CUDA arm (`crate::cuda::rope`); the CPU arm has no
/// such ceiling but re-exports this constant so a call site can apply ONE
/// domain check regardless of device.
pub const MAX_HEAD_DIM: usize = 256;

/// Fused rotate-half RoPE. See the module doc for the full design.
#[derive(Debug, Clone, Copy)]
pub struct RopeFused {
    /// When `true`, `sin` is negated before use in the elementwise
    /// formula (`out = x*cos + rotate_half(x)*(-sin)`) — how `bwd` reuses
    /// this exact `KernelOp` to compute `dx` from `grad_res`. See the
    /// module doc's "`bwd`: RoPE with the sign of `sin` flipped" section.
    pub negate_sin: bool,
}

impl RopeFused {
    pub fn new(negate_sin: bool) -> Self {
        Self { negate_sin }
    }
}

impl super::sealed::Sealed for RopeFused {}

/// Validates and derives `(hidden, period, total_rows)` shared by every
/// arm (CPU, CUDA glue) of this op — see the module doc's "Broadcast
/// without a broadcast" section for what `period`/`total_rows` mean.
/// `hidden == 0` is signalled via `period == 0` in the returned tuple; the
/// caller checks that and takes the empty fast path rather than dividing
/// by it. `pub(crate)`: `crate::cuda::rope` imports this exact check
/// rather than re-deriving it (the same "shared, not duplicated" choice
/// `ops::softmax::softmax_dims` and `ops::layer_norm::hidden_of` make).
pub(crate) fn rope_dims(
    l_x: &Layout,
    l_cos: &Layout,
    l_sin: &Layout,
    op: &'static str,
) -> Result<(usize, usize, usize)> {
    let hidden = *l_x.dims().last().ok_or_else(|| {
        Error::Msg(format!(
            "{op}: input must have rank >= 1 to define a last (head_dim) dimension"
        ))
    })?;
    if l_cos.dims() != l_sin.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l_cos.shape().clone(),
            rhs: l_sin.shape().clone(),
            op,
        });
    }
    let Some(&cos_last) = l_cos.dims().last() else {
        return Err(Error::Msg(format!(
            "{op}: cos/sin must have rank >= 1 to define a last (head_dim) dimension"
        )));
    };
    if cos_last != hidden {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l_x.shape().clone(),
            rhs: l_cos.shape().clone(),
            op,
        });
    }
    if hidden == 0 {
        return Ok((0, 0, 0));
    }
    if !hidden.is_multiple_of(2) {
        return Err(Error::Msg(format!(
            "{op}: head_dim={hidden} must be even — rotate-half splits it into two \
             equal halves"
        )));
    }
    let cos_elems = l_cos.shape().elem_count();
    if cos_elems == 0 || !cos_elems.is_multiple_of(hidden) {
        return Err(Error::Msg(format!(
            "{op}: cos/sin element count {cos_elems} is not a positive multiple of \
             head_dim={hidden}"
        )));
    }
    let period = cos_elems / hidden;
    // SUFFICIENT check (not merely "a multiple of"): the axis immediately
    // before `hidden` (`x.dims()[rank-2]`) must equal `period` exactly,
    // UNLESS `period == 1` (a single table row broadcasting over every
    // row of `x`, regardless of that axis's size — `row % 1 == 0` for
    // every `row`, so there is no possible "wrong axis" for a one-row
    // table to walk; this is the only period value that is safe
    // independent of `x`'s shape). `total_rows % period == 0` alone is
    // NECESSARY but not sufficient for `period > 1` — e.g. `x = [3, 2,
    // hidden]` against a `cos`/`sin` table of `period = 3` has
    // `total_rows = 6`, a clean multiple of `3`, but row-major
    // flattening makes `row % period` walk the WRONG axis (`x`'s size-2
    // axis, not its size-3 one), silently reading table row 0 for x-rows
    // that should read table row 1 and vice versa — a confident wrong
    // number, not a shape error, which is exactly the family-D failure
    // this op's domain must not admit. Requiring the axis just before
    // `hidden` to equal `period` is what makes `row % period` (this
    // file's math below, and the CUDA kernel's identical indexing) provably
    // correct: for any row-major `[d0, ..., d_{k-2}, d_{k-1}, hidden]`
    // layout, `row = flat_idx / hidden` ranges over `d0 * ... * d_{k-1}`
    // and `row % d_{k-1}` is EXACTLY the index along `d_{k-1}` — the
    // fastest-varying dimension among the non-`hidden` axes — which only
    // coincides with "the table's own position axis" when `d_{k-1}`
    // (`x.dims()[rank-2]`) IS `period` (or `period` is the trivial `1`).
    let x_dims = l_x.dims();
    let axis_before_hidden = if x_dims.len() >= 2 {
        x_dims[x_dims.len() - 2]
    } else {
        // Rank-1 `x` (`[hidden]` alone, no outer axis at all): there is
        // exactly one row, so it can only correctly pair with a
        // single-row (`period == 1`) table — which the `period == 1`
        // exemption below already covers.
        1
    };
    if period != 1 && axis_before_hidden != period {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l_x.shape().clone(),
            rhs: l_cos.shape().clone(),
            op,
        });
    }
    let x_elems = l_x.shape().elem_count();
    let total_rows = x_elems / hidden;
    Ok((hidden, period, total_rows))
}

impl CustomOp3 for RopeFused {
    fn name(&self) -> &'static str {
        "rope_fused"
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
        let (hidden, period, total_rows) = rope_dims(l1, l2, l3, op)?;
        if s1.dtype() != s2.dtype() || s1.dtype() != s3.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op,
            });
        }
        if hidden == 0 {
            // `s1`'s dtype agreement with `s2`/`s3` is already checked
            // above, so passing `s1` for both of `super::empty_like`'s
            // storage arguments is safe: its dtype-mismatch arm can never
            // fire for `(s1, s1)`. `super::empty_like` (shared with
            // `LayerNormFused`/`SoftmaxLastDimFused`) already returns
            // `l1.shape().clone()`, which for this call IS the correct
            // output shape (`RopeFused`'s output is always shaped like
            // `x`) — no separate shape override needed, unlike the CUDA
            // glue's OWN unary empty-alloc path (`cuda::alloc_empty`),
            // which is not tied to `LayerNormFused`/`SoftmaxLastDimFused`'s
            // two-storage dtype check and stays a distinct helper.
            return super::empty_like(s1, s1, l1, op);
        }
        let (x1, x2) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op })?;
        let (c1, c2) = l2
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op })?;
        let (s_1, s_2) = l3
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op })?;
        let sign = if self.negate_sin { -1.0 } else { 1.0 };
        match (s1, s2, s3) {
            (CpuStorage::F32(x), CpuStorage::F32(cos), CpuStorage::F32(sin)) => {
                let out = rope_fwd_f32(
                    &x[x1..x2],
                    &cos[c1..c2],
                    &sin[s_1..s_2],
                    total_rows,
                    period,
                    hidden,
                    sign,
                );
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::BF16(x), CpuStorage::BF16(cos), CpuStorage::BF16(sin)) => {
                let out = rope_fwd_bf16(
                    &x[x1..x2],
                    &cos[c1..c2],
                    &sin[s_1..s_2],
                    total_rows,
                    period,
                    hidden,
                    sign,
                );
                Ok((CpuStorage::BF16(out), l1.shape().clone()))
            }
            (CpuStorage::F16(x), CpuStorage::F16(cos), CpuStorage::F16(sin)) => {
                let out = rope_fwd_f16(
                    &x[x1..x2],
                    &cos[c1..c2],
                    &sin[s_1..s_2],
                    total_rows,
                    period,
                    hidden,
                    sign,
                );
                Ok((CpuStorage::F16(out), l1.shape().clone()))
            }
            (s1, _, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), op)),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        crate::cuda::rope::cuda_fwd(self.negate_sin, s1, l1, s2, l2, s3, l3)
    }

    /// `dx` is ALWAYS `Some` (same rule as `LayerNormFused`'s `dx` slot —
    /// `x` may be an intermediate on a path to a `Var`, see `Axpy`'s doc on
    /// this exact hazard) — computed by reusing THIS op with `sin`
    /// negated (module doc). `dcos`/`dsin` check `track_op()` (NOT
    /// `is_variable()` alone — module doc's "`bwd`: RoPE with the sign of
    /// `sin` flipped" section explains why the narrower check is a real
    /// predicate hole, not just a stylistic difference) directly on the
    /// actual `cos`/`sin` arguments `bwd` already receives (no separate
    /// construction-data flag is needed the way `dgamma_needed` is).
    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        arg3: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let dx = super::apply3(
            grad_res,
            arg2,
            arg3,
            RopeFused {
                negate_sin: !self.negate_sin,
            },
        )?;
        let sign = if self.negate_sin { -1.0 } else { 1.0 };
        let dcos = if arg2.track_op() {
            Some(rope_grad_table(arg1, grad_res, arg2, false, 1.0)?)
        } else {
            None
        };
        let dsin = if arg3.track_op() {
            Some(rope_grad_table(arg1, grad_res, arg3, true, sign)?)
        } else {
            None
        };
        Ok((Some(dx), dcos, dsin))
    }
}

/// `rotate_half(x) = cat(-x[..,half:], x[..,:half])`, via ordinary
/// `Tensor` ops — used only by [`rope_grad_table`]'s dead-in-practice
/// `dsin` path (see the module doc), not the fused forward/`dx` kernel.
fn rotate_half_tensor(x: &Tensor) -> Result<Tensor> {
    let hidden = x.dim(D::Minus1)?;
    let half = hidden / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// `d(out)/d(table)` summed over the broadcast ("period") axis, for
/// whichever of `cos`/`sin` `table` is (selected by `is_sin`, NOT by
/// `sign` — the two are independent: `is_sin` picks the coefficient
/// SHAPE (`x` for `cos`, `rotate_half(x)` for `sin`), `sign` is a
/// separate multiplier that is `-1.0` exactly when this op instance
/// negates `sin`, per the module doc). Ordinary `Tensor` composition
/// (reshape + sum), not a further fused kernel — deliberately, since
/// `cos`/`sin` are never `Var`s in any call site this crate ships (see
/// the module doc); this exists so a future caller that DID make them
/// trainable gets a correct gradient rather than a silently-`None` one,
/// without over-investing a dedicated kernel in a path with no current
/// exerciser.
fn rope_grad_table(
    x: &Tensor,
    grad_res: &Tensor,
    table: &Tensor,
    is_sin: bool,
    sign: f64,
) -> Result<Tensor> {
    let hidden = x.dim(D::Minus1)?;
    let coeff = if is_sin {
        (rotate_half_tensor(x)? * sign)?
    } else {
        x.clone()
    };
    let term = (grad_res * coeff)?;
    let period = table.elem_count() / hidden;
    let total = term.elem_count();
    let rows_outer = total / (period * hidden);
    term.reshape((rows_outer, period, hidden))?
        .sum(0)?
        .reshape(table.shape().clone())
}

/// Fixed fold order (family J): rows walked `0..total_rows` in ascending
/// order, columns `0..hidden` within each row — a given `(x, cos, sin)`
/// triple always yields the same output bit-for-bit. No reduction is
/// performed here (this op is purely elementwise, unlike LayerNorm), so
/// determinism follows directly from the absence of any accumulation
/// order to fix.
pub(crate) fn rope_fwd_row_f32(x: &[f32], cos: &[f32], sin: &[f32], sign: f32, out: &mut [f32]) {
    let hidden = x.len();
    let half = hidden / 2;
    for col in 0..hidden {
        let rh = if col < half {
            -x[col + half]
        } else {
            x[col - half]
        };
        out[col] = x[col] * cos[col] + rh * sin[col] * sign;
    }
}

fn rope_fwd_f32(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    total_rows: usize,
    period: usize,
    hidden: usize,
    sign: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; total_rows * hidden];
    for r in 0..total_rows {
        let seq_idx = r % period;
        let xr = &x[r * hidden..(r + 1) * hidden];
        let cr = &cos[seq_idx * hidden..(seq_idx + 1) * hidden];
        let sr = &sin[seq_idx * hidden..(seq_idx + 1) * hidden];
        let outr = &mut out[r * hidden..(r + 1) * hidden];
        rope_fwd_row_f32(xr, cr, sr, sign, outr);
    }
    out
}

/// BF16 accumulates in f32 (the same accumulation semantics as
/// `layer_norm`'s BF16 arms and the CUDA kernel below), rounding to bf16
/// exactly once on the way out.
fn rope_fwd_row_bf16(x: &[bf16], cos: &[bf16], sin: &[bf16], sign: f32, out: &mut [bf16]) {
    let hidden = x.len();
    let half = hidden / 2;
    for col in 0..hidden {
        let xv = x[col].to_f32();
        let rh = if col < half {
            -x[col + half].to_f32()
        } else {
            x[col - half].to_f32()
        };
        let c = cos[col].to_f32();
        let s = sin[col].to_f32();
        out[col] = bf16::from_f32(xv * c + rh * s * sign);
    }
}

fn rope_fwd_bf16(
    x: &[bf16],
    cos: &[bf16],
    sin: &[bf16],
    total_rows: usize,
    period: usize,
    hidden: usize,
    sign: f32,
) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; total_rows * hidden];
    for r in 0..total_rows {
        let seq_idx = r % period;
        let xr = &x[r * hidden..(r + 1) * hidden];
        let cr = &cos[seq_idx * hidden..(seq_idx + 1) * hidden];
        let sr = &sin[seq_idx * hidden..(seq_idx + 1) * hidden];
        let outr = &mut out[r * hidden..(r + 1) * hidden];
        rope_fwd_row_bf16(xr, cr, sr, sign, outr);
    }
    out
}

/// [`rope_fwd_row_bf16`]'s exact twin, substituting `half::f16` —
/// f32-accumulate, round-once, per the per-op f16 reference-regime table
/// (`docs/maintainer/cuda-kernel-guide.md`).
fn rope_fwd_row_f16(x: &[f16], cos: &[f16], sin: &[f16], sign: f32, out: &mut [f16]) {
    let hidden = x.len();
    let half = hidden / 2;
    for col in 0..hidden {
        let xv = x[col].to_f32();
        let rh = if col < half {
            -x[col + half].to_f32()
        } else {
            x[col - half].to_f32()
        };
        let c = cos[col].to_f32();
        let s = sin[col].to_f32();
        out[col] = f16::from_f32(xv * c + rh * s * sign);
    }
}

fn rope_fwd_f16(
    x: &[f16],
    cos: &[f16],
    sin: &[f16],
    total_rows: usize,
    period: usize,
    hidden: usize,
    sign: f32,
) -> Vec<f16> {
    let mut out = vec![f16::ZERO; total_rows * hidden];
    for r in 0..total_rows {
        let seq_idx = r % period;
        let xr = &x[r * hidden..(r + 1) * hidden];
        let cr = &cos[seq_idx * hidden..(seq_idx + 1) * hidden];
        let sr = &sin[seq_idx * hidden..(seq_idx + 1) * hidden];
        let outr = &mut out[r * hidden..(r + 1) * hidden];
        rope_fwd_row_f16(xr, cr, sr, sign, outr);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    fn fused(negate_sin: bool, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        crate::ops::apply3(x, cos, sin, RopeFused::new(negate_sin))
    }

    /// Hand-computed reference, `period = 1`, `hidden = 4`: a single
    /// rotation angle. `x = [1, 2, 3, 4]`, `theta` chosen so `cos = 0.5`,
    /// `sin = sqrt(3)/2` (60 degrees) at every duplicated column pair —
    /// exactly the redundant-table shape this op's domain requires (see
    /// the module doc).
    #[test]
    fn cpu_fwd_f32_matches_hand_computed_rotation() {
        let device = Device::Cpu;
        let c = 0.5f32;
        let s = 3f32.sqrt() / 2.0;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let cos = Tensor::from_slice(&[c, c, c, c], (1, 4), &device).unwrap();
        let sin = Tensor::from_slice(&[s, s, s, s], (1, 4), &device).unwrap();
        let out: Vec<f32> = fused(false, &x, &cos, &sin)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // out[0] = 1*c - 3*s; out[1] = 2*c - 4*s; out[2] = 3*c + 1*s; out[3] = 4*c + 2*s
        let expected = [
            1.0 * c - 3.0 * s,
            2.0 * c - 4.0 * s,
            3.0 * c + 1.0 * s,
            4.0 * c + 2.0 * s,
        ];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "{o} vs {e}");
        }
    }

    /// The sign-flip identity at the same hand-computable fixture: negating
    /// `sin` (what `bwd` reuses this op for) flips the off-diagonal terms'
    /// sign, nothing else — a readable failure if the sign convention
    /// (which half gets the negation) is ever flipped by accident.
    #[test]
    fn negate_sin_flips_only_the_rotate_half_terms_sign() {
        let device = Device::Cpu;
        let c = 0.5f32;
        let s = 3f32.sqrt() / 2.0;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let cos = Tensor::from_slice(&[c, c, c, c], (1, 4), &device).unwrap();
        let sin = Tensor::from_slice(&[s, s, s, s], (1, 4), &device).unwrap();
        let out: Vec<f32> = fused(true, &x, &cos, &sin)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let expected = [
            1.0 * c + 3.0 * s,
            2.0 * c + 4.0 * s,
            3.0 * c - 1.0 * s,
            4.0 * c - 2.0 * s,
        ];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "{o} vs {e}");
        }
    }

    #[test]
    fn period_broadcasts_across_leading_rows() {
        // period = 2 (two "positions"), 3 outer rows sharing the same two
        // table rows via `r % period` — batch=3, seq=2, hidden=2.
        let device = Device::Cpu;
        let hidden = 2;
        let period = 2;
        let outer = 3;
        let cos = Tensor::from_slice(&[1.0f32, 1.0, 0.0, 0.0], (period, hidden), &device).unwrap();
        let sin = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0], (period, hidden), &device).unwrap();
        let xv: Vec<f32> = (0..outer * period * hidden).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&xv, (outer, period, hidden), &device).unwrap();
        let out: Vec<f32> = fused(false, &x, &cos, &sin)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for b in 0..outer {
            // seq_idx 0: cos=[1,1], sin=[0,0] -> out == x (identity rotation)
            let row0 = &xv[(b * period) * hidden..(b * period + 1) * hidden];
            let got0 = &out[(b * period) * hidden..(b * period + 1) * hidden];
            assert_eq!(got0, row0, "batch {b}, seq 0 must be an identity rotation");
        }
    }

    #[test]
    fn hidden_zero_is_a_no_op_not_a_division_by_zero_panic() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[] as &[f32], (3, 0), &device).unwrap();
        let cos = Tensor::from_slice(&[] as &[f32], (1, 0), &device).unwrap();
        let sin = Tensor::from_slice(&[] as &[f32], (1, 0), &device).unwrap();
        let out = fused(false, &x, &cos, &sin).unwrap();
        assert_eq!(out.elem_count(), 0);
    }

    #[test]
    fn odd_head_dim_is_refused_not_silently_truncated() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 3), &device).unwrap();
        let cos = Tensor::from_slice(&[1.0f32, 1.0, 1.0], (1, 3), &device).unwrap();
        let sin = Tensor::from_slice(&[0.0f32, 0.0, 0.0], (1, 3), &device).unwrap();
        let err = fused(false, &x, &cos, &sin).expect_err("odd head_dim must be refused");
        assert!(matches!(err, Error::Msg(_)));
    }

    #[test]
    fn non_contiguous_x_is_refused_not_silently_misread() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (4, 2),
            &device,
        )
        .unwrap()
        .t()
        .unwrap();
        assert!(!x.is_contiguous());
        // `x`'s shape (post-transpose) is `[2, 4]`: the axis before
        // `hidden=4` is `2`, so `cos`/`sin` must have `period == 2` (NOT
        // `1`) to pass the shape/period check and reach the contiguity
        // check this test actually exercises.
        let cos = Tensor::from_slice(&[1.0f32; 8], (2, 4), &device).unwrap();
        let sin = Tensor::from_slice(&[0.0f32; 8], (2, 4), &device).unwrap();
        let err = fused(false, &x, &cos, &sin).expect_err("non-contiguous x must be refused");
        assert!(matches!(err, Error::RequiresContiguous { .. }));
    }

    /// The auditor-flagged misindexing case: `total_rows` (`6`) IS a
    /// clean multiple of `period` (`3`), which the OLD (weaker) check
    /// accepted — but `x`'s axis immediately before `hidden` is `2`, not
    /// `3`, so `row % period` would silently walk the wrong axis (see
    /// `rope_dims`'s doc). Refused, not silently misindexed.
    #[test]
    fn axis_before_hidden_mismatched_with_period_is_refused_even_when_total_rows_divides() {
        let device = Device::Cpu;
        let hidden = 2;
        let x = Tensor::from_slice(&[0.0f32; 3 * 2 * 2], (3, 2, hidden), &device).unwrap();
        let cos = Tensor::from_slice(&[1.0f32; 3 * 2], (3, hidden), &device).unwrap();
        let sin = Tensor::from_slice(&[0.0f32; 3 * 2], (3, hidden), &device).unwrap();
        let err = fused(false, &x, &cos, &sin).expect_err(
            "total_rows (6) divides period (3) but the axis before hidden (2) does not equal \
             period (3) — this must be refused, not silently misindexed",
        );
        assert!(matches!(err, Error::ShapeMismatchBinaryOp { .. }));
    }

    #[test]
    fn total_rows_not_a_multiple_of_period_is_refused() {
        let device = Device::Cpu;
        // period = 3 but total_rows = 2 (not a multiple) — an irregular
        // row/table pairing, refused rather than silently misindexed.
        let x = Tensor::from_slice(&[0.0f32; 4], (2, 2), &device).unwrap();
        let cos = Tensor::from_slice(&[1.0f32; 6], (3, 2), &device).unwrap();
        let sin = Tensor::from_slice(&[0.0f32; 6], (3, 2), &device).unwrap();
        let err = fused(false, &x, &cos, &sin)
            .expect_err("total_rows not a multiple of period must be refused");
        assert!(matches!(err, Error::ShapeMismatchBinaryOp { .. }));
    }

    #[test]
    fn dtype_mismatch_is_refused() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let cos = Tensor::from_slice(&[bf16::from_f32(1.0); 4], (1, 4), &device).unwrap();
        let sin = Tensor::from_slice(&[bf16::from_f32(0.0); 4], (1, 4), &device).unwrap();
        let err = fused(false, &x, &cos, &sin).expect_err("dtype mismatch must be refused");
        assert!(matches!(err, Error::DTypeMismatchBinaryOp { .. }));
    }

    #[test]
    fn bf16_forward_matches_f32_accumulation_rounded_once() {
        let device = Device::Cpu;
        let cf = 0.5f32;
        let sf = 3f32.sqrt() / 2.0;
        let xv = [1.0f32, 2.0, 3.0, 4.0];
        let xb: Vec<bf16> = xv.iter().map(|&v| bf16::from_f32(v)).collect();
        let cb = [bf16::from_f32(cf); 4];
        let sb = [bf16::from_f32(sf); 4];
        let x = Tensor::from_slice(&xb, (1, 4), &device).unwrap();
        let cos = Tensor::from_slice(&cb, (1, 4), &device).unwrap();
        let sin = Tensor::from_slice(&sb, (1, 4), &device).unwrap();
        let out: Vec<bf16> = fused(false, &x, &cos, &sin)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let xf: Vec<f64> = xb.iter().map(|v| v.to_f32() as f64).collect();
        let cff = cb[0].to_f32() as f64;
        let sff = sb[0].to_f32() as f64;
        let expected: Vec<f32> = vec![
            (xf[0] * cff - xf[2] * sff) as f32,
            (xf[1] * cff - xf[3] * sff) as f32,
            (xf[2] * cff + xf[0] * sff) as f32,
            (xf[3] * cff + xf[1] * sff) as f32,
        ];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o.to_f32() - e).abs() < 1e-2, "{o} vs {e}");
        }
    }

    /// F16's exact twin of `bf16_forward_matches_f32_accumulation_
    /// rounded_once` above.
    #[test]
    fn f16_forward_matches_f32_accumulation_rounded_once() {
        let device = Device::Cpu;
        let cf = 0.5f32;
        let sf = 3f32.sqrt() / 2.0;
        let xv = [1.0f32, 2.0, 3.0, 4.0];
        let xh: Vec<f16> = xv.iter().map(|&v| f16::from_f32(v)).collect();
        let ch = [f16::from_f32(cf); 4];
        let sh = [f16::from_f32(sf); 4];
        let x = Tensor::from_slice(&xh, (1, 4), &device).unwrap();
        let cos = Tensor::from_slice(&ch, (1, 4), &device).unwrap();
        let sin = Tensor::from_slice(&sh, (1, 4), &device).unwrap();
        let out: Vec<f16> = fused(false, &x, &cos, &sin)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let xf: Vec<f64> = xh.iter().map(|v| v.to_f32() as f64).collect();
        let cff = ch[0].to_f32() as f64;
        let sff = sh[0].to_f32() as f64;
        let expected: Vec<f32> = vec![
            (xf[0] * cff - xf[2] * sff) as f32,
            (xf[1] * cff - xf[3] * sff) as f32,
            (xf[2] * cff + xf[0] * sff) as f32,
            (xf[3] * cff + xf[1] * sff) as f32,
        ];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o.to_f32() - e).abs() < 1e-2, "{o} vs {e}");
        }
    }

    #[test]
    fn gradcheck_dx_f32_vs_central_finite_differences() {
        let device = Device::Cpu;
        let x0: [f32; 8] = [-2.0, -0.75, -0.1, 0.3, 1.2, 4.0, 0.6, -1.3];
        // hidden = 4, period = 1, two rows sharing the same table row.
        let cos0 = [0.8f32, 0.6, 0.8, 0.6];
        let sin0 = [0.6f32, 0.8, 0.6, 0.8];
        let x = Var::from_tensor(&Tensor::from_slice(&x0, (2, 4), &device).unwrap()).unwrap();
        let cos = Tensor::from_slice(&cos0, (1, 4), &device).unwrap();
        let sin = Tensor::from_slice(&sin0, (1, 4), &device).unwrap();

        let out = fused(false, &x, &cos, &sin).unwrap();
        let grads = out.backward().unwrap();
        let dx: Vec<f32> = grads
            .get(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let sum_fwd = |x: &Tensor| -> f64 {
            fused(false, x, &cos, &sin)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64
        };

        let fd_eps = 2e-3f32;
        let tol = 5e-2f64;
        for i in 0..x0.len() {
            let mut xp = x0;
            xp[i] += fd_eps;
            let mut xm = x0;
            xm[i] -= fd_eps;
            let xp_t = Tensor::from_slice(&xp, (2, 4), &device).unwrap();
            let xm_t = Tensor::from_slice(&xm, (2, 4), &device).unwrap();
            let numeric = (sum_fwd(&xp_t) - sum_fwd(&xm_t)) / (2.0 * fd_eps as f64);
            assert!(
                (numeric - dx[i] as f64).abs() < tol,
                "dx[{i}]: numeric {numeric} vs analytic {}",
                dx[i]
            );
        }
    }

    /// Chain-rule oracle: `x` is an INTERMEDIATE (`w.affine(2, 0)`) on a
    /// path to a `Var` — `is_variable() == false`, exactly the case
    /// `Axpy`'s own regression test exercises. `dx`'s slot must still be
    /// populated and correctly chain through the `affine`.
    #[test]
    fn bwd_chains_through_an_intermediate_non_variable_x() {
        let device = Device::Cpu;
        let w0: [f32; 4] = [0.5, -1.0, 2.0, 0.25];
        let cos0 = [0.8f32, 0.6, 0.8, 0.6];
        let sin0 = [0.6f32, 0.8, 0.6, 0.8];
        let w = Var::from_tensor(&Tensor::from_slice(&w0, (1, 4), &device).unwrap()).unwrap();
        let x = w.affine(2.0, 0.0).unwrap();
        assert!(!x.is_variable());
        let cos = Tensor::from_slice(&cos0, (1, 4), &device).unwrap();
        let sin = Tensor::from_slice(&sin0, (1, 4), &device).unwrap();

        let out = fused(false, &x, &cos, &sin).unwrap();
        let grads = out.backward().unwrap(); // must not panic
        let dw: Vec<f32> = grads
            .get(&w)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Reference: dw = 2 * dx, where dx is this op's own gradient at x0.
        let x_direct =
            Var::from_tensor(&Tensor::from_slice(&w0.map(|v| 2.0 * v), (1, 4), &device).unwrap())
                .unwrap();
        let out2 = fused(false, &x_direct, &cos, &sin).unwrap();
        let grads2 = out2.backward().unwrap();
        let dx: Vec<f32> = grads2
            .get(&x_direct)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, (a, b)) in dw.iter().zip(dx.iter()).enumerate() {
            assert!((a - 2.0 * b).abs() < 1e-4, "dw[{i}]: {a} vs 2*{b}");
        }
    }

    #[test]
    fn cos_sin_gradients_are_none_when_not_variables() {
        let device = Device::Cpu;
        let x0: [f32; 4] = [1.0, -1.0, 2.0, 0.5];
        let cos0 = [0.8f32, 0.6, 0.8, 0.6];
        let sin0 = [0.6f32, 0.8, 0.6, 0.8];
        let x = Var::from_tensor(&Tensor::from_slice(&x0, (1, 4), &device).unwrap()).unwrap();
        let cos = Tensor::from_slice(&cos0, (1, 4), &device).unwrap();
        let sin = Tensor::from_slice(&sin0, (1, 4), &device).unwrap();
        assert!(!cos.is_variable());
        assert!(!sin.is_variable());

        let out = fused(false, &x, &cos, &sin).unwrap();
        let grads = out.backward().unwrap();
        assert!(grads.get(&x).is_some());
        assert!(grads.get(&cos).is_none());
        assert!(grads.get(&sin).is_none());
    }

    #[test]
    fn cos_sin_gradients_are_populated_when_variables() {
        let device = Device::Cpu;
        let x0: [f32; 8] = [1.0, -1.0, 2.0, 0.5, -0.3, 1.7, 0.2, -2.1];
        let cos0 = [0.8f32, 0.6, 0.8, 0.6];
        let sin0 = [0.6f32, 0.8, 0.6, 0.8];
        let x = Tensor::from_slice(&x0, (2, 4), &device).unwrap();
        let cos = Var::from_tensor(&Tensor::from_slice(&cos0, (1, 4), &device).unwrap()).unwrap();
        let sin = Var::from_tensor(&Tensor::from_slice(&sin0, (1, 4), &device).unwrap()).unwrap();

        let out = fused(false, &x, &cos, &sin).unwrap();
        let grads = out.backward().unwrap(); // must not panic
        let dcos: Vec<f32> = grads
            .get(&cos)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let dsin: Vec<f32> = grads
            .get(&sin)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Hand-derived: dcos[j] = sum_rows(dy[r,j]*x[r,j]); ones-seed dy
        // (Tensor::backward's default) makes dy == 1 everywhere, so
        // dcos[j] = sum_rows(x[r,j]), dsin[j] = sum_rows(rotate_half(x)[r,j]).
        let rows = 2;
        let hidden = 4;
        let mut expected_dcos = [0f32; 4];
        let mut expected_dsin = [0f32; 4];
        for r in 0..rows {
            for j in 0..hidden {
                expected_dcos[j] += x0[r * hidden + j];
                let half = hidden / 2;
                let rh = if j < half {
                    -x0[r * hidden + j + half]
                } else {
                    x0[r * hidden + j - half]
                };
                expected_dsin[j] += rh;
            }
        }
        for (a, b) in dcos.iter().zip(expected_dcos.iter()) {
            assert!((a - b).abs() < 1e-4, "dcos: {a} vs {b}");
        }
        for (a, b) in dsin.iter().zip(expected_dsin.iter()) {
            assert!((a - b).abs() < 1e-4, "dsin: {a} vs {b}");
        }
    }

    /// The `track_op()` class fix (audit): `cos`/`sin` derived from a
    /// `Var` through an intermediate op (`* 1.0`, the SAME tracked-but-
    /// not-a-`Var` construction `low_rank_residual_linear.rs`'s own
    /// regression test uses) have `is_variable() == false` but
    /// `track_op() == true` -- an `is_variable()`-only gate at `bwd` would
    /// return `dcos`/`dsin = None` here while candle's OWN `sorted_nodes`
    /// walk (`backprop.rs`) still expects a gradient entry for these
    /// tracked nodes, PANICKING at `backprop.rs:175` ("grad not
    /// populated") instead of this test's clean `Ok` with a real, correct
    /// gradient. Reached through the PUBLIC `RopeFused`/`apply3` surface
    /// (`jammi_kernels::ops::{apply3, RopeFused}`), a real
    /// `Tensor::backward()` call.
    #[test]
    fn cos_sin_gradients_are_populated_when_tracked_but_not_variables() {
        let device = Device::Cpu;
        let x0: [f32; 8] = [1.0, -1.0, 2.0, 0.5, -0.3, 1.7, 0.2, -2.1];
        let cos0 = [0.8f32, 0.6, 0.8, 0.6];
        let sin0 = [0.6f32, 0.8, 0.6, 0.8];
        let x = Tensor::from_slice(&x0, (2, 4), &device).unwrap();
        let cos_var =
            Var::from_tensor(&Tensor::from_slice(&cos0, (1, 4), &device).unwrap()).unwrap();
        let sin_var =
            Var::from_tensor(&Tensor::from_slice(&sin0, (1, 4), &device).unwrap()).unwrap();
        // Tracked (has an Op::Affine) but NOT itself a Var -- the exact
        // cell the panic lived in.
        let cos = (cos_var.as_tensor() * 1.0).unwrap();
        let sin = (sin_var.as_tensor() * 1.0).unwrap();
        assert!(!cos.is_variable() && cos.track_op());
        assert!(!sin.is_variable() && sin.track_op());

        let out = fused(false, &x, &cos, &sin).unwrap();
        let grads = out
            .backward()
            .expect("tracked-non-Var cos/sin must populate real gradients, never panic");
        let dcos: Vec<f32> = grads
            .get(cos_var.as_tensor())
            .expect("the gradient must reach cos_var, the tracked-non-Var cos's own ancestor")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let dsin: Vec<f32> = grads
            .get(sin_var.as_tensor())
            .expect("the gradient must reach sin_var, the tracked-non-Var sin's own ancestor")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Reference: the SAME literal-Var case `cos_sin_gradients_are_
        // populated_when_variables` already proves correct -- the
        // tracked-non-Var path must produce the IDENTICAL numeric
        // gradient (the `* 1.0` intermediate is algebraically a no-op).
        let cos_ref =
            Var::from_tensor(&Tensor::from_slice(&cos0, (1, 4), &device).unwrap()).unwrap();
        let sin_ref =
            Var::from_tensor(&Tensor::from_slice(&sin0, (1, 4), &device).unwrap()).unwrap();
        let out_ref = fused(false, &x, &cos_ref, &sin_ref).unwrap();
        let grads_ref = out_ref.backward().unwrap();
        let dcos_ref: Vec<f32> = grads_ref
            .get(&cos_ref)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let dsin_ref: Vec<f32> = grads_ref
            .get(&sin_ref)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            dcos, dcos_ref,
            "a tracked-non-Var cos's gradient must be bit-identical to the literal-Var case's \
             own gradient (the `* 1.0` intermediate is a no-op)"
        );
        assert_eq!(
            dsin, dsin_ref,
            "a tracked-non-Var sin's gradient must be bit-identical to the literal-Var case's \
             own gradient (the `* 1.0` intermediate is a no-op)"
        );
    }
}
