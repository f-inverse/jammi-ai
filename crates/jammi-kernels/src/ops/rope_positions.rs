//! Fused RoPE rotate-half on the FlashAttention-2-packed `[total, 3, h,
//! d]` `qkv` buffer — P6 Stage B B3-dense. `RopeFused` (`ops::rope`)
//! cannot take this layout directly: its `rope_dims` requires the axis
//! immediately before `hidden` to equal `period` (or `period == 1`), and
//! for a packed `qkv` that axis is `h` (the head axis), which is neither
//! — applying `RopeFused` here would silently read the wrong table row
//! for every row after the first (`ops::rope::rope_dims`'s own doc names
//! this exact family-D hazard). This op decodes `token = row / (3*h)`
//! (`row = flat_idx / d`) directly, so it walks the correct axis
//! regardless of `h`.
//!
//! ## Scope: dense only (`position = token % seq`)
//!
//! The P6 Stage B v5 contract's general mechanism is `positions[r] = r -
//! cu[seq(r)]` (a per-row lookup table, needed once a batch has real
//! padding and `cu_seqlens` is non-uniform). For the DENSE fast path
//! (`cu_seqlens` uniform, every sequence length `== seq`) that reduces to
//! the closed form `position = token % seq` — the SAME modulo
//! [`super::rope::RopeFused`] already uses, just walking a different
//! axis. This commit implements ONLY the dense closed form (one `seq:
//! usize` field, no positions `Tensor`/device array at all) — the
//! general table form is explicitly future work (the padded regime), not
//! implemented here; a future generalization would add a `positions`
//! argument alongside (or instead of) `seq` without changing this op's
//! per-element math, mirroring how `rope_positions.cu`'s shared
//! `rope_rotate` device function already factors the math out from the
//! indexing.
//!
//! ## V slot pass-through
//!
//! `qkv`'s slot 2 (V) is copied through unchanged — RoPE only ever
//! applies to Q/K (contract v5 §3.6) — because this op's OUTPUT is the
//! single tensor `flash_attention_varlen` consumes directly (no separate
//! V tensor to reassemble later): the packed buffer must remain a valid,
//! complete `qkv` after this op runs.
//!
//! ## `bwd`: the same sign-flip reuse `RopeFused` already established
//!
//! Forward is `out = x*cos + rotate_half(x)*sin` on the Q/K slots (module
//! doc of `ops::rope` derives why); the SAME algebra applies per-element
//! here (V is a pure identity map either direction), so `bwd` reuses this
//! exact `KernelOp` with `negate_sin` flipped — no permutation of
//! `grad_res` needed, exactly `RopeFused::bwd`'s own mechanism.
//!
//! ## Why `KernelOp` (`Copy`), not `StatefulKernelOp`
//!
//! Unlike `crate::ops::flash_attention`'s ops, this op holds no `Saved`
//! field and no device array construction data at all — dense-scope
//! `seq: usize` is plain `Copy` data, so this op fits the crate's
//! ordinary stateless-`KernelOp` family (`super::apply3`), the same as
//! `RopeFused` itself. It has no dependency on `crate::flash`'s FFI
//! boundary either (a pure elementwise transform, no CUTLASS), so it is
//! gated by this crate's plain `cuda` feature, not `flash-attn` — usable
//! (and testable) independent of whether the vendored FlashAttention-2
//! kernels are compiled in.
//!
//! ## Domain (family D)
//!
//! `qkv`: rank 4, `[total, 3, h, d]`, contiguous, `d` even, `total ==
//! qkv.dim(0)`. `cos`/`sin`: `[period, d]` (any leading dims of size 1
//! accepted, same convention as `ops::rope`), contiguous, `period == seq`
//! (or `period == 1`, a single shared table row). `seq == 0` is refused
//! (the modulo is undefined) UNLESS `total == 0` too (the fully
//! degenerate empty-batch case, which takes the `d == 0`-style empty
//! fast path). `d == 0` degenerates to an empty output, same as
//! `ops::rope`'s `hidden == 0` case.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, Error, Layout, Result, Shape, Tensor};
use half::bf16;

const OP: &str = "rope_positions_fused";

/// Fused RoPE rotate-half on the packed `[total, 3, h, d]` buffer. See
/// the module doc.
#[derive(Debug, Clone, Copy)]
pub struct RopePositionsFused {
    /// Dense sequence length (`total = batch * seq`). Every token's
    /// position is `token % seq`. See the module doc's scope note.
    pub seq: usize,
    /// Same convention as [`super::rope::RopeFused::negate_sin`]: `true`
    /// is how `bwd` reuses this forward kernel to compute `dx`.
    pub negate_sin: bool,
}

impl RopePositionsFused {
    pub fn new(seq: usize, negate_sin: bool) -> Self {
        Self { seq, negate_sin }
    }
}

impl super::sealed::Sealed for RopePositionsFused {}

/// Validates and derives `(total, h, d)` shared by every arm (CPU, CUDA
/// glue) of this op. `pub(crate)`: `crate::cuda::rope_positions` imports
/// this exact check rather than re-deriving it (the same "shared, not
/// duplicated" choice `ops::rope::rope_dims` documents for itself).
pub(crate) fn rope_positions_dims(
    l_qkv: &Layout,
    l_cos: &Layout,
    l_sin: &Layout,
    seq: usize,
) -> Result<(usize, usize, usize)> {
    let dims = l_qkv.dims();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(Error::Msg(format!(
            "{OP}: qkv must be rank-4 [total, 3, h, d], got {dims:?}"
        )));
    }
    let total = dims[0];
    let h = dims[2];
    let d = dims[3];

    if l_cos.dims() != l_sin.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l_cos.shape().clone(),
            rhs: l_sin.shape().clone(),
            op: OP,
        });
    }
    let Some(&cos_last) = l_cos.dims().last() else {
        return Err(Error::Msg(format!(
            "{OP}: cos/sin must have rank >= 1 to define a last (head_dim) dimension"
        )));
    };
    if cos_last != d {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l_qkv.shape().clone(),
            rhs: l_cos.shape().clone(),
            op: OP,
        });
    }
    if d == 0 {
        return Ok((total, h, 0));
    }
    if !d.is_multiple_of(2) {
        return Err(Error::Msg(format!(
            "{OP}: head_dim={d} must be even -- rotate-half splits it into two equal halves"
        )));
    }
    let cos_elems = l_cos.shape().elem_count();
    if cos_elems == 0 || !cos_elems.is_multiple_of(d) {
        return Err(Error::Msg(format!(
            "{OP}: cos/sin element count {cos_elems} is not a positive multiple of head_dim={d}"
        )));
    }
    let period = cos_elems / d;
    if period != 1 && period != seq {
        return Err(Error::Msg(format!(
            "{OP}: cos/sin table covers {period} positions, expected exactly seq={seq} \
             (or a single shared row, period=1) -- a table covering a different span would \
             silently index the wrong position for a dense forward"
        )));
    }
    if total > 0 && seq == 0 {
        return Err(Error::Msg(format!(
            "{OP}: seq=0 with a nonempty qkv (total={total}) -- position = token % seq is \
             undefined"
        )));
    }
    // Dense-only scope (module doc): `total` MUST be `batch * seq` for some
    // integer `batch`. `token % seq` is arithmetically well-defined even
    // when `total` is not a multiple of `seq`, but it is then SEMANTICALLY
    // wrong -- the tail rows of the last, incomplete "batch" would wrap
    // into positions that belong to a batch element that never existed,
    // silently misindexing rather than refusing a shape outside this op's
    // domain. Mirrors `rope_dims`'s own "silently misindexed" guard.
    if seq > 0 && !total.is_multiple_of(seq) {
        return Err(Error::Msg(format!(
            "{OP}: total={total} is not a multiple of seq={seq} -- this op's DENSE scope \
             requires total == batch * seq for some integer batch, or `position = token % seq` \
             silently wraps into a batch element that does not exist"
        )));
    }
    Ok((total, h, d))
}

#[allow(clippy::too_many_arguments)]
fn rope_positions_fwd_f32(
    qkv: &[f32],
    cos: &[f32],
    sin: &[f32],
    total: usize,
    h: usize,
    d: usize,
    seq: usize,
    sign: f32,
) -> Vec<f32> {
    let half = d / 2;
    let mut out = vec![0.0f32; total * 3 * h * d];
    for token in 0..total {
        let seq_idx = if seq == 0 { 0 } else { token % seq };
        let table_base = seq_idx * d;
        for slot in 0..3usize {
            for h_idx in 0..h {
                let row_base = ((token * 3 + slot) * h + h_idx) * d;
                if slot == 2 {
                    out[row_base..row_base + d].copy_from_slice(&qkv[row_base..row_base + d]);
                    continue;
                }
                for c in 0..d {
                    let xv = qkv[row_base + c];
                    let rh = if c < half {
                        -qkv[row_base + c + half]
                    } else {
                        qkv[row_base + c - half]
                    };
                    let cc = cos[table_base + c];
                    let ss = sin[table_base + c];
                    out[row_base + c] = xv * cc + rh * ss * sign;
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn rope_positions_fwd_bf16(
    qkv: &[bf16],
    cos: &[bf16],
    sin: &[bf16],
    total: usize,
    h: usize,
    d: usize,
    seq: usize,
    sign: f32,
) -> Vec<bf16> {
    let half = d / 2;
    let mut out = vec![bf16::ZERO; total * 3 * h * d];
    for token in 0..total {
        let seq_idx = if seq == 0 { 0 } else { token % seq };
        let table_base = seq_idx * d;
        for slot in 0..3usize {
            for h_idx in 0..h {
                let row_base = ((token * 3 + slot) * h + h_idx) * d;
                if slot == 2 {
                    out[row_base..row_base + d].copy_from_slice(&qkv[row_base..row_base + d]);
                    continue;
                }
                for c in 0..d {
                    let xv = f32::from(qkv[row_base + c]);
                    let rh = if c < half {
                        -f32::from(qkv[row_base + c + half])
                    } else {
                        f32::from(qkv[row_base + c - half])
                    };
                    let cc = f32::from(cos[table_base + c]);
                    let ss = f32::from(sin[table_base + c]);
                    out[row_base + c] = bf16::from_f32(xv * cc + rh * ss * sign);
                }
            }
        }
    }
    out
}

impl CustomOp3 for RopePositionsFused {
    fn name(&self) -> &'static str {
        OP
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
        let (total, h, d) = rope_positions_dims(l1, l2, l3, self.seq)?;
        if s1.dtype() != s2.dtype() || s1.dtype() != s3.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: OP,
            });
        }
        if d == 0 {
            return super::empty_like(s1, s1, l1, OP);
        }
        let (x1, x2) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: OP })?;
        let (c1, c2) = l2
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: OP })?;
        let (s_1, s_2) = l3
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: OP })?;
        let sign = if self.negate_sin { -1.0 } else { 1.0 };
        match (s1, s2, s3) {
            (CpuStorage::F32(x), CpuStorage::F32(cos), CpuStorage::F32(sin)) => {
                let out = rope_positions_fwd_f32(
                    &x[x1..x2],
                    &cos[c1..c2],
                    &sin[s_1..s_2],
                    total,
                    h,
                    d,
                    self.seq,
                    sign,
                );
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::BF16(x), CpuStorage::BF16(cos), CpuStorage::BF16(sin)) => {
                let out = rope_positions_fwd_bf16(
                    &x[x1..x2],
                    &cos[c1..c2],
                    &sin[s_1..s_2],
                    total,
                    h,
                    d,
                    self.seq,
                    sign,
                );
                Ok((CpuStorage::BF16(out), l1.shape().clone()))
            }
            (s1, _, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), OP)),
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
        crate::cuda::rope_positions::cuda_fwd(self.seq, self.negate_sin, s1, l1, s2, l2, s3, l3)
    }

    /// `dx` (the gradient wrt the packed, pre-rotation `qkv`) reuses THIS
    /// op with `sin` negated -- `RopeFused::bwd`'s exact mechanism,
    /// applies unchanged here (V's identity map is its own transpose).
    /// `dcos`/`dsin`: unlike `RopeFused`, this op does NOT implement a
    /// real table gradient -- `cos`/`sin` are non-`Var` leaf tables in
    /// every call site this op ships behind (the SAME premise
    /// `RopeFused`'s module doc states for itself), and composing one for
    /// this op's packed-buffer-with-V-passthrough indexing is real,
    /// currently-unexercised work. Rather than silently return `None` for
    /// a hypothetical future trainable table (the exact landmine
    /// `LayerNormFused`'s doc warns a hardcoded `false`/`None` would be),
    /// a caller that DOES pass a `Var` `cos`/`sin` gets a typed error, not
    /// a silently-missing gradient.
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
            RopePositionsFused {
                seq: self.seq,
                negate_sin: !self.negate_sin,
            },
        )?;
        if arg2.is_variable() || arg3.is_variable() {
            return Err(Error::Msg(format!(
                "{OP}: cos/sin gradient is not implemented -- every call site this op ships \
                 behind treats cos/sin as non-Var leaf tables (see the module doc); a caller \
                 that made them trainable would need a real dcos/dsin implementation, not the \
                 silently-None gradient this error replaces"
            )));
        }
        let _ = arg1; // domain-only; `dx` above is the whole gradient wrt qkv.
        Ok((Some(dx), None, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::RopeFused;
    use candle_core::{DType, Device};

    fn fused(
        seq: usize,
        negate_sin: bool,
        qkv: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        super::super::apply3(qkv, cos, sin, RopePositionsFused::new(seq, negate_sin))
    }

    /// A real (non-trivial-angle) rotary table, `[period, hidden]`,
    /// duplicated-column convention (`RopeFused`'s own module doc,
    /// re-required here since this op consumes the identical table
    /// shape).
    fn rope_table(period: usize, hidden: usize, theta_base: f64) -> (Tensor, Tensor) {
        let half = hidden / 2;
        let mut cos = vec![0f32; period * hidden];
        let mut sin = vec![0f32; period * hidden];
        for pos in 0..period {
            for i in 0..half {
                let theta = theta_base.powf(-2.0 * (i as f64) / (hidden as f64));
                let angle = pos as f64 * theta;
                let (s, c) = angle.sin_cos();
                cos[pos * hidden + i] = c as f32;
                cos[pos * hidden + i + half] = c as f32;
                sin[pos * hidden + i] = s as f32;
                sin[pos * hidden + i + half] = s as f32;
            }
        }
        let device = Device::Cpu;
        (
            Tensor::from_vec(cos, (period, hidden), &device).unwrap(),
            Tensor::from_vec(sin, (period, hidden), &device).unwrap(),
        )
    }

    /// Packs a `[b, h, s, d]` tensor's own values into slot `slot` of a
    /// FRESH `[b*s, 3, h, d]` `qkv` tensor (the other two slots filled
    /// from `filler`, a distinct tensor of the same shape, so a
    /// pass-through bug on the wrong slot cannot hide behind identical
    /// data). Mirrors the real call site's `qkv.reshape((b*s, 3, h,
    /// d))`-after-`[b, s, 3, h, d]`-view shape, but built explicitly here
    /// (the real call site's `qkv` is ALREADY `[b, s, 3*h*d]` from one
    /// `Wqkv` GEMM; this test constructs the THREE logical tensors a
    /// from-scratch oracle needs instead).
    fn pack_bhsd_into_qkv(x_bhsd: &Tensor, filler: &Tensor, slot: usize) -> Tensor {
        let (b, h, s, d) = x_bhsd.dims4().unwrap();
        let x_bshd = x_bhsd.transpose(1, 2).unwrap().contiguous().unwrap(); // [b,s,h,d]
        let filler_bshd = filler.transpose(1, 2).unwrap().contiguous().unwrap();
        let mut slots = Vec::with_capacity(3);
        for i in 0..3 {
            let src = if i == slot { &x_bshd } else { &filler_bshd };
            slots.push(src.reshape((b * s, 1, h, d)).unwrap());
        }
        Tensor::cat(&slots, 1).unwrap()
    }

    fn unpack_qkv_slot(qkv: &Tensor, slot: usize, b: usize, s: usize) -> Tensor {
        let (total, _, h, d) = qkv.dims4().unwrap();
        assert_eq!(total, b * s);
        qkv.narrow(1, slot, 1)
            .unwrap()
            .reshape((b, s, h, d))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
    }

    fn to_bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// THE oracle (P6 Stage B B3-dense, contract v5 §3.6): `rope_positions`
    /// on the packed layout is bit-identical to `RopeFused` applied to the
    /// SAME data in `[b, h, s, d]` form (the block arm's own operand
    /// shape, `gather_bhsd`'s target), for a real (non-trivial-angle)
    /// table, non-trivial data, b1 AND b8. RED control: sign flipped (the
    /// `negate_sin` this op's own `bwd` reuses) must NOT match.
    fn bit_identity_case(b: usize, h: usize, s: usize, d: usize) {
        bit_identity_case_with_red_control(b, h, s, d, true)
    }

    fn bit_identity_case_with_red_control(
        b: usize,
        h: usize,
        s: usize,
        d: usize,
        check_red_control: bool,
    ) {
        let device = Device::Cpu;
        let n = b * h * s * d;
        // Non-trivial, non-symmetric data: distinct per element, includes
        // negative values (`sin(k)`-derived, never all-zero or constant).
        let xv: Vec<f32> = (0..n).map(|k| (k as f32 * 0.037).sin() * 3.0).collect();
        let x_bhsd = Tensor::from_vec(xv, (b, h, s, d), &device).unwrap();
        let fv: Vec<f32> = (0..n)
            .map(|k| (k as f32 * 0.091 + 1.0).cos() * 2.0)
            .collect();
        let filler_bhsd = Tensor::from_vec(fv, (b, h, s, d), &device).unwrap();
        let (cos, sin) = rope_table(s, d, 10_000.0);

        // Reference: RopeFused directly on [b,h,s,d] (the block arm's
        // operand shape).
        let reference = super::super::apply3(&x_bhsd, &cos, &sin, RopeFused::new(false)).unwrap();

        // rope_positions on the packed [b*s,3,h,d] buffer, slot 0 = q.
        let qkv = pack_bhsd_into_qkv(&x_bhsd, &filler_bhsd, 0);
        let out = fused(s, false, &qkv, &cos, &sin).unwrap();
        let got_q = unpack_qkv_slot(&out, 0, b, s);
        assert_eq!(
            to_bits(&got_q),
            to_bits(&reference),
            "rope_positions on slot 0 (q) must be bit-identical to RopeFused on [b,h,s,d], \
             b={b} h={h} s={s} d={d}"
        );

        // V slot (2) must pass through byte-identical to the filler data
        // packed into it (RoPE never touches V) -- `qkv`'s OWN slot 2
        // (not the output) is the expected value, read before the op runs.
        let got_v = unpack_qkv_slot(&out, 2, b, s);
        let expected_v = unpack_qkv_slot(&qkv, 2, b, s);
        assert_eq!(
            to_bits(&got_v),
            to_bits(&expected_v),
            "rope_positions must pass V (slot 2) through unchanged, b={b} h={h} s={s} d={d}"
        );

        // RED control: negate_sin=true must NOT reproduce the (positive-
        // sign) reference on non-trivial data. Skipped at `s=1`: with a
        // single position, `angle = pos*theta = 0` for every `i`, so
        // `sin` is identically zero and flipping its sign is genuinely a
        // no-op -- not a weakened control, a degenerate domain where the
        // control itself is vacuous (the boundary case this test exists
        // to cover on its OWN, `bit_identical_single_position_s_one`,
        // still gets the full bit-identity check above).
        if check_red_control {
            let out_negated = fused(s, true, &qkv, &cos, &sin).unwrap();
            let got_q_negated = unpack_qkv_slot(&out_negated, 0, b, s);
            assert_ne!(
                to_bits(&got_q_negated),
                to_bits(&reference),
                "RED control: sign-flipped rope_positions must NOT match the reference, \
                 b={b} h={h} s={s} d={d}"
            );
        }

        // Slot 1 (K) must ALSO be bit-identical to `RopeFused` on the SAME
        // data -- RoPE applies to Q *and* K (contract v5 §3.6), and the
        // slot-0 check above says nothing about slot 1: a defect that
        // rotates only slot 0 (leaving K a pass-through, e.g. a
        // `slot == 2` condition mutated to `slot >= 1`) would sail through
        // every assertion above while silently returning an unrotated K to
        // every downstream attention call. Packed independently (`x_bhsd`
        // now at slot 1, `filler_bhsd` at slots 0 and 2) so this cannot
        // hide behind slot 0's already-rotated data.
        let qkv_k = pack_bhsd_into_qkv(&x_bhsd, &filler_bhsd, 1);
        let out_k = fused(s, false, &qkv_k, &cos, &sin).unwrap();
        let got_k = unpack_qkv_slot(&out_k, 1, b, s);
        assert_eq!(
            to_bits(&got_k),
            to_bits(&reference),
            "rope_positions on slot 1 (k) must be bit-identical to RopeFused on [b,h,s,d], \
             b={b} h={h} s={s} d={d}"
        );
        if check_red_control {
            let out_k_negated = fused(s, true, &qkv_k, &cos, &sin).unwrap();
            let got_k_negated = unpack_qkv_slot(&out_k_negated, 1, b, s);
            assert_ne!(
                to_bits(&got_k_negated),
                to_bits(&reference),
                "RED control: sign-flipped rope_positions on slot 1 (k) must NOT match the \
                 reference, b={b} h={h} s={s} d={d}"
            );
        }
    }

    #[test]
    fn bit_identical_to_rope_fused_on_bhsd_b1_s_small() {
        bit_identity_case(1, 2, 5, 4);
    }

    #[test]
    fn bit_identical_to_rope_fused_on_bhsd_b8_s_small() {
        bit_identity_case(8, 3, 7, 6);
    }

    #[test]
    fn bit_identical_to_rope_fused_on_bhsd_head_dim_matches_production() {
        // head_dim=64 (ModernBERT-large's real head_dim), a smaller
        // (b, s) so the CPU test stays fast — the pod leg covers full
        // production (b, s) at this head_dim in bf16.
        bit_identity_case(2, 4, 9, 64);
    }

    #[test]
    fn degenerate_d_zero_is_empty_not_a_panic() {
        let device = Device::Cpu;
        let qkv = Tensor::zeros((3, 3, 2, 0), DType::F32, &device).unwrap();
        let cos = Tensor::zeros((4, 0), DType::F32, &device).unwrap();
        let sin = Tensor::zeros((4, 0), DType::F32, &device).unwrap();
        let out = fused(4, false, &qkv, &cos, &sin).unwrap();
        assert_eq!(out.dims(), qkv.dims());
        assert_eq!(out.elem_count(), 0);
    }

    /// Family D: a table whose period disagrees with `seq` must be
    /// refused, not silently misindexed.
    #[test]
    fn table_period_mismatch_is_refused() {
        let device = Device::Cpu;
        let qkv = Tensor::zeros((6, 3, 2, 4), DType::F32, &device).unwrap(); // total=6, seq should be e.g. 3 or 6
        let (cos, sin) = rope_table(5, 4, 10_000.0); // period=5, matches neither seq=2 nor seq=3
        let err = fused(2, false, &qkv, &cos, &sin).unwrap_err();
        assert!(format!("{err}").contains("cos/sin table covers"));
    }

    /// Family D: `total` not a multiple of `seq` must be refused, not
    /// silently misindexed (`token % seq` is well-defined arithmetic even
    /// then, but semantically wrong -- mirrors `rope_dims`'s
    /// `total_rows_not_a_multiple_of_period_is_refused`). The table's own
    /// period matches `seq` exactly, so this exercises ONLY the new
    /// `total % seq` guard, not the pre-existing period check.
    #[test]
    fn total_not_a_multiple_of_seq_is_refused() {
        let device = Device::Cpu;
        // total=5, seq=2: 5 is not a multiple of 2.
        let qkv = Tensor::zeros((5, 3, 2, 4), DType::F32, &device).unwrap();
        let (cos, sin) = rope_table(2, 4, 10_000.0); // period=2, matches seq=2 exactly
        let err = fused(2, false, &qkv, &cos, &sin).unwrap_err();
        assert!(
            format!("{err}").contains("is not a multiple of seq"),
            "expected the total%seq guard's message, got: {err}"
        );
    }

    /// Family D boundary: `seq=0` with a nonempty qkv is refused, not a
    /// division-by-zero / modulo-by-zero panic.
    #[test]
    fn seq_zero_with_nonempty_qkv_is_refused_not_a_panic() {
        let device = Device::Cpu;
        let qkv = Tensor::zeros((2, 3, 1, 4), DType::F32, &device).unwrap();
        let (cos, sin) = rope_table(1, 4, 10_000.0);
        let err = fused(0, false, &qkv, &cos, &sin).unwrap_err();
        assert!(format!("{err}").contains("seq=0"));
    }

    /// A single identical token position (`s=1`, the degenerate/boundary
    /// "one point" case family D asks every op to cover) still matches
    /// `RopeFused` bit-for-bit.
    #[test]
    fn bit_identical_single_position_s_one() {
        bit_identity_case_with_red_control(2, 2, 1, 4, false);
    }

    #[test]
    fn cos_sin_variable_gradient_is_a_typed_error_not_silent_none() {
        use candle_core::Var;
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 3usize, 4usize);
        let n = b * h * s * d;
        let xv: Vec<f32> = (0..n).map(|k| k as f32 * 0.1).collect();
        let x_bhsd = Tensor::from_vec(xv, (b, h, s, d), &device).unwrap();
        let qkv = pack_bhsd_into_qkv(&x_bhsd, &x_bhsd, 0);
        let (cos, sin) = rope_table(s, d, 10_000.0);
        let cos = Var::from_tensor(&cos).unwrap();
        let out = fused(s, false, &qkv, &cos, &sin).unwrap();
        let loss = out.sum_all().unwrap();
        let err = loss.backward().unwrap_err();
        assert!(format!("{err}").contains("gradient is not implemented"));
    }
}
