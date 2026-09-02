# Writing a fused CUDA kernel for jammi

*Single source of truth for kernel work in `crates/jammi-kernels`.* The local agent skill
`.claude/skills/jammi-cuda-kernels/` is a thin pointer at this file — `.claude/*` is gitignored
(`.gitignore:64-69` allowlists only `agents/`, `hooks/`, `evals/`, `settings.json`, `AGENTS.md`),
so the knowledge lives here and the skill indexes it, never copies it.

Adapted in spirit from HuggingFace's `cuda-kernels` agent skill, but the substrate is different and
most of their scaffolding does not transfer. **What transfers:** architecture-aware optimisation
patterns, vectorised bf16 access, and the isolated-plus-end-to-end benchmark discipline.
**What does NOT transfer, and must not be adopted:** the Kernel Hub distribution model
(`get_kernel()`, Nix multi-variant builds, publish-and-fetch-precompiled) adds an external
dependency for something jammi compiles in-tree and cuts against the self-contained build; and the
whole `build.toml` / `torch-ext/` / PyTorch-C++-binding layout assumes a torch extension. jammi is
Rust + candle. Take their knowledge, not their plumbing.

One line of theirs IS load-bearing and jammi should follow it: *"For attention, prefer the model
library's existing optimized path when one already exists — Flash Attention 2 is usually the right
baseline for attention, while custom kernels are especially useful for operations like RMSNorm and
other targeted hotspots."* Hand-write the elementwise/reduction hotspots. Do not hand-write attention.

## 1. The substrate

A jammi kernel is a candle `CustomOp1|2|3` in `crates/jammi-kernels/src/ops/<op>.rs` with:
* `cpu_fwd` — the reference arm, and on CPU-only hosts the ONLY arm the local gate compiles.
* `cuda_fwd` — in `src/cuda/<op>.rs`, launching a `.cu` kernel built by `bindgen_cuda` (`build.rs`).
* `bwd` — the gradient. Optional but if present it is where the bugs are (see §3).
Dispatch goes through the admission predicate + `admit(...)` + `DispatchCounters`, so every call
site records `<op>_fused` / `<op>_eager`. Arities in use today: 13 × CustomOp1, 35 × CustomOp2,
13 × CustomOp3.

`bindgen_cuda` does NOT pass `-use_fast_math`; read `build.rs`'s comments before assuming a flag.
A vendored third-party tree (FlashAttention) uses a hand-rolled `nvcc` path behind its own feature
instead — that is the exception, not the pattern.

## 2. The roofline method — do this BEFORE writing any code

Never optimise by intuition. Compute the traffic model, then state achieved GB/s and the fraction
of roofline. This is what found jammi's worst kernel.

* **A100 SXM4 80GB = 2039 GB/s** (HBM2e, NVIDIA datasheet). Do NOT use 1555 GB/s — that is the
  **40GB** variant, and using it understates how bad a kernel is.
* Traffic = bytes that MUST cross HBM: every input read once + every output written once.
  Worked example, softmax over `[8,16,512,512]` bf16: scores = 8·16·512·512·2 = **67.1 MB**;
  fwd must read scores + read mask + write P ≈ **138 MB**; bwd (dscores) reads p, dp and writes ds
  ≈ **201 MB**.
* achieved GB/s = traffic / measured kernel time. Divide by 2039 for the roofline fraction.
* A well-written bandwidth-bound kernel lands in the tens of percent. HF's agent-generated RMSNorm
  measured **22-35%** on H100; treat 25-50% as a realistic target and >80% as excellent.
* **The anti-pattern this repo actually shipped:** `softmax_fwd_bf16` launched ONE 256-thread block
  per 512-element row (2 elements per thread) and made FIVE strided passes with three block-wide
  tree reductions and scalar 2-byte loads → 1.887 ms/launch = ~73 GB/s = **3.5% of roofline, 29x
  off**. Its own backward hit 301 GB/s on MORE traffic — when a bwd beats its fwd, the fwd is the bug.

Fixes, in the order that usually pays:
1. **Vectorise.** Load 128 bits at a time (8 × bf16) — never scalar 2-byte loads.
2. **Give each block enough work.** 2 elements/thread means launch overhead and reduction latency
   dominate. For short rows put multiple rows per block, or one warp per row.
3. **Cut passes.** Online softmax (Milakov & Gimelshein, arXiv:1805.02867) computes max and sum in
   one pass — 3 memory accesses per element instead of 4, ~1.3x on its own.
4. **Fuse the neighbours.** An additive mask belongs inside the softmax read, not in a separate
   `badd`. (jammi already does this — `SoftmaxLastDimFused` takes the mask as its second operand.)

## 3. The oracles — a fused kernel is not believed until these pass

Ordered by how much pain each one has already saved. Every claim below is from a real escape.

**The kernel-acceptance-oracle standard.** An oracle is not evidence because it exists; it is
evidence only if it (a) genuinely ran (never silently skipped in a way that still reports green),
(b) checks every bound it claims to check, (c) grounds every number and every floor it asserts
against in a real producer, (d) was validated out-of-sample of wherever it was calibrated, and (e)
demonstrates — not merely asserts — real separation between healthy noise and the defect it exists
to catch. The eight rules below (`KO-1` through `KO-8`) are the standing checklist an auditor runs
over any oracle, novel or not; five of them (`KO-2`, `KO-3`, `KO-4`, `KO-5`, `KO-7`) are mechanically
enforced today by `ci/scripts/check_kernel_oracles.py` and `ci/scripts/check_doc_numbers_have_producers.py`
(KO-4) / `ci/scripts/check_cuda_run_artifacts.py` (KO-3, optional per artifact); the remaining three
(`KO-1`, `KO-6`, `KO-8`) require running code or human judgment a static scan cannot make and stay
auditor-only. Each mechanical id below is an INSTANCE of one of the numbered rules 3.1-3.9 that
follow — tagged inline — never a parallel standard: the checklist and the numbered rules are the
same discipline read two ways, not two disciplines.

<!-- BEGIN KERNEL-ORACLE-STANDARD-IDS -->
- `KO-1` — producer-injected controls (auditor-only; generalizes 3.6)
- `KO-2` — bound coverage parity (mechanical; instances 3.7)
- `KO-3` — separation in the artifact (mechanical, optional per artifact leg; instances 3.8)
- `KO-4` — floors cite a producer (mechanical, advisory leg — the bare scan is `continue_on_error` in ci.yml; instances 3.9)
- `KO-5` — off-sample bounds (mechanical, marker-scoped; instances 3.2)
- `KO-6` — live signal (auditor-only; generalizes 3.5)
- `KO-7` — unrun-is-RED (mechanical, total over every scanned file; instances 3.5)
- `KO-8` — independent reference (auditor-only; generalizes 3.1 and 3.3)
<!-- END KERNEL-ORACLE-STANDARD-IDS -->

`KO-1` (producer-injected controls) is not mechanical because whether a RED control's failure is
CAUSALLY the specific defect under test, rather than some unrelated red result, is a semantic
judgment about what a perturbation means — the same discipline esc-046's leg 3 states ("a
deliberately mis-ordered reference MUST read GREEN pre-fix and RED post-fix") but verifying the
perturbation is the real one, not a stand-in, needs a human (or `fix-verifier`) reading the diff.

`KO-6` (live signal) is not mechanical because "this value is genuinely nonzero at runtime, not
zeroed/detached/short-circuited" (esc-045's `||g_f32|| > 0` signal clause) can only be known by
executing the oracle — a static scan cannot evaluate a tensor norm.

`KO-8` (independent reference) is not mechanical because "the reference computation does not share
the arm-under-test's own bug" (esc-044's root cause: an eager reference rebuilt in the test body,
GREEN under revert BY CONSTRUCTION) requires reading the reference's provenance against upstream
truth (PyTorch/HF/PEFT at source, never this repo's own prior belief about them) — exactly the
`validate-kernel-semantics-at-aten-source` discipline, which is a research act, not a grep.

**Pressure-test design rule: a metric is inadmissible until its own noise floor is measured.** A
bound is not "the fused arm agrees with the reference" in the abstract; it is only meaningful
relative to how much the EXTERNAL reference stack itself disagrees with ITS OWN higher-precision
version, at the SAME operating point (same shape, same seed, same dtype) — never a constant pulled
from a different run or a different amplitude. esc-045's backward budget is the working instance:
`mean_cos(jammi bf16, jammi f32) >= mean_cos(torch bf16, torch f32) minus the same run's own noise
leg (torch bf16-eager vs bf16-sdpa)` — the reference stack's (torch's) own bf16-vs-f32 spread, on
the identical fixture, IS the budget; a hand-picked absolute constant would be unfalsifiable (there
would be no way to tell "the fused kernel is worse" from "bf16 is just noisy at this shape" apart).
Until that external-stack spread is measured at the operating point, the metric has no basis for a
pass/fail line at all.

**3.1 Match the eager reference's GEMM OPERAND FORM, not just its maths (esc-044). [`KO-8`]
candle's own `Op::Matmul` backward differentiates through transposed **views**
(`grad.matmul(&rhs.t())`); both candle's CPU `gemm` and cuBLAS pick packing/blocking/split-k from
operand STRIDES. A fused bwd that materialises `pᵀ`/`vᵀ`/`dsᵀ` with `.contiguous()` issues a
different kernel than the arm it replaces and diverges systematically. That defect flattened a real
28-layer training loss while **every committed test stayed green**. The rule from `CLAUDE.md` is
*"two things that are the same thing at a different scale are one thing"* — one definition of how
each GEMM is issued, called by both arms; assert the `(rows, cols, row_stride, col_stride)` of every
operand, captured FROM the op, never rebuilt in the test body.

**3.2 A compounding defect is invisible at one call — key the oracle on GROWTH. [`KO-5`]**
esc-044's single-layer divergence was ~2e-3 relative, INSIDE the bf16 bound, and reached O(1) over 28
layers. So assert `r(L) = Σ|fused−eager| / Σ|eager|` over an L-deep stack against **the same run's own
r(1)** — `r(L_max) <= C · max(r(1), measured_floor)`, C small. Never against an absolute ULP constant.

**3.3 Agreement is not accuracy. [`KO-8`]** Two bf16 arms can agree and both be wrong. Anchor with a
**higher-precision reference**: run the same composition in F32 and compare each arm to it. Report
Σ|arm−ref| for both; accept only if the fused arm is no further than eager. (This is how a batch-1
anomaly was correctly closed as ordinary rounding rather than chased as a defect.)

**3.4 Test at PRODUCTION shape and amplitude.** Legs that run batch 2, seq 128, or amplitude 0.1
when production is batch 8, seq 512, `max|qkv|` 9-18 are decoration. jammi shipped a defect that
every parity leg missed for exactly this reason. Include **batch 1** at the op level — it is where a
`[1,1,S,S]` per-batch mask and a broadcast-over-batch mask become shape-indistinguishable.

**3.5 Zero dispatch is RED, never green. [`KO-6`, `KO-7`]** Assert the `DispatchCounters` delta shows the fused arm
actually ran (`fused == layers·steps`, `eager == 0`). The repo's end-to-end `learns_on_gpu` tests
were green on a broken build partly because they train a head_dim-16 model that never reaches a
head_dim-64 kernel at all.

**3.6 The learning gate (fused kernels only). [`KO-1`]** Same build, arm forced on vs off, same seed and
data, loss sequences compared **elementwise** across a shape sweep. Equal ⇒ value-neutral. Use
**batch ≥ 2**: jammi-bench's loss is `mean(relu(margin − cos(a,p) + cos(a,n)))` over `batch`
triplets, so at batch 1 it is ONE hinge — a binary switch, useless as an oracle.

**3.7 Write comparisons affirmatively. [`KO-2`]** `assert!(x.is_finite() && x <= bound)`, never
`assert!(!(x > bound))` — a NaN must FAIL, not read as a fit. Count non-finite elements before
comparing anything.

**3.8 No absolute ULP floor [`KO-3`]** in a discriminating assertion — a `k · ulp(max)` floor charges every
element the allowance of the largest and hides exactly the divergence you are hunting.

**3.9 No number without a producer. [`KO-4`]** A doc comment or `assert!` message that states a
precise-looking measurement — a mismatch count (`5145/16384`), a percentage (`26% of elements`), a
bare cosine (`0.796`) — reads as evidence, so a reader (or a fix agent citing it as ground truth)
must be able to re-derive or re-locate it: cite the real producer inline, `see <test_fn>` /
`printed by <test_fn>` (a real function, grep-verifiable) or `measured by <artifact path>` (a
tracked file), or tag it `no-producer: <reason>` when the number is genuinely *derived*, not
measured (e.g. `2^-7` bf16 ULP). `ci/scripts/check_doc_numbers_have_producers.py` enforces this
over `crates/jammi-kernels/{src,tests}`, `crates/jammi-encoders/src`, `crates/jammi-lora/src`, and
`crates/jammi-bench/src`, fail-closed with file:line and the offending number. The scan leg reports
(advisory) until precision on main reaches >= 80% real; the self-test and only-shrinks legs are
required — currently 33/69 = 47.8% real, 36/69 = 52.2% noise (see
`ci/doc_number_allowlist_classification.md`).

## 3.10 f16 per-op reference-regime table

**Why this table exists before any f16 tolerance is written (KO-8, "match the eager reference's
operand form").** The eager reference's own arithmetic is NOT uniform across ops: some upcast to
F32 internally and round once on the way out; others compute dtype-native (in the tensor's own
16-bit type) at specific mid-loop points, matching what candle's own composed ops (`Tensor::affine`,
`broadcast_add`, `GeluErf`) actually do at that dtype. A blanket `2^-9` (f16's ULP-relative
constant, `no-producer: derived from f16's 10-bit mantissa`) tolerance would be either too loose
(hiding a real divergence in an op that should be f32-accumulate-exact) or meaningless (an op that
genuinely rounds mid-loop needs a *behavioral* boundary oracle, not a tolerance at all — see
`docs/maintainer/cuda-kernel-guide.md` §3's KO-8 and the f16 boundary-contract doctrine in D4 of
campaign #443's numerics contract). Every row below states: (a) the eager reference's regime
(`f32-internal` = upcasts to F32, computes, rounds back once; `dtype-native` = computes in the
tensor's own 16-bit type at the stated step, matching what candle's un-fused ops would do), (b) the
rounding-POINT count (how many times a value crosses a 16-bit rounding boundary), and (c) whether
the op has a CPU F16 reference arm today, and what backs it.

| Op | Eager regime | Rounding points | CPU F16 arm |
|---|---|---|---|
| `layer_norm` (`LayerNormFused`/`LayerNormBwdDx`/`LayerNormBwdDgamma`) | f32-internal (mean/var/xhat accumulate in f32; `jammi-encoders/src/layer_norm.rs:353-370`'s `LayerNorm::slow` upcasts F16\|BF16→F32, casts back once) | 1 (final cast to f16) | **Present** — `ln_fwd_f16`/`ln_bwd_dx_f16`/`ln_bwd_dgamma_f16`, `crates/jammi-kernels/src/ops/layer_norm.rs` |
| `softmax_last_dim` (`SoftmaxLastDimFused`/`SoftmaxBwdDScores`) | dtype-native at two steps (the scale-multiply and the mask-add each round to the 16-bit dtype immediately, matching `candle_nn::ops::softmax`'s native `broadcast_add` and `Tensor::affine`'s own rounding point — `cuda/softmax.cu:75-101`'s `bf16_mul_rounded`/`bf16_add_rounded`); every step AFTER the mask add (max/exp/sum/normalize) stays f32 | 2 (scale-mul, mask-add); `dscores` bwd is 1 (final cast) | **Present** — `softmax_row_f16`/`softmax_fwd_f16`/`dscores_row_f16`/`dscores_f16`, `crates/jammi-kernels/src/ops/softmax.rs` |
| `geglu` (`GegluFused`/`GegluBwdDWiOut`) | dtype-native, two-op eager shape reproduced deliberately (candle's own `GeluErf::f16` arm ALSO computes in f64, mirroring its `bf16` arm exactly — `candle-core-0.11.0/src/op.rs:1002-1009` — but this op's OWN activation is computed in f32, matching the upstream HF/`kernels-community` "fp32 opmath" reference more closely than candle's f64 arm; see the module doc's "bf16 boundary-rounding" section) | 2 fwd (round activation, round product); 2 bwd (round `d_gate`, round `d_up`, each independently, f32-accumulated) | **Present** — `geglu_fwd_row_f16`/`geglu_fwd_f16`/`geglu_bwd_row_f16`/`geglu_bwd_f16`, `crates/jammi-kernels/src/ops/geglu.rs` |
| `rope`/`rope_positions` (`RopeFused`/`RopePositionsFused`) | f32-internal (accumulate in f32, matching `layer_norm`'s BF16 arms and the CUDA kernel) | 1 (final cast) | **Present** — `rope_fwd_row_f16`/`rope_fwd_f16` (`ops/rope.rs`), `rope_positions_fwd_f16` (`ops/rope_positions.rs`) |
| `dropout` | dtype-independent decision (Philox mask is a pure function of position, not value) + f32-internal scale multiply on a KEPT element | 1 (KEPT element only; a DROPPED element is exact zero, no rounding) | **Present** — `dropout_f16`, `crates/jammi-kernels/src/ops/dropout.rs` (Metal host-fallback arm deliberately NOT widened — out of this campaign's CUDA-only scope) |
| `scaled_cast_add` (`ScaledCastAdd`) | f32-internal (esc-046 fix: widen `base` to f32, add the already-f32 scaled `lora`, round the sum once — matches PEFT's own promote-add-cast-once model) | 1 | **Present** — `scaled_cast_add_f16_f32`/`scaled_cast_add_f32_f16`/`scaled_cast_add_f16_f16`, `crates/jammi-kernels/src/ops/scaled_cast_add.rs` (mirrors the existing 4-combo F32/BF16 matrix with 3 new F16 combos) |
| `cast_scale`/`cast_add` (`CastScaleBf16F32`/`CastAddBf16`; F16: `CastScaleF16F32`/`CastAddF16`) | N/A — **each type is structurally dtype-monomorphic, not dtype-generic** | N/A | **Present, as a SEPARATE pair of types** — `CastScaleF16F32` (`crates/jammi-kernels/src/ops/cast_scale.rs:427`) and `CastAddF16` (`:508`), each with its own CPU arm and its own CUDA arm (`crates/jammi-kernels/src/cuda/cast_scale_f16.cu:1`). They are not match arms on the BF16 types: those are domain-restricted to BF16 by construction (`CastScaleBf16F32`'s own doc: "this op's domain is BF16-only rather than accepting F32 too — nothing to fuse there"), so the F16 analogs carry their own double-rounding-safety argument at F16's 11-bit significand (`24 >= 2*11+2` holds with EQUALITY — at the boundary, not far past it the way BF16's margin is; each type's doc states this explicitly rather than inheriting BF16's). `low_rank_residual_linear`'s F16 backward admits both (`crates/jammi-kernels/src/ops/low_rank_residual_linear.rs:814`, `:911`), and both are pinned bit-identical to the eager two-kernel chain by `cast_scale_f16_bit_identical_to_the_eager_two_kernel_chain_on_cuda_across_scales` (`crates/jammi-kernels/tests/cuda_parity.rs:12280`) and `cast_add_f16_bit_identical_to_the_eager_two_kernel_chain_on_cuda_with_red_control` (`:12360`) — cited by TEST NAME first, line second: a line number alone rots the moment a neighbouring test is added or deleted, a name does not. |
| `attention_block` (`AttentionBlockFused`) | f32-only on CPU by a **real, disclosed candle limitation for BF16** (candle-core 0.11's CPU backend has no BF16 `MatMul` impl) — **but this limitation does NOT extend to F16**: `candle-core-0.11.0/src/cpu_backend/mod.rs:1382-1385`'s `MatMul::f` accepts `DType::F16 \| F32 \| F64` (the `gemm` crate ships a real `gemm-f16` backend, confirmed present in this workspace's own dependency tree), so an F16 CPU matmul arm is architecturally possible where a BF16 one never was. Rounding regime unstated (`f32 accumulate throughout` per the module doc's own F32-only CPU domain — an F16 arm would need to decide de novo whether QK^T/PV GEMMs run in f16-native or f32-accumulated precision, i.e. this is a fresh design decision, not a mechanical copy) | N/A (none designed) | **None.** The CPU forward matches `(CpuStorage::F32(qkv), CpuStorage::F32(mask))` only (`crates/jammi-kernels/src/ops/attention_block.rs:779`) — a ~500-line monomorphic `attention_fwd_f32` with its own `AttentionFwdF32Params` struct, RoPE-packing, mask-broadcast, and paired `bwd_core`/gradient-GEMM-layout machinery. An F16 arm is a second monomorphic forward, not a match-arm extension, and would have to decide de novo whether the QK^T/PV GEMMs run f16-native or f32-accumulated. |
| `mem_efficient_attention` (`MemEfficientAttentionFused`) | Same F32-only-CPU-by-BF16-matmul-limitation shape as `attention_block`; F16 matmul is architecturally possible for the same `gemm-f16` reason above | N/A (none designed) | **None.** Same shape as `attention_block`: the CPU forward matches `(CpuStorage::F32, CpuStorage::F32)` only (`crates/jammi-kernels/src/ops/mem_efficient_attention.rs:834`), a chunked/flash-style forward of ~2900 lines with its own GEMM-layout oracle machinery. |
| `low_rank_residual_linear`/`lora_linear` (`LowRankResidualLinear`) | f32-internal for the LoRA branch (x upcast to F32 once, both GEMMs and the dropout in f32, `ScaledCastAdd` epilogue rounds once); the BASE GEMM runs in the tensor's own dtype, so on CPU it is exactly candle's `gemm-f16` matmul for F16 — where BF16 has no CPU `MatMul` impl at all | 1 (the epilogue's cast of the scaled LoRA delta into `base`'s dtype) | **Present, and dtype-generic rather than a hand-written arm** — the CPU dtype gate admits `F32 \| BF16 \| F16` (`crates/jammi-kernels/src/ops/low_rank_residual_linear.rs:620`) and every step is composed from candle ops that are themselves dtype-generic, so F16 needed no monomorphic forward of its own. This is the asymmetry the structural finding below names: the same gate admits BF16, but a BF16 base fails inside candle's own `matmul` with a typed error, while F16 runs and is pinned bit-exact against the manual composition (`:1193`). |

**Structural finding (family K, "diagnose the structure before reaching for a tool"):**
candle-core 0.11's CPU backend refuses `BF16` for `matmul` (no `gemm-bf16` backend exists in the
`gemm` crate) but DOES accept `F16` (`gemm-f16` is a real, present dependency of this workspace —
it resolves in `Cargo.lock`). Every "CPU F32-only" domain comment in `attention_block.rs`/
`mem_efficient_attention.rs`/`low_rank_residual_linear.rs` that cites the BF16 matmul gap as its
reason does NOT automatically apply to F16: F16 CPU matmul is possible where BF16's is not, and
`low_rank_residual_linear`'s own F16 CPU arm is what that possibility already bought. Whether an
op has an F16 CPU arm today therefore tracks whether its CPU forward is composed from
dtype-generic ops (it does) or is a monomorphic `_f32` function (it does not) — never the BF16
matmul gap.

### 3.10.1 Dispatch status of the table's non-admitted-looking rows

Having an f16 arm is not the same as being *provable* on a device. Two different proof
mechanisms exist, and `ci/release-feature-manifest.json` is the single source of truth for
which one applies to which op — that file's `_schema_doc` defines each mechanism; this
subsection only states where the rows above land, so a reader of the table does not have
to guess.

| Op | Status | What proves it |
|---|---|---|
| `cast_scale` | **Admitted**, under dtype-keyed registry keys `cast_scale_bf16_f32` / `cast_scale_f16_f32` | A dispatch-registry delta on those keys in the `capability_surface` probe. The manifest names it dtype-neutrally as `cast_scale` in `fused_op_admission`; the name→key mapping is `jammi-kernels`'s own static table. |
| `cast_add` | **Admitted**, under `cast_add_bf16` / `cast_add_f16` | Same mechanism; both admit through `admit_cast_boundary` in `crates/jammi-kernels/src/ops/low_rank_residual_linear.rs`, called from that op's BF16 and F16 backward arms. |
| `rope_positions` | **Internal sub-kernel** of the flash-attention op | No `admit()` of its own: `crates/jammi-kernels/src/ops/flash_attention.rs` launches `crate::cuda::rope_positions::cuda_fwd` directly. It is proven by compiling in the lane's build *and* by its parent's admitted dispatch being observed (a delta on the flash cascade's key) — never by a counter of its own, which does not exist. |
| `scaled_cast_add` | **Internal sub-kernel** of `low_rank_residual_linear` | Same mechanism: `ScaledCastAdd::new` is constructed bare in that op's epilogue, so the parent's observed dispatch is the proof. |

The distinction is load-bearing when reading a capability report: an internal sub-kernel
is neither "unreachable" nor "independently admitted" — it runs exactly when its parent
does. Wiring a real `admit()` call site (or an admitted parent that launches it) is what
moves an op between these rows, in the same unit as that code edit.

There is no third, compiled-only status, and a kernel that would need one does not stay.
A kernel with no `admit()` site and no admitted parent has no row here and no manifest
category: the only thing a shipped build says about it is that it compiled, which no
capability report can act on. Such a kernel is wired or deleted in the same unit as its
authoring, decided by measuring its share of shipped-leg GPU time against a threshold
fixed before the numbers are seen — see
`crates/jammi-kernels/artifacts/cuda-runs/2026-09-01-axpy-census-bdeb80c-a100-pcie.json`,
where a pre-registered rule (wire iff the share reaches 2% of per-step GPU time on some
shipped leg *and* one dispatch site covers at least half of it) measured 0.007% / 0.026% /
0.027% across the three shipped dtype legs and the kernel was deleted.

## 4. Benchmarking

Two levels, always both. HF's own warning: their RMSNorm was **1.88x isolated but ~6% end-to-end**,
because it was a small share of runtime. An isolated number alone is not a result.
* Isolated: kernel time at production shapes; report achieved GB/s and % of roofline (§2).
* End-to-end: `jammi-bench finetune-step` with the dispatch counters, against the same box's torch
  reference. Compare `s_per_step_p50` and `peak_vram_bytes`.
* Profile with `nsys` and difference two step counts (N=5 vs N=10) to isolate per-step work.
* Commit the artifact under `crates/jammi-kernels/artifacts/cuda-runs/<date>-<unit>-<sha7>-<gpu>.json`
  carrying the **git_sha of the tip it measured**. A green artifact whose sha is not an ancestor of
  the branch is evidence about the ORACLE, not the code.
* After a squash merge of a branch carrying green artifacts, the merger stamps `merged_as`/
  `merged_via_pr` in the same day — until then every PR fails rule (d).
* **Exclusive box, timing lock.** A shared GPU under concurrent agent builds moves a timing number by
  several times the exclusive-box noise band; rent the box exclusively and hold a timing lock for the
  duration of the measured legs, or the number is not comparable to anything.
* **Ratios travel across boxes; milliseconds do not.** A same-box `jammi ÷ torch` (or `torch ÷ jammi`)
  ratio is meaningful across a GPU/driver/torch-wheel change because both arms moved together; a raw
  millisecond number from one box says nothing about another box's — always re-derive the torch
  denominator on the box you measure jammi on, never carry a torch number in from a different session.
* **Attribute a kernel-time delta by GRID, not by launch count.** Averaging launch counts across
  unequal-size dispatches misattributes time to the wrong kernel family; attribute by element mass
  times the removed family's own bandwidth, and read the grid dimensions (not just the kernel name)
  off the profile before naming what a kernel does.

## 5. Host and build hazards (they silently invalidate results)

* `cargo` is not on PATH in non-interactive shells — `export PATH="$HOME/.cargo/bin:$PATH"`.
* Give every worktree a UNIQUE `CARGO_TARGET_DIR`; a shared one serves stale artifacts. On a
  RunPod pod this is what `gpu-dev.sh target <session> <name>` gives each **tree**: a fresh,
  per-tree clone of the pod's build-substrate seed (`docs/maintainer/dev-gpu.md`) — the seed
  itself is member-free (zero `jammi-*` artifacts) precisely so a clone can never serve one back
  stale; only third-party dependency artifacts are shared.
* For `cargo mutants`: COPY mode, and `CARGO_TARGET_DIR` **unset**. A shared target dir makes cargo
  report "Fresh" for MUTATED sources and scores every mutant against unmutated artifacts — a whole
  run was invalidated that way.
* Copy mode duplicates the workspace + target per job. Budget `~25 GB + S_src + S_seed +
  N*S_clone` once the build-substrate seed/clone is in use (S values pending —
  `ci/scripts/perf/pod_build_timings.sh` is the producer) `+ 3 GB/other agent + 2 GB/mutants
  job`; a mutation session wants a pod ≥ 120 GB (`RP_DISK_GB=... ci/scripts/gpu-dev.sh up`).
* `tests/cuda_parity.rs` is `required-features = ["cuda"]`, so no CPU-only gate compiles it. Whatever
  you put there is checked by the pod lane alone — say so rather than implying local green covers it.

## 6. Process

Mutating work goes through the rigor chain (`AGENTS.md`); a kernel change is not exempt.
The two steps that catch kernel bugs specifically: **pressure-test the design before code**
(a wrong kernel design compiles and passes its own tests), and **fix-verifier / red-green** — revert
the fix, keep the test, require RED. If the test is green on the broken build, it is not an oracle,
whatever else it asserts.
