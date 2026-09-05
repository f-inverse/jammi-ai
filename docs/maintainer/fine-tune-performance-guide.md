# Fusing the fine-tune step — a guide to the performance track

How jammi's Rust/candle training step went from 0.44× to 1.07× of PyTorch on the same GPU (2026-08-23 → 08-26), and what it took to prove the numbers were real. Written for an ML engineer who wants to understand the low-level computation of modern ML through one real track: candle `CustomOp`s, the autograd tape's cost, bf16 rounding placement, FlashAttention-2, and — above all — how to prove a fused kernel is both faster and faithful.

Sources: GitHub #352 / #356 / #374 / #428; PRs #357–#391; the five session ledgers under `.jammi/ledger/` (`perf-fusion-20260823`, `perf-continuation-20260824`, `perf-close-20260824`, `perf-s2-20260825`, `perf-s4-20260826`) — these are session-local working notes, **not tracked in this repository**: a row cite like `s2:245` records where a number came from, but no clone can re-resolve it; the evidence a reader *can* re-verify is the committed artifacts under `crates/jammi-kernels/artifacts/cuda-runs/`, which the claims gate binds by value; `docs/maintainer/cuda-kernel-guide.md`; the escape ledger `.jammi/escapes.jsonl` (tracked). Every number in the ten tables below is mechanically bound to a tracked artifact or escaped into a committed, shrink-only ledger (`ci/scripts/check_perf_claims.py`, `ci/perf_claims_allowlist.txt`) — a table cell that drifts from its artifact, or a new cell landing with no producer, is a CI failure; the surrounding PROSE (this paragraph included) is not gated the same way. Where the sources disagree, the disagreement is printed, not smoothed.

Notation: `b8·s512` = batch 8 triplets (the bench batches anchor/positive/negative into one forward, so the attention batch is 24 rows), sequence 512. `s/step` is the p50 of measured optimizer steps. "Same box" means both legs ran on the same physical GPU in the same session. `s2:245` = session-2 ledger row 245; `s4:11` = session-4 ledger row 11; `esc-044` = a row of the escape ledger.

---

## 0. How to read this

**jammi** is an embedding engine on [candle](https://github.com/huggingface/candle) (Rust; eager execution; tape-based autograd). One of its jobs is fine-tuning an encoder — here `answerdotai/ModernBERT-large`: 28 layers, hidden 1024, 16 heads of dimension 64 — with LoRA adapters, in bf16, on a GPU. The comparison stack is PyTorch + HF Transformers + PEFT doing the identical work.

The bar was stated before the first measurement (PLAN.md, 2026-08-23, adopted verbatim into #352): *is jammi's optimizer step at least 0.9× as fast as PyTorch's on the same GPU, at equal precision, without OOM-ing where torch fits, and without changing what the model learns?* Units are seconds per optimizer step, peak VRAM, and later gradient fidelity vs an f32 truth — never GPU utilization, because "a naive kernel can be 100% utilised and slow".

What makes the track worth a guide is not the result but the path: two thirds of the effort went into *proving* that fused kernels were both faster and faithful, and the proofs failed more often than the kernels did. If you take one section, take §7.

---

## 1. The state before: busy, not productive

### The measurement that opened the track (#352, 2026-08-23T18:56Z; b8·s128, bf16, dropout 0)

| stack | s/step | triplets/s | peak VRAM | GPU util | mem-ctrl util |
|---|---:|---:|---:|---:|---:|
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger; c5=ledger; c6=ledger -->
| PyTorch + PEFT (2024-era wheel) | 0.331 | 24.07 | 5.76 GB | 43% | 0% |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger; c5=ledger -->
| jammi (eager candle) | 0.593 | 13.50 | 35.8 GB | 100% | 25% |

The issue states an A100-SXM4; the ledger's reproduction of the same 0.592 s/step records an A100 80GB PCIe (`fusion row 1`). The issue gives eager VRAM three ways (35.8 / 57.9 / 40.0 GB) for the same config. Both discrepancies are typical of the track and are why every later number carries its box.

The signature to learn is the last two columns. `nvidia-smi` "utilization" means *percent of time at least one kernel is resident*, not achieved occupancy. A GPU that is 100% busy while its memory controller idles at 25% is running many small, low-occupancy kernels back to back — busy, not productive, not bandwidth-limited. Torch at 43% did the same work in half the time because its kernels were bigger.

Also in the issue: batch 16 OOMs where torch uses 11.6 GB; the shipped default seq 512 OOMs at batch 8; host-generated dropout masks cost 2.9× (2.072 s / 56.7 GB at dropout 0.05 vs 0.719 s / 40.0 GB).

### Four hypotheses, measured and eliminated

1. **Attention / missing FlashAttention.** Batch-1 sequence scan: seq 64 → 0.446 s, 128 → 0.440, 256 → 0.438, 512 → 0.598, 1024 → 1.354. Eight times the sequence (64× the attention FLOPs) cost 11% more time below 256. And `candle-flash-attn` cannot train at all: its op implements `CustomOp3` with no `bwd`.
2. **Raw `cudaMalloc` per op.** cudarc uses the pooled `cuMemAllocAsync`. Consequence: every VRAM figure in this track is a pool high-water mark. "The OOM is the hard fact."
3. **Dispatch-bound.** Batching the three triplet forwards removed two thirds of dispatches and bought 1.21×. Dispatch-bound would have been ~3×.
4. **HBM saturation.** Memory-controller utilization 25%.

### The diagnosis

candle is eager with no fusion layer and no graph capture: every elementwise op is its own kernel reading from and writing to HBM. A LayerNorm was ~12 ops (57 per step); softmax 5 (each `[B,H,S,S]` intermediate retained); RoPE ~12 per Q/K including a `cat` copy; each of the 112 LoRA sites ~7. Order 2,600 candle ops per forward, and **every intermediate pinned on the autograd tape** — the 6× memory.

### The profile that replaced the estimates (`fusion rows 1–7`, A100 PCIe)

- **15,035 kernel launches per optimizer step**; 562 ms of kernel time in a 592 ms wall (launch gaps ~5%), yet `CUDA_LAUNCH_BLOCKING=1` costs 6.1× — launch-*heavy* without being launch-*bound*.
- Forward 2,611 kernels / 134 ms; backward 9,123 / 420 ms; AdamW 3,301 launches for 7.6 ms (1% of time, 22% of launches).
- **candle's backprop does `zeros_like` + a full-tensor add for every gradient contribution, even the first**: ~2,900 `badd` per step ≈ 220 ms ≈ 39% of GPU time was tape accumulation, not model math.
- Broadcast/strided elementwise kernels 5–20× below bandwidth. GEMMs ~10% of GPU time.

Nobody could have predicted the third bullet from the model's math. It came from counting kernels by name in a profile.

---

## 2. Where a step's time goes: the mental model

**Memory-bound work (the roofline).** Ceiling = `min(peak FLOP/s, bandwidth × FLOPs-per-byte)`. LayerNorm, softmax, RoPE, GeGLU, casts and the AdamW update do a handful of FLOPs per element, so their best time is *bytes moved ÷ HBM bandwidth*. House method (kernel guide §2): traffic = every input read once + every output written once; achieved GB/s = traffic ÷ time; divide by the A100-SXM4-80GB's 2039 GB/s. (Three roofline constants were used in the track — 1935 PCIe, 1555 for the 40 GB part, 2039 SXM4-80 — giving different "× off roofline" for the same kernel. Write the constant down.) A good bandwidth-bound kernel lands in the tens of percent; the shipped softmax forward measured 3.5%.

**Launch-bound work.** Each launch costs ~5–10 µs regardless of size. The fused AdamW step runs at 9.2–9.8 µs per call while element counts vary 5× — pure launch latency, 2–12% of roofline. Its lever was never bandwidth.

**The tape tax.** When `Tensor::backward()` walks the graph, every gradient contribution goes through `GradStore::or_insert`, which on a vacant entry returns `zeros_like(tensor)`, and then *every* backward arm does `sum_grad = sum_grad.add(arg_grad)`. There is no "if vacant, move" fast path (`candle-core-0.11.0/src/backprop.rs:768-777`, `s2:18`):

```
candle 0.11, every contribution incl. the first:
  grad arrives ──▶ or_insert → zeros_like (full-size fill) ──▶ sum_grad.add(arg) ──▶ stored
                              badd_bf16 ≈ 0.137 ms per launch on a [24,512,1024] tensor

torch AccumulateGrad:
  grad arrives ──▶ move (no kernel) ──▶ 2nd+ contribution: add
```

At the P3 tip, b8·s512: `badd_bf16` = 563 launches/step = 27 layers × 19 nodes + 18 + 32; **496 (88%) are accumulation ≈ 102 ms of 116 ms**. A pre-norm transformer's tape is nearly a chain, so almost every node has one consumer and its add is pure waste. After the first fusion round this term *grew* from 39% to 50–53% of GPU time (`cont row 11`): fusing elementwise chains removed cheap kernels faster than it removed tape nodes. **The lever is node count.**

The standing constraint — no vendoring, patching, or upstream PRs to candle — meant the fix could never be a move fast path in `or_insert`. It had to be fewer nodes: wider fused ops, each one `CustomOp` node regardless of the work inside.

---

## 3. The levers: hypothesized vs measured

### Round one: the C-series (2026-08-23/24, PR #357)

Fuse the chains the profile named — LayerNorm (C2), RoPE (C3), masked softmax (C4), GeGLU (C5), the LoRA epilogue (C6), device Philox dropout (C7) — as candle `CustomOp`s with real backwards, behind a `cuda` feature.

| tip (b8·s128, PCIe, 15 steps) | s/step | Δ | modeled |
|---|---:|---:|---:|
<!-- claims: c1=ledger -->
| eager | 0.5915 | — | — |
<!-- claims: c1=ledger; c2=ledger; c3=ledger -->
| C2 LayerNorm | 0.5089 | −82.6 ms | ~90 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger -->
| C3 RoPE | 0.4410 | −67.9 ms | ~84 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger -->
| C4 softmax+mask | 0.3701 | −70.9 ms | ~79 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger -->
| C5 GeGLU | 0.3097 | −60.4 ms | ~118 (whole MLP cluster; GEMMs never in scope) |
<!-- claims: c1=ledger; c2=ledger; c3=ledger -->
| C6 LoRA epilogue | 0.2750 | −34.7 ms | ~50 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger -->
| C7 Philox dropout (d0) | 0.2695 | VRAM 33.5 → 22.4 GB | — |

`fusion rows 30, 36, 50`. Dropout 0.05: host mask path 1.364 s / 39.6 GB → device Philox 0.284 s / 19.7 GB.

**2.2× over eager, and NOT MET.** Rebuilt on the current stack (torch 2.13, transformers 5.15), torch's own b8·s128 step was 0.1184 s, not 0.331: the denominator had moved 2.8×. jammi-fused landed at 0.42–0.44× (`fusion row 52`); its VRAM delta was still ~5× torch's; b8·s512 with dropout OOMed where torch fit. *A comparison's reference is a moving part; rebuild it on the box you measure on, every time.*

Killed before it started: **C1b**, a vendored patch to candle's tape accumulation, on the no-candle-dependency rule. The plan estimated the residual tape cost at 22–66 ms; it was ~3× that.

### Round two: node count (2026-08-24, P1–P5)

Post-#357 census on an SXM4 (`cont row 11`): 251 ms GPU at b8·s128, of which `badd_bf16 + badd_f32` = 50%; attention ~7% (12% at s512). Torch same pod: 105 ms, 5,906 launches, 7 memcpys.

| lever | mechanism | predicted | measured (same box, SXM4) | verdict |
|---|---|---|---|---|
<!-- claims: c1=ledger; c2=ledger; c3=crates/jammi-kernels/artifacts/cuda-runs/2026-08-24-p1-softmax-fold-bf8e807-a100-sxm4.json#/rows/b8-s512-d0/base/s_per_step_p50; c4=crates/jammi-kernels/artifacts/cuda-runs/2026-08-24-p1-softmax-fold-bf8e807-a100-sxm4.json#/rows/b8-s512-d0/tip/s_per_step_p50; c5=crates/jammi-kernels/artifacts/cuda-runs/2026-08-24-p1-softmax-fold-bf8e807-a100-sxm4.json#/rows/b8-s512-d0/base/peak_vram_bytes as GB; c6=crates/jammi-kernels/artifacts/cuda-runs/2026-08-24-p1-softmax-fold-bf8e807-a100-sxm4.json#/rows/b8-s512-d0/tip/peak_vram_bytes as GB; c7=diff(crates/jammi-kernels/artifacts/cuda-runs/2026-08-24-p1-softmax-fold-bf8e807-a100-sxm4.json#/rows/b8-s128-d0/tip/peak_vram_bytes,crates/jammi-kernels/artifacts/cuda-runs/2026-08-24-p1-softmax-fold-bf8e807-a100-sxm4.json#/rows/b8-s128-d0/base/peak_vram_bytes) as MiB -->
| **P1** fold 1/√d into fused softmax | deletes one `[B,H,S,S]` affine node per layer | −5.6 GB, ~12 ms at s512 | s512 1.078 → 1.038 s, 77.5 → 71.8 GB; s128 flat (+32 MiB, disclosed) | landed #362 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger; c5=ledger; c6=ledger; c7=ledger; c8=ledger; c9=ledger; c10=ledger; c11=ledger; c12=ledger; c13=ledger; c14=ledger; c15=ledger; c16=ledger; c17=ledger; c18=ledger -->
| **P2** `LowRankResidualLinear` — one CustomOp3 per LoRA site, no dW for frozen W | ~11 → 3 nodes per site; 60% of the add mass | GPU 251 → 160–190 ms (first draft 140–150) | s128 0.2668 → 0.2098 (−21%), 23.7 → 8.7 GB; s512 1.037 → 0.780 (−25%), 71.9 → 39.1 GB | landed #363 |
<!-- claims: c1=ledger; c2=ledger; c3=crates/jammi-kernels/artifacts/cuda-runs/2026-08-24-p1-softmax-fold-bf8e807-a100-sxm4.json#/rows/b8-s512-d0/base/s_per_step_p50; c4=ledger; c5=crates/jammi-kernels/artifacts/cuda-runs/2026-08-24-p1-softmax-fold-bf8e807-a100-sxm4.json#/rows/b8-s512-d0/base/peak_vram_bytes as GB; c6=ledger; c7=ledger; c8=ledger; c9=ledger -->
| **P3** `AttentionBlockFused` — RoPE + QKᵀ + softmax + PV in one node, P recomputed in bwd | drop three retained `[B,H,S,S]` per layer | −16.6 GB at s512, +27 ms | s512 1.078 → 1.039 s, 77.5 → 60.85 GB (−16.65); then the esc-044 fix alone: 0.772 → 0.658 s | landed #368 |
<!-- claims: c1=ledger; c2=ledger -->
| **P4** CUDA-graph capture | hide launch gaps | ≤0.5–6.7% (GPU busy ≈ wall) | — | killed at pressure-test |
<!-- claims: c1=ledger; c2=ledger; c3=ledger -->
| **P4b** device-side gradient clip (225 host syncs) | the product step always clips; the bench never did | −3.9% at s128 | clip on: +12.7 ms (s512) over the no-clip bench step — a baseline correction, not a saving | #373 cut → #381 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger; c5=ledger; c6=ledger; c7=ledger; c8=ledger -->
| **P5** jammi-owned explicit backward | remove GradStore adds wholesale | 400+ → 155 → 100 → 62–93 → 33 → 19 ms | never built (six design rounds) | parked at the 19 ms atom |

Sources: `cont rows 12, 17, 34, 60, 63, 77; close rows 38, 54; s2:57–155`.

### Round three: the five-lever plan and the census (2026-08-25)

With P1+P2+P3 stacked and esc-044 fixed, an *exclusive* A100 measured: **b8·s512 jammi 0.668 s (band 0.58% over three runs) vs torch-sdpa 0.4292 s = 0.642×** (`s2:12, 22`). The bar needed ≤0.4769 s: remove 191 ms. The census of the 673 ms GPU step (a100b, nsys, grouped by kernel *and grid*):

| kernel | ms/step | launches | what it is |
|---|---:|---:|---|
<!-- claims: c1=ledger; c2=ledger; c3=ledger -->
| `badd_bf16` | 116.1 | 563 | 496 are GradStore accumulation |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger -->
| `softmax_fwd_bf16` | 105.6 | 56 | attention; 2/layer because the block bwd recomputes P; ~3.5% of roofline |
<!-- claims: c1=ledger; c2=ledger; c3=ledger -->
| `ucopy_bf16` | 83.9 | 364 | 100% inside the attention block |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger -->
| `Kernel2` | 52.7 | 364 | six grid configs; only `gridDim.z=384` (26.5 ms) are attention |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger; c5=ledger; c6=ledger -->
| `cast_bf16_f32` / `scaled_cast_add` / `cast_f32_bf16` | 40.4 / 21.0 / 11.1 | 337 / 112 / 112 | the LoRA site's dtype motion |
<!-- claims: c1=ledger; c2=ledger -->
| `ampere_sgemm_32x128_nn` | 40.1 | 336 | the LoRA site's rank-16 f32 GEMMs |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger -->
| `affine_f32` | 25.8 | 2129 | 2016 AdamW; 112 the LoRA bwd |
<!-- claims: c1=ledger; c2=ledger; c3=ledger -->
| bf16 GEMMs (z=1) | 53.7 | 140 | *projection* GEMMs (Wqkv, Wi) — not attention |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger; c5=ledger; c6=ledger -->
| `rope_fwd` / `softmax_bwd` / `layer_norm_bwd_dx` | 19.3 / 18.8 / 9.0 | 168 / 28 / 56 | |

`s2:51, 70`. Total 673.0 ms GPU, 7,796 launches, 1,401 memcpys; torch-sdpa: ~422 ms, 5,672 launches, 7 memcpys. GEMMs matched torch kernel-for-kernel.

> **A lead error, corrected by the census itself.** The first reading attributed the 56/step bf16 GEMMs to attention, putting it at ~315–334 ms. The pressure-test pointed at the grid: those GEMMs have `gridDim.z = 1` and x-tiles 3072/5248 — the Wqkv/Wi *projections*. Attention-core kernels carry `z = b·h = 384`. Corrected attention-attributable: **243.8 ms**. The same census refuted "attention ∝ s², so FA2 cannot help at s128": at s128 the block's kernels are launch-bound (~38% of a 183 ms step).

The five-lever plan: AdamW ~26 ms, cast boundary ~72 ms, FA2 ~160 ms, the esc-045 bf16 fix as correctness gate, P5 ~102 ms as margin.

| lever | projection history | measured (one build, forced arm off/on) | PR |
|---|---|---|---|
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger; c5=ledger; c6=ledger; c7=ledger; c8=ledger; c9=crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-cast-w1-80f02fb-a100-sxm4.json#/legs/b8_s512_disabled_r1/s_per_step_p50; c10=crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-cast-w1-80f02fb-a100-sxm4.json#/legs/b8_s512_fused_r1/s_per_step_p50; c11=neg(crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-cast-w1-80f02fb-a100-sxm4.json#/deltas/b8_s512_p50_ms/delta_ms); c12=crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-cast-w1-80f02fb-a100-sxm4.json#/legs/b8_s128_disabled_r1/s_per_step_p50; c13=crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-cast-w1-80f02fb-a100-sxm4.json#/legs/b8_s128_fused_r1/s_per_step_p50 -->
| **Cast boundary W1** (fuse `cast+affine` and `cast+add` in the LoRA bwd) | 72.5 → 13.4 → 40–48 → 28–31 → 31–42 ms | b8·s512 0.6744 → **0.6349 (−39.6 ms)**; s128 0.1978 → 0.1883; bit-identical to the two-kernel chain | #377 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger; c5=ledger; c6=ledger; c7=ledger; c8=ledger; c9=legacy(crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4.json#/a100b_full_step_ab_reference/summary/s512/disabled_eager_p50_r1_r2/0); c10=legacy(crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4.json#/a100b_full_step_ab_reference/summary/s512/fused_p50_r1_r2/0); c11=legacy(neg(diff(mean(crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4.json#/a100b_full_step_ab_reference/summary/s512/disabled_eager_p50_r1_r2/0,crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4.json#/a100b_full_step_ab_reference/summary/s512/disabled_eager_p50_r1_r2/1),mean(crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4.json#/a100b_full_step_ab_reference/summary/s512/fused_p50_r1_r2/0,crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4.json#/a100b_full_step_ab_reference/summary/s512/fused_p50_r1_r2/1))) as ms); c12=ledger -->
| **AdamW in-place step** (`InplaceOp2/3`; 3 launches/Var, zero `Var::set` memcpys) | −40 → "~26" → ~30 wall → −20.5 isolated | optimizer phase 23.1 → 2.59 ms (8.9×); full step b8·s512 0.6759 → **0.6589 (−16.4 ms)**; s128 −16.5; bit-identical to candle's chain | #380 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger; c5=ledger; c6=ledger; c7=ledger; c8=ledger; c9=ledger; c10=ledger; c11=ledger; c12=ledger; c13=ledger -->
| **FlashAttention-2 dense arm** | −285 → −210..−240 → −140 → [159,183]; kernel 61–85 ms projected vs **39 measured** | b8·s512 block 0.6756 → **flash 0.4626 (−213 ms)** same box, 0.937× torch; s128 −43 ms | #389 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger -->
| **P5** | 102 → 33 → 19 ms atom | not built | — |
| **esc-045 fix** ("the gate") | — | did not gate the shipped levers: FA2 replaces the softmax site; the others are bit-identical; the metric turned out chaotic (§6) | #374 open |

Sources: `s2:89, 97, 109, 137, 146, 151, 164, 192`. The AdamW "26 ms" was a mis-attribution (112 of the 2129 affines belong to the LoRA backward; AdamW's data volume is ~0.5 GB) — its credit was launch overhead, which is why the in-place design won over a true multi-tensor kernel that would have needed a candle patch.

### The stacked result (committed producer `ci/scripts/perf/stacked_sweep.sh`)

| shape | jammi stacked s/step | torch-sdpa | jammi all-off | torch ÷ jammi |
|---|---:|---:|---:|---:|
<!-- claims: c1=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s512/stacked_p50_min_s; c2=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s512/torch_p50_min_s; c3=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s512/alloff_p50_s; c4=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s512/ratio_torch_over_stacked -->
| b8·s512 | **0.4212** | 0.4512 | 0.6554 | **1.071×** |
<!-- claims: c1=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s128/stacked_p50_min_s; c2=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s128/torch_p50_min_s; c3=pct(crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s128/legs/torch_r1/s_per_step_p50,crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s128/legs/torch_r2/s_per_step_p50); c4=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s128/alloff_p50_s; c5=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s128/ratio_torch_over_stacked -->
| b8·s128 | 0.1307 | 0.1319 (r1/r2 spread 8.3%) | 0.1892 | 1.009× |
<!-- claims: c1=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b1s128/stacked_p50_min_s; c2=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b1s512/stacked_p50_min_s; c3=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b1s128/torch_p50_min_s; c4=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b1s512/torch_p50_min_s; c5=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b1s128/alloff_p50_s; c6=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b1s512/alloff_p50_s; c7=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b1s128/ratio_torch_over_stacked; c8=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b1s512/ratio_torch_over_stacked -->
| b1·s128 / b1·s512 | 0.0401 / 0.0738 | 0.1260 / 0.1216 | 0.0648 / 0.1157 | 3.14× / 1.65× |
<!-- claims: c1=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b16s128/stacked_p50_min_s; c2=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s256/stacked_p50_min_s; c3=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b16s128/torch_p50_min_s; c4=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s256/torch_p50_min_s; c5=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b16s128/alloff_p50_s; c6=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s256/alloff_p50_s; c7=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b16s128/ratio_torch_over_stacked; c8=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s256/ratio_torch_over_stacked -->
| b16·s128 / b8·s256 | 0.2178 / 0.2206 | 0.2243 / 0.2273 | 0.3129 / 0.3219 | 1.030× / 1.030× |
<!-- claims: c1=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b16s512/stacked_p50_min_s; c2=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s1024/stacked_p50_min_s; c3=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b16s512/torch_p50_min_s; c4=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s1024/torch_p50_min_s; c5=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b16s512/alloff_p50_s; c6=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s1024/alloff_p50_s; c7=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b16s512/ratio_torch_over_stacked; c8=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s1024/ratio_torch_over_stacked -->
| b16·s512 / b8·s1024 | 0.8186 / 0.8300 | 0.8790 / 0.9461 | 1.2650 / 1.4781 | 1.074× / 1.140× |

Artifact `crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json` + 40 raw runs; a100c A100 80GB PCIe, driver 570.172.08, torch 2.13.0+cu126 / transformers 5.15.1 / peft 0.20.0 (`s4:25`). Counters on every stacked leg: flash 840 fused / 0 declined, AdamW 6720 fused / 0 eager. Caveats the artifact prints: the b8·s128 torch leg's own spread exceeds the margin at that shape (an earlier run read 0.950×); the all-off leg disabled the flash and AdamW arms only; the ratio uses the min of two torch runs — the estimator least favourable to jammi.

---

## 4. How the levers are built

Paths are in `crates/jammi-kernels` unless stated; `[fa2]` marks what arrived with the FlashAttention-2 arm (PR #389, on main as `6c526f9`).

### The substrate: one tape node, whatever happens inside

A fused op is a struct implementing candle's `CustomOp1/2/3` with `cpu_fwd` (the reference arm), `cuda_fwd` (launches PTX), and `bwd(args…, res, grad_res)` returning one gradient per argument. `tensor.apply_op3(y, z, op)` stores the op in the result's graph node; `backward()` reaches it and accumulates via `or_insert`. jammi routes every call through wrappers (`ops/mod.rs:205`, wrappers from `:213`) whose `KernelOp` bound is `Copy + Send + Sync + Sealed` — `Copy` is a structural proof of statelessness; `Sealed` keeps downstream crates from adding ops.

**candle 0.11 has no `save_for_backward`.** `bwd` gets only the arguments, the result and the incoming gradient; anything else must be recomputed. LayerNorm recomputes mean/rstd, GeGLU recomputes `gelu(gate)`, the attention block recomputes RoPE, scores and the probability matrix.

The exception is `Saved<T>` `[fa2]` (`ops/saved.rs`): a write-once/read-once slot for FlashAttention's per-row log-sum-exp (`lse`, f32, a different shape than the output). `apply_op3` allocates a fresh `Arc` per call, so the slot is scoped to one forward; a second `bwd` on the node or a `bwd` before `fwd` is a typed error. Such ops cannot be `Copy`, so `StatefulKernelOp` exists with a grep-discipline test forbidding `Clone`/`Arc` wrapping.

**PTX.** `build.rs` compiles `src/cuda/*.cu` to PTX for `compute_80`; each op embeds its PTX with `include_str!` and loads it through candle's public `CudaDevice::get_or_load_custom_func`. nvcc's `--fmad=true` is left on; kernels needing bit-exactness pin operations with `__fmul_rn`/`__fadd_rn`. Lesson from the FA2 work: a header-only edit served *stale PTX* because a directory-level `rerun-if-changed` does not track edits to existing files and bindgen_cuda's skip check compares a `.ptx`'s mtime against its own `.cu` only. Build systems are numerics.

### Admission: deciding fused vs eager, and proving which one ran

Every fused call site evaluates its own predicate (device, dtype, contiguity, shape bounds, each with a named reason) and passes it to `admit(mode, op, predicate_name, holds, counters)` (`admission.rs:739-755`). In `Strict` mode (`JAMMI_KERNELS_STRICT=1`, which the bench sets) a miss is an error, so "fell back everywhere" can never pass as a green fused measurement. Two atomics per op count `fused` and `eager` dispatches; the bench prints the delta over the timed loop. **Counters are the proof a kernel ran** — the end-to-end "learns on GPU" tests were green on a broken build because they trained a head_dim-16 model that never reached the head_dim-64 kernel; hence "zero dispatch is RED".

`JAMMI_KERNELS_DISABLE=op1,op2|all` (#375) forces ops eager *in the same binary* — the instrument behind every A/B. A typo never reads as success: `unmatched_disables()` turns a never-fired key into a hard error. Op keys are not flat: `attention_block_fused` subsumes `rope_fused` and `softmax_last_dim_fused` on the training path.

The flash arm introduced a three-way chain (flash → block → eager) where a decline means "try the next arm". A `bool` cannot say that, so `[fa2]` adds `PredicateOutcome { Holds, DomainMiss, CapabilityMiss }` and `admit_cascade` with a `declined` counter. The 17 two-arm predicates were deliberately not migrated: reclassifying their misses as never-erroring would defang `Strict`.

### Each fused op as a mini-lesson

Common shape: elementwise ops launch one thread per element in a grid-stride loop; reductions launch one 256-thread block per row with a shared-memory tree reduction. None of the hand-written kernels use warp shuffles or vectorized loads — the pattern the kernel guide's own §2 names as the 3.5%-of-roofline anti-pattern. jammi's wins are tape-node elimination and rounding control; the only tensor-core code is the vendored FlashAttention-2.

**LayerNorm** (`LayerNormFused`, CustomOp2 over (x, γ)). Replaces a dozen ops each retaining a `[rows, hidden]` intermediate. One block per row, three passes, f32 accumulation, one rounding at the store. Backward in one launch:

```
t          = dy · γ
mean_t     = Σ t / H          mean_t_x̂ = Σ (t · x̂) / H
dx         = rstd · (t − mean_t − x̂ · mean_t_x̂)        // layer_norm.cu:172
```

`dγ = Σ_rows dy·x̂` is two kernels to stay O(rows·hidden) with no `atomicAdd`; whether it is needed is frozen into the op from `gamma.is_variable()`.

**LayerNorm site, biased** (#460). The bias-free fused site above domain-excludes every architecture whose LayerNorm carries an affine bias — BERT and DistilBERT, never ModernBERT (`ModernBertConfig` cannot even express one) — so a biased checkpoint's own norm call has always fallen straight through to the eager composition, unconditionally, regardless of every other admission check holding. Both the bias-free and biased forward shapes dispatch through the SAME `layer_norm_fused` admit key — one admission key, one dispatch-proof counter pair, regardless of which shape actually ran — but the underlying kernel mechanism is NOT one shared row-math definition reused across the bias-free and biased arms. On CUDA, every pre-existing bias-free kernel symbol is append-only: byte-for-byte untouched, unreachable from anything the biased site adds. Each biased CUDA entry point (`layer_norm_fwd_{f32,bf16,f16}_biased`) instead carries its OWN per-dtype row-body template (`ln_fwd_row_body_f32`/`ln_fwd_row_body_bf16` in `crates/jammi-kernels/src/cuda/layer_norm.cu`, `ln_fwd_row_body_f16` in `crates/jammi-kernels/src/cuda/layer_norm_f16.cu`) — a SEPARATE, textually duplicated copy of that dtype's row math, not a definition the bias-free kernel also calls. This is an accepted drift surface, the direct cost of keeping the bias-free kernel's bytes provably untouched rather than refactoring a shared body two call sites would depend on; bit-identity of the bias-free arm therefore holds by construction (nothing below the pre-existing kernels is reachable from them), not by ongoing discipline to keep two reachable copies in sync. On CPU, the row math IS genuinely shared: both the bias-free and biased row functions call the SAME `mean_var_f32`/`mean_var_bf16`/`mean_var_f16` helpers for their mean/variance reduction — only the epilogue (whether `beta` is added) differs between the two. Backward's new bias gradient, `dbeta_from_grad`, is an ORDINARY `Tensor` composition (`to_dtype` then `sum` over every batch dimension) — not a fused reduction kernel of its own, on either backend; a combined γ+β reduction kernel would compute an unneeded extra `dgamma` launch in the one lattice cell that actually needs `dbeta` without `dgamma`. The eval-mode arm is untouched by any of this: eval never reaches the training-only fused arm regardless of whether the norm the caller loaded is biased, so serving numerics carry no dependence on this widening at all. See `crates/jammi-kernels/src/ops/layer_norm.rs`'s own module doc for the exact predicate and kernel-launch shape this widening lands under.

**GeGLU** (`GegluFused`, CustomOp1 over the packed `wi_out`). Four tape tensors become one node. **It rounds twice on purpose**: `act = bf16(gate·Φ(gate))`, then `out = bf16(f32(act)·up)`, because the reference — HF's two-op `act(gate) * up` and the `kernels-community` kernel — itself rounds the activation before the multiply. One rounding would be a *different*, over-precise computation. Backward: `d_gate = dy·up·(Φ + gate·φ)`, `d_up = dy·gate·Φ`, one launch.

**RoPE** (`RopeFused`, CustomOp3 over (x, cos, sin)). One thread per element: `out = bf16(x·cos + rotate_half(x)·sin·sign)`. Backward = forward with the sine negated: pairing columns j and j+half, the forward is the rotation `[[cos, −sin],[sin, cos]]`, its Jacobian is the same orthogonal matrix, and `dx = Jᵀ·dy` is the rotation by −θ.

**Masked softmax** (`SoftmaxLastDimFused`, CustomOp2 over (scores, mask)). Replaces the scale affine + mask add + max/sub/exp/sum/div, each keeping a `[B,H,S,S]` intermediate — the memory lever (~22 GB at seq 1024, batch 8). Output-only backward `dscores = (dy − Σ dy·y)·y`, so nothing quadratic survives the forward except `y`.

> **The bf16-boundary trap.** Eager's `broadcast_add` on bf16 rounds `scores + mask` to bf16. At `MASKED_LOGIT = −10 000` the bf16 ULP is **64**, so every masked score annihilates to the same value and a fully-masked row becomes uniform `1/n`. A first fused version added the mask in f32, produced `softmax(scores)` on those rows, and diverged O(1) — invisible to every f32 oracle. Fully-masked rows are reachable (pad queries). The kernel reproduces eager's destructive rounding at exactly that step. **"Match eager" at bf16 means matching *where* the reference rounds, not rounding as little as possible.**

**Attention block** (`AttentionBlockFused`, CustomOp3 over (qkv, rope_pack, mask)). No new kernel: a composed interior of cuBLAS strided-batched GEMMs, gathers into contiguous scratch, and direct calls into the RoPE and softmax kernels, all inside one `cuda_fwd`. `head_dim == 64` is load-bearing: `1/√64 = 0.125` is a power of two, so scaling Q before the GEMM is bit-exact to scaling scores after. The backward recomputes P and issues five gradient GEMMs through one shared definition of a gradient GEMM's operand form — because of esc-044 (§5).

**LoRA site** (`LowRankResidualLinear`, CustomOp3 over (x, W, [Aᵀ; B; bias-block])). `ab`'s dim-0 pack holds `Aᵀ` then `B`; a bias-carrying frozen base (every BERT/DistilBERT LoRA site) rides as a third block, `ceil(out/rank)` zero-padded `F32` rows appended to the same stack — the row-axis layout is what keeps every GEMM slice a zero-copy `narrow` whether or not that third block exists. Forward: `out = round_once(f32(x@Wᵀ ⊕ b) + scale·(dropout(x)@Aᵀ@Bᵀ))`, where `⊕ b` is candle's own storage-level `broadcast_add` in the BASE dtype — bit-identical to eager `Linear::forward` (the module doc's step 1b) — before the sum widens to f32 and rounds ONCE on the way out; W frozen, and candle's `Op::Matmul` backward would compute a `dW` GEMM for it regardless — the fused op returns `None`. jammi picks bit-exact parity with candle-eager over torch/PEFT's own single-rounding cuBLASLt epilogue fold (unreachable from candle, which never exposes an f32 accumulator out of `matmul`) — see the op's own module doc for the full three-variant rounding enumeration and every citation behind it, rather than restating them here. The honest cost of admitting a bias: one small `to_dtype` cast plus one `broadcast_add` launch per site per forward (paid even at F32 — `to_dtype` never no-ops a same-dtype pair), and one tiny `zeros_like` of the padded bias block per site per backward (so `Tensor::cat`'s own backward has a same-shaped slot to narrow, even though the frozen bias itself contributes no gradient of its own). Node counts (`crates/jammi-kernels/src/ops/low_rank_residual_linear.rs`'s own oracles): bias-free eager 10 / fused 6 (`fused_site_retains_fewer_tape_nodes_than_the_eager_composition`); bias-carrying eager 11 / fused 6 (`fused_site_with_bias_retains_fewer_tape_nodes_than_the_eager_biased_composition`) — the eager arm's own bias add is itself a new tracked node, while the fused arm's cost is unchanged because the pack just grows a `Tensor::cat` argument. The cast-boundary fusions (#377) live in its backward: `f32(x)·scale + 0.0f` (the `+0.0f` is required for signed-zero identity with candle's `affine_f32`) and round-then-native-bf16-add, both proven bit-identical to the chains they replace.

**Measured: the bias-carrying site's activation gain (#428).** The #356 close-out profile's ablation census projected `C-LORA` (this site) at `ACTIVATE` for both BERT and DistilBERT from a same-build ablation differential, pending a real fused-vs-eager port and measurement (`crates/jammi-kernels/artifacts/cuda-runs/2026-08-31-profile-356-closeout-7820d697-a100-sxm4.json`). `ci/scripts/perf/lora_bias_ab.sh` closed that out on an A100 80GB PCIe (n=3 repeats/cell, per-step wall `(wall_600 − wall_100)/500`, both arms under `JAMMI_KERNELS_STRICT=1`, eager forced via `JAMMI_KERNELS_DISABLE=lora_linear_fused`, every leg counter-proved via `lora_linear_fused_dispatches`), at `b8W512f32` (the wire shape the fused-vs-fused control floor is measured on) and `b32W64bf16`. BERT's forced-eager arm measured 0.933241 s/step against 0.763558 s/step fused at the wire shape (an 18.18% gain over a 0.35% same-arm control floor) and 0.237753 vs 0.164589 at `b32W64bf16` (30.77%); DistilBERT measured 0.469170 vs 0.388349 at the wire shape (17.23% over a 0.12% floor) and 0.122467 vs 0.087061 at `b32W64bf16` (28.91%). Both models ACTIVATE at both shapes. The BERT legs loaded the stock, unconverted `bert-base-uncased` checkpoint (26 `gamma` / 26 `beta` tensors) through the #423 legacy-name loader, never the gamma/beta-renamed form.

| model | shape | fused s/step | control s/step | gain vs eager | floor | verdict |
|---|---|---:|---:|---:|---:|---|
<!-- claims: c1=crates/jammi-kernels/artifacts/cuda-runs/2026-09-05-lora-bias-428-c69dbd7-a100-pcie.json#/notes/verdicts/bert/fused_median_wire_s_per_step; c2=crates/jammi-kernels/artifacts/cuda-runs/2026-09-05-lora-bias-428-c69dbd7-a100-pcie.json#/notes/verdicts/bert/control_median_s_per_step; c3=crates/jammi-kernels/artifacts/cuda-runs/2026-09-05-lora-bias-428-c69dbd7-a100-pcie.json#/notes/verdicts/bert/gain_by_shape/b8W512f32 as %; c4=crates/jammi-kernels/artifacts/cuda-runs/2026-09-05-lora-bias-428-c69dbd7-a100-pcie.json#/notes/verdicts/bert/floor as % -->
| BERT | b8W512f32 | 0.763558 | 0.766233 | 18.18% | 0.35% | ACTIVATE |
<!-- claims: c1=crates/jammi-kernels/artifacts/cuda-runs/2026-09-05-lora-bias-428-c69dbd7-a100-pcie.json#/notes/verdicts/bert/gain_by_shape/b32W64bf16 as % -->
| BERT | b32W64bf16 | — | — | 30.77% | — | ACTIVATE |
<!-- claims: c1=crates/jammi-kernels/artifacts/cuda-runs/2026-09-05-lora-bias-428-c69dbd7-a100-pcie.json#/notes/verdicts/distilbert/fused_median_wire_s_per_step; c2=crates/jammi-kernels/artifacts/cuda-runs/2026-09-05-lora-bias-428-c69dbd7-a100-pcie.json#/notes/verdicts/distilbert/control_median_s_per_step; c3=crates/jammi-kernels/artifacts/cuda-runs/2026-09-05-lora-bias-428-c69dbd7-a100-pcie.json#/notes/verdicts/distilbert/gain_by_shape/b8W512f32 as %; c4=crates/jammi-kernels/artifacts/cuda-runs/2026-09-05-lora-bias-428-c69dbd7-a100-pcie.json#/notes/verdicts/distilbert/floor as % -->
| DistilBERT | b8W512f32 | 0.388349 | 0.387883 | 17.23% | 0.12% | ACTIVATE |
<!-- claims: c1=crates/jammi-kernels/artifacts/cuda-runs/2026-09-05-lora-bias-428-c69dbd7-a100-pcie.json#/notes/verdicts/distilbert/gain_by_shape/b32W64bf16 as % -->
| DistilBERT | b32W64bf16 | — | — | 28.91% | — | ACTIVATE |

Producer `ci/scripts/perf/lora_bias_ab.sh`; artifact `crates/jammi-kernels/artifacts/cuda-runs/2026-09-05-lora-bias-428-c69dbd7-a100-pcie.json` (NVIDIA A100 80GB PCIe, driver 570.172.08, RunPod pod f8pflfnhmuorsg); n=3 repeats per cell; per-step wall by `(wall_600 − wall_100)/500`; both arms run under `JAMMI_KERNELS_STRICT=1`. "control s/step" is the fused-vs-fused control arm (same binary, forced-off nothing) the "floor" column derives from — `|fused − control| / fused` — and is measured only at the wire shape; `b32W64bf16`'s absolute fused/eager times are not a committed top-level field of the artifact (only its `gain_by_shape` fraction is), so per this guide's own rule (h) the table cites only what `ci/scripts/check_perf_claims.py` can bind by value against a tracked field — the full-precision absolutes for that shape are given in the paragraph above instead, which is prose and not gated the same way (see the Sources note at the top of this guide).

**AdamW** (in-place via `InplaceOp2/3`). A `CustomOp` returns new storage; splicing it into a `Var` costs a D2D memcpy per Var per step (672 of the step's 1,345). `InplaceOpN` mutates the Var's storage in place: three launches per Var, zero copies. Bit-identity to candle's eager chain required pinning every op with `__fmul_rn`/`__fadd_rn` and reproducing candle's `x*mul + 0.0f` signed-zero laundering; a red-control kernel using `fmaf` proves the harness sees that class.

### FlashAttention-2, the one tensor-core arm

Vendored Dao-AILab v2.8.3.post1 with no edits, CUTLASS as a pinned submodule; only `run_mha_{fwd,bwd}<bf16, 64, false>` compile — one native cubin per arch in the compiled set (the encoder refuses any device outside that enumerated set — see `third_party/flash-attention/VENDORED.md`'s "Supported archs" for the current membership and per-arch validation status). Three shim headers replace the PyTorch includes; a torch-free C ABI turns every `TORCH_CHECK` into a status code with a `struct_size` guard. Hand-rolled nvcc with upstream's flags including `--use_fast_math` — the one exempt translation unit, because every upstream wheel and parity oracle is calibrated to it.

Sequences are packed into `qkv [total_q, 3, H, 64]` with a `cu_seqlens [B+1]` prefix sum; the kernels index from host totals and do not cross-check against the device array, so `CuSeqlens` is a validated type whose geometry is derived from the same host lengths, with an `unsafe` escape hatch. The forward has no atomics; the backward is deterministic iff `deterministic` (private zeroed `dq_accum` splits reduced in fixed order) — pinned on. Projected determinism cost 1.6–1.7×; measured at b8: 0.3733 vs 0.3759 ms, free (`s2:137`). Projected kernel cost 61–85 ms/step; measured ~39.

**One op over (qkv, cos, sin).** The first wiring composed a RoPE op with the flash op and regressed VRAM by +1.8–2.5 GB. Attribution: candle's `BackpropOp::new3` clones every tracked argument into the result's node, so the two-op form retained a *second* rotated buffer per layer. `FlashVarlenAttentionFusedRope` rotates at the storage level inside one `cuda_fwd` and recomputes the rotation in `bwd`: bit-identical values, flash peak 18.91 → 16.43 GB, now 0.2 GB below the block arm (`s2:214`).

**Windows and the decline lattice.** ModernBERT alternates global and local layers (`local_attention = 128`); the flash arm passes radius 64 as `window_size = (64, 64)`, cross-checked against FA's `mask.h`, HF's `sliding_window − 1` adapter and HF's eager mask, with `w±1` as red controls. `op_disabled` is consulted FIRST, before any predicate work — an operator-disabled run short-circuits straight to the block arm, never paying for the gates below. Then the cheap capability gates: feature compiled? CUDA? arch in the compiled set? bf16? head_dim 64? Then the domain fences on the resolved per-row lengths: `trusted_lengths_match_mask` (caller-supplied lengths are validated against the mask on-device, never trusted bare), `mask_is_prefix_every_row`, `every_row_length_ge_1`. A miss anywhere declines to the block arm (counted), never errors under `Strict`. Density is no longer a decline term — `build_flash_forward_decision` fuses BOTH a dense and a genuinely-padded batch once every gate clears; `is_dense` only picks which arm dispatches (no transport for dense, reason `domain_ok_dense`; the encoder-boundary unpad/repad transport for padded, reason `domain_ok_padded`) — the 1.07× is measured on the dense branch only, and the padded-shape measurement is pending its own pod run.

### The bench and its torch twin

`jammi-bench finetune-step`: three encoder forwards, a triplet hinge, one backward, one AdamW step; synthetic uniform token ids, so it measures *cost*, never learning. `torch_finetune_step.py` is matched argument for argument (`attn_implementation` read back from the config; `--attn eager` = semantic twin, `--attn sdpa` = the throughput bar; LoRA init distribution-matched; TF32 off). `ab_merge.py` refuses to compare legs whose `FINETUNE_IDENTITY_FIELDS` differ (the tuple is declared once, in `ci/scripts/perf/identity_fields.py`, and imported — 18 entries, including the padded-batch `row_lengths` vector); the raw attention string (`attn_requested`/`attn_implementation`) is recorded as provenance and never compared, while the reference *class* it implies is compared via the `attention_arm` identity field; the clip determinant `max_grad_norm` is in the comparison tuple (null = clip off is a value, never MISSING) and in the K7-completeness const (`FinetuneStepTier::IDENTITY_FIELDS`, a strict superset). A stdlib-`unittest` suite (`ci/scripts/perf/test_identity_fields_subset.py`) pins the claim mechanically: every Python comparison-tuple entry must be named in the corresponding Rust K7-completeness const, and the tuple cardinalities (18, 11) are asserted as numbers, not promises.

---

## 5. Numerics at the bf16 boundary

bf16 keeps f32's 8-bit exponent and 7 explicit mantissa bits. ULP at 1.0 ≈ 0.0078; at 100 it is 1.0; at ~6,700 (the layer-18 residual magnitude) 32; at −10,000, 64. There are 2¹⁶ f32 ULPs per bf16 ULP. Every jammi kernel reads bf16, accumulates in f32, and rounds at the store; cuBLAS bf16 GEMMs accumulate in f32 (`CUBLAS_COMPUTE_32F`).

### Rounding placement: three cases where the obvious fix was wrong

- **esc-047 — the eager arm was the outlier.** jammi's *eager* LayerNorm rounded x̂ to bf16 before multiplying by γ; eager RoPE rounded three times. A domain agent proposed changing the *fused* kernel to match. ATen's `layer_norm_kernel.cu` computes bf16 LN in float and casts once; HF's `apply_rotary_pos_emb` does `(q.float()*cos) + (rotate_half(q.float())*sin)` then one `.to(dtype)`. The fix went the other way (PR #382; 67,546 of 262,144 elements differed).
- **esc-046 — round-then-add vs add-then-round.** PEFT computes `result(bf16) + delta(f32)` in f32 and casts the sum once; jammi rounded the delta first. 176/4096 elements differ, max one ULP at |base| ≈ 100. Both jammi arms carry the same deviation, so a same-build A/B is blind to it — why "agreement is not accuracy" is a rule. Fixed on main (`6be93f0`): the LoRA epilogue now rounds once at the wider dtype, matching PEFT, with FMA pinned in all four `scaled_cast_add` kernels.
- **The softmax backward, validated at the wrong call path.** Round 3 of the esc-045 fix rounded `dy·y` to bf16 because ATen's bf16 `SoftMax.cu` does. HF ModernBERT never runs that path: `modeling_modernbert.py:180` computes `softmax(..., dtype=torch.float32)`, so the change moved jammi *away* from the reference. Validate against the model's **call path**, quoting the line and its dtypes.

### Reduction order: a GEMM's operand form is part of its numerics (esc-044)

candle's CPU `gemm` and cuBLAS choose packing/blocking/split-k from operand *strides*; a transposed view (OP_T) and a transposed copy (OP_N) of the same matrix can reduce in different orders. `AttentionBlockFused::bwd` materialised `pᵀ`, `vᵀ`, `dsᵀ` while candle's own `Op::Matmul` backward differentiates through views. Per GEMM the difference was `r(1) = [0, 1.04e-7, 3.87e-8]` — inside every bf16 bound, every oracle green — and it grew ~700–1000× to `r(28)`. On the real model at b8·s512 the fused arm's loss went flat (0.318 → 0.291) while the same build forced eager learned (→ 0.1006). Fix: one definition of a gradient GEMM's operand form; dropping the copies was also a 15% speedup. The oracle asserts *growth* against the same run's own `r(1)`: `r(28) ≤ 4·max(r(1), 1e-9)` — never an absolute constant, because an absolute floor is what hid the defect.

### The monotone-rounding argument

Round-to-nearest-even is monotone, so two pre-rounding f32 values less than one bf16 ULP apart land on adjacent grid points at most; a two-ULP bf16 gap needs a pre-rounding delta above 2⁻⁸ of the magnitude (~130,000 f32 ULPs), which neither FMA contraction nor a cancellation-free elementwise formula can produce. When a RoPE parity leg failed by 2 ULPs on one box and the bound was loosened from k=1 to k=2 "defensively", the audit refused with exactly this (`s4:27`): both arms round once; the observation is box-specific nondeterminism, and k=2 masks it. The bound went back to 1; the divergence got its own escape (esc-048; not reproducible an hour later under compute-sanitizer; green on two other boxes). *A bound is a derivation; a bound you cannot derive is a bound you cannot defend.*

### Smaller facts that bite

- FMA contraction is ~1 f32 ULP and fatal to a bit-identity claim; pin with `__fmul_rn`/`__fadd_rn`.
- libm `erff` vs hardware `erff`: GeGLU's two arms compute the error function differently, unlike RoPE/casts and every other pure-arithmetic op; its parity leg needs 8 cancellation ULPs where pure-arithmetic legs need 3.
- `half::bf16::from_f64` truncates to 32 bits before rounding and disagrees with `from_f32(x as f32)` on some inputs; never call a `from_f64` reference "single-rounding".
- Exact-integer fixtures (values in −4..4) make a cross-device GEMM bit-exact regardless of summation order.

---

## 6. The esc-045 saga: when the metric is the defect

On 2026-08-25 the jammi-vs-torch gradient oracle (#372's `torch_grad_oracle.py`: same checkpoint, the same LoRA weights on both stacks, one forward+backward, dropout 0) ran for the first time. Cosine of `dL/dB` over the 112 `lora_b` tensors, b8·s128, seed 42 (#374; `s2:26–27`):

| pair | mean cosine | notes |
|---|---:|---|
<!-- claims: c1=ledger; c2=ledger -->
| jammi f32 vs torch f32 | 0.9999998 | per-layer ‖g‖ identical to 4 sig figs |
<!-- claims: c1=ledger; c2=ledger -->
| torch bf16 vs torch f32 | 0.932 | 0 negative tensors |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger; c5=ledger -->
| jammi bf16 vs jammi f32 | **0.337** | 31 negative; ‖g‖ ratio 2–13× from layer 18 down |
<!-- claims: c1=ledger -->
| torch bf16-eager vs bf16-sdpa (noise floor) | 0.825 | |

jammi's math is right; its bf16 backward was 6.5× further from truth than torch's. Seven rounds followed. The bisection said any *one* of LayerNorm, GeGLU or softmax going eager restored ~0.9 ("three independent rounding errors do not compose that way"); the block and LoRA ops were bit-identical to their eager arms. Stream races and allocator garbage were refuted. The first compute-sanitizer "0 errors" had instrumented `env`, a process that ran no CUDA. Round 3's source-validated softmax fix pointed the wrong way (§5). Round 5 confirmed a mechanism (bf16 logits before softmax) that round 6 — the first run with a *live* LoRA init instead of ZerosB — retracted. Round 7 re-measured round 6's ranking at seed 43 and b8·s128, and it *reversed both times*. Then the torch column ran (`s4:11`, `s4:42`):

| operating point (b4·s128) | block-fused | eager | torch bf16 | flash (FA2 tip) |
|---|---:|---:|---:|---:|
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger -->
| gaussian, seed 42 | 0.610 | 0.767 | 0.796 | 0.790 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger -->
| post-step, seed 42 | −0.350 | −0.198 | **−0.201** | −0.319 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger -->
| gaussian, seed 43 | 0.300 | 0.451 | 0.425 | 0.104 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger -->
| post-step, seed 43 | 0.563 | 0.229 | 0.679 | 0.143 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger -->
| gaussian, seed 44 | −0.264 | 0.133 | 0.485 | 0.680 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger -->
| post-step, seed 44 | −0.070 | −0.087 | **−0.062** | 0.095 |

**Torch's own bf16 backward collapses too** (range −0.20…0.80 over six points). A single-step gradient cosine here has no resolving power for either stack. The only statistic with power is paired and sign-based: block − torch is negative at 6/6 points (p = 1/64); flash − torch is 4 positive / 6 negative over ten points — no consistent sign. The FA2 arm removed the systematic deficit; what remains is the metric's chaos.

**What the saga teaches.** A real asymmetry was found by a real oracle — and then the oracle, used past its resolving power, spent seven rounds failing to name an op. #374 remains open; the honest closing instrument is a learning curve — a few hundred steps on a real pair dataset with a held-out evaluation, three arms, ≥3 seeds, accept if the fused arm sits inside the seed spread of the other two. The bench's own losses cannot do this: its synthetic triplets saturate to loss 0 within 25 steps for every arm.

---

## 7. How to prove a fused kernel

Each rule cites the `cuda-kernel-guide.md` §3 discipline it is an instance of, the
kernel-oracle-standard id (`KO-1`..`KO-8`, `cuda-kernel-guide.md` §3) that covers it, and the
escape that paid for it. The standard's mechanical subset is live in CI: `KO-2`/`KO-5`/`KO-7`
are enforced by `ci/scripts/check_kernel_oracles.py` (wired in `ci.yml`), `KO-3` by
`ci/scripts/check_cuda_run_artifacts.py` (optional per artifact leg), `KO-4` by
`ci/scripts/check_doc_numbers_have_producers.py` (advisory scan leg); `KO-1`/`KO-6`/`KO-8` are
auditor-only by design (each needs running code or a judgment a static scan cannot make). A rule
with no close §3 analog (2, 12 — this guide's own additions, not folded into the kernel guide)
prints `—` in the guide column; a rule whose substance no KO id mechanizes prints
"judgment-level" in the KO column.

| # | rule | `cuda-kernel-guide.md` | KO | the escape that paid for it |
|---|---|---|---|---|
| 1 | Two references; never mixed — semantic/rounding parity targets torch/HF `eager` (the model's call path) and PEFT for the LoRA epilogue; the throughput bar is torch-sdpa; FA2 op parity is torch's vendored FA2 kernel | §3.3 agreement is not accuracy | KO-8 (auditor-only) | declaring sdpa "the reference" would have invalidated every rounding decision; jammi-eager is never a reference for jammi-fused unless pinned to torch-eager at that site |
| 2 | The oracle lives in the crate that owns the arm | — | KO-8 (auditor-only — an in-test reimplementation is exactly the non-independent reference KO-8 refuses) | #382's first oracles compared a kernel to an in-test reimplementation updated in the same diff — reverting production left them green; the biting tests call the real `LayerNorm::slow` / `RotaryEmbedding::apply` in `jammi-encoders` |
| 3 | Same-build forced-arm A/B, elementwise, across a shape sweep including batch 1 — compare loss *sequences*, never a loss *value* | §3.6 the learning gate | KO-1 (auditor-only) | caught esc-044 at b8·s512 (b8·s128 hid it) and a second defect at b1·s512; its limit is esc-046, where both arms carry the same deviation |
| 4 | RED controls per leg, conjunctive, with a printed magnitude — a *real semantic mutant* through the op's public surface, never an edit to the test's own copy of the array; a control must assert on *its own* leg, measured on hardware | §3.7 write comparisons affirmatively | KO-1 (the mutant is real; auditor-only) + KO-2 (bound coverage; check_kernel_oracles.py) | the FA2 controls were `!out_ok \|\| !dqkv_ok` and the gradient arm had never fired; the fix asserts `out_ok && !dqkv_ok` on a backward-only window drop, GREEN with the injection off; a dropped-scale control was inside its bound on real hardware |
| 5 | No absolute floors — bounds are per element and relative, with any near-zero floor *measured* from the same run | §3.8 no absolute ULP floor | KO-3 (check_cuda_run_artifacts.py, optional per artifact leg) | the pre-#386 bf16 legs bounded with a fixed floor for every element below 1.0; esc-044's signature fit inside it |
| 6 | Bounds hold off-sample, not fitted to the seeds on hand | §3.2 key the oracle on growth against the same run's own r(1), never a fitted constant | KO-5 (check_kernel_oracles.py) | the FA2 encoder oracle's seed-fitted bounds gave false RED on fresh seeds and overlapped the mutants; deleted, not re-fitted |
| 7 | Live signal — the fixture's own cotangent/gradient must be nonzero before a mutant can be seen | §3.5 zero dispatch is RED (same failure shape one level up: a leg that cannot register a defect) | KO-6 (auditor-only — a static scan cannot evaluate a tensor norm) | a gradient leg's loss was identically the batch size; a real mutant *improved* and passed; fixed with a seed-keyed random cotangent and a nonzero reference sum |
| 8 | Unrun is RED — a GPU-less host must never silently pass as green | §3.5 zero dispatch is RED | KO-7 (check_kernel_oracles.py, total over every scanned file) | CUDA tests skip on a GPU-less host unless `JAMMI_REQUIRE_CUDA`/`JAMMI_REQUIRE_FLASH` is set, in which case a missing device panics instead of reading green |
| 9 | Producers for every number — a doc comment or artifact must cite what produced it | §3.9 no number without a producer | KO-4 (check_doc_numbers_have_producers.py, advisory scan leg) | the repo's only committed b8·s512 artifact once provenanced the *defective* pre-esc-044 build |
| 10 | The two-term Higham bound, per leg — a relative term plus an absolute term at the operands' own scale | §3.8 no absolute ULP floor (the two-term form is how the floor is derived, not assumed) | KO-3 (the §3.8 family; the two-term derivation itself is auditor judgment) | one shared absolute term dominated the elementwise legs until split by reduction term count |
| 11 | Cotangent fixtures — fixed, sign-mixed, production-amplitude, never `dy = 1` | §3.4 test at production shape and amplitude | judgment-level (§3.4 carries no KO id; the vacuous-cotangent failure it prevents is KO-6's territory) | under `dy = 1`, LayerNorm's centered backward is identically zero and the leg compared 0.0 to 0.0 |
| 12 | Mutation testing on touched files | — | judgment-level (no mechanical gate) | a crate-wide `cargo mutants` at the P2 tip found survivors five audit rounds had missed |

> **The principle under all of it.** An acceptance oracle is a claim that a defect would be caught, not that a number was computed. Before it may gate, it must be shown to be in a state where the defect it excludes could register. Every apparatus failure above — constant loss, bound wider than the metric's range, seed-fitted bound, a leg the defect does not reach, a control perturbing the test rather than the producer, a control vacuous on hardware, a skip that reads green — is that one bug. The FA2 kernel was verified once and its oracle rebuilt six times; the gate that ends that loop is a checklist the auditor verifies conformance to (`s4:22`; the mechanical subset is enforced by `ci/scripts/check_kernel_oracles.py`).

**What a static gate cannot see (#383).** A per-op ablation gate's budget was `3·(max − min)` of the reference arm's own seed spread: 2.49–3.80 on the artifact, while the gated quantity is a difference of two cosines, ≤ 2.0 by construction. An op whose ablation made the gradient exactly anti-parallel *passed*. The reference arm's spread was 1.27 at a median cosine of 0.016 — no resolving power. Check a budget against the metric's *range* before checking it against data; a derived budget inherits the noise of its source; measure the reference stack's spread at the operating point first. esc-044 escaped through a floor too *low* to see 1e-7; #383 could not fire because its budget was too *high* to see −1. Same error.

---

## 8. Measuring honestly

Three of the disciplines below are general (they apply to any fused-kernel benchmark, not just this
track) and now live in `cuda-kernel-guide.md` §4: exclusive box + timing lock; ratios travel across
boxes while milliseconds do not; attribute a kernel-time delta by grid, not by launch count. What
stays here is unique to this track's own runs:

- **VRAM is per box** (14.65 GB on driver 570 vs 16.36 GB on driver 595 for the same build). Torch 2.11 vs 2.13 changes nothing same-box (0.642 vs 0.643). A cu130 wheel on a driver-570 box silently runs on CPU.
- **Micro-benchmarks:** nothing allocated in the timed region (a 151 MB allocation read 5% of roofline where the kernel ran at 53–65%); per-iteration sync; ≥20 warmups, ≥200 iterations; min and median; run twice. Always end-to-end beside isolated.
- **Flag spreads, never smooth them.** The committed sweep prints the b8·s128 torch spread and the earlier 0.950× reading.
- **The product step is heavier than the bench step:** the trainer always clips (+12.7 ms at s512); every headline ratio is the unclipped bench step. Recorded, not hidden.
- **Time from `date -u`.**

---

## 9. Pods, builds, and the process

**The GPU dev loop.** `ci/scripts/gpu-dev.sh` rents disposable RunPod A100s (no network volume — it would pin one datacenter and delete the cloud-tier / PCIe-vs-SXM failover). Lessons paid in dollars: every pod carries its TTL in its *name* and a sweeper reaps it at that age regardless of in-pod timers — four dev pods vanished at once at the default 8 h (fixed: 72 h dev default, verify-before-terminate, refuse `up` over a live alias; #387/#388). "Pod never became reachable" was ssh-agent exhaustion — twelve local identities tried before the session key, `MaxAuthTries` hit, a healthy pod terminated, eight times in one night (`IdentitiesOnly=yes`, #358). `pgrep -f`/`pkill -f` matching their own command line stalled or killed the caller three separate times. Pin SHAs, not branch names. Agents never touch pod lifecycle.

**Build times, measured.** Cold `cargo build --release -p jammi-bench --features cuda` on a 252-vCPU pod: 284 s, 749 units; the serial tail is datafusion → jammi-db → jammi-ai → jammi-bench because the bench links the whole engine (762 crates). sccache as configured gives **zero** cross-target-dir reuse on the pod image and costs +33% wall (populate 457 s; the next fresh target 473 s with 187 hits / 1,121 misses; wrapper off 344 s) — after a first "warm" reading retracted because the cache held only debug units. The replacement (a per-pod seed target, cleaned member-free at seed time and copied per worktree) went through five pressure-test rounds, each refuting a mechanism on a fixture before code: rsync-preserved mtimes make a pushed change *older* than the seed's artifacts; bindgen_cuda's PTX skip check sits *below* cargo's fingerprint layer; cargo names test binaries by target, not package; a launcher-held `flock` dies with the launcher; `tmux -t jammi` prefix-matches every `jammi-<tree>`. Transferable rule: *derive every enumeration from the tool that owns the fact* — `cargo clean --workspace` instead of globs, the kernel's lock instead of a pid file, rsync's own file list instead of a git tree hash.

**The rigor chain, as it actually ran.** Every unit went scope → plan + pressure-test → contract → implement → adversarial + discipline audits + citation check → oracle → red/green verification → ship. Pressure-tests killed wrong designs before code (P4 capture on ROI; the attention-GEMM attribution; bounds transplanted from a different comparison pair; six lead-invented FA2 mechanisms that fell at source). Audits caught what compiled and passed its own tests (esc-044's growth; the constant-loss leg; the seed-fitted bounds; a fix that moved away from the reference; the k=2 relay).

> **The loop, and its cures.** Audit advisories turned into fix rounds (#382 ×6, #384 ×4, #387/#388 ×8); the FA2 kernel was verified once and its oracle rebuilt six times, each rebuild designed after the previous audit. Cures that held: *advisories are not work* — only a finding that changes a number or correctness gets a round; class B folds into the next commit. *Scoped re-audits* check the prior findings closed plus a new-defect sweep. *Probe the class before dispatching the fix*; a special-case with "does not apply to the others" is the loop's signature. *One push per PR*, after pod gates and the audit are green. And "no endless loops" means verify contract inputs at source before the pressure-test — never cap rounds or park units.

---

## 10. Where it stands

| b8·s512, dropout 0 | s/step | VRAM | box |
|---|---:|---:|---|
| eager, 2026-08-23 | OOM | — | PCIe |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger -->
| C-series end | 1.083–1.096 | 77.5–78.8 GB | PCIe |
<!-- claims: c1=crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p2-5932520-a100-sxm4.json#/bench_legs/4/s_per_step_p50; c2=crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p2-5932520-a100-sxm4.json#/bench_legs/4/peak_vram_bytes as GB -->
| P1 + P2 | 0.7817 | 39.58 GB | SXM4 |
<!-- claims: c1=crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p3-e32ed90-a100-sxm4.json#/bench_legs/5/s_per_step_p50; c2=crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p3-e32ed90-a100-sxm4.json#/bench_legs/5/peak_vram_bytes as GB -->
| + P3 (defective bwd) | 0.7742 | 17.06 GB | SXM4 |
<!-- claims: c1=ledger; c2=ledger; c3=ledger; c4=ledger; c5=ledger; c6=ledger -->
| + esc-044 fix, exclusive box | 0.6682 ± 0.002 | 14.65 GB | a100 / drv 570; torch 0.4292 → 0.642× |
<!-- claims: c1=crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-cast-w1-80f02fb-a100-sxm4.json#/legs/b8_s512_fused_r1/s_per_step_p50; c2=ledger -->
| + cast W1 (#377) | 0.6349 | +0.3 GB | SXM4 |
<!-- claims: c1=legacy(crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4.json#/a100b_full_step_ab_reference/summary/s512/fused_p50_r1_r2/0); c2=legacy(crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4.json#/a100b_full_step_ab_reference/summary/s512/disabled_eager_p50_r1_r2/0) -->
| + AdamW (#380) | 0.6589 (from an eager arm of 0.6759 — a base that predates W1) | — | timing box |
<!-- claims: c1=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s512/stacked_p50_min_s; c2=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s512/legs/stacked_r1/peak_vram_bytes as GB; c3=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s512/torch_p50_min_s; c4=crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json#/shapes/b8s512/ratio_torch_over_stacked -->
| + FA2, stacked (#389) | **0.4212** | 14.55 GB | a100c PCIe; torch 0.4512 → **1.071×** |

Rows are not one same-box chain; each row's comparison is in its box column.

On main: the discriminating parity legs (#386); esc-044 closed with a committed artifact (#390); the FA2 dense arm (#389, a merge commit rather than a squash: its cuda-run artifacts carry branch SHAs); the device-side gradient clip with torch-parity rounding (#381); the esc-046 LoRA-epilogue rounding fix (#398); the kernel-oracle-standard's mechanical gate (`ci/scripts/check_kernel_oracles.py`, wired in `ci.yml`); the pod-build substrate (#397, `docs/maintainer/pod-build-guide.md`); and, of the #356 close-out profile's four branch decisions, `C-LORA` — ported and measured (#428, §4): BERT and DistilBERT both ACTIVATE at both measured shapes. Open: #374 esc-045 with the learning-curve instrument as its closing metric; the padded FA2 regime; P5's 19 ms atom (fuse LRRL(Wi)+GeGLU); the `Kernel2` identity in the census; the flaky power test as its own unit; and the #356 profile's remaining three branches, `C-LN`/`C-ATTN`/`C-MLP`, each still `UNRESOLVED` in that profile's own verdict block.

---

## 11. Checklists

### Before you claim a kernel is faster

1. Same box, one build, both arms (forced via `JAMMI_KERNELS_DISABLE`), exclusive GPU under the timing lock.
2. Counters on every leg: fused == layers·steps, eager == 0; the all-off leg shows `requested == fired`.
3. Run twice; report min and median; print the band (exclusive-box noise ~±0.4%).
4. Sweep shapes including batch 1 and the longest supported sequence.
5. Micro-bench hygiene: no allocation in the timed region; sync per iteration; warmups; achieved GB/s and % of a named roofline constant.
6. Attribute by grid, not launch count.
7. Isolated *and* end-to-end; levers stack additively only when one stacked build says so.
8. VRAM same-box only; record driver, torch wheel, and that torch ran on the GPU.
9. Commit the artifact: tracked producer, `git_sha` an ancestor at a pushed commit, gate green; stamp `merged_as` the day of a squash merge.
10. Every number in a doc names its producer, or it is not written.

### Before you claim it is correct

1. Name the reference; quote the model's call-path line with its dtypes.
2. Validate rounding placement at ATen/HF/PEFT source, at production shape and amplitude, batch 1 included, with live (never ZerosB) adapter gradients.
3. Match the eager arm's GEMM operand form; assert strides on both sides.
4. Match *where* eager rounds at bf16 — the `MASKED_LOGIT` add is the trap.
5. Anchor both arms to an f32/f64 truth; agreement is not accuracy.
6. The oracle calls the real production function in the crate that owns the arm.
7. Every bound names the mutation it catches, prints its ratio, has no absolute floor, is asserted conjunctively per leg, and was proven RED on hardware.
8. Bounds derived off-sample; metrics shown seed- and shape-stable with the reference stack's own spread measured first; prefer paired or sign-based statistics.
9. Same-build forced-arm loss sequences elementwise identical across the sweep; growth oracles against the same run's own r(1).
10. Red→green proven by reverting the fix; a CUDA leg counts only when a committed artifact records it running.
11. Mutation testing on touched files; `RUST_BACKTRACE=1` before the PR.
12. Then the audit runs anyway — your spot-check is a pre-filter, never the gate.

---

## 12. Glossary

- **Roofline / arithmetic intensity** — throughput ceiling `min(peak FLOP/s, bandwidth × FLOPs-per-byte)`; low-intensity ops are bandwidth-bound.
- **Memory-bound vs launch-bound** — time ∝ bytes moved vs time ∝ number of launches (~5–10 µs each).
- **Autograd tape / tape node** — candle records each op as a node holding clones of its inputs; every node costs memory until released and a `zeros_like + add` per gradient contribution.
- **CustomOp** — candle's user-defined op with its own `cpu_fwd`/`cuda_fwd`/`bwd`; one tape node regardless of the work inside.
- **bf16 / ULP** — 8-bit exponent, 7 explicit mantissa bits; ULP is the spacing of adjacent values at a magnitude. Bounds are stated in ULPs of the element, never of the maximum.
- **FMA contraction** — nvcc fuses `a*b + c` into one rounding by default; pin with `__fmul_rn`/`__fadd_rn` where bit-identity matters.
- **Reduction order** — floating-point addition is not associative; tree reductions, cuBLAS blocking and view-vs-copy operand forms all change the bits.
- **varlen / `cu_seqlens`** — sequences packed back-to-back with a prefix-sum of row offsets instead of padding.
- **LSE** — per-row log-sum-exp saved in f32 by FlashAttention so its backward recomputes probabilities without materialising `[S,S]`.
- **Sliding window** — a local layer where row r attends keys with `|r − c| ≤ w`; ModernBERT alternates global and local (radius 64) layers.
- **Escape (esc-NNN)** — a defect that reached a branch or main past the gates, recorded with a falsifiable symptom spec and the gate that missed it.
