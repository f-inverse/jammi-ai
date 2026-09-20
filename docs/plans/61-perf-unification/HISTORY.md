# 61 — perf-unification: history of the fine-tune performance track

The narration of how jammi's fine-tune step went from 0.44× to 1.07× of PyTorch, moved verbatim
out of `docs/maintainer/fine-tune-performance-guide.md` (which keeps the current mechanism,
measured state and method). Section numbers and cross-references below are the guide's own at
the time the text was moved.

## Preamble

How jammi's Rust/candle training step went from 0.44× to 1.07× of PyTorch on the same GPU (2026-08-23 → 08-26), and what it took to prove the numbers were real. Written for an ML engineer who wants to understand the low-level computation of modern ML through one real track: candle `CustomOp`s, the autograd tape's cost, bf16 rounding placement, FlashAttention-2, and — above all — how to prove a fused kernel is both faster and faithful.

Sources: GitHub #352 / #356 / #374 / #428; PRs #357–#391; `docs/maintainer/cuda-kernel-guide.md`; and the committed run artifacts under `crates/jammi-kernels/artifacts/cuda-runs/`, which are the evidence a reader can re-verify. A row cite like `s2:245` points into session working notes that were never tracked in this repository: it records where a number came from, but no clone can re-resolve it. Where the sources disagree, the disagreement is printed, not smoothed.

Notation: `b8·s512` = batch 8 triplets (the bench batches anchor/positive/negative into one forward, so the attention batch is 24 rows), sequence 512. `s/step` is the p50 of measured optimizer steps. "Same box" means both legs ran on the same physical GPU in the same session. `s2:245` = session-2 ledger row 245; `s4:11` = session-4 ledger row 11; `esc-044` = a row of the escape ledger.

The bar was stated before the first measurement (PLAN.md, 2026-08-23, adopted verbatim into #352): *is jammi's optimizer step at least 0.9× as fast as PyTorch's on the same GPU, at equal precision, without OOM-ing where torch fits, and without changing what the model learns?* Units are seconds per optimizer step, peak VRAM, and later gradient fidelity vs an f32 truth — never GPU utilization, because "a naive kernel can be 100% utilised and slow".

What makes the track worth a guide is not the result but the path: two thirds of the effort went into *proving* that fused kernels were both faster and faithful, and the proofs failed more often than the kernels did. If you take one section, take §7.

## 1. The state before: busy, not productive

### The measurement that opened the track (#352, 2026-08-23T18:56Z; b8·s128, bf16, dropout 0)

| stack | s/step | triplets/s | peak VRAM | GPU util | mem-ctrl util |
|---|---:|---:|---:|---:|---:|
| PyTorch + PEFT (2024-era wheel) | 0.331 | 24.07 | 5.76 GB | 43% | 0% |
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

## 3. The levers: hypothesized vs measured

### Round one: the C-series (2026-08-23/24, PR #357)

Fuse the chains the profile named — LayerNorm (C2), RoPE (C3), masked softmax (C4), GeGLU (C5), the LoRA epilogue (C6), device Philox dropout (C7) — as candle `CustomOp`s with real backwards, behind a `cuda` feature.

| tip (b8·s128, PCIe, 15 steps) | s/step | Δ | modeled |
|---|---:|---:|---:|
| eager | 0.5915 | — | — |
| C2 LayerNorm | 0.5089 | −82.6 ms | ~90 |
| C3 RoPE | 0.4410 | −67.9 ms | ~84 |
| C4 softmax+mask | 0.3701 | −70.9 ms | ~79 |
| C5 GeGLU | 0.3097 | −60.4 ms | ~118 (whole MLP cluster; GEMMs never in scope) |
| C6 LoRA epilogue | 0.2750 | −34.7 ms | ~50 |
| C7 Philox dropout (d0) | 0.2695 | VRAM 33.5 → 22.4 GB | — |

`fusion rows 30, 36, 50`. Dropout 0.05: host mask path 1.364 s / 39.6 GB → device Philox 0.284 s / 19.7 GB.

**2.2× over eager, and NOT MET.** Rebuilt on the current stack (torch 2.13, transformers 5.15), torch's own b8·s128 step was 0.1184 s, not 0.331: the denominator had moved 2.8×. jammi-fused landed at 0.42–0.44× (`fusion row 52`); its VRAM delta was still ~5× torch's; b8·s512 with dropout OOMed where torch fit. *A comparison's reference is a moving part; rebuild it on the box you measure on, every time.*

Killed before it started: **C1b**, a vendored patch to candle's tape accumulation, on the no-candle-dependency rule. The plan estimated the residual tape cost at 22–66 ms; it was ~3× that.

### Round two: node count (2026-08-24, P1–P5)

Post-#357 census on an SXM4 (`cont row 11`): 251 ms GPU at b8·s128, of which `badd_bf16 + badd_f32` = 50%; attention ~7% (12% at s512). Torch same pod: 105 ms, 5,906 launches, 7 memcpys.

| lever | mechanism | predicted | measured (same box, SXM4) | verdict |
|---|---|---|---|---|
| **P1** fold 1/√d into fused softmax | deletes one `[B,H,S,S]` affine node per layer | −5.6 GB, ~12 ms at s512 | s512 1.078 → 1.038 s, 77.5 → 71.8 GB; s128 flat (+32 MiB, disclosed) | landed #362 |
| **P2** `LowRankResidualLinear` — one CustomOp3 per LoRA site, no dW for frozen W | ~11 → 3 nodes per site; 60% of the add mass | GPU 251 → 160–190 ms (first draft 140–150) | s128 0.2668 → 0.2098 (−21%), 23.7 → 8.7 GB; s512 1.037 → 0.780 (−25%), 71.9 → 39.1 GB | landed #363 |
| **P3** `AttentionBlockFused` — RoPE + QKᵀ + softmax + PV in one node, P recomputed in bwd | drop three retained `[B,H,S,S]` per layer | −16.6 GB at s512, +27 ms | s512 1.078 → 1.039 s, 77.5 → 60.85 GB (−16.65); then the esc-044 fix alone: 0.772 → 0.658 s | landed #368 |
| **P4** CUDA-graph capture | hide launch gaps | ≤0.5–6.7% (GPU busy ≈ wall) | — | killed at pressure-test |
| **P4b** device-side gradient clip (225 host syncs) | the product step always clips; the bench never did | −3.9% at s128 | clip on: +12.7 ms (s512) over the no-clip bench step — a baseline correction, not a saving | #373 cut → #381 |
| **P5** jammi-owned explicit backward | remove GradStore adds wholesale | 400+ → 155 → 100 → 62–93 → 33 → 19 ms | never built (six design rounds) | parked at the 19 ms atom |

Sources: `cont rows 12, 17, 34, 60, 63, 77; close rows 38, 54; s2:57–155`.

### Round three: the five-lever plan and the census (2026-08-25)

With P1+P2+P3 stacked and esc-044 fixed, an *exclusive* A100 measured: **b8·s512 jammi 0.668 s (band 0.58% over three runs) vs torch-sdpa 0.4292 s = 0.642×** (`s2:12, 22`). The bar needed ≤0.4769 s: remove 191 ms. The census of the 673 ms GPU step (a100b, nsys, grouped by kernel *and grid*):

| kernel | ms/step | launches | what it is |
|---|---:|---:|---|
| `badd_bf16` | 116.1 | 563 | 496 are GradStore accumulation |
| `softmax_fwd_bf16` | 105.6 | 56 | attention; 2/layer because the block bwd recomputes P; ~3.5% of roofline |
| `ucopy_bf16` | 83.9 | 364 | 100% inside the attention block |
| `Kernel2` | 52.7 | 364 | six grid configs; only `gridDim.z=384` (26.5 ms) are attention |
| `cast_bf16_f32` / `scaled_cast_add` / `cast_f32_bf16` | 40.4 / 21.0 / 11.1 | 337 / 112 / 112 | the LoRA site's dtype motion |
| `ampere_sgemm_32x128_nn` | 40.1 | 336 | the LoRA site's rank-16 f32 GEMMs |
| `affine_f32` | 25.8 | 2129 | 2016 AdamW; 112 the LoRA bwd |
| bf16 GEMMs (z=1) | 53.7 | 140 | *projection* GEMMs (Wqkv, Wi) — not attention |
| `rope_fwd` / `softmax_bwd` / `layer_norm_bwd_dx` | 19.3 / 18.8 / 9.0 | 168 / 28 / 56 | |

`s2:51, 70`. Total 673.0 ms GPU, 7,796 launches, 1,401 memcpys; torch-sdpa: ~422 ms, 5,672 launches, 7 memcpys. GEMMs matched torch kernel-for-kernel.

> **A lead error, corrected by the census itself.** The first reading attributed the 56/step bf16 GEMMs to attention, putting it at ~315–334 ms. The pressure-test pointed at the grid: those GEMMs have `gridDim.z = 1` and x-tiles 3072/5248 — the Wqkv/Wi *projections*. Attention-core kernels carry `z = b·h = 384`. Corrected attention-attributable: **243.8 ms**. The same census refuted "attention ∝ s², so FA2 cannot help at s128": at s128 the block's kernels are launch-bound (~38% of a 183 ms step).

The five-lever plan: AdamW ~26 ms, cast boundary ~72 ms, FA2 ~160 ms, the esc-045 bf16 fix as correctness gate, P5 ~102 ms as margin.

| lever | projection history | measured (one build, forced arm off/on) | PR |
|---|---|---|---|
| **Cast boundary W1** (fuse `cast+affine` and `cast+add` in the LoRA bwd) | 72.5 → 13.4 → 40–48 → 28–31 → 31–42 ms | b8·s512 0.6744 → **0.6349 (−39.6 ms)**; s128 0.1978 → 0.1883; bit-identical to the two-kernel chain | #377 |
| **AdamW in-place step** (`InplaceOp2/3`; 3 launches/Var, zero `Var::set` memcpys) | −40 → "~26" → ~30 wall → −20.5 isolated | optimizer phase 23.1 → 2.59 ms (8.9×); full step b8·s512 0.6759 → **0.6589 (−16.4 ms)**; s128 −16.5; bit-identical to candle's chain | #380 |
| **FlashAttention-2 dense arm** | −285 → −210..−240 → −140 → [159,183]; kernel 61–85 ms projected vs **39 measured** | b8·s512 block 0.6756 → **flash 0.4626 (−213 ms)** same box, 0.937× torch; s128 −43 ms | #389 |
| **P5** | 102 → 33 → 19 ms atom | not built | — |
| **esc-045 fix** ("the gate") | — | did not gate the shipped levers: FA2 replaces the softmax site; the others are bit-identical; the metric turned out chaotic (§6) | #374 open |

Sources: `s2:89, 97, 109, 137, 146, 151, 164, 192`. The AdamW "26 ms" was a mis-attribution (112 of the 2129 affines belong to the LoRA backward; AdamW's data volume is ~0.5 GB) — its credit was launch overhead, which is why the in-place design won over a true multi-tensor kernel that would have needed a candle patch.

## 6. The esc-045 saga: when the metric is the defect

On 2026-08-25 the jammi-vs-torch gradient oracle (#372's `torch_grad_oracle.py`: same checkpoint, the same LoRA weights on both stacks, one forward+backward, dropout 0) ran for the first time. Cosine of `dL/dB` over the 112 `lora_b` tensors, b8·s128, seed 42 (#374; `s2:26–27`):

| pair | mean cosine | notes |
|---|---:|---|
| jammi f32 vs torch f32 | 0.9999998 | per-layer ‖g‖ identical to 4 sig figs |
| torch bf16 vs torch f32 | 0.932 | 0 negative tensors |
| jammi bf16 vs jammi f32 | **0.337** | 31 negative; ‖g‖ ratio 2–13× from layer 18 down |
| torch bf16-eager vs bf16-sdpa (noise floor) | 0.825 | |

jammi's math is right; its bf16 backward was 6.5× further from truth than torch's. Seven rounds followed. The bisection said any *one* of LayerNorm, GeGLU or softmax going eager restored ~0.9 ("three independent rounding errors do not compose that way"); the block and LoRA ops were bit-identical to their eager arms. Stream races and allocator garbage were refuted. The first compute-sanitizer "0 errors" had instrumented `env`, a process that ran no CUDA. Round 3's source-validated softmax fix pointed the wrong way (§5). Round 5 confirmed a mechanism (bf16 logits before softmax) that round 6 — the first run with a *live* LoRA init instead of ZerosB — retracted. Round 7 re-measured round 6's ranking at seed 43 and b8·s128, and it *reversed both times*. Then the torch column ran (`s4:11`, `s4:42`):

| operating point (b4·s128) | block-fused | eager | torch bf16 | flash (FA2 tip) |
|---|---:|---:|---:|---:|
| gaussian, seed 42 | 0.610 | 0.767 | 0.796 | 0.790 |
| post-step, seed 42 | −0.350 | −0.198 | **−0.201** | −0.319 |
| gaussian, seed 43 | 0.300 | 0.451 | 0.425 | 0.104 |
| post-step, seed 43 | 0.563 | 0.229 | 0.679 | 0.143 |
| gaussian, seed 44 | −0.264 | 0.133 | 0.485 | 0.680 |
| post-step, seed 44 | −0.070 | −0.087 | **−0.062** | 0.095 |

**Torch's own bf16 backward collapses too** (range −0.20…0.80 over six points). A single-step gradient cosine here has no resolving power for either stack. The only statistic with power is paired and sign-based: block − torch is negative at 6/6 points (p = 1/64); flash − torch is 4 positive / 6 negative over ten points — no consistent sign. The FA2 arm removed the systematic deficit; what remains is the metric's chaos.

**What the saga teaches.** A real asymmetry was found by a real oracle — and then the oracle, used past its resolving power, spent seven rounds failing to name an op. #374 remains open; the honest closing instrument is a learning curve — a few hundred steps on a real pair dataset with a held-out evaluation, three arms, ≥3 seeds, accept if the fused arm sits inside the seed spread of the other two. The bench's own losses cannot do this: its synthetic triplets saturate to loss 0 within 25 steps for every arm.

## 9. Pods, builds, and the process

**The GPU dev loop.** `ci/scripts/gpu-dev.sh` rents disposable RunPod A100s (no network volume — it would pin one datacenter and delete the cloud-tier / PCIe-vs-SXM failover). Lessons paid in dollars: every pod carries its TTL in its *name* and a sweeper reaps it at that age regardless of in-pod timers — four dev pods vanished at once at the default 8 h (fixed: 72 h dev default, verify-before-terminate, refuse `up` over a live alias; #387/#388). "Pod never became reachable" was ssh-agent exhaustion — twelve local identities tried before the session key, `MaxAuthTries` hit, a healthy pod terminated, eight times in one night (`IdentitiesOnly=yes`, #358). `pgrep -f`/`pkill -f` matching their own command line stalled or killed the caller three separate times. Pin SHAs, not branch names. Agents never touch pod lifecycle.

**Build times, measured.** Cold `cargo build --release -p jammi-bench --features cuda` on a 252-vCPU pod: 284 s, 749 units; the serial tail is datafusion → jammi-db → jammi-ai → jammi-bench because the bench links the whole engine (762 crates). sccache as configured gives **zero** cross-target-dir reuse on the pod image and costs +33% wall (populate 457 s; the next fresh target 473 s with 187 hits / 1,121 misses; wrapper off 344 s) — after a first "warm" reading retracted because the cache held only debug units. The replacement (a per-pod seed target, cleaned member-free at seed time and copied per worktree) went through five pressure-test rounds, each refuting a mechanism on a fixture before code: rsync-preserved mtimes make a pushed change *older* than the seed's artifacts; bindgen_cuda's PTX skip check sits *below* cargo's fingerprint layer; cargo names test binaries by target, not package; a launcher-held `flock` dies with the launcher; `tmux -t jammi` prefix-matches every `jammi-<tree>`. Transferable rule: *derive every enumeration from the tool that owns the fact* — `cargo clean --workspace` instead of globs, the kernel's lock instead of a pid file, rsync's own file list instead of a git tree hash.

**The rigor chain, as it actually ran.** Every unit went scope → plan + pressure-test → contract → implement → adversarial + discipline audits + citation check → oracle → red/green verification → ship. Pressure-tests killed wrong designs before code (P4 capture on ROI; the attention-GEMM attribution; bounds transplanted from a different comparison pair; six lead-invented FA2 mechanisms that fell at source). Audits caught what compiled and passed its own tests (esc-044's growth; the constant-loss leg; the seed-fitted bounds; a fix that moved away from the reference; the k=2 relay).

> **The loop, and its cures.** Audit advisories turned into fix rounds (#382 ×6, #384 ×4, #387/#388 ×8); the FA2 kernel was verified once and its oracle rebuilt six times, each rebuild designed after the previous audit. Cures that held: *advisories are not work* — only a finding that changes a number or correctness gets a round; class B folds into the next commit. *Scoped re-audits* check the prior findings closed plus a new-defect sweep. *Probe the class before dispatching the fix*; a special-case with "does not apply to the others" is the loop's signature. *One push per PR*, after pod gates and the audit are green. And "no endless loops" means verify contract inputs at source before the pressure-test — never cap rounds or park units.

## 10. Where it stands

| b8·s512, dropout 0 | s/step | VRAM | box |
|---|---:|---:|---|
| eager, 2026-08-23 | OOM | — | PCIe |
| C-series end | 1.083–1.096 | 77.5–78.8 GB | PCIe |
| P1 + P2 | 0.7817 | 39.58 GB | SXM4 |
| + P3 (defective bwd) | 0.7742 | 17.06 GB | SXM4 |
| + esc-044 fix, exclusive box | 0.6682 ± 0.002 | 14.65 GB | a100 / drv 570; torch 0.4292 → 0.642× |
| + cast W1 (#377) | 0.6349 | +0.3 GB | SXM4 |
| + AdamW (#380) | 0.6589 (from an eager arm of 0.6759 — a base that predates W1) | — | timing box |
| + FA2, stacked (#389) | **0.4212** | 14.55 GB | a100c PCIe; torch 0.4512 → **1.071×** |

Rows are not one same-box chain; each row's comparison is in its box column.

On main: the discriminating parity legs (#386); esc-044 closed with a committed artifact (#390); the FA2 dense arm (#389, a merge commit rather than a squash: its cuda-run artifacts carry branch SHAs); the device-side gradient clip with torch-parity rounding (#381); the esc-046 LoRA-epilogue rounding fix (#398); the kernel-oracle-standard's mechanical gate (`ci/scripts/check_kernel_oracles.py`, wired in `ci.yml`); the pod-build substrate (#397, `docs/maintainer/pod-build-guide.md`); and all four of the #356 close-out profile's branch decisions are now ported AND measured: `C-LORA` (#428, §4): BERT and DistilBERT both ACTIVATE at both measured shapes; `C-LN` (#460, §4, `crates/jammi-kernels/artifacts/cuda-runs/2026-09-05-ln-bias-460-6c6d79c-a100-pcie.json`): BERT and DistilBERT both ACTIVATE at both measured shapes, landed by architectural direction rather than a pre-registered activation bar (§4's own landing-rule note); `C-ATTN` (#462, §4, `crates/jammi-kernels/artifacts/cuda-runs/2026-09-06-attn-block-462-869c65f-a100-pcie.json`): BERT and DistilBERT both ACTIVATE at both measured shapes (the largest gain of the four sites), landed by architectural direction; `C-MLP` (#463, §4, `crates/jammi-kernels/artifacts/cuda-runs/2026-09-06-gelu-erf-463-869c65f-a100-pcie.json`): BERT and DistilBERT both ACTIVATE at both measured shapes, landed by architectural direction, the wire-shape gain exceeding the census's own launch-count projection for the reason given in §4. The #356 close-out profile artifact itself (`crates/jammi-kernels/artifacts/cuda-runs/2026-08-31-profile-356-closeout-7820d697-a100-sxm4.json`) keeps its own `UNRESOLVED` strings verbatim — a committed artifact is never edited after the fact, even once the branch it named is resolved elsewhere; resolution lives in this guide and the two sites' own A/B artifacts, not by mutating the census. Open: #374 esc-045 with the learning-curve instrument as its closing metric; the padded FA2 regime; P5's 19 ms atom (fuse LRRL(Wi)+GeGLU); the `Kernel2` identity in the census; the flaky power test as its own unit; the three cross-modal towers' training-step profile, pre-registered and not yet measured (§4, "The cross-modal towers"); and the one gap the #462/#463 measurements name rather than close: the BERT family (BERT, DistilBERT) has no flash-attention transport wired at all — `attention_block_flash` declines by construction on every leg of #462's own A/B, a `CapabilityMiss` this guide's own §4 "Attention site" paragraph counts but does not close.
