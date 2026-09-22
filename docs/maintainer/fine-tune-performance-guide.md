# Fusing the fine-tune step — a guide to the performance track

How jammi's Rust/candle LoRA fine-tune step reaches 1.07× PyTorch's throughput at b8·s512 on the same GPU. Written for an ML engineer who wants to understand the low-level computation of modern ML through one real track: candle `CustomOp`s, the autograd tape's cost, bf16 rounding placement, FlashAttention-2, and — above all — how to prove a fused kernel is both faster and faithful.

The evidence a reader can re-verify is the committed run artifacts under `crates/jammi-kernels/artifacts/cuda-runs/`; `docs/maintainer/cuda-kernel-guide.md` is the general kernel discipline this guide applies. Where two measurements disagree, the disagreement is printed, not smoothed.

Notation: `b8·s512` = batch 8 triplets (the bench batches anchor/positive/negative into one forward, so the attention batch is 24 rows), sequence 512. `s/step` is the p50 of measured optimizer steps. "Same box" means both legs ran on the same physical GPU in the same session.

---

## 0. How to read this

**jammi** is an embedding engine on [candle](https://github.com/huggingface/candle) (Rust; eager execution; tape-based autograd). One of its jobs is fine-tuning an encoder — here `answerdotai/ModernBERT-large`: 28 layers, hidden 1024, 16 heads of dimension 64 — with LoRA adapters, in bf16, on a GPU. The comparison stack is PyTorch + HF Transformers + PEFT doing the identical work.

The bar: *is jammi's optimizer step at least 0.9× as fast as PyTorch's on the same GPU, at equal precision, without OOM-ing where torch fits, and without changing what the model learns?* Units are seconds per optimizer step, peak VRAM, and gradient fidelity vs an f32 truth — never GPU utilization, because "a naive kernel can be 100% utilised and slow".

The guide runs from cause to proof: §1 is the mental model of where a step's time goes; §2 names the levers and the one same-box measurement of all of them stacked; §3 is how each lever is built, including the BERT-family and cross-modal tower sites and their measured gains; §4 is bf16 numerics; §5 is how to prove a fused kernel (if you take one section, take §5); §6 is measuring honestly; §7 the pods and builds the measurements run on; §8 where the track stands and what is open; §9 the checklists; §10 a glossary. How the track unfolded — the opening measurements, each lever's measurement as it landed, and the investigation of the bf16 gradient metric — is recorded in `docs/plans/61-perf-unification/HISTORY.md`.

---

## 1. Where a step's time goes: the mental model

**Memory-bound work (the roofline).** Ceiling = `min(peak FLOP/s, bandwidth × FLOPs-per-byte)`. LayerNorm, softmax, RoPE, GeGLU, casts and the AdamW update do a handful of FLOPs per element, so their best time is *bytes moved ÷ HBM bandwidth*. House method (kernel guide §2): traffic = every input read once + every output written once; achieved GB/s = traffic ÷ time; divide by the A100-SXM4-80GB's 2039 GB/s. (Three roofline constants are in use for this hardware family — 1935 PCIe, 1555 for the 40 GB part, 2039 SXM4-80 — giving different "× off roofline" for the same kernel. Write the constant down.) A good bandwidth-bound kernel lands in the tens of percent; the shipped softmax forward measured 3.5%.

**Launch-bound work.** Each launch costs ~5–10 µs regardless of size. The fused AdamW step runs at 9.2–9.8 µs per call while element counts vary 5× — pure launch latency, 2–12% of roofline. Its lever is not bandwidth.

**Busy is not productive.** `nvidia-smi` "utilization" means *percent of time at least one kernel is resident*, not achieved occupancy. A GPU that is 100% busy while its memory controller idles at 25% is running many small, low-occupancy kernels back to back — busy, not productive, not bandwidth-limited. Eager candle has no fusion layer and no graph capture: every elementwise op is its own kernel reading from and writing to HBM, and every intermediate is pinned on the autograd tape.

**The tape tax.** When `Tensor::backward()` walks the graph, every gradient contribution goes through `GradStore::or_insert`, which on a vacant entry returns `zeros_like(tensor)`, and then *every* backward arm does `sum_grad = sum_grad.add(arg_grad)`. There is no "if vacant, move" fast path (`candle-core-0.11.0/src/backprop.rs`):

```
candle 0.11, every contribution incl. the first:
  grad arrives ──▶ or_insert → zeros_like (full-size fill) ──▶ sum_grad.add(arg) ──▶ stored
                              badd_bf16 ≈ 0.137 ms per launch on a [24,512,1024] tensor

torch AccumulateGrad:
  grad arrives ──▶ move (no kernel) ──▶ 2nd+ contribution: add
```

In a b8·s512 census of the step with the attention block fused but before the flash arm, `badd_bf16` = 563 launches/step = 27 layers × 19 nodes + 18 + 32; **496 (88%) are accumulation ≈ 102 ms of 116 ms**. A pre-norm transformer's tape is nearly a chain, so almost every node has one consumer and its add is pure waste. Fusing elementwise chains removes cheap kernels faster than it removes tape nodes, so accumulation's share of GPU time *grows* as elementwise fusion lands (39% → 50–53% after the first fused ops). **The lever is node count.**

jammi does not vendor, patch, or depend on upstream changes to candle, so the fix cannot be a move fast path in `or_insert`. It is fewer nodes: wider fused ops, each one `CustomOp` node regardless of the work inside.

---

## 2. The levers and the stacked result

Each shipped lever removes tape nodes, launches or copies; §3 is how each is built.

| lever | what it removes |
|---|---|
| Fused LayerNorm, RoPE, masked softmax, GeGLU, LoRA epilogue, device Philox dropout | a dozen-op chain per site, each op a kernel and a retained intermediate; host-generated dropout masks |
| 1/√d folded into the fused softmax | one `[B,H,S,S]` affine node per layer |
| `LowRankResidualLinear` — one `CustomOp3` per LoRA site | ~11 → 3 nodes per site, and the `dW` GEMM for the frozen W |
| `AttentionBlockFused` — RoPE + QKᵀ + softmax + PV in one node, P recomputed in bwd | three retained `[B,H,S,S]` per layer |
| Cast-boundary fusions in the LoRA backward | the LoRA site's separate `cast`/`affine`/`add` kernels |
| In-place AdamW (`InplaceOp2/3`) | one D2D `Var::set` memcpy per Var per step; 3 launches per Var |
| FlashAttention-2 dense arm | the block's composed attention interior |
| Device-side gradient clip | the host syncs of the clip the product step always runs |

Two levers are not built. CUDA-graph capture can hide only launch gaps, and GPU busy ≈ wall on this step, so its ceiling is 0.5–6.7%. A jammi-owned explicit backward that removes the remaining `GradStore` adds is bounded at a ~19 ms atom (fusing LRRL(Wi) + GeGLU).

### The stacked result (the `train-step` ladder; producer `ci/scripts/perf/finetune_step_ab.sh`)

| shape | jammi stacked s/step | torch-sdpa | jammi all-off | torch ÷ jammi |
|---|---:|---:|---:|---:|
| b8·s512 | **0.4212** | 0.4512 | 0.6554 | **1.071×** |
| b8·s128 | 0.1307 | 0.1319 (r1/r2 spread 8.3%) | 0.1892 | 1.009× |
| b1·s128 / b1·s512 | 0.0401 / 0.0738 | 0.1260 / 0.1216 | 0.0648 / 0.1157 | 3.14× / 1.65× |
| b16·s128 / b8·s256 | 0.2178 / 0.2206 | 0.2243 / 0.2273 | 0.3129 / 0.3219 | 1.030× / 1.030× |
| b16·s512 / b8·s1024 | 0.8186 / 0.8300 | 0.8790 / 0.9461 | 1.2650 / 1.4781 | 1.074× / 1.140× |

Artifact `crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json` + 40 raw runs; A100 80GB PCIe, driver 570.172.08, torch 2.13.0+cu126 / transformers 5.15.1 / peft 0.20.0. Counters on every stacked leg: flash 840 fused / 0 declined, AdamW 6720 fused / 0 eager. Caveats the artifact prints: the b8·s128 torch leg's own spread exceeds the margin at that shape (another run read 0.950×); the all-off leg disabled the flash and AdamW arms only; the ratio uses the min of two torch runs — the estimator least favourable to jammi.

---

## 3. How the levers are built

Paths are in `crates/jammi-kernels` unless stated.

### The substrate: one tape node, whatever happens inside

A fused op is a struct implementing candle's `CustomOp1/2/3` with `cpu_fwd` (the reference arm), `cuda_fwd` (launches PTX), and `bwd(args…, res, grad_res)` returning one gradient per argument. `tensor.apply_op3(y, z, op)` stores the op in the result's graph node; `backward()` reaches it and accumulates via `or_insert`. jammi routes every call through wrappers (`ops/mod.rs`) whose `KernelOp` bound is `Copy + Send + Sync + Sealed` — `Copy` is a structural proof of statelessness; `Sealed` keeps downstream crates from adding ops.

**candle 0.11 has no `save_for_backward`.** `bwd` gets only the arguments, the result and the incoming gradient; anything else must be recomputed. LayerNorm recomputes mean/rstd, GeGLU recomputes `gelu(gate)`, the attention block recomputes RoPE, scores and the probability matrix.

The exception is `Saved<T>` (`ops/saved.rs`): a write-once/read-once slot for FlashAttention's per-row log-sum-exp (`lse`, f32, a different shape than the output). `apply_op3` allocates a fresh `Arc` per call, so the slot is scoped to one forward; a second `bwd` on the node or a `bwd` before `fwd` is a typed error. Such ops cannot be `Copy`, so `StatefulKernelOp` exists with a grep-discipline test forbidding `Clone`/`Arc` wrapping.

**PTX.** `build.rs` compiles `src/cuda/*.cu` to PTX for `compute_80`; each op embeds its PTX with `include_str!` and loads it through candle's public `CudaDevice::get_or_load_custom_func`. nvcc's `--fmad=true` is left on; kernels needing bit-exactness pin operations with `__fmul_rn`/`__fadd_rn`. A header-only edit can serve *stale PTX*: a directory-level `rerun-if-changed` does not track edits to existing files, and bindgen_cuda's skip check compares a `.ptx`'s mtime against its own `.cu` only. Build systems are numerics.

### Admission: deciding fused vs eager, and proving which one ran

Every fused call site evaluates its own predicate (device, dtype, contiguity, shape bounds, each with a named reason) and passes it to `admit(mode, op, predicate_name, holds, counters)` (`admission.rs`). In `Strict` mode (`JAMMI_KERNELS_STRICT=1`, which the bench sets) a miss is an error, so "fell back everywhere" can never pass as a green fused measurement. Two atomics per op count `fused` and `eager` dispatches; the bench prints the delta over the timed loop. **Counters are the proof a kernel ran** — an end-to-end "learns on GPU" test that trains a head_dim-16 model never reaches the head_dim-64 kernel and stays green on a broken build; hence "zero dispatch is RED".

`JAMMI_KERNELS_DISABLE=op1,op2|all` forces ops eager *in the same binary* — the instrument behind every A/B. A typo never reads as success: `unmatched_disables()` turns a never-fired key into a hard error. Op keys are not flat: `attention_block_fused` subsumes `rope_fused` and `softmax_last_dim_fused` on the training path.

The flash arm has a three-way chain (flash → block → eager) where a decline means "try the next arm". A `bool` cannot say that, so the chain uses `PredicateOutcome { Holds, DomainMiss, CapabilityMiss }` and `admit_cascade` with a `declined` counter. The two-arm predicates stay `bool`: reclassifying their misses as never-erroring would defang `Strict`.

### Each fused op as a mini-lesson

Common shape: elementwise ops launch one thread per element in a grid-stride loop; reductions launch one 256-thread block per row with a shared-memory tree reduction. None of the hand-written kernels use warp shuffles or vectorized loads — the pattern the kernel guide's own §2 names as the 3.5%-of-roofline anti-pattern. jammi's wins are tape-node elimination and rounding control; the only tensor-core code is the vendored FlashAttention-2.

**LayerNorm** (`LayerNormFused`, CustomOp2 over (x, γ)). Replaces a dozen ops each retaining a `[rows, hidden]` intermediate. One block per row, three passes, f32 accumulation, one rounding at the store. Backward in one launch:

```
t          = dy · γ
mean_t     = Σ t / H          mean_t_x̂ = Σ (t · x̂) / H
dx         = rstd · (t − mean_t − x̂ · mean_t_x̂)        // layer_norm.cu
```

`dγ = Σ_rows dy·x̂` is two kernels to stay O(rows·hidden) with no `atomicAdd`; whether it is needed is frozen into the op from `gamma.is_variable()`.

**LayerNorm site, biased.** BERT and DistilBERT carry an affine bias on their LayerNorms (ModernBERT never does — `ModernBertConfig` cannot express one). The bias-free and biased forward shapes dispatch through the SAME `layer_norm_fused` admit key — one admission key, one dispatch-proof counter pair — but not through one shared CUDA row body. Each biased CUDA entry point (`layer_norm_fwd_{f32,bf16,f16}_biased`) carries its OWN per-dtype row-body template (`ln_fwd_row_body_f32`/`ln_fwd_row_body_bf16` in `crates/jammi-kernels/src/cuda/layer_norm.cu`, `ln_fwd_row_body_f16` in `crates/jammi-kernels/src/cuda/layer_norm_f16.cu`), a textually duplicated copy of that dtype's row math that the bias-free kernels never call. That duplication is an accepted drift surface: it is what makes the bias-free arm's bit-identity hold by construction (nothing the biased site adds is reachable from the bias-free kernels) rather than by keeping two reachable copies in sync. On CPU the row math IS shared: both row functions call the same `mean_var_f32`/`mean_var_bf16`/`mean_var_f16` helpers, and only the epilogue (whether `beta` is added) differs. The bias gradient, `dbeta_from_grad`, is an ordinary `Tensor` composition (`to_dtype` then `sum` over every batch dimension) on both backends; a combined γ+β reduction kernel would pay an unneeded `dgamma` launch in the one lattice cell that needs `dbeta` without `dgamma`. Eval never reaches the training-only fused arm, so serving numerics do not depend on it. `crates/jammi-kernels/src/ops/layer_norm.rs`'s module doc has the exact predicate and kernel-launch shape.

**GeGLU** (`GegluFused`, CustomOp1 over the packed `wi_out`). Four tape tensors become one node. **It rounds twice on purpose**: `act = bf16(gate·Φ(gate))`, then `out = bf16(f32(act)·up)`, because the reference — HF's two-op `act(gate) * up` and the `kernels-community` kernel — itself rounds the activation before the multiply. One rounding would be a *different*, over-precise computation. Backward: `d_gate = dy·up·(Φ + gate·φ)`, `d_up = dy·gate·Φ`, one launch.

**Dense erf-GELU, BERT family.** GeGLU above fuses the GATED `act(gate)·up` form; BERT's and DistilBERT's FFN activation is the other GELU shape — a single `x·0.5·(1+erf(x/√2))` with no gate multiply — called as `activations::gelu_erf(&hidden, training)` (`crates/jammi-encoders/src/bert.rs`) and `activations::gelu_erf(&mid, training)?` (`crates/jammi-encoders/src/distilbert.rs`). `GeluErfFused` (admit key `gelu_erf_fused`) is one `CustomOp1` node over exactly that call, with a one-kernel ATen-form backward (`dx = dy·(Φ(x) + x·φ(x))`) replacing the ~12-op composition candle's own `GeluErf` autograd rule expands `Tensor::gelu_erf()` into. It deliberately tracks a DIFFERENT cdf formulation than `GegluFused`: GeGLU's CPU reference calls `libm::erff` (`crates/jammi-kernels/src/ops/geglu.rs`), while this op matches candle's own `gelu_erf()` bit-for-bit at every dtype it implements — CUDA `normcdff`, CPU `erf_f32` — including the CUDA 16-bit arms' double rounding (the cdf itself rounds to the working dtype before the final multiply does). Two ops computing the same mathematical function through two different upstream references are kept apart on purpose, not reconciled into one implementation. It is wired behind the `training` flag at exactly these two call sites, so eval-mode serving numerics are untouched. The measured gain (below) is larger than a launch-count census projects: the fused backward removes ~12 launches AND the transient intermediates candle's expanded rule allocates per launch, a cost class a launch-count method under-weights.

**RoPE** (`RopeFused`, CustomOp3 over (x, cos, sin)). One thread per element: `out = bf16(x·cos + rotate_half(x)·sin·sign)`. Backward = forward with the sine negated: pairing columns j and j+half, the forward is the rotation `[[cos, −sin],[sin, cos]]`, its Jacobian is the same orthogonal matrix, and `dx = Jᵀ·dy` is the rotation by −θ.

**Masked softmax** (`SoftmaxLastDimFused`, CustomOp2 over (scores, mask)). Replaces the scale affine + mask add + max/sub/exp/sum/div, each keeping a `[B,H,S,S]` intermediate — the memory lever (~22 GB at seq 1024, batch 8). Output-only backward `dscores = (dy − Σ dy·y)·y`, so nothing quadratic survives the forward except `y`.

> **The bf16-boundary trap.** Eager's `broadcast_add` on bf16 rounds `scores + mask` to bf16. At `MASKED_LOGIT = −10 000` the bf16 ULP is **64**, so every masked score annihilates to the same value and a fully-masked row becomes uniform `1/n`. Adding the mask in f32 instead produces `softmax(scores)` on those rows and diverges O(1) — invisible to every f32 oracle. Fully-masked rows are reachable (pad queries). The kernel reproduces eager's destructive rounding at exactly that step. **"Match eager" at bf16 means matching *where* the reference rounds, not rounding as little as possible.**

**Attention block** (`AttentionBlockFused`, CustomOp3 over (qkv, rope_pack, mask)). No new kernel: a composed interior of cuBLAS strided-batched GEMMs, gathers into contiguous scratch, and direct calls into the RoPE and softmax kernels, all inside one `cuda_fwd`. `head_dim == 64` is load-bearing: `1/√64 = 0.125` is a power of two, so scaling Q before the GEMM is bit-exact to scaling scores after. The backward recomputes P and issues five gradient GEMMs through one shared definition of a gradient GEMM's operand form — because the operand form is part of the numerics (§4).

**Attention site, BERT family.** `rope: bool` and `fully_masked: FullyMaskedPolicy` are both `pub` fields on `AttentionBlockFused` (`crates/jammi-kernels/src/ops/attention_block.rs`), and `FullyMaskedPolicy::Propagate` (`crates/jammi-kernels/src/ops/softmax.rs`) is the op's generically-correct default — match candle-eager's own output on a fully-masked row, never the production-kernel zero-output convention `Zeros` asserts. BERT and DistilBERT train through a shared, model-agnostic per-layer cascade seam — the same `mem_efficient_attention → attention_block_fused → eager` admission chain ModernBERT uses — with `rope: false` (a per-module cached, never-read placeholder pack takes RoPE's third-argument slot) and `fully_masked: Propagate` (exact-arithmetic-equivalent to the eager composition on an all-padding row at f32 — `mean_pool`'s divisor-floor comment in `crates/jammi-encoders/src/pooling.rs` treats an all-padding row as a supported input; the bf16 divergence is disclosed in the seam's own doc). The flash TRANSPORT is not part of this seam — it is a ModernBERT-only encoder-boundary protocol (unpad/repad, rank-2 compaction) — so BERT/DistilBERT construct a declined flash decision at the seam edge: a named reason under `PredicateOutcome`'s `CapabilityMiss` (`crates/jammi-kernels/src/admission.rs`) — "the calling forward has not wired this arm's transport protocol" — counted on the block's cascade-decline counters every forward, never silently. The bridge from BERT/DistilBERT's separate `q`/`k`/`v` tensors to the packed `qkv` input is an ordinary `Tensor::cat`; its backward is not free (a `narrow` plus an `or_insert` zero-fill plus an add per argument, per layer, per step), and that cost rides on BOTH arms of this site's A/B, never hidden inside either arm's number.

**LoRA site** (`LowRankResidualLinear`, CustomOp3 over (x, W, [Aᵀ; B; bias-block])). `ab`'s dim-0 pack holds `Aᵀ` then `B`; a bias-carrying frozen base (every BERT/DistilBERT LoRA site) rides as a third block, `ceil(out/rank)` zero-padded `F32` rows appended to the same stack — the row-axis layout is what keeps every GEMM slice a zero-copy `narrow` whether or not that third block exists. Forward: `out = round_once(f32(x@Wᵀ ⊕ b) + scale·(dropout(x)@Aᵀ@Bᵀ))`, where `⊕ b` is candle's own storage-level `broadcast_add` in the BASE dtype — bit-identical to eager `Linear::forward` (the module doc's step 1b) — before the sum widens to f32 and rounds ONCE on the way out; W is frozen, and candle's `Op::Matmul` backward would compute a `dW` GEMM for it regardless — the fused op returns `None`. jammi picks bit-exact parity with candle-eager over torch/PEFT's own single-rounding cuBLASLt epilogue fold (unreachable from candle, which never exposes an f32 accumulator out of `matmul`) — the op's module doc has the full three-variant rounding enumeration and its citations. The cost of admitting a bias: one small `to_dtype` cast plus one `broadcast_add` launch per site per forward (paid even at F32 — `to_dtype` never no-ops a same-dtype pair), and one tiny `zeros_like` of the padded bias block per site per backward (so `Tensor::cat`'s backward has a same-shaped slot to narrow, even though the frozen bias contributes no gradient). Node counts (`crates/jammi-kernels/src/ops/low_rank_residual_linear.rs`'s oracles): bias-free eager 10 / fused 6 (`fused_site_retains_fewer_tape_nodes_than_the_eager_composition`); bias-carrying eager 11 / fused 6 (`fused_site_with_bias_retains_fewer_tape_nodes_than_the_eager_biased_composition`) — the eager arm's bias add is itself a tracked node, while the fused arm's cost is unchanged because the pack just grows a `Tensor::cat` argument. The cast-boundary fusions live in its backward: `f32(x)·scale + 0.0f` (the `+0.0f` is required for signed-zero identity with candle's `affine_f32`) and round-then-native-bf16-add, both proven bit-identical to the chains they replace.

### The BERT-family sites, measured

Each BERT-family site above has a fused-vs-eager A/B on an A100 80GB PCIe, produced by `ci/scripts/perf/lora_bias_ab.sh` at the artifact's own `git_sha`. Common method: n=3 repeats per cell; per-step wall `(wall_600 − wall_100)/500`; both arms under `JAMMI_KERNELS_STRICT=1`; every leg counter-proved; the eager arm forces only the site under test off via `JAMMI_KERNELS_DISABLE`, with every other BERT-family fused key left ON on both arms (the cross-op invariant that isolates one site). Shapes are `b8W512f32` (the wire shape, where the fused-vs-fused control floor is measured) and `b32W64bf16`. "control s/step" is the fused-vs-fused control arm the "floor" column derives from — `|fused − control| / fused`; `b32W64bf16`'s absolute times are not a committed top-level field of the artifacts (only `gain_by_shape` is), so those cells print `—`. Every site ACTIVATEs for both models at both shapes. The LayerNorm, attention and erf-GELU sites had no pre-registered activation bar — they land by architectural direction (one common path per site covering every family), so ACTIVATE there is the measured classification only (each artifact's `notes.landing_rule`); the LoRA site's verdict is judged against its pre-registered bar.

| site (forced-eager keys) | model | shape | fused s/step | control s/step | gain vs eager | floor |
|---|---|---|---:|---:|---:|---:|
| LoRA (`lora_linear_fused`) | BERT | b8W512f32 | 0.763558 | 0.766233 | 18.18% | 0.35% |
| | BERT | b32W64bf16 | — | — | 30.77% | — |
| | DistilBERT | b8W512f32 | 0.388349 | 0.387883 | 17.23% | 0.12% |
| | DistilBERT | b32W64bf16 | — | — | 28.91% | — |
| LayerNorm (`layer_norm_fused`) | BERT | b8W512f32 | 0.686294 | 0.685976 | 8.87% | 0.05% |
| | BERT | b32W64bf16 | — | — | 25.11% | — |
| | DistilBERT | b8W512f32 | 0.346907 | 0.347852 | 9.42% | 0.27% |
| | DistilBERT | b32W64bf16 | — | — | 25.24% | — |
| attention (`attention_block_fused,softmax_last_dim_fused`) | BERT | b8W512f32 | 0.413199 | 0.413605 | 40.67% | 0.10% |
| | BERT | b32W64bf16 | — | — | 15.30% | — |
| | DistilBERT | b8W512f32 | 0.211661 | 0.211983 | 40.16% | 0.15% |
| | DistilBERT | b32W64bf16 | — | — | 17.30% | — |
| erf-GELU (`gelu_erf_fused`) | BERT | b8W512f32 | 0.414392 | 0.413523 | 4.68% | 0.21% |
| | BERT | b32W64bf16 | — | — | 6.55% | — |
| | DistilBERT | b8W512f32 | 0.212233 | 0.211716 | 4.86% | 0.24% |
| | DistilBERT | b32W64bf16 | — | — | 7.56% | — |

Artifacts, in table order: `crates/jammi-kernels/artifacts/cuda-runs/2026-09-05-lora-bias-428-c69dbd7-a100-pcie.json`, `2026-09-05-ln-bias-460-6c6d79c-a100-pcie.json` (`AB_OP=ln`), `2026-09-06-attn-block-462-869c65f-a100-pcie.json` (`AB_OP=attn`), `2026-09-06-gelu-erf-463-869c65f-a100-pcie.json` (`AB_OP=gelu`), all in the same directory. The full-precision eager absolutes: LoRA — BERT 0.933241 s/step eager at the wire shape, 0.237753 eager vs 0.164589 fused at `b32W64bf16`; DistilBERT 0.469170 eager at the wire shape, 0.122467 vs 0.087061 at `b32W64bf16`. LayerNorm — BERT 0.753107 eager at the wire shape, 0.161657 vs 0.121071 at `b32W64bf16`; DistilBERT 0.382965 eager at the wire shape, 0.085596 vs 0.063990 at `b32W64bf16`.

Notes per site. The LoRA legs load the stock, unconverted `bert-base-uncased` checkpoint (26 `gamma` / 26 `beta` tensors) through the legacy-name loader, never the gamma/beta-renamed form. The attention A/B must force `softmax_last_dim_fused` eager alongside `attention_block_fused` — the block subsumes the softmax step, so leaving the softmax key on would keep it dispatching fused on the "eager" arm and understate the eager cost — and `attention_block_flash` declines on both arms by the BERT family's counted `CapabilityMiss`. The attention site is the largest single gain, as the BERT-family ablation census (`crates/jammi-kernels/artifacts/cuda-runs/2026-08-31-profile-356-closeout-7820d697-a100-sxm4.json`) projected. That census artifact keeps its own `UNRESOLVED` strings verbatim — a committed artifact is never edited after the fact; the per-site A/B artifacts above are where each branch is resolved.

**AdamW** (in-place via `InplaceOp2/3`). A `CustomOp` returns new storage; splicing it into a `Var` costs a D2D memcpy per Var per step (672 of the step's 1,345). `InplaceOpN` mutates the Var's storage in place: three launches per Var, zero copies. Bit-identity to candle's eager chain requires pinning every op with `__fmul_rn`/`__fadd_rn` and reproducing candle's `x*mul + 0.0f` signed-zero laundering; a red-control kernel using `fmaf` proves the harness sees that class. Its gain is launch overhead, not bandwidth: AdamW's data volume is ~0.5 GB, which is why an in-place design beats a true multi-tensor kernel that would need a candle patch.

### FlashAttention-2, the one tensor-core arm

Vendored Dao-AILab v2.8.3.post1 with no edits, CUTLASS as a pinned submodule; only `run_mha_{fwd,bwd}<bf16, 64, false>` compile — one native cubin per arch in the compiled set (the encoder refuses any device outside that enumerated set — see `third_party/flash-attention/VENDORED.md`'s "Supported archs" for the current membership and per-arch validation status). Three shim headers replace the PyTorch includes; a torch-free C ABI turns every `TORCH_CHECK` into a status code with a `struct_size` guard. Hand-rolled nvcc with upstream's flags including `--use_fast_math` — the one exempt translation unit, because every upstream wheel and parity oracle is calibrated to it.

Sequences are packed into `qkv [total_q, 3, H, 64]` with a `cu_seqlens [B+1]` prefix sum; the kernels index from host totals and do not cross-check against the device array, so `CuSeqlens` is a validated type whose geometry is derived from the same host lengths, with an `unsafe` escape hatch. The forward has no atomics; the backward is deterministic iff `deterministic` (private zeroed `dq_accum` splits reduced in fixed order) — pinned on. Determinism measured free at b8: 0.3733 vs 0.3759 ms. The kernel costs ~39 ms/step.

**One op over (qkv, cos, sin).** Composing a RoPE op with the flash op regresses VRAM by +1.8–2.5 GB: candle's `BackpropOp::new3` clones every tracked argument into the result's node, so the two-op form retains a *second* rotated buffer per layer. `FlashVarlenAttentionFusedRope` rotates at the storage level inside one `cuda_fwd` and recomputes the rotation in `bwd`: bit-identical values, flash peak 18.91 → 16.43 GB, 0.2 GB below the block arm.

**Windows and the decline lattice.** ModernBERT alternates global and local layers (`local_attention = 128`); the flash arm passes radius 64 as `window_size = (64, 64)`, cross-checked against FA's `mask.h`, HF's `sliding_window − 1` adapter and HF's eager mask, with `w±1` as red controls. `op_disabled` is consulted FIRST, before any predicate work — an operator-disabled run short-circuits straight to the block arm, never paying for the gates below. Then the cheap capability gates: feature compiled? CUDA? arch in the compiled set? bf16? head_dim 64? Then the domain fences on the resolved per-row lengths: `trusted_lengths_match_mask` (caller-supplied lengths are validated against the mask on-device, never trusted bare), `mask_is_prefix_every_row`, `every_row_length_ge_1`. A miss anywhere declines to the block arm (counted), never errors under `Strict`. Density is not a decline term — `build_flash_forward_decision` fuses BOTH a dense and a genuinely-padded batch once every gate clears; `is_dense` only picks which arm dispatches (no transport for dense, reason `domain_ok_dense`; the encoder-boundary unpad/repad transport for padded, reason `domain_ok_padded`) — the 1.07× is measured on the dense branch only; the padded regime has no committed measurement.

### The cross-modal towers: what is reachable, and what the profile found

CLIP-text, OpenCLIP-vision and HTSAT-CLAP audio are LoRA-trainable on the same
`MaybeLoraLinear` seam the BERT family uses (`docs/maintainer/MAINTAINER-GUIDE.md` §2.5),
so several of this section's fused primitives are reachable on a tower training step **with
no new kernel work at all**: the fused LoRA site (`lora_linear_fused`, every wrapped linear
on every tower), the fused biased LayerNorm (`layer_norm_fused` — all three towers
carry a bias on their LayerNorms), and, on HTSAT only, the fused
dense erf-GELU (`gelu_erf_fused`): that tower's Swin-block MLP and its projection head's
`"gelu"` arm both call the house seam `activations::gelu_erf(x, training)`, so the same
op §3's BERT-family paragraph measures admits there on tensor state, with eval bytes
unchanged (the seam's `training == false` arm is the unchanged `Tensor::gelu_erf` call).
Two primitives remain out of reach: CLIP's attention would need the house `MASKED_LOGIT`
sentinel to reach `attention_block_fused` (a declared eval-bit change, so it is a port with
its own oracle, not a wiring change), HTSAT's `head_dim` of 24 declines the block arm's
domain outright, and CLIP's `quick_gelu` activation has no fused seam at all — so the CLIP
towers have no `gelu_erf_fused` admit site, and naming that key in a forced-eager CLIP leg
would be an unmatched disable the bench refuses.

**The training-step profile.** The workload, method and two-sided ACTIVATE/DECLINE/UNRESOLVED
thresholds were fixed (`docs/plans/66-tower-profile/`) before any leg ran; 11 of the 12
legs (`A1`/`A2`/`D1`/`D2` × the three towers) pass the profile's validity gate — `htsat-A2`
fails its UNATTRIBUTED bound and is excluded from every finding, see the profile README's
deviations — and the BF16 pre-flight (P2) passes on every tower, measured on
`crates/jammi-kernels/artifacts/cuda-runs/2026-09-07-profile-421-towers-c1b0b0ba-a100-sxm4.json`
(A100-SXM4-80GB):

| tower | leg | dtype | wall s/step | front s | busy s | residual s | front % | busy % |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CLIP-text | A1 | f32 | 0.1004 | 0.0000 | 0.0608 | 0.0396 | 0.0 | 60.5 |
| CLIP-text | A2 (bf16) | bf16 | 0.0965 | 0.0000 | 0.0414 | 0.0551 | 0.0 | 42.9 |
| CLIP-text | D1 (LoRA+LN eager) | f32 | 0.1486 | 0.0000 | 0.0895 | 0.0591 | 0.0 | 60.2 |
| CLIP-text | D2 (LoRA eager) | f32 | 0.1329 | 0.0000 | 0.0794 | 0.0535 | 0.0 | 59.7 |
| CLIP-vision | A1 | f32 | 0.1147 | 0.0235 | 0.0652 | 0.0260 | 20.5 | 56.9 |
| CLIP-vision | A2 (bf16) | bf16 | 0.1086 | 0.0241 | 0.0386 | 0.0459 | 22.2 | 35.6 |
| CLIP-vision | D1 (LoRA+LN eager) | f32 | 0.1658 | 0.0243 | 0.0976 | 0.0439 | 14.7 | 58.8 |
| CLIP-vision | D2 (LoRA eager) | f32 | 0.1450 | 0.0243 | 0.0873 | 0.0334 | 16.8 | 60.2 |
| HTSAT | A1 | f32 | 1.5500 | 1.2509 | 0.2348 | 0.0643 | 80.7 | 15.1 |
| HTSAT | A2 (bf16) † | bf16 | 1.5005 | 1.2516 | 0.1883 | 0.0606 | 83.4 | 12.6 |
| HTSAT | D1 (LoRA+LN+GELU eager) | f32 | 1.6895 | 1.2493 | 0.3488 | 0.0914 | 73.9 | 20.6 |
| HTSAT | D2 (LoRA eager) | f32 | 1.6329 | 1.2550 | 0.2953 | 0.0826 | 76.9 | 18.1 |

`front` is a direct measurement (`media_front_end_wall_s`), never `wall − busy`; a text leg
carries `front = 0` by the boundary declared above (tokenization stays in the residual).
† `htsat-A2` fails the profile's validity gate (its UNATTRIBUTED share of GPU busy is over
the bound) and is excluded from every finding — see the profile README's Deviations
section.

The already-fused chains' realized gains (eager twin minus fused, per step): C-LORA +32.6
ms wall on CLIP-text (32.4 % of the A1 baseline wall), +30.3 ms on CLIP-vision (26.4 %),
+82.9 ms on HTSAT (5.4 %); C-LN +15.6 ms on CLIP-text, +20.8 ms on CLIP-vision; on HTSAT,
D1 disables `layer_norm_fused` AND `gelu_erf_fused` together, so its D1-minus-D2 delta
(+56.6 ms, 3.6 % of the A1 baseline wall) is the JOINT C-LN + C-GELU-HTSAT gain, not C-LN
alone.

**All four candidate ports are UNRESOLVED — the profile licenses no port.** No candidate
clears ACTIVATE (`s_wall>=10%` on any decision-grade leg) or DECLINE (both
`s_wall+U_wall<5%` AND `s_busy+U_busy<5%` on every decision-grade leg). With
`kernel_census.py` keying kernels on demangled names (`docs/maintainer/MAINTAINER-GUIDE.md`
§2.5), both CLIP-tower A2 legs are decision-grade for attribution (`htsat-A2` fails the
validity gate and is excluded from every finding — HTSAT has no candidate port in this
profile, so that never blocks a verdict; see `docs/plans/66-tower-profile/README.md`), so
no verdict below is F32-only. This is not uniform across candidates or axes: `C-MLP`'s own
measured `s_wall` (no `U` term) is only 3.95 %/3.11 % on CLIP-text and 3.36 %/2.66 % on
CLIP-vision — well under the 5 % DECLINE floor on wall, combined or not — it is the
combined *busy* share (`s_busy+U_busy`, 7.24 %/8.29 % CLIP-text, 6.80 %/8.98 % CLIP-vision)
that lands in the 5–10 % band and keeps DECLINE from firing. Verbatim reasons
(artifact `candidate_decisions[]`): `C-ATTN-CLIP-text` — UNRESOLVED, "neither ACTIVATE
(s_wall>=10% on any decision-grade leg) nor DECLINE (combined share <5% on every
decision-grade leg) — clip-text-A1: s_wall+U_wall=0.1030, s_busy+U_busy=0.1703;
clip-text-A2: s_wall+U_wall=0.0878, s_busy+U_busy=0.2045"; `C-MLP-CLIP-text` — UNRESOLVED,
"…clip-text-A1: s_wall+U_wall=0.0438, s_busy+U_busy=0.0724; clip-text-A2:
s_wall+U_wall=0.0356, s_busy+U_busy=0.0829" (elided prefix identical to
`C-ATTN-CLIP-text`'s above; full text at artifact `candidate_decisions[1].reason`);
`C-ATTN-CLIP-vision` — UNRESOLVED, "…clip-vision-A1: s_wall+U_wall=0.0641,
s_busy+U_busy=0.1128; clip-vision-A2: s_wall+U_wall=0.0501, s_busy+U_busy=0.1410"
(`candidate_decisions[2].reason`); `C-MLP-CLIP-vision` — UNRESOLVED, "…clip-vision-A1:
s_wall+U_wall=0.0387, s_busy+U_busy=0.0680; clip-vision-A2: s_wall+U_wall=0.0319,
s_busy+U_busy=0.0898" (`candidate_decisions[3].reason`). The two-sided rule refuses to
manufacture a verdict a 5–10 % share does not support on either side — that refusal, not a
missing signal, is why nothing ports.

**Findings.** The HTSAT training step is CPU front-end-bound: front-end share of wall is 81
% on the F32 decision leg (`htsat-A1`; `htsat-A2` excluded: fails the validity
gate) (audio decode/resample/STFT/mel dominating wall time), arm-invariant — the parallel
media front end below addresses it. CLIP-vision's own image decode/preprocess front end is
20–22 % of wall on the F32/BF16 decision legs. Both media corpus producers cycle only 16
distinct train clips (families × instances = 24 files at any `--rows`) — a page-cached
working set, not a realistic-corpus I/O cost — so both front-end numbers are a real
per-item CPU decode/preprocess compute cost, never disk I/O (artifact
`notes.recorded_deviations`; full caveat: `docs/plans/66-tower-profile/README.md`). At
batch 8 the CLIP training steps are launch-bound (3638–3722 launches/step across the four
F32/BF16 A-arm CLIP legs); BF16 cuts GPU busy 32–41 % per tower while wall drops only 4–5
%. `C-ATTN-HTSAT` is measured, not a candidate port: 33 % of GPU busy (~5 % of wall) on the
F32 decision leg — HTSAT's head_dim of 24 at every stage sits outside the fixed-head-dim
port tier, so this stays a measured, open number, never folded into UNATTRIBUTED. Full
per-leg table and deviations: `docs/plans/66-tower-profile/README.md`.

**The producers.** `ci/scripts/perf/gen_fixed_shape_image_corpus.py` and
`ci/scripts/perf/gen_fixed_length_audio_corpus.py` emit the media corpora (fixed shape and
fixed duration, so the timed region carries the tower's real front-end cost and nothing
that varies row to row); `ci/scripts/perf/gen_fixed_width_corpus.py` emits the CLIP-text
one. All three also emit a HELD-OUT split on `--heldout-rows`, writing a
`heldout_ids.txt` plus its own `heldout_triplets.jsonl` beside the train corpus, because
`finetune-run` REQUIRES `--heldout-ids` + `--heldout-jsonl` on every leg — a corpus without
one is not runnable, so the producer that owns the corpus owns the split too. The media
producers additionally take `--heldout-batch` (required alongside `--heldout-rows`, refused
unless it divides the held-out row count) and `--heldout-families`, which RESERVES that
many of the family pool for the split, making the held-out rows id-, file- and
family-disjoint from train; the text producer's split is id- and text-disjoint and carries
the same width guarantee. On every producer the held-out rows carry the train schema and
the train corpus's own pinned shape/duration/width, and requesting them leaves the train
split byte-identical to a no-held-out run at the same seed — so a driver can generate
train-only and train+held-out corpora at the same seed and difference them. Each producer's
suite is a guard in `ci/guards.toml`.

**The run flags the profile pins.** `jammi-bench finetune-run --task
{text_embedding,image_embedding,audio_embedding}` selects the tower; `--objective triplet`
pins the joined-forward row count (a media task refuses MNRL outright); `--lora-init
{zeros_b,gaussian}` selects the adapter initialization (`zeros_b` is the default;
`gaussian` exists because a `zeros_b` adapter has a zero `B`, hence a zero gradient into `A`
at step 0, which makes it useless as a dtype pre-flight); and `--expect-kernels-disabled
<keys>` turns a forced-eager leg's premise into a checked one. That last flag is what makes
an eager twin a datum rather than an assumption: the run refuses at START unless this
flag's value EQUALS this process's real `JAMMI_KERNELS_DISABLE` exactly (equality, not
subset: a subset check would let an ambient or leftover env var through undetected, and
`--arm alloff` already forces an exact-set match of its own two keys, so no legitimate leg
names a chain key on top of alloff's pair), and refuses at the END unless no requested
disable went unmatched AND every named key's fused dispatch counter read zero across the
run. `--arm fused` with no `--expect-kernels-disabled` at all makes NO claim about
`JAMMI_KERNELS_DISABLE` — an operator may legitimately run a fused leg with OTHER,
unrelated op keys disabled, so this binary does not, and must not, refuse an unlabeled
fused leg on that basis; the two-sided witness that a profile decision leg's fused side was
genuinely unlabeled lives outside this binary, in the driver and merger. A leg that
fails the alloff arm-level check or the `--expect-kernels-disabled` START/END checks exits
non-zero and writes no row — INVALID, never a datum. The claim itself is recorded on the
report as provenance (`kernels_disabled_expected`), distinct from the process-OBSERVED
`kernels_disabled_requested`/`kernels_disabled_fired` pair it was checked against.

**The front end is measured, never differenced.** `finetune-run` reports
`media_front_end_wall_s` on every media leg, read from
`TrainingResult::media_front_end_wall` and summed across the resume-cycled epoch legs of a
run. Its boundary is DECLARED, not implied, and it has exactly ONE such boundary: on an
`EncoderAdapters` target, the wall around the decode + preprocess call — including the
device tensor build those preprocess functions perform internally, which cannot be
separated without splitting them — with the tower forward EXCLUDED, accumulated only while
the loop is in training mode (held-out eval passes are excluded, so the number is a strict
subset of the training wall), and `null` rather than `0.0` on a text leg because
tokenization is not a media front end and stays in the residual. On a `ProjectionHead`
target, of any modality, the field never accumulates at all — the frozen base's decode,
preprocess and forward are one inseparable call there with no seam to time in isolation, so
charging that combined call to a field documented as front-end-only would give it a second,
incompatible meaning (a residual computed against it would double-subtract the tower
forward); reporting `0` is the honest reading, per that field's own doc. It is a MEASURED
field — neither identity nor provenance — for the same
reason the dispatch counters are, and it is deliberately NOT `wall − gpu_busy`: a
difference would absorb launch latency, sync stalls and the optimizer's own CPU time into a
number labelled "front end". Having it directly is what lets a profile report `launch/sync
residual = wall − front − busy` instead of folding all three together.

The three towers compose `attention_softmax` directly, with no admission counter behind
it, so **all-zero attention counters are the healthy reading for a tower leg** — the
BERT-family all-zero validity control would refuse every valid media leg. A tower family
is judged on the LoRA-linear dispatch counters instead: zero of those over a run that took
optimizer steps proves the adapted encoder was never forwarded.

**The witnessed `calls` term: `FusibleSiteCensus`.** The profile's positive-proof equation
is `fused + eager == calls × batches` per admission key. `fused`/`eager` are measured (the
process-wide `jammi_kernels::admission` counters) and `batches` is measured (training
forwards taken); `calls` — how many times ONE forward reaches a given seam — is witnessed by
`AnyEncoder::fusible_site_census() -> FusibleSiteCensus`
(`crates/jammi-encoders/src/fusible_census.rs`): `lora_sites_wrapped` (sites the built tower's
ONE training forward actually takes a `lora_linear_fused` admission decision for —
`jammi_lora::MaybeLoraLinear::takes_lora_linear_admission` over the tower's own site
traversal, a NARROWER predicate than `is_lora`: a `Lora` site over a `FrozenBase::Quantized`
base (a QLoRA backbone) is adapted but composes before ever reaching `admit()`, so it is not
counted here — paired with `lora_linear_fused`), `layer_norms` (house `LayerNorm` instances the
built tower actually holds on its forward path, paired with `layer_norm_fused`), and
`gelu_seam_calls_per_forward` (calls to `activations::gelu_erf` per forward, paired with
`gelu_erf_fused`). Every count is WALKED off the built structure — the `Vec<Layer>` the
loader actually built, the `MaybeLoraLinear` arm each site landed on, the `Option<LayerNorm>`
a layer actually holds — never derived by config arithmetic (`2 × layers`, and similar): a
structural walk disagrees with a formula the moment the built tower stops matching its
config (a site the selector declined, a pre-norm a family omits on layer 0, an HTSAT stage
with no `downsample`), which is the only case a `calls` witness earns anything over a
comment doing the same arithmetic by hand. The census is per-forward at `training == true`
only: every seam short-circuits before any admission decision in eval, so an eval forward
contributes `0` to BOTH sides of the equation — never "all eager" — and `0` is itself a
real, falsifiable answer for a family with no such seam at all (ModernBERT's GeGLU FFN and
both OpenCLIP towers' `quick_gelu` hold `gelu_seam_calls_per_forward == 0`, for two
different architectural reasons named on the field's own doc). These are STRUCTURAL counts
pinned per tower on the committed tiny fixtures by each encoder's own
`fusible_site_census_is_the_exact_per_forward_seam_call_count` test — not a measurement of
any run — and read (`lora_sites_wrapped`/`layer_norms`/`gelu_seam_calls_per_forward`): tiny
BERT 6/3/1, DistilBERT 12/5/2, ModernBERT 16/9/0, CLIP-text 4/3/0, OpenCLIP-vision 4/4/0,
HTSAT 53/21/8-or-9 (the audio tower's GELU term is 8 on the fixture's `"relu"` projection
head and 9 on a `"gelu"` twin over the identical geometry — the real `laion/clap-htsat-fused`
checkpoint's `projection_hidden_act` is read from its own config at run time and recorded by
the census, never hardcoded). `FinetuneRunTier::fusible_site_census`
(`crates/jammi-bench/src/report.rs`) records the census as PROVENANCE — a structural
property of the build the bench captures every epoch and refuses if it changes mid-run —
never IDENTITY (it is nothing a caller declared, so two legs cannot disagree about it the
way they can about `lora_rank`) — and it is where a profile's positive-proof
equation takes its `calls` term from, rather than re-deriving it.

**The `--epochs 1 --grad-accum 1` convention.** `steps_measured` counts TRAINING forwards
only under that pinning: at `--epochs > 1`, `finetune-run`'s resume-chained legs double-count
`global_step` (measured directly — a `--epochs 2` run reports 6 for 4 actual forwards).
Every profiled leg pins `--epochs 1 --grad-accum 1` for exactly this reason:
`batches == steps_measured` in the positive-proof equation is only true under that pinning.

### The media front end: parallelized across rayon's global pool

The per-item decode/preprocess work a media leg does before the tower forwards is spread
across rayon's GLOBAL pool for HTSAT and CLIP-vision, without touching a decoder body.
candle installs no private pool of its own, so this is the one pool the process ever
schedules media-batch work on.

**Measured: the HTSAT/CLIP-vision front-end A/B.** The run
(`crates/jammi-kernels/artifacts/cuda-runs/2026-09-08-frontend-0a8562c4-a100-pcie.json`,
A100 80GB PCIe, tip `0a8562c4` vs the sequential base `c1b0b0ba`, at the tip binary's own
resolved rayon global-pool width and the profile's fixed per-step item count, over several
interleaved base/tip repeats) measured HTSAT's front-end and full-step wall per-step
means, base against tip, and CLIP-vision's report-only front-end per-step means, base
against tip — every cell in the table below is bound to the committed artifact's own
field. The HTSAT bar's ratio, its observed interval, and the two-sided machine-model
bound it is judged against are bound the same way below, as are the driver-default and
the run's own measured serial-tail ratio: a bound falls strictly inside the interval, so
the bar is UNRESOLVED, invariant under both ratios — not ACTIVATE. The parallel front end
is kept because bit identity holds (pool sizes 1/5/7/24 against the sequential reference,
`crates/jammi-ai/tests/it/media_front_end.rs`) and the n=1 image serving path stays within
its always-on gross latency bar (3x before_min + before_spread, same suite); the 5 % n=1
bar is opt-in (`JAMMI_FRONTEND_N1_LATENCY=1`) and no serving-latency measurement is
recorded, so no serving-regression claim tighter than that bar is made, and NO
parallel-efficiency claim is made. §9's first checklist applies: every number in a doc
names its producer, or it is not written.

| HTSAT/CLIP-vision front-end quantity | value (s or ratio) |
|---|---:|
| HTSAT front-end s/step, base mean | 1.335 |
| HTSAT front-end s/step, tip mean | 0.108 |
| HTSAT step-wall s/step, base mean | 1.567 |
| HTSAT step-wall s/step, tip mean | 0.344 |
| CLIP-vision front-end s/step, base mean | 0.0293 |
| CLIP-vision front-end s/step, tip mean | 0.0078 |
| HTSAT bar ratio | 0.0809 |
| HTSAT bar ratio_lo | 0.0772 |
| HTSAT bar ratio_hi | 0.0871 |
| HTSAT bar lower bound | 0.0448 |
| HTSAT bar upper bound | 0.0864 |
| serial-tail ratio, driver-default | 0.0033 |
| serial-tail ratio, measured | 0.00355 |

**The mechanism.** Two parallel stages, the same shape on both towers:

1. **Decode.** A shared per-item decode body per modality
   (`image_preprocess`'s and `audio_preprocess`'s private `decode_*_results`)
   backs two DIFFERENT public error contracts, not one. The trainer's
   `image_encoder_input`/`audio_encoder_input` call `decode_image_batch`/
   `decode_audio_batch`, which hard-fail the whole job on the
   LOWEST-INDEX decode error — a corrupt training item is a refusal, not a
   row to skip. Serving's `arrow_to_images`/`arrow_to_audio` instead call the
   `_per_row_indexed` variants (`decode_image_batch_per_row_indexed`,
   `decode_audio_batch_per_row_indexed`), which return EVERY row's own
   outcome: a corrupt row's bytes produce that Arrow row's own `Err` (surfaced
   as that row's `_status=false` with an Arrow-row-indexed `_error` message in
   `BackendOutput`, per `docs/guide/src/generate-image-embeddings.md`'s
   error-handling table), the rest of the batch still embeds, and a null row
   keeps its own `None`/"Null or missing …" treatment. Both variants decode in
   parallel across the batch on rayon's global pool; a path-valued Arrow
   column reads its bytes SEQUENTIALLY first (`std::fs::read` never runs
   inside the pool).
2. **Preprocess.** `preprocess_image_batch`/`preprocess_clap_fusion` preallocate the
   batch's output buffer once and have each item write its own disjoint, fixed-stride
   chunk via `par_chunks_mut` — filters and the STFT window are hoisted out of the
   per-item closure, and a release-mode (not `debug_assert!`) per-item length check
   guards every chunk write. Preprocess has no per-row variant: the trainer and
   serving both call the lowest-index-hard-fail `_indexed` form (serving has already
   dropped every decode-failed or null row before preprocessing runs, so a preprocess
   failure there is never a corrupt-item skip either).

There is no thread-count knob anywhere in this path: chunk count is always the batch
size, so the effective parallelism is `min(pool_size, batch_size)`, emergent from
whichever pool the process happens to run under — never configured. "The LOWEST-INDEX
failing row is the one surfaced" holds for the training path's decode AND preprocess
stages, and for the preprocess stage on the serving path — the same order a sequential
loop fails in — but NOT for the serving path's decode stage, which surfaces every row's
own outcome instead of collapsing to one. An empty batch is refused before any chunking is
attempted.

**Provenance: `rayon_pool_threads`.** The train-run leg records `rayon_pool_threads` as a
provenance field — `rayon::current_num_threads()` at report time, i.e. the rayon GLOBAL
POOL SIZE the run's process resolved to, not how many of those threads actually touched a
given batch's chunks. It is machine/build provenance, the same class
`device_name`/`host.logical_cpus` occupy — a fact about the box and the process, never a
determinant of what a step computes, so it is never an identity field.

### The bench and its torch twin

`jammi-bench finetune-step`: three encoder forwards, a triplet hinge, one backward, one AdamW step; synthetic uniform token ids, so it measures *cost*, never learning. `torch_finetune_step.py` is matched argument for argument (`attn_implementation` read back from the config; `--attn eager` = semantic twin, `--attn sdpa` = the throughput bar; LoRA init distribution-matched; TF32 off). `ab_merge.py` refuses to compare legs whose `FINETUNE_IDENTITY_FIELDS` differ (the tuple is declared once, in `ci/scripts/perf/identity_fields.py`, and imported — 18 entries, including the padded-batch `row_lengths` vector); the raw attention string (`attn_requested`/`attn_implementation`) is recorded as provenance and never compared, while the reference *class* it implies is compared via the `attention_arm` identity field; the clip determinant `max_grad_norm` is in the comparison tuple (null = clip off is a value, never MISSING) and in the identity-completeness const (`FinetuneStepTier::IDENTITY_FIELDS`, a strict superset). A stdlib-`unittest` suite (`ci/scripts/perf/test_identity_fields_subset.py`) pins the claim mechanically: every Python comparison-tuple entry must be named in the corresponding Rust identity-completeness const, and the tuple cardinalities (18, 11) are asserted as numbers, not promises.

`jammi-bench finetune-run` is the multi-step tier over the same machinery. Every producer fills one leg type (`crates/jammi-bench/src/leg.rs`): a workload payload whose identity is declared once, in `Payload::IDENTITY_FIELDS`, beside provenance that is recorded and never compared, the measurements every axis of the ladder reads, and the facts its premises check. The train-run identity is declared in `TrainRunPayload::IDENTITY_FIELDS`, and read from there by the comparator: a `finetune-run` leg is a leg of the `train-run` parity ladder (`jammi-bench ladder train-run`, `docs/plans/69-parity-ladder/README.md`), whose one operator refuses legs that disagree on any identity field — between the two legs of a seed and across every leg of the comparison. Every field lands in exactly one of three classes, and the class is the argument, not the field's type. **IDENTITY** is a premise two legs must AGREE on to be comparable at all: `task` (which TOWER of a multi-tower checkpoint was trained — the strongest identity field on this tier after the checkpoint digests, because one `checkpoint_weights_sha256` covers both OpenCLIP towers), `lora_init` (a `gaussian` leg starts from a different point on the loss surface than a `zeros_b` leg at the identical seed and selectors), and `train_media_sha256`/`heldout_media_sha256` (the media corpus's own CONTENT digests, a digest-of-digests in manifest and scoring order — the manifest digests beside them name PATHS, so swapping the bytes behind those paths moves every measured loss while leaving every other identity field byte-identical; `null` on a text leg, where the manifest IS the content, is a stated value and not a missing one). **PROVENANCE** is caller-declared or build-observed, recorded and never compared: `kernels_disabled_expected`, the `--expect-kernels-disabled` claim, in exactly `arm`'s sense. **MEASURED** is an outcome of running, in neither tuple: `media_front_end_wall_s`, the same class every dispatch counter and `train_run_wall_s` carry — naming it in a comparison tuple would make two legs at different front-end costs incomparable, which is the opposite of what it is for.

`jammi-bench encode-step` is the serving-side peer, and the producer of the `encode` ladder's engine rungs (`docs/plans/69-parity-ladder/README.md`): rows in a table → one artifact per key, through `direct` (the loaded model called on the rows, no plan), `plan` (the real `generate_text_embeddings`/`infer` plan at one partition) and `plan-partitioned` (at N) — a task (`--task embed|infer`), a size (a `--rows` sweep of a seeded VARIABLE-length corpus, so padding costs what it costs), a device and a checkpoint are parameters of the one workload, never tiers of their own. Every (unit, take) is measured in a child process (`VmHWM` never falls; the device sampler wraps a process from outside), with more than one rung served INTERLEAVED in it (forward then reversed, so a drifting box lands on each alike — the legs an edge's speed is read from; a rung's space is read from a session of its own). A leg carries its per-iteration time series, never only a summary; a `plan` leg that commits a table carries where each serve's time went inside the result-table sink (`sink_phases`: the time to the last output batch, the Parquet write, the ANN insert, the segment persist — the sink's own account, off its `SINK_PHASES_TARGET` event); every leg carries the artifact's digest in key order, and a unit's first take its vectors beside it. The producer decides nothing: `rung` and `partitions` are provenance, the exact edges' equality (`direct`, `plan`, `plan-partitioned` persist byte-identical artifacts, for both tasks) is a hermetic test here and the comparator's outcome axis there, and a rung that is not deterministic across its serves or lost a row is an error, never a leg. `torch_encode.py` is the `torch` rung by the same contract: the SAME corpus file, the same `tokenizers` library and truncation bound, pooling resolved by `pooling_from_config`'s rules, the same dtype and batch size, embeddings persisted inside the span — and, with `--ann-index`, the same `usearch` graph the engine's sink builds beside every table it commits, without which the reference has done less work than the leg beside it. `--order corpus` with `--attn eager` is the semantic twin (the plan's own chunks, hence its padding); `--order length-sorted` with `--attn sdpa` is what `sentence-transformers`' `encode()` does, the bar a user holds the engine to. `encode_ab.sh` runs the legs — interleaved engine rungs, both torch orders, each engine rung alone — as a palindrome, and hands each legs directory to `jammi-bench ladder encode`.

---

## 4. Numerics at the bf16 boundary

bf16 keeps f32's 8-bit exponent and 7 explicit mantissa bits. ULP at 1.0 ≈ 0.0078; at 100 it is 1.0; at ~6,700 (the layer-18 residual magnitude) 32; at −10,000, 64. There are 2¹⁶ f32 ULPs per bf16 ULP. Every jammi kernel reads bf16, accumulates in f32, and rounds at the store; cuBLAS bf16 GEMMs accumulate in f32 (`CUBLAS_COMPUTE_32F`).

### Rounding placement: three cases where the obvious fix is wrong

- **The eager arm can be the outlier.** An eager LayerNorm that rounds x̂ to bf16 before multiplying by γ, or an eager RoPE that rounds three times, is not the reference — changing the *fused* kernel to match it would be wrong. ATen's `layer_norm_kernel.cu` computes bf16 LN in float and casts once; HF's `apply_rotary_pos_emb` does `(q.float()*cos) + (rotate_half(q.float())*sin)` then one `.to(dtype)`. jammi's eager arms follow the references (the two placements differ on 67,546 of 262,144 elements).
- **Round-then-add vs add-then-round.** PEFT computes `result(bf16) + delta(f32)` in f32 and casts the sum once; rounding the delta first differs on 176/4096 elements, max one ULP at |base| ≈ 100. A deviation both jammi arms share is invisible to a same-build A/B — why "agreement is not accuracy" is a rule. The LoRA epilogue rounds once at the wider dtype, matching PEFT, with FMA pinned in all four `scaled_cast_add` kernels.
- **The softmax backward, validated at the right call path.** ATen's bf16 `SoftMax.cu` rounds `dy·y` to bf16, but HF ModernBERT never runs that path: `modeling_modernbert.py` computes `softmax(..., dtype=torch.float32)`, so copying ATen's rounding would move jammi *away* from the reference. Validate against the model's **call path**, quoting the line and its dtypes.

### Reduction order: a GEMM's operand form is part of its numerics

candle's CPU `gemm` and cuBLAS choose packing/blocking/split-k from operand *strides*; a transposed view (OP_T) and a transposed copy (OP_N) of the same matrix can reduce in different orders. An `AttentionBlockFused::bwd` that materialises `pᵀ`, `vᵀ`, `dsᵀ` while candle's own `Op::Matmul` backward differentiates through views differs per GEMM by `r(1) = [0, 1.04e-7, 3.87e-8]` — inside every bf16 bound, every oracle green — and the difference grows ~700–1000× to `r(28)`. On the real model at b8·s512 that fused arm's loss went flat (0.318 → 0.291) while the same build forced eager learned (→ 0.1006). Hence one definition of a gradient GEMM's operand form, shared by both arms; dropping the copies is also a 15% speedup. The oracle asserts *growth* against the same run's own `r(1)`: `r(28) ≤ 4·max(r(1), 1e-9)` — never an absolute constant, because an absolute floor is exactly what a 1e-7 defect hides under.

### The monotone-rounding argument

Round-to-nearest-even is monotone, so two pre-rounding f32 values less than one bf16 ULP apart land on adjacent grid points at most; a two-ULP bf16 gap needs a pre-rounding delta above 2⁻⁸ of the magnitude (~130,000 f32 ULPs), which neither FMA contraction nor a cancellation-free elementwise formula can produce. So when a RoPE parity leg where both arms round once fails by 2 ULPs on one box, the bound stays at k=1: the observation is box-specific nondeterminism, and loosening to k=2 "defensively" would mask it. *A bound is a derivation; a bound you cannot derive is a bound you cannot defend.*

### Smaller facts that bite

- FMA contraction is ~1 f32 ULP and fatal to a bit-identity claim; pin with `__fmul_rn`/`__fadd_rn`.
- libm `erff` vs hardware `erff`: GeGLU's two arms compute the error function differently, unlike RoPE/casts and every other pure-arithmetic op; its parity leg needs 8 cancellation ULPs where pure-arithmetic legs need 3.
- `half::bf16::from_f64` truncates to 32 bits before rounding and disagrees with `from_f32(x as f32)` on some inputs; never call a `from_f64` reference "single-rounding".
- Exact-integer fixtures (values in −4..4) make a cross-device GEMM bit-exact regardless of summation order.

### A single-step gradient cosine cannot judge bf16 fidelity

The jammi-vs-torch gradient oracle (same checkpoint, the same LoRA weights on both stacks, one forward+backward, dropout 0) agrees at f32: mean cosine of `dL/dB` over the 112 `lora_b` tensors is 0.9999998, per-layer ‖g‖ identical to 4 significant figures. At bf16 it cannot rank arms: torch's own bf16 backward against its f32 truth ranges from −0.20 to 0.80 over six operating points (b4·s128; seeds 42–44, before and after one optimizer step), and rankings between arms reverse between seeds. The only statistic with power there is paired and sign-based. The instrument that can judge bf16 fidelity is a learning curve — a few hundred steps on a real pair dataset with a held-out evaluation, three arms, ≥3 seeds, accepting the fused arm if it sits inside the seed spread of the other two. The bench's own losses cannot do this: its synthetic triplets saturate to loss 0 within 25 steps for every arm. The per-point measurements are in `docs/plans/61-perf-unification/HISTORY.md`.

---

## 5. How to prove a fused kernel

Each rule cites the `cuda-kernel-guide.md` §3 discipline it is an instance of, the
kernel-oracle-standard id (`KO-1`..`KO-8`, `cuda-kernel-guide.md` §3) that covers it, and the
failure it prevents. `KO-7` holds by construction (a GPU test is compiled only under a
`live-*` feature and cannot obtain "no device"), `KO-3` is checked per artifact by
`ci/scripts/check_cuda_run_artifacts.py`, and the rest are held in review. A rule
with no close §3 analog (2, 12 — this guide's own additions, not folded into the kernel guide)
prints `—` in the guide column; a rule whose substance no KO id mechanizes prints
"judgment-level" in the KO column.

| # | rule | `cuda-kernel-guide.md` | KO | the failure it prevents |
|---|---|---|---|---|
| 1 | Two references; never mixed — semantic/rounding parity targets torch/HF `eager` (the model's call path) and PEFT for the LoRA epilogue; the throughput bar is torch-sdpa; FA2 op parity is torch's vendored FA2 kernel | §3.3 agreement is not accuracy | KO-8 (review-only) | declaring sdpa "the reference" would invalidate every rounding decision; jammi-eager is never a reference for jammi-fused unless pinned to torch-eager at that site |
| 2 | The oracle lives in the crate that owns the arm | — | KO-8 (review-only — an in-test reimplementation is exactly the non-independent reference KO-8 refuses) | an oracle comparing a kernel to an in-test reimplementation updated in the same diff stays green when production is reverted; the biting tests call the real `LayerNorm::slow` / `RotaryEmbedding::apply` in `jammi-encoders` |
| 3 | Same-build forced-arm A/B, elementwise, across a shape sweep including batch 1 — compare loss *sequences*, never a loss *value* | §3.6 the learning gate | KO-1 (review-only) | the operand-form defect (§4) shows at b8·s512 and hides at b8·s128; another shows only at b1·s512; its limit is a deviation both arms carry (§4, round-then-add) |
| 4 | RED controls per leg, conjunctive, with a printed magnitude — a *real semantic mutant* through the op's public surface, never an edit to the test's own copy of the array; a control must assert on *its own* leg, measured on hardware | §3.7 write comparisons affirmatively | KO-1 (the mutant is real; review-only) + KO-2 (bound coverage; check_kernel_oracles.py) | a disjunctive control (`!out_ok \|\| !dqkv_ok`) passes while the gradient arm never fires; assert `out_ok && !dqkv_ok` on a backward-only window drop, GREEN with the injection off; a dropped-scale control can sit inside its bound on real hardware |
| 5 | No absolute floors — bounds are per element and relative, with any near-zero floor *measured* from the same run | §3.8 no absolute ULP floor | KO-3 (check_cuda_run_artifacts.py, optional per artifact leg) | a fixed floor for every bf16 element below 1.0 is wide enough to contain the operand-form defect's signature |
| 6 | Bounds hold off-sample, not fitted to the seeds on hand | §3.2 key the oracle on growth against the same run's own r(1), never a fitted constant | KO-5 (review) | seed-fitted bounds give false RED on fresh seeds and overlap the mutants; delete them, do not re-fit |
| 7 | Live signal — the fixture's own cotangent/gradient must be nonzero before a mutant can be seen | §3.5 zero dispatch is RED (same failure shape one level up: a leg that cannot register a defect) | KO-6 (review-only — a static scan cannot evaluate a tensor norm) | a gradient leg whose loss is identically the batch size lets a real mutant *improve* and pass; use a seed-keyed random cotangent and a nonzero reference sum |
| 8 | Unrun is RED — a GPU-less host must never silently pass as green | §3.5 zero dispatch is RED | KO-7 (by construction) | a GPU test compiles only under `live-gpu-tests` and acquires its device through `jammi-test-resources`, so without a device it fails instead of reading green |
| 9 | Producers for every number — a doc comment or artifact must cite what produced it | §3.9 no number without a producer | KO-4 (review-only) | an artifact without a checked producer can provenance a defective build as the headline number |
| 10 | The two-term Higham bound, per leg — a relative term plus an absolute term at the operands' own scale | §3.8 no absolute ULP floor (the two-term form is how the floor is derived, not assumed) | KO-3 (the §3.8 family; the two-term derivation itself is review judgment) | one shared absolute term dominates the elementwise legs unless split by reduction term count |
| 11 | Cotangent fixtures — fixed, sign-mixed, production-amplitude, never `dy = 1` | §3.4 test at production shape and amplitude | judgment-level (§3.4 carries no KO id; the vacuous-cotangent failure it prevents is KO-6's territory) | under `dy = 1`, LayerNorm's centered backward is identically zero and the leg compares 0.0 to 0.0 |
| 12 | Mutation testing on touched files | — | judgment-level (no mechanical gate) | a crate-wide `cargo mutants` run finds survivors that review misses |

> **The principle under all of it.** An acceptance oracle is a claim that a defect would be caught, not that a number was computed. Before it may gate, it must be shown to be in a state where the defect it excludes could register. Every apparatus failure above — constant loss, bound wider than the metric's range, seed-fitted bound, a leg the defect does not reach, a control perturbing the test rather than the producer, a control vacuous on hardware, a skip that reads green — is that one bug. The gate against it is a checklist reviewers verify conformance to; the mechanical subset is enforced by `ci/scripts/check_kernel_oracles.py`.

**What a static gate cannot see.** A per-op ablation gate whose budget is `3·(max − min)` of the reference arm's own seed spread measured 2.49–3.80, while the gated quantity is a difference of two cosines, ≤ 2.0 by construction: an op whose ablation made the gradient exactly anti-parallel *passes*. The reference arm's spread was 1.27 at a median cosine of 0.016 — no resolving power. Check a budget against the metric's *range* before checking it against data; a derived budget inherits the noise of its source; measure the reference stack's spread at the operating point first. A floor too *low* to see 1e-7 (§4, reduction order) and a budget too *high* to see −1 are the same error.

---

## 6. Measuring honestly

Three of the disciplines below are general (they apply to any fused-kernel benchmark, not just this
track) and live in `cuda-kernel-guide.md` §4: exclusive box + timing lock; ratios travel across
boxes while milliseconds do not; attribute a kernel-time delta by grid, not by launch count. What
stays here is specific to this track's runs:

- **VRAM is per box** (14.65 GB on driver 570 vs 16.36 GB on driver 595 for the same build). Torch 2.11 vs 2.13 changes nothing same-box (0.642 vs 0.643). A cu130 wheel on a driver-570 box silently runs on CPU. Every VRAM figure is a pool high-water mark (cudarc uses the pooled `cuMemAllocAsync`); the OOM is the hard fact.
- **A comparison's reference is a moving part.** Rebuild torch on the box you measure on, every time: the same b8·s128 torch step measured 0.331 s on a 2024-era wheel and 0.1184 s on torch 2.13 / transformers 5.15.
- **Micro-benchmarks:** nothing allocated in the timed region (a 151 MB allocation read 5% of roofline where the kernel ran at 53–65%); per-iteration sync; ≥20 warmups, ≥200 iterations; min and median; run twice. Always end-to-end beside isolated.
- **Flag spreads, never smooth them.** The committed sweep prints the b8·s128 torch spread and another run's 0.950× reading.
- **The product step is heavier than the bench step:** the trainer always clips (+12.7 ms at s512); every headline ratio is the unclipped bench step. Recorded, not hidden.
- **A learning comparison is a paired decision, not a loss value.** Whether a fused arm learns as well as its reference is the kernel edge of the `train-run` ladder: `ci/scripts/perf/finetune_run_ab.sh` runs `jammi-bench finetune-run` legs for `resident-reference` (flash cascade and fused AdamW disabled) and `resident` at 12 seeds, two same-seed repeats each, plus the `lr = 0` control at two seeds, and `jammi-bench ladder train-run --from resident-reference --to resident --axes outcome` judges them. Before any number counts the ladder checks each leg's premises (constant schedule, padded admission, the init-anchored train probe moved, tie fraction under its cap, the dispatch counters prove the arm), identity across every leg, that the control ran at `lr = 0` and did *not* learn, and that no repeat strays from its first run by more than the seeds' differences spread. The decision is the exact sign test over `d = resident − reference` held-out loss at the count its level implies (11 of 12 at 0.0064) with the mean agreeing: a degradation is `RED`, an improvement `RED_FOR_INVESTIGATION`, anything the measurement cannot vouch for `INVALID`. Parity is a separate, stronger claim — the interval of the mean difference inside `±δ`, `δ` being what a deliberately mutated arm showed the instrument resolves — and `GREEN` without it means *no difference detected*, never *equivalent*. Mutant builds are judged by the same command (`--mutant LABEL:PATCH_SHA256`). The verdict is `ladder_verdict.json`; its `causes` name everything behind a non-`GREEN` status.
- **Time from `date -u`.**

---

## 7. Pods and builds

**The GPU dev loop.** `ci/scripts/gpu-dev.sh` rents disposable RunPod A100s (no network volume — it would pin one datacenter and lose the cloud-tier / PCIe-vs-SXM failover). Every pod carries its TTL in its *name* and a sweeper reaps it at that age regardless of in-pod timers, so the dev default is 72 h, `down` verifies before terminating, and `up` refuses over a live alias. ssh uses `IdentitiesOnly=yes`: without it, a host with many local identities hits `MaxAuthTries` before the session key and a healthy pod reads as unreachable. `pgrep -f`/`pkill -f` match their own command line and can stall or kill the caller. Pin SHAs, not branch names.

**Build times, measured.** Cold `cargo build --release -p jammi-bench --features cuda` on a 252-vCPU pod: 284 s, 749 units; the serial tail is datafusion → jammi-db → jammi-ai → jammi-bench because the bench links the whole engine (762 crates). sccache as configured gives **zero** cross-target-dir reuse on the pod image and costs +33% wall (populate 457 s; the next fresh target 473 s with 187 hits / 1,121 misses; wrapper off 344 s). The pod build instead uses a per-pod seed target, cleaned member-free at seed time and copied per worktree (`docs/maintainer/pod-build-guide.md`). Its design rests on facts each checked on a fixture: rsync-preserved mtimes make a pushed change *older* than the seed's artifacts; bindgen_cuda's PTX skip check sits *below* cargo's fingerprint layer; cargo names test binaries by target, not package; a launcher-held `flock` dies with the launcher; `tmux -t jammi` prefix-matches every `jammi-<tree>`. Transferable rule: *derive every enumeration from the tool that owns the fact* — `cargo clean --workspace` instead of globs, the kernel's lock instead of a pid file, rsync's own file list instead of a git tree hash.

---

## 8. Where it stands

The fused stack runs the ModernBERT-large LoRA step at **1.071×** torch-sdpa at b8·s512 (0.4212 vs 0.4512 s/step, 14.55 GB) and at or above parity across the sweep in §2 (`crates/jammi-kernels/artifacts/cuda-runs/2026-08-26-p6-stacked-sweep-eee7e6a-a100c-pcie.json`). On the BERT family, all four fused sites — LoRA, biased LayerNorm, the shared attention cascade seam, and dense erf-GELU — ACTIVATE for BERT and DistilBERT at both measured shapes (§3, with their artifacts). On the cross-modal towers the reachable fused sites deliver the realized gains in §3's tower profile, the profile licenses no new port (all four candidates UNRESOLVED), and the HTSAT front end is parallelized with bit identity (§3, "The media front end"). The kernel-oracle standard's mechanical subset is enforced in CI (`ci/scripts/check_kernel_oracles.py`); the pod build runs on the seed-target substrate (`docs/maintainer/pod-build-guide.md`).

Open:

- **bf16 gradient fidelity has no deciding measurement.** The single-step cosine cannot judge it (§4); the learning-curve instrument has not been run.
- **The padded FA2 regime is unmeasured.** The 1.07× is the dense branch only.
- **The explicit backward is not built.** ~19 ms of `GradStore` accumulation remains (fuse LRRL(Wi) + GeGLU).
- **`Kernel2` is unattributed.** The census kernel with six grid configs is identified only for its `gridDim.z=384` attention launches.
- **The encode ladder has no GPU measurement against PyTorch.** `encode_ab.sh` produces every rung's legs (engine rungs interleaved, both torch orders, agreement-checked by the comparator); no run of it on a real checkpoint and a real device is committed, so whether end-to-end embedding generation is at par is not yet a measured claim. What CPU legs over the tiny fixture already show is structural: over short rows, where the forward is cheap, the embeddings sink's ANN build — whose cost grows faster than its rows — is the larger part of a big serve (the `plan` leg's `sink_phases` puts numbers on it).
- **The BERT family has no flash-attention transport.** `attention_block_flash` declines by construction on every BERT/DistilBERT leg — a counted `CapabilityMiss` (§3, "Attention site").

---

## 9. Checklists

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
12. Then review runs anyway — your spot-check is a pre-filter, never the gate.

---

## 10. Glossary

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
