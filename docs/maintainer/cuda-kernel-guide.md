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

**3.1 Match the eager reference's GEMM OPERAND FORM, not just its maths (esc-044).**
candle's own `Op::Matmul` backward differentiates through transposed **views**
(`grad.matmul(&rhs.t())`); both candle's CPU `gemm` and cuBLAS pick packing/blocking/split-k from
operand STRIDES. A fused bwd that materialises `pᵀ`/`vᵀ`/`dsᵀ` with `.contiguous()` issues a
different kernel than the arm it replaces and diverges systematically. That defect flattened a real
28-layer training loss while **every committed test stayed green**. The rule from `CLAUDE.md` is
*"two things that are the same thing at a different scale are one thing"* — one definition of how
each GEMM is issued, called by both arms; assert the `(rows, cols, row_stride, col_stride)` of every
operand, captured FROM the op, never rebuilt in the test body.

**3.2 A compounding defect is invisible at one call — key the oracle on GROWTH.**
esc-044's single-layer divergence was ~2e-3 relative, INSIDE the bf16 bound, and reached O(1) over 28
layers. So assert `r(L) = Σ|fused−eager| / Σ|eager|` over an L-deep stack against **the same run's own
r(1)** — `r(L_max) <= C · max(r(1), measured_floor)`, C small. Never against an absolute ULP constant.

**3.3 Agreement is not accuracy.** Two bf16 arms can agree and both be wrong. Anchor with a
**higher-precision reference**: run the same composition in F32 and compare each arm to it. Report
Σ|arm−ref| for both; accept only if the fused arm is no further than eager. (This is how a batch-1
anomaly was correctly closed as ordinary rounding rather than chased as a defect.)

**3.4 Test at PRODUCTION shape and amplitude.** Legs that run batch 2, seq 128, or amplitude 0.1
when production is batch 8, seq 512, `max|qkv|` 9-18 are decoration. jammi shipped a defect that
every parity leg missed for exactly this reason. Include **batch 1** at the op level — it is where a
`[1,1,S,S]` per-batch mask and a broadcast-over-batch mask become shape-indistinguishable.

**3.5 Zero dispatch is RED, never green.** Assert the `DispatchCounters` delta shows the fused arm
actually ran (`fused == layers·steps`, `eager == 0`). The repo's end-to-end `learns_on_gpu` tests
were green on a broken build partly because they train a head_dim-16 model that never reaches a
head_dim-64 kernel at all.

**3.6 The learning gate (fused kernels only).** Same build, arm forced on vs off, same seed and
data, loss sequences compared **elementwise** across a shape sweep. Equal ⇒ value-neutral. Use
**batch ≥ 2**: jammi-bench's loss is `mean(relu(margin − cos(a,p) + cos(a,n)))` over `batch`
triplets, so at batch 1 it is ONE hinge — a binary switch, useless as an oracle.

**3.7 Write comparisons affirmatively.** `assert!(x.is_finite() && x <= bound)`, never
`assert!(!(x > bound))` — a NaN must FAIL, not read as a fit. Count non-finite elements before
comparing anything.

**3.8 No absolute ULP floor** in a discriminating assertion — a `k · ulp(max)` floor charges every
element the allowance of the largest and hides exactly the divergence you are hunting.

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

## 5. Host and build hazards (they silently invalidate results)

* `cargo` is not on PATH in non-interactive shells — `export PATH="$HOME/.cargo/bin:$PATH"`.
* Give every worktree a UNIQUE `CARGO_TARGET_DIR`; a shared one serves stale artifacts.
* For `cargo mutants`: COPY mode, and `CARGO_TARGET_DIR` **unset**. A shared target dir makes cargo
  report "Fresh" for MUTATED sources and scores every mutant against unmutated artifacts — a whole
  run was invalidated that way.
* Copy mode duplicates the workspace + target per job. Budget `~25 GB + 3 GB/agent + 2 GB/mutants
  job`; a mutation session wants a pod ≥ 120 GB (`RP_DISK_GB=... ci/scripts/gpu-dev.sh up`).
* `tests/cuda_parity.rs` is `required-features = ["cuda"]`, so no CPU-only gate compiles it. Whatever
  you put there is checked by the pod lane alone — say so rather than implying local green covers it.

## 6. Process

Mutating work goes through the rigor chain (`AGENTS.md`); a kernel change is not exempt.
The two steps that catch kernel bugs specifically: **pressure-test the design before code**
(a wrong kernel design compiles and passes its own tests), and **fix-verifier / red-green** — revert
the fix, keep the test, require RED. If the test is green on the broken build, it is not an oracle,
whatever else it asserts.
