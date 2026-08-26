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
