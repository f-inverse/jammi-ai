# 66 — tower profile close-out (#421 PR B)

> Plan group. The pre-registered training-step profile of the three cross-modal towers
> (CLIP-text, OpenCLIP-vision, HTSAT-CLAP audio) that issue #421 step 3 ("PROFILE FIRST",
> rule 12) requires before any kernel port on those towers is considered. See
> [`CONTRACT.md`](./CONTRACT.md) for the frozen v2.5 contract (pre-registered method,
> workload, instrument-validity and decision rule, BEFORE any measurement); this file
> records what was measured against it.

## What was pre-registered

`CONTRACT.md` fixed, before any leg ran: the workload (`--objective triplet`, `rows = 3B =
24`, fixed-shape synthetic media, real stock checkpoints `laion/CLIP-ViT-B-32-laion2B-
s34B-b79K` and `laion/clap-htsat-fused`), the twelve legs (`A1`/`A2`/`D1`/`D2` × the three
towers — wire-default F32, CUDA-opt-in BF16, and two forced-eager twins that isolate
`lora_linear_fused` and the joint `layer_norm_fused`/`gelu_erf_fused` chains), the
kernel-grain census method (N/M wall differencing, a witnessed `calls` term off
`FusibleSiteCensus`, a direct front-end timer rather than `wall − busy`), the validity gate
(counter equations, `--expect-kernels-disabled` satisfied, UNATTRIBUTED ≤ 5 % of GPU busy),
and the two-sided decision rule (ACTIVATE / DECLINE / UNRESOLVED) for the four candidate
ports the contract names: `C-ATTN-CLIP-text`, `C-ATTN-CLIP-vision`, `C-MLP-CLIP-text`,
`C-MLP-CLIP-vision`. HTSAT attention (`C-ATTN-HTSAT`, head_dim 24 at every stage) was
declared OUT OF TIER up front — a named, measured chain whose port decision is a separate
line of work, never folded into UNATTRIBUTED and never decided under #421.

## What was measured

<!-- profile-421-generated: measured-summary -->
11 of 12 legs pass the contract's validity gate — `htsat-A2` fails it (UNATTRIBUTED
share_gpu_busy=0.0566 > 0.05) and is excluded from every finding; the BF16 pre-flight (P2) passes
on all three towers. Source:
`crates/jammi-kernels/artifacts/cuda-runs/2026-09-07-profile-421-towers-c1b0b0ba-a100-sxm4.json`
(NVIDIA A100-SXM4-80GB, driver 580.126.16, nsys 2025.3.2.474, git sha
`c1b0b0bad1f79a4ad6c298400e6ea19cc1ca633c`).
<!-- /profile-421-generated -->

| tower | leg | dtype | wall s/step | front s/step | GPU busy s/step | residual s/step | front % of wall | busy % of wall |
|---|---|---|---:|---:|---:|---:|---:|---:|
<!-- profile-421-generated: measured-towers-table -->
| CLIP-text | A1 | f32 | 0.1004 | 0.0000 | 0.0608 | 0.0396 | 0.0 | 60.5 |
| CLIP-text | A2 | bf16 | 0.0965 | 0.0000 | 0.0414 | 0.0551 | 0.0 | 42.9 |
| CLIP-text | D1 (LoRA+LN eager) | f32 | 0.1486 | 0.0000 | 0.0895 | 0.0591 | 0.0 | 60.2 |
| CLIP-text | D2 (LoRA eager) | f32 | 0.1329 | 0.0000 | 0.0794 | 0.0535 | 0.0 | 59.7 |
| CLIP-vision | A1 | f32 | 0.1147 | 0.0235 | 0.0652 | 0.0260 | 20.5 | 56.9 |
| CLIP-vision | A2 | bf16 | 0.1086 | 0.0241 | 0.0386 | 0.0459 | 22.2 | 35.6 |
| CLIP-vision | D1 (LoRA+LN eager) | f32 | 0.1658 | 0.0243 | 0.0976 | 0.0439 | 14.7 | 58.8 |
| CLIP-vision | D2 (LoRA eager) | f32 | 0.1450 | 0.0243 | 0.0873 | 0.0334 | 16.8 | 60.2 |
| HTSAT | A1 | f32 | 1.5500 | 1.2509 | 0.2348 | 0.0643 | 80.7 | 15.1 |
| HTSAT | A2 | bf16 | 1.5005 | 1.2516 | 0.1883 | 0.0606 | 83.4 | 12.6 |
| HTSAT | D1 (LoRA+LN+GELU eager) | f32 | 1.6895 | 1.2493 | 0.3488 | 0.0914 | 73.9 | 20.6 |
| HTSAT | D2 (LoRA eager) | f32 | 1.6329 | 1.2550 | 0.2953 | 0.0826 | 76.9 | 18.1 |
<!-- /profile-421-generated -->

`front` is a direct measurement (`media_front_end_wall_s`), never `wall − busy`; text legs
carry `front = 0` by the contract's own stated boundary (tokenization stays in the
residual). `residual = wall − front − busy` is launch/sync overhead, computed by
`profile_421_merge.py`.

### Realized gains (eager twin minus fused, per step)

Measured by the same artifact's `realized_gains` array:

<!-- profile-421-generated: realized-gains -->
- **C-LORA** (`lora_linear_fused`, D2 minus its tower's A1): CLIP-text +32.6 ms/step (32.4
  % of the A1 baseline wall), CLIP-vision +30.3 ms/step (26.4 %), HTSAT +82.9 ms/step (5.4
  %).
- **C-LN** (`layer_norm_fused`, D1 minus D2, isolating the LayerNorm kernel on top of the
  already-fused LoRA site): CLIP-text +15.6 ms/step (15.6 % of A1 baseline wall),
  CLIP-vision +20.8 ms/step (18.2 %).
- **C-LN + C-GELU-HTSAT joint** (HTSAT's D1 disables `layer_norm_fused` AND
  `gelu_erf_fused` together, so its D1-minus-D2 delta is the two chains combined, not C-LN
  alone): +56.6 ms/step (3.6 % of the A1 baseline wall).
<!-- /profile-421-generated -->

These three chains get numbers, not ACTIVATE/DECLINE/UNRESOLVED verdicts — they are
already fused on every tower whose domain predicate admits (`CONTRACT.md`'s scope facts),
so there is nothing left to port; the realized-gain measurement quantifies what landing
them bought.

### Candidate-port decisions (verbatim from the artifact's `candidate_decisions`)

<!-- profile-421-generated: decision-grade-note -->
All four candidate ports the contract named are **UNRESOLVED** — decided on BOTH the F32
(A1) and BF16 (A2) decision legs of each tower (the pass-4 census-key fix, below, makes
both CLIP-tower A2 legs decision-grade for attribution — `htsat-A2` is merge-VALID but
fails the contract's validity gate and is excluded from every finding, see the deviation
below; HTSAT has no candidate port under this contract, so that never blocks a
candidate-port decision — no candidate is F32-only by consequence):
<!-- /profile-421-generated -->

<!-- profile-421-generated: candidate-reasons -->
- **`C-ATTN-clip-text`** — UNRESOLVED: "neither ACTIVATE (s_wall>=10% on any decision-grade
  leg) nor DECLINE (combined share <5% on every decision-grade leg) — clip-text-A1:
  s_wall+U_wall=0.1030, s_busy+U_busy=0.1703; clip-text-A2: s_wall+U_wall=0.0878,
  s_busy+U_busy=0.2045"
- **`C-MLP-clip-text`** — UNRESOLVED: "neither ACTIVATE (s_wall>=10% on any decision-grade
  leg) nor DECLINE (combined share <5% on every decision-grade leg) — clip-text-A1:
  s_wall+U_wall=0.0438, s_busy+U_busy=0.0724; clip-text-A2: s_wall+U_wall=0.0356,
  s_busy+U_busy=0.0829"
- **`C-ATTN-clip-vision`** — UNRESOLVED: "neither ACTIVATE (s_wall>=10% on any
  decision-grade leg) nor DECLINE (combined share <5% on every decision-grade leg) —
  clip-vision-A1: s_wall+U_wall=0.0641, s_busy+U_busy=0.1128; clip-vision-A2:
  s_wall+U_wall=0.0501, s_busy+U_busy=0.1410"
- **`C-MLP-clip-vision`** — UNRESOLVED: "neither ACTIVATE (s_wall>=10% on any
  decision-grade leg) nor DECLINE (combined share <5% on every decision-grade leg) —
  clip-vision-A1: s_wall+U_wall=0.0387, s_busy+U_busy=0.0680; clip-vision-A2:
  s_wall+U_wall=0.0319, s_busy+U_busy=0.0898"
<!-- /profile-421-generated -->

<!-- profile-421-generated: decline-band-summary -->
**No port is licensed under #421.** No candidate clears ACTIVATE (`s_wall>=10%` on any
decision-grade leg) or DECLINE (both `s_wall+U_wall<5%` AND `s_busy+U_busy<5%` on every
decision-grade leg) — the per-leg numbers are quoted verbatim above. This is not uniform
across candidates or axes: `C-MLP`'s own measured `s_wall` (no `U` term) is only 3.95
%/3.11 % on CLIP-text (A1/A2) and 3.36 %/2.66 % on CLIP-vision — well under the 5 %
DECLINE floor on the wall axis, combined or not (`s_wall+U_wall` above is 4.38 %/3.56 %
and 3.87 %/3.19 %, still under 5 %) — it is the combined *busy* share (`s_busy+U_busy`,
7.24 %/8.29 % CLIP-text, 6.80 %/8.98 % CLIP-vision) that lands in the contract's 5–10 %
band and is what keeps DECLINE from firing. The two-sided rule does exactly what it was
pre-registered to do: it refuses to manufacture a verdict a 5–10 % share does not
support, on either side.
<!-- /profile-421-generated -->

### Findings (verbatim text from the artifact's `findings`)

<!-- profile-421-generated: findings -->
- **`htsat-front-end-bound`**: "The HTSAT training step is CPU front-end-bound: front-end
  share of wall is 81% on the F32 decision leg (htsat-A1) (rule: front_share_of_wall >= 50%
  on every leg read; audio decode/resample/STFT/mel dominating wall time). htsat-A2 is
  excluded from this finding: it fails the contract's validity gate (htsat-A2: attribution
  decision_grade is False, not True (recorded reason: UNATTRIBUTED share_gpu_busy=0.0566 >
  0.05)). Front-end time itself is 1.249-1.255 s/step across every contract-valid F32 x
  A/D-arm leg read (htsat-A1, htsat-D1, htsat-D2) (relative spread 0.5%, within the 5%
  arm-invariance rule), so this cost is arm-invariant."
- **`clip-vision-front-end-share`**: "CLIP-vision's image decode/preprocess front end is
  20-22% of wall on the F32/BF16 decision legs."
- **`clip-launch-bound-batch8`**: "At batch 8 the CLIP training steps are launch-bound:
  3638-3722 launches/step across the four F32/BF16 A-arm CLIP legs (text and vision) (rule:
  launches/step >= 1000 and launch/sync residual share of wall (residual_s_per_step /
  wall_s_per_step) >= 20%, both on every named leg); switching to BF16 cuts GPU busy by
  32-41% per tower while wall drops by only 4-5%."
- **`c-attn-htsat-out-of-tier`**: "C-ATTN-HTSAT is measured, not a candidate port: 33% of
  GPU busy (~5% of wall) on the F32 decision leg (htsat-A1). HTSAT attention (head_dim 24
  at every stage) sits OUTSIDE the fixed-head-dim port tier by the contract's own
  declaration — this stays a measured, OPEN number, never folded into UNATTRIBUTED and
  never decided under this issue."
<!-- /profile-421-generated -->

The HTSAT front-end finding is the next line of work, not under #421 — it is being closed
on `perf/421-frontend` (PR #471), a separate, sibling unit; this doc does not duplicate its
mechanism, only points at it.

## Deviations from a clean single pass

- **esc-088, run1 all-12-legs INVALID.** `profile_421_legs.sh`'s corpus tuple shared stdout
  with its own producers, corrupting every leg's manifest at the pre-fix tip. Fixed by PR
  #469 (`fix/421-driver-corpus-stdout`, four rounds); this artifact is the full 12-leg
  re-run (run2) on the merged fix.
<!-- profile-421-generated: census-key-root-cause -->
- **The census-key root cause (pass-4, `perf/421-attribution`).** `kernel_census.py` keyed
  each GPU-kernel bucket on `shortName` alone; cutlass's `Kernel2<...>` template wrapper
  gives every bf16 GEMM tile instantiation the same literal `shortName`, so three distinct
  cutlass instantiations on `clip-text-A2` collapsed into one anonymous row that tripped
  the attribution's 1 % known-kernel-name gate. Fixed by keying on `COALESCE(demangledName,
  shortName)` instead — a strict, sum-preserving refinement (a bucket can only split, never
  merge two old buckets into fewer new ones): every top-line number
  (`gpu_kernel_us_per_step`, wall/front/busy per step) is unchanged; only the
  per-instantiation breakdown resplit. Both CLIP-tower A2 legs are decision-grade for
  attribution under the fix.
<!-- /profile-421-generated -->
<!-- profile-421-generated: htsat-a2-deviation -->
- **`htsat-A2` (bf16) is merge-VALID but fails the contract's validity gate (not
  decision-grade for attribution)**: its UNATTRIBUTED share of GPU busy is 5.66 %, over the
  contract's 5 % validity bound (window-partition copies and the audio front end's own
  activation are still undeclared chains at the identical element count as a generic
  residual-stream permute/reshape copy — the attribution module declares neither rather
  than guess). `htsat-A1` (f32) clears the bound and is decision-grade. HTSAT has no
  candidate port under this contract in the first place, so `htsat-A2`'s own
  non-decision-grade status never blocks a candidate-port decision. It is excluded from
  every finding that would read it: the HTSAT front-end finding reads `htsat-A1` only and
  asserts no dtype-invariance.
<!-- /profile-421-generated -->
<!-- profile-421-generated: corpus-pool-note -->
- Both media corpus producers emit families × instances = 24 files at any `--rows`, so the
  M-leg's 4800 rows cycle 16 distinct train clips (a page-cached working set) — the
  HTSAT/vision front-end finding is a real per-item CPU decode/preprocess compute cost, not
  a realistic-corpus I/O cost.
<!-- /profile-421-generated -->
<!-- profile-421-generated: hermetic-test-count-note -->
- **`CONTRACT.md`'s own §D5 "45 hermetic tests" figure was already stale at freeze.**
  `profile_421_merge.py`'s hermetic suite had grown to 55 tests by the freeze commit
  (`perf/421-profile-p1` @ aace002f) and stays at 55 at `c1b0b0ba` (`python3
  ci/scripts/perf/test_profile_421_merge.py` → "Ran 55 tests"); the count was true earlier
  on `perf/421-profile-p1` but drifted before the freeze landed. Not corrected in the
  frozen body (a pre-registration's text is never edited after freezing — see
  `CONTRACT.md`'s own `citations-resolve-at` header note), recorded here instead: the count
  is descriptive prose about the suite's size, not a method parameter any decision rule
  reads, so this staleness never affected a verdict.
  `docs/maintainer/fine-tune-performance-guide.md` and `CHANGELOG.md` both already avoid
  citing a bare, drifting count for this suite.
<!-- /profile-421-generated -->
<!-- profile-421-generated: basename-ambiguity-note -->
- **`CONTRACT.md`'s Scope-facts section carries three bare-basename citations that are
  mechanically AMBIGUOUS at its own pinned epoch (`bff1fad6`), not stale — resolved through
  a declared header map, never a frozen-body edit.** `check_citations.py`'s
  `docs/plans/66-tower-profile`-scoped citation form resolves a bare ``
  `<basename>.rs:<line>` `` by searching the pinned tree for a unique match; at `bff1fad6`
  this repo already has THREE `layer_norm.rs` files
  (`crates/jammi-encoders/src/layer_norm.rs`,
  `crates/jammi-kernels/src/cuda/layer_norm.rs`,
  `crates/jammi-kernels/src/ops/layer_norm.rs`) and TEN `main.rs` files across crate/test
  binaries, so `` `layer_norm.rs:129, 552-583` `` (Scope facts, para 1) and the two ``
  `main.rs:115-223` ``/`` `main.rs:1389-1400` `` citations (Scope facts, para 5) each
  resolve to more than one candidate by basename alone. The frozen body is never edited
  post-freeze (this same section's own `citations-resolve-at` header note) to spell them
  out as full paths — instead `CONTRACT.md`'s HEADER ZONE (never frozen) carries a second
  HTML comment, `<!-- citations-basename-map:
  layer_norm.rs=crates/jammi-encoders/src/layer_norm.rs;
  main.rs=crates/jammi-bench/src/main.rs -->`, naming the two intended targets.
  `check_citations.py` resolves each mapped basename to its declared path, validated
  against the pinned tree (the path must exist there and its own basename must match the
  map key) so the map can only disambiguate a genuine ambiguity, never silently re-point a
  citation — see `check_citations.py`'s own module doc for the full narrowing-not-asserting
  argument. An unmapped ambiguous basename anywhere else still fails closed exactly as
  before.
<!-- /profile-421-generated -->
- **`CONTRACT.md`'s §D6 item 1 (`CONTRACT.md:113`) names the wrong function for the
  checkpoint-content refusal — the frozen prose itself is not wrong, only stale on a name.**
  It attributes the refusal of a `$MODEL_DIR_CLIP` carrying `config.json`/`model.safetensors`
  and a `$MODEL_DIR_CLAP` missing the HF triad to `preflight_probe`, but the shipped driver
  implements that refusal in `_checkpoint_identity_probe`
  (`ci/scripts/perf/profile_421_legs.sh:387-411`, landed in 32db48bb); `preflight_probe`
  (`ci/scripts/perf/profile_421_legs.sh:418-503`) checks pinned flags and producer output
  only. The behavior the contract pre-registered exists and runs before every leg — only the
  function name in the frozen prose is stale. Not corrected in the frozen body (same
  never-edit-after-freeze doctrine as the "45 hermetic tests" bullet above), recorded here
  instead.

- **`CONTRACT.md`'s own Status line (line 7) still reads "v2.4 … Not yet frozen" inside a
  body whose header marks it frozen and whose title and §D6 declare v2.5.** The line is
  frozen-body prose that was not updated when v2.5 froze (§D6 records v2.5 = v2.4 + the
  round-3 folds). Not corrected in the frozen body (a pre-registration's text is never
  edited after freezing — see the `citations-resolve-at` header note), recorded here
  instead: the freeze itself is witnessed by the commit history (the freeze commit and the
  citation epoch `bff1fad6` both precede the measured build `c1b0b0ba`), and the Status
  line is descriptive prose no decision rule reads, so it never affected a verdict.

## PR trail

| PR | branch | what it landed |
|---|---|---|
| #465 | `feat/421-tower-lora` | LoRA-wraps CLIP-text, OpenCLIP-vision, HTSAT; image/audio triplet fine-tuning end to end — makes the towers trainable at all |
| #466 | `cookbook/421-media-tower-lora-recipes` | `image_search`/`audio_search` recipes exercise tower LoRA fine-tuning on every PR |
| #467 | `perf/421-profile-p1` | PR B-1 pre-flight code: the HTSAT GELU seam, `--expect-kernels-disabled`, the held-out split, the front-end timer, `FusibleSiteCensus`, `profile_421_legs.sh`/`profile_421_merge.py` |
| #468 | `fix/421-k4-snapshot-x86-unpinned` | K4 bit-snapshot oracle pinned same-box; drops the unpinned x86 fleet row |
| #469 | `fix/421-driver-corpus-stdout` | esc-088: the driver's corpus tuple no longer shares stdout with its own producers |
| #470 | `perf/421-attribution` | Post-export chain attribution (`profile_421_attribute.py`, `kernel_census.py`), the pass-4 census-key fix |
| — (pending) | `perf/421-artifact` | **this unit**: the close-out artifact, the frozen contract copy, this README, the guide/CHANGELOG/maintainer-guide updates |
| #471 (open) | `perf/421-frontend` | The HTSAT/CLIP-vision front-end finding's follow-on: parallelizes the media front end across rayon's global pool |
| #472 (open) | `perf/421-followups` | Other close-out follow-ups from this profile |
| #473 (open) | `cookbook/421-tower-chapter` | The cookbook chapter for the tower training-step profile (lands after #471 merges) |

Landing status is stated as of this docs commit; an "(open)" PR that merges before this
unit's own PR lands is updated by the lead at merge time, not backfilled here.
