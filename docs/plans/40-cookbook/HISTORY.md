# 40 — cookbook: chapter history

Journey-shaped passages moved verbatim out of the cookbook chapters, which describe the engine as
it is. Each section names the chapter it came from.

## `chapters/22-precision/finetune-acceleration.qmd` — the pending-state race

**What this section used to concede, and what changed.** Until recently the server this chapter
started was necessarily an all-in-one deployment: mounting the `train` tier is what makes
`fine_tune` callable at all, and that same tier ran the claiming worker *inside the very process
that accepted the submission*. Between the submit RPC returning and the status RPC landing, that
worker's claim loop could already have claimed the job, resolved the device and replaced the
marker with a `"determined"` report — a genuine race, not a flake, and one a slower (2-vCPU)
render machine lost more often than a fast one. So this section asserted the race-free fact
(never `None`) plus *whichever of the two legal live states it caught*, byte-exactly inside that
state's own branch. That two-branch form was the honest previous state, and it is worth saying
why: it never loosened an assertion, it conceded only the **choice** between two assertions,
because a state the chapter could not deterministically observe is a state it must not claim to
have measured.

## `chapters/22-precision/finetune-acceleration.qmd` — the f16 memory limits

**The eager-vs-fused memory story, and a real fix.** The reporter's own target shape — batch 16,
sequence 128, ModernBERT-large [@warner2024modernbert] — could not run `f16` fine-tuning **at
all** on an 80GB A100: it OOM'd in the backward pass, while the identical shape at `bf16`/`f32`
completed. The confound was real: under the disabled-fusion arm, `f16` declines every fused
kernel (the very `Miss`es this chapter's live cells show above) and runs the fully-eager
composition, while `bf16`/`f32` ran mostly fused — so the eager composition's own memory profile
was never isolated from the *dtype*. Root cause, once isolated: `cudarc` (and candle's own CUDA
backend) carries no caching allocator — every distinct tensor shape a training loop ever requests
is a fresh `cuMemAlloc`, so a loop whose per-step shapes are drawn from an unbounded set grows the
allocator's reserved footprint with the **count of distinct shapes ever seen**, independent of
dtype. The fix bucketed each batch's natural tokenizer-padded width up to the nearest
power-of-two (a bounded, 5-rung ladder at `max_seq_length=128`), at the trainer's own
batch-construction seam — never inside `jammi-encoders`' eager arithmetic, where padding an
activation inside a mean/variance reduction would corrupt the math. Post-fix: the reporter's
exact scenario (`f16`, batch 16, sequence 128) **completes**, `steps_measured` at the full
expected count, peak memory **44.3 GB, flat** after the initial ramp (1 Hz `nvidia-smi`
sampling) — bounded, not the pre-fix unbounded 0→49 GB→78 GB growth trajectory that OOM'd.
`bf16`/`f32` complete at the same shape too, with finite, comparable losses.

**A second, campaign-induced limit at batch 8, sequence 512 — root-caused and closed.** The
stress shape OOM'd too, both `bf16` and `f16`, under the disabled-fusion arm, on legs the
pre-bucketing baseline ran clean — an early-step, not gradual, ramp to roughly 63 GB. This looked
like a second, independent memory ceiling; it was not. The actual mechanism was in the bucketing
fix itself: `evaluate_held_out`'s eval pass reused the SAME bucket-UP-to-the-nearest-power-of-two
padding the training-step fix above introduces, so a real held-out batch whose natural width was
321 tokens got rounded up to the 512 rung — `max_seq_length`'s own cap — a roughly 2.5×
softmax-intermediate blow-up relative to its true width. This was a defect the bucketing fix
introduced on the EVAL path, not a pre-existing eager-attention ceiling, and — being
dtype-independent — it explains why `bf16`, not only `f16`, was affected. The fix:
`encode_texts` now dispatches on `self.training_mode` — bucket-up padding stays on the
training-step path (unbounded per-step shape churn is the growth mechanism the first fix above
targets), while `evaluate` / `evaluate_held_out` use a `tokenize_natural_width` sibling instead,
matching the tokenizer's own natural per-batch width, exactly as it behaved before the bucketing
fix existed. Re-measured on the pod after the fix: `batch 8, sequence 512` now **completes** for
both `bf16` and `f16`, under both the fused and the disabled-fusion arm, with a full 1 Hz memory
trace — stable around **60.4 GB peak**, no OOM. The honest arc: a real limit existed, its root
cause was a second-order effect of this campaign's own first fix (never a pre-existing
eager-attention ceiling), and it is closed — not reported as still open.

From the chapter's `bf16`-vs-`f16` guidance list:

- **`f16` at the reporter's own shape (batch 16, sequence 128) now runs, bounded, at 44.3 GB,
  and the batch 8, sequence 512 stress shape now runs too, at ~60.4 GB.** Both were real fixes,
  not workarounds — the second one root-caused to this campaign's own bucketing fix leaking onto
  the eval path, not a pre-existing eager-attention ceiling. Both OOMs this campaign started from
  are closed.
