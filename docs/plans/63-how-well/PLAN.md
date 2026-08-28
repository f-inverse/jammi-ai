# How-well unit plan v1 (C16 lift) — condensed persistence
(Full text with citations in the Plan agent's task output; this file carries the operative content. Authored 2026-08-28 vs the M2 train tip efc116d.)

## Reconstruction essentials
- C16 verbatim (CONTRACT.md:229-237): no feasible statistic; recall@10 dead (fine-tune moves 0.002-0.0105 over 0.538 base, tol 0.03 — producer EXECUTION-STATUS.md:803-806); held-out loss needs: public per-pair seam (Trainer::evaluate PRIVATE at trainer.rs:2248), non-default early-stopping metric, non-hinge objective or measured tie fraction, calibration run first.
- Guide :324 already prescribes the instrument: fixed-budget learning run, real pairs, held-out eval, >=3 seeds, fused inside the others' seed spread.
- The seam is a REFACTOR: evaluate() exists (dropout-safe, empty-refusing); missing = public surface + per-example (pre-mean_all) losses + stable example ids. MNRL per-example is batch-coupled -> batch partition becomes identity; per-example stats diagnostic-only in v1.
- Stats machinery exists (jammi-numerics bootstrap/mannwhitney/welch); MISSING: exact sign test.
- Floor discipline: esc-045 saga (guide:109-119) — metric inadmissible until own noise floor measured; paired/sign fallback (rule 8, :439); K_MAX deletion lesson (mean-anchored, off-sample verification, never per-seed max).
- Fixture: cookbook arxiv 1500 pairs (committed), 0.538 frozen base; split content NOT committed -> heldout_ids_sha256 becomes identity. Step times from T6 artifact -> 507 steps/run.
- C16 parenthetical partially STALE: engine default already ValLoss (jammi-wire fine_tune.rs:163-169); residual true instance parallel_train.rs:166. Supersession record must resolve precisely.
- Memeff ruling: no how-well cell today (unreachable = vacuous leg); when CUDA-wired, ONE added cell in the same matrix at a long-seq shape with dispatch-premise assertion — never its own protocol.

## Plan
(a) Public seam: jammi-wire HeldOutLoss{per_example, mean, count, tie_fraction} + ExampleLoss{example_id, loss}; Trainer::evaluate_held_out (private evaluate delegates); compute_loss_per_example beside compute_loss; dropout bracket moved inside; typed refusals (empty, kind-mismatch, non-finite); CPU-hermetic tests (sum-consistency, determinism, tie-fraction=1.0 on saturated hinge, no-RNG-perturbation).
(b) Floor: N seeds x ALLOFF arm full runs -> seed spread (mean-anchored; max diagnostic); r1/r2 same-seed repeat (nonzero delta = RED finding); committed artifact = C16's calibration run; dynamic-range verdict IN the artifact (movement/floor >> 1 or STOP with recorded negative result).
(c) A/B: N seeds x 2 arms (fused cascade vs ALLOFF=attention_block_flash,adamw_step_fused), identical everything (budget fixed, early-stop capped identically); d_i = fused - alloff per seed; PRIMARY = exact two-sided sign test (new sign_test in jammi-numerics); RED iff >=7/8 positive AND mean>0 (pre-registered, conjunctive); per-pair stats diagnostic; magnitude bound ONLY as a corridor derived from measured floor + RED-control magnitude, off-sample verified, else sign-only; two-sided (anomalous improvement = RED-for-investigation); premise legs conjunctive (dispatch counters per arm, learning-happened > floor, tie cap); RED control: lr=0 arm x2 seeds fails learning-happened.
(d) Bench/CI: finetune-run tier (drives the REAL TrainingLoopBuilder + the public seam); FinetuneRunTier::IDENTITY_FIELDS superset of the 18 (adds epochs, lr, schedule, warmup_steps, weight_decay, grad-accum, validation_fraction, split_rule+seed, dataset_sha256, heldout_ids_sha256, heldout_batch_partition_sha256, embedding_loss+temp, matryoshka_dims, early_stopping patience+metric, eval_cadence); FINETUNE_RUN_IDENTITY_FIELDS in identity_fields.py; producer finetune_run_ab.sh (stacked_sweep model) + merger with leg-premise refusal computing the sign test INTO the artifact; gate OFF merge path (gpu-prove precedent: nightly/dispatch/label), require-env panic-not-skip; merge-path CI checks committed artifacts only.
(e) C16 supersession: append-dated pointer (never edit); each precondition quoted+discharged; ValLoss staleness resolved; recall@10 finding restated with producer.
(f) Cost: ~2.0-2.5h one A100 (~$4 secure); off-sample verify +$2; mutant RED column +$1; all under ~$10. N-vs-cost is wall-clock+power, not dollars.

## Open questions (pressure-tester rules)
1. N=8 convention vs N=12 power (sign-test cells); seed-set rotation vs off-sample-bound-only.
2. Dynamic-range pre-registration probe: CPU-hermetic reduced-scale (free) predictive enough vs GPU-only single-seed (~$0.50)?
3. MNRL batch-coupling: partition-as-identity (v1) vs batch-independent scoring rule for clean per-pair pairing?
4. Checkpoint: large (T6-timed) vs base (cookbook-run) — which does the shipping claim need?
5. Memeff cell fixture: long-doc pair fixture is new + own dynamic-range question — sequence with M2 dispatch wiring.
6. C16 stale parenthetical: pre-ValLoss tip vs parallel_train — which was meant?
7. RED-control depth: lr=0 only vs real kernel-mutant column (+$1/+30min) before any bound gates.
8. Gate granularity: nightly floor+A/B together vs standing floor artifact refreshed on identity change + nightly A/B.

STATUS: awaiting pressure-test. Sequence: after embedding-surface unit, before v0.48.0.

---
# v2 deltas (pressure-test REFINE folded, 2026-08-28 — plan CONVERGED)

CONFIRMED: the instrument (paired-by-seed, exact distribution-free sign test) is right; the seam is genuinely a refactor (dropout bracket + empty-refusal already in evaluate(); flash arm on the trainer's forward path; arm selection process-scoped via OnceLock); cost arithmetic reproduces exactly; memeff ruling confirmed (zero tree hits).

THE TWELVE DELTAS (binding):
1. COMMIT the derived held-out fixture under cookbook/fixtures/ (engine-owned; cookbook/book is forbidden to crates/** by check_cookbook_one_way.sh:15) — the pairs are currently DELETED by build_finetune_cache.py:505-507 and abstracts come from a network download; heldout_ids_sha256 must hash something the checkout contains. ODC-BY 1.0 attribution appended to NOTICE.
2. Held-out set sized to a MULTIPLE of batch_size (explicit committed id list, not validation_fraction rounding) — collapses batch-mean vs example-mean and fixes every example at batch_size-1 in-batch negatives.
3. N=12; RED iff >=11 of 12 d_i share a sign AND mean agrees in sign; alpha2=0.0064 pre-registered; exact tail via u128 binomial sums (never float CDF).
4. d_i = FINAL-EPOCH evaluate_held_out() example-mean, pre-registered; explicitly NOT TrainingResult::final_loss (= best_val_loss, a min-over-epochs order statistic, trainer.rs:1038/1048). Full per-epoch trajectory recorded in the artifact.
5. Early stopping DISABLED via early_stopping_patience: 10_000 (the repo's own never-stops idiom), both arms — not "capped identically".
6. r1/r2 same-seed repeat: delta MEASURED AND REPORTED as the determinism floor; RED only if it exceeds the cross-seed spread (no CUDA bitwise contract exists — the measurement must be able to report what it measures).
7. Kernel-mutant RED column REQUIRED in v1 (+$1/+30min); the gate's sensitivity claim is "detects a regression >= mutant M", never movement/floor (the sign test is scale-free — power depends on P(d_i>0), not movement). Late-epoch movement reported beside total.
8. Premise leg asserts admission.is_dense per arm (counters record fused/declined, not dense/padded); pre-register which flash branch the fixture exercises — the variable-length arxiv pairs take the PADDED transport, a different branch than T6's dense-shape 1.55x; add a dense cell or scope the verdict accordingly.
9. HeldOutLoss carries batch_partition_sha256 + in_batch_negatives_per_example ON THE STRUCT (its value is a property of (model, partition)).
10. Triage arm pre-registered (flash-only vs adamw-only), run on RED — ALLOFF bundles two levers and a RED cannot otherwise localize.
11. CI: own workflow gpu-howwell.yml, workflow_dispatch + PR label run-howwell ONLY, NO schedule in v1 (nightly = ~$150/month; add schedule only after the measured FP rate exists). Reachability allowlist: if the build tuple is byte-identical to stacked_sweep's row, AMEND that row's reason to name both scripts (the gate keys on tuple text alone). Satisfy check_producer_provenance_gates (A)/(B). Producer is NOT stacked_sweep-shaped (needs the committed fixture, no cookbook stack, no server, no network).
12. C16 supersession records the parenthetical as WRONG-WHEN-WRITTEN (ValLoss default since 3b8c0978 2026-06-10; C16 landed 2026-08-26): the true residuals are final_loss=best_val_loss (min-statistic) and parallel_train.rs:166.
Also: evaluate()'s batch-mean semantics stay UNTOUCHED for monitoring (changing them perturbs checkpoint_best and pinned values) — the public seam computes the example-mean as a NEW quantity; sum-consistency test targets the seam, not the legacy mean. Checkpoint: ModernBERT-large primary (the perf claim's class); base column added iff the release text makes a base-class quality claim (cookbook's 0.538→0.556 is base-class — likely yes, +$1.5). Probe: GPU-only, 3 runs (~$1) — CPU-f32 cannot observe the arm (flash doesn't exist on CPU). Seeds: fixed committed 12-seed gate set + disjoint >=6-seed off-sample reserve; no rotation. Budget note: the finetune-run tier needs job_id/worker_id/Arc<Catalog>/artifact_dir plumbing (TrainingLoopBuilder's real surface) — priced as a heavier tier build.

STATUS: CONVERGED. Sequence: after the embedding-surface unit, before v0.48.0.
