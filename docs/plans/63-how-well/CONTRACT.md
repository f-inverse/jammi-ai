# CONTRACT — how-well unit (C16 lift), phase 2

Binding plan: `PLAN.md` v1 + v2 deltas (the twelve deltas are binding where they amend).
Sequenced AFTER unit 62 lands; branch `feat/63-how-well` from the post-62 main tip. One PR.
Authored by the lead 2026-08-28 (persisted early so the record leaves the session; the
implementation dispatch happens when 62 merges).

## Frame

C16 (61/CONTRACT.md:229-237) gates learning OUTCOME for the fused attention arms: held-out
loss, paired by seed, exact two-sided sign test. Pre-registered decision rule (binding, v2
delta 3/4): N=12 seeds x 2 arms (fused cascade vs ALLOFF=attention_block_flash,adamw_step_fused);
d_i = FINAL-EPOCH `evaluate_held_out()` example-mean (explicitly NOT
`TrainingResult::final_loss` = best_val_loss, a min-over-epochs order statistic,
trainer.rs:1038/1048); RED iff >=11 of 12 d_i share a sign AND the mean agrees in sign;
alpha2=0.0064; exact tail via u128 binomial sums, never float CDF. Early stopping DISABLED
via `early_stopping_patience: 10_000` both arms. Two-sided: anomalous improvement is
RED-for-investigation. Premise legs conjunctive: `admission.is_dense` per arm (v2 delta 8 —
pre-register which flash branch the fixture exercises; variable-length arxiv pairs take the
PADDED transport; add a dense cell or scope the verdict), learning-happened > floor, tie cap.
RED control: lr=0 arm x2 seeds fails learning-happened. Kernel-mutant RED column REQUIRED in
v1 — the sensitivity claim is "detects a regression >= mutant M", never movement/floor.
Triage arm pre-registered (flash-only vs adamw-only), run on RED.

## H1 — public per-pair seam (domains: wire-server for jammi-wire, ai-core for jammi-ai)

- jammi-wire: `HeldOutLoss{per_example, mean, count, tie_fraction, batch_partition_sha256,
  in_batch_negatives_per_example}` + `ExampleLoss{example_id, loss}` (v2 delta 9: partition
  digest + negatives-per-example live ON the struct — properties of (model, partition)).
- jammi-ai: `Trainer::evaluate_held_out` public; the private `evaluate` DELEGATES; its
  batch-mean semantics stay UNTOUCHED for monitoring (changing them perturbs checkpoint_best
  and pinned values) — the seam computes the example-mean as a NEW quantity.
  `compute_loss_per_example` beside `compute_loss`; dropout bracket moved inside; typed
  refusals (empty, kind-mismatch, non-finite).
- Tests (CPU-hermetic): sum-consistency targets the SEAM (batch-sum of per-example == seam
  mean x count), determinism, tie_fraction==1.0 on saturated hinge, no-RNG-perturbation
  (calling the seam does not perturb the training RNG stream).
- MNRL per-example is batch-coupled → the batch partition IS identity; held-out set sized to
  a MULTIPLE of batch_size via an explicit committed id list, never validation_fraction
  rounding (v2 delta 2), fixing every example at batch_size-1 in-batch negatives.

## H2 — exact sign test (domain: numerics)

- `sign_test` in jammi-numerics beside bootstrap/mannwhitney/welch, mirroring their API
  conventions: exact two-sided binomial tail via u128 integer sums; ties reported, never
  silently dropped without a recorded count; typed refusals (n=0, all-tie).
- Tests: exact small-n values against hand-computed tails; symmetry; tie handling; the
  pre-registered (12, 11) cell's alpha2=0.0064 pinned as a golden.

## H3 — committed fixture (domains: cookbook for fixtures, docs-ci for NOTICE)

- COMMIT the derived held-out fixture under `cookbook/fixtures/` (engine-owned; cookbook/book
  is forbidden to crates/** by check_cookbook_one_way.sh:15). The arxiv pairs are currently
  DELETED by build_finetune_cache.py:505-507 and abstracts come from a network download —
  `heldout_ids_sha256` must hash something the checkout CONTAINS (v2 delta 1). Fixture =
  committed id list + pair content for the held-out split, sized to a multiple of batch_size.
- ODC-BY 1.0 attribution appended to NOTICE (docs-ci).
- Seeds: fixed committed 12-seed gate set + disjoint >=6-seed off-sample reserve; no rotation.

## H4 — finetune-run tier + producer + workflow (domains: bench, docs-ci)

- bench: FinetuneRunTier driving the REAL TrainingLoopBuilder + the public seam (needs
  job_id/worker_id/Arc<Catalog>/artifact_dir plumbing — priced as a heavier tier build).
  `FinetuneRunTier::IDENTITY_FIELDS` = superset of the existing 18 + epochs, lr, schedule,
  warmup_steps, weight_decay, grad-accum, validation_fraction, split_rule+seed,
  dataset_sha256, heldout_ids_sha256, heldout_batch_partition_sha256, embedding_loss+temp,
  matryoshka_dims, early_stopping patience+metric, eval_cadence. Cardinality pin Rust side.
- docs-ci: FINETUNE_RUN_IDENTITY_FIELDS mirror in identity_fields.py + pins; producer
  `finetune_run_ab.sh` (NOT stacked_sweep-shaped: committed fixture, no cookbook stack, no
  server, no network) + merger computing the sign test INTO the artifact with leg-premise
  refusal; full per-epoch trajectory recorded in the artifact; r1/r2 same-seed repeat delta
  MEASURED AND REPORTED as the determinism floor — RED only if it exceeds the cross-seed
  spread (v2 delta 6).
- CI: own workflow `gpu-howwell.yml`, workflow_dispatch + PR label `run-howwell` ONLY, NO
  schedule in v1 (v2 delta 11). require-env panic-not-skip; merge-path CI checks committed
  artifacts only. Reachability allowlist: if the build tuple is byte-identical to
  stacked_sweep's row, AMEND that row's reason to name both scripts. Satisfy
  check_producer_provenance_gates (A)/(B). Workflow/gate-adjacent edits staged for
  human/admin merge where SWARM_GATE_TOUCHED fires.

## H5 — pod campaign (lead-run)

Order: (0) GPU-only dynamic-range probe, 3 runs (~$1; CPU-f32 cannot observe the arm).
(1) Floor: 12 x ALLOFF full runs → seed spread (mean-anchored; max diagnostic); r1/r2
same-seed repeat → determinism floor; committed artifact = C16's calibration run;
dynamic-range verdict IN the artifact (STOP with recorded negative result if movement/floor
inadequate). (2) A/B: 12 x fused arm (ALLOFF legs reused from floor where identity permits);
sign test computed into the artifact by the merger. (3) Kernel-mutant RED column (+$1) +
lr=0 control x2 seeds. (4) Off-sample bound verification only if a magnitude corridor is
claimed. Checkpoint: ModernBERT-large primary; base-class column iff the release text makes
a base-class quality claim. Budget: ~$10 A100 class total. Pod discipline per memory
(push-stamp, no in-job filtering, verdict lines).

## H6 — C16 supersession (domain: docs-ci)

Append-dated supersession record (never edit C16's text): each precondition quoted +
discharged; the parenthetical recorded WRONG-WHEN-WRITTEN (ValLoss has been the default since
7deadd4b 2026-04-30 — 3b8c0978 2026-06-10 only relocated the already-defaulted enum to
jammi-wire; C16 landed 2026-08-26); true residuals = final_loss=best_val_loss (min-statistic)
and parallel_train.rs:166; recall@10 dead-range finding restated with its producer.

## Acceptance (phase 6)

1. The seam: `evaluate_held_out` public, per-example losses + stable ids, sum-consistency /
   determinism / tie-fraction / no-RNG-perturbation tests — RED at base (evaluate is private,
   no per-example surface exists).
2. `sign_test` exists with exact-tail goldens — RED at base (no sign test in jammi-numerics).
3. The committed fixture exists and `heldout_ids_sha256` hashes checkout content — RED at base.
4. FinetuneRunTier + both-side identity pins + producer + gpu-howwell.yml — RED at base.
5. The pod campaign's floor artifact carries the dynamic-range verdict; the A/B artifact
   carries the computed sign test with conjunctive premise legs; the mutant column is proven
   RED. (Artifact-backed, ancestry-true shas.)

## Standing clauses

As unit 62's: own worktree + unique CARGO_TARGET_DIR; exact full gates per-step `$?`; SHAs
not branch names; gate/workflow edits human-merged; numbers measured not derived; verifiers
end with the fenced json verdict block LAST; class_enumeration required in phase-4/5/6
verdicts; no consumer names; docs reflect current state.

## Amendment 2026-08-28 (lead, post-H4a): objective selection under the triplet-shaped fixture

H4a surfaced that the committed H3 fixture is triplet-shaped (anchor/positive/negative per
row), while the Frame's "embedding_loss+temp" phrasing anticipated MNRL. The chapter itself
trains both families; the fixture's (anchor, positive) projection serves MNRL losslessly
(in-batch negatives replace the mined negative; held-out ids unchanged). PRE-REGISTERED
RESOLUTION, decided by H5 step 0's dynamic-range probe (both objectives, 1 seed each, both
arms — still ~$1):
- Selection rule: choose the objective whose probe shows tie_fraction under the cap AND the
  larger held-out movement/floor ratio; MNRL is the default on ties or ambiguity. The
  non-selected objective is dropped from v1 (no second protocol), its probe result recorded
  in the calibration artifact.
- Identity semantics per objective: MNRL → temperature non-null, margin null(NullMeans:
  "objective is mnrl"); Triplet → margin non-null, temperature null(NullMeans: "objective is
  triplet"). embedding_loss stays in identity either way.
- The tier must therefore run BOTH objectives over the committed fixture (triplet natively;
  MNRL via the (anchor, positive) projection in committed order) — H4a delta.
- The tie-cap premise leg stays conjunctive regardless of objective (C16's own hinge
  warning; MNRL keeps it as a cheap invariant).

## Amendment 2026-08-28b (lead, from unit-64 scoping): train-text provisioning

The unit-64 gap-analyzer surfaced that H4's "no network" producer discipline and the
committed fixture are mutually unsatisfiable as written: the fixture commits held-out TEXT
but only ids+per-pair-sha256 for the 1372 train pairs, so no checkout contains training
text. RULING: the producer gains a PRE-RUN provisioning step — outside every timed/measured
leg — that fetches the train text via the fixture's own derive script and VERIFIES every
pair byte-content against the committed train_ids_sha256.json (content-addressed transport:
the committed hashes are the trust surface, the network is not); any mismatch is a loud
refusal before any leg runs. "No network" is thereby narrowed to: no network during
measured legs, and no unverified content ever. dataset_sha256 remains reconstructable from
checkout + verified content. The H5 campaign runs the provisioning step once per pod.

## Amendment 2026-08-29 (docs-ci, unit-63 re-audit round-2, finding 5): FinetuneRunTier
## identity reshape — 32 fields, not H4's original 35

H4's own `FinetuneRunTier::IDENTITY_FIELDS` list above ("superset of the existing 18 +
epochs, lr, schedule, warmup_steps, weight_decay, grad-accum, validation_fraction,
split_rule+seed, dataset_sha256, heldout_ids_sha256, heldout_batch_partition_sha256,
embedding_loss+temp, matryoshka_dims, early_stopping patience+metric, eval_cadence") is
WRONG-WHEN-WRITTEN as of this amendment — kept verbatim above, superseded here. The
unit-63 adversarial-audit's finding 5 (identity-completeness) reshaped the actual set from
35 entries to 32 (`ci/scripts/perf/identity_fields.py::FINETUNE_RUN_IDENTITY_FIELDS`,
mirroring `crates/jammi-bench/src/report.rs`'s `FinetuneRunTier::IDENTITY_FIELDS` exactly):

- ADDED: `heldout_pairs_sha256` — sha256 of the `--heldout-jsonl` file's own bytes, measured
  at load; the held-out fixture's TEXT is a total determinant of every per-example loss
  `d_i` and was hashed nowhere before this fix (only the id ORDER, via `heldout_ids_sha256`,
  was anchored).
- RENAMED: `dataset_sha256` → `train_pairs_file_sha256` — the old name collided with the
  committed fixture manifest's OWN `dataset_sha256` (a Merkle digest over per-pair content
  hashes, built off-process), a DIFFERENT quantity under the SAME spelling, so neither
  anchored the other. The new name states exactly what it hashes: the `--train-jsonl`
  file's own raw bytes, measured off the file this run actually read.
- MOVED TO PROVENANCE: `split_rule`, `batched_forward`, `steps_measured` — none could vary
  independently of an already-admitted field or a build-time constant (`split_rule` is a
  hardcoded literal, `batched_forward` is always `true`, `steps_measured` is a MEASURED
  outcome of running, not a premise the run was configured under).
- DROPPED: `split_seed` — a pure, literal duplicate of `seed` (the split function takes no
  separate seed parameter).
- KEPT despite also being a pure function of already-identity inputs:
  `heldout_batch_partition_sha256` (a genuine cross-arm equality guard against the
  partitioning ALGORITHM diverging, not a redundant echo of inputs).

Net: 35 − 4 (`split_rule`, `split_seed`, `batched_forward`, `steps_measured`) + 1
(`heldout_pairs_sha256`) = 32. Contract and code now agree; this amendment is the record of
that reconciliation, never a further code change.
