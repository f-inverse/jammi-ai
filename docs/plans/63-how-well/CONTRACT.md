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

### Appended note, 2026-08-29 (docs-ci, unit-63 round-5 audit): golden supersession plan

`ci/scripts/perf/fixtures/finetune_run_golden/modernbert_fused.json` and
`modernbert_alloff.json` are, as of this note, a STAGED CLOSURE: their
`checkpoint_config_sha256` names the CPU-hermetic `tiny_modernbert_ner`
fixture (`head_dim=16`) while their dispatch counters are composited from
real `head_dim=64` GPU legs — a combination no real producer invocation can
emit (see the fixture's own `PROVENANCE.md`, "Emittability status" section,
for the two precise arithmetic contradictions this leaves open). The
campaign's first real ModernBERT-large (`head_dim == 64`) `finetune-run`
probe leg — run at `finetune_run_ab.sh`'s own checkpoint/batch/seq shape,
for both the `fused` and `alloff` arms — REPLACES both committed golden
files VERBATIM (identity, provenance, premise, measurement, and
dispatch-counter fields all sourced from that one real leg's own report),
at which point the emittability claim becomes true and
`GoldenProducerAnchoredFieldSetTests`'s field-set pin re-verifies against
the replacement without modification. No skip, `xfail`, or `TODO` marker
gates this — the plan is this prose record plus the fixture's own
`PROVENANCE.md` "Supersession plan" section, not a pinned-but-disabled
test.

## Amendment 2026-08-29b (lead, post-campaign-v1): probe bug fix + premise-failure handling + mutant dose ladder

Basis (committed FIRST, auditable): docs/plans/63-how-well/measurements/campaign-v1/ at
04ceb25c — the v1 run is INVALID (seed-4 alloff premise) and mutant M1 measured UNDETECTED.
Pressure-tested (rule 9) before this amendment; the findings it encodes:

1. NOT a rule amendment — an instrument bug fix. `learning_happened_delta`'s baseline is
   taken after the first resume-cycle has already trained epoch 0 (finetune_run.rs's probe
   ordering), so the premise measures epoch-1..final — excluding the largest learning
   epoch — while its field doc claims "over the run"; and its endpoint choice was never
   pre-registered (the "CONTRACT H4" citation at the definition site names a string this
   contract never contained — a fabricated citation, corrected herewith). The fix is
   statable with zero reference to seed 4:
   (a) the probe anchors at the UNTRAINED model (one evaluate_held_out on the train-probe
       batch before the first run(); LoRA init is ZerosB, so an lr=0 leg still reads
       exactly 0.0 and the strict-> floor still bites);
   (b) the producer emits the probe as a RAW per-epoch series INCLUDING the init point —
       never a pre-derived scalar — and the MERGER derives the premise from the series
       (rule: init_probe - final_probe > floor; the rule lives where rules live);
   (c) premise-failure handling is now pre-registered: the primary verdict keeps the
       strict 12-clean-or-INVALID rule unchanged; the merged artifact ALWAYS also emits a
       non-parameterised, explicitly non-decisional `premise_failure_diagnostic` block
       (failed seeds, failing legs, their raw series) — it can never promote
       INVALID->GREEN and the merger accepts NO operator override for premise failures.
2. Pre-published predictions (falsifiable, recorded BEFORE the v2 run):
   (i) the corrected probe does not touch the training path (the no-RNG-perturbation seam
       property is pinned by test), so v2's d-column is predicted to reproduce v1's
       BIT-IDENTICALLY, with dispatch counters legitimately shifted by exactly one eval's
       worth per leg;
   (ii) conditional on (i), the v2 verdict is predicted now: GREEN — v1's diagnostic
       d-column reads n_neg=8/12, mean_d=-0.0238, p=0.2266 (no concordant degradation;
       the fused arm trends BETTER). An amendment that publishes its own outcome in
       advance cannot be shopping for one. If (i) fails, the amendment's basis is void
       and the discrepancy is a finding to investigate before any verdict is claimed.
3. Mutant column, corrected design: M1 is recorded as a NON-DETECTION (a sign-flipping
   early transient — wrong space of reasoning; see campaign-v1/README.md), not a bound.
   v2 pre-registers a one-parameter, monotone, SUSTAINED dose family: the fused AdamW
   update scaled by (1+eps), eps in {0.02, 0.10, 0.50}, each dose's predicted per-seed
   effect stated in held-out example-mean units in mutants/README.md BEFORE the spend;
   each dose column is produced by substituting the mutant INTO THE FUSED ARM and merged
   against the SAME v2 alloff legs under the SAME >=11/12+mean rule (the gate's own
   statistic — mutant-vs-fused is explicitly NOT the sensitivity claim); the reported
   sensitivity is the pair of adjacent doses straddling detection. Acceptance 5's
   "mutant column proven RED" is discharged by the smallest detected dose.
4. Boundary constraints: no --allow-premise-failure, no waived-seed list, no rescale
   switch anywhere in the merger; the off-sample reserve stays reserved for step 4;
   decaying LR schedules stay disabled for this tier until the resume-cycle LR-horizon
   defect (total_steps recomputed per cycle) is fixed — the campaign's constant/0-warmup
   setting is unaffected.

## Addendum 2026-08-29c (lead): signed dose family — the prediction table falsified the positive-eps direction pre-spend

Amendment 2026-08-29b item 3 named eps in {0.02, 0.10, 0.50} (update scale 1+eps). The
REQUIRED pre-spend prediction table (mutants/README.md, derived solely from the committed
campaign-v1 measurements) predicts the (1+eps) direction as held-out IMPROVEMENT, not
degradation — the lr0-vs-trained secant slope is positive for both measured seeds, so more
effective lr lowers held-out loss over this range. A positive-eps dose therefore cannot
discharge "mutant column proven RED (degradation)". This is the prediction discipline
working: the design is falsified BEFORE the spend, on committed data, not after it.

Resolution (still pre-spend, still one monotone one-parameter family — update scaled by
(1+eps), eps now SIGNED): the ladder becomes eps in {-0.50, -0.10, +0.50}.
- Negative doses (silent lr DEFLATION — the undertrained-regression class) carry the
  predicted degradation direction (secant symmetric: predicted per-seed shift
  +|eps|*slope_seed, sign-consistent across both measured seeds).
- +0.50 is retained deliberately as the two-sided falsification cell for the improvement
  prediction itself (if it reads RED_FOR_INVESTIGATION-shaped improvement, the prediction
  is confirmed; if it degrades, the secant extrapolation is refuted and the README must
  record that).
- Acceptance 5's "mutant column proven RED" is discharged by the smallest detected
  DEGRADATION dose (expected among the negative eps values); the reported sensitivity is
  the adjacent-dose pair straddling detection within the negative branch.
Patches M_eps_-0.10 / M_eps_-0.50 are cut from the same single-constant template; the
prediction table gains their rows (same arithmetic, committed data only) before any leg runs.

**Postscript, 2026-08-29c** (docs-ci, unit-63 round-7 audit advisory (a)): amendment
2026-08-29b item 2(ii)'s own diagnostic-d-column sentence ("v1's diagnostic d-column reads
n_neg=8/12, mean_d=-0.0238, p=0.2266") mixes two different denominators — the sign counts
(n_pos/n_neg) and mean_d/p are all computed over the v1 run's 11 CLEAN (premise-passing)
seeds (seed 4's alloff leg is the run's one premise failure, per that item's own basis
paragraph), never the pre-registered 12-seed gate count. The correct, denominator-consistent
statement: n_pos=3, n_neg=8 of 11; mean_d=-0.0238, p=0.2266, both also over the same 11. Item
2(ii)'s own prose is left exactly as originally recorded (append-only); this postscript is
the correction of record. measurements/campaign-v1/README.md carries the identical fix in
place, with its own dated correction note.

**Postscript, 2026-08-29 (docs-ci, unit-63 "RED-proof column"):** the signed `(1+eps)`
ladder above DEMONSTRATED the detector (11/12 concordance, `p=0.00635 < alpha2`) but landed
every scheduled dose in the IMPROVEMENT direction — no member of this family can discharge
acceptance 5's "mutant column proven RED (degradation)" (see
`measurements/dose-ladder/README.md`'s own finding 3). `ab_merge.py` therefore gains a
**RED-proof label class**: a `dose_label` carrying the literal prefix `redproof-` (e.g.
`redproof-nobc`, `redproof-signflip`, mutants/README.md's own `M_nobc`/`M_signflip` pair,
both outside the `(1+eps)` lr-scale family by construction) participates fully in
`build_mutant_dose_column` (premises, partner premises, identity, `detected` computation —
unchanged) but is partitioned OUT of the eps-family scans (`sensitivity`,
`two_sided_falsification`, `dose_anomalies`, and the duplicate-EPS arm of
`mutant_dose_ladder_reject_duplicate_doses`) BEFORE those scans ever run — a partition on
the label prefix, never a widening of `_dose_label_eps`'s own strict eps-only domain; the
duplicate-LABEL and duplicate-PATCH_SHA arms still apply across the full set, so a
RED-proof column remains subject to the same same-patch-measured-twice refusal any eps
column is. The merged artifact gains `red_proof` (one
`{dose_label, patch_sha256, detected, n_pos, n_neg, mean_d, p_value, clean_pair_count}` entry
per RED-proof column) and `red_proof_verdict` — `"PROVEN"` iff at least one RED-proof column
reads `detected == "RED"`, otherwise `"NOT_PROVEN"` naming every column's own `detected` (a
column reading `RED_FOR_INVESTIGATION` is recorded as-is, an anomaly for a mutant expected to
degrade, never a second way to reach `"PROVEN"`). `red_proof_verdict == "PROVEN"` contributes
NOTHING to the exit code — it is the expected outcome and never masks another dose-ladder
failure cause (a PROVEN merge can still exit non-zero on `sensitivity_error`, an `INVALID`
dose column, `dose_anomalies`, or a non-GREEN primary status); `"NOT_PROVEN"` contributes
non-zero, named in the merge's own stderr and artifact. This retires the prior "run
`redproof-nobc`/`redproof-signflip` in
their own, separate invocation and treat its exit 1 as expected" convention
(mutants/README.md's own former "Minimal labeling convention" section) — reinterpreting a
failure exit as success was never acceptable; the RED-proof verdict is now read directly off
this merge's own first-class field, in the SAME artifact the primary decision lands in.

**Postscript, 2026-08-29d** (docs-ci, unit-63 round-13 audit finding 2): the postscript above
underspecified `red_proof_verdict`'s own null state. `ab_merge.py`'s `except
(MutantDoseLadderSensitivityError, RedProofLabelError)` handler resets `sensitivity`/
`two_sided_falsification`/`dose_anomalies` (+ `sensitivity_error`) on ANY dose-ladder refusal,
but `red_proof_verdict` must distinguish two different refused states, never collapse them to
the same `null`: no RED-proof label was ever present in the supplied dose set (kept `null`,
exactly as before — nothing to report), versus at least one RED-proof-labeled column WAS
scheduled but the refusal fired before RED-proof evaluation ever ran (`partition_red_proof_
dose_columns` itself, or any eps-family scan downstream of it, raising before `build_red_
proof_summary` is reached). The second state is now recorded as an explicit `NOT_PROVEN`-class
verdict naming the refusal (`"NOT_PROVEN (dose set refused before RED-proof evaluation: <exc>)"`),
never left `null` byte-identical to the first — a `null` `red_proof_verdict` means only "no
RED-proof column was scheduled", never "one was scheduled but its own evaluation never ran".
This lets both the existing NOT_PROVEN exit fold (`main()`'s own dose-ladder exit-code branch)
and `runpod_gpu_howwell.sh`'s own GREEN-but-nonzero cause namer (`howwell_dose_ladder_cause.py`)
fire on this state instead of falling through to an unexplained-contradiction "unknown" cause.
