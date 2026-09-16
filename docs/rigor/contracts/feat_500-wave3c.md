# Contract — `feat/500-wave3c`: the remainder of plan 67's wave 3, one consolidated PR

Base: `main` @ `b5a7aab9` (PR #580 merged: U5b-1a-A2 root identity, U5b-0 leaf inventory,
the merge-path runner). Slug `feat_500-wave3c`; pressure row in
`docs/rigor/feat_500-wave3c.jsonl`; oracle record in `docs/rigor/feat_500-wave3c.oracle.jsonl`
at the final tip, exported last, after every non-rigor edit.

## 1. Scope and build order

The plan's sizing doc pins two hard orders inside the remainder (U4b before U5b-1b-ii;
U5b-1b-ii before U5b-1b-iii and U5b-2); everything else is independent and was built
concurrently, one implementer per unit in its own worktree off `856ec8dd`, consolidated here
as one commit per unit:

| # | Unit | Built | Depends on |
|---|---|---|---|
| 1 | U4b — rank context; gather rule; lockstep; single-node `Local` gang; `[worker] local_ranks` | concurrently (wave A) | main |
| 2 | U5b-1b-i — `Peer` collective + two-phase round protocol; peer-listener decode cap | concurrently (wave A) | main |
| 3 | U5a-2 — `HostAdmission`, drain, re-verification, admit-and-hold; the world>1 admission conjunct (#566) | concurrently (wave A) | main |
| 4 | U5b-1b-ii (db) — `[distributed] max_world_size`; migration 037; assembly reason table, cooldown/counter; claim-side cooldown term; CAS call-site helper | concurrently (wave A) | main |
| 5 | U5b-1b-ii (body) — coordinator: membership → assignment → dispatch → assembly; `serveable_world` at submit | after 1–4 | 1, 2, 3, 4 |
| 6 | U5b-1b-iii — `world_size == 1` rank body; runner-role writer split; `Outcome`; resume pin | after 5 | 5 |
| 7 | U5b-2 — watchdog; abort with no terminal write; released-vs-failed; chaos | after 6 | 5, 6 |

Not in this PR: U7b-A2b live (RunPod cluster runs, with the user); the pod and cluster legs'
artifacts (label-gated, run after merge).

Each unit's section below is the implementer's contract file, folded by the lead after the
lead opened every cited line and re-ran every named oracle on the consolidated tree; every
deviation from `docs/plans/67-distributed-training/UNITS.md` is stated with its reason and the
plan rows carry a dated correction in the same commit.

## 2. U4b (landed as twelve commits on this branch; original tip `c987001f`)

The implementer's contract, folded by the lead after checking: the presence-set reduce, the step-bounded epoch loop, the flag-reduced divergence decision, the `world > 1` refusals for the streamed, mining, GradCache and precomputed arms, `TrainingBatch::Classification { logits, .. }`, `RESUME_STATE_SCHEMA_VERSION = 2` with `load_bundle -> Ok(None)` on a version mismatch. The pressure round's blocks 1–4 are closed by this shape. Three cuts the implementer named are SCHEDULED on this branch, not accepted: (i) `run_spec` spawning `[worker] local_ranks` ranks over `Local` (built with the coordinator body, §6, the same `run_spec` topology site); (ii) the per-rank dropout seed wiring, which needs the LoRA init seed split from the dropout seed (§2a); (iii) the streamed arm's zero-row-rank fix in `stream.rs`, so a residency-bounded stream trains at `world > 1` instead of being refused (§2a). Consolidation notes: the twelve commits applied without conflict; the `Collective` trait now takes a per-call `&BlockingCall` (U5b-1b-i), so `RankContext`'s five wrappers are threaded with the witness in the integration commit (§2b).

### 1. Scope shipped

**`crates/jammi-ai/src/fine_tune/trainer.rs`**
- `RankContext { collective: Arc<dyn Collective>, partition: PartitionSpec }` — the trainer's
  own gang identity. `RankContext::single_rank` (Noop over `PartitionSpec::single_rank`) is
  `TrainingLoopBuilder`'s default when `.rank_context(..)` is never called, so every pre-U4b
  caller/test is unaffected byte-for-byte. Five wrapper methods
  (`all_gather`/`all_reduce_sum`/`all_reduce_max_flags`/`broadcast`/`barrier`) are the ONLY
  route the trainer/optimizer use to reach the underlying `Collective` (design pressure round
  finding 9 — the seam the concurrently-built `Peer` collective, U5b-1b-i, needs to add a
  per-call witness argument to). `RankContext::canonical_vars_digest(names)` — a stable digest
  of the canonical `trainable_vars` order, exposed for U5b-1b-i's round descriptor
  (`agreement`). `RankContext::dropout_seed`/the free `rank_dropout_seed(base_seed, rank)` —
  `f(seed, rank)` per DESIGN.md §4, rank 0 reproducing `base_seed` exactly. **Deviation**: this
  function is unit-tested in isolation
  (`rank_context_dropout_seed_is_identity_at_rank_zero_and_distinct_elsewhere`) but is NOT wired
  into any model-construction call site (production or test) in this unit — see Uncovered §3.
- `compute_loss_gathered` — the gather rule. `Contrastive`/`Pairs`/`Triplet` gather the
  post-projection encoder outputs + scores (already computed locally by `encode_chunk`).
  `Classification` gathers the LOGITS: `TrainingBatch::Classification` was reshaped
  (`data.rs`) to carry `logits` instead of `embeddings` — `classify()` now runs inside
  `encode_chunk`, on this rank's own local embeddings, symmetric with `Regression`'s
  `head_forward`-before-loss shape (design pressure round finding 5). `Ner` stays refused.
  Every arm's gathered tensor is downstream of every trainable parameter, so `all_gather`'s
  detached-remote-slot backward closes the W× gradient hazard (proved by an executed
  RED-PROOF mutation, §2).
- The Resident production epoch loop is now bounded by the fixed, once-computed
  `train_batches_per_epoch` (a function of `self.rank_ctx.world()`, not a hardcoded `1`),
  never "this rank's own chunk came back empty" — the two coincide at W=1 (proved: a
  zero-row chunk can only ever fall at `step == train_batches_per_epoch`, one past the loop)
  but diverge at W>1, where a zero-row RANK's slice at a step a PEER rank still has data for
  would otherwise skew the gang's collective-call count (executed RED-PROOF: a hang, §2).
  `encode_texts` gained an empty-batch guard: a real encoder forward errors on a zero token
  count (measured: `rope_fused: cos/sin element count 0 is not a positive multiple of
  head_dim`), so a zero-row tensor of the right trailing width is built directly, no model
  call at all.
- `process_batch_loss`'s divergence decision now reduces through `all_reduce_max_flags`
  BEFORE the accumulate/divergence-count decision (a forced divergence on one rank is seen,
  and acted on identically, by every rank); at W=1 (`Noop`) this is the identity, so the W=1
  window is byte-unchanged.
- `save_resume_checkpoint`/`save_epoch_checkpoint` are no-ops for `rank_ctx.rank() != 0`
  (DESIGN.md §4: rank 0 alone writes/publishes) — the artifact-store-presence check and the
  new `gather_dropout_positions` collective call both happen BEFORE the rank gate, so every
  rank still takes the same collective calls in lockstep.
- `gather_dropout_positions` — an `all_gather` of every rank's own per-layer dropout Philox
  positions to rank 0 (real collective call, every rank participates even though only rank 0's
  caller reads the result). `restore_from_checkpoint` selects THIS rank's own entry from the
  per-rank map (never rank 0's).
- The Streamed arm is refused, typed, at `world > 1` (its own `next_chunk` still collapses an
  empty-but-in-bound chunk into end-of-epoch — a named, deferred gap, Uncovered §1); mining/
  GradCache/a Precomputed loader are likewise refused at `world > 1` (design pressure round
  finding 4) — restating (mining/GradCache) or newly adding (Precomputed) the run-time edge of
  the submit-time refusal `RankAdmission::admit` already applies.
- Rung pinning: NOT wired (design pressure round finding 6 — the gather is dim-0 over pooled
  `[rows, hidden]`, so dim-1, which a bucket rung governs, is irrelevant to gather agreement;
  every call site still passes `None`). `batch_bucket.rs`'s doc corrected to state this.

**`crates/jammi-ai/src/fine_tune/partition.rs`**: `PartitionSpec::for_gang` (the production,
un-gated, validated arbitrary-rank constructor — `for_rank`, test/`test-hooks`-only, is now a
thin wrapper over it), `PartitionSpec::rank()`/`world()` accessors, `PartitionSpec::
counts_for_step` (every rank's row count for a step, derived — never exchanged).

**`crates/jammi-ai/src/fine_tune/optimizer.rs`**: `canonical_reduce(rank_ctx, trainable_vars,
grads)` — lays `grads` out in canonical order, `all_reduce_sum`s it AND a second presence-set
reduce (a per-var 1.0/0.0 indicator), and writes a var back to `grads` only when present on at
least one rank — absent-on-every-rank stays absent (design pressure round finding 2: absent is
not zero for `AdamW::step`, which skips an absent `Var` entirely — moment decay, bias
correction, weight decay all move `θ`).

**`crates/jammi-ai/src/fine_tune/resume.rs`**: `RESUME_STATE_SCHEMA_VERSION` bumped 1→2;
`dropout_positions: HashMap<u32, HashMap<String, u64>>` (per rank, was flat).
`ResumeState::schema_version` gets `#[serde(default = "unversioned_schema_version")]` (→ `0`,
never a real version); `load_bundle` now returns `Result<Option<RestoredCheckpoint>>` — a
version mismatch (including the now-parseable absent-field case) is `Ok(None)` + one
`tracing::warn!`, never a hard `Err` (design pressure round finding 7 — greenfield, no reader
for an old shape, so a hard failure would strand every live older checkpoint). A genuinely
torn moments file still hard-errors once the version check passes.
`crates/jammi-ai/src/fine_tune/worker.rs::discover_resume` updated to match.

**`crates/jammi-ai/src/fine_tune/data.rs`**: `TrainingBatch::Classification { logits, labels }`
(was `{ embeddings, labels }`).

**`crates/jammi-ai/src/pipeline/recompute.rs`**: the K1 `FineTune` (retrain) arm's exhaustive
destructuring extended with `collective: _, local_ranks: _` (bound the moment the manifest
fields landed, as designed).

**`crates/jammi-db/src/store/manifest.rs`** (co-owned with U2a/U3, authorized for this unit):
`ProducingDescriptor::FineTune` gains `collective: String` (canonical lowercase token) and
`local_ranks: u32`; the completeness test (`FineTuneFields`/`fine_tune_descriptor`/
`fine_tune_fields`/`fine_tune_every_field_moves_the_hash`) extended for both.
**`crates/jammi-db/src/store/artifact.rs`**: its own `fine_tune_descriptor()` test fixture
updated to match (the only other production-shape construction site in that crate).

**`crates/jammi-ai/src/fine_tune/worker.rs`**: `train_fine_tune`'s manifest-descriptor
construction populates `collective`/`local_ranks` from `session.inner_config().worker` at
claim time.

**`crates/jammi-db/src/config/mod.rs`, `config/tests.rs`, `config/host_memory.rs`,
`crates/jammi-ai/src/fine_tune/collective/mod.rs`, `docs/guide/src/configuration.md`**:
`WorkerConfig::world_size` → `local_ranks`, `WorkerTopology::world_size()` → `local_ranks()`
(U4b S8) — every OTHER `world_size` (spec.rs, manifest.rs's `FineTune::world_size`,
jobs_repo.rs, gang.rs, the wire) is a DIFFERENT concept and is untouched, per the design
pressure round's explicit instruction. `docs/guide/src/configuration.md` documented no
`[worker] world_size` at all before this unit, so this is an ADD of the whole gang-knobs block
(`local_ranks`, `collective`, `rank_timeout_secs`), not a rename in the guide.

#### Deviations from UNITS.md / the brief, with reasons

1. **`run_spec` does not spawn `[worker] local_ranks` local ranks.** UNITS.md's own text names
   this ("`run_spec` spawns `[worker] local_ranks` ranks on one host over `Local`"). Not built.
   The gather rule, lockstep, canonical reduce, and per-rank resume are fully built and tested
   hermetically by constructing `RankContext`/`Local`/`LocalGang` directly and driving the REAL
   `TrainingLoop::run` (never a mock), which is what every acceptance oracle in UNITS.md
   actually asks for — none of the five hermetic oracles names `run_spec` as part of what it
   measures. Wiring `run_spec` itself to fan out N ranks (own model load per rank, own
   `RankContext`, per-rank dropout seed derivation, thread lifecycle inside the worker's
   existing claim/heartbeat/cancel machinery) is a distinct, large orchestration change I judged
   out of proportion to attempt and verify soundly in the remaining time without risking a
   half-tested regression in worker.rs's existing (heavily-tested) claim loop. Named as a
   follow-up, not silently absorbed. See Uncovered §2.
2. **`RankContext::dropout_seed`/`rank_dropout_seed` is not wired into any model-construction
   call site.** Investigated concretely (not merely deferred by assertion): every existing
   LoRA-layer constructor (`LoraLinear::new`, `LoraBuildConfig`) takes ONE `seed` field that
   governs BOTH weight init and dropout. Per-rank-varying that ONE field (as a naive wiring
   would) would ALSO vary weight INIT per rank, breaking "every rank starts with identical
   weights" — a real correctness hazard, not a style question. Splitting the init seed from
   the dropout seed is a LoRA-layer change this unit's files_in_scope does not include. The
   free function and its own property (rank 0 identity, other ranks distinct and
   deterministic) are built and tested in isolation; wiring it is filed as a follow-up. See
   Uncovered §3.
3. **The Streamed arm's own zero-row-rank hazard is not fixed, only refused at `world > 1`.**
   `EpochSource`'s `next_chunk` (now serving the Streamed arm alone; the dead `Resident`
   variant was removed) still collapses an empty-but-in-bound chunk into end-of-epoch — the
   SAME hazard the Resident arm's fix closes. Fixing it requires threading the
   `train_batches_per_epoch` bound through the stream's own pump/consumer protocol
   (`stream.rs`), which U2c built under its own, separate contract; reworking it was judged
   out of this unit's scope (files_in_scope names `trainer.rs`'s stream CONSUMER, not
   `stream.rs`'s pump). A typed refusal at `world > 1` (K2: refuse rather than compute past a
   valid domain) closes the gap honestly instead of leaving it silently reachable.
4. **Acceptance (c) is not a separately-built test.** UNITS.md's (c) ("W=2 × B vs W=1 × 2B
   within pre-registered ε at `lora_dropout=0`") is, on inspection, the SAME comparison as (b)
   under the framing this unit already built: `gather_exactness_w2_matches_w1_within_pre_
   registered_epsilon` runs W=2 at per-rank batch 2 (global batch 4) against a W=1 reference at
   batch 4 (`= 2×B`), both at `lora_dropout` effectively disabled via eval mode (see that
   test's own doc for why), within a pre-registered ε. I did not duplicate it as a second test;
   the contract's Properties table below cites it for both (b) and (c).

### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| At every world size, `RankContext`/`Noop` wiring changes ZERO bytes of a W=1 run | `cargo test -p jammi-ai --features test-hooks --lib fine_tune::` (319 passed) — the WHOLE pre-existing suite is this oracle | Not re-mutated (this is the pre-existing K4 regression suite, unchanged); any of the wiring changes in this unit would have reported here first if it had perturbed W=1 |
| `PartitionSpec::for_gang`/`for_rank` agree and validate their bounds; `counts_for_step` matches `rows_for_step` per rank, including the zero-row-rank case | `fine_tune::partition::tests::for_gang_validates_and_for_rank_agrees_with_it`, `..::counts_for_step_matches_rows_for_step_for_every_rank_including_a_zero_row_rank` | Covered by construction (assertions on out-of-bounds inputs; the zero-row case is asserted directly, `vec![2,0]`) |
| `canonical_reduce`: at W=1, changes NOTHING (absence and presence both survive unchanged); at a real gang, sums a shared var, resolves a rank-absent var to the other rank's own value, and restores absence where NO rank had the var | `fine_tune::optimizer::tests::canonical_reduce_at_world_one_leaves_absence_and_presence_unchanged`, `..::canonical_reduce_sums_a_real_gang_and_restores_absence_where_no_rank_had_it` (a real 2-thread `Local` gang) | RED-PROOF (executed, reverted before commit — see commit `989dc3ff`'s own message): the FIRST-CUT shape of this function (before the presence-set fix) unconditionally inserted a zero-filled tensor for every var; reverting to that shape makes the absent-var assertion fail (`grads.get(..).is_none()` becomes `Some`) |
| `RankContext::canonical_vars_digest`: order- and content-sensitive | `fine_tune::trainer::tests::canonical_vars_digest_is_order_sensitive_and_content_sensitive` | Covered by construction (asserts reordering and content changes both move the digest) |
| `RankContext::dropout_seed`/`rank_dropout_seed`: rank 0 is the identity; every other rank is distinct and deterministic | `fine_tune::trainer::tests::rank_context_dropout_seed_is_identity_at_rank_zero_and_distinct_elsewhere` | Covered by construction (asserts rank 0 == base_seed exactly, ranks 1..8 pairwise distinct, and repeat-call determinism) |
| **(b/c) Gather exactness**: on a REAL 2-rank `Local` gang, the global loss AND the summed adapter gradient equal a W=1 reference (same rows, one combined batch) within a PRE-REGISTERED ε = `1e-4`, on a fixture whose per-rank batches bucket to DIFFERENT natural widths | `fine_tune::trainer::encode_texts_bucketing_oracle::gather_exactness_w2_matches_w1_within_pre_registered_epsilon` | EXECUTED (applied, run, reverted — see commit `bb6a3c12`): `compute_loss_gathered`'s `Contrastive` arm changed to `.clone()` instead of gathering; red output: `rank 0: gathered global loss 0.37497652 must match the W=1 reference 1.0657526 within 0.0001` |
| **(d) Lockstep — zero-row rank**: a gang whose last global step gives one rank zero rows completes (no hang, no rank-count skew) | `fine_tune::trainer::gang_lockstep_oracle::a_zero_row_rank_the_gang_completes` (a real 2-rank `Local` gang through the FULL production `run()`) | EXECUTED (applied, run, reverted — commit `b1ffecf6`): restored the pre-U4b "this rank's own empty chunk ends the epoch" loop termination; confirmed a HANG (did not complete in 25 s vs 0.34 s healthy), left to resolve naturally at `Local`'s own 120 s rendezvous timeout: `all_gather: timed out after 120s waiting for every peer to arrive` |
| **(d) Lockstep — a Var absent from one rank's `GradStore`**: the gang completes | `..::a_var_absent_from_one_ranks_gradstore_the_gang_completes` (a real gang, `after_backward` removes rank 1's gradient for one Var at step 1) | Covered directly by `canonical_reduce`'s own presence-set RED-PROOF above (same mechanism); this test is the run()-level integration proof that the mechanism is actually wired into the production step boundary |
| **(d) Lockstep — forced divergence on one rank**: both ranks end in the SAME typed refusal, never one erroring while its peer hangs/trains on | `..::forced_divergence_on_one_rank_both_ranks_end_in_the_same_refusal` (a real gang, `after_backward` poisons rank 1's gradient to NaN at step 1) | This property IS its own RED-PROOF-shaped test: before U4b, `RankContext`/`canonical_reduce` do not exist, so there is no second rank for a poisoned gradient to propagate to at all — the assertion (`both ranks' errors name step 1`) has no meaning pre-U4b |
| **(a) Equal-topology reproducibility, W=2 twice**: two independent from-scratch 2-rank gangs (same seed/data) produce byte-identical rank-0 final weights | `fine_tune::trainer::gang_determinism_oracle::w2_twice_is_byte_identical` | EXECUTED (applied, run, reverted — commit `68299c5d`): changed the second run's epoch count 2→3; confirmed red, reverted |
| **(a) Equal-topology reproducibility, resume across a kill**: a 2-rank gang cooperatively cancelled after one epoch, then resumed from the SAME shared artifact store, reproduces an uninterrupted run's final weights byte-for-byte | `..::resume_after_a_kill_matches_an_uninterrupted_run` | EXECUTED (applied, run, reverted — commit `68299c5d`'s own message): commented out `optimizer.load_state(..)` in `restore_from_checkpoint`; confirmed a real byte divergence, then reverted. (This test's own construction ALSO caught two real test-harness bugs along the way — a shrunk `config.epochs` for the "killed" leg, and firing `cancel` during the checkpointed epoch's own last step — both of which silently made the test pass for the WRONG reason (no resume ever happened); both are recorded in the test's own doc as the reason an executed RED-PROOF, not a green result alone, is load-bearing here.) |
| **(e) W=1 via `Noop` is byte-identical to the existing golden(s)** | The WHOLE pre-existing `fine_tune::` suite (319 tests, unchanged pass) is this oracle — no new test needed since `RankContext::single_rank`/`Noop` is a pure addition | N/A — regression-only claim |
| Manifest `collective`/`local_ranks` are real hash determinants on `ProducingDescriptor::FineTune` | `jammi-db::store::manifest::tests::fine_tune_every_field_moves_the_hash` (extended) | Covered by construction (`assert_each_change_moves_hash` over both new fields) |
| The K1 retrain arm's exhaustive destructuring cannot silently drop a new manifest field | `cargo check -p jammi-ai` failed to compile before `recompute.rs` was updated (E0027, "pattern does not mention fields `collective`, `local_ranks`") | The compile error itself IS the executed proof — the completeness test is the type system |
| `WorkerConfig`/`WorkerTopology`'s rename preserves every existing validation/round-trip behavior under the new name | `jammi-db::config::tests` (216 passed, including the renamed `load_refuses_local_ranks_zero`/`..wider_than_the_devices`) | Renaming is the only change; the pre-existing assertions are the oracle |

### 3. Uncovered

1. **The Streamed arm's per-rank zero-row-rank hazard** (see Deviation §3 above) — refused,
   typed, at `world > 1`, never fixed. A real gang against a residency-bounded per-rank stream
   is therefore not exercised at all; only the Resident (eager) production arm is proven at
   `world > 1`.
2. **`worker.rs::run_spec` spawning `[worker] local_ranks` local ranks** — not built (Deviation
   §1). The claim-loop-to-gang-spawn wiring, per-rank model loading, and per-rank seed
   derivation (which needs Uncovered §3 first) are a named follow-up.
3. **`RankContext::dropout_seed` wiring** — built and unit-tested in isolation, not connected
   to any model constructor (Deviation §2). Consequently `gang_determinism_oracle`'s two tests
   use `lora_dropout = 0.0` and do NOT exercise the per-rank dropout-position gather/restore
   determinant specifically (restoring the wrong rank's positions is vacuous when every rank's
   map is empty) — stated plainly in that test's own doc, not hidden.
4. **Acceptance (f)**: "two real devices in one session hold two entries in the production
   model-cache map for one model id" — hermetically UNCOVERED, exactly as UNITS.md itself
   states ("only one device exists off the pod... a pod-leg obligation, not a hermetic one").
   Not attempted, not faked. U4a's own hermetic assertion (a mirror map, not the production
   insert) is the only hermetic proxy this plan defines for it, and it is unit U4a's, not this
   one's, to ship.
5. **The pod leg** (`Nccl`, 2×A100; digest pair + per-step delta against a GPU-measured ε) —
   explicitly named in UNITS.md as NOT this unit's; labelled UNCOVERED, not attempted.
6. **`docs/maintainer/MAINTAINER-GUIDE.md`'s `PRODUCING-DESCRIPTOR-VARIANTS` block** — not
   updated for the two new manifest fields. U4b's own `files_in_scope` in UNITS.md does not
   name this file (only U2a's and U3's do); left for the lead's doc-parity pass
   (`check_doc_parity.py`), which I did not run myself (outside the trimmed gate list I was
   given).
7. **Live-Postgres arm for the jammi-db tests touched** — not run. Every jammi-db test I added
   or renamed in this unit (`config::tests`, `store::manifest::tests`, `store::artifact::
   tests`) is pure Rust logic (config parsing/validation, hash-determinism fixtures) with no
   catalog/SQL backend involved, so no `::postgres` arm exists for them to run under.

### 4. Gates

Per the lead's mid-task gate-trim message (COMMON.md's Gates section rewritten): only the
crates/tests touched, one filter per module, no workspace-wide builds, no `cargo doc`, no
live-Postgres lane (none applicable — see Uncovered §7), no `merge_path.sh`.

| Command | Exit | Notes |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | 0 | |
| `cargo clippy -p jammi-db --all-targets -- -D warnings` | 0 | |
| `cargo test -p jammi-ai --features test-hooks --lib fine_tune::` | 0 | 319 passed, 0 failed |
| `cargo test -p jammi-ai --features test-hooks --lib pipeline::` | 0 | 33 passed, 0 failed |
| `cargo test -p jammi-db --lib store::` | 0 | 121 passed, 0 failed |
| `cargo test -p jammi-db --lib config::` | 0 | 216 passed, 0 failed |

`RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ai --no-deps` was NOT run per the lead's trimmed
gate instruction (explicitly excluded: "no `cargo doc`"); the lead runs the full merge path once
on the consolidated tip.

### 5. Commits

```
c987001f feat(db,ai): #500 U4b — manifest topology fields: collective, local_ranks on ProducingDescriptor::FineTune
68299c5d test(ai): #500 U4b acceptance (a) — equal-topology reproducibility, a real two-rank Local gang, from scratch and across a resume
b1ffecf6 test(ai): #500 U4b acceptance (d) — lockstep, a real two-rank Local gang driven through the full production run()
bb6a3c12 test(ai): #500 U4b acceptance (b) — gather exactness, a real two-rank Local gang vs a W=1 reference
c6f8c213 feat(db,ai,docs): #500 U4b S8 — rename [worker] world_size to local_ranks
9001f0cb feat(ai): #500 U4b — route every collective call through RankContext; expose the canonical-vars digest
5a342266 feat(ai): #500 U4b — per-rank resume, rank-0-only checkpoint writes, and a soft schema-version fallback
49b6d9b3 feat(ai): #500 U4b — no-gather-story arms refused at world > 1; Classification carries logits
989dc3ff fix(ai): #500 U4b — canonical_reduce restores presence, not just the sum (design pressure round)
2eb0ba1a feat(ai): #500 U4b — RankContext, the gather rule, lockstep divergence flags, and the step-bounded epoch loop
f4d84145 feat(ai): #500 U4b — canonical_reduce: zero-filled, canonical-order all_reduce_sum at the optimizer-step boundary
11c26ddc feat(ai): #500 U4b — PartitionSpec::for_gang/counts_for_step, the production multi-rank constructor and the gather-count primitive
```


## 2a. U4b tail (landed as one commit; original `1261041b`) — streamed arm at `world > 1`; LoRA init/dropout seed split; per-rank dropout positions exercised

The implementer's contract, folded by the lead after checking: the `world > 1` refusal for a streamed source is gone and both epoch loops are `for step in 0..train_batches_per_epoch`; `EpochSource::next_chunk` returns the chunk, never a sentinel; `LoraLinear::{new_seeded, new_with_base_seeded}` split the init seed from the dropout seed with the original constructors as `seed, seed` wrappers; the coordinator's files were not touched. Its two Uncovered items (the worker's construction call sites wired to the `_for_rank` builders with a per-rank dropout seed; the `LoraBuildConfig`/`EncoderAdapters` seed split) belong to the same `run_spec` topology site the coordinator body owns and are folded into §6's scope.

Base: `feat/500-wave3c` @ `9b0dcb57`. Branch `unit/u4bt`, tip `1261041b`.
Worktree `(the unit worktree)`
(the harness-assigned worktree; already detached at the named base tip — no
second worktree was created). Target dir
`(scratchpad)/targets/u4bt`,
`RUSTC_WRAPPER=sccache`, one `--features test-hooks` set for `jammi-ai` for
the whole session.

#### 1. Scope shipped

**Item 1 — the streamed arm at `world > 1`** (`crates/jammi-ai/src/fine_tune/stream.rs`,
`crates/jammi-ai/src/fine_tune/trainer.rs`):

- `stream.rs::run_pump` computes a `step_bound: Option<usize>` once at pump
  start: `Some(partition::batches_per_epoch(window.len(), spec.world(),
  spec.batch()))` for `Slice::PerRank`, `None` for `Slice::All` (unchanged
  contract — validation is never per-rank-partitioned, so an empty range
  there stays unconditionally terminal). A new helper,
  `walk_past_empty_steps`, replaces the two "if range.is_empty() { emit
  terminal; return; }" call sites: for `PerRank`, an empty range short of
  the bound is sent as a REAL, non-terminal chunk (the zero-row-rank case,
  DESIGN.md §4) and the loop advances to the next step; the bound itself
  is the only terminal condition, and at the bound nothing further is sent
  (the last real, possibly-empty step already went out). `Slice::All`
  keeps sending its one terminal chunk exactly as before.
- `EpochSource::next_chunk(step) -> Result<TextChunk>` (was
  `Result<Option<TextChunk>>`): always the real chunk for `step <
  train_batches_per_epoch` — the caller's own bounded loop is now the ONLY
  end-of-epoch signal, never a sentinel this method returns. A `None` from
  the underlying stream this early is a typed internal-invariant error.
- `trainer.rs::run`'s Streamed arm: deleted the `world > 1` refusal (kept
  the F6 whole-set-arm invariant check, unrelated to this item); the loop
  changed from `loop { let Some(chunk) = epoch_source.next_chunk(step)?
  else { break }; ... step += 1 }` to `for step in 0..
  train_batches_per_epoch { let chunk = epoch_source.next_chunk(step)?;
  ... }`, mirroring the Resident arm's own step-bounded loop exactly. Also
  switched `compute_loss` → `compute_loss_gathered(call, &batch, &counts)`
  (with `counts = partition_spec.counts_for_step(train_count, step)`) since
  `world > 1` is now reachable here and DESIGN.md §4 requires every rank to
  compute the identical GLOBAL loss over the gathered batch — at `world ==
  1` `all_gather` is `Noop`'s identity, so this is byte-identical to the
  prior `compute_loss` call there (no new W=1 oracle needed; the existing
  `fine_tune::` suite pins it).
- `partition.rs`: added `PartitionSpec::batch()` (the per-rank batch size
  the spec was built with) — `run_pump` needs it to derive its own
  `step_bound`; no other change to `partition.rs`.

**Deviation from a literal reading of the brief**: the brief's citation for
where the refusal/fix lives (`trainer.rs` ~1065–1080) matched; no deviation
here beyond the `compute_loss` → `compute_loss_gathered` switch, which the
brief's own text ("removing U4b's typed refusal by fixing the stream's
epoch bound") implies but does not spell out — I judged it required, not
optional: without it, two ranks reading a Streamed source at `world > 1`
would each compute a DIFFERENT local loss, never lockstep, defeating the
whole point of lifting the refusal.

**Item 2 — the LoRA init seed split from the dropout seed**
(`crates/jammi-lora/src/lora_linear.rs`, `crates/jammi-lora/src/seeded.rs`,
`crates/jammi-ai/src/fine_tune/lora.rs`):

- `LoraLinear::new`/`new_with_base` (jammi-lora) are now thin
  `init_seed == dropout_seed` wrappers over two NEW functions,
  `new_seeded`/`new_with_base_seeded`, which take `init_seed`/`dropout_seed`
  independently: the A/B weight-init draw (`seed_for_param`) is keyed by
  `init_seed`, the dropout-mask draw (`DropoutMasks::new`) by
  `dropout_seed`. Every EXISTING call site in the workspace (worker.rs,
  jammi-encoders' `lora_site.rs`, every test) is unaffected byte-for-byte —
  none of them were touched, since `new`/`new_with_base` keep their exact
  original signatures.
- `DropoutMasks` (seeded.rs) gains `seed(&self) -> u64` (`pub(crate)`,
  returns `run_seed`); `LoraLinear` gains `dropout_run_seed(&self) ->
  Option<u64>` (public), exposing the per-layer Philox SEED for the oracle
  — `dropout_position()` (the forward COUNT) is deliberately
  rank-invariant (both ranks of a gang take the same number of training
  forwards per step in the common, no-zero-row-rank case), so it cannot by
  itself distinguish two ranks whose dropout SEED differs; `dropout_run_seed`
  can.
- `jammi-ai/src/fine_tune/lora.rs`: `build_head_layer` (the shared
  per-layer builder) is now `build_head_layer_for_rank(.., dropout_seed:
  u64)`, calling `LoraLinear::new_seeded` with `init_seed = config.seed`,
  `dropout_seed` from the caller. Each of the four public builders
  (`build_classification_head`, `build_distribution_head`, `build_ner_head`,
  `build_projection_head`) is now a thin `dropout_seed == config.seed`
  wrapper over a new `_for_rank` sibling (`build_classification_head_for_rank`,
  etc.) that takes an explicit `dropout_seed: u64` and threads it to every
  layer it builds. The four original functions keep their EXACT original
  signatures — every existing call site (worker.rs, ~30 trainer.rs tests,
  jammi-bench) is unaffected byte-for-byte.

**Deviation, with the reason** (named, not silently absorbed): the brief
cites `fine_tune/target.rs` as "the LoRA build path" to wire the split
into. `target.rs` has no construction call sites at all (verified: it is
the `TrainingTarget` enum + its dispatch methods only); the ACTUAL
production LoRA construction happens in `worker.rs` (excluded from this
unit — the concurrent implementer's file) for BOTH the `ProjectionHead`
path (`lora::build_*_head`) and the `EncoderAdapters` path
(`jammi_lora::LoraBuildConfig`, consumed by `jammi-encoders::lora_site.rs`).
I could not wire the ACTUAL per-rank dropout seed into worker.rs's
construction call without touching a file I was told not to touch. What I
shipped instead: the split itself (jammi-lora's `_seeded` constructors),
threaded all the way through jammi-ai's OWN `lora.rs` builders via
`_for_rank` siblings that DO NOT exist yet at worker.rs's call sites — the
same shape U4b's own already-landed contract chose for
`RankContext::dropout_seed`'s wiring (Deviation 2 there: "the free function
and its own property... are built and tested in isolation; wiring it is
filed as a follow-up") and for `run_spec`'s N-rank fan-out (Deviation 1
there). This unit closes ONE MORE LAYER of that same, previously-declared
gap (the LoRA-layer seed split itself, plus `gang_determinism_oracle`'s own
in-tree wiring via `run_gang_rank`, which builds models directly and
bypasses worker.rs entirely — exactly how every other gang oracle in this
file already proves the mechanism). Wiring `worker.rs`'s actual
construction call sites (`build_classification_head_for_rank` etc. with a
REAL per-rank `dropout_seed` derived at claim time) is still a named,
deferred follow-up, together with `run_spec`'s N-rank fan-out itself (the
same follow-up U4b already named — deriving a per-rank dropout seed is
part of "per-rank model load per rank... per-rank dropout seed derivation"
in that unit's own Deviation 1 text).

The `EncoderAdapters`/`LoraBuildConfig` path is UNCOVERED by this unit for
the same reason: `LoraBuildConfig.seed` (jammi-lora's `config.rs`) is a
single field consumed by `jammi-encoders::lora_site.rs`; splitting it would
require editing every `LoraBuildConfig { .. }` construction site across
jammi-encoders/jammi-bench AND worker.rs's own construction at
`worker.rs:6154` — the same worker.rs-exclusion blocks it. Not attempted;
see Uncovered §2 below.

**Item 3 — the per-rank dropout-position gather/restore, exercised**
(`crates/jammi-ai/src/fine_tune/trainer.rs`, `gang_determinism_oracle`):

- `gang_config` (dropout always `0.0`) replaced by `gang_config_with_dropout(epochs,
  lora_dropout)`; `drive_gang`/`run_gang_rank` gained `lora_dropout: f64` and
  `train_rows: usize` parameters (every existing call site updated to pass
  `0.0`/`8`, unchanged behavior). `run_gang_rank` now builds the model via
  `build_projection_head_for_rank(HIDDEN, &config, &varmap, &vb,
  rank_ctx.dropout_seed(config.seed))` instead of the old rank-blind
  `build_projection_head`.
- Two new tests: `w2_twice_is_byte_identical_with_dropout` (the SAME
  property as `w2_twice_is_byte_identical`, at `lora_dropout = 0.3`, `8`
  rows — equal partition, no zero-row-rank) and
  `resume_after_a_kill_matches_an_uninterrupted_run_with_dropout` (the SAME
  property as its sibling, at `lora_dropout = 0.3`, but `6` rows — chosen
  so `W=2, B=2` leaves rank 1 with a real zero-row trailing step every
  epoch, i.e. the two ranks' own `dropout_position()` COUNTS genuinely
  diverge over the run). The `8`-row equal-partition fixture would make a
  cross-rank position swap numerically vacuous (both ranks' counts always
  coincide), which is exactly why the resume-with-dropout row uses `6`
  instead — stated in that test's own doc.

#### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| A `Streamed` source at `world > 1`, whose last global step leaves one rank empty, completes through the REAL production `run()` (no hang, no rank-count skew) | `fine_tune::trainer::gang_lockstep_oracle::a_zero_row_rank_via_a_streamed_source_the_gang_completes` (a real 2-rank `Local` gang over a materialised table) | EXECUTED (applied, run to resolution, reverted): restored the pre-unit "row_count()==0 ends the epoch" check in the Streamed consumer loop; rank 1 exits one step early, rank 0 deadlocks — `all_gather: timed out after 120s waiting for every peer to arrive`, 120.23s wall clock vs ~0.2s healthy |
| At `W=1`, the Streamed arm's per-step body is byte-unchanged (Noop gather is the identity) | The whole pre-existing `fine_tune::` suite (347 tests) — no new W=1 oracle needed | N/A — regression-only claim; any perturbation would have reported here first |
| `PartitionSpec::batch()` returns the value the spec was built with | Covered by construction (`run_pump`'s own `step_bound` derivation, exercised by every `Slice::PerRank` streamed test, `p1`–`p_t2` in `tests/it/training_set_stream.rs`, 16 tests, all green) | N/A — a pure accessor; the zero-row-rank oracle above is the load-bearing proof of its CONSUMER |
| Two ranks built with the SAME `init_seed` (`config.seed`) and DIFFERENT `dropout_seed` report byte-identical `lora_a`/`lora_b` before any forward, and two DIFFERENT `dropout_run_seed()` values, each equal to the seed it was constructed with | `fine_tune::trainer::lora_seed_split_oracle::dropout_seed_split_leaves_init_identical_and_dropout_distinct` | EXECUTED (applied, run, reverted): threaded `dropout_seed` into BOTH `LoraLinear::new_seeded`'s `init_seed` and `dropout_seed` arguments inside `build_head_layer_for_rank` — pre-forward `lora_a` bytes diverge between rank 0/1; reverted |
| `LoraLinear::new`/`new_with_base`'s existing `init_seed == dropout_seed` behavior is unchanged | The whole pre-existing `jammi-lora`/`jammi-ai` test suites (unchanged pass) | N/A — regression-only; `new`/`new_with_base`'s bodies are pure delegation to the `_seeded` siblings with `seed, seed` |
| A 2-rank gang at `lora_dropout > 0`: two independent from-scratch runs produce byte-identical rank-0 final weights | `fine_tune::trainer::gang_determinism_oracle::w2_twice_is_byte_identical_with_dropout` | EXECUTED (applied, run, reverted): changed the second call's epoch count `2` → `3` — confirmed a real byte divergence (`lora_a`/`lora_b` differ between the two "independent" runs), then reverted; `git status --short` empty afterwards |
| A 2-rank gang at `lora_dropout > 0`, with a fixture whose two ranks' own forward COUNTS genuinely diverge (a real zero-row-rank step every epoch): a gang killed after 1 epoch and resumed reproduces an uninterrupted run byte-for-byte | `fine_tune::trainer::gang_determinism_oracle::resume_after_a_kill_matches_an_uninterrupted_run_with_dropout` | EXECUTED (applied, run, reverted): changed `restore_from_checkpoint`'s `state.dropout_positions.get(&self.rank_ctx.rank())` to `state.dropout_positions.get(&0u32)` (always rank 0's entry) — confirmed a real byte divergence between the uninterrupted and resumed legs' final weights (`lora_a`/`lora_b` differ), then reverted |
| `check_citations.py`: every `PATH:LINE` citation still resolves after the jammi-lora insertion shifted lines | `python3 ci/scripts/perf/check_citations.py` | Covered directly: ran once BEFORE re-anchoring (6 stale citations reported, all at the exact two lines the insertion moved), fixed, ran again (0 stale, 1036 files scanned) |

#### 3. Uncovered

1. **`LoraBuildConfig`/`EncoderAdapters`'s dropout seed is NOT split.**
   `jammi_encoders::lora_site.rs` still derives one seed for both A/B init
   and dropout from `LoraBuildConfig.seed`, itself constructed from a single
   field at every call site (including `worker.rs:6154`, excluded from this
   unit). A real gang training an `EncoderAdapters` target still shares one
   dropout seed across ranks today. Named, not silently absorbed — same
   shape as U4b's own Deviation 2 (the ProjectionHead path this unit closes
   was the SAME kind of gap).
2. **`worker.rs`'s actual construction call sites are not wired to the
   `_for_rank` builders.** `build_classification_head`/`build_distribution_head`/
   `build_ner_head`/`build_projection_head`'s ORIGINAL (rank-blind)
   signatures are still what worker.rs calls; the per-rank dropout seed
   this unit built is only exercised in-tree, by `gang_determinism_oracle`'s
   own direct construction (bypassing worker.rs), exactly as U4b's own
   `run_spec`-fan-out deviation was left. A real production multi-rank
   `Local` gang (once `run_spec` itself spawns N ranks — still not built,
   per U4b's own named Deviation 1) would need this wiring too.
3. **The Streamed arm's `world > 1` fix is proven only for `Slice::PerRank`
   with a SINGLE trailing empty step per rank per epoch.** `walk_past_empty_
   steps`'s `while` loop is written to handle a RUN of consecutive empty
   steps (defensively), but I did not construct a fixture where more than
   one leading/trailing step is empty for the same rank in one epoch — per
   `partition.rs`'s own module doc, this can only happen when `world *
   batch > train_count` even at step 0 (a single-step epoch), which the
   `a_zero_row_rank_via_a_streamed_source_the_gang_completes` fixture does
   not construct (it has 3 real steps, one trailing empty). The multi-step
   generalization is exercised by construction (the loop's own shape) but
   not by a dedicated fixture.
4. **Live-Postgres arm** — not applicable; no `jammi-db` test was added or
   touched by this unit.

#### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | 0 | |
| `cargo clippy -p jammi-lora --all-targets -- -D warnings` | 0 | run in addition to the brief's list since this unit touches jammi-lora (COMMON.md: "ONLY the crates you touched") |
| `cargo test -p jammi-ai --features test-hooks --lib fine_tune::` | 0 | 347 passed, 0 failed (includes the gang oracles and every new test this unit adds) |
| `cargo test -p jammi-ai --features test-hooks --lib stream::` | 0 | 0 tests matched (unchanged from base — `stream.rs` has no unit tests of its own; every oracle for it lives in `tests/it`) |
| `cargo test -p jammi-ai --features test-hooks --test it -- streamed` | 0 | 1 test matched (the literal substring `streamed`); see the next row for the real filter |
| `cargo test -p jammi-ai --features test-hooks --test it -- training_set_stream` | 0 | 16 passed, 0 failed — the real filter for U2c's per-rank stream it-rows |
| `cargo test -p jammi-ai --features test-hooks --test it -- training_set::` | 0 | 13 passed, 0 failed — the eager-path it-rows in the same file family |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1036 file(s) scanned, all PATH:LINE citations resolve` (after re-anchoring the 6 citations the jammi-lora insertion shifted) |

`RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ai --no-deps` was NOT run —
outside this unit's trimmed gate list (COMMON.md: the lead runs the full
merge path once on the consolidated tip).

#### 5. Commits

```
1261041b feat(ai,lora): #500 U4b tail — streamed arm at world>1, LoRA init/dropout seed split, per-rank dropout position exercised
```

#### Files touched (repo-relative)

- `crates/jammi-ai/src/fine_tune/stream.rs` (item 1)
- `crates/jammi-ai/src/fine_tune/trainer.rs` (items 1, 2, 3)
- `crates/jammi-ai/src/fine_tune/partition.rs` (item 1, `PartitionSpec::batch()`)
- `crates/jammi-ai/src/fine_tune/lora.rs` (item 2)
- `crates/jammi-lora/src/lora_linear.rs` (item 2 — outside `jammi-ai`; see
  the deviation note above for why: `LoraLinear::new`/`new_with_base`, the
  actual seed-split site, live here, not in `jammi-ai`)
- `crates/jammi-lora/src/seeded.rs` (item 2, `DropoutMasks::seed`)
- `ci/scripts/perf/finetune_ab.sh`, `crates/jammi-kernels/src/admission.rs`,
  `crates/jammi-kernels/src/ops/mod.rs` (mechanical: re-anchored 6 stale
  `PATH:LINE` citations the jammi-lora insertion shifted — no behavior
  change)

#### Scope amendments (for the lead)

- Touched `crates/jammi-lora/**` (a crate not named in my dispatch's
  "crate owned" framing) because `LoraLinear::new`/`new_with_base` — the
  ACTUAL definition site of the seed the brief describes as
  "`crates/jammi-ai/src/fine_tune/lora.rs`" — live there, not in jammi-ai;
  jammi-ai's `lora.rs` is a thin wrapper crate over it. No conflict with the
  concurrent implementer (their exclusions are `worker.rs`/`spec.rs`/
  `crates/jammi-server/src/grpc/**`, none of which touch jammi-lora).
- Touched `crates/jammi-kernels/src/{admission.rs,ops/mod.rs}` and
  `ci/scripts/perf/finetune_ab.sh` ONLY to re-anchor `PATH:LINE` citations
  that my jammi-lora insertion shifted (mechanical, no behavior change) —
  `check_citations.py` failed until this was done.


## 2b. Integration (landed as one commit; original `18e0db19`) — the witness through `RankContext`; the round inbox in the hold loop; the trybuild oracle as `compile_fail` doctests; doc parity and citations

The implementer's contract, folded by the lead after checking: one production minting site (`worker.rs`, `BlockingCall::spawn_blocking`), no `cfg` added to the trainer, `trybuild` and `tests/ui/**` gone, `FrameOutcome` exhaustive over the oneof with `unreachable!` arms. The consolidated tree compiles and every unit's filtered oracle is green on it; the second seam the brief did not name (`config/tests.rs` still naming `world_size`) is closed. The empty-frame arm U5a-2 labelled UNCOVERED is executed here.

Worktree `wt-int`, branch `unit/int`, base `feat/500-wave3c` @ `00b1ad38`, tip `18e0db19`.
Every path is repo-relative; every oracle named was executed on the tip (§4 has the commands, exit
codes and counts); every mutation in §2 was applied to the committed tip, run through ONE filtered
oracle, and reverted (`git checkout -- <file>`; `git status --short` empty afterwards). Target dir
`…/scratchpad/targets/int`, `RUSTC_WRAPPER=sccache`, one `--features` set per crate for the whole
session (`jammi-ai`/`jammi-server`: `test-hooks`; `jammi-db`: `live-postgres-tests,test-hooks`).

#### 1. Scope shipped

The consolidated base did not compile at TWO seams (the brief named one): (a) `RankContext`'s five
wrappers called the `Collective` verbs without the witness — the five `E0061`s
`cargo clippy -p jammi-ai` reports at `trainer.rs:369–389` on the base; (b) `jammi-db`'s lib-test
target: the db slice's `config/tests.rs::distributed_config_loads_independently_of_worker` and
`DistributedConfig`'s two intra-doc links still named `WorkerConfig::world_size`, which U4b S8
renamed to `local_ranks` (`E0609` at `config/tests.rs:818`; the links would fail the docs lane).
Both are closed; nothing else was touched beyond the four items.

**Item 1 — the witness through `RankContext`** (`crates/jammi-ai/src/fine_tune/trainer.rs`,
`optimizer.rs`, `worker.rs`):
- `RankContext::{all_gather, all_reduce_sum, all_reduce_max_flags, broadcast, barrier}` take
  `call: &BlockingCall` first and forward it (doc rewritten to the shipped shape: the wrappers
  forward the ONE witness `run` receives).
- `TrainingLoop::run(&mut self, call: &BlockingCall, source)` is the ONE place the trainer receives
  it; `call` is threaded to every path that reaches a wrapper: `compute_loss_gathered(call, ..)`
  (12 gathers), `process_batch_loss(call, ..)` (the lockstep flag reduce and the window-boundary
  `canonical_reduce`), the trailing-window `canonical_reduce`, `save_resume_checkpoint(call, ..)` →
  `capture_resume_bundle(call, ..)` → `gather_dropout_positions(call)`.
  `optimizer::canonical_reduce(call, rank_ctx, vars, grads)`. No other production path reaches a
  wrapper (`save_epoch_checkpoint`, `restore_from_checkpoint`, `evaluate*`, `run_gradcache_epoch`
  make no collective call — verified by grep over every `rank_ctx.` / verb call site).
- Production mints at exactly one place: `worker.rs` `train_fine_tune`'s
  `BlockingCall::spawn_blocking(move |call| …run_fine_tune_blocking(&call, params)…)`;
  `run_fine_tune_blocking(call, params)` passes it to `training_loop.run(call, source)`.
- Tests mint only through the three minting sites: `collective::tests::witness` (a scoped
  `spawn_scoped` helper) is now `pub(crate)` and re-exported under `cfg(test)` from
  `collective/mod.rs`; `run_text_loop`/`run_gang_rank` take `call`; the 12 `run_text_loop` tests
  wrap their bodies in `witness` (the thread-local counter reads stay on the run's own thread);
  the gang oracles spawn each rank with `BlockingCall::spawn_thread`; the `it` tests
  (`fine_tune.rs` ×3, `ft_correctness_sweep.rs` ×3, `ft_determinism.rs`) and jammi-bench's
  `finetune_run::{run, run_impl}` (+ `main.rs`, 6 test call sites) use
  `BlockingCall::spawn_blocking`; `persist` (resume test helper) takes `&mut TrainingLoop` and
  captures under `witness` (a `&TrainingLoop` is not `Send`: the loop holds a `Cell`).
  `tests/gpu_capability/gang_nccl.rs` (CUDA-gated) threads the witness by inspection (§3).
- No second minting site, the token is not `Send`, no test-only constructor.

**Item 2 — the round inbox in the hold loop** (`crates/jammi-server/src/grpc/gang.rs`):
- `run_rank` builds `(inbox, member) = gang_rounds::member_link(events.clone())` BEFORE `Admitted`
  is queued (fails only outside a runtime context → `Status::internal`, unreachable in a handler);
  `HeldSession { inbox: RoundInbox, member: Option<MemberLink> }`.
- `dispatch_round_frame(frame) -> FrameOutcome`: a round frame (`RoundInbox::is_round_frame`) is
  `deliver`ed and the session stays held (`FrameOutcome::Held`); a refused delivery (the link is
  gone) ends the session with a `FailedPrecondition` trailer; the refusal match is EXHAUSTIVE on
  `RankControl`'s oneof (`None` → the `InvalidArgument` empty-frame violation as before;
  `Assign`/`Cancel` and the four round arms are `unreachable!` with the reason — a new oneof arm is
  a compile error here). `on_control_frame` is async and returns `FrameOutcome`; the hold loop's
  inbound arm breaks only on `End`. A transport error is reported to the inbox
  (`RoundInbox::fail`) before the silent end. Still exactly four `select!` arms.
- The session keeps the link for its whole life and drops `inbox` then `member` last at the end of
  `hold` (the link's forwarder holds the last clone of the event sender — this is what closes the
  response stream; every pre-existing "stream closes after `Aborted`" oracle still passes).
- The seat U5b-1b-iii takes: `member: Option<MemberLink>` on the held session plus the `test-hooks`
  seam `GangServer::take_member_links() -> UnboundedReceiver<MemberLink>` (`offer_member_link` hands
  each admitted session's link to the registered taker; `None` when taken). Deviation from
  "the session keeps it": under `test-hooks` with a taker registered, the taker owns it — the ONLY
  way an oracle can build the member's `Peer` over the REAL handler's link (the type is not
  `Clone`); a plain build has no taker and the session always keeps it.
- Fixtures: `gang_service.rs` gained `world_two_ready` (the admitted world-2 row without the stream
  opened), `open_rank_at(addr, first)`, and `pub(crate)` on the fixtures the new oracles reuse.

**Item 3 — the trybuild oracle as doctests** (`crates/jammi-ai/src/fine_tune/collective/mod.rs`):
four doctests on `BlockingCall`'s docs — the passing `spawn_blocking` control;
`compile_fail,E0277` (the witness carried into a `tokio::spawn`ed future); `compile_fail,E0061`
(no witness to pass); `compile_fail,E0624` (the private `mint`). Removed:
`tests/it/blocking_call.rs` (+ `mod blocking_call;`), `tests/ui/**`, `tests/ui_pass/**`, the
`trybuild` dev-dependency and the workspace `trybuild = "1"` (`Cargo.lock` pruned: `trybuild`,
`toml`, `toml_writer`, `serde_spanned`, `target-tuple`). Deviation from UNITS.md's "(g) … a compile
error (`trybuild`)": the brief's own instruction; UNITS.md's plan row is the lead's to date.

**Item 4 — docs and citations**: `docs/maintainer/MAINTAINER-GUIDE.md` §2.8a (the inbound arm's
delivery/refusal/closed-link ends, the link built at admission, `RoundInbox::fail`), §2.8c (the
witness threading and its one production minting site; the doctest oracle replacing trybuild; the
real-handler round oracle). `check_doc_parity.py` green (unchanged). Eleven stale `PATH:LINE`
citations re-anchored by identifier: nine were stale at the consolidated base
(`MAINTAINER-GUIDE.md` → `stream.rs` ×6, `data.rs`, `recompute.rs`, `trainer.rs`;
`gguf.rs` → `config/mod.rs`), two moved under my `finetune_run.rs` insertions
(`jammi-encoders/src/test_support.rs`).

#### 2. Properties

| Property (quantified) | Executed oracle (path::name; lane) | Executed mutation that reds it (first red line) |
|---|---|---|
| P21 (compile-time): for every `Collective` verb, a call whose witness would reach a runtime worker thread does not compile — a `BlockingCall` cannot cross into a `tokio::spawn`ed future, no witness exists to pass on a worker thread, no fourth minting site exists — while the same verb from `spawn_blocking` compiles and runs | `cargo test -p jammi-ai --features test-hooks --doc -- BlockingCall`: `BlockingCall (line 169)` control ok; `(line 189) compile fail` (E0277) ok; `(line 205)` (E0061) ok; `(line 217)` (E0624) ok — 4 passed | M1: `_thread_bound: PhantomData<*const ()>` → `PhantomData<()>` (the witness made `Send`) → RED: `BlockingCall (line 189) - compile fail ... FAILED` / `Test compiled successfully, but it's marked compile_fail` (the other three stay red-as-expected: they do not depend on `Send`) |
| P21 at the trainer's production entry: `run_fine_tune_blocking` (hence `TrainingLoop::run` and every wrapper) is reachable only under the witness minted at `worker.rs`'s `BlockingCall::spawn_blocking`; handing the run to a runtime worker is a compile error | the compiler on the tip: `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` exit 0 | M2: inside the production closure, `run_fine_tune_blocking(&call, params)` → `Handle::current().block_on(tokio::spawn(async move { run_fine_tune_blocking(&call, params) })).unwrap()` → RED: `error: future cannot be sent between threads safely --> crates/jammi-ai/src/fine_tune/worker.rs:2951:31` (`could not compile jammi-ai (lib)`) |
| Every path that reaches a wrapper receives the ONE witness `run` was given — no wrapper is reachable without it | the compiler (the five `E0061`s on the base become zero on the tip; every call site enumerated in §1 compiles) + `cargo test -p jammi-ai --features test-hooks --lib fine_tune::` 343 passed (the U4b gang oracles — lockstep, gather exactness, determinism/resume — drive real 2-rank `Local` gangs through the full production `run` under `spawn_thread` witnesses) | Covered by construction: removing `call` from any one wrapper call is the base's `E0061` (the executed base clippy run, `int-scratch/clippy-ai-0.log`, is that red) |
| W=1 is byte-unchanged and the trainer is never `cfg`-forked | the WHOLE pre-existing `fine_tune::` suite (343, up from U4b's 319 by U5b-1b-i's own additions) and `--test it -- peer_gang host_admission jobs_shutdown` (32) pass unchanged in substance; no `cfg` was added to the trainer (`git diff 00b1ad38..HEAD -- crates/jammi-ai/src/fine_tune/trainer.rs` adds no `#[cfg`) | Regression-only claim (as U4b's contract states it): a perturbation of the W=1 window reports here first |
| Item 2: through the REAL `GangServer::run_rank` on a loopback listener, an admitted `world_size == 2` member session delivers one round's frames to its inbox, and the `Peer` member built over the session's own `MemberLink` folds every tensor verb (gather, reduce, flags; chunked under a 1 KiB cap) byte-for-byte to `Local`; the job row is untouched | `crates/jammi-server/tests/it/gang_rounds.rs::a_round_through_the_real_run_rank_handler_reaches_the_member_link_and_equals_local` (`test-hooks`; admitted via `dial_member` against the real handler, the link taken through `take_member_links`) | M4: `dispatch_round_frame` refuses round frames with an `InvalidArgument` trailer (the base tree's dispatch) instead of delivering → RED: `panicked at crates/jammi-server/tests/it/gang_rounds.rs:220:10: gather: FineTune("all_gather: round 0: the stream ended while waiting for the coordinator's result — nothing applied")` |
| Item 2: a non-round, non-session frame (`control: None`, the only such value of the oneof) on an admitted stream still ends the session with the `InvalidArgument` protocol-violation trailer, the row untouched | `crates/jammi-server/tests/it/gang_service.rs::run_rank_empty_frame_on_an_admitted_stream_is_invalid_argument` (the empty frame IS sendable by the generated client — U5a-2's "cannot be built without a raw codec" was wrong; this row executes the arm U5a-2 labelled UNCOVERED) | M3: the `is_round_frame` guard skipped (`|| true`: every frame routed to `deliver`) → RED: `panicked at crates/jammi-server/tests/it/gang_service.rs:2482:10: an empty frame is a status, never an event: Some(RankEvent { event: Some(Aborted(Aborted { reason: NoBody })) })` (the session stayed held and parked) |
| Item 2: a round frame after the member's link closed (its owner dropped it) ends the session with a `FailedPrecondition` trailer naming the closed link — never buffered, never a bare close; row untouched | `gang_rounds.rs::a_round_frame_after_the_member_link_closed_ends_the_session_with_a_trailer` (`test-hooks`; world 1) | M5: a refused delivery treated as `Held` (`deliver(..).await \|\| true`) → RED: `panicked at crates/jammi-server/tests/it/gang_service.rs:550:29: no stream event within 5s` |
| Item 2: every pre-existing hold-loop end (Cancel, second Assign, drain, refuted/unavailable/store-unavailable, park, supersession, busy slot) is unchanged, and the stream still closes after its terminal event with the link now dropped last | `cargo test -p jammi-server --features test-hooks --test it -- gang api_freeze tenant_isolation_oracle`: 59 passed (56 pre-existing + 3 new) | unchanged oracles (the "stream must close after Aborted" assertion in `expect_aborted` is the one a link dropped too early or too late would red) |
| Item 4: every `PATH:LINE` citation resolves at HEAD; every documented enumeration matches its enum | `python3 ci/scripts/perf/check_citations.py` → `check-citations: 1036 file(s) scanned, all PATH:LINE citations resolve (…; 2 exempt as non-ancestor legacy evidence)`; `python3 ci/scripts/check_doc_parity.py` → `doc-parity: all bindings in parity.` | mechanical gates; on the base the citation gate is RED with 12 stale anchors (executed: the first run on the tree, before re-anchoring) |
| Second seam: `jammi-db`'s lib-test target compiles and the `[distributed]`/`[worker]` independence test holds under the renamed knob | `cargo clippy -p jammi-db --all-targets --features test-hooks -- -D warnings` exit 0; `cargo test -p jammi-db --features live-postgres-tests,test-hooks --lib config::tests::distributed_config_loads_independently_of_worker` 1 passed | the base's `E0609` at `config/tests.rs:818` is the executed red (`int-scratch/clippy-db.log`) |

#### 3. Uncovered

- **`tests/gpu_capability/gang_nccl.rs`** (`required-features = ["live-gpu-tests"]`, the verb calls
  inside `#[cfg(feature = "cuda")]`): threaded by inspection (`assert_gang_checks(call, ..)` under
  `spawn_thread`/`spawn_scoped`; the post-abort `all_reduce_max_flags` under a scoped witness) —
  not compiled here (no CUDA toolchain on this host); CI's gated-surface clippy step is the oracle.
  At the base this file did not compile under `cuda` either (it called the verbs without the
  witness — U5b-1b-i's own Uncovered).
- **`RoundInbox::fail` on a transport error**: wired in the hold loop's `Some(Err(status))` arm; no
  hermetic oracle injects a transport error into an admitted inbound stream (the existing suite has
  none either). Stated, not executed.
- **Backpressure while delivering**: `deliver` awaits inbox room (64 frames) inside the inbound arm's
  handler, so the drain/re-verify/park arms are not polled during that wait. Reachable only once a
  consumer exists that stalls (U5b-1b-iii's body; the watchdog is U5b-2's). Not exercised.
- **`member_link` failing at admission** (`Status::internal`): needs a handler outside a runtime
  context; unreachable under tonic, not executed.
- **jammi-bench's own test targets** (`crates/jammi-bench` `run`/`run_impl` callers): compiled by the
  workspace clippy of the merge path, not run here (outside the brief's gate list; the edits are the
  `spawn_blocking` → `BlockingCall::spawn_blocking` shape at 7 sites and the two signatures).
- **Trainer-level (b) with `Peer` substituted for `Local`** (U5b-1b-i's deferred row): not built —
  it needs the coordinator body's assignment (§6), outside these four items.
- **The E0277 code on the `tokio::spawn` doctest**: rustdoc reports the doctest green under the
  `compile_fail,E0277` fence and M1 turns exactly that block green-compiling (red test); clippy on
  the same shape at the production site prints the diagnostic without a bracketed code (M2). I did
  not separately capture rustdoc's raw stderr for the block.

#### 4. Gates (all on the tip `18e0db19`; logs in `<scratchpad>/int-scratch/`)

| Command | Exit | Result |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | clean |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | 0 | 0 warnings (`clippy-ai-3.log`) |
| `cargo clippy -p jammi-server --all-targets --features test-hooks -- -D warnings` | 0 | 0 warnings (`clippy-server-1.log`) |
| `cargo clippy -p jammi-db --all-targets --features test-hooks -- -D warnings` | 0 | 0 warnings (`clippy-db-2.log`; the base: `E0609`, `clippy-db.log`) |
| `cargo test -p jammi-ai --features test-hooks --lib fine_tune::` | 0 | 343 passed; 0 failed |
| `cargo test -p jammi-ai --features test-hooks --lib collective` | 0 | 69 passed; 0 failed |
| `cargo test -p jammi-ai --features test-hooks --test it -- peer_gang host_admission jobs_shutdown` | 0 | 32 passed; 0 failed |
| `cargo test -p jammi-ai --features test-hooks --doc -- BlockingCall` | 0 | 4 passed (3 compile-fail + control); 0 failed |
| `cargo test -p jammi-server --features test-hooks --test it -- gang api_freeze tenant_isolation_oracle` | 0 | 59 passed; 0 failed |
| `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1:54329/jammi_test cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- assembly_outcome migrations gang_rank_admission --test-threads=1` | 0 | 73 passed (22 `::postgres` arms executed); 0 failed |
| `cargo test -p jammi-db --features live-postgres-tests,test-hooks --lib config::tests::distributed_config_loads_independently_of_worker` | 0 | 1 passed |
| `python3 ci/scripts/check_doc_parity.py` | 0 | `doc-parity: all bindings in parity.` |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1036 file(s) scanned, all PATH:LINE citations resolve (HEAD for living files, each artifact's own recorded git_sha for committed evidence reachable from HEAD; 2 exempt as non-ancestor legacy evidence).` |
| `git status --short` after every gate and every mutation revert | — | 0 entries |

Mutation runs (each `exit=101`, reverted): `mut-M1.log` … `mut-M5.log`; runner `mutations.sh`.

#### 5. Commits (`git log --oneline 00b1ad38..HEAD`; the worktree's local `main` is not the base)

```
18e0db19 feat(ai,server,db,docs): #500 int — the witness through RankContext; the round inbox in the hold loop; the trybuild oracle as compile_fail doctests; doc parity and citations on the consolidated tree
```

31 files, +1224/−732 (`trainer.rs` +927/− mostly the 12 wrapped test bodies re-indented by rustfmt).


## 3. U5b-1b-i (landed as one commit on this branch; original `7f653e20`)

The implementer's contract, folded by the lead after checking: the proto diff deletes no declaration (comment lines only) and `api_freeze_baseline.txt` is byte-identical; all five trait verbs take `&BlockingCall`; both peer-listener services and the client carry the decode cap; the trybuild scratch was deleted. The pressure round's blocks 5 and 6 are closed by the bound round index / fatal commit-phase fault and by the hermetic loopback target. Consolidation notes: the one cherry-pick conflict was a `worker.enabled` citation both units re-anchored, resolved against the combined `runtime.rs`; the trybuild oracle (247 s, 10 GB per cold run) is to be replaced by a `compile_fail` doctest carrying the same error codes at the consolidation build, per the implementer's own deviation 6.

Branch `unit/u5b1bi` in `(the unit worktree)`, base `856ec8dd` (PR #580 tip), tip `7f653e2058eb50e57eb5dbe95b8e2a40becd7100`.
Every path below is repo-relative to that worktree; every test named was executed on this branch
(the Gates section has the commands, exit codes and counts) and every mutation listed was
executed against the committed tip and reverted (`git checkout -- <file>`), its first red line
quoted from the run.

### 1. Scope shipped

**ai-core (`crates/jammi-ai`)**
- `src/fine_tune/collective/mod.rs` — `Descriptor`, `TensorSignature` and the closed `Verb` enum
  lifted here (public), `Descriptor::agrees_with` = the derived equality; two NEW determinants:
  `round: u64` (the round index — `Local` stamps it under its rendezvous lock from the shared
  generation, `Peer` from its own counter, the wire carries it) and `agreement: Option<String>`
  (the caller-bound opaque digest; `Local::with_agreement` / `Peer::with_agreement` bind it per
  rank; U4b binds its canonical trainable-variable key-name digest; this unit carries and
  compares it, binds nothing). `BlockingCall`: the `!Send + !Sync` witness with a PRIVATE
  constructor and three minting sites (`spawn_blocking`, `spawn_thread`, `spawn_scoped`). The
  `Collective` trait's five verbs take `call: &BlockingCall` — ON THE TRAIT (binding change 3):
  the trainer holds `&dyn Collective` and never names `Peer`, so a witness on `Peer`'s inherent
  methods would be invisible at the one call site that matters; `Noop`, `Local`, `Nccl` take and
  ignore it. `pub mod peer` + re-exports (`Peer`, `MemberLink`, `CoordinatorLink`, `LinkFault`,
  `RankReadFault`).
- `src/fine_tune/collective/local.rs` — uses the lifted types; `Contribution::kind() -> Verb`;
  `Local::with_agreement`; `Shared::exchange` stamps `descriptor.round = generation` under the
  lock; every verb takes the witness; the white-box tests moved to the minting sites; the
  per-field sweep gained `round` and `agreement` arms (`round: _`, `agreement: _` destructure —
  a field added without an arm fails to compile); new `a_round_mismatch_disagrees`,
  `an_agreement_mismatch_disagrees_including_bound_versus_unbound`.
- `src/fine_tune/collective/noop.rs`, `nccl.rs` — the witness parameter (ignored). `nccl.rs`
  compiles only under `cuda`; see Uncovered.
- `src/fine_tune/collective/peer.rs` (NEW, ~2000 lines) — `Peer` (coordinator = rank 0 over
  `Vec<CoordinatorLink>`, member = rank `1..world` over one `MemberLink`), `with_timeout`,
  `with_agreement`, `device`; the two-phase round (`Peer::round` → `coordinate` / `participate`);
  the rank-ordered fold on the coordinator's device with `Local`'s operation sequence; the Arrow
  IPC codec (`encode_tensors` / `decode_tensors`: f32 `Float32`, f16 `Float16`, bf16 `UInt16`
  bits; one length-prefixed IPC stream per tensor); chunking under `max_message_bytes − 64` and
  reassembly against `reassembly_bound` (elements × size + 4 KiB per tensor); the wire
  conversions with K2 range checks (`descriptor_to_wire` / `descriptor_from_wire`,
  `verb_from_wire`, `dtype_to_wire` / `dtype_from_wire`); the links (`MemberLink::from_channels`,
  `CoordinatorLink::from_channels`, `CoordinatorLink::over_client(channel, assign,
  max_message_bytes)` — opens `RunRank`, sends `Assign`, requires `Admitted`, caps the CLIENT's
  inbound decode at the same value); the rank read path (`RankReadFault::StoreUnavailable` +
  `abort_reason()` → `AbortReason::StoreUnavailable`, `verify_leaves` over any ranged reader,
  `verify_partition_leaves` over a `JammiObjectStore` one `get_range` per leaf).
- `src/fine_tune/collective/peer_tests.rs` (NEW) — the in-process oracles: a channel-wired gang
  with per-direction TAPS (forward / swallow / cut / close-inbound-then-forward / replace) so a
  fault is injected at an exact frame; the `Local` twin for byte comparison; a recording
  `object_store::ObjectStore` wrapper for the bounded-read oracle.
- `src/fine_tune/collective/tests.rs` — existing tests under the witness (`run_gang` mints via
  `BlockingCall::spawn_thread`; scoped spawns via `spawn_scoped`; test-thread bodies via a
  scoped `witness` helper). All 47 pass unchanged in substance.
- `tests/it/peer_gang.rs` (NEW) — two tasks over a REAL loopback tonic `RunRank` stream (a
  hold-loop-shaped test `GangService`): fold parity at f32/f16/bf16 with chunking, the client
  recv cap at `n−1`/`n`/`n+1` by `encoded_len()`, the wire deadline, the corrupted-leaf abort.
- `tests/it/blocking_call.rs` + `tests/ui/*.rs` (+ `.stderr`) + `tests/ui_pass/*.rs` (NEW) — the
  trybuild oracle.
- `Cargo.toml` — `half` and `tokio-stream` move under/into `local` (bf16/f16 element type and
  `ReceiverStream` are the `Peer` codec's and links'); dev-deps `trybuild` (workspace) and
  `object_store` (the recording wrapper). `half` leaves the `cuda` list (it is unconditional now).

**wire-server**
- `crates/jammi-wire/proto/jammi/v1/gang.proto` — ADDITIVE ONLY: `RankControl` gains
  `round_result = 3`, `round_chunk = 4`, `round_commit = 5`, `round_fault = 6`; `RankEvent` gains
  `round_contribution = 4`, `round_chunk = 5`, `round_ack = 6`, `round_fault = 7`; new messages
  `RoundDescriptor { round, verb: RoundVerb (closed), world, optional root, Counts counts,
  repeated TensorSignature, optional agreement }`, `Counts`, `TensorSignature { dims,
  ElementType }`, `RoundPayload { round, descriptor, chunk_count, flags }`, `RoundChunk`,
  `RoundAck`, `RoundCommit`, `RoundFault { round, detail }`; enums `RoundVerb`, `ElementType`.
  Nothing renamed or removed (`git diff 856ec8dd -- crates/jammi-wire/proto/jammi/v1/gang.proto`
  shows additions only). `api_freeze_baseline.txt` is byte-unchanged and
  `api_freeze::wire_surface_equals_the_frozen_baseline` passes (it decodes only PACKAGE/RPC).
- `crates/jammi-server/src/runtime.rs` — `OssServer::bind`: BOTH peer-listener services carry
  `.max_decoding_message_size(max_message_bytes)` from `self.session.inner_config().server.limits`
  (per-service setter, binding change 5).
- `crates/jammi-server/src/limits.rs` — N5 rustdoc restated to quantify over every listener
  (public chain, `peer_bind`'s two services) and the coordinator's client as the third site.
- `crates/jammi-server/src/grpc/gang_rounds.rs` (NEW; binding change 7) — the seam for U5a-2's
  hold loop: `RoundInbox { is_round_frame, deliver, fail }`, `member_link(events) ->
  (RoundInbox, MemberLink)` over the session's own `Sender<Result<RankEvent, Status>>`,
  `dial_member(&PeerAddr, Assign, max_message_bytes) -> CoordinatorLink`. `GangServer::run_rank`
  is NOT touched; the `select!` wiring is the lead's at consolidation.
- `crates/jammi-server/tests/it/gang_rounds.rs` (NEW) — (d) on every listener and both
  peer-listener services at `n−1`/`n`/`n+1` encoded bytes; one real round through
  `RoundInbox::deliver` + `dial_member` vs `Local`.
- `crates/jammi-server/Cargo.toml` — dev-dep `candle-core` (the seam test's tensors; already
  linked through `jammi-ai`).

**db** — `crates/jammi-db/src/storage/object_store_handle.rs`: `JammiObjectStore::get_range`
(one ranged read; the read path's primitive). No catalog/schema change.

**docs** — `docs/maintainer/MAINTAINER-GUIDE.md` § 2.8c (new: the witness, the descriptor, the
wire, the reduce, the two-phase round, links and the server seam, the decode cap on every
listener, the rank's read path) + one sentence in § 2.8a; a stale `worker.enabled` citation at
guide line 498 re-anchored (my `runtime.rs` insertion moved it: `runtime.rs:2068` →
`runtime.rs:2086`); `docs/guide/src/operability.md` (`max_message_bytes` row: every listener,
`n` decodes / `n+1` refused), `docs/guide/src/configuration.md` (the key's comment).

#### Deviations from UNITS.md / the brief (each with the reason and the code)

1. **No change to `tests/distributed/{main.rs,harness.rs}`** (binding change 2). That target is
   `required-features = ["live-distributed-tests"]` (`crates/jammi-ai/Cargo.toml`, the
   `[[test]] name = "distributed"` block) and its harness needs Postgres + MinIO; nothing in this
   unit is fleet-dependent. The two-process brief is met as TWO TASKS over a real loopback tonic
   listener in the hermetic `it` target (`crates/jammi-ai/tests/it/peer_gang.rs`), which runs in
   `cargo test -p jammi-ai --features test-hooks`. No test name was added to `distributed.yml`.
2. **The wire dtype enum is `ElementType`, not `DType`.** prost strips only an enum-name-shaped
   prefix (`D_TYPE_`), so `DType { DTYPE_F32 }` generated `DType::DtypeF32`; `ElementType {
   ELEMENT_TYPE_F32 }` generates `ElementType::F32`. The Rust-side match over the WIRE enum is
   literally exhaustive (`Ok(Unspecified) | Err(_)` refused). The match over candle's `DType` is
   as exhaustive as Rust allows: `candle_core::DType` is `#[non_exhaustive]`
   (`~/.cargo/registry/src/*/candle-core-0.11.0/src/dtype.rs`), so a `_` arm is MANDATORY; every
   known variant is listed and the wildcard refuses a later-added one
   (`peer.rs::dtype_to_wire`).
3. **`Descriptor::agrees_with` returns `bool`** (was `Result<(), ()>` in local.rs): the type is
   now public and clippy's `result_unit_err` refuses a public `Result<_, ()>`; the callers read
   `if !a.agrees_with(&b)`.
4. **The commit point** (binding change 1, as shipped): a fault before the coordinator observes
   the last ACK leaves no rank applied; a fault DURING the commit fan-out is fatal on every rank
   (coordinator applies nothing, faults every member, refuses every later round). The one
   residual state: a member the commit did reach applied the round and returned `Ok` before the
   fan-out failed — physically unretractable with a one-way commit — and its NEXT contribution
   is refused by the coordinator's fault. So the shipped guarantee is "no rank ever CONTINUES
   past a round every rank did not apply"; the lead's oracle (a fault between the fan-out to
   rank 1 and rank 2 faults every rank and no next contribution is accepted) is executed
   (`a_fault_during_the_commit_fan_out_is_fatal_on_every_rank_and_no_next_contribution_is_accepted`).
   Stated in `peer.rs`'s module doc, `gang.proto`'s round-protocol comment and § 2.8c.
5. **`MemberLink::over_stream` does not exist.** U5a-2's hold loop owns the inbound `Streaming` in
   its `select!`; consuming it whole would be the wrong shape. The seam is `gang_rounds::member_link`
   + `RoundInbox::deliver`, exercised by `crates/jammi-server/tests/it/gang_rounds.rs`.
6. **The trybuild cost** (acceptance (g), as briefed): trybuild builds the ui cases in its own
   target subdir (`<target>/tests/trybuild`), rebuilding the dependency tree — 247 s and 10 GB on
   this host for the first run (incremental after). A `compile_fail` doctest would assert the same
   error codes at zero extra build; I shipped trybuild because the brief and binding change 3 name
   it. The lead may prefer the doctest form; the property is the same.
7. **The trainer's call sites are untouched**: at `856ec8dd` no trainer code calls a `Collective`
   verb (`grep -rn "\.all_gather(\|\.all_reduce_sum(\|\.barrier(" crates/jammi-ai/src` outside
   `collective/` is empty), so the trait change breaks nothing in-tree; U4b's helper threads the
   witness at consolidation (binding change 3).

### 2. Properties

| # | Property (quantified) | Executed oracle (path::name, lane) | Executed mutation → red |
|---|---|---|---|
| P1 (a) | For every verb, every rank and each of f32/f16/bf16, `Peer`'s result bits equal `Local`'s over the same inputs (world 3, unequal counts incl. a zero-row rank, mixed-dtype reduce, root=1 broadcast) | `fine_tune::collective::peer_tests::peer_fold_over_the_wire_equals_local_fold_byte_for_byte_at_f32_f16_bf16` (lib, test-hooks); over the real wire with chunking: `it::peer_gang::peer_fold_over_a_loopback_run_rank_stream_equals_local_at_f32_f16_bf16_with_chunking` | M1: the reduce fold skips rank 0's own contribution → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:264:5` — assertion `left == right` failed: every rank's result bits over the wire must equal the in-process fold's |
| P1' | Two gangs over the same inputs produce identical bits | `peer_tests::peer_collectives_are_deterministic_across_two_gangs` | (covered by M1's class; a nondeterministic fold order would fail P1 against `Local`) |
| P2 (b) | At W=2, for a contrastive stream (`[rows,8]` + `[rows]` gathers, f32+bf16 grads, flags, scaler broadcast) and a regression stream (`[rows,1]` + `[rows]`), over 2 steps with unequal counts, every rank's bytes equal `Local`'s | `peer_tests::peer_w2_fold_matches_local_w2_on_regression_and_contrastive_streams` — fold-level; the trainer-level row is DEFERRED to consolidation: the lead runs U4b's W=2 trainer parity oracle with `Peer` substituted for `Local` (its name is U4b's to report) | M2: the gather concatenates slices in reverse rank order → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:361:9` — assertion `left == right` failed: contrastive fixture: Peer-W2 bytes must equal Local-W2 on every rank at every step |
| P3 (c) | Every round wait on every rank (coordinator for a contribution; member for the result; member for the commit) expires at the gang deadline with an error naming the round and what it waited for | `peer_tests::every_round_wait_on_every_rank_expires_at_the_gang_deadline_naming_the_round`; over the wire: `it::peer_gang::a_silent_member_over_the_wire_expires_the_coordinators_wait_at_the_deadline_naming_the_round` | M3: the timeout message drops `round {k}` → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:400:5` — the coordinator's wait must expire naming the round: Fine-tune error: barrier: timed out after 400ms waiting for rank 2's contribution — nothing applied |
| P4 (c) | A member's stream ending between the coordinator's publish and the last ACK leaves NO rank applied for that round, every rank's fault names the round, and no rank waits out the deadline | `peer_tests::a_disconnect_between_publish_and_the_last_ack_leaves_no_rank_applied_and_names_the_round` | M4: the member applies the held result without waiting for the commit → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:489:18` — round 0 must not apply on any rank: () |
| P5 (binding 1) | A fault during the commit fan-out (rank 2's stream gone between the fan-out to rank 1 and to rank 2) is fatal on every rank: the coordinator applies nothing, rank 2 applies nothing, rank 1 (reached) applied round 0, and NO rank's next contribution is accepted, promptly | `peer_tests::a_fault_during_the_commit_fan_out_is_fatal_on_every_rank_and_no_next_contribution_is_accepted` | M5: the coordinator ignores a failed commit send and applies → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:560:10` — the coordinator applies nothing: 4 |
| P6 (e) | A `root` disagreement, a `counts` disagreement, an `agreement` disagreement (bound≠bound, bound vs unbound), an unknown wire `verb` (99) and a stale wire `round` are each a typed refusal on EVERY rank naming BOTH descriptors | `peer_tests::a_root_disagreement_…`, `…a_counts_disagreement_…`, `…an_agreement_slot_disagreement_…`, `…an_unknown_wire_verb_…`, `…a_contribution_from_a_stale_round_…` (all `…_naming_both_sides…`) | M6: `agrees_with` compares `verb` only → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:595:9` — counts: rank 0 was not refused symmetrically: Fine-tune error: all_gather: round 0: rank 1's contribution: tensor 0: 4 elements on the wire where the agreed shape [1, 2] holds 2; M7: `Peer` never signs its bound agreement → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:667:18` — the ranks bound different agreements: (); M8: an unknown wire verb defaults to `Barrier` → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:702:14` — an unknown wire verb is refused: (); M9: the wire `round` is ignored on decode → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:737:14` — a stale round never folds: () |
| P7 | Every field of `Descriptor` is a determinant of agreement (`agrees_with` ≡ derived `==` over a per-field sweep incl. `round`, `agreement`; a field added without a sweep arm fails to compile) | `fine_tune::collective::local::descriptor_tests::{agrees_with_matches_derived_equality_over_a_per_field_mutation_sweep, a_round_mismatch_disagrees, an_agreement_mismatch_disagrees_including_bound_versus_unbound}` | M6 (above) also reds the sweep |
| P8 | A rank's own argument refusal (a counts vector that cannot describe the gang) faults its peers with a `RoundFault` before returning, so the peer is refused in < deadline/4 | `peer_tests::a_domain_refusal_on_one_rank_faults_its_peers_before_the_deadline` | M20: the member's prepare-failure fault is not sent → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:767:5` — the coordinator waited 4.004808958s for a fault the member had already raised |
| P9 | A dtype outside f32/f16/bf16 is refused at the seam on the calling rank (naming the dtype) before any descriptor exists, and its peers are faulted | `peer_tests::a_dtype_outside_f32_f16_bf16_is_refused_at_the_seam_and_faults_the_peers` | M19: f64 is carried as `F32` on the wire → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:793:9` — rank 0: unexpected: Fine-tune error: all_reduce_sum: round 0: timed out after 5s waiting for rank 1's contribution — nothing applied |
| P10 | After any fault, every verb on every rank refuses promptly (< deadline/2) quoting the fault | `peer_tests::every_verb_after_a_fault_refuses_promptly_on_every_rank_quoting_the_fault` | M21: the fault is not recorded → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:835:13` — rank 0: unexpected: Fine-tune error: all_gather: round 1: faulted by a peer — broadcast: round 0: faulted by a peer — broadcast: round 0: rank 0's round descriptor is Descriptor { round: 0, verb: Broadcast, world: 2, root: Some(0), counts: None, tensors: [TensorSignature { dims: [1, 1], dtype: F32 }], agreement: None } but rank 1's is Descriptor { round: 0, verb: Broadcast, world: 2, root: Some(1), counts: None, tensors: [TensorSignature { dims: [1, 1], dtype: F32 }], agreement: None } — the ranks disagree about what this round computes, so no rank may be handed a result |
| P11 | A tensor larger than `max_message_bytes` travels as ≥ ⌈size/cap⌉ chunks each ≤ `cap − 64` bytes and reassembles byte-exact (= `Local`) | `peer_tests::a_tensor_larger_than_the_message_cap_travels_in_chunks_under_the_cap_and_reassembles_exact`; over the real listener (4 KiB cap, 10 KiB gathers): `it::peer_gang::peer_fold_over_a_loopback_…_with_chunking` | M14: no chunking (one chunk) → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:879:5` — a 16 KiB tensor under a 1 KiB cap needs many chunks: [17228] |
| P12 (K2) | A payload announcing more bytes than its agreed descriptor implies is refused at the first chunk past the bound, before buffering more | `peer_tests::a_payload_past_the_bound_its_descriptor_implies_is_refused_before_buffering` | M15: the bound check is dropped → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:932:5` — unexpected: Fine-tune error: all_reduce_sum: round 0: timed out after 5s waiting for chunk 64 of rank 1's contribution — nothing applied |
| P13 | Every result lands on the rank's device; in `all_gather` only the rank's OWN slot carries a gradient (grad = own rows, never ×world) | `peer_tests::results_land_on_the_ranks_device_and_only_the_own_gather_slot_is_attached` | M16: the own slot is spliced in detached → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:965:9` — assertion `left == right` failed: rank 0's gradient must be its OWN slot's, never scaled by the world size |
| P14 | A member's `Aborted{reason}` mid-round faults the coordinator naming the reason and the round | `peer_tests::a_member_that_aborts_its_session_faults_the_coordinator_naming_the_reason` | M23: the reason is not named → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:998:5` — unexpected: Fine-tune error: barrier: round 0: the member ended its session while waiting for rank 1's contribution — nothing applied |
| P15 (f) | `verify_leaves` reads exactly one leaf's range per leaf in inventory order, never wider, stops at the first bad leaf; a corrupted row group is `RankReadFault::StoreUnavailable` naming that leaf, `abort_reason() == StoreUnavailable`, "member-scoped"; a read error and a short read are the same class | `peer_tests::verify_leaves_reads_exactly_one_leaf_range_at_a_time_and_names_a_corrupted_leaf_member_scoped` | M10: the digest is never compared → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:1114:6` — a corrupted leaf is caught: () |
| P16 (f, bounded memory) | `verify_partition_leaves` over a `file://` `JammiObjectStore` issues one BOUNDED `get_opts` range per leaf, exactly the leaf's range, and never an unbounded (whole-object) read; a corrupted leaf is named | `peer_tests::verify_partition_leaves_over_a_file_store_reads_one_bounded_range_per_leaf_and_names_a_corrupted_leaf` (recording `ObjectStore` wrapper) | M11: whole-object `get_bytes` then slice → RED: `panicked at crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:1248:5` — assertion `left == right` failed: every read is one leaf's bounded range |
| P17 (f, end to end) | Over a real stream, a member that finds a corrupted leaf before its first collective ends the session `Aborted(StoreUnavailable)`; the coordinator's first round faults naming it, having folded nothing | `it::peer_gang::a_corrupted_leaf_is_caught_on_the_member_before_any_collective_and_ends_the_session_member_scoped` | covered by M10 (the member would verify clean and run the round → `expect_err` on the coordinator fails) — not separately executed |
| P18 (d, listeners) | On the `peer_bind` listener (`GangService` AND `PeerService`) and on the public listener, a message of exactly `n = max_message_bytes` encoded bytes decodes, `n−1` decodes, `n+1` is `OUT_OF_RANGE` naming `the limit is: n bytes` (RED at base for the peer listener: no setter, tonic's 4 MiB default) | `jammi-server it::gang_rounds::a_frame_of_exactly_max_message_bytes_decodes_on_every_listener_and_one_more_byte_is_refused_naming_the_configured_cap` | M12: the `GangService` setter dropped → RED: `panicked at crates/jammi-server/tests/it/gang_rounds.rs:102:13` — assertion `left == right` failed: GangService: code: 'The system is not in a state required for the operation's execution', message: "gang admission refused"; M24: the `PeerService` setter dropped → RED: `panicked at crates/jammi-server/tests/it/gang_rounds.rs:123:13` — assertion `left == right` failed: PeerService: code: 'Client specified an invalid argument', message: "storage_precision is unspecified" |
| P19 (d, client) | The coordinator's client decodes a member frame of exactly `n` and `n−1` encoded bytes and refuses `n+1` naming the configured cap (never tonic's default) | `it::peer_gang::the_client_recv_cap_is_the_configured_cap_at_n_minus_1_n_and_n_plus_1_encoded_bytes` | M13: the client setter dropped → RED: `panicked at crates/jammi-ai/tests/it/peer_gang.rs:359:13` — a 2049-byte frame over a 2048-byte cap must be refused naming the configured cap: Fine-tune error: barrier: round 0: faulted by a peer — xxx…(the 2043-byte padding detail; the frame DECODED under the dropped cap) |
| P20 (seam) | A round frame delivered through `RoundInbox::deliver` reaches the member's `Peer` and `dial_member` builds the coordinator's link: one real round of gather/reduce/flags over a hold-loop-shaped handler equals `Local` | `jammi-server it::gang_rounds::a_round_delivered_through_the_inbox_and_dialed_through_dial_member_equals_local` | M17: `is_round_frame` excludes `RoundCommit` → RED: `panicked at crates/jammi-server/tests/it/gang_rounds.rs:222:38` — reduce: FineTune("all_reduce_sum: round 1: timed out after 20s waiting for rank 1's contribution — nothing applied") |
| P21 (g) | A `Collective` verb reached from a runtime worker thread does not compile (a `BlockingCall` cannot cross into a `tokio::spawn`ed future; no witness exists to pass; the constructor is private), while the same verb from `spawn_blocking` compiles | `it::blocking_call::a_collective_verb_from_a_runtime_worker_thread_does_not_compile` (trybuild: `tests/ui/{verb_from_a_runtime_worker_thread, verb_without_a_witness, mint_outside_a_spawn_site}.rs` red with pinned `.stderr`; `tests/ui_pass/verb_from_spawn_blocking.rs` green) | M18: the witness made `Send` (`PhantomData<()>`) → RED: `panicked at /Users/vijaychakilam/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/trybuild-1.0.121/src/run.rs:103:13` — 1 of 4 tests failed |
| P22 (api_freeze) | The frozen wire surface is unchanged: `api_freeze_baseline.txt` byte-identical to base, the live descriptor equals it | `jammi-server it::api_freeze::wire_surface_equals_the_frozen_baseline`; `git diff 856ec8dd -- crates/jammi-server/tests/it/api_freeze_baseline.txt` is empty | not a mutation of mine; the oracle is the freeze guard's own (an added rpc reds it) |
| P23 | The existing `Local`/`Noop` properties hold under the witness | `fine_tune::collective::tests` (47) + `local::{rendezvous_state_tests, constructor_verb_tests, descriptor_tests}` | unchanged oracles |

### 3. Uncovered

- **`Nccl` under the witness.** `crates/jammi-ai/src/fine_tune/collective/nccl.rs` is
  `#[cfg(feature = "cuda")]`; this host (macOS, no CUDA toolchain) cannot compile it. The edit is
  the signature-only witness parameter on the five verbs plus `barrier` forwarding `call` to
  `all_reduce_max_flags`; UNCOVERED here — the CI gated-surface clippy step compiles it.
- **`Local`'s stamp of `Descriptor::round`** (`Shared::exchange` sets `descriptor.round =
  generation`): no oracle observes the stamped value on `Local` beyond the field being a
  determinant (P7). The executed stale-round oracle on `Local` remains its generation check
  (`a_round_holding_a_superseded_contribution_is_refused_by_the_rank_completing_it`).
- **Real multi-host / GPU residency.** Every Peer oracle runs on `Device::Cpu` (loopback or
  channels); a `to_device` failure on real hardware after a round published is the same class
  `local.rs` labels UNCOVERED (U4b's pod leg / U7b-A2b live).
- **Real listener admission.** `GangServer::run_rank` at this base admits nothing (ends
  `Unimplemented`), so no oracle runs `Peer` through the REAL peer listener's handler; the
  loopback handlers in `peer_gang.rs` / `gang_rounds.rs` mirror the hold loop's shape (admit,
  deliver round frames, events on the response stream). The hold-loop `select!` wiring is U5a-2's
  + the lead's at consolidation.
- **P17's own mutation** was not executed separately (its refutation is M10's class); **P1'** and
  **P22** likewise rest on the named oracles without a dedicated mutation.
- **Trainer-level (b)** is DEFERRED to consolidation as stated in P2.

### 4. Gates (trimmed set per the lead; every command run in this worktree with
`CARGO_TARGET_DIR=…/targets/u5b1bi`)

```
cargo fmt --all -- --check
  exit 0  
cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings
  exit 0      Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.41s
cargo clippy -p jammi-server --all-targets --features test-hooks -- -D warnings
  exit 0      Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.48s
cargo clippy -p jammi-db --all-targets --features test-hooks -- -D warnings
  exit 0      Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.17s
cargo clippy -p jammi-wire --all-targets -- -D warnings
  exit 0      Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.17s
cargo test -p jammi-ai --features test-hooks --lib collective   (tests.rs 47 + local white-box + peer_tests 23)
  exit 0  ok. 69 passed; 0 failed; 0 ignored; 0 measured; 704 filtered out; finished in 1.21s;
cargo test -p jammi-ai --features test-hooks --test it -- peer_gang blocking_call   (4 loopback + 1 trybuild)
  exit 0  ok. 5 passed; 0 failed; 0 ignored; 0 measured; 584 filtered out; finished in 21.13s;
cargo test -p jammi-server --features test-hooks --test it -- api_freeze tenant_isolation_oracle gang_service grpc_limits gang_rounds
  exit 0  ok. 35 passed; 0 failed; 0 ignored; 0 measured; 208 filtered out; finished in 4.63s;
python3 ci/scripts/perf/check_citations.py
  exit 0  check-citations: 1038 file(s) scanned, all PATH:LINE citations resolve (HEAD for living files, each artifact's own recorded git_sha for committed evidence reachable from HEAD; 2 exempt as non-ancestor legacy evidence).
python3 ci/scripts/check_no_consumer_names.py
  exit 0  no-consumer-names: OK — no governance-verb leak in the diff, no philosophy leak-smell token in the engine tree, allowlist clean.
python3 ci/scripts/check_dep_direction.py < <(cargo metadata --format-version 1 --locked)
  exit 0  OK: 675 crates in the OSS default-members normal-dep closure; all are local workspace paths or crates.io
bash ci/scripts/check_cookbook_one_way.sh
  exit 0  one-way guard: clean — no crates/** reference to cookbook/book/.
git status --short after every gate and mutation revert: 0 entries (0 = clean)
git diff 856ec8dd -- crates/jammi-server/tests/it/api_freeze_baseline.txt: 0 lines
gang.proto diff vs base: 162	5	crates/jammi-wire/proto/jammi/v1/gang.proto  (added/deleted lines: every deleted line is a rewritten comment line; no declaration removed — see the diff)
```

### 5. Commits (`git log --oneline 856ec8dd..HEAD`; the worktree's local `main` is not the base)

```
7f653e20 feat(collective): #500 U5b-1b-i — the Peer collective + round protocol
```


## 4. U5a-2 (landed as three commits on this branch; originals `8332a289`, `db82aef9`, `54bbe341`)

The implementer's contract, folded by the lead after opening the cited lines: the strict resolver takes an explicit tenant with the strict predicate and no admin arm; the holder flip is at the hold site and the prologue self-release test is intact; `dispatch_round_frame` is the one wiring site U5b-1b-i's round machinery joins; `in_flight`/`InFlightGuard` are gone. The pressure round's block 7 (no route to the job's tenant) is closed by the rebuilt tenant carrier; its advisory on the refusal-vs-abort observable split is closed by P1/P9.

Branch `unit/u5a2` in worktree `wt-u5a2`, cut from `856ec8dd`. Crates touched: `jammi-db`,
`jammi-ai`, `jammi-server` (+ one proto comment in `jammi-wire`, docs). Every claim below is
stated as it EXISTS on the branch tip; every path is repo-relative; test names are
`file::fn`.

### 1. Scope shipped

#### jammi-db (edits local to `get_job_for_rank`/`RankAdmissionRow` and the strict resolver)
- `crates/jammi-db/src/catalog/jobs_repo.rs::RankAdmissionRow` gains `tenant_id: Option<String>`
  (the row's `jobs.tenant_id` as RAW TEXT), `training_set_ref: Option<String>`,
  `training_set_location: Option<String>`; `Catalog::get_job_for_rank`'s one statement selects
  the three columns. **Deviation, stated:** the tenant is carried as text, never parsed in the row
  mapper — the U5a-1 round-2 excision deleted `tenant_id` precisely because its `parse::<TenantId>()`
  turned a found row into `Err` (indistinguishable from a driver fault, contradicting the method's
  own "infallible on content" contract, `docs/rigor/contracts/feat_500-C-U5a-1.md` Addendum 3).
  The handler parses it; an unparseable value is a row fact (`GangRefusalReason::TenantUndecodable`),
  exactly like an undecodable `world_size`. Pinned by
  `crates/jammi-db/tests/it/gang_rank_admission.rs::get_job_for_rank_carries_the_tenant_text_and_the_filled_pair`
  (sqlite + postgres), whose last arm plants `tenant_id = 'not-a-uuid'` by raw SQL and asserts
  `Ok(Some(row))` with the text verbatim.
- `crates/jammi-db/src/catalog/result_repo.rs::Catalog::get_result_table_for_tenant(name, tenant:
  Option<TenantId>)` — the strict predicate `tenant_id = $t OR (tenant_id IS NULL AND $t IS NULL)`,
  an explicit tenant argument, **no admin arm and no read of `current_tenant()`** (deviation from
  the excised verb, which kept the repo-wide admin-scope branch: a resolver whose tenant is an
  explicit argument must not be widened by ambient scope; the call site guard is kept on top of it).
  Tests: `crates/jammi-db/tests/it/result_tables.rs::get_result_table_for_tenant_never_matches_a_null_tenant_row_for_a_real_tenant`
  and `::get_result_table_for_tenant_resolves_only_the_owning_tenant`, both `test_case`-parameterized
  sqlite/postgres, each with a `with_admin_scope` arm proving ambient scope does not widen it.

#### jammi-ai (`crates/jammi-ai/src/fine_tune/worker.rs`, `crates/jammi-ai/src/session.rs`)
- `HostAdmission { phase: watch<WorkerPhase>, holder: watch<Holder>, registry: Arc<InstanceRegistration> }`
  owned by `InferenceSession` (`InferenceSession::host_admission()`); `instance_registration()`
  delegates to `registry`. `phase` moves out of `WorkerShared` (`WorkerShared::phase()` delegates;
  `set_phase_for_test` delegates). `HostAdmission::{begin_drain (Running→Draining, never regresses
  Releasing), begin_release, phase_receiver, holder, holder_receiver, probe_claim, job_running
  (pub(crate)), try_hold_rank, admit_rank, hold_for_test (test-hooks)}`.
- `Holder = Free | ClaimProbe | JobRun | Rank{job_id, attempt}`; `HolderBusy = ClaimProbe | JobRun |
  Rank{..}`; guards `ClaimGuard` (ClaimProbe/JobRun → Free on drop) and `RankHold` (→ Free on drop
  only if the cell still names this exact `(job_id, attempt)`).
- The claim loop (`JobWorker::run_until`): `probe_claim()` (Free→ClaimProbe) immediately before
  `claim_next`; a held slot skips the claim and sleeps the idle poll; `register_job_hold_or_release`
  flips ClaimProbe→JobRun once the lease hold is registered; the guard drops after
  `run_claimed_job_under` returns. **Deviation from README r27's sketch** ("`claim_next`'s `Some`
  arm flips ClaimProbe→JobRun"): the flip is at the HOLD SITE, so the claim→hold prologue stays a
  `ClaimProbe` and `release_and_stop`'s 2e still waits one heartbeat for the prologue's own
  self-release (zero net attempts, OPS D10) instead of aborting a claim whose lease would only fall
  to expiry. Pinned by `crates/jammi-ai/tests/it/host_admission.rs::the_claim_loop_moves_the_holder_free_probe_run_free`
  (the prologue park reads `ClaimProbe`) and the unchanged
  `crates/jammi-ai/tests/it/jobs_shutdown.rs::release_with_the_loop_paused_in_the_claim_to_hold_prologue_self_releases`.
- `WorkerShared::in_flight`, `InFlightGuard` DELETED (cut, no rebuild owed: the holder kind is the
  fact they approximated). `EmbeddedWorker` holds `admission`; `begin_drain` → `admission.begin_drain()`;
  `release_and_stop` 2a → `begin_release()`, 2e reads the holder KIND (`JobRun` aborts now; anything
  else waits one heartbeat). `JobWorker` holds `admission` (its own `Arc`, keeps no session alive) so
  `run()`/`run_claimed_job` build `WorkerShared` over it; a direct `run_claimed_job` and an inline
  `run_now` never touch the holder. `InferenceSession::release_job_leases` flips `begin_release()`
  first. `/metrics` `jammi_worker_jobs_in_flight` = `holder == JobRun` (`crates/jammi-server/src/routes/health.rs`).

#### jammi-server (`crates/jammi-server/src/grpc/gang.rs`, `runtime.rs`, tests, docs)
- `GangServer::new(session, lease, heartbeat)`; `runtime.rs` passes `LeaseIntervals::heartbeat()`
  and calls `session.host_admission().begin_drain()` on both DRAIN arms (a worker-less server has no
  `EmbeddedWorker::begin_drain` to flip the phase).
- `run_rank` order: bounded first frame → wire K2 → ambient admin scope → `get_job_for_rank`
  (`admission_catalog_fault` on `Err`) → status/claimant/attempt/lease → `WorldSizeFact` →
  `assign.world != row.world_size` → **world>1 conjunct on `row.world_size > 1`**: pair filled
  (`TrainingSetPairMissing`), tenant text parses (`TenantUndecodable`), then
  `resolve_training_set_identity(store, tenant, ref, location) -> Result<TrainingSetOutcome, JammiError>`
  (`Err` = the strict resolver's catalog read faulting → `admission_catalog_fault`; outcomes
  `Verified | AdminScopeRefused | Unresolved | NotReady | SidecarAbsent | DigestMismatch | StoreFault`
  → `GangRefusalReason::{AdminScope, TrainingSetUnresolved, TrainingSetNotReady,
  TrainingSetSidecarAbsent, TrainingSetDigestMismatch, TrainingSetStoreFault}`) → `fresh_instance`
  (`admission_catalog_fault` on `Err`) → **only then** `HostAdmission::admit_rank(job_id,
  row.attempts, heartbeat)` (`Err(HolderBusy)` → one fixed `Unavailable("gang admission: this
  host's job slot is busy")`) → `Admitted` (`mpsc` channel, `ReceiverStream`) → `tokio::spawn(HeldSession::hold())`.
  `GangRefusalReason::MultiHostUnsupported` is DELETED; the enum names sixteen determinants.
- `HeldSession::hold`: FOUR `select!` arms — inbound (`on_control_frame`: `Cancel` →
  `Aborted{Cancelled}`; second `Assign` → `InvalidArgument` trailer; everything else →
  `dispatch_round_frame`, the ONE wiring site for the round protocol, today only an empty frame →
  trailer; the client's half-close disables the arm, the session stays held; a transport error
  ends silently), phase watch (`!= Running` or sender gone → `Aborted{Drain}`),
  `interval_at(now + heartbeat, heartbeat)` tick → `reverify` (`ReverifyEnd::{Refuted, Unavailable,
  StoreUnavailable}` with `abort_reason()`, `scope()`, `counts_toward_assembly_attempts()`), park
  `sleep(lease)` → `Aborted{NoBody}`. One event (or one trailer) then the stream closes and the
  `RankHold` drops.
- **Observables split at admission (pressure-round fold 2):** pre-admission = `Err(Status)` (the
  one fixed `FailedPrecondition` for every I-GANG determinant; `Unavailable` for catalog fault /
  busy slot; `InvalidArgument` for wire K2); post-admission = in-stream `Admitted` then one
  `Aborted{reason}`, or a status trailer for the admitted-stream K2 violation. R3 is not weakened:
  a reason is named only on a session the caller was admitted to on the job's own coordinates.
- Docs (same commit set): `docs/maintainer/MAINTAINER-GUIDE.md` §2.8a rewritten (the lattice incl.
  the world>1 conjunct, `HostAdmission`, the hold loop, observables split, config);
  `docs/guide/src/security.md` I-GANG (derivation, non-disclosure incl. the post-admission reasons,
  admit-and-hold); `configuration.md`, `deploy-server.md`, `api-stability.md`, `gang.proto`
  comment, `api_freeze_baseline.txt` comment, DESIGN.md/README.md allowlist sentences; seven
  pre-existing `PATH:LINE` citations re-anchored after the line shifts.
- Tests: `crates/jammi-server/tests/it/gang_service.rs` (rewritten around the restored world>1
  fixtures), `gang_terminal_write_oracle.rs` (new), `gang_rank_admission_oracle.rs` (+ the strict
  resolver's caller oracle), `gang_admission_catalog_fault_oracle.rs` (floor 2→3),
  `tenant_isolation_oracle.rs` (derivation claim + assertion), `gang_training_spec_parity.rs`
  (sqlite + postgres arms as two named fns over one body — this crate has no `test_case` dev-dep;
  no dependency added), `health.rs` (gauge read off the holder).

#### Cuts and their rebuilds
- `WorkerShared::in_flight`/`InFlightGuard`: cut; rebuilt as the holder kind (above).
- `GangRefusalReason::MultiHostUnsupported` and the `world_size != 1` refusal: cut; rebuilt as the
  world>1 conjunct (rows in §2).
- `run_rank_refuses_world_gt_one_when_caller_world_matches_the_row`: deleted with its determinant;
  its role (the direction-(a) control) is now `run_rank_refuses_when_assign_world_mismatches_row_world_size`'s
  `TrainingSetPairMissing` control.
- The excised admin-scope arm of the strict verb: not restored (deviation above); its property is
  the db tests' `with_admin_scope` arms.

### 2. Properties (quantified) — executed oracle — executed mutation that reds it

Lanes: SRV = `cargo test -p jammi-server --features test-hooks --test it -- <filter>`; SRV-plain =
the same without `--features`; AI = `cargo test -p jammi-ai --features test-hooks --test it --`;
DB = `cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- --test-threads=1`
(sqlite arm always; postgres arm with `JAMMI_TEST_PG_URL`). Mutation ids M1–M13 are the executed
runs recorded in `u5a2-scratch/mutations.txt` (each: applied → one filtered test → reverted).

| # | Property (over every input / exit arm) | Executed oracle | Executed mutation → red (first line) |
|---|---|---|---|
| P1 | For every `RunRank` call, the admission decision (every I-GANG determinant incl. the world>1 conjunct and freshness) is complete BEFORE the holder is consulted; a refused call never touches the holder; `Admitted` is emitted only after a successful CAS (a busy slot is `Err(Status)`, never a stream). | SRV `gang_service::run_rank_refuses_unavailable_at_once_while_a_loop_job_runs` (busy slot → `Unavailable` + no reason recorded; an absent job under a busy slot → `FailedPrecondition`/`NotFound`) | M3: slot consulted before the row read → M3 (holder checked before the row read): `gang_service.rs:2192: assertion left == right failed` (an absent job answered `Unavailable`, not `FailedPrecondition`) |
| P2 | For every `row.world_size > 1` call and NEVER for `world_size == 1`: the pair must be filled, the row's tenant must parse, the strict resolver under the ROW's tenant must find a `ready` row, its sidecar must verify `artifact == training_set_ref`; every determinant refuses the one fixed `FailedPrecondition`; keyed on the row's decoded fact, never `assign.world`. | SRV `gang_service::run_rank_refuses_when_assign_world_mismatches_row_world_size` (both directions + controls: direction (a)'s control reaches `TrainingSetPairMissing`, direction (b)'s control is `Admitted`), `::run_rank_refuses_a_training_set_another_tenant_owns`, `::run_rank_refuses_a_null_tenant_training_set_for_a_tenant_bound_job`, `::run_rank_refuses_world_gt_one_when_the_sidecar_predates_the_leaf_inventory`, `::run_rank_world_two_own_tenant_training_set_is_admitted_held_and_parks_no_body` (the admitting control) | M1: world gate deleted → M1 (`if false && assign.world != world_size`): `gang_service.rs:1713: assertion left == right failed: must record WorldMismatch specifically, not the pair determinant`; M2: tenant forced `None` → M2 (`let tenant = None` before the resolver): `gang_service.rs:2019: the strict resolver must not resolve the NULL-tenant row (a later determinant refusing instead would mean it did)`; M7: leaf-less sidecar accepted → M7 (`Ok(None) => Verified`): `gang_service.rs:1583: every fixture this function builds must refuse` (the leaf-less sidecar admitted) |
| P3 | Non-disclosure: all sixteen determinants refuse with pairwise-identical `(code, message)` on the wire; the `test-hooks` seam distinguishes every one. | SRV-plain + SRV `gang_service::run_rank_refusal_is_non_disclosing_across_every_determinant`; SRV `::run_rank_last_refusal_reason_distinguishes_every_determinant`; `every_gang_refusal_reason` re-validated by an exhaustive match | (shape inherited from U5a-1, unchanged: any one arm interpolating a reason reds the pairwise oracle naming the pair; not re-executed this round — the sixteen-row scenario table is the executed novelty) |
| P4 | Tenant is derived from the job row, never accepted: no caller metadata is read; the strict resolver never resolves another tenant's or a NULL-tenant row for a real tenant, under ambient admin scope or not; the resolution site refuses ambient admin scope before the resolver runs. | SRV `gang_service::run_rank_never_reads_a_caller_supplied_tenant` (`jammi-session-id` metadata ignored → `Admitted`), `::resolution_site_refuses_under_admin_scope_before_the_strict_resolver_runs` (control `Verified` outside, `AdminScopeRefused` inside); DB `result_tables::get_result_table_for_tenant_never_matches_a_null_tenant_row_for_a_real_tenant::{sqlite,postgres}`, `::get_result_table_for_tenant_resolves_only_the_owning_tenant::{sqlite,postgres}`; the derivation claim in `tenant_isolation_oracle::gang_service_is_unimplemented_on_the_public_listener` | M11: strict predicate relaxed → M11 (predicate relaxed to `OR tenant_id IS NULL`): `result_tables.rs:550: the strict resolver must never match a NULL-tenant row for a real tenant` (sqlite arm); M2 (above) |
| P5 | The admission row carries the row's own tenant as text and the filled pair on both backends; the read is infallible on content (a garbage tenant is `Ok(Some)`). | DB `gang_rank_admission::get_job_for_rank_carries_the_tenant_text_and_the_filled_pair::{sqlite,postgres}` | M12: tenant parsed in the mapper → M12 (`parse::<TenantId>().expect(..)` in the mapper): `jobs_repo.rs:2087: panicked` (the garbage-tenant arm is no longer `Ok(Some)`) (sqlite arm) |
| P6 | Holder lattice (c2'): `Free` admits; `JobRun`/another `Rank` refuse at once; the same job at an equal attempt refuses, at a greater attempt supersedes in place and the elder's drop leaves the successor's hold; `ClaimProbe` is waited ≤ one bound then admits-if-freed or refuses; a hold's drop frees only its own cell. | AI `host_admission::{free_admits_a_rank_and_dropping_the_hold_frees_the_slot, a_job_run_or_another_rank_refuses_at_once, the_same_job_at_a_greater_attempt_takes_the_slot_and_the_elder_leaves_it, a_claim_probe_is_waited_out_then_admitted_if_freed_or_refused_at_the_bound}`; SRV `gang_service::{run_rank_refuses_unavailable_at_once_while_a_loop_job_runs, run_rank_waits_out_a_claim_probe_then_admits_if_freed_or_refuses_unavailable}` (test-hooks), `::a_held_rank_refuses_other_ranks_and_the_same_job_at_a_greater_attempt_takes_the_slot` (plain) | M9: `RankHold::drop` frees regardless of identity → M9 (`RankHold::drop` frees any `Rank`): `host_admission.rs:197: the superseded elder's drop must not free the successor's slot` |
| P7 | Exclusion (d2', OPS D6): the loop moves the holder `Free→ClaimProbe→JobRun→Free` around every claim (the flip at the hold site, the prologue a probe); an idle loop never calls `claim_next` while a rank is held and claims the moment it is freed; an inline `run_now` is outside the exclusion; a `JobRun`-holding peer refuses a rank, an idle peer admits. | AI `host_admission::{the_claim_loop_moves_the_holder_free_probe_run_free, an_idle_loop_never_claims_while_a_rank_is_held, an_inline_run_now_never_touches_the_holder}`; the gauge `health::gauges::in_flight_gauge_is_one_during_a_loop_claimed_job_and_zero_during_run_now`; the reshaped `jobs_shutdown` suite (20 rows) | M8: flip at `claim_next`'s `Some` arm → M8 (`job_running()` at `claim_next`'s `Some` arm): `host_admission.rs:306: the claim committed but the hold is not registered: still a probe` (read `JobRun`); M10: probe ignores a held slot → M10 (`probe_claim` overwrites a held slot): `host_admission.rs:365: claim_next must not be called while a rank is held` |
| P8 | OPS D10: RELEASE's abort decision reads the holder kind, never a count — a `Rank` beside an idle loop is never loop work (cooperative `Stopped`, hold untouched, phase `Releasing`); `JobRun` aborts now. | AI `host_admission::release_and_stop_beside_a_held_rank_exits_cooperatively_and_flips_the_phase`; `jobs_shutdown::release_with_the_loop_paused_in_the_claim_to_hold_prologue_self_releases` (the prologue still self-releases) | M13: a held `Rank` treated like `JobRun` → M13 (`holder == Holder::Free` → a held `Rank` aborts now): first executed against the ORIGINAL oracle (an idle loop) it stayed GREEN — the loop had exited at 2a before 2e ran, so the arms were indistinguishable; the oracle was rewritten to park the loop after reclaim, re-executed green, then the same mutation re-executed: `host_admission.rs:457: a held Rank is not loop work: 2e waits for the cooperative exit, never aborts` (`Aborted` ≠ `Stopped`) |
| P9 | The hold loop has exactly four arms and every end is one stream event: `Cancel` → `Cancelled`; a second `Assign` → `InvalidArgument` TRAILER (K2); phase leaving `Running` → `Drain` at once (both via the session cell and via the real server shutdown path, worker-less); park bound → `NoBody` after ≥ two ticks. | SRV `gang_service::{run_rank_cancel_on_an_admitted_stream_ends_cancelled, run_rank_second_assign_on_an_admitted_stream_is_invalid_argument, run_rank_held_session_ends_drain_when_the_host_drains, run_rank_held_session_ends_drain_on_server_shutdown, run_rank_every_i_gang_determinant_satisfied_is_admitted_held_and_parks_no_body}` | M4: drain arm never fires → M4 (phase arm replaced by `pending()`): `gang_service.rs:570: expected Aborted{Drain}, got Aborted { reason: NoBody }` (parks to `NoBody` instead) |
| P10 | Re-verification (i2'): the three ends are pairwise distinct on the wire, in scope and in the count rule; a row fact moving → `Refuted`; the catalog faulting → `Unavailable`; THIS host's store faulting (`Storage`/`Io`) → `StoreUnavailable`; the artifact's sidecar no longer verifying → `Refuted` (never `StoreUnavailable`). | SRV `gang_service::{run_rank_held_session_ends_refuted_when_the_row_no_longer_holds, run_rank_held_session_ends_unavailable_when_the_catalog_faults, run_rank_held_session_ends_store_unavailable_when_this_hosts_store_faults, run_rank_held_session_ends_refuted_when_the_sidecar_stops_verifying}`; lib `grpc::gang::tests::reverify_ends_are_pairwise_distinguishable_in_reason_scope_and_count` | M5: store fault classified `Refuted` → M5 (`StoreFault → ReverifyEnd::Refuted`): `gang_service.rs:570: expected Aborted{StoreUnavailable}, got Aborted { reason: Refuted }` |
| P11 | Terminal-write scope (g2'): the peer names no `jobs` writer (the set derived from `jobs_repo.rs` itself) and every end (cancel, K2 trailer, drain, refuted, unavailable, store-unavailable, park, supersession) leaves the job row byte-identical to its pre-admission snapshot. | SRV `gang_terminal_write_oracle::the_gang_handler_names_no_jobs_writer` (+ its two self-tests); the `row_facts` before/after equality in every hold-loop row above | M6: `fail_job(` named as code in `run_rank` → M6 (`let _ = stringify!(fail_job());` in `run_rank`): `gang_terminal_write_oracle.rs:283: gang.rs names the jobs writer fail_job( as code` |
| P12 | Every admission-time catalog read maps `Err` through `admission_catalog_fault` (three sites), never `map_engine_error`; `get_job_for_rank`'s and `get_result_table_for_tenant`'s only production callers are the gang handler. | SRV `gang_admission_catalog_fault_oracle::run_rank_never_calls_map_engine_error` (floor 3), `gang_rank_admission_oracle::{only_the_gang_run_rank_handler_calls_get_job_for_rank, only_the_gang_resolution_site_calls_get_result_table_for_tenant}` | (allowlist-both-directions shape: a fourth caller file reds the oracle naming it — the U5a-1 methodology; the raised floor was exercised by the `2 → 3` self-test fixture change) |
| P13 | Producer→consumer `world_size` parity holds on both backends. | SRV `gang_training_spec_parity::get_job_for_rank_world_size_matches_the_real_training_spec_producer_{sqlite,postgres}` (postgres executed with `JAMMI_TEST_PG_URL`) | (decode mutations inherited from U5a-1's db tests) |

### 3. Uncovered

- **UNCOVERED — a `StoreUnavailable` from a genuine I/O fault on a local file.** The executed
  member-scoped store fault is `JammiError::Storage(SchemeNotEnabled)` from `open_parquet` on an
  `s3://` URL this build compiles no driver for (hermetic, network-free). A permission/I-O error
  on a `file://` sidecar is the same `JammiError::Storage`/`Io` arm by construction
  (`ManifestError::Storage → JammiError::Storage`, `object_store` errors → `StorageError`) but is
  not executed: a `chmod`-based fixture silently passes under a root CI lane (memory note
  `jammi-ci-root-permission-fault-tests`) and no fault-injection seam exists in the store; adding
  one was out of scope.
- **UNCOVERED — the `FIRST_ASSIGN_BOUND` timeout arm** (a silent client for 10 s): inherited from
  U5a-1, still not timed in the suite.
- **UNCOVERED — `AbortReason::Drain` on the sender-dropped branch** (the `InferenceSession` dropping
  while a rank is held): the session outlives every server in the fixtures; the arm is the same
  `wait_for` `Err` path as a phase flip and is stated, not executed.
- **UNCOVERED — the round-frame dispatch point receiving a real round frame**: no such frame exists
  in `RankControl`'s oneof at this head; only the empty-frame protocol violation reaches
  `dispatch_round_frame` and is not separately executed (a `control: None` frame cannot be built
  by the generated client without a raw codec; the second-`Assign` row executes the same
  end-with-trailer path).
- **Non-disclosure over the sixteen determinants on the plain lane** is executed; the
  "a leaking arm reds the pairwise oracle" mutation is inherited from U5a-1 and not re-executed.
- The world>1 admission row from a REAL coordinator's `fill_training_set_identity` call on the wire
  path (U5b-1b-ii's materialization step) does not exist yet; every world>1 fixture fills the pair
  through the db CAS directly.

### 4. Gates (trimmed set per the lead; exit codes and counts)

All with `CARGO_TARGET_DIR=…/targets/u5a2`; one `--features` set per crate for the whole session
(`jammi-db`: `live-postgres-tests,test-hooks`; `jammi-ai`/`jammi-server`: `test-hooks`). The final
tree (tip below) is exactly the tree these ran on: every mutation was reverted by `git checkout`
and `git status` is clean; the only edit after the server runs was the ai TEST rewrite of the
RELEASE-beside-a-rank row, re-run green with its mutation red, then `cargo fmt --check` and
`cargo clippy -p jammi-ai` re-run green.

| Command | exit | result |
|---|---|---|
| `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1:54329/jammi_test cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- --test-threads=1 get_result_table_for_tenant get_job_for_rank_carries_the_tenant_text` | 0 | 6 passed (3 tests × sqlite + postgres); 0 failed |
| `cargo test -p jammi-ai --features test-hooks --test it -- jobs_shutdown host_admission` | 0 | 28 passed (20 `jobs_shutdown` + 8 `host_admission`); 0 failed |
| `cargo test -p jammi-ai --features test-hooks --test it -- release_and_stop_beside_a_held_rank` (after the oracle rewrite) | 0 | 1 passed |
| `cargo test -p jammi-server --features test-hooks --test it -- --test-threads=4 gang tenant_isolation_oracle::gang_service_is_unimplemented` | 0 | 47 passed (gang_service 37, gang_rank_admission_oracle 4, gang_admission_catalog_fault_oracle 3, gang_terminal_write_oracle 3, gang_training_spec_parity 2 — postgres arm skipped here —, tenant_isolation_oracle 1); 0 failed |
| `JAMMI_TEST_PG_URL=… cargo test -p jammi-server --features test-hooks --test it -- --test-threads=1 in_flight_gauge gang_training_spec_parity` | 0 | 3 passed (the gauge row; parity sqlite + postgres, postgres EXECUTED) |
| `cargo test -p jammi-server --features test-hooks --lib -- gang` | 0 | 1 passed (`reverify_ends_are_pairwise_distinguishable_in_reason_scope_and_count`) |
| `cargo test -p jammi-server --test it -- gang tenant_isolation_oracle::gang_service_is_unimplemented` (PLAIN lane, the count) | 0 | 44 passed; 0 failed — the `test-hooks` lane runs 3 more (`run_rank_last_refusal_reason_distinguishes_every_determinant`, `run_rank_refuses_unavailable_at_once_while_a_loop_job_runs`, `run_rank_waits_out_a_claim_probe_then_admits_if_freed_or_refuses_unavailable`), invisible to the plain lane by `#[cfg]` |
| `cargo clippy -p jammi-server --all-targets --features test-hooks -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | 0 | clean (a pre-existing 4-space doc continuation in the 2e bullet tripped `doc_overindented_list_items` once I restructured that bullet; re-indented) |
| `cargo clippy -p jammi-db --all-targets --features live-postgres-tests,test-hooks -- -D warnings` | 0 | clean |
| `cargo fmt --all -- --check` | 0 | clean |
| `python3 ci/scripts/perf/check_citations.py` (not in the trimmed set; run because my insertions shifted lines) | 0 | `1030 file(s) scanned, all PATH:LINE citations resolve` — after re-anchoring seven citations (six in `MAINTAINER-GUIDE.md`, one in `pinned_source_gate.rs`) that my worker.rs/runtime.rs/session.rs insertions had moved |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | OK |

Executed mutations: M1–M12 red on the first run; M13 green on the first run (vacuous oracle — an
idle loop exits at 2a before 2e; recorded honestly), the oracle rewritten, then M13 red — see §2.
`u5a2-scratch/mutations.txt` and `mut-M*.log` hold every run's output.

### 5. Commits (`git log --oneline 856ec8dd..HEAD`)

```
54bbe341 feat(wire-server): #500 U5a-2 — admit-and-hold: the world>1 conjunct (#566), the holder CAS, the four-arm hold loop, re-verification's three ends
db82aef9 feat(ai-core): #500 U5a-2 — HostAdmission: session-owned phase, the holder CAS lattice, the claim loop's probe
8332a289 feat(db): #500 U5a-2 — admission row carries the job's tenant text and training-set pair; strict tenant-pinned result-table resolver
```
(no trailers, per the brief; 30 files, +4331/−785 against `856ec8dd`; `git status` clean.)


## 5. U5b-1b-ii — database slice (landed as one commit on this branch; original `6dab678b`)

The implementer's contract, folded by the lead after opening the cited lines (the cooldown term is `lease_expired_clause("next_assembly_after", ..)` verbatim; `lease.rs` carries a zero diff; `AssemblyOutcome::effect` has no wildcard arm; the pressure round's block 8 is closed by this shape).

### 1. Scope shipped

Files touched (all inside `crates/jammi-db` and its docs; `[worker]` untouched, per the brief):

- `crates/jammi-db/src/config/mod.rs` — new `DistributedConfig { max_world_size: u32 }`
  section (default 1, `deny_unknown_fields`), `JammiConfig.distributed` field, `Default`
  wiring, and a `config.distributed.validate()?` call in `load_from` (refuses `0`).
- `crates/jammi-db/src/config/tests.rs` — 4 tests for the new section (default, TOML
  round-trip, zero-refusal at `validate()` and at `load_from`, independence from
  `[worker]`).
- `crates/jammi-db/src/catalog/schema.rs` — `MIGRATION_037_JOBS_ASSEMBLY_FAILURES_NEXT_AFTER`:
  `jobs.assembly_failures INTEGER NOT NULL DEFAULT 0`, `jobs.next_assembly_after` (nullable
  `TEXT`, same representation `lease_expires_at` uses).
- `crates/jammi-db/src/catalog/migrations.rs` — the `("037_jobs_assembly_failures_next_after",
  schema::MIGRATION_037_JOBS_ASSEMBLY_FAILURES_NEXT_AFTER)` tuple appended.
- `crates/jammi-db/src/catalog/jobs_repo.rs`:
  - `AssemblyOutcome` (9 variants: `Refuted`, `AllRootDivergent`, `Unavailable`,
    `StoreUnavailable`, `ShortListed`, `NoBody`, `Drain`, `Cancelled`, `Success`) and a
    private `AssemblyEffect` (`Neither | CooldownOnly | CooldownAndCounted | Reset`) — a
    proper 4-way enum rather than a `(bool, bool)` pair, so "counted but not cooled" (a
    combination the design never calls for) is unrepresentable. `AssemblyOutcome::effect`
    is an exhaustive match (no wildcard): a new variant with no rule is a compile error.
  - `assembly_backoff(failures: u32) -> Duration` — bounded exponential (`2s * 2^k`,
    `k` clamped to 8 before the shift to avoid overflow regardless of `failures`, final
    ceiling 300s / 5 minutes, both named constants).
  - `Catalog::record_assembly_outcome(job_id, attempt, outcome) -> Result<bool>` — a
    locking read of the current `assembly_failures` (guarded by `job_id`+`attempts`, `FOR
    UPDATE` on Postgres; SQLite's `BEGIN IMMEDIATE` already serializes), then ONE `UPDATE`
    that writes both columns per the rule, using `super::lease::lease_deadline_expr` (the
    SAME helper `claim_next`'s own lease stamp uses) for the cooldown deadline. Returns
    `false` with no write when `attempt` no longer names the row's current attempt.
  - `claim_next`'s candidate subselect gains `AND {cooldown_clause}`, where
    `cooldown_clause = lease_expired_clause("next_assembly_after", kind, &mut params)` —
    **reused verbatim**, not a new function (see the pressure-round correction below).
  - `TrainingSetAssembly { Won, Reused, Moved }` and
    `Catalog::materialize_or_reuse_training_set(job_id, claimed_by, attempts, ref,
    location) -> Result<TrainingSetAssembly>` — a thin, DESIGN.md-vocabulary wrapper of
    `fill_training_set_identity`, mapping `Filled → Won`, `Reused → Reused`, `Aborted →
    Moved`.
- `crates/jammi-db/tests/it/migrations.rs` — `037_...` appended to `EXPECTED_MIGRATION_NAMES`
  and the 029-replay `DELETE FROM applied_migrations ... IN (...)` list (037 ALTERs `jobs`);
  a new column-presence assertion after the replayed reopen (the fifth K5 pin site's teeth);
  a new `migration_037_is_ordered_after_036_and_adds_assembly_failures_next_after` oracle
  (sqlite+postgres) mirroring 034/035/036's own shape, plus a default-value assertion
  (`assembly_failures = 0`, `next_assembly_after = NULL` on a row naming neither column).
- `crates/jammi-db/tests/it/assembly_outcome.rs` (new) — every behavioural oracle below.
- `crates/jammi-db/tests/it/main.rs` — `mod assembly_outcome;`.
- `docs/guide/src/configuration.md`, `docs/maintainer/MAINTAINER-GUIDE.md` — `[distributed]
  max_world_size` documented (the guide fence is inside the ONE big TOML fence
  `docs_config_fences.rs`'s real-loader oracle already walks, so it is exercised, not just
  asserted in prose).

#### Deviations from the brief, with the cited reason

1. **No `cooldown_ready_clause` function.** The brief (`U5b-1b-ii-db.md`) asked for a new
   `(next_assembly_after IS NULL OR next_assembly_after <= <backend now>)` clause. Mid-task
   the lead relayed a pressure-round finding, verified against `crates/jammi-db/src/catalog/lease.rs:13-26,
   47-56, 60-73, 147-158`: SQLite's arm deliberately binds the APPLICATION clock (never a
   SQL clock function, which truncates precision) while Postgres's reads the DATABASE's
   `now()`, and the two backends store different representations. The binding correction:
   reuse `lease_expired_clause` VERBATIM (its `<` predicate, not a new `<=` one) rather than
   invent a second, near-duplicate clock helper. I cut the `cooldown_ready_clause` function
   and its lease.rs tests entirely and wired `claim_next`'s cooldown term straight to
   `lease_expired_clause("next_assembly_after", kind, &mut params)` — `lease.rs` ends the
   unit with a NET-ZERO diff (`git diff --stat crates/jammi-db/src/catalog/lease.rs` is
   empty), so it is not part of this unit's shipped diff at all.
2. **Acceptance (b) restated, per the same relay:** "no second clock source: the cooldown
   predicate is rendered by the lease module's helpers" (a source-scan oracle), plus the
   process-clock-skew oracle scoped to the Postgres arm only. Both built exactly as
   directed — see properties 5 and 6 below.
3. **`record_assembly_outcome` is a locking SELECT + one UPDATE, not a single bare UPDATE.**
   The brief's prose says "one UPDATE applying that rule"; I read that as "one coherent,
   atomically-guarded write" rather than "zero reads permitted" — embedding the exponential-
   backoff arithmetic (`2^k` clamped, ceiling, per-dialect `LEAST`/`MIN`) directly in raw SQL
   for two dialects risked exactly the overflow/precision class of bug `assembly_backoff`'s
   own doc comment warns about, for no benefit: the read is a locked, guarded, same-
   transaction SELECT (`FOR UPDATE` on Postgres; SQLite's `BEGIN IMMEDIATE` already
   serializes per `catalog::migrations`' own module doc and the concurrent-migrate tests),
   and the SUBSEQUENT write is exactly one `UPDATE` statement per call, guarded by the same
   `job_id`/`attempts` CAS `fill_training_set_identity` uses. `assembly_backoff` itself is a
   pure, independently testable Rust function (see property 4's escalation oracle), not
   duplicated per-dialect SQL.
4. **`materialize_or_reuse_training_set` maps `Aborted → Moved`, not a bespoke third state.**
   `fill_training_set_identity` (existing, `crates/jammi-db/src/catalog/jobs_repo.rs`) already
   distinguishes "won"/"reused"/"aborted, no write" exhaustively; the brief's ask ("returns
   whether it WON or REUSED; a moved claim ... aborts with NO write, a distinct typed
   outcome") is satisfied by a thin rename into the coordinator's own vocabulary
   (`TrainingSetAssembly`), not a new state machine.

### 2. Properties

| Property (quantified) | Executed oracle (test path::name, lane/features) | Executed mutation that reds it (change, red output first line) |
|---|---|---|
| (a) A higher-priority job inside its cooldown never blocks a lower-priority ready job, on either backend. | `assembly_outcome::cooldown_job_never_blocks_a_lower_priority_ready_job::{sqlite,postgres}` (`--features live-postgres-tests,test-hooks`) | Removed `AND {cooldown_clause}` from both `candidate` branches in `claim_next` (jobs_repo.rs). Red (both dialects, verified separately with/without `JAMMI_TEST_PG_URL`): `assertion \`left == right\` failed: the higher-priority job must be skipped while cooling down ... left: "cooling-high" right: "ready-low"`. |
| (b1) No second clock source: every cooldown-bearing SQL fragment in `jobs_repo.rs` is rendered through `catalog::lease`'s helpers, never a hand-written `CURRENT_TIMESTAMP`/`datetime('now'`/`chrono::Utc::now()`. | `assembly_outcome::cooldown_sql_has_no_second_clock_source` (hermetic, no backend) | Added `const MUTATION_PROBE_TOKEN: &str = "CURRENT_TIMESTAMP";` to jobs_repo.rs. Red: `jobs_repo.rs must never hand-write a clock-bearing SQL fragment ... found the literal "CURRENT_TIMESTAMP"`. |
| (b2, Postgres-only) The cooldown predicate is governed ENTIRELY by the Postgres SERVER's own clock — a value set via `now() ± interval` and compared via the same `now()`, never a value this test computes from its own process clock. | `assembly_outcome::postgres_cooldown_is_governed_by_the_server_clock_alone` (`--features live-postgres-tests`, requires `JAMMI_TEST_PG_URL`) | Same mutation as (a) (cooldown clause removed). Red: `the hour-ahead cooldown must still hold the second row back`. |
| (c1) The assembly outcome table is exhaustive and total: every one of the 8 non-success variants gets exactly its documented (counted?, cooled?) pair, on both backends. | `assembly_outcome::assembly_outcome_table_is_exhaustive_and_each_rule_fires_once::{sqlite,postgres}` | Reassigned `AllRootDivergent` from `CooldownAndCounted` to `CooldownOnly` in `AssemblyOutcome::effect`. Red: `assertion \`left == right\` failed: all-root-divergent: counted=true left: 0 right: 1`. |
| (c2) Consecutive COUNTED failures escalate the backoff (bounded exponential on the counter), on both backends. | `assembly_outcome::consecutive_counted_failures_escalate_the_backoff::{sqlite,postgres}` | Pinned `assembly_backoff`'s exponent to `0` regardless of `failures` (constant backoff). Red: `the second counted failure's backoff (1.999...s remaining) must exceed the first's (1.999...s remaining)`. |
| (c3) A success resets BOTH the counter and the cooldown, even after a counted failure armed both. | `assembly_outcome::success_resets_the_counter_and_the_cooldown::{sqlite,postgres}` | Remapped `Success` to `AssemblyEffect::Neither` (no reset). Red: `assertion \`left == right\` failed: success must reset the counter to 0 left: 1 right: 0`. |
| (d) `record_assembly_outcome`: a moved claim (stale `attempts`) writes NOTHING, on either backend. | `assembly_outcome::record_assembly_outcome_moved_claim_aborts_without_a_write::{sqlite,postgres}` | Dropped the `attempts = $2` guard from the internal locking SELECT (job_id-only). Red: `a stale attempt number must write nothing at all`. |
| Migration 037 ordered after 036; adds `jobs.assembly_failures` (NOT NULL) / `jobs.next_assembly_after` (nullable TEXT), on both backends. | `migrations::migration_037_is_ordered_after_036_and_adds_assembly_failures_next_after::{sqlite,postgres}` | RED at base: the migration, the const, and the test are all new in this diff — the test cannot even compile against the pre-change tree (`schema::MIGRATION_037_...` does not exist). |
| The fifth K5 pin site: 037 is on the 029-replay's ledger-clearing `IN (...)` list, so a manufactured pre-029 reopen replays it and the recreated `jobs` table carries both columns. | `migrations::migration_029_copies_training_jobs_rows_into_jobs_as_queued` | Dropped `'037_jobs_assembly_failures_next_after'` from the `DELETE FROM applied_migrations ... IN (...)` list. Red: `reopened jobs must carry 'assembly_failures' (migration 037 replayed): []`. |
| (f1) `materialize_or_reuse_training_set`: first call WINS; a repeat under the same claim, same values, REUSEs; a concurrent racer REUSEs (never a second write); a moved claim aborts with NO write. | `assembly_outcome::materialize_or_reuse_training_set_{first_call_wins,second_call_same_values_reuses,concurrent_racer_reuses_never_overwrites,moved_claim_aborts_without_a_write}::{sqlite,postgres}` | Remapped `TrainingSetFillOutcome::Aborted → TrainingSetAssembly::Reused` (instead of `Moved`). Red: `assertion \`left == right\` failed left: Reused right: Moved`. |
| `[distributed] max_world_size` defaults to 1, round-trips through TOML, refuses `0` at both `validate()` and `load_from`, and loads independently of `[worker]`. | `config::tests::distributed_config_{default_is_single_rank,toml_round_trips_max_world_size,zero_max_world_size_is_refused_at_load,loads_independently_of_worker}` (`--features test-hooks`, `--lib`) | Disabled the `max_world_size == 0` branch (`if false && ...`) in `DistributedConfig::validate`. Red: `called \`Result::unwrap_err()\` on an \`Ok\` value: ()`. |
| The `[distributed]` guide fence parses under the real loader (doc-parity). | `docs_config_fences::docs_toml_fences_parse_under_the_real_loader` (pre-existing oracle, exercised by the new fence content) | Not separately mutated — this is the SAME generic oracle every guide fence already rides; a broken `[distributed]` fence would fail it exactly as any other section's would (verified green with the addition in place). |

### 3. Uncovered

- **The body's own call sites** (`jammi-ai`'s `fine_tune/worker.rs`/`spec.rs`, `jammi-server`'s
  `grpc/gang.rs`) that will actually call `record_assembly_outcome` and
  `materialize_or_reuse_training_set`, and the `serveable_world`/`[distributed]
  max_world_size` submit-time cross-check — explicitly out of scope for this unit (U4b,
  U5b-1b-i, U5a-2 build them); labelled DEFERRED per the brief, not a gap in this unit's own
  verb-level oracles.
- **A true OS-clock-skew injection** (actually setting the test process's wall clock away
  from the Postgres server's) is not attempted — no clock-mocking harness exists in this
  crate, and the lead's relayed correction explicitly reframes acceptance (b) as the
  structural "no second clock source" source-scan plus the Postgres-server-clock-authority
  behavioural test, which together are the achievable, hermetic proxy for the same property.
  Labelled, not silently assumed.
- **`assembly_backoff`'s behaviour beyond the clamp exponent (8)** — failures counts above
  8 all saturate at the same 300s ceiling; not separately probed at e.g. `failures = 1000`
  for overflow-freedom beyond the `checked_shl`/`expect` invariant already encoded (the
  `expect` message states why it cannot panic: the exponent is clamped to `< 32` before the
  shift). No test drives `failures` near `u32::MAX`; the `unwrap_or(u32::MAX)` fallback in
  `record_assembly_outcome`'s `u32::try_from(new_failures)` (i32 → u32) is likewise
  unexercised at that boundary (`new_failures` is bounded by `i32::MAX` well before it, via
  `saturating_add`, so the fallback arm is dead in practice but retained as a defensive
  total function).

### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | |
| `cargo clippy -p jammi-db --all-targets --features live-postgres-tests,test-hooks -- -D warnings` | 0 | |
| `cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- assembly_outcome migrations docs_config_fences --test-threads=1` | 0 | 53 passed (20 `assembly_outcome`, 30 `migrations`, 3 `docs_config_fences`), 0 failed, `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1:54329/jammi_test` set (live scratch Postgres 16, reachable on 54329) |
| `cargo test -p jammi-db --features live-postgres-tests,test-hooks --lib config::tests::distributed -- --test-threads=1` | 0 | 4 passed |
| Every mutation in §2 | red confirmed, then reverted | `git status --short` clean of mutation artifacts after each revert; `git diff --stat crates/jammi-db/src/catalog/lease.rs` empty (net zero touch, per the pressure-round correction) |

One test binary reused for the whole session (`CARGO_TARGET_DIR=<scratch>/targets/u5b1bii-db`,
one `--features live-postgres-tests,test-hooks` set throughout).

### 5. Commits

```
6dab678b feat(db): #500 U5b-1b-ii (db) — assembly cooldown/counter, [distributed] max_world_size, the CAS call-site helper
```
(one commit on `unit/u5b1bii-db`, off tip `856ec8dd`)


## 6. U5b-1b-ii — coordinator body (landed as four commits; original tip `a8ab9667`; the U4b-tail commit it carried was dropped as already on this branch)

The implementer's contract, folded by the lead after checking: `assembly_outcome` is a total match with only `Moved → None`; the lease is settled by `AssemblyOutcome::counts_toward_failures` (an uncounted outcome hands the lease back at once — OPS D10 at the job level, deviation 2 accepted: leaving it to expire would burn an attempt on every uncounted outcome); `RankAdmission` holds no catalog handle; a retry binds the recorded training set through the tenant-pinned `get_result_table`, never the strict resolver; the dropout-position gather is four `f32` limbs (the F64 row was a dtype the `Peer` wire refuses by design — a defect only a real `Peer` gang reaches, found by this unit's server oracle). The first end-to-end `Peer` gang through the real `run_rank` handler publishes bytes identical to `Local`. Two items surfaced here are scheduled, not accepted: the `jammi-bench` crate does not compile on this branch (U4b's `load_bundle -> Option` and §2b's witness left two call sites behind; no unit's trimmed gates built the bench crate) — fixed in wave C; the agreement digest (U4b's `canonical_vars_digest`) is bound nowhere — bound by U5b-1b-iii at the rank body's post-target-build seam.

Worktree `wt-u5b1bii`, branch `unit/u5b1bii`, base `feat/500-wave3c` @ `9b0dcb57`. Every path
is repo-relative; every oracle named was executed on the tip (§4); every mutation in §2 was
applied to the committed tree, run through ONE filtered test, and reverted (`git checkout --
<file>`, `git status --short` empty afterwards — `u5b1bii-scratch/mutations.py`, logs
`mut-M*.log`). One `--features` set per crate for the whole session (`jammi-ai`,
`jammi-server`: `test-hooks`), `RUSTC_WRAPPER=sccache`, `CARGO_TARGET_DIR=…/targets/u5b1bii`.

#### 1. Scope shipped

**`crates/jammi-ai/src/fine_tune/spec.rs`** — `RankAdmission { serveable_world, collective,
cuda_build }` (was `devices`): `from_config` reads `[distributed] max_world_size`; `new(serveable_world,
..)`; `serveable_world()`; `admit` refuses `world_size > serveable_world` naming
`[distributed] max_world_size` — the device count is no longer read, and the type holds no
catalog handle (no catalog read is expressible from `admit`). Doc rewritten to the fleet-bound
semantics (a count within the serveable world but beyond this host's devices submits and is
decided by assembly).

**`crates/jammi-ai/src/fine_tune/worker.rs`**
- `TopologyDecision::decide(world_size, local_ranks)` → `Single` (`<= 1`) | `Local { world }`
  (`<= local_ranks`) | `Peer { world }`; called once per `run_spec` FineTune/GraphFineTune arm.
  A `GraphFineTune` at `Peer` is refused typed at the coordinator's edge (no training-set table
  for a member to be admitted against) — `WorkerJobError::Failed`, never a smaller gang.
- `train_fine_tune(.., topology: RankTopology)`: `Single` is byte-unchanged (`rank_ctx: None`,
  the builder's `single_rank` default; rank 0 on `BlockingCall::spawn_blocking` as before);
  `Local { world }` builds a `LocalGang::with_timeout([select_device(for_device(devices[r]))..],
  rank_timeout_secs)`, one `RankContext::new(gang.rank(r), PartitionSpec::for_gang(r, world,
  batch_size, BlockByGlobalBatch))` per rank, a per-rank model-cache entry
  (`ModelCache::get_or_load_on(devices[r], ..)`), a per-rank source (`TrainingSource::replicate`),
  and spawns ranks `1..world` on `BlockingCall::spawn_thread` (each entering the runtime's
  `Handle`) — the SECOND production minting site (docs: collective/mod.rs, guide §2.8c/§2.8d);
  the other ranks are joined before rank 0's result is read and any rank's error/panic fails
  the run (no publish over an incomplete gang); `Peer { world, coordinator }` gives rank 0
  `RankContext::new(coordinator, for_gang(0, ..))`.
- `RunFineTuneParams { rank, rank_ctx, .. }`; `run_fine_tune_blocking` sets `.rank_context` when
  given; `compute_and_persist_acceleration_report(.., persist: bool, ..)` — every rank computes,
  rank 0 alone persists.
- `MemberDialer` trait (the engine's one transport seam) + `HostAdmission::{install_member_dialer
  (write-once), member_dialer}`.
- `WorkerJobError::Abandoned(String)` + its arm in `run_claimed_job_under` (no terminal write,
  checkpoint GC, warn).
- The coordinator body: `TrainingSetIdentityPair` (sidecar digest, table name — built in
  `run_spec` beside the materialization descriptor), `CoordinatorEnd` (13 variants,
  `VARIANTS`/`ordinal`, `Display`), `assembly_outcome(&CoordinatorEnd) -> Option<AssemblyOutcome>`
  (TOTAL match, no wildcard; `Moved => None`; `MemberAborted { reason }` one-to-one over
  `AbortReason`, `Unspecified → Unavailable`; `TrainingFailed`/`Published → Success`),
  `ShortListing`, `assign_ranks(&[GangMember], world)` (pure: sorted by `instance_id` bytes, rank
  `r` = the `r`-th, short → `Err`), `cancel_links`, `JobWorker::coordinate` (record + settle) and
  `JobWorker::assemble_and_run` (CAS → gates → listing with this process's own `MemberRoot` →
  assignment → `peer_addr_of` + dial per member → `Peer::coordinator(..).with_timeout(..)` →
  `train_fine_tune(Peer)` → `end_members` → classify: `member_aborts` → `MemberAborted`,
  `fault` → `LinkFault`, else `TrainingFailed`). Settle rule: `Published → Ok(artifact)`;
  `Cancelled → Err(Cancelled)` (the existing arm decides request vs lease loss);
  `TrainingFailed → Err(Failed)`; `Moved → Err(Abandoned)` with no record and no release; every
  other end records, then `release_job_lease` iff `!outcome.counts_toward_failures()`, then
  `Err(Abandoned)`.
- **The retry binds the recorded training set**: `run_spec(.., recorded_pair)` — the claimed
  `JobRecord`'s `training_set_ref`/`training_set_location` (written by a prior attempt's CAS,
  write-once) — and `bind_recorded_training_set` resolves that table by name through the job's
  tenant-pinned catalog (`get_result_table`, never the strict resolver, whose only caller is
  the gang handler by an enumerating oracle), requires `ready`, verifies the sidecar's
  `artifact` against the recorded digest (the same verify a member is admitted against), binds
  it on the session context (`ResultStore::bind_result_table`, the producer's own reuse arm) and
  hands the arm a `TrainingSetTable { outcome: Reused }`; a first attempt materialises as
  before. Without it a registered source (anchored unpinned) mints a fresh table per attempt
  and the CAS reads every retry as `Moved` — the executed red of M16.
- `training_test_hooks`: `note_topology`/`topology_for`, `note_assembly_listing`/
  `assembly_listings_for`, `note_coordinator_end`/`coordinator_ends_for` (`test-hooks` only).

**The seed split, wired (the lead's binding addition after the U4b tail landed at
`2d46508b`; that commit is cherry-picked onto this branch as its own commit — see §5 — because
the `_for_rank` builders and `new_with_base_seeded` exist nowhere else, and the demanded
oracles must execute):**
- `worker.rs::run_fine_tune_blocking`: `dropout_seed = rank_ctx.map_or(config.seed, |c|
  c.dropout_seed(config.seed))` (rank 0 / the single rank: the identity); the three head
  builders are the `_for_rank` siblings at that seed; `BuildEncoderAdaptersParams.dropout_seed`
  → `LoraBuildConfig { seed: config.seed, dropout_seed }`.
- `crates/jammi-lora/src/config.rs`: `LoraBuildConfig.dropout_seed: u64` (doc: `seed` keys
  the init draw, `dropout_seed` the mask draw); `frozen()` sets both `0`.
- `crates/jammi-encoders/src/{lora_site.rs, bert.rs, distilbert.rs, modernbert.rs}`: the four
  tower-side `LoraLinear::new_with_base(.., seed, ..)` calls → `new_with_base_seeded(.., seed,
  dropout_seed, ..)` (the BERT family is not on the `LoraSite` seam, so its own builders are
  threaded too).
- Every other `LoraBuildConfig { .. }` construction site (29 across 18 files: jammi-ai's
  `candle.rs` serving-side load, `trainer.rs` tests, `tower_adapters.rs`, the `gpu_capability`
  target — gated, by inspection; jammi-lora's `adapter.rs` and `tests/it.rs`; jammi-bench's
  three; jammi-encoders' `modernbert.rs` test, `test_support.rs` and its it/tests) gains
  `dropout_seed: <the same expression as seed>` — the `init == dropout` shape every one of
  them had.
- `training_test_hooks::{RankTarget, note_rank_target, rank_targets_for}` (`test-hooks`):
  recorded in `run_fine_tune_blocking` after the target is built and before any step.
- `crates/jammi-encoders/src/lora_site.rs` `#[cfg(test)] mod tests`: the site-level oracle.

**`crates/jammi-ai/src/fine_tune/trainer.rs`** — `gather_dropout_positions` encodes each
rank's `u64` positions as four 16-bit limbs in an `f32` `(1, 4n)` tensor
(`encode_dropout_positions`/`decode_dropout_positions`, free `pub(crate)` fns + a codec
oracle): the F64 row the base gathered is a dtype the `Peer` wire refuses by design (U5b-1b-i
P9), so every `Peer` gang with an artifact store faulted at its first epoch boundary — found
by the server oracle's attempt 2 (`all_gather: round 10: … F64 is refused at the seam`). The
new encoding is exact for EVERY `u64` (F64 was exact only below `2^53`) and byte-neutral for
`Local`/`Noop` (the U4b determinism/resume-with-dropout oracles re-executed green).

**`crates/jammi-ai/src/fine_tune/collective/peer.rs`** — `Link.session_abort: Option<i32>`
recorded in `Link::recv` from `Inbound::abort_reason` (only a member's `RankEvent::Aborted`
carries one); `CoordinatorLink::{session_abort, cancel_session}` (a `Cancel` frame via
`try_send`, never blocks); `Peer::{fault, member_aborts, end_members}`.
**`collective/mod.rs`** — the two minting sites' docs. **`collective/peer_tests.rs`** — the P14
oracle extended with the typed record.
**`crates/jammi-ai/src/fine_tune/data.rs`** — `TrainingDataLoader::replicate` (rows/batches
copied, the pool reservation stays with rank 0's loader). **`source.rs`** — `StreamedSet:
Clone`, `TrainingSource::replicate`.
**`crates/jammi-db/src/catalog/jobs_repo.rs`** — `AssemblyOutcome::counts_toward_failures`
(derived from `effect`, never a second table). **`config/mod.rs`** — `local_ranks` doc current.
**`crates/jammi-server/src/grpc/gang_rounds.rs`** — `GangDialer: MemberDialer` over
`dial_member`. **`runtime.rs`** — `OssServer::bind` installs it beside the gang listener.
**`crates/jammi-server/Cargo.toml`** — dev-dep `candle-nn` (the it oracle's projection head;
already in the build through `jammi-ai`). **`tests/it/gang_rounds.rs`** —
`mount_real_gang_server` `pub(crate)`.
**Tests** — `crates/jammi-ai/tests/it/gang_coordinator.rs` (new), `rank_admission.rs`
(re-anchored on the serveable world; one new pure oracle), `crates/jammi-server/tests/it/
gang_coordinator.rs` (new), worker.rs unit tests (three new).
**Docs** — `docs/maintainer/MAINTAINER-GUIDE.md` (§2.8c minting sites; NEW §2.8d the
coordinator body; the `[distributed]` and Train-flow sentences), `docs/guide/src/configuration.md`
(`local_ranks`, `[distributed]`), 11 `PATH:LINE` citations re-anchored under my insertions
(10 in the guide, 1 in `gguf.rs`).

##### Deviations from UNITS.md / the brief, with the reason and the code

1. **The `Moved` arm records no `AssemblyOutcome`** ("every exit arm records exactly one" vs
   "a `Moved` claim exits with no write" — the brief's own two sentences): `assembly_outcome`
   returns `None` for exactly `CoordinatorEnd::Moved` (`worker.rs`, the total match), and the
   table oracle asserts exactly one `None`. A moved row is another attempt's to describe.
2. **The lease is settled by the outcome's counting class, not left for reclaim on every end.**
   DESIGN §4 names leave-`running`-for-reclaim as the only requeue path; the brief says a
   member's `Unavailable` is "cooled, not counted". Leaving the lease to expire would burn one of
   `MAX_ATTEMPTS` (`reclaim_expired_jobs` counts `attempts - releases`) on every uncounted
   outcome — the opposite of "not counted" at the job level. So an uncounted outcome hands the
   lease back at once (`Catalog::release_job_lease`, `releases + 1`, lease NULL, arm 1a
   reclaims it within one idle poll, OPS D10) and a counted one leaves it to expire; both keep the
   row `running` for reclaim — the same arm, reached sooner. `AssemblyOutcome::
   counts_toward_failures` (jobs_repo.rs) is the one rule, derived from `effect()`.
3. **`AllRootDivergent` is never produced.** Root identity is a predicate INSIDE
   `list_gang_members` (U5b-1a-A2), so divergent-root members are invisible to the body and an
   all-divergent fleet is `ShortListed`. Stated in `assembly_outcome`'s doc and guide §2.8d.
4. **A coordinator-side catalog fault, an unresolvable address, a refused dial, a `Peer` that
   cannot be built and a mid-run link fault all map to `Unavailable`** (cooled, not counted):
   the table has no coordinator-side transient row, and `Unavailable`'s rule is the transient
   one. A run failure with no gang fault (`TrainingFailed`) maps to `Success` — "assembly
   proceeded to a run", `AssemblyOutcome::Success`'s own doc — and the caller records `failed`
   exactly as a W=1 run's failure is recorded.
5. **A host with no `MemberRoot` or no installed dialer ends `HostCannotCoordinate → ShortListed`**
   rather than a terminal failure: the job is cooled and released so a member-capable host claims
   it; the reason is logged and in the end's `Display`.
6. **The Peer oracle's byte comparison is conditional at this tip** (`crates/jammi-server/tests/
   it/gang_coordinator.rs`): `run_spec` binds `Streamed` for every column-source `fine_tune`
   (`source::whole_set_arm` is `None` outside mining/GradCache — both refused at `world > 1`),
   and the trainer refuses a `Streamed` source at `world > 1` (U4b's contract §2 deviation 3; the
   U4b tail, §2a, is the concurrent unit building that arm). The oracle admits exactly two ends
   for attempt 2 — `Published` (then bytes must equal the `LocalGang` reference) or the trainer's
   pinned streamed refusal (then `failed`, the error names it, the member ends "nothing applied")
   — and is red on any other. With the U4b tail cherry-picked onto this branch (§5) the
   `Published` branch EXECUTES (`t-srv-branch.log`, `attempt-2 end: published (ordinal 12)`):
   rank 0's adapter over `Peer` through the real hold loop is byte-identical to `Local`'s. The
   refusal branch is what the oracle admits on a tree without that tail, never on this one.
7. **The local fan-out oracle is a `graph_fine_tune`** (the one `Resident` source `run_spec`
   binds at this tip, deviation 6): the property — the fan-out through the real `run_spec`
   publishes bytes equal to a U4b-shaped direct `LocalGang` run over the same rows, config, seed
   and base model — is the same; the rows come from the same seeded `GraphSampler`.
8. **No agreement digest is bound** on `Local`/`Peer` ranks (U4b's contract said it binds
   `canonical_vars_digest`; nothing in the tree does — `grep with_agreement` finds only the
   constructors and tests). Binding needs a post-target-build seam on the collective; not built
   here, stated in §3.
9. **The dialer is injected, not called.** `jammi-ai` cannot depend on `jammi-server`, so the
   brief's "dial with `gang_rounds::dial_member`" is met by `MemberDialer` (declared in
   `worker.rs`) implemented by `gang_rounds::GangDialer` over `dial_member` and installed in
   `OssServer::bind` (`runtime.rs`, one statement — outside `grpc/**`, the only wiring line).
10. **`peer.rs`, `data.rs`, `source.rs`, `jobs_repo.rs`, `config/mod.rs`, `runtime.rs`,
    `gguf.rs`, `jammi-server/Cargo.toml`** are touched beyond the brief's files_in_scope, each a
    small seam the body needs (listed above); `stream.rs`, `lora.rs`, `target.rs` and the
    trainer's streamed arm are untouched. **`trainer.rs` IS touched at one function**
    (`gather_dropout_positions`, plus two free codec fns and their oracle) — after the U4b tail
    landed, so no concurrent edit exists — because the F64 gather is a defect only a `Peer` gang
    reaches, and this unit is the first to run one.

#### 2. Properties

| Property (quantified) | Executed oracle (path::name; lane) | Executed mutation that reds it (first red line) |
|---|---|---|
| (d) For every spec, `world_size > serveable_world` is refused typed naming `[distributed] max_world_size` from configuration alone (no catalog handle exists on the type); every `world_size <= serveable_world` passes this arm regardless of this host's devices, on both embedded entrances, the `jobs` table unchanged on refusal | `it::rank_admission::a_count_past_the_serveable_world_is_refused_from_configuration_alone` (pure); `::every_unservable_rank_count_is_refused_at_both_submit_entrances` (the one-device session admits W=2 within a serveable world of 2 and writes exactly one row) | M1: `if false && world_size > self.serveable_world` (spec.rs) → RED `rank_admission.rs:439: two ranks on a serveable world of one: ()` (the W=2 spec was admitted) |
| (d) A W=2 job on a one-device host with serveable world 2 REACHES ASSEMBLY: topology `Peer{2}`, the CAS writes the pair, the empty listing ends `ShortListed` recorded on the row (failures 0, `next_assembly_after` set), the lease released (`releases 1`, lease NULL), status `running`, no error — RED at base (the base refuses W=2 at submit on one device) | `it::gang_coordinator::a_two_rank_job_beyond_this_hosts_devices_reaches_assembly_and_lands_short_listed` | M5: `record_assembly_outcome` skipped (`outcome.filter(\|_\| false)`) → RED `gang_coordinator.rs:551: ShortListed is cooled down: Row { .. next_assembly_after: None .. }`; M7: the release skipped (`if false && !counts`) → RED `gang_coordinator.rs:555: assertion left == right failed: an uncounted outcome hands the lease back at once` (`releases 0`) |
| The CAS `Moved` arm writes nothing: a claim whose attempt moved leaves every assembly/identity/lease/status column byte-identical | `it::gang_coordinator::a_moved_claim_exits_the_coordinator_body_with_no_write` | M4: `Ok(Moved) => {}` (the body proceeds past a moved claim) → RED `gang_coordinator.rs:606: assertion left == right failed` (the row changed: ShortListed recorded and the lease released on a row the attempt did not own) |
| (e) Rank assignment is a pure function of the sorted listing: every one of the 24 permutations of a 4-member listing yields the same `rank → instance_id`; ranks are `1..world` over the first `world−1` in byte order; a listing shorter than `world−1` is `ShortListing`, never a smaller gang | `lib fine_tune::worker::tests::rank_assignment_is_a_pure_function_of_the_sorted_listing` | M3: the sort in `assign_ranks` removed → RED `worker.rs:7778: assertion left == right failed: rank r is the r-th member in instance_id byte order; the surplus member is unused` |
| (e) Through the REAL `GangServer::run_rank`: a member whose slot is busy answers `Unavailable` — the attempt ends `MemberRefused` naming rank 1/member, recorded `Unavailable` (failures 0, cooled), lease released, nothing terminal, no slot held; the NEXT attempt (after the cooldown, through the loop's reclaim + claim) RE-LISTS the same sorted listing, is admitted, builds the `Peer`, runs rank 0, records `Success` (cooldown reset), ends the member session (`Cancel` → slot `Free`); the run's end is `Published` with bytes == `LocalGang` reference, or exactly the pinned streamed refusal (deviation 6) — RED at base (W=2 unsubmittable; no body) | `jammi-server it::gang_coordinator::a_member_answering_unavailable_ends_the_attempt_cooled_and_the_next_attempt_relists_and_runs_the_gang` (`test-hooks`) | M8: the dialer never installed (`runtime.rs`) → RED `gang_coordinator.rs:495: the real handler's Unavailable ends the attempt naming the member: this host cannot coordinate: no gang listener is mounted in this process (no member dialer installed)`; M11: `end_members` sends no `Cancel` (`filter(\|_link\| false)`) → see the M11 row below |
| The dropout-position gather carries every `u64` exactly in a dtype every collective accepts: each position round-trips through four `f32` limbs (`0`, `0xFFFF`, `2^16`, `2^24+1`, `2^53+1`, `u64::MAX`), a malformed row (length, a non-integral or over-wide limb) is refused; a `Peer` gang's epoch boundary no longer faults | `lib fine_tune::trainer::dropout_position_codec_tests::every_u64_position_round_trips_through_four_f32_limbs`; the server oracle's attempt 2 past round 10; `gang_determinism_oracle::{w2_twice_is_byte_identical_with_dropout, resume_after_a_kill_matches_an_uninterrupted_run_with_dropout}` (Local, unchanged bytes) | M17: the limb shift dropped on decode (`value \|= raw as u64`) → RED `trainer.rs:15534: assertion left == right failed` (`2^16` decodes as `1`) |
| A retry of a job whose row names a training set BINDS that table (ready, digest-verified) and its CAS is `Reused`, so the next attempt reaches the run instead of reading its own retry as a moved claim | the server oracle's attempt 2 (`ends[1]` is past dispatch; `training_set_ref` unchanged across attempts) | M16: the bind skipped (`Some(pair) if false`) → RED `gang_coordinator.rs:562: attempt 2 must reach the run (Published or a run failure), got: the claim moved before assembly (no write)` |
| Every `CoordinatorEnd` maps to exactly one `AssemblyOutcome` and only `Moved` maps to none: `assembly_outcome` is a total match (a new variant is a compile error), one sample per `ordinal` in `0..VARIANTS` (a variant without a sample reds), each sample's row as documented; `MemberAborted{reason}` one-to-one over every frozen `AbortReason` with `Unspecified → Unavailable`; the settle rule: exactly `Refuted`/`AllRootDivergent` count | `lib fine_tune::worker::tests::every_coordinator_end_records_exactly_one_assembly_outcome_and_only_moved_writes_nothing` | M2: `ShortListed => AssemblyOutcome::Unavailable` → RED `worker.rs:7717: assertion left == right failed: short listing: 0 fresh member(s) where 1 are needed`; M13: `counts_toward_failures` keyed on `CooldownOnly` (jobs_repo.rs) → RED `worker.rs:7741: assertion failed: O::Refuted.counts_toward_failures()` |
| The topology is decided from `(world_size, local_ranks)` alone: `W<=1 → Single`; `1<W<=L → Local{W}`; `W>L → Peer{W}`; no hybrid | `lib fine_tune::worker::tests::topology_is_decided_from_world_size_and_local_ranks_alone` | M6: `world_size < local_ranks` (W=2,L=2 → Peer) → RED `gang_coordinator.rs:643: assertion left == right failed` (topology `Peer{2}`, not `Local{2}`; the graph job is then refused at the coordinator's edge) |
| The local fan-out: a `local_ranks = 2` host runs a W=2 job through the REAL `run_spec` as `Local{2}` (no coordinator body), completes, and publishes an adapter byte-identical to a U4b-shaped `LocalGang` run (each rank on `spawn_thread`, direct `TrainingLoop::run`) over the same sampled rows, config, seed, base model | `it::gang_coordinator::a_local_ranks_two_host_fans_a_two_rank_job_out_through_run_spec_and_publishes_the_gangs_bytes` | M9: rank 0 given a single-rank context (`None`) while rank 1 joins the 2-rank gang → RED `gang_coordinator.rs:652: assertion left == right failed: Row { status: "failed" .. }` — rank 1 timed out at the 10 s rank deadline, the run failed, nothing published (10.16 s wall vs 0.3 s healthy); M6 (above) |
| The seed split through the REAL `run_spec` (`local_ranks = 2`, `lora_dropout = 0.3`): the two ranks' dropout seeds differ (rank 0 = `config.seed`), every head layer's `dropout_run_seed` is its rank's own, the ranks' pre-step adapter weight digests are equal — and the published bytes still equal the `LocalGang` reference built per rank at the same seeds | `it::gang_coordinator::a_local_ranks_two_host_fans_…` (the seed assertions in the same oracle) | M14: `build_projection_head_for_rank(.., &FineTuneConfig { seed: dropout_seed, ..config }, .., dropout_seed)` → RED `gang_coordinator.rs:689: assertion left == right failed: both ranks start from byte-identical adapter weights` |
| At the tower seam (`LoraSite::wrap`, every non-BERT tower's sites): two sites with the same `seed` and different `dropout_seed`s have byte-identical `lora_a` and report each their own `dropout_run_seed`; a different `seed` moves `lora_a` | `jammi-encoders lib lora_site::tests::the_site_keys_init_by_seed_and_the_masks_by_dropout_seed` | M15: `lora_site.rs` passes `self.lora.seed` for both seeds → RED `lora_site.rs:216: assertion left == right failed` (`dropout_run_seed` = `Some(7)`, not `Some(100)`) |
| A member's `Aborted{reason}` mid-round is recorded TYPED on the coordinator's link for that rank (`Peer::member_aborts == [(1, StoreUnavailable)]`), beside the permanent fault; a member records none | `lib fine_tune::collective::peer_tests::a_member_that_aborts_its_session_faults_the_coordinator_naming_the_reason` (extended) | M10: `session_abort` never recorded in `Link::recv` → RED `peer_tests.rs:1006: assertion left == right failed: the coordinator records the member's abort reason typed, on rank 1's link` (`[]` vs `[(1, 5)]`) |
| The member's slot is free within seconds of the attempt's end | the server oracle's final poll (`holder() == Free` within 5 s) | M11: `end_members` sends no `Cancel` → stayed GREEN (`mut-M11.log`: `1 passed`). The slot frees either way: dropping the coordinator's `Peer` drops its links, the client stream tears down and the hold loop ends on the transport error. So this row pins the SLOT, not `end_members`'s own effect (a cooperative `Aborted{Cancelled}` at the member instead of a transport error) — that effect is UNCOVERED (§3); `end_members` is kept for the clean end, stated as untested |
| W=1 is byte-unchanged: the single-rank path passes `rank_ctx: None` and the builder's own default | the pre-existing `fine_tune::` lib suite and `it -- fine_tune` (§4) | regression-only (as U4b's and §2b's contracts state it) |

#### 3. Uncovered

- **`end_members`'s own effect** (a cooperative `Aborted{Cancelled}` at the member rather
  than a transport error): the slot frees through the transport teardown regardless, so M11
  stayed green; no server-side observation of a held session's end reason exists to pin it.
- **A mid-run member abort / link fault through the coordinator body** — the healthy `Peer`
  run now executes end to end (deviation 6), but no oracle injects a member abort or a link
  fault into a running gang (U5b-2's chaos row); the classification tail
  (`MemberAborted`/`LinkFault`) is executed only through the table oracle and the typed record
  (P14 extended).
- **A rank of a `Local` gang failing while rank 0 succeeds** ("rank 0's artifact is never
  published over a gang that did not complete"): built (`rank_failures`), no oracle manufactures
  a one-sided rank failure through `run_spec`; the symmetric case (rank 0 given a single-rank
  context, rank 1 times out) is M9's executed red.
- **`AllRootDivergent`** — unreachable from the body (deviation 3); labelled, not faked.
- **A deterministic coordinator-side collective refusal** (P9's dtype refusal, a descriptor
  disagreement) maps to `LinkFault → Unavailable`, uncounted and released: such a job retries
  under the 2 s… backoff indefinitely (the design's transient class); a typed split of "this
  rank's own refusal" from "a peer's/transport fault" inside `Peer` is not built.
- **Agreement binding** (deviation 8).
- **The BERT family's and the encoder-adapters path's seed split through `run_spec`**: the
  tower-side threading is by the same mechanical change at four sites and the `LoraSite`
  unit oracle executes the seam every non-BERT tower goes through; `run_spec`'s fan-out oracle
  is a projection-head job (`AnyEncoder` exposes no per-site `dropout_run_seed`, only
  positions), so an encoder-adapters gang's distinct masks are proven at the seam, not
  end to end. The `gpu_capability` construction sites are gated (`live-gpu-tests`) and
  compiled by CI's gated-surface clippy, not here.
- **GPU placement of local ranks** (rank `r` on a real `devices[r]`, one model-cache entry per
  device): the fan-out oracle runs both ranks on the CPU (declared ordinals `[0, 1]` degrade off
  the pod); U4b's own (f) covers the cache-map shape; the pod leg is U7b's.
- **`Handle::enter` on the rank threads**: exercised by the fan-out oracle (rank 1's resume
  discovery and checkpoint calls `block_on` the entered handle); a rank thread panicking outside
  `catch_unwind` (`Ok(Err(_))` join arm) is stated, not executed.
- **The `Abandoned` arm's checkpoint GC** after a mid-run fault: shares `gc_epoch_checkpoints`
  with the cancelled arm (its own oracles); not separately executed here.
- **Two real processes**: the server oracle mounts the real handler over the coordinator's own
  engine (one `HostAdmission`, `run_claimed_job` holds no probe so the slot is the member's);
  two OS processes over one catalog is the fleet leg.

#### 4. Gates

COMMON.md's trimmed set, plus clippy on the two extra crates this unit touches; every command
with `RUSTC_WRAPPER=sccache`, `CARGO_TARGET_DIR=…/targets/u5b1bii`, on the final tip (the
last edits after the mutation round were the three clippy fixes and one `eprintln!`; every
oracle below was re-executed after them). Logs in `u5b1bii-scratch/` (`gates.log`,
`gates2.log`, `t-*.log`, `mut-*.log`).

| Command | Exit | Result |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | clean |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-server --all-targets --features test-hooks -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-db --all-targets --features test-hooks -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-lora --all-targets -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-encoders --all-targets -- -D warnings` | 0 | clean |
| `cargo test -p jammi-ai --features test-hooks --lib -- fine_tune::` | 0 | 351 passed, 0 failed (the WHOLE pre-existing suite + this unit's worker/codec/peer rows — the W=1 byte-identity regression oracle) |
| `cargo test -p jammi-ai --features test-hooks --lib -- dropout_position_codec fine_tune::worker::tests::` | 0 | 38 passed |
| `cargo test -p jammi-ai --features test-hooks --test it -- gang_coordinator rank_admission` | 0 | 13 passed (3 + 10) |
| `cargo test -p jammi-server --features test-hooks --test it -- gang_coordinator gang_rounds` | 0 | 5 passed (1 + 4); `--nocapture`: `attempt-2 end: published (ordinal 12)` |
| `cargo test -p jammi-encoders --lib -- lora_site::tests` | 0 | 1 passed |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1038 file(s) scanned, all PATH:LINE citations resolve (…; 2 exempt as non-ancestor legacy evidence)` — 47 citations re-anchored in total under this unit's insertions (guide, `gguf.rs`, jammi-bench's README/`torch_finetune_step.py`/`grad_oracle.rs`, `ab_merge.py`) |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | OK |
| `cargo check -p jammi-bench` | 101 | PRE-EXISTING at the base (`9b0dcb57`): `finetune_run.rs:2121 builder.resume(restored)` expects `RestoredCheckpoint`, gets `Option` (U4b's `load_bundle` change), and the bin's `#[cfg(test)]` calls `run_impl(&params, true)` with two args (§2b's witness change) — verified by `git show 9b0dcb57:…` carrying the same lines; this unit's bench diff is four `dropout_seed: <seed>` lines, which compile (the errors are elsewhere). Not fixed here (jammi-bench is the `bench` owner's; outside my files); the merge path's workspace clippy is where it surfaces. |
| Mutations M1–M17 (`mutations.py`, one filtered test each, `git checkout -- <file>` after, `git status --short` clean) | — | 15 red on the first run (§2, first red lines quoted); M11 GREEN (a vacuous row, relabelled: §2/§3); M12 was never assigned |
| `git status --short` at the tip | — | 0 entries |

Not run, per COMMON.md's trimmed set: `cargo doc`, the live-Postgres lane (no jammi-db test
added: `counts_toward_failures` is pure and pinned by the worker table oracle), `merge_path.sh`,
workspace-wide builds. The `gpu_capability` construction sites (`capability_surface.rs`) are
`live-gpu-tests`-gated: edited by the same mechanical rule, compiled by CI's gated-surface
clippy, not here.

#### 5. Commits (`git log --oneline 9b0dcb57..HEAD`; base `feat/500-wave3c` @ `9b0dcb57`)

```
a8ab9667 fix(ai,server,docs): #500 U5b-1b-ii — clippy clean on every target; attempt 2's end on record; citations re-anchored
aaf10c03 feat(ai,lora,encoders,bench): #500 U5b-1b-ii — the seed split wired per rank; a retry binds the recorded training set; the dropout-position gather carried as f32 limbs
6cb183c3 feat(ai,lora): #500 U4b tail — streamed arm at world>1, LoRA init/dropout seed split, per-rank dropout position exercised
acb47662 test(server): #500 U5b-1b-ii — the Peer oracle asserts the attempt's end before joining the member thread, bounded
5004081c feat(ai,server,db,docs): #500 U5b-1b-ii — the coordinator body: topology fan-out, membership → assignment → dispatch → assembly; serveable_world at submit
```

`6cb183c3` is the U4b tail's own commit (`2d46508b` on `feat/500-wave3c`), cherry-picked
onto this branch — not this unit's work: it carries the `_for_rank` builders,
`new_with_base_seeded` and the streamed arm the lead's binding addition wires and this unit's
oracles execute against; it touches none of this unit's files and applied without conflict.
The lead's consolidation takes the four `U5b-1b-ii` commits and drops it (an empty
cherry-pick on a branch that already has it). Tip: `a8ab966725b4b7d3ede7641e554971681b0807b9`.


## 7. U5b-1b-iii

(built after §6)

## 8. U5b-2

(built after §7)

## 9. Pressure round (phase 1, executed at `856ec8dd` before the code landed) — REFINE, eight blocks folded

The design of the four wave-A units was attacked read-only against the base tree; the lead
opened every cited line before relaying. Each block below names the decision that folds it,
sent to the implementer while its unit was still being built; the commit closing each is
recorded at consolidation.

| # | Finding (cited) | Decision folded |
|---|---|---|
| 1 | U4b (b) "bit-for-bit summed adapter gradient vs W=1" is unachievable under f32 reassociation (`batch_bucket.rs:304-315` already documents 1e-5 for the padding half) and is satisfiable only by every rank computing the whole global gradient and skipping the reduce | (b) restated as a pre-registered ε whose discriminator is the W× hazard; bit-identity kept only for W=2-twice and resume-vs-uninterrupted; the "trainable op after the gather" mutation stays the hazard oracle |
| 2 | Zero ≠ absent for AdamW (`adamw.rs:245` skips an absent Var, steps a zero-gradient one; `optimizer.rs:626-633` documents absence as designed) | the reduce reduces the presence set (union across ranks) and restores "absent on every rank ⇒ absent" before `clip_and_step`; never a `world > 1` guard |
| 3 | Both epoch loops end on THIS rank's first empty chunk (`trainer.rs:1400-1404`, `1453-1456`); the optimizer-step boundary counts only non-diverged batches (`2881-2893`, `2903`) | epoch end = the global step count from the partition rule; a zero-row rank takes the step with a synthesized `[0, hidden]` tensor; the divergence decision is flag-reduced BEFORE accumulation so all ranks skip together (at W=1 `Noop` returns the rank's own flags — the W=1 window is unchanged); the one-rank divergence oracle uses the `cfg(test)` poke seam and says so |
| 4 | Hard-negative mining (`1303-1315`), GradCache (`1344-1362`) and the Precomputed arm (`1317-1343`) have no gather story; `gradcache.rs`/`hard_negative_miner.rs` in no scope | typed refusal at the spec admission edge for `world_size > 1` with mining, GradCache or a precomputed loader; their gather is a named follow-up; (b)/(d) oracles use the real loaders |
| 5 | The lifted `Descriptor` drops the round generation `Local` keeps outside it (`local.rs:174-193`); a commit-phase fault leaves one rank un-applied and the next fold is silently wrong | the round index is bound into the agreed descriptor; any commit-phase fault is fatal on every rank |
| 6 | `tests/distributed` already exists behind `live-distributed-tests` with a Postgres+MinIO harness and never gates a PR (`Cargo.toml:335-338`, `distributed.yml:24-26`) — (a),(c),(e),(f) would report green without executing | the Peer-vs-Local fold, deadline, commit-phase fault, descriptor disagreement, corrupted leaf and decode cap move to a hermetic two-process target in the ordinary test lane; fleet-dependent rows join the matrix by name |
| 7 | `RankAdmissionRow` carries no tenant (`jobs_repo.rs:212-250`, deleted in #566 round 2); `get_job` is tenant-filtered; admin scope forbidden — no route to the job's tenant for R2(b) | the admission row carries the row's OWN `jobs.tenant_id` (rebuilt carrier, its own oracle on both backends); the strict resolver pins to it |
| 8 | "Backend SQL clock on both dialects" contradicts the lease discipline: SQLite binds the application clock by design (`lease.rs:13-26`), and the two backends store different representations (`47-56`) | the cooldown reuses `lease_expired_clause`/`lease_deadline_expr`/`lease_now`/`lease_deadline` — no second clock source; the process-skew oracle is Postgres-only; a source-scan oracle pins "no literal clock outside the lease helpers" |

Advisories folded in the same messages: no rung exchange and no pin in U4b (the gather is dim 0
over pooled `[rows, hidden]`; `batch_bucket.rs:96-104`'s rationale rewritten; (c)'s ε covers
padding variance); the `agreement` slot is bound by U4b's canonical key-name digest; the
`BlockingCall` witness is a per-call trait argument owned by U5b-1b-i, threaded through U4b's
single collective helper at consolidation; an older resume bundle reads as no checkpoint (warn),
and W=1's byte-identity property is the artifact, not the bundle; the decode cap is per service
(both services on the peer listener, and the client recv side), pinned at exactly
`max_message_bytes`; classification logits are gathered by moving `classify()` into batch
construction (`TrainingBatch::Classification` reshaped, `data.rs` in scope); the
`[worker] world_size` → `local_ranks` rename touches the config field and accessors only and
the guide block is an ADD; stale premises corrected (U5a-1 froze no round messages;
`get_result_table_for_tenant` is deleted); the run_rank seam is pinned — U5a-2 owns admission,
the holder CAS and the hold loop; U5b-1b-i's round machinery lives in its own module and is wired
into the inbound arm at consolidation.

## 10. Gates (the merge path)

`bash ci/scripts/merge_path.sh` on the consolidated tip with `JAMMI_TEST_PG_URL` pointing at a
local PostgreSQL 16 in the CI lane's shape — run ONCE by the lead, never per implementer; the
phase-5 oracle dispatched after every other stage is green; only `docs/rigor/**` committed
after it.
