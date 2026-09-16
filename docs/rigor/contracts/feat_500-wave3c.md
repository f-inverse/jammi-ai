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

Also on this branch, outside the numbered order: C1 (the cookbook session-close gate for the
non-pytest lanes, issue #539), #574 (a malformed lease timestamp as a row fact on both backends),
U7b-A2b (the two-host cluster driver, its label-only workflow and the id-secrecy scan — once
"filed, not scheduled"; built here because the wave's cluster-leg artifact cannot exist without
it), and the three `main` defects of §9a. The pod-leg and cluster-leg artifacts are produced by
live RunPod runs ON THIS BRANCH and committed as their own `artifact(...)` commits under the
registry's rules. The pod leg's is committed (§8d). The cluster leg ran once here (workflow
35055952682, the `run-cluster` label at `2e43af22`) and exited 75: no SECURE data center offered
`NVIDIA A100-SXM4-80GB` at `MEDIUM` for the CLUSTER product (every one `LOW`; the same for H100 and
H200 when the lead read the catalog), nothing billed. Its artifact is committed from the first
capacity-green run — `workflow_dispatch` on `main` after this merges, or a re-label here — and plan
row C9 stays open on that run alone; the driver's per-data-center threshold is kept (A1 in its
header: co-placement needs ONE data center, and the account-wide figure does not establish it).

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
construction populated `collective`/`local_ranks` from `session.inner_config().worker` at
claim time — U4b's stage, where the config was the only source. Since §6 decides the run's
topology before this site, it records the topology the run executes at (§8d, the phase-5
oracle's finding).

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


## 2c. The pod-leg artifact producer (U4b's pod-leg obligation, plan row B8; landed as three commits, originals `6a9ced97`/`b7df5ff0`/`27d52e94`)

The implementer's contract, folded by the lead: ε = 2e-4 registered in its own commit first (its sha re-anchored to this branch's cherry-picked registration commit `e7440afd`, an ancestor of every later tip); the artifact writer is the one function the pod run calls, proven against rule (k) on CPU with a red that came from the pre-registration rule itself; per-EPOCH loss deltas (the only granularity an integration test can observe; the registry requires none). **Pod-leg run 3 (workflow 35053688974, pod krqjd2cks5zqhg, 1 pod × 2 A100-SXM4-80GB, 2026-09-16T04:04Z, measured tree `02eb3943`): the property holds on hardware.** Two same-seed W=2 runs over `Nccl` produced byte-identical adapter digests (`71dd7958…`), every per-epoch loss delta against the W=1 × 2B reference was 0.0 under ε = 2e-4, `verdict: pass`, `status: GREEN`; the artifact `crates/jammi-kernels/artifacts/cuda-runs/2026-09-16-500-u4b-gang-pod-02eb3943-a100-sxm4.json` passes the registry checker with full history and is committed on this branch (plan row B8). The JOB was red: the producer's two shape oracles run the checker on the pod, whose depth-1 clone had no history for the ancestry rule — fixed in `ci/scripts/runpod_gpu_gang.sh` (a blobless `--unshallow` after the clone); run 4 is the green job.

#### 1. Scope shipped

- **`crates/jammi-ai/tests/gpu_capability/gang_pod_leg.rs`** (NEW). Two-commit
  shape, per the brief's rule (k) ordering requirement:
  1. `6a9ced97614890d8c50880b33b3bd70d2e2a67d1` — registers
     `GANG_POD_LEG_EPSILON` (2e-4) and its derivation alone, before any test
     reads it and before this leg's own use of it exists.
  2. `b7df5ff0` — the rest of the module: the `#[test]
     gang_pod_leg_two_ranks_over_nccl_reproduce_and_match_w1`, the pure
     artifact writer (`GangPodArtifact`/`build_gang_pod_artifact`/
     `write_gang_pod_artifact`), and the hermetic `synthetic_artifact_tests`
     module — plus `GANG_POD_LEG_EPSILON_REGISTERED_SHA` (commit 1's own
     sha, learned only once it existed).
- **`crates/jammi-ai/tests/gpu_capability/main.rs`**: added `mod gang_pod_leg;`.
- **`docs/maintainer/dev-gpu.md`**: the gang-leg "The artifact" paragraph now
  names the producer test (it previously described the artifact appearing
  with no stated writer).
- No other file touched (`ci/scripts/check_cuda_run_artifacts.py`'s rule (k)
  registry did not need a new field — every field the pod leg's schema
  requires already exists in `GANG_POD_FIELD_REGISTRY`).

##### Deviations from the brief, with the code cited (the lead opens the cited
code before accepting these)

1. **Per-EPOCH, not per-step, `gang.per_step_loss_delta`.** The brief's item
   2 asks for `per_step_loss_delta[] = |loss_w2[t] − loss_w1[t]| per global
   step`. The CPU-hermetic oracle this mirrors
   (`gather_exactness_w2_matches_w1_within_pre_registered_epsilon`,
   `crates/jammi-ai/src/fine_tune/trainer.rs:14997`) reads per-step loss via
   `TrainingLoop::compute_loss_gathered`/`encode_chunk`
   (`crates/jammi-ai/src/fine_tune/trainer.rs:507` — `TrainingLoop`'s
   `target`/`base_model`/etc. fields are all PRIVATE, and neither
   `compute_loss_gathered` nor `encode_chunk` is `pub`). This unit's scope is
   `crates/jammi-ai/tests/gpu_capability/**` only; `trainer.rs` is out of
   scope (shared-declaration-adjacent for this wave, and the brief itself
   says "No other file"). The only per-run-progress granularity an external
   integration test CAN observe is the per-EPOCH `avg_train_loss`
   `harness::loss_capture` already captures off the trainer's own
   `tracing::info!("Epoch complete", …)` event — confirmed at
   `crates/jammi-ai/tests/gpu_capability/harness.rs:434-567` (the exact
   mechanism `fine_tune_learns.rs`'s P2 already uses,
   `crates/jammi-ai/tests/gpu_capability/fine_tune_learns.rs:88-90`). Named
   in the module doc's "Deviations" section. The registry field itself does
   not encode a granularity requirement — `_gang_check_delta_series`
   (`ci/scripts/check_cuda_run_artifacts.py:1182`) only requires a non-empty
   list of finite numbers — so this is a wording deviation, not a schema
   violation.
2. **No `Collective` trait import**, unlike `gang_nccl.rs`'s module-level
   `use jammi_ai::fine_tune::collective::Collective;`
   (`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:100-101`). That import
   exists there because `assert_gang_checks` calls collective methods
   (`all_reduce_sum`, `all_gather`, `all_reduce_max_flags`, `barrier`,
   `is_aborted`) DIRECTLY on a `Nccl` value
   (`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:267-343`), and those
   methods are trait methods, not inherent ones. `gang_pod_leg.rs` never
   calls a collective method directly — training runs entirely through
   `RankContext`/`TrainingLoop::run`
   (`crates/jammi-ai/src/fine_tune/trainer.rs:322-341` for `RankContext`,
   `:1084` for `run`), which invoke the collective internally. The one place
   this module constructs an `Arc<dyn Collective>` is `Arc::new(nccl_rank)`
   handed to `RankContext::new(collective: Arc<dyn Collective>, ..)`
   (`crates/jammi-ai/src/fine_tune/trainer.rs:340`), whose parameter type
   already names the trait — the unsized coercion type-checks without a
   local `use` (confirmed: `cargo check --features live-gpu-tests` is clean
   with no `Collective` import anywhere in this file).
3. **Own copies of small (<20-line) helpers**, never edits to files this
   unit does not own:
   - `serial_cuda_device_or_require`/`second_cuda_device_or_require` mirror
     `gang_nccl.rs`'s same-named functions
     (`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:111-149`), which are
     private to that module (no `pub`) and that file is out of this unit's
     scope (owned by a concurrent implementer per the brief's file list).
   - `claimed_job` mirrors `trainer.rs`'s `test_fixtures::claimed_job`
     (`crates/jammi-ai/src/fine_tune/trainer.rs:7306-7348`), `pub(super)` —
     not reachable across the crate boundary from an integration test, and
     `trainer.rs` is out of scope.
   - `ephemeral_artifact_store`/`ephemeral_hub_source` mirror
     `gguf_quantized_gpu.rs`'s private helpers of the same shape
     (`crates/jammi-ai/tests/gpu_capability/gguf_quantized_gpu.rs:831-853`).
   All four are direct structural copies of already-proven, already-tested
   patterns (not derivations), duplicated only because the originals are
   private/out-of-file-scope. Each is cited above so the lead can diff it
   against its original.
4. **`load_tiny_bert_on` loads directly onto a named CUDA device** via
   `ModelResolver` + `CandleBackend::load(&resolved, &DeviceConfig{gpu_device,
   devices: vec![gpu_device], ..})` rather than the async `ModelCache`'s
   scheduler-budget path (`session.model_cache().get_or_load_on(..)`) —
   mirrors `gguf_quantized_gpu.rs`'s own precedent for placing a
   `LoadedModel` on a specific device
   (`crates/jammi-ai/tests/gpu_capability/gguf_quantized_gpu.rs:821-929`).
   `ModelCache::get_or_load_on` needs a `[gpu] devices` config declaring
   BOTH cuda:0 and cuda:1 up front and per-device budget bookkeeping this
   leg's tiny fixture does not need; the resolver+backend path is the
   simpler, already-proven route to "this exact model, on this exact
   device".

#### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| The full non-`cuda` arm of `gang_pod_leg.rs` (module doc, non-gated helpers, the writer, the hermetic tests) type-checks under `--features live-gpu-tests` (no `cuda`) | `cargo check -p jammi-ai --features live-gpu-tests --test gpu_capability` — exit 0, no warnings | N/A (a compile gate, not a numeric property) — confirmed the two `dead_code` warnings this file first produced (`gang_pod_artifact_filename`/`write_gang_pod_artifact`, only called from the `cuda`-gated real test) by running the check BEFORE wiring `write_gang_pod_artifact` into `synthetic_artifact_tests::run_checker_written`, then confirmed clean after |
| `build_gang_pod_artifact`'s clean, well-formed `pass` output (written through `write_gang_pod_artifact`, the SAME function the real pod run calls) satisfies EVERY field `ci/scripts/check_cuda_run_artifacts.py`'s rule (k) POD registry requires | `synthetic_artifact_tests::synthetic_pass_artifact_satisfies_rule_k` — `cargo test -p jammi-ai --features live-gpu-tests --test gpu_capability gang_pod_leg -- --test-threads=1`: **ok** | ORGANIC red, captured before commit 2 landed: with `GANG_POD_LEG_EPSILON_REGISTERED_SHA` == the CURRENT `git rev-parse HEAD` (uncommitted commit 2 not yet made, so HEAD was still commit 1 itself), the test failed with `` `gang.epsilon.registered_sha` equals the artifact's own `git_sha` (6a9ced97614890d8c50880b33b3bd70d2e2a67d1) — the ε must be registered BEFORE the tree that was measured, never in the same commit `` — i.e. the pre-registration ordering check genuinely bites when ε and the measured tree coincide. Committing (advancing HEAD past the registration commit) turned it green. |
| `build_gang_pod_artifact`'s well-formed `fail` output (reason set, status `RED`, an unequal digest pair) ALSO satisfies rule (k) cleanly — a failing run is representable, not refused | `synthetic_artifact_tests::synthetic_fail_artifact_satisfies_rule_k` — **ok** | Same organic red as above (this test hit the identical epsilon-ordering finding before commit 2 landed); green after |
| Dropping `gang.epsilon` from an otherwise-clean artifact is NAMED by the checker (the non-vacuous half: a checker that never fires would pass the shape test vacuously) | `synthetic_artifact_tests::missing_epsilon_is_named_by_the_checker` — **ok**, asserts a finding containing `gang.epsilon` | The test body itself IS the mutation (`artifact["gang"].remove("epsilon")`) executed every run; its own assertion is the red-proof that the checker bites — confirmed by reading the checker's actual returned finding during a manual run: `` `gang.epsilon` is missing — the pre-registered tolerance, its derivation, and the commit it was registered at… `` |
| `GANG_POD_LEG_EPSILON_REGISTERED_SHA` is a well-formed 40-lowercase-hex sha | `synthetic_artifact_tests::epsilon_registered_sha_is_well_formed` — **ok** | Standing shape check (a malformed constant would fail this directly); the ancestry/ordering claim itself is exercised by the two tests above via the real checker, not re-derived here |
| The pod-leg training/artifact-writing property itself (U4b acceptance (a)/(c): reproducible digest pair, per-epoch loss within ε) | **UNCOVERED here** — see §3 | N/A — no CUDA toolchain in this authoring environment; every line touching a CUDA device is `#[cfg(feature = "cuda")]`-gated and did not compile/run in this session |

Gate-command exit codes double as an executed "does the whole non-cuda arm
still assemble" oracle; see §4.

#### 3. Uncovered

- **The real pod-leg property itself** (byte-identical digest pair across
  two W=2 runs; the per-epoch loss delta within `GANG_POD_LEG_EPSILON`
  against a real W=1×2B reference) — no CUDA toolchain in this authoring
  sandbox (no `nvcc`/`libnccl`). Everything that exercises it is
  `#[cfg(feature = "cuda")]`-gated and did not run here. The lead runs the
  leg on the pod (`ci/scripts/runpod_gpu_gang.sh`) and returns the
  compiler's/driver's output on a failure, per the brief.
- **Whether `load_tiny_bert_on`'s `DeviceConfig{gpu_device: 1, devices:
  vec![1], ..}` actually places the model on `cuda:1`** rather than `cuda:0`
  — this is exactly the kind of off-by-one only a real second device can
  refute; untestable without CUDA.
- **`nvidia_smi_field`'s `--id=<ordinal>` flag actually scoping to the named
  GPU** — untestable without `nvidia-smi`/a real device; a malformed query
  degrades to `"unknown"` rather than panicking (by design — the module doc
  states this is descriptive metadata, not a correctness input), but that
  degrade path itself is unexercised here too.
- **The dropout-position gather at `lora_dropout > 0`** is out of this
  leg's scope by the brief's own acceptance (`lora_dropout = 0.0`
  pinned) — U4b tail's `dropout_seed_split_*` oracles own that property.

#### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo fmt -p jammi-ai -- --check` | 0 | Clean after one `cargo fmt -p jammi-ai` pass (formatting-only diffs: line wraps, arg lists) |
| `cargo fmt --all -- --check` | 0 | Also clean (no other crate touched) |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` (brief's literal command) | 0 | Does **not** compile `gpu_capability` at all — that target's `required-features = ["live-gpu-tests"]` (`crates/jammi-`ai/Cargo.toml` (line 355 at the unit tip)) is not satisfied by `test-hooks` alone, so this command is a no-op for the file this unit ships. Reported honestly rather than cited as coverage it is not. |
| `cargo clippy -p jammi-ai --test gpu_capability --features live-gpu-tests,test-hooks --all-targets -- -D warnings` (the gated-surface variant that DOES compile this file, matching CI's own gated-surface clippy step) | 0 | Clean, no warnings |
| `cargo check -p jammi-ai --features live-gpu-tests --test gpu_capability` | 0 | Clean, no warnings (the two `dead_code` warnings hit during authoring, fixed — see §2) |
| `cargo test -p jammi-ai --features live-gpu-tests --test gpu_capability gang_pod_leg -- --nocapture --test-threads=1` | 0 | **5 passed; 0 failed.** `gang_pod_leg_two_ranks_over_nccl_reproduce_and_match_w1` passes VACUOUSLY here (`skip_without_gpu!()` returns immediately — no CUDA feature compiled in this session); the 4 `synthetic_artifact_tests` pass FOR REAL (see §2) |
| `python3 ci/scripts/check_cuda_run_artifacts.py --self-test` | 0 | `cuda-run-artifacts self-test: OK` — the checker's OWN self-tests (including every rule (k) fixture) still pass; this unit did not touch the checker |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `1042 file(s) scanned, all PATH:LINE citations resolve` (same 2 pre-existing EXEMPT legacy citations as before this unit's changes; nothing new) |

#### 5. Commits

`git log --oneline 109b9c39..HEAD` (this unit's branch base, `feat/500-wave3c @ 109b9c39`
— `main..HEAD` would additionally list the whole wave-3c stack this branch
carries, since the local `main` ref in this worktree is behind that base):

```
27d52e94 docs(maintainer): #500 U4b — name the pod-leg gang artifact's own producer
b7df5ff0 test(ai): #500 U4b — the pod-leg gang artifact producer over Nccl
6a9ced97 test(ai): #500 U4b — register the pod-leg gang epsilon ahead of the test and run that measure against it
```

Tip: `27d52e9425e5720657f5b348ce0a16db5da7dcfc`


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
`worker.rs` (line 6154 at the unit tip) — the same worker.rs-exclusion blocks it. Not attempted;
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
   field at every call site (including `worker.rs` (line 6154 at the unit tip), excluded from this
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
`cargo clippy -p jammi-ai` reports at `trainer.rs` (line 369 at the unit tip)–389` on the base; (b) `jammi-db`'s lib-test
target: the db slice's `config/tests.rs::distributed_config_loads_independently_of_worker` and
`DistributedConfig`'s two intra-doc links still named `WorkerConfig::world_size`, which U4b S8
renamed to `local_ranks` (`E0609` at `config/tests.rs` (line 818 at the unit tip); the links would fail the docs lane).
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
| Second seam: `jammi-db`'s lib-test target compiles and the `[distributed]`/`[worker]` independence test holds under the renamed knob | `cargo clippy -p jammi-db --all-targets --features test-hooks -- -D warnings` exit 0; `cargo test -p jammi-db --features live-postgres-tests,test-hooks --lib config::tests::distributed_config_loads_independently_of_worker` 1 passed | the base's `E0609` at `config/tests.rs` (line 818 at the unit tip) is the executed red (`int-scratch/clippy-db.log`) |

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
| `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1 port 54329/jammi_test cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- assembly_outcome migrations gang_rank_admission --test-threads=1` | 0 | 73 passed (22 `::postgres` arms executed); 0 failed |
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
guide line 498 re-anchored (my `runtime.rs` insertion moved it: `runtime.rs` (line 2068 at the unit tip) →
`runtime.rs` (line 2086 at the unit tip)); `docs/guide/src/operability.md` (`max_message_bytes` row: every listener,
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
| P21 (g) | A `Collective` verb reached from a runtime worker thread does not compile (a `BlockingCall` cannot cross into a `tokio::spawn`ed future; no witness exists to pass; the constructor is private), while the same verb from `spawn_blocking` compiles | `it::blocking_call::a_collective_verb_from_a_runtime_worker_thread_does_not_compile` (trybuild: `tests/ui/{verb_from_a_runtime_worker_thread, verb_without_a_witness, mint_outside_a_spawn_site}.rs` red with pinned `.stderr`; `tests/ui_pass/verb_from_spawn_blocking.rs` green) | M18: the witness made `Send` (`PhantomData<()>`) → RED: `panicked at trybuild 1.0.121's own `run.rs`, line 103:13` — 1 of 4 tests failed |
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
| P1 | For every `RunRank` call, the admission decision (every I-GANG determinant incl. the world>1 conjunct and freshness) is complete BEFORE the holder is consulted; a refused call never touches the holder; `Admitted` is emitted only after a successful CAS (a busy slot is `Err(Status)`, never a stream). | SRV `gang_service::run_rank_refuses_unavailable_at_once_while_a_loop_job_runs` (busy slot → `Unavailable` + no reason recorded; an absent job under a busy slot → `FailedPrecondition`/`NotFound`) | M3: slot consulted before the row read → M3 (holder checked before the row read): `gang_service.rs` (line 2192 at the unit tip): assertion left == right failed` (an absent job answered `Unavailable`, not `FailedPrecondition`) |
| P2 | For every `row.world_size > 1` call and NEVER for `world_size == 1`: the pair must be filled, the row's tenant must parse, the strict resolver under the ROW's tenant must find a `ready` row, its sidecar must verify `artifact == training_set_ref`; every determinant refuses the one fixed `FailedPrecondition`; keyed on the row's decoded fact, never `assign.world`. | SRV `gang_service::run_rank_refuses_when_assign_world_mismatches_row_world_size` (both directions + controls: direction (a)'s control reaches `TrainingSetPairMissing`, direction (b)'s control is `Admitted`), `::run_rank_refuses_a_training_set_another_tenant_owns`, `::run_rank_refuses_a_null_tenant_training_set_for_a_tenant_bound_job`, `::run_rank_refuses_world_gt_one_when_the_sidecar_predates_the_leaf_inventory`, `::run_rank_world_two_own_tenant_training_set_is_admitted_held_and_parks_no_body` (the admitting control) | M1: world gate deleted → M1 (`if false && assign.world != world_size`): `gang_service.rs` (line 1713 at the unit tip): assertion left == right failed: must record WorldMismatch specifically, not the pair determinant`; M2: tenant forced `None` → M2 (`let tenant = None` before the resolver): `gang_service.rs` (line 2019 at the unit tip): the strict resolver must not resolve the NULL-tenant row (a later determinant refusing instead would mean it did)`; M7: leaf-less sidecar accepted → M7 (`Ok(None) => Verified`): `gang_service.rs` (line 1583 at the unit tip): every fixture this function builds must refuse` (the leaf-less sidecar admitted) |
| P3 | Non-disclosure: all sixteen determinants refuse with pairwise-identical `(code, message)` on the wire; the `test-hooks` seam distinguishes every one. | SRV-plain + SRV `gang_service::run_rank_refusal_is_non_disclosing_across_every_determinant`; SRV `::run_rank_last_refusal_reason_distinguishes_every_determinant`; `every_gang_refusal_reason` re-validated by an exhaustive match | (shape inherited from U5a-1, unchanged: any one arm interpolating a reason reds the pairwise oracle naming the pair; not re-executed this round — the sixteen-row scenario table is the executed novelty) |
| P4 | Tenant is derived from the job row, never accepted: no caller metadata is read; the strict resolver never resolves another tenant's or a NULL-tenant row for a real tenant, under ambient admin scope or not; the resolution site refuses ambient admin scope before the resolver runs. | SRV `gang_service::run_rank_never_reads_a_caller_supplied_tenant` (`jammi-session-id` metadata ignored → `Admitted`), `::resolution_site_refuses_under_admin_scope_before_the_strict_resolver_runs` (control `Verified` outside, `AdminScopeRefused` inside); DB `result_tables::get_result_table_for_tenant_never_matches_a_null_tenant_row_for_a_real_tenant::{sqlite,postgres}`, `::get_result_table_for_tenant_resolves_only_the_owning_tenant::{sqlite,postgres}`; the derivation claim in `tenant_isolation_oracle::gang_service_is_unimplemented_on_the_public_listener` | M11: strict predicate relaxed → M11 (predicate relaxed to `OR tenant_id IS NULL`): `result_tables.rs` (line 550 at the unit tip): the strict resolver must never match a NULL-tenant row for a real tenant` (sqlite arm); M2 (above) |
| P5 | The admission row carries the row's own tenant as text and the filled pair on both backends; the read is infallible on content (a garbage tenant is `Ok(Some)`). | DB `gang_rank_admission::get_job_for_rank_carries_the_tenant_text_and_the_filled_pair::{sqlite,postgres}` | M12: tenant parsed in the mapper → M12 (`parse::<TenantId>().expect(..)` in the mapper): `jobs_repo.rs` (line 2087 at the unit tip): panicked` (the garbage-tenant arm is no longer `Ok(Some)`) (sqlite arm) |
| P6 | Holder lattice (c2'): `Free` admits; `JobRun`/another `Rank` refuse at once; the same job at an equal attempt refuses, at a greater attempt supersedes in place and the elder's drop leaves the successor's hold; `ClaimProbe` is waited ≤ one bound then admits-if-freed or refuses; a hold's drop frees only its own cell. | AI `host_admission::{free_admits_a_rank_and_dropping_the_hold_frees_the_slot, a_job_run_or_another_rank_refuses_at_once, the_same_job_at_a_greater_attempt_takes_the_slot_and_the_elder_leaves_it, a_claim_probe_is_waited_out_then_admitted_if_freed_or_refused_at_the_bound}`; SRV `gang_service::{run_rank_refuses_unavailable_at_once_while_a_loop_job_runs, run_rank_waits_out_a_claim_probe_then_admits_if_freed_or_refuses_unavailable}` (test-hooks), `::a_held_rank_refuses_other_ranks_and_the_same_job_at_a_greater_attempt_takes_the_slot` (plain) | M9: `RankHold::drop` frees regardless of identity → M9 (`RankHold::drop` frees any `Rank`): `host_admission.rs` (line 197 at the unit tip): the superseded elder's drop must not free the successor's slot` |
| P7 | Exclusion (d2', OPS D6): the loop moves the holder `Free→ClaimProbe→JobRun→Free` around every claim (the flip at the hold site, the prologue a probe); an idle loop never calls `claim_next` while a rank is held and claims the moment it is freed; an inline `run_now` is outside the exclusion; a `JobRun`-holding peer refuses a rank, an idle peer admits. | AI `host_admission::{the_claim_loop_moves_the_holder_free_probe_run_free, an_idle_loop_never_claims_while_a_rank_is_held, an_inline_run_now_never_touches_the_holder}`; the gauge `health::gauges::in_flight_gauge_is_one_during_a_loop_claimed_job_and_zero_during_run_now`; the reshaped `jobs_shutdown` suite (20 rows) | M8: flip at `claim_next`'s `Some` arm → M8 (`job_running()` at `claim_next`'s `Some` arm): `host_admission.rs` (line 306 at the unit tip): the claim committed but the hold is not registered: still a probe` (read `JobRun`); M10: probe ignores a held slot → M10 (`probe_claim` overwrites a held slot): `host_admission.rs` (line 365 at the unit tip): claim_next must not be called while a rank is held` |
| P8 | OPS D10: RELEASE's abort decision reads the holder kind, never a count — a `Rank` beside an idle loop is never loop work (cooperative `Stopped`, hold untouched, phase `Releasing`); `JobRun` aborts now. | AI `host_admission::release_and_stop_beside_a_held_rank_exits_cooperatively_and_flips_the_phase`; `jobs_shutdown::release_with_the_loop_paused_in_the_claim_to_hold_prologue_self_releases` (the prologue still self-releases) | M13: a held `Rank` treated like `JobRun` → M13 (`holder == Holder::Free` → a held `Rank` aborts now): first executed against the ORIGINAL oracle (an idle loop) it stayed GREEN — the loop had exited at 2a before 2e ran, so the arms were indistinguishable; the oracle was rewritten to park the loop after reclaim, re-executed green, then the same mutation re-executed: `host_admission.rs` (line 457 at the unit tip): a held Rank is not loop work: 2e waits for the cooperative exit, never aborts` (`Aborted` ≠ `Stopped`) |
| P9 | The hold loop has exactly four arms and every end is one stream event: `Cancel` → `Cancelled`; a second `Assign` → `InvalidArgument` TRAILER (K2); phase leaving `Running` → `Drain` at once (both via the session cell and via the real server shutdown path, worker-less); park bound → `NoBody` after ≥ two ticks. | SRV `gang_service::{run_rank_cancel_on_an_admitted_stream_ends_cancelled, run_rank_second_assign_on_an_admitted_stream_is_invalid_argument, run_rank_held_session_ends_drain_when_the_host_drains, run_rank_held_session_ends_drain_on_server_shutdown, run_rank_every_i_gang_determinant_satisfied_is_admitted_held_and_parks_no_body}` | M4: drain arm never fires → M4 (phase arm replaced by `pending()`): `gang_service.rs` (line 570 at the unit tip): expected Aborted{Drain}, got Aborted { reason: NoBody }` (parks to `NoBody` instead) |
| P10 | Re-verification (i2'): the three ends are pairwise distinct on the wire, in scope and in the count rule; a row fact moving → `Refuted`; the catalog faulting → `Unavailable`; THIS host's store faulting (`Storage`/`Io`) → `StoreUnavailable`; the artifact's sidecar no longer verifying → `Refuted` (never `StoreUnavailable`). | SRV `gang_service::{run_rank_held_session_ends_refuted_when_the_row_no_longer_holds, run_rank_held_session_ends_unavailable_when_the_catalog_faults, run_rank_held_session_ends_store_unavailable_when_this_hosts_store_faults, run_rank_held_session_ends_refuted_when_the_sidecar_stops_verifying}`; lib `grpc::gang::tests::reverify_ends_are_pairwise_distinguishable_in_reason_scope_and_count` | M5: store fault classified `Refuted` → M5 (`StoreFault → ReverifyEnd::Refuted`): `gang_service.rs` (line 570 at the unit tip): expected Aborted{StoreUnavailable}, got Aborted { reason: Refuted }` |
| P11 | Terminal-write scope (g2'): the peer names no `jobs` writer (the set derived from `jobs_repo.rs` itself) and every end (cancel, K2 trailer, drain, refuted, unavailable, store-unavailable, park, supersession) leaves the job row byte-identical to its pre-admission snapshot. | SRV `gang_terminal_write_oracle::the_gang_handler_names_no_jobs_writer` (+ its two self-tests); the `row_facts` before/after equality in every hold-loop row above | M6: `fail_job(` named as code in `run_rank` → M6 (`let _ = stringify!(fail_job());` in `run_rank`): `gang_terminal_write_oracle.rs` (line 283 at the unit tip): gang.rs names the jobs writer fail_job( as code` |
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
| `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1 port 54329/jammi_test cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- --test-threads=1 get_result_table_for_tenant get_job_for_rank_carries_the_tenant_text` | 0 | 6 passed (3 tests × sqlite + postgres); 0 failed |
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
| `cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- assembly_outcome migrations docs_config_fences --test-threads=1` | 0 | 53 passed (20 `assembly_outcome`, 30 `migrations`, 3 `docs_config_fences`), 0 failed, `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1 port 54329/jammi_test` set (live scratch Postgres 16, reachable on 54329) |
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
| (d) For every spec, `world_size > serveable_world` is refused typed naming `[distributed] max_world_size` from configuration alone (no catalog handle exists on the type); every `world_size <= serveable_world` passes this arm regardless of this host's devices, on both embedded entrances, the `jobs` table unchanged on refusal | `it::rank_admission::a_count_past_the_serveable_world_is_refused_from_configuration_alone` (pure); `::every_unservable_rank_count_is_refused_at_both_submit_entrances` (the one-device session admits W=2 within a serveable world of 2 and writes exactly one row) | M1: `if false && world_size > self.serveable_world` (spec.rs) → RED `rank_admission.rs` (line 439 at the unit tip): two ranks on a serveable world of one: ()` (the W=2 spec was admitted) |
| (d) A W=2 job on a one-device host with serveable world 2 REACHES ASSEMBLY: topology `Peer{2}`, the CAS writes the pair, the empty listing ends `ShortListed` recorded on the row (failures 0, `next_assembly_after` set), the lease released (`releases 1`, lease NULL), status `running`, no error — RED at base (the base refuses W=2 at submit on one device) | `it::gang_coordinator::a_two_rank_job_beyond_this_hosts_devices_reaches_assembly_and_lands_short_listed` | M5: `record_assembly_outcome` skipped (`outcome.filter(\|_\| false)`) → RED `gang_coordinator.rs` (line 551 at the unit tip): ShortListed is cooled down: Row { .. next_assembly_after: None .. }`; M7: the release skipped (`if false && !counts`) → RED `gang_coordinator.rs` (line 555 at the unit tip): assertion left == right failed: an uncounted outcome hands the lease back at once` (`releases 0`) |
| The CAS `Moved` arm writes nothing: a claim whose attempt moved leaves every assembly/identity/lease/status column byte-identical | `it::gang_coordinator::a_moved_claim_exits_the_coordinator_body_with_no_write` | M4: `Ok(Moved) => {}` (the body proceeds past a moved claim) → RED `gang_coordinator.rs` (line 606 at the unit tip): assertion left == right failed` (the row changed: ShortListed recorded and the lease released on a row the attempt did not own) |
| (e) Rank assignment is a pure function of the sorted listing: every one of the 24 permutations of a 4-member listing yields the same `rank → instance_id`; ranks are `1..world` over the first `world−1` in byte order; a listing shorter than `world−1` is `ShortListing`, never a smaller gang | `lib fine_tune::worker::tests::rank_assignment_is_a_pure_function_of_the_sorted_listing` | M3: the sort in `assign_ranks` removed → RED `worker.rs` (line 7778 at the unit tip): assertion left == right failed: rank r is the r-th member in instance_id byte order; the surplus member is unused` |
| (e) Through the REAL `GangServer::run_rank`: a member whose slot is busy answers `Unavailable` — the attempt ends `MemberRefused` naming rank 1/member, recorded `Unavailable` (failures 0, cooled), lease released, nothing terminal, no slot held; the NEXT attempt (after the cooldown, through the loop's reclaim + claim) RE-LISTS the same sorted listing, is admitted, builds the `Peer`, runs rank 0, records `Success` (cooldown reset), ends the member session (`Cancel` → slot `Free`); the run's end is `Published` with bytes == `LocalGang` reference, or exactly the pinned streamed refusal (deviation 6) — RED at base (W=2 unsubmittable; no body) | `jammi-server it::gang_coordinator::a_member_answering_unavailable_ends_the_attempt_cooled_and_the_next_attempt_relists_and_runs_the_gang` (`test-hooks`) | M8: the dialer never installed (`runtime.rs`) → RED `gang_coordinator.rs` (line 495 at the unit tip): the real handler's Unavailable ends the attempt naming the member: this host cannot coordinate: no gang listener is mounted in this process (no member dialer installed)`; M11: `end_members` sends no `Cancel` (`filter(\|_link\| false)`) → see the M11 row below |
| The dropout-position gather carries every `u64` exactly in a dtype every collective accepts: each position round-trips through four `f32` limbs (`0`, `0xFFFF`, `2^16`, `2^24+1`, `2^53+1`, `u64::MAX`), a malformed row (length, a non-integral or over-wide limb) is refused; a `Peer` gang's epoch boundary no longer faults | `lib fine_tune::trainer::dropout_position_codec_tests::every_u64_position_round_trips_through_four_f32_limbs`; the server oracle's attempt 2 past round 10; `gang_determinism_oracle::{w2_twice_is_byte_identical_with_dropout, resume_after_a_kill_matches_an_uninterrupted_run_with_dropout}` (Local, unchanged bytes) | M17: the limb shift dropped on decode (`value \|= raw as u64`) → RED `trainer.rs` (line 15534 at the unit tip): assertion left == right failed` (`2^16` decodes as `1`) |
| A retry of a job whose row names a training set BINDS that table (ready, digest-verified) and its CAS is `Reused`, so the next attempt reaches the run instead of reading its own retry as a moved claim | the server oracle's attempt 2 (`ends[1]` is past dispatch; `training_set_ref` unchanged across attempts) | M16: the bind skipped (`Some(pair) if false`) → RED `gang_coordinator.rs` (line 562 at the unit tip): attempt 2 must reach the run (Published or a run failure), got: the claim moved before assembly (no write)` |
| Every `CoordinatorEnd` maps to exactly one `AssemblyOutcome` and only `Moved` maps to none: `assembly_outcome` is a total match (a new variant is a compile error), one sample per `ordinal` in `0..VARIANTS` (a variant without a sample reds), each sample's row as documented; `MemberAborted{reason}` one-to-one over every frozen `AbortReason` with `Unspecified → Unavailable`; the settle rule: exactly `Refuted`/`AllRootDivergent` count | `lib fine_tune::worker::tests::every_coordinator_end_records_exactly_one_assembly_outcome_and_only_moved_writes_nothing` | M2: `ShortListed => AssemblyOutcome::Unavailable` → RED `worker.rs` (line 7717 at the unit tip): assertion left == right failed: short listing: 0 fresh member(s) where 1 are needed`; M13: `counts_toward_failures` keyed on `CooldownOnly` (jobs_repo.rs) → RED `worker.rs` (line 7741 at the unit tip): assertion failed: O::Refuted.counts_toward_failures()` |
| The topology is decided from `(world_size, local_ranks)` alone: `W<=1 → Single`; `1<W<=L → Local{W}`; `W>L → Peer{W}`; no hybrid | `lib fine_tune::worker::tests::topology_is_decided_from_world_size_and_local_ranks_alone` | M6: `world_size < local_ranks` (W=2,L=2 → Peer) → RED `gang_coordinator.rs` (line 643 at the unit tip): assertion left == right failed` (topology `Peer{2}`, not `Local{2}`; the graph job is then refused at the coordinator's edge) |
| The local fan-out: a `local_ranks = 2` host runs a W=2 job through the REAL `run_spec` as `Local{2}` (no coordinator body), completes, and publishes an adapter byte-identical to a U4b-shaped `LocalGang` run (each rank on `spawn_thread`, direct `TrainingLoop::run`) over the same sampled rows, config, seed, base model | `it::gang_coordinator::a_local_ranks_two_host_fans_a_two_rank_job_out_through_run_spec_and_publishes_the_gangs_bytes` | M9: rank 0 given a single-rank context (`None`) while rank 1 joins the 2-rank gang → RED `gang_coordinator.rs` (line 652 at the unit tip): assertion left == right failed: Row { status: "failed" .. }` — rank 1 timed out at the 10 s rank deadline, the run failed, nothing published (10.16 s wall vs 0.3 s healthy); M6 (above) |
| The seed split through the REAL `run_spec` (`local_ranks = 2`, `lora_dropout = 0.3`): the two ranks' dropout seeds differ (rank 0 = `config.seed`), every head layer's `dropout_run_seed` is its rank's own, the ranks' pre-step adapter weight digests are equal — and the published bytes still equal the `LocalGang` reference built per rank at the same seeds | `it::gang_coordinator::a_local_ranks_two_host_fans_…` (the seed assertions in the same oracle) | M14: `build_projection_head_for_rank(.., &FineTuneConfig { seed: dropout_seed, ..config }, .., dropout_seed)` → RED `gang_coordinator.rs` (line 689 at the unit tip): assertion left == right failed: both ranks start from byte-identical adapter weights` |
| At the tower seam (`LoraSite::wrap`, every non-BERT tower's sites): two sites with the same `seed` and different `dropout_seed`s have byte-identical `lora_a` and report each their own `dropout_run_seed`; a different `seed` moves `lora_a` | `jammi-encoders lib lora_site::tests::the_site_keys_init_by_seed_and_the_masks_by_dropout_seed` | M15: `lora_site.rs` passes `self.lora.seed` for both seeds → RED `lora_site.rs` (line 216 at the unit tip): assertion left == right failed` (`dropout_run_seed` = `Some(7)`, not `Some(100)`) |
| A member's `Aborted{reason}` mid-round is recorded TYPED on the coordinator's link for that rank (`Peer::member_aborts == [(1, StoreUnavailable)]`), beside the permanent fault; a member records none | `lib fine_tune::collective::peer_tests::a_member_that_aborts_its_session_faults_the_coordinator_naming_the_reason` (extended) | M10: `session_abort` never recorded in `Link::recv` → RED `peer_tests.rs` (line 1006 at the unit tip): assertion left == right failed: the coordinator records the member's abort reason typed, on rank 1's link` (`[]` vs `[(1, 5)]`) |
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
| `cargo check -p jammi-bench` | 101 | PRE-EXISTING at the base (`9b0dcb57`): `finetune_run.rs` (line 2121 at the unit tip) builder.resume(restored)` expects `RestoredCheckpoint`, gets `Option` (U4b's `load_bundle` change), and the bin's `#[cfg(test)]` calls `run_impl(&params, true)` with two args (§2b's witness change) — verified by `git show 9b0dcb57:…` carrying the same lines; this unit's bench diff is four `dropout_seed: <seed>` lines, which compile (the errors are elsewhere). Not fixed here (jammi-bench is the `bench` owner's; outside my files); the merge path's workspace clippy is where it surfaces. |
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


## 7. U5b-1b-iii (landed as one commit after its rebase onto `a072ad59`; original `2264c28e`)

The implementer's contract, folded by the lead after checking: `LeaseHolder` is a separate type so a `Rank` has nothing to pass to a job-row writer (the executed compile error); every writer site takes it; the body runs through the real hold loop and `RankEvent::Outcome` is consumed on receipt with the coordinator publishing only when every member's digest equals its own; the agreement digest is bound at the top of `run`. Two rebase-time decisions accepted on the evidence: the chaos rows drive rank 1 through the REAL body (the tap survives for body-less sessions only), and the resume pin is WITHDRAWN — U5b-2's executed chaos rows have a retired attempt's successor gang resume from the shared root, which the listing's root-identity predicate guarantees every member shares (issue #543 re-scoped to fleets without a shared root). Plan row U5b-1b-iii carries the dated correction.

Worktree `wt-u5b1biii`, branch `unit/u5b1biii`, base `feat/500-wave3c` @ `82f80b84`. Every path
is repo-relative; every oracle named was executed on the tip (§4); every mutation in §2 was
applied to the committed tree, run through ONE filtered test (or one `cargo check`), and
reverted (`git checkout -- <file>`, `git status --short` empty afterwards — `u5b1biii-scratch/
mutations.py`, logs `mut-M*.log`). One `--features` set per crate for the whole session
(`jammi-ai`, `jammi-server`, `jammi-db`: `test-hooks`), `RUSTC_WRAPPER=sccache`,
`CARGO_TARGET_DIR=…/targets/u5b1biii`.

#### 1. Scope shipped

**`crates/jammi-ai/src/fine_tune/role.rs` (new; `fine_tune::role`)** — the single-writer rule
as types. `LeaseHolder { LoopClaimer, Coordinator }` (the ONE writer of an attempt's row:
today's in-process path incl. a `Local` gang's rank 0, which never traverses the coordinator
body; rank 0 of a `Peer` gang) and `RunnerRole { Holder(LeaseHolder), Rank { rank } }` (what a
body runs AS; `implied_by_rank`, `rank()`, `lease_holder() -> Option<LeaseHolder>` — `None` for a
`Rank`, which is what makes every durable write unreachable from a rank body). The derived shape
the brief left open: the brief's `RunnerRole::{LoopClaimer, Coordinator, Rank{rank}}` would let a
`Rank` be PASSED to a job-row writer and refused at run time; splitting the holder out as its own
type makes the job-row writers take `LeaseHolder`, so a `Rank` has no value to pass and the write
is a compile error (§2, the executed M12).

**`crates/jammi-ai/src/fine_tune/worker.rs`**
- Every job-row-writing site on the run path takes a `LeaseHolder` (REQUIRED, by value): the
  module doc's new "Runner roles and the job-row writers" section carries the per-site table,
  derived by `grep -n 'record_failed(\|finish_job_with_model(\|persist_acceleration_report(\|register_job_hold_or_release(' worker.rs`
  minus doc lines — 17 rows: `register_job_hold_or_release(.., holder)` (the lease-hold
  registration, the `Releasing` self-release arm, the holder accounting `job_running()` — row 1;
  the `test-hooks` recorder `note_lease_holder` sits here), the 13 production `record_failed(holder,
  ..)` sites (rows 2–9, 11–15: `run_claimed_job_under` ×3, `publish_and_finalize` ×5,
  `run_claimed_compute_job` ×5), the finalize CAS inside `publish_and_finalize(holder, ..)` (row
  10), `persist_acceleration_report(holder, ..)` and its two markers `mark_acceleration_
  {not_applicable,undetermined}(holder, ..)` behind `compute_and_persist_acceleration_report(..,
  role: RunnerRole, ..)` — a `Rank` computes and discards, `role.lease_holder()` is the gate (row
  16), and `coordinate`'s `record_assembly_outcome`/`release_job_lease` as `LeaseHolder::
  Coordinator` (row 17). The two test call sites of `record_failed` pass `LoopClaimer`.
- `lease_holder_for(spec, local_ranks) -> LeaseHolder` (pub): the ONE derivation — `Coordinator`
  exactly when a column-source `fine_tune` decides `TopologyDecision::Peer` over this host's
  `[worker] local_ranks` (the SAME `decide` call `run_spec` makes), `LoopClaimer` otherwise (a
  single rank, a `Local` gang, every `graph_fine_tune`, a context predictor). Derived once in
  `run_claimed_job_under` right after the spec deserialises and threaded: `run_spec(.., holder)`,
  `train_fine_tune(.., holder, topology)` (rank 0 gets `RunnerRole::Holder(holder)`, every other
  local rank `Rank { rank }`), `publish_and_finalize(holder, ..)`, the post-run `record_failed`s.
  `coordinate`/`assemble_and_run` run as `LeaseHolder::Coordinator` by their own name.
- `RunFineTuneParams { role: RunnerRole }` replaces `rank: u32` (`rank = role.rank()`);
  `run_fine_tune_blocking` records `note_runner_role` and sets `.runner_role(role)` on the trainer
  builder.
- `bind_training_source(session, table, columns, task, detected, common) -> TrainingSource`:
  `run_spec`'s FineTune source selection (the `whole_set_arm` predicate, the `Resident` eager
  read-back with reservation, the `Streamed` set with F5's pre-pass and F3's vocabulary) factored
  out, called by rank 0 in `run_spec` and by the rank body — the ranks' loaders derive from one
  definition. `bind_recorded_training_set` returns `(TrainingSetTable, MaterializationManifest)`
  (the member verifies its leaves against the sidecar it just verified) and documents its error
  classes: every refusal is `JammiError::FineTune`, every other variant is a read faulting.
- **The rank body**: `MemberAssignment { job_id, attempt, rank, world, coordinator_instance_id,
  tenant, training_set_ref, training_set_location, spec_json }`, `RankOutcome { Trained {
  artifact_digest }, Failed { reason }, Aborted(AbortReason) }`, `pub async fn run_member_rank(
  session, assignment, link: MemberLink, cancel) -> RankOutcome` (+ `member_rank_body` under the
  row's tenant scope): decode the spec (a column-source `fine_tune` only; the spec's `world_size`
  must equal the assignment's), `bind_recorded_training_set` (refusal → `Aborted(Refuted)`;
  `Storage`/`Io` → `Aborted(StoreUnavailable)`; other → `Aborted(Unavailable)` — the hold loop's
  re-verification classes), `verify_partition_leaves` over the sidecar's whole inventory BEFORE the
  first collective (a bad leaf → `Aborted(fault.abort_reason())`), `bind_training_source`, load the
  base model, `Peer::member(rank, world, link, device, max_message_bytes).with_timeout(
  rank_timeout_secs)`, `RankContext::new(peer, for_gang(rank, world, batch, BlockByGlobalBatch))`,
  then `run_fine_tune_blocking` on `BlockingCall::spawn_blocking` — the THIRD production minting
  site (documented in `collective/mod.rs` on `mint`/`spawn_blocking`) — as `RunnerRole::Rank {
  rank }` with `worker_id = coordinator_instance_id` (the lease holder's id, which a `Rank` never
  writes under). Its end: `adapter_files_digest(training.artifact_dir)` → `Trained`; a typed
  error/panic/join error → `Failed`. `adapter_files_digest(dir)` (pub): SHA-256 over every regular
  file directly in `dir` in name order as `name`, NUL, `len` (LE u64), bytes — exactly
  `publish_artifact`'s file set, computed by the ONE function on both sides.
- **`Outcome` consumed**: `assemble_and_run` step (7) `reconcile_member_ends(&coordinator,
  &artifact)` after `train_fine_tune` returns `Ok`: `Peer::collect_member_ends` on a blocking
  thread, then per rank — `Trained` with rank 0's own `adapter_files_digest` → ok; a differing
  digest or `Failed { reason }` → `CoordinatorEnd::TrainingFailed` (the job's own failure,
  recorded `failed`, nothing published); `Aborted(raw)` → `MemberAborted { rank, reason }`;
  `Ended(why)` → `LinkFault`. `end_members` (now step 8) runs whichever way. The `test-hooks`
  recorder `note_member_end`/`member_ends_for(job_id)` keys on the output model id.
- **The resume pin — WITHDRAWN at the rebase (§6, deviation 12).** It was built as
  `CoordinatorEnd::ResumeRefused` + `ArtifactStore::has_resume_checkpoint` + an assembly-time
  refusal on the pre-rebase tip `74b7e5e0`; on the consolidation tip the successor gang of a
  retired attempt RESUMES from the shared store (U5b-2's executed chaos rows), and the real
  member body resumes through `run_fine_tune_blocking`'s `discover_resume` exactly as rank 0
  and a `Local` rank do. Nothing of the pin remains on the tip.
- `training_test_hooks`: `note_lease_holder`/`lease_holders_for(job) -> Vec<(attempt,
  LeaseHolder)>`, `note_runner_role`/`runner_roles_for(job) -> Vec<RunnerRole>`, `note_member_end`/
  `member_ends_for(job) -> Vec<(rank, Debug)>`, `fail_member_outcome(job, reason)`/`take_member_
  failure` (the fault injection at the body's natural end).
- Unit tests: the total-table oracle now asserts exactly `{Moved, ResumeRefused}` record nothing;
  `the_lease_holder_is_the_coordinator_exactly_when_a_fine_tune_decides_peer` (a 3×4 grid over
  `fine_tune` and `graph_fine_tune`).

**`crates/jammi-ai/src/fine_tune/trainer.rs`** — `TrainingLoop.role: RunnerRole`;
`TrainingLoopBuilder::runner_role(role)`; `build()` derives the role from the rank context when
unset (`RunnerRole::implied_by_rank(rank_ctx.rank())` — rank 0 the loop claimer, the pre-role
default, so every existing caller changes zero bytes) and REFUSES a role whose `rank()` differs
from `rank_ctx.rank()` (typed: "the role and the rank must agree"); `save_resume_checkpoint`
(after the lockstep gather) and `save_epoch_checkpoint` write only when `self.role.
lease_holder()` is `Some` — the runner-role gate lives HERE, `store/artifact.rs` stays
role-agnostic. **The agreement binding**: `RankContext::bind_agreement(&self, names)` →
`Collective::bind_agreement(canonical_vars_digest(names))`, called at the top of
`TrainingLoop::run` right after `optim_param_names` (the varmap is final, the first collective is
ahead) over `optimizer::sorted_trainable_var_names(&self.varmap)` (new; the SAME lock and sort
as `sorted_trainable_vars`). New test module `runner_role_and_agreement_oracle` (3 oracles, §2).

**`crates/jammi-ai/src/fine_tune/collective/{mod,noop,local,nccl,peer}.rs`** —
`Collective::bind_agreement(&self, digest: String) -> Result<()>` on the trait (object-safe;
`Noop`/`Nccl` accept and ignore — a gang of one has no peer, NCCL carries no descriptor;
`Local`/`Peer` bind once through the shared `bind_agreement_once` over a `OnceLock<String>`:
the same digest again is a no-op, a different one a typed error naming both; `with_agreement`
builders kept, now over the `OnceLock`). `peer.rs`: `Link.outcome: Option<Outcome>` recorded in
`recv` beside `session_abort` (`Inbound::outcome`); `MemberEnd { Trained { artifact_digest },
Failed { reason }, Aborted(i32), Ended(String) }` (exported); `Peer::collect_member_ends(&self,
&BlockingCall) -> Vec<(u32, MemberEnd)>` — coordinator only, each link read under the gang
deadline until an `Outcome`/`Aborted`/close, an already-recorded frame honoured without a read.
`mod.rs`: three production minting sites documented.

**`crates/jammi-db`** — `RankAdmissionRow.spec: String` (the `spec` column verbatim, already
selected by `get_job_for_rank`'s statement; the rank body reconstructs its job from it — no spec
travels on the wire); `ArtifactStore::has_resume_checkpoint(tenant, job_id) -> bool` (the resume
manifest's existence, never a fetch).

**`crates/jammi-server/src/grpc/gang_rounds.rs`** — `RoundInbox { frames, forwarder:
AbortHandle }`; `RoundInbox::sever(self)` aborts the link's outbound forwarder and closes the
inbound side, so no frame of the body's rides the response stream after the session's terminal
event (and the forwarder's clone of the event sender goes with it, which is what closes the
stream). **`gang.rs`** — `run_rank` spawns `run_member_rank` for every `world_size > 1` session
(the identity is the world>1 conjunct's product) with the session's link, `row.spec`, the
row's tenant and pair, BEFORE `Admitted` is queued; a `world_size == 1` session keeps (or, under
`test-hooks`, offers) its link and parks. `HeldSession { body: Option<JoinHandle<RankOutcome>>,
body_cancel: Arc<AtomicBool> }`; `SessionEnd { Aborted, Outcome(outcome::Result), Violation }`;
the hold loop has FIVE arms of which a session takes four — the body's end (`Trained`/`Failed`
→ `RankEvent::Outcome`; the prologue's `Aborted(reason)` → `Aborted`; a task that ended without
an outcome → `Outcome{Failed}`) OR the park bound (`if body.is_none()`), beside inbound, drain
(untouched — U5b-2's) and the tick; a foreign end sets `body_cancel`; every end severs the
inbox. `outcome_event`. The `test-hooks` tap `take_member_links` now hands out body-less
(`world_size == 1`) sessions' links only. Module doc and `hold`'s doc restated.

**Tests** — `crates/jammi-server/tests/it/gang_coordinator.rs` rewritten around the real body
(two oracles over the production `peer_addr` listener; the LocalGang reference kept);
`gang_service.rs`: `world_two_ready` now materialises a REAL training set (`materialize_
projection_table` over a registered `pairs` CSV, under the tenant) and claims a REAL `fine_tune`
spec (`world_two_spec_json`, tiny_bert) so an admitted session's body reconstructs a runnable
job; `admitted_world_two` opens rank 1 (rank 0 is the coordinator's own); the world-2 park oracle
became `…_is_admitted_runs_its_body_and_never_parks`; new `run_rank_body_refuses_a_partition_
whose_leaf_does_not_verify_as_store_unavailable`; module doc restated. `gang_rounds.rs`: the
world-2 tap oracle (`a_round_through_the_real_run_rank_handler_reaches_the_member_link_and_
equals_local`) deleted — its seat is the body's; its property (the member's fold over the real
hold loop equals `Local`) is the coordinator oracle's byte equality; the body-less trailer oracle
kept. `crates/jammi-ai/tests/it/gang_coordinator.rs`: the K4 pinned row
`a_single_rank_job_runs_as_the_loop_claimer_and_never_traverses_the_coordinator`, the resume-pin
row `a_peer_job_with_a_resume_checkpoint_is_refused_at_assembly_and_fails_typed`, role
assertions on the `Local` fan-out row.

**Docs (same commit set)** — `docs/maintainer/MAINTAINER-GUIDE.md`: §2.8a (the five-arm hold
loop, the body's end, the sever), §2.8c (three minting sites; the tap's scope), §2.8d (steps
7–8, the terminal write on receipt, the resume pin, `ResumeRefused → nothing`, the oracles), NEW
§2.8e (the rank body and the runner roles, the agreement binding), the Train-flow sentence;
`docs/guide/src/security.md` (admit-and-hold: the body, the single writer); 11 `PATH:LINE`
citations re-anchored by identifier under my insertions (worker.rs ×5, trainer.rs ×1,
artifact.rs ×5).

##### Deviations from UNITS.md / the brief, with the reason and the code

1. **`RunnerRole` is two types, not one enum.** The brief's `RunnerRole::{LoopClaimer, Coordinator,
   Rank{rank}}` on every writer would make "a `Rank` attempts a write" a run-time arm. Splitting
   `LeaseHolder` out (`role.rs`) makes the writers take a type a `Rank` cannot produce
   (`RunnerRole::lease_holder() -> Option<_>`): M12 is the executed compile error.
2. **`TrainedOutcome.artifact_digest` is the MEMBER's own adapter digest, not rank 0's published
   digest carried by a last round.** The member digests bytes it wrote (its local adapter files),
   never bytes it did not; every rank holds identical weights after the last step (DESIGN §4), so
   the coordinator's equality check (`reconcile_member_ends`) is the gang's own attestation that it
   converged to ONE artifact, and a differing digest fails the attempt rather than publishing over
   it. Carrying rank 0's digest to members would need a collective after the trainer's last save —
   a round the trainer does not make.
3. **The coordinator WAITS for every member's end before publishing** (`assemble_and_run` step 7).
   §6 published on rank 0's own result alone; "the terminal write on receipt" is now literal:
   `Published` requires every member's `Trained` with the coordinator's digest; a member's
   `Failed` is `TrainingFailed` (terminal `failed`, nothing published — the server's second
   oracle), an `Aborted` after the run is `MemberAborted` through the table.
4. **A body-bearing session never parks** — the hold loop's park arm is gated `if body.is_none()`
   (U5a-2's "exactly four arms" pin restated as five-of-which-four). A body's bounds are the gang
   deadline on every round wait and the re-verification tick. `NoBody` is now exactly what its
   proto comment says: a body-less session's park.
5. **The body's `Outcome` is emitted by the hold loop, not sent through the link.** The body
   RETURNS its `RankOutcome` (the fifth arm's `JoinHandle`), and the hold loop emits the one
   terminal event — so "every end is ONE stream event" stays a hold-loop property and a foreign
   end can never be followed by the body's own frame (the inbox is severed).
6. **The body's prologue classifies its refusals as the tick does** (`Refuted`/`StoreUnavailable`/
   `Unavailable`), reusing `bind_recorded_training_set`'s error classes — so the two world-2
   re-verification oracles (a store fault / a stripped sidecar manufactured AFTER admission) read
   the same wire event whichever of the body's bind or the tick sees the fault first.
7. **`ResumeRefused` records NO assembly outcome** (a second `None` beside `Moved`; §6's "only
   `Moved` writes nothing" restated). No assembly happened and the job goes terminal `failed`;
   recording `Success` would reset a cooldown on a row that is about to be `failed`.
8. **`RankAdmissionRow.spec`** (jammi-db) and **`ArtifactStore::has_resume_checkpoint`** (jammi-db)
   are outside the brief's files_in_scope: the body reconstructs its job from the row (no spec on
   the wire — DESIGN §4), and the pin is a manifest-existence probe (never a bundle fetch).
9. **The world-2 server fixtures now carry a REAL training set and a REAL spec** — with a body at
   admission, a synthetic `{"common":{"world_size":2}}` row would end every world-2 session
   `Outcome{Failed}` at once (an undeserialisable spec) and the re-verification rows could not be
   held. The fixtures are the coordinator's own producer (`materialize_projection_table`), so they
   are what a member is admitted against in production.
10. **The `gang_rounds.rs` world-2 tap oracle is deleted**, not kept behind a flag: the tap's seat
    is the body's. The tap survives for body-less sessions only (the trailer oracle needs a link
    the session would otherwise just hold).
11. **`whole_set_arm` in `run_spec` moved into `bind_training_source`** (the F6 predicate is
    still the ONE decision, now shared with the member); the guide's citation re-anchored to it.

#### 2. Properties

Lanes: AI-LIB = `cargo test -p jammi-ai --features test-hooks --lib -- <filter>`; AI-IT =
`cargo test -p jammi-ai --features test-hooks --test it -- <filter>`; SRV = `cargo test -p
jammi-server --features test-hooks --test it -- <filter>`. Mutation ids are `mutations.py`'s;
each was applied to the committed tip, run through the ONE named test, reverted.

| Property (quantified) | Executed oracle (path::name; lane) | Executed mutation that reds it (first red line) |
|---|---|---|
| (compile-time) For every job-row-writing site on the run path — the lease-hold registration with its `Releasing` self-release arm and holder accounting, every `record_failed` (13 production sites, 2 test sites), the finalize CAS in `publish_and_finalize`, the acceleration-report write and its two markers, the coordinator's assembly-outcome/lease-release writes — the call is unreachable without a `LeaseHolder`; a `Rank` body holds `RunnerRole::Rank`, which has no `LeaseHolder` to pass, so a job-row write from the rank body does not compile | the compiler on the tip: `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` exit 0 (every site compiles WITH the parameter; the base's signatures, without it, would not — a missed site is `E0061`) | M12: a `record_failed(RunnerRole::Rank { rank }, ..)` call added to `member_rank_body` → `cargo check -p jammi-ai --features test-hooks` RED: ``error[E0308]: mismatched types --> `worker.rs` (line 5643 at the unit tip):9 … RunnerRole::Rank { rank } … expected `LeaseHolder`, found `RunnerRole`` (`record_failed`'s first parameter is `holder: LeaseHolder`)` |
| (derivation) For every `(kind, world_size in 1..=4, local_ranks in 1..=3)`: the attempt's `LeaseHolder` is `Coordinator` exactly when a column-source `fine_tune` decides `TopologyDecision::Peer`, `LoopClaimer` otherwise; a `graph_fine_tune` never coordinates; `W == 1` is the `LoopClaimer` on every kind and host (K4) | AI-LIB `fine_tune::worker::tests::the_lease_holder_is_the_coordinator_exactly_when_a_fine_tune_decides_peer` | M5: `Peer → LoopClaimer` in `lease_holder_for` → RED: ``worker.rs` (line 8568 at the unit tip): assertion left == right failed: fine_tune W=2 L=1` (`left: LoopClaimer`)`; the same mutation against SRV `gang_coordinator::a_member_whose_body_fails…` (the hold registered as the loop claimer on a Peer attempt) → RED: ``gang_coordinator.rs` (line 647 at the unit tip): assertion left == right failed` (`left: [(1, LoopClaimer)]` — the Peer attempt's hold registered as the loop claimer)` |
| K4, pinned: a `world_size == 1` `fine_tune` through the REAL claim → `run_claimed_job` → `run_spec` is today's loop path — topology `Single`, the hold registered as `LoopClaimer`, exactly one rank ran as `Holder(LoopClaimer)`, no coordinator end, no assembly listing, no identity pair written, the row `completed` with a published adapter | AI-IT `gang_coordinator::a_single_rank_job_runs_as_the_loop_claimer_and_never_traverses_the_coordinator` | M4: `TopologyDecision::decide` answering `Peer` for `world_size <= 1` (the job enters the coordinator body) → RED: ``gang_coordinator.rs` (line 751 at the unit tip): assertion left == right failed` (`left: Some(Peer { world: 1 })` — the single-rank job entered the coordinator body)` |
| The role and the rank agree by construction on every `TrainingLoop`: unset → derived from the rank context (rank 0 the loop claimer, the pre-role default; rank `r` → `Rank{r}`); an explicit role whose `rank()` contradicts the context is refused at `build`, the matching one builds | AI-LIB `fine_tune::trainer::runner_role_and_agreement_oracle::a_runner_role_that_contradicts_the_rank_context_is_refused_at_build` | M3: the agreement guard made vacuous (`\|\| true`) → RED: ``trainer.rs` (line 8769 at the unit tip): the role contradicts the rank: ()` (a holder role on a rank-1 context built)` |
| The trainer's durable-write gate, on a REAL two-rank `Local` gang through the full `TrainingLoop::run`, each rank over its OWN artifact store: every rank takes the epoch-boundary dropout gather (both complete), the holder (rank 0) writes the resume checkpoint into its store, the `Rank` (rank 1) writes NOTHING into its own | AI-LIB `…runner_role_and_agreement_oracle::a_rank_of_a_gang_never_writes_the_resume_checkpoint_and_the_holder_does` | M1: the role gate in `save_resume_checkpoint` removed → RED: ``trainer.rs` (line 8677 at the unit tip): assertion left == right failed: the lease holder (rank 0) alone writes the resume checkpoint; the Rank writes nothing` (`left: [true, true]` — rank 1's store holds a bundle)` |
| The agreement binding: two ranks whose canonical trainable-variable layouts differ (the same `Var`s, shapes and order; different NAMES on rank 1) are refused at the first reduce on BOTH ranks, the refusal naming BOTH digests — never folded | AI-LIB `…runner_role_and_agreement_oracle::two_ranks_whose_canonical_layouts_differ_are_refused_naming_both_digests` | M2: the `bind_agreement` call at the top of `run` removed (bind nothing) → both ranks complete `Ok` (the positions line up, the fold proceeds silently) → RED: ``trainer.rs` (line 8717 at the unit tip): rank 0 must refuse: the ranks' canonical layouts differ, yet the fold proceeded`` |
| (superseded at the rebase, §6) The total exit table and the resume pin — M6/M7 were executed on `74b7e5e0` (`worker.rs` (line 8488 at the unit tip) … left: Some(Success)`; `gang_coordinator.rs` (line 815 at the unit tip) … short listing: 0 fresh member(s)`) and are WITHDRAWN with the pin; on the tip the total table is the base's 13-variant one (`assembly_outcome`: only `Moved → None`; U5b-2's `lease_settlement` oracle covers every ordinal) and a `Peer` gang's resume is EXECUTED by §8's chaos rows through the real body (every successor attempt resumes from epoch 1's bundle and publishes the reference bytes) | AI-LIB `fine_tune::worker::tests::every_coordinator_end_records_exactly_one_assembly_outcome_and_only_moved_writes_nothing`; SRV `gang_chaos::*` (4 rows, §6) | the chaos rows' own executed mutations are U5b-2's (§8); no new mutation of mine on this row |
| `RankEvent::Outcome` end to end through the REAL `GangServer::run_rank` on the production `peer_bind` listener and the REAL rank body: attempt 1 (member busy) ends `MemberRefused`, cooled, released, hold as `Coordinator`; attempt 2 re-lists, admits, the body runs rank 1 as `Rank{1}` beside rank 0 as `Holder(Coordinator)` (never `LoopClaimer`), the member's session ends `Outcome{Trained{digest}}` which the coordinator reads (`member_ends_for == [(1, Trained{..})]`), the attempt is `published` (ordinal 12), the row `completed` through the loop's own finalize, the published adapter byte-identical to the U4b-shaped `LocalGang` reference, the slot free | SRV `gang_coordinator::a_member_answering_unavailable_ends_the_attempt_cooled_and_the_next_attempt_runs_the_real_rank_body_to_a_published_artifact` | M8: the digest comparison inverted (`!= own` passes) → the equal digest is a mismatch → `TrainingFailed`, nothing published → RED: ``gang_coordinator.rs` (line 513 at the unit tip): assertion left == right failed: attempt 2 publishes over the real rank body: the run failed: rank 1 ended Trained with adapter digest c3432070… where rank 0's is c3432070…: the gang did not converge to one artifact` (ordinal 11, not 12 — the equal digests were read as a mismatch; on record: the member's digest EQUALS rank 0's on the healthy tree)`; M13: the hold loop maps a `Trained` body end to `Aborted{NoBody}` instead of `Outcome` → the coordinator reads `MemberAborted` → RED: ``gang_coordinator.rs` (line 513 at the unit tip): assertion left == right failed: attempt 2 publishes over the real rank body: rank 1 ended its session: Aborted(NoBody)` (ordinal 9, `MemberAborted`)` |
| The terminal write on receipt, failure arm: a member whose body reports `Outcome{Failed{reason}}` ends the attempt `TrainingFailed("rank 1: …")` (ordinal 11), the row `failed` with that reason (site 4 of the writer table under `Coordinator`), `Success` recorded (assembly proceeded), NO model row registered, nothing published, both roles recorded, the slot free | SRV `gang_coordinator::a_member_whose_body_fails_ends_the_attempt_failed_under_the_coordinator_and_publishes_nothing` | M9: a member's `Failed` ignored by `reconcile_member_ends` → the attempt publishes → RED: ``gang_coordinator.rs` (line 613 at the unit tip): assertion left == right failed: the attempt ends TrainingFailed on the member's Outcome{Failed}: published` (`left: 12` — the attempt published over the failed member)` |
| A body-bearing (`world_size == 2`) session NEVER parks: through the real handler its body reconstructs the job from the row, binds and verifies the training set, loads the model and sends its round-0 contribution (round frames are the ONLY frames on the stream); past the park bound (`LEASE`) no terminal event has been emitted; `Cancel` ends it `Aborted{Cancelled}` with the stream closing right after (no body frame follows), the row untouched, the slot free | SRV `gang_service::run_rank_world_two_own_tenant_training_set_is_admitted_runs_its_body_and_never_parks` | M10: the park arm's `if body.is_none()` guard removed → `Aborted{NoBody}` inside the bound → RED: ``gang_service.rs` (line 2289 at the unit tip): a body-bearing session emits no terminal event at the park bound: RankEvent { event: Some(Aborted(Aborted { reason: NoBody })) }``; M11: no body spawned for a `world_size > 1` session → parks, no round frame → RED: ``gang_service.rs` (line 2289 at the unit tip): a body-bearing session emits no terminal event at the park bound: RankEvent { event: Some(Aborted(Aborted { reason: NoBody })) }` (no round frame ever sent: nothing ran)` |
| The body's own pre-collective verify: a training set whose Parquet bytes were corrupted inside a row group AFTER the sidecar attested them (a fault the sidecar-level tick cannot see) ends the session `Aborted{StoreUnavailable}` from the body's leaf verify, member-scoped, within the prologue, the row untouched | SRV `gang_service::run_rank_body_refuses_a_partition_whose_leaf_does_not_verify_as_store_unavailable` | M15: `verify_partition_leaves` handed no leaves → the body binds, loads and waits at its first collective; no terminal event → RED: ``gang_service.rs` (line 709 at the unit tip): assertion left == right failed: expected Aborted{StoreUnavailable}, got RankEvent { event: Some(Outcome(Outcome { result: Some(Failed(FailedOutcome { reason: "rank 1: DataFusion error: Parquet error: … Unexpected PageType -1406" })) })) }` — without the leaf verify the corrupt bytes reach the streamed pre-pass as an untyped decode failure, never the member-scoped abort` |
| The two world-2 re-verification ends are unchanged with a body alive: a store fault / a stripped sidecar manufactured after admission still end `Aborted{StoreUnavailable}` / `Aborted{Refuted}` whichever of the body's bind or the tick sees it first, the row untouched | SRV `gang_service::run_rank_held_session_ends_store_unavailable_when_this_hosts_store_faults`, `::run_rank_held_session_ends_refuted_when_the_sidecar_stops_verifying` (re-executed over the real training-set fixture) | regression rows (U5a-2's M5 shape); the body's classification arms are by inspection the tick's (`bind_recorded_training_set`'s documented error classes) |
| Every pre-existing hold-loop end (cancel, second Assign, drain ×2, refuted/unavailable, park for `world_size == 1`, busy slot, probe wait, supersession) and every admission determinant is unchanged; the peer names no `jobs` writer | SRV `-- gang` (the whole gang-prefixed set, §4) incl. `gang_terminal_write_oracle` | unchanged oracles |
| W=1 and the in-process `Local` gang are byte-unchanged (the roles derive to today's values; the checkpoint gates fire for the same ranks as `rank != 0` did): the whole pre-existing `fine_tune::` lib suite (incl. U4b's determinism/resume-with-dropout oracles over a `Local` gang), the `Local` fan-out row (now also asserting `[(1, LoopClaimer)]` and roles `{Holder(LoopClaimer), Rank{1}}`), and the loop/shutdown/admission suites that drive `register_job_hold_or_release` under `LoopClaimer` | AI-LIB `fine_tune::`; AI-IT `gang_coordinator jobs_shutdown host_admission acceleration_report fine_tune` (§4) | regression-only (as §6 states it) — a perturbation of the W=1 window reports here first |

#### 3. Uncovered

- **A member's `Aborted{reason}` AFTER the run** (its session refuted/drained between its last
  collective and its `Outcome`): mapped `MemberAborted` through the table (`reconcile_member_
  ends`), executed only through the table oracle — no hermetic oracle manufactures a refutation
  in that window.
- **A `LinkFault` at the outcome read** (`MemberEnd::Ended`: the member's stream closed or fell
  silent past the gang deadline before its `Outcome`): built, not executed — the healthy oracle
  and the injected failure both deliver an `Outcome`.
- **A differing digest from a REAL divergence**: M8 executes the comparison's refusal by
  inverting it; no oracle manufactures two ranks that complete with different adapter bytes
  (the collective's agreement and lockstep oracles are what would have to be broken first).
- **A foreign end while the body is mid-round** (Cancel/Refuted/Drain during training, not at
  the first collective): the never-parks and re-verification oracles end sessions whose body waits
  at its FIRST collective; the sever's effect on a body between rounds (its next verb ends
  `Disconnected`, its `RoundFault` dropped by the aborted forwarder) is executed only there.
- **`body_cancel` at an epoch boundary**: set on every foreign end; the trainer's check is the
  existing one, but every executed foreign end reaches the body through the severed inbox first,
  so the flag's own effect is not separately observed.
- **The rank body's `Aborted(Unavailable)` arm** (a catalog fault binding the training set) and
  its "spec's `world_size` differs from the assignment" `Failed` arm: not manufactured.
- **`Nccl::bind_agreement`** (accept-and-ignore): CUDA-gated, compiled by CI's gated-surface
  clippy, not here.
- **A `Resident` source in a `Peer` member** (`bind_training_source`'s eager arm): unreachable for
  a gang by admission (mining/GradCache refused at `world > 1`), so the member always takes
  `Streamed`; the eager arm is executed by rank 0's W=1/`Local` rows only.
- **Two real processes**: the server oracles run the member's body in the SAME process as the
  coordinator (one engine, one `HostAdmission`; the member dials its own `peer_bind` listener);
  two OS processes over one catalog is the fleet leg (U7b).
- **The member body's resume with a bundle written by a DIFFERENT root** (a fleet whose members
  do not share the coordinator's result root): unreachable by the listing's root-identity
  predicate; the chaos rows resume over one shared root only.

#### 4. Gates

COMMON.md's trimmed set plus clippy on the third crate this unit touches (`jammi-db`); every
command with `RUSTC_WRAPPER=sccache`, `CARGO_TARGET_DIR=…/targets/u5b1biii`. Logs in
`u5b1biii-scratch/`. The tip `74b7e5e0` is `9bbf36a7` (on which every mutation and the clippy
runs executed) plus ONE `cargo fmt` reflow of two statements in
`crates/jammi-ai/tests/it/gang_coordinator.rs` (whitespace only; the amended commit); the
`gang_coordinator` it filter and the citation gate were re-executed on the amended tip.

| Command | Exit | Result |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | clean (`fmt-final2.log`, on the tip) |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | 0 | 0 warnings (`clippy-final-jammi-ai.log`) |
| `cargo clippy -p jammi-server --all-targets --features test-hooks -- -D warnings` | 0 | 0 warnings |
| `cargo clippy -p jammi-db --all-targets --features test-hooks -- -D warnings` | 0 | 0 warnings |
| `cargo test -p jammi-ai --features test-hooks --lib -- runner_role_and_agreement_oracle fine_tune::role fine_tune::worker::tests::every_coordinator_end fine_tune::worker::tests::the_lease_holder` | 0 | 6 passed (`t-ai-lib-2.log`) |
| `cargo test -p jammi-ai --features test-hooks --lib -- fine_tune::` | 0 | 356 passed, 0 failed — the WHOLE pre-existing suite + this unit's rows (the W=1 / `Local` byte-identity regression oracle, incl. U4b's determinism and resume-with-dropout rows) (`t-ai-lib-ft.log`) |
| `cargo test -p jammi-ai --features test-hooks --test it -- gang_coordinator` | 0 | 5 passed (K4 pinned row, resume pin, Local fan-out with roles, ShortListed, Moved) (`t-ai-it-3.log`, on the tip) |
| `cargo test -p jammi-ai --features test-hooks --test it -- jobs_shutdown host_admission acceleration_report fine_tune` | 0 | 90 passed, 0 failed (`register_job_hold_or_release`/`record_failed`/the report writers under `LoopClaimer`, unchanged) (`t-ai-it-reg.log`) |
| `cargo test -p jammi-server --features test-hooks --test it -- gang` | 0 | 54 passed, 0 failed (2 `gang_coordinator` + 1 `gang_rounds` + every `gang_service`/oracle row; `t-srv-4.log`) |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1039 file(s) scanned, all PATH:LINE citations resolve (…; 2 exempt as non-ancestor legacy evidence)` — 11 citations re-anchored by identifier under this unit's insertions (on the tip) |
| Mutations M1–M13, M15 + M5s (`mutations.py`; one filtered test or one `cargo check` each; `git checkout -- <file>` after; `git status --short` clean after every one) | — | 15 red on the first run whose mutation compiled (§2, first red lines quoted); M7 and M11 were first written as non-exhaustive `match` arms (a compile error, not the property's red) and re-written to compile before the quoted runs (`mutations-summary.log`, `mutations-summary-2.log`) |
| `git status --short` at the tip | — | 0 entries |

Not run, per COMMON.md's trimmed set: `cargo doc`, the live-Postgres lane (no `jammi-db` test
added — `RankAdmissionRow.spec` is read by the existing `get_job_for_rank` statement and
exercised by every world-2 server row; `has_resume_checkpoint` is exercised by the resume-pin
and every `Peer` assembly row over the file store), `merge_path.sh`, workspace-wide builds,
`jammi-bench`. `Nccl::bind_agreement` is CUDA-gated (CI's gated-surface clippy).

#### 5. Commits (`git log --oneline 82f80b84..HEAD`; base `feat/500-wave3c` @ `82f80b84`)

```
74b7e5e0 feat(ai,server,db,docs): #500 U5b-1b-iii — the rank body over the real hold loop; the runner-role writer split; RankEvent::Outcome consumed on receipt; the resume pin; the agreement binding
```

One commit (19 files + `role.rs`, +2683/−690). Tip: `74b7e5e01a856e35650b4aba29a0cee9ff6fdf8e`.

Seams for the lead (U5b-2, the watchdog): the hold loop's drain arm is untouched (its
`break SessionEnd::Aborted(AbortReason::Drain)` is the one line U5b-2's `Released` emission
replaces); a foreign end's body handling (`body_cancel` + `RoundInbox::sever`) is shared by every
non-body end, so a `Released` end needs no new wiring. `run_claimed_job_under`'s lease-lost arm,
the `lost` flag and `release_job_lease` were not edited beyond the `LeaseHolder` parameter
threaded through `record_failed`/`register_job_hold_or_release`.

#### 6. Rebase onto a072ad59 (`feat/500-wave3c` moved from `82f80b84` while this unit was built)

`git rebase a072ad59` in the unit worktree; six files conflicted. Each conflict, its two sides,
and the resolution — by intent, never by hunk:

| File | Conflict | Resolution |
|---|---|---|
| `crates/jammi-ai/src/fine_tune/worker.rs` (1/3) | `coordinate`'s doc: U5b-2's released-vs-failed settlement paragraph vs mine ("every end but `Moved` and `ResumeRefused`", the `Coordinator` role sentence) | U5b-2's paragraph kept whole; my one sentence ("Runs as `LeaseHolder::Coordinator` — writer table row 17") appended; the `ResumeRefused` mention dropped (see the pin below) |
| `worker.rs` (2/3) | `coordinate`'s settle arms: U5b-2's `lease_settlement(&end)` → `Release`/`Expire`/`Untouched` then a two-tuple `match (end, artifact)` vs my three-tuple match with the `ResumeRefused → Failed` arm and the per-outcome release | U5b-2's shape taken verbatim (the settlement is the ONE lease rule now; my `coordinate` never edited the settlement, only threaded the holder) — `LeaseHolder::Coordinator` is bound above the outcome record as before |
| `worker.rs` (3/3) | the `tests` module: U5b-2's `every_coordinator_end_settles_its_lease_by_the_released_vs_failed_split` vs my `the_lease_holder_is_the_coordinator_exactly_when_a_fine_tune_decides_peer` (both inserted before `rank_assignment_is_a_pure_function…`) | both kept, U5b-2's first; the shared closing braces re-stitched |
| `crates/jammi-db/src/catalog/jobs_repo.rs` (2) | `RankAdmissionRow`: #574's `lease: LeaseFact` (replacing `lease_live`/`remaining`) vs my `spec: String` beside the old pair; the constructor likewise | both fields: `spec` (mine) beside `lease` (#574's); `lease_live`/`remaining` gone with #574 |
| `crates/jammi-ai/tests/it/gang_coordinator.rs` | U5b-2's `a_live_building_training_set_row_left_by_a_crashed_coordinator_is_never_met_by_the_successor` vs my K4 row and resume-pin row (same insertion point) | U5b-2's row + my K4 row kept; my resume-pin row DELETED (the pin is withdrawn, below) |
| `crates/jammi-server/tests/it/gang_coordinator.rs` (4) | U5b-2 made thirteen helpers `pub(crate)` (incl. `RankEnv`/`rank_env`/`try_run_rank`/`run_rank` for the chaos sibling) vs my rewrite (which deleted the test-thread rank-1 helpers and rewrote the oracle around the real body) | my rewrite, with every helper the chaos sibling still needs `pub(crate)` (`pairs`, `gang_config`, `tiny_bert_model`, `write_pairs_csv`, `two_rank_spec`, `Row` + fields, `row`, `published_adapter_bytes`, `file_store`, `reference_rank0_adapter_bytes`); `RankEnv`/`rank_env`/`try_run_rank`/`run_rank` are gone — their consumer (`gang_chaos.rs`'s test-thread rank 1) is gone too (below) |
| `docs/guide/src/security.md` | #574's Postgres-only bullet (client-side lease decode) vs my admit-and-hold bullet | both: my admit-and-hold text (the body, the single writer) above #574's bullet |
| `docs/maintainer/MAINTAINER-GUIDE.md` | §2.8d's post-record paragraph: U5b-2's settlement + watchdog + chaos prose vs my `ResumeRefused → nothing` | U5b-2's prose taken whole; my resume-pin sentences in §2.8d's step list replaced by the resume-through-the-shared-store sentence (below); my §2.8e and the rest untouched |

Landed upstream and folded without conflict: `LeaseSettlement`/`lease_settlement` (a TOTAL
match — no `ResumeRefused` arm is needed since the variant is withdrawn; `VARIANTS` is 13 again,
U5b-2's settlement oracle covers every ordinal); the cancel fix's
`checkpoint_before_spawn_blocking(job_id)` (auto-merged in `train_fine_tune`); #574's
`LeaseFact` in `gang.rs` (my `run_rank` edits sit below it); the guard fixes.

**The resume pin is WITHDRAWN (deviation 12).** The brief's "`world_size > 1` resume is
REFUSED at assembly (resume-state broadcast across ranks is #543, never built here)" cannot stand
on this tree: §8's four chaos rows (`crates/jammi-server/tests/it/gang_chaos.rs`) have the
successor gang of a retired attempt RESUME from epoch 1's checkpoint and publish bytes equal to
an uninterrupted run — executed, on the consolidation branch. Their rank 1 did it by hand
(`run_rank_1` fetched `{job_id}/_resume/` from the shared store); the real body does it through
`run_fine_tune_blocking`'s `discover_resume` over the member's OWN artifact store — the fleet's
shared root, the identity every member was admitted on — exactly as an in-process `Local` rank
does, per-rank dropout positions included (U4b's bundle). The shared store IS the cross-rank
broadcast; #543 remains the issue for a deployment whose ranks do not share a root, which this
design's root-identity predicate already refuses at the listing. Removed: `CoordinatorEnd::
ResumeRefused`, `ArtifactStore::has_resume_checkpoint`, the pin in `assemble_and_run`, the
resume-pin it oracle, M6/M7 (§2's rows are superseded by the chaos rows' executed resume), the
docs' pin prose. The lead decides whether the plan row keeps the pin's wording.

**The chaos rows run against the REAL body** (the lead's question, decided): the tap
(`GangServer::take_member_links`) hands out body-less sessions' links only, so
`gang_chaos.rs`'s test-thread rank 1 over a tapped link is gone. Instead two `test-hooks` seams
in the engine — `training_test_hooks::wrap_member_collective(job_id, wrap)` (one-shot: the next
member body of that job applies `wrap` to its `Peer` right after building it, before the first
collective — the chaos rows' `ChaosRank`, now over `Arc<dyn Collective>` and forwarding
`bind_agreement`, is that wrapper) and `note_rank_outcome`/`rank_outcomes_for(job_id)` (every
`run_member_rank` return recorded, whether or not the session lived to emit it). The rows arm the
wrapper before the coordinator dials (`arm_rank_1`) and read the body's end
(`wait_rank_ends`): the retired body ends `Failed{.. chaos ..}` (silent, split brain) or
`Failed{.. nothing applied ..}` (drain — its round on the severed link), the successor's body
`Trained{..}`; a body killed WITH its member's runtime (the dropped-stream row) records no end
at all — asserted. Why the real body rather than a suppression flag: the rows' properties are
the coordinator's failure path over a member that dies/stalls/drains INSIDE a real round, and a
rank 1 that is the production body (its own bind, leaf verify, streamed source, resume discovery,
role) is the stronger oracle for exactly the seam this unit ships; a suppression flag would have
kept a second, test-only rank body alive beside the real one. `RankEnv`/`rank_env`/
`try_run_rank`/`run_rank`/`run_rank_1`/`spawn_rank_1`/`joined` and the member's tap are
deleted, not flagged.

##### 6a. Gates on the rebased tip `2264c28e` (every command re-run on the finished tree; logs `u5b1biii-scratch/f2-*.log`, `f3-*.log`)

| Command | Exit | Result |
|---|---|---|
| `cargo test -p jammi-server --features test-hooks --test it -- --test-threads=1 gang_chaos gang_coordinator gang_service` | 0 | 43 passed, 0 failed (the 4 chaos rows through the REAL body; the 2 coordinator oracles; every `gang_service` row over the real training-set fixture; `f3-srv.log`, on the tip) |
| `cargo test -p jammi-ai --features test-hooks --test it -- gang_coordinator jobs_shutdown host_admission` | 0 | 33 passed, 0 failed (`f2-ai-it.log`) |
| `cargo test -p jammi-ai --features test-hooks --lib -- fine_tune::` | 0 | 358 passed, 0 failed (`f2-ai-lib.log`) |
| `JAMMI_TEST_PG_URL=… cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- gang_rank_admission --test-threads=1` | 0 | 25 passed, 0 failed — sqlite and postgres arms (`f2-db.log`) |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | 0 | clean (`f2-clippy-jammi-ai.log`) |
| `cargo clippy -p jammi-server --all-targets --features test-hooks -- -D warnings` | 0 | clean (`f3-clippy-server.log`, on the tip) |
| `cargo clippy -p jammi-db --all-targets --features test-hooks -- -D warnings` | 0 | clean |
| `cargo fmt --all -- --check` | 0 | clean (`f3-fmt.log`) |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1042 file(s) scanned, all PATH:LINE citations resolve (…; 2 exempt …)` — 8 more citations re-anchored under the rebase's line moves (guide → worker.rs ×3, artifact.rs ×5) |
| `git status --short` at the tip | — | 0 entries; `git merge-base --is-ancestor a072ad59 HEAD` true |

The §2 mutations M1–M5s, M8–M13, M15 were executed on the pre-rebase commit `74b7e5e0`; the
code they mutate is byte-identical on the rebased tip except where §6 states a change (the settle
arms — U5b-2's shape, whose own settlement oracle is on the tip; the withdrawn pin — M6/M7
superseded). Not re-executed after the rebase.

##### 6b. Commits (`git log --oneline a072ad59..HEAD`)

```
2264c28e feat(ai,server,db,docs): #500 U5b-1b-iii — the rank body over the real hold loop; the runner-role writer split; RankEvent::Outcome consumed on receipt; the agreement binding
```

One commit (20 files incl. `role.rs`, +2874/−911 against `a072ad59`). Tip:
`2264c28ec15b7c53993330e91c1e47a5307048ac`. The pre-rebase tip `74b7e5e0` (§5) is superseded.


## 8. U5b-2 (landed as two commits; original tip `81dc653a`)

The implementer's contract, folded by the lead after checking: `run_claimed_job_under` carries a zero diff; `lease_settlement` is a total match with no wildcard; the hermetic chaos rows live in the server it-suite over a two-host loopback fleet and the SIGKILL rows in `tests/distributed` with their names in the workflow matrix. Three deviations accepted with their code: no flag flip (every watchdog-named failure is detected inside rank 0's own collective call, so the run has already returned through `Abandoned`; the flag would have no reader); no `BackOff` disposition (the live same-named `building` row is unreachable at this tip — table names carry nanos + a uuid — refuted by an executed oracle rather than shipped as dead code); mid-run uncounted member faults spend the attempt per DESIGN §4, narrowing §6's release rule to assembly ends and `Drain`. One finding filed for U5b-1b-iii: a rank resumed out of lockstep folds rounds whose descriptors agree and fails late — the agreement digest must bind the resume epoch.

Worktree `wt-u5b2`, branch `unit/u5b2`, base `feat/500-wave3c` @ `82f80b84`. Every path is
repo-relative; every oracle named was executed on the tip (§4); every mutation in §2 was applied
to the committed tree, run through ONE filtered test, and reverted (`git checkout -- <file>`,
`git status --short` empty after each — `u5b2-scratch/mutations.py`, logs `mut-M*.log`,
`mutations.out`). One `--features` set per crate for the session (`jammi-ai`, `jammi-server`:
`test-hooks`; the distributed target: `live-distributed-tests`, its own required feature),
`RUSTC_WRAPPER=sccache`, `CARGO_TARGET_DIR=…/targets/u5b2`. `jammi-db` is untouched.

#### 1. Scope shipped

**`crates/jammi-ai/src/fine_tune/worker.rs`** (the coordinator's failure path only; the rank
body, `RankEvent::Outcome`'s producer and every job-row-writing site are U5b-1b-iii's and are
untouched — `run_claimed_job_under` carries a zero diff):
- `LeaseSettlement { Release, Expire, Untouched }` and `lease_settlement(&CoordinatorEnd)`, a
  TOTAL match (no wildcard; a new end is a compile error): `MemberAborted{Drain}` → `Release`
  (OPS D10); `MemberAborted{Refuted | Unavailable | StoreUnavailable | NoBody | Cancelled |
  Unspecified}` and `LinkFault` → `Expire` (a rank failure spends the attempt at the successor's
  claim; `attempts + 1` is `claim_next`'s, never the coordinator's); every assembly end
  (`HostCannotCoordinate`, `CatalogFault`, `ShortListed`, `MemberUnreachable`, `MemberRefused`,
  `PeerRefused`, `Drain`) → by its recorded outcome's counting class (§6's rule, kept: all
  uncounted today → `Release`); `Moved`, `Published`, `TrainingFailed`, `Cancelled` →
  `Untouched` (the caller's arms).
- `JobWorker::coordinate` settles by `lease_settlement` after recording the assembly outcome
  (`Release` → `release_job_lease`; `Expire` → a warn line naming the end; the end returns
  `Abandoned(end)` — no terminal write), logs the settlement on the "coordinator attempt ended"
  line, and its doc states the watchdog (below), the split and OPS D6.
- `WorkerJobError::Abandoned`'s doc re-anchored on the split. The §6 table oracle's settle
  sentence and one assertion message re-worded (the counting class only); a NEW lib oracle
  pins the settlement over every variant (§2).

**The per-attempt watchdog is the coordinator's own `Peer`, stated, not a second task.** Every
member's stream is read by rank 0's rounds: a member's `Aborted{reason}` is recorded typed on its
link (`CoordinatorLink::session_abort`), a dropped stream ends the wait `Transport`/
`Disconnected`, a rank silent past `[worker] rank_timeout_secs` ends it `Timeout` (the round
deadline `Peer::with_timeout`) — each fails rank 0's collective call, faults every member in the
same round (`fault_all` → `RoundFault`) and, at the body's step (7), ends every session
(`Peer::end_members` → `Cancel`, the stream close); `assemble_and_run` classifies the end from
the links. The `Peer` is built for the attempt and dropped with it: a fault retires exactly the
attempt it belongs to. OPS D6: a member's slot is `Rank` for its whole session and a peer never
claims while it holds a rank (`HostAdmission`), so ending a session aborts no claim transaction.

**`crates/jammi-server/tests/it/gang_chaos.rs`** (new) — the hermetic chaos rows over a
two-HOST loopback fleet: a coordinator engine (its own `InferenceSession`, the production
`GangDialer` installed by the one statement `OssServer::bind` performs) and a member engine
(its own session; the REAL `GangServer::run_rank` mounted on a tokio runtime of its own, so
"the member process died" is that runtime dropped: listener, connections, hold loops, tapped
links' forwarders), both over ONE shared SQLite catalog and ONE shared `file://` result root —
the shape two `jammi-server` replicas take. Rank 1 is the test's thread over the tapped
`MemberLink` (the `test-hooks` tap, the rank body's seat) running the real `TrainingLoop`,
wrapped in a chaos collective (`ChaosRank`) that fires at the FIFTH `all_reduce_sum` — epoch 2's
first optimizer step (`canonical_reduce` is two reduces per step, two steps per epoch), past
epoch 1's resume checkpoint. Rank 1 performs the rank body's resume discovery
(`fetch_resume_checkpoint` for the real job from the fleet's artifact store, `load_bundle`,
`TrainingLoopBuilder::resume`) so the successor gang resumes in lockstep. Four rows (§2), each
completing under a new gang with published bytes EQUAL to an uninterrupted `LocalGang` run.
**`gang_coordinator.rs`** (server it): thirteen §6 helpers made `pub(crate)` for the sibling.

**`crates/jammi-ai/tests/it/gang_coordinator.rs`** — the planted live `building` row oracle
(the `BackOff` refutation, §1 deviation 2).

**`crates/jammi-ai/tests/distributed/{gang_chaos.rs (new), harness.rs, main.rs}`,
`.github/workflows/distributed.yml`** — the process-level SIGKILL rows: every spawned worker is
gang-capable (`worker_toml` renders `[server] peer_bind`/`peer_advertise` on a per-process peer
port, `[distributed] max_world_size = 2`, `[worker] rank_timeout_secs = 10`; `shared_config`
submits at the same bound), `submit_gang_fine_tune` submits a `world_size = 2` spec through
`run_training_spec`; `killed_peer_job_is_reclaimed_and_completed_by_a_new_gang` and
`killed_coordinator_job_is_reclaimed_and_completed_by_a_new_gang` SIGKILL rank 1 / rank 0 of a
three-process fleet mid-run and require `completed` by a new gang, `attempts - releases >= 2`
(the crashed attempt spent), one model row, the resume checkpoint reaped; both names added to
the chaos leg's matrix (advisory, as the plan's lane says). Compiled by the workflow's own
compile-check command (§4); NOT executed here (§3).

**Docs (same commit set):** `docs/maintainer/MAINTAINER-GUIDE.md` §2.8d (the settlement
paragraph rewritten; a new "per-attempt watchdog" paragraph incl. the BackOff refutation and
the chaos oracles), `docs/guide/src/deploy-server.md` (DRAIN: a held rank ends `Drain`, the
job's lease handed back; every other mid-run loss spends an attempt),
`docs/guide/src/fine-tuning.md` (Training safety: multi-host runs),
`docs/guide/src/configuration.md` (`rank_timeout_secs`).

##### Deviations from UNITS.md / the brief, with the reason and the code

1. **No flag flip; the run exits through `Abandoned`, not the lease-lost arm.** The brief:
   "flip the hold's `lost` flag so the run exits through `run_claimed_job_under`'s lease-lost
   arm". At the coordinator every failure the watchdog names is detected INSIDE rank 0's own
   collective call (`peer.rs`: `Link::recv` → `WaitEnd::{Timeout, Disconnected, Transport}` and
   `Frame::Session` for a member's `Aborted`; `Peer::round` sets `fault`), so `train_fine_tune`
   has RETURNED with the error before `coordinate` classifies the end — the hold and the cancel
   watcher are dropped at `run_spec`'s return (`worker.rs::run_claimed_job_under`, `drop(hold);
   drop(cancel_watcher);`), and a flag flipped then has no reader (the trainer checks `cancel`
   only at its epoch-boundary top and before the checkpoint write, both already past). The
   lease-lost arm and the `Abandoned` arm perform the same actions (checkpoint GC, a warn, no
   write; `run_claimed_job_under`'s two arms); `Abandoned(end)` carries the reason, the
   lease-lost arm would log "lease lost" for a lease that was not lost. Routing through the
   flag would be a mechanism with no observable and a misleading log — cut; the property
   (no terminal write on every mid-run end) is P3 with its executed mutation M3.
2. **No `BackOff` disposition on the training path — the state is unreachable at this tip,
   refutation executed.** `ResultStore::create_table` names every table
   `{source}__{task}__{model}__{nanos}_{uuid8}` (`store/mod.rs`, "Nanoseconds plus a short uuid
   suffix"); `training_set::materialize_projection_table` anchors a registered source
   `InputAnchor::unpinned_at_instant`, and `exact_match_candidates` (`store/freshness.rs`)
   short-circuits any unpinned anchor and only ever considers `ready` rows
   (`find_ready_result_tables_by_definition`); the job row's write-once pair is built from the
   FINISHED table (`run_spec`'s FineTune arm, `table.record.table_name` after
   `materialize_projection_table`/`bind_recorded_training_set`, which requires `ready`) and
   written by the coordinator body's CAS after it. So no successor — a retry, or a second job —
   ever finds a live same-named `building` row: it materializes its own table (README r31's own
   "honestly always a miss"). Executed: `it::gang_coordinator::
   a_live_building_training_set_row_left_by_a_crashed_coordinator_is_never_met_by_the_successor`
   plants a `building` TrainingSet row over the same source/task under a live renewing lease,
   runs the attempt, and asserts the attempt's own `ready` table is recorded, the body is
   reached, and the orphan is byte-untouched (still `building`, same writer, lease live) — the
   lease's to reap after expiry (`ResultStore::recover` → `claim_expired_building_table`, which
   exists and is exercised by the store's own recovery tests). M6 (deterministic naming) reds
   it: that is what "same-named" would cost. Shipping a disposition nothing can reach would be
   dead code and an oracle that cannot go red.
3. **A mid-run `MemberAborted{Unavailable | StoreUnavailable | NoBody}` and every `LinkFault`
   spend the attempt** although their recorded `AssemblyOutcome` is uncounted (§6 deviation 2
   released every uncounted outcome). DESIGN.md §4 is explicit ("reclaim requeues the job within
   the remaining lease window"; only the DRAIN/RELEASE case releases) and the brief's "every
   other reason → the failed-attempt path"; §6's own §3 named the consequence of releasing:
   a deterministic member fault retries forever under the cooldown. The counting class still
   governs the cooldown (`record_assembly_outcome` unchanged) and every ASSEMBLY end (no run
   started), so §6's rule is narrowed, not replaced. Consequence stated: at this tip a member
   with no rank body parks `NoBody` after the member's lease, which now spends an attempt —
   correct for the design (a member that never runs a body is a failed rank) and moot once
   U5b-1b-iii's body consumes the link.
4. **The drain hook in `gang.rs` is untouched.** U5a-2's phase arm already ends every held
   session `Aborted{Drain}` on `begin_drain`; the coordinator receives it typed
   (`session_abort`, P4's executed row) and no determinant is missing — nothing to extend.
5. **`tests/distributed/gang_chaos.rs` is written and compiled, not executed** (§3); the
   `harness.rs` gang knobs apply to EVERY spawned worker (greenfield: one fleet shape, no
   per-test flag) — the pre-existing rows submit `world_size = 1` and are unaffected by an idle
   gang listener.
6. **The chaos rows live in `crates/jammi-server/tests/it/`**, not `jammi-ai/tests/it`: the
   real `run_rank` hold loop is `jammi-server`'s, and the two-host fleet needs it.

#### 2. Properties

Lanes: SRV = `cargo test -p jammi-server --features test-hooks --test it -- --test-threads=1
<filter>`; AI = `cargo test -p jammi-ai --features test-hooks --test it -- <filter>`; LIB =
`cargo test -p jammi-ai --features test-hooks --lib -- <filter>`. "M0" is the base tree's
`worker.rs` (`git show 82f80b84:…`) under this unit's tests — the RED-at-base check.

| # | Property (over every input / exit arm) | Executed oracle | Executed mutation → first red line |
|---|---|---|---|
| P1 | The settlement is a total function of the end: for EVERY `CoordinatorEnd` (one sample per ordinal in `0..VARIANTS`, a variant without a sample reds it) and every frozen `AbortReason`: exactly `MemberAborted{Drain}` releases; every other `MemberAborted` and `LinkFault` expire; every assembly end settles by its outcome's counting class; `Moved`/`Published`/`TrainingFailed`/`Cancelled` are untouched | LIB `fine_tune::worker::tests::every_coordinator_end_settles_its_lease_by_the_released_vs_failed_split` | M1 (`LinkFault`/non-drain aborts → `Release`): `worker.rs` (line 7928 at the unit tip): the settlement of the gang faulted: timed out (ordinal 10)` (left `Release`, right `Expire`); M2 (`Drain` → `Expire`): `worker.rs` (line 7928 at the unit tip): the settlement of rank 1 ended its session: Aborted(Drain) (ordinal 9)` |
| P2 (a) | A member's stream dropped mid-round (its process gone) retires the attempt as `LinkFault` naming the round and the transport (`all_reduce_sum: round N: the stream failed/ended …`), recorded `Unavailable` (cooled, `assembly_failures` 0); the row is `running`, `error` NULL, `attempts` 1, `releases` 0, the lease armed (left to expire); epoch 1's resume checkpoint exists; the member's slot frees; the successor's claim is attempt 2 with `releases` 0 (the attempt spent), only after the lease expired; attempt 2 over the restarted member publishes bytes equal to the `LocalGang` reference; one model row | SRV `gang_chaos::a_member_stream_dropped_mid_round_retires_the_attempt_spent_and_a_new_gang_completes_it` | **M0 (base) RED:** `gang_chaos.rs` (line 671 at the unit tip): a rank failure spends the attempt: the lease is never released — left: 1, right: 0`; M1: the same line; M7 (rank 0 never resumes: `discover_resume` → `Ok(None)`): `gang_chaos.rs` (line 795 at the unit tip): attempt 2 publishes: the gang faulted: all_gather: round 11: timed out … waiting for rank 1's contribution` (a gang out of lockstep with its checkpoint never publishes) |
| P3 (a) | A member silent past `[worker] rank_timeout_secs` mid-round retires the attempt after ≥ the deadline as `LinkFault` naming `timed out after 3s waiting for rank 1` and the round; the same row facts as P2 (no terminal write, attempt spent); the coordinator's `Cancel` frees the member's slot within 5 s WHILE its rank thread is still stalled; the next gang completes with `(attempts, releases) == (2, 0)` and the reference bytes | SRV `gang_chaos::a_member_silent_past_the_rank_timeout_retires_the_attempt_spent_and_a_new_gang_completes_it` | **M0 RED** and M1: `gang_chaos.rs` (line 671 at the unit tip) (as P2); M3 (mid-run ends returned as `Failed` → `record_failed`): `gang_chaos.rs` (line 661 at the unit tip): no terminal write: Row { status: "failed", … error: Some("the gang faulted: all_reduce_sum: round 14: timed out after 3s …") }`; M5 (`end_members` removed): `gang_chaos.rs` (line 383 at the unit tip): the member's slot must free once the session ended, holder: Rank { .. attempt: 1 }` — the slot then frees only at the member's re-verification after the lease expiry, not on the stream close |
| P4 | A member's host DRAINing mid-round ends its session `Aborted{Drain}`; the coordinator ends `MemberAborted{Drain}` ("rank 1 ended its session: Aborted(Drain)"), records `Drain` (neither counted nor cooled: `assembly_failures` 0, `next_assembly_after` NULL) and RELEASES: `releases` 1, lease NULL, `attempts` 1, `running`, no error; the rank's own round ends on its closed link ("nothing applied"); the next attempt is claimable within one poll (< the lease window) over a fresh member (the draining host is listed no more; the assignment names each host per attempt) and completes: `(attempts, releases) == (2, 1)` — zero net attempts (OPS D10) — with the reference bytes | SRV `gang_chaos::a_member_aborted_drain_mid_round_releases_the_lease_and_the_next_attempt_completes_at_once` | M2: `gang_chaos.rs` (line 665 at the unit tip): Released: the lease is handed back (OPS D10) — left: 0, right: 1`. Green at base (M0): §6's counting-class rule already released `Drain`; this row pins it against M2 |
| P5 | Split brain: attempt 1's coordinator loses its lease (its keeper dies) while its rank 1 is held and stalled; a second coordinator reclaims only after the expiry (attempt 2, `releases` 0), dials the same member, whose `RunRank` at the greater attempt takes the slot from the elder hold (U5a-2's fence) on the FIRST dial; the elder session is refuted at its next tick, the stale runner's attempt ends `Cancelled` (its own lost lease) or `MemberAborted{Refuted}` and writes nothing (`record_assembly_outcome` is a moved claim; its arm is lease-lost/`Abandoned`); attempt 2 completes with `(attempts, releases) == (2, 0)`, `claimed_by` the successor, one model row, the reference bytes | SRV `gang_chaos::an_older_attempts_stale_runner_is_fenced_by_the_successor_and_writes_nothing` | M4 (`try_hold_rank` refuses a greater attempt): `gang_chaos.rs` (line 1053 at the unit tip): the successor's attempt publishes — left: 5, right: 12` (the successor's dial was refused `Unavailable` → `MemberRefused`; a third attempt would follow). Green at base (M0): the fence and the lease-lost arm are U5a-2's/the loop's; pinned end to end here |
| P6 | The successor attempt of every retired attempt resumes from epoch 1's job-level checkpoint and publishes bytes EQUAL to an uninterrupted two-rank `LocalGang` run of the same fixture; exactly one model row exists after the retry | the byte-equality and model-count assertions in P2–P5 (`assert_completed_like_the_reference`); P2's `fetch_resume_checkpoint(..).is_some()` after attempt 1 | M7 (above): attempt 2 never publishes |
| P7 | A crashed coordinator's live `building` training-set row over the same source/task is never met by the successor: the attempt materializes its OWN `ready` table, records ITS name on the job row, reaches the coordinator body, and leaves the orphan byte-untouched (`building`, same writer, lease live) | AI `gang_coordinator::a_live_building_training_set_row_left_by_a_crashed_coordinator_is_never_met_by_the_successor` | M6 (`create_table` names tables `{source}__{task}__{model}`, no nanos/uuid): `gang_coordinator.rs` (line 772 at the unit tip): assertion left == right failed: [] — left: 0, right: 1` (the successor's `create_table` collides with the orphan's name; no attempt reaches the body) |
| P8 | Every U5b-1b-ii coordinator oracle and every admission/shutdown row of the loop is unchanged by the split | AI `gang_coordinator rank_admission jobs_shutdown host_admission` (42 passed: 4 + 10 + 19 + 9); SRV `gang_coordinator` (the §6 Peer row, `attempt-2 end: published`) | regression-only |

#### 3. Uncovered

- **The process-level SIGKILL rows** (`tests/distributed/gang_chaos.rs`): no MinIO here and,
  at this tip, no rank body in the server (U5b-1b-iii), so a cross-process gang cannot complete
  at all (a member parks `NoBody` — spending an attempt under this split — and three attempts
  fail the job). Written to the plan's shape, compiled by the workflow's own compile-check
  command, listed advisory in `distributed.yml`; first executed by the nightly lane after
  consolidation with U5b-1b-iii. Two timing choices are stated, not measured: the 4 s settle
  before the kill (the gang must be mid-run) and rank 1's selection by the sorted `workers`
  listing (`assign_ranks`'s order without the root predicate — every spawned worker shares the
  root).
- **The cluster leg (2 pods × 2 GPUs, `Nccl`)** — not this unit's; UNCOVERED.
- **A resumed gang out of lockstep is not refused, only fails late** (found by this unit's
  rows, M7 and the first run before rank 1 resumed): a rank that starts from a different epoch
  than rank 0 folds rounds whose descriptors agree (same shapes, same round index) over
  different weight states until the shorter run ends; the round descriptor carries no epoch/
  step and the agreement slot is bound nowhere (§6 deviation 8, scheduled for U5b-1b-iii). The
  rank body must resume from the same `_resume` bundle (the test's rank 1 does); nothing in the
  wire enforces it. Filed here for the lead; not fixed (the rank body's seam).
- **The elder session's end reason in P5** admits both `Cancelled` (the keeper's exit guard
  flips the flag before the member's refutation arrives) and `MemberAborted{Refuted}`; which
  one lands is a race between the keeper thread's death and the member's 3 s tick — the row
  facts are identical either way and asserted; the reason is not pinned.
- **`Peer::fault_all`'s `RoundFault` reaching a member mid-round** is exercised only as the
  member's link closing (P4's "nothing applied" is the `Disconnected` arm; the stalled rank of
  P3 never reads its inbox); a member reading the fault frame itself is U5b-1b-i's oracles.
- **A member's `Aborted{Refuted | Unavailable | StoreUnavailable}` mid-run** is settled by the
  same `Expire` arm as P2/P3 (P1 pins the mapping); only `Drain` and the fence's `Refuted` are
  produced end to end here.
- **The `Release` failure arm** (`release_job_lease` erroring after a `Drain`, "left to
  expiry") is a warn; not executed.
- **Two real processes over Postgres** for the hermetic rows: two engines in one process over
  one SQLite file (WAL, 5 s busy timeout) — the Postgres arm of the same fleet is the
  distributed lane's.

#### 4. Gates

COMMON.md's trimmed set on the final tip (every command with `RUSTC_WRAPPER=sccache`,
`CARGO_TARGET_DIR=…/targets/u5b2`; logs in `u5b2-scratch/`: `gates.log`, `t-*.log`,
`mut-*.log`, `mutations.out`).

| Command | Exit | Result |
|---|---|---|
| `cargo test -p jammi-ai --features test-hooks --lib -- lease_settlement every_coordinator_end` | 0 | 2 passed (the new settlement oracle + the §6 table oracle) |
| `cargo test -p jammi-ai --features test-hooks --test it -- gang_coordinator rank_admission jobs_shutdown host_admission` | 0 | 42 passed, 0 failed (gang_coordinator 4 incl. P7, rank_admission 10, jobs_shutdown 19, host_admission 9) — `t-ai-it1.log` |
| `cargo test -p jammi-server --features test-hooks --test it -- --test-threads=1 gang_chaos` | 0 | 4 passed, 0 failed, 22.5 s — `t-chaos3.log` (the four rows of §2) |
| `cargo test -p jammi-server --features test-hooks --test it -- gang_coordinator` (the §6 Peer row, run on the base binary for timing: `attempt-2 end: published`) | 0 | 1 passed — `timing-base.log`; re-executed inside the M-round binaries unchanged |
| M0 (base `worker.rs`, this unit's rows) | 101 | 2 failed (P2, P3 at `gang_chaos.rs` (line 671 at the unit tip), `releases` 1 ≠ 0), 2 passed (P4, P5) — `mut-M0.log` |
| M1–M7 (`mutations.py`, one filtered test each, reverted, tree clean after each) | 101 each | every run red on its first try; first red lines quoted in §2 — `mutations.out`, `mut-M*.log` |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-server --all-targets --features test-hooks -- -D warnings` | 0 | clean |
| `cargo test -p jammi-ai --features live-distributed-tests --test distributed --no-run` (the workflow's compile-check step) | 0 | compiles — `gates.log`, `gates2.log` |
| `cargo clippy -p jammi-ai --features live-distributed-tests --test distributed -- -D warnings` | 0 | clean after the `WorkerPorts` fold (a first run tripped `too_many_arguments` on `worker_toml`, `gates.log`; refactored, not allowed) |
| `cargo fmt --all -- --check` | 0 | clean |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1040 file(s) scanned, all PATH:LINE citations resolve (…; 2 exempt as non-ancestor legacy evidence)` — no PATH:LINE citation shifted (the guide edits are prose; the cited `worker.rs` anchors are symbol-level) |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | OK |
| `git status --short` at the tip | — | 0 entries |

The chaos rows, the ai filters and the lib oracles were executed on the tree of `4fec9ce3`
(commit 2 touches only the `distributed` target, its harness and the workflow — none of it in
the `it`/lib binaries); the clippy runs on the two crates preceded the `WorkerPorts` fold, which
lives in the `distributed` target alone (not compiled by `--all-targets --features test-hooks`:
`required-features`), whose own clippy and compile check ran after it.

Not run, per COMMON.md: `cargo doc`, the live-Postgres lane (no `jammi-db` change),
`merge_path.sh`, workspace builds, the distributed rows themselves.

#### 5. Commits (`git log --oneline 82f80b84..HEAD`)

```
81dc653a test(ai,ci): #500 U5b-2 — the process-level SIGKILL rows for a two-rank gang; every distributed-lane worker gang-capable
4fec9ce3 feat(ai,server,docs): #500 U5b-2 — the released-vs-failed split on the coordinator's failure path; hermetic chaos rows over a two-host loopback fleet
```
(no trailers, per the brief; `git status --short` clean.) Tip:
`81dc653a4432e2cd7954777aa808a1d0fe7fd31b`.


## 8a. Issue #574 (landed as one commit; original `96da013d`)

The implementer's contract, folded by the lead after checking: `LeaseFact::{Live, Dead, Undecodable}` decoded in Rust on the admission read; the SQL-side parse stays on the claim/reclaim path (the backend clock matters there; the admission read never writes or reaps); `GangRefusalReason::LeaseUndecodable` is the seventeenth determinant, non-disclosing on the wire.

Branch `unit/i574` in worktree `wt-i574`, cut from `feat/500-wave3c` @ `82f80b84`.
Crate touched: `jammi-db` (`catalog::{jobs_repo, lease}`, `tests/it/{gang_rank_admission,
gang_instance_freshness}`), `jammi-server` (`grpc::gang`, `tests/it/gang_service`), docs
(`docs/maintainer/MAINTAINER-GUIDE.md` §2.8a, `docs/guide/src/security.md` I-GANG). Every
claim below is stated as it EXISTS on the branch tip; every path is repo-relative; test
names are `file::fn`.

#### 1. Scope shipped

- `crates/jammi-db/src/catalog/lease.rs` gains `LeaseFact` (`Live { remaining: Duration } |
  Dead | Undecodable` — the `WorldSizeFact` pattern from U5a-2), `parse_app_clock_stamp`
  (the app-clock ISO-8601-with-`Z` shape `now_sortable()`/`lease_now()`/`lease_deadline()`
  all write, flexible fractional width), `parse_lease_expires_at` (backend-aware:
  `parse_app_clock_stamp` on SQLite, Postgres's own default `timestamptz`-cast-to-`text`
  rendering — `YYYY-MM-DD HH:MM:SS[.ffffff]±HH[:MM]`, fractional part omitted when zero —
  on Postgres, format verified against a live Postgres 16 via `psql`), `decode_lease_expires_at`
  (`kind, text: Option<&str>, now -> LeaseFact`, infallible), and
  `last_seen_at_is_fresh(text: &str, margin, now) -> bool` (backend-INDEPENDENT: `last_seen_at`
  is always an app-clock stamp on either backend, since `Catalog::upsert_instance` /
  `reregister_instance` / `touch_instance` never write the database clock there). Both
  boundaries match the SQL predicates they replace exactly: `Live` iff `deadline >= now`
  (the negation of `lease_expired_clause`'s strict `<`); fresh iff `seen >= now - margin`
  (the negation of `stale_before_clause`'s strict `<`).
- `crates/jammi-db/src/catalog/jobs_repo.rs`: `RankAdmissionRow::{lease_live: bool,
  remaining: Duration}` REPLACED by `RankAdmissionRow::lease: LeaseFact` (a reshape, not a
  bolt-on — every call site updated atomically, no compatibility field kept).
  `Catalog::get_job_for_rank`'s SELECT drops the SQL-side `lease_remaining_seconds_expr`
  computed column entirely and selects `lease_expires_at` as raw TEXT; the row mapper calls
  `decode_lease_expires_at(kind, text, chrono::Utc::now())`. `Catalog::fresh_instance` drops
  the SQL-side `stale_before_clause` predicate from its `WHERE`, selects only `last_seen_at`
  by primary key, and calls `last_seen_at_is_fresh` in Rust.
- `crates/jammi-server/src/grpc/gang.rs`: `GangRefusalReason` gains `LeaseUndecodable`
  (doc'd against `LeaseFact::Undecodable`, issue #574). `run_rank`'s lease check becomes a
  `match row.lease { Live => admit, Dead => LeaseDead, Undecodable => LeaseUndecodable }`,
  same fixed `FailedPrecondition` either way. `reverify`'s `row_holds` conjunct becomes
  `matches!(row.lease, LeaseFact::Live { .. })` — `Undecodable` mid-hold collapses into
  `Refuted` exactly like every other row fact that stops holding (no new `ReverifyEnd`
  needed).
- `crates/jammi-db/tests/it/{gang_rank_admission,gang_instance_freshness}.rs`: every
  `row.lease_live`/`row.remaining` call site ported to `row.lease.is_live()`/
  `row.lease.remaining()`/`LeaseFact` matches; two new parity oracles added (below).
- `crates/jammi-server/tests/it/gang_service.rs`: `refusal_scenario` gains a
  `GangRefusalReason::LeaseUndecodable` arm (raw SQL plants `lease_expires_at =
  'not-a-timestamp'` on an otherwise-admitting world-1 fixture); `every_gang_refusal_reason`'s
  `WITNESSES` list and its exhaustive `assert_every_variant_is_a_witness` match both gain the
  variant (the match has NO wildcard arm — a future variant fails this file to compile until
  matched, per U5a-2's own methodology); doc counts updated sixteen → seventeen.
- Docs: `MAINTAINER-GUIDE.md` §2.8a rung 3 (the lease row fact, `LeaseUndecodable`) and rung 5
  (`fresh_instance` decodes client-side now) rewritten; the "sixteen, `GangRefusalReason`"
  non-disclosure sentence → seventeen. `docs/guide/src/security.md`'s I-GANG section: the
  determinant list gains "or undecodable"; the "Multi-host gang admission is a Postgres-only
  deployment shape" bullet rewritten — it no longer claims the ADMISSION read needs Postgres's
  shared clock (that claim became false the moment the read moved to Rust); the real reason
  (SQLite is a single process, no second host) is kept, and the CLAIM/RECLAIM write paths are
  named as the ones that still require the shared clock, with the reasoning (a wrongful reap is
  destructive; a stale admission read self-corrects at the next heartbeat).

##### Deviations from the brief, with the reason and the code

- **The brief's candidate shape `LeaseFact::{Live { remaining }, Dead, Undecodable}` was
  taken verbatim** (not "the shape you derive") — it is exactly the `WorldSizeFact` pattern
  already established in this file (`jobs_repo.rs` (lines 414–424 at the unit tip)), so no alternate shape was
  justified.
- **`fresh_instance` was NOT given its own 3-state fact type.** The brief says "a malformed
  `last_seen_at` is 'not fresh', a row fact, never a read fault" — it does not ask for a new
  `GangRefusalReason` variant for freshness, and `Catalog::fresh_instance`'s own doc
  (`jobs_repo.rs`, cited above) already states `false` covers "an absent OR a stale row
  alike... disclosing nothing about which". A malformed value joins that SAME class; adding a
  distinguishable enum there would be a variant with no consumer (family L: no generic-nucleus
  seam asked for it, and `GangServer::run_rank` has exactly one `CoordinatorNotFresh` site that
  reads a `bool`). If a future caller needs the distinction, `last_seen_at_is_fresh` already
  returns a clean boundary to build a fact type on top of.
- **The split decision (SQL stays on claim/reclaim, Rust decodes on the admission read) is
  argued in `lease.rs`'s own doc on `decode_lease_expires_at`, not merely asserted**: reaping
  a live claimant under app-clock skew is destructive (the module's own pre-existing "whose
  clock" doc, `lease.rs` (lines 13–25 at the unit tip)); the admission read never writes or reaps, so a few-hundred-ms
  skew self-corrects at the next re-verification tick (`heartbeat`-cadence). This is the one
  place this unit reasons beyond mechanical translation, and it is the property the RED-first
  Postgres mutations (below) exist to keep honest: if the split were wrong in the OTHER
  direction (claim/reclaim also needing to move to Rust), that is out of this issue's scope
  and not touched.

#### 2. Properties (quantified) — executed oracle — executed mutation that reds it

Lanes: DB = `cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it --
<filter> --test-threads=1` (sqlite arm always; postgres arm EXECUTED against
`JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1 port 54329/jammi_test`, a live Postgres 16); DB-lib =
`cargo test -p jammi-db --features live-postgres-tests,test-hooks --lib -- <filter>`; SRV =
`cargo test -p jammi-server --features test-hooks --test it -- <filter>`.

| # | Property (over every input / exit arm) | Executed oracle | Executed mutation → red (first line) |
|---|---|---|---|
| P1 | For a `jobs.lease_expires_at` value that is neither `NULL` nor parseable for a backend, `Catalog::get_job_for_rank` returns `Ok(Some(row))` with `RankAdmissionRow::lease == LeaseFact::Undecodable`, IDENTICALLY on sqlite and (EXECUTED, live) postgres — never `Err` on either. | DB `gang_rank_admission::get_job_for_rank_undecodable_lease_is_a_row_fact_on_both_backends::{sqlite,postgres}` | M1: reintroduce `lease_remaining_seconds_expr`'s computed column into the SELECT (even though nothing reads it back) → postgres arm reds: `BackendDriver(Sqlx(Database(PgDatabaseError { ... code: "22007", message: "invalid input syntax for type timestamp with time zone: \"not-a-timestamp\"" ...})))` — the EXACT fault class issue #574 reports; sqlite arm stays green (proving the divergence, not just a regression) |
| P2 | For an `instances.last_seen_at` value that does not parse as a timestamp, `Catalog::fresh_instance` returns `Ok(false)`, IDENTICALLY on sqlite and (EXECUTED, live) postgres — never `Err` on either. | DB `gang_instance_freshness::fresh_instance_malformed_last_seen_at_is_not_fresh_on_both_backends::{sqlite,postgres}` | M2: reintroduce `stale_before_clause`'s computed predicate into the SELECT (unread) → postgres arm reds with the same `22007 invalid input syntax` `PgDatabaseError`; sqlite arm stays green |
| P3 | `decode_lease_expires_at`'s Live/Dead boundary is EXACTLY `lease_expired_clause`'s own negation (`deadline >= now`, never the strict `>`), for `NULL`, malformed, exactly-`now`, future, and past inputs. | DB-lib `catalog::lease::tests::decode_lease_expires_at_is_infallible_on_every_input` | M3: `deadline >= now` → `deadline > now` in `decode_lease_expires_at`: `lease.rs` (line 578 at the unit tip): assertion left == right failed / left: Dead / right: Live { remaining: 0ns }` (the exactly-`now` case moved off the boundary) |
| P4 | `parse_lease_expires_at` decodes STRICTLY per-backend: SQLite's arm never accepts Postgres's own text rendering and vice versa (a cross-decode would silently misread a value this backend never wrote in that shape). | DB-lib `catalog::lease::tests::{parse_lease_expires_at_postgres_accepts_its_own_default_text_rendering, parse_lease_expires_at_sqlite_uses_the_app_clock_shape_only}` | M4: swap the two `match kind` arms in `parse_lease_expires_at` → BOTH tests red: `lease.rs` (line 512 at the unit tip)/535: assertion left == right failed / left: None / right: Some(...)` (each backend's own format text no longer parses under its own arm) |
| P5 | `last_seen_at_is_fresh`'s boundary is EXACTLY `stale_before_clause`'s own negation (`seen >= now - margin`); a value that does not parse is never fresh. | DB-lib `catalog::lease::tests::last_seen_at_is_fresh_matches_stale_before_clauses_boundary`; DB `gang_instance_freshness::{fresh_instance_true_just_inside_the_liveness_margin, fresh_instance_false_just_outside_the_liveness_margin}::{sqlite,postgres}` (the pre-existing boundary pair, unaffected by the reshape, re-run green on live postgres) | (boundary asserted directly in the unit test; the pre-existing just-inside/just-outside pair already carries its own mutation proof per U5a-2's era — not re-executed this round, since `last_seen_at_is_fresh`'s comparator is a direct, unchanged port of `stale_before_clause`'s) |
| P6 | The gang admission handler refuses `LeaseFact::Undecodable` under its OWN `GangRefusalReason::LeaseUndecodable`, distinguishable from `LeaseDead` only under `test-hooks`, with the SAME fixed wire status as every other determinant (non-disclosure, now over seventeen determinants). | SRV `gang_service::{run_rank_refusal_is_non_disclosing_across_every_determinant, run_rank_last_refusal_reason_distinguishes_every_determinant}` (both re-execute all seventeen `refusal_scenario`s, `LeaseUndecodable` among them) | M5: conflate `LeaseUndecodable`'s `record_refusal` call with `GangRefusalReason::LeaseDead` in `gang.rs` → `run_rank_last_refusal_reason_distinguishes_every_determinant` reds: `gang_service.rs` (line 1968 at the unit tip): assertion left == right failed / left: Some(LeaseDead) / right: Some(LeaseUndecodable)` |
| P7 | Every OTHER column on the admission row stays populated when the lease is undecodable (the malformed lease is isolated to `RankAdmissionRow::lease` alone, never contaminating `status`/`claimed_by`/`attempts`). | DB `gang_rank_admission::get_job_for_rank_undecodable_lease_is_a_row_fact_on_both_backends::{sqlite,postgres}` (asserts `status`/`claimed_by`/`attempts` after planting the malformed lease) | (structural: the same test as P1; not separately mutated — the row-mapper shape makes a partial decode a compile error, the same argument U5a-2's own analogous property relies on) |

Every mutation above was applied → the SINGLE filtered test that names it re-run → the red
output captured verbatim → `git diff` reverted with `Edit` back to the exact committed text
→ `git status` clean confirmed before the next mutation. No mutation round left residue in
the committed tree (verified: `git status --short` empty and `git diff --stat` empty after
the last revert, before writing this contract).

#### 3. Uncovered

- **UNCOVERED — a mid-hold lease going `Undecodable` at re-verification.** `reverify`'s
  `row_holds` conjunct (`matches!(row.lease, LeaseFact::Live { .. })`) collapses `Undecodable`
  into `Refuted` by construction (the same boolean-AND shape that already collapses a `status`
  change), and the row-holds machinery is exercised generically by
  `run_rank_held_session_ends_refuted_when_the_row_no_longer_holds` (a `status` mutation, not a
  lease one) plus every admitted-and-parked test implicitly proving `row.lease` still reads
  `Live` across multiple heartbeats post-admission (`run_rank_every_i_gang_determinant_satisfied_is_admitted_held_and_parks_no_body`,
  `HEARTBEAT * 3`+ waits with no `Refuted`). No test plants `lease_expires_at =
  'not-a-timestamp'` on an ALREADY-ADMITTED session and asserts the SPECIFIC `Refuted` end from
  THAT determinant — the generic "row no longer holds → Refuted" shape is proven, the
  lease-specific instance of it is not separately isolated. Adding one is a same-shape
  extension of `run_rank_held_session_ends_refuted_when_the_row_no_longer_holds`'s own pattern,
  left for a follow-up since the brief's deliverable is the ADMISSION read (rung 3), not
  re-verification's determinant enumeration.
- **UNCOVERED — `peer_addr_of`'s own `stale_before_clause("last_seen_at", ...)` call
  (`jobs_repo.rs`, a different verb than `fresh_instance`).** The SAME backend-divergence class
  exists there (a malformed `last_seen_at` would fault on Postgres, read as absent-or-stale on
  SQLite) but `peer_addr_of` is not on the `RunRank` admission path and the brief scopes this
  unit to `get_job_for_rank`/`fresh_instance` by name ("the only other read on the RunRank
  admission path") — out of scope, not touched, and not claimed fixed.
- **UNCOVERED — `list_gang_members`'s own `stale_before_clause("i.last_seen_at", ...)`
  call** (a THIRD site with the same class, used for gang membership listing, not
  admission) — likewise out of the brief's named scope.
- The Postgres timestamptz text format (`parse_lease_expires_at`'s Postgres arm) was verified
  against ONE live Postgres 16 instance's DEFAULT `DateStyle` (`ISO, MDY`). A deployment that
  changes `DateStyle` (a session/database-level Postgres setting, not exposed by this crate's
  config) would write and read a DIFFERENT text rendering than this parser expects — this was
  already true before the fix in the sense that the OLD code's `col::timestamptz` cast was
  DateStyle-independent (Postgres compares typed values, not text), so this is a genuinely NEW
  determinant this fix introduces: a `DateStyle` this crate never configures becoming
  non-default would turn every WELL-FORMED lease into `Undecodable` on read, not merely a
  malformed one. Labelled here, not defended: this crate does not set `DateStyle` anywhere
  (grepped `crates/jammi-db` for `DateStyle`/`datestyle`, no hits), so the deployment's default
  (which every fresh Postgres ships as `ISO, MDY`) is relied on implicitly.

#### 4. Gates (trimmed set per the brief; exit codes and counts)

All with `CARGO_TARGET_DIR=<scratchpad>/targets/i574`, `RUSTC_WRAPPER=sccache`; one
`--features` set per crate for the whole session (`jammi-db`: `live-postgres-tests,test-hooks`;
`jammi-server`: `test-hooks`).

| Command | exit | result |
|---|---|---|
| `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1 port 54329/jammi_test cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- gang_rank_admission gang_instance_freshness --test-threads=1` | 0 | 37 passed (postgres arm EXECUTED live, not skipped); 0 failed |
| `cargo test -p jammi-db --features live-postgres-tests,test-hooks --lib -- catalog::lease::` | 0 | 18 passed; 0 failed |
| `cargo test -p jammi-server --features test-hooks --test it -- gang_service gang_rank_admission_oracle gang_admission_catalog_fault_oracle` | 0 | 43 passed; 0 failed |
| `cargo clippy -p jammi-db --all-targets --features live-postgres-tests,test-hooks -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-server --all-targets --features test-hooks -- -D warnings` | 0 | clean |
| `cargo fmt --all -- --check` | 0 | clean |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `1038 file(s) scanned, all PATH:LINE citations resolve` (2 pre-existing EXEMPT historical artifacts, unrelated to this unit) |

Executed mutations: M1–M5 all red on their first run (see §2); every mutation reverted via
`Edit` back to the exact committed text before the next; `git status --short` empty and
`git diff --stat` empty confirmed after the last revert.

#### 5. Commits (`git log --oneline 82f80b84..HEAD`)

```
96da013d fix(db,server): #500 #574 — gang admission decodes lease/freshness facts in Rust, never SQL, on both backends
```
(no trailers, per COMMON.md; 8 files, +690/−136 against `82f80b84`; `git status` clean.)


## 8b. C1 — the cookbook session-lifecycle gate for the non-pytest lanes, issue #539 (landed as one commit; original `1124512c`)

The implementer's contract, folded by the lead after running the gate and its 13-check self-test on the merged tree (clean over 80 files). Mechanism (b), the AST gate, was chosen over the registry observer with the reason stated (a quarto chapter is one persistent kernel with no wrappable entry point; a static gate reds on the PR with no render). Its four labelled limits (intra-procedural taint only; `jammi.connect` attribute-call only; the support library out of the scanned roots; with-item ordering) each carry the grep that shows zero live sites today.

#### Scope shipped

- `ci/scripts/check_cookbook_session_lifecycle.py` (new): an AST gate over
  `cookbook/book/scripts/**`, `cookbook/quickstart/**`, `cookbook/recipes/**`
  (every `.py`) and `cookbook/book/chapters/**/*.qmd` (every executed
  ```{python}``` cell). It walks the real AST block extent of every
  `with tempfile.TemporaryDirectory() as X:` statement (its own `.items` and
  `.body`, at any nesting depth — `ast.walk`, no line count, no indentation
  column) and flags a `jammi.connect(...)` call whose first argument
  (positional or `target=` keyword) mentions `X` — an f-string, a `BoolOp`
  (`args.target or f"file://{X}"`), string concatenation, or `X.name` — unless
  that call is itself a with-item's own `context_expr` (of the same statement
  or of a nested `with` inside the block). `--self-test` runs 13 fixture
  checks and exits non-zero on any failure.
- `.github/workflows/ci.yml`: two new `guard` matrix legs — `cookbook
  session-lifecycle gate` (the real scan) and `cookbook session-lifecycle gate
  (self-test)` — so `ci/scripts/merge_path.sh --only guards` runs both.
- `cookbook/book/tests/conftest.py`: the module docstring's closing sentence
  ("a static gate over those is filed as issue #539") is rewritten to name the
  shipped gate instead of the filed issue (current-state docs, same commit).
- `docs/maintainer/MAINTAINER-GUIDE.md` §1.5: "four coupling artifacts" ->
  "five", with a new item 5 describing both session-lifecycle mechanisms (the
  pytest registry rail and this gate) and why crediting only a with-item is
  sound on every exit path.

##### Deviations from the brief, with the code cited for each

- **Mechanism chosen: (b), the static AST analysis — not (a), the registry
  observer.** The issue itself frames this as an open choice ("or both if
  each covers what the other cannot"). Rejected (a) because none of the four
  non-pytest lanes has one process entry point this repo controls the way
  `pytest`'s `conftest.py` controls test collection: `cookbook/book/scripts/*.py`
  and `cookbook/recipes/*/example.py` are each `python <file>.py` — a
  standalone process a harness could wrap, sure, but the fourth lane, a
  `.qmd` chapter, is executed by quarto as a persistent kernel across many
  cells (confirmed by reading a real chapter, e.g.
  `cookbook/book/chapters/21-`unified-client/unified-client.qmd` (lines 122–126 at the unit tip) binds
  `embedded`/`remote` in one cell and both names are read by cells nearly 300
  lines later) — there is no importable module boundary a Python-level
  `atexit`/`observe()` harness could wrap around "one `.qmd` chapter's
  render" without reimplementing quarto's own render loop, and doing so would
  only report a leak once the WHOLE chapter finished (deep in the nightly
  render, `.github/workflows/cookbook-render.yml`, never a PR gate) — failing
  the "RED on a PR, no render needed" requirement the brief's own "runs in
  `ci.yml`'s guard matrix" line states. (b) runs over the committed source
  text directly, in every PR, with no render.
- **Scope excludes `cookbook/book/tests/**`.** The issue's own body states
  "No static shape gate exists for those lanes and none is needed" for the
  pytest lane. I read `cookbook/book/tests/conftest.py` in full and confirmed
  its `_no_leaked_sessions` fixture already subscribes to
  `clients/python/jammi/_sessions.py`'s registry for every test, independent
  of binding shape — a second, less precise mechanism over the same files
  would be redundant coverage the issue explicitly declines. (Checked for
  false-positive risk anyway: `cookbook/book/tests/test_unified_client_cache.py:110`
  binds `embedded = jammi.connect(f"file://{tmp_path}")` against pytest's own
  `tmp_path` fixture, never a `TemporaryDirectory` with-item, so this gate
  would not even see it as tainted if the scope were widened — verified by
  running the gate against that path with the exclusion lifted, zero
  offenses.)
- **`jammi.connect` only, not an aliased `from jammi import connect`.** Same
  limit the excised gate's `_CONNECT` regex had, stated in the issue's own
  design comment ("taint = the tempdir's bound name appearing in the FIRST
  argument of any `jammi.connect(...)` call"). Swept by hand:
  `grep -rn "^from jammi import" cookbook/book/scripts cookbook/book/chapters
  cookbook/quickstart cookbook/recipes` returns only `Session`/`Capability`
  imports, never `connect` — zero live occurrences today.
- **`cookbook/book/jammi_cookbook/**` (the shared support library) is out of
  scope.** The issue's property names "build scripts, recipes, quickstart,
  executed chapter cells" — a library imported by those lanes, not itself a
  lane. Checked: `grep -rln "TemporaryDirectory" cookbook/book/jammi_cookbook`
  returns nothing, so this is not a live gap today; a reviewer adding a
  `TemporaryDirectory`+`jammi.connect` site there should not rely on this
  gate to catch it (documented as a known limit in the module docstring is
  NOT done for this specific one — flagging it here instead, since it did not
  come up as a real site to word a limit around).

#### Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| Every `jammi.connect(...)` call under the four non-pytest lane roots, tainted by an enclosing `TemporaryDirectory`'s bound name, is a with-item whose context manager is that connect call directly (same statement or nested `with`) — else the gate reports it by `path:lineno`. | `check_cookbook_session_lifecycle.py --self-test`, checks `assignment-form-flagged`, `wrapper-with-item-flagged`, `exit-stack-flagged`, `finally-body-control-flow-still-flagged`, `two-tempdirs-one-statement-flagged`, `boolop-fallback-assignment-flagged` (6/13); plus the module-level scan `python3 ci/scripts/check_cookbook_session_lifecycle.py` over the real committed tree. | For the self-test fixtures: each fixture IS the mutation of the negative-control shape (e.g. `wrapper-with-item-flagged` is `with-item-shape-silent`'s exact code with the with-item's CM changed from `jammi.connect(...)` to `_Wrap(jammi.connect(...))`) — asserted both ways in the same run, so a change that broke crediting either direction reds one of the two. For the real tree: mutated `cookbook/book/scripts/build_cdc_cache.py`'s with-item connect (`args.target or f"file://{catalog}"`) into a bare assignment closed only in a `try/finally`; ran `python3 ci/scripts/check_cookbook_session_lifecycle.py`; first line of red output: ``::error::an embedded engine outlives the TemporaryDirectory rmtree is about to remove (ENOTEMPTY: Errno 39 on Linux / 66 on macOS):`` followed by ``cookbook/book/scripts/build_cdc_cache.py:274: `jammi.connect(args.target or f'file://{catalog}')` is tainted by TemporaryDirectory `catalog` and is not a with-item...`` (exit 1); reverted the file (`git diff` empty), re-ran, gate green again. |
| A wrapper with-item (context manager is not the session itself) is flagged — the exact shape the excised regex/indent gate could not distinguish from a genuine with-item. | `--self-test`'s `wrapper-with-item-flagged`. | The fixture's own construction: `_Wrap(jammi.connect(f'file://{d}')) as w` — mutating it back to `jammi.connect(f'file://{d}') as w` (dropping `_Wrap`) makes the check assert `len(...) == 1` against an empty list, i.e. it reds if the credit logic stopped distinguishing wrapper from direct — verified by hand-running that reverted fixture through `offenses_for(...)`, got `[]` as expected for the un-mutated with-item shape. |
| An `ExitStack.enter_context(jammi.connect(...))` (never a with-item at all) is flagged. | `--self-test`'s `exit-stack-flagged`. | Fixture is the mutation of the with-item shape into an `ExitStack`; `_credited_connect_ids` correctly excludes it since the call is an argument to `enter_context`, not any with-item's `context_expr`. |
| A close written inside a `finally:` clause — even one reachable only after an early `return` inside a NESTED `try/finally` (the "finally-body control flow" the issue's title names) — is still flagged, because crediting it is unsound in general (indistinguishable from a close on an aliasing name or inside an `if`). | `--self-test`'s `finally-body-control-flow-still-flagged`. | Fixture nests a `try/finally` with a conditional early `return` inside an outer `try/finally: db.close()`, itself inside the tempdir's `with` block — `_credited_connect_ids` never inspects `finally` bodies at all, so this stays red regardless of how deep the control flow nests; verified the fixture actually contains a `return` inside a `finally`-guarded nested `try` (not just a flat `finally`) by re-reading the fixture source in the self-test file. |
| Two `TemporaryDirectory` items bound in the SAME `with` statement (one of the five shapes the excised regex/indent gate missed) each taint independently; a connect referencing either, not itself a with-item, is flagged. | `--self-test`'s `two-tempdirs-one-statement-flagged`. | Fixture: `with TemporaryDirectory() as a, TemporaryDirectory() as b:` then a bare assignment connect on `f'file://{a}/{b}'` — both names are in `names`, `_tainted_by` matches on either — flags once (deduped by `id(call)`), not twice. |
| A parenthesized multi-line `with (` header (3.10+ grammar, the excised gate's ">20-line bail-out" shape) parses identically to the unparenthesized form; the with-item shape stays silent regardless of how many lines the header spans. | `--self-test`'s `parenthesized-header-with-item-silent`. | `ast.parse` produces the same `With` node either way — there is no line-count parameter to mutate; instead verified by also running the REAL site `cookbook/book/scripts/check_api_reference.py:215-219` (an actual 5-line parenthesized header with-item) through the real-tree scan, which reports zero offenses for it. |
| A `.qmd` chapter's `python`-fenced cells are extracted and concatenated (blank-padded so line numbers still index the real file); non-python fences and prose are excluded; a tainted-and-unclosed shape inside a cell is flagged at the cell's own original line number. | `--self-test`'s `qmd-cell-extraction-flags-and-keeps-lineno` (asserts both the offense count AND the exact `lineno` against `qmd_text.splitlines().index(...)`). | Fixture includes a decoy ` ```{r}` cell containing the literal text `1 + 1` and a prose paragraph that MENTIONS `jammi.connect` in backticks — mutating the extractor to not gate cell membership on the `{python}` tag (e.g. treating any fenced block as Python) would either raise `SyntaxError` on the `{r}` cell's content in a real multi-language chapter or shift line numbers; the assertion on the exact lineno catches a padding-offset regression, and the offense count catches a scope-widening regression. |
| A held, never-`rmtree`'d catalog directory (`tempfile.mkdtemp(...)`, not `TemporaryDirectory`) is out of scope by construction — no removal race exists for it. | `--self-test`'s `held-directory-out-of-scope`; corroborated by the real tree: `cookbook/recipes/image_search/0{2,3,4}-*.py` and the `audio_search` siblings all connect against a persistent `ARTIFACT_DIR`, and the real-tree scan reports zero offenses for them. | Fixture: `db = jammi.connect(f'file://{ARTIFACT_DIR}')` with no enclosing `TemporaryDirectory` at all — `_tempdir_with_statements` finds no `With` node carrying a `TemporaryDirectory` item, so nothing is even attempted; flipping `ARTIFACT_DIR` to a `TemporaryDirectory`-bound name (as done in `assignment-form-flagged`) reproduces a red, confirming the negative here is not merely "the gate never runs." |
| The real, committed `cookbook/**` tree (80 files: `cookbook/book/scripts`, `cookbook/quickstart`, `cookbook/recipes`, `cookbook/book/chapters/**/*.qmd`) is clean. | `--self-test`'s `real-tree-is-clean` control (asserted first, so a gate that reds everywhere is caught before any positive control is trusted); `python3 ci/scripts/check_cookbook_session_lifecycle.py` exit 0. | Any of the mutations above, applied to the real tree instead of a fixture, reds this control — executed once for real (`build_cdc_cache.py`, see the first row) rather than only argued. |

#### Uncovered

- **Intra-procedural taint only** (a `TemporaryDirectory` passed as a
  parameter into a helper several calls deep, which then derives the real
  catalog directory via `tempfile.mkdtemp(dir=parameter)`, is invisible).
  Real, current instance: `cookbook/book/scripts/build_recompute_cache.py`'s
  `emit` (lines 388-402, read directly) -> `run_cache`/`_fresh_chain` (lines
  183-259, read directly) chain — both call sites close correctly via
  `try/finally` (`_fresh_chain`'s `db.close()` at line 198, `run_cache`'s at
  line 259), verified by eye, but this gate does not see them at all (no
  `jammi.connect` call appears anywhere inside `emit`'s own `with
  tempfile.TemporaryDirectory() as work_root:` block's AST subtree at line
  392, since the actual connect lives in a DIFFERENT function, reached only
  through a plain function call `run_cache(work, model, src_path)` at line
  398). Same limit the excised gate carried (issue #536's fix commit
  `d63b8ef5`/`caedd168` region); a reviewer introducing a NEW multi-hop chain
  like this should not rely on this gate.
- **`jammi.connect` attribute-call only** — an aliased `from jammi import
  connect` binding a bare `connect(...)` call is invisible (see Scope
  deviations above; zero live sites today).
- **`cookbook/book/jammi_cookbook/**` is out of the scanned roots** (see
  Scope deviations above; zero live `TemporaryDirectory` sites there today,
  confirmed by grep, so this is a labelled gap rather than a demonstrated
  miss).
- **Item ordering within one `with` statement is not enforced.** If a
  connect item appeared BEFORE its tempdir sibling in the same statement's
  `items` list, this gate would still treat it as tainted (my `names` set is
  built from ALL `TemporaryDirectory` items in the statement, not filtered by
  position) even though the name would not actually be bound yet at that
  point in real execution (a `NameError` at runtime, not a leak). This is a
  theoretical false-positive-on-broken-code path only: no cookbook site puts
  a tempdir item after its dependent connect item (checked: every real
  with-item site in `cookbook/**` puts the `TemporaryDirectory` item(s)
  first), and code with this shape would fail at import/run time before this
  gate's finding would ever matter in practice.
- **The `merge_path.sh --only guards` run's one unrelated failed LEG** ("pod
  build substrate", `rc=1`, log `target/merge-path/025.log`) internally runs
  `ci/scripts/test_pod_substrate.sh`, whose own tally is `195 passed, 19
  failed, 0 skipped`. Re-read the 19 `FAIL` lines directly (not just the tail
  I first looked at): every one traces to `cargo`/`cargo metadata`/`cargo
  package`/a real `cargo build` being unavailable or refusing in this
  sandbox (`which cargo` -> not found on this worktree's `PATH` at all) —
  e.g. `cargo package --list failed for a NON-network reason`, `cargo
  metadata --frozen ... failed (or produced no output)`, `real cargo build
  of the a2fix fixture workspace failed`, plus several fixture-harness
  assertions (`(b/adopt)`, `(n)`, `(i)`, `(p1/A5)`, `(p2/A5)`) that
  themselves shell out to a `cargo`-backed seed pipeline and so fail the same
  way once the toolchain is absent. Unrelated to this cookbook-only, no-Rust
  unit (I never touch `ci/scripts/test_pod_substrate.sh`,
  `pod_build_timings.sh`, or any RunPod/cargo-provisioning script). Not fixed
  here (out of this unit's scope per the brief: "your worktree ... no Rust
  crate") — `which cargo` is a property of this shell's `PATH`, not of any
  git branch content, so this failure is independent of the diff and would
  reproduce identically checking out any commit in this same shell.

#### Gates

| Command | Exit | Result |
|---|---|---|
| `python3 ci/scripts/check_cookbook_session_lifecycle.py --self-test` | 0 | `self-test: all 13/13 checks passed` |
| `python3 ci/scripts/check_cookbook_session_lifecycle.py` | 0 | `cookbook session-lifecycle gate: clean -- 80 file(s) scanned...` |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1028 file(s) scanned, all PATH:LINE citations resolve...` (2 pre-existing EXEMPT entries, unrelated) |
| `ruff check jammi_cookbook scripts tests` (run from `cookbook/book/`, the exact invocation `.github/workflows/cookbook-book.yml`'s `Lint` step uses) | 0 | `All checks passed!` (my only edit under that lint scope, `conftest.py`, is a docstring-only change) |
| `bash ci/scripts/merge_path.sh --only guards --skip-pg --skip-mdbook` | 1 (re-run with output redirected to a file and `$?` read directly afterward, not through a pipe — my first run piped through `tail -100` and mis-showed exit 0, since a pipeline's status is its LAST command's, `tail`'s, not `merge_path.sh`'s; corrected here) | `RAN 89  FAILED 1` (of the 89 guard-matrix LEGS; `exit "$failed"` at the bottom of `merge_path.sh` is exactly the failed-leg count, 1; the failing leg's own internal suite, `test_pod_substrate.sh`, further reports `195 passed, 19 failed`) — both new legs (`cookbook session-lifecycle gate`, `cookbook session-lifecycle gate (self-test)`) `ok`; the failing leg (`pod build substrate`) is the pre-existing, unrelated `cargo`-absent environment gap detailed in Uncovered above |
| (informational, not in the brief's list) bare `ruff check ci/scripts/check_cookbook_session_lifecycle.py` | 3 findings | `ci/scripts/` carries no `pyproject.toml`/ruff config of its own and is not linted by any CI job (confirmed: running the same bare `ruff check` against the pre-existing, CI-passing `ci/scripts/check_release_manifest.py` also produces findings under ruff's DEFAULT ruleset) — fixed the two real ones anyway (`typing.Iterable` -> `collections.abc.Iterable`; narrowed the defensive `except Exception` to `except (ValueError, TypeError)`) and left the executable-bit finding alone (every sibling `check_*.py` in `ci/scripts/` is non-executable; `chmod +x` would be a gratuitous convention break) |

#### Commits

```
1124512c fix(cookbook): #500 C1 — AST session-lifecycle gate for the non-pytest lanes (issue #539)
```
(branched from `b5a7aab9`, one commit; `git log --oneline main..HEAD` in this
worktree shows the same single commit — local `main` is already at
`b5a7aab9`, the wave-3b merge tip, so the base and the branch point coincide.)


## 8c. U7b-A2b — the two-host cluster driver, its workflow and the id-secrecy scan (landed as four commits; original tip `a180db21`)

The implementer's contract, folded by the lead after checking the four deliverable files, the label-only trigger with no `schedule:`, the re-registered cluster producer, and the F1–F3/A1–A4 revert-red oracles. It was "filed, not scheduled" in the plan; it is built here because the wave's cluster-leg artifact (row C9) cannot exist without it. The one real cluster run is the lead's, through the new `gpu-cluster.yml` dispatch on this branch.

Base: `feat/500-wave3c` @ `b3978107`. Branch `unit/a2b`. Four commits:
`0fdffde5` (driver/workflow/scan, re-registered gates, docs), `53c5d44e`
(dedicated F1/F2/F3/A1 oracles + the `_rpc_wait_for_members_ready`
extraction), `c752cccd` (a flaky test-harness bug in F1's own SIGINT arm,
found and fixed — §3), `a180db21` (one stale present-tense doc claim in
dev-gpu.md's own P8 paragraph, found and fixed — §1).

#### 1. Scope shipped

- **`ci/scripts/runpod_gpu_cluster.sh`** (new, 1084→~1140 lines): restored
  from the pre-excision tree (`git show 23ef24a9:ci/scripts/
  runpod_gpu_cluster.sh`, the commit immediately before U7b's own round-3
  excision `55276624`) and patched to close every round-3 finding (§2
  below). Deviation from a byte-for-byte restore: the reachability wait
  loop (originally inline in the executed-only block) is extracted into a
  new sourceable function, `_rpc_wait_for_members_ready` (`ci/scripts/
  `runpod_gpu_cluster.sh` (lines 350–431 at the unit tip)) — required to make A1 testable at all
  (the inline loop was reachable only by executing the whole driver).
  Behavior-preserving: `bash ci/scripts/test_gpu_cluster_lane.sh` stayed
  green across the extraction, before any new test was added.
- **`ci/scripts/gang_id_secrecy_scan.py`** (new, 824→~840 lines): restored
  verbatim from the same pre-excision tree, then patched for A2 (§2).
- **`.github/workflows/gpu-cluster.yml`** (new, 158 lines): restored
  verbatim — `run-cluster` PR label or `workflow_dispatch` only, no
  `schedule:`, `permissions: contents: read`, job-level (never workflow-
  level) `concurrency` with `cancel-in-progress: false` (a cancel would
  SIGKILL the runner, the EXIT trap never runs, and the rented cluster is
  orphaned).
- **`ci/scripts/test_gpu_cluster_lane.sh`** (new, 986→~1290 lines):
  restored verbatim, then extended with the F1/F2/F3/A1 oracles (§3).
- **Re-registered** (each REVERTED from the round-3 excision, since a
  driver now exists again): `check_cuda_run_artifacts.py`'s
  `GANG_LEG_PRODUCER_PATH["cluster"]` row and its self-test fixtures (the
  `expect_hit`/`expect_hit_only` P-E2 arms reverted to `expect_clean`, the
  F4 leg/producer-mismatch pair reverted to compare BOTH real drivers, a
  tracked `runpod_gpu_cluster.sh` stand-in re-added) — `_gang_check_leg_
  producer_binding`'s own fail-closed-on-no-row behavior (P-E2) is KEPT as
  a permanent invariant for any future third leg, only its docstring
  restated; `check_gpu_prove_once.py`'s `PAID_POD_LANE_TABLE["ci/scripts/
  runpod_gpu_cluster.sh"] = "gpu-cluster.yml"` row; `execution_surface_
  reachability_allowlist.txt`'s cluster-leg tuple row (plus the shared
  `--no-run` compile-check row's comment updated "two origins" →
  "three origins"); `ci.yml`'s two cluster-only guard-matrix rows (`gpu
  cluster lane suite`, `gang id secrecy scan self-test`) plus the PyYAML
  install condition extended to cover the lane's own `--read-on-block` use;
  `test_check_gpu_prove_once.py`'s `CLUSTER_YML_GOOD` fixture and the
  real-tree derived-driver-set assertions (both list sites, alphabetically
  re-sorted).
- **`ci/scripts/runpod_lib.sh`** (shared, +19/-4 lines): `_rp_rest`'s two
  `curl` calls gain `--max-time "$RP_REST_MAX_TIME"` (new var, default 30s,
  `ci/scripts/runpod_lib.sh:119`, `_rp_rest` at `:384`) — F1's other half
  (§2); the F13 tripwire comment reverted from "today the pod leg... a
  future cluster leg" back to "the pod leg... and the cluster leg... alike"
  (the cluster leg's remote text does call `_rp_zero_test_tripwire_lines`
  again, verified: `ci/scripts/runpod_gpu_cluster.sh:196-201`).
- **`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs`** (doc-comments +
  one runtime skip-message string only, no logic changed): restated to
  describe the shipped driver as the primary path and the by-hand
  procedure (dev-gpu.md) as its fallback. Deviation from the round-3
  excision's own inverse: the file has ALSO been rebuilt on the
  `BlockingCall`-witness API by other wave-3c work landed on the shared
  base branch since 23ef24a9 (unrelated to this unit); I edited the
  CURRENT tree's doc comments in place rather than reverting to the old
  23ef24a9 text, to avoid clobbering that unrelated, already-landed change.
- **`docs/maintainer/dev-gpu.md`**: cluster-leg section rewritten to
  describe the shipped driver as primary, the by-hand steps as fallback;
  states the cost bounds as the literal `$3.82`/`$26.71` figures (G4's own
  test greps for these exact strings — previously only `$3.816/h` was
  stated, which is the RATE, not either derived bound); folds in the
  lead's own executed pre-flight (below).
- **`docs/plans/67-distributed-training/UNITS.md`**: U7b-A2b's own section
  gets a dated correction (2026-09-16: built in wave 3c, no longer "filed,
  NOT scheduled"), restating each of F1-F3/A1-A4 as closed by construction;
  two forward-referencing claims inside U7a's own section ("no producer is
  registered... until U7b-A2b ships a driver", "has nothing to scan until
  U7b-A2b's driver exists") corrected to past tense now that both are
  false as CURRENT-state claims.
- **`docs/maintainer/pod-build-guide.md`** (NOT in the brief's own touch
  list — a forced, mechanical follow-on): `runpod_lib.sh`'s own comment
  restatements shifted every `PATH:LINE` citation into it by a constant
  offset per region (+8 for old lines 112-369, +15 for old lines ≥370, +0
  before); re-anchored by that exact mapping (verified against the actual
  new-tree content at each target line before accepting), with one
  hand-fix (`RP_TTL_HOURS` citations, old line 111, inside the +8 region's
  own insertion point, so 0 offset there specifically) — `check_citations.py`
  0 stale, 1038 files scanned, is the executed proof this mapping is
  correct, not merely plausible. **Scope amendment**: recorded below.
- **A4** (advisory, already resolved on this base tree before this unit —
  no code change needed): `_rp_cluster_payload`'s request shape stays the
  fixed 2×1 literal (`compute.gpuCountPerPod: 1, compute.podCount: 2`,
  `ci/scripts/runpod_lib.sh:1500-1509`-ish) — no caller needs any other
  shape; verified unchanged since M1 landed.

**No engine crate touched beyond gang_nccl.rs's doc comments/skip message**
(no Rust logic, no doctest blocks — verified: `grep -c '```' gang_nccl.rs`
finds none near my edits). No live RunPod call in this unit.

#### 2. Round-3 findings F1–F3 and advisories A1–A4, each closed by construction

- **F1 — the scan reachable only from a skippable EXIT-only trap sequenced
  behind untimed REST calls.** Closed two ways: (a) `_rpc_cleanup_cluster`
  (`ci/scripts/runpod_gpu_cluster.sh:869-1004`) now calls `_rpc_scan_or_
  destroy` FIRST, before `_rpc_self_remove_status`/`rp_cluster_delete`; (b)
  the trap registration (`:1016-1019`) is `EXIT`, `HUP`, `INT`, `TERM` —
  never `EXIT` alone — each signal passing its own conventional 128+n code
  explicitly (`local rc="${1:-$?}"`, deterministic regardless of what `$?`
  holds when a signal, rather than a normal `exit`, invokes the trap); the
  function disarms all four traps at its own entry (`trap - EXIT INT TERM
  HUP`) to prevent re-entry. `_rp_rest`'s own curl calls (`ci/scripts/
  `runpod_lib.sh` (lines 384–403 at the unit tip)) now carry `--max-time "$RP_REST_MAX_TIME"`
  (default 30s, `:119`) — previously unbounded.
- **F2 — a false "WILL BE DESTROYED" claim under `RP_SESSION`.**
  `_rpc_scan_or_destroy` (`ci/scripts/runpod_gpu_cluster.sh:813-864`) now
  `rm -rf`s the relocated quarantine directory SYNCHRONOUSLY, in the same
  function call, never deferring to `rp_cleanup`'s own `RP_WORK_IS_TEMP`-
  conditional teardown (which an exported `RP_SESSION` clears at
  `runpod_lib.sh` source time — the by-hand fallback's own exact shape).
  The log line is now past tense ("destroyed there, now, unconditionally"),
  stated truthfully.
- **F3 — the in-place fallback's globs missing `..`-prefixed names.** The
  fallback `rm -rf` (`ci/scripts/runpod_gpu_cluster.sh:850` area) gains a
  third glob, `..?*`, alongside `*` and `.[!.]*`.
- **A1 — `rp_cluster_pods`'s member-count check used `-ge` rather than an
  exact match.** The wait loop (now `_rpc_wait_for_members_ready`) tracks
  two DISTINCT boolean flags (`rank0_seen`/`rank1_seen`), never a raw tally
  — a duplicate/stale row naming the same rank twice can no longer satisfy
  readiness without both real ranks ever being seen.
- **A2 — the scan's hex needles were not whitespace-stripped.**
  `_scan_bytes` (`ci/scripts/gang_id_secrecy_scan.py:179-200`) extends the
  whitespace-stripped fallback (previously `base64`-only) to `hex`-prefixed
  needles too; `raw` deliberately excluded (stripping whitespace from
  arbitrary binary risks a different false-negative class).
- **A3 — the tee'd run log was never explicitly flushed/waited.** The
  driver captures `_RPC_TEE_PID=$!` immediately after `exec > >(tee -a
  "$RUN_LOG") 2>&1`; `_rpc_cleanup_cluster`'s own last act, after `rp_
  cleanup`, is `exec 1>&- 2>&-` (closing the fds feeding `tee`, so it sees
  EOF) then `wait "$_RPC_TEE_PID"`, before the final `exit "$rc"`.
- **A4 — `_rp_cluster_payload`'s request shape was hardcoded.** Kept FIXED,
  as already disposed at the excision commit — no caller on this tree
  needs any other shape; verified unchanged (§1).

#### 3. Properties (a table)

| Property (quantified) | Executed oracle | Executed mutation (red output, first line) |
|---|---|---|
| F1: the cleanup trap invokes the scan strictly before either of its own REST calls, on every path through `_rpc_cleanup_cluster` | `ci/scripts/test_gpu_cluster_lane.sh` "F1: the id-secrecy scan runs BEFORE..." (:413) | Executed: swapped the two blocks in `_rpc_cleanup_cluster` (cluster-id REST block first, scan call after) on the committed file, reran `bash ci/scripts/test_gpu_cluster_lane.sh` — `FAIL - F1: expected 'SCAN_CALLED REST_CALLED '; got order: 'REST_CALLED SCAN_CALLED ' — the scan is no longer sequenced first` (87 passed, 1 failed); restored the original file byte-for-byte (`git status --short` clean afterward), reran — 88 passed, 0 failed. |
| F1: the trap fires under SIGTERM/SIGHUP (dynamic), and all four registrations (EXIT/HUP/INT/TERM) exist verbatim (static, for INT — see below) | `test_gpu_cluster_lane.sh` "F1: the cleanup trap fires under SIG${sig}..." (TERM/HUP, dynamic) + "F1: all four trap registrations..." (static) | Dropping the three signal-specific `trap` lines (only `trap _rpc_cleanup_cluster EXIT` left) reds both the dynamic TERM/HUP arms AND the static all-four-present check — not independently re-executed as a fresh mutation in this unit (the design-time repro that led to this shape already exercised it; the static check's own executed history is below). |
| F1's own test-harness bug, found and fixed mid-unit: the SIGINT-delivery arm was flaky, not a driver defect | n/a (a defect in the TEST, not the property) | Executed: a dynamic SIGINT-delivery arm (send SIGINT to a `&`-backgrounded subshell registering all four traps) was added first, passed 4/4 direct runs, then FAILED under `merge_path.sh --only guards`'s own nested invocation (`gpu-cluster-lane: 87 passed, 1 failed`, naming exactly the SIGINT arm). Isolated with two standalone repros: `sigtest.sh` (backgrounded via `&`, job control off) — SIGINT never arrives, the trap never fires, the process stays alive; `sigtest2.sh` (the identical script run in the FOREGROUND, no backgrounding) — SIGINT fires the trap immediately (`GOT_INT` then `GOT_EXIT`, exit 130). This is bash's own documented behavior (SIGINT/SIGQUIT, and ONLY those two, forced to SIG_IGN for an asynchronous list command in a job-control-off shell, before that child's own `trap` ever runs — irreversible once already SIG_IGN at shell entry) — an artifact of THIS test's own need to background a driver-emulating subshell to signal it while running, never something the real driver (which runs in the foreground of its own CI step) is subject to. Replaced the dynamic SIGINT arm with a static grep-based check (matching this file's own `RpSshoRequiresRpInitTest`-class precedent for properties a dynamic test cannot reliably exercise); a first version of that static check itself had a bug (BSD grep's basic-regex mode does not treat `\|` as alternation — only one of the four patterns matched), caught the same way — `FAIL - F1: expected exactly 4 trap registration lines...; got: 1019:trap '_rpc_cleanup_cluster 143' TERM` (one line, not four) — fixed with `grep -nE` and unescaped `|`; reran, 4/4 lines matched. `bash ci/scripts/test_gpu_cluster_lane.sh` run 5x consecutively after the fix: 88 passed, 0 failed every time; `merge_path.sh --only guards` rerun after the fix: `gpu cluster lane suite` is `ok` (§5). |
| F2: a dirty carrier is destroyed synchronously regardless of `RP_SESSION`/`RP_WORK_IS_TEMP` | `test_gpu_cluster_lane.sh` "F2: under RP_SESSION..." (:519) + "F2 revert-RED..." (:565) | Reverting `_rpc_scan_or_destroy`'s destroy step to the original quarantine-and-defer text (move only, no `rm -rf`, "WILL BE DESTROYED" restored) on a scratch copy: `gpu-cluster-destroy-*` now SURVIVES under `$RP_WORK` — executed, `find "$F2_SESSION_ROOT" ... -name "gpu-cluster-destroy-*"` finds a match (red for the FIXED assertion, green for the revert-RED's own inverted assertion). |
| F3: the in-place destroy fallback matches `..`-prefixed names | `test_gpu_cluster_lane.sh` "F3: the in-place fallback..." (:602) + "F3 revert-RED..." (:641) | Reverting the glob set to `*`/`.[!.]*` only (dropping `..?*`) on a scratch copy: the planted `..leak` file SURVIVES — executed, `[ -e "$F3_ARTIFACT_REVERT/..leak" ]` true (red for the fixed assertion). |
| A1: readiness requires rank 0 AND rank 1 each seen, never a raw count | `test_gpu_cluster_lane.sh` "A1: two rows BOTH claiming rank 0..." (:904), "A1: rank 0 and rank 1..." (:917), "A1 revert-RED..." (:984) | Reverting to `ok_count`/`-ge 2` (both the loop's own break condition and the post-loop refusal check) on a scratch copy: the duplicate-rank-0/no-rank-1 fixture breaks the loop "ready" and proceeds into the member-resolution phase with rank 1 unset — executed, a DIFFERENT (generic) downstream failure than the fixed code's accurate, named refusal for the SAME fixture. |
| A2: hex-encoded, line-wrapped ids are still detected (whitespace-stripped fallback, not base64-only) | `gang_id_secrecy_scan.py --self-test`: `test_line_wrapped_hex_lower_is_still_a_hit` (:778), `test_line_wrapped_hex_upper_is_still_a_hit` (:791) | Reverting `_scan_bytes`'s guard from `name.startswith("base64") or name.startswith("hex")` back to `name.startswith("base64")`: both new tests FAIL — executed, `Ran 31 tests ... FAILED (failures=2)`, both naming the two hex-wrap tests; restored, `OK`. |
| A3: this process explicitly waits for its own `tee` process substitution before exiting | Not independently oracled by a NEW dedicated test in this unit (see §4, Uncovered) — closed by construction, reviewable by inspection (`ci/scripts/runpod_gpu_cluster.sh` around `_RPC_TEE_PID=$!` and the `exec 1>&- 2>&-`/`wait` pair). | None executed — see §4. |
| The re-registered cluster-leg producer row: a complete cluster-leg `gang` artifact is accepted, and a leg/producer mismatch (either direction) is refused | `check_cuda_run_artifacts.py --self-test` (whole-suite oracle, gang-leg rows) | Reverting `GANG_LEG_PRODUCER_PATH` to omit the `cluster` key (the round-3 P-E2 state) while running the reverted-back self-test fixtures (which now assert `expect_clean` for a complete cluster artifact) reproduces the round-3-era `expect_hit` self-test failure class — verified by construction (the round-3 excision commit's own diff is the executed record of this exact class; not independently re-executed as a fresh mutation in this unit, since the pre/post states are the two historical commits themselves, `23ef24a9`→`55276624`→this unit). |

#### 4. Uncovered (named, never claimed closed)

- **A3's own oracle**: no NEW dedicated mocks-only test drives the
  tee-wait mechanism end-to-end (e.g. a stubbed slow `tee` in `PATH` with a
  timing race) — the mechanism is reviewable by inspection and the
  existing P-A oracles all pass through `_rpc_cleanup_cluster` (which now
  includes the wait), but none specifically PROVES bytes would otherwise
  be lost without it. Labelled UNCOVERED, not claimed closed by a test.
- **The one real cluster run** (member sshd reachability on a genuine
  cluster, `ens1` as the overlay iface, member self-removal): still
  UNMEASURED. The lead's own executed pre-flight (2026-09-16, pod
  `rln5hfmn4viu06`, REST v2, RTX A4000, SECURE, `EUR-IS-1`) settled ONLY
  whether REST v2's `args` reaches `bash -c` on `RP_IMAGE` — confirmed yes,
  matching the GraphQL path S4 already measured — but that probe was a
  single ordinary POD, never a cluster: it showed only `eth0`/`lo` on
  `/sys/class/net`, no overlay network. This unit ships no live RunPod
  call; the one real cluster run moves to the lead, after merge, under the
  standing authorization (≤ 1h billed, ≤ 2 runs, label-only until a
  flake-free streak).
- **Pre-existing findings on the base tree, NOT introduced by this unit**
  (verified via `git diff b3978107..HEAD --stat` touching none of these
  paths): `merge_path.sh --only guards --skip-pg --skip-mdbook` (run 3
  times across this unit's own commits — §5) reports 3 FAILs out of 89
  commands every time it is at its own steady state (runs 1 and 3; run 2's
  own 4th FAIL was this unit's own flaky test-harness bug, found and fixed
  — §3, not a pre-existing finding): (a) "pod build substrate"
  (`test_pod_substrate.sh`) — environment gaps in this dispatched agent's
  own shell (`cargo: command not found`, no `--reflink` support), unrelated
  to any file this unit touches; (b) "kernel oracle standard"
  (`check_kernel_oracles.py`) — a stale `fn-in-literal reviewed` marker in
  `crates/jammi-server/tests/it/gang_terminal_write_oracle.rs`, last
  touched by commit `456a4937` (U5a-2, a different unit, before this
  unit's own base tip); (c) "arch validation freshness"
  (`check_arch_validation_freshness.py`) — `crates/jammi-kernels/src/
  admission.rs` outdated the arch-80/86/89/90 waivers, a different unit's
  own concurrent change. None of these three logs name any file this
  unit's own diff touches.

#### 5. Gates (real exit codes)

| Command | Exit | Notes |
|---|---|---|
| `bash -n ci/scripts/runpod_lib.sh` | 0 | |
| `bash -n ci/scripts/runpod_gpu_cluster.sh` | 0 | |
| `bash -n ci/scripts/test_gpu_cluster_lane.sh` | 0 | |
| `shellcheck -S warning ci/scripts/runpod_gpu_cluster.sh` | 0 | |
| `shellcheck -S warning ci/scripts/test_gpu_cluster_lane.sh` | 0 | |
| `shellcheck -S warning ci/scripts/runpod_lib.sh` | 1 | SC2034, `group_rcs_assoc_keys` — pre-existing on the base tree (verified against `git show HEAD:ci/scripts/runpod_lib.sh` before my commit; now shifted +15 lines), unrelated to this unit's own hunks |
| `bash ci/scripts/test_runpod_cluster_lib.sh` | 0 | 69 passed |
| `bash ci/scripts/test_gpu_cluster_lane.sh` | 0 | 88 passed; stable across 9+ consecutive runs (5 direct, 4 inside `merge_path.sh`'s own wrapping), after the SIGINT-oracle fix (§3) |
| `python3 ci/scripts/gang_id_secrecy_scan.py --self-test` | 0 | 31 passed (29 pre-existing + 2 new hex-wrap arms) |
| `python3 ci/scripts/check_gpu_prove_once.py` | 0 | |
| `python3 ci/scripts/test_check_gpu_prove_once.py` | 0 | 207 passed |
| `python3 ci/scripts/check_cuda_run_artifacts.py --self-test` | 0 | |
| `python3 ci/scripts/check_cuda_run_artifacts.py` | 0 | |
| `python3 ci/scripts/check_execution_surface_reachability.py` | 0 | one pre-existing, unrelated "suspicious unregistered line" note (`ci/scripts/perf/test_grad_oracle_cross_producer_parity.py:37`) |
| `python3 ci/scripts/perf/check_citations.py` | 0 | 1038 files scanned, 0 stale, 2 exempt (pre-existing, unrelated legacy artifact citations) |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | |
| `python3 ci/scripts/check_doc_parity.py` | 0 | |
| `bash ci/scripts/test_gpu_dev_lifecycle.sh` | 0 | 157 passed |
| `bash ci/scripts/test_gpu_gang_lane.sh` | 0 | 54 passed |
| `bash ci/scripts/test_gpu_prove_lane.sh` | 0 | 74 passed (one printed FAIL line is an intentional RED-control fixture assertion, not a suite failure — pre-existing shape) |
| `bash ci/scripts/merge_path.sh --only guards --skip-pg --skip-mdbook` | 1 (runs 1, 2, 3 — each `merge_path`'s OWN summary line, not the shell's, since it reports per-command results and exits 1 whenever ANY command failed) | Run 1 (before the F1/F2/F3/A1 oracle commit): `RAN 89 FAILED 3` — the three pre-existing, out-of-scope findings named in §4. Run 2 (after that commit, before the SIGINT-oracle fix): `RAN 89 FAILED 4` — the same 3, PLUS `gpu cluster lane suite` (the flaky SIGINT arm, §3) — this is what surfaced the flake. Run 3 (final, after all four commits including the SIGINT fix): **`RAN 89 FAILED 3`** — back to exactly the three pre-existing findings; `gpu cluster lane suite` is `ok`, confirmed under the exact nested `bash -e -c` invocation shape that caught the flake in run 2. Full log: `(scratchpad)/a2b-scratch/merge_path_guards4.out`. |

#### 6. Cost ceiling restated

`2 × $1.908/GPU/h = $3.816/h` (S4's measured SECURE-cluster rate). (i)
terminate-succeeds: `1h × $3.816/h = $3.82` per run. (ii) sweep-only (the
EXIT trap's own delete fails, member self-removal unmeasured): `(1+6)h ×
$3.816/h = $26.71`. Standing authorization (2026-09-13): ≤ 1h billed wall
per run, ≤ 2 runs, label-only (`run-cluster`) until a flake-free streak.
This unit ships NO live RunPod call; the one real cluster run belongs to
the lead, after merge.

#### 7. The lead's own pre-flight (folded into this unit's record)

Mid-task, the lead reported an executed pre-flight I had not run and could
not run myself (a real billable RunPod transaction): a single-pod REST v2
create (pod `rln5hfmn4viu06`, RTX A4000, SECURE, `EUR-IS-1`, created
2026-09-16T02:55:54Z, deleted 03:07Z) with `args: "bash -c '...'"` on
`ghcr.io/f-inverse/jammi-ai-ci-cuda:latest`. The container log printed the
args-echoed marker (`PREFLIGHT-ARGS-OK`), then `nvidia-smi -L` (`GPU 0:
NVIDIA RTX A4000`), then `/sys/class/net` (`bonding_masters eth0 lo`); the
read-back `Pod.args` on a subsequent GET matched the sent text exactly.
This closes U7b's own contract §1 gap (whether REST v2's `args` field
reaches `bash -c` the way the pod path's GraphQL `dockerArgs` field
measurably does) as EXECUTED — folded into `docs/maintainer/dev-gpu.md`'s
cluster-leg section and `docs/plans/67-distributed-training/UNITS.md`'s
own dated correction (§1 above). Still unmeasured, honestly, because this
probe was a single ordinary pod, never a cluster (§4): `ens1` as a cluster
member's own overlay iface, member sshd reachability on a real cluster,
and member self-removal.

#### 8. Scope amendment

`docs/maintainer/pod-build-guide.md` is outside the brief's own touch list
(`ci/scripts/**`, `.github/workflows/**`, `docs/maintainer/dev-gpu.md`, the
plan rows) but was forced into scope by `runpod_lib.sh`'s own comment
restatements shifting every `PATH:LINE` citation into it — COMMON.md's own
warning ("your insertions shift other files' PATH:LINE anchors") names
exactly this class. Re-anchored by the shift's own exact mapping, verified
against `check_citations.py` (0 stale, 1038 files). No other file outside
the brief's list was touched.

#### 9. Commits

```
$ git log --oneline b3978107..HEAD
a180db21 docs(maintainer): #500 U7b-A2b — dev-gpu.md's P8/schedule-visibility paragraph named the driver as still absent
c752cccd fix(ci): #500 U7b-A2b — F1's own SIGINT oracle was flaky, unrelated to the driver
53c5d44e test(ci): #500 U7b-A2b — dedicated RED-then-GREEN oracles for round-3's F1/F2/F3/A1, each with an executed revert-RED mutation
0fdffde5 feat(ci,docs): #500 U7b-A2b — the two-host cluster driver, its own workflow, and the id-secrecy scan (re-attempt, round-3 findings closed by construction)
```


## 8d. Residuals found by the full local run and the draft PR's CI (each root-caused; two filed)

- Four server tests red on the consolidated tree: the terminal-write oracle's source mask kept the CHAR count while its offsets are BYTES (a multi-byte character in a comment shifted the fn-body slice) — the mask is byte-preserving now, in both oracles; three remote-compute rows submit two-rank specs under the new serveable-world bound — the server fixture declares `n` ranks serveable beside `n` devices.
- The Postgres test-hooks lane: #574's malformed-stamp row, left on the shared database, faulted the sibling prune test's SQL-side sweep — the test deletes its row; the product-side residual (one unreadable stamp faults every SQL-side sweep: `prune_instances`, `list_gang_members`, `peer_addr_of`, claim/reclaim) is **issue #585**, UNCOVERED here; the root fix is a schema-edge domain on the stamp columns, which would also retire #574's `Undecodable` arm.
- Four CI lanes (OSS-only build, Python, Smoke, dep-DAG freshness) red on one cause: a `rank` binding in the rank body read only under test-hooks, dead under every other build's denied warnings.
- The draft PR's CI on `2e43af22`: `check_flash_attn_closure` refused the cluster driver's two cuda-bearing
  tuples as unlisted — the gate enumerates every gated tuple's origin under `ci/scripts/**` and admits it
  only from `PROVE_SCOPE` or `EXEMPT_SCOPE`, and A2b registered the driver with every other gate but this
  one. The driver has its own `EXEMPT_SCOPE` row (the same reason as the pod leg's: a distributed-training
  correctness surface that declares nothing in `prove_lane.crates`) and the self-test fixture the row
  requires (a row without a fixture reds the GOOD fixture by construction). Rustdoc under `-D warnings`
  refused three intra-doc links in the `worker.rs`/`role.rs` module docs that named `LeaseHolder`,
  `RunnerRole` and `LeaseHolder::LoopClaimer` bare from a scope where they are not imported; they name
  `crate::fine_tune::role::…` now. Both in one commit; the merge path's static, guard and swarm stages were
  re-run on it (§10). The tests stage's evidence from `2e43af22` carries for that delta (a gate's scope
  table, its fixture, doc comments); the topology-determinant fix below changes production code in
  `worker.rs`, so the hermetic lane was re-run on the final tip for every crate that can observe
  `jammi-ai` (`jammi-ai`, `jammi-server`, `jammi-bench` — the reverse dependency closure minus
  `jammi-python`, which the lane excludes; §10). The Postgres lanes' evidence carries: `jammi-db`'s two
  lanes cannot observe `jammi-ai`, and the `jammi-server` introspection lane runs no fine-tune job
  (`grpc_introspection.rs` names none).
- The phase-5 oracle (PASS at `2f27d6a1`, §10) found the manifest's `collective` determinant recording
  the `[worker] collective` SELECTION (`auto|nccl|cpu`) where the field's own contract says the
  collective the run reduced over (`noop|local|peer|nccl`): `auto` resolves differently on different
  hosts, so the recorded token was neither the determinant nor a stable name for it; `local_ranks`
  recorded the `[worker] local_ranks` CAPACITY where the field says the width this host ran at. Both
  are total functions of the `RankTopology` §6 decides before the descriptor is built, and the
  descriptor reads them off it (`RankTopology::collective_token`/`host_ranks`: `Single` → `noop`/1,
  `Local { world }` → `local`/`world`, `Peer` → `peer`/1 — rank 0 alone on this host); the unit oracle
  `rank_topology_records_the_executed_collective_and_host_ranks` samples every arm with a
  non-degeneracy check, and each match is exhaustive. The identity consequence of the old token was
  bounded (the environment folds the device; `Peer` ≡ `Local` bit-for-bit by §3's parity oracles), which
  is why the oracle did not block; it is fixed at the root rather than reconciled in the doc. The oracle's
  second residual — required fields added to the persisted `FineTune` descriptor at an unchanged
  `MANIFEST_VERSION` make a pre-existing FineTune sidecar a hard decode error — is the variant's
  designed behaviour for determinant growth (K1 replay for this variant is retrain; no external
  consumers; the release is held), stated here, not changed.
- The oracle's second run (PASS at `0ae786fc`) returned three doc findings, each fixed where it lives:
  the field's own definition (`ProducingDescriptor::FineTune::local_ranks`) still sourced the value from
  `[worker] local_ranks` "at claim time" and called it the gang width — the same class as the fix above,
  on the consumer side; `docs/guide/src/format-stability.md` claimed an older-or-equal manifest version
  is "readable by construction (the layout only grew)", which the required topology determinants at an
  unchanged `MANIFEST_VERSION` falsify for a pre-existing fine-tune sidecar — the reject-newer paragraph
  now states what the reader does (a body lacking a required field is the typed `ManifestError::Serde`;
  the one named older shape, no `leaves`, is absent); and the `worker.rs` module-doc links whose bare
  form resolves (the types are imported there) lost the explicit targets rustdoc refuses as redundant
  (`role.rs`'s module doc keeps its qualified link: nothing is in scope there). Its third observation —
  `Cargo.lock`'s `heck 0.4.1 → 0.5.0` re-point on the `prost-build`/`snafu-derive` build-dep edges, a
  resolver dedupe beside the four added edges — is informational.
- The oracle's third run (PASS at `c21492b9`) checked the three doc statements against the code and refuted
  the half of the format-stability paragraph kept from `main`: the materialization manifest's reader refuses
  ANY version inequality (`MaterializationManifest::from_json_bytes`, `!=`, as its own rustdoc states — an
  older version names a superseded determinant set), so the guide's table row, bullet and section calling it
  "reject-newer" (`found > MANIFEST_VERSION`) were wrong on `main` and stayed wrong through the previous
  rewrite. The three statements now say exact-version, keep reject-newer for the ordered formats
  (`.rowmap`, ANN `.manifest.json`, whose readers compare with `>`), and state the decode-before-stamp
  order. Doc-only; the gates and the oracle were re-run on it.
- The pod leg: run 3 measured the property (§2c) and its artifact is committed; run 4 (workflow 35054325406, on `ddd68928`) is the fully green job after the pod's clone was deepened blobless for the registry's ancestry rule.

## 9. Pressure round (phase 1, executed at `856ec8dd` before the code landed) — REFINE, eight blocks folded

The design of the four wave-A units was attacked read-only against the base tree; the lead
opened every cited line before relaying. Each block below names the decision that folds it,
sent to the implementer while its unit was still being built; the commit closing each is
recorded at consolidation.

| # | Finding (cited) | Decision folded |
|---|---|---|
| 1 | U4b (b) "bit-for-bit summed adapter gradient vs W=1" is unachievable under f32 reassociation (`batch_bucket.rs` (lines 304–315 at the unit tip) already documents 1e-5 for the padding half) and is satisfiable only by every rank computing the whole global gradient and skipping the reduce | (b) restated as a pre-registered ε whose discriminator is the W× hazard; bit-identity kept only for W=2-twice and resume-vs-uninterrupted; the "trainable op after the gather" mutation stays the hazard oracle |
| 2 | Zero ≠ absent for AdamW (`adamw.rs` (line 245 at the unit tip) skips an absent Var, steps a zero-gradient one; `optimizer.rs` (lines 626–633 at the unit tip) documents absence as designed) | the reduce reduces the presence set (union across ranks) and restores "absent on every rank ⇒ absent" before `clip_and_step`; never a `world > 1` guard |
| 3 | Both epoch loops end on THIS rank's first empty chunk (`trainer.rs` (lines 1400–1404 at the unit tip), `1453-1456`); the optimizer-step boundary counts only non-diverged batches (`2881-2893`, `2903`) | epoch end = the global step count from the partition rule; a zero-row rank takes the step with a synthesized `[0, hidden]` tensor; the divergence decision is flag-reduced BEFORE accumulation so all ranks skip together (at W=1 `Noop` returns the rank's own flags — the W=1 window is unchanged); the one-rank divergence oracle uses the `cfg(test)` poke seam and says so |
| 4 | Hard-negative mining (`1303-1315`), GradCache (`1344-1362`) and the Precomputed arm (`1317-1343`) have no gather story; `gradcache.rs`/`hard_negative_miner.rs` in no scope | typed refusal at the spec admission edge for `world_size > 1` with mining, GradCache or a precomputed loader; their gather is a named follow-up; (b)/(d) oracles use the real loaders |
| 5 | The lifted `Descriptor` drops the round generation `Local` keeps outside it (`local.rs` (lines 174–193 at the unit tip)); a commit-phase fault leaves one rank un-applied and the next fold is silently wrong | the round index is bound into the agreed descriptor; any commit-phase fault is fatal on every rank |
| 6 | `tests/distributed` already exists behind `live-distributed-tests` with a Postgres+MinIO harness and never gates a PR (`Cargo.toml` (lines 335–338 at the unit tip), `distributed.yml` (lines 24–26 at the unit tip)) — (a),(c),(e),(f) would report green without executing | the Peer-vs-Local fold, deadline, commit-phase fault, descriptor disagreement, corrupted leaf and decode cap move to a hermetic two-process target in the ordinary test lane; fleet-dependent rows join the matrix by name |
| 7 | `RankAdmissionRow` carries no tenant (`jobs_repo.rs` (lines 212–250 at the unit tip), deleted in #566 round 2); `get_job` is tenant-filtered; admin scope forbidden — no route to the job's tenant for R2(b) | the admission row carries the row's OWN `jobs.tenant_id` (rebuilt carrier, its own oracle on both backends); the strict resolver pins to it |
| 8 | "Backend SQL clock on both dialects" contradicts the lease discipline: SQLite binds the application clock by design (`lease.rs` (lines 13–26 at the unit tip)), and the two backends store different representations (`47-56`) | the cooldown reuses `lease_expired_clause`/`lease_deadline_expr`/`lease_now`/`lease_deadline` — no second clock source; the process-skew oracle is Postgres-only; a source-scan oracle pins "no literal clock outside the lease helpers" |

Advisories folded in the same messages: no rung exchange and no pin in U4b (the gather is dim 0
over pooled `[rows, hidden]`; `batch_bucket.rs` (lines 96–104 at the unit tip)'s rationale rewritten; (c)'s ε covers
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

## 9a. Main defects found while wave 3c was built — root-caused and folded here (the user's instruction: one PR)

Three defects on `main` surfaced from CI on the #580 merge tip and were each root-caused with a RED→GREEN oracle on their own branch, then folded into this branch with their standalone PRs (#581, #582, #583) closed:

| Defect | Root cause | Fix (commits on this branch) |
|---|---|---|
| The Shape B compose smoke had failed on every `main` run since wave 1 (#521): after RELEASE the restarted container never came back | `docker compose kill -s SIGINT` is recorded by the daemon as a MANUAL stop, which exempts the container from `restart: unless-stopped`; the script's step 4 could never observe a restart through that API. Same runner image and Docker 28.0.4 on the last green run (2026-09-12, before the script existed) and the first red one | SIGINT is delivered to the container's init process from the host pid namespace (`docker inspect`'s `.State.Pid`, `sudo -n kill`), what an operator's `kill` does; the post-release proof is the END STATE inside the lease window (`releases = 1`, `attempts = 2`, running/completed, lease held while running) — the instant between the release and the successor's reclaim is a few hundred milliseconds wide and any snapshot of it is a race by construction; the compose comment and the Kubernetes README no longer claim a restart follows a compose kill. Verified green on both arches by three dispatched runs of the main-only workflow |
| DRAIN on a Postgres catalog intermittently stalled the full 30 s `close_pool_and_drain` ceiling with `remaining=1` | No jammi task held the connection (every `PoolConnection` is closure-scoped). sqlx-core 0.8.6's `Pool::close` sweeps the idle queue then waits on the permits (sqlx-core 0.8.6, `src/pool/inner.rs`, the close pass at lines 97–115); a concurrently returning connection pushes onto idle BEFORE releasing its permit (the same file, `release` at lines 200–211), so a return landing after the pass's last sweep leaves `close()` returned with one idle connection nothing ever sweeps | `crates/jammi-db/src/catalog/backend.rs::close_pool_and_drain` loops `Pool::close` passes until `size() == 0`; the ceiling remains only for a driver-level close that never completes; the warning names the consequence per backend. A `test-hooks` per-pool `after_release` park widens the window deterministically: `close_barrier_tests` (sqlite, postgres) RED 60.65 s → GREEN 0.62 s; `serve_shutdown_modes` idle/after-traffic DRAIN RED 33–35 s → GREEN within the park + < 1 s. Residual, not fixed: the lease keeper's SQLite close waits its 2 s sidecar ceiling on every SQLite DRAIN/RELEASE |
| The hermetic lane was red 3 of 3 on this code since #580, each time a DIFFERENT `jobs_cancel` test at its 15 s "observe promptly" bound | Neither observer was slow: `training_test_hooks::pause_slot()` was a process-global unkeyed pause (unkeyed since #485); `checkpoint_before_spawn_blocking()`, called by every `train_fine_tune` before `spawn_blocking`, took whichever pause was armed, so a sibling 20 000-epoch training test reaching it inside the 14–29 ms window after the arm parked forever and its cancel/lease-loss signal had no trainer to be observed by. Both CI timelines show the arming test passing within ~50 ms of the victim's wait beginning and the `FAILED` exactly 15.0 s later; `jobs_shutdown.rs` had documented the hazard and routed one test around it | The pause is a registry keyed by job id (the shape every other park in `training_test_hooks` has); a sibling's checkpoint passes through. Tip: 5 of 6 red unloaded; fixed: 10/10 green unloaded and 10/10 under a load average of 20.7 with the fine-tune, lib and jammi-db suites looping; mutation (take the first armed entry regardless of job) reds the unit oracle and reproduces the CI failure. The 15 s bounds are untouched. Consolidation note: the one cherry-pick conflict was `worker.rs`'s test-hooks module, where the coordinator body had added its recorders in the same region; resolved by keeping both, type-checked |

## 10. Gates (the merge path)

**Expected red at merge: `SWARM_GATE_TOUCHED`.** This branch adds one gate definition,
`ci/scripts/check_cookbook_session_lifecycle.py` (C1, issue #539 — a plan unit whose deliverable
IS a gate), and U7b-A2b extends `check_gpu_prove_once.py`'s P7 judgement and
`check_cuda_run_artifacts.py`'s cluster-leg producer registry. Gate definitions are
human-amend-only by design, so the swarm guard reds on them and the PR is admin-merged once every
other check is green (the standing authorization: admin merge when `SWARM_GATE_TOUCHED` is the
sole red). The reviewer's attention belongs on those three files.


`bash ci/scripts/merge_path.sh` on the consolidated tip with `JAMMI_TEST_PG_URL` pointing at a
local PostgreSQL 16 in the CI lane's shape — run ONCE by the lead, never per implementer; the
phase-5 oracle dispatched after every other stage is green; only `docs/rigor/**` committed
after it.

**What ran, on which tip (the lead's runs, logs in the session scratchpad):**

| Tip | Stages | Result |
|---|---|---|
| `2e43af22` | static, guards, swarm, tests (all four; the three Postgres lanes included) | 117 ok; red: rustdoc (the three module-doc links, §8d) and `SWARM_GATE_TOUCHED` |
| `2f27d6a1` | closure row + doc links | `check_flash_attn_closure` PASS, rustdoc links verified; the phase-5 oracle PASS here found the topology-determinant defect (§8d) |
| `0ae786fc` | static, guards, swarm | rustdoc red again (explicit targets rustdoc refuses as redundant where the bare link resolves, §8d); the phase-5 oracle PASS here returned the three doc findings (§8d) |
| `c21492b9` | static, guards, swarm; then the hermetic lane for `jammi-ai`, `jammi-server`, `jammi-bench` (`jammi-ai`'s reverse dependency closure, §8d) | 112 ok, the only red `SWARM_GATE_TOUCHED` (expected, above); tests: 17 targets, 2024 passed, 0 failed; the phase-5 oracle PASS here returned the exact-version finding (§8d) |
| `6fd5aeff` (docs only) | static, guards, swarm | 112 ok, the only red `SWARM_GATE_TOUCHED`; the phase-5 oracle PASS at this tip is the record below (its `head_sha`), and every commit after it is `docs/rigor/**` only |

The phase-5 oracle's record of the final tip is `docs/rigor/feat_500-wave3c.oracle.jsonl`; the
pressure row is `docs/rigor/feat_500-wave3c.jsonl`.
