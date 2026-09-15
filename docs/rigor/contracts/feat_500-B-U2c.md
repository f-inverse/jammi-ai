# CONTRACT — feat/500-B-U2c: streaming training-set loader with a residency bound (#544)

**Contract of record.** slug: `feat_500-B-U2c` · this file is the committed mechanism
contract `ci/scripts/check_rigor_record.py` requires under `docs/rigor/contracts/**` for
the rigor record at `docs/rigor/feat_500-B-U2c.jsonl`. Design contract of record:
`docs/plans/67-distributed-training/UNITS.md` §U2c (v4.6); the full design history (M1–M3,
the phase-0.5/phase-1 pressure folds, the c3b production-binding fold, the c3c writer fold)
lives in the session scratchpad's `CONTRACT-U2c.md` — every property below is restated
against the tree AS IT IS, not copied from that working document. Every citation is
`path::construct` (never `path:line`, which drifts under an unrelated edit), re-tagged
**(at `b27828be`)** — `fix(ai): #500 U2c c3d -- the streamed source's queries observe the job's tenant across the pump's spawn`, the branch's current HEAD as of this contract's own commit (c4b). This contract was first written at `4b03d5a9` (`fix(db): #500 U2c — a training set whose registration cannot declare its order says so`); two code commits landed after that write and are folded below rather than left stale: `7f493d87` (`fix(bench): #500 U2c -- the fine-tune bench trains through TrainingSource like production`) and `b27828be` (`fix(ai): #500 U2c c3d -- the streamed source's queries observe the job's tenant across the pump's spawn`). `git log --format=%h main..HEAD` for this branch:
`b27828be`, `7f493d87`, `803c8a18`, `4b03d5a9`, `af528c74` (merge), `aa3e940f`, `a9083ca6`,
`76d81b75`, `1f75f46f`, `a4b21cba`, `0b66c22f`, `a49bfa76` — `b27828be` is HEAD at this
contract's own commit time; every construct below has been RE-RESOLVED against `b27828be`'s
tree (not merely carried forward from the `4b03d5a9` write).

## Scope

`crates/jammi-db/src/store/mod.rs` (the committed order, both registration paths, the
writer's single-partition sort), `crates/jammi-db/src/session.rs` (the session memory
pool, `sql_stream`, `single_partition_context`), `crates/jammi-db/src/config/mod.rs` +
`crates/jammi-db/src/config/host_memory.rs` (`memory_limit` grammar and floor),
`crates/jammi-db/src/error.rs` (`ResourcesExhausted`, shared-declaration class),
`crates/jammi-server/src/grpc/wire.rs` (the wire mapping, a scope-amendment hunk this
unit's c2 flagged and the lead approved), `crates/jammi-ai/src/fine_tune/{stream.rs, decode.rs, source.rs, data.rs, training_set.rs, worker.rs, trainer.rs}` (the per-rank
stream, the Arrow decoder factored out of `worker.rs`, the `Resident`/`Streamed`
production binding).

## M1 — the provider declares the committed order (db)

**Mechanism.** `crates/jammi-db/src/store/mod.rs::training_set_sort_keys` is the ONE
source — every projected column, ascending, NULLs first, in declared order; both
`crates/jammi-db/src/store/mod.rs::training_set_order_by` (the SQL `ORDER BY` renderer)
and `crates/jammi-db/src/store/mod.rs::training_set_file_sort_order` (the DataFusion
`SortExpr` renderer) render from it, never re-deriving the direction/NULL placement
independently. `crates/jammi-db/src/store/mod.rs::ResultStore::bind_result_table` passes
the declared order to `register_table` for every single-fragment `TrainingSet` row, on
BOTH registration paths (fresh materialization inside `BuildingTable::finish`, and crash
recovery via `load_existing_tables`); `crates/jammi-db/src/store/mod.rs::ResultStore::training_set_registration_sort_order` is where the order is read back off the
table's `.materialization.json` sidecar — it legitimately returns `Ok(None)` when no
sidecar exists (a pre-migration-021 table) or the sidecar's descriptor is not a
`TrainingSet` variant, and BOTH arms `warn!`, naming the table and the reason, before
returning (this unit's own commit `4b03d5a9`; the sidecar-present, wrong-descriptor arm
already warned before this unit, the no-sidecar-at-all arm did not).

**Properties (over every input).**
- P1 (no sort in the plan). For every `TrainingSet` table and every
  `target_partitions ∈ {1, N>1}`, the physical plan of a committed-order read contains no
  `SortExec`; at `N>1` it contains a `SortPreservingMergeExec` (or a single output
  partition). A provider declaring `NULLS LAST` on a nullable sort column reinstates
  `SortExec` (the positive control).
- P1′ (registration cannot silently fail to declare). Every arm of
  `training_set_registration_sort_order` that returns `Ok(None)` is observed: a
  `tracing::warn!` naming the table fires before the return.

**Oracles, by name.**
- `crates/jammi-db/tests/it/materialization.rs::a_training_sets_registration_declares_its_order_so_the_read_back_plans_no_sort` (P1, both
  registration paths, with the schema-nullability non-vacuity check).
- `crates/jammi-ai/tests/it/training_set_stream.rs::p1_the_loader_derived_state_plans_with_no_sort_and_no_merge` (P1, the stream's own
  loader-derived, single-partition read).
- `crates/jammi-db/src/store/mod.rs::tests::the_two_order_renderers_agree_for_a_permuted_column_list` (the one-source property the
  two renderers share; excludes: a third hand-spelling entering by a future edit — caught
  only by this test's own permutation, not mechanically enforced).
- `crates/jammi-db/tests/it/materialization.rs::registration_warns_when_a_training_sets_sidecar_is_absent` (P1′, this unit's new test;
  verified RED when the `warn!` is reverted to a bare `Ok(None)`, executed not asserted).

## M2 — `engine.memory_limit` becomes the engine's memory pool (db; `error.rs` shared-declaration class)

**Mechanism.** `crates/jammi-db/src/config/mod.rs::EngineConfig::memory_limit_bytes` is the
ONE reader of `[engine] memory_limit`, parsing `"<n>%"` (of
`crates/jammi-db/src/config/host_memory.rs::total_physical_memory_bytes`, the lower of the
host physical total and a readable cgroup ceiling), `"<n>GB"`/`"<n>MB"`/`"<n>KB"`, or
`"<n>"` bytes; a resolved value below `crates/jammi-db/src/config/mod.rs::EngineConfig::MEMORY_LIMIT_FLOOR_BYTES` (64 MiB) is refused, typed, naming the key and the
floor. `crates/jammi-db/src/session.rs::JammiSession::build` resolves this once and installs
a `GreedyMemoryPool` on the SAME `RuntimeEnvBuilder`/`SessionStateBuilder` chain `ctx`'s
state is built from. `crates/jammi-db/src/session.rs::JammiSession::memory_pool` is how an
engine-side consumer registers its own `MemoryConsumer` against that pool;
`crates/jammi-db/src/session.rs::JammiSession::sql_stream` is `sql`'s streamed twin
(`execute_stream`, the tenant analyzer rule applied identically). Every over-budget grow —
a DataFusion operator's own reservation or an engine-side consumer's — surfaces as
`crates/jammi-db/src/error.rs::JammiError::ResourcesExhausted { limit_bytes, detail }`; at
the wire edge `crates/jammi-server/src/grpc/wire.rs::map_engine_error` maps it to
`Code::ResourceExhausted` (a scope-amendment hunk this unit's c2 flagged, wire-server owns
it, lead-approved).

**Properties.**
- P2 (typed, never counted). Every byte a loader holds resident is a `MemoryReservation` on
  the session pool; exceeding `memory_limit_bytes` surfaces as `ResourcesExhausted` from the
  loader's public path, never a panic, never a private-counter assertion. The eager read's
  collected batches reserve their whole collected size through the same pool
  (`crates/jammi-ai/src/fine_tune/training_set.rs::reserve_eager_batches`, the
  `training_set_eager`-named consumer).
- B5 (`memory_limit` grammar, load-time refusal). `"007"` (7 bytes) is refused by the floor,
  naming the key and the floor.

**Oracles, by name.**
- `crates/jammi-db/src/config/mod.rs::tests` — the `memory_limit` grammar/floor unit tests
  (percent, binary units, bytes, and the floor refusal).
- `crates/jammi-ai/tests/it/training_set_stream.rs::p3_streamed_read_completes_under_a_small_pool_while_eager_fails` (P2, the pool
  high-water-mark oracle through a consumer-filtered wrapper, never the aggregate pool).
- `crates/jammi-server/src/grpc/wire.rs::tests` — the `ResourcesExhausted` → `Code::ResourceExhausted` mapping (the wire-edge half of the scope amendment).

## M3 — the per-rank stream (ai-core)

**Mechanism.** `crates/jammi-ai/src/fine_tune/stream.rs::TrainingSetStream::open` derives a
loader-local, `target_partitions = 1` `SessionState`
(`crates/jammi-db/src/session.rs::single_partition_context`) from the caller's session,
runs ONE bounded-memory pre-pass over the window (schema check plus, for a numeric target,
a null/NaN aggregate — `crates/jammi-ai/src/fine_tune/decode.rs::check_schema_matches_format`), then spawns a pump that walks
`crates/jammi-ai/src/fine_tune/training_set.rs::read_back_sql`'s ordered stream batch by
batch, decoding ONLY the rows the current step's `crates/jammi-ai/src/fine_tune/stream.rs::RowWindow`/`crates/jammi-ai/src/fine_tune/stream.rs::Slice` selects
(`crates/jammi-ai/src/fine_tune/decode.rs::append_selected_rows`, the SAME decoder the
eager path uses), and emits one `OwnedChunk` per step through a bounded channel
(`crates/jammi-ai/src/fine_tune/stream.rs::StreamConfig::new`, refusing a zero prefetch
depth). `crates/jammi-ai/src/fine_tune/stream.rs::PRODUCTION_PREFETCH_DEPTH` (`= 2`) is the
one named constant production trains at
(`crates/jammi-ai/src/fine_tune/worker.rs::run_spec`), the regression pin for an earlier
`prefetch = 2` deadlock. `crates/jammi-ai/src/fine_tune/stream.rs::TrainingSetStream::next_chunk` is the blocking call the trainer's per-step loop drives.

**Properties.**
- P3 (the bound). `live_bytes(r) ≤ S₁ + (prefetch + 1)·C + carry + Σ E`, `S₁` = ONE
  in-flight DataFusion scan batch (not pool-accounted, DataFusion's own pipeline — the
  merge term is ZERO by construction, since the loader's own query plans at one output
  partition with no `SortPreservingMergeExec`), `C` = one chunk of B rows (pool-accounted
  via the stream's own named `MemoryConsumer("training_set_stream[rank r]")`, never the
  aggregate pool high-water mark), `carry` ≤ B−1 rows pending, `Σ E` = the K3 scaler's
  whole-prefix pass and the mining/GradCache whole-set loaders (named exemptions,
  pool-accounted but not bounded by this inequality).
- P4 (liveness over every held chunk). For every `prefetch ≥ 1` and a consumer holding
  chunk t across the ask for t+1, the stream completes and serves exactly `window.len()`
  rows; no lease-return dependency.
- P5 (per-rank slicing). For W = 2, rank 0's and rank 1's streams are independent objects;
  for every step t, rank r's chunk equals the eager `text_chunk_for_rank(spec_r, t)`
  exactly; the two row sets are disjoint and their union is the window.
- P6 (parity). (i) A W=1 stream's chunks concatenate to `read_back_sql`'s collected rows in
  order, at `target_partitions ∈ {1, 4}`. (ii) The pinned per-platform adapter prints
  (`refactor_parity`, `regression_refactor_parity`,
  `gradcache_completes_at_w1_with_a_pinned_adapter_digest`) are unchanged with the
  production path now streaming.
- P7 (every refusal is behavioural). Each refusal (`prefetch = 0`, the early-end window
  overrun, a format with no row-level partition refused at `open`, pool exhaustion) has a
  test reaching it through the public path; deleting the refusal's branch turns that test
  red.

**Oracles, by name.**
- `crates/jammi-ai/tests/it/training_set_stream.rs::p6i_w1_stream_concatenation_matches_read_back_sql_at_various_partition_counts` (P6.i).
- `crates/jammi-ai/tests/it/training_set_stream.rs::p4_liveness_over_every_held_chunk_at_every_accepted_prefetch` (P4; fixture 70,000 rows,
  B = 13, prefetch ∈ {1,2,3,4}, 60s wall-clock timeout — the regression pin for the excised
  arm's `prefetch = 2` deadlock).
- `crates/jammi-ai/tests/it/training_set_stream.rs::p5_two_rank_world_slices_match_eager_text_chunk_for_rank_exactly` (P5).
- `crates/jammi-ai/tests/it/training_set_stream.rs::p7_early_end_when_window_exceeds_table_rows`,
  `crates/jammi-ai/tests/it/training_set_stream.rs::p7_classification_without_a_vocabulary_is_refused_at_open` (P7).
- `crates/jammi-ai/tests/it/training_set_stream.rs::f3_classification_streams_given_a_vocabulary_and_matches_the_eager_class_indices`,
  `crates/jammi-ai/tests/it/training_set_stream.rs::f3_a_label_seen_only_in_the_validation_suffix_is_still_counted` (F3: classification's
  whole-table vocabulary is NOT an exemption — reversed from c3's first cut).

## The production binding (c3b/§11 folds) — `TrainingSource`, `whole_set_arm`, F1–F6

**Mechanism.** `crates/jammi-ai/src/fine_tune/source.rs::TrainingSource` is
`{Resident(TrainingDataLoader), Streamed(Box<StreamedSet>)}`;
`crates/jammi-ai/src/fine_tune/source.rs::whole_set_arm` is the ONE predicate both
`crates/jammi-ai/src/fine_tune/worker.rs::run_spec`'s source selection and
`crates/jammi-ai/src/fine_tune/trainer.rs::f6_streamed_refusal_oracle`'s refusal (the
trainer's own dispatch) call, so the two decisions can never come apart. A `Streamed`
source's construction calls only `crates/jammi-ai/src/fine_tune/training_set.rs::materialize_projection_table` — never `read_back`/`read_back_with_reservation` — so no
`Vec<RecordBatch>` over the training set is ever collected (F1). A `Resident` loader's
construction reads back through `crates/jammi-ai/src/fine_tune/training_set.rs::read_back_with_reservation` and attaches the live `MemoryReservation` via
`crates/jammi-ai/src/fine_tune/data.rs::TrainingDataLoader::with_reservation` — held for
the loader's lifetime (moved into whichever half of a later
`crates/jammi-ai/src/fine_tune/data.rs::TrainingDataLoader::split` carries it), not
checked-then-released (P-R). The K3 scaler's vector and the classification vocabulary are
SEPARATE, bounded, whole-table passes through the same decoder (F2/F3), never the per-step
chunk stream; every `EpochSource::Stream` consumer terminates on the first zero-row chunk
(F4); the refusal domain is pre-passed once over the whole table before step 0 (F5); the
worker's source selection and the trainer's arm dispatch share the ONE `whole_set_arm`
predicate (F6).

**Properties.**
- P-M (writer residency, c3c). The training-set write plans at ONE output partition, never
  a `SortPreservingMergeExec`: `crates/jammi-db/src/store/mod.rs::ResultStore::plan_training_set_rows` derives its sort through `crates/jammi-db/src/session.rs::single_partition_context` — the SAME derivation `TrainingSetStream::open`
  uses for its own reads. The deployment rule (stated where the write is planned,
  `crates/jammi-db/src/store/mod.rs::ResultStore::materialize_training_set`'s doc): a
  single `engine.batch_size` batch larger than `engine.memory_limit` cannot be sorted — the
  arithmetic an F1 fixture states directly: 800 rows × ~100 KB/row ≈ 80 MiB under a 64 MiB
  pool needs `batch_size = 32` (≈ 3.1 MB/batch) rather than the `8192` default (≈ 800 MB/batch,
  far larger than the pool regardless of partition count).
- P-F1 (the unit's own property, end to end). A full regression job at W=1 over a table
  whose eager collected size exceeds the pool COMPLETES (`source_kind_for == "streamed"`),
  and the eager collect of the SAME table under the SAME pool refuses, naming
  `training_set_eager`.
- P-R (the Resident loader owns its bytes). `TrainingSource::Resident` carries the eager
  `MemoryReservation` for the loader's lifetime; the pool's `reserved()` is ≥ the collected
  size while a Resident job trains and returns to baseline after.
- P-P (one prefetch depth). `PRODUCTION_PREFETCH_DEPTH` is one named constant with its
  reason, never a literal at the call site.

**Oracles, by name.**
- `crates/jammi-db/tests/it/materialization.rs::the_writers_single_partition_derivation_plans_one_sort_and_no_merge` (P-M(i), at
  `target_partitions ∈ {1, 4}`, FILE-backed — see below for why never `MemTable`-backed).
- `crates/jammi-ai/tests/it/training_set_stream.rs::f1_a_table_whose_eager_read_exceeds_the_pool_trains_to_completion_through_the_stream`
  (P-M(ii)/P-F1, the executed 80 MiB-under-64 MiB numbers).
- `crates/jammi-ai/tests/it/training_set_stream.rs::p_r_a_resident_loader_holds_its_eager_reservation_while_training_runs` (P-R).
- `crates/jammi-ai/tests/it/training_set_stream.rs::f4_a_validation_window_not_a_multiple_of_batch_size_completes` (F4).
- `crates/jammi-ai/tests/it/training_set_stream.rs::f5_a_nan_target_in_the_validation_suffix_refuses_before_step_zero_under_train_loss` (F5).
- `crates/jammi-ai/src/fine_tune/trainer.rs::f6_streamed_refusal_oracle::a_streamed_source_under_a_mining_eligible_config_is_refused` (F6, the trainer-side half;
  the worker-side half — "mining/GradCache selects `Resident`" — is observed end to end by
  `crates/jammi-ai/tests/it/training_set.rs::gradcache_completes_at_w1_with_a_pinned_adapter_digest`'s `source_kind_for` assertion).
- `crates/jammi-ai/src/fine_tune/source.rs::tests::split_index_matches_the_resident_split_boundary` (the split-boundary equality every
  `total ∈ 0..=1000` and every admitted fraction).

**The writer's source universe (addendum, this contract).** P-M holds for the source
universe production actually registers a training-set write over: a `ListingTable`
(a registered CSV/Parquet source, or a pinned result table's own provider — its
file-group count follows `target_partitions`), a Postgres/MySQL federated source
(`crates/jammi-db/src/source/postgres.rs`, `crates/jammi-db/src/source/mysql.rs`, planned
through `datafusion_federation::FederationOptimizerRule` — one partition by construction),
and the mutable provider's own `crates/jammi-db/src/store/mutable/provider.rs::MutableTableProvider::scan`'s `MemTable::try_new` (always `vec![vec![batch]]` — one
partition; `MemTable::try_new` has exactly one call site under `crates/*/src` in the
workspace). The edge this excludes, found by the lead's probe
(`scratchpad/logs/probe-memtable.log`, session `12f161bf`): a hand-built MULTI-partition
`MemTable` under `single_partition_context` plans a `SortPreservingMergeExec` over N
per-partition `SortExec`s that still collapses to ONE output partition — the writer's own
`plan_training_set_rows`'s `partition_count() != 1` guard cannot see that shape, because a
`MemTable`'s partition count is fixed at construction and never collapses just because
`target_partitions` changed (unlike a `ListingTable`'s file groups). No production source
registers one, so the edge is unreachable in practice, not mechanically closed by the
guard alone — this is why `the_writers_single_partition_derivation_plans_one_sort_and_no_merge`
is FILE-backed (see its own doc comment,
`crates/jammi-db/tests/it/materialization.rs::the_writers_single_partition_derivation_plans_one_sort_and_no_merge`) rather than
`MemTable`-backed: a `MemTable` fixture would not stand in for production's actual shape.

## The streamed source's tenant scope across `block_on` (c3d) — P-T1–P-T3

**Mechanism.** `crates/jammi-ai/src/fine_tune/source.rs::StreamedSet::tenant` captures the job's
tenant (`crates/jammi-ai/src/session.rs::InferenceSession::tenant`) inside
`crates/jammi-ai/src/fine_tune/worker.rs::run_spec`, while that task is still executing inside
the caller's `crates/jammi-ai/src/session.rs::InferenceSession::with_tenant_scoped` task-local
scope (`crates/jammi-ai/src/fine_tune/worker.rs::run_claimed_job_under` installs it for the
whole job). `crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::open_streamed_source` drives
`crates/jammi-ai/src/fine_tune/stream.rs::TrainingSetStream::open` through
`tokio::runtime::Handle::block_on` from the `spawn_blocking` pool — a FRESH top-level poll on a
different OS thread that does NOT inherit the async task's task-local
(`crates/jammi-db/src/tenant_scope.rs::TenantBinding::current` reads the override installed on
the CURRENT task only, falling back to the session's sticky shared value otherwise). Unscoped,
every query `open` issues — the schema/null-NaN pre-pass, the ordered `read_back_sql` plan, the
pump's own planning — resolves a tenant-owned training-set table as `Unscoped`-invisible: "table
… not found", exactly like a peer's private table. `open_streamed_source` therefore re-enters
`with_tenant_scoped` INSIDE its own `block_on`'d future (a `match streamed.tenant` over `Some`
scoping the `open` future through `with_tenant_scoped` and `None` awaiting it directly), covering
every nested `.await` `open` makes — never assumed inherited.

**Properties.**
- P-T1 (capture point). `StreamedSet::tenant` is captured (`InferenceSession::tenant`) while the
  worker's `run_spec` task is still executing inside the caller's `with_tenant_scoped` scope —
  never read later, never re-derived from the session's sticky binding; `None` only for a
  genuinely unscoped run (the queue-drain worker's own claim, or a test session that never bound
  a tenant).
- P-T2 (every query the stream issues observes the job's tenant). For a `Some(tenant)`
  `StreamedSet`, every query `open_streamed_source`'s `block_on`'d future issues resolves against
  `tenant`, never `Unscoped`: the re-entry happens INSIDE the `block_on` future, so it covers
  every nested `.await` `open` makes, including ones a later edit adds inside `open`'s body —
  never a property that depends on enumerating `open`'s call graph by hand.
- P-T3 (the sticky binding does not stand in for a scope — testing rule). A test that only sets
  the session's STICKY binding (`bind_tenant`/`with_tenant`) cannot exercise P-T2:
  `TenantBinding::current_tenant` falls back to the sticky value whenever no task-local override
  is installed on the CURRENT task — including the `spawn_blocking` thread `block_on` runs on —
  so a sticky-bound session's blocking-thread call would "accidentally" resolve the right tenant
  even without this fix. Every tenant oracle for this mechanism scopes via `with_tenant_scoped`
  (task-local) on BOTH the submit and the wait, matching production's per-request scoping (the
  gRPC layer's per-request interceptor; `run_claimed_job_under`'s own doc on why the claim itself
  stays unscoped) exactly — never the sticky binding.

**Oracles, by name.**
- `crates/jammi-ai/tests/it/training_set_stream.rs::p_t2_a_tenant_scoped_job_trains_through_the_stream_over_exactly_its_own_rows`
  (P-T1/P-T2: a job submitted AND awaited inside one `with_tenant_scoped` block, whose source is
  registered under that SAME bound tenant AFTER the bind — so its training-set table is
  owner-gated, not GLOBAL). RED at `7f493d87` (before this commit), quoted verbatim from the
  oracle's own doc comment: `job.wait()` returns `Err(FineTune("DataFusion error: Error during planning: table 'datafusion.public.jammi.training__text_embedding__training-set__…' not found"))`.
  GREEN after `b27828be`. The oracle also asserts
  `crates/jammi-ai/src/fine_tune/worker.rs::training_test_hooks::streamed_total_rows_for` equals
  the tenant's own row count exactly — ruling out an unscoped fallback that happened to resolve
  SOME table, not just an outright not-found.
- `crates/jammi-server/tests/it/grpc_job.rs::training_under_a_tenant_scope_succeeds_over_the_wire`
  (the pre-existing server-level regression test this unit's oracle reproduces at the jammi-ai
  level; a full-workspace run RED before this commit — panic: "assertion left == right failed:
  tenant-scoped training should complete, got 'failed' / left: \"failed\" / right: \"completed\""
  — GREEN after, 240 passed/0 failed in the same suite).

**Why `crates/jammi-ai/tests/it/fine_tune.rs::worker_run_span_carries_job_and_tenant` never
caught this.** That test binds its tenant AFTER its source is already added, so that source's
result tables are GLOBAL (`owner = None`, visible unscoped) — it passes regardless of the bug.
`p_t2_…` binds the tenant FIRST, so its training-set table is genuinely owner-gated; this
ordering, not the tenant-scoping mechanism itself, is what makes the property falsifiable.

## The bench call site (c4b addendum) — `jammi-bench`'s fine-tune bench trains like production

**Mechanism, by construct.** `crates/jammi-bench/src/finetune_run.rs::run_impl` no longer
hoists one `train_loader` reused by reference across every epoch (`TrainingLoop::run` — see
`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::run` — took an owned `TrainingSource`,
not `&TrainingDataLoader`, once this unit's binding landed): `run_impl`'s own doc comment on the
call site states the reasoning this contract restates rather than duplicates. Per epoch,
`run_impl` rebuilds `train_loader` fresh from `train_rows` (the SAME borrowed fixture rows the
loader was always built from, never mutated by a prior epoch's run) and passes
`TrainingSource::Resident(train_loader)` — reproducing byte-identical content and order every
epoch, exactly what the old borrow-and-reuse shape obtained.

**What this call site measures, and what it does NOT.** This tier's `train_loader` is built by
`TrainingDataLoader::from_triplets`/`from_pairs`/`from_media_triplets` over in-memory fixture
rows — `run_impl` drives `TrainingLoopBuilder` directly off a committed corpus file, never
through an `InferenceSession`/materialised table
(`crates/jammi-ai/src/fine_tune/training_set.rs::read_back_with_reservation`/
`materialize_projection_table` have no caller on this path). It therefore carries no
`MemoryReservation` and is correctly passed as `TrainingSource::Resident` with
`crates/jammi-ai/src/fine_tune/data.rs::TrainingDataLoader::with_reservation` never called — the
same "reservation-free by construction" state every non-worker caller of `from_triplets` et al.
already documents. This tier's numbers (wall time, dispatch counters, loss/held-out
trajectories) are measured OUTSIDE the loader-residency pool this unit adds: they report
training math identically to the eager path DataFusion pool accounting never touches, never the
pool's own bound (P2/P3/P-F1), which `crates/jammi-ai/tests/it/training_set_stream.rs`'s own
oracles exercise instead. This call site never exercises M2/M3/the production binding's
`Streamed` arm — it is a P1-adjacent smoke path (does `TrainingLoop::run` still accept and run a
`Resident` source correctly), not a residency-bound oracle.

**Citation ripple.** `crates/jammi-encoders/src/test_support.rs`'s own doc comment cites two
line numbers inside `finetune_run.rs` (`lora_linear_fused_dispatch_before`/
`lora_linear_fused_dispatch_after`, the process-wide fused-dispatch counter reads) that moved
when `run_impl`'s doc comment grew; both were corrected in the same commit (`7f493d87`) that
made the call-site change — this contract records the ripple existed and was closed in one
commit, not a separately-tracked stale citation.

## History — the c3b "F1 uncovered" episode (HISTORY, not a live finding)

c3 shipped `decode.rs` + `stream.rs` and every ai-side oracle above, but left
`TrainingLoop::run` bound to the eager `TrainingDataLoader` — production did not yet train
through the stream, so P6.ii was vacuous. c3b built the `TrainingSource`/`whole_set_arm`
binding (this section's own mechanism) and attempted the P-F1 oracle (a table whose eager
read exceeds the pool trains to completion through the stream); six recorded attempts
tuned the STREAM's own prefetch/window/consumer shape and still could not make the oracle
pass, and were recorded as `uncovered`. The lead's probe (private clone at `a9083ca6`)
found the actual cause was never the stream: at one partition the stream's own ordered
read-back already planned as a bare scan with no `SortExec`/`SortPreservingMergeExec`, and
P3 already completed on an 80 MiB table under a 64 MiB pool through the stream primitive.
The full job failed inside a `SortPreservingMergeExec` needing a few MB with the pool
already nearly full — that merge belonged to the training-set WRITER
(`ResultStore::materialize_training_set`'s explicit `DataFrame::sort` at the session's own
`target_partitions`, then a real merge over N partition-local sorted runs that filled the
pool before the merge itself could reserve its few MB), not the per-rank reader. c3c (this
unit's writer fold, §12 of the design contract) fixed the writer to plan its sort at ONE
output partition (`single_partition_context`, the same derivation the stream already used
for reads) — eliminating the merge entirely — which is what the P-M/P-F1 properties above
state and what `f1_a_table_whose_eager_read_exceeds_the_pool_trains_to_completion_through_the_stream`
now passes. The stream.rs module-doc's earlier claim of a structural bound the writer did
not yet honor, and the corresponding uncovered block, are DELETED (not refined) by that
fix — nothing in the current tree references either.

**The workspace-run discovery (c4b).** After c3c, a `cargo test --workspace` run (never `-p jammi-ai -p jammi-db` alone, this unit's own crate-scoped gates) surfaced two further defects neither crate-scoped suite could have seen. (i) `crates/jammi-bench/src/finetune_run.rs::run_impl`'s bench binary failed to compile: a mismatched-types error at its `training_loop.run(&train_loader)` call site against the new `crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::run(&mut self, source: TrainingSource)` signature — because `jammi-bench` is a separate crate outside `-p jammi-ai -p jammi-db`'s build graph entirely, so neither package-scoped `cargo test` ever compiles it; fixed by `7f493d87` (the bench call-site addendum above). (ii) `crates/jammi-server/tests/it/grpc_job.rs::training_under_a_tenant_scope_succeeds_over_the_wire`, a pre-existing tenant-isolation regression test, failed its completion assertion — because that server-level test is the first caller in the tree to submit AND await a job through a genuinely task-local tenant scope over a `Streamed` source; every `jammi-ai`-level tenant oracle at the time used the session's STICKY binding, which the P-T3 testing rule above shows masks exactly this bug class. Fixed by `b27828be` (c3d, the P-T1–P-T3 section above). Neither defect was reachable from `-p jammi-ai`/`-p jammi-db` in isolation: the first is a foreign crate's compile error, the second needed a cross-crate (`jammi-server`) caller that scopes both halves of the tenant boundary the way production's gRPC layer does — which is exactly why the merge-path gate set runs the full workspace, not each touched crate alone.

## The closing round (2026-09-15) — the audit's two blocks and four advisories, lead-applied

The closing adversarial audit (round 1) blocked on two items and the closing citation check on a stale
pre-amend sha in a test's doc comment (fixed by construct in that comment). Discipline PASSED. The user then
took the unit over from the swarm; the fixes below were applied by the lead and the closers were NOT re-run.

- **The bench timer spans `run()` only.** `crates/jammi-bench/src/finetune_run.rs::run_impl` had moved the
  per-epoch loader build (`RowSet::loader`, a clone of the whole fixture corpus, media bytes included) inside
  the `train_run_wall_s` span, whose contract in `crates/jammi-bench/src/report.rs` excludes loader construction.
  The build is hoisted above the timer. Oracle:
  `crates/jammi-bench/src/finetune_run.rs::train_run_wall_s_excludes_the_loader_build` — a test-only,
  thread-local sleep hook in `RowSet::loader` makes the build's cost large and deterministic and asserts it is
  outside the measured span.
- **P1 pins the shipped derivation.**
  `crates/jammi-ai/tests/it/training_set_stream.rs::p1_the_loader_derived_state_plans_with_no_sort_and_no_merge`
  called a hand-rolled `SessionStateBuilder::new_from_existing(..).with_config(..)` — the construction c3c
  replaced because it replaces the caller's default catalog. It now calls
  `crates/jammi-db/src/session.rs::single_partition_context`, the function the loader ships with.
- **An unreadable pre-pass aggregate is a typed refusal.** `crates/jammi-ai/src/fine_tune/stream.rs::validate_window`
  read both aggregate outputs through `unwrap_or(0.0)`, so a window whose subquery yielded no rows (a `row_count`
  overstating the table → SQL `NULL`) silently passed the whole-table null/NaN refusal. It refuses now, naming
  the aggregate and the window. Oracle:
  `crates/jammi-ai/tests/it/training_set_stream.rs::p_b3_a_window_whose_aggregate_subquery_matches_no_rows_refuses_at_open`.
- **`split` consumes.** `crates/jammi-ai/src/fine_tune/data.rs::TrainingDataLoader::split` takes `self`, so a
  second split of a reservation-carrying loader — which would hand back zero bytes with no error — is
  unrepresentable.
- **An unreadable sidecar is non-fatal.** `crates/jammi-db/src/store/mod.rs::training_set_registration_sort_order`
  treats an unreadable manifest like an absent one (warn naming the table and the error; registration succeeds;
  `Ok(None)`; the explicit `ORDER BY` still sorts). Oracle:
  `crates/jammi-db/tests/it/materialization.rs::registration_warns_when_a_training_sets_sidecar_is_unreadable`.
  The one sidecar read per training-set row at session start is stated as a cost in that function's doc.
- **The guide's writer source universe states the truth.** A versioned result table's provider is a
  `UnionExec` over one `ListingTable` per fragment (`crates/jammi-db/src/store/masked_provider.rs`), N partitions
  regardless of `target_partitions`, which the writer's `partition_count` guard cannot see; no training-set
  source is one today (every `source_sql` names the source schema's table; pinned providers are read through
  `read_table` only), and the guide says so instead of claiming the provider follows `target_partitions`.

Lead verification: `cargo fmt`, the agent's `cargo clippy --workspace`, `cargo test -p jammi-bench/-db/-ai` (green),
and a `cargo test --workspace` in which only `jobs_cancel::a_claimed_training_jobs_cancel_request_is_honoured_at_the_next_epoch_boundary`
failed (a poll-cadence timing assertion seen only under full-suite load, passing in isolation twice before this
round); the unit's suites run again on the consolidated wave-3 branch before its single PR.

## The consolidated PR's hermetic lane (2026-09-15) — P4's test defect, and the product defect under it

**What CI found.** PR #579's "Test (hermetic)" job failed
`training_set_stream.rs::p4_liveness_over_every_held_chunk_at_every_accepted_prefetch`
at both consolidated tips it ran (`0672f9ca`, `2ec9f638`): `prefetch=1:
deadlocked past the 120s timeout`. Locally the test passed on every run but
routinely tripped `cargo test`'s "running for over 60 seconds" notice.

**The test defect.** The consumer loop dropped chunk `t` BEFORE asking for
`t+1` (`drop(chunk)` at the end of the body, the ask at the top of the next
iteration), so it never held a chunk across the ask — the exact condition P4
states. Its wall clock was a 5 ms sleep on every one of the 5,384 chunks per
prefetch value (70,000 rows / 13): 27 s of sleep per prefetch value, four
values, plus the runner's scheduling, never the property. The rewrite holds
structurally (the ask is issued while the previous chunk is alive; the
previous chunk is dropped only after the ask returns) and slows the consumer
only where a full buffer matters: the first `2 × MAX_PREFETCH` chunks and
`MAX_PREFETCH` chunks either side of the 65,536-row group boundary.

**The product defect the test defect hid.** With the per-chunk sleep gone the
test still took 171.87 s (four passes over 70,000 rows, ~43 s a pass). Cause:
`decode::append_selected_rows` read each column through
`extract_string_column`, which materialises the WHOLE column as owned
`String`s, and the pump called it once per ROW (`&[row_in_batch]`) — every
13-row chunk copied every string of an 8,192-row batch, twice. The function's
own doc ("applied to the whole batch once … cloned ONLY for `indices`")
described the shape the code did not have.

**The fix (this commit).** `decode.rs` gains typed cell views —
`StringCells` (`Utf8View`/`Utf8`/`LargeUtf8`, and an owned array only for the
`cast` fallback) and `BinaryCells` — behind `string_cells`/`binary_cells`,
which carry the extractors' type policy and both refusals; `extract_string_
column`/`extract_binary_column` are DEFINED as the cells turned into a `Vec`,
so there is one policy. `DecodedBatch` decodes a batch ONCE (the text/media
views plus the whole-column numeric reads whose null/NaN policy must scan
every slot anyway) and `append(indices, vocab, acc)` clones exactly the
indexed cells, in the caller's order; the pump builds it lazily per batch
(a batch this rank selects nothing from is never decoded), collects each
step's indices, and appends at every range end and at the batch end when a
range continues into the next batch. The one-shot `append_selected_rows`
wrapper is deleted, not kept as a stale entry point.

**Executed.** `cargo test -p jammi-ai --test it training_set_stream`: 16
passed in 19.91 s with P4 among them (P5/P6 parity and the exact-count
assertions are the oracles that a flush placed wrongly — a range straddling a
batch boundary appended only at range end, or only at batch end — would
fail). `cargo test -p jammi-ai --lib`: 749 passed, three new unit tests
(`decode::decoded_batch_tests`: cells read by index in the caller's order
with a NULL slot keeping the text path's `""` contract; `extract_string_
column == cells.to_vec()` on every family including the cast fallback; a
binary column refused as text by the view too). `cargo clippy -p jammi-ai -p
jammi-bench --all-targets -- -D warnings` and `RUSTDOCFLAGS="-D warnings"
cargo doc --no-deps -p jammi-ai -p jammi-db` clean. What the unit tests
exclude: the `Utf8View` family and the pump's flush placement — both
exercised by the `it` suite above (DataFusion returns Parquet text as
`Utf8View`; the 70,000-row fixture spans batch boundaries at every prefetch).
The phase-5 oracle re-ran at this tip (a mechanism change after the
recorded PASS makes that record stale by the gate's own rule) and PASSED,
with both pump flushes mutation-tested (deleting the batch-end flush fails
P4/P5/P6.i; deleting the range-end flush fails nine oracles). Its one
non-blocking observation is folded here: `DecodedBatch`'s doc claimed every
refusal message equals the eager path's; the eager Pairs/Triplet arms carry
an image/audio-triplet operator hint the stream's per-column message never
had (unchanged from the per-row entry point this fold replaced). The doc
now states the divergence as one of wording, not of acceptance; sharing one
message constructor per column between both entry points is the fold that
would close it and is left named, not done, in this unit.

## Gates run at contract time

**At `4b03d5a9` (the contract's first write, c4).** `cargo fmt --all --check` (0), `cargo clippy -p jammi-db --all-targets -- -D warnings` (0), `cargo test -p jammi-db` (521 lib + 480 it + 3 doc, 0 failed, 1 pre-existing ignored), `python3 ci/scripts/perf/check_citations.py` (0, 1024 files scanned). `cargo clippy -p jammi-ai`/`cargo test -p jammi-ai` are c1–c3c's own gates (unit branch, prior commits).

**At `b27828be` (c4b, this write).** A `cargo test --workspace` run over `7f493d87`/`b27828be`
(exit `0`) is the gate that actually caught both defects the "workspace-run discovery"
paragraph above narrates and confirms them fixed: `jammi_ai` lib 748 passed/0 failed,
`jammi_ai` it 568 passed/0 failed/1 ignored (includes `p_t2_…`), `jammi_bench` unittests 202
passed/0 failed (the bench compiles and runs again), `jammi_db` lib 521 passed/0 failed,
`jammi_db` it 495 passed/0 failed/1 ignored, `jammi_server` it (`grpc_job`/`grpc_remote_compute`
etc.) 240 passed/0 failed — including `training_under_a_tenant_scope_succeeds_over_the_wire`.
This contract's own docs-only commit re-runs `python3 ci/scripts/perf/check_citations.py`,
`python3 ci/scripts/check_doc_parity.py`, `python3 ci/scripts/check_no_consumer_names.py`, and
`python3 ci/scripts/check_rigor_record.py`, plus this contract file's own construct-citation
self-check (every backtick-quoted `path.rs::construct` re-resolved against `b27828be`'s tree;
zero bare `path:line` tokens) — real exit codes reported in this commit's own message body.

## Filed

#551 (`RelationKey`/`parquet_path` reader; stays filed, no new route added — the stream's
allow-list entry delegates to `read_back_sql`, the one site spelling `sql_relation()`).
U4b (the rank body binding to `Slice::PerRank` at W>1). Mining/GradCache streaming
(exemptions, stated not built).
