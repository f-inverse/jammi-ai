# Distributed training on DataFusion — design (#500)

Multi-GPU and multi-node training as engine mechanism: a training job declares a world size, N
ranks train one job cooperatively, gang failure semantics are explicit, identity and parity are
scoped, and placement stays the deployer's runtime. This document states the positions taken, the
mechanisms, the invariants they keep, and the alternatives rejected.

Citations name a symbol rather than a line: a Rust item as `crates/<crate>/src/<path>.rs::Item`
(`Type::method` for a nested item), a manifest key as `Cargo.toml::[dependencies].<dep>`, a wire
message as `<file>.proto::Message`, a doc section as `<path>.md#heading`.

## 1. What MLlib did to Spark, and what that means here

The MLlib paper's win over Mahout was building on Spark core's data abstraction: RDD caching
made iteration cheap. MLlib's optimizers ran gradient steps as `map` + `treeAggregate`; later
measurement (Sparker, 2021) found `treeAggregate` taking ~67% of end-to-end time with the
driver as the bottleneck. Databricks' fix was Project Hydrogen's barrier execution mode
(SPARK-24374): all-or-nothing scheduling, a barrier call, task host info and stage-wide restart.
TorchDistributor uses it only to launch ranks; NCCL does every collective.

The data engine's durable contributions to training are therefore the **data plane** and
**gang scheduling**, never the collective. Jammi is to DataFusion what MLlib was to Spark core;
the gang is to Ballista what Project Hydrogen was to the Spark scheduler. Ballista 54.x has no
barrier mode, retries failed tasks individually (harmful for a gang), and lists executor-failure
handling on the adaptive path as a gap; `datafusion-distributed` has no scheduler and fails the
whole query on node loss (a gang's failure model). Neither is a hard dependency of training; the
Ballista extension (§9) proves the operators run unchanged under a real distributor.

The position, in four statements:

1. **The training set is a plan and a producer** — a content-addressed result table; sharding is
   partitioning of its train prefix by a fixed rule; epochs re-read it with bounded memory (§2).
2. **A trained model is a producer** — its descriptor folds the training-set digest, the
   base-model identity, the canonical spec, the seed and the topology; replay is retrain (§3).
3. **The gang is a co-scheduled unit, not a per-step query** — every rank gathers the global
   batch's representations and computes the identical global loss; adapter gradients are summed;
   collectives run in-binary behind one trait; any rank failure fails the attempt; the jobs
   fleet's lease, reclaim and resume checkpoint give restart (§4).
4. **Not** the SGD step or gradient exchange as DataFusion operators or aggregates. That is the
   `treeAggregate` shape whose cost the Sparker measurement records.

**Workload and payload.** LoRA adapters sit inside the tower, so an encoder-adapters run has no
frozen forward; each rank runs the full tower forward/backward on its slice. What crosses the
wire per optimizer step is (a) the gathered representations of the global batch — `W·B·d`
floats per representation column (anchor, positive, optional negative) plus the scores — and
(b) the adapter gradients (rank-8 matrices). At W=4, B=32, d=768, f32: about 0.4 MB per
column and single-digit MB for the adapter gradients. Both are small enough that the exact CPU
collective is viable far beyond a test twin; NCCL matters when the tensors already live on
GPUs. The **projection-head** target (`TrainingTarget::ProjectionHead`,
`crates/jammi-ai/src/fine_tune/target.rs`) runs the same per-batch pattern (§5).

## 2. The training set is a producer

`ProducingDescriptor::TrainingSet { source, columns, task, format, order_rule: "full_tuple_v1" }`
(`crates/jammi-db/src/store/manifest.rs::ProducingDescriptor::TrainingSet`). Materialization runs
the source plan through the session, sorts by the full projected tuple (identical tuples are
identical rows, so their mutual order is immaterial), and writes an immutable Parquet result
table of kind `TrainingSet` (`crates/jammi-db/src/catalog/result_repo.rs::ResultTableKind::TrainingSet`)
with the standard attestation (`crates/jammi-db/src/store/manifest.rs::MaterializationManifest`).

The descriptor carries **no topology and no split**: world size, batch and validation fraction do
not enter the table's identity. Reuse across jobs additionally requires equal, *pinned* input
anchors — the engine's existing rule
(`crates/jammi-db/src/store/manifest.rs::AnchorKind::UnpinnedAtInstant`): a plain registered
source exposes no version or digest surface, anchors as `UnpinnedAtInstant`, and the cache probe
never matches an unpinned anchor. Two jobs over the same plain source therefore each materialize
their own table; reuse needs a pinned source. The training set is materialized with
`job_attempt: None`: it is a shared producer output, never the claiming attempt's
`jobs.partial_result`, so the attempt-scoped partial-result adoption rules
(`crates/jammi-ai/src/jobs.rs::PartialResultDisposition`) never apply to it. Table names are
unique per materialization, so a crashed coordinator's `building` row is never met by name by
its successor; it is reclaimed by lease expiry like any other abandoned building table
(`crates/jammi-db/src/catalog/result_repo.rs::Catalog::claim_expired_building_table`).

`GraphFineTune` goes through the same funnel over a different input seam, because a graph run's
rows are the output of a biased walk, not of a query the engine can express as durable SQL. The
worker re-reads the node/edge sources in a committed order (`GRAPH_READ_ORDER_RULE_V1`), samples
them (`crates/jammi-ai/src/fine_tune/graph_sampler.rs::GraphSampler::sample`), and hands the
sampled pairs to the same materialization as a `Batches` (`RecordBatch`-stream) input with a
leading `_ordinal` column, under `ProducingDescriptor::GraphTrainingSet`. The written table is
still `ResultTableKind::TrainingSet`. The node/edge sources anchor `UnpinnedAtInstant`, exactly
like the tabular arm's plain-source case. Media blob columns are stored as they are for every
other result table.

**Row-group alignment is not part of the descriptor.** Aligning row groups to `W·B` would put
topology into the table's identity and break sharing across world sizes; the reader slices
instead.

**Split.** The job's `validation_fraction` defines the train prefix
(`crates/jammi-ai/src/fine_tune/data.rs::TrainingDataLoader::split`, the `TextRows` arm):
`val_count = round(rows × fraction)`, train rows `[0, rows − val_count)`, validation rows after.
The tests-only `Precomputed` loader arm
(`crates/jammi-ai/src/fine_tune/data.rs::TrainingDataLoader::from_precomputed`) hands tensors
straight to the trainer, splits by batch count, and stays outside the table path. Every step
quantity is a function of the **global** batch: `batches_per_epoch = ceil(train_count / (W·B))`;
`global_step`, the LR horizon (`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::run`)
and the trailing-window loss scale
(`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::process_batch_loss`) all index by global
batch, so W ranks take exactly the steps W=1 takes at batch W·B.

**Reading the committed order.** A Parquet scan gives no row-order guarantee: the table is written
with 65,536-row row groups and the session plans at `[engine] execution_threads` partitions, so a
table larger than one row group comes back interleaved unless the reader asks for the order.
`crates/jammi-ai/src/fine_tune/training_set.rs::read_back_sql` is the one place that asks,
rendering `ORDER BY` from the table's own recorded order columns
(`crates/jammi-db/src/store/manifest.rs::ProducingDescriptor::training_set_order_columns` — the
projected tuple for a projection, `_ordinal` for a graph sample), never a caller-supplied key. A
reader-class allow-list in the same module enumerates every reader of the relation. The result
table's `ListingTable` declares that same order as its file sort order
(`crates/jammi-db/src/store/mod.rs::training_set_file_sort_order`, rendered from the single
source of truth, NULL placement included), so DataFusion elides the pipeline-breaking `SortExec`
the `ORDER BY` would otherwise plan. Without that declaration the whole table arrives on the
first poll and no stream built on top of it can be bounded.

**Two sources.** The trainer is handed a `TrainingSource`
(`crates/jammi-ai/src/fine_tune/source.rs::TrainingSource`):

- `Streamed` — a per-rank, residency-bounded stream over the table
  (`crates/jammi-ai/src/fine_tune/stream.rs::TrainingSetStream`). A background pump walks the
  DataFusion stream batch by batch, decodes each batch once, keeps only the rows the current
  step's chunk needs, and hands finished chunks to the trainer over a bounded channel. A fresh
  stream opens each epoch over the train window and a second one over the validation window. At
  world `W` every rank opens its own whole-prefix ordered stream and filters it with the per-rank
  partition predicate: row groups are 65,536 rows while the partition rule strides at `W·B`, so
  there is no row-group-level pushdown to exploit, and each rank scans the whole prefix and keeps
  only its own rows. This `W×` read amplification is the stated cost of per-rank independence; a
  stream *shared* across ranks (rank r observing rank r′'s rows) is the alternative it forbids.
  The residency bound is stated term by term in the module doc — one in-flight scan batch,
  `(prefetch + 1)` chunks accounted against the session's `[engine] memory_limit` memory pool
  through a named `MemoryConsumer`, the carry-over, and the named exemptions — so exceeding the
  bound is a typed error from the pool, never an assertion over the loader's own counters.
- `Resident` — the whole train prefix read eagerly through `read_back_sql`. The whole-set arms
  below, the `Precomputed` test path and a `GraphFineTune` run use it; these are the stated
  exemptions from the residency bound.

**Partition rule v1 ("block-by-global-batch")** over the train prefix
(`crates/jammi-ai/src/fine_tune/partition.rs::PartitionSpec::rows_for_step`): with per-rank batch
B and world W, global batch t is rows `[t·W·B, (t+1)·W·B)`; rank r reads
`[t·W·B + r·B, t·W·B + (r+1)·B)`. The union over ranks at step t is exactly the W=1 batch of
size W·B at step t. The trailing global batch is **kept** (drop-last would change W=1 bytes);
ranks then hold unequal counts, and when `train_count mod (W·B) ≤ r·B` rank r holds **zero**
rows: it encodes nothing and contributes a 0-row tensor to every gather of that step, with a zero
in the counts vector every rank derives from the rule. Sequence bucketing (`batch_bucket.rs`) runs
per rank batch.

**`TargetScaler`.** The scaler is fitted over the **train prefix only** — fitting over the whole
table would leak validation targets and change W=1 bytes. The target column of the train prefix
is collected, in committed order, into **one** `Vec<f32>` and
`crates/jammi-ai/src/fine_tune/regression_loss.rs::TargetScaler::from_targets` is called **once**:
it is a two-pass whole-tensor reduction and f32 summation is grouping-sensitive, so a chunked or
streamed accumulation would move the low bits. Every rank computes the scaler inside its own run
from the same committed prefix with the same reduction, so every rank holds identical μ/σ and no
value crosses the wire. The collected vector (4 bytes per row) is a named exemption from the
stream's residency bound. μ/σ persist in the resume bundle.

**Whole-set arms.** Hard-negative mining (the refresh-boundary check in
`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::run` rebuilds a mined loader from the
model at each refresh epoch) and GradCache
(`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::run_gradcache_epoch`, the whole train
prefix as one in-batch-negative batch) are structurally whole-set consumers. They run at W=1
only: `world_size > 1` with `hard_negatives.mine == true` or `cached == true` is a typed refusal
at submit (`crates/jammi-ai/src/fine_tune/spec.rs::RankAdmission::admit`). The predicate is `mine`
(`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::mining_eligible`), not
`refresh_every > 0`: `refresh_every` defaults to 1, so keying on it would refuse every multi-rank
job. The gather primitive (§4) is what would lift this restriction.

## 3. A trained model is a producer

`ProducingDescriptor::FineTune` folds:

| field | why it moves the bytes |
|---|---|
| training-set definition hash + artifact digest + row count | the data and its order |
| **canonical serialization of the whole `TrainingSpec` variant** — `FineTuneConfig` (`crates/jammi-wire/src/fine_tune.rs::FineTuneConfig`: LoRA rank/alpha/dropout, `use_rslora`, `rank_pattern`, `init_lora_weights`, lr, epochs, batch, `max_seq_length`, losses, `matryoshka_dims`, `quantile_levels`, `validation_fraction`, early stopping, `cached`, `hard_negatives`, …), `TrainingCommon`, method, task, seed | everything the trainer reads |
| base model identity (`ModelIdentity`: id, backend, precision, content digest, quantization) | the frozen weights |
| backbone dtype; fused-kernel admission profile | bits per op |
| `world_size`, per-rank batch, partition rule version, the collective the run reduced over (`noop` / `local` / `peer`), the ranks this host ran | the summation order and the batch layout |
| `MaterializationEnv` (engine version, device kind, model identities) | as for every producer |

A hand-enumerated table of "fields that matter" would be green and wrong the day a field is
added, so the descriptor takes the whole spec, and an **exhaustive destructuring** completeness
test (no `..`) lives beside the owning types in `jammi-ai` (and in `jammi-wire` for
`FineTuneConfig`): a new field fails compilation instead of escaping the hash.

**Crate layering.** `jammi-db` depends on no jammi crate but `jammi-numerics`
(`crates/jammi-db/Cargo.toml::[dependencies].jammi-numerics`), and every descriptor variant holds
primitives and db-local types (`crates/jammi-db/src/store/manifest.rs::ProducingDescriptor`).
`FineTuneConfig` is `jammi-wire`, `TrainingSpec`/`TrainingCommon` are `jammi-ai`
(`crates/jammi-ai/src/fine_tune/spec.rs::TrainingSpec`), `TrainingFormat` is `jammi-ai`
(`crates/jammi-ai/src/fine_tune/data.rs::TrainingFormat`). So both training descriptors carry an
**opaque, versioned canonical encoding** — `spec_canonical: String` (sorted-key canonical JSON)
with `spec_schema_version: u32` — produced by `jammi-ai` from the owning types, plus db-local
primitives (`ModelTask`, ids, digests, the topology fields). `TrainingSet.format` is likewise a
canonical string.

**The kernel admission profile.** A different fused-vs-eager admission outcome can change the bits
a run produces at the same nominal precision, so `MaterializationEnv` carries a canonical string
tag for it, built by the fine-tune worker from facts known before training runs (compiled build
features, the admission mode, the disabled-op set, the job's declared training dtype) and handed
across the crate boundary
(`crates/jammi-db/src/store/manifest.rs::MaterializationEnv::with_kernel_admission_profile`). One
residual is stated on the field: the device fold records a CUDA ordinal, never the GPU's compute
capability, so two runs on different-architecture GPUs can hash identically despite a
hardware-driven admission difference. Measured on candle 0.11: `CUBLAS_WORKSPACE_CONFIG` is a
kernel-selection input that must be *consistent* across ranks (`:4096:8` reproduces the unset
default, `:16:8` differs), so it is an input to reproducibility, not a setting the engine should
default.

**Naming and the catalog.** The catalog name `jammi:fine-tuned:{job_id}` stays the handle and the
re-claim idempotency key (`crates/jammi-ai/src/fine_tune/worker.rs::JobWorker::train_fine_tune`).
Migration `033_model_materialization` adds nullable `models.definition_hash` and
`models.input_anchors` (append-only; nullable because `ContextPredictor` has no materialization).
The definition probes (`crates/jammi-db/src/catalog/model_repo.rs::Catalog::find_models_by_definition`,
`Catalog::probe_model_by_definition`) never match NULL and restrict to `artifact_path IS NOT NULL`,
the servable set. There is no `models.manifest_path` column: the sidecar's path is always the
fixed name `materialization.json` under the model's artifact prefix. The manifest is written last
into the artifact prefix by the lease holder before the finalize CAS, and the two catalog columns
are written only after that CAS has committed this attempt's `artifact_path` — no unfinalized
row is ever a cache-hit candidate.

**Cache policy.** `cache` lives on `TrainingSpec::FineTune` itself, not `TrainingCommon`;
`TrainingSpec::GraphFineTune` has no `cache` field, so `lora_common_from_proto`
(`crates/jammi-ai/src/wire/training.rs::lora_common_from_proto`) refuses `cache = Use` for that
kind, typed, at decode — the one place that can still see both the kind and the requested value. A
stray `cache` key under `graph_fine_tune` in a persisted `jobs.spec` row is dropped at
deserialize, since the type has nowhere to put it; a hard error on unknown keys across the
persisted-row format is a separate reshape (<https://github.com/f-inverse/jammi-ai/issues/548>).
Model-level reuse by definition hash is not implemented: `CachePolicy::Use` on the column-source
`FineTune` kind is refused, typed, at submit, before anything durable is written. The intended
shape of reuse — a hit registers **its own name** pointing at the reused prefix (two rows, one
prefix, reported on the job's own result), probing by definition hash alone because the
training-set digest is already inside the hash — is recorded at
<https://github.com/f-inverse/jammi-ai/issues/562>. An unset `cache` field encodes identically on
both transports: the Rust client omits the field rather than assigning the enum's `Bypass`
discriminant, so an explicit `Bypass` and an unset field put the same bytes on the wire, matching
the embedded Python encoding.

**Deleting and reclaiming.** Deleting a model row is always allowed. The underlying bytes are
reclaimed only when no live `models` row, in any tenant, names the object's exact key or its
immediate containing directory as `artifact_path`.
`crates/jammi-db/src/store/reconcile.rs::ResultStore::prefix_is_referenced` is an admin-scoped
whole-catalog scan of `models.artifact_path` that
`crates/jammi-db/src/store/reconcile.rs::ResultStore::delete_unreferenced_prefix` consults before
every `models/`-prefix byte-delete — reconcile's own reap, the worker's abandon path, the
worker's epoch-checkpoint sweep, and the trainer's mid-run retention prune all reach it — refusing
typed (`StorageError::Referenced { prefix, count }`) while any row still references the prefix.
Reconcile's attribution set is built from the same admin-scoped scan, never the tenant-scoped
`list_models`, and a prefix it finds still referenced is reported with its count on
`ReconcileReport` (`crates/jammi-db/src/store/reconcile.rs`) and carried on the wire. Catalog
referential integrity between two rows naming the same prefix only becomes relevant once
model-level reuse exists (<https://github.com/f-inverse/jammi-ai/issues/547>).

The recompute replay arm (`crates/jammi-ai/src/pipeline/recompute.rs`) for `FineTune` is
**retrain**; for a training set it is re-materialize, through the same function a fresh run calls.

## 4. The gang

### Roles

A training-kind job is claimed by a `JobWorker` through `claim_next`
(`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::claim_next`, `FOR UPDATE SKIP LOCKED`,
`attempts + 1`). The claimant holds the only lease
(`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::heartbeat_job`, driven by the lease keeper)
and is rank 0. How the job's ranks are laid out is decided from the job's own `world_size` and
this host's `[worker] local_ranks`, and nothing else
(`crates/jammi-ai/src/fine_tune/worker.rs::TopologyDecision`):

- `world_size <= 1` → `Single`: the single-rank path over `Noop`.
- `1 < world_size <= local_ranks` → `Local`: every rank runs in this process, rank r pinned to
  `[gpu] devices[r]`.
- `world_size > local_ranks` → `Peer`: this process is the coordinator and rank 0; ranks
  `1..world_size` are fleet members it assembles and dials.

Every kind that trains from a training-set table runs under every topology: `fine_tune` and
`graph_fine_tune`. From the table on, the two kinds share one path: a reader asks the table's
descriptor for its committed order and the spec for what to decode
(`crates/jammi-ai/src/fine_tune/spec.rs::TrainingSpec::training_set_view`), never which producer
wrote the table. A `Peer` member binds the table by the identity on the job row and ends `Trained`
at the digest rank 0 published
(`crates/jammi-server/tests/it/gang_coordinator.rs::graph_fine_tune_runs_as_a_peer_gang`; the
in-process case is
`crates/jammi-ai/tests/it/gang_coordinator.rs::a_local_ranks_two_host_fans_a_two_rank_job_out_through_run_spec_and_publishes_the_gangs_bytes`).
`context_predictor` never runs on a gang: its spec variant has no `TrainingCommon`, so a
multi-rank predictor job is unrepresentable past the wire decode, which refuses it, typed — the
last edge that can still see a count a caller chose.

### The single-writer rule

The lease holder is the one writer of a job's row, of its durable checkpoints and of its published
artifact; every other rank of a gang writes nothing durable. The rule is stated as types
(`crates/jammi-ai/src/fine_tune/role.rs::LeaseHolder`,
`crates/jammi-ai/src/fine_tune/role.rs::RunnerRole`): a `LeaseHolder` is either the loop claimer
(the in-process path, including a `Local` gang's rank 0) or the coordinator (rank 0 of a `Peer`
gang); a `RunnerRole` is `Holder(LeaseHolder)` or `Rank { rank }`. Every job-row-writing site on
the run path takes a `LeaseHolder` as a required parameter, so a missed site is a compile error
and a `Rank` body, which holds no `LeaseHolder`, has nothing to pass — the write is unreachable by
type. The trainer's own durable writes (the resume and epoch checkpoints) are gated on the same
role inside the training loop, never inside the store, which stays role-agnostic. The gang has
one output artifact, written once by rank 0 at the finalize CAS; there is no per-rank fragment and
no peer-side write into the artifact prefix. The module doc of
`crates/jammi-ai/src/fine_tune/worker.rs` enumerates every writer site with the holder roles that
can reach it.

### Membership

Peers are catalog rows, not a configured list. A static peer list in `[worker]` was rejected: the
retrieval data plane (`../68-compute-tier-substrate/units/DIST-DATA-PLANE.md`, D9) already needs
fleet membership, and two membership mechanisms would be a knob neither needs.

- A replica becomes gang-reachable by setting `[server] peer_advertise`.
  `crates/jammi-db/src/catalog/instance.rs::InstanceRegistration::from_config` is the one choke
  point: `peer_advertise` unset yields a non-member registration (NULL columns, no check at all);
  `peer_advertise` set requires `peer_bind`, parses the address, and wraps the resolved
  result-table root. `ServerConfig::validate` is not the home of this check because it cannot see
  `artifact_dir`. Every writer of the `instances` (+`workers`) row builds this one value.
- The row carries the configured result root verbatim
  (`crates/jammi-db/src/catalog/instance.rs::MemberRoot`: `[storage] result_root` when set, else
  `{artifact_dir}/jammi_db`) paired with its identity across spellings
  (`crates/jammi-db/src/catalog/instance.rs::RootIdentity` — scheme aliasing, symlinks, case and
  slash folding), written to `instances.result_root_identity`.
- `crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::list_gang_members` takes a `GangListing
  { kind, self_instance, root_identity, lease }` and returns the instances whose `workers.kinds`
  contains the kind, whose `peer_addr` is set, whose `last_seen_at` is fresh, which are not
  draining or warming, which are not the caller, and whose root identity equals the caller's.
  Kinds are split on `,` and matched as whole tokens in Rust (`fine_tune` must not match
  `graph_fine_tune`). The member order is byte order on `instance_id`, sorted in Rust — never a
  SQL `ORDER BY`, whose collation is backend-dependent, and never `RETURNING` order, which is
  unspecified on both backends.
- **Root identity equality is necessary, never sufficient, for shared storage.** Two members whose
  roots spell identically but sit on different filesystems pass the listing; sufficiency is
  established only by the attestation verify below. A fleet with no shared result root cannot form
  a `Peer` gang at all — the listing is empty
  (`crates/jammi-db/tests/it/gang_membership.rs::file_and_s3_rooted_members_are_not_gang_members_of_each_other`).
- Freshness is `crates/jammi-db/src/catalog/lease.rs::instance_liveness_margin` (`2 × lease`);
  deletion is `crates/jammi-db/src/catalog/lease.rs::instance_prune_window` (`3 × lease`),
  strictly beyond the margin, so a member judged merely stale is never also eligible for deletion.
  On a failed touch the lease keeper re-upserts the whole tuple — the `instances` row and, when
  the registration carries worker facts, the `workers` row, in one transaction
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::reregister_instance`) — because
  `Catalog::touch_instance` is a pure `UPDATE` that can never resurrect a pruned row. A process
  whose row was pruned during a transient outage rejoins on its next heartbeat without a restart.
- `crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::peer_addr_of` is the one by-id address
  resolution verb: fresh-only under the same margin, no kind or self filter, unreachable from any
  public RPC, ignoring any caller tenant.
- A replica that sets `peer_advertise` to be gang-reachable also joins the retrieval ring, since
  that ring is every row with `peer_addr` set. Capability-scoping the ring is not done.

### The wire: `GangService.RunRank`

One bidirectional RPC, `GangService.RunRank(stream RankControl) returns (stream RankEvent)`
(`crates/jammi-wire/proto/jammi/v1/gang.proto::GangService`), served only on the internal
`[server] peer_bind` listener beside `PeerService`
(`../68-compute-tier-substrate/units/DIST-DATA-PLANE.md`, D7) — never on the tenant-scoped public
chain, which answers `UNIMPLEMENTED` for these paths, and never advertised by `GetServerInfo`. The
coordinator dials each member and sends exactly one `Assign { job_id, attempt, rank, world,
coordinator_instance_id }`.

**Nothing dialable and no data identity travels on the wire.** An earlier shape with URLs in the
assignment and a `FetchPartition` RPC would have made the peer listener an authenticated but
unauthorized compute and server-side-request primitive. Instead the training-set identity lives
on the `jobs` row (`jobs.training_set_ref` / `training_set_location`, migration
`034_jobs_training_set_identity`, a paired-nullability `CHECK`), written once by the coordinator
through a compare-and-set
(`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::fill_training_set_identity`: a concurrent
racer reuses the winner's pair, a moved claim aborts with no write), and resolved host-side by each
member. A rank reads its partition from the shared object store; there is no wire fetch of bytes
every member can already read. Addresses are resolved from `instances.peer_addr` by instance id.

**Authorization — invariant I-GANG: the job row is the capability.**
`crates/jammi-server/src/grpc/gang.rs::GangServer::run_rank` decides, in this order, before a
single stream event is emitted:

1. the wire-level edges (`world == 0`, `rank >= world`), before any row is read;
2. ambient admin scope, refused outright;
3. the row predicate through
   `crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::get_job_for_rank` — a db-owned verb that
   reads by primary key with no tenant predicate and never under admin scope, reachable only from
   the gang handler (`Catalog::get_job` is tenant-filtered and the peer path forbids admin scope):
   `status = 'running'`, `claimed_by` = the caller's coordinator, at the caller's attempt, lease
   live;
4. the **row's own** `world_size`, decoded from the same `spec` JSON the claiming worker
   reconstructs its run from — the caller's `Assign.world` must merely agree with it;
5. for `world_size > 1`: the training-set identity pair is filled on the row, and the row's own
   `tenant_id` pins a strict tenant-scoped resolution of a `ready` result table
   (`crates/jammi-db/src/catalog/result_repo.rs::Catalog::get_result_table_for_tenant`, never the
   relaxed read) whose sidecar manifest verifies `artifact == training_set_ref`;
6. the coordinator's own liveness
   (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::fresh_instance`, one row by primary key —
   the member never performs the full roster read).

The tenant is **derived, never accepted**: `Assign` carries none, no session-tenant extension is
read, and the only tenant the handler uses is the `jobs` row's own `tenant_id`. Every determinant
above collapses to the same `FailedPrecondition` with one fixed message, so the listener discloses
neither a job's existence, its claimant, its attempt, its tenant, nor another tenant's table
(`crates/jammi-server/src/grpc/gang.rs::GangRefusalReason` is visible only to same-process tests).
A catalog fault at admission is `Unavailable`, never `FailedPrecondition`. `RunRank` sits in its
own `GANG_LISTENER_ALLOWLIST` bucket in
`crates/jammi-server/tests/it/tenant_isolation_oracle.rs`, never appended to the peer-search
bucket, and its `PACKAGE`/`RPC` lines are in `crates/jammi-server/tests/it/api_freeze_baseline.txt`.
Message shape and arity are frozen by review: the freeze guard decodes only package and RPC
tokens, so later changes may add fields or oneof variants but never rename or remove. Mutual
transport auth stays the deployer's runtime, as for every surface.

**Two observables, split at admission.** Before admission every outcome is the call's own result
(`Err(Status)`), and no stream exists. After admission the call has returned `Ok(stream)`, so every
later outcome is delivered in the stream: `Admitted`, then exactly one terminal event —
`Aborted{reason}` or the rank body's `Outcome` — or a status trailer for a protocol violation (a
second `Assign`). An admitted session's end may name its reason
(`Refuted`/`Unavailable`/`StoreUnavailable`/`Drain`/`Cancelled`/`NoBody`): the caller already
holds the job's coordinates and was admitted on them, so a reason discloses nothing a
pre-admission refusal withholds.

### A peer is a fleet worker with one admission holder

A peer is a `JobWorker` process whose `[worker] kinds` include the job's kind and whose `peer_bind`
is set. There is no new worker state: `RunRank` contends for the host's single holder cell
(`crates/jammi-ai/src/fine_tune/worker.rs::HostAdmission`,
`crates/jammi-ai/src/fine_tune/worker.rs::Holder`), every transition a compare-and-set with no
lock held across an `.await`:

- The claim loop moves `Free → ClaimProbe` immediately before `claim_next` and `ClaimProbe →
  JobRun` on a claim (`crates/jammi-ai/src/fine_tune/worker.rs::JobWorker::run_until`). So a peer
  never claims while it runs a rank, never receives a rank while running its own claimed job, and
  is reachable whenever idle.
- An admitted rank moves `Free → Rank{job_id, attempt}`. A `ClaimProbe` is waited on for at most
  one heartbeat; `JobRun` or another `Rank` refuses `Unavailable` at once — transient, no assembly
  budget consumed.
- **Attempt fence on `job_id`.** A held rank of the same job at a lesser attempt is taken over;
  an equal or greater held attempt refuses. The fence keys on the job, not on `(job_id, rank)`,
  because a rank can move hosts between attempts.
- Ending a session never aborts a claim transaction: a member's holder is `Rank` for the whole
  session and the claim loop cannot be inside `claim_next` while it is. This preserves the rule
  that nothing aborts the loop while a client-driven claim `COMMIT` may be in flight
  (`../68-compute-tier-substrate/units/OPS-COMPUTE-TIER-OPERABILITY.md`, D6).
- Inline `run_now` (`crates/jammi-ai/src/jobs.rs::run_now`, compute specs only) never touches the
  holder; it deliberately runs beside a loop-claimed job or an admitted rank.

An admitted session is **held** by a spawned loop with these arms: the inbound stream (`Cancel`
ends the session cooperatively; round frames are delivered to the session's round inbox); the
host's phase watch (a DRAIN or RELEASE ends every held rank with `Drain`, the only host-initiated
cut); a re-verification tick, once per heartbeat, that re-decides the row predicate, the
training-set identity and the coordinator's liveness and ends the session `Refuted`,
`Unavailable` or `StoreUnavailable` (`crates/jammi-server/src/grpc/gang.rs::ReverifyEnd` — the
three are pairwise distinct in scope and in whether they count against assembly); and either the
rank body's end or, for a body-less `world_size == 1` session, the park bound (`NoBody`, one lease
window after admission). The peer writes nothing terminal on behalf of a rank: every end is a
stream event and the job row is untouched; a source-scan oracle enumerates the catalog's `jobs`
writers and asserts none is named in the handler.

### The collective

One trait, selected by topology and configuration
(`crates/jammi-ai/src/fine_tune/collective/mod.rs::Collective`):

```
trait Collective {
  fn all_gather(&self, call: &BlockingCall, local: &Tensor, counts: &[usize]) -> Result<Tensor>;
  fn all_reduce_sum(&self, call: &BlockingCall, tensors: &mut [Tensor]) -> Result<()>;   // canonical trainable_vars order
  fn all_reduce_max_flags(&self, call: &BlockingCall, flags: u32) -> Result<u32>;
  fn broadcast(&self, call: &BlockingCall, t: &mut Tensor, root: u32) -> Result<()>;
  fn barrier(&self, call: &BlockingCall) -> Result<()>;
  fn rank(&self) -> u32; fn world(&self) -> u32;
  fn bind_agreement(&self, digest: String) -> Result<()>;
}
```

`BlockingCall` (`crates/jammi-ai/src/fine_tune/collective/mod.rs::BlockingCall`) is a witness
token minted only at a `spawn_blocking` boundary and required by every verb, so calling a
collective from a runtime-worker thread is a compile error, never a runtime refusal. The trainer
holds one `&dyn Collective` and is never `cfg`-forked.

- **`Noop`** (W=1): gather and reduce are identity.
- **`Local`** (in-process ranks on the devices of one host): rank-ordered reduce on rank 0's
  device, deterministic. Every round carries a descriptor (verb, world, root, counts, per-tensor
  signature, and an agreement slot bound to the canonical trainable-variable key digest); a round
  publishes only once every rank's descriptor is equal, and a disagreement before publication is a
  symmetric typed error on every rank naming both descriptors.
- **`Peer`** (`crates/jammi-ai/src/fine_tune/collective/peer.rs::Peer`, across processes over the
  `RunRank` stream): rank-ordered coordinator-reduce. Every member sends its descriptor and its
  tensors as Arrow IPC; the coordinator folds in rank order on its own device with the same
  operation sequence `Local` runs, so the two arms are byte-identical over the same inputs.
  Elements travel exactly (f32 as `Float32`, f16 as `Float16`, bf16 as its `UInt16` bit pattern;
  every other dtype is refused). A payload larger than `[server.limits] max_message_bytes` is
  split into `RoundChunk`s; the peer listener applies the same configured decode cap the public
  chain applies rather than tonic's 4 MiB default. The round is **two-phase**: a member holds the
  coordinator's result unapplied, ACKs, and applies only on `RoundCommit`, which the coordinator
  sends after observing every member's ACK. A fault before the last ACK leaves no rank applied for
  that round, and every rank's error names the round. A fault during the commit fan-out is fatal on
  every rank; the one residual state — a member the commit reached has applied round k while the
  gang faulted — cannot continue, because its next contribution is answered by the coordinator's
  fault. The round index is a descriptor field, so a stale contribution is a typed disagreement.
  Every wait on every rank expires at the gang deadline (`[worker] rank_timeout_secs`) with a typed
  error naming the round and what it waited for.
- **`Nccl`** (`crates/jammi-ai/src/fine_tune/collective/nccl.rs`, compiled under the `cuda`
  feature through candle's `cudarc/nccl` re-export; `Comm::from_devices` for single-process
  multi-GPU, `Comm::from_rank` across processes). It implements the same trait and is exercised on
  the GPU legs (§10). Three measured facts shape it:
  - NCCL has no `allgatherv`. Unequal counts are pad-to-max on the way in and narrow on the way
    out; the result is bitwise equal to a CPU concat for f32 and bf16.
  - A dead peer is not detected. The host call does not block; the wait is in
    `stream.synchronize()`, which sits indefinitely against a dead peer. The escape is
    `ncclCommAbort` from another thread, after which `synchronize` returns `Ok` over a **garbage
    buffer**. The abort flag, never the sync's return value, is the failure signal, and every
    operation checks it after synchronizing.
  - A communicator must never be aborted twice (the second call segfaults), and cudarc's
    `Drop for Comm` is an abort. The double abort is made unrepresentable: the comm lives in a
    `Mutex<Option<Comm>>` and `crates/jammi-ai/src/fine_tune/collective/nccl.rs::Nccl::abort` takes
    it out and drops it.

  NCCL exchanges buffers and nothing else, so this arm has no descriptor agreement: a gang whose
  ranks disagree about counts or root produces a wrong result or a hang. Agreement is kept upstream
  of the collective — every rank derives its counts from the same partition rule and walks the same
  canonical trainable-variable order.

`[worker] collective = auto | nccl | cpu` is configuration, not a build feature
(`crates/jammi-db/src/config/mod.rs::CollectiveSelection`); `nccl` on a build without CUDA
refuses at session open. The model descriptor records the collective the run actually reduced
over.

### The step (the gather rule)

Per global batch t, each rank encodes its slice, then `all_gather`s the step's tensors using the
counts every rank derives from the partition rule. Every rank then computes the **identical global
loss** over the gathered batch through the existing loss functions
(`crates/jammi-ai/src/fine_tune/trainer.rs::dispatch_contrastive_loss`, `mnrl_loss`,
`cross_entropy_loss`, the regression/quantile losses), so each loss's own 1/n runs over the global
n.

The alternative — each rank computes a local loss and gradients are averaged — is rejected because
the default embedding loss (CoSENT), AnglE and MNRL are **batch-coupled**: their terms range over
pairs of rows in the batch, so the average of per-rank gradients is not the gradient of the
global-batch loss. With the gather rule, batch-coupled objectives keep exactly their W=1
semantics.

Periodic parameter averaging (each rank trains locally for a round, then the ranks average their
adapters — local SGD, the `treeAggregate` shape of §1) is rejected for the same reason plus one
more: it trades exactness for communication the workload does not need to save. With adapter
payloads in the kilobyte-to-megabyte range, synchronous data parallelism is exact at negligible
cost, so the convergence risk of local SGD buys nothing.

**Invariant: the gather point is downstream of every trainable parameter; nothing trainable
consumes a gathered remote slot** — otherwise the summed gradient of that parameter is W× too
large. Gather points per `TrainingBatch` arm:

- contrastive / pairs / triplet gather the encoder outputs (post-projection,
  `crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::encode_texts`) and the scores;
- **classification gathers the logits** — the trainable head is applied inside the loss
  (`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::compute_loss`, head at
  `TrainingLoop::classify`), so gathering `embeddings` would route remote rows through a trainable
  parameter;
- regression gathers the head output (`TrainingLoop::head_forward` runs pre-loss) and the targets;
- NER stays refused (`TrainingLoop::compute_loss_per_example`, the `Ner` arm).

Matryoshka prefixes narrow dim 1 only (`crates/jammi-ai/src/fine_tune/trainer.rs::matryoshka_sum`)
and are orthogonal to a dim-0 gather. Backward runs through a gather whose backward keeps only the
local slots (rank r's own rows), so no gradient crosses the wire during backward. At each
optimizer-step boundary the adapter `GradStore` is laid out in the canonical `trainable_vars` order
with zeros for absent entries — a sparse `GradStore` is a real shape
(`crates/jammi-ai/src/fine_tune/optimizer.rs::clip_and_step`), and without zero-filling the reduce
set could differ per rank — then `all_reduce_sum`med, then `clip_and_step`. Gradient accumulation
counts global batches. Every rank holds identical weights after the step.

### Lockstep control flow

The step boundary is the global batch index, never a rank-local counter. Divergence
(`loss.is_nan() || loss > 100`,
`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::process_batch_loss`), the three-strikes
abort, early stopping and epoch exit are decided by `all_reduce_max_flags` at the same boundary on
every rank; validation loss is computed by rank 0 and the stop flag broadcast. No rank can reach a
collective a different number of times than its peers. Rank-local, data-dependent control flow is
exactly what this rule removes.

### Checkpoints and resume

Rank 0 alone writes the resume checkpoint at epoch boundaries and publishes; other ranks'
checkpoint calls are unreachable by role. Each rank's dropout Philox position
(`crates/jammi-ai/src/fine_tune/resume.rs::ResumeState::dropout_positions`) is gathered to rank 0
at the epoch boundary and stored **per rank** in the bundle, and each rank's dropout seed derives
from the job seed and the rank. A resumed gang at equal topology therefore reproduces an
uninterrupted one. At `W > 1` every rank discovers the resume checkpoint against the same
tenant-scoped prefix under the shared result root
(`{tenant}/{job_id}/_resume/`,
`crates/jammi-db/src/store/artifact.rs::ArtifactStore::put_resume_checkpoint`), which a zombie
writer cannot regress because the write is gated on the held lease
(`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::save_resume_checkpoint`). A `Peer` gang
and a `Local` gang publish byte-identical adapters from the same checkpoint, and a corrupted
checkpoint fails the attempt loudly rather than silently restarting
(`crates/jammi-server/tests/it/gang_resume_parity.rs`).

### Shared storage and the partitioned attestation

Root identity is only necessary for shared storage; each member proves sufficiency by verifying
what it reads. The materialization manifest carries a keyed **leaf inventory** in addition to the
whole-object digest (`crates/jammi-db/src/store/manifest.rs::LeafDigest`,
`crates/jammi-db/src/store/manifest.rs::LeafKey`): `RowGroup { index, offset, length }` for a
result table, `File { name }` for a model bundle. A rank verifies its read path one row group at a
time — bounded memory, never whole-artifact buffering — and
`crates/jammi-db/src/store/mod.rs::ResultStore::verify_partitions` names the first divergent leaf.
A corrupted leaf is caught on the member before any collective step and ends the session
member-scoped (`StoreUnavailable`), never counted against the assembly. A sidecar with no `leaves`
reads as absent (re-materialize), never as a whole-artifact read accepted in its place.

The inventory is **additive** to `manifest.artifact`. Making `artifact` a fold over the leaves was
rejected: `artifact` is the base of the version-identity chain and is reused as the base
fragment's digest, and row groups do not partition a Parquet file (footer, page index and bloom
filters belong to no leaf), so a fold would have moved every downstream anchor and weakened the
subject it attests.

### The coordinator

`crates/jammi-ai/src/fine_tune/worker.rs::JobWorker::coordinate` runs rank 0 of a `Peer` gang in
the process that claimed the job:

1. the write-once CAS of the training-set identity pair (a moved claim exits with no write);
2. the membership listing with this process's own root identity — the verb filters, the body
   filters nothing;
3. rank assignment as a pure function of the sorted listing. There is no substitution: a member
   answering `Unavailable` ends the current attempt, and the next attempt re-lists;
4. dispatch — `peer_addr_of`, then dial with the `Assign`;
5. the run as rank 0 over `Peer`;
6. exactly one assembly outcome recorded on the row, then the lease settled (below).

`world_size` is checked at submit against `[distributed] max_world_size` only
(`crates/jammi-ai/src/fine_tune/spec.rs::RankAdmission`): a world size within it but beyond this
host's own devices submits and is decided by assembly, never refused at submit.

**Assembly outcomes are a total table**
(`crates/jammi-db/src/catalog/jobs_repo.rs::AssemblyOutcome`; a new variant with no rule is a
compile error). Migration `037_jobs_assembly_failures_next_after` adds the durable counter and
cooldown:

| outcome | counted | cooled down |
|---|---|---|
| `Refuted`, `AllRootDivergent` | yes | yes |
| `Unavailable`, `StoreUnavailable`, `ShortListed` | no | yes |
| `NoBody`, `Drain`, `Cancelled` | no | no |
| `Success` | resets both | |

The coordinator body never produces `AllRootDivergent`: root identity is a predicate inside the
membership listing, so a divergent member is simply not listed and the attempt ends `ShortListed`.

The cooldown term sits in `claim_next`'s **candidate subselect**, on the same backend clock the
lease columns use — never the outer `UPDATE` guard and never a process clock — so a
higher-priority job inside its cooldown does not block a lower-priority ready job, and a skewed
process clock never changes when a cooldown expires.

### Failure and release

**An aborted attempt lands no terminal write.** `fail_job` is terminal
(`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::fail_job`: `status = 'failed'`, no attempt
bump). The fleet's only requeue path is to leave the row `running`: reclaim
(`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::reclaim_expired_jobs_inner`, arm 1a) requeues
it once the lease expires, and `attempts + 1` happens at the successor's claim. So a gang fault
never calls `fail_job`; it retires the attempt and lets reclaim requeue the job within the
remaining lease window (≤ `[lease] duration_secs`). There is no per-task retry anywhere.

The per-attempt watchdog is the coordinator's own `Peer`: every member's stream is read by its
rounds, so a member's `Aborted{reason}`, a dropped stream, or a rank silent past
`[worker] rank_timeout_secs` ends rank 0's collective call with the gang faulted; every member is
faulted in the same round and its session ended. The `Peer` is built for the attempt and dropped
with it, so a fault retires exactly the attempt it belongs to. This is the lease keeper's shape —
bounded by the attempt, created by the claimant for the job it holds — and is allowed under the
actuator rule (`crates/jammi-ai/src/pipeline/recompute.rs`, module doc: "the engine ships the
actuator; it never ships the control loop that pulls it").

**A released rank is a release, not a failure**
(`crates/jammi-ai/src/fine_tune/worker.rs::lease_settlement`):

- a member's `Aborted{Drain}` — its host draining, a rolling restart of the peer tier — hands the
  lease back at once through
  `crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::release_job_lease` (`releases + 1`, lease
  NULL), so the restart costs the job zero net attempts
  (`../68-compute-tier-substrate/units/OPS-COMPUTE-TIER-OPERABILITY.md`, D10);
- every other mid-run gang fault leaves the lease to expire: an attempt spent at the successor's
  claim, so a member that keeps failing exhausts the job's attempts instead of retrying forever;
- an assembly end settles by its outcome's counting class: an uncounted outcome hands the lease
  back at once, a counted one leaves it to expire.

Coordinator death expires the lease. Either way the next attempt resumes from the job-level resume
checkpoint. The executed cases are in `crates/jammi-server/tests/it/gang_chaos.rs` (a dropped
member stream, a silent member, a draining member, a stale older-attempt runner fenced by its
successor) and `crates/jammi-ai/tests/distributed/gang_chaos.rs` (a killed peer, a killed
coordinator).

### Device-plural session

`[gpu] devices = [..]` gives one `GpuScheduler` per device and a `ModelCache` keyed by
`crates/jammi-ai/src/model/cache.rs::CacheKey` — `{ model_id, device, task: Option<_>, backend:
Option<_> }`, where `None` is a distinct key value, never a wildcard — applied to both the entries
map and the single-flight in-flight map (`crates/jammi-ai/src/model/cache.rs::CacheInner`). `Local`
ranks are threads pinned to devices.

## 5. The frozen forward of the projection-head target

For `ProjectionHead` training the tower is frozen, but the frozen forward is not a producer
table: `encode_texts`/`encode_media` (`crates/jammi-ai/src/fine_tune/trainer.rs`) call
`project_frozen_embedding` inline, per training batch, to run the frozen tower and hand the pooled
output to the head's own LoRA layer; evaluation uses the identical call. Nothing under
`fine_tune/` drives `InferenceExec` or reads an `Embedding` result table for this path. At
`world_size > 1` each rank runs its own slice's frozen forward inside the normal step, and the
head's output is a gather point like any other target's (§4). No partition set exists to fan out
over, so there is no peer-to-peer fetch of a shared table, and peers exchange data only through
the collective; there is no second peer-to-peer surface.

Partitioned inference is a separate mechanism: `InferenceExec` fans out N ways in one process
below `crates/jammi-ai/src/operator/ordinal_split_exec.rs::OrdinalSplitExec`. That node has no
wire form across a distributor — its mechanism is one in-process guarded pull shared by its N
partitions, which has no meaning across executors (the reasoning is in
`crates/jammi-ballista/src/codec.rs`'s module doc). A cross-executor range split of `InferenceExec`
is tracked at <https://github.com/f-inverse/jammi-ai/issues/540>.

## 6. Properties the tests hold

| Property | Kind | Where |
|---|---|---|
| **Refactor parity**: W=1 through the table-backed loader, the train-prefix scaler and `Noop` produces adapter bytes identical to the pinned cookbook fine-tune fixtures (`cookbook/book/artifacts/finetune_*/checksums.json`) | byte | cookbook; hermetic |
| **Remote equals embedded, over the wire**: W=2 across two processes (`Peer`) equals W=2 in-process (`Local`), byte-for-byte. Rank 0 is always in-process, so W=1 never crosses the wire and is covered by the `Noop` row | byte | hermetic loopback; distributed lane |
| **Equal-topology reproducibility**: two runs, same W and spec → identical bytes; also across a resume (kill at epoch k, resume, compare to uninterrupted) | byte on `Local`/`Peer`; on GPU legs see below | hermetic; gpu-gang |
| **W-invariance**: W × B versus W=1 × W·B, identical loss per step within ε, at `lora_dropout = 0` (dropout and bucketing make W-invariance non-exact even for decomposable losses); ε is measured on the leg that gates — a CPU ε is never inherited by a GPU leg | tolerance | hermetic; gpu-gang |
| **Gather exactness**: for CoSENT, AnglE, MNRL, **classification** and **quantile regression**, the W=2 global loss and the summed adapter gradient at step t equal the W=1 loss and gradient on the same rows bit-for-bit on CPU, on a fixture whose `train_count` is not a multiple of W·B (a zero-row rank occurs) | byte | hermetic |
| **Lockstep**: one rank's batch forced to diverge; one rank's batch yields no gradient for a Var; a zero-row rank; the gang completes | property | hermetic |
| **Gang failure**: a killed peer → job requeued, completed by a new gang from the checkpoint, exactly one model, no orphan prefix promoted; a killed coordinator → same via lease; an older attempt's stale runner is fenced and writes nothing; a draining member releases the lease at zero net attempts | property | server it-suite; distributed lane |
| **Authorization**: `RunRank` for a job not running / not claimed by the caller / at another attempt / lease expired / wrong world / coordinator not fresh is refused with one indistinguishable status | property | server it-suite |
| **Cache reuse refused**: `CachePolicy::Use` on a fine-tune is refused on every durable submit edge; two model rows may share one prefix; reaping respects references | property | hermetic |
| **Distributor-agnosticism** (§9): codec round-trip for every operator; a gang placed through a scheduler publishes the same bytes as the peer-based run and is never re-launched | byte / property | `jammi-ballista` and `jammi-ai` it-suites; distributed lane |

**GPU byte claims.** Measured on candle 0.11: a LoRA-shaped forward/backward/SGD is byte-identical
across processes and across two A100s with no environment pins. At world size 2 the reduction is
commutative, so the digest pair is **asserted** equal on a passing GPU leg, and the per-step loss
delta is compared to an ε registered for that leg before its first gating run. At world size ≥ 3
the NCCL collective-algorithm pin set (`NCCL_ALGO`, `NCCL_PROTO`, channel counts) is unproven, so a
digest pair there is recorded, not asserted. Two known sources of GPU non-determinism — the
atomic-add `dQ` accumulation in flash-attention's backward outside its `deterministic` mode
(`crates/jammi-kernels/third_party/flash-attention/src/flash_bwd_kernel.h`) and NCCL's channel
split — are why GPU byte-equality is a per-leg measured claim rather than an engine guarantee.

## 7. Configuration and placement

```
[gpu]         device = 0 ; devices = [0, 1]
[worker]      enabled = true ; kinds = "all" ; local_ranks = 1 ; rank_timeout_secs = 120 ; collective = "auto"   # auto|nccl|cpu
[distributed] max_world_size = 1
[server]      peer_bind = "..." ; peer_advertise = "..."
[ballista]    scheduler_bind = "..."
[ballista.executor] scheduler_address = "..." ; bind = "..." ; grpc_bind = "..." ; advertise_host = "..." ; work_dir = "..." ; task_slots = N
```

Three unrelated world-size concepts, loaded independently with no cross-check:

- `[worker] local_ranks` — how many ranks **this host** places on its own `[gpu] devices` for a
  job it runs entirely in-process (`local_ranks <= devices.len()` is enforced at config load).
- `[distributed] max_world_size` — the widest `Peer` gang any coordinator on this deployment may
  accept: one rank per **fleet member**, never per local device
  (`crates/jammi-db/src/config/mod.rs::DistributedConfig`).
- per-job `world_size` on `TrainingCommon` (identity-relevant; `#[serde(default)]` = 1, so queued
  specs written before the field existed still deserialize), checked against `max_world_size` at
  submit.

Catalog migrations this design appends: `033_model_materialization`,
`034_jobs_training_set_identity`, `035_instances_peer_addr_result_root`,
`036_instances_result_root_identity`, `037_jobs_assembly_failures_next_after`,
`038_compute_cluster_state`. The per-row-group leaf digests are a sidecar-object change and append
none.

Placement is the deployer's runtime. The Kubernetes reference overlay runs the compute tier as a
StatefulSet with a headless Service and `nvidia.com/gpu: N`, with a dedicated single-replica
scheduler Deployment (`deploy/kubernetes/overlays/shape-d/`): gang members need stable network
identities, which a Deployment's pods do not have. Compose lists services; Slurm and Ray are
placement options only.

## 8. Non-goals

Sharded model or optimizer state (FSDP-like); elastic gangs; SGD or gradient exchange as
DataFusion operators or aggregates; `ContextPredictor` on a gang; hard-negative mining and
GradCache at `world_size > 1` (typed refusal); a sixth pluggable backend; an engine-level GPU
byte-equality guarantee (§6 states what is measured); changing Ballista itself — an accelerator
resource dimension in its executor specification is carried by the engine's own catalog and
placement policy (§9), never by a fork or an upstream request.

## 9. The Ballista extension

**Discipline.** `jammi-kernels` extends candle at the seam candle exposes (`CustomOp1/2/3`), keeps
one call path, vendors verbatim at a pinned version only where no seam exists, and believes
nothing before its oracles pass. `jammi-ballista` follows the same discipline: extend at the seams
the library exposes, never fork. The crate is not a leaf — it encodes jammi operators and hosts
the model cache and catalog — so it sits between `jammi-ai`/`jammi-db` and `jammi-server`,
publishable and lockstep. It is a crate rather than a `jammi-server` module or a cargo feature
because topology is configuration: a feature would be a library-vs-server gate, and a crate lets a
library embedder host the roles too. `jammi-server` depends on it unconditionally and decides at
runtime, from `[ballista]`, whether a process hosts either role. Neither `jammi-ai` nor `jammi-db`
depends on it; the placed-gang submitter and runner seams `jammi-ai` exposes are *installed* by
this crate's roles.

**Seams used (Ballista 54.1, read from source).**

| Gap | Seam | What jammi installs |
|---|---|---|
| operators cross the wire | `SchedulerConfig.override_{logical,physical}_codec`, `ExecutorProcessConfig.override_*_codec` | `crates/jammi-ballista/src/codec.rs::JammiCodec`: `InferenceExec`, `AnnSearchExec`, `AsofJoinExec`, `KeyCheckExec`, `GangExec`, in the crate's own `jammi.ballista.v1` package. Every buffer it writes starts with the magic `[0x07, 'J', 'M', 'B']`; `0x07` is field 0 / wire type 7, which can never begin a valid protobuf message, so nothing Ballista's own codec writes can alias it. A buffer without the magic delegates whole to Ballista's codec. Decode rebuilds each operator against the decoding process's own session — model cache, result store and context are never serialized. `MaskExec` and `OrdinalSplitExec` are refused typed (§5) |
| no executor-side state across plans | `ExecutorProcessConfig.override_execution_engine`; `create_query_stage_exec` rewrites `ShuffleReaderExec` nodes and wraps the writer | `crates/jammi-ballista/src/engine.rs::JammiExecutionEngine`: refuses, typed, a stage whose `InferenceExec`/`GangExec` names a device kind different from this executor's own, and a `GangExec` stage with more than one partition. Shuffle stays Ballista's local `work_dir`; this seam is where an object-store shuffle would go once a cross-executor read is proven |
| cluster state in memory only | `ClusterState` + `JobState` traits; `BallistaCluster::new(Arc<dyn ClusterState>, Arc<dyn JobState>)` | `crates/jammi-ballista/src/cluster.rs::CatalogClusterState` / `CatalogJobState` over distributor-neutral tables (`compute_executors`, `compute_jobs`, migration `038_compute_cluster_state`) through generic CRUD in `jammi-db`; this module is the only place Ballista's vocabulary meets the catalog |
| no accelerator dimension | `ClusterState::bind_schedulable_tasks`; `TaskDistributionPolicy::Custom(Arc<dyn DistributionPolicy>)` | `crates/jammi-ballista/src/placement.rs::DevicePlacement` (below). The Rust `ExecutorSpecification { task_slots: u32 }` has no accelerator attribute slot; the proto side's `oneof resource { TaskSlots(u32) }` is extensible, so an accelerator dimension is an upstream variant, not a schema break. Until then jammi carries device kinds out of band in `compute_executors.devices` |
| task retry rejoins a dead gang | `SchedulerConfig.task_max_failures`, `stage_max_failures` (scheduler-global) | both 0: retries are the jobs table's `attempts`/reclaim. Measured: retries are off for jammi operators by error classification (a panic → `Internal`, any operator error → `ExecutionError`, both non-retryable) rather than by the knob; the knobs still close Ballista's own I/O-retry and fetch-failure arms |
| scheduler control loop | `expire_dead_executors` starts unconditionally in `SchedulerServer::init` | see "Executor loss" below |
| push launching, transport | `TaskLauncher`, `override_create_grpc_client_endpoint`, `use_tls` | not used |

**Roles are listener-shaped knobs**, the same class as `peer_bind` and `health_listen`: a replica
hosts a scheduler iff `[ballista] scheduler_bind` is set, an executor iff
`[ballista.executor] scheduler_address` is set; both may be set on one process; unset is a single
node. Not a tier, not a CLI role. The roles are hosted on jammi's own two-mode shutdown
(`crates/jammi-ballista/src/roles.rs`), never Ballista's `start_server`/`start_executor_process`,
which install their own `ctrl_c` handlers and would race it. DRAIN stops task admission and waits
for an in-flight placed gang; only RELEASE tears an executor down at once. The six configured
addresses are checked for collisions by one cross-section validator over the whole config, and
`advertise_host` is required whenever `bind` is unspecified.

**Placement** (`DevicePlacement`, round-robin over executor slots with three refinements):

1. **Submitter exclusion.** A `GangExec` stage is never bound to the executor whose id equals the
   descriptor's own submitter: that host holds an admission for the whole await, so binding the
   gang task back to it would deadlock the placed run against itself. This is decided before
   topology.
2. **Kind match, not "is GPU-bound".** A stage whose plan carries a required device kind — a
   `GangExec` (its descriptor's stamped kind, CPU included) or an `InferenceExec` (its required
   constructor argument; the codec never invents or rewrites it) — binds only to an executor whose
   own registration lists that exact kind. `crates/jammi-ballista/src/engine.rs::required_device_kind`
   is the one predicate the placement policy, the engine's stage check and the client's
   pre-submission refusal all read. A "GPU-bound" predicate would refuse a `GangExec`
   unconditionally on an all-CPU cluster and would let a CPU-kind `InferenceExec` bind to an
   executor reporting no devices. `compute_executors.devices` is the join's sole authority, never
   `workers.devices` and never a join on `instance_id`. The policy reads the live stage plan in
   `active_jobs` directly.
3. **Re-launch guard.** A `GangExec` stage whose job row is already `claimed_by` an executor other
   than its submitter is never bound to any slot (next paragraph).

The slot compare-and-set against the catalog's committed `available_slots` runs per candidate
before the graph's task info is stamped; a lost CAS leaves the task unstamped for the executor's
own retry. Ballista's built-in binders are `pub(crate)`, so a custom `ClusterState` must bring its
own; in pull-staged mode `bind_schedulable_tasks` is bypassed entirely and only the `Custom` policy
is honoured, so the role knobs pin push-staged scheduling.

**Executor loss resets tasks with no knob to stop it.** `expire_dead_executors` sweeps executor
heartbeats — the same class as `reclaim_expired_jobs` and `prune_instances`, which the fleet
already runs each tick — but the same loop also posts `ExecutorLost` →
`reset_stages_on_lost_executor`: a running stage's lost task has its slot freed, and a successful
stage's completed tasks are re-failed as `ResultLost` (`retryable: true, count_to_failures:
false`), which `update_task_status` resets **without consulting `task_max_failures`**. The
`ExecutorLost` arm itself posts no `ReviveOffers` and no failure, so re-launch on a surviving
executor is conditional on a later independent event (a new executor registering, or a subsequent
task-status success, under push-staged scheduling); absent one, the freed task can sit unscheduled.
Ballista's `task_attempt` counter is not bumped by this reset, so a guard cannot key on it. The
guard is therefore keyed on jammi's own job row: the claim is transferred to the placed executor
at launch, and the bind-time predicate above refuses any second launch. A job row the policy
cannot read is folded into the same refusal. With this in place, killing an executor mid-gang
fails the attempt and requeues it through jammi's lease path.

**Scheduler restart and multiple schedulers.** Executor registrations and heartbeats survive a
scheduler restart from the catalog-backed `ClusterState`. An in-flight job does not:
`ExecutionGraphBox` has no serialisation in 54.1 and `JobState::try_acquire_job` is never called by
the scheduler, so `compute_jobs` mirrors ownership and status text only, and recovery is jammi's
own reclaim — a re-run as a new Ballista job id — never Ballista-side job survival.
`ballista-scheduler` is built with `default-features = false`, and that is load-bearing: the REST
API's `get_running_jobs` errors on a status row with no execution graph, which is exactly the state
a restarted scheduler holds. Two schedulers over one store serve one cluster only for
**sequential** jobs; concurrent jobs hang on whichever scheduler loses the slot race, with no
public path to wake it (`revive_offers` and the query-stage event loop are `pub(crate)`,
`job_resubmit_interval_ms` has no readers, `cluster_state_events` is unconsumed). The supported
shape is active/standby.

**The gang under Ballista is one placed task.** `crates/jammi-ai/src/operator/gang_exec.rs::GangExec`
is a single-partition, zero-child operator whose `execute` dispatches to the process's installed
placed-gang runner, which runs the same coordinator body as §4 on a kind-matching executor; the
ranks are fleet members reached over `peer_bind`. The submitting worker moves its holder to an
awaiting state, submits the descriptor, and hands the attempt off
(`crates/jammi-ai/src/fine_tune/worker.rs::JobWorker::run_placed_gang`; the submitter's exit arms
are total: a stream that produced a batch, or a `claimed_by` that moved, means the executor owns
the attempt and the submitter writes nothing; otherwise the row is left `running` for reclaim).
The runner is reached through a process-global installed once per process rather than a
`TaskContext` extension, because an extension set on the submitting session's config never crosses
the wire to the executor's reconstructed one. Modelling gang ranks as Ballista tasks with
all-or-nothing binding was rejected: it would be a second gang mechanism with its own parity
obligation, and it would need a stage kind Ballista does not have. The executed cases are in `crates/jammi-ai/tests/it/gang_placed.rs`.

**Device pinning does not move bytes.** The executor process runs on its configured
`[gpu] devices`; device kind is already in `MaterializationEnv`; the ordinal is not
output-affecting; the shuffle writer never reorders a partition-ordered sink.

**Publishing.** `ci/scripts/publish_crates.sh`'s `PUBLISH_ORDER` array enumerates publishable
crates in topological order; `jammi-ballista` sits before `jammi-server` (a release tag would
otherwise half-publish).

**Ballista for the compute plane only.** The retrieval data plane does not use Ballista
(`../68-compute-tier-substrate/units/DIST-DATA-PLANE.md`, D2). Of that decision's grounds, the
missing accelerator resource dimension and the unproven object-store shuffle stand; cluster state
is pluggable through public traits, as this crate shows; and the scheduler's liveness loop, with
retries off and the re-launch guard in place, never makes consumer work runnable on its own.

**Spice's fork, for comparison.** Multi-active HA via object-store state, object-store shuffle,
mTLS, bidirectional control streams, catalog/UDF sync, their own shuffle format; one binary with
`--role scheduler`; batch only; no GPU. Jammi keeps HA state at the `ClusterState` seam on the
catalog and forks nothing.

## 10. Hardware proof legs

The CPU collectives are proven hermetically in CI. The `Nccl` arm is proven on rented hardware by
two label- or dispatch-triggered lanes, each committing a `gang` artifact (world, collective,
per-rank device, digest pair, measured per-step loss delta, and the ε registered for the leg)
under `crates/jammi-kernels/artifacts/cuda-runs/`, validated by
`ci/scripts/check_cuda_run_artifacts.py`:

- **Pod leg** — one pod, two GPUs, `Comm::from_devices`
  (`docs/maintainer/dev-gpu.md#the-gang-leg--two-gpus-in-one-pod`).
- **Two-host leg** — two hosts, one GPU each, `Comm::from_rank`: rank 0 mints the NCCL id, rank 1
  joins from it, both reduce a known vector
  (`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs::gang_nccl_two_hosts_reduce_a_known_vector`;
  `docs/maintainer/dev-gpu.md#the-cluster-leg--two-hosts-one-gpu-each`). Two transports carry it;
  the proof does not care which fabric moves the bytes. The provider's clustered product measured
  near-zero capacity in practice, so the default transport is two ordinary pods in one data center
  joined by the provider's private networking, each member deriving its `NCCL_SOCKET_IFNAME` from
  its private address at run time rather than from a literal. The NCCL id is an opaque secret: a
  scan over every carrier the driver produces (`ci/scripts/gang_id_secrecy_scan.py`) runs before
  anything is uploaded.

Measured provisioning facts that shape the reaper: cluster members expose no per-pod actions, so
cleanup deletes the cluster rather than a member pod, and pod listings omit cluster members unless
asked for them; the pod sweep is fail-closed on its cluster-member exclusion set. No paid lane runs
on a schedule unless its workflow is on a reviewed allow-list (`ci/scripts/check_gpu_prove_once.py`).

The cu12 server packaging bundles `libnccl` alongside the CUDA runtime libraries because
`candle-core/nccl` adds a hard `DT_NEEDED` on it; the bundled set is a fixed hand list rather than
a `DT_NEEDED` closure walk because `libnvrtc-builtins` is `dlopen`'d by `libnvrtc`, not linked, so
a closure walk cannot be trusted to reach it (`packaging/server-cu12/README.md`).

## References

- MLlib: Machine Learning in Apache Spark, JMLR 17 (2016) — https://www.jmlr.org/papers/volume17/15-237/15-237.pdf
- Sparker (ICPP 2021) — treeAggregate ≈ 67% of time — https://dl.acm.org/doi/fullHtml/10.1145/3472456.3472499
- SPARK-24374 SPIP: Barrier Execution Mode — https://issues.apache.org/jira/browse/SPARK-24374
- TorchDistributor — https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor
- Ballista 53.0.0 / 54.0.0 / 54.1.0 — https://datafusion.apache.org/blog/output/2026/05/24/datafusion-ballista-53.0.0/ , …/2026/07/12/datafusion-ballista-54.0.0/ , …/2026/08/09/datafusion-ballista-54.1.0/
- Ballista architecture — https://datafusion.apache.org/ballista/contributors-guide/architecture.html ; extension hooks in `ballista/core/src/extension.rs`; task-level retry in `ballista/scheduler/src/state/task_manager.rs` (read at 54.1)
- Ballista 54.1 seams (read from source): `ballista/scheduler/src/cluster/mod.rs` (`ClusterState`, `JobState`, `DistributionPolicy`), `scheduler/src/config.rs` (`TaskDistributionPolicy::Custom`, `task_max_failures`), `scheduler/src/scheduler_process.rs` (`start_server(cluster, …)`), `scheduler/src/scheduler_server/mod.rs` (`new_with_task_launcher`, `expire_dead_executors`), `scheduler/src/scheduler_server/query_stage_scheduler.rs` (`QueryStageScheduler::on_receive`, the `ExecutorLost` and `TaskUpdating` arms), `executor/src/executor_process.rs` (`override_execution_engine`), `executor/src/execution_engine.rs` (`ExecutionEngine`, reader rewrite), `core/src/serde/scheduler/mod.rs` (`ExecutorSpecification`)
- datafusion-distributed — https://github.com/datafusion-contrib/datafusion-distributed
- cudarc `nccl` — https://docs.rs/cudarc/latest/cudarc/nccl/index.html ; candle-core 0.11 `nccl = ["cuda", "cudarc/nccl"]`
- candle `llama_multiprocess` — https://github.com/huggingface/candle/blob/main/candle-examples/examples/llama_multiprocess/main.rs
- Apache Ballista at Spice AI — https://spice.ai/blog/apache-ballista-at-spice-ai
- Gather-with-local-backward for in-batch-negative losses under data parallelism: the pattern behind sentence-transformers' cached/gathered MNRL and `torch.distributed.nn.all_gather` usage; here each rank computes the full loss, so no backward communication is needed.
