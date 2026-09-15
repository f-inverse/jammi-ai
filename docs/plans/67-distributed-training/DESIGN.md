# DESIGN — distributed training on DataFusion (#500), v4

Companion to `README.md` (rulings) and `UNITS.md` (contracts). Every mechanism names the
principle it derives from and the code it lands on. Every citation names a symbol, resolved at
this branch's head — a Rust item as `crates/<crate>/src/<path>.rs::Item` (fn, struct, enum,
const, mod or test name; `Type::method` for a nested item), a manifest key as
`Cargo.toml::[dependencies].<dep>`, a wire message as `<file>.proto::Message`/`::Enum`, or a
doc section as `<path>.md#heading` — never a `path:line` byte offset, which drifts as the crate
grows. A trailing `(~:NNN)` is a locus hint only, never load-bearing.

## 1. What MLlib did to Spark, and what that means here

The MLlib paper's win over Mahout was building on Spark core's data abstraction: RDD caching
made iteration cheap. MLlib's optimizers ran gradient steps as `map` + `treeAggregate`; later
measurement (Sparker, 2021) found `treeAggregate` taking ~67% of end-to-end time with the
driver as the bottleneck. Databricks' fix was Project Hydrogen's barrier execution mode
(SPARK-24374): all-or-nothing scheduling, a barrier call, task host info and stage-wide restart.
TorchDistributor uses it only to launch ranks; NCCL does every collective.

The data engine's durable contributions to training are therefore the **data plane** and
**gang scheduling**, never the collective. jammi is to DataFusion what MLlib was to Spark core;
the gang unit is to Ballista what Project Hydrogen was to the Spark scheduler. Ballista 54.x
has no barrier mode, retries failed tasks individually (harmful for a gang), and lists
executor-failure handling on the adaptive path as a gap; `datafusion-distributed` has no
scheduler and fails the whole query on node loss (a gang's failure model). Neither is a hard
dependency; the Ballista unit proves the operators run unchanged under a real distributor.

**Workload and payload.** LoRA adapters sit inside the tower, so an encoder-adapters run has no
frozen forward; each rank runs the full tower forward/backward on its slice. What crosses the
wire per optimizer step is (a) the gathered representations of the global batch — `W·B·d`
floats per representation column (anchor, positive, optional negative) plus the scores — and
(b) the adapter gradients (rank-8 matrices). At W=4, B=32, d=768, f32: about 0.4 MB per
column and single-digit MB for the adapter gradients. Both are small enough that the exact CPU
collective is viable far beyond a test twin; NCCL matters when the tensors already live on
GPUs. The **projection-head** target (`TrainingTarget::ProjectionHead`,
`crates/jammi-ai/src/fine_tune/target.rs`) runs the same per-batch pattern: `encode_texts`/
`encode_media` call `project_frozen_embedding` inline, per batch, to run the frozen tower
forward before the head's own LoRA layer projects it — never a separate stage or a materialized
table (§5) — and evaluation uses the identical call.

## 2. The training set is a producer

`ProducingDescriptor::TrainingSet { source, columns, task, format, order_rule: "full_tuple_v1" }`
(`crates/jammi-db/src/store/manifest.rs::ProducingDescriptor::TrainingSet` gains the variant). Materialization runs the
source plan through the session, sorts by the full projected tuple (identical tuples are
identical rows, so their mutual order is immaterial), and writes an immutable Parquet result
table of kind `TrainingSet` (`crates/jammi-db/src/catalog/result_repo.rs::ResultTableKind::TrainingSet`) with the standard
attestation (`crates/jammi-db/src/store/manifest.rs::MaterializationManifest`). The descriptor carries **no
topology and no split**, so the table is shared by every job over the same source, columns,
task and format ONLY when the input anchors are equal AND pinned — the engine's existing rule
(`crates/jammi-db/src/store/manifest.rs::AnchorKind::UnpinnedAtInstant`): a plain, unpinned
source anchors as `UnpinnedAtInstant` and the cache probe short-circuits any unpinned anchor, so it
is honestly always a miss. Whatever their world size, batch or validation fraction, two jobs over
the same PLAIN source still each materialize their own table; reuse needs a pinned source.
Input anchors are the source anchors. `GraphFineTune` does not go through this table: its seeded,
deterministic sampled pairs (`crates/jammi-ai/src/fine_tune/graph_sampler.rs::GraphSampler::sample`)
are sampled in memory and trained on directly; a graph training set's own table is
https://github.com/f-inverse/jammi-ai/issues/538. Media blob columns are stored as today.

**Split.** The job's `validation_fraction` defines the train prefix exactly as today
(`crates/jammi-ai/src/fine_tune/data.rs::TrainingDataLoader::split`, the `TextRows` arm):
`val_count = round(rows × fraction)`, train rows `[0, rows − val_count)`,
validation rows after. The tests-only `Precomputed` loader arm
(`crates/jammi-ai/src/fine_tune/data.rs::TrainingDataLoader::from_precomputed`; split by
batch count in `TrainingDataLoader::split`'s `Precomputed` arm) hands tensors straight to the trainer and stays outside the table
path, unchanged. Every step quantity is a function of the **global** batch:
`batches_per_epoch = ceil(train_count / (W·B))`; `global_step`, the LR horizon
(`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::run`, the horizon computed ahead of
`compute_lr`) and the trailing-window loss scale
(`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::process_batch_loss`) all
index by global batch, so W ranks take exactly the steps W=1 takes at batch W·B. U2b lands
the formula at W=1.

**Loader.** `TrainingDataLoader` (`crates/jammi-ai/src/fine_tune/data.rs::TrainingDataLoader`) reads the train prefix eagerly through
`read_back_sql` (`training_set_order_by` applied — the result-table `ListingTable`
(`crates/jammi-db/src/store/mod.rs::build_result_table_provider`) declares no file sort order (a plain
`ParquetReadOptions::default().to_listing_options(...)`), so `read_back_sql`'s `ORDER BY` plans a
pipeline-breaking `SortExec` at `target_partitions` ∈ {1, N}; **U2c** declares the sort order —
rendered from `training_set_order_by` via `ListingOptions::with_file_sort_order` — so DataFusion
elides that `SortExec`; a reader-class allow-list oracle enumerates every reader of
`sql_relation()`), converting
rows into `TrainingRow`s per format; the partition rule (below) slices this in-memory sequence
per rank. A residency-bounded per-rank stream over the table's row groups is NOT part of this
plan: U2b's own design carried it, a design fix round excised it (issue #544) after a
lease-held-across-a-carry-over deadlock, and it is rebuilt as its own unit, **U2c** — a
whole-prefix ORDERED stream per rank with a per-rank `rows_for_step` filter, never a stream
shared across ranks (the resulting W× read/decode amplification is accepted, stated, and
measured, never hidden) — scheduled before U4b binds a per-rank reader to it.

**Partition rule v1 ("block-by-global-batch")** over the train prefix: with per-rank batch B
and world W, global batch t is rows `[t·W·B, (t+1)·W·B)`; rank r reads
`[t·W·B + r·B, t·W·B + (r+1)·B)`. The union over ranks at step t is exactly the W=1 batch of
size W·B at step t. The trailing global batch is **kept** (today keeps and specially scales
it, so drop-last would change W=1 bytes); ranks then hold unequal counts, and when
`train_count mod (W·B) ≤ r·B` rank r holds **zero** rows: it encodes nothing and contributes a
0-row tensor to every gather of that step, with a zero in the counts vector every rank derives
from the rule. Sequence bucketing (`batch_bucket.rs`) runs per rank batch; the W-invariance
oracle pins the bucket rung (§6).

**`TargetScaler`** (K3): rank 0 collects the target column of the train prefix, in committed
order, into **one** `Vec<f32>` on the trainer's device and calls `from_targets` **once**
(`crates/jammi-ai/src/fine_tune/regression_loss.rs::TargetScaler::from_targets` is a two-pass whole-tensor reduction; f32 summation is
grouping-sensitive, so a chunked accumulation would move the low bits) — bit-identical to
today (`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::run`, the scaler-construction site). The eager loader holds no residency bound for the scaler to be
exempt from; once U2c's per-rank stream lands, the scaler's whole-prefix reduction will need
the same named exemption from ITS bound (a 4-bytes-per-row allowance is that unit's own
contract). μ/σ ship in the `RankAssignment` and persist for resume as today.

**Whole-set arms.** Hard-negative mining (`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::run`, the refresh-boundary check that rebuilds a mined loader at each
refresh epoch from the model) and GradCache (`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::run_gradcache_epoch`, the whole train prefix as
one in-batch-negative batch) are structurally whole-set consumers. In this plan they run at
W=1 only: `world_size > 1` with `hard_negatives.mine == true` or `cached == true` is a typed
K2 refusal at submit time (`mine` is the real gate, `crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::mining_eligible`; `refresh_every` defaults
to 1 and `== 0` is already refused when mining). They read the eager loader's whole train
prefix directly and need no residency accounting under this plan; once U2c's per-rank stream
exists under other arms, mining and GradCache stay outside it (their own named exemption, not
built here). The gather primitive (§4) is what lifts this later.

## 3. A trained model is a producer

`ProducingDescriptor::FineTune` folds:

| field | why it moves the bytes |
|---|---|
| training-set definition hash + artifact digest + row count | the data and its order |
| **canonical serialization of the whole `TrainingSpec` variant** — `FineTuneConfig` (`crates/jammi-wire/src/fine_tune.rs::FineTuneConfig`: LoRA rank/alpha/dropout, `use_rslora`, `rank_pattern`, `init_lora_weights`, lr, epochs, batch, `max_seq_length`, losses, `matryoshka_dims`, `quantile_levels`, `validation_fraction`, early stopping, `cached`, `hard_negatives`, …), `TrainingCommon`, method, task, seed | everything the trainer reads |
| base model identity (`ModelIdentity`: id, backend, precision, content digest, quantization) | the frozen weights |
| backbone dtype; fused-kernel admission profile (declared, UNCOVERED — see below) | bits per op |
| `world_size`, per-rank batch, partition rule version, collective backend, reduction policy | the summation order and the batch layout |
| `MaterializationEnv` (engine version, device kind, model identities) | as for every producer |

**Crate layering.** `jammi-db` depends on no jammi crate but `jammi-numerics`
(`crates/jammi-db/Cargo.toml::[dependencies].jammi-numerics`), and every existing variant holds primitives and db-local
types (`crates/jammi-db/src/store/manifest.rs::ProducingDescriptor`). `FineTuneConfig` is `jammi-wire`, `TrainingSpec`/`TrainingCommon`
are `jammi-ai` (`crates/jammi-ai/src/fine_tune/spec.rs::TrainingSpec`), `TrainingFormat` is `jammi-ai` (`crates/jammi-ai/src/fine_tune/data.rs::TrainingFormat`). So both new
variants carry an **opaque, versioned canonical encoding** — `spec_canonical: String`
(sorted-key canonical JSON) with `spec_schema_version: u32` — produced by `jammi-ai` from the
owning types, plus db-local primitives (`ModelTask`, ids, digests, the topology fields).
`TrainingSet.format` is likewise a canonical string. The **exhaustive destructuring**
completeness test (no `..`) lives in `jammi-ai` (and `jammi-wire` for `FineTuneConfig`), so a
new field fails compilation instead of escaping the hash; U4b extends it with the topology
fields.

The catalog name `jammi:fine-tuned:{job_id}` stays as the handle and re-claim idempotency key
(`crates/jammi-ai/src/fine_tune/worker.rs::JobWorker::train_fine_tune`; the id format is documented
again on `ModelRegistration::model_id`). The `model_materialization` migration (033)
adds nullable `models.definition_hash`, `models.input_anchors` (append-only, K5; nullable
because `ContextPredictor` has no materialization; the probe never matches NULL —
`find_models_by_definition`/`probe_model_by_definition` additionally restrict to
`artifact_path IS NOT NULL`, the SERVABLE set). No `models.manifest_path` column: the
sidecar's path is always the fixed name `materialization.json` under the model's artifact
prefix, never a recorded column. The manifest is written last into the artifact prefix by
the coordinator before the finalize CAS, and the two catalog columns are written only after
that CAS has already committed this attempt's `artifact_path` — no unfinalized row is ever a
cache-hit candidate.

`cache` lives on `TrainingSpec::FineTune` itself, not `TrainingCommon` — `TrainingSpec::
GraphFineTune` has no `cache` field at all, so `lora_common_from_proto` refuses `cache = Use`
for that kind, typed, at decode (the one place that can still see both the kind and the
requested value). A stray `cache` key found under `graph_fine_tune` in a persisted
`jobs.spec` row is dropped at deserialize rather than refused, since the type has nowhere to
put it; a hard error on unknown keys across the persisted-row format is a separate reshape
(<https://github.com/f-inverse/jammi-ai/issues/548>). Model-level cache reuse is not
implemented: `CachePolicy::Use` on the column-source `FineTune` kind is also refused, typed, at
submit (`InferenceSession::submit_fine_tune_spec_deduped`) rather than probing by definition
hash. The probe-by-definition design is filed as the property to implement — a hit
registers **its own name** pointing at the reused prefix (two rows, one prefix, reported on
the job's own result via `cache_outcome`), probing by definition hash alone (the anchors leg
removed as redundant: the training-set digest is already inside the hash)
(<https://github.com/f-inverse/jammi-ai/issues/562>). An unset `cache` field encodes
identically on both transports: the Rust client omits the field (rather than assigning the
enum's `Bypass` discriminant) so an explicit `Bypass` and an unset field put the SAME bytes on
the wire, matching the embedded Python encoding.

Deleting a model row is always allowed. Catalog referential integrity between two rows that
might name the same prefix only becomes relevant once model-level reuse ships
(<https://github.com/f-inverse/jammi-ai/issues/547>). The underlying bytes are reclaimed only
when no live `models` row, in any tenant, names the object's exact key or its immediate
containing directory as `artifact_path` — the guard rule every `models/` byte-delete this
design's reclaim paths perform runs through: `ResultStore::prefix_is_referenced` is an
admin-scoped whole-catalog scan of `models.artifact_path` that
`ResultStore::delete_unreferenced_prefix` consults before every `models/`-prefix byte-delete
(reconcile's own reap, the worker's abandon path, the worker's epoch-checkpoint sweep, and the
trainer's mid-run retention prune all reach it), refusing typed (`StorageError::Referenced
{ prefix, count }`) while any row still references the prefix; reconcile's attribution set is
built from the same admin-scoped scan, never the tenant-scoped `list_models`, and a prefix it
finds still referenced this way is reported (with its count) via
`ReconcileReport.referenced`/`referenced_count` (`crates/jammi-db/src/store/reconcile.rs`),
carried on the wire (`catalog.proto`, tags 15/16).

The fused-kernel admission profile the training loop actually resolves (fused vs. eager per
operator) is declared on `MaterializationEnv` but UNCOVERED: nothing writes a real value into
it, so it does not yet distinguish two runs that differ only in that outcome
(<https://github.com/f-inverse/jammi-ai/issues/546>).

The recompute replay arm (`pipeline/recompute.rs`, K1) for `FineTune` is **retrain**.

## 4. The gang

**Roles (on the jobs fleet).** A training-kind job (`fine_tune`, `graph_fine_tune`) is claimed
by a `JobWorker` through `claim_next` (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::claim_next`,
`FOR UPDATE SKIP LOCKED`, `attempts + 1`); that process is the coordinator and rank 0 and holds
the only lease (`heartbeat_job`, `crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::heartbeat_job`, driven by the lease keeper).
`context_predictor` is refused at `world_size > 1`. The coordinator materializes or reuses the
training set (with `job_attempt: None` — a shared producer output, never this attempt's
`partial_result`), computes the scaler, resolves `W−1` **members** via U5b-1a's
`list_gang_members(GangListing { kind, self_instance, window })` (`workers.kinds`
∋ kind, `instances.peer_addr` set — the column U5b-1a appends and DIST's placement consumes —
and `last_seen_at` fresh under `instance_liveness_margin()`) FILTERED, by the coordinator itself,
on U5b-1a-A2's root-identity predicate against `instances.result_root` (`list_gang_members`'s own
admission predicate does NOT consult the root at all — contract `feat_500-C-U5b-1a` §12, the
round-5 excision; root identity across spellings, and the predicate built on it, is U5b-1a-A2's
question, a precondition of this coordinator), agreeing with this coordinator's own root —
NECESSARY, never SUFFICIENT, for shared storage; sufficiency is the attestation VERIFY (§2's
whole-artifact sidecar / U5b-0's per-partition leaf inventory) — and, from U8b, `workers.devices`
sufficient), mints the NCCL id when the collective is `nccl`, and sends each member:

```
RankAssignment { job_id, attempt, coordinator_instance_id, rank, world_size,
                 peers[rank -> instance_id], collective, nccl_id?, training_set_table_id,
                 base_model_id, spec, partition_rule, scaler?, resume_from_checkpoint: bool }
```

No URL travels on the wire: the peer resolves the table and the base model by id through the
tenant-scoped catalog and derives storage URLs itself.

**A peer is a fleet worker with a busy slot.** `RunRank` takes the worker's single job slot
(`JobSlot`): the claim loop takes it **before** `claim_next`, **holds it across** the inline
`run_claimed_job_under` and **releases it before** the idle sleep
(`crates/jammi-ai/src/fine_tune/worker.rs::JobWorker::run_until`), so a
peer never claims while it runs a rank, never aborts a claim transaction (68 OPS D6), never
receives a rank while training its own job, and is reachable whenever idle. Handler order: same
`job_id` with a lesser attempt → abort that runner and take the slot; lesser-or-equal → refuse;
otherwise try-lock; busy → typed `Unavailable`. No new worker state. The full-roster read above
(`list_gang_members`) is the coordinator's OWN dispatch-time tool; a peer's inbound-`RunRank`
freshness check never calls it — it reads the coordinator's own `instances` row alone, through
U5a-1's `fresh_instance(coordinator_instance_id)`, one row by primary key.

**Authorization (invariant I-GANG: the job row is the capability).** The service is mounted on
the internal `[server] peer_bind` listener (68 DIST D7), never on the tenant-scoped public chain.
The peer reads the `jobs` row through a new db-owned verb `get_job_for_rank(job_id)` — by
primary key, no tenant predicate, never admin scope (`get_job` is tenant-filtered,
`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::get_job`; D7 forbids `with_admin_scope` on the peer path), reachable only from the
gang handler — verifies `status = 'running'`, `claimed_by = coordinator_instance_id` and a live
lease, then **derives the tenant from the row** (`jobs.tenant_id`) and pins every subsequent
catalog read to it. Nothing dialable travels on the wire: the assignment carries
`peers[rank → instance_id]`; each peer resolves addresses through U5b-1a's `peer_addr_of(instance_id,
window) -> Option<PeerAddr>` and refuses a rank naming any instance that resolves to `None` — not
a fresh member (the NCCL id, an opaque secret, is the only out-of-band value). `RunRank` sits in its own `GANG_LISTENER_ALLOWLIST` bucket in `tenant_isolation_oracle.rs` (its text
states the ground the shipped W=1 lattice actually has — no tenant value read on the path; the
derivation claim returns with U5a-2's cross-tenant-denial case, #566), unioned like D7's, with the public-listener `UNIMPLEMENTED` assertion, and its
`api_freeze_baseline.txt` lines land in the same commit. Peers fence on **`job_id`**: a `RunRank`
at attempt N aborts every local runner of that job with attempt < N; a lesser or equal attempt is
refused; `(job_id, rank)` is the runner's identity only. Mutual transport auth stays the
deployer's runtime, as for every surface today.

**Collective.** One trait, four implementations, selected by configuration:

```
trait Collective {
  fn all_gather(&self, local: &Tensor, counts: &[usize]) -> Result<Tensor>;   // backward: local slots only
  fn all_reduce_sum(&self, tensors: &mut [Tensor]) -> Result<()>;            // canonical trainable_vars order
  fn all_reduce_max_flags(&self, flags: u32) -> Result<u32>;
  fn broadcast(&self, t: &mut Tensor, root: u32) -> Result<()>;
  fn barrier(&self) -> Result<()>;
  fn rank(&self) -> u32; fn world(&self) -> u32;
}
```

`Noop` (W=1: gather is identity, reduce is identity), `Local` (in-process ranks on N devices of
one host: rank-ordered reduce on rank 0's device, deterministic), `Peer` (across processes over
the `RunRank` bidirectional stream: rank-ordered coordinator-reduce in Arrow IPC, exact),
`Nccl` (`candle-core/nccl`: `Comm::from_devices` single-process multi-GPU, `Comm::from_rank`
across processes; `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS` pinned in
the rank environment). `Nccl` compiles under the existing `cuda` feature; the trainer is never
`cfg`-forked.

**Step (the gather rule).** Per global batch t, each rank encodes its slice, then
`all_gather`s the step's tensors using the counts every rank derives from the partition rule.
**Invariant: the gather point is downstream of every trainable parameter; nothing trainable
consumes a gathered remote slot** — otherwise the summed gradient of that parameter is W× too
large. Gather points per `TrainingBatch` arm: contrastive / pairs / triplet gather the encoder
outputs (post-projection, `crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::encode_texts`) and the scores; **classification gathers the
logits from `classify()`** (the trainable head is applied inside the loss today,
`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::compute_loss` (the `Classification` arm), head at `TrainingLoop::classify`), never `embeddings`; regression gathers the head
output (`head_forward` already runs pre-loss, `TrainingLoop::head_forward`) and the targets; NER stays refused
as today (`TrainingLoop::compute_loss_per_example`, the `Ner` arm). Every rank then computes the **identical global loss** over the gathered
batch through the existing loss functions (`dispatch_contrastive_loss`, `crates/jammi-ai/src/fine_tune/trainer.rs::dispatch_contrastive_loss`,
`mnrl_loss`, `cross_entropy_loss`, the regression/quantile losses) — batch-coupled objectives
keep exactly their W=1 semantics, and each loss's own 1/n runs over the global n. Matryoshka
prefixes narrow dim 1 only (`crates/jammi-ai/src/fine_tune/trainer.rs::matryoshka_sum`) and are orthogonal to a dim-0 gather. Backward runs through a gather whose backward keeps only the local slots (rank
r's own rows), so no gradient crosses the wire; at each optimizer-step boundary the adapter
`GradStore` is laid out in the canonical `trainable_vars` order with zeros for absent entries
(`crates/jammi-ai/src/fine_tune/optimizer.rs::clip_and_step` documents absent entries as a real shape), `all_reduce_sum`med, then
`clip_and_step` (`crates/jammi-ai/src/fine_tune/optimizer.rs::clip_and_step`). Gradient accumulation counts global batches. Every rank
holds identical weights after the step. Rank 0 alone writes the resume checkpoint at epoch
boundaries and publishes; other ranks' checkpoint calls are no-ops — except that each rank's
dropout Philox position (`crates/jammi-ai/src/fine_tune/resume.rs::ResumeState::dropout_positions`, per process today) is gathered
to rank 0 at the epoch boundary and stored **per rank** in the bundle, and each rank's dropout
seed derives as `f(seed, rank)`; a resumed gang at equal topology therefore reproduces an
uninterrupted one, and W=1 keeps today's single-entry shape.

**Lockstep control flow.** The step boundary is the global batch index, never a rank-local
counter. Divergence (`loss.is_nan() || loss > 100`, `crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::process_batch_loss`), the 3-strikes
abort, early stopping and epoch exit are decided by `all_reduce_max_flags` at the same
boundary on every rank (validation loss is computed by rank 0 and the stop flag broadcast).
No rank can reach a collective a different number of times than its peers.

**Failure and release.** `fail_job` is terminal (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::fail_job`); the fleet's
only requeue path is the leave-`running`-for-reclaim arm
(`crates/jammi-ai/src/fine_tune/worker.rs::JobWorker::run_claimed_job_under`, the lease-lost arm) → reclaim
arm 1a (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::reclaim_expired_jobs_inner`, the "Arm 1a" block) → `attempts + 1` at the successor's claim (`Catalog::claim_next`). So any
`RankEvent::error`, stream drop, or rank silent for `rank_timeout_secs` fails the attempt like
this: the coordinator cancels every rank (stream close + NCCL communicator abort), aborts the
attempt (no publish, no finalize), and flips its hold's `lost` flag (`cancel` *is*
`hold.lost_flag()`, `crates/jammi-ai/src/fine_tune/worker.rs::JobWorker::run_claimed_job_under`) so its own run exits through that arm with no
terminal write; reclaim requeues the job within the remaining lease window (≤ `[lease]
duration_secs`). A rank ended by a DRAIN/RELEASE on its host (68 OPS) sends
`RankEvent::Released`; the coordinator first calls OPS's `release_job_lease` (`releases + 1`,
lease NULL; the CAS admits the holder) and then flips the same flag, so a rolling restart of the
peer tier costs zero net attempts (OPS D10). Coordinator death expires the lease. A live
same-named `building` training-set row left by a crashed coordinator is met with the `BackOff`
disposition and reclaimed through `claim_expired_building_table` after expiry (README r31). The per-attempt watchdog is the lease keeper's
shape — bounded by the attempt it belongs to, retiring only that attempt — and is allowed under
the actuator rule (`crates/jammi-ai/src/pipeline/recompute.rs`'s module doc, "the engine ships
the actuator; it never ships the control loop that pulls it", `Cascade::Downstream`; 68 DIST D5).
Either way the next attempt resumes from the job-level resume checkpoint
(`{tenant}/{job_id}/_resume/`, `crates/jammi-db/src/store/artifact.rs::ArtifactStore::put_resume_checkpoint`), which a zombie writer cannot regress
(the write is gated on the held lease, `crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::save_resume_checkpoint`). No per-task retry anywhere.

**Device-plural session.** `[gpu] devices = [..]` gives one `GpuScheduler` per device and a
`ModelCache` keyed by a `CacheKey { model_id, device, task: Option<_>, backend: Option<_> }`
shared with plan 65's rekey — `None` is a distinct key value, never a wildcard — applied to
both the entries map and the single-flight `in_flight` map (`crates/jammi-ai/src/model/cache.rs::CacheInner`); `Local` ranks
are threads pinned to devices.

## 5. The distributed frozen forward (head target)

For `ProjectionHead` training the tower is frozen, but the frozen forward is not a producer
table: `encode_texts`/`encode_media` (`trainer.rs`) call `project_frozen_embedding` inline, per
training batch, to run the frozen tower and hand the pooled output to the head's own LoRA layer
— nothing under `fine_tune/` drives `InferenceExec` or reads an `Embedding` result table for this
path. At `world_size > 1` each rank runs its own slice's frozen forward inside the normal step,
and the head's output is a gather point like any other target's (§4's gather rule); no partition
set exists to fan out over, so there is no peer-to-peer fetch of a shared table. The gang has one
output artifact, written once by rank 0 — the lease holder — at the finalize CAS (§3; the resume
checkpoint is likewise lease-holder-only, `crates/jammi-db/src/store/artifact.rs::ArtifactStore::put_resume_checkpoint`), never
assembled from separate per-peer outputs. Peers exchange data only through the collective (§4's `Peer`
implementation over the `RunRank` stream); there is no second peer-to-peer surface. A
partition-aware inference operator that streams `InferenceExec` across partitions and fans a
table out to peers is a distinct, unscheduled piece of work — GitHub issue #540 — because nothing
in this design produces a table for it to partition.

## 6. Oracles

| Oracle | Kind | Where |
|---|---|---|
| **Refactor parity**: W=1 with the new loader, scaler-over-train-prefix and `Noop` produces adapter bytes identical to the base commit on every cookbook fine-tune fixture (`cookbook/book/artifacts/finetune_*/checksums.json`) | byte | cookbook 6.5; hermetic |
| **K4 (real)**: W=2 over the wire (`Peer`, two processes) equals W=2 in-process (`Local`), byte-for-byte; rank 0 is always in-process, so W=1 never crosses the wire and is covered by the `Noop` parity row | byte | distributed lane |
| **Equal-topology reproducibility**: two runs, same W and plan → identical bytes; also across a resume (kill at epoch k, resume, compare to uninterrupted) | byte on `Local`/`Peer` (hermetic); on GPU legs at world size 2 the digest pair is ASSERTED on a pass verdict — S5 measured candle 0.11 LoRA-shaped forward/backward/SGD byte-identical across two A100s with no env pins at world 2 — and the per-step loss delta is compared to an ε pre-registered per leg before the first gating run (or the max delta over ≥ 3 same-seed baseline runs on that box); at world size ≥ 3 the NCCL collective-algorithm pin set is unproven (reduction is commutative at world 2), so the digest pair there stays recorded, not asserted, until the cluster leg characterizes it | hermetic; gpu-gang |
| **W-invariance**: W × B versus W=1 × W·B, identical loss per step within ε, at `lora_dropout = 0` and a pinned bucket rung; ε measured on the leg that gates (CPU ε never inherited by GPU) | tolerance | hermetic; gpu-gang |
| **Gather exactness**: for CoSENT, AnglE, MNRL, **classification** and **quantile regression**, the W=2 global loss and the summed adapter gradient at step t equal the W=1 loss and gradient on the same rows bit-for-bit on CPU, on a fixture whose `train_count` is not a multiple of W·B (a zero-row rank occurs) | byte | hermetic |
| **Lockstep**: one rank's batch forced to diverge; one rank's batch yields no gradient for a Var; the gang completes | property | hermetic |
| **Gang failure**: kill −9 a peer → job requeued, completed by a new gang from the checkpoint, exactly one model, no orphan prefix promoted; kill −9 the coordinator → same via lease; split-brain: attempt N+1 dispatched while N is live on the peer → N aborted, N+1 runs | property | distributed lane |
| **Authorization**: `RunRank` for a job not running / not claimed by the caller / lease expired is refused | property | server it-suite |
| **Cache reuse refused**: `CachePolicy::Use` on a fine-tune is refused on every durable submit edge (reuse: https://github.com/f-inverse/jammi-ai/issues/562); two model rows may share one prefix; reaping respects references | property | hermetic |
| **Distributor-agnosticism**: an embedding job through a Ballista scheduler + 2 executors hosted by the jammi binary → identical bytes to the same query's single-executor plan; a gang job through the scheduler → identical bytes to U5b's peer-based run | byte | U8 |

## 7. Configuration and placement

```
[gpu]        device = 0 ; devices = [0, 1]
[worker]     enabled = true ; kinds = "all" ; local_ranks = 1 ; rank_timeout_secs = 120 ; collective = "auto"   # auto|nccl|cpu
[distributed] max_world_size = 1   # widest Peer gang any coordinator on this deployment may accept (67 U5b-1b-ii)
[server]     peer_bind = "..." ; peer_advertise = "..."          # peer_bind: 68 DIST-1; peer_advertise + instances.peer_addr/result_root: 67 U5b-1a; members are catalog rows, not a list
[ballista]   scheduler_bind = "..." ; executor = { scheduler_address = "...", work_dir = "..." }   # U8a
```
`[worker] local_ranks` (renamed from `world_size` in U4b S8) is how many ranks THIS HOST places
on its own `[gpu] devices` for a job it runs entirely in-process — orthogonal to
`[distributed] max_world_size`, which bounds a `Peer` gang across FLEET MEMBERS; the two knobs
load independently with no cross-check. Per-job `world_size` lives in `TrainingCommon`
(identity-relevant; `#[serde(default)]` = 1) — a third, unrelated concept, checked against
`[distributed] max_world_size` at submit.
Placement is the deployer's runtime: Kubernetes runs the compute tier as a StatefulSet with a
headless service (or an indexed Job) with `nvidia.com/gpu: N`; Compose lists services; Slurm
and Ray are placement options only (#482; owned by U9b after 68 K and OPS merge).

## 8. Non-goals (declared)

Sharded model or optimizer state (FSDP-like); elastic gangs; SGD or gradient exchange as
DataFusion operators or aggregates; `ContextPredictor` on a gang; hard-negative mining and
GradCache at `world_size > 1` (typed refusal in this plan); a sixth pluggable backend; a GPU
byte-equality claim before S5.

## 9. The Ballista extension (U8a, U8b)

**Discipline.** `jammi-kernels` extends candle at the seam candle exposes (`CustomOp1/2/3`), keeps
one call path, vendors verbatim at a pinned version only where no seam exists
(`third_party/flash-attention`, `VENDORED.md`), and believes nothing before its oracles pass.
`jammi-ballista` follows the same discipline; the crate is not a leaf (it encodes jammi
operators and hosts the model cache and catalog) — it sits between `jammi-ai`/`jammi-db` and
`jammi-server`, publishable and lockstep, with no cargo feature.

**Seams used (Ballista 54.1, read from source 2026-09-10).**

| Gap 68 named | Seam | What jammi installs |
|---|---|---|
| operators cross the wire | `SchedulerConfig.override_{logical,physical}_codec`, `ExecutorProcessConfig.override_*_codec` | `JammiCodec`: `InferenceExec`, `AnnSearchExec`, `AsofJoinExec`, `GangExec` ↔ the U5 descriptor messages |
| no executor-side state across plans | `ExecutorProcessConfig.override_execution_engine: Option<Arc<dyn ExecutionEngine>>`; `create_query_stage_exec(job, stage, task, partitions, plan, work_dir, config)` rewrites `ShuffleReaderExec` nodes and wraps the writer | `JammiExecutionEngine`: model cache held across plans, device pinned to `[gpu] devices`. Shuffle stays Ballista's local `work_dir` in v1; this seam is where an object-store shuffle would go once a spike proves the cross-executor read (D2's condition 3 stands) |
| cluster state in memory only | `ClusterState` + `JobState` traits; `BallistaCluster::new(Arc<dyn ClusterState>, Arc<dyn JobState>)`; `start_server(cluster, addr, config)` | U8a: Ballista's in-memory state. U8b: `CatalogClusterState`/`CatalogJobState` in `jammi-ballista` over jammi-db's public backend, on distributor-neutral tables from the `compute_cluster_state` migration (executor registrations, heartbeats, job graphs) — executor registrations and heartbeats survive a scheduler restart from a custom `ClusterState` (S6 probe 6); an in-flight job does not (`ExecutionGraphBox` has no serialisation in 54.1, `JobState::try_acquire_job` is never called), so recovery is a re-submit of the stored plan as a new Ballista job id through jammi's own `attempts`/reclaim semantics, not Ballista-side job survival. Two schedulers over one store serve one cluster only for sequential jobs (S6 probe 7); concurrent jobs hang on whichever scheduler loses the slot race with no public path to wake it — active/standby, a second upstream contribution is owed for active/active |
| no accelerator dimension | `ClusterState::bind_schedulable_tasks(distribution, active_jobs, executors) -> Vec<BoundTask>`; `TaskDistributionPolicy::Custom(Arc<dyn DistributionPolicy>)` | `DevicePlacement`: executor id ↔ `workers.devices`; a task is GPU-bound iff its stage plan (from `active_jobs`' execution graph, decoded through `JammiCodec`) contains a `GangExec` or an `InferenceExec` whose descriptor names a CUDA device; such a task binds only to a device-bearing executor. The Rust `ExecutorSpecification { task_slots: u32 }` (not `vcores`) has no accelerator attribute slot; the proto side's `oneof resource { TaskSlots(u32) }` is extensible, so the accelerator dimension is an upstream variant PR, not a schema break — the one upstream PR 67 owes |
| task retry rejoins a dead gang | `SchedulerConfig.task_max_failures`, `stage_max_failures` (global) | both 0: retries are the jobs table's. Holds for jammi operators by error classification (panic → `Internal`, any operator error → `ExecutionError`, both non-retryable, S6 probe 4) rather than by the knob; the knobs still close Ballista's own I/O-retry and fetch-failure arms |
| scheduler control loop | `expire_dead_executors` starts in `init()` | membership liveness (the class of `reclaim_expired_jobs`/`prune_instances`) **and** a task reset with no knob to stop it: the same loop posts `ExecutorLost`, which resets the lost executor's running tasks and re-fails its completed tasks as `ResultLost` without consulting `task_max_failures` (S6 probe 5) — `DevicePlacement` needs a bind-time guard that refuses to re-bind an already-launched `(job_id, stage_id, partition)` (README r42) |
| push launching, transport | `TaskLauncher`, `override_create_grpc_client_endpoint`, `use_tls` | as needed; not in v1 |

**Roles.** Listener-shaped knobs: a replica hosts a scheduler iff `[ballista] scheduler_bind` is
set, an executor iff `[ballista] executor.scheduler_address` is set; both may be set on one
process; unset = single node. Same class as `peer_bind` and `health_listen`.

**Gang under Ballista.** One task: `GangExec` (single partition) whose `execute` runs the U5b
coordinator; `DevicePlacement` puts it on a device-bearing executor; the ranks are fleet members
reached over `peer_bind` as in U5b. Bytes equal U5b's (K4 shape). A gang stage kind in Ballista
stays a future upstream option, not a dependency.

**Publishing.** `ci/scripts/publish_crates.sh`'s `PUBLISH_ORDER` array enumerates publishable crates by name in
topological order; `jammi-ballista` is inserted before `jammi-server` in the same commit (a
`v*` tag would otherwise half-publish). `check_dep_direction.py` encodes no layering and is
not touched.

**Oracles.** Codec round-trip for every operator; an embedding job via `submit_physical_plan`
across two executors byte-identical to the same query's single-executor plan; a W=2 gang job through the scheduler
byte-identical to U5b's and never task-retried; killing an executor mid-gang fails the job and
requeues it through jammi's lease path — holds only with U8b's bind-time re-launch guard in
place (README r42; without it, Ballista re-launches the gang task on its own, S6 probe 5); U8b:
scheduler restart keeps registered executors and re-submits an in-flight job as a new Ballista
job id through jammi's own reclaim (README r43's job-survival correction); a GPU stage never
binds to a device-less executor.

**Spice's fork, for the record.** Multi-active HA via object-store state, object-store shuffle,
mTLS, bidirectional control streams, catalog/UDF sync, their own shuffle format; one binary with
`--role scheduler`; batch only; no GPU; upstreaming planned, none landed. jammi's version keeps
HA state at the `ClusterState` seam on the catalog and forks nothing.

## References

- MLlib: Machine Learning in Apache Spark, JMLR 17 (2016) — https://www.jmlr.org/papers/volume17/15-237/15-237.pdf
- Sparker (ICPP 2021) — treeAggregate ≈ 67% of time — https://dl.acm.org/doi/fullHtml/10.1145/3472456.3472499
- SPARK-24374 SPIP: Barrier Execution Mode — https://issues.apache.org/jira/browse/SPARK-24374
- TorchDistributor — https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor
- Ballista 53.0.0 / 54.0.0 / 54.1.0 — https://datafusion.apache.org/blog/output/2026/05/24/datafusion-ballista-53.0.0/ , …/2026/07/12/datafusion-ballista-54.0.0/ , …/2026/08/09/datafusion-ballista-54.1.0/
- Ballista architecture — https://datafusion.apache.org/ballista/contributors-guide/architecture.html ; extension hooks in `ballista/core/src/extension.rs`; task-level retry in `ballista/scheduler/src/state/task_manager.rs` (read 2026-09-10)
- datafusion-distributed — https://github.com/datafusion-contrib/datafusion-distributed
- cudarc `nccl` — https://docs.rs/cudarc/latest/cudarc/nccl/index.html ; candle-core 0.11 `nccl = ["cuda", "cudarc/nccl"]`
- candle `llama_multiprocess` — https://github.com/huggingface/candle/blob/main/candle-examples/examples/llama_multiprocess/main.rs
- Ballista 54.1 seams (read from source 2026-09-10): `ballista/scheduler/src/cluster/mod.rs` (`ClusterState`, `JobState`, `DistributionPolicy`), `scheduler/src/config.rs` (`TaskDistributionPolicy::Custom`, `task_max_failures`), `scheduler/src/scheduler_process.rs` (`start_server(cluster, …)`), `scheduler/src/scheduler_server/mod.rs` (`new_with_task_launcher`, `expire_dead_executors`), `executor/src/executor_process.rs` (`override_execution_engine`), `executor/src/execution_engine.rs` (`ExecutionEngine`, reader rewrite), `core/src/serde/scheduler/mod.rs` (`ExecutorSpecification`)
- Apache Ballista at Spice AI — https://spice.ai/blog/apache-ballista-at-spice-ai
- Gather-with-local-backward for in-batch-negative losses under data parallelism: the pattern behind sentence-transformers' cached/gathered MNRL and `torch.distributed.nn.all_gather` usage; here each rank computes the full loss, so no backward communication is needed.
