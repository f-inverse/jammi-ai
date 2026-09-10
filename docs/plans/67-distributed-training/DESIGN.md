# DESIGN — distributed training on DataFusion (#500), v2

Companion to `README.md` (rulings) and `UNITS.md` (contracts). Every mechanism names the
principle it derives from and the code it lands on. Citations are read against main at
`7561658e` (2026-09-10) and were re-verified by the lead and by two pressure-testers.

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
GPUs. The frozen-forward-as-distributed-stage applies to the **projection-head** target
(`crates/jammi-ai/src/fine_tune/target.rs:249`) and to evaluation.

## 2. The training set is a producer

`ProducingDescriptor::TrainingSet { source, columns, task, format, order_rule: "full_tuple_v1" }`
(`crates/jammi-db/src/store/manifest.rs:305` gains the variant). Materialization runs the
source plan through the session, sorts by the full projected tuple (identical tuples are
identical rows, so their mutual order is immaterial), and writes an immutable Parquet result
table of kind `TrainingSet` (`crates/jammi-db/src/catalog/result_repo.rs:27`) with the standard
attestation (`MaterializationManifest`, `manifest.rs:874`). The descriptor carries **no
topology and no split**: the table is shared by every job over the same source, columns, task
and format, whatever their world size, batch or validation fraction. Input anchors are the
source anchors. `GraphFineTune` materializes its seeded, deterministic sampled pairs
(`graph_sampler.rs:374`) the same way. Media blob columns are stored as today.

**Split.** The job's `validation_fraction` defines the train prefix exactly as today
(`data.rs:477-481`): `val_count = round(rows × fraction)`, train rows `[0, rows − val_count)`,
validation rows after. `batches_per_epoch` derives from `train_count` and the batch size (the
trailing-window scale at `trainer.rs:2505-2514` is unchanged in form).

**Loader.** `TrainingDataLoader` (`data.rs:200`, today `Vec<TrainingRow>`) becomes a per-epoch
`RecordBatch` stream over the table's row groups with a prefetch bound; each head's constructor
becomes a per-batch converter. A rank slice that straddles row groups reads the covering groups
and slices (a reader concern; no materialization alignment).

**Partition rule v1 ("block-by-global-batch")** over the train prefix: with per-rank batch B
and world W, global batch t is rows `[t·W·B, (t+1)·W·B)`; rank r reads
`[t·W·B + r·B, t·W·B + (r+1)·B)`. The union over ranks at step t is exactly the W=1 batch of
size W·B at step t. The trailing global batch may be short; ranks then hold unequal counts and
every rank knows every count from the rule. Sequence bucketing (`batch_bucket.rs`) runs per
rank batch; the W-invariance oracle pins the bucket rung (§6).

**`TargetScaler`** (K3): rank 0 streams the target column of the train prefix in committed
order into the same `from_targets` reduction (`regression_loss.rs:169`) — bit-identical to
today (`trainer.rs:824-836`) — and ships μ/σ in the `RankAssignment`; it persists for resume as
today.

**Whole-set arms.** Hard-negative mining (`trainer.rs:1120`, a mined loader rebuilt at each
refresh epoch from the model) and GradCache (`trainer.rs:1616-1626`, the whole train prefix as
one in-batch-negative batch) are structurally whole-set consumers. In this plan they run at
W=1 only: `world_size > 1` with `hard_negatives.refresh_every > 0` or `cached = true` is a
typed K2 refusal at submit time, and the residency bound exempts them (they stream the table in
but hold what they need). The gather primitive (§4) is what lifts this later.

## 3. A trained model is a producer

`ProducingDescriptor::FineTune` folds:

| field | why it moves the bytes |
|---|---|
| training-set definition hash + artifact digest + row count | the data and its order |
| **canonical serialization of the whole `TrainingSpec` variant** — `FineTuneConfig` (`crates/jammi-wire/src/fine_tune.rs:237-267`: LoRA rank/alpha/dropout, `use_rslora`, `rank_pattern`, `init_lora_weights`, lr, epochs, batch, `max_seq_length`, losses, `matryoshka_dims`, `quantile_levels`, `validation_fraction`, early stopping, `cached`, `hard_negatives`, …), `TrainingCommon`, method, task, seed | everything the trainer reads |
| base model identity (`ModelIdentity`: id, backend, precision, content digest, quantization) | the frozen weights |
| backbone dtype; fused-kernel admission profile | bits per op |
| `world_size`, per-rank batch, partition rule version, collective backend, reduction policy | the summation order and the batch layout |
| `MaterializationEnv` (engine version, device kind, model identities) | as for every producer |

U3's completeness test is an **exhaustive destructuring** of `FineTuneConfig`, `TrainingCommon`
and the descriptor (no `..`), so a new field fails compilation instead of escaping the hash;
U4b extends the same test with the topology fields when they appear.

The catalog name `jammi:fine-tuned:{job_id}` stays as the handle and re-claim idempotency key
(`worker.rs:1176-1184`). Migration 029 `model_materialization` adds nullable
`models.definition_hash`, `models.input_anchors`, `models.manifest_path` (append-only, K5;
nullable because `ContextPredictor` has no materialization; the probe never matches NULL). The
manifest is written last into the artifact prefix by the coordinator before the finalize CAS.
`CachePolicy::Use` probes by definition hash + anchors; on a hit the job completes by
registering **its own name** pointing at the reused prefix (two rows, one prefix); a prefix is
reaped only when no model row references it (reconcile attribution,
`crates/jammi-db/src/store/reconcile.rs:14-17`). The recompute replay arm
(`pipeline/recompute.rs`, K1) for `FineTune` is **retrain**.

## 4. The gang

**Roles.** The worker that claims the job (`claim_next_training_job`,
`training_repo.rs:951`) is the coordinator and rank 0; it holds the only lease and heartbeat.
It materializes or reuses the training set, computes the scaler, resolves `W−1` peers from
`[training] peers`, mints the NCCL id when the collective is `nccl`, and sends each peer:

```
RankAssignment { job_id, tenant, attempt, coordinator_worker_id, rank, world_size,
                 peers[rank -> address], collective, nccl_id?, training_set_table_id,
                 base_model_id, spec, partition_rule, scaler?, resume_from_checkpoint: bool }
```

No URL travels on the wire: the peer resolves the table and the base model by id through the
tenant-scoped catalog and derives storage URLs itself.

**Authorization.** Before running, a peer verifies through a tenant-scoped, read-only catalog
read that `job_id` is `running`, `claimed_by == coordinator_worker_id`, and the lease is live;
otherwise `RunRank` is refused with a typed status. Peers key running ranks by `(job_id, rank)`
and fence on `attempt`: a strictly greater attempt aborts the older; a lesser or equal one is
refused. `FetchPartition` takes a result-table id and a partition index, resolved the same way.
Mutual transport auth stays the deployer's runtime, as for every surface today.

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
`all_gather`s each representation column and the scores using the counts every rank derives
from the partition rule. Every rank computes the **identical global loss** over the gathered
batch through the existing loss functions (`dispatch_contrastive_loss`, `trainer.rs:4343-4356`,
`mnrl_loss`, classification/regression heads) — batch-coupled objectives keep exactly their
W=1 semantics. Backward runs through a gather whose backward keeps only the local slots (rank
r's own rows), so no gradient crosses the wire; at each optimizer-step boundary the adapter
`GradStore` is laid out in the canonical `trainable_vars` order with zeros for absent entries
(`optimizer.rs:600-611` documents absent entries as a real shape), `all_reduce_sum`med, then
`clip_and_step` (`optimizer.rs:612`). Gradient accumulation counts global batches. Every rank
holds identical weights after the step. Rank 0 alone writes the resume checkpoint at epoch
boundaries and publishes; other ranks' checkpoint calls are no-ops.

**Lockstep control flow.** The step boundary is the global batch index, never a rank-local
counter. Divergence (`loss.is_nan() || loss > 100`, `trainer.rs:2560-2567`), the 3-strikes
abort, early stopping and epoch exit are decided by `all_reduce_max_flags` at the same
boundary on every rank (validation loss is computed by rank 0 and the stop flag broadcast).
No rank can reach a collective a different number of times than its peers.

**Failure.** Any `RankEvent::error`, stream drop, or rank silent for `rank_timeout_secs` fails
the attempt: the coordinator cancels every rank (stream close + NCCL communicator abort),
aborts the attempt (no publish, no finalize), and the job returns to `queued` with
`attempts + 1` through the existing fail/reclaim path. Coordinator death expires the lease.
Either way the next attempt resumes from the job-level resume checkpoint
(`{tenant}/{job_id}/_resume/`, `artifact.rs:300-323`), which a zombie writer cannot regress
(the write is gated on the held lease, `trainer.rs:3601`). No per-task retry anywhere.

**Device-plural session.** `[gpu] devices = [..]` gives one `GpuScheduler` per device and a
`ModelCache` keyed by a `CacheKey { model_id, device, task: Option<_>, backend: Option<_> }`
shared with plan 65's rekey; `Local` ranks are threads pinned to devices.

## 5. The distributed frozen forward (head target) and the partition-aware operator

For `ProjectionHead` training the tower is frozen, so the features are an `Embedding` result
table over the training set — the existing embedding producer (`pipeline/embedding.rs`), which
today collects every batch before writing (`embedding.rs:184-191`) and drives `InferenceExec`
at partition 0 only (`inference_exec.rs:144`). U6 makes `InferenceExec` inherit its input's
partitioning, streams batches into the `ResultSink` in partition order, and lets the
coordinator fan partitions out to peers via `FetchPartition` so each peer computes a disjoint
slice with the model loaded once per process. The table must be byte-identical to the
single-process table (K4 shape).

## 6. Oracles

| Oracle | Kind | Where |
|---|---|---|
| **Refactor parity**: W=1 with the new loader, scaler-over-train-prefix and `Noop` produces adapter bytes identical to the base commit on every cookbook fine-tune fixture (`cookbook/book/artifacts/finetune_*/checksums.json`) | byte | cookbook 6.5; hermetic |
| **K4 (real)**: W=1 through the gang path (coordinator dispatching to itself) equals the in-process trainer | byte | server it-suite |
| **Equal-topology reproducibility**: two runs, same W and plan → identical bytes | byte on `Local`/`Peer` (hermetic); digest pair + per-step loss delta ≤ ε on GPU legs until S5 promotes | hermetic; gpu-gang |
| **W-invariance**: W × B versus W=1 × W·B, identical loss per step within ε, at `lora_dropout = 0` and a pinned bucket rung; ε measured on the leg that gates (CPU ε never inherited by GPU) | tolerance | hermetic; gpu-gang |
| **Gather exactness**: for CoSENT, AnglE and MNRL, the W=2 global loss at step t equals the W=1 loss on the same rows bit-for-bit on CPU (same expression, same inputs) | byte | hermetic |
| **Lockstep**: one rank's batch forced to diverge; one rank's batch yields no gradient for a Var; the gang completes | property | hermetic |
| **Gang failure**: kill −9 a peer → job requeued, completed by a new gang from the checkpoint, exactly one model, no orphan prefix promoted; kill −9 the coordinator → same via lease; split-brain: attempt N+1 dispatched while N is live on the peer → N aborted, N+1 runs | property | distributed lane |
| **Authorization**: `RunRank` for a job not running / not claimed by the caller / lease expired is refused | property | server it-suite |
| **Cache reuse**: same spec on the same training-set digest with `CachePolicy::Use` → no second training; two model rows, one prefix; reaping respects references | property | hermetic |
| **Distributor-agnosticism**: same operators through a Ballista scheduler + 2 executors hosted by the jammi binary → identical bytes to the peer path | byte | U8 |

## 7. Configuration and placement

```
[gpu]      device = 0 ; devices = [0, 1]
[training] world_size = 1 ; peers = [] ; rank_timeout_secs = 120 ; collective = "auto"   # auto|nccl|cpu
```
Per-job `world_size` lives in `TrainingCommon` (identity-relevant). Placement is the deployer's
runtime: Kubernetes runs the compute tier as a StatefulSet with a headless service (or an
indexed Job); Compose lists services; Slurm and Ray are placement options only (#482).

## 8. Non-goals (declared)

Sharded model or optimizer state (FSDP-like); elastic gangs; SGD or gradient exchange as
DataFusion operators or aggregates; `ContextPredictor` on a gang; hard-negative mining and
GradCache at `world_size > 1` (typed refusal in this plan); a sixth pluggable backend; a GPU
byte-equality claim before S5.

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
- Gather-with-local-backward for in-batch-negative losses under data parallelism: the pattern behind sentence-transformers' cached/gathered MNRL and `torch.distributed.nn.all_gather` usage; here each rank computes the full loss, so no backward communication is needed.
