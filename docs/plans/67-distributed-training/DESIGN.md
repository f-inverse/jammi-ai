# DESIGN — distributed training on DataFusion (#500)

Companion to `README.md` (rulings) and `UNITS.md` (contracts). Every mechanism below names the
principle it derives from and the code it lands on. Citations are read against main at
`7561658e` (2026-09-10) and re-verified by the lead before use.

## 1. What MLlib did to Spark, and what that means here

The MLlib paper's win over Mahout was building on Spark core's data abstraction: RDD caching
made iteration cheap and Mahout on MapReduce lost on scheduling overhead. MLlib's optimizers
ran gradient steps as `map` + `treeAggregate`. Later measurement (Sparker, 2021) found
`treeAggregate` taking ~67% of end-to-end time with the driver as the bottleneck. Databricks'
fix was not a better aggregate; it was Project Hydrogen's barrier execution mode (SPARK-24374):
"in Spark a task in a stage doesn't depend on any other tasks in the same stage … in MPI all
workers start at the same time and pass messages around." Barrier mode adds all-or-nothing
scheduling, a barrier call, task host info and stage-wide restart. TorchDistributor uses it only
to launch ranks; NCCL does every collective.

So the data engine's durable contributions to training are the **data plane** and **gang
scheduling**; never the collective. jammi is to DataFusion what MLlib was to Spark core (a
library on the engine's data abstraction), and the gang unit is to Ballista what Project
Hydrogen was to the Spark scheduler (a co-scheduled unit the distributor lacks). Ballista 54.x
has no barrier mode, retries failed tasks individually (harmful for a gang: a retried rank
rejoins a dead communicator), and lists executor-failure handling on the adaptive path as a
gap; `datafusion-distributed` (contrib) has no scheduler and fails the whole query on any node
loss, which is exactly a gang's failure model. Neither is adopted as a hard dependency; the
Ballista unit proves the operators run unchanged under a real distributor.

One correction to the first review: LoRA adapters sit **inside** the tower, so an
encoder-adapters run has no frozen forward; every step is a full tower forward/backward on each
rank. The all-reduce payload is only the adapter gradients (rank-8 matrices, a few MB), which
makes data-parallel LoRA communication-cheap and makes the CPU collective viable well beyond a
test twin. The frozen-forward-as-distributed-stage applies to the **projection-head** target
(`crates/jammi-ai/src/fine_tune/target.rs:249`, a head on a frozen base) and to evaluation.

## 2. The training set is a producer

`ProducingDescriptor::TrainingSet { source, columns, task, format, order_rule: "full_tuple_v1" }`
(`crates/jammi-db/src/store/manifest.rs:305` gains the variant). Materialization runs the
source plan through the session, sorts by the full projected tuple (rows with identical tuples
are identical, so their mutual order is immaterial), and writes an immutable Parquet result
table of kind `TrainingSet` (`crates/jammi-db/src/catalog/result_repo.rs:27`) with the standard
`.materialization.json` attestation (`MaterializationManifest`, `manifest.rs:874`). Input
anchors are the source anchors; the definition hash folds descriptor + `MaterializationEnv`.
`GraphFineTune` materializes its sampled pairs the same way (the sampler is seeded and
deterministic, `crates/jammi-ai/src/fine_tune/graph_sampler.rs:374`), so both LoRA kinds
consume one table shape. Binary media columns are stored as they are today (blob columns).

The loader (`crates/jammi-ai/src/fine_tune/data.rs:200`, today `Vec<TrainingRow>`) becomes a
per-epoch `RecordBatch` stream over the table's row groups with a prefetch bound; each head's
constructor becomes a per-batch converter. Row count and per-row byte size are recorded in the
manifest so the trainer computes `batches_per_epoch` and the trailing-window scale
(`crates/jammi-ai/src/fine_tune/trainer.rs:2505-2514`) without a pass over the data.

**Partition rule v1** ("block-by-global-batch"): with per-rank batch B and world W, global batch
t is committed rows `[t·W·B, (t+1)·W·B)`; rank r reads `[t·W·B + r·B, t·W·B + (r+1)·B)`. Row
groups are sized to a multiple of W·B at materialization so a rank's slice never straddles a
row group. The trailing global batch may be short; ranks may then hold unequal counts, and the
collective reduces `(Σ grad, count)` so the step is exactly the global-batch mean. Sequence
bucketing (`batch_bucket.rs`) is applied per rank batch; padding is masked, so W-invariance of
the trajectory holds up to floating point, which is why that oracle is tolerance-based (§6).

The `TargetScaler` μ/σ (K3) is computed once over the whole materialized table at
materialization and stored in the manifest, so every rank de-standardizes identically and a
resumed attempt cannot see a different scaler.

## 3. A trained model is a producer

`ProducingDescriptor::FineTune` folds (K7 completeness table, audited per field in U3's test):

| field | why it moves the bytes |
|---|---|
| training-set definition hash + artifact digest + row count | the data and its order |
| base model identity (`ModelIdentity`: id, backend, precision, content digest, quantization) | the frozen weights |
| task, method, objective/loss, LoRA config (rank, alpha, dropout, target modules, layers) | the trainable graph |
| optimizer config (lr, schedule, warmup, weight decay, clip, grad-accum, epochs, per-rank batch) | the step |
| seed; dropout policy | the RNG draws (LoRA-A init, dropout masks) |
| target scaler policy + persisted μ/σ | K3 |
| backbone dtype; fused-kernel admission profile | bits per op |
| `world_size`, partition rule version, collective backend, reduction policy | the summation order |
| `MaterializationEnv` (engine version, device kind, model identities) | as for every producer |

The catalog name `jammi:fine-tuned:{job_id}` stays as the handle and re-claim idempotency key
(`crates/jammi-ai/src/fine_tune/worker.rs:1176-1184`). Migration 029 `model_materialization`
adds `models.definition_hash`, `models.input_anchors`, `models.manifest_path` (append-only, K5);
the manifest is written last into the artifact prefix, next to `manifest.json`, by the
coordinator before the finalize CAS. `CachePolicy::Use` probes by definition hash + anchors and
returns the existing model instead of training. The recompute pipeline's replay arm
(`crates/jammi-ai/src/pipeline/recompute.rs`, K1) for `FineTune` is **retrain**: equal-topology
training is reproducible by construction, unlike `External`.

## 4. The gang

**Roles.** The worker that claims the job (`claim_next_training_job`,
`crates/jammi-db/src/catalog/training_repo.rs:951`) is the coordinator and rank 0. It holds the
only lease and heartbeat (unchanged). It materializes or reuses the training set (§2), resolves
`W−1` peers from `[training] peers` (deployer-supplied addresses; the engine never places), mints
the NCCL unique id when the collective is `nccl`, and sends each peer a `RankAssignment`:

```
RankAssignment { job_id, tenant, rank, world_size, peers[rank -> address], collective,
                 nccl_id?, training_set { table_id, artifact_url, digest, row_count },
                 base_model, spec, partition_rule, resume_checkpoint_url?, attempt }
```

Peers execute `RunRank` (server-streaming `RankEvent`: heartbeat, step metrics, done, error).
Rank 0 runs its own rank in-process. Nothing about ranks touches the catalog.

**Collective.** One trait, four implementations, selected by configuration:

```
trait Collective { fn all_reduce_sum(&self, grads: &mut [Tensor], count: u64) -> Result<u64>;
                   fn broadcast(&self, t: &mut Tensor, root: u32) -> Result<()>;
                   fn barrier(&self) -> Result<()>; fn rank(&self) -> u32; fn world(&self) -> u32; }
```

`Noop` (W=1, the single-device path: same trainer, same code), `Local` (in-process ranks on N
devices of one host: rank-ordered reduce on rank 0's device, deterministic), `Peer` (across
processes over the `RunRank` bidirectional stream: each rank sends `(Σ grad, count)` in Arrow
IPC to rank 0, rank 0 sums in rank order and streams the result back; exact, deterministic,
adequate for LoRA payloads), `Nccl` (`candle-core/nccl`: `Comm::from_devices` for `Local`-shaped
single-process multi-GPU, `Comm::from_rank` with the coordinator-minted id across processes;
`NCCL_ALGO`/`NCCL_PROTO` pinned in the rank environment so the reduction order is fixed for a
given topology). `Nccl` is compiled under the existing `cuda` feature; the trainer is never
`cfg`-forked.

**Step.** Per rank: forward/backward on its slice → at each optimizer-step boundary
(`gradient_accumulation_steps` respected as today, `trainer.rs:2582`), `all_reduce_sum` over the
adapter `GradStore` → divide by the reduced count → `clip_and_step` (`optimizer.rs:612`). Every
rank holds identical weights after the step. Rank 0 alone writes the resume checkpoint at epoch
boundaries and publishes the artifact; the other ranks' checkpoint calls are no-ops.

**Failure.** Any `RankEvent::error`, any stream drop, or a rank silent for
`rank_timeout_secs` fails the attempt: the coordinator cancels every rank (stream close +
NCCL communicator abort), aborts the attempt (no publish, no finalize), and the job returns to
`queued` with `attempts + 1` through the existing `fail`/reclaim path. Coordinator death expires
the lease; reclaim proceeds as today. Either way the next attempt resumes from the job-level
resume checkpoint (`{tenant}/{job_id}/_resume/`, `crates/jammi-db/src/store/artifact.rs:300-323`).
No per-task retry exists anywhere in the gang path. This is the barrier-mode failure model.

**Device-plural session.** `[gpu] devices = [0,1,…]` makes the session hold one `GpuScheduler`
per device and a `ModelCache` keyed by `(ModelId, device)` (today keyed by `ModelId` alone,
`crates/jammi-ai/src/model/cache.rs:206-210`); `Local` ranks are threads pinned to devices.

## 5. The distributed frozen forward (head target) and the partition-aware operator

For `ProjectionHead` training the tower is frozen, so the features are an `Embedding` result
table over the training set — the existing embedding producer
(`crates/jammi-ai/src/pipeline/embedding.rs`), which today collects every batch before writing
(`embedding.rs:184-191`) and drives `InferenceExec` at partition 0 only
(`inference_exec.rs:144`). U6 makes `InferenceExec` inherit its input's partitioning, streams
batches straight into the `ResultSink`, and lets the coordinator fan partitions out to peers via
`GangService.FetchPartition` (server-streaming Arrow IPC, tenant-scoped) so each peer computes
a disjoint slice with the model loaded once per process. The resulting table must be
byte-identical to the single-process table (K4 shape). The head then trains on the features
with W=1 or a small gang, unchanged machinery.

## 6. Oracles

| Oracle | Kind | Where |
|---|---|---|
| **Refactor parity**: at W=1, the new loader + `Noop` collective produce adapter bytes identical to the base commit on the cookbook fixture | byte | cookbook 6.5 goldens; hermetic |
| **K4 (real)**: W=1 through the gang path (coordinator dispatching to itself as a peer) equals the in-process trainer, byte-for-byte | byte | server it-suite |
| **Equal-topology reproducibility**: two runs, same W, plan and hardware → identical bytes | byte | hermetic (`Local`, `Peer`); gpu-gang (`Nccl`, same box) |
| **W-invariance**: W ranks × B versus W=1 × W·B, identical loss per step within tolerance ε (fp reduction order) | tolerance | hermetic; gpu-gang |
| **Gang failure**: kill −9 a peer → job requeued, completed by a new gang from the checkpoint, no partial artifact; kill −9 the coordinator → lease reclaim, same outcome | property | distributed lane, chaos leg |
| **Cache reuse**: same spec on the same training-set digest with `CachePolicy::Use` → no second training | property | hermetic |
| **Distributor-agnosticism**: the same operators through a Ballista scheduler + 2 executors hosted by the jammi binary → identical bytes to the peer path | byte | U8, `ballista` feature |

## 7. Configuration and placement

```
[gpu]      device = 0 ; devices = [0, 1]           # plural is a superset of device
[training] world_size = 1 ; peers = [] ; rank_timeout_secs = 120 ; collective = "auto"  # auto|nccl|cpu
```
Per-job `world_size` lives in `TrainingCommon` (identity-relevant). Placement is the deployer's
runtime: Kubernetes runs the compute tier as a StatefulSet with a headless service (or an
indexed Job) so rank addresses are stable and startup is ordered; Compose lists services; Slurm
and Ray are placement options only. This is the #482 consequence, kept.

## 8. Non-goals (declared)

Sharded model or optimizer state (FSDP-like); elastic gangs; SGD or gradient exchange as
DataFusion operators or aggregates; `ContextPredictor` on a gang; a sixth pluggable backend.

## References

- MLlib: Machine Learning in Apache Spark, JMLR 17 (2016) — https://www.jmlr.org/papers/volume17/15-237/15-237.pdf
- Sparker: Efficient Reduction for More Scalable Machine Learning with Spark (ICPP 2021) — treeAggregate ≈ 67% of time — https://dl.acm.org/doi/fullHtml/10.1145/3472456.3472499
- SPARK-24374 SPIP: Barrier Execution Mode — https://issues.apache.org/jira/browse/SPARK-24374
- TorchDistributor — https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor
- Ballista 53.0.0 / 54.0.0 / 54.1.0 release notes — https://datafusion.apache.org/blog/output/2026/05/24/datafusion-ballista-53.0.0/ , …/2026/07/12/datafusion-ballista-54.0.0/ , …/2026/08/09/datafusion-ballista-54.1.0/
- Ballista architecture — https://datafusion.apache.org/ballista/contributors-guide/architecture.html
- Ballista extension hooks (`with_ballista_physical_extension_codec`, `ballista/core/src/extension.rs`) and task-level retry (`max task failure attempts`, `ballista/scheduler/src/state/task_manager.rs`) — read from source 2026-09-10
- datafusion-distributed — https://github.com/datafusion-contrib/datafusion-distributed
- cudarc `nccl` module (`Comm::from_devices`, `Comm::from_rank`, `all_reduce_in_place`) — https://docs.rs/cudarc/latest/cudarc/nccl/index.html ; candle-core 0.11 `nccl = ["cuda", "cudarc/nccl"]`
- candle `llama_multiprocess` (one process per rank, id via file, inference only) — https://github.com/huggingface/candle/blob/main/candle-examples/examples/llama_multiprocess/main.rs
