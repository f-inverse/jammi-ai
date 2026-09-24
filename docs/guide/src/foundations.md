# Built on DataFusion and Ballista

Jammi is a DataFusion engine that deploys as a Ballista cluster. Everything
it adds sits on an extension point those projects ship for that purpose: no
fork, no vendored copy, no patch carried against upstream. This page maps
what Jammi adds, the seam each addition uses, and what that buys you. The
[Design Philosophy](./philosophy.md#extending-third-party-libraries-at-their-seams)
states the rule this follows.

## The idea: a model run is a plan operator

The common way to put a model into a query engine is a user-defined
function: the engine hands a batch of rows to opaque code and gets a column
back. The engine can't see the model, so it can't place the model's work,
bound its memory, or say what produced its output.

In Jammi, a model run is a node of the physical plan, like a sort or a
join. `InferenceExec` embeds, classifies or scores a relation's rows, and
`TrainingExec` runs a claimed fine-tuning job. The optimizer, the scheduler
and the materialization contract all see these nodes. Everything below
follows from that choice.

## What rides on DataFusion

| Jammi adds | DataFusion seam | Jammi type |
|---|---|---|
| Inference as a plan stage: rows numbered, costed, cut into forward chunks, admitted to a device and forwarded | `ExecutionPlan` | `InferenceExec`, `NumberedInputExec`, `RowCostExec`, `KeyCheckExec` (`jammi-datafusion`) |
| An inference fan-out the optimizer keeps | `PhysicalOptimizerRule` | `InferenceFanOut` |
| Fine-tuning as a plan stage | `ExecutionPlan` | `TrainingExec`, bound to a `TrainingRunner` |
| Durable result tables, and `CREATE TABLE … AS` that writes one | `ExecutionPlan`, `UserDefinedLogicalNodeCore`, `ExtensionPlanner` | `ResultTableSinkExec`, `StoreStatementNode`, `MaterializationPlanner` |
| Vector search, as-of joins and graph propagation as operators | `ExecutionPlan` | `AnnSearchExec`, `AsofJoinExec`, `InitialStateExec` / `HopFoldExec` / `ReadoutExec` |
| Versioned reads over append-only tables, and mutable companion tables | `TableProvider` | `MaskedTableProvider`, `MutableTableProvider` |
| SQL functions over model outputs | `TableFunctionImpl`, `ScalarUDFImpl`, `AggregateUDFImpl` | `annotate(…)`, `jammi_content_hash(…)`, `vector_mean` / `vector_sum` / `vector_max` |
| A memory bound shared fairly by the operators that are using it | `MemoryPool` | `ActiveSpillPool` |

`jammi-datafusion` depends on no part of the Jammi engine. It binds to a
model through one pair of traits, `ModelRuntime` and `BoundModel`. A
DataFusion user who wants a model stage in their own engine can take the
crate and bring their own model cache; Jammi's engine is one such consumer.

## What rides on Ballista

| Jammi adds | Ballista seam | Jammi type |
|---|---|---|
| Jammi's operators crossing the scheduler/executor boundary, and every other node crossing through Ballista's own codec unchanged | `PhysicalExtensionCodec` | `JammiCodec` (`jammi-ballista`) |
| Placement by device kind: a stage runs only on an executor that registered the device its plan requires | `TaskDistributionPolicy::Custom` | `DevicePlacement` |
| A per-stage device check before a stage runs, refusing typed rather than running on the wrong device | `ExecutionEngine` (wrapping Ballista's default) | `JammiExecutionEngine` |
| Cluster and job state in the catalog, so every scheduler sees one fleet | `ClusterState`, `JobState` | `CatalogClusterState`, `CatalogJobState` |

The scheduler, executor and client roles run inside the same
`jammi-server` binary, chosen by the `[ballista]` configuration section.
Ballista's own retries are off (`task_max_failures = stage_max_failures =
0`): a task fault reaches Jammi's own attempt and reclaim accounting, so
there's one retry discipline, not two competing ones.

## What rides on candle

Model math runs on [candle](https://github.com/huggingface/candle), a Rust
tensor library. Jammi extends it through candle's own `CustomOp` seam:

- **`jammi-kernels`:** fused CUDA kernels (attention, layer norm, GELU,
  AdamW, LoRA residuals). Each has a CPU reference arm, with candle's
  eager composition as the fallback.
- **`jammi-lora`:** low-rank adapters.
- **`jammi-encoders`:** the encoder towers.

No Python runs in the serving or training path.

## What this buys you

- **The planner sees the model.** Device placement is a property of the
  plan (`DevicePlacement` reads the kind an `InferenceExec` or
  `TrainingExec` declares). A model's residency is admitted against its
  device's memory budget (`[gpu]`), and a forward against the device's
  forward slots; host-side operators share the session's memory pool. A
  forward the device refuses for memory is retried at half the rows, and
  fails the query only when a single row does not fit.
- **Identical bytes at any fan-out.** The rows a model forwards together
  are decided once, by row cost, and carried as a chunk id the exchange
  hashes on. A plan fanned over one partition or sixteen, in one process or
  across a Ballista cluster, forwards identical chunks and writes identical
  bytes.
- **Model outputs under a correctness contract.** Every result table
  records what produced it: the producing descriptor and the environment,
  including a typed record of each model's run (local weights with their
  content digest and precision, a remote endpoint's declaration, or an
  import). Definition hashes, verification and recompute apply to model
  outputs as they do to any other table
  ([The Materialization Contract](./materialization-contract.md); the book's
  recompute chapter, `cookbook/book/chapters/20-recompute/`, runs a
  recompute over unmoved inputs and asserts it byte-identical).
- **Training on the same plane as queries.** A fine-tuning job is placed,
  admitted and recorded by the same machinery as an embedding plan.
- **One binary, every topology.** Embedded, single server and Ballista
  cluster are configuration, not forks
  ([Reference Topologies](./reference-topologies.md)).

## What is measured, and what is not yet

Claims about speed, space and learning are made only from committed
verdicts of the parity ladder (`ci/scripts/perf/campaign.sh`). The ladder
compares the engine with a PyTorch twin built on the same box, under
stationarity and noise gates.

The session committed under
`ci/artifacts/parity-ladder-runs/2026-09-23-a100-parity/` (A100 80 GB,
engine commit `168c71dc`) established:

- **Learning:** over twelve seeds, the fine-tuning run's held-out loss is
  non-inferior to PyTorch's and equivalent within the derived margin.
- **Speed:** the fine-tuning run takes 0.606× PyTorch's wall time by
  medians; the training step alone takes 0.885×.
- **Space:** 0.516× the device memory and 0.805× the host memory of the same
  PyTorch run.
- **The plane:** an embedding plan fanned over four partitions costs 0.95× the
  single plan, and 1.61× on a Ballista executor. A streamed training job
  costs 1.0007× the resident one.

That session also recorded ladders it couldn't judge: graph propagation,
graph sampling, the context predictor and the structure encoder were
INVALID, and the placed training rung was refused. Those workloads carry no
claim until a committed verdict judges them. A verdict states the commit it
measured, so a number here holds for that commit, not necessarily for
today's head.
