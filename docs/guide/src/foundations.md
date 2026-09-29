# Built on DataFusion

Jammi is a DataFusion engine. Everything it adds sits on an extension point
DataFusion and candle ship for that purpose: no
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
join. `InferenceExec` embeds, classifies or scores a relation's rows. The
optimizer, the memory pool and the materialization contract all see that
node. Everything below follows from that choice.

## What rides on DataFusion

| Jammi adds | DataFusion seam | Jammi type |
|---|---|---|
| Inference as a plan stage: rows numbered, costed, cut into forward chunks, admitted to a device and forwarded | `ExecutionPlan` | `InferenceExec`, `NumberedInputExec`, `RowCostExec`, `KeyCheckExec` (`jammi-datafusion`) |
| An inference fan-out the optimizer keeps | `PhysicalOptimizerRule` | `InferenceFanOut` |
| Durable result tables, and `CREATE TABLE … AS` that writes one | `UserDefinedLogicalNodeCore`, `ExtensionPlanner`, `ExecutionPlan` | `StoreStatementNode`, `MaterializationPlanner`, `StoreStatementExec` |
| Vector search, as-of joins and graph propagation as operators | `ExecutionPlan` | `VectorSearchExec`, `AsofJoinExec`, `InitialStateExec` / `HopFoldExec` / `ReadoutExec` |
| Versioned reads over append-only tables, and mutable companion tables | `TableProvider` | `MaskedTableProvider`, `MutableTableProvider` |
| SQL functions over model outputs | `TableFunctionImpl`, `ScalarUDFImpl`, `AggregateUDFImpl` | `annotate(…)`, `jammi_content_hash(…)`, `vector_mean` / `vector_sum` / `vector_max` |
| A memory bound shared fairly by the operators that are using it | `MemoryPool` | `ActiveSpillPool` |

`jammi-datafusion` depends on no part of the Jammi engine. It binds to a
model through one pair of traits, `ModelRuntime` and `BoundModel`. A
DataFusion user who wants a model stage in their own engine can take the
crate and bring their own model cache; Jammi's engine is one such consumer.

## What rides on candle

Model math runs on [candle](https://github.com/huggingface/candle), a Rust
tensor library. Jammi extends it through candle's own `CustomOp` seam:

- **`jammi-kernels`:** fused CUDA kernels (attention, layer norm, GELU,
  AdamW, LoRA residuals). Each has a CPU reference arm, with candle's
  eager composition as the fallback.
  It also opens every CUDA device, with cuBLAS held to the `f32` compute
  type candle asks for: whichever way a card's cuBLAS splits a `bf16`
  matmul, the partial sums are added in `f32` and rounded once.
- **`jammi-lora`:** low-rank adapters.
- **`jammi-encoders`:** the encoder towers.

No Python runs in the serving or training path.

## What this buys you

- **The planner sees the model.** A model's residency is admitted against its
  device's `[gpu] memory_limit` budget, and a forward against the device's
  forward slots; host-side operators share the session's memory pool. A
  forward the device refuses for memory is retried at half the rows, and
  fails the query only when a single row does not fit.
- **A remote model is one more device.** A model served at a declared
  endpoint runs through the same `InferenceExec`: its forwards are requests,
  admitted by the endpoint's `max_in_flight` the way a local model's are
  admitted by its device, and its declaration is the run a table records
  ([Use a Remote Model](./remote-models.md)).
- **Identical bytes at any fan-out.** The rows a model forwards together
  are decided once, by row cost, and carried as a chunk id the exchange
  hashes on. A plan fanned over one partition or sixteen forwards identical
  chunks and writes identical bytes ([The Cookbook → Fan-out inference](https://f-inverse.github.io/jammi-ai/cookbook/chapters/26-fanout/fanout.html)
  builds one table at one, two and four partitions and asserts the artifact
  digests equal).
- **Model outputs under a correctness contract.** Every result table
  records what produced it: the producing descriptor and the environment,
  including a typed record of each model's run (local weights with their
  content digest and precision, a remote endpoint's declaration, or an
  import). Definition hashes, verification and recompute apply to model
  outputs as they do to any other table
  ([The Materialization Contract](./materialization-contract.md); the book's
  recompute chapter, `cookbook/book/chapters/20-recompute/`, runs a
  recompute over unmoved inputs and asserts it byte-identical).
- **Training as a durable job.** A fine-tuning job is admitted, claimed
  under a lease and recorded like any other job; its training set is a
  result table under the same materialization contract, and the engine
  assembles its ranks as one gang
  ([Fine-Tuning](./fine-tuning.md)).
- **One binary, every topology.** Embedded, a single server, and a fleet of
  servers claiming jobs from one catalog are configuration, not forks
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
- **Fan-out:** an embedding plan fanned over four partitions costs 0.95× the
  single plan. A streamed training job costs 1.0007× the resident one.

That session also recorded ladders it couldn't judge: graph propagation,
graph sampling, the context predictor and the structure encoder were
INVALID. Those workloads carry no
claim until a committed verdict judges them. A verdict states the commit it
measured, so a number here holds for that commit, not necessarily for
today's head.
