# jammi-datafusion

Jammi's DataFusion extension: what the engine adds to DataFusion at
DataFusion's own seams — `ExecutionPlan` nodes, a `PhysicalOptimizerRule`
and the wire form that carries them to another process — over the
vocabulary every model stage shares (what a model computes, where it is
loaded from, the kind of device a plan runs on).

`inference` is the forward stage: a relation's rows are numbered and
chunked by a token budget once, below every exchange; each chunk is
prepared on the host, admitted against its device and forwarded; the
output carries the task's columns behind a common prefix; and the whole
plan is placeable on a Ballista executor. It binds to a model through one
trait pair, `ModelRuntime` and `BoundModel`, so a consumer brings its own
model cache, device admission and forward.

`training` is the training stage: a claimed training job — its
coordinates, never its spec — runs as one task on the process whose
executor holds the job's device, through a `TrainingRunner` the consumer
implements with its own claim transfer, training loop and publish, and
yields the job's outcome as one row. The node binds its runner at
construction, so a codec's decode binds the decoding process's own; a
process that runs no training binds `NoTrainingRunner`.

The crate depends on DataFusion, Arrow, `jammi-numerics` and `chrono`
alone.
