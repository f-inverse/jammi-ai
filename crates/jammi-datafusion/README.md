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
model cache, device admission and forward. The crate depends on
DataFusion, Arrow and `jammi-numerics` alone.
