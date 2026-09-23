# jammi-inference

Run a model over a DataFusion relation as a physical stage: the rows are
numbered and chunked by a token budget once, below every exchange; each
chunk is prepared on the host, admitted against its device and forwarded;
the output carries the task's columns behind a common prefix; and the
whole plan is placeable on a Ballista executor through the wire form this
crate provides.

The crate extends DataFusion at its own seams — `ExecutionPlan` nodes and
a `PhysicalOptimizerRule` — and binds to a model through one trait pair,
`ModelRuntime` and `BoundModel`, so a consumer brings its own model cache,
device admission and forward. It depends on no engine: only DataFusion,
Arrow and `jammi-numerics`.
