//! Graph feature propagation — the decoupled-GNN forward pass (SGC / APPNP),
//! a deterministic data-plane operation, no autograd and no architecture.
//!
//! # What it computes
//!
//! The forward pass of a graph convolution is, mathematically, repeatedly
//! replacing each node's embedding with an aggregate of its neighbours'
//! embeddings — `Âᵏ·X`. SGC showed the per-layer nonlinearities are removable:
//! precompute the propagated features, then learn a simple head (the *decoupled*
//! GNN). This module is the **propagate** half — it takes an embedding table and
//! a graph and emits a new embedding table where each row is its `k`-hop
//! neighbourhood aggregate. The **learn** half (fine-tune a head on the
//! propagated features) is the existing fine-tune path; the recommended order is
//! propagate-then-fine-tune, never two independent smoothing passes.
//!
//! # The recurrence
//!
//! With **self-loops** `Ã = A + I` and the symmetric normalisation
//! `Â = D̃^{-1/2}(A+I)D̃^{-1/2}` over the augmented degrees `d̃ = deg + 1`, the
//! default ([`PropagationWeighting::DegreeNormalized`]) iterates the APPNP
//! recurrence with an `α`-teleport restart:
//!
//! ```text
//! X⁽ᵏ⁾ = (1−α)·Â·X⁽ᵏ⁻¹⁾ + α·X⁽⁰⁾
//! ```
//!
//! The self-loop is load-bearing twice over: it removes the `−1` eigenmode that
//! makes a plain `D^{-1/2}AD^{-1/2}` oscillate, and it makes an **isolated node
//! propagate to its own `X⁽⁰⁾`** for free (its only neighbour is itself). The
//! `α`-restart is the oversmoothing fix: each hop re-anchors every node to its
//! original embedding, so deep propagation does not collapse the representations
//! into one low-rank subspace (APPNP). `α` defaults to `0.1`, hops to `2`, and
//! the hop count is capped at [`DEFAULT_HOP_CAP`].
//!
//! The graph a propagation runs over is the subgraph the edge relation induces
//! on the embedding table's keys: an edge naming a key the table does not hold
//! is no edge, for the degrees as much as for the sum.
//!
//! # One propagation, two verbs
//!
//! A propagation is four choices — where `X⁽⁰⁾` comes from, the operator, the
//! depth, and how the per-hop blocks are read out ([`readout`]).
//! [`InferenceSession::propagate_embeddings`] takes `X⁽⁰⁾` from an embedding
//! table; [`InferenceSession::generate_structure_embeddings`](crate::pipeline::graph_structure)
//! generates it from the graph ([`seed`]) for nodes that have no content to
//! embed. Both build the same plan.
//!
//! # How a hop runs
//!
//! Every weighting is one operator with a different coefficient rule: a hop
//! joins the node state to the adjacency `(g, n, w)` on `n`, adds each node's
//! own row as its self-loop, and folds each group's terms in neighbour order
//! ([`hop::HopFoldExec`]):
//!
//! | weighting | term factor | group factor |
//! |---|---|---|
//! | [`Uniform`](PropagationWeighting::Uniform) | `1` | `1/count` — the mean, `D̃⁻¹Ã` |
//! | [`DegreeNormalized`](PropagationWeighting::DegreeNormalized) | `1/√d̃_n` | `1/√d̃_g` — `D̃^{-1/2}ÃD̃^{-1/2}` |
//! | [`EdgeSimilarity`](PropagationWeighting::EdgeSimilarity) | `max(weight, 0)`, self-loop `1` | `1/Σw` |
//!
//! The augmented degrees are exact integer counts computed once, for the
//! initial state, and carried on every state row; a `neighbor_graph` cosine
//! similarity lives in `[−1, 1]`, and a negative one is anti-signal that
//! clamps to zero rather than subtracting, the self-loop's weight of one
//! keeping every `Σw` positive.
//!
//! The whole propagation is one physical plan ([`plan`]) — initial state, the
//! hops, the readout — written through the same embedding sink every
//! embedding producer uses. Its joins, shuffles and sorts are stock operators
//! that hold pool reservations and spill, so the graph is bounded by the
//! spill disk and `[engine] memory_limit` is honoured; and the sink places
//! the plan on the compute plane when one can hold it.
//!
//! # Determinism
//!
//! Every fold, teleport, and readout runs in `f64` with a single final `f32`
//! cast, and each group's fold runs over one total order — its neighbours'
//! keys, bytewise — in one pass by one accumulator, which the hop operator
//! requires of its input rather than hopes for ([`hop`] says why an aggregate
//! function cannot promise it). The output is byte-identical whatever
//! `target_partitions` is, whatever order the edge relation's rows arrive in,
//! and whether or not a hop spilled; the rows are written in key order, so
//! the stored object is too.
//!
//! # Homophily
//!
//! Smoothing helps only when neighbours share signal. On a *heterophilous*
//! graph (neighbours tend to differ) propagation aggregates harmful signal and
//! is beaten by a structure-ignoring baseline; the homophily diagnostic
//! ([`InferenceSession::homophily_by_edge_type`]) measures this first, and the
//! answer there is learned attention. This module transports adjacency; it never
//! judges what an edge means.
//!
//! # References
//! - Wu et al. 2019, *Simplifying Graph Convolutional Networks (SGC)*:
//!   <https://arxiv.org/abs/1902.07153>
//! - Gasteiger et al. 2019, *Predict then Propagate (APPNP)*:
//!   <https://arxiv.org/abs/1810.05997>
//! - Xu et al. 2018, *Jumping Knowledge Networks*:
//!   <https://arxiv.org/abs/1806.03536>
//! - Zhu et al. 2020, *Beyond Homophily in GNNs*:
//!   <https://arxiv.org/abs/2006.11468>

pub mod hop;
pub mod plan;
pub mod readout;
pub mod seed;
pub mod state;

use std::sync::Arc;

use datafusion::prelude::DataFrame;

use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::error::{JammiError, Result};
use jammi_db::session::QueryContext;
use jammi_db::store::manifest::{InputAnchor, MaterializationEnv, ProducingDescriptor};
use jammi_db::store::{CacheOutcome, CachePolicy, SinkKind};

use crate::pipeline::graph_neighbourhood::{EdgeDirection, EdgeSourceRef, DEFAULT_HOP_CAP};
use crate::session::InferenceSession;

use plan::{
    adjacency_order, adjacency_relation, emit_plan, hop_plan, EdgeRead, Emit, FeatureSource,
    HopInput, HopPlanSpec,
};
use readout::BlockReadout;

/// Default number of propagation hops. Two hops is the homophily sweet spot;
/// beyond ~3 hops aggregation is over-smoothing (the `α`-restart mitigates it
/// but does not license unbounded depth).
pub const DEFAULT_PROPAGATE_HOPS: usize = 2;

/// Default APPNP teleport probability. Each hop re-mixes `α` of every node's
/// original `X⁽⁰⁾` back in, anchoring it against over-smoothing.
pub const DEFAULT_TELEPORT_ALPHA: f64 = 0.1;

/// How neighbour contributions are weighted when a hop aggregates them.
///
/// The default [`DegreeNormalized`](PropagationWeighting::DegreeNormalized) is
/// the symmetric `Â = D̃^{-1/2}(A+I)D̃^{-1/2}` of SGC/APPNP, paired with the
/// `α`-teleport restart — the PageRank-decay form that keeps propagation
/// anchored against over-smoothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PropagationWeighting {
    /// Random-walk normalisation `D̃^{-1}Ã`: each node's new vector is the plain
    /// mean of itself and its neighbours.
    Uniform,
    /// Symmetric normalisation `D̃^{-1/2}(A+I)D̃^{-1/2}` (the default). Combined
    /// with the `α`-teleport this is the APPNP/PageRank-decay propagation.
    #[default]
    DegreeNormalized,
    /// Edge-weighted mean `Σ(w·x)/Σw` over the neighbourhood, where `w` is the
    /// declared edge weight clamped to `max(weight, 0)` ("fixed attention" from
    /// a `neighbor_graph` similarity edge) and the self-loop weighs `1`. An
    /// absent weight is a present, full-strength edge (`1`).
    EdgeSimilarity,
}

/// What the propagation emits — how the per-hop blocks `[X⁽⁰⁾, …, X⁽ᴷ⁾]` are
/// read out into one row. Every variant is one point of a single family (a
/// weighted, optionally normalised, summed-or-concatenated readout of the
/// blocks; see [`readout`]), folded by one operator.
///
/// A typed mode rather than a flag, because the readout decides the output's
/// dimensionality and the vector space it lives in — a same-space hazard a
/// `bool` would hide.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum PropagationOutput {
    /// Emit only the final `X⁽ᴷ⁾` — a `d`-dimensional embedding table in the
    /// input's vector space.
    #[default]
    Final,
    /// Concatenate `[X⁽⁰⁾ ‖ X⁽¹⁾ ‖ … ‖ X⁽ᴷ⁾]`, each per-hop block L2-normalised
    /// before concat so the raw block's larger norm does not dominate cosine
    /// search (Jumping Knowledge). The output is `(K+1)·d`-dimensional and
    /// indexes in **its own space** — do not search it against `d`-dim vectors.
    JumpingKnowledge,
    /// The weighted sum `Σₖ wₖ · X⁽ᵏ⁾/‖X⁽ᵏ⁾‖` of the L2-normalised blocks — a
    /// `d`-dimensional table mixing every diffusion scale it gives weight to.
    /// `weights[k]` weighs block `k`, so there are exactly `hops + 1` of them
    /// (`weights[0]` weighs `X⁽⁰⁾` itself); a zero weight leaves its block
    /// unread. Indexes in its own space, like the concat.
    WeightedSum {
        /// One finite weight per block `0..=hops`, not all zero.
        weights: Vec<f64>,
    },
}

/// Which declared edge relation to propagate over and how the walk reads it. The
/// edge source and direction reuse the shared graph-neighbourhood config types.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropagateRequest {
    /// The source whose embedding table holds `X⁽⁰⁾`.
    pub source_id: String,
    /// The specific embedding table to propagate, or `None` to resolve the
    /// source's default embedding table.
    pub embedding_table: Option<String>,
    /// The edge relation defining the graph.
    pub edge_source: EdgeSourceRef,
    /// How the direction of an edge is read when building the neighbourhood.
    pub direction: EdgeDirection,
    /// Number of hops. Clamped to `[1, hop_cap]`.
    pub hops: usize,
    /// The hard depth cap `hops` is clamped to.
    pub hop_cap: usize,
    /// Neighbour-contribution weighting.
    pub weighting: PropagationWeighting,
    /// APPNP teleport probability (the share of `X⁽⁰⁾` re-mixed each hop), in
    /// `[0, 1]`; `0.0` disables the restart (plain SGC smoothing).
    pub alpha: f64,
    /// How the per-hop blocks are read out.
    pub output: PropagationOutput,
}

impl PropagateRequest {
    /// A propagation with the over-smoothing-safe defaults: the symmetric
    /// degree-normalised `Â` with an `α=0.1` teleport restart, two hops capped at
    /// [`DEFAULT_HOP_CAP`], out-edges, and final-block output.
    pub fn new(source_id: impl Into<String>, edge_source: EdgeSourceRef) -> Self {
        Self {
            source_id: source_id.into(),
            embedding_table: None,
            edge_source,
            direction: EdgeDirection::Out,
            hops: DEFAULT_PROPAGATE_HOPS,
            hop_cap: DEFAULT_HOP_CAP,
            weighting: PropagationWeighting::DegreeNormalized,
            alpha: DEFAULT_TELEPORT_ALPHA,
            output: PropagationOutput::Final,
        }
    }

    /// Builder: set the embedding table to propagate (default: the source's
    /// resolved embedding table).
    pub fn with_embedding_table(mut self, table: impl Into<String>) -> Self {
        self.embedding_table = Some(table.into());
        self
    }

    /// Builder: set how an edge's direction is read.
    pub fn with_direction(mut self, direction: EdgeDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Builder: set the hop count (clamped to `[1, hop_cap]` at run time).
    pub fn with_hops(mut self, hops: usize) -> Self {
        self.hops = hops;
        self
    }

    /// Builder: set the neighbour weighting.
    pub fn with_weighting(mut self, weighting: PropagationWeighting) -> Self {
        self.weighting = weighting;
        self
    }

    /// Builder: set the APPNP teleport probability.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Builder: set how the per-hop blocks are read out.
    pub fn with_output(mut self, output: PropagationOutput) -> Self {
        self.output = output;
        self
    }

    /// The depth actually run: `hops` clamped to `[1, hop_cap]`.
    pub fn effective_hops(&self) -> usize {
        self.hops.clamp(1, self.hop_cap.max(1))
    }
}

/// The model-id provenance recorded on a propagated embedding table. Short and
/// fixed (the source embedding table is recorded separately in `derived_from`),
/// so the generated table name stays a sane length.
const PROPAGATE_MODEL_ID: &str = "graph_propagate";

/// Refuse a teleport probability outside `[0, 1]` — `α·X⁽⁰⁾ + (1−α)·ÂX` is a
/// convex mix, and a NaN would poison every row.
pub(crate) fn check_alpha(alpha: f64) -> Result<()> {
    if (0.0..=1.0).contains(&alpha) {
        Ok(())
    } else {
        Err(JammiError::Config(format!(
            "a propagation's teleport probability must lie in [0, 1], got {alpha}"
        )))
    }
}

/// The model-id provenance recorded on a propagation's adjacency snapshot.
const ADJACENCY_MODEL_ID: &str = "graph_adjacency";

/// The model-id provenance recorded on a hop's state.
const STATE_MODEL_ID: &str = "graph_state";

/// A relation a propagation wrote for its own reading — its adjacency
/// snapshot, a hop's state — as the `building` table that holds it and the
/// relation that reads it. See [`InferenceSession::write_working_table`]
/// for the lifecycle.
struct WorkingTable {
    /// `Some` until the table is reclaimed.
    building: Option<jammi_db::store::BuildingTable>,
    relation: DataFrame,
}

impl WorkingTable {
    /// Abort the table: the row ends `failed` and its bytes are deleted. A
    /// delete that fails is left for `reconcile`, which reaps a failed row's
    /// objects by its ordinary rule — never a reason to fail a propagation
    /// that has already landed, or to mask the error of one that has not.
    async fn abort(building: jammi_db::store::BuildingTable) {
        let table = building.table_name().to_string();
        if let Err(e) = building.abort().await {
            tracing::warn!(
                table,
                error = %e,
                "graph propagation: a working table's bytes were not all reclaimed; left for \
                 reconcile"
            );
        }
    }

    /// Reclaim the table now, and wait for it.
    async fn reclaim(mut self) {
        if let Some(building) = self.building.take() {
            Self::abort(building).await;
        }
    }
}

/// A propagation dropped mid-flight — cancelled — reclaims its working
/// tables the way one that ended does: the same abort, run on the runtime
/// that was driving it. With no runtime left the handle's own drop lets the
/// lease lapse, and the sweep reclaims the row as it does a vanished
/// process's.
impl Drop for WorkingTable {
    fn drop(&mut self) {
        let Some(building) = self.building.take() else {
            return;
        };
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(Self::abort(building));
        }
    }
}

/// The operator and depth of one propagation — what the two propagating verbs
/// share, beside the features and the table they differ in.
pub(crate) struct PropagationShape<'a> {
    pub edge_source: &'a EdgeSourceRef,
    pub direction: EdgeDirection,
    pub weighting: PropagationWeighting,
    pub alpha: f64,
    pub hops: usize,
    pub readout: BlockReadout,
    /// The block width `d`.
    pub dimensions: usize,
}

/// The table a propagation lands as, and the contract it is recorded under.
pub(crate) struct PropagationTable<'a> {
    pub source_id: &'a str,
    pub model_id: &'a str,
    pub derived_from: Option<&'a str>,
    pub key_column: Option<&'a str>,
    pub descriptor: &'a ProducingDescriptor,
    pub env: &'a MaterializationEnv,
    pub inputs: Vec<InputAnchor>,
}

impl InferenceSession {
    /// Propagate an embedding table's features over a declared graph — the
    /// thin [`Self::run_now`] wrapper: submits a
    /// [`crate::jobs::ComputeSpec::Propagate`] and returns the terminal
    /// [`ResultTableRecord`] + [`CacheOutcome`] [`Self::run_now`] produced, so
    /// a direct call and a queued-and-claimed `propagate` job of the same spec
    /// run identical code (`Self::propagate_embeddings_materialize`, through
    /// [`crate::jobs::execute_compute`]).
    pub async fn propagate_embeddings(
        self: &Arc<Self>,
        request: &PropagateRequest,
        cache: CachePolicy,
    ) -> Result<(ResultTableRecord, CacheOutcome)> {
        let spec = crate::jobs::ComputeSpec::Propagate {
            request: request.clone(),
            cache,
        };
        self.run_table_spec("propagate_embeddings", spec).await
    }

    /// Run a compute spec whose result is a table through [`Self::run_now`],
    /// and read the table's record back.
    pub(crate) async fn run_table_spec(
        self: &Arc<Self>,
        verb: &'static str,
        spec: crate::jobs::ComputeSpec,
    ) -> Result<(ResultTableRecord, CacheOutcome)> {
        match self.run_now(spec).await? {
            crate::jobs::JobResult::Table {
                table,
                cache_outcome,
            } => {
                let record = self
                    .catalog()
                    .get_result_table(&table)
                    .await?
                    .ok_or_else(|| {
                        JammiError::Catalog(format!(
                            "{verb}: run_now's own table '{table}' vanished before it could be \
                             read back"
                        ))
                    })?;
                Ok((record, cache_outcome))
            }
            crate::jobs::JobResult::Model { .. } => Err(JammiError::Other(format!(
                "{verb}: run_now returned a training JobResult for a compute spec"
            ))),
        }
    }

    /// `propagate_embeddings`'s actual materializer — see
    /// [`InferenceSession::infer_materialize`]'s doc for the `job_attempt`
    /// convention every `*_materialize` method shares.
    ///
    /// Pins the source's embedding table, and plans
    /// [`PropagateRequest::effective_hops`] hops of the configured weighting
    /// over the tenant-scoped edge relation ([`plan`]). The output is a normal
    /// `kind=Model` embedding table with a sidecar index, `derived_from` the
    /// source table.
    pub(crate) async fn propagate_embeddings_materialize(
        self: &Arc<Self>,
        request: &PropagateRequest,
        cache: CachePolicy,
        job_attempt: Option<jammi_db::catalog::result_repo::JobAttempt<'_>>,
    ) -> Result<(ResultTableRecord, CacheOutcome)> {
        check_alpha(request.alpha)?;
        let hops = request.effective_hops();
        let readout = BlockReadout::lower(&request.output, hops)?;

        let table = self
            .catalog()
            .resolve_embedding_table(&request.source_id, request.embedding_table.as_deref())
            .await?;
        // ONE resolution of the source table's current version: its anchor
        // (`inputs` below) and every row the plan reads both derive from this
        // single pin, so a version publish racing this materialization can
        // never straddle the two.
        let pin = self.result_store().pin_current_version(table).await?;
        let table = pin.record();
        let dimensions = table
            .dimensions()
            .ok_or_else(|| {
                JammiError::Other(format!(
                    "propagate_embeddings: embedding table '{}' carries no dimensions",
                    table.table_name
                ))
            })?
            .get();

        // The descriptor and both input anchors (the source embedding table
        // and the edge relation) are resolvable before anything is read, so
        // the cache probe runs ahead of the plan.
        let descriptor = ProducingDescriptor::GraphPropagation {
            source_table: table.table_name.clone(),
            edge_source: request.edge_source.to_binding(),
            kernel_id: PROPAGATE_MODEL_ID.to_string(),
            direction: propagation_direction(request.direction),
            hops,
            alpha_bits: request.alpha.to_bits(),
            weighting: propagation_weighting(request.weighting),
            output: propagation_output(&request.output),
            dimensions: readout.out_dim(dimensions),
        };
        let env = MaterializationEnv::new(self.compute_device(), Vec::new());
        let inputs = vec![
            pin.input_anchor(),
            self.edge_source_anchor(&request.edge_source).await?,
        ];
        if let Some(reused) = self.probe_cache(cache, &descriptor, &env, &inputs).await? {
            return Ok(reused);
        }

        // Reads through `pinned_provider` — the SAME resolution the anchor was
        // computed from, never a second, independent read of `current_version`.
        let ctx = self
            .context()
            .out_of_core(readout.state_row_bytes(dimensions));
        let provider = self.result_store().pinned_provider(&ctx, &pin).await?;
        let features = ctx.read_table(provider).map_err(JammiError::from)?;

        let record = self
            .materialize_propagation(
                &ctx,
                PropagationShape {
                    edge_source: &request.edge_source,
                    direction: request.direction,
                    weighting: request.weighting,
                    alpha: request.alpha,
                    hops,
                    readout,
                    dimensions,
                },
                FeatureSource::Table(Box::new(features)),
                PropagationTable {
                    source_id: &request.source_id,
                    model_id: PROPAGATE_MODEL_ID,
                    derived_from: Some(table.table_name.as_str()),
                    // Every output row is keyed by a `_row_id` read verbatim
                    // off the source table, so the output's keys came from
                    // exactly the origin column the source records — the
                    // provenance is inherited, never re-asserted.
                    key_column: table.key_column.as_deref(),
                    descriptor: &descriptor,
                    env: &env,
                    inputs,
                },
                job_attempt,
            )
            .await?;
        Ok((record, CacheOutcome::Computed))
    }

    /// The top-of-producer cache probe: under [`CachePolicy::Use`], the ready
    /// table an exact prior run of `descriptor` over `inputs` left.
    pub(crate) async fn probe_cache(
        &self,
        cache: CachePolicy,
        descriptor: &ProducingDescriptor,
        env: &MaterializationEnv,
        inputs: &[InputAnchor],
    ) -> Result<Option<(ResultTableRecord, CacheOutcome)>> {
        if cache != CachePolicy::Use {
            return Ok(None);
        }
        let definition =
            jammi_db::store::manifest::MaterializationManifest::definition_of(descriptor, env)
                .map_err(jammi_db::store::manifest_to_jammi)?;
        Ok(self
            .result_store()
            .probe_cache_record(&definition, inputs)
            .await?
            .map(|reused| {
                let outcome =
                    CacheOutcome::Reused(jammi_db::store::ReusedArtifact::Table(reused.name()));
                (reused, outcome)
            }))
    }

    /// Plan `shape` over `features` and land it as `table` — the one funnel
    /// both propagating verbs materialize through.
    ///
    /// The adjacency is snapshotted first ([`Self::snapshot_adjacency`]) and
    /// every stage reads the snapshot, so the propagation runs over one graph
    /// however the edge source moves meanwhile. The snapshot, and every
    /// stage's state, is reclaimed when the propagation ends, whichever way
    /// it ends.
    pub(crate) async fn materialize_propagation(
        self: &Arc<Self>,
        ctx: &QueryContext,
        shape: PropagationShape<'_>,
        features: FeatureSource,
        table: PropagationTable<'_>,
        job_attempt: Option<jammi_db::catalog::result_repo::JobAttempt<'_>>,
    ) -> Result<ResultTableRecord> {
        let (edges, weighted) = self.edge_scan(ctx, shape.edge_source).await?;
        let relation = adjacency_relation(
            EdgeRead {
                edges,
                weighted,
                direction: shape.direction,
                weighting: shape.weighting,
            },
            &features,
        )?;
        let snapshot = self
            .snapshot_adjacency(ctx, table.source_id, relation)
            .await?;
        #[cfg(feature = "test-hooks")]
        crate::jobs::compute_test_hooks::maybe_park(
            table.source_id,
            crate::jobs::compute_test_hooks::ParkPoint::AfterAdjacencySnapshot,
        )
        .await;

        let landed = self
            .land_propagation(ctx, &snapshot, shape, features, table, job_attempt)
            .await;
        snapshot.reclaim().await;
        landed
    }

    /// Write `plan`'s rows as a working table of kind
    /// [`ResultTableKind::Working`] and hand back the handle that holds it,
    /// with the relation that reads it (in `order`, when the rows were
    /// committed in one).
    ///
    /// The table is a `building` row this process holds under its lease and
    /// never promotes. That is its whole lifecycle: it is aborted — the row
    /// failed, its bytes deleted — when its reader is done with it, whether
    /// the propagation landed, failed, or was dropped mid-flight
    /// ([`WorkingTable`]'s `reclaim` and `Drop` are the one abort); and a
    /// propagation whose process is gone stops renewing the lease, after
    /// which the recovery sweep reclaims the row as it does any dead
    /// writer's. It is written through the sink, so it lands in the shared
    /// store wherever the compute plane runs the write, and a placed reader
    /// finds it there.
    async fn write_working_table(
        self: &Arc<Self>,
        ctx: &QueryContext,
        source_id: &str,
        model_id: &str,
        plan: Arc<dyn datafusion::physical_plan::ExecutionPlan>,
        order: Option<Vec<datafusion::logical_expr::SortExpr>>,
    ) -> Result<WorkingTable> {
        let schema = plan.schema();
        let mut building = self
            .result_store()
            .create_table(
                source_id,
                jammi_db::ModelTask::TextEmbedding,
                ResultTableKind::Working,
                None,
                model_id,
                None,
                None,
                None,
                None,
            )
            .await?;
        self.result_store()
            .write_result_table(&mut building, SinkKind::Rows, plan, ctx.task_ctx())
            .await?;
        let provider = self
            .result_store()
            .building_provider(ctx, &building, schema, order)
            .await?;
        let relation = ctx.read_table(provider).map_err(JammiError::from)?;
        Ok(WorkingTable {
            building: Some(building),
            relation,
        })
    }

    /// Snapshot `relation` — an [`adjacency_relation`] — as the working
    /// table every stage of the propagation reads.
    async fn snapshot_adjacency(
        self: &Arc<Self>,
        ctx: &QueryContext,
        source_id: &str,
        relation: DataFrame,
    ) -> Result<WorkingTable> {
        let plan = relation
            .create_physical_plan()
            .await
            .map_err(|e| JammiError::Other(format!("graph propagation: adjacency plan: {e}")))?;
        self.write_working_table(
            ctx,
            source_id,
            ADJACENCY_MODEL_ID,
            plan,
            Some(adjacency_order()),
        )
        .await
    }

    /// Run the hops over `snapshot` one stage at a time ([`hop_plan`]) —
    /// each stage's state a working table the next reads, reclaimed once
    /// read — and land the last through the embedding sink as `table`: a
    /// `kind=Model` embedding table written where the compute plane says and
    /// finished under the caller's contract. Refuses a propagation that has
    /// no node to write.
    async fn land_propagation(
        self: &Arc<Self>,
        ctx: &QueryContext,
        snapshot: &WorkingTable,
        shape: PropagationShape<'_>,
        features: FeatureSource,
        table: PropagationTable<'_>,
        job_attempt: Option<jammi_db::catalog::result_repo::JobAttempt<'_>>,
    ) -> Result<ResultTableRecord> {
        let stage = |block: Option<usize>| HopPlanSpec {
            adjacency: snapshot.relation.clone(),
            weighting: shape.weighting,
            alpha: shape.alpha,
            readout: shape.readout.clone(),
            block,
        };
        // Stages 1..K−1 write state; the first reads the features. What is
        // left at the end is the last stage's plan, read out inline.
        let mut input = HopInput::Features(features);
        let mut held: Option<WorkingTable> = None;
        for block in 1..shape.hops {
            let state = hop_plan(ctx, input, &stage(Some(block))).await?;
            let written = self
                .write_working_table(ctx, table.source_id, STATE_MODEL_ID, state, None)
                .await?;
            if let Some(previous) = held.replace(written) {
                previous.reclaim().await;
            }
            input = HopInput::State(Box::new(
                held.as_ref()
                    .map(|w| w.relation.clone())
                    .expect("the stage just written"),
            ));
        }
        let last = hop_plan(ctx, input, &stage((shape.hops >= 1).then_some(shape.hops))).await?;
        let out_dim = shape.readout.out_dim(shape.dimensions);
        let plan = emit_plan(
            last,
            shape.readout,
            Emit {
                dimensions: shape.dimensions,
                source_id: table.source_id,
                model_id: table.model_id,
            },
        )?;

        // Every `?` between here and `finish` unwinds through the handle's
        // Drop (a best-effort `building -> failed` CAS).
        let mut building = self
            .result_store()
            .create_table(
                table.source_id,
                jammi_db::ModelTask::TextEmbedding,
                ResultTableKind::Model,
                table.derived_from,
                table.model_id,
                Some(out_dim as i32),
                table.key_column,
                None,
                job_attempt,
            )
            .await?;
        let embedding = &self.inner_config().embedding;
        let summary = self
            .result_store()
            .write_result_table(
                &mut building,
                SinkKind::Embeddings {
                    dimensions: out_dim,
                    ann: embedding.ann,
                    checkpoint_interval: embedding.checkpoint_interval,
                },
                plan,
                ctx.task_ctx(),
            )
            .await?;
        if let Some(previous) = held.take() {
            previous.reclaim().await;
        }
        if summary.rows == 0 {
            return Err(JammiError::Config(format!(
                "{}: the graph has no node to embed — the edge relation and the node set it is \
                 read against share no key",
                table.model_id
            )));
        }
        building
            .finish(
                self.context(),
                summary.rows as usize,
                jammi_db::store::manifest::Materialization::new(
                    table.descriptor,
                    table.env,
                    table.inputs,
                ),
            )
            .await
    }

    /// Resolve the input anchor for the edge relation a propagation reads — the
    /// second input alongside the embedding table. A `NeighborGraph` edge source
    /// is an immutable result table, pinned by its content digest
    /// (`ResultDigest`); a `Registered` external source has no version surface in
    /// open-core, so it is anchored as `UnpinnedAtInstant` — honest about the
    /// reproducibility gap rather than fabricating a pin.
    ///
    /// Resolves through [`jammi_db::store::ResultStore::pin_current_version`]
    /// (a `NeighborGraph` table is excluded from embedding refresh, so it never
    /// carries a `current_version` and takes the unversioned, hash-the-Parquet
    /// arm), so the anchor has no public shape a caller could get without also
    /// being able to get the matching content.
    pub(crate) async fn edge_source_anchor(
        self: &Arc<Self>,
        edge_source: &EdgeSourceRef,
    ) -> Result<InputAnchor> {
        match edge_source {
            EdgeSourceRef::NeighborGraph { table_name } => {
                let record = self
                    .catalog()
                    .get_result_table(table_name)
                    .await?
                    .ok_or_else(|| {
                        JammiError::Catalog(format!(
                            "propagate: edge relation '{table_name}' not found in the catalog"
                        ))
                    })?;
                Ok(self
                    .result_store()
                    .pin_current_version(record)
                    .await?
                    .input_anchor())
            }
            EdgeSourceRef::Registered { source_id, .. } => Ok(InputAnchor::unpinned_at_instant(
                source_id.clone(),
                chrono::Utc::now().to_rfc3339(),
            )),
        }
    }

    /// The SQL relation of the registered source `source_id`, once it is
    /// known to hold every one of `columns` — a column it lacks is the typed
    /// [`JammiError::Schema`], naming it, rather than a planner error from
    /// wherever the column is first read.
    pub(crate) async fn require_source_columns(
        &self,
        ctx: &QueryContext,
        source_id: &str,
        columns: impl IntoIterator<Item = &String>,
    ) -> Result<String> {
        let table = self.find_table_name(source_id).await?;
        let relation = jammi_db::sql::source_relation(source_id, &table);
        let held = ctx
            .sql(&format!("SELECT * FROM {relation} LIMIT 0"))
            .await
            .map_err(|e| JammiError::Other(format!("source '{source_id}': schema read: {e}")))?;
        match columns
            .into_iter()
            .find(|column| held.schema().field_with_unqualified_name(column).is_err())
        {
            Some(missing) => Err(JammiError::Schema {
                table: source_id.to_string(),
                column: missing.clone(),
                expected: "a column of the source".into(),
                actual: "missing".into(),
            }),
            None => Ok(relation),
        }
    }

    /// The declared edges of `edge_source` as a relation over `ctx`, with
    /// canonical `_src`/`_dst` (`Utf8`) columns and, when the source carries
    /// one, a `_weight` (`Float64`) — and whether it does.
    ///
    /// The scan runs through the generic SQL surface so the tenant-scope
    /// analyzer rule injects the `tenant_id` predicate — a cross-tenant
    /// endpoint is filtered before it reaches the adjacency. A registered
    /// source's bound columns are checked against its schema first, so a
    /// column it lacks is refused by [`Self::require_source_columns`].
    async fn edge_scan(
        &self,
        ctx: &QueryContext,
        edge_source: &EdgeSourceRef,
    ) -> Result<(DataFrame, bool)> {
        let scan_error = |e| JammiError::Other(format!("propagate: edge scan: {e}"));
        match edge_source {
            EdgeSourceRef::NeighborGraph { table_name } => {
                // neighbor_graph tables register as the bare literal
                // `jammi.{name}`; `similarity` is the edge weight.
                let sql = format!(
                    "SELECT arrow_cast(src, 'Utf8') AS _src, \
                     arrow_cast(dst, 'Utf8') AS _dst, \
                     arrow_cast(similarity, 'Float64') AS _weight \
                     FROM {}",
                    jammi_db::store::result_table_relation(table_name)
                );
                Ok((ctx.sql(&sql).await.map_err(scan_error)?, true))
            }
            EdgeSourceRef::Registered {
                source_id,
                src_column,
                dst_column,
                weight_column,
                ..
            } => {
                let bound = [Some(src_column), Some(dst_column), weight_column.as_ref()];
                let relation = self
                    .require_source_columns(ctx, source_id, bound.into_iter().flatten())
                    .await?;
                let quote = jammi_db::sql::quote_ident;
                let mut projection = format!(
                    "arrow_cast({}, 'Utf8') AS _src, arrow_cast({}, 'Utf8') AS _dst",
                    quote(src_column),
                    quote(dst_column)
                );
                if let Some(weight) = weight_column {
                    projection.push_str(&format!(
                        ", arrow_cast({}, 'Float64') AS _weight",
                        quote(weight)
                    ));
                }
                let sql = format!("SELECT {projection} FROM {relation}");
                Ok((
                    ctx.sql(&sql).await.map_err(scan_error)?,
                    weight_column.is_some(),
                ))
            }
        }
    }
}

/// Map the AI-crate [`EdgeDirection`] to the manifest's transport-neutral mirror.
pub(crate) fn propagation_direction(
    direction: EdgeDirection,
) -> jammi_db::store::manifest::PropagationDirection {
    use jammi_db::store::manifest::PropagationDirection as M;
    match direction {
        EdgeDirection::Out => M::Out,
        EdgeDirection::In => M::In,
        EdgeDirection::Undirected => M::Undirected,
    }
}

/// Map the AI-crate [`PropagationWeighting`] to the manifest's mirror.
pub(crate) fn propagation_weighting(
    weighting: PropagationWeighting,
) -> jammi_db::store::manifest::PropagationWeighting {
    use jammi_db::store::manifest::PropagationWeighting as M;
    match weighting {
        PropagationWeighting::Uniform => M::Uniform,
        PropagationWeighting::DegreeNormalized => M::DegreeNormalized,
        PropagationWeighting::EdgeSimilarity => M::EdgeSimilarity,
    }
}

/// Map the AI-crate [`PropagationOutput`] to the manifest's mirror. The
/// weights are recorded by their IEEE-754 bit patterns, so the descriptor
/// stays bit-exact and `Eq`.
fn propagation_output(output: &PropagationOutput) -> jammi_db::store::manifest::PropagationOutput {
    use jammi_db::store::manifest::PropagationOutput as M;
    match output {
        PropagationOutput::Final => M::Final,
        PropagationOutput::JumpingKnowledge => M::JumpingKnowledge,
        PropagationOutput::WeightedSum { weights } => M::WeightedSum {
            weight_bits: weights.iter().map(|w| w.to_bits()).collect(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> PropagateRequest {
        PropagateRequest::new(
            "src",
            EdgeSourceRef::NeighborGraph {
                table_name: "g".into(),
            },
        )
    }

    #[test]
    fn request_defaults_are_oversmoothing_safe() {
        let req = request();
        assert_eq!(req.weighting, PropagationWeighting::DegreeNormalized);
        assert_eq!(req.alpha, DEFAULT_TELEPORT_ALPHA);
        assert_eq!(req.hops, DEFAULT_PROPAGATE_HOPS);
        assert_eq!(req.hop_cap, DEFAULT_HOP_CAP);
        assert_eq!(req.output, PropagationOutput::Final);
    }

    #[test]
    fn effective_hops_clamps_to_cap() {
        assert_eq!(request().with_hops(10).effective_hops(), DEFAULT_HOP_CAP);
        assert_eq!(
            request().with_hops(0).effective_hops(),
            1,
            "hops floor at 1"
        );
    }

    #[test]
    fn alpha_outside_the_unit_interval_is_refused() {
        for alpha in [0.0, 0.1, 1.0] {
            assert!(check_alpha(alpha).is_ok());
        }
        for alpha in [-0.1, 1.5, f64::NAN, f64::INFINITY] {
            assert!(matches!(check_alpha(alpha), Err(JammiError::Config(_))));
        }
    }

    #[test]
    fn weighted_sum_output_records_its_weights_bit_exactly() {
        use jammi_db::store::manifest::PropagationOutput as M;
        let output = PropagationOutput::WeightedSum {
            weights: vec![0.0, 1.0, 0.5],
        };
        assert_eq!(
            propagation_output(&output),
            M::WeightedSum {
                weight_bits: vec![0.0_f64.to_bits(), 1.0_f64.to_bits(), 0.5_f64.to_bits()],
            }
        );
    }
}
