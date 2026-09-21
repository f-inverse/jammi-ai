//! Graph-structure embeddings — an embedding table from an edge relation
//! alone, for nodes that have no content an encoder could read: a transaction
//! graph, an id-only entity graph, a citation graph without abstracts.
//!
//! Every other way structure enters an embedding — a graph-supervised
//! fine-tune, [`propagate_embeddings`](crate::pipeline::graph_propagation), a
//! declared-edge context — starts from content. This verb starts from the
//! graph. It is training-free and deterministic, a data-plane encoder: no
//! message passing is learned, and the output is an ordinary embedding table,
//! consumed through `search`.
//!
//! # What it computes
//!
//! A [graph propagation](crate::pipeline::graph_propagation) whose initial
//! features are generated rather than read, and whose per-hop blocks are
//! mixed rather than dropped:
//!
//! ```text
//! X⁽⁰⁾ = the structural seed        (a very sparse random projection row per
//!                                     node, scaled by d̃^β — `seed`)
//! X⁽ᵏ⁾ = P · X⁽ᵏ⁻¹⁾                 (k = 1..K, no teleport)
//! out  = Σₖ wₖ · X⁽ᵏ⁾ / ‖X⁽ᵏ⁾‖₂     (k = 0..K, row-wise norms)
//! ```
//!
//! Row `j` of `Pᵏ·R` is a random projection of node `j`'s `k`-step walk
//! distribution, so two nodes whose walks land in the same places — the same
//! community, the same role around a hub — get close rows, whatever their
//! keys are. The weights choose which walk lengths the embedding listens to.
//! This is FastRP (Chen et al. 2019), on this engine's operator.
//!
//! # The operator, and where it departs from the paper
//!
//! `P` is the propagation's own random-walk operator
//! ([`PropagationWeighting::Uniform`], the default): `P = D̃⁻¹(A + I)`, the
//! transition matrix of the *lazy* walk over the self-loop-augmented graph,
//! where the paper uses `D⁻¹A`. The self-loop is an invariant here, and it
//! buys three things the paper's operator lacks:
//!
//! - **Any graph is admissible.** The paper's normalisation argument needs
//!   `|λ| < 1` for every non-principal eigenvalue, which fails for a
//!   bipartite graph (`λ = −1`: walks alternate sides forever, and odd and
//!   even blocks never agree). `A + I` has an odd cycle at every node, so `P`
//!   is aperiodic and its spectrum lies in `(−1, 1]` on every graph — user
//!   ↔ item, account ↔ merchant and author ↔ paper graphs included.
//! - **An isolated node is a fixed point**, not a division by zero: its only
//!   neighbour is itself, so every block is its seed row and its output is
//!   that row's direction.
//! - **Block `k` sees every walk of length `≤ k`** (a lazy walk may wait), so
//!   the blocks are nested scales rather than disjoint parities.
//!
//! Three more departures, each stated where it lives: the seed drops the
//! paper's global `(2m)^β` factor and offers a fixed sparsity rather than
//! `s = √n`, so a row never depends on the graph's size
//! ([`seed`](crate::pipeline::graph_propagation::seed)); block `0` — the seed
//! itself — is a readable block with its own weight (`weights[0]`), where the
//! paper's sum starts at `k = 1`; and each block is L2-normalised before it
//! is weighed, as the paper's reference implementation does and its
//! Algorithm 1 does not say. The reference implementation's final
//! per-dimension standardisation is not applied: it is a statistic of the
//! whole table, and a row here is a function of its node's neighbourhood.
//!
//! A directed read (`Out`/`In`) is supported and means what it says — the
//! walk follows the edges one way — but the degree scale's derivation is the
//! undirected one, and [`EdgeDirection::Undirected`] is the default.
//!
//! # What is stable under change
//!
//! A node's seed row is a function of `(seed, key, dimensions, sparsity)` and
//! its own degree. Adding a node or an edge leaves every other seed row's
//! direction bit-identical, and the output row of any node farther than `K`
//! hops from the change bit-identical too: block `k` of node `j` reads only
//! `j`'s `k`-hop neighbourhood. At `β = 0` the radius is `K − 1` — the seed
//! rows are degree-free, so the change enters only through the touched
//! node's own mean. Inside the radius the output moves, as it must: the
//! structure changed.
//!
//! # Searching a structure table
//!
//! No encoder maps a query into this space, so there is nothing to search it
//! *with* but the table itself: query by row key (query-by-example — "nodes
//! placed like this one"), or by a vector read from the same table. A vector
//! of another width is refused by the index; a text-encoder vector of the
//! same width is a point of an unrelated space and ranks noise. The table
//! carries a `key_column` only when the request names one, and `search`
//! hydrates source columns only then.
//!
//! # References
//! - Chen, Sultan, Tian, Chen, Skiena 2019, *Fast and Accurate Network
//!   Embeddings via Very Sparse Random Projection (FastRP)*:
//!   <https://arxiv.org/abs/1908.11512>
//! - Wu et al. 2019, *Simplifying Graph Convolutional Networks* (the
//!   self-loop-augmented operator): <https://arxiv.org/abs/1902.07153>

use std::sync::Arc;

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};
use jammi_db::store::manifest::{MaterializationEnv, ProducingDescriptor};
use jammi_db::store::{CacheOutcome, CachePolicy};

use crate::pipeline::graph_neighbourhood::{EdgeDirection, EdgeSourceRef};
use crate::pipeline::graph_propagation::plan::FeatureSource;
use crate::pipeline::graph_propagation::readout::BlockReadout;
use crate::pipeline::graph_propagation::seed::SeedSpec;
use crate::pipeline::graph_propagation::{
    propagation_direction, propagation_weighting, PropagationOutput, PropagationShape,
    PropagationTable, PropagationWeighting,
};
use crate::session::InferenceSession;

/// Default embedding width. A random projection's distortion falls as
/// `1/√d`; 256 lanes hold community structure with margin (the guide's
/// planted-partition measurement) at half the index cost of the paper's 512.
pub const DEFAULT_STRUCTURE_DIMENSIONS: usize = 256;

/// Default readout weights over blocks `0..=4`: the seed and the one-hop
/// block unread, the two-, three- and four-step scales mixed equally. A
/// node's own seed row carries no structure, and the one-hop block of a
/// sparse graph is mostly the node's own few neighbours' noise; the signal
/// is in the longer walks.
pub const DEFAULT_STRUCTURE_WEIGHTS: [f64; 5] = [0.0, 0.0, 1.0, 1.0, 1.0];

/// Default degree exponent `β`: no degree scaling. A negative `β` damps the
/// pull of hubs (the paper tunes it in `[−1, 0]`).
pub const DEFAULT_STRUCTURE_BETA: f64 = 0.0;

/// Default projection sparsity `s = 3` (Achlioptas): two thirds of a seed row
/// is zero.
pub const DEFAULT_STRUCTURE_SPARSITY: f64 = 3.0;

/// Default seed.
pub const DEFAULT_STRUCTURE_SEED: u64 = 0;

/// The depth bound of a structure encoding. Its blocks are distinct diffusion
/// scales read out side by side, not one smoothing pass, so the bound is not
/// the smoothing cap: it is where a further block stops telling nodes apart
/// (`λ₂ᵏ` has decayed) while still costing a full shuffle of the graph.
pub const DEFAULT_STRUCTURE_HOP_CAP: usize = 8;

/// The model-id provenance recorded on a structure table.
const STRUCTURE_MODEL_ID: &str = "graph_structure";

/// A graph-structure encoding: which edge relation, read how, into what.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructureRequest {
    /// The source the embedding table belongs to — what `search` is later
    /// called with. With only a graph in hand, the edge source itself.
    pub source_id: String,
    /// The column of `source_id` holding the node keys, when it has one:
    /// recorded as the table's key provenance, so `search` hydrates the
    /// source's columns onto its results. `None` leaves results as keys and
    /// scores.
    pub key_column: Option<String>,
    /// The edge relation defining the graph. Every key it names is a node; a
    /// node with no edge is declared by a self-edge `(v, v)`.
    pub edge_source: EdgeSourceRef,
    /// How an edge's direction is read.
    pub direction: EdgeDirection,
    /// The walk operator's weighting.
    pub weighting: PropagationWeighting,
    /// The embedding width `d`.
    pub dimensions: usize,
    /// The readout weight of each block `0..=K` — so the depth is
    /// `weights.len() − 1` hops. `weights[0]` weighs the seed itself.
    pub weights: Vec<f64>,
    /// The degree exponent `β` of the seed scale `d̃^β`.
    pub beta: f64,
    /// The projection sparsity `s ≥ 1`.
    pub sparsity: f64,
    /// Keys every node's projection stream.
    pub seed: u64,
    /// The most hops `weights` may ask for.
    pub hop_cap: usize,
}

impl StructureRequest {
    /// An encoding with the defaults: an undirected read under the
    /// random-walk operator, [`DEFAULT_STRUCTURE_DIMENSIONS`] lanes,
    /// [`DEFAULT_STRUCTURE_WEIGHTS`], no degree scaling, `s = 3`, seed `0`.
    pub fn new(source_id: impl Into<String>, edge_source: EdgeSourceRef) -> Self {
        Self {
            source_id: source_id.into(),
            key_column: None,
            edge_source,
            direction: EdgeDirection::Undirected,
            weighting: PropagationWeighting::Uniform,
            dimensions: DEFAULT_STRUCTURE_DIMENSIONS,
            weights: DEFAULT_STRUCTURE_WEIGHTS.to_vec(),
            beta: DEFAULT_STRUCTURE_BETA,
            sparsity: DEFAULT_STRUCTURE_SPARSITY,
            seed: DEFAULT_STRUCTURE_SEED,
            hop_cap: DEFAULT_STRUCTURE_HOP_CAP,
        }
    }

    /// Builder: name the source column holding the node keys.
    pub fn with_key_column(mut self, key_column: impl Into<String>) -> Self {
        self.key_column = Some(key_column.into());
        self
    }

    /// Builder: set how an edge's direction is read.
    pub fn with_direction(mut self, direction: EdgeDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Builder: set the walk operator's weighting.
    pub fn with_weighting(mut self, weighting: PropagationWeighting) -> Self {
        self.weighting = weighting;
        self
    }

    /// Builder: set the embedding width.
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Builder: set the per-block readout weights (and with them the depth).
    pub fn with_weights(mut self, weights: impl Into<Vec<f64>>) -> Self {
        self.weights = weights.into();
        self
    }

    /// Builder: set the degree exponent `β`.
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Builder: set the projection sparsity `s`.
    pub fn with_sparsity(mut self, sparsity: f64) -> Self {
        self.sparsity = sparsity;
        self
    }

    /// Builder: set the seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Builder: set the depth bound.
    pub fn with_hop_cap(mut self, hop_cap: usize) -> Self {
        self.hop_cap = hop_cap;
        self
    }

    /// The depth the weights ask for, `weights.len() − 1`. Refuses
    /// ([`JammiError::Config`]) no weights at all, and a depth past
    /// [`Self::hop_cap`] — a weight cannot be clamped away silently.
    pub fn hops(&self) -> Result<usize> {
        let hops = self.weights.len().checked_sub(1).ok_or_else(|| {
            JammiError::Config(
                "a structure encoding needs at least one readout weight (the seed block's)".into(),
            )
        })?;
        if hops > self.hop_cap {
            return Err(JammiError::Config(format!(
                "a structure encoding of {hops} hops ({} weights) exceeds its hop cap of {}",
                self.weights.len(),
                self.hop_cap
            )));
        }
        Ok(hops)
    }
}

impl InferenceSession {
    /// Encode a graph's structure into an embedding table — the thin
    /// [`Self::run_now`] wrapper over a
    /// [`crate::jobs::ComputeSpec::GraphStructure`], so a direct call and a
    /// queued job of the same spec run identical code. See the
    /// [module doc](crate::pipeline::graph_structure) for what is computed.
    pub async fn generate_structure_embeddings(
        self: &Arc<Self>,
        request: &StructureRequest,
        cache: CachePolicy,
    ) -> Result<(ResultTableRecord, CacheOutcome)> {
        let spec = crate::jobs::ComputeSpec::GraphStructure {
            request: request.clone(),
            cache,
        };
        self.run_table_spec("generate_structure_embeddings", spec)
            .await
    }

    /// `generate_structure_embeddings`'s materializer — see
    /// [`InferenceSession::infer_materialize`]'s doc for the `job_attempt`
    /// convention every `*_materialize` method shares.
    ///
    /// Validates the request, then lands the structural-seed propagation
    /// through the funnel [`Self::propagate_embeddings`] lands through: a
    /// `kind=Model` embedding table with a sidecar index.
    pub(crate) async fn generate_structure_embeddings_materialize(
        self: &Arc<Self>,
        request: &StructureRequest,
        cache: CachePolicy,
        job_attempt: Option<jammi_db::catalog::result_repo::JobAttempt<'_>>,
    ) -> Result<(ResultTableRecord, CacheOutcome)> {
        let hops = request.hops()?;
        let seed = SeedSpec::new(
            request.seed,
            request.dimensions,
            request.sparsity,
            request.beta,
        )?;
        let readout = BlockReadout::lower(
            &PropagationOutput::WeightedSum {
                weights: request.weights.clone(),
            },
            hops,
        )?;
        let ctx = self
            .context()
            .out_of_core(readout.state_row_bytes(request.dimensions));
        if let Some(key_column) = &request.key_column {
            self.require_source_columns(&ctx, &request.source_id, [key_column])
                .await?;
        }

        let descriptor = ProducingDescriptor::GraphStructure {
            edge_source: request.edge_source.to_binding(),
            kernel_id: STRUCTURE_MODEL_ID.to_string(),
            direction: propagation_direction(request.direction),
            weighting: propagation_weighting(request.weighting),
            seed: request.seed,
            dimensions: request.dimensions,
            sparsity_bits: request.sparsity.to_bits(),
            beta_bits: request.beta.to_bits(),
            weight_bits: request.weights.iter().map(|w| w.to_bits()).collect(),
        };
        let env = MaterializationEnv::new(self.compute_device(), Vec::new());
        let inputs = vec![self.edge_source_anchor(&request.edge_source).await?];
        if let Some(reused) = self.probe_cache(cache, &descriptor, &env, &inputs).await? {
            return Ok(reused);
        }

        let record = self
            .materialize_propagation(
                &ctx,
                PropagationShape {
                    edge_source: &request.edge_source,
                    direction: request.direction,
                    weighting: request.weighting,
                    alpha: 0.0,
                    hops,
                    readout,
                    dimensions: request.dimensions,
                },
                FeatureSource::StructuralSeed(seed),
                PropagationTable {
                    source_id: &request.source_id,
                    model_id: STRUCTURE_MODEL_ID,
                    derived_from: None,
                    key_column: request.key_column.as_deref(),
                    descriptor: &descriptor,
                    env: &env,
                    inputs,
                },
                job_attempt,
            )
            .await?;
        Ok((record, CacheOutcome::Computed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> StructureRequest {
        StructureRequest::new(
            "edges",
            EdgeSourceRef::Registered {
                source_id: "edges".into(),
                src_column: "src".into(),
                dst_column: "dst".into(),
                type_column: None,
                weight_column: None,
                as_of_column: None,
            },
        )
    }

    #[test]
    fn the_weights_are_the_depth() {
        assert_eq!(
            request().hops().unwrap(),
            DEFAULT_STRUCTURE_WEIGHTS.len() - 1
        );
        assert_eq!(request().with_weights([1.0]).hops().unwrap(), 0);
    }

    #[test]
    fn no_weights_and_a_depth_past_the_cap_are_refused() {
        let none = request().with_weights(Vec::<f64>::new());
        assert!(matches!(none.hops(), Err(JammiError::Config(_))));
        let deep = request().with_weights(vec![1.0; DEFAULT_STRUCTURE_HOP_CAP + 2]);
        assert!(matches!(deep.hops(), Err(JammiError::Config(_))));
        let allowed = deep.with_hop_cap(DEFAULT_STRUCTURE_HOP_CAP + 1);
        assert_eq!(allowed.hops().unwrap(), DEFAULT_STRUCTURE_HOP_CAP + 1);
    }

    #[test]
    fn the_request_survives_a_spec_round_trip() {
        let request = request()
            .with_key_column("account_id")
            .with_direction(EdgeDirection::Out)
            .with_weighting(PropagationWeighting::EdgeSimilarity)
            .with_dimensions(64)
            .with_weights([0.0, 1.0, 0.25])
            .with_beta(-0.5)
            .with_sparsity(8.0)
            .with_seed(u64::MAX);
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            serde_json::from_str::<StructureRequest>(&json).unwrap(),
            request
        );
    }
}
