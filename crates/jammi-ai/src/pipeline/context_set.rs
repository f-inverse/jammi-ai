//! Context-set assembly + the permutation-invariant set encoder.
//!
//! A Neural Process conditions a prediction on a context set
//! `C = {(xᵢ, yᵢ)}`. In a database that set is not an abstraction to invent —
//! it is a retrieval joined to its labels: `C = search(target, k) ⋈
//! value_columns`. This module makes that first-class and ships the
//! permutation-invariant set encoder `ρ(Σ φ(·))` (DeepSets) as the *encode +
//! aggregate* half of an NP.
//!
//! The encoder is **fixed** pooling — `mean | sum | max` over the neighbour
//! vectors — and reuses the engine's vector-aggregation UDAF
//! ([`crate::query::register_vector_agg_udafs`], the same operator graph
//! propagation pools through). There is exactly one aggregation implementation;
//! this module calls it via SQL. Learned/attention pooling (which context
//! element matters) is out of scope here: that is the AttnCNP point on the
//! spectrum, owned downstream. Fixed symmetric pooling is the DeepSets/CNP
//! expressiveness ceiling, stated rather than hidden.
//!
//! Leakage is a first-class contract. A target retrieving *itself* (or a
//! same-key duplicate) as its own context trivially leaks the answer when a
//! `value_column` is the prediction target, so `exclude_self` defaults true and
//! drops every same-key neighbour before pooling. A `split` predicate scopes
//! the context to a train split so a target's own outcome stays held out.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch};
use datafusion::prelude::{col, lit};

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};

use crate::pipeline::graph_neighbourhood::{EdgeDirection, EdgeGather};
use crate::session::InferenceSession;

/// How a context set's candidate members were assembled — a *fact* about the
/// assembly, carried on [`ContextRepresentation::source`]. It is **not** an
/// exchangeability judgment: the engine records how the context was built and
/// lets governance decide whether a marginal conformal claim over it is sound
/// (the S16-G coverage doctrine — the engine surfaces the fact, governance
/// chooses the lever).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextSourceKind {
    /// Embedding-similarity neighbours (`search(target, k)`).
    Ann,
    /// Declared-edge neighbours (a bounded, target-anchored walk).
    Edges,
    /// The union of both, pooled once.
    Hybrid,
}

/// How a [`Hybrid`](ContextSource::Hybrid) context merges its ANN and declared-
/// edge candidate sets. An enum (not a bool) so per-edge-type channels can be
/// added without a breaking reshape; v1 ships `Union`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HybridMerge {
    /// Union the candidate key sets (ANN first, in similarity order; then the
    /// declared-edge neighbours not already present), dedup, pool once.
    #[default]
    Union,
}

/// The candidate-set source for a target's context: embedding-similar rows, a
/// declared-edge walk, or both. The source selects only how the candidate keys
/// are produced; everything after the gather (exclude-self → split → pool →
/// hydrate) is the same S16 pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum ContextSource {
    /// S16 retrieval: `search(query, k)` over the source's embedding table.
    Ann {
        /// Neighbourhood size.
        k: usize,
    },
    /// A declared-edge walk anchored at the target (the target's row key, passed
    /// as [`ContextRequest::exclude_key`], is the gather anchor).
    Edges(EdgeGather),
    /// Union of the ANN and declared-edge candidate sets, pooled once.
    Hybrid {
        /// ANN neighbourhood size for the retrieval arm.
        ann_k: usize,
        /// The declared-edge gather for the edge arm.
        edges: EdgeGather,
        /// How the two candidate sets merge.
        merge: HybridMerge,
    },
}

impl ContextSource {
    /// The assembly-fact tag for this source.
    pub fn kind(&self) -> ContextSourceKind {
        match self {
            ContextSource::Ann { .. } => ContextSourceKind::Ann,
            ContextSource::Edges(_) => ContextSourceKind::Edges,
            ContextSource::Hybrid { .. } => ContextSourceKind::Hybrid,
        }
    }

    /// The post-exclusion/-split key cap, if the source bounds it by count. ANN
    /// is `k`-bounded; an edge walk is bounded by hops/fan-out, not a count, so
    /// it (and a hybrid union) keeps every gathered member.
    fn max_keys(&self) -> Option<usize> {
        match self {
            ContextSource::Ann { k } => Some(*k),
            ContextSource::Edges(_) | ContextSource::Hybrid { .. } => None,
        }
    }
}

/// Which permutation-invariant reduction the set encoder folds the neighbour
/// vectors with. Maps one-to-one onto the engine's vector-aggregation UDAF
/// names; none is universally right (DeepSets), so the caller chooses and the
/// decoder always also receives [`ContextRepresentation::context_size`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SetAggregator {
    /// Element-wise mean. Discards set size; the default symmetric pool.
    #[default]
    Mean,
    /// Element-wise sum. Encodes set size into the pooled vector.
    Sum,
    /// Element-wise maximum. Robust but lossy.
    Max,
}

impl SetAggregator {
    /// The SQL UDAF this aggregator pools through — the *same* element-wise
    /// vector-aggregation function registered by
    /// [`crate::query::register_vector_agg_udafs`]. One operator, two callers
    /// (graph propagation and context encoding); this is the call site, not a
    /// second implementation.
    const fn udaf_name(self) -> &'static str {
        match self {
            SetAggregator::Mean => "vector_mean",
            SetAggregator::Sum => "vector_sum",
            SetAggregator::Max => "vector_max",
        }
    }
}

/// A request to assemble and encode a target's context set.
///
/// The context is `search(query, k)` over `source`'s embedding table, with the
/// target's own row excluded ([`Self::exclude_self`]) and optionally scoped to a
/// train split ([`Self::split`]); the encoder pools the retrieved neighbour
/// vectors with [`Self::aggregator`].
#[derive(Debug, Clone)]
pub struct ContextRequest {
    /// Source whose embedding table is retrieved against and whose rows the
    /// context members are pooled and hydrated from.
    pub source_id: String,
    /// The specific embedding table to retrieve/pool/hydrate against, or `None`
    /// to resolve the source's default (newest) embedding table. Pinning it makes
    /// a recompute faithful: a materialised context set is itself a `kind=model`
    /// table for the same source, so re-resolving the *default* after it exists
    /// would shadow the original source table — the pin holds the pooling on the
    /// table the recipe actually ran over.
    pub embedding_table: Option<String>,
    /// The target's query vector — the point whose neighbourhood is the ANN /
    /// hybrid context. (A pure-edge context anchors on `exclude_key` instead.)
    pub query: Vec<f32>,
    /// Where the candidate members come from: ANN retrieval, a declared-edge
    /// walk, or both. The ANN neighbourhood size lives here ([`ContextSource::Ann`]).
    pub source: ContextSource,
    /// Label / outcome columns hydrated from the source for each context row.
    pub value_columns: Vec<String>,
    /// The pooling reduction the set encoder applies.
    pub aggregator: SetAggregator,
    /// Drop any neighbour whose key equals `exclude_key` before pooling — the
    /// leakage guard. Defaults on through [`Self::new`].
    pub exclude_self: bool,
    /// The target's own row key, dropped from the context when `exclude_self`
    /// is set. `None` for a free-vector query that corresponds to no stored row.
    pub exclude_key: Option<String>,
    /// Optional SQL predicate (over the source's columns) scoping the context to
    /// a split, e.g. `"split = 'train'"`. Holds the train/target leakage line.
    pub split: Option<String>,
}

impl ContextRequest {
    /// A context request with the leakage-safe defaults: `exclude_self` on,
    /// mean pooling, no split scope. `exclude_key` is left unset — pass the
    /// target's row key through [`Self::exclude_key`] when the query vector
    /// belongs to a stored row.
    pub fn new(source_id: impl Into<String>, query: Vec<f32>, k: usize) -> Self {
        Self {
            source_id: source_id.into(),
            embedding_table: None,
            query,
            source: ContextSource::Ann { k },
            value_columns: Vec::new(),
            aggregator: SetAggregator::Mean,
            exclude_self: true,
            exclude_key: None,
            split: None,
        }
    }
}

/// The encoded context set: the pooled conditioning vector plus the carried
/// metadata an NP decoder conditions on.
///
/// `context_size` is carried **separately** from `context_vector` (never folded
/// into it) so a decoder can use the count signal without it corrupting the
/// pooled representation — the size-invariance contract. `value_rows` carries
/// the requested `value_columns` of each context member for label-aware
/// decoding.
#[derive(Debug, Clone)]
pub struct ContextRepresentation {
    /// The fixed-width pooled vector `ρ(Σ φ(xᵢ))`. `None` for an empty context
    /// (no neighbour survived `exclude_self`/`split`) — a degenerate set is not
    /// silently averaged into a one-element "representation".
    pub context_vector: Option<Vec<f32>>,
    /// Number of neighbours that survived exclusion and entered the pool.
    pub context_size: usize,
    /// Keys of the context members, in retrieval (descending-similarity) order.
    pub context_keys: Vec<String>,
    /// The hydrated `value_columns` of the context members — one batch carrying
    /// the requested label/outcome columns, in retrieval order. Empty batches
    /// when no `value_columns` were requested.
    pub value_rows: Vec<RecordBatch>,
    /// How this context was assembled (ANN / declared-edge / hybrid). A *fact*
    /// the decoder and governance read — never an exchangeability judgment the
    /// engine makes.
    pub source: ContextSourceKind,
}

impl ContextRepresentation {
    /// Whether the context set is empty — no neighbour survived self-exclusion
    /// and split scoping. A degenerate context the decoder should treat as
    /// low-confidence / fall back to the prior, never as a representation.
    pub fn is_empty(&self) -> bool {
        self.context_size == 0
    }
}

impl InferenceSession {
    /// Assemble a target's context set and encode it into a fixed-width
    /// [`ContextRepresentation`].
    ///
    /// `C = search(query, k) ⋈ value_columns`, with the target's own row
    /// excluded (`exclude_self`) and optionally scoped to a split, then pooled
    /// permutation-invariantly by the requested [`SetAggregator`]. The pooling
    /// reuses the engine's vector-aggregation UDAF — one aggregation operator,
    /// called here via SQL.
    ///
    /// Honours the embeddings-through-`search` stance: the neighbour vectors are
    /// pooled inside the engine and never cross the API boundary as raw vectors;
    /// only the single pooled `context_vector` is returned.
    pub async fn assemble_context(
        self: &Arc<Self>,
        request: &ContextRequest,
    ) -> Result<ContextRepresentation> {
        let table = self
            .catalog()
            .resolve_embedding_table(&request.source_id, request.embedding_table.as_deref())
            .await?;

        // The candidate set — the only part that differs by source. Everything
        // after this is the one shared tail (exclude-self → split → truncate →
        // pool → hydrate), so ANN, edge, and hybrid contexts are leakage-scoped
        // and pooled identically.
        let mut context_keys = self.gather_candidates(request, &table).await?;

        // Self-exclusion (the leakage guard): drop every neighbour whose key
        // matches the target's own key. A free-vector query carries no key, so
        // nothing is dropped; an edge walk already excludes its anchor.
        let exclude = request
            .exclude_self
            .then_some(request.exclude_key.as_deref())
            .flatten();
        if let Some(exclude) = exclude {
            context_keys.retain(|key| key.as_str() != exclude);
        }

        // Split scope (the train/target leakage line): keep only the candidates
        // whose source row satisfies the split predicate, before any count cap.
        // Applied here rather than during retrieval because the predicate is over
        // the source's columns, which the ANN index does not carry. A predicate
        // no candidate satisfies yields an empty context — defined, not a crash.
        // Expressing the split by a graph-locality column (component/cohort)
        // keeps graph-adjacent rows on one side of the train/eval line.
        if let Some(split) = request.split.as_deref() {
            context_keys = self
                .filter_keys_by_split(
                    &request.source_id,
                    request.embedding_table.as_deref(),
                    &context_keys,
                    split,
                )
                .await?;
        }

        // An ANN context is `k`-bounded (over-fetched, then capped here); an edge
        // walk is bounded by hops/fan-out, so it keeps every gathered member.
        if let Some(max) = request.source.max_keys() {
            context_keys.truncate(max);
        }

        let context_vector = self
            .pool_context_vectors(&table, &context_keys, request.aggregator)
            .await?;

        let value_rows = if request.value_columns.is_empty() {
            Vec::new()
        } else {
            self.hydrate_value_columns(
                &request.source_id,
                request.embedding_table.as_deref(),
                &context_keys,
                &request.value_columns,
            )
            .await?
        };

        Ok(ContextRepresentation {
            context_vector,
            context_size: context_keys.len(),
            context_keys,
            value_rows,
            source: request.source.kind(),
        })
    }

    /// Produce the candidate member keys for a context, in retrieval order — the
    /// only step that differs by [`ContextSource`]. ANN over-fetches and ranks by
    /// similarity; an edge source runs the bounded, target-anchored walk; a
    /// hybrid unions the two (ANN first, then declared-edge members not already
    /// present). Self-exclusion / split / cap / pool / hydrate all follow in the
    /// shared tail of [`assemble_context`].
    async fn gather_candidates(
        self: &Arc<Self>,
        request: &ContextRequest,
        table: &ResultTableRecord,
    ) -> Result<Vec<String>> {
        match &request.source {
            ContextSource::Ann { k } => {
                self.ann_candidates(table, &request.query, *k, request.exclude_self)
                    .await
            }
            ContextSource::Edges(gather) => {
                let target = self.edge_anchor(request)?;
                Ok(self.gather_edge_candidates(gather, target).await?.keys)
            }
            ContextSource::Hybrid {
                ann_k,
                edges,
                merge,
            } => {
                let mut keys = self
                    .ann_candidates(table, &request.query, *ann_k, request.exclude_self)
                    .await?;
                let target = self.edge_anchor(request)?;
                let edge_keys = self.gather_edge_candidates(edges, target).await?.keys;
                match merge {
                    HybridMerge::Union => {
                        let existing: HashSet<&str> = keys.iter().map(String::as_str).collect();
                        let extra: Vec<String> = edge_keys
                            .into_iter()
                            .filter(|k| !existing.contains(k.as_str()))
                            .collect();
                        keys.extend(extra);
                    }
                }
                Ok(keys)
            }
        }
    }

    /// The ANN candidate keys: `search(query, k)` over the embedding table,
    /// over-fetching by one under self-exclusion so a self-hit never shrinks the
    /// surviving context below `k`.
    async fn ann_candidates(
        &self,
        table: &ResultTableRecord,
        query: &[f32],
        k: usize,
        exclude_self: bool,
    ) -> Result<Vec<String>> {
        let fetch_k = if exclude_self { k.saturating_add(1) } else { k };
        let neighbours = self
            .result_store()
            .search_vectors(self.context(), table, query, fetch_k)
            .await?;
        Ok(neighbours
            .into_iter()
            .map(|(key, _similarity)| key)
            .collect())
    }

    /// The target key an edge walk anchors on — the target's own row key, passed
    /// through `exclude_key`. A pure-edge context over a free-vector query has no
    /// anchor and is a typed error, not a silent empty gather.
    fn edge_anchor<'a>(&self, request: &'a ContextRequest) -> Result<&'a str> {
        request.exclude_key.as_deref().ok_or_else(|| {
            JammiError::Other(
                "declared-edge context requires the target's row key (exclude_key) as the \
                 gather anchor"
                    .into(),
            )
        })
    }

    /// Pool the stored vectors of `context_keys` from `table`'s embedding
    /// Parquet into one fixed-width vector via the vector-aggregation UDAF.
    ///
    /// Filters the registered embedding table to the context keys and runs
    /// `vector_<agg>(vector)` over them — the aggregate's *value* is
    /// permutation-invariant by construction, so it does not depend on the order
    /// the keys arrive in. Under a fixed execution plan the pooled vector is also
    /// stable run-to-run; it is not byte-identical across arbitrary partitionings
    /// (`f64` `+` is non-associative — see the vector-aggregation UDAF), which the
    /// downstream conformal calibration tolerates (sub-rounding differences are
    /// immaterial). An empty key set yields `None` rather than a pool over
    /// nothing.
    async fn pool_context_vectors(
        &self,
        table: &ResultTableRecord,
        context_keys: &[String],
        aggregator: SetAggregator,
    ) -> Result<Option<Vec<f32>>> {
        if context_keys.is_empty() {
            return Ok(None);
        }

        // Resolve the shared vector-aggregation UDAF by name and call it over the
        // neighbour vectors — the single aggregation operator, this the call
        // site. `udaf` comes from the `FunctionRegistry` the session state
        // implements.
        use datafusion::execution::FunctionRegistry;
        let pool_udaf = self
            .context()
            .state()
            .udaf(aggregator.udaf_name())
            .map_err(|e| {
                JammiError::Other(format!(
                    "Context pool: UDAF '{}' not registered: {e}",
                    aggregator.udaf_name()
                ))
            })?;

        // Result tables register under the single bare literal `jammi.{name}`
        // (`register_parquet_table`), so reach this one through
        // `TableReference::bare` — a `&str` would be re-parsed and split on the
        // dot, missing the registered table whenever the name carries a hyphen
        // from a sanitized local model path.
        let table_ref =
            datafusion::sql::TableReference::bare(format!("jammi.{}", table.table_name));
        let keys: Vec<datafusion::prelude::Expr> =
            context_keys.iter().map(|k| lit(k.as_str())).collect();
        let pooled = self
            .context()
            .table(table_ref.clone())
            .await
            .map_err(|e| JammiError::Other(format!("Context pool: resolve '{table_ref}': {e}")))?
            // Typed IN-list over the keys — the arbitrary row keys are bound
            // values, never interpolated into SQL text.
            .filter(col("_row_id").in_list(keys, false))
            .map_err(|e| JammiError::Other(format!("Context pool: filter: {e}")))?
            .aggregate(vec![], vec![pool_udaf.call(vec![col("vector")])])
            .map_err(|e| JammiError::Other(format!("Context pool: aggregate: {e}")))?
            .collect()
            .await
            .map_err(|e| JammiError::Other(format!("Context pool: collect: {e}")))?;

        extract_single_vector(&pooled, &table.table_name)
    }

    /// Keep only the candidate keys whose source row satisfies `split`,
    /// preserving retrieval order.
    ///
    /// Scans the source restricted to `split` (a predicate over the source's
    /// columns, e.g. `split = 'train'`), projecting the key column, then retains
    /// the candidates present in that qualifying set. The leakage line: when the
    /// context feeds a train/eval target, the context rows come from the train
    /// split with the target's own outcome held out.
    async fn filter_keys_by_split(
        &self,
        source_id: &str,
        embedding_table: Option<&str>,
        context_keys: &[String],
        split: &str,
    ) -> Result<Vec<String>> {
        if context_keys.is_empty() {
            return Ok(Vec::new());
        }
        let table = self
            .catalog()
            .resolve_embedding_table(source_id, embedding_table)
            .await?;
        let key_col = table.key_column.as_deref().ok_or_else(|| {
            JammiError::Other(format!(
                "Context split: source '{source_id}' embedding table has no key column"
            ))
        })?;
        let source_table = self.find_table_name(source_id)?;

        let keys: Vec<datafusion::prelude::Expr> =
            context_keys.iter().map(|k| lit(k.as_str())).collect();

        // The split predicate is the user's own SQL over the source columns, so
        // it is applied as a `WHERE` clause inside the scan; the candidate keys
        // stay bound values in a typed IN-list, never interpolated into text.
        let batches = self
            .context()
            .sql(&format!(
                "SELECT arrow_cast(\"{key_col}\", 'Utf8') AS _context_key \
                 FROM \"{source_id}\".public.\"{source_table}\" WHERE {split}"
            ))
            .await
            .map_err(|e| JammiError::Other(format!("Context split: scan: {e}")))?
            .filter(col("_context_key").in_list(keys, false))
            .map_err(|e| JammiError::Other(format!("Context split: filter: {e}")))?
            .collect()
            .await
            .map_err(|e| JammiError::Other(format!("Context split: collect: {e}")))?;

        let qualifying = collect_keys(&batches, "_context_key");
        Ok(context_keys
            .iter()
            .filter(|k| qualifying.contains(*k))
            .cloned()
            .collect())
    }

    /// Hydrate the requested `value_columns` of `context_keys` from the source,
    /// preserving retrieval order.
    ///
    /// `_row_id` equals the source key-column value, so the context members join
    /// back to their labels/outcomes directly. The result is ordered by the
    /// retrieval rank (descending similarity) the keys were collected in.
    async fn hydrate_value_columns(
        &self,
        source_id: &str,
        embedding_table: Option<&str>,
        context_keys: &[String],
        value_columns: &[String],
    ) -> Result<Vec<RecordBatch>> {
        if context_keys.is_empty() {
            return Ok(Vec::new());
        }
        let table = self
            .catalog()
            .resolve_embedding_table(source_id, embedding_table)
            .await?;
        let key_col = table.key_column.as_deref().ok_or_else(|| {
            JammiError::Other(format!(
                "Context hydrate: source '{source_id}' embedding table has no key column"
            ))
        })?;
        let source_table = self.find_table_name(source_id)?;

        let select_list = value_columns
            .iter()
            .map(|c| format!("\"{c}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let keys: Vec<datafusion::prelude::Expr> =
            context_keys.iter().map(|k| lit(k.as_str())).collect();

        let batches = self
            .context()
            .sql(&format!(
                "SELECT {select_list}, arrow_cast(\"{key_col}\", 'Utf8') AS _context_key \
                 FROM \"{source_id}\".public.\"{source_table}\""
            ))
            .await
            .map_err(|e| JammiError::Other(format!("Context hydrate: scan: {e}")))?
            .filter(col("_context_key").in_list(keys, false))
            .map_err(|e| JammiError::Other(format!("Context hydrate: filter: {e}")))?
            .collect()
            .await
            .map_err(|e| JammiError::Other(format!("Context hydrate: collect: {e}")))?;

        Ok(order_by_keys(batches, context_keys))
    }
}

/// Collect every value of a string column across `batches` into a set.
fn collect_keys(batches: &[RecordBatch], column: &str) -> std::collections::HashSet<String> {
    use arrow::array::StringArray;
    batches
        .iter()
        .filter_map(|b| {
            b.column_by_name(column)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        })
        .flat_map(|keys| (0..keys.len()).map(|i| keys.value(i).to_string()))
        .collect()
}

/// Reorder hydrated `value_columns` rows to match `context_keys` (retrieval
/// order), dropping the carried `_context_key` column. The filter above returns
/// rows in scan order; the context's rank order is the retrieval order, so the
/// rows are sliced per key and re-emitted in it here.
fn order_by_keys(batches: Vec<RecordBatch>, context_keys: &[String]) -> Vec<RecordBatch> {
    use arrow::array::StringArray;

    // Build per-key single-row slices keyed off `_context_key`, then emit them
    // in `context_keys` order — one single-row batch per surviving context
    // member, in retrieval rank.
    let mut row_slices: std::collections::HashMap<String, Vec<RecordBatch>> =
        std::collections::HashMap::new();
    for batch in &batches {
        let Some(key_col) = batch
            .column_by_name("_context_key")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        else {
            continue;
        };
        let value_only = drop_named_column(batch, "_context_key");
        for row in 0..batch.num_rows() {
            let key = key_col.value(row).to_string();
            row_slices
                .entry(key)
                .or_default()
                .push(value_only.slice(row, 1));
        }
    }

    let mut ordered = Vec::new();
    for key in context_keys {
        if let Some(slices) = row_slices.get(key) {
            ordered.extend(slices.iter().cloned());
        }
    }
    ordered
}

/// Project `batch` to every column except `name`.
fn drop_named_column(batch: &RecordBatch, name: &str) -> RecordBatch {
    let keep: Vec<usize> = (0..batch.num_columns())
        .filter(|&i| batch.schema().field(i).name() != name)
        .collect();
    batch.project(&keep).expect("projecting existing columns")
}

/// Extract the single pooled `FixedSizeList<Float32>` cell from the UDAF's
/// one-row, one-column output, or `None` when the group was empty (the
/// aggregate evaluates a null vector over no rows).
fn extract_single_vector(batches: &[RecordBatch], table: &str) -> Result<Option<Vec<f32>>> {
    let Some(batch) = batches.iter().find(|b| b.num_rows() > 0) else {
        return Ok(None);
    };
    let list = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| JammiError::Schema {
            table: table.to_string(),
            column: "context_vector".into(),
            expected: "FixedSizeList<Float32>".into(),
            actual: format!("{}", batch.column(0).data_type()),
        })?;
    if list.is_null(0) {
        return Ok(None);
    }
    let values = list.value(0);
    let floats = values
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| JammiError::Schema {
            table: table.to_string(),
            column: "context_vector".into(),
            expected: "FixedSizeList<Float32>".into(),
            actual: format!("{}", values.data_type()),
        })?;
    Ok(Some((0..floats.len()).map(|i| floats.value(i)).collect()))
}

/// A batch of pooled context vectors keyed by target, the materialised form of
/// [`assemble_context`](InferenceSession::assemble_context) over many targets
/// that all shared one assembly recipe. The unique target keys become the
/// `_row_id` of a normal embedding-shaped result table.
///
/// The whole batch is pooled under one [`ContextRequest`] recipe — the
/// determinant set of *how* every row was produced (candidate source, value
/// columns, aggregator, self-exclusion, split). That recipe is carried here so
/// the materialization descriptor records the real producer completely; the
/// per-target `query` / `exclude_key` of each individual request are the inputs
/// the recipe ran over (and become the row keys), not the recipe's definition,
/// so they are not part of the shared recipe.
#[derive(Debug)]
pub struct MaterializedContext<'a> {
    /// Each target's key paired with its pooled context vector.
    pub rows: &'a [(String, Vec<f32>)],
    /// The pooled vectors' fixed width.
    pub dimensions: usize,
    /// The assembly recipe every row in the batch was pooled under — the
    /// `ContextSet` descriptor's determinant set.
    pub recipe: &'a ContextRequest,
}

impl InferenceSession {
    /// Materialise per-target pooled context vectors into a normal
    /// embedding-shaped result table — searchable and joinable like any other
    /// embedding table, with its own sidecar ANN index — for batch NP
    /// workflows.
    ///
    /// Delegates the write to [`jammi_db::store::ResultStore::materialize_embedding_table`],
    /// which builds the same `(_row_id, _source_id, _model_id, vector)` Parquet
    /// and sidecar index every embedding table gets, so a materialised context
    /// set is a first-class member of the same table family.
    pub async fn materialize_context(
        &self,
        context: MaterializedContext<'_>,
        cache: jammi_db::store::CachePolicy,
    ) -> Result<(ResultTableRecord, jammi_db::store::CacheOutcome)> {
        // De-duplicate target keys so the table's `_row_id` stays a key: a
        // batch must not carry two pooled vectors for the same target.
        let seen: BTreeSet<&str> = context.rows.iter().map(|(k, _)| k.as_str()).collect();
        if seen.len() != context.rows.len() {
            return Err(JammiError::Other(
                "materialize_context: duplicate target key in context rows".into(),
            ));
        }

        let recipe = context.recipe;
        let source_id = recipe.source_id.as_str();

        // The materialization contract: a context set pools from the source's
        // raw rows (no model invoked here — the encoder is the pooling kernel),
        // so the environment carries the engine version + device with an empty
        // model set. The descriptor records the full assembly recipe — the real
        // producer — so two materialisations under different recipes (a different
        // candidate source, value columns, aggregator, self-exclusion, or split)
        // hash differently. There is no single source result table the batch
        // derives from, so the input is the source itself, which has no version
        // surface in open-core → `UnpinnedAtInstant` (honest, not a fabricated
        // pin).
        let descriptor = jammi_db::store::manifest::ProducingDescriptor::ContextSet {
            encoder_id: "jammi:context-set".to_string(),
            source_id: source_id.to_string(),
            embedding_table: recipe.embedding_table.clone(),
            candidate_source: candidate_source_for(&recipe.source),
            value_columns: recipe.value_columns.clone(),
            aggregator: aggregator_for(recipe.aggregator),
            exclude_self: recipe.exclude_self,
            split: recipe.split.clone(),
            dimensions: context.dimensions,
        };
        let env =
            jammi_db::store::manifest::MaterializationEnv::new(self.compute_device(), Vec::new());
        let inputs = vec![jammi_db::store::manifest::InputAnchor::unpinned_at_instant(
            source_id,
            chrono::Utc::now().to_rfc3339(),
        )];

        // Cache probe before the write. A context set anchors its source
        // `UnpinnedAtInstant` (a raw source with no version surface), so a `Use`
        // request is honestly always a miss; the probe runs for surface
        // uniformity and provable off-ness. The probe is before any Parquet
        // write, so a hit would leave no `building` orphan to reap.
        if cache == jammi_db::store::CachePolicy::Use {
            let def_hash = jammi_db::store::manifest::MaterializationManifest::definition_of(
                &descriptor,
                &env,
            )
            .map_err(jammi_db::store::manifest_to_jammi)?;
            if let Some(reused) = self
                .result_store()
                .probe_cache_record(&def_hash, &inputs)
                .await?
            {
                let table = reused.table_name.clone();
                return Ok((reused, jammi_db::store::CacheOutcome::Reused { table }));
            }
        }

        let record = self
            .result_store()
            .materialize_embedding_table(
                self.context(),
                jammi_db::store::EmbeddingTableSpec {
                    source_id,
                    model_id: "jammi:context-set",
                    // A pooled context set is keyed by *target* keys and pools
                    // each target's neighbours from the source's raw rows —
                    // there is no single source result table the whole batch
                    // derives from, so there is no FK-lineage anchor to record.
                    derived_from: None,
                    dimensions: context.dimensions,
                },
                context.rows,
                jammi_db::store::manifest::Materialization::new(&descriptor, &env, inputs),
            )
            .await?;
        Ok((record, jammi_db::store::CacheOutcome::Computed))
    }
}

/// Map the AI-crate [`ContextSource`] to the manifest's transport-neutral
/// candidate-source mirror — the determinant that selects which neighbours a
/// target pools. The ANN neighbourhood size and the declared-edge gather knobs
/// are output-affecting, so each is carried into the descriptor.
fn candidate_source_for(
    source: &ContextSource,
) -> jammi_db::store::manifest::ContextCandidateSource {
    use jammi_db::store::manifest::ContextCandidateSource as M;
    match source {
        ContextSource::Ann { k } => M::Ann { k: *k },
        ContextSource::Edges(gather) => M::Edges {
            gather: edge_gather_for(gather),
        },
        ContextSource::Hybrid {
            ann_k,
            edges,
            merge,
        } => M::Hybrid {
            ann_k: *ann_k,
            gather: edge_gather_for(edges),
            // `merge` is an output-determinant (it selects which keys survive
            // into the pool). With only `Union` today a variance test can't yet
            // distinguish two merges, but it is recorded for forward-completeness
            // so the next merge channel can't silently regress the descriptor.
            merge: merge_for(*merge),
        },
    }
}

/// Map the AI-crate [`HybridMerge`] to the manifest's merge-channel mirror.
fn merge_for(merge: HybridMerge) -> jammi_db::store::manifest::ContextHybridMerge {
    use jammi_db::store::manifest::ContextHybridMerge as M;
    match merge {
        HybridMerge::Union => M::Union,
    }
}

/// Map the AI-crate [`SetAggregator`] to the manifest's pooling mirror.
fn aggregator_for(aggregator: SetAggregator) -> jammi_db::store::manifest::ContextAggregator {
    use jammi_db::store::manifest::ContextAggregator as M;
    match aggregator {
        SetAggregator::Mean => M::Mean,
        SetAggregator::Sum => M::Sum,
        SetAggregator::Max => M::Max,
    }
}

/// Map the AI-crate [`EdgeGather`] to the manifest's transport-neutral mirror,
/// carrying every output-affecting knob of the walk. `hops` is recorded as the
/// *effective* (post-clamp) depth, so the descriptor never needs the cap; the
/// minimum edge weight is recorded by its IEEE-754 bit pattern so the descriptor
/// stays bit-exact and `Eq`/`Hash`-able.
fn edge_gather_for(gather: &EdgeGather) -> jammi_db::store::manifest::ContextEdgeGather {
    jammi_db::store::manifest::ContextEdgeGather {
        edge_source: gather.edge_source.to_binding(),
        hops: gather.effective_hops(),
        fanout: gather.fanout,
        direction: edge_direction_for(gather.direction),
        edge_types: gather.edge_types.clone(),
        min_weight_bits: gather.min_weight.map(f64::to_bits),
        as_of: gather.as_of.clone(),
    }
}

/// Map the AI-crate [`EdgeDirection`] to the manifest's walk-direction mirror.
fn edge_direction_for(direction: EdgeDirection) -> jammi_db::store::manifest::PropagationDirection {
    use jammi_db::store::manifest::PropagationDirection as M;
    match direction {
        EdgeDirection::Out => M::Out,
        EdgeDirection::In => M::In,
        EdgeDirection::Undirected => M::Undirected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregator_maps_to_the_shared_udaf_names() {
        // The encoder pools through the *same* UDAF names the engine registers
        // (`register_vector_agg_udafs`) — one aggregation operator, this the
        // call site. A drift here is a second implementation sneaking in.
        assert_eq!(SetAggregator::Mean.udaf_name(), "vector_mean");
        assert_eq!(SetAggregator::Sum.udaf_name(), "vector_sum");
        assert_eq!(SetAggregator::Max.udaf_name(), "vector_max");
    }

    #[test]
    fn new_request_is_leakage_safe_by_default() {
        let r = ContextRequest::new("patents", vec![0.0; 4], 5);
        assert!(r.exclude_self, "self-exclusion must default on");
        assert_eq!(r.aggregator, SetAggregator::Mean);
        assert!(r.split.is_none());
        assert_eq!(
            r.source,
            ContextSource::Ann { k: 5 },
            "the three-arg constructor builds an ANN source"
        );
        assert_eq!(r.source.kind(), ContextSourceKind::Ann);
    }

    #[test]
    fn empty_representation_reports_empty() {
        let rep = ContextRepresentation {
            context_vector: None,
            context_size: 0,
            context_keys: Vec::new(),
            value_rows: Vec::new(),
            source: ContextSourceKind::Ann,
        };
        assert!(rep.is_empty());
    }
}
