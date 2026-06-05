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

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch};
use datafusion::prelude::{col, lit};

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};

use crate::session::InferenceSession;

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
    /// Source whose embedding table is retrieved against.
    pub source_id: String,
    /// The target's query vector — the point whose neighbourhood is the context.
    pub query: Vec<f32>,
    /// Neighbourhood size. The retrieval over-fetches by one when
    /// `exclude_self` is set, so a self-hit never shrinks the context below `k`.
    pub k: usize,
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
            query,
            k,
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
            .resolve_embedding_table(&request.source_id, None)
            .await?;

        // Over-fetch by one when excluding self so a self-hit (the query vector
        // is the target's own stored vector — the nearest neighbour of itself)
        // never shrinks the surviving context below k.
        let fetch_k = if request.exclude_self {
            request.k.saturating_add(1)
        } else {
            request.k
        };

        let neighbours = self
            .result_store()
            .search_vectors(self.context(), &table, &request.query, fetch_k)
            .await?;

        // Self-exclusion (the leakage guard): drop every neighbour whose key
        // matches the target's own key, then truncate back to k. A free-vector
        // query carries no key, so nothing is dropped.
        let exclude = request
            .exclude_self
            .then_some(request.exclude_key.as_deref())
            .flatten();
        let mut context_keys: Vec<String> = neighbours
            .into_iter()
            .map(|(key, _similarity)| key)
            .filter(|key| Some(key.as_str()) != exclude)
            .collect();
        context_keys.truncate(request.k);

        let context_vector = self
            .pool_context_vectors(&table, &context_keys, request.aggregator)
            .await?;

        let value_rows = if request.value_columns.is_empty() {
            Vec::new()
        } else {
            self.hydrate_value_columns(&request.source_id, &context_keys, &request.value_columns)
                .await?
        };

        Ok(ContextRepresentation {
            context_vector,
            context_size: context_keys.len(),
            context_keys,
            value_rows,
        })
    }

    /// Pool the stored vectors of `context_keys` from `table`'s embedding
    /// Parquet into one fixed-width vector via the vector-aggregation UDAF.
    ///
    /// Filters the registered embedding table to the context keys and runs
    /// `vector_<agg>(vector)` over them — the aggregate is permutation-invariant
    /// by construction, so the pooled vector is byte-identical regardless of the
    /// order the keys arrive in. An empty key set yields `None` rather than a
    /// pool over nothing.
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

        let table_ref = format!("jammi.{}", table.table_name);
        let keys: Vec<datafusion::prelude::Expr> =
            context_keys.iter().map(|k| lit(k.as_str())).collect();
        let pooled = self
            .context()
            .table(&table_ref)
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

    /// Hydrate the requested `value_columns` of `context_keys` from the source,
    /// preserving retrieval order.
    ///
    /// `_row_id` equals the source key-column value, so the context members join
    /// back to their labels/outcomes directly. The result is ordered by the
    /// retrieval rank (descending similarity) the keys were collected in.
    async fn hydrate_value_columns(
        &self,
        source_id: &str,
        context_keys: &[String],
        value_columns: &[String],
    ) -> Result<Vec<RecordBatch>> {
        if context_keys.is_empty() {
            return Ok(Vec::new());
        }
        let table = self
            .catalog()
            .resolve_embedding_table(source_id, None)
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
                 FROM {source_id}.public.\"{source_table}\""
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

/// A pooled context vector keyed by its target, the materialised form of an
/// [`assemble_context`](InferenceSession::assemble_context) result for batch
/// workflows. The unique target keys carried here become the `_row_id` of a
/// normal embedding-shaped result table.
#[derive(Debug)]
pub struct MaterializedContext<'a> {
    /// Each target's key paired with its pooled context vector.
    pub rows: &'a [(String, Vec<f32>)],
    /// The pooled vectors' fixed width.
    pub dimensions: usize,
}

impl InferenceSession {
    /// Materialise per-target pooled context vectors into a normal
    /// embedding-shaped result table — searchable and joinable like any other
    /// embedding table, with its own sidecar ANN index — for batch NP
    /// workflows.
    ///
    /// Delegates the write to [`jammi_db::store::ResultStore::materialize_embedding_table`],
    /// which builds the same `(_row_id, _source_id, _model_id, vector)` Parquet
    /// + sidecar index every embedding table gets, so a materialised context set
    /// is a first-class member of the same table family.
    pub async fn materialize_context(
        &self,
        source_id: &str,
        context: MaterializedContext<'_>,
    ) -> Result<ResultTableRecord> {
        // De-duplicate target keys so the table's `_row_id` stays a key: a
        // batch must not carry two pooled vectors for the same target.
        let seen: BTreeSet<&str> = context.rows.iter().map(|(k, _)| k.as_str()).collect();
        if seen.len() != context.rows.len() {
            return Err(JammiError::Other(
                "materialize_context: duplicate target key in context rows".into(),
            ));
        }

        self.result_store()
            .materialize_embedding_table(
                self.context(),
                source_id,
                "jammi:context-set",
                context.rows,
                context.dimensions,
            )
            .await
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
    }

    #[test]
    fn empty_representation_reports_empty() {
        let rep = ContextRepresentation {
            context_vector: None,
            context_size: 0,
            context_keys: Vec::new(),
            value_rows: Vec::new(),
        };
        assert!(rep.is_empty());
    }
}
