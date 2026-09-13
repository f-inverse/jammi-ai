//! The rows a fine-tune trains on, as a **producer output** rather than a run's
//! scratch space.
//!
//! A training run no longer re-runs its source query into memory. It
//! materialises the projected rows once, through
//! [`ResultStore::materialize_training_set`](jammi_db::store::ResultStore::materialize_training_set), into an immutable Parquet result
//! table of kind `TrainingSet` carrying a definition hash and the standard
//! manifest attestation, and then reads that table back. Two runs over the same
//! source query, columns, task and format name the same table — whatever their
//! world size, batch or validation split, none of which enter the table's
//! identity.
//!
//! # The order contract has two halves, and this module owns the reader's
//!
//! The producer commits the rows sorted by the **full projected tuple**
//! ([`TRAINING_SET_ORDER_RULE_V1`](jammi_db::store::manifest::TRAINING_SET_ORDER_RULE_V1)). A Parquet scan gives no row-order
//! guarantee: the table is written with 64K row groups and the session plans at
//! `[engine] execution_threads` partitions, so a table larger than one row group
//! comes back interleaved unless the reader asks for the order. [`read_back_sql`]
//! is the one place that asks, and it renders the key through
//! [`training_set_order_by`] rather than hand-writing it — a reader that spelled
//! the direction or the NULL placement differently would read rows in an order
//! the table's own descriptor does not claim, and nothing would report it.
//!
//! # Anchors
//!
//! A registered source exposes no version or digest surface, so its rows are
//! anchored [`AnchorKind::UnpinnedAtInstant`](jammi_db::store::manifest::AnchorKind::UnpinnedAtInstant)
//! — the same honest anchor the embedding producer records for the same reason
//! (`pipeline/embedding.rs`). The tabular arm reads one such source and records
//! one anchor; the graph arm's sampled pairs are a function of BOTH the node
//! source and the edge source, so it records BOTH as separate anchors sharing
//! ONE read instant (see `materialize_sampled_pairs`'s own "Anchors" section)
//! — the two-anchor, one-instant shape `pipeline/asof/verb.rs` uses for its
//! spine and facts relations. The engine's reuse probe never matches an
//! unpinned anchor, so a training set
//! over a plain source is never reused across runs; the anchor still rides the
//! manifest, so staleness reports the same honest `Undecidable` it reports for
//! every unpinned input. The engine does own a resolver from a relation name to
//! a `ready` result table's content digest
//! (`Catalog::get_result_table` → `ResultStore::pin_current_version`), but no
//! fine-tune source can reach it: a source is resolved through
//! `SessionContext::catalog(source_id)`, and a result table is registered as the
//! BARE table `jammi.{name}` in the default catalog, never as a catalog of its
//! own. That path is therefore uncovered here rather than speculatively wired —
//! see `training_set::a_result_table_cannot_be_a_fine_tune_source` in the
//! integration suite for the executed probe.

use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use jammi_db::error::{JammiError, Result};
use jammi_db::sql::{quote_ident, source_relation};
use jammi_db::store::manifest::InputAnchor;
use jammi_db::store::{training_set_order_by, TrainingSetSpec, TrainingSetTable};

use super::graph_sampler::SampledPair;
use crate::model::ModelTask;
use crate::session::InferenceSession;

/// An Arrow error on the in-memory pair relation, as a `JammiError`.
///
/// `JammiError` has no `From<ArrowError>` (Arrow errors reach it through
/// DataFusion), so the box goes through `DataFusionError::ArrowError` — the
/// same route `pipeline/asof` takes — rather than being flattened to a string.
fn arrow_err(e: arrow::error::ArrowError) -> JammiError {
    JammiError::from(datafusion::error::DataFusionError::ArrowError(
        Box::new(e),
        None,
    ))
}

/// The `anchor` / `positive` / `negative` column names a graph fine-tune's
/// sampled pairs are projected under, in declared order. Also the full-tuple
/// order key, so this order is output-affecting.
const GRAPH_PAIR_COLUMNS: [&str; 3] = ["anchor", "positive", "negative"];

/// The SQL that reads a materialised training set back in its **committed
/// order** — the reader's half of the order contract, in one place.
///
/// `SELECT *` over the registered table (its schema is exactly the projection,
/// in declared order) with [`training_set_order_by`] re-applied. The relation is
/// spelled with [`TrainingSetTable::sql_relation`]: a result-table name carries
/// hyphens (a sanitized model id) and dots (a nanosecond timestamp), so the
/// unquoted form re-parses as arithmetic and as a multi-part reference — never
/// the table.
pub fn read_back_sql(table: &TrainingSetTable, columns: &[String]) -> String {
    format!(
        "SELECT * FROM {} {}",
        table.sql_relation(),
        training_set_order_by(columns)
    )
}

/// Materialise `columns` of a registered `source` as a training set, then read
/// the committed rows back in order.
///
/// The projection runs unordered — the producer owns the sort, and asking the
/// source for an order it is about to re-impose would only plan the sort twice.
pub async fn materialize_projection(
    session: &InferenceSession,
    source_id: &str,
    columns: &[String],
    task: ModelTask,
    format: &str,
) -> Result<(TrainingSetTable, Vec<RecordBatch>)> {
    let table_name = session.find_table_name(source_id)?;
    let projection = columns
        .iter()
        .map(|c| quote_ident(c))
        .collect::<Vec<_>>()
        .join(", ");
    let source_sql = format!(
        "SELECT {projection} FROM {}",
        source_relation(source_id, &table_name)
    );
    materialize_and_read(
        session,
        TrainingSetSpec {
            source_id,
            source_sql: &source_sql,
            columns,
            task,
            format,
            // The source has no version surface to pin, so it is anchored at
            // the instant it was read — the same honest anchor the embedding
            // producer records for the same reason, constructed here rather
            // than behind a helper so the anchor value never travels apart
            // from the read it describes.
            inputs: vec![InputAnchor::unpinned_at_instant(
                source_id,
                chrono::Utc::now().to_rfc3339(),
            )],
            device: session.compute_device(),
        },
    )
    .await
}

/// Materialise a spec through the producer and read the table back in its
/// committed order.
///
/// The read runs on the SAME [`SessionContext`] the verb was handed: the verb
/// binds the table there on both the computed and the reused path, so the
/// read-back resolves without a second registration and a reused table is read
/// exactly like a fresh one.
async fn materialize_and_read(
    session: &InferenceSession,
    spec: TrainingSetSpec<'_>,
) -> Result<(TrainingSetTable, Vec<RecordBatch>)> {
    let columns = spec.columns.to_vec();
    let table = session
        .result_store()
        .materialize_training_set(session.context(), spec)
        .await?;
    let batches = session.sql(&read_back_sql(&table, &columns)).await?;
    Ok((table, batches))
}

/// A [`MemTable`] registered on a session for the length of one
/// materialization, deregistered on drop.
///
/// The sampled pairs are already resident as `Vec<SampledPair>`; the MemTable is
/// a second resident copy of the same rows, and it exists only so the producer
/// — whose input is a SQL relation — can plan over them. Holding it past the
/// write would leave one such copy per distinct graph spec resident for the
/// life of the session, so the registration is scoped to the call that needs it.
///
/// There is no "displaced occupant" to restore on drop: DataFusion 54.1's
/// [`MemorySchemaProvider::register_table`](https://docs.rs/datafusion-catalog/54.1.0/datafusion_catalog/memory/struct.MemorySchemaProvider.html#method.register_table)
/// checks `table_exist` and refuses (`Err`) rather than displacing whenever the
/// name is already bound, so `register_table` never returns `Ok(Some(_))` for
/// an occupied relation — see [`register_pairs_relation`], which turns that
/// refusal into a typed error before this guard is ever constructed. This
/// guard only ever removes the ONE binding it created.
struct RegisteredPairs {
    ctx: SessionContext,
    relation: String,
}

impl Drop for RegisteredPairs {
    fn drop(&mut self) {
        match self.ctx.deregister_table(self.relation.as_str()) {
            Ok(Some(_)) => {}
            Ok(None) => {
                // Something else removed the binding before we got to it (a
                // session-level cleanup, or a bug elsewhere) — not actionable
                // here, but distinct from a clean deregister so it is visible
                // in the log rather than silently swallowed.
                tracing::warn!(
                    relation = %self.relation,
                    "sampled-pair relation was already unbound at drop time"
                );
            }
            Err(e) => {
                tracing::warn!(
                    relation = %self.relation,
                    error = %e,
                    "sampled-pair relation could not be deregistered; its rows stay resident"
                );
            }
        }
    }
}

/// Register `provider` under `relation` on `ctx`, guarded so the binding is
/// released when the returned [`RegisteredPairs`] drops.
///
/// A relation name is the spec's identity (source + sample config), so two
/// concurrent materializations of the identical graph spec on the SAME session
/// contend for the SAME name. DataFusion's schema provider refuses the second
/// registration outright (see [`RegisteredPairs`]'s doc) rather than displacing
/// the first — this call turns that refusal into a typed [`JammiError::FineTune`]
/// naming the relation, raised here before any row is written, rather than
/// treating the overlap as a race to survive.
fn register_pairs_relation(
    ctx: &SessionContext,
    relation: String,
    provider: Arc<dyn datafusion::datasource::TableProvider>,
) -> Result<RegisteredPairs> {
    ctx.register_table(relation.as_str(), provider)
        .map_err(|e| {
            JammiError::FineTune(format!(
                "a graph training set is already materializing sampled pairs under \
                 relation {relation} on this session: {e}"
            ))
        })?;
    Ok(RegisteredPairs {
        ctx: ctx.clone(),
        relation,
    })
}

/// Materialise a graph fine-tune's seeded sampled pairs as a training set,
/// through the SAME producer the tabular path uses, then read them back in
/// committed order.
///
/// The pairs are the output of a deterministic biased walk, not of a query, so
/// they are bound to the session as an in-memory relation for the length of the
/// materialization and the producer plans over that. The relation's NAME is the
/// canonical JSON of the graph sources and the sample config — not a per-run
/// id — because the relation name is the `source` the definition hash folds: a
/// per-run id would make two identical graph fine-tunes two different tables
/// while a config the name omitted would make two different ones collide.
///
/// **This table is not replayable in a later session.** Its recorded source
/// names a relation that lives only while the materialization runs, so a
/// `recompute` of it fails at the planner naming the missing relation. That is
/// the honest consequence of sampling being a producer the engine cannot express
/// as SQL; nothing here reports a replay that did not happen.
///
/// # Anchors
///
/// The pairs are a function of BOTH the node source and the edge source (the
/// sampler walks the edges to pick positives, and every anchor/positive text
/// is read off the nodes), so both are recorded as inputs — never just the
/// one this call happens to use for the catalog row's `source_id` lineage
/// column. Neither exposes a version surface, so each is honestly anchored
/// [`AnchorKind::UnpinnedAtInstant`](jammi_db::store::manifest::AnchorKind::UnpinnedAtInstant)
/// at ONE shared read instant — the same two-anchor, one-instant shape
/// `pipeline/asof/verb.rs` records for its spine and facts relations.
pub(crate) async fn materialize_sampled_pairs(
    session: &InferenceSession,
    node_source: &str,
    edge_source: &str,
    spec_identity: &str,
    pairs: &[SampledPair],
    has_negatives: bool,
) -> Result<(TrainingSetTable, Vec<RecordBatch>)> {
    let columns: Vec<String> = GRAPH_PAIR_COLUMNS
        .iter()
        .take(if has_negatives { 3 } else { 2 })
        .map(|c| (*c).to_string())
        .collect();

    let mut anchors = Vec::with_capacity(pairs.len());
    let mut positives = Vec::with_capacity(pairs.len());
    let mut negatives = Vec::with_capacity(pairs.len());
    for pair in pairs {
        anchors.push(pair.anchor.clone());
        positives.push(pair.positive.clone());
        if has_negatives {
            // The uniform-shape contract of the sampler: `has_negatives` was
            // read off the pair set, so every pair carries at least one. A pair
            // that does not is a broken sampler invariant, reported rather than
            // padded with an empty string (which would train the model to pull
            // away from "").
            let negative = pair.hard_negatives.first().ok_or_else(|| {
                JammiError::FineTune(
                    "graph pair set declares mined hard negatives but a pair supplied none".into(),
                )
            })?;
            negatives.push(negative.clone());
        }
    }

    let mut fields = vec![
        Field::new("anchor", DataType::Utf8, false),
        Field::new("positive", DataType::Utf8, false),
    ];
    let mut arrays: Vec<arrow::array::ArrayRef> = vec![
        Arc::new(StringArray::from(anchors)),
        Arc::new(StringArray::from(positives)),
    ];
    if has_negatives {
        fields.push(Field::new("negative", DataType::Utf8, false));
        arrays.push(Arc::new(StringArray::from(negatives)));
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).map_err(arrow_err)?;
    let provider = MemTable::try_new(schema, vec![vec![batch]]).map_err(JammiError::from)?;

    let relation = format!("jammi_sampled_pairs:{spec_identity}");
    let ctx = session.context();
    let _registered = register_pairs_relation(ctx, relation.clone(), Arc::new(provider))?;

    let projection = columns
        .iter()
        .map(|c| quote_ident(c))
        .collect::<Vec<_>>()
        .join(", ");
    let source_sql = format!("SELECT {projection} FROM {}", quote_ident(&relation));
    // Both inputs the sampled pairs are a function of, anchored at the SAME
    // read instant — see the "Anchors" section above.
    let now = chrono::Utc::now().to_rfc3339();
    let inputs = vec![
        InputAnchor::unpinned_at_instant(node_source, now.clone()),
        InputAnchor::unpinned_at_instant(edge_source, now),
    ];
    materialize_and_read(
        session,
        TrainingSetSpec {
            source_id: node_source,
            source_sql: &source_sql,
            columns: &columns,
            task: ModelTask::TextEmbedding,
            format: if has_negatives {
                super::data::TrainingFormat::Graph {
                    has_negatives: true,
                }
            } else {
                super::data::TrainingFormat::Graph {
                    has_negatives: false,
                }
            }
            .format_tag(),
            inputs,
            device: session.compute_device(),
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_table(value: &str) -> Arc<dyn datafusion::datasource::TableProvider> {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringArray::from(vec![value.to_string()])) as arrow::array::ArrayRef],
        )
        .unwrap();
        Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())
    }

    /// Two overlapping registrations of one spec identity in one session: the
    /// second is a typed refusal naming the relation, raised by
    /// [`register_pairs_relation`] before any row is written — never a silent
    /// displace-and-restore. DataFusion's `MemorySchemaProvider::register_table`
    /// (54.1.0, confirmed by reading
    /// `datafusion-catalog-54.1.0/src/memory/schema.rs:63-72`) checks
    /// `table_exist` and returns `exec_err!` for an occupied name rather than
    /// ever returning `Ok(Some(previous))`, so there is no "displaced occupant"
    /// state to construct or restore; the first guard's own registration is the
    /// only thing standing in the second call's way.
    #[tokio::test]
    async fn registering_sampled_pairs_over_an_occupied_relation_is_a_typed_refusal() {
        let ctx = SessionContext::new();
        let relation = "jammi_sampled_pairs:overlap-probe".to_string();

        let _first = register_pairs_relation(&ctx, relation.clone(), tiny_table("first"))
            .expect("the first registration over an unoccupied name succeeds");

        let second = register_pairs_relation(&ctx, relation.clone(), tiny_table("second"));
        match second {
            Err(JammiError::FineTune(message)) => {
                assert!(
                    message.contains(&relation),
                    "the refusal must name the occupied relation: {message}"
                );
            }
            Err(other) => panic!(
                "expected a typed FineTune refusal naming {relation}, got a different error variant: {other:?}"
            ),
            Ok(_) => panic!(
                "expected the second registration over {relation} to be refused, but it succeeded"
            ),
        }
    }

    /// A single guard's drop removes exactly the ONE binding it created — the
    /// binding bound under `ctx` before the drop is this guard's own provider
    /// (`Arc::ptr_eq`, not just "something is bound"), and the relation is gone
    /// after.
    #[tokio::test]
    async fn registered_pairs_drop_removes_exactly_its_own_binding() {
        let ctx = SessionContext::new();
        let relation = "jammi_sampled_pairs:drop-probe".to_string();
        let provider = tiny_table("only");

        let guard = register_pairs_relation(&ctx, relation.clone(), Arc::clone(&provider))
            .expect("registering over an unoccupied name succeeds");

        let bound_before = ctx
            .table_provider(relation.as_str())
            .await
            .expect("the relation is bound before the drop");
        assert!(
            Arc::ptr_eq(&bound_before, &provider),
            "the bound provider must be this guard's own, by identity"
        );

        drop(guard);
        assert!(
            !ctx.table_exist(&relation).unwrap(),
            "the guard's drop must remove the relation it registered"
        );
    }
}
