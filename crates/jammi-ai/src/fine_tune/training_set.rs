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
//! (`pipeline/embedding.rs`). This module's one producer arm reads one such
//! source and records one anchor. The engine's reuse probe never matches an
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
//!
//! # The graph arm does not go through this module
//!
//! A graph fine-tune's sampled pairs are the output of a deterministic biased
//! walk, not of a query the engine can express as durable SQL, and re-running
//! that walk from the recorded spec is exactly `reconstruct_graph_loader`'s
//! job (`worker.rs`). They are sampled in memory and trained on directly, the
//! way `origin/main` always did it; no `TrainingSet` table is written, and
//! nothing here registers or deregisters a relation on the shared session for
//! them. Giving the graph arm a table of its own is
//! <https://github.com/f-inverse/jammi-ai/issues/538>.

use arrow::array::RecordBatch;
use jammi_db::error::Result;
use jammi_db::sql::{quote_ident, source_relation};
use jammi_db::store::manifest::InputAnchor;
use jammi_db::store::{training_set_order_by, TrainingSetSpec, TrainingSetTable};

use crate::model::ModelTask;
use crate::session::InferenceSession;

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
/// The read runs on the SAME `SessionContext` the verb was handed: the verb
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
