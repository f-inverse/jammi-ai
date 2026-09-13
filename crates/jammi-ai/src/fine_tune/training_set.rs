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
struct RegisteredPairs {
    ctx: SessionContext,
    relation: String,
    /// Whatever `register_table` displaced when this guard's relation was
    /// bound — `None` on the ordinary path (nothing occupied the
    /// content-addressed name before us; this is what a sequential
    /// registration under an occupied name actually returns too, since
    /// DataFusion's default schema provider REFUSES a sequential duplicate
    /// with `Err` rather than displacing it). `Some` means an overlapping
    /// materialization (the SAME `spec_identity` racing this one, or a
    /// coincidental name collision) won a genuine check-then-insert race and
    /// displaced an existing binding; `Drop` restores exactly that occupant
    /// rather than deregistering outright, so this guard undoes only what ITS
    /// OWN registration did.
    displaced: Option<Arc<dyn datafusion::datasource::TableProvider>>,
}

impl Drop for RegisteredPairs {
    fn drop(&mut self) {
        if let Some(provider) = self.displaced.take() {
            // An overlap: restore the prior occupant so the concurrent
            // materialization that is still reading it does not lose its
            // relation out from under it.
            if let Err(e) = self.ctx.register_table(self.relation.as_str(), provider) {
                tracing::warn!(
                    relation = %self.relation,
                    error = %e,
                    "could not restore the sampled-pair relation this materialization displaced"
                );
            }
            return;
        }
        // The ordinary path: nothing occupied this name before us, so our own
        // registration is the only thing to remove.
        match self.ctx.deregister_table(self.relation.as_str()) {
            Ok(Some(_)) => {}
            Ok(None) => {
                // Nothing was bound here at drop time even though nothing
                // occupied the name when WE registered: some other
                // registration under this identical name displaced ours
                // in between (an overlap this guard did not observe because
                // it only knows what it itself displaced). Not actionable
                // here, but distinct from a clean deregister — worth its own
                // line so the overlap is visible in the log.
                tracing::warn!(
                    relation = %self.relation,
                    "sampled-pair relation was already unbound at drop time \
                     (displaced by another registration under the same name)"
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
    let displaced = ctx
        .register_table(relation.as_str(), Arc::new(provider))
        .map_err(JammiError::from)?;
    let _registered = RegisteredPairs {
        ctx: ctx.clone(),
        relation: relation.clone(),
        displaced,
    };

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

    /// The fold's overlap probe, job-level: a mechanism to force two
    /// `materialize_sampled_pairs` calls to interleave their register/drop on
    /// the SAME relation name (e.g. a `run_job_now` that dispatches a claimed
    /// job synchronously) does not exist anywhere in this codebase — `grep
    /// -rn run_job_now` across the whole workspace has no hits. The two-worker
    /// concurrency infrastructure in `jobs_cancel.rs`/`jobs_shutdown.rs`
    /// targets job-CLAIM races (two workers racing `claim_next`), not this
    /// in-process table-registration race, and this contract does not
    /// authorize adding a new test-hook seam to build one. **UNCOVERED** at
    /// the job level, for that reason.
    ///
    /// What IS constructible, and exercised below instead: [`RegisteredPairs`]'s
    /// drop semantics, driven directly and deterministically — no timing
    /// window to hit, and no test hook to add. Two guards over the SAME
    /// relation name (the overlap the type exists to survive) must restore
    /// the first occupant when the second one drops, and the first guard
    /// (which displaced nothing) must fully remove the binding when it drops
    /// last.
    ///
    /// The state a genuine overlap leaves behind (B's provider bound, A's
    /// provider the value `register_table` handed back as `displaced`) is
    /// constructed directly rather than produced by calling `register_table`
    /// twice in sequence: DataFusion's own `MemorySchemaProvider` checks
    /// `table_exist` before inserting and REFUSES a sequential duplicate with
    /// `Err("... already exists")` — confirmed below — so the `Ok(Some(_))`
    /// displacement arm this guard defends against is reachable only through
    /// a genuine check-then-insert TOCTOU race between two REAL concurrent
    /// threads, which a single-threaded test cannot reproduce by calling the
    /// same function twice.
    #[tokio::test]
    async fn registered_pairs_drop_restores_a_displaced_overlap_and_fully_removes_the_last() {
        let ctx = SessionContext::new();
        let relation = "jammi_sampled_pairs:overlap-probe";

        let a_provider = tiny_table("first");
        let first_displaced = ctx
            .register_table(relation, Arc::clone(&a_provider))
            .unwrap();
        assert!(
            first_displaced.is_none(),
            "nothing should occupy the name before A's registration"
        );

        // A sequential second registration under the SAME name is refused,
        // never a silent displacement — the executed refutation of "this
        // guard's `Some` arm is reachable by calling `register_table` twice".
        let sequential_duplicate = ctx.register_table(relation, tiny_table("second"));
        assert!(
            sequential_duplicate.is_err(),
            "a sequential duplicate registration must be refused, not silently \
             displace — got {sequential_duplicate:?}"
        );

        // Construct the state the genuine (thread-race) overlap would leave:
        // B's provider now bound in the ctx, A's provider recorded as what
        // B's registration displaced.
        ctx.deregister_table(relation).unwrap();
        ctx.register_table(relation, tiny_table("second")).unwrap();
        let a = RegisteredPairs {
            ctx: ctx.clone(),
            relation: relation.to_string(),
            displaced: None,
        };
        let b = RegisteredPairs {
            ctx: ctx.clone(),
            relation: relation.to_string(),
            displaced: Some(a_provider),
        };

        // B finishes first: its drop must RESTORE A's provider, not remove
        // the binding outright — A may still be reading it.
        drop(b);
        assert!(
            ctx.table_exist(relation).unwrap(),
            "B's drop must restore the displaced provider, not remove the binding"
        );

        // A finishes last: nothing occupied the name when A registered
        // (`displaced: None`), so A's drop is the ordinary path and fully
        // removes the binding.
        drop(a);
        assert!(
            !ctx.table_exist(relation).unwrap(),
            "the last guard's drop must fully remove the relation"
        );
    }

    /// The ordinary, non-overlapping path: a single guard's drop removes
    /// exactly the relation it registered, and `deregister_table`'s `Ok(None)`
    /// arm (nothing bound at drop time) is reachable without a panic when a
    /// THIRD party removed the binding first — the distinct log line this
    /// fold added, exercised for compile/run correctness rather than log
    /// content (no test harness here captures `tracing` output).
    #[tokio::test]
    async fn registered_pairs_drop_tolerates_a_binding_already_removed() {
        let ctx = SessionContext::new();
        let relation = "jammi_sampled_pairs:already-gone-probe";
        let displaced = ctx.register_table(relation, tiny_table("only")).unwrap();
        assert!(displaced.is_none());
        let guard = RegisteredPairs {
            ctx: ctx.clone(),
            relation: relation.to_string(),
            displaced,
        };
        // Something else removes the binding before the guard drops.
        ctx.deregister_table(relation).unwrap();
        assert!(!ctx.table_exist(relation).unwrap());
        drop(guard); // Must not panic on `Ok(None)`.
        assert!(!ctx.table_exist(relation).unwrap());
    }
}
