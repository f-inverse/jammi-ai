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

/// The per-job-unique relation a graph fine-tune's sampled pairs are bound to
/// for the length of one materialization: the spec identity (source + sample
/// config) plus the claiming job's id.
///
/// Per-job — never spec-only — because a real session's
/// [`ResultTableSchemaProvider`](jammi_db::store::ResultTableSchemaProvider)'s
/// `register_table` inserts UNCONDITIONALLY and hands back whatever it displaced; it never
/// refuses an occupied name the way DataFusion's in-memory
/// `MemorySchemaProvider::register_table` (the provider behind
/// `SessionContext::new()`, confirmed by reading
/// `datafusion-catalog-54.1.0/src/memory/schema.rs:63-72`) does. Two
/// concurrent materializations of the SAME graph spec — two jobs sharing one
/// definition — therefore cannot be kept from contending by any occupancy
/// check on a real session: the second call's registration would silently
/// take over the first's relation, and whichever call finishes (and
/// deregisters) first would tear the name out from under the other. Naming
/// each job's relation uniquely removes the contention instead of trying to
/// detect it after the fact.
fn pairs_relation_name(spec_identity: &str, job_id: &str) -> String {
    format!("jammi_sampled_pairs:{spec_identity}:{job_id}")
}

/// A relation registered on a session for the length of one materialization,
/// deregistered by [`Drop`] on every exit — success, an error propagated
/// through `?`, or the owning task being cancelled while parked mid-`.await`
/// (Rust drops a future's live locals on cancellation exactly as it does on a
/// normal return). This guard owns nothing but the ONE name it registered;
/// with relations named per-job ([`pairs_relation_name`]) there is no
/// "occupied name" case left to refuse, so cleanup is its only job.
struct DeregisterOnDrop {
    ctx: SessionContext,
    relation: String,
}

impl Drop for DeregisterOnDrop {
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

/// Test-only pause inside [`materialize_sampled_pairs`], between registering
/// the sampled-pair relation and running the materialization plan over it —
/// exactly the window a same-named registration from a concurrent call could
/// once step on, before per-job naming made that structurally impossible. No
/// production path observes anything here beyond the `maybe_park` call, which
/// returns at once when nothing is armed for the relation. One-shot per
/// relation, mirroring `fine_tune::worker::loop_test_hooks`'s park/release
/// shape.
#[cfg(feature = "test-hooks")]
pub mod graph_materialize_test_hooks {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};

    use tokio::sync::Notify;

    struct Armed {
        relation: String,
        parked: Arc<AtomicBool>,
        parked_notify: Arc<Notify>,
        released: Arc<AtomicBool>,
        release_notify: Arc<Notify>,
    }

    fn armed() -> &'static Mutex<Vec<Armed>> {
        static ARMED: OnceLock<Mutex<Vec<Armed>>> = OnceLock::new();
        ARMED.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// The test's side of one armed park: wait for the call to arrive, then
    /// let it continue. Dropping the handle without releasing leaves the call
    /// parked forever — release it explicitly.
    pub struct ParkHandle {
        parked: Arc<AtomicBool>,
        parked_notify: Arc<Notify>,
        released: Arc<AtomicBool>,
        release_notify: Arc<Notify>,
    }

    impl ParkHandle {
        /// Resolve once the call has reached the park point.
        pub async fn wait_parked(&self) {
            while !self.parked.load(Ordering::SeqCst) {
                self.parked_notify.notified().await;
            }
        }

        /// Let the parked call continue.
        pub fn release(&self) {
            self.released.store(true, Ordering::SeqCst);
            self.release_notify.notify_one();
        }
    }

    /// Arm one park for the next materialization that registers `relation`.
    /// One-shot: the park disarms as soon as it is taken.
    pub fn arm(relation: &str) -> ParkHandle {
        let parked = Arc::new(AtomicBool::new(false));
        let parked_notify = Arc::new(Notify::new());
        let released = Arc::new(AtomicBool::new(false));
        let release_notify = Arc::new(Notify::new());
        armed()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Armed {
                relation: relation.to_string(),
                parked: Arc::clone(&parked),
                parked_notify: Arc::clone(&parked_notify),
                released: Arc::clone(&released),
                release_notify: Arc::clone(&release_notify),
            });
        ParkHandle {
            parked,
            parked_notify,
            released,
            release_notify,
        }
    }

    pub(super) async fn maybe_park(relation: &str) {
        let taken = {
            let mut list = armed().lock().unwrap_or_else(PoisonError::into_inner);
            list.iter()
                .position(|a| a.relation == relation)
                .map(|i| list.remove(i))
        };
        let Some(armed) = taken else {
            return;
        };
        armed.parked.store(true, Ordering::SeqCst);
        armed.parked_notify.notify_one();
        while !armed.released.load(Ordering::SeqCst) {
            armed.release_notify.notified().await;
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
/// canonical JSON of the graph sources and the sample config PLUS `job_id`
/// ([`pairs_relation_name`]) — never spec-only, so two jobs racing the
/// identical graph spec cannot contend for one name (see that function's
/// doc). That name is embedded in the recorded `source` SQL this call passes
/// to the producer, and `source` is what the definition hash folds — so,
/// honestly, two otherwise-identical graph fine-tunes now record two
/// different definition hashes. This is not a regression to paper over: a
/// graph training set was already per-job in every way that matters (its
/// anchors are [`AnchorKind::UnpinnedAtInstant`], never reused, never served
/// to a later run), and the executed SQL must equal the recorded SQL — a
/// canonical name substituted into the recording but not into execution (or
/// vice versa) would make the manifest a lie about what actually ran.
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
    job_id: &str,
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

    let relation = pairs_relation_name(spec_identity, job_id);
    let ctx = session.context();
    ctx.register_table(relation.as_str(), Arc::new(provider))
        .map_err(JammiError::from)?;
    let _guard = DeregisterOnDrop {
        ctx: ctx.clone(),
        relation: relation.clone(),
    };
    #[cfg(feature = "test-hooks")]
    graph_materialize_test_hooks::maybe_park(&relation).await;

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
    use arrow::array::Array;

    /// A real [`InferenceSession`] — the fixture a production job actually
    /// runs on, whose default schema is [`jammi_db::store::ResultTableSchemaProvider`]
    /// (`session.rs`'s `wrap_with` installs it before any table is loaded),
    /// never DataFusion's `MemorySchemaProvider` behind a bare
    /// `SessionContext::new()`. The two providers disagree on exactly the
    /// property this module's naming scheme depends on (see
    /// [`pairs_relation_name`]'s doc), so a test standing in for production
    /// behaviour has to be built on this one.
    async fn real_session(dir: &tempfile::TempDir) -> Arc<InferenceSession> {
        Arc::new(
            InferenceSession::new(jammi_test_utils::test_config(dir.path()))
                .await
                .unwrap(),
        )
    }

    fn pair(anchor: &str, positive: &str) -> SampledPair {
        SampledPair {
            anchor: anchor.to_string(),
            positive: positive.to_string(),
            hard_negatives: Vec::new(),
        }
    }

    /// The distinct text values a batch set's `column` column carries, in
    /// batch/row order — used to tell "these are call A's rows" from "these
    /// are call B's rows" without trusting row count alone.
    fn column_values(batches: &[RecordBatch], column: &str) -> Vec<String> {
        use arrow::array::{LargeStringArray, StringViewArray};

        let mut values = Vec::new();
        for batch in batches {
            let array = batch
                .column_by_name(column)
                .unwrap_or_else(|| panic!("column {column} is present"));
            if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
                values.extend((0..a.len()).map(|i| a.value(i).to_string()));
            } else if let Some(a) = array.as_any().downcast_ref::<StringViewArray>() {
                values.extend((0..a.len()).map(|i| a.value(i).to_string()));
            } else if let Some(a) = array.as_any().downcast_ref::<LargeStringArray>() {
                values.extend((0..a.len()).map(|i| a.value(i).to_string()));
            } else {
                panic!(
                    "column {column} is none of Utf8/Utf8View/LargeUtf8, got {:?}",
                    array.data_type()
                );
            }
        }
        values
    }

    /// Oracle (1): two overlapping graph materializations of ONE spec
    /// identity (two jobs racing the identical graph fine-tune), sequenced by
    /// the test-hooks park so call A is parked between registering its
    /// relation and running its materialization plan while call B runs to
    /// completion. Each call must read back only its OWN rows, and neither
    /// call's relation must exist once both have returned.
    ///
    /// Before per-job naming (this module's predecessor, `RegisteredPairs` /
    /// `register_pairs_relation`), both calls shared ONE relation name (spec
    /// identity only) and a real session's `ResultTableSchemaProvider`
    /// (unlike the `MemorySchemaProvider` the deleted type's own tests used)
    /// never refuses the second registration — it silently displaces the
    /// first, so B's completion (which deregisters "the" relation) tears the
    /// name out from under A while A is still parked on it. Reproduced as a
    /// one-line revert of [`pairs_relation_name`] to `spec_identity` alone;
    /// see the fix round 3 report for the transcript.
    #[cfg(feature = "test-hooks")]
    #[tokio::test(flavor = "multi_thread")]
    async fn overlapping_graph_materializations_of_one_spec_read_back_their_own_rows() {
        let dir = tempfile::tempdir().unwrap();
        let session = real_session(&dir).await;

        let spec_identity = "shared-graph-spec";
        let relation_a = pairs_relation_name(spec_identity, "job-a");
        let relation_b = pairs_relation_name(spec_identity, "job-b");

        let pairs_a = vec![pair("a-anchor", "a-positive")];
        let pairs_b = vec![pair("b-anchor", "b-positive")];

        let park = graph_materialize_test_hooks::arm(&relation_a);

        let session_a = Arc::clone(&session);
        let handle_a = tokio::spawn(async move {
            materialize_sampled_pairs(
                &session_a,
                "nodes",
                "edges",
                spec_identity,
                "job-a",
                &pairs_a,
                false,
            )
            .await
        });

        park.wait_parked().await;

        let result_b = materialize_sampled_pairs(
            &session,
            "nodes",
            "edges",
            spec_identity,
            "job-b",
            &pairs_b,
            false,
        )
        .await;
        let (_table_b, batches_b) =
            result_b.expect("B's own materialization must succeed while A is parked");

        park.release();
        let (_table_a, batches_a) = handle_a.await.expect("A's task must not panic").expect(
            "A's own materialization must succeed once resumed, even though B raced it \
                 under the SAME spec identity",
        );

        assert_eq!(
            column_values(&batches_a, "anchor"),
            vec!["a-anchor".to_string()],
            "A must read back only its OWN rows, not B's"
        );
        assert_eq!(
            column_values(&batches_b, "anchor"),
            vec!["b-anchor".to_string()],
            "B must read back only its OWN rows, not A's"
        );

        assert!(
            !session.context().table_exist(&relation_a).unwrap(),
            "A's relation must not survive its own materialization"
        );
        assert!(
            !session.context().table_exist(&relation_b).unwrap(),
            "B's relation must not survive its own materialization"
        );
    }

    /// Oracle (2a): a materialization that registers its relation and then
    /// fails (K2's empty-training-set refusal, reached only after the
    /// relation is registered — `pairs` is empty, so the write side sees zero
    /// rows) still deregisters on the way out, via [`DeregisterOnDrop`]'s
    /// normal `?`-propagated drop. Already true of the deleted
    /// `RegisteredPairs` guard too (its `Drop` ran on any unwind); this
    /// re-pins the same property on the excised mechanism, on a real session.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_relation_is_gone_after_an_empty_training_set_refusal() {
        let dir = tempfile::tempdir().unwrap();
        let session = real_session(&dir).await;

        let relation = pairs_relation_name("empty-spec", "job-empty");
        let err = materialize_sampled_pairs(
            &session,
            "nodes",
            "edges",
            "empty-spec",
            "job-empty",
            &[],
            false,
        )
        .await
        .expect_err("zero sampled pairs must refuse as an empty training set, never a 0-row table");
        assert!(
            matches!(err, JammiError::EmptyTrainingSet { .. }),
            "expected the typed K2 refusal, got {err:?}"
        );
        assert!(
            !session.context().table_exist(&relation).unwrap(),
            "the relation registered before the refusal must not survive it"
        );
    }

    /// Oracle (2b): a materialization whose owning task is cancelled while
    /// parked mid-`.await` (a `run_claimed_job` future dropped, or its task
    /// aborted, the same action a job cancellation or a worker shutdown
    /// takes) still deregisters its relation — [`DeregisterOnDrop`]'s `Drop`
    /// runs on a future's live locals exactly as it does on a normal return.
    #[cfg(feature = "test-hooks")]
    #[tokio::test(flavor = "multi_thread")]
    async fn the_relation_is_gone_after_the_materialization_task_is_cancelled() {
        let dir = tempfile::tempdir().unwrap();
        let session = real_session(&dir).await;

        let spec_identity = "cancel-spec";
        let relation = pairs_relation_name(spec_identity, "job-cancel");
        let pairs = vec![pair("anchor", "positive")];

        let park = graph_materialize_test_hooks::arm(&relation);
        let session_c = Arc::clone(&session);
        let handle = tokio::spawn(async move {
            materialize_sampled_pairs(
                &session_c,
                "nodes",
                "edges",
                spec_identity,
                "job-cancel",
                &pairs,
                false,
            )
            .await
        });

        park.wait_parked().await;
        handle.abort();
        let _ = handle.await;

        assert!(
            !session.context().table_exist(&relation).unwrap(),
            "the relation must not survive its owning task being cancelled while it was live"
        );
    }
}
