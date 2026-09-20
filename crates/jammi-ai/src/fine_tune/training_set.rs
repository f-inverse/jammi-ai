//! The rows a fine-tune trains on, as a **producer output** rather than a run's
//! scratch space.
//!
//! A training run does not re-run its source query into memory. It
//! materialises the projected rows once, through
//! [`ResultStore::materialize_training_set`](jammi_db::store::ResultStore::materialize_training_set), into an immutable Parquet result
//! table of kind `TrainingSet` carrying a definition hash and the standard
//! manifest attestation, and then reads that table back. Two runs over the same
//! source query, columns, task and format name the same table — whatever their
//! world size, batch or validation split, none of which enter the table's
//! identity.
//!
//! # The order contract has two halves, and the type holds the reader's
//!
//! The producer commits the rows sorted by the **full projected tuple**
//! ([`TRAINING_SET_ORDER_RULE_V1`](jammi_db::store::manifest::TRAINING_SET_ORDER_RULE_V1)). A Parquet scan gives no row-order
//! guarantee: the table is written with 64K row groups and the session plans at
//! `[engine] execution_threads` partitions, so a table larger than one row group
//! comes back interleaved unless the reader asks for the order. Every read of
//! a training set's rows — the eager [`read_back`] here, the per-rank stream
//! and its load-time pre-pass and label vocabulary (`stream.rs`) — is a
//! [`TrainingSetTable::scan`](jammi_db::store::TrainingSetTable::scan): the
//! handle plans the sort from the table's OWN recorded order columns, the
//! sort is a node of the plan it returns, and the handle exposes no relation,
//! no SQL text and no unordered form, so a reader composes on the scan
//! (`collect`, `execute_stream`, a `limit`, an aggregate) and cannot spell
//! the direction, the NULL placement or the key any other way. The bare
//! catalog name (`table_name()`, or a `ResultTableRecord`'s) is the table's
//! identity, and a relation minted from it reads an arbitrary registered
//! result table under that relation's own contract — not a training set.
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
//! # The graph arm goes through the SAME funnel, over a DIFFERENT input seam
//!
//! A graph fine-tune's sampled pairs are the output of a deterministic biased
//! walk, not of a query the engine can express as durable SQL, so this
//! module's `training_set_spec`/`materialize_projection*` helpers (SQL-only)
//! are not it. Instead `worker.rs::materialize_graph_training_set` re-reads the
//! node/edge sources (ordered — `GRAPH_READ_ORDER_RULE_V1`), samples
//! them, and hands the sampled pairs to
//! [`ResultStore::materialize_training_set`](jammi_db::store::ResultStore::materialize_training_set)
//! as a [`TrainingSetInput::Batches`] stream (a leading `_ordinal` column,
//! never re-sorted — the producer's own committed order IS the
//! sampler's emission order) under a
//! [`jammi_db::store::manifest::ProducingDescriptor::GraphTrainingSet`]
//! descriptor, never this module's `ProducingDescriptor::TrainingSet`.
//! The written table is still `ResultTableKind::TrainingSet` — the SAME
//! catalog kind, registration, and reuse-probe machinery this module's arm
//! uses — and `pipeline/recompute.rs`'s `recompute_graph_training_set`
//! replays it through the identical shared sample-then-materialise
//! function a fresh run calls. A `Peer` gang does NOT admit a member
//! against this table: `run_spec` refuses a graph fine-tune's
//! `TopologyDecision::Peer` by name (`world_size` above `[worker]
//! local_ranks`) — a member's own rank body has no read path over this
//! table in the SAME `_ordinal`-committed order rank 0 reads, and no
//! executed oracle proves the resulting per-rank shards combine into a
//! correct gradient; `Single` and in-process `Local` gangs are unaffected.
//! The empty-negative-pool refusal and the named
//! `training_set_graph_sample` residency reservation are the graph
//! arm's own, with no tabular-arm counterpart.

use arrow::array::RecordBatch;
use jammi_db::error::Result;
use jammi_db::sql::{quote_ident, source_relation};
use jammi_db::store::manifest::{InputAnchor, ProducingDescriptor};
use jammi_db::store::{TrainingSetInput, TrainingSetSpec, TrainingSetTable};

use crate::model::ModelTask;
use crate::session::InferenceSession;

/// The single constructor every SQL-sourced production call site in this
/// crate builds a [`TrainingSetSpec`] through: a future field added to the
/// spec is added in exactly ONE place, rather than re-derived independently
/// at each of [`materialize_projection`] and `pipeline/recompute.rs`'s
/// `recompute_training_set`. The graph arm's `Batches`-sourced spec
/// (`fine_tune::worker::materialize_graph_training_set`) builds
/// `TrainingSetSpec` directly rather than through this function — its
/// `input` is a `RecordBatch` stream, not a `source_sql` string, so the two
/// arms' constructors do not share a signature to begin with.
///
/// A thin pass-through by design — it changes nothing about what a caller
/// supplies, only WHERE the seven fields are named — so it cannot move a
/// [`TrainingSetSpec::definition_hash`]; pinned by `training_set_spec_matches_
/// a_hand_built_spec_byte_for_byte` below.
pub(crate) fn training_set_spec<'a>(
    source_id: &'a str,
    source_sql: &'a str,
    columns: &'a [String],
    task: ModelTask,
    format: &'a str,
    inputs: Vec<InputAnchor>,
    device: jammi_db::store::manifest::ComputeDevice,
) -> TrainingSetSpec<'a> {
    TrainingSetSpec {
        source_id,
        input: TrainingSetInput::Sql(source_sql),
        columns,
        task,
        descriptor: ProducingDescriptor::training_set(source_sql, columns.to_vec(), task, format),
        inputs,
        device,
    }
}

/// Materialise `columns` of a registered `source` as a training set, then read
/// the committed rows back in order.
///
/// The projection runs unordered — the producer owns the sort, and asking the
/// source for an order it is about to re-impose would only plan the sort twice.
///
/// The EAGER entry point: collects the whole read-back into memory (see
/// [`read_back`]'s own doc for the reservation this pays). A `Streamed`
/// [`super::source::TrainingSource`] calls [`materialize_projection_table`]
/// instead — the table-only form — and never reaches this function, so no
/// `Vec<RecordBatch>` is ever collected for it.
pub async fn materialize_projection(
    session: &InferenceSession,
    source_id: &str,
    columns: &[String],
    task: ModelTask,
    format: &str,
) -> Result<(TrainingSetTable, Vec<RecordBatch>)> {
    let table = materialize_projection_table(session, source_id, columns, task, format).await?;
    let batches = read_back(session, &table).await?;
    Ok((table, batches))
}

/// Materialise `columns` of a registered `source` as a training set,
/// returning the table ONLY — no row is ever read. The table-only entry
/// point [`materialize_projection`] (the eager arm) and a `Streamed`
/// [`super::source::TrainingSource`] (the worker's stream arm) both build
/// their [`TrainingSetSpec`] identically through `training_set_spec`, so a
/// table this function materialises names EXACTLY the definition hash
/// [`materialize_projection`] would have computed from the same inputs.
pub async fn materialize_projection_table(
    session: &InferenceSession,
    source_id: &str,
    columns: &[String],
    task: ModelTask,
    format: &str,
) -> Result<TrainingSetTable> {
    let table_name = session.find_table_name(source_id).await?;
    let projection = columns
        .iter()
        .map(|c| quote_ident(c))
        .collect::<Vec<_>>()
        .join(", ");
    let source_sql = format!(
        "SELECT {projection} FROM {}",
        source_relation(source_id, &table_name)
    );
    materialize(
        session,
        training_set_spec(
            source_id,
            &source_sql,
            columns,
            task,
            format,
            // The source has no version surface to pin, so it is anchored at
            // the instant it was read — the same honest anchor the embedding
            // producer records for the same reason.
            vec![InputAnchor::unpinned_at_instant(
                source_id,
                chrono::Utc::now().to_rfc3339(),
            )],
            session.compute_device(),
        ),
    )
    .await
}

/// Materialise a spec through the producer. Reads NO row —
/// the reader's half of the order contract ([`read_back`]) is a SEPARATE
/// call, made only by a caller that actually wants the rows in memory.
pub async fn materialize(
    session: &InferenceSession,
    spec: TrainingSetSpec<'_>,
) -> Result<TrainingSetTable> {
    session
        .result_store()
        .materialize_training_set(session.context(), spec)
        .await
}

/// Read a materialised training set back in its committed order, collected
/// into memory — the EAGER arm's whole-table read, the table's own
/// [`TrainingSetTable::scan`] collected.
///
/// The read runs on the SAME `SessionContext` the verb was handed: the verb
/// binds the table there on both the computed and the reused path, so the
/// read-back resolves without a second registration and a reused table is read
/// exactly like a fresh one.
///
/// **The eager reservation route.** The collected batches'
/// `RecordBatch::get_array_memory_size()` sum is reserved against a
/// `MemoryConsumer("training_set_eager")` on `session.memory_pool()` right
/// after collection — the SAME pool a per-rank [`super::stream::
/// TrainingSetStream`] reserves against — so an eager read that collected
/// more bytes than `[engine] memory_limit` allows surfaces the typed
/// [`jammi_db::error::JammiError::ResourcesExhausted`] naming
/// `training_set_eager`, never a silent over-budget hold. The reservation is
/// checked and then released here (this crate does not thread a residency
/// guard through every `Vec<RecordBatch>` return site this function has), so
/// this is a load-time check on what was just collected, not a continuously
/// held accounting of how long the caller keeps the batches afterward —
/// stated, not hidden. A caller that instead needs the residency HELD for as
/// long as it keeps the rows (the production `Resident` binding,
/// `worker.rs::run_spec`) calls [`read_back_with_reservation`], not this
/// function.
pub async fn read_back(
    session: &InferenceSession,
    table: &TrainingSetTable,
) -> Result<Vec<RecordBatch>> {
    let batches = table.scan(session.context()).await?.collect().await?;
    // Checked, then released immediately — this function's own contract
    // (its doc above), unlike `read_back_with_reservation`'s.
    reserve_eager_batches(session, &batches)?.free();
    Ok(batches)
}

/// [`read_back`]'s twin for a caller that must KEEP the eager reservation
/// alive after this call returns: reserves the SAME
/// `training_set_eager`-named bytes and hands the live
/// [`MemoryReservation`](datafusion::execution::memory_pool::MemoryReservation)
/// back rather than freeing it — the caller attaches it to whatever owns the
/// rows for as long as they stay resident
/// (`super::data::TrainingDataLoader::with_reservation`) so the pool
/// reflects an eager Resident loader's true residency for its whole
/// lifetime, not just the instant this call returns.
pub async fn read_back_with_reservation(
    session: &InferenceSession,
    table: &TrainingSetTable,
) -> Result<(
    Vec<RecordBatch>,
    datafusion::execution::memory_pool::MemoryReservation,
)> {
    let batches = table.scan(session.context()).await?.collect().await?;
    let reservation = reserve_eager_batches(session, &batches)?;
    Ok((batches, reservation))
}

/// The eager reservation check (see [`read_back`]'s doc): reserve
/// `batches`' total `get_array_memory_size()` against `session.memory_pool()`
/// under a dedicated `MemoryConsumer`, surfacing a typed
/// [`jammi_db::error::JammiError::ResourcesExhausted`] naming
/// `training_set_eager` when the pool refuses it. Returns the GROWN,
/// still-live reservation — a caller that wants the old "check, then
/// release" behaviour calls `.free()` on it itself (as [`read_back`] does);
/// a caller that wants the residency held calls [`read_back_with_reservation`]
/// and keeps the reservation this returns.
fn reserve_eager_batches(
    session: &InferenceSession,
    batches: &[RecordBatch],
) -> jammi_db::error::Result<datafusion::execution::memory_pool::MemoryReservation> {
    let total_bytes: usize = batches.iter().map(RecordBatch::get_array_memory_size).sum();
    let pool = session.memory_pool();
    let reservation = datafusion::execution::memory_pool::MemoryConsumer::new("training_set_eager")
        .register(&pool);
    reservation
        .try_grow(total_bytes)
        .map_err(jammi_db::error::JammiError::from)?;
    Ok(reservation)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `training_set_spec` is a thin pass-through, so it must name the
    /// exact same [`TrainingSetSpec::
    /// definition_hash`] as a hand-built struct literal over the identical
    /// seven fields — the "unification must not change any hash" property,
    /// pinned directly rather than by re-running a whole fixture through the
    /// engine.
    #[test]
    fn training_set_spec_matches_a_hand_built_spec_byte_for_byte() {
        let columns = vec!["anchor".to_string(), "positive".to_string()];
        let inputs = vec![InputAnchor::unpinned_at_instant(
            "training",
            "2024-01-01T00:00:00Z".to_string(),
        )];
        let device = jammi_db::store::manifest::ComputeDevice::Cpu;

        let via_helper = training_set_spec(
            "training",
            "SELECT anchor, positive FROM training",
            &columns,
            ModelTask::TextEmbedding,
            "pairs",
            inputs.clone(),
            device.clone(),
        );
        let hand_built = TrainingSetSpec {
            source_id: "training",
            input: TrainingSetInput::Sql("SELECT anchor, positive FROM training"),
            columns: &columns,
            task: ModelTask::TextEmbedding,
            descriptor: ProducingDescriptor::training_set(
                "SELECT anchor, positive FROM training",
                columns.clone(),
                ModelTask::TextEmbedding,
                "pairs",
            ),
            inputs,
            device,
        };
        assert_eq!(
            via_helper.definition_hash().unwrap(),
            hand_built.definition_hash().unwrap(),
            "training_set_spec must be a pure pass-through: it cannot move the definition hash \
             relative to constructing the SAME fields directly"
        );
    }
}
