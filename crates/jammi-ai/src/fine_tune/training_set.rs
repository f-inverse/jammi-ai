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
//! # The order contract has two halves, and this module owns the reader's
//!
//! The producer commits the rows sorted by the **full projected tuple**
//! ([`TRAINING_SET_ORDER_RULE_V1`](jammi_db::store::manifest::TRAINING_SET_ORDER_RULE_V1)). A Parquet scan gives no row-order
//! guarantee: the table is written with 64K row groups and the session plans at
//! `[engine] execution_threads` partitions, so a table larger than one row group
//! comes back interleaved unless the reader asks for the order. [`read_back_sql`]
//! is the one place that asks, through
//! [`TrainingSetTable::relation`](jammi_db::store::TrainingSetTable::relation)'s
//! `TrainingSetRelation::select_ordered` — which renders
//! [`training_set_order_by`](jammi_db::store::training_set_order_by) from the
//! table's OWN recorded order columns, never a caller-supplied key — rather
//! than hand-writing the clause; a reader that spelled the direction or the
//! NULL placement differently would read rows in an order the table's own
//! descriptor does not claim, and nothing would report it. The spelling
//! scan below (`reader_class_allow_list`) is the disclosed residual this
//! type does not close: the BARE catalog name is reachable off any
//! `ResultTableRecord` (`Catalog::get_result_table`, or
//! `ResultTableRecord::table_name` directly), independently of
//! `TrainingSetTable`, so a hand-built `format!("SELECT * FROM \"jammi.{{}}\"",
//! record.table_name)` remains representable — the scan is what still catches
//! that route.
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

/// The SQL that reads a materialised training set back in its **committed
/// order** — the reader's half of the order contract, in one place.
///
/// A thin call into [`TrainingSetTable::relation`]'s
/// `TrainingSetRelation::select_ordered`: the relation and its
/// order clause are minted together, from the table's OWN recorded order
/// columns, by that one method, which takes no projection argument at all
/// — there is no caller-supplied value left to trust or
/// distrust for the `ORDER BY`, so this function takes none either:
/// a parameter this function never read would be exactly the
/// band-aid shape a caller could mistake for still influencing the SQL.
///
/// `Result`, not a bare `String`: [`TrainingSetTable::relation`]
/// refuses an empty order-column list, typed — a `TrainingSetTable` this
/// crate ever hands out never actually has one, but this function does not
/// swallow that refusal into a panic or an assumption; it propagates.
pub fn read_back_sql(table: &TrainingSetTable) -> Result<String> {
    Ok(table.relation()?.select_ordered())
}

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
/// into memory — the EAGER arm's whole-table read.
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
    let batches = session.sql(&read_back_sql(table)?).await?;
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
    let batches = session.sql(&read_back_sql(table)?).await?;
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

/// The reader-class allow-list: every call site in the workspace, outside test
/// code, that reaches a training-set table's relation by NAME through one of
/// the routes below, keyed by `path:function` rather than `path:line` — a
/// line number drifts under an unrelated edit, a function name does not —
/// with the ONE property each entry must hold: it applies
/// [`training_set_order_by`](jammi_db::store::training_set_order_by) itself
/// (or is reviewed as not reading a training-set relation at all — see the
/// needles below). A caller that reads a relation by name without
/// applying the order loses it silently on a multi-row-group table scanned by
/// more than one partition.
///
/// # The guarantee, and its residual
///
/// [`TrainingSetTable::relation`](jammi_db::store::TrainingSetTable::relation)
/// makes the SAFE route the only one with no order key to supply — every
/// production reader in this crate that names a training-set relation goes
/// through it (`read_back_sql`, below), and that route cannot render an
/// unordered read. It does NOT make the UNSAFE routes unreachable: the bare
/// catalog name is reachable off any `ResultTableRecord` — a fresh
/// `Catalog::get_result_table(name)` call, or `ResultTableRecord::table_name`
/// directly on a value obtained some other way — entirely independently of
/// `TrainingSetTable`, so a hand-built
/// `format!("SELECT * FROM \"jammi.{{}}\"", record.table_name)` remains
/// representable. This scan is what catches that; it is a residual check,
/// not a closed set over meaning. Its needles:
/// - `.sql_relation(` — the dot-call form of
///   [`TrainingSetTable::sql_relation`](jammi_db::store::TrainingSetTable::sql_relation).
/// - `sql_relation(&` — its UFCS form.
/// - `registered_name(` — a needle guarding the name against reuse by a
///   future symbol.
/// - `result_table_relation(` —
///   [`jammi_db::store::result_table_relation`], `sql_relation`'s
///   general-purpose sibling minter every other registered-relation reader in
///   the workspace calls.
/// - `.record.table_name` / `.table_name()` — the catalog-record route: a
///   caller that reaches a `ResultTableRecord`'s bare name through EITHER
///   spelling. Neither the field-access nor the method-call form is specific
///   to training sets — `PinnedSource::table_name`, `BuildingTable::table_name`,
///   `BuildingVersion::table_name` are catalog-identity accessors on
///   DIFFERENT types entirely — so this needle catches many reviewed,
///   unrelated call sites too; each is reviewed once and allow-listed below
///   with a note of which type it belongs to and why it never reaches a
///   training-set relation.
///
/// # This scan's universe is exactly these named routes — nothing wider
///
/// The scan matches literal invocation/field-access syntax, so it covers a
/// caller only if the caller spells one of the needles above. A caller that
/// re-derives the bare name some OTHER way (string concatenation from parts,
/// a value threaded through several functions before reaching a `format!`)
/// is invisible to it — the residual this module's doc states, not a claim
/// this scan closes.
///
/// Every needle match is a single-LINE `contains` check, so a rustfmt-split
/// expression whose needle text spans two lines — `.record`/`.table_name` on
/// separate lines, e.g. a method-chain break rustfmt inserts for a long line
/// — is invisible to it the same way, even though the compiled code reaches
/// the identical field. `.sql_relation(`, `registered_name(`, and
/// `result_table_relation(` are single tokens rustfmt never splits mid-call,
/// so they do not carry this gap. `sql_relation(&` is NOT a single token —
/// rustfmt CAN break a call's argument list onto its own line after `(` — so
/// it carries the same gap as `.record.table_name`/`.table_name()`; no call
/// site in the workspace spells `TrainingSetTable::sql_relation(&..)` in UFCS
/// form (every real call is the dot-call form), so that gap is currently
/// vacuous, not closed.
///
/// The accessor-impl exclusion below (`accessor_impl_line_ranges`) treats
/// its `(start, end)` span as INCLUSIVE of `end`, the impl block's
/// closing-brace LINE — so a needle appearing on that SAME line, after the
/// `}` (e.g. `} let _ = record.table_name;`), would be excluded as if it were
/// still inside the impl block. `cargo fmt --check` (a required CI step)
/// rejects that shape — rustfmt always places an item's closing brace alone
/// on its own line — so no rustfmt-clean commit can produce it.
///
/// One exclusion, by construction rather than by allow-listing: the
/// accessors' OWN implementations
/// (`crates/jammi-db/src/store/mod.rs::{table_name,sql_relation,relation}`)
/// read `self.record.table_name` (or call `self.sql_relation()`) to build
/// their OWN return value — that is the method constructing its result, never
/// a caller reaching for the relation key, so the scan skips matches whose
/// LINE falls inside `TrainingSetTable`'s or `TrainingSetRelation`'s own
/// `impl` block SPAN (see `accessor_impl_line_ranges`, which parses the file
/// with `syn` rather than a text heuristic) — never a function-NAME match
/// file-wide, which would also silently swallow `PinnedSource::table_name`'s
/// identically-named but unrelated method in the SAME file; that hit is
/// reviewed and allow-listed below instead.
///
/// [`read_back_sql`] itself carries NO entry in [`ALLOWED`] — it spells none
/// of this scan's needles (`TrainingSetTable::relation` is not one), which
/// is the point: the safe route needs no allow-list entry because it has no
/// spelling to review. Its own order guarantee is pinned separately, by
/// `training_set::read_back_re_applies_the_committed_order_across_row_groups`
/// (`tests/it/training_set.rs`).
///
/// Every entry in [`ALLOWED`] is therefore the catalog-record-route
/// residual: a reviewed call that reaches a bare table name for a purpose
/// OTHER than building a training-set SQL read (index bookkeeping, an error
/// message, a non-training-set relation) — see each entry's own comment.
///
/// This test finds every call site itself (never hand-transcribes the count)
/// by walking every `crates/*/src/**/*.rs` file from the workspace root and
/// grepping for each route's invocation syntax — so a NEW caller anywhere in
/// the workspace, not just this crate, fails it, and a call site that moves
/// to a different function name (rename) requires a conscious edit to this
/// allow-list rather than silently staying "covered". Every needle is
/// assembled at runtime (never spelled as one contiguous literal in this
/// module's own source) so this scan does not match its own doc comments,
/// messages, or the `const` below.
#[cfg(test)]
mod reader_class_allow_list {
    /// `(workspace-relative path, enclosing function name)` for every
    /// production call site that has been reviewed and accepted.
    const ALLOWED: &[(&str, &str, usize)] = &[
        // --- `.table_name()` needle, the catalog-record route:
        // none of the sites below reach a TRAINING-SET relation.
        // `run_spec` reads `TrainingSetTable::table_name` TWICE — the
        // identity pair a `Peer` gang admits against, and a materialization-
        // manifest error message — both catalog-key STRINGs, never
        // interpolated into SQL.
        ("crates/jammi-ai/src/fine_tune/worker.rs", "run_spec", 2),
        // `PinnedSource::table_name`, twice — a context-pool error message
        // AND a vector-extraction key, not a training-set relation.
        (
            "crates/jammi-ai/src/pipeline/context_set.rs",
            "pool_context_vectors",
            2,
        ),
        // `TrainingSetTable::table_name`, the freshly-materialised REPLAY's
        // own identity, returned as the recomputed table's name — never
        // interpolated into SQL.
        (
            "crates/jammi-ai/src/pipeline/recompute.rs",
            "recompute_training_set",
            1,
        ),
        // The SAME accessor, in this module's own unit-style fixture
        // (`#[cfg(test)]` inside `src/`, so this scan still walks it), TWICE
        // — a catalog-row fetch and the refusal assertion, both naming the
        // table for the test's own bookkeeping, never building SQL.
        (
            "crates/jammi-ai/src/pipeline/recompute.rs",
            "recompute_training_set_refuses_when_its_own_manifest_read_finds_no_sidecar",
            2,
        ),
        // `PinnedSource::table_name`, an embedding-delta refusal's table
        // label.
        (
            "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
            "infer_delta",
            1,
        ),
        // `PinnedSource::table_name` — the refusal/error label of the
        // pinned-provider row-set read; the rows come through
        // `pinned_provider`, never a relation string.
        (
            "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
            "current_state",
            1,
        ),
        // `PinnedSource::table_name` — the `QuerySource::Stored` table label
        // a query-by-example search reports, never a relation string.
        ("crates/jammi-ai/src/session.rs", "search_by_id", 1),
        // `PinnedSource::table_name`, once each — the typed-vector reader's
        // error label; both read through `pinned_provider` (or the raw base
        // bytes), never a relation string.
        ("crates/jammi-db/src/store/mod.rs", "read_vectors", 1),
        ("crates/jammi-db/src/store/mod.rs", "read_vector_by_key", 1),
        // `PinnedSource::input_anchor`'s OWN body (`&self.record.table_name`)
        // — the anchor's `source` is the catalog name, never a relation; the
        // same construction shape as `PinnedSource::table_name` below.
        ("crates/jammi-db/src/store/mod.rs", "input_anchor", 1),
        // `BuildingTable::table_name` — the freshly-built embedding table's
        // own identity string, not a training-set relation.
        ("crates/jammi-ai/src/pipeline/embedding.rs", "run", 1),
        // `PinnedSource::table_name`, three call sites across
        // `context_predictor.rs` — a gang-episode/provenance table label and
        // a peer-vector read's catalog key, never a training-set relation.
        (
            "crates/jammi-ai/src/pipeline/context_predictor.rs",
            "episode_for_task",
            1,
        ),
        (
            "crates/jammi-ai/src/pipeline/context_predictor.rs",
            "read_member_vectors",
            1,
        ),
        (
            "crates/jammi-ai/src/pipeline/context_predictor.rs",
            "predict_with_context_predictor_provenanced",
            1,
        ),
        // `BuildingTable::table_name`, twice — `jammi-db`'s OWN ANN-index
        // segment bookkeeping (a precision-mismatch error message AND the
        // `max_index_segment_id` key), no relation SQL anywhere on this path.
        ("crates/jammi-db/src/store/mod.rs", "append_segment", 2),
        // `BuildingVersion::table_name`, twice — same shape, the version-row
        // sibling of `append_segment`.
        (
            "crates/jammi-db/src/store/mod.rs",
            "append_segment_for_version",
            2,
        ),
        // `PinnedSource::table_name`'s OWN body (`&self.record.table_name`) —
        // NOT excluded by the accessor-impl-span mechanism above, since
        // `PinnedSource`'s impl block is a DIFFERENT, unrelated span
        // (`PinnedSource::record`/`table_name` are the same residual-route
        // class the module doc names). Reviewed here instead: the
        // method builds its OWN return value, the identical shape the
        // construction-exclusion covers for `TrainingSetTable`/
        // `TrainingSetRelation`, just not folded into that exclusion's span.
        ("crates/jammi-db/src/store/mod.rs", "table_name", 1),
        // `BuildingTable::table_name` — a test-utility helper's own table
        // label for its refusal path.
        ("crates/jammi-test-utils/src/lib.rs", "abandon_building", 1),
        // `ResultTableName::table_name` — a reused table's identity encoded
        // onto the wire's `CacheOutcome` message, never a relation string.
        (
            "crates/jammi-wire/src/cache_outcome.rs",
            "cache_outcome_to_proto",
            1,
        ),
        // --- `result_table_relation(` needle: every reviewed site — the
        // registration write/teardown and the `TableReference` reads that
        // derive the relation from the one minter, plus this crate's own
        // quoted-relation reads. Each read already applies its OWN order (a
        // primary-key equality scan, an ANN-ordered nearest-neighbour read,
        // or an unordered full scan with no committed order to lose) — none
        // is a training-set relation.
        // `ResultStore::bind_provider` — the ONE registration write, binding
        // a provider under the relation; never a read.
        ("crates/jammi-db/src/store/mod.rs", "bind_provider", 1),
        // Registration teardown on source removal — never a read.
        (
            "crates/jammi-db/src/store/result_schema.rs",
            "deregister_result_tables",
            1,
        ),
        // The `neighbor_graph` edge relation, read as a bare `TableReference`
        // — an unordered edge-list scan with no committed order to lose.
        (
            "crates/jammi-ai/src/pipeline/graph_neighbourhood.rs",
            "load_neighbor_graph_edges",
            1,
        ),
        // The SAME minter in `store/mod.rs`'s own unit fixtures (`#[cfg(test)]`
        // inside `src/`, so this scan still walks them), once each — minting
        // the relation a `TrainingSetRelation` fixture wraps, whose only
        // rendering (`select_ordered`) always carries the order clause.
        (
            "crates/jammi-db/src/store/mod.rs",
            "training_set_relation_select_ordered_renders_the_recorded_order_by",
            1,
        ),
        (
            "crates/jammi-db/src/store/mod.rs",
            "training_set_relation_with_no_order_columns_is_unconstructible",
            1,
        ),
        (
            "crates/jammi-ai/src/session.rs",
            "infer_ordered_read_back_sql",
            1,
        ),
        (
            "crates/jammi-ai/src/pipeline/graph_propagation.rs",
            "edge_scan_sql",
            1,
        ),
        ("crates/jammi-db/src/index/exact.rs", "vector_scan", 1),
        ("crates/jammi-bench/src/search_rss.rs", "scan_only_drain", 1),
        (
            "crates/jammi-bench/src/search_rss.rs",
            "naive_collect_all_search",
            1,
        ),
        (
            "crates/jammi-bench/src/propagate.rs",
            "read_sorted_vectors",
            1,
        ),
        ("crates/jammi-bench/src/corpus.rs", "load_vectors", 1),
        // `recompute_graph_training_set` returns the freshly
        // materialised graph table's bare name to its caller alongside the
        // cache outcome — a value handed UP, never a relation string built
        // here; the table's own read goes through `TrainingSetTable::relation`.
        (
            "crates/jammi-ai/src/pipeline/recompute.rs",
            "recompute_graph_training_set",
            1,
        ),
    ];

    /// The accessors' own implementations (`crates/jammi-db/src/store/mod.rs`)
    /// — excluded by construction, not by allow-listing, since a match there
    /// is the method building its own return value, never a caller reaching
    /// for the relation key. See the module doc's "One exclusion" note.
    const ACCESSOR_IMPL_FILE: &str = "crates/jammi-db/src/store/mod.rs";

    /// The `Self` type names whose top-level `impl` blocks are excluded —
    /// NEVER a function-name match file-wide: `PinnedSource` has its own
    /// `table_name`/`record` methods in the SAME file, and a name-only
    /// exclusion would silently swallow its body as if it were
    /// `TrainingSetTable::table_name`'s.
    const ACCESSOR_IMPL_TYPES: &[&str] = &["TrainingSetTable", "TrainingSetRelation"];

    /// `(start, end)` 0-based line indices (inclusive) for every top-level
    /// `impl` block in `text` whose `Self` type is one of
    /// [`ACCESSOR_IMPL_TYPES`] — via the REAL parser (`syn::parse_file`), not
    /// a line-text heuristic: a line-based "the next line that is exactly
    /// `}` at zero indentation closes the block" heuristic silently EXTENDS
    /// the excluded span past a rustfmt-clean trailing comment on that same
    /// closing brace (`} // end impl TrainingSetRelation` is not the LITERAL
    /// line `"}"`, so the heuristic keeps walking past the true end), and a
    /// hand-built read placed right after such a brace would be swallowed
    /// by the widened span, invisible to the scan. `syn`'s span for a parsed
    /// `ItemImpl` ends at the closing brace TOKEN itself, regardless of what
    /// shares its line or how any doc comment inside the block spells
    /// `{`/`}` in prose (a tokenizer never confuses a comment's or a string
    /// literal's braces with real block structure, unlike a naive text
    /// scan).
    fn accessor_impl_line_ranges(text: &str) -> Vec<(usize, usize)> {
        use syn::spanned::Spanned;

        let file = syn::parse_file(text).unwrap_or_else(|e| {
            panic!(
                "crates/jammi-db/src/store/mod.rs failed to parse as Rust via `syn` — this \
                 scan's accessor-impl exclusion cannot run: {e}"
            )
        });
        // Named, not just spanned: a PER-NAME count is asserted
        // below, not merely a total across all of `ACCESSOR_IMPL_TYPES` — a
        // total-length check alone cannot tell "two `impl TrainingSetRelation`
        // blocks and zero `impl TrainingSetTable` blocks" (2 total, matching
        // `ACCESSOR_IMPL_TYPES.len() == 2` by coincidence) from the intended
        // "exactly one of each", which would silently leave `TrainingSetTable`
        // entirely unexcluded while reporting no staleness at all.
        let named_spans: Vec<(&'static str, (usize, usize))> = file
            .items
            .iter()
            .filter_map(|item| match item {
                syn::Item::Impl(item_impl) => Some(item_impl),
                _ => None,
            })
            .filter_map(|item_impl| {
                let syn::Type::Path(type_path) = item_impl.self_ty.as_ref() else {
                    return None;
                };
                let name = type_path.path.segments.last()?.ident.to_string();
                let matched = *ACCESSOR_IMPL_TYPES.iter().find(|t| **t == name)?;
                let span = item_impl.span();
                // `proc_macro2::LineColumn::line` is 1-based; this scan's own
                // line index (`lines.iter().enumerate()`) is 0-based.
                Some((
                    matched,
                    (
                        span.start().line.saturating_sub(1),
                        span.end().line.saturating_sub(1),
                    ),
                ))
            })
            .collect();
        for type_name in ACCESSOR_IMPL_TYPES {
            let count = named_spans.iter().filter(|(n, _)| n == type_name).count();
            assert_eq!(
                count, 1,
                "expected exactly one top-level `impl {type_name}` block, found {count} — this \
                 scan's exclusion set is stale (the type was renamed, removed, or gained/lost an \
                 impl block); a total-count check across all of ACCESSOR_IMPL_TYPES would not \
                 catch this NAME's count going to 0 as long as some OTHER name's count rose to \
                 compensate, so each name is asserted separately"
            );
        }
        let mut spans: Vec<(usize, usize)> =
            named_spans.into_iter().map(|(_, span)| span).collect();
        spans.sort_unstable();
        spans
    }

    /// Every route this scan matches, each assembled from separate literal
    /// parts so the exact contiguous text never appears once in this file
    /// (which would otherwise match itself, its own doc comments, and its
    /// own messages).
    fn needles() -> Vec<String> {
        vec![
            format!(".{}(", "sql_relation"),
            format!("{}(&", "sql_relation"),
            format!("{}(", "registered_name"),
            format!("{}(", "result_table_relation"),
            format!(".{}.{}", "record", "table_name"),
            format!(".{}()", "table_name"),
        ]
    }

    /// A route's own definition line (`fn sql_relation(` /
    /// `fn registered_name(` / `fn result_table_relation(`) is not a call
    /// site — the return-type accessor/minter being DEFINED, never invoked.
    /// Distinct from `accessor_impl_line_ranges`'s exclusion, which covers calls
    /// made FROM inside those functions' bodies. `.table_name()`'s leading
    /// dot already excludes every `fn table_name(` definition line across
    /// the workspace (a definition never starts with a dot), so it needs no
    /// entry here.
    fn is_definition_line(line: &str) -> bool {
        line.contains(&format!("fn {}(", "sql_relation"))
            || line.contains(&format!("fn {}(", "registered_name"))
            || line.contains(&format!("fn {}(", "result_table_relation"))
    }

    fn workspace_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("crates/jammi-ai/../.. must be the workspace root")
    }

    /// Every `crates/*/src/**/*.rs` file under the workspace root — `src/`
    /// only, so a test fixture calling the reader method (there are several,
    /// deliberately, to build committed-order oracles) never enters this
    /// production-code sweep.
    fn all_workspace_src_files(root: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let crates_dir = root.join("crates");
        for crate_entry in std::fs::read_dir(&crates_dir)
            .unwrap_or_else(|e| panic!("read_dir({}): {e}", crates_dir.display()))
        {
            let crate_entry = crate_entry.unwrap();
            if !crate_entry.file_type().unwrap().is_dir() {
                continue;
            }
            let src_dir = crate_entry.path().join("src");
            if src_dir.is_dir() {
                walk_rs_files(&src_dir, &mut out);
            }
        }
        out
    }

    fn walk_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in
            std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir({}): {e}", dir.display()))
        {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                walk_rs_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    /// The name of the nearest `fn`/`async fn` declaration at or before
    /// `line_idx` (0-based) in `lines` — a plain textual scan, adequate for
    /// this codebase's style of one function body per reader-method call
    /// site (never a closure or a nested `fn`).
    fn enclosing_fn_name(lines: &[&str], line_idx: usize) -> Option<String> {
        let fn_line = regex_lite_find_fn(lines, line_idx)?;
        let after_fn = fn_line.split("fn ").nth(1)?;
        let name: String = after_fn
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            None
        } else {
            Some(name)
        }
    }

    /// Walk backward from `line_idx` for a line containing `"fn "` — no
    /// external regex dependency needed for this narrow a scan.
    fn regex_lite_find_fn<'a>(lines: &[&'a str], line_idx: usize) -> Option<&'a str> {
        (0..=line_idx).rev().map(|i| lines[i]).find(|l| {
            l.trim_start().starts_with("fn ")
                || l.trim_start().starts_with("pub fn ")
                || l.trim_start().starts_with("pub(crate) fn ")
                || l.trim_start().starts_with("async fn ")
                || l.trim_start().starts_with("pub async fn ")
                || l.trim_start().starts_with("pub(crate) async fn ")
        })
    }

    #[test]
    fn every_production_sql_relation_call_site_is_on_the_allow_list() {
        let root = workspace_root();
        let needles = needles();
        // One entry PER OCCURRENCE, never deduped: a (path, fn) key alone
        // would let a SECOND hand-built read inside an already-allow-listed
        // function pass silently. The assertion below compares counts.
        let mut found: Vec<(String, String)> = Vec::new();
        for path in all_workspace_src_files(&root) {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let lines: Vec<&str> = text.lines().collect();
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let accessor_spans = if rel == ACCESSOR_IMPL_FILE {
                accessor_impl_line_ranges(&text)
            } else {
                Vec::new()
            };
            for (i, line) in lines.iter().enumerate() {
                // Skip comment/doc lines outright — a mention of a route in
                // prose is not an invocation of it.
                if line.trim_start().starts_with("//") {
                    continue;
                }
                // A route's own definition line is not a call site.
                if is_definition_line(line) {
                    continue;
                }
                if !needles.iter().any(|n| line.contains(n)) {
                    continue;
                }
                // The accessors' own bodies (`sql_relation` calling
                // `result_table_relation` on itself, `relation` calling
                // `sql_relation`) are excluded by construction, scoped to
                // the exact `impl TrainingSetTable`/`impl TrainingSetRelation`
                // spans — NEVER by function name file-wide, which would also
                // (wrongly) swallow `PinnedSource::table_name`'s identically-
                // named but unrelated method in the SAME file. See the
                // module doc's "One exclusion" note.
                if accessor_spans.iter().any(|(s, e)| i >= *s && i <= *e) {
                    continue;
                }
                let func = enclosing_fn_name(&lines, i).unwrap_or_else(|| {
                    panic!(
                        "{rel}:{}: reader-method call with no enclosing `fn` found by this \
                         scan — widen `regex_lite_find_fn`'s prefix list",
                        i + 1
                    )
                });
                found.push((rel.clone(), func));
            }
        }

        // Count occurrences per (path, fn) — the key `ALLOWED` now carries a
        // third field for.
        let mut found_counts: std::collections::BTreeMap<(String, String), usize> =
            std::collections::BTreeMap::new();
        for site in found {
            *found_counts.entry(site).or_insert(0) += 1;
        }
        for ((path, func), count) in &found_counts {
            let allowed_count = ALLOWED
                .iter()
                .find(|(p, f, _)| *p == path.as_str() && *f == func.as_str())
                .map(|(_, _, c)| *c)
                .unwrap_or(0);
            assert!(
                *count <= allowed_count,
                "{path}:{func} reaches a relation's registered/catalog name (via `sql_relation`, \
                 its UFCS form, `registered_name`, `result_table_relation`, a bare `record` \
                 field's `table_name`, or a bare `table_name` call) {count} time(s), only \
                 {allowed_count} reviewed/allowed at this site — review the NEW occurrence: \
                 either it reaches a training-set relation and must apply \
                 `training_set_order_by` (or pin `target_partitions = 1` on an `ORDER BY`-free \
                 scan), or it is unrelated (a different type's catalog identity, an already-\
                 ordered read) and the count in ALLOWED belongs raised, with a note of which."
            );
        }
        for (path, func, allowed_count) in ALLOWED {
            let actual = found_counts
                .get(&((*path).to_string(), (*func).to_string()))
                .copied()
                .unwrap_or(0);
            assert_eq!(
                actual, *allowed_count,
                "{path}:{func}: ALLOWED claims {allowed_count} occurrence(s), the scan finds \
                 {actual} — the allow-list is stale (a site moved, was refactored away, or this \
                 entry's count was never accurate); narrow or correct it rather than leaving a \
                 count that does not describe reality."
            );
        }
    }
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
