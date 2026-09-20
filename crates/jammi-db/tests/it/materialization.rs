//! The materialization contract — adversarial oracle over the `MatchVerdict`
//! surface, the single-funnel guarantee, and recovery's manifest awareness.
//!
//! Every result table is published through the one `building -> ready` boundary
//! [`jammi_db::store::BuildingTable::finish`], which writes a `.materialization.json`
//! attestation (definition hash over the producing descriptor + the
//! output-affecting environment, plus the as-of input anchors) *before* the
//! status flip. `verify_materialization` recomputes the Parquet digest and
//! reports a verdict; the engine never acts on one.
//!
//! These tests construct each verdict directly (a non-vacuous oracle), prove the
//! funnel persists both the sidecar and the catalog summary columns, and prove
//! recovery distinguishes a torn manifest-less write (reaped) from a legitimate
//! pre-contract table (honest `MissingManifest`). The SIGKILL crash-injection
//! peer lives in `materialization_crash_recovery.rs` (feature `test-hooks`).

use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use datafusion::prelude::SessionContext;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::result_repo::ResultTableName;
use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::model_task::ModelTask;
use jammi_db::store::manifest::{
    AnchorKind, ArtifactDigest, ComputeDevice, ComputePrecision, DefinitionHash, InputAnchor,
    MatchVerdict, MaterializationEnv, MaterializationManifest, ModelContentDigest, ModelIdentity,
    ProducingDescriptor,
};
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::store::{
    BuildingTable, CacheOutcome, PinnedSource, ResultStore, ReusedArtifact, StaleReason, Staleness,
    TrainingSetInput, TrainingSetSpec,
};
use tempfile::tempdir;
use test_case::test_case;

use crate::common;
use crate::common::fresh_catalog;

const DIMS: usize = 4;

fn store(dir: &std::path::Path, catalog: Arc<Catalog>) -> ResultStore {
    ResultStore::new(dir, catalog, AnnIndexConfig::default()).unwrap()
}

async fn create_building(store: &ResultStore) -> BuildingTable {
    create_building_for(store, "docs").await
}

async fn create_building_for(store: &ResultStore, source_id: &str) -> BuildingTable {
    store
        .create_table(
            source_id,
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "test-model",
            Some(DIMS as i32),
            Some("_row_id"),
            Some("body"),
            None,
        )
        .await
        .unwrap()
}

/// A source id unique to this test's temp directory.
///
/// The SQLite arm gets a fresh catalog per test (the catalog file lives in the
/// temp dir); the Postgres arm shares ONE database across the whole run, so a
/// per-source count assertion would read another test's rows. Keying the
/// source on the temp dir's own random component makes every such assertion
/// range over exactly the rows its own test created, on both backends.
fn unique_source(dir: &tempfile::TempDir, stem: &str) -> String {
    let suffix = dir
        .path()
        .file_name()
        .and_then(|s| s.to_str())
        .expect("a temp dir has a UTF-8 final component");
    format!("{stem}-{suffix}")
}

async fn write_embedding_parquet(store: &ResultStore, info: &BuildingTable, n: usize) -> usize {
    let schema = embedding_table_schema(DIMS);
    let row_ids: Vec<String> = (0..n).map(|i| format!("row-{i}")).collect();
    let row_id_arr = StringArray::from_iter_values(row_ids.iter().map(|s| s.as_str()));
    let source_arr = StringArray::from_iter_values((0..n).map(|_| "docs"));
    let model_arr = StringArray::from_iter_values((0..n).map(|_| "test-model"));
    let flat: Vec<f32> = (0..n)
        .flat_map(|i| (0..DIMS).map(move |d| (i * DIMS + d) as f32))
        .collect();
    let item = Arc::new(arrow_schema::Field::new(
        "item",
        arrow_schema::DataType::Float32,
        false,
    ));
    let vectors =
        FixedSizeListArray::try_new(item, DIMS as i32, Arc::new(Float32Array::from(flat)), None)
            .unwrap();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(row_id_arr),
            Arc::new(source_arr),
            Arc::new(model_arr),
            Arc::new(vectors),
            jammi_db::store::content_hash::null_hash_column(n),
        ],
    )
    .unwrap();
    let mut writer = store.open_writer(info.parquet_url(), schema).await.unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.close().await.unwrap()
}

fn descriptor() -> ProducingDescriptor {
    ProducingDescriptor::Embedding {
        model_id: "test-model".into(),
        task: ModelTask::TextEmbedding,
        source_id: "docs".into(),
        columns: vec!["body".into()],
        key_column: "_row_id".into(),
        dimensions: DIMS,
    }
}

fn env() -> MaterializationEnv {
    MaterializationEnv::new(
        ComputeDevice::Cpu,
        vec![ModelIdentity {
            model_id: "test-model".into(),
            backend: "candle".into(),
            compute_precision: ComputePrecision::F32,
            content_digest: ModelContentDigest::Sha256("it-fixture-digest".into()),
            quantization: None,
        }],
    )
}

/// Materialise a table through the funnel and return its record + the manifest
/// the funnel computed.
async fn materialize(
    store: &ResultStore,
    ctx: &SessionContext,
    inputs: Vec<InputAnchor>,
) -> (ResultTableRecord, DefinitionHash) {
    let info = create_building(store).await;
    let rows = write_embedding_parquet(store, &info, 3).await;
    // `finish` = renew the lease → write the attestation → promote by CAS →
    // register; it returns the promoted catalog record.
    let record = info
        .finish(
            ctx,
            rows,
            jammi_db::store::manifest::Materialization::new(&descriptor(), &env(), inputs),
        )
        .await
        .unwrap();
    let definition_hash = DefinitionHash(
        record
            .definition_hash
            .clone()
            .expect("the funnel persists the definition hash"),
    );
    (record, definition_hash)
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn verdict_match_for_an_untouched_table(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let (record, def) =
        materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 1)]).await;

    // No expectation: Match.
    assert_eq!(
        store
            .verify_materialization(&common::pin(&store, &record.table_name).await, None)
            .await
            .unwrap(),
        MatchVerdict::Match
    );
    // Correct expected definition: Match.
    assert_eq!(
        store
            .verify_materialization(&common::pin(&store, &record.table_name).await, Some(&def))
            .await
            .unwrap(),
        MatchVerdict::Match
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn verdict_mismatch_against_a_wrong_expected_hash(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let (record, _def) =
        materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 1)]).await;

    let wrong = DefinitionHash("deadbeef".into());
    let verdict = store
        .verify_materialization(&common::pin(&store, &record.table_name).await, Some(&wrong))
        .await
        .unwrap();
    match verdict {
        MatchVerdict::Mismatch { expected, found } => {
            assert_eq!(expected, "deadbeef");
            assert_ne!(found, "deadbeef");
        }
        other => panic!("expected Mismatch, got {other:?}"),
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn verdict_mismatch_when_the_data_is_tampered(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let (record, _def) =
        materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 1)]).await;

    // Tamper the Parquet bytes after attestation: the recomputed digest diverges
    // from the manifest's, so the data is no longer the attested artifact.
    let handle = store
        .open_parquet(&jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap())
        .unwrap();
    let path = handle.data_path().unwrap();
    handle
        .put_bytes(&path, bytes::Bytes::from_static(b"not a parquet anymore"))
        .await
        .unwrap();

    assert!(matches!(
        store
            .verify_materialization(&common::pin(&store, &record.table_name).await, None)
            .await
            .unwrap(),
        MatchVerdict::Mismatch { .. }
    ));
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn verdict_match_with_unpinned_inputs(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let (record, _def) = materialize(
        &store,
        &ctx,
        vec![
            InputAnchor::mutable_version("pinned", 5),
            InputAnchor::unpinned_at_instant("federated", "2026-06-17T00:00:00Z"),
        ],
    )
    .await;

    match store
        .verify_materialization(&common::pin(&store, &record.table_name).await, None)
        .await
        .unwrap()
    {
        MatchVerdict::MatchWithUnpinnedInputs { unpinned } => {
            assert_eq!(unpinned, vec!["federated".to_string()]);
        }
        other => panic!("expected MatchWithUnpinnedInputs, got {other:?}"),
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn verdict_missing_manifest_for_a_pre_contract_table(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));

    // A pre-contract table: bytes + a `ready` catalog row, but NO manifest
    // sidecar and NO definition_hash summary column — exactly a table created
    // before migration 021.
    let info = create_building(&store).await;
    let rows = write_embedding_parquet(&store, &info, 3).await;
    catalog
        .update_result_table_status(info.table_name(), ResultTableStatus::Ready, rows)
        .await
        .unwrap();
    let record = store
        .catalog()
        .get_result_table(info.table_name())
        .await
        .unwrap()
        .unwrap();
    assert!(
        record.definition_hash.is_none(),
        "pre-contract row has no summary"
    );

    assert_eq!(
        store
            .verify_materialization(&common::pin(&store, &record.table_name).await, None)
            .await
            .unwrap(),
        MatchVerdict::MissingManifest
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn the_funnel_persists_sidecar_and_summary_columns(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let (record, def) =
        materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 1)]).await;

    // The catalog summary columns mirror the sidecar.
    assert_eq!(record.definition_hash.as_deref(), Some(def.as_str()));
    let anchors_json = record.input_anchors_json.expect("anchors summary present");
    let anchors: Vec<InputAnchor> = serde_json::from_str(&anchors_json).unwrap();
    assert_eq!(anchors.len(), 1);
    assert_eq!(anchors[0].kind, AnchorKind::MutableVersion);

    // The full sidecar reads back and agrees with the summary.
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("sidecar present");
    assert_eq!(manifest.definition_hash, def);
    assert_eq!(
        manifest.manifest_version,
        jammi_db::store::manifest::MANIFEST_VERSION
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn recovery_reaps_a_torn_manifestless_building_row(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));

    // Construct the torn state the crash window leaves: a `building` row whose
    // Parquet is valid but whose manifest never landed (no `.materialization.json`,
    // status still `building`).
    let info = create_building(&store).await;
    write_embedding_parquet(&store, &info, 3).await;
    let url = jammi_db::storage::StorageUrl::parse(&info.parquet_url().to_string()).unwrap();
    assert!(
        store
            .read_materialization_manifest(&url)
            .await
            .unwrap()
            .is_none(),
        "the torn state has no manifest"
    );
    // The writer is dead and its lease has run out (asserted `building` under
    // a live lease first, so the transition below is observed, not assumed).
    let table_name = jammi_test_utils::abandon_building(&catalog, info).await;

    store.recover().await.unwrap();

    // Recovery cannot reconstruct the producing descriptor, so a manifest-less
    // valid Parquet is reaped to `failed`, never promoted manifest-less.
    let record = store
        .catalog()
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(record.status, "failed");
    assert!(
        record.lease_expires_at.is_none(),
        "a terminal row carries no lease"
    );
    let handle = store.open_parquet(&url).unwrap();
    let path = handle.data_path().unwrap();
    assert!(
        !handle.exists(&path).await.unwrap(),
        "the torn Parquet bytes are reaped"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn recovery_promotes_a_building_row_whose_manifest_landed(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));

    // A `building` row whose Parquet AND manifest sidecar both landed, but whose
    // status flip never committed (a crash *after* the sidecar write, *before*
    // the catalog flip). Recovery promotes it with summary columns from the
    // sidecar.
    let info = create_building(&store).await;
    let rows = write_embedding_parquet(&store, &info, 3).await;
    let manifest = jammi_db::store::manifest::MaterializationManifest::compute(
        &descriptor(),
        &env(),
        vec![InputAnchor::mutable_version("docs", 1)],
        store_artifact_digest(&store, &info).await,
        store_artifact_leaves(&store, &info).await,
        "run-x".into(),
        "2026-06-17T00:00:00Z".into(),
    )
    .unwrap();
    write_sidecar(&store, &info, &manifest).await;
    let table_name = jammi_test_utils::abandon_building(&catalog, info).await;

    store.recover().await.unwrap();

    let record = store
        .catalog()
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(record.status, "ready");
    // A claim mints a FRESH id, `"{store.writer_id()}
    // /claim-{uuid}"` — never the store's raw process-wide id.
    assert!(
        record
            .writer_id
            .as_deref()
            .is_some_and(|w| w.starts_with(&format!("{}/claim-", store.writer_id()))),
        "recovery claimed the row before promoting: the recoverer is the writer of record, got {:?}",
        record.writer_id
    );
    assert_eq!(record.row_count, rows);
    assert_eq!(
        record.definition_hash.as_deref(),
        Some(manifest.definition_hash.as_str())
    );
    assert_eq!(
        store
            .verify_materialization(&common::pin(&store, &record.table_name).await, None)
            .await
            .unwrap(),
        MatchVerdict::Match
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn recovery_reaps_a_post_contract_ready_table_whose_sidecar_vanished(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let (record, _def) =
        materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 1)]).await;
    assert_eq!(record.status, "ready");

    // Corrupt: delete the sidecar of a post-contract `ready` row (its summary
    // column is set, so it is NOT a pre-contract table).
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    delete_sidecar(&store, &record.parquet_path).await;
    assert!(store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .is_none());

    store.recover().await.unwrap();

    let after = store
        .catalog()
        .get_result_table(&record.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        after.status, "failed",
        "a post-contract ready table missing its manifest is reaped, not left queryable"
    );
}

// --- the training-set producer ---------------------------------------------
//
// A training set is a producer output shared across runs, not a run's scratch space — shared on the
// engine's standing reuse key, the definition hash AND the recorded input anchors, so an unpinned
// source is never reused. These tests pin the db half of that contract: the kind and the manifest,
// reuse over a pinned source, the two ways a reuse is refused (an unpinned anchor, an advanced
// one), the refusal of an empty projection, the committed full-tuple order under a partitioned
// plan, the exclusion from embedding resolution, and the promise that materializing never touches a
// `building` row this call does not own.

/// The training-set fixture's columns, in the declared order that is also the
/// order key.
fn ts_columns() -> Vec<String> {
    vec!["q".to_string(), "a".to_string()]
}

fn ts_schema() -> arrow_schema::SchemaRef {
    Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("q", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("a", arrow_schema::DataType::Utf8, true),
    ]))
}

fn ts_batch(rows: &[(Option<&str>, Option<&str>)]) -> RecordBatch {
    let q: StringArray = rows.iter().map(|(q, _)| *q).collect();
    let a: StringArray = rows.iter().map(|(_, a)| *a).collect();
    RecordBatch::try_new(ts_schema(), vec![Arc::new(q), Arc::new(a)]).unwrap()
}

/// A session whose plans run at `partitions` target partitions, with the
/// fixture registered as `rows` across `batches` (one partition per batch), so
/// a scan really is partitioned and the producer's own merge is the only thing
/// keeping the committed order total.
fn ts_session(partitions: usize, batches: Vec<RecordBatch>) -> SessionContext {
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionConfig;

    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_target_partitions(partitions.max(1)),
    );
    let partitioned: Vec<Vec<RecordBatch>> = batches.into_iter().map(|b| vec![b]).collect();
    let table = MemTable::try_new(ts_schema(), partitioned).unwrap();
    ctx.register_table("rows", Arc::new(table)).unwrap();
    ctx
}

const TS_SOURCE_SQL: &str = "SELECT * FROM rows";

fn ts_spec<'a>(source_id: &'a str, columns: &'a [String], format: &'a str) -> TrainingSetSpec<'a> {
    TrainingSetSpec {
        source_id,
        input: TrainingSetInput::Sql(TS_SOURCE_SQL),
        columns,
        task: ModelTask::TextEmbedding,
        descriptor: ProducingDescriptor::training_set(
            TS_SOURCE_SQL,
            columns.to_vec(),
            ModelTask::TextEmbedding,
            format,
        ),
        // A registered relation exposes no version surface, so the honest
        // anchor is the read instant — the shape a real caller passes over an
        // unpinned source, and the one the reuse probe never matches on.
        inputs: vec![InputAnchor::unpinned_at_instant(
            source_id,
            "2026-09-13T00:00:00Z",
        )],
        device: ComputeDevice::Cpu,
    }
}

/// The SQL a pinned spec projects: the parent result table, registered on the
/// run's own session under the bare name `parent` by [`pinned_session`].
const PINNED_SOURCE_SQL: &str = "SELECT \"q\", \"a\" FROM parent";

/// A pinned training source: a `ready` result table (itself materialised from
/// the in-memory fixture) resolved ONCE into a [`PinnedSource`], so the anchor
/// a spec carries and the rows a run reads come from the same resolution. Its
/// anchor is an `AnchorKind::ResultDigest` — the only anchor kind the reuse
/// probe can honestly match, because a digest that still holds proves the rows
/// did not move.
async fn pinned_training_source(
    store: &ResultStore,
    dir: &tempfile::TempDir,
    rows: Vec<RecordBatch>,
) -> PinnedSource {
    let columns = ts_columns();
    let source = unique_source(dir, "pinned-parent");
    let parent = store
        .materialize_training_set(&ts_session(1, rows), ts_spec(&source, &columns, "parent"))
        .await
        .unwrap();
    // `pin_current_version` takes an owned `ResultTableRecord`, fetched
    // through the catalog — `TrainingSetTable` carries no
    // whole-row accessor.
    let record = store
        .catalog()
        .get_result_table(parent.table_name())
        .await
        .unwrap()
        .expect("the producer promoted a catalog row");
    store.pin_current_version(record).await.unwrap()
}

/// A fresh session reading `pin` under the bare name `parent`, through the
/// provider of the pin's own resolution.
async fn pinned_session(store: &ResultStore, pin: &PinnedSource) -> SessionContext {
    let ctx = SessionContext::new();
    let provider = store.pinned_provider(&ctx, pin).await.unwrap();
    ctx.register_table("parent", provider).unwrap();
    ctx
}

/// [`ts_spec`] over the pinned parent: the same definition determinants, with
/// `anchor` as the single recorded input.
fn pinned_spec<'a>(
    source_id: &'a str,
    columns: &'a [String],
    format: &'a str,
    anchor: InputAnchor,
) -> TrainingSetSpec<'a> {
    pinned_spec_multi(source_id, columns, format, vec![anchor])
}

/// [`pinned_spec`]'s general form: the caller supplies the full recorded
/// input set directly, over the ONE registered `parent` relation — the shape
/// the two-anchor short-circuit oracle needs (a pinned anchor plus an
/// unpinned one, neither of which the SQL itself has to distinguish).
fn pinned_spec_multi<'a>(
    source_id: &'a str,
    columns: &'a [String],
    format: &'a str,
    inputs: Vec<InputAnchor>,
) -> TrainingSetSpec<'a> {
    TrainingSetSpec {
        source_id,
        input: TrainingSetInput::Sql(PINNED_SOURCE_SQL),
        columns,
        task: ModelTask::TextEmbedding,
        descriptor: ProducingDescriptor::training_set(
            PINNED_SOURCE_SQL,
            columns.to_vec(),
            ModelTask::TextEmbedding,
            format,
        ),
        inputs,
        device: ComputeDevice::Cpu,
    }
}

/// The SQL a two-anchor pinned spec projects: BOTH parents, registered by
/// [`pinned_session_two`] under `parent_a`/`parent_b` — a spec naming two
/// anchors actually reads rows from both relations, rather than naming a
/// second anchor no query touches.
const PINNED_SOURCE_SQL_TWO: &str =
    "SELECT \"q\", \"a\" FROM parent_a UNION ALL SELECT \"q\", \"a\" FROM parent_b";

/// [`pinned_session`]'s two-relation counterpart: registers `pin_a` and
/// `pin_b` on one fresh session under the bare names [`PINNED_SOURCE_SQL_TWO`]
/// reads from.
async fn pinned_session_two(
    store: &ResultStore,
    pin_a: &PinnedSource,
    pin_b: &PinnedSource,
) -> SessionContext {
    let ctx = SessionContext::new();
    let provider_a = store.pinned_provider(&ctx, pin_a).await.unwrap();
    ctx.register_table("parent_a", provider_a).unwrap();
    let provider_b = store.pinned_provider(&ctx, pin_b).await.unwrap();
    ctx.register_table("parent_b", provider_b).unwrap();
    ctx
}

/// [`pinned_spec`]'s two-anchor counterpart, over [`PINNED_SOURCE_SQL_TWO`].
fn pinned_spec_two<'a>(
    source_id: &'a str,
    columns: &'a [String],
    format: &'a str,
    inputs: Vec<InputAnchor>,
) -> TrainingSetSpec<'a> {
    TrainingSetSpec {
        source_id,
        input: TrainingSetInput::Sql(PINNED_SOURCE_SQL_TWO),
        columns,
        task: ModelTask::TextEmbedding,
        descriptor: ProducingDescriptor::training_set(
            PINNED_SOURCE_SQL_TWO,
            columns.to_vec(),
            ModelTask::TextEmbedding,
            format,
        ),
        inputs,
        device: ComputeDevice::Cpu,
    }
}

/// Re-attest a parent table's `.materialization.json` sidecar to a NEW
/// artifact digest — models the parent having been independently recomputed
/// to new content, which [`ResultStore::current_anchor`] (via
/// [`ResultStore::staleness`]) reads as the parent's current `ResultDigest`.
/// The same technique `tests/it/freshness.rs`'s
/// `reattest_parent_with_new_digest` uses, duplicated here for this file's
/// own fixtures rather than shared across `it` test modules.
async fn reattest_with_new_digest(
    store: &ResultStore,
    parent: &ResultTableRecord,
    new_digest: ArtifactDigest,
) {
    let url = jammi_db::storage::StorageUrl::parse(&parent.parquet_path).unwrap();
    let original = store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("parent has a manifest");
    let updated = MaterializationManifest {
        artifact: new_digest,
        ..original
    };
    let handle = store.open_parquet(&url).unwrap();
    let sidecar = handle.sibling_path("materialization.json").unwrap();
    handle
        .put_bytes(&sidecar, updated.to_json_bytes().unwrap().into())
        .await
        .unwrap();
}

/// The committed rows, read straight off the Parquet object in FILE order —
/// never back through a DataFusion scan, whose partitioning is exactly what
/// this oracle must not be at the mercy of. Returns the rows and the file's
/// row-group count.
async fn committed_rows(
    store: &ResultStore,
    parquet_path: &str,
) -> (Vec<(Option<String>, Option<String>)>, usize) {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let url = jammi_db::storage::StorageUrl::parse(parquet_path).unwrap();
    let handle = store.open_parquet(&url).unwrap();
    let path = handle.data_path().unwrap();
    let bytes = handle.get_bytes(&path).await.unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
    let row_groups = builder.metadata().num_row_groups();
    let reader = builder.build().unwrap();
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch.unwrap();
        let q = string_column(&batch, "q");
        let a = string_column(&batch, "a");
        for i in 0..batch.num_rows() {
            out.push((q[i].clone(), a[i].clone()));
        }
    }
    (out, row_groups)
}

/// A Parquet string column read without assuming which Arrow string type the
/// reader hands back: the default parquet reader yields `Utf8View` in some
/// configurations and `Utf8` in others, and a `downcast_ref::<StringArray>()`
/// that assumed one would silently read nothing under the other.
fn string_column(batch: &RecordBatch, name: &str) -> Vec<Option<String>> {
    let column = batch.column_by_name(name).expect("column present");
    let cast = arrow::compute::cast(column, &arrow_schema::DataType::Utf8).unwrap();
    let values = cast.as_any().downcast_ref::<StringArray>().unwrap();
    (0..values.len())
        .map(|i| {
            if values.is_null(i) {
                None
            } else {
                Some(values.value(i).to_string())
            }
        })
        .collect()
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_training_set_lands_as_a_ready_kinded_table_with_its_attestation(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = ts_session(
        1,
        vec![ts_batch(&[
            (Some("q2"), Some("a2")),
            (Some("q1"), Some("a1")),
        ])],
    );
    let columns = ts_columns();
    let source = unique_source(&dir, "tickets");

    let materialized = store
        .materialize_training_set(&ctx, ts_spec(&source, &columns, "pairs"))
        .await
        .unwrap();

    // The whole catalog row, fetched through the catalog —
    // `TrainingSetTable` carries no whole-row accessor; `kind`/`row_count`/
    // `parquet_path` each have their own narrow accessor, used directly
    // below, but `status`/`derived_from`/the catalog's own indexed
    // `definition_hash` summary do not.
    let record = store
        .catalog()
        .get_result_table(materialized.table_name())
        .await
        .unwrap()
        .expect("the producer promoted a catalog row");
    assert_eq!(materialized.kind(), ResultTableKind::TrainingSet);
    assert_eq!(record.status, ResultTableStatus::Ready.to_string());
    assert_eq!(materialized.row_count(), 2);
    assert!(matches!(materialized.outcome, CacheOutcome::Computed));
    // The catalog's summary column is the hash the verb reports.
    assert_eq!(
        record.definition_hash.as_deref(),
        Some(materialized.definition_hash.as_str())
    );
    // The row is nobody's partial result (r31): it was created with no job
    // attempt, so no job row was ever touched.
    assert!(record.derived_from.is_none());

    // The attestation is on disk and names this producer with these exact
    // determinants.
    let manifest = store
        .read_materialization_manifest(
            &jammi_db::storage::StorageUrl::parse(materialized.parquet_path()).unwrap(),
        )
        .await
        .unwrap()
        .expect("a training set carries a materialization attestation");
    assert_eq!(manifest.definition_hash, materialized.definition_hash);
    match &manifest.descriptor {
        ProducingDescriptor::TrainingSet {
            source,
            columns: recorded,
            task,
            format,
            order_rule,
        } => {
            assert_eq!(source, "SELECT * FROM rows");
            assert_eq!(recorded, &columns);
            assert_eq!(*task, ModelTask::TextEmbedding);
            assert_eq!(format, "pairs");
            assert_eq!(order_rule, jammi_db::store::TRAINING_SET_ORDER_RULE_V1);
        }
        other => panic!("expected a TrainingSet descriptor, got {other:?}"),
    }

    // The table reads back under the name a caller queries it by.
    let rows = ctx
        .sql(&format!(
            "SELECT \"q\" FROM {} {}",
            materialized.sql_relation(),
            jammi_db::store::training_set_order_by(&columns)
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
}

/// The declared-order registration wiring, over BOTH sites that build a TrainingSet
/// table's provider: fresh materialization's own registration (inside
/// `BuildingTable::finish`) and crash-recovery's (`ResultStore::load_existing_tables`,
/// on an entirely fresh session that never saw the write) both declare the
/// producer's committed order on the `ListingTable`, so the read-back query
/// ([`training_set_order_by`]'s clause, applied over the SAME columns the
/// table was materialised from) plans no `SortExec` — the table asserts its
/// own order rather than the plan re-proving it by sorting.
///
/// The full fixture-based oracle (a multi-row-group, >1-file-group table, and
/// the NULLS-LAST positive control that must reinstate `SortExec`) lives in
/// `jammi-ai`'s `tests/it/training_set.rs` beside the row-order oracle: this
/// test instead pins the two REGISTRATION call sites, independent of
/// `jammi-ai`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_training_sets_registration_declares_its_order_so_the_read_back_plans_no_sort(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = ts_session(
        4,
        vec![
            ts_batch(&[(Some("q2"), Some("a2"))]),
            ts_batch(&[(Some("q1"), Some("a1"))]),
        ],
    );
    let columns = ts_columns();
    let source = unique_source(&dir, "tickets");

    let materialized = store
        .materialize_training_set(&ctx, ts_spec(&source, &columns, "pairs"))
        .await
        .unwrap();

    let query = format!(
        "SELECT * FROM {} {}",
        materialized.sql_relation(),
        jammi_db::store::training_set_order_by(&columns)
    );

    // Fresh materialization's own registration (inside `finish`) declared the
    // committed order: the read-back plan carries no `SortExec`.
    let plan = ctx
        .sql(&query)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let text = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );
    assert!(
        !text.contains("SortExec"),
        "fresh materialization's registration must declare the order: {text}"
    );

    // The sort columns are nullable in the resolved schema -- otherwise NULLS
    // FIRST is indistinguishable from NULLS LAST and "no SortExec" would be a
    // vacuous claim about a schema that could not have forced one anyway.
    let schema = plan.schema();
    for column in &columns {
        let field = schema.field_with_name(column).unwrap();
        assert!(
            field.is_nullable(),
            "'{column}' must be nullable for the no-SortExec claim to be non-vacuous"
        );
    }

    // Recovery's registration path (`load_existing_tables` -> the SAME
    // `bind_result_table`) declares the identical order on a session that
    // never ran the write -- reading the manifest sidecar back, not reusing
    // any in-process state from the write above.
    let ctx2 = SessionContext::new();
    store.load_existing_tables(&ctx2).await.unwrap();
    let plan2 = ctx2
        .sql(&query)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let text2 = format!(
        "{}",
        datafusion::physical_plan::displayable(plan2.as_ref()).indent(true)
    );
    assert!(
        !text2.contains("SortExec"),
        "recovery's registration must declare the order too: {text2}"
    );
}

/// A training-set row whose `.materialization.json` sidecar is absent (a
/// pre-migration-021 table — the same shape
/// [`verdict_missing_manifest_for_a_pre_contract_table`] models for the
/// verify path) still registers: `training_set_registration_sort_order`
/// returns `Ok(None)` rather than refusing the row, because
/// [`training_set_order_by`]'s explicit `ORDER BY` clause still sorts the
/// read correctly — only the `SortExec`-free plan is lost, not
/// correctness. The fallback STATES itself: a `tracing::warn!`
/// naming the table fires on recovery's registration path
/// (`load_existing_tables` -> `bind_result_table` ->
/// `training_set_registration_sort_order`), captured here the same way
/// `jammi-ai`'s `model::cache::load_bookkeeping_tests::catalog_read_error_skips_bookkeeping_write`
/// captures a `tracing::warn!` — a real `tracing_subscriber::fmt` subscriber
/// writing into an in-memory buffer this test inspects, never a log-crate
/// shim. Deleting the `warn!` call (reverting to a bare `Ok(None)`) turns
/// this test red without changing any other assertion in this file.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn registration_warns_when_a_training_sets_sidecar_is_absent(backend: BackendKind) {
    use std::io;
    use std::sync::Mutex;
    use tracing_subscriber::fmt::MakeWriter;

    #[derive(Clone, Default)]
    struct BufferWriter(Arc<Mutex<Vec<u8>>>);
    impl io::Write for BufferWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl<'w> MakeWriter<'w> for BufferWriter {
        type Writer = BufferWriter;
        fn make_writer(&'w self) -> Self::Writer {
            self.clone()
        }
    }

    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = ts_session(1, vec![ts_batch(&[(Some("q1"), Some("a1"))])]);
    let columns = ts_columns();
    let source = unique_source(&dir, "no-sidecar");

    let materialized = store
        .materialize_training_set(&ctx, ts_spec(&source, &columns, "pairs"))
        .await
        .unwrap();
    assert_eq!(materialized.kind(), ResultTableKind::TrainingSet);

    // Simulate a pre-migration-021 row: bytes + a `ready` catalog row, but no
    // manifest sidecar — the SAME corruption
    // `recovery_reaps_a_post_contract_ready_table_whose_sidecar_vanished`
    // applies to the verify path, applied here to the registration path.
    delete_sidecar(&store, materialized.parquet_path()).await;

    let buffer = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(BufferWriter(buffer.clone()))
        .with_ansi(false)
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    // Recovery's registration path re-reads the sidecar from a session that
    // never saw the write, exactly `a_training_sets_registration_declares_
    // its_order_so_the_read_back_plans_no_sort`'s recovery half above — but
    // now with no sidecar to read.
    let ctx2 = SessionContext::new();
    store.load_existing_tables(&ctx2).await.unwrap();

    // Registration still succeeds (correctness is preserved: the explicit
    // `ORDER BY` still sorts the read).
    let query = format!(
        "SELECT * FROM {} {}",
        materialized.sql_relation(),
        jammi_db::store::training_set_order_by(&columns)
    );
    let rows = ctx2.sql(&query).await.unwrap().collect().await.unwrap();
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

    let log = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
    assert!(
        log.contains(materialized.table_name()),
        "the missing-sidecar warning must name the table, got: {log}"
    );
    assert!(
        log.contains("no materialization manifest sidecar"),
        "the missing-sidecar warning must state the reason, got: {log}"
    );
}

/// An UNREADABLE sidecar
/// (present but not valid JSON — an object-store error hits the same code
/// path) is treated like an ABSENT one, never fatal to registration: the row
/// still resolves (the explicit `ORDER BY` still sorts it correctly) and a
/// `tracing::warn!` names both the table and the underlying error.
///
/// If `training_set_registration_sort_order` propagated the read error via
/// `?`, `bind_result_table` itself would return `Err` WITHOUT ever calling
/// `register_table` — the row would never enter `ctx`'s schema at all.
/// `load_existing_tables_inner` catches that per-row (`if let Err(e) =
/// self.bind_result_table(..) { warn!(..) }`), so `load_existing_tables`
/// itself returns `Ok(())` either way and cannot tell the two apart; the real
/// oracle is whether the row is still QUERYABLE afterward — the `SELECT`
/// below would otherwise fail ("table ... not found") for the wrong reason
/// (a corrupt HINT sidecar, not a corrupt table).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn registration_warns_when_a_training_sets_sidecar_is_unreadable(backend: BackendKind) {
    use std::io;
    use std::sync::Mutex;
    use tracing_subscriber::fmt::MakeWriter;

    #[derive(Clone, Default)]
    struct BufferWriter(Arc<Mutex<Vec<u8>>>);
    impl io::Write for BufferWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl<'w> MakeWriter<'w> for BufferWriter {
        type Writer = BufferWriter;
        fn make_writer(&'w self) -> Self::Writer {
            self.clone()
        }
    }

    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = ts_session(1, vec![ts_batch(&[(Some("q1"), Some("a1"))])]);
    let columns = ts_columns();
    let source = unique_source(&dir, "corrupt-sidecar");

    let materialized = store
        .materialize_training_set(&ctx, ts_spec(&source, &columns, "pairs"))
        .await
        .unwrap();
    assert_eq!(materialized.kind(), ResultTableKind::TrainingSet);

    corrupt_sidecar(&store, materialized.parquet_path()).await;

    let buffer = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(BufferWriter(buffer.clone()))
        .with_ansi(false)
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let ctx2 = SessionContext::new();
    store.load_existing_tables(&ctx2).await.unwrap();

    // The real oracle: the row must still be QUERYABLE after registration —
    // `load_existing_tables` itself always returns `Ok(())` (a per-row
    // failure there is caught and only warned about), so this SELECT, not
    // the call above, is what distinguishes "the row registered without a
    // declared sort order" from "the row never registered at all".
    let query = format!(
        "SELECT * FROM {} {}",
        materialized.sql_relation(),
        jammi_db::store::training_set_order_by(&columns)
    );
    let rows = ctx2
        .sql(&query)
        .await
        .expect("the row must still be registered despite the unreadable sidecar")
        .collect()
        .await
        .unwrap();
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

    let log = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
    assert!(
        log.contains(materialized.table_name()),
        "the unreadable-sidecar warning must name the table, got: {log}"
    );
    assert!(
        log.contains("could not be read"),
        "the unreadable-sidecar warning must state the reason, got: {log}"
    );
}

/// The training-set WRITER's full-tuple sort plans at
/// exactly ONE output partition and never builds a
/// `SortPreservingMergeExec` — the SAME single-partition derivation
/// ([`jammi_db::session::single_partition_context`]) that
/// [`ResultStore::materialize_training_set`]'s own `plan_training_set_rows`
/// calls (a private method; reproduced here byte-for-byte the way this
/// file's own read-back plan-shape test above reproduces the registration's
/// EXPLAIN), exercised at the session's OWN `target_partitions` in `{1, 4}`.
///
/// A FILE-backed source, not [`ts_session`]'s `MemTable`: a `MemTable`'s
/// partition count is fixed at construction and never collapses just
/// because `target_partitions` changed, so it cannot stand in for
/// production's actual shape here — every real `materialize_training_set`
/// caller's `source_sql` scans a `ListingTable` (`session.add_source`, a
/// registered CSV/Parquet source, or a pinned result table's own
/// `ListingTable`-backed provider), whose file-GROUP count DOES follow
/// `target_partitions` (empirically: an 80 MiB single file plans as exactly
/// one file group at `target_partitions = 1`, the mechanism
/// `crates/jammi-ai/tests/it/training_set_stream.rs`'s
/// `f1_a_table_whose_eager_read_exceeds_the_pool_trains_to_completion_
/// through_the_stream` exercises end to end). `repartition_file_min_size` is
/// forced to `1` so even this test's small file splits into multiple groups
/// at the OUTER session's `target_partitions = 4` — otherwise the pin would
/// be vacuous there (a file this small would never split on its own). The
/// hazard is exactly a `target_partitions > 1` write building a real
/// `SortPreservingMergeExec` that fills the pool before it can reserve its
/// own few MB.
#[test_case(1 ; "target_partitions_1")]
#[test_case(4 ; "target_partitions_4")]
#[tokio::test]
async fn the_writers_single_partition_derivation_plans_one_sort_and_no_merge(
    target_partitions: usize,
) {
    use datafusion::common::Column;
    use datafusion::logical_expr::Expr;
    use datafusion::physical_plan::ExecutionPlanProperties;
    use datafusion::prelude::{CsvReadOptions, SessionConfig};

    let dir = tempdir().unwrap();
    let csv_path = dir.path().join("rows.csv");
    let mut body = String::from("q,a\n");
    for i in 0..64u32 {
        body.push_str(&format!("q{i:04},a{i:04}\n"));
    }
    std::fs::write(&csv_path, body).unwrap();

    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_target_partitions(target_partitions),
    );
    ctx.register_csv("rows", csv_path.to_str().unwrap(), CsvReadOptions::new())
        .await
        .unwrap();
    ctx.sql("SET datafusion.optimizer.repartition_file_min_size = 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let columns = ts_columns();

    // Byte-for-byte `plan_training_set_rows`'s own construction
    // (`crates/jammi-db/src/store/mod.rs`): a `Column::new_unqualified`
    // projection (never the parsing `col(..)` helper) over the SAME
    // single-partition derivation, sorted ascending, NULLS FIRST.
    let single_partition_ctx = jammi_db::session::single_partition_context(&ctx);
    let projection: Vec<Expr> = columns
        .iter()
        .map(|c| Expr::Column(Column::new_unqualified(c.clone())))
        .collect();
    let sorted = single_partition_ctx
        .sql("SELECT * FROM rows")
        .await
        .unwrap()
        .select(projection.clone())
        .unwrap()
        .sort(
            projection
                .into_iter()
                .map(|e| e.sort(true, true))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let plan = sorted.create_physical_plan().await.unwrap();

    assert_eq!(
        plan.output_partitioning().partition_count(),
        1,
        "the writer's single-partition derivation must plan exactly one output partition \
         regardless of the session's own target_partitions ({target_partitions})"
    );
    let text = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );
    assert!(
        !text.contains("SortPreservingMergeExec"),
        "the writer's plan must never merge partition-local sorted runs: {text}"
    );
}

/// A projected column
/// name is data, never a fragment of SQL to re-parse. `"meta.id"` and
/// `"id"` are both admitted by `TrainingSetSpec::validate_columns` (no rule
/// there forbids a dot or mixed case), so a training set materialized over
/// exactly these two column names is reachable from
/// `materialize_training_set` — the registration renderer must bind
/// `"meta.id"` as ONE verbatim column name, never split it into a
/// `meta`-qualified reference to `id` (which would falsely collapse onto
/// the SAME schema field the second, bare `"id"` column also names).
///
/// Asserted directly against the resolved physical plan's own
/// `output_ordering` (never string-matched against `EXPLAIN` text, which a
/// cosmetic Display change could accidentally satisfy either way): the
/// LEADING declared sort column must be the schema field literally named
/// `"meta.id"`, not `"id"`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn the_file_sort_order_declares_a_dotted_column_verbatim_not_as_a_qualified_reference(
    backend: BackendKind,
) {
    use datafusion::datasource::MemTable;
    use datafusion::physical_expr::expressions::Column as PhysicalColumn;

    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));

    let schema: arrow_schema::SchemaRef = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("meta.id", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("id", arrow_schema::DataType::Utf8, true),
    ]));
    let meta_id: StringArray = vec![Some("z"), Some("m")].into_iter().collect();
    let id: StringArray = vec![Some("a"), Some("b")].into_iter().collect();
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(meta_id), Arc::new(id)]).unwrap();
    let ctx = SessionContext::new();
    let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.register_table("rows", Arc::new(table)).unwrap();

    let columns = vec!["meta.id".to_string(), "id".to_string()];
    let source = unique_source(&dir, "dotted");
    const DOTTED_SQL: &str = "SELECT \"meta.id\", \"id\" FROM rows";
    let spec = TrainingSetSpec {
        source_id: &source,
        input: TrainingSetInput::Sql(DOTTED_SQL),
        columns: &columns,
        task: ModelTask::TextEmbedding,
        descriptor: ProducingDescriptor::training_set(
            DOTTED_SQL,
            columns.clone(),
            ModelTask::TextEmbedding,
            "pairs",
        ),
        inputs: vec![InputAnchor::unpinned_at_instant(
            &source,
            "2026-09-15T00:00:00Z",
        )],
        device: ComputeDevice::Cpu,
    };

    let materialized = store.materialize_training_set(&ctx, spec).await.unwrap();

    let query = format!(
        "SELECT * FROM {} {}",
        materialized.sql_relation(),
        jammi_db::store::training_set_order_by(&columns)
    );
    let plan = ctx
        .sql(&query)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    // No SortExec: the (small, single-file-group) table's declared order is
    // still trusted for the read-back plan.
    let text = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );
    assert!(!text.contains("SortExec"), "no SortExec expected: {text}");

    // The defect itself: the declared ordering's LEADING column must be the
    // schema field literally named "meta.id" -- a parsing renderer would
    // have declared "id" here instead (both "meta.id" and the bare "id"
    // collapsing onto the same misresolved schema field).
    let ordering = plan
        .properties()
        .output_ordering()
        .expect("a declared file sort order");
    assert_eq!(ordering.len(), 2, "both projected columns are declared");
    let leading = ordering[0]
        .expr
        .downcast_ref::<PhysicalColumn>()
        .expect("the leading sort expr is a bare column reference");
    assert_eq!(
        leading.name(),
        "meta.id",
        "the leading declared sort key must be the dotted column verbatim, not a \
         misresolved 'id' — got the physical plan: {text}"
    );

    // The rows themselves come back in the true committed order (meta.id
    // ascending: "m" then "z") -- correctness, not merely the metadata.
    let rows = ctx.sql(&query).await.unwrap().collect().await.unwrap();
    let out = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
    let meta_id_out = string_column(&out, "meta.id");
    assert_eq!(
        meta_id_out,
        vec![Some("m".to_string()), Some("z".to_string())],
        "rows must read back in the committed full-tuple order"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn two_runs_over_one_pinned_definition_share_one_training_set(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let columns = ts_columns();
    let source = unique_source(&dir, "tickets");
    let rows = vec![ts_batch(&[
        (Some("q1"), Some("a1")),
        (Some("q2"), Some("a2")),
    ])];
    // Sharing is only ever offered over a PINNED source: the parent is a
    // result table resolved once, and both runs anchor on that resolution's
    // digest.
    let pin = pinned_training_source(&store, &dir, rows).await;
    let anchor = pin.input_anchor();
    assert_eq!(
        anchor.kind,
        AnchorKind::ResultDigest,
        "the fixture must pin the source, or this oracle proves nothing about reuse"
    );

    // Two runs, each on its OWN session — the second must find the table
    // through the catalog, not through a registration the first left behind.
    // The sessions themselves are kept (not discarded as temporaries) so the
    // digest check below can query EACH one independently, after both
    // materialize calls have returned.
    let first_ctx = pinned_session(&store, &pin).await;
    let first = store
        .materialize_training_set(
            &first_ctx,
            pinned_spec(&source, &columns, "pairs", anchor.clone()),
        )
        .await
        .unwrap();
    let second_ctx = pinned_session(&store, &pin).await;
    let second = store
        .materialize_training_set(
            &second_ctx,
            pinned_spec(&source, &columns, "pairs", anchor.clone()),
        )
        .await
        .unwrap();

    assert!(matches!(first.outcome, CacheOutcome::Computed));
    assert_eq!(
        second.outcome,
        CacheOutcome::Reused(ReusedArtifact::Table(ResultTableName::new(
            first.table_name()
        ))),
        "the second run must report the reuse, never hand back a copy in silence"
    );
    assert_eq!(second.table_name(), first.table_name());
    assert_eq!(second.definition_hash, first.definition_hash);

    // The reuse path (`ResultStore::bind_result_table`, the in-tree binder
    // `fine_tune/training_set.rs` reaches through `materialize_training_set`)
    // must rebind the SAME immutable artifact bytes it wrote once, not
    // merely the same name. Proven by reading the rows back through the
    // SECOND, INDEPENDENT `SessionContext` (`second_ctx`) `second` bound its
    // table on and comparing them against the first session's own read —
    // never by reading `first`/`second`'s `parquet_path` sidecar twice: a
    // `Reused` outcome carries the IDENTICAL path by construction
    // (`second.table_name() == first.table_name()`, asserted above), so two
    // `read_materialization_manifest` calls against that one shared path
    // compare a file with itself regardless of what reuse actually did.
    // Querying through two distinct `SessionContext`s is the only way to
    // exercise `bind_result_table`'s OWN rebind twice with an independent
    // read each time.
    let order_by = jammi_db::store::training_set_order_by(&columns);
    let first_rows = first_ctx
        .sql(&format!(
            "SELECT * FROM {} {order_by}",
            first.sql_relation()
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let second_rows = second_ctx
        .sql(&format!(
            "SELECT * FROM {} {order_by}",
            second.sql_relation()
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        arrow::util::pretty::pretty_format_batches(&first_rows)
            .unwrap()
            .to_string(),
        arrow::util::pretty::pretty_format_batches(&second_rows)
            .unwrap()
            .to_string(),
        "the reuse path must resolve the identical rows through a SECOND, independent \
         SessionContext — not merely the same name, and not the same sidecar path read twice"
    );

    // One table, not two.
    let tables = catalog
        .find_result_tables(&source, None, None)
        .await
        .unwrap();
    let training_sets: Vec<_> = tables
        .iter()
        .filter(|t| t.kind == ResultTableKind::TrainingSet)
        .collect();
    assert_eq!(
        training_sets.len(),
        1,
        "two runs over one definition must not leave two tables"
    );

    // …and the sharing is not "reuse whatever exists": one determinant moved
    // (the format) is a different training set.
    let other_format = store
        .materialize_training_set(
            &pinned_session(&store, &pin).await,
            pinned_spec(&source, &columns, "triplets", anchor),
        )
        .await
        .unwrap();
    assert!(matches!(other_format.outcome, CacheOutcome::Computed));
    assert_ne!(other_format.table_name(), first.table_name());
}

/// `ResultStore::install_result_schema`'s property: its own doc says "Idempotent:
/// re-installing the same provider preserves the tables it already holds."
/// Two calls on ONE `SessionContext` — an explicit one here, then the
/// implicit second one `materialize_training_set`'s own write path makes
/// through `bind_result_table` -> `register_table` -> `install_result_schema`
/// — must both succeed (no error on either), and the table the SECOND call's
/// write registers must still resolve afterwards through the SAME session:
/// "preserves the tables it already holds", executed rather than merely
/// quoted from the doc comment.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn install_result_schema_twice_on_one_session_binds_the_same_schema_and_errors_on_neither(
    backend: BackendKind,
) {
    use datafusion::datasource::MemTable;

    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), catalog);
    let columns = ts_columns();
    let source = unique_source(&dir, "install-schema-twice");
    let ctx = SessionContext::new();

    // Call 1: explicit, on an otherwise-untouched session. The fixture table
    // is registered AFTER this call, so it lands on `store`'s own
    // `ResultTableSchemaProvider` (the provider `install_result_schema`
    // installs as the session's default schema) rather than the plain
    // `MemorySchemaProvider` a session starts with — the shape a real caller
    // hits, since `install_result_schema` REPLACES whatever default schema
    // was there before (`register_schema`'s own documented "replaced"
    // semantics), which would otherwise silently drop a table registered
    // before this call rather than after it.
    store.install_result_schema(&ctx).unwrap();
    let mem_table = MemTable::try_new(
        ts_schema(),
        vec![vec![ts_batch(&[(Some("q1"), Some("a1"))])]],
    )
    .unwrap();
    ctx.register_table("rows", Arc::new(mem_table)).unwrap();

    // Call 2: implicit, inside `materialize_training_set`'s own write path,
    // on the SAME session — must not error even though the schema name is
    // already occupied by call 1's own provider, and must not drop the
    // fixture table call 1's registration landed on (the SAME `Arc` both
    // calls install, per `install_result_schema`'s own "idempotent" doc).
    let table = store
        .materialize_training_set(&ctx, ts_spec(&source, &columns, "install-schema-twice"))
        .await
        .unwrap();

    // The table call 2 registered still resolves through the session both
    // calls shared — the "preserves the tables it already holds" half, not
    // merely the "doesn't error" half.
    let rows = ctx
        .sql(&format!("SELECT * FROM {}", table.sql_relation()))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
}

/// The uniqueness oracle for the fresh path fine_tune/ reaches
/// (`ResultStore::materialize_training_set` -> `create_table`,
/// `store/mod.rs:1183`): a BURST of concurrent `create_table` calls over the
/// identical definition must never collide on one table name.
///
/// A burst of 64, real OS threads (`worker_threads = 8`), not two sequential
/// awaited calls — `create_table`'s own doc (`store/mod.rs:1178-1181`)
/// states the reason the name carries a `uuid8` suffix on top of the
/// nanosecond timestamp: "two tokio tasks call create_table within the same
/// nanosecond". Two sequential calls almost always differ in wall-clock
/// nanoseconds on their own, proving nothing about the suffix; a `tokio::
/// join!` of exactly two also does not reproduce a collision under the
/// mutation below on this host (`chrono::Utc::now()`'s effective resolution
/// is finer than the gap between two cooperatively-scheduled calls), because
/// two COOPERATIVELY SCHEDULED tasks on one OS thread (Tokio's own `join!`
/// never spawns a second OS-thread-parallel task) never race on wall-clock
/// reads at all, regardless of count. **The discriminator this test exists
/// to exercise is OS-thread PARALLELISM (`tokio::spawn` onto a multi-worker
/// runtime), not the width 64 specifically**: a `tokio::spawn` burst of
/// only TWO tasks on the same `worker_threads = 8` runtime also reproduces
/// the race under the mutation below, non-deterministically — two threads
/// racing a `chrono::Utc::now()` read only SOMETIMES land in the same
/// nanosecond bucket, so a 2-way burst's collision rate on any given host or
/// load regime is not a number this test can pin (it has been measured at
/// materially different rates across load regimes on this host, so no rate
/// is stated here). The 64-way burst below is the WITNESS this doc relies on
/// instead — see the mutation result just below for its own measured
/// determinism — because 64 concurrent OS-thread-parallel reads make the
/// SAME per-pair race the 2-way case only sometimes hits overwhelmingly
/// likely to land at least once.
///
/// Mutation-checked, not merely asserted: removing the `_{suffix}` segment
/// from `create_table`'s name builder (`format!("{source_id}__{task_str}__
/// {sanitized}__{timestamp}_{suffix}")` -> `format!("{source_id}__{task_str}__
/// {sanitized}__{timestamp}")`) fails this test with the OBSERVED
/// failure `BackendDriver(Constraint { table: "<unknown>", detail: "UNIQUE
/// constraint failed: result_tables.table_name" })` — the catalog's own
/// unique constraint on `table_name` catching the collision the suffix
/// exists to prevent, not merely a `HashSet` bookkeeping assertion in this
/// test.
///
/// **This test is itself timing-sensitive.**
/// Measured on this host: the `sqlite` arm ALONE (its own process, its own
/// `--test it -- create_table_names_a_concurrent_burst_uniquely_over_one_definition`
/// invocation) fails deterministically under the mutation above, 5/5 runs.
/// The SAME `sqlite` arm can also PASS once when co-run immediately after
/// the `postgres` arm inside one `--test-threads=1` process — a warmed-up
/// tokio thread pool (already-spun-up worker threads,
/// different scheduling latency than a cold start) narrows the race window
/// below what 64 concurrent `chrono::Utc::now()` reads reliably hit. Call
/// the `sqlite` arm ALONE before treating a single co-run pass as this
/// test's own green; a `--test-threads=1` full-file run's own pass is not,
/// by itself, evidence the mutation was reverted.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn create_table_names_a_concurrent_burst_uniquely_over_one_definition(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let source = unique_source(&dir, "concurrent");

    let mut handles = Vec::new();
    for _ in 0..64 {
        let store = store.clone();
        let source = source.clone();
        handles.push(tokio::spawn(async move {
            create_building_for(&store, &source)
                .await
                .table_name()
                .to_string()
        }));
    }
    let mut names = std::collections::HashSet::new();
    for h in handles {
        let name = h.await.unwrap();
        assert!(
            names.insert(name.clone()),
            "two concurrent create_table calls over the identical definition collided on {name} \
             — the uuid8 suffix (store/mod.rs:1181) is exactly what prevents this"
        );
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn an_unpinned_source_is_never_reused(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let columns = ts_columns();
    let source = unique_source(&dir, "tickets");
    let rows = vec![ts_batch(&[
        (Some("q1"), Some("a1")),
        (Some("q2"), Some("a2")),
    ])];

    // The registered relation exposes no version surface, so both runs anchor
    // on a read INSTANT. An instant does not prove the rows are the ones the
    // first run read: between the two runs the relation may have gained,
    // lost, or rewritten every row, and the engine has no way to tell. So the
    // second run recomputes — a training set over changed data is never
    // served as the old one.
    let first = store
        .materialize_training_set(
            &ts_session(1, rows.clone()),
            ts_spec(&source, &columns, "pairs"),
        )
        .await
        .unwrap();
    let second = store
        .materialize_training_set(&ts_session(1, rows), ts_spec(&source, &columns, "pairs"))
        .await
        .unwrap();

    // Non-vacuity: the two requests are the SAME definition — only the
    // unpinned anchors keep them apart, so a definition-only probe would have
    // reused here.
    assert_eq!(first.definition_hash, second.definition_hash);
    assert_eq!(
        ts_spec(&source, &columns, "pairs").inputs[0].kind,
        AnchorKind::UnpinnedAtInstant,
        "the fixture must anchor unpinned, or this oracle proves nothing"
    );
    assert!(matches!(first.outcome, CacheOutcome::Computed));
    assert!(
        matches!(second.outcome, CacheOutcome::Computed),
        "an unpinned source must never be reused, got {:?}",
        second.outcome
    );
    assert_ne!(second.table_name(), first.table_name());

    let training_sets: Vec<String> = catalog
        .find_result_tables(&source, None, None)
        .await
        .unwrap()
        .into_iter()
        .filter(|t| t.kind == ResultTableKind::TrainingSet)
        .map(|t| t.table_name)
        .collect();
    assert_eq!(
        training_sets.len(),
        2,
        "two unpinned runs must leave two tables, found {training_sets:?}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_reused_training_set_requires_equal_anchors(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let columns = ts_columns();
    let source = unique_source(&dir, "tickets");
    let rows = vec![ts_batch(&[
        (Some("q1"), Some("a1")),
        (Some("q2"), Some("a2")),
    ])];
    let pin = pinned_training_source(&store, &dir, rows).await;
    let anchor = pin.input_anchor();

    let first = store
        .materialize_training_set(
            &pinned_session(&store, &pin).await,
            pinned_spec(&source, &columns, "pairs", anchor.clone()),
        )
        .await
        .unwrap();
    assert!(matches!(first.outcome, CacheOutcome::Computed));

    // The same parent table, at a DIFFERENT digest — what a recompute of the
    // parent leaves a second run holding. The definition is untouched, so the
    // anchor is the only thing that says these are two different training
    // sets.
    let advanced = InputAnchor::result_digest(
        pin.table_name(),
        &ArtifactDigest(format!("{:0>64}", "deadbeef")),
    );
    assert_ne!(advanced, anchor);
    let second = store
        .materialize_training_set(
            &pinned_session(&store, &pin).await,
            pinned_spec(&source, &columns, "pairs", advanced),
        )
        .await
        .unwrap();

    assert_eq!(
        first.definition_hash, second.definition_hash,
        "the two requests must share a definition, or the anchor is not what \
         this oracle is measuring"
    );
    assert!(
        matches!(second.outcome, CacheOutcome::Computed),
        "a different input anchor is a different training set, got {:?}",
        second.outcome
    );
    assert_ne!(second.table_name(), first.table_name());

    // …and the pinned reuse itself still works: the ORIGINAL anchor hits the
    // first table, so this test's `Computed` is the anchor's doing and not a
    // probe that never matches anything.
    let third = store
        .materialize_training_set(
            &pinned_session(&store, &pin).await,
            pinned_spec(&source, &columns, "pairs", anchor),
        )
        .await
        .unwrap();
    assert_eq!(
        third.outcome,
        CacheOutcome::Reused(ReusedArtifact::Table(ResultTableName::new(
            first.table_name()
        )))
    );
}

// --- two-anchor reuse: anchor-SET equality, not per-member matching --------
//
// Every oracle above pins ONE anchor. A real graph training set records
// TWO (`sources.node_source` and `sources.edge_source`), so the reuse probe's
// contract — exact SET equality over the whole recorded input list, and the
// unpinned short-circuit firing on ANY member — has to hold once the set has
// more than one element, not just vacuously at size 1.

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn two_pinned_equal_anchors_reuse_one_table(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let columns = ts_columns();
    let source = unique_source(&dir, "tickets");
    let pin_a =
        pinned_training_source(&store, &dir, vec![ts_batch(&[(Some("qa1"), Some("aa1"))])]).await;
    let pin_b =
        pinned_training_source(&store, &dir, vec![ts_batch(&[(Some("qb1"), Some("ab1"))])]).await;
    let anchor_a = pin_a.input_anchor();
    let anchor_b = pin_b.input_anchor();
    assert_ne!(
        anchor_a, anchor_b,
        "the two anchors must genuinely differ, or this oracle proves nothing about SET reuse over two members"
    );

    // Two runs, each on its OWN session with BOTH anchors pinned equal.
    let first = store
        .materialize_training_set(
            &pinned_session_two(&store, &pin_a, &pin_b).await,
            pinned_spec_two(
                &source,
                &columns,
                "pairs",
                vec![anchor_a.clone(), anchor_b.clone()],
            ),
        )
        .await
        .unwrap();
    assert!(matches!(first.outcome, CacheOutcome::Computed));

    let second = store
        .materialize_training_set(
            &pinned_session_two(&store, &pin_a, &pin_b).await,
            pinned_spec_two(&source, &columns, "pairs", vec![anchor_a, anchor_b]),
        )
        .await
        .unwrap();
    assert_eq!(
        second.outcome,
        CacheOutcome::Reused(ReusedArtifact::Table(ResultTableName::new(
            first.table_name()
        ))),
        "two anchors, both pinned and both equal, must reuse the first table"
    );
    assert_eq!(second.table_name(), first.table_name());
    assert_eq!(second.definition_hash, first.definition_hash);

    let training_sets: Vec<_> = catalog
        .find_result_tables(&source, None, None)
        .await
        .unwrap()
        .into_iter()
        .filter(|t| t.kind == ResultTableKind::TrainingSet)
        .collect();
    assert_eq!(
        training_sets.len(),
        1,
        "two runs over one two-anchor definition must not leave two tables"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn one_unpinned_member_short_circuits_reuse_even_beside_a_pinned_equal_match(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let columns = ts_columns();
    let source = unique_source(&dir, "tickets");
    let pin =
        pinned_training_source(&store, &dir, vec![ts_batch(&[(Some("q1"), Some("a1"))])]).await;
    let anchor = pin.input_anchor();
    // The SAME instant on both calls: if the short-circuit required EVERY
    // member to be unpinned (rather than firing on ANY), this literal-equal
    // unpinned anchor beside the byte-identical pinned one would look like a
    // sound exact match.
    let unpinned = InputAnchor::unpinned_at_instant(&source, "2026-09-13T00:00:00Z");
    let inputs = vec![anchor, unpinned];

    let first = store
        .materialize_training_set(
            &pinned_session(&store, &pin).await,
            pinned_spec_multi(&source, &columns, "pairs", inputs.clone()),
        )
        .await
        .unwrap();
    assert!(matches!(first.outcome, CacheOutcome::Computed));

    // Store-level oracle: even the bare exact-match probe over the identical
    // set resolves NO candidate — the short-circuit fires on the mere
    // presence of the unpinned member, so `exact_match_candidates` is never
    // satisfied by the pinned member's byte-identical match either.
    assert_eq!(
        store
            .lookup_cached(&first.definition_hash, &inputs)
            .await
            .unwrap(),
        None,
        "a request set holding ANY unpinned member must resolve no candidate, even one whose \
         other member is a byte-identical pinned match"
    );

    let second = store
        .materialize_training_set(
            &pinned_session(&store, &pin).await,
            pinned_spec_multi(&source, &columns, "pairs", inputs),
        )
        .await
        .unwrap();
    assert!(
        matches!(second.outcome, CacheOutcome::Computed),
        "one unpinned member in a two-anchor set must force recompute, got {:?}",
        second.outcome
    );
    assert_ne!(second.table_name(), first.table_name());

    let training_sets: Vec<String> = catalog
        .find_result_tables(&source, None, None)
        .await
        .unwrap()
        .into_iter()
        .filter(|t| t.kind == ResultTableKind::TrainingSet)
        .map(|t| t.table_name)
        .collect();
    assert_eq!(
        training_sets.len(),
        2,
        "two runs with an unpinned member must leave two tables, found {training_sets:?}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn two_pinned_anchors_where_only_the_second_differs_is_not_reused(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let columns = ts_columns();
    let source = unique_source(&dir, "tickets");
    let pin_a =
        pinned_training_source(&store, &dir, vec![ts_batch(&[(Some("qa1"), Some("aa1"))])]).await;
    let pin_b =
        pinned_training_source(&store, &dir, vec![ts_batch(&[(Some("qb1"), Some("ab1"))])]).await;
    let anchor_a = pin_a.input_anchor();
    let anchor_b = pin_b.input_anchor();

    let first = store
        .materialize_training_set(
            &pinned_session_two(&store, &pin_a, &pin_b).await,
            pinned_spec_two(&source, &columns, "pairs", vec![anchor_a.clone(), anchor_b]),
        )
        .await
        .unwrap();
    assert!(matches!(first.outcome, CacheOutcome::Computed));

    // ONLY the SECOND anchor differs between the two requests — the FIRST
    // element (`anchor_a`) is byte-identical across both. A comparison that
    // checked membership by POSITION (or only the first recorded element)
    // rather than SET equality over the whole list would wrongly call this a
    // hit.
    let advanced_b = InputAnchor::result_digest(
        pin_b.table_name(),
        &ArtifactDigest(format!("{:0>64}", "advanceddigest")),
    );
    assert_ne!(advanced_b, pin_b.input_anchor());
    let second = store
        .materialize_training_set(
            &pinned_session_two(&store, &pin_a, &pin_b).await,
            pinned_spec_two(&source, &columns, "pairs", vec![anchor_a, advanced_b]),
        )
        .await
        .unwrap();

    assert_eq!(
        first.definition_hash, second.definition_hash,
        "the two requests share one definition; the anchor SET is the only thing that moved"
    );
    assert!(
        matches!(second.outcome, CacheOutcome::Computed),
        "one anchor differing in a two-anchor SET must not reuse, got {:?}",
        second.outcome
    );
    assert_ne!(second.table_name(), first.table_name());

    let training_sets: Vec<_> = catalog
        .find_result_tables(&source, None, None)
        .await
        .unwrap()
        .into_iter()
        .filter(|t| t.kind == ResultTableKind::TrainingSet)
        .collect();
    assert_eq!(
        training_sets.len(),
        2,
        "one anchor differing must leave two tables, not share one"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn staleness_over_a_two_anchor_manifest_reports_on_both_relations(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let columns = ts_columns();
    let source = unique_source(&dir, "tickets");
    let pin_a =
        pinned_training_source(&store, &dir, vec![ts_batch(&[(Some("qa1"), Some("aa1"))])]).await;
    let pin_b =
        pinned_training_source(&store, &dir, vec![ts_batch(&[(Some("qb1"), Some("ab1"))])]).await;
    let anchor_a = pin_a.input_anchor();
    let anchor_b = pin_b.input_anchor();

    let training_set = store
        .materialize_training_set(
            &pinned_session_two(&store, &pin_a, &pin_b).await,
            pinned_spec_two(&source, &columns, "pairs", vec![anchor_a, anchor_b]),
        )
        .await
        .unwrap();
    // `staleness` takes a whole `ResultTableRecord`, fetched through the
    // catalog — `TrainingSetTable` carries no whole-row
    // accessor.
    let ts_record = store
        .catalog()
        .get_result_table(training_set.table_name())
        .await
        .unwrap()
        .expect("the producer promoted a catalog row");
    assert_eq!(
        store
            .staleness(&ts_record, &training_set.definition_hash)
            .await
            .unwrap(),
        Staleness::Fresh,
        "unchanged, both parents must read Fresh"
    );

    // Both parents are independently recomputed to NEW digests — the
    // training set's manifest still records the OLD ones for both relations.
    reattest_with_new_digest(
        &store,
        pin_a.record(),
        ArtifactDigest::of_bytes(b"parent-a-v2"),
    )
    .await;
    reattest_with_new_digest(
        &store,
        pin_b.record(),
        ArtifactDigest::of_bytes(b"parent-b-v2"),
    )
    .await;

    match store
        .staleness(&ts_record, &training_set.definition_hash)
        .await
        .unwrap()
    {
        Staleness::Stale { reasons } => {
            let advanced: std::collections::HashSet<&str> = reasons
                .iter()
                .filter_map(|r| match r {
                    StaleReason::InputAdvanced { source, .. } => Some(source.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                advanced.contains(pin_a.table_name()) && advanced.contains(pin_b.table_name()),
                "staleness over a two-anchor manifest must report BOTH relations, got {reasons:?}"
            );
        }
        other => panic!("expected Stale with both relations reported, got {other:?}"),
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn an_empty_projection_is_refused_before_any_row_exists(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    // A source that exists and has the right schema — and zero rows.
    let ctx = ts_session(1, vec![ts_batch(&[])]);
    let columns = ts_columns();
    let source = unique_source(&dir, "tickets");

    let err = store
        .materialize_training_set(&ctx, ts_spec(&source, &columns, "pairs"))
        .await
        .expect_err("an empty training set must be refused, never materialized");
    assert!(
        matches!(err, jammi_db::error::JammiError::EmptyTrainingSet { .. }),
        "expected the typed empty-training-set refusal, got {err:?}"
    );

    // The refusal's real content: no row, in ANY status, and no bytes.
    let tables = catalog
        .find_result_tables(&source, None, None)
        .await
        .unwrap();
    assert!(
        tables.is_empty(),
        "the refusal must leave no catalog row behind, found {:?}",
        tables.iter().map(|t| &t.table_name).collect::<Vec<_>>()
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn the_committed_order_is_the_full_projected_tuple(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let columns = ts_columns();
    let source = unique_source(&dir, "tickets");

    // The fixture is deliberately hostile to an unordered scan: the rows are
    // scrambled, split across four partitions, and read at
    // `target_partitions = 4`; the tuple's SECOND column is what separates
    // rows that tie on the first (so a key-column-only sort would not be
    // total); and NULLs are present in both columns (so the NULL placement is
    // exercised, not assumed).
    //
    // It is also large enough that the writer flushes more than one row group
    // (65_536 rows per group), so the order oracle ranges over a real
    // multi-row-group file rather than passing vacuously on a single group.
    const ROWS: usize = 70_000;
    let mut partitions: Vec<Vec<(Option<String>, Option<String>)>> = vec![Vec::new(); 4];
    for i in 0..ROWS {
        // A scrambling permutation with no fixed point in the sort order.
        let n = (i * 37) % ROWS;
        // Every `q` value appears twice, so `a` is the separator.
        let q = Some(format!("q{:05}", n / 2));
        let a = Some(format!("a{n:05}"));
        partitions[i % 4].push((q, a));
    }
    // Degenerate rows the sort must place, not skip: a NULL in each column and
    // a fully-NULL row.
    partitions[0].push((None, Some("a-null-q".to_string())));
    partitions[1].push((Some("q-null-a".to_string()), None));
    partitions[2].push((None, None));

    let batches: Vec<RecordBatch> = partitions
        .iter()
        .map(|rows| {
            let borrowed: Vec<(Option<&str>, Option<&str>)> = rows
                .iter()
                .map(|(q, a)| (q.as_deref(), a.as_deref()))
                .collect();
            ts_batch(&borrowed)
        })
        .collect();
    let ctx = ts_session(4, batches);

    let materialized = store
        .materialize_training_set(&ctx, ts_spec(&source, &columns, "pairs"))
        .await
        .unwrap();

    let (rows, row_groups) = committed_rows(&store, materialized.parquet_path()).await;
    assert!(
        row_groups > 1,
        "the order oracle is vacuous on a single row group; the fixture produced {row_groups}"
    );
    assert_eq!(rows.len(), ROWS + 3);

    // `full_tuple_v1`: ascending on (q, a), NULLs first. Compared as the
    // producer's own key — an `Option` orders `None` before `Some`, which IS
    // "NULLs first".
    let mut expected = rows.clone();
    expected.sort();
    assert_eq!(
        rows, expected,
        "the committed file order must be the canonical full-tuple order"
    );
    // Non-vacuity: the fixture as the scan sees it — partition 0 first, then
    // 1, 2, 3, each in its own arrival order — is NOT already sorted, so the
    // assertion above had something to catch. (This is exactly what a
    // `CoalescePartitionsExec` over the unmerged sort would have committed.)
    let input_order: Vec<(Option<String>, Option<String>)> =
        partitions.iter().flatten().cloned().collect();
    let mut sorted_input = input_order.clone();
    sorted_input.sort();
    assert_ne!(
        input_order, sorted_input,
        "the fixture must not arrive already ordered, or the oracle proves nothing"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_training_set_never_resolves_as_a_sources_embedding_table(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = ts_session(1, vec![ts_batch(&[(Some("q1"), Some("a1"))])]);
    let columns = ts_columns();
    let source = unique_source(&dir, "docs");

    // The training set carries a genuine embedding `task` (the task its rows
    // train), which is exactly the trap: only the KIND separates it from a
    // model output for the same source.
    let mut spec = ts_spec(&source, &columns, "pairs");
    spec.task = ModelTask::TextEmbedding;
    let training_set = store.materialize_training_set(&ctx, spec).await.unwrap();
    let ts_record = store
        .catalog()
        .get_result_table(training_set.table_name())
        .await
        .unwrap()
        .expect("the producer promoted a catalog row");
    assert_eq!(ts_record.task, ModelTask::TextEmbedding);

    // With ONLY the training set present, the source has no embedding table.
    let err = catalog
        .resolve_embedding_table(&source, None)
        .await
        .expect_err("a training set must not resolve as an embedding table");
    assert!(
        err.to_string().contains("No ready embedding table"),
        "{err}"
    );

    // And with a real embedding table present, resolution picks that one even
    // though the training set is the newer row.
    let info = create_building_for(&store, &source).await;
    let rows = write_embedding_parquet(&store, &info, 2).await;
    let embedding = info
        .finish(
            &ctx,
            rows,
            jammi_db::store::manifest::Materialization::new(&descriptor(), &env(), vec![]),
        )
        .await
        .unwrap();
    let resolved = catalog
        .resolve_embedding_table(&source, None)
        .await
        .unwrap();
    assert_eq!(resolved.table_name, embedding.table_name);
    assert_eq!(resolved.kind, ResultTableKind::Model);
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn materializing_never_touches_a_live_building_row(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = ts_session(1, vec![ts_batch(&[(Some("q1"), Some("a1"))])]);
    let columns = ts_columns();
    let source = unique_source(&dir, "tickets");

    // A live `building` training-set row for the same source — what a crashed
    // coordinator leaves behind. Its reclaim is the lease's job; what must
    // hold HERE is that a second materialization neither deletes it, promotes
    // it, nor writes over it.
    let abandoned = store
        .create_table(
            &source,
            ModelTask::TextEmbedding,
            ResultTableKind::TrainingSet,
            None,
            "training-set",
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    let abandoned_name = abandoned.table_name().to_string();
    let before = catalog
        .get_result_table(&abandoned_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(before.status, ResultTableStatus::Building.to_string());

    let materialized = store
        .materialize_training_set(&ctx, ts_spec(&source, &columns, "pairs"))
        .await
        .unwrap();
    assert_ne!(materialized.table_name(), abandoned_name);

    let after = catalog
        .get_result_table(&abandoned_name)
        .await
        .unwrap()
        .expect("the live building row must still exist");
    assert_eq!(after.status, ResultTableStatus::Building.to_string());
    assert_eq!(after.writer_id, before.writer_id);
    assert_eq!(after.lease_expires_at, before.lease_expires_at);
    assert_eq!(after.row_count, before.row_count);
    // Let the handle release its own row cleanly so the test leaves no
    // heartbeat running.
    abandoned.abort().await.unwrap();
}

// --- helpers reaching the store's manifest sidecar for torn-state setup ----

async fn store_artifact_digest(
    store: &ResultStore,
    info: &BuildingTable,
) -> jammi_db::store::manifest::ArtifactDigest {
    let handle = store.open_parquet(info.parquet_url()).unwrap();
    let path = handle.data_path().unwrap();
    let bytes = handle.get_bytes(&path).await.unwrap();
    jammi_db::store::manifest::ArtifactDigest::of_bytes(&bytes)
}

async fn store_artifact_leaves(
    store: &ResultStore,
    info: &BuildingTable,
) -> Vec<jammi_db::store::manifest::LeafDigest> {
    let handle = store.open_parquet(info.parquet_url()).unwrap();
    let path = handle.data_path().unwrap();
    let bytes = handle.get_bytes(&path).await.unwrap();
    jammi_db::store::manifest::parquet_leaves(&bytes).unwrap()
}

async fn write_sidecar(
    store: &ResultStore,
    info: &BuildingTable,
    manifest: &jammi_db::store::manifest::MaterializationManifest,
) {
    let handle = store.open_parquet(info.parquet_url()).unwrap();
    let sidecar = handle.sibling_path("materialization.json").unwrap();
    handle
        .put_bytes(&sidecar, manifest.to_json_bytes().unwrap().into())
        .await
        .unwrap();
}

async fn delete_sidecar(store: &ResultStore, parquet_path: &str) {
    let url = jammi_db::storage::StorageUrl::parse(parquet_path).unwrap();
    let handle = store.open_parquet(&url).unwrap();
    let sidecar = handle.sibling_path("materialization.json").unwrap();
    handle.vanish_for_test(&sidecar).await.unwrap();
}

/// Overwrite a training-set row's `.materialization.json` sidecar with bytes
/// that are not valid JSON at all — an UNREADABLE sidecar, distinct from an
/// ABSENT one: `read_materialization_manifest`
/// finds the object present (`handle.exists` is `true`) but
/// `MaterializationManifest::from_json_bytes` fails to parse it, so the call
/// returns `Err`, never `Ok(None)`.
async fn corrupt_sidecar(store: &ResultStore, parquet_path: &str) {
    let url = jammi_db::storage::StorageUrl::parse(parquet_path).unwrap();
    let handle = store.open_parquet(&url).unwrap();
    let sidecar = handle.sibling_path("materialization.json").unwrap();
    handle
        .put_bytes(&sidecar, b"not valid json".to_vec().into())
        .await
        .unwrap();
}

// ── The leaf inventory, through the store ─────────────────────────────────
//
// The unit-level oracles (`store::manifest::tests::leaves`) prove the
// footer walk over a three-row-group object; these prove the FUNNEL writes
// the inventory, `verify_partitions` names the corrupt row group, a
// footer-only mutation is the whole-object digest's to catch (no leaf
// covers it), and a pre-`leaves` sidecar reads as absent on both verbs.

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn the_funnel_writes_one_leaf_per_row_group_and_verify_partitions_matches(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();
    let (record, _def) =
        materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 1)]).await;
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("the funnel wrote a sidecar");
    // The footer is the oracle for the count, read here independently.
    let handle = store.open_parquet(&url).unwrap();
    let bytes = handle
        .get_bytes(&handle.data_path().unwrap())
        .await
        .unwrap();
    let footer = parquet::file::metadata::ParquetMetaDataReader::new()
        .parse_and_finish(&bytes)
        .unwrap();
    assert_eq!(manifest.leaves.len(), footer.num_row_groups());
    assert!(!manifest.leaves.is_empty());
    assert_eq!(
        store.verify_partitions(&record).await.unwrap(),
        jammi_db::store::manifest::PartitionVerdict::Match
    );
    assert_eq!(
        store
            .verify_materialization(&common::pin(&store, &record.table_name).await, None)
            .await
            .unwrap(),
        MatchVerdict::Match
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_corrupted_row_group_is_named_by_its_leaf_and_a_footer_mutation_by_the_artifact(
    backend: BackendKind,
) {
    use jammi_db::store::manifest::{LeafKey, PartitionVerdict};
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();
    let (record, _def) =
        materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 1)]).await;
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .unwrap();
    let handle = store.open_parquet(&url).unwrap();
    let path = handle.data_path().unwrap();
    let original = handle.get_bytes(&path).await.unwrap().to_vec();
    let LeafKey::RowGroup {
        index,
        offset,
        length,
    } = manifest.leaves[0].key.clone()
    else {
        panic!("a result table's leaf is a row group");
    };
    // Inside the first row group: the leaf names it.
    let mut inside = original.clone();
    inside[(offset + length / 2) as usize] ^= 0xff;
    handle
        .put_bytes(&path, bytes::Bytes::from(inside))
        .await
        .unwrap();
    match store.verify_partitions(&record).await.unwrap() {
        PartitionVerdict::Mismatch { key, .. } => assert!(
            matches!(key, LeafKey::RowGroup { index: i, .. } if i == index),
            "{key:?}"
        ),
        other => panic!("expected the corrupt row group to be named, got {other:?}"),
    }
    assert!(matches!(
        store
            .verify_materialization(&common::pin(&store, &record.table_name).await, None)
            .await
            .unwrap(),
        MatchVerdict::Mismatch { .. }
    ));
    // In the footer (past the last row group's range, inside the metadata):
    // no leaf covers it, so the inventory still matches — the whole-object
    // digest is the subject that catches it.
    let last = manifest.leaves.last().unwrap();
    let LeafKey::RowGroup {
        offset: last_off,
        length: last_len,
        ..
    } = last.key.clone()
    else {
        panic!("row-group leaf");
    };
    let mut footered = original.clone();
    footered[(last_off + last_len) as usize + 4] ^= 0x01;
    handle
        .put_bytes(&path, bytes::Bytes::from(footered))
        .await
        .unwrap();
    assert!(matches!(
        store
            .verify_materialization(&common::pin(&store, &record.table_name).await, None)
            .await
            .unwrap(),
        MatchVerdict::Mismatch { .. }
    ));
    match store.verify_partitions(&record).await {
        Ok(PartitionVerdict::Match) => {}
        // The flip may have made the footer unreadable: then no inventory
        // can be derived at all — still never a false leaf mismatch.
        Err(e) => assert!(e.to_string().contains("parquet footer"), "{e}"),
        other => panic!("a footer mutation names no leaf: {other:?}"),
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_pre_leaves_sidecar_reads_as_absent_on_both_verbs(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();
    let (record, _def) =
        materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 1)]).await;
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .unwrap();
    // Rewrite the sidecar as the pre-inventory format: the same object
    // without `leaves`.
    let mut value = serde_json::to_value(&manifest).unwrap();
    value.as_object_mut().unwrap().remove("leaves");
    let handle = store.open_parquet(&url).unwrap();
    let sidecar = handle.sibling_path("materialization.json").unwrap();
    handle
        .put_bytes(&sidecar, serde_json::to_vec(&value).unwrap().into())
        .await
        .unwrap();
    assert!(store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        store
            .verify_materialization(&common::pin(&store, &record.table_name).await, None)
            .await
            .unwrap(),
        MatchVerdict::MissingManifest
    );
    assert_eq!(
        store.verify_partitions(&record).await.unwrap(),
        jammi_db::store::manifest::PartitionVerdict::MissingManifest
    );
    // A NEWER version stays an error, never a miss.
    let mut newer = serde_json::to_value(&manifest).unwrap();
    newer.as_object_mut().unwrap().insert(
        "manifest_version".into(),
        serde_json::json!(jammi_db::store::manifest::MANIFEST_VERSION + 1),
    );
    handle
        .put_bytes(&sidecar, serde_json::to_vec(&newer).unwrap().into())
        .await
        .unwrap();
    assert!(store
        .read_materialization_manifest(&url)
        .await
        .err()
        .is_some());
}

// ─── The `Batches` producer input ─────────────────────────────────────────

fn ordinal_schema() -> arrow_schema::SchemaRef {
    Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("_ordinal", arrow_schema::DataType::UInt64, false),
        arrow_schema::Field::new("anchor", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("positive", arrow_schema::DataType::Utf8, false),
    ]))
}

fn ordinal_batch(rows: &[(u64, &str, &str)]) -> RecordBatch {
    let ord = arrow::array::UInt64Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>());
    let anchor: StringArray = rows.iter().map(|r| Some(r.1)).collect();
    let positive: StringArray = rows.iter().map(|r| Some(r.2)).collect();
    RecordBatch::try_new(
        ordinal_schema(),
        vec![Arc::new(ord), Arc::new(anchor), Arc::new(positive)],
    )
    .unwrap()
}

/// A one-shot [`jammi_db::store::TrainingSetInput::Batches`] stream over
/// `batches`, deliberately NOT sorted alphabetically by `(anchor, positive)`
/// — the sampler's own per-anchor emission order, exactly the shape
/// `plan_training_set_rows` must commit without re-imposing a full-tuple
/// sort.
fn ordinal_stream(batches: Vec<RecordBatch>) -> datafusion::execution::SendableRecordBatchStream {
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    Box::pin(RecordBatchStreamAdapter::new(
        ordinal_schema(),
        futures::stream::iter(batches.into_iter().map(Ok)),
    ))
}

fn graph_descriptor_fixture() -> ProducingDescriptor {
    ProducingDescriptor::graph_training_set(
        "kb_nodes",
        "kb_edges",
        "id",
        "text",
        "src",
        "dst",
        ModelTask::TextEmbedding,
        "pairs",
        jammi_db::store::manifest::GraphSampleFields {
            seed: 1,
            walk_length: 3,
            walks_per_node: 2,
            return_p_bits: 1.0_f64.to_bits(),
            in_out_q_bits: 1.0_f64.to_bits(),
            hard_negatives: 0,
            exclude_hops: 1,
        },
    )
}

/// A `Batches` input's rows commit and read back in EXACTLY the caller's
/// emission order — never the tabular arm's full-tuple alphabetic sort. Two
/// batches, each internally NOT alphabetic (`z`, `m`, `a`), so a full-tuple
/// `SortExec` (if one ran) would visibly permute them; the oracle is
/// `training_set_order_by` over just `["_ordinal"]`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn batches_input_commits_and_reads_back_in_emission_order(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let rows: Vec<(u64, &str, &str)> = vec![
        (0, "n3", "z"),
        (1, "n3", "m"),
        (2, "n3", "a"),
        (3, "n1", "z"),
        (4, "n1", "m"),
        (5, "n1", "a"),
    ];
    let batches = vec![ordinal_batch(&rows[..3]), ordinal_batch(&rows[3..])];
    let source = unique_source(&dir, "graph-batches");
    let order_columns = vec!["_ordinal".to_string()];
    let now = "2026-09-17T00:00:00Z";
    let spec = TrainingSetSpec {
        source_id: &source,
        input: TrainingSetInput::Batches {
            schema: ordinal_schema(),
            stream: ordinal_stream(batches),
        },
        columns: &order_columns,
        task: ModelTask::TextEmbedding,
        descriptor: graph_descriptor_fixture(),
        inputs: vec![
            InputAnchor::unpinned_at_instant("kb_nodes", now),
            InputAnchor::unpinned_at_instant("kb_edges", now),
        ],
        device: ComputeDevice::Cpu,
    };

    let materialized = store.materialize_training_set(&ctx, spec).await.unwrap();
    assert_eq!(materialized.row_count(), 6);
    assert_eq!(materialized.kind(), ResultTableKind::TrainingSet);

    let expected: Vec<(u64, String, String)> = rows
        .iter()
        .map(|(o, a, p)| (*o, a.to_string(), p.to_string()))
        .collect();

    async fn read_rows(ctx: &SessionContext, query: &str) -> Vec<(u64, String, String)> {
        let got = ctx.sql(query).await.unwrap().collect().await.unwrap();
        let mut out = Vec::new();
        for batch in &got {
            let ord = batch
                .column_by_name("_ordinal")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .unwrap();
            let anchor = string_column(batch, "anchor");
            let positive = string_column(batch, "positive");
            for i in 0..batch.num_rows() {
                out.push((
                    ord.value(i),
                    anchor[i].clone().unwrap(),
                    positive[i].clone().unwrap(),
                ));
            }
        }
        out
    }

    // (a) The reader's half of the contract: an EXPLICIT `ORDER BY` over the
    // order key reproduces emission order (works regardless of physical
    // write order, since this is a real sort).
    let ordered_query = format!(
        "SELECT * FROM {} {}",
        materialized.sql_relation(),
        jammi_db::store::training_set_order_by(&order_columns)
    );
    assert_eq!(
        read_rows(&ctx, &ordered_query).await,
        expected,
        "an explicit ORDER BY over the order key must reproduce emission order"
    );

    // (b) The producer's half: a PLAIN scan with NO `ORDER BY` at all
    // already comes back in emission order, because the write path never
    // re-sorted the rows in the first place (a `Batches` input commits
    // EXACTLY the caller's order, so even an unordered read of this
    // single-file, single-row-group table matches it). Mutation-checked:
    // adding a `.sort()` to `plan_training_set_rows`'s `Batches` arm makes
    // this specific assertion fail (rows come back alphabetised) while
    // assertion (a) above still passes.
    let plain_query = format!("SELECT * FROM {}", materialized.sql_relation());
    assert_eq!(
        read_rows(&ctx, &plain_query).await,
        expected,
        "the Batches arm must not re-impose a sort — even an unordered read of the freshly \
         written table must already be in emission order"
    );

    // A gang member binds the SAME table on its OWN, independent session —
    // never the coordinator's `ctx`. A fresh session's `bind_result_table`
    // must declare the SAME `_ordinal` file sort order
    // (`ResultStore::training_set_registration_sort_order`'s
    // `GraphTrainingSet` arm) so a member's plain scan agrees with the
    // coordinator's own read, with no explicit `ORDER BY` needed on either
    // side. This is infrastructure a `Batches`-sourced table's cross-session
    // rebinding must hold regardless of which caller re-binds it.
    let member_ctx = SessionContext::new();
    // The handle has no whole-row accessor: a fresh session binds the
    // catalog's own row, fetched by name.
    let materialized_record = store
        .catalog()
        .get_result_table(materialized.table_name())
        .await
        .unwrap()
        .unwrap();
    store
        .bind_result_table(&member_ctx, &materialized_record)
        .await
        .unwrap();
    let member_query = format!("SELECT * FROM {}", materialized.sql_relation());
    assert_eq!(
        read_rows(&member_ctx, &member_query).await,
        expected,
        "a table re-bound on an INDEPENDENT session must still read back in emission order"
    );
}

/// An EMPTY `Batches` stream's refusal names `source_id`, never a
/// fabricated SQL string (there is none to quote).
#[tokio::test]
async fn batches_input_empty_stream_names_source_id_not_sql() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(BackendKind::Sqlite, dir.path()).await;
    let store = store(dir.path(), catalog);
    let ctx = SessionContext::new();

    let source = unique_source(&dir, "graph-batches-empty");
    let order_columns = vec!["_ordinal".to_string()];
    let now = "2026-09-17T00:00:00Z";
    let spec = TrainingSetSpec {
        source_id: &source,
        input: TrainingSetInput::Batches {
            schema: ordinal_schema(),
            stream: ordinal_stream(Vec::new()),
        },
        columns: &order_columns,
        task: ModelTask::TextEmbedding,
        descriptor: graph_descriptor_fixture(),
        inputs: vec![
            InputAnchor::unpinned_at_instant("kb_nodes", now),
            InputAnchor::unpinned_at_instant("kb_edges", now),
        ],
        device: ComputeDevice::Cpu,
    };

    let err = store
        .materialize_training_set(&ctx, spec)
        .await
        .expect_err("an empty Batches stream must be refused, never a 0-row table");
    let msg = format!("{err}");
    assert!(
        msg.contains(&source),
        "the refusal must name source_id ('{source}'), got: {msg}"
    );
    assert!(
        !msg.to_uppercase().contains("SELECT"),
        "the refusal must never fabricate a SQL string for a Batches input, got: {msg}"
    );
}

/// The mutation half: skipping the ordinal-sortedness assertion (a
/// deliberately UN-ordered `_ordinal` column within one batch) must be
/// caught, typed, rather than silently committed out of order — the
/// production analogue of the harness's "drop the ORDER BY" mutation.
#[tokio::test]
async fn batches_input_out_of_order_ordinal_within_a_batch_is_refused() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(BackendKind::Sqlite, dir.path()).await;
    let store = store(dir.path(), catalog);
    let ctx = SessionContext::new();

    // `_ordinal` goes 0, 2, 1 WITHIN one batch — not non-decreasing.
    let bad_rows: Vec<(u64, &str, &str)> = vec![(0, "n0", "x"), (2, "n1", "y"), (1, "n2", "z")];
    let batches = vec![ordinal_batch(&bad_rows)];
    let source = unique_source(&dir, "graph-batches-unsorted");
    let order_columns = vec!["_ordinal".to_string()];
    let now = "2026-09-17T00:00:00Z";
    let spec = TrainingSetSpec {
        source_id: &source,
        input: TrainingSetInput::Batches {
            schema: ordinal_schema(),
            stream: ordinal_stream(batches),
        },
        columns: &order_columns,
        task: ModelTask::TextEmbedding,
        descriptor: graph_descriptor_fixture(),
        inputs: vec![
            InputAnchor::unpinned_at_instant("kb_nodes", now),
            InputAnchor::unpinned_at_instant("kb_edges", now),
        ],
        device: ComputeDevice::Cpu,
    };

    let err = store
        .materialize_training_set(&ctx, spec)
        .await
        .expect_err("an out-of-order `_ordinal` batch must be refused, never silently committed");
    assert!(
        format!("{err}").contains("not sorted"),
        "the refusal should name the sortedness violation: {err}"
    );
}
