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
use jammi_db::catalog::backend_postgres::PostgresBackend;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
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
    BuildingTable, CacheOutcome, PinnedSource, ResultStore, StaleReason, Staleness, TrainingSetSpec,
};
use tempfile::tempdir;
use test_case::test_case;

const DIMS: usize = 4;

/// Build a catalog on `backend`, running migrations. Returns `None` for the
/// Postgres arm when `JAMMI_TEST_PG_URL` is unset, so callers skip (never
/// `#[ignore]`) exactly like [`jammi_test_utils::make_test_session`].
async fn fresh_catalog(backend: BackendKind, dir: &std::path::Path) -> Option<Arc<Catalog>> {
    let backend_impl = match backend {
        BackendKind::Sqlite => {
            let b = SqliteBackend::open(&dir.join("catalog.db")).await.unwrap();
            jammi_db::catalog::backend::BackendImpl::Sqlite(b)
        }
        BackendKind::Postgres => {
            let url = jammi_test_utils::pg_url_for_tests()?;
            let pg = PostgresBackend::open_with_options(&url, 8, None)
                .await
                .unwrap();
            jammi_db::catalog::backend::BackendImpl::Postgres(pg)
        }
    };
    backend_impl.migrate().await.unwrap();
    Some(Arc::new(Catalog::from_backend(backend_impl)))
}

/// Fetch a backend-parameterized catalog, skipping the test (with a warning)
/// when the Postgres arm has no `JAMMI_TEST_PG_URL`.
macro_rules! fresh_catalog_or_skip {
    ($backend:expr, $dir:expr) => {
        match fresh_catalog($backend, $dir.path()).await {
            Some(c) => c,
            None => {
                eprintln!("skipping {:?}: JAMMI_TEST_PG_URL unset", $backend);
                return;
            }
        }
    };
}

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
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let (record, def) =
        materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 1)]).await;

    // No expectation: Match.
    assert_eq!(
        store.verify_materialization(&record, None).await.unwrap(),
        MatchVerdict::Match
    );
    // Correct expected definition: Match.
    assert_eq!(
        store
            .verify_materialization(&record, Some(&def))
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
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let (record, _def) =
        materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 1)]).await;

    let wrong = DefinitionHash("deadbeef".into());
    let verdict = store
        .verify_materialization(&record, Some(&wrong))
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
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
        store.verify_materialization(&record, None).await.unwrap(),
        MatchVerdict::Mismatch { .. }
    ));
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn verdict_match_with_unpinned_inputs(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
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

    match store.verify_materialization(&record, None).await.unwrap() {
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
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
        store.verify_materialization(&record, None).await.unwrap(),
        MatchVerdict::MissingManifest
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn the_funnel_persists_sidecar_and_summary_columns(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
        store.verify_materialization(&record, None).await.unwrap(),
        MatchVerdict::Match
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn recovery_reaps_a_post_contract_ready_table_whose_sidecar_vanished(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let (record, _def) =
        materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 1)]).await;
    assert_eq!(record.status, "ready");

    // Corrupt: delete the sidecar of a post-contract `ready` row (its summary
    // column is set, so it is NOT a pre-contract table).
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    delete_sidecar(&store, &record).await;
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
// A training set is a producer output shared across runs, not a run's scratch
// space (r31) — shared on the engine's standing reuse key, the definition hash
// AND the recorded input anchors, so an unpinned source is never reused.
// These tests pin the db half of that contract: the kind and the manifest,
// reuse over a pinned source, the two ways a reuse is refused (an unpinned
// anchor, an advanced one), the K2 refusal of an empty projection, the committed full-tuple order under a partitioned plan, the
// exclusion from embedding resolution, and the promise that materializing
// never touches a `building` row this call does not own.

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

fn ts_spec<'a>(source_id: &'a str, columns: &'a [String], format: &'a str) -> TrainingSetSpec<'a> {
    TrainingSetSpec {
        source_id,
        source_sql: "SELECT * FROM rows",
        columns,
        task: ModelTask::TextEmbedding,
        format,
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
    store.pin_current_version(parent.record).await.unwrap()
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
        source_sql: PINNED_SOURCE_SQL,
        columns,
        task: ModelTask::TextEmbedding,
        format,
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
        source_sql: PINNED_SOURCE_SQL_TWO,
        columns,
        task: ModelTask::TextEmbedding,
        format,
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
    record: &ResultTableRecord,
) -> (Vec<(Option<String>, Option<String>)>, usize) {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
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
    let catalog = fresh_catalog_or_skip!(backend, dir);
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

    assert_eq!(materialized.record.kind, ResultTableKind::TrainingSet);
    assert_eq!(
        materialized.record.status,
        ResultTableStatus::Ready.to_string()
    );
    assert_eq!(materialized.record.row_count, 2);
    assert!(matches!(materialized.outcome, CacheOutcome::Computed));
    // The catalog's summary column is the hash the verb reports.
    assert_eq!(
        materialized.record.definition_hash.as_deref(),
        Some(materialized.definition_hash.as_str())
    );
    // The row is nobody's partial result (r31): it was created with no job
    // attempt, so no job row was ever touched.
    assert!(materialized.record.derived_from.is_none());

    // The attestation is on disk and names this producer with these exact
    // determinants.
    let manifest = store
        .read_materialization_manifest(
            &jammi_db::storage::StorageUrl::parse(&materialized.record.parquet_path).unwrap(),
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

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn two_runs_over_one_pinned_definition_share_one_training_set(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
    let first = store
        .materialize_training_set(
            &pinned_session(&store, &pin).await,
            pinned_spec(&source, &columns, "pairs", anchor.clone()),
        )
        .await
        .unwrap();
    let second = store
        .materialize_training_set(
            &pinned_session(&store, &pin).await,
            pinned_spec(&source, &columns, "pairs", anchor.clone()),
        )
        .await
        .unwrap();

    assert!(matches!(first.outcome, CacheOutcome::Computed));
    assert_eq!(
        second.outcome,
        CacheOutcome::Reused {
            table: first.table_name().to_string()
        },
        "the second run must report the reuse, never hand back a copy in silence"
    );
    assert_eq!(second.table_name(), first.table_name());
    assert_eq!(second.definition_hash, first.definition_hash);

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

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn an_unpinned_source_is_never_reused(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
        CacheOutcome::Reused {
            table: first.table_name().to_string()
        }
    );
}

// --- two-anchor reuse: anchor-SET equality, not per-member matching --------
//
// Every oracle above pins ONE anchor. A real graph training set (M1) records
// TWO (`sources.node_source` and `sources.edge_source`), so the reuse probe's
// contract — exact SET equality over the whole recorded input list, and the
// unpinned short-circuit firing on ANY member — has to hold once the set has
// more than one element, not just vacuously at size 1.

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn two_pinned_equal_anchors_reuse_one_table(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
        CacheOutcome::Reused {
            table: first.table_name().to_string()
        },
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
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
    assert_eq!(
        store
            .staleness(&training_set.record, &training_set.definition_hash)
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
        .staleness(&training_set.record, &training_set.definition_hash)
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
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
        "expected the typed K2 refusal, got {err:?}"
    );

    // K2's real content: no row, in ANY status, and no bytes.
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
    let catalog = fresh_catalog_or_skip!(backend, dir);
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

    let (rows, row_groups) = committed_rows(&store, &materialized.record).await;
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
    let catalog = fresh_catalog_or_skip!(backend, dir);
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
    assert_eq!(training_set.record.task, ModelTask::TextEmbedding);

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
    let catalog = fresh_catalog_or_skip!(backend, dir);
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

async fn delete_sidecar(store: &ResultStore, record: &ResultTableRecord) {
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let handle = store.open_parquet(&url).unwrap();
    let sidecar = handle.sibling_path("materialization.json").unwrap();
    handle.delete_if_exists(&sidecar).await.unwrap();
}
