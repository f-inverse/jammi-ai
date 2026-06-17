//! The materialization contract — adversarial oracle over the `MatchVerdict`
//! surface, the single-funnel guarantee, and recovery's manifest awareness.
//!
//! Every result table is published through the one `building -> ready` boundary
//! [`ResultStore::finalize_with_manifest`], which writes a `.materialization.json`
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

use arrow::array::{FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use datafusion::prelude::SessionContext;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::model_task::ModelTask;
use jammi_db::store::manifest::{
    AnchorKind, ComputeDevice, DefinitionHash, InputAnchor, MatchVerdict, MaterializationEnv,
    ModelIdentity, ProducingDescriptor,
};
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::store::{ResultStore, ResultTableInfo};
use tempfile::tempdir;

const DIMS: usize = 4;

async fn fresh_catalog(dir: &std::path::Path) -> Arc<Catalog> {
    let backend = SqliteBackend::open(&dir.join("catalog.db")).await.unwrap();
    let backend = jammi_db::catalog::backend::BackendImpl::Sqlite(backend);
    backend.migrate().await.unwrap();
    Arc::new(Catalog::from_backend(backend))
}

fn store(dir: &std::path::Path, catalog: Arc<Catalog>) -> ResultStore {
    ResultStore::new(dir, catalog, AnnIndexConfig::default()).unwrap()
}

async fn create_building(store: &ResultStore) -> ResultTableInfo {
    store
        .create_table(
            "docs",
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "test-model",
            Some(DIMS as i32),
            Some("_row_id"),
            Some("body"),
        )
        .await
        .unwrap()
}

async fn write_embedding_parquet(store: &ResultStore, info: &ResultTableInfo, n: usize) -> usize {
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
        ],
    )
    .unwrap();
    let mut writer = store.open_writer(&info.parquet_url, schema).await.unwrap();
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
    let manifest = store
        .finalize_with_manifest(
            ctx,
            &info.table_name,
            &info.parquet_url,
            rows,
            jammi_db::store::manifest::Materialization::new(&descriptor(), &env(), inputs),
        )
        .await
        .unwrap();
    let record = store
        .catalog()
        .get_result_table(&info.table_name)
        .await
        .unwrap()
        .expect("record after materialize");
    (record, manifest.definition_hash)
}

#[tokio::test]
async fn verdict_match_for_an_untouched_table() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn verdict_mismatch_against_a_wrong_expected_hash() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn verdict_mismatch_when_the_data_is_tampered() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn verdict_match_with_unpinned_inputs() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn verdict_missing_manifest_for_a_pre_contract_table() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));

    // A pre-contract table: bytes + a `ready` catalog row, but NO manifest
    // sidecar and NO definition_hash summary column — exactly a table created
    // before migration 021.
    let info = create_building(&store).await;
    let rows = write_embedding_parquet(&store, &info, 3).await;
    catalog
        .update_result_table_status(&info.table_name, ResultTableStatus::Ready, rows)
        .await
        .unwrap();
    let record = store
        .catalog()
        .get_result_table(&info.table_name)
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

#[tokio::test]
async fn the_funnel_persists_sidecar_and_summary_columns() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn recovery_reaps_a_torn_manifestless_building_row() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));

    // Construct the torn state the crash window leaves: a `building` row whose
    // Parquet is valid but whose manifest never landed (no `.materialization.json`,
    // status still `building`).
    let info = create_building(&store).await;
    write_embedding_parquet(&store, &info, 3).await;
    let url = jammi_db::storage::StorageUrl::parse(&info.parquet_url.to_string()).unwrap();
    assert!(
        store
            .read_materialization_manifest(&url)
            .await
            .unwrap()
            .is_none(),
        "the torn state has no manifest"
    );

    store.recover().await.unwrap();

    // Recovery cannot reconstruct the producing descriptor, so a manifest-less
    // valid Parquet is reaped to `failed`, never promoted manifest-less.
    let record = store
        .catalog()
        .get_result_table(&info.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(record.status, "failed");
    let handle = store.open_parquet(&info.parquet_url).unwrap();
    let path = handle.data_path().unwrap();
    assert!(
        !handle.exists(&path).await.unwrap(),
        "the torn Parquet bytes are reaped"
    );
}

#[tokio::test]
async fn recovery_promotes_a_building_row_whose_manifest_landed() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

    store.recover().await.unwrap();

    let record = store
        .catalog()
        .get_result_table(&info.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(record.status, "ready");
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

#[tokio::test]
async fn recovery_reaps_a_post_contract_ready_table_whose_sidecar_vanished() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

// --- helpers reaching the store's manifest sidecar for torn-state setup ----

async fn store_artifact_digest(
    store: &ResultStore,
    info: &ResultTableInfo,
) -> jammi_db::store::manifest::ArtifactDigest {
    let handle = store.open_parquet(&info.parquet_url).unwrap();
    let path = handle.data_path().unwrap();
    let bytes = handle.get_bytes(&path).await.unwrap();
    jammi_db::store::manifest::ArtifactDigest::of_bytes(&bytes)
}

async fn write_sidecar(
    store: &ResultStore,
    info: &ResultTableInfo,
    manifest: &jammi_db::store::manifest::MaterializationManifest,
) {
    let handle = store.open_parquet(&info.parquet_url).unwrap();
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
