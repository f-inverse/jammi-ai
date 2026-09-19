//! Crash-consistency of the materialization contract under `SIGKILL`, in the
//! window the contract must survive: a result table's Parquet bytes are durable
//! but its `.materialization.json` manifest has not yet been written and the
//! `building -> ready` flip has not committed.
//!
//! [`jammi_db::store::BuildingTable::finish`] is the single `building -> ready`
//! boundary. It writes the manifest sidecar BEFORE the status flip (the same
//! ordering the ANN sidecar uses), so a crash never leaves a `ready` table
//! without a manifest. The hardest window is *before* the sidecar lands: a valid
//! Parquet, no manifest, status still `building`. Recovery cannot reconstruct
//! the producing descriptor, so it must reap that row to `failed`, never promote
//! it manifest-less.
//!
//! The parent runs [`manifestless_parquet_child`] in its own process with
//! `JAMMI_TEST_MATERIALIZATION_CHECKPOINT` set; the child drives `finish`, whose
//! test hook fires after the lease renew and the durable Parquet and before the
//! manifest write, and parks. The parent `SIGKILL`s it, restarts on the same
//! directory, runs `recover()`, and asserts the row is `failed` and the bytes
//! reaped. SQLite only; compiled under `test-hooks`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use jammi_db::catalog::result_repo::ResultTableKind;
use jammi_db::config::AnnIndexConfig;
use jammi_db::model_task::ModelTask;
use jammi_db::session::JammiSession;
use jammi_db::store::manifest::{
    ComputeDevice, InputAnchor, MaterializationEnv, ProducingDescriptor,
};
use jammi_db::store::mutable::test_hook::MATERIALIZATION_CHECKPOINT_ENV;
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::store::ResultStore;

use crate::common;

/// The lease the child's store holds its building row under: short, so the
/// parent's recovery after the SIGKILL reclaims it within seconds.
fn short_lease() -> jammi_db::catalog::lease::LeaseIntervals {
    jammi_db::config::LeaseConfig {
        duration_secs: 3,
        heartbeat_secs: 1,
    }
    .intervals()
    .unwrap()
}
const TABLE_SOURCE: &str = "crash_docs";
const DIMS: usize = 4;

async fn child_workload() {
    let dir = std::env::var(common::ARTIFACT_DIR_ENV).expect("child needs artifact dir");
    let dir = PathBuf::from(dir);
    let session = JammiSession::new(common::test_config(&dir))
        .await
        .expect("child session");

    // A short lease so the parent's post-kill recovery sees the dead writer's
    // row as reclaimable within seconds rather than the 30 s default.
    let store = ResultStore::new(
        &dir,
        Arc::clone(session.catalog()),
        AnnIndexConfig::default(),
    )
    .unwrap()
    .with_lease_intervals(short_lease());

    let info = store
        .create_table(
            TABLE_SOURCE,
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "crash-model",
            Some(DIMS as i32),
            Some("_row_id"),
            Some("body"),
            None,
        )
        .await
        .unwrap();

    // Write a valid, closed Parquet — the durable bytes the crash leaves behind.
    let schema = embedding_table_schema(DIMS);
    let row_ids = StringArray::from_iter_values(["row-0", "row-1"]);
    let sources = StringArray::from_iter_values([TABLE_SOURCE, TABLE_SOURCE]);
    let models = StringArray::from_iter_values(["crash-model", "crash-model"]);
    let item = Arc::new(arrow_schema::Field::new(
        "item",
        arrow_schema::DataType::Float32,
        false,
    ));
    let flat: Vec<f32> = (0..2 * DIMS).map(|i| i as f32).collect();
    let vectors =
        FixedSizeListArray::try_new(item, DIMS as i32, Arc::new(Float32Array::from(flat)), None)
            .unwrap();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(row_ids),
            Arc::new(sources),
            Arc::new(models),
            Arc::new(vectors),
            jammi_db::store::content_hash::null_hash_column(2),
        ],
    )
    .unwrap();
    let mut writer = store.open_writer(info.parquet_url(), schema).await.unwrap();
    writer.write_batch(&batch).await.unwrap();
    let rows = writer.close().await.unwrap();

    let descriptor = ProducingDescriptor::Embedding {
        model_id: "crash-model".into(),
        task: ModelTask::TextEmbedding,
        source_id: TABLE_SOURCE.into(),
        columns: vec!["body".into()],
        key_column: "_row_id".into(),
        dimensions: DIMS,
    };
    let env = MaterializationEnv::new(ComputeDevice::Cpu, Vec::new());
    let inputs = vec![InputAnchor::unpinned_at_instant(
        TABLE_SOURCE,
        "1970-01-01T00:00:00Z",
    )];

    // The hook fires inside `finish` AFTER the lease renew and the Parquet
    // bytes are durable and BEFORE the manifest sidecar is written; it parks
    // the child here.
    info.finish(
        session.context(),
        rows,
        jammi_db::store::manifest::Materialization::new(&descriptor, &env, inputs),
    )
    .await
    .unwrap();

    unreachable!("hook parks the child; SIGKILL is the only exit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "child process of manifestless_parquet_is_reaped_under_sigkill"]
async fn manifestless_parquet_child() {
    child_workload().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn manifestless_parquet_is_reaped_under_sigkill() {
    let dir = tempfile::tempdir().unwrap();
    common::kill_child_at_checkpoint(
        "materialization_crash_recovery::manifestless_parquet_child",
        dir.path(),
        (MATERIALIZATION_CHECKPOINT_ENV, "1"),
    )
    .await;

    // Restart on the same dir. The dead child's row is `building` under a
    // lease that is still live for a few seconds: recovery must leave it
    // alone until the lease expires (a live-lease row may belong to a writer
    // that is merely slow), then reap it — a valid Parquet with no manifest
    // goes to `failed`.
    let restart = JammiSession::new(common::test_config(dir.path()))
        .await
        .expect("restart session");
    let store = ResultStore::new(
        dir.path(),
        Arc::clone(restart.catalog()),
        AnnIndexConfig::default(),
    )
    .unwrap();
    let building_before = restart
        .catalog()
        .list_result_tables_by_status(jammi_db::catalog::status::ResultTableStatus::Building)
        .await
        .unwrap();
    assert_eq!(
        building_before.len(),
        1,
        "precondition: the killed child left exactly one `building` row"
    );
    assert!(
        building_before[0].lease_expires_at.is_some(),
        "precondition: the row carries the dead writer's lease"
    );
    let reap_deadline = Instant::now() + Duration::from_secs(20);
    loop {
        store.recover().await.expect("recover after crash");
        let building = restart
            .catalog()
            .list_result_tables_by_status(jammi_db::catalog::status::ResultTableStatus::Building)
            .await
            .unwrap();
        if building.is_empty() {
            break;
        }
        assert!(
            Instant::now() < reap_deadline,
            "recovery never reaped the dead writer's row once its lease expired"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let failed = restart
        .catalog()
        .list_result_tables_by_status(jammi_db::catalog::status::ResultTableStatus::Failed)
        .await
        .unwrap();
    assert_eq!(
        failed.len(),
        1,
        "the manifest-less torn write was reaped to exactly one `failed` row"
    );
    let url = jammi_db::storage::StorageUrl::parse(&failed[0].parquet_path).unwrap();
    let handle = store.open_parquet(&url).unwrap();
    let path = handle.data_path().unwrap();
    assert!(
        !handle.exists(&path).await.unwrap(),
        "the torn Parquet bytes were reaped"
    );
}
