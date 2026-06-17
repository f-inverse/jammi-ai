//! Crash-consistency of the materialization contract under `SIGKILL`, in the
//! window the contract must survive: a result table's Parquet bytes are durable
//! but its `.materialization.json` manifest has not yet been written and the
//! `building -> ready` flip has not committed.
//!
//! [`ResultStore::finalize_with_manifest`] is the single `building -> ready`
//! boundary. It writes the manifest sidecar BEFORE the status flip (the same
//! ordering the ANN sidecar uses), so a crash never leaves a `ready` table
//! without a manifest. The hardest window is *before* the sidecar lands: a valid
//! Parquet, no manifest, status still `building`. Recovery cannot reconstruct
//! the producing descriptor, so it must reap that row to `failed`, never promote
//! it manifest-less.
//!
//! Mechanics mirror `mutable_crash_recovery.rs`: the parent respawns the test
//! binary with `JAMMI_TEST_MATERIALIZATION_CHECKPOINT` set; the child drives
//! `finalize_with_manifest`, whose test-hook fires after the Parquet is durable
//! and before the manifest write, writes the ready file, and parks. The parent
//! `SIGKILL`s, restarts on the same dir, runs `recover()`, and asserts the row
//! is `failed` and the bytes reaped.
//!
//! Gated behind `feature = "test-hooks"` (the SIGKILL harness, SQLite only).

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
use jammi_db::store::mutable::test_hook::{MATERIALIZATION_CHECKPOINT_ENV, READY_FILE_ENV};
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::store::ResultStore;

use crate::common;

const CHILD_MARKER_ENV: &str = "JAMMI_TEST_CRASH_CHILD";
const ARTIFACT_DIR_ENV: &str = "JAMMI_TEST_ARTIFACT_DIR";
const TABLE_SOURCE: &str = "crash_docs";
const DIMS: usize = 4;

async fn child_workload() {
    let dir = std::env::var(ARTIFACT_DIR_ENV).expect("child needs artifact dir");
    let dir = PathBuf::from(dir);
    let session = JammiSession::new(common::test_config(&dir))
        .await
        .expect("child session");

    let store = ResultStore::new(
        &dir,
        Arc::clone(session.catalog()),
        AnnIndexConfig::default(),
    )
    .unwrap();

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
        ],
    )
    .unwrap();
    let mut writer = store.open_writer(&info.parquet_url, schema).await.unwrap();
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

    // The hook fires inside finalize_with_manifest AFTER the Parquet is durable
    // and BEFORE the manifest sidecar is written; it parks the child here.
    store
        .finalize_with_manifest(
            session.context(),
            &info.table_name,
            &info.parquet_url,
            rows,
            jammi_db::store::manifest::Materialization::new(&descriptor, &env, inputs),
        )
        .await
        .unwrap();

    unreachable!("hook parks the child; SIGKILL is the only exit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn manifestless_parquet_is_reaped_under_sigkill() {
    const NAME: &str =
        "materialization_crash_recovery::manifestless_parquet_is_reaped_under_sigkill";
    if std::env::var(CHILD_MARKER_ENV).is_ok() {
        child_workload().await;
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let ready_file = dir.path().join("ready");
    let exe = std::env::current_exe().expect("current_exe for child spawn");

    let mut child = tokio::process::Command::new(&exe)
        .args(["--exact", "--nocapture", NAME])
        .env(CHILD_MARKER_ENV, "1")
        .env(ARTIFACT_DIR_ENV, dir.path())
        .env(READY_FILE_ENV, &ready_file)
        .env(MATERIALIZATION_CHECKPOINT_ENV, "1")
        .spawn()
        .expect("spawn child test process");

    let pid = child.id().expect("child pid available") as i32;

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if tokio::fs::try_exists(&ready_file)
            .await
            .expect("try_exists ready file")
        {
            break;
        }
        if Instant::now() > deadline {
            let _ = child.kill().await;
            panic!("child never reached the materialization checkpoint within 30s");
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!("child exited before the materialization checkpoint: {status:?}");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // SAFETY: `pid` is the child we just spawned; `SIGKILL` is unconditional and
    // synchronous. No allocator or thread state is touched.
    let rc = unsafe { libc::kill(pid, libc::SIGKILL) };
    assert_eq!(
        rc,
        0,
        "libc::kill failed: {}",
        std::io::Error::last_os_error()
    );
    let _ = child.wait().await;

    // Restart on the same dir and run the real recovery sweep. The torn row has
    // a valid Parquet but no manifest, so recovery reaps it to `failed`.
    let restart = JammiSession::new(common::test_config(dir.path()))
        .await
        .expect("restart session");
    let store = ResultStore::new(
        dir.path(),
        Arc::clone(restart.catalog()),
        AnnIndexConfig::default(),
    )
    .unwrap();
    store.recover().await.expect("recover after crash");

    // Find the (single) crash-target row and assert it failed with no live bytes.
    let building = restart
        .catalog()
        .list_result_tables_by_status(jammi_db::catalog::status::ResultTableStatus::Building)
        .await
        .unwrap();
    assert!(
        building.is_empty(),
        "recovery left no row `building` — reconciliation is terminal"
    );
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
