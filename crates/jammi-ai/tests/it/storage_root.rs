//! `InferenceSession` honours `config.storage.result_root`: when set, result
//! tables are rooted there (here a hermetic `memory://` URL standing in for an
//! `r2://`/`s3://` deploy root) rather than on local disk under `artifact_dir`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use candle_core::{Device, Tensor};
use jammi_ai::session::InferenceSession;
use jammi_db::config::StorageConfig;
use jammi_db::model_task::ModelTask;
use jammi_db::storage::{StorageRegistry, StorageUrl};
use jammi_db::store::ArtifactStore;
use tempfile::TempDir;

use crate::common;

/// With `storage.result_root` set to a `memory://` URL, the session's result
/// store creates tables under that root and round-trips a batch back — proving
/// the configured cloud root threads from `JammiConfig` into the `ResultStore`
/// without touching local disk for the table data. The catalog (SQLite under
/// the temp `artifact_dir`) is unaffected.
#[tokio::test]
async fn inference_session_roots_result_tables_at_configured_memory_root() {
    let dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.storage = StorageConfig {
        result_root: Some("memory:///jammi_results".into()),
        cloud: None,
    };

    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    let store = session.result_store();

    let info = store
        .create_table(
            "patents",
            ModelTask::Classification,
            jammi_db::catalog::result_repo::ResultTableKind::Model,
            None,
            "model",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert!(
        info.parquet_url
            .as_str()
            .starts_with("memory:///jammi_results/"),
        "result table not rooted at the configured memory root: {}",
        info.parquet_url
    );

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec!["x", "y"])) as ArrayRef],
    )
    .unwrap();
    let mut writer = store
        .open_writer(&info.parquet_url, Arc::clone(&schema))
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    let rows = writer.close().await.unwrap();
    assert_eq!(rows, 2);

    // Nothing was written under the local jammi_db dir — the table lives in
    // the in-memory root.
    let local_db = dir.path().join("jammi_db");
    let has_parquet = local_db.exists()
        && std::fs::read_dir(&local_db)
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().is_some_and(|x| x == "parquet"));
    assert!(
        !has_parquet,
        "result-table parquet leaked to local disk under {local_db:?}"
    );
}

/// Cross-host model-artifact sharing (Shape-D): a model artifact a worker writes
/// to a shared object store on one host is fetched and loaded by a *different*
/// host with no shared local disk.
///
/// The two hosts are modelled as two [`ArtifactStore`]s over the SAME shared
/// object store (a `memory://` root standing in for `s3://`/`r2://`, with the
/// driver shared via one [`StorageRegistry`] so the in-memory bytes are visible
/// to both — exactly as a real bucket is) but with **distinct local fetch
/// caches** (each host's own disk). Host A writes a real safetensors bundle
/// under a unique per-attempt prefix; host B, whose cache is empty, fetches by
/// that prefix — the only place the bytes can come from is the shared object
/// store — verifies it (manifest + sha256), and loads the weights into candle.
/// Proving the artifact written on host A is usable on host B is precisely the
/// cross-host worker-fleet guarantee.
#[tokio::test]
async fn artifact_written_on_host_a_is_loadable_on_host_b() {
    // One shared object store (the "bucket"): both hosts' stores resolve the
    // same `memory://` driver through this single registry.
    let registry = StorageRegistry::new();
    let root = StorageUrl::parse("memory:///jammi_results/models").unwrap();

    let host_a_cache = TempDir::new().unwrap();
    let host_b_cache = TempDir::new().unwrap();
    let store_a =
        ArtifactStore::with_root(root.clone(), registry.clone(), host_a_cache.path().into())
            .unwrap();
    let store_b = ArtifactStore::with_root(root, registry, host_b_cache.path().into()).unwrap();

    // Host A trains and publishes a real safetensors adapter under a unique
    // per-attempt prefix, exactly as the fine-tune worker does.
    let device = Device::Cpu;
    let weight = Tensor::randn(0.0f32, 1.0, (4, 8), &device).unwrap();
    let buf = {
        let mut map = HashMap::new();
        map.insert("lora.weight".to_string(), weight);
        let tmp = host_a_cache.path().join("tmp.safetensors");
        candle_core::safetensors::save(&map, &tmp).unwrap();
        std::fs::read(&tmp).unwrap()
    };

    let files = vec![
        ("adapter.safetensors".to_string(), Bytes::from(buf.clone())),
        (
            "adapter_config.json".to_string(),
            Bytes::from_static(b"{\"adapter_type\":\"projection_head\"}"),
        ),
    ];
    let prefix = store_a
        .put_artifact(&["job-x", "worker-a", "0"], &files)
        .await
        .unwrap();

    // Host B has never seen this artifact: its local cache is empty.
    assert!(
        std::fs::read_dir(host_b_cache.path())
            .unwrap()
            .next()
            .is_none(),
        "host B's cache starts empty — it has no local copy of the artifact"
    );

    // Host B fetches by the prefix host A recorded. The bytes can only come from
    // the shared object store; they land in host B's own cache, verified.
    let local = store_b.fetch_artifact(&prefix).await.unwrap();
    assert!(
        local.dir().starts_with(host_b_cache.path()),
        "host B materialises the artifact into its own local cache"
    );

    // The fetched adapter loads into candle and round-trips byte-identically —
    // the weights trained on host A are usable on host B.
    let loaded =
        candle_core::safetensors::load(local.dir().join("adapter.safetensors"), &device).unwrap();
    let got = loaded.get("lora.weight").expect("weight present");
    assert_eq!(got.dims(), &[4, 8], "loaded weight has the trained shape");
    let raw = std::fs::read(local.dir().join("adapter.safetensors")).unwrap();
    assert_eq!(raw, buf, "host B reads back the exact bytes host A wrote");
}
