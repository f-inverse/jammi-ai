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
use jammi_datafusion::ModelTask;
use jammi_db::catalog::instance::MemberRoot;
use jammi_db::config::StorageConfig;
use jammi_db::storage::{StorageRegistry, StorageUrl};
use jammi_db::store::ArtifactStore;
use tempfile::TempDir;

use crate::common;

/// `InferenceSession`'s result store is rooted at EXACTLY
/// `JammiConfig::resolved_result_root()`'s own value — the SAME string a
/// gang member's `instances.result_root` row carries verbatim — never a second, independently
/// re-derived path, for both arms (`storage.result_root` unset and set). Proven by creating a table
/// and checking its `parquet_url` starts with the resolved root.
async fn assert_store_rooted_at_resolved_root(config: jammi_db::config::JammiConfig) {
    let expected = StorageUrl::parse(&config.resolved_result_root().unwrap()).unwrap();
    let session = InferenceSession::new(config).await.unwrap();
    let store = session.result_store();
    let info = store
        .create_table(
            "root_parity_probe",
            ModelTask::Classification,
            jammi_db::catalog::result_repo::ResultTableKind::Model,
            None,
            "model",
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert!(
        info.parquet_url().as_str().starts_with(expected.as_str()),
        "store root {} does not match resolved_result_root() {}",
        info.parquet_url(),
        expected
    );
}

/// `storage.result_root` UNSET: the store roots at `{artifact_dir}/jammi_db`,
/// exactly what `resolved_result_root()` names.
#[tokio::test]
async fn store_root_matches_resolved_result_root_when_unset() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    assert_store_rooted_at_resolved_root(config).await;
}

/// `storage.result_root` SET (to a `memory://` root): the store roots
/// exactly there, again matching `resolved_result_root()`.
#[tokio::test]
async fn store_root_matches_resolved_result_root_when_set() {
    let dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.storage = StorageConfig {
        result_root: Some("memory:///jammi_root_parity".into()),
        cloud: None,
    };
    assert_store_rooted_at_resolved_root(config).await;
}

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
            None,
        )
        .await
        .unwrap();
    assert!(
        info.parquet_url()
            .as_str()
            .starts_with("memory:///jammi_results/"),
        "result table not rooted at the configured memory root: {}",
        info.parquet_url()
    );

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec!["x", "y"])) as ArrayRef],
    )
    .unwrap();
    let mut writer = store
        .open_writer(info.parquet_url(), Arc::clone(&schema))
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

// ---------------------------------------------------------------------------
// The `instances.result_root` column carries `resolved_result_root()` VERBATIM —
// the SAME string the result store is rooted at — through a REAL session.
// No filesystem access, no interpretation, no scheme aliasing.
// ---------------------------------------------------------------------------

/// With `[server] peer_advertise`/`peer_bind` set, `config` produces a
/// member row whose `result_root` equals `resolved_result_root()`, which in
/// turn equals the prefix every table this session creates is rooted at.
async fn assert_member_row_matches_resolved_root(
    mut config: jammi_db::config::JammiConfig,
    port: u16,
) {
    config.server.peer_bind = Some(format!("0.0.0.0:{port}"));
    config.server.peer_advertise = Some(format!("127.0.0.1:{port}"));

    let expected = config.resolved_result_root().unwrap();
    // The store parses `resolved_result_root()` as a `StorageUrl` (a bare
    // local path becomes a `file://` URL — `build_result_store`'s own
    // parse); the `instances.result_root` column, checked below, carries
    // the PRE-parse string verbatim.
    let expected_store_root = StorageUrl::parse(&expected).unwrap();
    let expected_identity = MemberRoot::resolved(&config)
        .unwrap()
        .identity()
        .as_str()
        .to_string();
    let session = InferenceSession::new(config).await.unwrap();
    let store = session.result_store();
    let info = store
        .create_table(
            "member_row_root_probe",
            ModelTask::Classification,
            jammi_db::catalog::result_repo::ResultTableKind::Model,
            None,
            "model",
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert!(
        info.parquet_url()
            .as_str()
            .starts_with(expected_store_root.as_str()),
        "store root {} does not match resolved_result_root() {}",
        info.parquet_url(),
        expected_store_root
    );

    let instance_id = session.instance_id().to_string();
    let row: Option<(Option<String>, Option<String>)> = session
        .catalog()
        .backend_arc()
        .transaction(
            jammi_db::catalog::backend::TxOptions::default(),
            move |tx| {
                let instance_id = instance_id.clone();
                Box::pin(async move {
                    tx.query_opt(
                        "SELECT result_root, result_root_identity FROM instances \
                         WHERE instance_id = $1",
                        &[jammi_db::catalog::backend::SqlValue::TextOwned(instance_id)],
                        |row| {
                            Ok((
                                row.try_get::<String>("result_root")?,
                                row.try_get::<String>("result_root_identity")?,
                            ))
                        },
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    let (root, identity) = row.expect("the member row exists");
    assert_eq!(
        root.as_deref(),
        Some(expected.as_str()),
        "instances.result_root must carry resolved_result_root() verbatim"
    );
    assert_eq!(
        identity.as_deref(),
        Some(expected_identity.as_str()),
        "instances.result_root_identity must be the identity MemberRoot::resolved derives"
    );
}

/// `result_root` UNSET: the member row carries `{artifact_dir}/jammi_db`.
#[tokio::test]
async fn member_row_matches_resolved_root_when_unset() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    assert_member_row_matches_resolved_root(config, 19301).await;
}

/// `result_root` an explicit `file://` root.
#[tokio::test]
async fn member_row_matches_resolved_root_for_file_scheme() {
    let dir = TempDir::new().unwrap();
    let root_dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.storage = StorageConfig {
        result_root: Some(format!("file://{}", root_dir.path().to_string_lossy())),
        cloud: None,
    };
    assert_member_row_matches_resolved_root(config, 19302).await;
}

/// `result_root` a `memory://` root: a MEMBER (peer_advertise set) is
/// refused at session construction, naming the root and the way out — an
/// in-memory store can never be shared with a gang peer.
/// The same root with no `peer_advertise` is a plain library session.
#[tokio::test]
async fn a_member_with_a_memory_result_root_is_refused_at_session_construction() {
    let dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.storage = StorageConfig {
        result_root: Some("memory:///jammi_member_row_probe".into()),
        cloud: None,
    };
    let library = InferenceSession::new(config.clone()).await;
    assert!(
        library.is_ok(),
        "a library session may use an in-memory root"
    );
    drop(library);
    config.server.peer_bind = Some("0.0.0.0:19303".to_string());
    config.server.peer_advertise = Some("127.0.0.1:19303".to_string());
    let err = match InferenceSession::new(config).await {
        Ok(_) => panic!("a member with an in-memory root must be refused"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("memory:///jammi_member_row_probe") && err.contains("peer_advertise"),
        "{err}"
    );
}

/// `result_root` a `gcs://` ALIAS scheme — proving the alias is never
/// folded (to `gs://` or anything else) on the membership path either.
#[tokio::test]
async fn member_row_matches_resolved_root_for_a_cloud_alias_scheme() {
    let dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.storage = StorageConfig {
        result_root: Some("gcs://bucket/jammi_member_row_probe".into()),
        cloud: None,
    };
    let expected = config.resolved_result_root().unwrap();
    assert_eq!(expected, "gcs://bucket/jammi_member_row_probe");

    config.server.peer_bind = Some("0.0.0.0:19304".to_string());
    config.server.peer_advertise = Some("127.0.0.1:19304".to_string());
    let session = InferenceSession::new(config).await.unwrap();
    let instance_id = session.instance_id().to_string();
    let row: Option<String> = session
        .catalog()
        .backend_arc()
        .transaction(
            jammi_db::catalog::backend::TxOptions::default(),
            move |tx| {
                let instance_id = instance_id.clone();
                Box::pin(async move {
                    tx.query_opt(
                        "SELECT result_root FROM instances WHERE instance_id = $1",
                        &[jammi_db::catalog::backend::SqlValue::TextOwned(instance_id)],
                        |row| row.try_get::<String>("result_root"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap()
        .flatten();
    assert_eq!(
        row.as_deref(),
        Some("gcs://bucket/jammi_member_row_probe"),
        "the gcs:// spelling must never fold to gs:// (or anything else) on the member row"
    );
    // …while its IDENTITY is the folded one, equal to the `gs://` spelling's.
    let mut gs = common::test_config(dir.path());
    gs.storage = StorageConfig {
        result_root: Some("gs://bucket/jammi_member_row_probe".into()),
        cloud: None,
    };
    let mut gcs = gs.clone();
    gcs.storage.result_root = Some("gcs://bucket/jammi_member_row_probe".into());
    assert_eq!(
        MemberRoot::resolved(&gcs).unwrap().identity(),
        MemberRoot::resolved(&gs).unwrap().identity()
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
    let catalog_dir = TempDir::new().unwrap();
    let catalog = jammi_db::catalog::Catalog::open(catalog_dir.path())
        .await
        .unwrap();
    let prefix = store_a
        .stage_attempt_artifact(&catalog, "job-x", "worker-a", 0, &files)
        .await
        .unwrap()
        .artifact()
        .url()
        .clone();

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
