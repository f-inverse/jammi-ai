//! Data-plane client registry-listing parity and the Flight-SQL tenant-scope
//! proof.
//!
//! These cover the wire surface added with the strict-client CLI:
//!
//! * **`sql` tenant scope (the #1 correctness proof)** — a `DataClient` that
//!   binds tenant A and runs a `SELECT` over the Flight SQL lane sees *only*
//!   tenant A's rows; a second session bound to tenant B sees only B's. This is
//!   the test that catches the silent-unscoped-exposure failure: `bind_tenant`
//!   and `sql` must stamp the *same* session id so the server resolves the query
//!   to the bound tenant rather than the unscoped (all-tenants) default.
//! * **`list_models` / `list_channels` / `list_mutable_tables` parity** — each
//!   remote listing (through the data client's composed `CatalogClient`) returns
//!   the same records a local `Session` over the same engine returns.
//!
//! Hermetic: a small `tenant_id`-columned Parquet fixture is written per test
//! (the analyzer scopes on the schema's own `tenant_id` column, so no
//! out-of-band tenant-column declaration is needed); no live network.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use jammi_ai::Session;
use jammi_client::DataClient;
use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::mutable::MutableTableDefinitionBuilder;
use jammi_db::ModelTask;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use tonic::transport::Endpoint;

use super::common::grpc::{start_engine_server, tenant_a, tenant_b, EngineServer, TENANT_A};

async fn remote(server: &EngineServer) -> DataClient {
    let endpoint = Endpoint::from_shared(format!("http://{}", server.addr)).expect("endpoint");
    DataClient::connect(endpoint)
        .await
        .expect("data client connect")
}

fn local(server: &EngineServer) -> Session {
    Session::new(Arc::clone(&server.engine))
}

/// Write a Parquet file with 10 rows split 6 (tenant A) + 4 (tenant B), keyed
/// by a literal `tenant_id` column so the engine's tenant analyzer scopes on it
/// directly. Returns the file path under `dir`.
fn write_tenant_parquet(dir: &TempDir) -> std::path::PathBuf {
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("note_id", DataType::Int64, false),
        Field::new("tenant_id", DataType::Utf8, true),
    ]));
    let note_ids = Int64Array::from((0..10_i64).collect::<Vec<_>>());
    let a = tenant_a().to_string();
    let b = tenant_b().to_string();
    let tenants: Vec<&str> = (0..10)
        .map(|i| if i < 6 { a.as_str() } else { b.as_str() })
        .collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(note_ids) as ArrayRef,
            Arc::new(StringArray::from(tenants)) as ArrayRef,
        ],
    )
    .expect("batch");
    let path = dir.path().join("notes.parquet");
    let file = std::fs::File::create(&path).expect("create parquet");
    let mut writer =
        ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

async fn count_via_remote(server: &EngineServer, tenant: &str) -> i64 {
    let session = remote(server).await;
    session
        .bind_tenant(tenant.parse().expect("tenant uuid"))
        .await
        .expect("bind_tenant");
    let batches = session
        .sql("SELECT COUNT(*) AS n FROM notes.public.notes")
        .await
        .expect("remote sql");
    let mut total = 0_i64;
    for batch in &batches {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 count column");
        for i in 0..col.len() {
            total += col.value(i);
        }
    }
    total
}

/// The mandatory tenant-scope proof: a `--tenant A` query over a `DataClient`
/// returns only tenant A's rows, and a `--tenant B` session sees a different
/// (B-only) count — over the Flight SQL lane, where `bind_tenant` and `sql`
/// stamp the same session id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_sql_is_tenant_scoped() {
    let server = start_engine_server().await;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_tenant_parquet(&dir);

    // Register the source through a remote session (no tenant bound — the
    // source registry is global here; the tenant scope applies to the *query*).
    let admin = remote(&server).await;
    admin
        .catalog()
        .add_source(
            "notes",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", path.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .expect("add_source");

    let count_a = count_via_remote(&server, TENANT_A).await;
    let count_b = count_via_remote(&server, &tenant_b().to_string()).await;

    assert_eq!(
        count_a, 6,
        "a DataClient bound to tenant A must see only A's 6 rows over Flight SQL"
    );
    assert_eq!(
        count_b, 4,
        "a DataClient bound to tenant B must see only B's 4 rows over Flight SQL"
    );

    server.shutdown.send(()).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_list_models_matches_local() {
    let server = start_engine_server().await;
    server
        .engine
        .catalog()
        .register_model(RegisterModelParams {
            model_id: "acme/embed-mini",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .expect("register_model");

    let remote_models = remote(&server)
        .await
        .catalog()
        .list_models()
        .await
        .expect("remote");
    let local_models = local(&server).list_models().await.expect("local");

    let remote_ids: Vec<&str> = remote_models.iter().map(|m| m.model_id.as_str()).collect();
    let local_ids: Vec<&str> = local_models.iter().map(|m| m.model_id.as_str()).collect();
    assert_eq!(
        remote_ids, local_ids,
        "remote list_models must match local list_models"
    );
    assert!(
        remote_ids.contains(&"acme/embed-mini"),
        "registered model must surface in the remote listing: {remote_ids:?}"
    );
    let registered = remote_models
        .iter()
        .find(|m| m.model_id == "acme/embed-mini")
        .expect("registered model present");
    assert_eq!(registered.backend, "candle");
    assert_eq!(registered.task, ModelTask::TextEmbedding);

    server.shutdown.send(()).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_list_channels_matches_local() {
    let server = start_engine_server().await;
    let spec = ChannelSpec {
        id: jammi_db::ChannelId::new("scored_by").expect("channel id"),
        priority: 3,
        columns: vec![
            ChannelColumn {
                name: "ranker".into(),
                data_type: ChannelColumnType::Utf8,
            },
            ChannelColumn {
                name: "rank_score".into(),
                data_type: ChannelColumnType::Float32,
            },
        ],
    };
    local(&server)
        .register_channel(&spec)
        .await
        .expect("register channel");

    let remote_channels = remote(&server)
        .await
        .catalog()
        .list_channels()
        .await
        .expect("remote");
    let local_channels = local(&server).list_channels().await.expect("local");

    assert_eq!(
        remote_channels.len(),
        local_channels.len(),
        "channel count must match across transports"
    );
    let ch = remote_channels
        .iter()
        .find(|c| c.id.as_str() == "scored_by")
        .expect("registered channel present remotely");
    assert_eq!(ch.priority, 3);
    let col_names: Vec<&str> = ch.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(col_names, vec!["ranker", "rank_score"]);

    server.shutdown.send(()).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_list_mutable_tables_matches_local() {
    let server = start_engine_server().await;
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("feature_id", DataType::Int64, false),
        Field::new("feature_value", DataType::Float64, true),
    ]));
    let def = MutableTableDefinitionBuilder::new(
        jammi_db::store::mutable::MutableTableId::new("feature_store").expect("id"),
        schema,
    )
    .primary_key(vec!["feature_id".to_string()])
    .tenant(None)
    .build()
    .expect("definition");
    local(&server)
        .create_mutable_table(def)
        .await
        .expect("create mutable table");

    let remote_tables = remote(&server)
        .await
        .catalog()
        .list_mutable_tables()
        .await
        .expect("remote");
    let local_tables = local(&server).list_mutable_tables().await.expect("local");

    let remote_ids: Vec<String> = remote_tables.iter().map(|d| d.id.to_string()).collect();
    let local_ids: Vec<String> = local_tables.iter().map(|d| d.id.to_string()).collect();
    assert_eq!(
        remote_ids, local_ids,
        "remote list_mutable_tables must match local"
    );
    assert!(
        remote_ids.iter().any(|id| id == "feature_store"),
        "registered mutable table must surface remotely: {remote_ids:?}"
    );

    server.shutdown.send(()).ok();
}
