//! `RemoteSession` over the wire, proven interchangeable with `LocalSession`.
//!
//! An in-process gRPC chain (`runtime::serve_grpc_chain`) hosts a real
//! `InferenceSession`. A `jammi_ai::RemoteSession` connects to it over a real
//! HTTP/2 channel and a `jammi_ai::LocalSession` wraps the *same* engine `Arc`.
//! The three properties Stage 3b-1 must establish are each pinned here:
//!
//! * **Round-trip parity** — `generate_embeddings` → `search` → `remove_source`
//!   plus `encode_query` through the remote transport return the same results a
//!   local session returns against the same engine (realistic `tiny_bert` text
//!   embeddings over the `patents` corpus, never dummy vectors).
//! * **Error parity (the #1 proof)** — a real failure on this path (searching a
//!   source with no embedding table) returns the *same `JammiError` variant*
//!   from `RemoteSession` as from `LocalSession`, not merely the same gRPC code.
//!   This is what the typed-error wire detail buys; a heuristic reverse-map
//!   could not satisfy it.
//! * **Tenant** — `bind_tenant` (async) over the wire is observed by a later
//!   `tenant()` read; the binding is keyed by the client's session id.
//!
//! Hermetic: the encoder is the local `tiny_bert` cookbook fixture and the
//! corpus is the bundled `patents.parquet`; no live network, no download.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use jammi_ai::audit::{verify_with_store, EnvSigningKeyStore, MASTER_KEY_ENV};
use jammi_ai::{
    Jammi, LocalSession, Modality, PerQueryAudit, QueryInput, RemoteSession, SearchQuery,
    SearchRequest, Session, Target,
};
use jammi_db::error::JammiError;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::trigger::{DeliveredBatch, Predicate, TopicDefinition, TopicId, TriggerError};
use jammi_db::AuditError;
use jammi_test_utils::{cookbook_fixture, fixture, test_config};
use tonic::transport::Endpoint;

use super::common::grpc::{
    start_engine_server, start_engine_server_with_trigger, tenant_a, EngineServer, TENANT_A,
};

/// 32-byte hex master key for the audit HMAC. Deterministic so signature
/// verification is reproducible; matches the engine-level audit tests.
const AUDIT_TEST_KEY: &str = "0000000000000000000000000000000000000000000000000000000000000001";

/// A realistic CDC-event topic schema: an id, a record kind, and an op code.
fn events_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("op", DataType::Utf8, false),
    ]))
}

/// One batch of CDC events matching [`events_schema`].
fn events_batch() -> RecordBatch {
    RecordBatch::try_new(
        events_schema(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["order", "order", "user"])),
            Arc::new(StringArray::from(vec!["c", "u", "d"])),
        ],
    )
    .expect("events batch")
}

/// A global (tenant-`None`) topic with a freshly-minted id. The trigger
/// round-trip tests use an unscoped session on both transports, so a global
/// topic keeps the publish/subscribe tenant scope identical (`None`) across
/// `LocalSession` and `RemoteSession` without a sticky-vs-task-local binding
/// mismatch on the shared engine; the minted id lets a later
/// `drop_topic(topic.id)` resolve on either transport.
fn events_topic() -> TopicDefinition {
    TopicDefinition {
        id: TopicId::new(),
        name: "cdc.events".to_string(),
        schema: events_schema(),
        tenant: None,
        broker_metadata: std::collections::BTreeMap::new(),
    }
}

fn tiny_bert_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

fn patents_connection() -> SourceConnection {
    SourceConnection {
        url: Some(format!("file://{}", fixture("patents.parquet").display())),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    }
}

/// Connect a `RemoteSession` to the in-process server.
async fn remote(server: &EngineServer) -> RemoteSession {
    let endpoint = Endpoint::from_shared(format!("http://{}", server.addr)).expect("endpoint");
    RemoteSession::connect(endpoint)
        .await
        .expect("remote session connect")
}

/// Wrap the server's engine `Arc` in a local session — the same engine the
/// remote calls reach, so any divergence is the transport's fault, not the
/// engine's.
fn local(server: &EngineServer) -> Session {
    Session::Local(LocalSession::new(Arc::clone(&server.engine)))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_round_trips_embeddings_and_search_like_local() {
    let server = start_engine_server().await;
    let remote = Session::Remote(remote(&server).await);
    let model_id = tiny_bert_model_id();

    // The local session registers + embeds the corpus directly on the shared
    // engine; both sessions then search the same persisted embedding table.
    // (AddSource over the remote transport has its own round-trip proof in
    // `remote_add_source_round_trips_like_local`.)
    let local = local(&server);
    local
        .add_source("patents", SourceType::File, patents_connection())
        .await
        .expect("add_source");

    let remote_table = remote
        .generate_embeddings(
            "patents",
            &model_id,
            &["abstract".to_string()],
            "id",
            Modality::Text,
        )
        .await
        .expect("remote generate_embeddings");

    assert_eq!(remote_table.status, "ready");
    assert!(remote_table.row_count > 0, "patents corpus embeds rows");
    assert!(remote_table.dimensions.is_some(), "dimensions recorded");
    assert_eq!(remote_table.source_id, "patents");
    // The remote arm reconstructs `task` from the requested modality (the wire
    // omits it as server-internal bookkeeping); it must match the tower.
    assert_eq!(remote_table.task, jammi_db::ModelTask::TextEmbedding);

    // encode_query parity: identical query, identical model → identical vector.
    let query = "quantum computing applications";
    let remote_vec = remote
        .encode_query(
            &model_id,
            QueryInput::Text(query.to_string()),
            Modality::Text,
        )
        .await
        .expect("remote encode_query");
    let local_vec = local
        .encode_query(
            &model_id,
            QueryInput::Text(query.to_string()),
            Modality::Text,
        )
        .await
        .expect("local encode_query");
    assert_eq!(
        remote_vec.len(),
        local_vec.len(),
        "remote/local query vectors share dimensionality"
    );
    assert_eq!(
        remote_vec, local_vec,
        "the same query through either transport encodes to the same vector"
    );

    // search parity: same request, same top-k keys + scores. The remote arm
    // rebuilds the result batch from the wire hits; compare on the
    // client-observable key/score the search verb surfaces.
    let request = |select: Vec<String>| SearchRequest {
        source_id: "patents".to_string(),
        query: SearchQuery::Vector(remote_vec.clone()),
        k: 5,
        filter: None,
        select,
    };
    let remote_hits = keys_and_scores(
        remote
            .search(request(Vec::new()))
            .await
            .expect("remote search"),
    );
    let local_hits = keys_and_scores(
        local
            .search(request(Vec::new()))
            .await
            .expect("local search"),
    );
    assert!(!remote_hits.is_empty(), "search returns hits");
    assert_eq!(
        remote_hits, local_hits,
        "remote and local search rank the same rows with the same scores"
    );

    // remove_source parity: after the remote drops the source, a search fails
    // on both transports (the source no longer resolves).
    remote
        .remove_source("patents")
        .await
        .expect("remote remove_source");
    let remote_err = remote
        .search(request(Vec::new()))
        .await
        .expect_err("search after remove must fail");
    let local_err = local
        .search(request(Vec::new()))
        .await
        .expect_err("local search after remove must fail too");
    assert_eq!(
        std::mem::discriminant(&remote_err),
        std::mem::discriminant(&local_err),
        "remote and local agree on the failure variant after removal"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// `add_source` over the wire, proven interchangeable with `LocalSession`. The
/// remote transport registers the `patents` corpus through
/// `EmbeddingService.AddSource` (the typed RPC the server and the TS gRPC-web
/// client already drive); the registration must reach the shared engine, so a
/// `generate_embeddings` against that source then succeeds — the source is
/// real and usable, not stubbed. Error parity is pinned too: re-registering the
/// same id fails inside the engine, and both transports reconstruct the same
/// `JammiError` variant + message.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_add_source_round_trips_like_local() {
    let server = start_engine_server().await;
    let remote = Session::Remote(remote(&server).await);
    let local = local(&server);

    // Register the corpus over the REMOTE transport. Before this call the engine
    // has no `patents` source; the registration must cross the wire and land on
    // the shared engine.
    remote
        .add_source("patents", SourceType::File, patents_connection())
        .await
        .expect("remote add_source");

    // The source is usable: embeddings generate over it through the remote
    // transport, producing a ready table with rows — proof the registration took
    // effect, not a silent no-op.
    let remote_table = remote
        .generate_embeddings(
            "patents",
            &tiny_bert_model_id(),
            &["abstract".to_string()],
            "id",
            Modality::Text,
        )
        .await
        .expect("generate_embeddings over the remote-registered source");
    assert_eq!(remote_table.status, "ready");
    assert!(
        remote_table.row_count > 0,
        "the remote-registered patents corpus embeds rows"
    );
    assert_eq!(remote_table.source_id, "patents");

    // The local session sees the same source on the shared engine — its own
    // `infer` resolves `patents`, which would error if the remote registration
    // had not reached the engine.
    let local_rows = local
        .infer(
            "patents",
            &tiny_bert_model_id(),
            jammi_db::ModelTask::TextEmbedding,
            &["abstract".to_string()],
            "id",
        )
        .await
        .expect("local infer resolves the remote-registered source");
    assert!(
        local_rows.iter().map(|b| b.num_rows()).sum::<usize>() > 0,
        "infer over the remote-registered source returns rows"
    );

    // Error parity: re-registering the same id fails inside the engine; both
    // transports reconstruct the identical variant + message.
    let local_err = local
        .add_source("patents", SourceType::File, patents_connection())
        .await
        .expect_err("local re-add of an existing source must fail");
    let remote_err = remote
        .add_source("patents", SourceType::File, patents_connection())
        .await
        .expect_err("remote re-add of an existing source must fail");
    assert_eq!(
        std::mem::discriminant(&local_err),
        std::mem::discriminant(&remote_err),
        "remote reconstructs the same add_source failure variant: {local_err:?} vs {remote_err:?}"
    );
    assert_eq!(
        local_err.to_string(),
        remote_err.to_string(),
        "remote carries the same add_source failure message the engine produced"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// THE error-parity proof. A search against a source that has no ready
/// embedding table fails inside the engine with `JammiError::Catalog`. The
/// remote transport must reconstruct that *exact* variant from the typed wire
/// detail — not just report `invalid_argument`. A heuristic reverse-map from
/// the gRPC code could not distinguish `Catalog` from `Source` / `Config` /
/// `Eval` / `Tenant`, all of which the server maps onto `invalid_argument`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_reconstructs_the_exact_error_variant_local_returns() {
    let server = start_engine_server().await;
    let remote = Session::Remote(remote(&server).await);
    let local = local(&server);

    // Register the corpus but never generate embeddings: a search-by-row-key
    // then fails resolving the (absent) embedding table — `JammiError::Catalog`.
    local
        .add_source("patents", SourceType::File, patents_connection())
        .await
        .expect("add_source");

    let request = || SearchRequest {
        source_id: "patents".to_string(),
        query: SearchQuery::RowKey("any-key".to_string()),
        k: 3,
        filter: None,
        select: Vec::new(),
    };

    let local_err = local
        .search(request())
        .await
        .expect_err("local search must fail");
    let remote_err = remote
        .search(request())
        .await
        .expect_err("remote search must fail");

    // Same variant AND same payload — faithful reconstruction, not a category.
    assert!(
        matches!(local_err, JammiError::Catalog(_)),
        "local search on a source with no embedding table is a Catalog error, got {local_err:?}"
    );
    match (&local_err, &remote_err) {
        (JammiError::Catalog(local_msg), JammiError::Catalog(remote_msg)) => {
            assert_eq!(
                local_msg, remote_msg,
                "the remote transport carries the same Catalog message the engine produced"
            );
        }
        other => panic!("remote did not reconstruct the Catalog variant: {other:?}"),
    }

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Error-parity, second distinct variant on THIS surface. `encode_query` runs
/// model inference: resolving a `local:` model whose directory does not exist
/// fails inside the engine with `JammiError::Model { model_id, message }` — a
/// struct variant distinct from the `Catalog` case above, and one the typed wire
/// detail must reconstruct field-for-field. A heuristic reverse-map from the
/// gRPC code (`invalid_argument`) could not recover `model_id` or distinguish
/// `Model` from `Source` / `Config` / `Tenant`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_reconstructs_a_model_error_from_an_inference_failure() {
    let server = start_engine_server().await;
    let remote = Session::Remote(remote(&server).await);
    let local = local(&server);

    // A syntactically valid `local:` model id pointing at a directory that does
    // not exist. The query input matches the modality, so the request clears
    // wire validation and reaches the engine's model resolver, which fails.
    let missing_model = "local:/nonexistent/jammi-test-model";
    let query = || {
        (
            missing_model.to_string(),
            QueryInput::Text("quantum computing".to_string()),
            Modality::Text,
        )
    };

    let (m, i, md) = query();
    let local_err = local
        .encode_query(&m, i, md)
        .await
        .expect_err("local encode_query on a missing model must fail");
    let (m, i, md) = query();
    let remote_err = remote
        .encode_query(&m, i, md)
        .await
        .expect_err("remote encode_query on a missing model must fail");

    // Same variant AND same fields — the struct variant crosses the wire intact.
    match (&local_err, &remote_err) {
        (
            JammiError::Model {
                model_id: local_id,
                message: local_msg,
            },
            JammiError::Model {
                model_id: remote_id,
                message: remote_msg,
            },
        ) => {
            assert_eq!(
                local_id, remote_id,
                "the remote transport carries the same model_id the engine produced"
            );
            assert_eq!(
                local_msg, remote_msg,
                "the remote transport carries the same model message the engine produced"
            );
        }
        other => panic!("expected a faithful Model error from both transports, got {other:?}"),
    }

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Tenant trio over the wire: a `bind_tenant` (now async) followed by a
/// `tenant()` read observes the bound tenant; the binding is keyed by the
/// client's own session id, so a second client sees nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_binds_and_reads_tenant_over_the_wire() {
    let server = start_engine_server().await;
    let bound = Session::Remote(remote(&server).await);
    let other = Session::Remote(remote(&server).await);

    assert_eq!(bound.tenant().await.expect("get tenant"), None);

    bound.bind_tenant(tenant_a()).await.expect("bind tenant");
    assert_eq!(
        bound.tenant().await.expect("get tenant"),
        Some(tenant_a()),
        "the bound client observes its tenant"
    );
    assert_eq!(
        other.tenant().await.expect("get tenant"),
        None,
        "a different session id sees no binding"
    );

    bound.unbind_tenant().await.expect("unbind tenant");
    assert_eq!(
        bound.tenant().await.expect("get tenant"),
        None,
        "after unbind the tenant is cleared"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Topics over the wire: a `RemoteSession` registers a topic, publishes a
/// batch, lists it, and drops it — and a `LocalSession` over the same engine
/// observes every effect identically. Proves the register → publish → list →
/// drop lifecycle is faithful across transports (the publish offset and the
/// listed `TopicDefinition` match what the in-process path produces).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_round_trips_the_topic_lifecycle_like_local() {
    let server = start_engine_server_with_trigger().await;
    let remote = Session::Remote(remote(&server).await);
    let local = local(&server);
    let topic = events_topic();

    // Register over the wire; the local session (same engine) must see it.
    remote
        .register_topic(&topic)
        .await
        .expect("remote register");
    let listed_local = local.list_topics().await.expect("local list");
    assert!(
        listed_local
            .iter()
            .any(|t| t.name == topic.name && t.id == topic.id),
        "the topic the remote registered (with its client-minted id) is visible locally"
    );

    // The remote `list_topics` reconstructs the SAME full definitions the local
    // one returns — id, name, schema, tenant — not just names.
    let listed_remote = remote.list_topics().await.expect("remote list");
    let remote_ours = listed_remote
        .iter()
        .find(|t| t.id == topic.id)
        .expect("remote list includes our topic");
    assert_eq!(remote_ours.name, topic.name);
    assert_eq!(remote_ours.tenant, topic.tenant);
    assert_eq!(
        remote_ours.schema.as_ref(),
        topic.schema.as_ref(),
        "remote list reconstructs the topic's payload schema"
    );

    // Publish over the wire; the offset is the engine-assigned one.
    let offset = remote
        .publish(&topic, events_batch())
        .await
        .expect("remote publish");
    assert_eq!(
        offset.value(),
        0,
        "first publish to a fresh topic is offset 0"
    );

    // Drop over the wire by the topic id; the local session no longer sees it.
    remote.drop_topic(topic.id).await.expect("remote drop");
    let after = local.list_topics().await.expect("local list after drop");
    assert!(
        !after.iter().any(|t| t.id == topic.id),
        "the dropped topic is gone on both transports"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Subscribe streaming over the wire. A `RemoteSession` opens a server-streaming
/// subscription from offset 0; after a publish, the stream yields the SAME
/// `DeliveredBatch` a `LocalSession`'s subscription yields against the same
/// engine — same offset, same rows. Proves the streaming verb (not just the
/// unary ones) is faithful end to end.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn remote_subscribe_stream_yields_the_same_batches_as_local() {
    let server = start_engine_server_with_trigger().await;
    let remote = Session::Remote(remote(&server).await);
    let local = local(&server);
    let topic = events_topic();

    local.register_topic(&topic).await.expect("register topic");
    // `Session::register_topic` registers the catalog row + backing table; the
    // broker's live-tail channel is provisioned separately (the CREATE TOPIC DDL
    // path does both). Register it on the broker so the subscribe live tail
    // attaches — the same engine both sessions share.
    server
        .engine
        .trigger_broker()
        .register_topic(&topic)
        .await
        .expect("broker register");

    // Publish one batch first (it lands at offset 0 in the backing table), then
    // open both subscriptions from offset 0. The deterministic backing-table
    // *replay* (not the racy live broadcast) delivers that batch to each stream,
    // so the parity assertion does not depend on subscribe/publish interleaving.
    local
        .publish(&topic, events_batch())
        .await
        .expect("publish");

    let mut remote_stream = remote
        .subscribe(&topic, Predicate::match_all(), Some(offset_zero()))
        .await
        .expect("remote subscribe");
    let mut local_stream = local
        .subscribe(&topic, Predicate::match_all(), Some(offset_zero()))
        .await
        .expect("local subscribe");

    let remote_delivered = next_delivered(&mut remote_stream).await;
    let local_delivered = next_delivered(&mut local_stream).await;

    assert_eq!(
        remote_delivered.offset.value(),
        local_delivered.offset.value(),
        "both transports deliver the batch at the same offset"
    );
    assert_eq!(
        remote_delivered.batch, local_delivered.batch,
        "the remote stream rebuilds the identical record batch the local stream yields"
    );
    assert_eq!(
        remote_delivered.batch,
        events_batch(),
        "and that batch is exactly what was published"
    );

    // Drop the streams before signalling shutdown: an open server-streaming
    // subscription keeps its connection (and the server task) alive, so
    // awaiting `server.handle` with a live stream would hang the test.
    drop(remote_stream);
    drop(local_stream);
    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Trigger error parity. A subscribe predicate that references a column the
/// topic schema does not have fails inside the engine's predicate parser with
/// `TriggerError::PredicateParse`. The remote transport must reconstruct that
/// EXACT variant from the typed trigger detail — not merely report
/// `invalid_argument`, which `PredicateUnsupported` and `BatchSchemaMismatch`
/// also map onto. A heuristic reverse-map from the gRPC code could not tell
/// `PredicateParse` from those siblings.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_reconstructs_the_exact_trigger_error_variant_local_returns() {
    let server = start_engine_server_with_trigger().await;
    let remote = Session::Remote(remote(&server).await);
    let local = local(&server);
    let topic = events_topic();
    local.register_topic(&topic).await.expect("register topic");

    // Parsing a predicate that names a column absent from the topic schema fails
    // with `PredicateParse`. This is the variant the engine produces in-process.
    let unknown_col = "no_such_column = 1";
    let local_err = match Predicate::from_sql(
        server.engine.context(),
        Arc::clone(&topic.schema),
        unknown_col,
    ) {
        Ok(_) => panic!("unknown-column predicate must fail to parse against the topic schema"),
        Err(e) => e,
    };
    assert!(
        matches!(local_err, TriggerError::PredicateParse(_)),
        "local predicate parse on an unknown column is PredicateParse, got {local_err:?}"
    );

    // Build a well-formed predicate that carries the same SQL (parsed against a
    // schema that *does* declare the column) and subscribe remotely: the SQL
    // crosses the wire, the server re-parses it against the REAL topic schema,
    // and fails with the same `PredicateParse`. The remote arm reconstructs the
    // exact variant from the attached detail.
    let remote_err = match remote
        .subscribe(
            &topic,
            predicate_referencing_unknown_column(unknown_col),
            None,
        )
        .await
    {
        Ok(_) => panic!("remote subscribe with an unknown-column predicate must fail"),
        Err(e) => e,
    };
    match (&local_err, &remote_err) {
        (TriggerError::PredicateParse(_), TriggerError::PredicateParse(_)) => {}
        other => panic!(
            "remote did not reconstruct the PredicateParse variant the engine produces: {other:?}"
        ),
    }

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Audit over the wire: a `RemoteSession` (tenant-bound) logs a record, fetches
/// it by id and via recent, and the fetched record's engine-computed signature
/// verifies — proving every field crossed the wire byte-for-byte. The fetched
/// record matches the one a `LocalSession` fetch returns for the same query id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_round_trips_audit_log_and_fetch_like_local() {
    let _g = audit_env_lock().lock().await;
    std::env::set_var(MASTER_KEY_ENV, AUDIT_TEST_KEY);

    let server = start_engine_server_with_trigger().await;
    let remote = Session::Remote(remote(&server).await);
    let local = local(&server);

    // The audit primitive requires a bound tenant. Bind the remote session over
    // the wire; the LocalSession reads back through the same engine, so scope it
    // identically via the engine's sticky binding for the fetch comparison.
    remote.bind_tenant(tenant_a()).await.expect("remote bind");

    let query_id = uuid::Uuid::now_v7();
    let record = PerQueryAudit::new(
        query_id,
        "openai/clip-vit-base",
        "rev-3",
        serde_json::json!({ "examiner_id": "42" }),
        vec!["doc-1".to_string(), "doc-2".to_string()],
        vec![0.91, 0.84],
    )
    .expect("record");

    remote
        .audit_log(vec![record])
        .await
        .expect("remote audit_log");

    // Fetch by id over the wire; the record is fully reconstructed (tenant,
    // signature, executed_at) so its signature verifies.
    let fetched = remote
        .audit_fetch_by_query_id(query_id)
        .await
        .expect("remote fetch")
        .expect("record present");
    assert_eq!(fetched.query_id, query_id);
    assert_eq!(fetched.tenant_id.as_deref(), Some(TENANT_A));
    assert!(!fetched.signature.is_empty(), "signature crossed the wire");
    verify_with_store(&fetched, &EnvSigningKeyStore)
        .expect("remote-fetched record signature verifies");

    // The local fetch (same engine, same tenant scope) returns the identical
    // record — the remote read is byte-for-byte the local read.
    let local_fetched = server
        .engine
        .with_tenant_scoped(tenant_a(), |_| local.audit_fetch_by_query_id(query_id))
        .await
        .expect("local fetch")
        .expect("record present locally");
    assert_eq!(
        fetched, local_fetched,
        "the remote-fetched audit record equals the local-fetched one field for field"
    );

    // fetch_recent over the wire surfaces the same record.
    let recent = remote
        .audit_fetch_recent(10)
        .await
        .expect("remote fetch_recent");
    assert!(
        recent.iter().any(|r| r.query_id == query_id),
        "fetch_recent includes the logged record"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Audit error parity. An audit log on an UNSCOPED session fails inside the
/// engine with `AuditError::NoTenantBinding`. The remote transport must
/// reconstruct that EXACT variant from the typed audit detail — not merely
/// report `failed_precondition`, which `MasterKey` shares.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_reconstructs_the_exact_audit_error_variant_local_returns() {
    let _g = audit_env_lock().lock().await;
    std::env::set_var(MASTER_KEY_ENV, AUDIT_TEST_KEY);

    let server = start_engine_server_with_trigger().await;
    let remote = Session::Remote(remote(&server).await);
    let local = local(&server);

    let record = || {
        PerQueryAudit::new(
            uuid::Uuid::now_v7(),
            "m",
            "v",
            serde_json::json!({}),
            vec![],
            vec![],
        )
        .expect("record")
    };

    // Local: an unscoped audit_log is NoTenantBinding.
    let local_err = local
        .audit_log(vec![record()])
        .await
        .expect_err("local unscoped audit_log must fail");
    assert!(
        matches!(local_err, AuditError::NoTenantBinding),
        "local unscoped audit_log is NoTenantBinding, got {local_err:?}"
    );

    // Remote: the unscoped session (no bind_tenant) hits the same engine path;
    // the typed detail reconstructs the IDENTICAL variant, not a category guess.
    let remote_err = remote
        .audit_log(vec![record()])
        .await
        .expect_err("remote unscoped audit_log must fail");
    match (&local_err, &remote_err) {
        (AuditError::NoTenantBinding, AuditError::NoTenantBinding) => {}
        other => panic!(
            "remote did not reconstruct the NoTenantBinding variant the engine produces: {other:?}"
        ),
    }

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// The SDK front door selects the transport. `Jammi::open(Target::Remote(_))`
/// connects to the in-process server and yields a `Session::Remote`;
/// `Jammi::open(Target::Local(_))` builds an embedded engine and yields a
/// `Session::Local`. Both are the same `Session` type, and a verb run through
/// either returns the same shape — so the two transports are interchangeable
/// through the `Session` the factory returns. The remote arm is compared to a
/// `LocalSession` over the *same* engine the server drives (so any divergence is
/// the transport's, not the engine's); the local-front-door arm proves
/// `Target::Local` independently opens a live embedded session that runs the
/// same verb.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn front_door_opens_interchangeable_local_and_remote_sessions() {
    let server = start_engine_server().await;
    let model_id = tiny_bert_model_id();
    let query = "quantum computing applications";

    // Front door, remote arm: Target::Remote → Session::Remote against the server.
    let endpoint = Endpoint::from_shared(format!("http://{}", server.addr)).expect("endpoint");
    let remote = Jammi::open(Target::Remote(endpoint))
        .await
        .expect("open remote session");
    assert!(
        matches!(remote, Session::Remote(_)),
        "Target::Remote opens a Session::Remote"
    );

    // A LocalSession over the SAME engine the server drives — the parity peer.
    let local_same_engine = local(&server);

    let remote_vec = remote
        .encode_query(
            &model_id,
            QueryInput::Text(query.to_string()),
            Modality::Text,
        )
        .await
        .expect("remote encode_query");
    let local_vec = local_same_engine
        .encode_query(
            &model_id,
            QueryInput::Text(query.to_string()),
            Modality::Text,
        )
        .await
        .expect("local encode_query");
    assert_eq!(
        remote_vec, local_vec,
        "the factory's remote Session and a local Session over the same engine \
         encode the same query identically — interchangeable through Session"
    );

    // Front door, local arm: Target::Local → Session::Local over a fresh
    // embedded engine, proving the same verb runs end to end through the
    // factory-opened embedded session (its own engine, so dimensionality — not
    // the exact vector — is the cross-engine-stable invariant).
    let dir = tempfile::tempdir().expect("tempdir");
    let embedded = Jammi::open(Target::Local(test_config(dir.path())))
        .await
        .expect("open local session");
    assert!(
        matches!(embedded, Session::Local(_)),
        "Target::Local opens a Session::Local"
    );
    let embedded_vec = embedded
        .encode_query(
            &model_id,
            QueryInput::Text(query.to_string()),
            Modality::Text,
        )
        .await
        .expect("embedded encode_query");
    assert_eq!(
        embedded_vec.len(),
        remote_vec.len(),
        "a verb through the factory-opened embedded Session yields the same \
         vector shape the remote Session does — same surface, either transport"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Offset 0 — the replay-from-earliest starting point for the subscribe tests.
fn offset_zero() -> jammi_db::trigger::Offset {
    jammi_db::trigger::Offset::new(0, chrono::Utc::now())
}

/// Pull the next delivered batch off a subscription stream, failing the test if
/// the stream ends, errors, or stalls. The timeout keeps a delivery miss a fast,
/// legible failure rather than a hang (a tailing subscription never ends on its
/// own, so an unbounded `next().await` would block forever on a regression).
async fn next_delivered<S>(stream: &mut S) -> DeliveredBatch
where
    S: futures::Stream<Item = Result<DeliveredBatch, TriggerError>> + Unpin,
{
    let next = tokio::time::timeout(std::time::Duration::from_secs(10), stream.next()).await;
    match next {
        Ok(Some(Ok(delivered))) => delivered,
        Ok(Some(Err(e))) => panic!("subscription yielded an error instead of a batch: {e:?}"),
        Ok(None) => panic!("subscription ended before delivering a batch"),
        Err(_) => panic!("subscription did not deliver a batch within the timeout"),
    }
}

/// Build a well-formed `Predicate` carrying `sql` as its source, parsed against
/// a schema that *declares* the referenced column — so the predicate is valid
/// locally and carries its SQL across the wire. The server re-parses that SQL
/// against the real topic schema (which lacks the column) and surfaces the
/// `PredicateParse` error there, exercising the server-side error-parity path.
fn predicate_referencing_unknown_column(sql: &str) -> Predicate {
    let permissive = Arc::new(Schema::new(vec![Field::new(
        "no_such_column",
        DataType::Int64,
        true,
    )]));
    let ctx = datafusion::execution::context::SessionContext::new();
    Predicate::from_sql(&ctx, permissive, sql)
        .expect("predicate parses against the permissive schema")
}

/// Serialize the audit tests that mutate the process-global
/// `JAMMI_AUDIT_MASTER_KEY`. Held across `.await`, so an async mutex.
fn audit_env_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Extract the (key, score) of each hit from a search result, the
/// client-observable identity of a hit, for cross-transport comparison.
fn keys_and_scores(batches: Vec<arrow::record_batch::RecordBatch>) -> Vec<(String, f32)> {
    use arrow::array::{Float32Array, StringArray};
    let mut out = Vec::new();
    for batch in &batches {
        let keys = batch
            .column_by_name("_row_id")
            .expect("_row_id column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_row_id is Utf8");
        let scores = batch
            .column_by_name("similarity")
            .expect("similarity column")
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("similarity is Float32");
        for row in 0..batch.num_rows() {
            out.push((keys.value(row).to_string(), scores.value(row)));
        }
    }
    out
}
