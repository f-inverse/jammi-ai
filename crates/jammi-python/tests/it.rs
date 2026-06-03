//! Integration test for the Rust-facing surface of `jammi-python`.
//!
//! `PyDatabase::session_arc` is what lets a downstream Rust crate
//! (e.g. a downstream Python-bindings layer) share the OSS database's
//! `Arc<InferenceSession>` — and therefore its schema-upgrade lock,
//! trigger broker, catalog cache, and tenant binding — instead of opening
//! a parallel session against the same artifact directory. Without that
//! sharing, two sessions race on schema migrations and observe one
//! another's tenant binding inconsistently.
//!
//! This test asserts both halves of the contract:
//!   1. The returned `Arc` aliases the database's session (proven by the
//!      strong count growing on every call).
//!   2. State mutated through one alias is visible through the other
//!      (proven by binding a tenant via the freshly-cloned `Arc` and
//!      reading it back through a second clone).

use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;

use _native::model_task::{ModelTaskArg, PyModelTask};
use _native::{connect_remote, PyDatabase};
use jammi_ai::local_session::{LocalSession, Modality, QueryInput, Session};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::config::JammiConfig;
use jammi_db::TenantId;
use jammi_server::grpc::session::SessionStore;
use jammi_server::runtime::serve_grpc_chain;
use jammi_test_utils::{cookbook_fixture, fixture};
use pyo3::prelude::*;
use pyo3::types::PyString;
use tempfile::tempdir;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

fn test_config(artifact_dir: &std::path::Path) -> JammiConfig {
    JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        gpu: jammi_db::config::GpuConfig {
            device: -1,
            ..Default::default()
        },
        inference: jammi_db::config::InferenceConfig {
            batch_size: 8,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn session_arc_shares_session_state_with_pydatabase() {
    let dir = tempdir().expect("tempdir");
    let db = PyDatabase::open(test_config(dir.path())).expect("open PyDatabase");

    // Baseline strong count for the session inside the database. Every
    // `session_arc()` call must increment it — that is what proves the
    // returned `Arc` aliases the database's session rather than a freshly
    // constructed parallel one.
    let first = db.session_arc();
    let count_after_first = Arc::strong_count(&first);

    let second = db.session_arc();
    let count_after_second = Arc::strong_count(&second);

    assert_eq!(
        count_after_second,
        count_after_first + 1,
        "session_arc() must clone the same Arc — strong count should grow \
         by exactly one per call (saw {count_after_first} then {count_after_second})",
    );
    assert!(
        Arc::ptr_eq(&first, &second),
        "both clones must point at the same InferenceSession allocation",
    );

    // Cross-handle state visibility: bind a tenant through `first`, read
    // it back through `second`. A parallel session would not observe the
    // write.
    let tenant =
        TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").expect("valid tenant uuid");
    first.bind_tenant(tenant);
    assert_eq!(
        second.tenant(),
        Some(tenant),
        "binding through one Arc must be visible through the other",
    );

    // And visible through the database's own Python-facing surface.
    let third = db.session_arc();
    assert_eq!(
        third.tenant(),
        Some(tenant),
        "binding must also be visible to subsequently-issued Arcs",
    );

    first.unbind_tenant();
    assert_eq!(
        second.tenant(),
        None,
        "unbinding through one Arc must clear the shared state",
    );
}

/// Both ways a Python caller can supply `task=` to a binding — the
/// `ModelTask` pyclass enum and the snake-case string — must extract to
/// the same `ModelTask` variant. Catches a regression in the string-vs-
/// pyclass branch of `ModelTaskArg::extract` that would let one shape
/// produce a different variant than the other.
#[test]
fn model_task_arg_accepts_pyclass_and_string_identically() {
    Python::initialize();

    Python::attach(|py| {
        // Path 1: typed PyClass instance — constructed via `Py::new`,
        // which is the same path the `#[pymodule] _native` registration
        // uses; sidesteps the `IntoPyObject` recursion that the
        // generated derive hits when called from inside `Python::attach`
        // on a freshly-loaded pyo3 0.28 type table.
        let typed_obj = Py::new(py, PyModelTask::TextEmbedding)
            .expect("Py::new PyModelTask")
            .into_any()
            .into_bound(py);
        let typed_arg: ModelTaskArg = typed_obj.as_borrowed().extract().expect("extract pyclass");
        assert_eq!(typed_arg.0, ModelTask::TextEmbedding);

        // Path 2: snake-case string.
        let str_obj = PyString::new(py, "text_embedding").into_any();
        let str_arg: ModelTaskArg = str_obj.as_borrowed().extract().expect("extract string");
        assert_eq!(str_arg.0, ModelTask::TextEmbedding);

        // Both shapes converge on the same variant.
        assert_eq!(typed_arg.0, str_arg.0);

        // Round-trip every variant via string so the test exercises the
        // full table.
        for variant in [
            ModelTask::TextEmbedding,
            ModelTask::ImageEmbedding,
            ModelTask::Classification,
            ModelTask::Ner,
        ] {
            let s = PyString::new(py, variant.as_db_str()).into_any();
            let arg: ModelTaskArg = s.as_borrowed().extract().expect("extract variant");
            assert_eq!(arg.0, variant);
        }

        // Unknown string surfaces a typed PyErr (not a panic).
        let bad = PyString::new(py, "not_a_task").into_any();
        let err: PyResult<ModelTaskArg> = bad.as_borrowed().extract();
        assert!(err.is_err(), "unknown task string must surface a PyErr");
    });
}

/// An in-process engine-backed gRPC server held alive for the duration of a
/// test. Dropping `_shutdown` tears it down; `_dir` roots the engine's temp
/// artifact dir; `_rt` keeps the server's runtime (and the spawned serve task)
/// running on a dedicated thread so the synchronous `RemoteDatabase` — which
/// builds and `block_on`s its *own* runtime, exactly as a Python caller does —
/// can drive it without nesting runtimes.
struct RemoteFixture {
    addr: SocketAddr,
    engine: Arc<InferenceSession>,
    _rt: tokio::runtime::Runtime,
    _shutdown: oneshot::Sender<()>,
    _dir: tempfile::TempDir,
}

/// Stand up the engine-backed gRPC chain on a dedicated runtime thread and
/// return its address plus the shared engine `Arc`. Mirrors the
/// `jammi-server` `start_engine_server` fixture, inlined here because that
/// crate's `tests/it/common` module is test-private.
fn start_remote_fixture() -> RemoteFixture {
    let dir = tempdir().expect("tempdir");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("server runtime");

    let cfg = test_config(dir.path());
    let session =
        rt.block_on(async { Arc::new(InferenceSession::new(cfg).await.expect("session")) });

    let listener = rt.block_on(async { TcpListener::bind("127.0.0.1:0").await.expect("bind") });
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);

    let store = SessionStore::new();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let flight_ctx = session.context().clone();
    let binding = session.tenant_binding_arc();
    let engine = Arc::clone(&session);
    rt.spawn(async move {
        serve_grpc_chain(
            addr,
            flight_ctx,
            binding,
            store,
            None,
            Some(session),
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await
        .expect("grpc server");
    });
    // Give the server a moment to bind before the client connects.
    rt.block_on(async { tokio::time::sleep(std::time::Duration::from_millis(100)).await });

    RemoteFixture {
        addr,
        engine,
        _rt: rt,
        _shutdown: shutdown_tx,
        _dir: dir,
    }
}

/// The remote Python binding (`connect_remote(endpoint=...)` →
/// `RemoteDatabase`) reuses the one Rust `RemoteSession` over the wire: a query
/// encoded through the remote Python session must equal the byte-identical
/// vector a `LocalSession` produces against the *same* engine. This is the
/// Shape-C parity proof — same engine, two transports, identical result — and
/// it is fully hermetic: the encoder is the bundled `tiny_bert` cookbook
/// fixture and the corpus is the bundled `patents.parquet`, no live network.
#[test]
fn remote_database_encode_query_matches_local_over_the_wire() {
    let srv = start_remote_fixture();
    let model_id = format!("local:{}", cookbook_fixture("tiny_bert").display());
    let query = "quantum computing applications";

    // Remote path: the synchronous Python-facing binding. It opens its own
    // runtime internally (as it would under a Python interpreter) and drives
    // the wire via `Session::Remote` → the Rust `RemoteSession` gRPC client.
    let remote_db = connect_remote(&format!("http://{}", srv.addr)).expect("connect_remote");
    let remote_vec = remote_db
        .encode_query_for_test(&model_id, query, Modality::Text)
        .expect("remote encode_query");

    // Local baseline: a `LocalSession` over the *same* engine `Arc` the server
    // drives, so any divergence is the transport's fault, not the engine's.
    let local = Session::Local(LocalSession::new(Arc::clone(&srv.engine)));
    let local_rt = tokio::runtime::Runtime::new().expect("local runtime");
    let local_vec = local_rt
        .block_on(local.encode_query(
            &model_id,
            QueryInput::Text(query.to_string()),
            Modality::Text,
        ))
        .expect("local encode_query");

    assert!(!remote_vec.is_empty(), "encode_query returns a vector");
    assert_eq!(
        remote_vec.len(),
        local_vec.len(),
        "remote/local query vectors share dimensionality",
    );
    assert_eq!(
        remote_vec, local_vec,
        "the same query through either transport encodes to the same vector",
    );
}

/// `add_source` + `generate_embeddings` + `search` entirely through the remote
/// Python binding return a non-empty hydrated result against a corpus
/// registered and embedded over the wire — the source verb and the compute
/// verbs are all reachable, not stubbed. `add_source` maps to
/// `EmbeddingService.AddSource` (the typed RPC), so the whole pipeline crosses
/// the wire through `RemoteDatabase`.
#[test]
fn remote_database_add_source_generate_embeddings_and_search() {
    let srv = start_remote_fixture();
    let model_id = format!("local:{}", cookbook_fixture("tiny_bert").display());

    let remote_db = connect_remote(&format!("http://{}", srv.addr)).expect("connect_remote");

    // Register the corpus over the wire through the remote binding's `add_source`
    // (no engine-side setup): the registration must reach the shared engine for
    // the embed/search below to resolve `patents`.
    remote_db
        .add_source_for_test(
            "patents",
            &fixture("patents.parquet").display().to_string(),
            "parquet",
        )
        .expect("remote add_source");

    let table = remote_db
        .generate_embeddings_for_test("patents", &model_id, &["abstract"], "id", Modality::Text)
        .expect("remote generate_embeddings");
    assert!(!table.is_empty(), "result table name is returned");

    let query = remote_db
        .encode_query_for_test(&model_id, "quantum computing", Modality::Text)
        .expect("remote encode_query");
    let hits = remote_db
        .search_for_test("patents", query, 5)
        .expect("remote search");
    assert!(hits > 0, "search over the embedded corpus returns hits");
}
