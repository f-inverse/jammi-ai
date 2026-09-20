//! Shared test helpers for jammi-db and jammi-ai integration tests.

pub mod child;
pub mod source_universe;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use jammi_db::catalog::backend::{BackendImpl, BackendKind};
use jammi_db::catalog::backend_postgres::PostgresBackend;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::session::JammiSession;

// The one null-hash embedding-batch builder every hand-built fixture routes
// through (the fifth `_content_hash` column is NULL on every table no
// embedding pipeline produced).
pub use jammi_db::store::schema::embedding_batch_with_null_hash;

/// The URL of the live Postgres the Postgres-backed tests run against, read
/// from `JAMMI_TEST_PG_URL`.
///
/// A test that calls this (directly or through a Postgres [`BackendKind`]) is
/// compiled only under its crate's `live-postgres-tests` feature; with that
/// feature on, the variable must be set.
///
/// # Panics
/// When `JAMMI_TEST_PG_URL` is unset or empty.
pub fn postgres_url() -> String {
    jammi_test_resources::env("JAMMI_TEST_PG_URL")
}

/// Build a [`JammiSession`] backed by `kind` for parameterized integration
/// tests. The caller passes an artifact dir (used by SQLite for the catalog
/// file and by both backends for result-table parquet); the Postgres variant
/// connects to [`postgres_url`] and runs migrations.
///
/// # Panics
/// When the session cannot be opened, or `kind` is Postgres and
/// `JAMMI_TEST_PG_URL` is unset.
pub async fn make_test_session(kind: BackendKind, artifact_dir: &Path) -> JammiSession {
    let config = test_config(artifact_dir);
    match kind {
        BackendKind::Sqlite => JammiSession::new(config)
            .await
            .expect("sqlite-backed session"),
        BackendKind::Postgres => {
            JammiSession::with_backend(config, open_backend(kind, artifact_dir).await)
                .await
                .expect("postgres-backed session")
        }
    }
}

/// A catalog backend of `kind`, not yet migrated: the SQLite file
/// `<dir>/catalog.db`, or a pool on the live Postgres at [`postgres_url`]
/// (which ignores `dir`).
///
/// # Panics
/// When the backend cannot be opened, or `kind` is Postgres and
/// `JAMMI_TEST_PG_URL` is unset.
pub async fn open_backend(kind: BackendKind, dir: &Path) -> BackendImpl {
    match kind {
        BackendKind::Sqlite => BackendImpl::Sqlite(
            SqliteBackend::open(&dir.join("catalog.db"))
                .await
                .expect("open sqlite backend"),
        ),
        BackendKind::Postgres => BackendImpl::Postgres(
            PostgresBackend::open_with_options(&postgres_url(), 8, None)
                .await
                .expect("open postgres backend"),
        ),
    }
}

/// The shared backends a multi-process distributed test runs against: one
/// Postgres catalog, one S3-compatible object store, and the credentials the
/// spawned server processes authenticate to it with.
///
/// A test that builds this is compiled only under its crate's
/// `live-distributed-tests` feature; with that feature on, every variable
/// [`DistributedBackends::from_env`] reads must be set.
pub struct DistributedBackends {
    /// Shared Postgres catalog URL (`JAMMI_TEST_PG_URL`).
    pub pg_url: String,
    /// S3 endpoint (`JAMMI_TEST_S3_ENDPOINT`, e.g. `http://127.0.0.1:9000`).
    pub s3_endpoint: String,
    /// Pre-created bucket every run roots its artifacts under (`JAMMI_TEST_S3_BUCKET`).
    pub s3_bucket: String,
    /// S3 access key (`AWS_ACCESS_KEY_ID`).
    pub access_key_id: String,
    /// S3 secret key (`AWS_SECRET_ACCESS_KEY`).
    pub secret_access_key: String,
    /// Region (`AWS_REGION`); `us-east-1` when unset, which an S3-compatible
    /// store without regions accepts.
    pub region: String,
}

impl DistributedBackends {
    /// Read the backends from the environment.
    ///
    /// # Panics
    /// Naming the first of `JAMMI_TEST_PG_URL`, `JAMMI_TEST_S3_ENDPOINT`,
    /// `JAMMI_TEST_S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
    /// that is unset or empty.
    pub fn from_env() -> Self {
        Self {
            pg_url: postgres_url(),
            s3_endpoint: jammi_test_resources::env("JAMMI_TEST_S3_ENDPOINT"),
            s3_bucket: jammi_test_resources::env("JAMMI_TEST_S3_BUCKET"),
            access_key_id: jammi_test_resources::env("AWS_ACCESS_KEY_ID"),
            secret_access_key: jammi_test_resources::env("AWS_SECRET_ACCESS_KEY"),
            region: std::env::var("AWS_REGION")
                .ok()
                .filter(|region| !region.is_empty())
                .unwrap_or_else(|| "us-east-1".to_string()),
        }
    }

    /// A fresh `s3://bucket/dist-<test>-<suffix>` root, so concurrent runs (or a
    /// re-run) never collide on the shared bucket. The prefix carries the test
    /// name so a failed run's objects can be traced to it.
    pub fn unique_result_root(&self, test: &str) -> String {
        format!("s3://{}/dist-{test}-{}", self.s3_bucket, unique_suffix())
    }

    /// The S3 [`jammi_db::storage::CloudConfig`] every object-store driver is
    /// given: the endpoint, the credentials, the region, and `allow_http` when
    /// the endpoint speaks plain HTTP. The spawned processes receive the same
    /// credentials through their `AWS_*` environment, so a harness reads
    /// exactly what they wrote.
    pub fn cloud(&self) -> jammi_db::storage::CloudConfig {
        jammi_db::storage::CloudConfig::S3(jammi_db::storage::S3Config {
            region: Some(self.region.clone()),
            endpoint: Some(self.s3_endpoint.clone()),
            access_key_id: Some(self.access_key_id.clone()),
            secret_access_key: Some(self.secret_access_key.clone().into()),
            session_token: None,
            allow_http: self.allows_http(),
        })
    }

    /// Whether the S3 endpoint is plain HTTP (a local S3-compatible store).
    pub fn allows_http(&self) -> bool {
        self.s3_endpoint.starts_with("http://")
    }
}

/// Backend-unique id suffix for any tenant id / source id / table name /
/// channel name a parameterized integration test creates.
///
/// The Postgres lane runs the whole matrix against ONE shared database (see
/// `make_test_session`), so two tests — or the two `BackendKind` arms of the
/// same test — must never write under the same catalog identifier: a fixed
/// literal collides with a sibling test's rows (or a prior run's), producing
/// cumulative-row-count and "already exists" failures that are a harness bug,
/// not a product regression. SQLite tests get a fresh on-disk catalog per
/// `tempdir()` and don't strictly need this, but calling it unconditionally
/// keeps one code path for both backends instead of a Postgres-only special
/// case.
pub fn unique_suffix() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let epoch_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{epoch_ns:x}_{n:x}")
}

/// An unused TCP port on localhost for a listener a SPAWNED process binds
/// later. Never `bind(:0)`-then-release, and never a fixed number above the
/// floor: both hand out a port from the kernel's ephemeral range, the same
/// range every outgoing `connect()` this test process makes (Postgres,
/// MinIO) draws its local port from, so the port can be taken by a client
/// socket before the child binds it ("failed to bind OSS server listeners:
/// Address already in use"). Ports come from a range BELOW every platform's
/// ephemeral floor (Linux 32768, macOS 49152), verified bindable at pick
/// time, and never handed out twice by this process.
pub fn free_port() -> u16 {
    use std::collections::HashSet;
    use std::hash::{BuildHasher, Hasher};
    use std::net::TcpListener;
    use std::sync::Mutex;
    static HANDED_OUT: Mutex<Option<HashSet<u16>>> = Mutex::new(None);
    const LO: u32 = 20_000;
    const SPAN: u32 = 12_000;
    let mut guard = HANDED_OUT.lock().expect("port ledger lock poisoned");
    let handed = guard.get_or_insert_with(HashSet::new);
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    let mut cursor = (h.finish() % u64::from(SPAN)) as u32;
    for _ in 0..SPAN {
        let port = (LO + cursor) as u16;
        cursor = (cursor + 1) % SPAN;
        if handed.contains(&port) {
            continue;
        }
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            handed.insert(port);
            return port;
        }
    }
    panic!(
        "no bindable port in {LO}..{} for the lane's fleet",
        LO + SPAN
    );
}

/// Workspace root — two levels up from any crate in `crates/<name>/`.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Root of the test fixtures directory (at workspace root). Houses the
/// generic test-only fixtures (`patents.parquet`, `assignees.csv`,
/// `golden_relevance.csv`, the tiny encoder fixtures that are not part of
/// the public cookbook surface, etc.).
pub fn fixtures_dir() -> PathBuf {
    workspace_root().join("tests").join("fixtures")
}

/// Root of the cookbook fixtures directory (at workspace root). Houses
/// the fixtures consumed by the OSS cookbook recipes — currently
/// `tiny_bert/`, `tiny_modernbert_classifier/`, and the synthetic data
/// files (`tiny_corpus.parquet`, `tiny_golden.json`, `tiny_labels.csv`,
/// `tiny_pairs.csv`). Integration tests that exercise the same model
/// fixtures the cookbook ships read from here so the recipe and the test
/// share one source of truth.
pub fn cookbook_fixtures_dir() -> PathBuf {
    workspace_root().join("cookbook").join("fixtures")
}

/// Path to a specific fixture file under `tests/fixtures/`.
pub fn fixture(name: &str) -> PathBuf {
    fixtures_dir().join(name)
}

/// Path to a specific fixture file under `cookbook/fixtures/`.
pub fn cookbook_fixture(name: &str) -> PathBuf {
    cookbook_fixtures_dir().join(name)
}

/// A text the `tiny_bert` fixture model can tell apart from every other
/// `(role, index)`: `role`, then `index` in base 36, each character its own
/// whitespace-separated word.
///
/// That model's vocabulary is 256 WordPiece tokens — the single characters
/// `0-9a-z`, punctuation and a few suffixes — so an ordinary English fixture
/// (`"anchor text 3"`, `"graph_node_text_3"`) tokenizes almost entirely to
/// `[UNK]`: rows that differ on the page are one row to the model, and a test
/// over them cannot see a wrong order, a wrong shard, or an anchor that equals
/// its positive. Every word this emits is a single in-vocabulary token, so
/// distinct arguments give distinct token sequences, for any number of rows.
///
/// # Panics
/// If `role` is not an ASCII lowercase letter or digit.
pub fn tiny_vocab_text(role: char, index: usize) -> String {
    assert!(
        role.is_ascii_digit() || role.is_ascii_lowercase(),
        "tiny_vocab_text: role {role:?} is not a single in-vocabulary character (0-9, a-z)"
    );
    let mut digits = Vec::new();
    let mut rest = index;
    loop {
        digits.push(char::from_digit((rest % 36) as u32, 36).expect("a base-36 digit"));
        rest /= 36;
        if rest == 0 {
            break;
        }
    }
    std::iter::once(role)
        .chain(digits.into_iter().rev())
        .map(String::from)
        .collect::<Vec<_>>()
        .join(" ")
}

/// URL for a `tests/fixtures/` fixture suitable for DataFusion's ListingTable.
pub fn fixture_url(name: &str) -> String {
    format!("file://{}", fixture(name).display())
}

/// URL for a `cookbook/fixtures/` fixture suitable for DataFusion's ListingTable.
pub fn cookbook_fixture_url(name: &str) -> String {
    format!("file://{}", cookbook_fixture(name).display())
}

/// A one-file parquet source under `dir` whose `id` key is NULL on exactly
/// one of its three rows — the shape every keyed pipeline refuses typed as
/// `InvalidKey { column: "id", null_count: 1 }`, in-process and placed.
/// Returns the source's `file://` URL.
pub fn write_null_key_source(dir: &Path) -> String {
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let src_dir = dir.join("null_key");
    std::fs::create_dir_all(&src_dir).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![Some(0i64), None, Some(2)])),
            Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(src_dir.join("part0.parquet")).unwrap();
    let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    format!("file://{}", src_dir.display())
}

/// Run `sql` as one Flight SQL statement against the server at `addr` —
/// `execute` for the ticket, then a raw `do_get` — so a statement's failure
/// arrives as the `Status` the server sent, its engine-error detail intact
/// (`jammi_wire::error_from_status` rebuilds the typed error), never
/// stringified by the SQL client's own error fold. No session header: the
/// statement runs unscoped.
pub async fn flight_statement(
    addr: std::net::SocketAddr,
    sql: &str,
) -> Result<Vec<arrow::array::RecordBatch>, tonic::Status> {
    use arrow_flight::decode::FlightRecordBatchStream;
    use arrow_flight::error::FlightError;
    use arrow_flight::flight_service_client::FlightServiceClient;
    use arrow_flight::sql::client::FlightSqlServiceClient;
    use futures::TryStreamExt;

    let channel = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
        .expect("flight endpoint")
        .connect()
        .await
        .map_err(|e| tonic::Status::unavailable(e.to_string()))?;
    let mut sql_client = FlightSqlServiceClient::new(channel.clone());
    let info = sql_client
        .execute(sql.to_string(), None)
        .await
        .map_err(|e| tonic::Status::internal(format!("execute: {e}")))?;
    let ticket = info
        .endpoint
        .first()
        .and_then(|e| e.ticket.clone())
        .expect("flight info carries one ticket");
    let stream = FlightServiceClient::new(channel)
        .do_get(tonic::Request::new(ticket))
        .await?
        .into_inner();
    FlightRecordBatchStream::new_from_flight_data(stream.map_err(FlightError::from))
        .try_collect()
        .await
        .map_err(|e| match e {
            FlightError::Tonic(status) => *status,
            other => tonic::Status::internal(other.to_string()),
        })
}

/// Convert a `file://...` URL back into a filesystem `PathBuf` for tests
/// that need to exercise on-disk file existence checks (e.g. asserting a
/// sidecar bundle was written, or peeking at the raw bytes a result-table
/// row references). Returns the input unchanged when no `file://` prefix
/// is present so the helper composes with callers that already strip it.
pub fn url_to_path(url: &str) -> PathBuf {
    PathBuf::from(url.strip_prefix("file://").unwrap_or(url))
}

/// Create a JammiConfig pointing at a temporary artifact directory.
pub fn test_config(artifact_dir: &Path) -> jammi_db::config::JammiConfig {
    jammi_db::config::JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        gpu: jammi_db::config::GpuConfig {
            device: -1,
            ..Default::default()
        },
        inference: jammi_db::config::InferenceConfig {
            batch_size: 8,
            ..Default::default()
        },
        logging: jammi_db::config::LoggingConfig {
            level: "debug".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Register a custom evidence channel with the catalog. Used by tests
/// that exercise the data-driven provenance machinery beyond the seeded
/// `vector` and `inference` channels.
pub async fn register_test_channel(
    catalog: &jammi_db::catalog::Catalog,
    id: &str,
    priority: i32,
    columns: &[(&str, jammi_db::catalog::channel_repo::ChannelColumnType)],
) -> jammi_db::error::Result<()> {
    let spec = jammi_db::catalog::channel_repo::ChannelSpec {
        id: jammi_db::ChannelId::new(id)?,
        priority,
        columns: columns
            .iter()
            .map(
                |(name, dtype)| jammi_db::catalog::channel_repo::ChannelColumn {
                    name: (*name).into(),
                    data_type: *dtype,
                },
            )
            .collect(),
    };
    catalog.channels().register(&spec).await
}

/// Build the materialization-contract triple for a **synthetic seed embedding
/// table** that an integration test materialises directly through
/// [`jammi_db::store::ResultStore::materialize_embedding_table`] (a fixture set
/// up before exercising propagation / context-prediction / serving).
///
/// The triple describes a context-set-shaped materialisation (no model invoked
/// — the vectors are synthetic) over a CPU device, with the source recorded as
/// an unpinned input at a fixed instant. Tests use it so the contract args are
/// declared once, not copy-pasted per fixture.
pub fn synthetic_seed_contract(
    encoder_id: &str,
    source_id: &str,
    dimensions: usize,
) -> (
    jammi_db::store::manifest::ProducingDescriptor,
    jammi_db::store::manifest::MaterializationEnv,
    Vec<jammi_db::store::manifest::InputAnchor>,
) {
    use jammi_db::store::manifest::{
        ComputeDevice, ContextAggregator, ContextCandidateSource, InputAnchor, MaterializationEnv,
        ProducingDescriptor,
    };
    let descriptor = ProducingDescriptor::ContextSet {
        encoder_id: encoder_id.to_string(),
        source_id: source_id.to_string(),
        embedding_table: None,
        candidate_source: ContextCandidateSource::Ann { k: 5 },
        value_columns: Vec::new(),
        aggregator: ContextAggregator::Mean,
        exclude_self: true,
        split: None,
        dimensions,
    };
    let env = MaterializationEnv::new(ComputeDevice::Cpu, Vec::new());
    let inputs = vec![InputAnchor::unpinned_at_instant(
        source_id,
        "1970-01-01T00:00:00Z",
    )];
    (descriptor, env, inputs)
}

/// Write a minimal `.materialization.json` sidecar beside a result table's
/// Parquet object, for recovery/store tests that construct a *promotable* torn
/// `building` state — a crash that landed the Parquet AND the manifest but never
/// committed the `building -> ready` flip. The manifest attests the given
/// Parquet bytes' digest under a synthetic context-set descriptor on CPU, so
/// recovery promotes the row (it has a manifest) and the summary columns
/// backfill from it.
pub async fn write_manifest_sidecar_for(
    store: &jammi_db::store::ResultStore,
    parquet_url: &jammi_db::storage::StorageUrl,
    source_id: &str,
    dimensions: usize,
) {
    use jammi_db::store::manifest::{
        ArtifactDigest, ComputeDevice, ContextAggregator, ContextCandidateSource, InputAnchor,
        MaterializationEnv, MaterializationManifest, ProducingDescriptor,
    };
    let handle = store.open_parquet(parquet_url).unwrap();
    let path = handle.data_path().unwrap();
    let bytes = handle.get_bytes(&path).await.unwrap();
    let manifest = MaterializationManifest::compute(
        &ProducingDescriptor::ContextSet {
            encoder_id: "synthetic-embed".into(),
            source_id: source_id.into(),
            embedding_table: None,
            candidate_source: ContextCandidateSource::Ann { k: 5 },
            value_columns: Vec::new(),
            aggregator: ContextAggregator::Mean,
            exclude_self: true,
            split: None,
            dimensions,
        },
        &MaterializationEnv::new(ComputeDevice::Cpu, Vec::new()),
        vec![InputAnchor::unpinned_at_instant(
            source_id,
            "1970-01-01T00:00:00Z",
        )],
        ArtifactDigest::of_bytes(&bytes),
        jammi_db::store::manifest::parquet_leaves(&bytes).unwrap(),
        "test-run".into(),
        "1970-01-01T00:00:00Z".into(),
    )
    .unwrap();
    let sidecar = handle.sibling_path("materialization.json").unwrap();
    handle
        .put_bytes(&sidecar, manifest.to_json_bytes().unwrap().into())
        .await
        .unwrap();
}

/// Detach a writer's [`jammi_db::store::BuildingTable`] handle WITHOUT any catalog transition
/// and force the row's lease into the past — the state a dead writer leaves
/// behind once its lease has run out. The torn-state fixtures every recovery
/// test constructs by hand need this: a live handle heartbeats its lease, so
/// a `recover()` run beside it would (correctly) skip the row. Returns the
/// table name. Asserts the row was `building` under a live lease first, so a
/// fixture that never held the lease cannot pass vacuously.
pub async fn abandon_building(
    catalog: &jammi_db::catalog::Catalog,
    building: jammi_db::store::BuildingTable,
) -> String {
    use jammi_db::catalog::backend::{SqlValue, TxOptions};
    use jammi_db::catalog::status::ResultTableStatus;
    use jammi_db::tenant_scope::TenantBinding;

    let name = building.table_name().to_string();
    let before = TenantBinding::admin_scope(catalog.get_result_table(&name))
        .await
        .unwrap()
        .expect("the building row exists");
    assert_eq!(
        before.status,
        ResultTableStatus::Building.to_string(),
        "abandon_building: precondition — the row is still `building`"
    );
    // The catalog's own backend-correct liveness predicate, not a Rust-side
    // string compare against `lease_expires_at` — that stored value is a
    // Postgres-clock expression's text rendering on Postgres, not a
    // `canonical_stamp_now()`-shaped string a naive `>` compare here would assume.
    assert!(
        TenantBinding::admin_scope(catalog.list_live_building_tables())
            .await
            .unwrap()
            .iter()
            .any(|t| t.table_name == name),
        "abandon_building: precondition — the row carries a live lease, got {:?}",
        before.lease_expires_at
    );
    building.detach();
    let stamped = name.clone();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE result_tables SET lease_expires_at = '1970-01-01T00:00:00.000000Z' \
                     WHERE table_name = $1",
                    &[SqlValue::TextOwned(stamped)],
                )
                .await
            })
        })
        .await
        .unwrap();
    name
}

/// A test query vector validated at the literal's own width — the width of
/// the vectors the test puts it against. The one way a test turns a literal
/// into the [`jammi_db::index::ValidatedQuery`] every search consumer takes;
/// an index or scan of another width refuses it as its own artifact's
/// mismatch, and a test of the CALLER's width fault goes through the entry
/// that holds the authority.
pub fn vq(v: &[f32]) -> jammi_db::index::ValidatedQuery {
    jammi_db::index::validate_query(v.to_vec(), v.len(), jammi_db::index::QuerySource::Caller)
        .expect("a finite literal test query validates")
}
