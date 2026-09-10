//! Escape row esc-100-trigger-replay-folds-declared-column-types:
//! **`register_topic` accepts column types publish/replay cannot handle**.
//!
//! Symptom: `topic_repo.rs` accepts 13 Arrow column types (Boolean,
//! Int8/16/32/64, UInt8/16/32/64, Float32/64, Utf8, Binary), but before this
//! fix the backing-table REPLAY path (`source/mutable.rs`'s
//! `fetch_scan_after_batch`) folded every integer width into `Int64Array` and
//! every non-numeric/non-boolean/non-text type (including `Binary`) into
//! `StringArray`. `Subscriber::group_replay_batches` then slices those
//! wrongly-typed columns and calls `RecordBatch::try_new` against the
//! topic's DECLARED schema (`Int32`, `Binary`, …) -- a hard schema/array
//! `DataType` mismatch that `try_new` refuses, independent of which backend
//! is under test (SQLite is enough to reproduce it; no Postgres round-trip
//! required).
//!
//! Control: an `Int64`/`Utf8`-only topic (the two types the fold happened to
//! decode into) round-trips fine -- already covered by every other test in
//! `trigger.rs`.
//!
//! Fix (this commit): `fetch_scan_after_batch` (and `provider.rs`'s general
//! mutable-table scan) decode width-faithfully per `DataType` and build the
//! EXACT Arrow array the schema declares.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{
    Array, BinaryArray, Float32Array, Int32Array, RecordBatch, UInt16Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use jammi_db::catalog::backend::BackendImpl;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::topic_repo::TopicRepo;
use jammi_db::catalog::Catalog;
use jammi_db::source::mutable::MutableTableRegistry;
use jammi_db::store::mutable::sqlite::SqliteMutableBackend;
use jammi_db::store::mutable::MutableBackend;
use jammi_db::tenant_scope::TenantBinding;
use jammi_db::trigger::{
    InMemoryBroker, Offset, Predicate, Publisher, Subscriber, TopicDefinition, TopicId,
    TriggerBroker,
};

fn topic_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("i32_col", DataType::Int32, false),
        Field::new("u16_col", DataType::UInt16, false),
        Field::new("u64_col", DataType::UInt64, false),
        Field::new("f32_col", DataType::Float32, false),
        Field::new("bytes_col", DataType::Binary, false),
    ]))
}

#[tokio::test]
async fn every_accepted_type_round_trips_through_replay() {
    let dir = tempfile::tempdir().unwrap();
    let sqlite = SqliteBackend::open(&dir.path().join("catalog.db"))
        .await
        .unwrap();
    let backend_impl = BackendImpl::Sqlite(sqlite);
    backend_impl.migrate().await.unwrap();

    let tenant_binding = TenantBinding::unscoped();
    let catalog = Arc::new(Catalog::from_backend_with_tenant(
        backend_impl,
        Some(tenant_binding.clone()),
    ));
    let backend_arc = catalog.backend_arc();
    let mutable_backend: Arc<dyn MutableBackend> =
        Arc::new(SqliteMutableBackend::new(Arc::clone(&backend_arc)));
    let registry = Arc::new(MutableTableRegistry::new(
        Arc::clone(&catalog),
        mutable_backend,
        tenant_binding,
    ));
    let broker: Arc<dyn TriggerBroker> = Arc::new(InMemoryBroker::new());
    let topic_repo = TopicRepo::new(Arc::clone(&catalog), Arc::clone(&registry));
    let publisher = Publisher::new(
        Arc::clone(&broker),
        Arc::clone(&backend_arc),
        Arc::clone(&registry),
    );
    let subscriber = Subscriber::new(Arc::clone(&broker), Arc::clone(&registry));

    // Register through `TopicRepo::register_topic` -- the persisted
    // type-name table, not an in-process `TopicDefinition` -- so the oracle
    // pins the actual catalog round-trip, not just the in-memory shape.
    let topic = TopicDefinition {
        id: TopicId::new(),
        name: "esc_100.every_type".to_string(),
        schema: topic_schema(),
        tenant: None,
        broker_metadata: BTreeMap::new(),
    };
    broker.register_topic(&topic).await.unwrap();
    topic_repo.register_topic(&topic).await.unwrap();
    let loaded = topic_repo
        .lookup_by_name(&topic.name, None)
        .await
        .unwrap()
        .expect("topic persisted");

    let batch = RecordBatch::try_new(
        Arc::clone(&loaded.schema),
        vec![
            Arc::new(Int32Array::from(vec![-7i32, 42])),
            Arc::new(UInt16Array::from(vec![1u16, 65535])),
            Arc::new(UInt64Array::from(vec![0u64, i64::MAX as u64])),
            Arc::new(Float32Array::from(vec![1.5f32, -2.25])),
            Arc::new(BinaryArray::from(vec![
                b"hello".as_ref(),
                b"\x00\x01\xff".as_ref(),
            ])),
        ],
    )
    .unwrap();

    publisher
        .publish_scoped(&loaded, None, batch.clone())
        .await
        .expect("publish accepts every declared column type");

    let from = Offset::new(0, chrono::Utc::now());
    let drained = subscriber
        .replay_only(&loaded, Predicate::match_all(), Some(from))
        .await
        .expect("replay must reconstruct the batch against the declared schema");
    assert_eq!(drained.len(), 1);
    let replayed = &drained[0].batch;
    assert_eq!(
        replayed.schema().as_ref(),
        loaded.schema.as_ref(),
        "replayed batch schema must equal the declared topic schema exactly \
         (no Int64/Utf8 fold)"
    );

    let i32_col = replayed
        .column_by_name("i32_col")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("i32_col must decode as Int32Array, not folded to Int64Array");
    assert_eq!(i32_col.values(), &[-7, 42]);

    let u16_col = replayed
        .column_by_name("u16_col")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt16Array>()
        .expect("u16_col must decode as UInt16Array");
    assert_eq!(u16_col.values(), &[1, 65535]);

    let u64_col = replayed
        .column_by_name("u64_col")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("u64_col must decode as UInt64Array");
    assert_eq!(u64_col.values(), &[0, i64::MAX as u64]);

    let f32_col = replayed
        .column_by_name("f32_col")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("f32_col must decode as Float32Array, not folded to Float64Array");
    assert_eq!(f32_col.values(), &[1.5, -2.25]);

    let bytes_col = replayed
        .column_by_name("bytes_col")
        .unwrap()
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("bytes_col must decode as BinaryArray, not folded to StringArray");
    assert_eq!(bytes_col.value(0), b"hello");
    assert_eq!(bytes_col.value(1), b"\x00\x01\xff");
}

/// `UInt64` values above `i64::MAX` are refused at publish (BIGINT is the
/// storage type on both backends; the range is validated at the edge).
#[tokio::test]
async fn uint64_above_i64_max_is_refused_at_publish() {
    let dir = tempfile::tempdir().unwrap();
    let sqlite = SqliteBackend::open(&dir.path().join("catalog.db"))
        .await
        .unwrap();
    let backend_impl = BackendImpl::Sqlite(sqlite);
    backend_impl.migrate().await.unwrap();
    let tenant_binding = TenantBinding::unscoped();
    let catalog = Arc::new(Catalog::from_backend_with_tenant(
        backend_impl,
        Some(tenant_binding.clone()),
    ));
    let backend_arc = catalog.backend_arc();
    let mutable_backend: Arc<dyn MutableBackend> =
        Arc::new(SqliteMutableBackend::new(Arc::clone(&backend_arc)));
    let registry = Arc::new(MutableTableRegistry::new(
        Arc::clone(&catalog),
        mutable_backend,
        tenant_binding,
    ));
    let broker: Arc<dyn TriggerBroker> = Arc::new(InMemoryBroker::new());
    let topic_repo = TopicRepo::new(Arc::clone(&catalog), Arc::clone(&registry));
    let publisher = Publisher::new(
        Arc::clone(&broker),
        Arc::clone(&backend_arc),
        Arc::clone(&registry),
    );

    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "u64_col",
        DataType::UInt64,
        false,
    )]));
    let topic = TopicDefinition {
        id: TopicId::new(),
        name: "esc_100.uint64_overflow".to_string(),
        schema: Arc::clone(&schema),
        tenant: None,
        broker_metadata: BTreeMap::new(),
    };
    broker.register_topic(&topic).await.unwrap();
    topic_repo.register_topic(&topic).await.unwrap();

    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(vec![u64::MAX]))]).unwrap();
    let err = publisher
        .publish_scoped(&topic, None, batch)
        .await
        .expect_err("a UInt64 value above i64::MAX must be refused, not silently truncated");
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("exceed") || msg.to_lowercase().contains("range"),
        "error should name the range violation: {msg}"
    );
}
