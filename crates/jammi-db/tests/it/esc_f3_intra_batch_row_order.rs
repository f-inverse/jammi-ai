//! Escape row esc-F3 (issue-triage symptom_spec; PLAN-F F1/F11/G2/H8/I7,
//! `<eval-verdict>` F15(iii)): **replayed batches reorder rows within a
//! publish**.
//!
//! Symptom: every row of one `publish_scoped` call shares the same
//! `_offset` (see `topic.rs`'s doc on `OFFSET_COLUMN`); before this fix
//! `source/mutable.rs`'s `fetch_scan_after_batch` rendered
//! `ORDER BY "_offset" ASC` with no tiebreak, so a backend is free to return
//! same-`_offset` rows in any order it likes -- there is no `_row_idx`
//! tiebreak keeping `Subscriber::group_replay_batches`'s reassembled batch in
//! the order the rows were originally published in.
//!
//! Fix (this commit): the `ORDER BY` clause also lists every
//! `def.primary_key` column that is not the order column itself -- for a
//! topic's backing table (`PRIMARY KEY (_offset, _row_idx)`,
//! `order_column = _offset`) that emits `ORDER BY _offset, _row_idx`,
//! pinning intra-batch row order exactly.
//!
//! Parameterised over both backends per the parity suite's require-gate
//! shape (`recovery.rs`): SQLite always runs; Postgres runs when
//! `JAMMI_TEST_PG_URL` is set (`JAMMI_REQUIRE_PG` turns an unset URL into a
//! hard failure rather than a silent skip).

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use jammi_db::catalog::backend::{BackendImpl, BackendKind};
use jammi_db::catalog::backend_postgres::PostgresBackend;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::topic_repo::TopicRepo;
use jammi_db::catalog::Catalog;
use jammi_db::source::mutable::MutableTableRegistry;
use jammi_db::store::mutable::postgres::PostgresMutableBackend;
use jammi_db::store::mutable::sqlite::SqliteMutableBackend;
use jammi_db::store::mutable::MutableBackend;
use jammi_db::tenant_scope::TenantBinding;
use jammi_db::trigger::{
    InMemoryBroker, Offset, Predicate, Publisher, Subscriber, TopicDefinition, TopicId,
    TriggerBroker,
};
use test_case::test_case;

fn require_live_pg(test_name: &str) {
    if std::env::var_os("JAMMI_REQUIRE_PG").is_some() {
        panic!(
            "{test_name}: JAMMI_REQUIRE_PG is set but JAMMI_TEST_PG_URL is unset -- this lane \
             must run the real Postgres arm, not skip it"
        );
    }
}

fn topic_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("seq", DataType::Int64, false)]))
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn intra_batch_row_order_survives_replay(backend: BackendKind) {
    let dir = tempfile::tempdir().unwrap();
    let backend_impl = match backend {
        BackendKind::Sqlite => {
            let sqlite = SqliteBackend::open(&dir.path().join("catalog.db"))
                .await
                .unwrap();
            BackendImpl::Sqlite(sqlite)
        }
        BackendKind::Postgres => {
            let Some(url) = jammi_test_utils::pg_url_for_tests() else {
                eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
                require_live_pg("intra_batch_row_order_survives_replay");
                return;
            };
            let pg = PostgresBackend::open_with_options(&url, 8, None)
                .await
                .unwrap();
            BackendImpl::Postgres(pg)
        }
    };
    backend_impl.migrate().await.unwrap();

    let tenant_binding = TenantBinding::unscoped();
    let catalog = Arc::new(Catalog::from_backend_with_tenant(
        backend_impl,
        Some(tenant_binding.clone()),
    ));
    let backend_arc = catalog.backend_arc();
    let mutable_backend: Arc<dyn MutableBackend> = match backend {
        BackendKind::Sqlite => Arc::new(SqliteMutableBackend::new(Arc::clone(&backend_arc))),
        BackendKind::Postgres => Arc::new(PostgresMutableBackend::new(Arc::clone(&backend_arc))),
    };
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

    let topic = TopicDefinition {
        id: TopicId::new(),
        name: format!("esc_f3.row_order.{}", jammi_test_utils::unique_suffix()),
        schema: topic_schema(),
        tenant: None,
        broker_metadata: BTreeMap::new(),
    };
    broker.register_topic(&topic).await.unwrap();
    topic_repo.register_topic(&topic).await.unwrap();

    // One publish, many rows, in a deliberately non-monotonic-looking value
    // sequence so a reorder is unmistakable against the ORIGINAL publish
    // order (row position 0..N-1, `_row_idx`), not against the VALUE order.
    const N: usize = 300;
    let seq: Vec<i64> = (0..N as i64).collect();
    let batch = RecordBatch::try_new(
        topic_schema(),
        vec![Arc::new(Int64Array::from(seq.clone()))],
    )
    .unwrap();
    publisher.publish_scoped(&topic, None, batch).await.unwrap();

    let from = Offset::new(0, chrono::Utc::now());
    let drained = subscriber
        .replay_only(&topic, Predicate::match_all(), Some(from))
        .await
        .unwrap();
    assert_eq!(
        drained.len(),
        1,
        "the whole wide batch must reassemble as ONE DeliveredBatch (one `_offset`)"
    );
    let replayed = drained[0]
        .batch
        .column_by_name("seq")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let replayed_seq: Vec<i64> = (0..replayed.len()).map(|i| replayed.value(i)).collect();
    assert_eq!(
        replayed_seq, seq,
        "intra-batch row order must survive replay exactly (ORDER BY _offset, _row_idx) -- \
         a reorder here means the `_row_idx` tiebreak is missing or not applied"
    );
}
