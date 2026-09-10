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

/// The first attempt at this escape row found insertion order preserved
/// even WITHOUT the tiebreak, because a fresh single-batch `INSERT` has no
/// reason to physically reorder on either backend. This test constructs
/// the way Postgres ACTUALLY reorders rows: it physically relocates an
/// EARLY row (`_row_idx = 0`) of an already-published batch, WITHOUT
/// changing its logical `_row_idx` or any column value, then replays and
/// asserts the logical (publish) order still comes back exactly.
///
/// The physical relocation itself is a `DELETE` + re-`INSERT` in one
/// writable-CTE statement (`WITH moved AS (DELETE ... RETURNING *) INSERT
/// ... SELECT * FROM moved`) rather than a plain `UPDATE`: a same-value
/// `UPDATE` on an unindexed column is HOT-eligible (Heap-Only Tuple) when
/// the page has slack, so Postgres keeps the new version on the SAME page
/// -- not a reliable reorder for a small (50-row) table. A fresh `INSERT`
/// is never HOT and lands wherever the heap's free-space map offers, which
/// for a table nobody has vacuumed since its initial bulk insert is, in
/// practice, past the still-live original rows.
///
/// RED (captured against this commit's own worktree by temporarily
/// reverting the `ORDER BY` in `source/mutable.rs::fetch_scan_after_batch`
/// to `order_col` only, real Postgres, `cargo test -p jammi-db --test it
/// --features live-postgres-tests intra_batch_row_order_survives_update_churn`;
/// thread id, a run-specific value, elided):
///
/// ```text
/// thread 'esc_f3_intra_batch_row_order::intra_batch_row_order_survives_update_churn_on_an_early_row_postgres' panicked at `.batch`, crates/jammi-db/tests/it/esc_f3_intra_batch_row_order.rs:300:5:
/// assertion `left == right` failed: intra-batch row order must survive an UPDATE that
/// physically reorders a row on Postgres's heap -- the ORDER BY _offset, _row_idx tiebreak
/// must pin logical order regardless of physical row placement
///   left: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
///          25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
///          47, 48, 49, 0]
///  right: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
///          24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
///          46, 47, 48, 49]
/// ```
///
/// Row 0 (the relocated row) came back LAST, at the physical end of the
/// heap -- reproducing the symptom the ORDER BY `_row_idx` tiebreak (this
/// commit's fix, currently in place and left untouched by this test) exists
/// to prevent. GREEN below with the fix restored. SQLite was not attempted
/// for this reproduction: SQLite's b-tree storage is organized by rowid/PK,
/// so a delete-then-reinsert is far more likely to reuse the same rowid
/// region or otherwise preserve scan order, and is not expected to reorder
/// the same way -- not claimed as evidence here; the escape row's
/// evidence is the Postgres run above.
#[tokio::test]
async fn intra_batch_row_order_survives_update_churn_on_an_early_row_postgres() {
    let Some(url) = jammi_test_utils::pg_url_for_tests() else {
        eprintln!(
            "skipping intra_batch_row_order_survives_update_churn_on_an_early_row_postgres: \
             JAMMI_TEST_PG_URL unset"
        );
        require_live_pg("intra_batch_row_order_survives_update_churn_on_an_early_row_postgres");
        return;
    };
    let pg = PostgresBackend::open_with_options(&url, 8, None)
        .await
        .unwrap();
    let backend_impl = BackendImpl::Postgres(pg);
    backend_impl.migrate().await.unwrap();

    let tenant_binding = TenantBinding::unscoped();
    let catalog = Arc::new(Catalog::from_backend_with_tenant(
        backend_impl,
        Some(tenant_binding.clone()),
    ));
    let backend_arc = catalog.backend_arc();
    let mutable_backend: Arc<dyn MutableBackend> =
        Arc::new(PostgresMutableBackend::new(Arc::clone(&backend_arc)));
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
        name: format!("esc_f3.update_churn.{}", jammi_test_utils::unique_suffix()),
        schema: topic_schema(),
        tenant: None,
        broker_metadata: BTreeMap::new(),
    };
    broker.register_topic(&topic).await.unwrap();
    topic_repo.register_topic(&topic).await.unwrap();

    const N: usize = 50;
    let seq: Vec<i64> = (0..N as i64).collect();
    let batch = RecordBatch::try_new(
        topic_schema(),
        vec![Arc::new(Int64Array::from(seq.clone()))],
    )
    .unwrap();
    publisher.publish_scoped(&topic, None, batch).await.unwrap();

    // Physically relocate row `_row_idx = 0` (the FIRST row of the publish)
    // to wherever the heap currently has free space, WITHOUT changing any
    // logical column value: a plain `UPDATE ... SET seq = seq` is HOT-
    // eligible (no indexed column changes and the page has slack for only
    // 50 narrow rows), so Postgres keeps the new tuple version on the SAME
    // page -- not a reliable reorder. A `DELETE` + re-`INSERT` in one
    // writable-CTE statement is not HOT-eligible (it is a brand-new tuple,
    // not an update chain), so the freshly inserted row lands whichever
    // heap position the free-space map currently offers -- which for a
    // table nobody has vacuumed since its initial bulk insert is, in
    // practice, past the still-live original 49 rows.
    let backing = topic.backing_table_name();
    backend_arc
        .transaction(
            jammi_db::catalog::backend::TxOptions::default(),
            move |tx| {
                let backing = backing.clone();
                Box::pin(async move {
                    let quoted = format!("\"{}\"", backing.replace('"', "\"\""));
                    let sql = format!(
                        "WITH moved AS (DELETE FROM {quoted} WHERE \"_row_idx\" = 0 RETURNING *) \
                         INSERT INTO {quoted} SELECT * FROM moved"
                    );
                    tx.execute(&sql, &[]).await?;
                    Ok::<(), jammi_db::catalog::backend::BackendError>(())
                })
            },
        )
        .await
        .unwrap();

    let from = Offset::new(0, chrono::Utc::now());
    let drained = subscriber
        .replay_only(&topic, Predicate::match_all(), Some(from))
        .await
        .unwrap();
    assert_eq!(
        drained.len(),
        1,
        "the whole batch must still reassemble as ONE DeliveredBatch after the UPDATE"
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
        "intra-batch row order must survive an UPDATE that physically reorders a row on \
         Postgres's heap -- the ORDER BY _offset, _row_idx tiebreak must pin logical order \
         regardless of physical row placement"
    );
}
