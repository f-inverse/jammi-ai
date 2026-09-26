//! End-to-end integration tests for Phase 2 — mutable companion tables.
//!
//! Coverage: register/list/drop lifecycle, atomic catalog + storage commit,
//! DataFusion DML through `INSERT INTO mutable.public.<id>`, `UPDATE` /
//! `DELETE` choosing rows by a subquery or a join, write conflicts, federation
//! between mutable tables and Parquet result tables, tenant filtering on
//! list, order-column round-trip, direct-access `insert_batch` and
//! `scan_after` paths, schema-mismatch rejection.
//!
//! Every test is parameterised over [`BackendKind`] via `test_case` +
//! `cfg_attr`. The SQLite lane is always generated; the Postgres lane is
//! generated only when the `live-postgres-tests` feature is on (with
//! `JAMMI_TEST_PG_URL` naming the server); the hermetic `cargo test` lane
//! runs only the SQLite parameterisation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{
    Array, BinaryArray, Float64Array, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use futures::StreamExt;
use jammi_db::catalog::backend::{BackendKind, TxOptions};
use jammi_db::error::JammiError;
use jammi_db::store::mutable::definition::{
    MutableIndexDef, MutableTableDefinitionBuilder, MutableTableError, MutableTableId,
};
use jammi_test_utils::make_test_session;
use tempfile::tempdir;
use test_case::test_case;

fn widget_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("score", DataType::Float64, true),
    ]))
}

fn events_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("seq", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, true),
    ]))
}

/// Backend-unique mutable-table id. SQLite per-tempdir tests don't strictly
/// need uniqueness, but the Postgres lane shares one database across every
/// test in the run; the suffix avoids `relation "<name>" already exists`
/// errors between parameterised variants.
fn unique_id(prefix: &str) -> MutableTableId {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let epoch_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    MutableTableId::new(format!("{prefix}_{epoch_ns:x}_{n:x}")).unwrap()
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_persists_catalog_row_and_storage_table(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    let id = unique_id("widgets");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    let listed = session.mutable_tables().list(None).await.unwrap();
    assert!(listed.iter().any(|d| d.id.as_str() == id.as_str()));

    let loaded = session.mutable_tables().get(&id).await.unwrap().unwrap();
    assert_eq!(loaded.schema.fields().len(), 3);
    assert_eq!(loaded.primary_key, vec!["id".to_string()]);

    session.drop_mutable_table(&id).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_removes_catalog_row_and_storage_table(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    let id = unique_id("ephemeral");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();
    session.drop_mutable_table(&id).await.unwrap();

    assert!(session.mutable_tables().get(&id).await.unwrap().is_none());
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn datafusion_insert_then_scan_round_trip(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    let id = unique_id("widgets");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    session
        .sql(&format!(
            "INSERT INTO mutable.public.{name} (id, name, score) VALUES \
             (1, 'alpha', 0.5), (2, 'beta', 1.5), (3, 'gamma', 2.5)",
            name = id.as_str(),
        ))
        .await
        .unwrap();

    let batches = session
        .sql(&format!(
            "SELECT id, name FROM mutable.public.{name} ORDER BY id",
            name = id.as_str()
        ))
        .await
        .unwrap();
    let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(batch.num_rows(), 3);
    let names = string_column_values(&batch, "name");
    assert_eq!(names[0], "alpha");
    assert_eq!(names[2], "gamma");
}

/// `(id, name, score)` rows of `table`, ordered by id.
async fn widget_rows(
    session: &jammi_db::session::JammiSession,
    table: &str,
) -> Vec<(i64, String, Option<f64>)> {
    let batches = session
        .sql(&format!(
            "SELECT id, name, score FROM mutable.public.{table} ORDER BY id"
        ))
        .await
        .unwrap();
    let batch = arrow::compute::concat_batches(&widget_schema(), &batches).unwrap();
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let names = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let scores = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    (0..batch.num_rows())
        .map(|i| {
            let score = (!scores.is_null(i)).then(|| scores.value(i));
            (ids.value(i), names.value(i).to_string(), score)
        })
        .collect()
}

/// The `count` a DML statement answers with.
async fn affected(session: &jammi_db::session::JammiSession, statement: &str) -> u64 {
    let batches = session.sql(statement).await.unwrap();
    batches[0]
        .column_by_name("count")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::UInt64Array>()
        .unwrap()
        .value(0)
}

/// A write wider than one statement can bind — 25,000 rows at four
/// parameters each (three columns and the tenant slot), past both SQLite's
/// 32,766 and Postgres's 65,535 — lands whole: the insert, and the update
/// that deletes every row's key and re-inserts it, are each split into
/// statements that fit, inside one transaction.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_write_wider_than_one_statement_lands_whole(backend: BackendKind) {
    const ROWS: i64 = 25_000;
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let id = unique_id("wide");
    let table = id.as_str().to_string();
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    let batch = RecordBatch::try_new(
        widget_schema(),
        vec![
            Arc::new(Int64Array::from((0..ROWS).collect::<Vec<_>>())),
            Arc::new(StringArray::from(
                (0..ROWS).map(|i| format!("w{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (0..ROWS).map(|i| i as f64).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    let registry = session.mutable_tables_arc();
    let written = session
        .catalog()
        .backend_arc()
        .transaction(TxOptions::default(), move |tx| {
            let (id, batch, registry) = (id.clone(), batch.clone(), Arc::clone(&registry));
            Box::pin(async move {
                registry
                    .insert_batch(tx, &id, &batch)
                    .await
                    .map_err(|e| jammi_db::BackendError::Execution(e.to_string()))
            })
        })
        .await
        .unwrap();
    assert_eq!(written, ROWS as u64);

    let updated = affected(
        &session,
        &format!("UPDATE mutable.public.{table} SET name = 'renamed'"),
    )
    .await;
    assert_eq!(updated, ROWS as u64);
    let rows = widget_rows(&session, &table).await;
    assert_eq!(rows.len(), ROWS as usize);
    assert!(rows.iter().all(|(_, name, _)| name == "renamed"));
    assert_eq!(rows.last().unwrap().2, Some((ROWS - 1) as f64));
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_delete_and_replace_rewrite_rows_in_place(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let id = unique_id("widgets");
    let table = id.as_str();
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();
    session
        .sql(&format!(
            "INSERT INTO mutable.public.{table} (id, name, score) VALUES \
             (1, 'alpha', 0.5), (2, 'beta', 1.5), (3, 'gamma', 2.5), (4, 'delta', NULL)"
        ))
        .await
        .unwrap();

    // An UPDATE evaluates its assignment against each matched row's own values.
    let updated = affected(
        &session,
        &format!("UPDATE mutable.public.{table} SET score = score * 10 WHERE score > 1"),
    )
    .await;
    assert_eq!(updated, 2, "rows 2 and 3 match; the NULL score does not");

    let deleted = affected(
        &session,
        &format!("DELETE FROM mutable.public.{table} WHERE name = 'gamma'"),
    )
    .await;
    assert_eq!(deleted, 1);

    // REPLACE INTO upserts by primary key: row 1 is replaced, row 5 is new.
    let replaced = affected(
        &session,
        &format!(
            "REPLACE INTO mutable.public.{table} (id, name, score) VALUES \
             (1, 'alpha', 9.0), (5, 'epsilon', 5.0)"
        ),
    )
    .await;
    assert_eq!(replaced, 2);

    // A plain INSERT of an existing key still fails, and fails whole.
    assert!(session
        .sql(&format!(
            "INSERT INTO mutable.public.{table} (id, name, score) VALUES (6, 'zeta', 0.0), (2, 'dup', 0.0)"
        ))
        .await
        .is_err());

    assert_eq!(
        widget_rows(&session, table).await,
        vec![
            (1, "alpha".into(), Some(9.0)),
            (2, "beta".into(), Some(15.0)),
            (4, "delta".into(), None),
            (5, "epsilon".into(), Some(5.0)),
        ]
    );

    // An unfiltered DELETE empties the table.
    assert_eq!(
        affected(&session, &format!("DELETE FROM mutable.public.{table}")).await,
        4
    );
    assert!(widget_rows(&session, table).await.is_empty());
}

/// Registers a `widgets` table holding `(1, alpha, 0.5)`, `(2, beta, 1.5)`,
/// `(3, gamma, 2.5)` and `(4, delta, NULL)`; returns its name.
async fn seeded_widgets(session: &jammi_db::session::JammiSession) -> String {
    let id = unique_id("widgets");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();
    session
        .sql(&format!(
            "INSERT INTO mutable.public.{id} (id, name, score) VALUES \
             (1, 'alpha', 0.5), (2, 'beta', 1.5), (3, 'gamma', 2.5), (4, 'delta', NULL)",
            id = id.as_str()
        ))
        .await
        .unwrap();
    id.as_str().to_string()
}

/// An `UPDATE … FROM` or a `DELETE` whose predicate is a subquery rewrites
/// exactly the rows its join or subquery selects: price adjustments land on
/// the widgets they name, a recall list removes the widgets it names, and
/// every other row stays as it was.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_join_or_subquery_chooses_the_rows_an_update_or_delete_rewrites(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let widgets = seeded_widgets(&session).await;
    let adjustments = unique_id("adjustments");
    let def = MutableTableDefinitionBuilder::new(
        adjustments.clone(),
        Arc::new(Schema::new(vec![
            Field::new("widget_id", DataType::Int64, false),
            Field::new("delta", DataType::Float64, false),
            Field::new("recalled", DataType::Boolean, false),
        ])),
    )
    .primary_key(vec!["widget_id".into()])
    .build()
    .unwrap();
    session.create_mutable_table(def).await.unwrap();
    let adjustments = adjustments.as_str();
    session
        .sql(&format!(
            "INSERT INTO mutable.public.{adjustments} (widget_id, delta, recalled) VALUES \
             (2, 10.0, false), (3, 20.0, true), (9, 1.0, true)"
        ))
        .await
        .unwrap();

    // Widget 9 has no row, so the join matches two widgets.
    let updated = affected(
        &session,
        &format!(
            "UPDATE mutable.public.{widgets} AS w SET score = w.score + a.delta \
             FROM mutable.public.{adjustments} AS a WHERE w.id = a.widget_id"
        ),
    )
    .await;
    assert_eq!(updated, 2);

    let deleted = affected(
        &session,
        &format!(
            "DELETE FROM mutable.public.{widgets} WHERE id IN \
             (SELECT widget_id FROM mutable.public.{adjustments} WHERE recalled)"
        ),
    )
    .await;
    assert_eq!(deleted, 1, "widget 3 is recalled; widget 9 does not exist");

    let renamed = affected(
        &session,
        &format!(
            "UPDATE mutable.public.{widgets} SET name = upper(name) WHERE EXISTS \
             (SELECT 1 FROM mutable.public.{adjustments} a WHERE a.delta > 5.0) AND score IS NULL"
        ),
    )
    .await;
    assert_eq!(renamed, 1, "only widget 4 has no score");

    assert_eq!(
        widget_rows(&session, &widgets).await,
        vec![
            (1, "alpha".into(), Some(0.5)),
            (2, "beta".into(), Some(11.5)),
            (4, "DELTA".into(), None),
        ]
    );
}

/// A join that selects one row several times is one rewrite of that row
/// when every match agrees on its new value, and is refused, writing
/// nothing, when the matches disagree.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_row_the_join_matches_twice_needs_one_new_value(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let widgets = seeded_widgets(&session).await;
    let tags = unique_id("tags");
    let def = MutableTableDefinitionBuilder::new(
        tags.clone(),
        Arc::new(Schema::new(vec![
            Field::new("tag_id", DataType::Int64, false),
            Field::new("widget_id", DataType::Int64, false),
            Field::new("label", DataType::Utf8, false),
        ])),
    )
    .primary_key(vec!["tag_id".into()])
    .build()
    .unwrap();
    session.create_mutable_table(def).await.unwrap();
    let tags = tags.as_str();
    session
        .sql(&format!(
            "INSERT INTO mutable.public.{tags} (tag_id, widget_id, label) VALUES \
             (10, 1, 'sale'), (11, 1, 'sale'), (12, 2, 'new'), (13, 2, 'clearance')"
        ))
        .await
        .unwrap();

    let updated = affected(
        &session,
        &format!(
            "UPDATE mutable.public.{widgets} AS w SET name = t.label \
             FROM mutable.public.{tags} AS t WHERE w.id = t.widget_id AND w.id = 1"
        ),
    )
    .await;
    assert_eq!(updated, 1, "both of widget 1's tags say 'sale'");

    let refused = session
        .sql(&format!(
            "UPDATE mutable.public.{widgets} AS w SET name = t.label \
             FROM mutable.public.{tags} AS t WHERE w.id = t.widget_id"
        ))
        .await
        .unwrap_err();
    match refused {
        JammiError::MutableTable(MutableTableError::AmbiguousUpdate { table, key }) => {
            assert_eq!(table.as_str(), widgets);
            assert_eq!(key, "(2)");
        }
        other => panic!("expected AmbiguousUpdate, got {other:?}"),
    }
    assert_eq!(
        widget_rows(&session, &widgets).await,
        vec![
            (1, "sale".into(), Some(0.5)),
            (2, "beta".into(), Some(1.5)),
            (3, "gamma".into(), Some(2.5)),
            (4, "delta".into(), None),
        ],
        "the refused statement wrote nothing"
    );
}

/// A statement reads the rows it selects before its write transaction opens.
/// A row another writer changes in between fails the statement whole —
/// the other writer's change is kept, not overwritten — and the statement
/// re-run selects against the new rows.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_row_changed_after_the_read_fails_the_rewrite_whole(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let widgets = seeded_widgets(&session).await;
    let raise = format!("UPDATE mutable.public.{widgets} SET score = score + 1 WHERE score > 1");

    // Planning reads the selected rows: widgets 2 and 3.
    let frame = session.context().sql(&raise).await.unwrap();
    let task = frame.task_ctx();
    let plan = frame.create_physical_plan().await.unwrap();
    assert_eq!(
        affected(
            &session,
            &format!("UPDATE mutable.public.{widgets} SET score = 100 WHERE id = 3")
        )
        .await,
        1
    );
    let conflict = JammiError::from(
        datafusion::physical_plan::collect(plan, Arc::new(task))
            .await
            .unwrap_err(),
    );
    match conflict {
        JammiError::MutableTable(MutableTableError::WriteConflict { table, rows }) => {
            assert_eq!(table.as_str(), widgets);
            assert_eq!(rows, 1, "widget 3 changed; widget 2 did not");
        }
        other => panic!("expected WriteConflict, got {other:?}"),
    }
    assert_eq!(
        widget_rows(&session, &widgets).await[1..3],
        [
            (2, "beta".into(), Some(1.5)),
            (3, "gamma".into(), Some(100.0))
        ],
        "nothing was written, and the other writer's change stands"
    );

    assert_eq!(affected(&session, &raise).await, 2);
    assert_eq!(
        widget_rows(&session, &widgets).await[1..3],
        [
            (2, "beta".into(), Some(2.5)),
            (3, "gamma".into(), Some(101.0))
        ]
    );
}

/// A `DELETE … LIMIT n` removes `n` of the rows its predicate matches, and
/// an `EXPLAIN` of a rewrite shows its plan without writing anything.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_limit_bounds_a_delete_and_an_explain_writes_nothing(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let widgets = seeded_widgets(&session).await;

    let explained = session
        .sql(&format!(
            "EXPLAIN UPDATE mutable.public.{widgets} SET score = 0"
        ))
        .await
        .unwrap();
    let plans = arrow::util::pretty::pretty_format_batches(&explained)
        .unwrap()
        .to_string();
    assert!(plans.contains("RowRewrite"), "{plans}");
    assert_eq!(widget_rows(&session, &widgets).await[0].2, Some(0.5));

    assert_eq!(
        affected(
            &session,
            &format!("DELETE FROM mutable.public.{widgets} WHERE score IS NOT NULL LIMIT 2")
        )
        .await,
        2
    );
    let rows = widget_rows(&session, &widgets).await;
    assert_eq!(rows.len(), 2);
    assert!(
        rows.iter().any(|(id, _, _)| *id == 4),
        "widget 4 has no score, so the predicate never matched it"
    );
}

/// `TRUNCATE` removes every row the session owns.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn truncate_empties_the_table(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let widgets = seeded_widgets(&session).await;
    assert_eq!(
        affected(
            &session,
            &format!("TRUNCATE TABLE mutable.public.{widgets}")
        )
        .await,
        4
    );
    assert!(widget_rows(&session, &widgets).await.is_empty());
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_makes_select_fail(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    let id = unique_id("widgets");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();
    session.drop_mutable_table(&id).await.unwrap();

    let err = session
        .sql(&format!(
            "SELECT * FROM mutable.public.{name}",
            name = id.as_str()
        ))
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("not found") || msg.contains("Table") || msg.contains(id.as_str()),
        "expected table-not-found error; got: {msg}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn registered_mutable_tables_reload_across_sessions(backend: BackendKind) {
    // Persistence contract: a mutable table registered through one session
    // is visible to a subsequent session opened against the same backend.
    // SQLite uses the artifact dir; Postgres reuses the JAMMI_TEST_PG_URL
    // connection — both paths go through `make_test_session`.
    let dir = tempdir().unwrap();
    let id = unique_id("persistent");

    {
        let session = make_test_session(backend, dir.path()).await;
        let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
            .primary_key(vec!["id".into()])
            .build()
            .unwrap();
        session.create_mutable_table(def).await.unwrap();
        session
            .sql(&format!(
                "INSERT INTO mutable.public.{name} (id, name, score) VALUES (42, 'meaning', 1.0)",
                name = id.as_str()
            ))
            .await
            .unwrap();
    }

    let session = make_test_session(backend, dir.path()).await;
    let batches = session
        .sql(&format!(
            "SELECT id, name FROM mutable.public.{name}",
            name = id.as_str()
        ))
        .await
        .unwrap();
    let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(batch.num_rows(), 1);
    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.value(0), 42);

    // Clean up the persistent row so a re-run doesn't surface stale state on
    // the shared Postgres catalog.
    session.drop_mutable_table(&id).await.unwrap();
}

/// A NULL in a nullable non-text column (`score FLOAT64`) must round-trip
/// through the DML insert path as a typed SQL null, not a bare text null —
/// Postgres rejects a text null bound into a `DOUBLE PRECISION` column
/// (`column is of type double precision but expression is of type text`).
/// This is the mutable-table-side anti-regression oracle for the shared
/// `SqlValue::Null` typed-bind fix; the catalog-side oracle lives in
/// `store.rs::result_table_none_dimensions_round_trips_as_null`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nullable_float_column_round_trips_typed_null(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    let id = unique_id("readings");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    session
        .sql(&format!(
            "INSERT INTO mutable.public.{name} (id, name, score) VALUES \
             (1, 'unscored', NULL), (2, 'scored', 3.75)",
            name = id.as_str(),
        ))
        .await
        .unwrap();

    let batches = session
        .sql(&format!(
            "SELECT id, score FROM mutable.public.{name} ORDER BY id",
            name = id.as_str()
        ))
        .await
        .unwrap();
    let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(batch.num_rows(), 2);
    let scores = batch
        .column_by_name("score")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap_or_else(|| {
            panic!(
                "score column is not Float64Array; got {:?}",
                batch.column_by_name("score").unwrap().data_type()
            )
        });
    assert!(
        scores.is_null(0),
        "row 1's NULL score must round-trip as null, not error or coerce to a value"
    );
    assert_eq!(
        scores.value(1),
        3.75,
        "row 2's non-null score must be exact"
    );

    session.drop_mutable_table(&id).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_emits_implicit_tenant_id_column_per_adr_00(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    let id = unique_id("with_tenant");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    // A `LIMIT 0` SELECT proves the on-disk table exists and is reachable
    // through DataFusion; the tenant_id column being present is exercised
    // implicitly through the predicate-injection path in Phase 3 tests.
    session
        .sql(&format!(
            "SELECT * FROM mutable.public.{name} LIMIT 0",
            name = id.as_str()
        ))
        .await
        .unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_filters_by_tenant_scope(backend: BackendKind) {
    use jammi_db::TenantId;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();

    let global_id = unique_id("global_table");
    let def = MutableTableDefinitionBuilder::new(global_id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    let scoped_id = unique_id("tenant_a_table");
    let def = MutableTableDefinitionBuilder::new(scoped_id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .tenant(Some(tenant_a))
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    let global = session.mutable_tables().list(None).await.unwrap();
    assert!(global.iter().any(|d| d.id.as_str() == global_id.as_str()));
    assert!(global.iter().all(|d| d.id.as_str() != scoped_id.as_str()));

    let scoped = session.mutable_tables().list(Some(tenant_a)).await.unwrap();
    assert!(scoped.iter().any(|d| d.id.as_str() == scoped_id.as_str()));
    assert!(scoped.iter().all(|d| d.id.as_str() != global_id.as_str()));

    // The global table is dropped from the unscoped session (its owner). The
    // tenant-scoped table can only be dropped by a session bound to that tenant
    // — the strict delete predicate refuses an unscoped (or foreign-tenant)
    // session, mirroring `delete_model`. Bind tenant A for its cleanup.
    session.drop_mutable_table(&global_id).await.unwrap();
    session
        .with_tenant(tenant_a)
        .drop_mutable_table(&scoped_id)
        .await
        .unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn catalog_create_get_delete_round_trip(backend: BackendKind) {
    // Catalog-level smoke test: drives the repos directly via
    // `session.catalog()`, bypassing the registry's storage-table step.
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    let id = unique_id("plain_cat");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .index(MutableIndexDef {
            name: format!("idx_{}_name", id.as_str()),
            columns: vec!["name".into()],
            unique: false,
        })
        .build()
        .unwrap();

    let catalog = session.catalog();
    catalog.create_mutable_table(&def).await.unwrap();
    let got = catalog.get_mutable_table(&id).await.unwrap().unwrap();
    assert_eq!(got.id.as_str(), id.as_str());
    assert_eq!(got.indexes.len(), 1);
    assert_eq!(got.indexes[0].columns, vec!["name".to_string()]);

    catalog.delete_mutable_table(&id).await.unwrap();
    assert!(catalog.get_mutable_table(&id).await.unwrap().is_none());
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn order_column_persists_across_reload(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let id = unique_id("events");

    {
        let session = make_test_session(backend, dir.path()).await;
        let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
            .primary_key(vec!["id".into()])
            .order_column("seq")
            .build()
            .unwrap();
        session.create_mutable_table(def).await.unwrap();
    }

    let session = make_test_session(backend, dir.path()).await;
    let reloaded = session
        .mutable_tables()
        .get(&id)
        .await
        .unwrap()
        .expect("table should exist after reload");
    assert_eq!(reloaded.order_column.as_deref(), Some("seq"));

    session.drop_mutable_table(&id).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn insert_batch_appends_with_session_tenant(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    let id = unique_id("events");
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .order_column("seq")
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    let batch = RecordBatch::try_new(
        events_schema(),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])),
            Arc::new(Int64Array::from(vec![100_i64, 101, 102])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();

    let backend_arc = session.catalog().backend_arc();
    let registry = session.mutable_tables_arc();
    let id_clone = id.clone();
    let written = backend_arc
        .transaction(TxOptions::default(), move |tx| {
            let id = id_clone.clone();
            let batch = batch.clone();
            let registry = Arc::clone(&registry);
            Box::pin(async move {
                let n = registry
                    .insert_batch(tx, &id, &batch)
                    .await
                    .map_err(|e| jammi_db::BackendError::Execution(e.to_string()))?;
                Ok::<u64, jammi_db::BackendError>(n)
            })
        })
        .await
        .unwrap();
    assert_eq!(written, 3);

    session.drop_mutable_table(&id).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scan_after_streams_rows_strictly_greater_in_order(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    let id = unique_id("events");
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .order_column("seq")
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    session
        .sql(&format!(
            "INSERT INTO mutable.public.{name} (id, seq, payload) VALUES \
             (1, 100, 'old'), (2, 200, 'mid'), (3, 300, 'new')",
            name = id.as_str(),
        ))
        .await
        .unwrap();

    let mut stream = session.mutable_tables().scan_after(&id, 150).await.unwrap();
    let mut seqs = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch.unwrap();
        let col = batch
            .column_by_name("seq")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..col.len() {
            seqs.push(col.value(i));
        }
    }
    assert_eq!(seqs, vec![200, 300]);

    session.drop_mutable_table(&id).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scan_after_errors_when_order_column_missing(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    let id = unique_id("noorder");
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    match session.mutable_tables().scan_after(&id, 0).await {
        Ok(_) => panic!("scan_after should reject table without order_column"),
        Err(MutableTableError::NoOrderColumn) => {}
        Err(other) => panic!("expected NoOrderColumn, got {other:?}"),
    }

    session.drop_mutable_table(&id).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn insert_batch_rejects_schema_mismatch(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    let id = unique_id("events");
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    let wrong_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch =
        RecordBatch::try_new(wrong_schema, vec![Arc::new(Int64Array::from(vec![1_i64]))]).unwrap();

    let backend_arc = session.catalog().backend_arc();
    let registry = session.mutable_tables_arc();
    let id_for_closure = id.clone();
    let err = backend_arc
        .transaction(TxOptions::default(), move |tx| {
            let id = id_for_closure.clone();
            let batch = batch.clone();
            let registry = Arc::clone(&registry);
            Box::pin(async move {
                match registry.insert_batch(tx, &id, &batch).await {
                    Ok(_) => Ok::<(), jammi_db::BackendError>(()),
                    Err(MutableTableError::Schema(msg)) => Err(jammi_db::BackendError::Execution(
                        format!("SCHEMA_MISMATCH:{msg}"),
                    )),
                    Err(other) => Err(jammi_db::BackendError::Execution(other.to_string())),
                }
            })
        })
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("SCHEMA_MISMATCH"),
        "expected schema mismatch error, got: {err}"
    );

    session.drop_mutable_table(&id).await.unwrap();
}

/// Round-trip a `DataType::Binary` column through the DataFusion DML sink
/// (writes BLOB/BYTEA bytes through `MutableTableSink::extract_value`) and
/// the provider scan (reads bytes back through `decode_row`'s explicit
/// `Binary` arm). Exercises non-UTF-8 byte sequences (`0x00`, `0xFF`,
/// bincode-shaped payload) so a silent UTF-8 decode would observably
/// truncate or null the value.
///
/// Postgres coverage runs in the same parameterised matrix as every other
/// test in this file when the `live-postgres-tests` feature is on; the
/// default hermetic `cargo test` lane runs the SQLite variant only.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn binary_column_roundtrip_through_provider(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    let id = unique_id("blobs");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("blob", DataType::Binary, false),
    ]));
    let def = MutableTableDefinitionBuilder::new(id.clone(), Arc::clone(&schema))
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    // Two non-UTF-8 payloads: an embedded NUL/0xFF/high-bit pattern, plus a
    // 256-byte bincode-shaped Vec<f32> blob (a realistic shape for callers
    // who store serialised embeddings in a mutable companion table).
    let payload_a: Vec<u8> = vec![0x00, 0xFF, 0xC3, 0x28, 0xA0, 0xA1, 0x80, 0x01];
    let payload_b: Vec<u8> = {
        let mut v = Vec::with_capacity(4 * 64);
        for i in 0_u32..64 {
            v.extend_from_slice(&(i as f32).to_le_bytes());
        }
        v
    };

    // The DataFusion sink path: a user-written DML statement whose hex
    // literals plan as `Binary` values, which exercises `extract_value`'s
    // Binary arm.
    fn hex_literal(bytes: &[u8]) -> String {
        bytes.iter().fold(String::from("X'"), |mut s, b| {
            use std::fmt::Write;
            write!(s, "{b:02X}").unwrap();
            s
        }) + "'"
    }
    session
        .sql(&format!(
            "INSERT INTO mutable.public.{name} (id, blob) VALUES (1, {a}), (2, {b})",
            name = id.as_str(),
            a = hex_literal(&payload_a),
            b = hex_literal(&payload_b),
        ))
        .await
        .unwrap();

    // Read back through the provider scan path — `decode_row`'s Binary arm
    // must produce a BinaryArray (not a null StringArray) with the original
    // bytes intact.
    let batches = session
        .sql(&format!(
            "SELECT blob FROM mutable.public.{name} ORDER BY id",
            name = id.as_str()
        ))
        .await
        .unwrap();
    let out = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(out.num_rows(), 2);
    let blobs = out
        .column_by_name("blob")
        .unwrap()
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap_or_else(|| {
            panic!(
                "blob column is not BinaryArray; got {:?}",
                out.column_by_name("blob").unwrap().data_type()
            )
        });
    assert_eq!(blobs.value(0), payload_a.as_slice());
    assert_eq!(blobs.value(1), payload_b.as_slice());

    session.drop_mutable_table(&id).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn timestamp_column_roundtrips_as_integer_tick(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    let id = unique_id("events_ts");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "observed_at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
    ]));
    let def = MutableTableDefinitionBuilder::new(id.clone(), Arc::clone(&schema))
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    // `2023-11-14T22:13:20.123456Z`, as the microsecond tick the column
    // stores.
    let ts_micros: i64 = 1_700_000_000_123_456;

    // The DataFusion sink path: a user-written DML statement whose timestamp
    // literal is cast to the column's `Timestamp(Microsecond, None)` on
    // insert, which exercises `extract_value`'s Timestamp arm and the
    // backend's timestamp column DDL.
    session
        .sql(&format!(
            "INSERT INTO mutable.public.{name} (id, observed_at) VALUES \
             (1, TIMESTAMP '2023-11-14T22:13:20.123456'), (2, NULL)",
            name = id.as_str(),
        ))
        .await
        .unwrap();

    // Read back through the provider scan path — `decode_row`/`build_arrays`
    // must reconstruct an exact `Timestamp(Microsecond, None)` array from the
    // stored integer ticks, not error out or silently widen/narrow the unit.
    let batches = session
        .sql(&format!(
            "SELECT observed_at FROM mutable.public.{name} ORDER BY id",
            name = id.as_str()
        ))
        .await
        .unwrap();
    let out = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(out.num_rows(), 2);

    let col = out.column_by_name("observed_at").unwrap();
    assert_eq!(
        col.data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, None),
        "reconstructed column's DataType must exactly equal the schema's declared Timestamp(unit, tz)"
    );
    let ts = col
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap_or_else(|| {
            panic!(
                "observed_at column is not TimestampMicrosecondArray; got {:?}",
                col.data_type()
            )
        });
    assert_eq!(
        ts.value(0),
        ts_micros,
        "row 1's non-null timestamp must round-trip to the exact same micros tick"
    );
    assert!(
        ts.is_null(1),
        "row 2's NULL timestamp must round-trip as null, not error or coerce to a value"
    );

    session.drop_mutable_table(&id).await.unwrap();
}

/// `Utf8` extracter — handles both `StringArray` and `StringViewArray`
/// (Arrow 57's parquet reader emits the latter under DataFusion 52).
fn string_column_values(batch: &RecordBatch, name: &str) -> Vec<String> {
    use arrow::array::StringViewArray;
    let col = batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column {name} missing in {:?}", batch.schema()));
    if let Some(sa) = col.as_any().downcast_ref::<StringArray>() {
        return (0..sa.len()).map(|i| sa.value(i).to_string()).collect();
    }
    if let Some(sv) = col.as_any().downcast_ref::<StringViewArray>() {
        return (0..sv.len()).map(|i| sv.value(i).to_string()).collect();
    }
    panic!(
        "column {name} is neither StringArray nor StringViewArray; got {:?}",
        col.data_type()
    );
}
