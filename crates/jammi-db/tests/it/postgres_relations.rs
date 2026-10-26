//! A Postgres source serves every relation of every schema its role may use:
//! tables, views and materialized views, each under its own schema, and
//! lists them `public` first.

use arrow::array::Int64Array;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::session::JammiSession;
use jammi_db::source::{SourceDefinition, SourceTable};
use jammi_test_utils::{make_test_session, unique_suffix};
use sqlx::{Connection, Executor, PgConnection};
use tempfile::{tempdir, TempDir};

async fn session() -> (JammiSession, String, [TempDir; 2]) {
    let source_dir = tempdir().unwrap();
    let url = jammi_test_utils::postgres_database_url(source_dir.path()).await;
    let mut db = PgConnection::connect(&url).await.unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1), (2); \
         CREATE VIEW v AS SELECT id FROM t; \
         CREATE MATERIALIZED VIEW mv AS SELECT id FROM t; \
         CREATE SCHEMA sales; CREATE TABLE sales.orders (id INTEGER); \
         INSERT INTO sales.orders VALUES (1), (2), (3); \
         CREATE SCHEMA \"Analytics\"; CREATE TABLE \"Analytics\".\"Daily Totals\" (id INTEGER); \
         INSERT INTO \"Analytics\".\"Daily Totals\" VALUES (7);",
    )
    .await
    .unwrap();
    db.close().await.unwrap();
    let dir = tempdir().unwrap();
    let session = make_test_session(BackendKind::Sqlite, dir.path()).await;
    let source = format!("pg_{}", unique_suffix());
    let definition = SourceDefinition::from_url(url.as_str(), None).unwrap();
    session
        .add_source(&source, definition.source_type, definition.connection)
        .await
        .unwrap();
    (session, source, [source_dir, dir])
}

async fn count(session: &JammiSession, relation: &str) -> i64 {
    let batches = session
        .sql(&format!("SELECT COUNT(*) AS n FROM {relation}"))
        .await
        .unwrap_or_else(|e| panic!("{relation}: {e:?}"));
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

#[tokio::test]
async fn every_relation_of_every_schema_answers() {
    let (session, source, _dirs) = session().await;
    for (relation, rows) in [
        ("public.t", 2),
        ("public.v", 2),
        ("public.mv", 2),
        ("sales.orders", 3),
        (r#""Analytics"."Daily Totals""#, 1),
    ] {
        assert_eq!(count(&session, &format!("{source}.{relation}")).await, rows, "{relation}");
    }
    let filtered = session
        .sql(&format!("SELECT id FROM {source}.sales.orders WHERE id >= 2 ORDER BY id"))
        .await
        .unwrap();
    let ids = filtered[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(ids, [2, 3]);
}

#[tokio::test]
async fn a_source_lists_its_tables_public_first() {
    let (session, source, _dirs) = session().await;
    let tables = session.source_tables(&source).await.unwrap();
    let table = |schema: &str, name: &str| SourceTable::new(&source, schema, name);
    assert_eq!(
        tables,
        [
            table("public", "mv"),
            table("public", "t"),
            table("public", "v"),
            table("Analytics", "Daily Totals"),
            table("sales", "orders"),
        ]
    );
    assert_eq!(
        table("Analytics", "Daily Totals").to_string(),
        format!(r#""{source}"."Analytics"."Daily Totals""#)
    );
}
