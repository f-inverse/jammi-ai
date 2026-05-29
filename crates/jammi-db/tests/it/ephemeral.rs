//! Integration tests for the ephemeral session-storage primitive (spec J6).
//!
//! Exercises the success criteria end-to-end against a real SQLite-backed
//! session: a working session context, session-scoped table visibility,
//! deletion on close, the `closed` lifecycle event (with deleted-row count),
//! timeout force-close, tenant isolation, and the audit-record-references-hash
//! integration pattern.

use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::ephemeral::{
    self, ActiveSessions, EphemeralSession, SessionLifecycleEvent, SessionLifecycleRecord,
    SESSION_LIFECYCLE_TOPIC,
};
use jammi_db::session::JammiSession;
use jammi_db::trigger::Predicate;
use jammi_test_utils::make_test_session;
use uuid::Uuid;

const TENANT_A: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a";
const TENANT_B: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8eff";

async fn session() -> Arc<JammiSession> {
    let dir = tempfile::tempdir().expect("tempdir");
    let s = make_test_session(BackendKind::Sqlite, dir.path())
        .await
        .expect("sqlite session");
    std::mem::forget(dir);
    Arc::new(s)
}

/// `(image_id VARCHAR, image_hash VARCHAR)` — the J6 motivating shape.
fn images_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("image_id", DataType::Utf8, false),
        Field::new("image_hash", DataType::Utf8, false),
    ]))
}

fn images_batch(rows: &[(&str, &str)]) -> arrow::array::RecordBatch {
    let ids: Vec<&str> = rows.iter().map(|(a, _)| *a).collect();
    let hashes: Vec<&str> = rows.iter().map(|(_, b)| *b).collect();
    arrow::array::RecordBatch::try_new(
        images_schema(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(hashes)),
        ],
    )
    .expect("images batch")
}

/// Criterion 1 + 2: a working session context whose tables are reachable by it.
#[tokio::test]
async fn open_create_insert_query() {
    let s = session().await;
    s.bind_tenant(TENANT_A.parse().unwrap());

    let mut ephem = EphemeralSession::open(
        Arc::clone(&s),
        Duration::from_secs(3600),
        ActiveSessions::new(),
    )
    .await
    .expect("open");

    ephem
        .create_ephemeral_table(
            "query_images",
            images_schema(),
            vec!["image_id".to_string()],
        )
        .await
        .expect("create");
    let n = ephem
        .insert(
            "query_images",
            images_batch(&[("img-1", "sha256:aaa"), ("img-2", "sha256:bbb")]),
        )
        .await
        .expect("insert");
    assert_eq!(n, 2);

    let batches = ephem
        .sql(
            "query_images",
            "SELECT image_hash FROM {table} ORDER BY image_id",
        )
        .await
        .expect("sql");
    let hashes: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            let col = b
                .column_by_name("image_hash")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            (0..b.num_rows())
                .map(|i| col.value(i).to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(hashes, vec!["sha256:aaa", "sha256:bbb"]);
    assert_eq!(ephem.count_rows("query_images").await.unwrap(), 2);
}

/// Criterion 3 + 4: close drops the table, and a `closed` event lands on the
/// lifecycle topic carrying the deleted-row count.
#[tokio::test]
async fn close_drops_tables_and_emits_closed_event() {
    let s = session().await;
    s.bind_tenant(TENANT_A.parse().unwrap());
    let tenant = TENANT_A.parse().unwrap();

    let mut ephem = EphemeralSession::open(
        Arc::clone(&s),
        Duration::from_secs(3600),
        ActiveSessions::new(),
    )
    .await
    .expect("open");
    ephem
        .create_ephemeral_table("imgs", images_schema(), vec!["image_id".to_string()])
        .await
        .unwrap();
    ephem
        .insert(
            "imgs",
            images_batch(&[("a", "h-a"), ("b", "h-b"), ("c", "h-c")]),
        )
        .await
        .unwrap();
    let session_id = ephem.session_id();
    let phys = ephem.physical_table_id("imgs").unwrap().clone();

    // Subscribe to the lifecycle topic (registered by the `opened` event on open).
    let topic = s
        .topic_repo()
        .lookup_by_name(SESSION_LIFECYCLE_TOPIC, Some(tenant))
        .await
        .unwrap()
        .expect("lifecycle topic registered on open");
    let mut sub = s
        .trigger_broker()
        .subscribe(topic.id, Predicate::match_all(), None)
        .await
        .unwrap();

    ephem.close().await.expect("close");

    // Criterion 3: the physical table is gone from the SQL surface.
    let err = s
        .sql(&format!(
            "SELECT * FROM mutable.public.\"{}\"",
            phys.as_str()
        ))
        .await;
    assert!(
        err.is_err(),
        "ephemeral table should be dropped after close"
    );

    // Criterion 4: a `closed` lifecycle event with deleted_row_count == 3.
    let record = next_lifecycle_for(&mut sub, session_id, SessionLifecycleEvent::Closed).await;
    assert_eq!(record.event, SessionLifecycleEvent::Closed);
    assert_eq!(record.tenant_id, TENANT_A);
    assert_eq!(record.deleted_row_count, 3);
}

/// Criterion 5: a session past its timeout is force-closed by the scanner with
/// a `timed_out` event, and its tables are deleted.
#[tokio::test]
async fn timeout_scanner_force_closes() {
    let s = session().await;
    s.bind_tenant(TENANT_A.parse().unwrap());
    let tenant = TENANT_A.parse().unwrap();
    let active = ActiveSessions::new();

    // Zero timeout: the session is immediately past its deadline.
    let mut ephem = EphemeralSession::open(Arc::clone(&s), Duration::from_secs(0), active.clone())
        .await
        .expect("open");
    ephem
        .create_ephemeral_table("imgs", images_schema(), vec!["image_id".to_string()])
        .await
        .unwrap();
    ephem
        .insert("imgs", images_batch(&[("a", "h-a")]))
        .await
        .unwrap();
    let session_id = ephem.session_id();
    let phys = ephem.physical_table_id("imgs").unwrap().clone();
    assert!(ephem.is_expired(chrono::Utc::now()));
    // Forget the handle so its Drop does not race the scanner — the scanner is
    // the path under test here.
    std::mem::forget(ephem);
    assert_eq!(active.len(), 1);

    let topic = s
        .topic_repo()
        .lookup_by_name(SESSION_LIFECYCLE_TOPIC, Some(tenant))
        .await
        .unwrap()
        .expect("lifecycle topic registered on open");
    let mut sub = s
        .trigger_broker()
        .subscribe(topic.id, Predicate::match_all(), None)
        .await
        .unwrap();

    // One manual scan stands in for a scanner tick.
    ephemeral::scan(&s, &active).await;

    assert_eq!(active.len(), 0, "expired session deregistered after scan");
    let gone = s
        .sql(&format!(
            "SELECT * FROM mutable.public.\"{}\"",
            phys.as_str()
        ))
        .await;
    assert!(gone.is_err(), "table dropped by timeout scan");

    let record = next_lifecycle_for(&mut sub, session_id, SessionLifecycleEvent::TimedOut).await;
    assert_eq!(record.event, SessionLifecycleEvent::TimedOut);
    assert_eq!(record.deleted_row_count, 1);
}

/// Criterion 6: a persistent table can store a hash that an ephemeral table
/// also held, and survives the ephemeral session's deletion — the persistent
/// record references the hash, not the deleted data.
#[tokio::test]
async fn persistent_record_references_hash_after_deletion() {
    use jammi_db::store::mutable::{MutableTableDefinitionBuilder, MutableTableId};
    let s = session().await;
    s.bind_tenant(TENANT_A.parse().unwrap());

    // Persistent companion table holding just the hash lineage.
    let lineage_id = MutableTableId::new("query_lineage").unwrap();
    let lineage_schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "image_hash",
        DataType::Utf8,
        false,
    )]));
    let def = MutableTableDefinitionBuilder::new(lineage_id, Arc::clone(&lineage_schema))
        .primary_key(vec!["image_hash".to_string()])
        .tenant(Some(TENANT_A.parse().unwrap()))
        .build()
        .unwrap();
    s.create_mutable_table(def).await.unwrap();

    let hash = "sha256:deadbeef";
    let mut ephem = EphemeralSession::open(
        Arc::clone(&s),
        Duration::from_secs(3600),
        ActiveSessions::new(),
    )
    .await
    .unwrap();
    ephem
        .create_ephemeral_table("imgs", images_schema(), vec!["image_id".to_string()])
        .await
        .unwrap();
    ephem
        .insert("imgs", images_batch(&[("img-1", hash)]))
        .await
        .unwrap();

    // Write the hash to the persistent table BEFORE closing the ephemeral one.
    s.sql(&format!(
        "INSERT INTO mutable.public.query_lineage (image_hash) VALUES ('{hash}')"
    ))
    .await
    .unwrap();

    ephem.close().await.unwrap();

    // The persistent record survives and still references the hash.
    let batches = s
        .sql("SELECT image_hash FROM mutable.public.query_lineage")
        .await
        .unwrap();
    let stored: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            let c = b
                .column_by_name("image_hash")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            (0..b.num_rows())
                .map(|i| c.value(i).to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(stored, vec![hash.to_string()]);
}

/// Criterion 7: tenant B cannot see tenant A's ephemeral data.
#[tokio::test]
async fn tenant_isolation() {
    let s = session().await;

    // Tenant A opens a session and stores a row.
    s.bind_tenant(TENANT_A.parse().unwrap());
    let mut ephem = EphemeralSession::open(
        Arc::clone(&s),
        Duration::from_secs(3600),
        ActiveSessions::new(),
    )
    .await
    .unwrap();
    ephem
        .create_ephemeral_table("imgs", images_schema(), vec!["image_id".to_string()])
        .await
        .unwrap();
    ephem
        .insert("imgs", images_batch(&[("a", "h-a")]))
        .await
        .unwrap();
    let phys = ephem.physical_table_id("imgs").unwrap().clone();

    // Under tenant A the row is visible.
    let a = s
        .sql(&format!(
            "SELECT * FROM mutable.public.\"{}\"",
            phys.as_str()
        ))
        .await
        .unwrap();
    assert_eq!(a.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

    // Switch to tenant B: the analyzer scopes the row out.
    s.bind_tenant(TENANT_B.parse().unwrap());
    let b = s
        .sql(&format!(
            "SELECT * FROM mutable.public.\"{}\"",
            phys.as_str()
        ))
        .await
        .unwrap();
    assert_eq!(
        b.iter().map(|b| b.num_rows()).sum::<usize>(),
        0,
        "tenant B sees zero of tenant A's ephemeral rows"
    );

    // Restore tenant A so the close runs under the right scope.
    s.bind_tenant(TENANT_A.parse().unwrap());
    ephem.close().await.unwrap();
}

/// An ephemeral session cannot be opened without a bound tenant.
#[tokio::test]
async fn open_requires_tenant_binding() {
    let s = session().await;
    // `EphemeralSession` (the Ok type) is not `Debug`, so `unwrap_err` is
    // unavailable; match the result directly instead.
    let result = EphemeralSession::open(
        Arc::clone(&s),
        Duration::from_secs(60),
        ActiveSessions::new(),
    )
    .await;
    match result {
        Err(ephemeral::EphemeralError::NoTenantBinding) => {}
        Err(other) => panic!("expected NoTenantBinding, got {other:?}"),
        Ok(_) => panic!("expected open to fail without a bound tenant"),
    }
}

/// Pull lifecycle records off a subscription until one matches `session_id` and
/// `want`, decoding the JSON `record` column. Bounded by a timeout so a missing
/// event fails the test rather than hanging.
async fn next_lifecycle_for(
    sub: &mut (impl StreamExt<
        Item = Result<jammi_db::trigger::DeliveredBatch, jammi_db::trigger::TriggerError>,
    > + Unpin),
    session_id: Uuid,
    want: SessionLifecycleEvent,
) -> SessionLifecycleRecord {
    let deadline = std::time::Duration::from_secs(5);
    loop {
        let delivered = tokio::time::timeout(deadline, sub.next())
            .await
            .expect("delivery within timeout")
            .expect("a batch")
            .expect("ok batch");
        let col = delivered
            .batch
            .column_by_name("record")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .expect("record column");
        for i in 0..col.len() {
            let rec: SessionLifecycleRecord = serde_json::from_str(col.value(i)).unwrap();
            if rec.session_id == session_id && rec.event == want {
                return rec;
            }
        }
    }
}
