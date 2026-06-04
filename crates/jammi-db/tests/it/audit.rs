//! Integration tests for the per-query audit primitive (spec J2).
//!
//! Exercises the success criteria end-to-end against a real SQLite-backed
//! session: signed writes, signature verification, tenant isolation, the
//! lineage size cap, SQL visibility, typed fetch, and trigger publication.

use std::sync::OnceLock;

use futures::StreamExt;
use jammi_db::audit::{
    self, EnvSigningKeyStore, PerQueryAudit, AUDIT_TABLE_NAME, AUDIT_TOPIC, MASTER_KEY_ENV,
};
use jammi_db::catalog::backend::BackendKind;
use jammi_db::session::JammiSession;
use jammi_db::trigger::Predicate;
use jammi_test_utils::make_test_session;
use tokio::sync::Mutex;
use uuid::Uuid;

const TENANT_A: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a";
const TENANT_B: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8eff";
const TEST_KEY: &str = "0000000000000000000000000000000000000000000000000000000000000001";

// The audit env vars (`JAMMI_AUDIT_MASTER_KEY`, the lineage cap) are
// process-global. An async-aware mutex serializes the tests that mutate them so
// the guard can be held across `.await` without tripping
// `clippy::await_holding_lock`, and a panicking test does not poison the lock
// for the rest of the suite.
fn env_lock() -> &'static Mutex<()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}

fn set_master_key() {
    std::env::set_var(MASTER_KEY_ENV, TEST_KEY);
}

async fn session() -> JammiSession {
    let dir = tempfile::tempdir().expect("tempdir");
    let s = make_test_session(BackendKind::Sqlite, dir.path())
        .await
        .expect("sqlite session");
    // Keep the catalog dir alive for the process; the harness exits cleanly.
    std::mem::forget(dir);
    s
}

fn sample(model: &str) -> PerQueryAudit {
    PerQueryAudit::new(
        Uuid::now_v7(),
        model,
        "rev-1",
        serde_json::json!({ "image_hashes": ["sha256:abc"], "examiner_id": "42" }),
        vec!["doc-1".to_string(), "doc-2".to_string()],
        vec![0.92, 0.88],
    )
    .expect("valid record")
}

#[tokio::test]
async fn log_then_fetch_and_signature_verifies() {
    let _g = env_lock().lock().await;
    set_master_key();
    let s = session().await.with_tenant(TENANT_A.parse().unwrap());

    let rec = sample("test/model");
    let qid = rec.query_id;
    s.audit().log(vec![rec]).await.expect("log");

    // Criterion 6: typed fetch by query id.
    let fetched = s
        .audit()
        .fetch_by_query_id(qid)
        .await
        .expect("fetch")
        .expect("present");
    assert_eq!(fetched.query_id, qid);
    assert_eq!(fetched.tenant_id.as_deref(), Some(TENANT_A));
    assert_eq!(fetched.model_id, "test/model");
    assert_eq!(fetched.top_k_result_ids, vec!["doc-1", "doc-2"]);

    // Criterion 1 + 3: the row carries a signature and it verifies.
    assert!(!fetched.signature.is_empty());
    audit::verify_with_store(&fetched, &EnvSigningKeyStore).expect("signature verifies");

    // fetch_recent returns the same record.
    let recent = s.audit().fetch_recent(10).await.expect("recent");
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0].query_id, qid);
}

#[tokio::test]
async fn tenant_isolation() {
    let _g = env_lock().lock().await;
    set_master_key();
    let s = session().await;

    // Tenant A writes two records.
    s.bind_tenant(TENANT_A.parse().unwrap());
    s.audit().log(vec![sample("m")]).await.unwrap();
    s.audit().log(vec![sample("m")]).await.unwrap();

    // Criterion 4: tenant A sees its own rows.
    let a_rows = s.audit().fetch_recent(100).await.unwrap();
    assert_eq!(a_rows.len(), 2, "tenant A sees its own rows");

    // Criterion 4: tenant B sees zero of tenant A's rows.
    s.bind_tenant(TENANT_B.parse().unwrap());
    let b_rows = s.audit().fetch_recent(100).await.unwrap();
    assert_eq!(b_rows.len(), 0, "tenant B sees zero of tenant A's rows");
}

#[tokio::test]
async fn two_tenants_can_both_log() {
    // Regression for the per-tenant audit-topic uniqueness defect: the
    // `topics` catalog table used to enforce a *global* `UNIQUE(name)`, so the
    // first tenant to `log` claimed `jammi.audit.search.v1` process-wide and
    // the SECOND tenant's first `log` crashed with
    // `UNIQUE constraint failed: topics.name`. Both tenants must succeed, and
    // each must see only its own audit records (delivered isolation preserved).
    let _g = env_lock().lock().await;
    set_master_key();
    let s = session().await;

    // Tenant A's first log registers its own `jammi.audit.search.v1` topic.
    let a_tenant = TENANT_A.parse().unwrap();
    s.bind_tenant(a_tenant);
    let a_rec = sample("model/a");
    let a_qid = a_rec.query_id;
    s.audit().log(vec![a_rec]).await.expect("tenant A log");

    // Tenant B's first log must ALSO register `jammi.audit.search.v1` — under
    // the old global unique this panicked; under `UNIQUE(name, tenant_id)` the
    // two per-tenant topics coexist.
    let b_tenant = TENANT_B.parse().unwrap();
    s.bind_tenant(b_tenant);
    let b_rec = sample("model/b");
    let b_qid = b_rec.query_id;
    s.audit()
        .log(vec![b_rec])
        .await
        .expect("tenant B log must succeed (second-tenant audit-topic register)");

    // Each tenant registered its own distinct, tenant-pinned audit topic.
    let a_topic = s
        .topic_repo()
        .lookup_by_name(AUDIT_TOPIC, Some(a_tenant))
        .await
        .unwrap()
        .expect("tenant A audit topic");
    let b_topic = s
        .topic_repo()
        .lookup_by_name(AUDIT_TOPIC, Some(b_tenant))
        .await
        .unwrap()
        .expect("tenant B audit topic");
    assert_eq!(a_topic.tenant, Some(a_tenant));
    assert_eq!(b_topic.tenant, Some(b_tenant));
    assert_ne!(
        a_topic.id, b_topic.id,
        "each tenant owns a distinct audit topic"
    );

    // Stored-data isolation: tenant B sees only its own record, not A's.
    s.bind_tenant(b_tenant);
    let b_rows = s.audit().fetch_recent(100).await.unwrap();
    assert_eq!(b_rows.len(), 1, "tenant B sees only its own audit row");
    assert_eq!(b_rows[0].query_id, b_qid);
    assert_eq!(b_rows[0].tenant_id.as_deref(), Some(TENANT_B));

    // ...and tenant A still sees only its own record, not B's.
    s.bind_tenant(a_tenant);
    let a_rows = s.audit().fetch_recent(100).await.unwrap();
    assert_eq!(a_rows.len(), 1, "tenant A sees only its own audit row");
    assert_eq!(a_rows[0].query_id, a_qid);
    assert_eq!(a_rows[0].tenant_id.as_deref(), Some(TENANT_A));
}

#[tokio::test]
async fn raw_sql_is_tenant_scoped() {
    let _g = env_lock().lock().await;
    set_master_key();
    let s = session().await;

    s.bind_tenant(TENANT_A.parse().unwrap());
    s.audit().log(vec![sample("m")]).await.unwrap();

    // Criterion 5: SELECT * via the SQL surface returns the calling tenant's
    // rows; the other tenant sees none.
    let sql = format!("SELECT * FROM mutable.public.\"{AUDIT_TABLE_NAME}\"");
    let a = s.sql(&sql).await.unwrap();
    let a_rows: usize = a.iter().map(|b| b.num_rows()).sum();
    assert_eq!(a_rows, 1);

    s.bind_tenant(TENANT_B.parse().unwrap());
    let b = s.sql(&sql).await.unwrap();
    let b_rows: usize = b.iter().map(|b| b.num_rows()).sum();
    assert_eq!(b_rows, 0);
}

#[tokio::test]
async fn lineage_cap_enforced() {
    let _g = env_lock().lock().await;
    set_master_key();
    std::env::set_var(audit::MAX_LINEAGE_BYTES_ENV, "64");
    let s = session().await.with_tenant(TENANT_A.parse().unwrap());

    let big = "x".repeat(200);
    let rec = PerQueryAudit::new(
        Uuid::now_v7(),
        "m",
        "v",
        serde_json::json!({ "blob": big }),
        vec!["d".to_string()],
        vec![0.5],
    )
    .unwrap();

    // Criterion 2: oversized lineage is rejected by construction.
    let err = s.audit().log(vec![rec]).await.unwrap_err();
    assert!(matches!(err, audit::AuditError::LineageTooLarge { .. }));
    std::env::remove_var(audit::MAX_LINEAGE_BYTES_ENV);
}

#[tokio::test]
async fn log_requires_tenant_binding() {
    let _g = env_lock().lock().await;
    set_master_key();
    let s = session().await;
    let err = s.audit().log(vec![sample("m")]).await.unwrap_err();
    assert!(matches!(err, audit::AuditError::NoTenantBinding));
}

#[tokio::test]
async fn master_key_missing_is_fatal_for_writes() {
    let _g = env_lock().lock().await;
    std::env::remove_var(MASTER_KEY_ENV);
    let s = session().await.with_tenant(TENANT_A.parse().unwrap());
    // Criterion 8 (data path): with no master key, signing — and thus the log
    // call — fails. `audit::ensure_master_key_present` is the startup check a
    // server can call to fail fast on the same condition, asserted here too.
    let err = s.audit().log(vec![sample("m")]).await.unwrap_err();
    assert!(matches!(err, audit::AuditError::MasterKey(_)));
    assert!(audit::ensure_master_key_present(&EnvSigningKeyStore).is_err());
}

#[tokio::test]
async fn published_to_trigger_topic() {
    let _g = env_lock().lock().await;
    set_master_key();
    let s = session().await.with_tenant(TENANT_A.parse().unwrap());

    // The first log registers the audit topic (via the catalog topic repo) and
    // provisions its backing table. Look it up so we subscribe to the exact id
    // the writer publishes to.
    s.audit().log(vec![sample("first")]).await.unwrap();
    let topic = s
        .topic_repo()
        .lookup_by_name(AUDIT_TOPIC, Some(TENANT_A.parse().unwrap()))
        .await
        .unwrap()
        .expect("audit topic registered after first log");

    // Subscribe for live fan-out, then log a second record.
    let mut sub = s
        .trigger_broker()
        .subscribe(topic.id, Predicate::match_all(), None)
        .await
        .unwrap();

    let rec = sample("second");
    let qid = rec.query_id;
    s.audit().log(vec![rec]).await.unwrap();

    // Criterion 7: a subscriber receives the JSON payload.
    let delivered = tokio::time::timeout(std::time::Duration::from_secs(5), sub.next())
        .await
        .expect("delivery within timeout")
        .expect("a batch")
        .expect("ok batch");
    // The publisher prepends engine-controlled columns (_offset/_row_idx/
    // _produced_at) to the topic payload; the audit JSON is in `record`.
    let col = delivered
        .batch
        .column_by_name("record")
        .and_then(|c| c.as_any().downcast_ref::<arrow::array::StringArray>())
        .expect("record column");
    let payload: PerQueryAudit = serde_json::from_str(col.value(0)).unwrap();
    assert_eq!(payload.query_id, qid);
    assert_eq!(payload.tenant_id.as_deref(), Some(TENANT_A));
    audit::verify_with_store(&payload, &EnvSigningKeyStore).expect("published payload verifies");
}

#[tokio::test]
async fn reserved_table_name_rejected_for_users() {
    let s = session().await;
    use jammi_db::store::mutable::{MutableTableDefinitionBuilder, MutableTableId};
    let id = MutableTableId::new(AUDIT_TABLE_NAME).unwrap();
    let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "query_id",
        arrow_schema::DataType::Utf8,
        false,
    )]));
    let def = MutableTableDefinitionBuilder::new(id, schema)
        .primary_key(vec!["query_id".to_string()])
        .build()
        .unwrap();
    let err = s.create_mutable_table(def).await.unwrap_err();
    assert!(
        err.to_string().contains("reserved"),
        "expected reserved-name rejection, got: {err}"
    );
}

/// Bench the bulk-insert log path at batch sizes 10/100/1000. `#[ignore]`d by
/// default (timing, not a correctness gate). Run explicitly with:
///   cargo test -p jammi-db --test it --release -- --ignored --nocapture bench_bulk_insert
#[tokio::test]
#[ignore = "timing bench; run explicitly with --ignored --nocapture"]
async fn bench_bulk_insert() {
    let _g = env_lock().lock().await;
    set_master_key();
    let s = session().await.with_tenant(TENANT_A.parse().unwrap());

    for &n in &[10usize, 100, 1000] {
        let records: Vec<PerQueryAudit> = (0..n).map(|_| sample("bench/model")).collect();
        let start = std::time::Instant::now();
        s.audit().log(records).await.expect("bulk log");
        let elapsed = start.elapsed();
        let per = elapsed.as_secs_f64() * 1000.0 / n as f64;
        println!(
            "audit bulk-insert: batch={n:>4}  total={:>8.2}ms  per_record={per:.4}ms",
            elapsed.as_secs_f64() * 1000.0
        );
    }
}
