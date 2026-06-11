//! Integration tests for per-query eval persistence (spec J9).
//!
//! Exercises the `_jammi_eval_per_query` catalog table end-to-end against a
//! real SQLite-backed catalog: bulk write + read-back, tenant isolation
//! (tenant A never sees tenant B's rows), verbatim cohort carry-through, and
//! the additive guarantee that the aggregate `eval_runs` path is untouched.

use std::str::FromStr;

use jammi_db::catalog::eval_repo::{EvalRunRecord, PerQueryEvalRecord};
use jammi_db::catalog::Catalog;
use jammi_db::tenant::{TenantContext, TenantId};
use jammi_db::tenant_scope::TenantBinding;
use tempfile::tempdir;

const TENANT_A: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a";
const TENANT_B: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8eff";

fn scoped(t: &str) -> TenantBinding {
    let b = TenantBinding::unscoped();
    b.set_shared(TenantContext::Scoped(TenantId::from_str(t).unwrap()));
    b
}

fn record(run: &str, query: &str, cohorts: &str, metrics: &str) -> PerQueryEvalRecord {
    PerQueryEvalRecord {
        eval_run_id: run.to_string(),
        query_id: query.to_string(),
        cohorts_json: cohorts.to_string(),
        metrics_json: metrics.to_string(),
    }
}

/// Bulk-insert several per-query rows, then read them back ordered by
/// `query_id`. Metrics and cohort JSON round-trip verbatim.
#[tokio::test]
async fn record_then_get_round_trips_per_query() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open_with_tenant(dir.path(), Some(scoped(TENANT_A)))
        .await
        .unwrap();

    let rows = vec![
        record(
            "run-1",
            "q-b",
            r#"{"split":"val"}"#,
            r#"{"recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"recall@10":1.0,"mrr":1.0,"ndcg":1.0,"distance":0.12}"#,
        ),
        record(
            "run-1",
            "q-a",
            "{}",
            r#"{"recall@1":0.0,"recall@3":0.5,"recall@5":0.5,"recall@10":1.0,"mrr":0.5,"ndcg":0.4,"distance":0.9}"#,
        ),
    ];
    catalog.record_eval_per_query(&rows).await.unwrap();

    let got = catalog.get_eval_per_query("run-1").await.unwrap();
    assert_eq!(got.len(), 2);
    // Ordered by query_id ascending: q-a then q-b.
    assert_eq!(got[0].query_id, "q-a");
    assert_eq!(got[0].cohorts_json, "{}");
    assert_eq!(got[1].query_id, "q-b");
    assert_eq!(got[1].cohorts_json, r#"{"split":"val"}"#);

    // Metrics JSON round-trips to the same numeric vector.
    let m: serde_json::Value = serde_json::from_str(&got[1].metrics_json).unwrap();
    assert_eq!(m["recall@1"], 1.0);
    assert_eq!(m["recall@10"], 1.0);
    assert_eq!(m["mrr"], 1.0);
    assert_eq!(m["distance"], 0.12);

    // An empty input is a no-op (no panic, nothing written for run-2).
    catalog.record_eval_per_query(&[]).await.unwrap();
    assert!(catalog
        .get_eval_per_query("run-2")
        .await
        .unwrap()
        .is_empty());
}

/// Tenant A writes per-query rows; tenant B (same database) sees none of them,
/// and vice versa. The read predicate is `tenant_id = $me OR IS NULL`.
#[tokio::test]
async fn per_query_rows_are_tenant_isolated() {
    let dir = tempdir().unwrap();

    let cat_a = Catalog::open_with_tenant(dir.path(), Some(scoped(TENANT_A)))
        .await
        .unwrap();
    let cat_b = Catalog::open_with_tenant(dir.path(), Some(scoped(TENANT_B)))
        .await
        .unwrap();

    cat_a
        .record_eval_per_query(&[record("run-shared", "q-a", "{}", r#"{"recall@1":1.0}"#)])
        .await
        .unwrap();
    cat_b
        .record_eval_per_query(&[record("run-shared", "q-b", "{}", r#"{"recall@1":0.0}"#)])
        .await
        .unwrap();

    let a_rows = cat_a.get_eval_per_query("run-shared").await.unwrap();
    assert_eq!(a_rows.len(), 1, "tenant A sees only its own row");
    assert_eq!(a_rows[0].query_id, "q-a");

    let b_rows = cat_b.get_eval_per_query("run-shared").await.unwrap();
    assert_eq!(b_rows.len(), 1, "tenant B sees only its own row");
    assert_eq!(b_rows[0].query_id, "q-b");
}

/// The per-query table is additive: writing per-query rows leaves the
/// aggregate `eval_runs` read path behaving exactly as before. A run recorded
/// with `record_eval_run` reads back identically whether or not per-query rows
/// also exist for it.
#[tokio::test]
async fn aggregate_path_unaffected_by_per_query_rows() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();

    // Register the FK target model, then record an aggregate run exactly as
    // the runner does.
    catalog
        .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
            model_id: "model-x",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: jammi_db::ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();

    let agg = EvalRunRecord {
        eval_run_id: "run-agg".into(),
        eval_type: "embedding".into(),
        model_id: Some("model-x::1".into()),
        source_id: "src".into(),
        golden_source: "golden".into(),
        k: Some(10),
        metrics_json: r#"{"recall_at_k":0.7}"#.into(),
        status: "completed".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
    };
    catalog.record_eval_run(&agg).await.unwrap();

    // Snapshot the aggregate row before any per-query writes.
    let before = catalog.get_eval_run("run-agg").await.unwrap().unwrap();

    catalog
        .record_eval_per_query(&[record(
            "run-agg",
            "q-1",
            "{}",
            r#"{"recall@1":1.0,"mrr":1.0,"ndcg":1.0,"distance":0.1}"#,
        )])
        .await
        .unwrap();

    let after = catalog.get_eval_run("run-agg").await.unwrap().unwrap();
    assert_eq!(before.eval_run_id, after.eval_run_id);
    assert_eq!(before.metrics_json, after.metrics_json);
    assert_eq!(before.k, after.k);
    assert_eq!(before.status, after.status);
    // And the per-query rows are independently readable.
    assert_eq!(
        catalog.get_eval_per_query("run-agg").await.unwrap().len(),
        1
    );
}
