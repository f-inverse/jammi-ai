//! Property 4 — cross-tenant isolation across processes.
//!
//! Two tenants' synthetic data and jobs share one catalog and one object store,
//! drained by the same worker fleet. Each job is submitted under its tenant's
//! scope, so it carries that `tenant_id`; the worker re-scopes the catalog to
//! the job's tenant for the whole run (the unscoped claim is intentional — one
//! worker drains every tenant's queue). The proof is that each completed model
//! is stamped to the correct tenant and is visible ONLY in that tenant's scope:
//! a worker running tenant-A's job cannot leak tenant-A's model into tenant-B's
//! view, and vice versa. A single worker drains the mixed queue, so the same
//! process runs both tenants' jobs under different scopes back-to-back.

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_db::TenantId;

use crate::harness::{self, Backends, Fleet, JobSize};

const TEST: &str = "cross_tenant_isolation";

/// Submit one tenant's job under its scope: register a tenant-private source and
/// submit a fine-tune over it, all inside `with_tenant_scoped` so both the
/// source rows and the queued job carry `tenant`. Returns `(job_id, model_id)`.
async fn submit_for_tenant(
    session: &Arc<InferenceSession>,
    tenant: TenantId,
    source: &str,
) -> (String, String) {
    session
        .with_tenant_scoped(tenant, |_scope| {
            let session = Arc::clone(session);
            let source = source.to_string();
            async move {
                harness::register_training_source(&session, &source).await;
                harness::submit_fine_tune(&session, &source, JobSize::Quick).await
            }
        })
        .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_tenant_queue_completes_each_under_its_own_scope() {
    let Some(backends) = Backends::from_env_or_skip(TEST) else {
        return;
    };
    let result_root = backends.unique_result_root(TEST);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;

    // Two distinct tenants (non-nil UUIDs).
    let tenant_a = TenantId::from_uuid(uuid::Uuid::new_v4()).unwrap();
    let tenant_b = TenantId::from_uuid(uuid::Uuid::new_v4()).unwrap();

    // One job per tenant, each over its own tenant-private source. The source
    // names are per-test-unique: the catalog's `sources` table keys on a global
    // `source_id` PRIMARY KEY (the tenant scope governs visibility, not key
    // uniqueness), so a fixed name would collide with a prior run's row on the
    // shared persistent catalog even across distinct tenants.
    let source_a = harness::unique_source_name(&format!("{TEST}-a"));
    let source_b = harness::unique_source_name(&format!("{TEST}-b"));
    let (job_a, model_a) = submit_for_tenant(&session, tenant_a, &source_a).await;
    let (job_b, model_b) = submit_for_tenant(&session, tenant_b, &source_b).await;

    // A SINGLE worker drains the mixed queue, running each job under its own
    // tenant scope back-to-back — the cross-tenant scenario on one process.
    let fleet = Fleet::spawn(&backends, &result_root, 1);

    // Both jobs reach `completed`, each stamped with its submitting tenant.
    for (job_id, tenant) in [(&job_a, tenant_a), (&job_b, tenant_b)] {
        let record = harness::poll_until(harness::TERMINAL_TIMEOUT, harness::POLL_INTERVAL, || {
            let session = &session;
            async move {
                let r = session.catalog().get_training_job(job_id).await.ok()?;
                (r.status == "completed").then_some(r)
            }
        })
        .await
        .unwrap_or_else(|| panic!("job {job_id} completes"));
        assert_eq!(
            record.tenant_id,
            Some(tenant),
            "the completed job is stamped with its submitting tenant"
        );
    }

    // Tenant isolation on the OUTPUT models: each tenant's model is visible in
    // its own scope and NOT in the other's. `pinned_to_tenant` reads see
    // `tenant_id = t ∪ NULL`, so a model stamped to tenant-B must be absent from
    // tenant-A's pinned view (it is neither A's nor engine-global).
    let cat_a = session.catalog().pinned_to_tenant(Some(tenant_a));
    let cat_b = session.catalog().pinned_to_tenant(Some(tenant_b));

    assert!(
        cat_a.get_model(&model_a).await.unwrap().is_some(),
        "tenant-A sees its own model {model_a}"
    );
    assert!(
        cat_b.get_model(&model_b).await.unwrap().is_some(),
        "tenant-B sees its own model {model_b}"
    );
    assert!(
        cat_a.get_model(&model_b).await.unwrap().is_none(),
        "tenant-A must NOT see tenant-B's model {model_b} (cross-tenant read isolation)"
    );
    assert!(
        cat_b.get_model(&model_a).await.unwrap().is_none(),
        "tenant-B must NOT see tenant-A's model {model_a} (cross-tenant read isolation)"
    );

    drop(fleet);
}
