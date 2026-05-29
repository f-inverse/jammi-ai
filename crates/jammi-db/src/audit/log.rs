//! Bulk write path for per-query audit records.
//!
//! Writing a batch performs, in order: tenant resolution, lineage size-cap
//! enforcement, per-tenant signing, table auto-creation, a single batched
//! insert through the mutable-table registry, and publication of the batch to
//! the audit trigger topic.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

use crate::catalog::backend::TxOptions;
use crate::session::JammiSession;
use crate::tenant::TenantId;
use crate::trigger::topic::TopicDefinition;
use crate::trigger::TopicId;

use super::error::AuditError;
use super::record::PerQueryAudit;
use super::signature;
use super::table::{self, AUDIT_TOPIC};

/// Environment variable overriding the `query_lineage` size cap.
pub const MAX_LINEAGE_BYTES_ENV: &str = "JAMMI_AUDIT_MAX_LINEAGE_BYTES";

/// Default `query_lineage` size cap: 8 KiB.
pub const DEFAULT_MAX_LINEAGE_BYTES: usize = 8 * 1024;

/// Maximum allowed size of `query_lineage` JSON, in bytes.
///
/// Tenants must reference image hashes / row ids in lineage, not raw payloads.
/// Override via `JAMMI_AUDIT_MAX_LINEAGE_BYTES`; default 8 KiB.
pub fn max_lineage_bytes() -> usize {
    std::env::var(MAX_LINEAGE_BYTES_ENV)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(DEFAULT_MAX_LINEAGE_BYTES)
}

/// Sign and persist a batch of audit records for the session's tenant.
///
/// The lineage size cap is enforced by construction: any record whose
/// `query_lineage` JSON exceeds the configured maximum is rejected with
/// [`AuditError::LineageTooLarge`] before anything is written, so compliance
/// posture is structural, not advisory. On success every record is inserted in
/// a single batch and the batch is published to the audit trigger topic.
pub async fn log_records(
    session: &JammiSession,
    records: Vec<PerQueryAudit>,
) -> Result<(), AuditError> {
    if records.is_empty() {
        return Ok(());
    }

    // 1. Resolve tenant from the session binding.
    let tenant = session.tenant().ok_or(AuditError::NoTenantBinding)?;
    let tenant_str = tenant.to_string();

    // 2. Enforce the lineage size cap before any side effects.
    let max = max_lineage_bytes();
    for rec in &records {
        let lineage_bytes = serde_json::to_vec(&rec.query_lineage)?.len();
        if lineage_bytes > max {
            return Err(AuditError::LineageTooLarge {
                actual: lineage_bytes,
                max,
            });
        }
    }

    // 3. Inject tenant + sign each record.
    let mut signed = Vec::with_capacity(records.len());
    for mut rec in records {
        rec.tenant_id = Some(tenant_str.clone());
        signature::sign_record(&mut rec)?;
        signed.push(rec);
    }

    // 4. Auto-create the audit table if absent.
    table::ensure_table_exists(session).await?;

    // 5. Single batched insert through the registry, inside a tenant-bound
    //    transaction (the write-side tenant guard asserts the match).
    let batch = build_batch(&signed)?;
    let id = table::audit_table_id()?;
    let registry = session.mutable_tables_arc();
    let backend = session.catalog().backend_arc();
    let tenant_for_tx = tenant;
    backend
        .transaction(TxOptions::default(), move |tx| {
            Box::pin(async move {
                tx.set_tenant(Some(tenant_for_tx));
                registry
                    .insert_batch(tx, &id, &batch)
                    .await
                    .map_err(|e| crate::catalog::backend::BackendError::Execution(e.to_string()))?;
                Ok(())
            })
        })
        .await
        .map_err(|e| AuditError::Storage(e.to_string()))?;

    // 6. Publish the batch to the audit trigger topic (idempotent register).
    publish_to_topic(session, &signed, tenant).await?;

    Ok(())
}

/// Arrow schema for the trigger-topic payload: one row per audit record, each
/// field a JSON-or-scalar column mirroring the stored shape.
fn topic_payload_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "record",
        DataType::Utf8,
        false,
    )]))
}

/// Build the insert `RecordBatch` matching [`table::audit_schema`].
fn build_batch(records: &[PerQueryAudit]) -> Result<RecordBatch, AuditError> {
    let schema = table::audit_schema();

    let query_id: Vec<String> = records.iter().map(|r| r.query_id.to_string()).collect();
    let model_id: Vec<String> = records.iter().map(|r| r.model_id.clone()).collect();
    let model_version: Vec<String> = records.iter().map(|r| r.model_version.clone()).collect();
    let lineage: Vec<String> = records
        .iter()
        .map(|r| serde_json::to_string(&r.query_lineage))
        .collect::<Result<_, _>>()?;
    let top_k: Vec<String> = records
        .iter()
        .map(|r| serde_json::to_string(&r.top_k_result_ids))
        .collect::<Result<_, _>>()?;
    let scores: Vec<String> = records
        .iter()
        .map(|r| serde_json::to_string(&r.retrieval_scores))
        .collect::<Result<_, _>>()?;
    let executed_at: Vec<i64> = records.iter().map(|r| r.executed_at_micros()).collect();
    let signature: Vec<String> = records.iter().map(|r| r.signature.clone()).collect();

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(query_id)),
            Arc::new(StringArray::from(model_id)),
            Arc::new(StringArray::from(model_version)),
            Arc::new(StringArray::from(lineage)),
            Arc::new(StringArray::from(top_k)),
            Arc::new(StringArray::from(scores)),
            Arc::new(Int64Array::from(executed_at)),
            Arc::new(StringArray::from(signature)),
        ],
    )
    .map_err(|e| AuditError::Storage(format!("build audit batch: {e}")))
}

/// Ensure the per-tenant audit topic exists, then publish the records' JSON
/// payloads as a single batch.
///
/// Follows the established trigger-stream pattern: the topic is registered
/// through the catalog [`crate::catalog::topic_repo::TopicRepo`] — which
/// provisions the Phase-2 mutable backing table and persists the catalog row —
/// and the batch is published via the [`crate::trigger::Publisher`], which
/// writes the backing table (the authoritative log) inside one transaction and
/// fans out to the broker. The topic is therefore durable and subscribable
/// through the standard `subscribe` path, not just a transient broker fan-out.
///
/// `TopicRepo::register_topic` is a one-shot insert (not idempotent), so this
/// looks the topic up first and registers only when absent — making repeated
/// `log` calls for the same tenant idempotent at the topic layer. The audit
/// topic is tenant-pinned, so each tenant gets its own `jammi.audit.search.v1`.
async fn publish_to_topic(
    session: &JammiSession,
    records: &[PerQueryAudit],
    tenant: TenantId,
) -> Result<(), AuditError> {
    let repo = session.topic_repo();

    // Resolve the existing topic, or register it on first use for this tenant.
    let topic = match repo
        .lookup_by_name(AUDIT_TOPIC, Some(tenant))
        .await
        .map_err(|e| AuditError::Broker(e.to_string()))?
    {
        Some(existing) => existing,
        None => {
            let definition = TopicDefinition {
                id: TopicId::new(),
                name: AUDIT_TOPIC.to_string(),
                schema: topic_payload_schema(),
                tenant: Some(tenant),
                broker_metadata: std::collections::BTreeMap::new(),
            };
            repo.register_topic(&definition)
                .await
                .map_err(|e| AuditError::Broker(e.to_string()))?;
            definition
        }
    };

    // Broker registration is idempotent on matching schema; ensure the driver
    // channel exists for live fan-out (it is not persisted across processes).
    session
        .trigger_broker()
        .register_topic(&topic)
        .await
        .map_err(|e| AuditError::Broker(e.to_string()))?;

    let payloads: Vec<String> = records
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<_, _>>()?;
    let batch = RecordBatch::try_new(
        Arc::clone(&topic.schema),
        vec![Arc::new(StringArray::from(payloads))],
    )
    .map_err(|e| AuditError::Broker(format!("build topic batch: {e}")))?;

    session
        .publisher()
        .publish_scoped(&topic, Some(tenant), batch)
        .await
        .map_err(|e| AuditError::Broker(e.to_string()))?;
    Ok(())
}
