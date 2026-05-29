//! Registration and publication of session-lifecycle events to the
//! `jammi.audit.session_lifecycle.v1` trigger topic.
//!
//! Mirrors the per-query audit topic path: the topic is registered through the
//! catalog [`crate::catalog::topic_repo::TopicRepo`] on first use (which mints
//! the topic id and provisions the durable backing table), then each record's
//! JSON is published through the [`crate::trigger::Publisher`] so the event is
//! both durable and subscribable via the standard `subscribe` surface. The
//! topic is tenant-pinned, so each tenant owns its own lifecycle stream.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

use crate::session::JammiSession;
use crate::tenant::TenantId;
use crate::trigger::topic::TopicDefinition;
use crate::trigger::TopicId;

use super::error::EphemeralError;
use super::event::{SessionLifecycleRecord, SESSION_LIFECYCLE_TOPIC};

/// Arrow schema for the lifecycle topic payload: one `record` column carrying
/// the JSON-encoded [`SessionLifecycleRecord`].
fn payload_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "record",
        DataType::Utf8,
        false,
    )]))
}

/// Ensure the per-tenant lifecycle topic exists, then publish one record.
///
/// `TopicRepo::register_topic` is a one-shot insert (not idempotent), so the
/// topic is looked up first and registered only when absent — making repeated
/// lifecycle publishes for the same tenant idempotent at the topic layer.
pub(super) async fn publish_lifecycle(
    session: &JammiSession,
    tenant: TenantId,
    record: &SessionLifecycleRecord,
) -> Result<(), EphemeralError> {
    let repo = session.topic_repo();

    let topic = match repo
        .lookup_by_name(SESSION_LIFECYCLE_TOPIC, Some(tenant))
        .await
        .map_err(|e| EphemeralError::Broker(e.to_string()))?
    {
        Some(existing) => existing,
        None => {
            let definition = TopicDefinition {
                id: TopicId::new(),
                name: SESSION_LIFECYCLE_TOPIC.to_string(),
                schema: payload_schema(),
                tenant: Some(tenant),
                broker_metadata: BTreeMap::new(),
            };
            repo.register_topic(&definition)
                .await
                .map_err(|e| EphemeralError::Broker(e.to_string()))?;
            definition
        }
    };

    // The broker driver's per-process channel is not persisted across restarts;
    // ensure it exists for live fan-out (idempotent on a matching schema).
    session
        .trigger_broker()
        .register_topic(&topic)
        .await
        .map_err(|e| EphemeralError::Broker(e.to_string()))?;

    let payload = serde_json::to_string(record)?;
    let batch = RecordBatch::try_new(
        Arc::clone(&topic.schema),
        vec![Arc::new(StringArray::from(vec![payload]))],
    )
    .map_err(|e| EphemeralError::Broker(format!("build lifecycle batch: {e}")))?;

    session
        .publisher()
        .publish_scoped(&topic, Some(tenant), batch)
        .await
        .map_err(|e| EphemeralError::Broker(e.to_string()))?;
    Ok(())
}
