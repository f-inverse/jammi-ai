//! The Jammi control-plane client.
//!
//! [`CatalogClient`] wraps the generated `CatalogServiceClient` over a shared
//! [`jammi_wire::SessionTransport`] and exposes every control verb the single
//! server-side `CatalogService` holds: the source/model registry, the channel
//! declarations, the mutable-table lifecycle, the topic-admin verbs, the
//! server-info handshake, and the tenant trio. It is candle-free — it speaks the
//! typed gRPC wire only and pulls no embedded engine.
//!
//! Every failure decodes the structured [`jammi_wire`] error detail the server
//! attaches, so a control verb returns the exact
//! [`jammi_db::error::JammiError`] variant the in-process path would return
//! — never a lossy gRPC-code-category guess. Tenant scope rides on the session
//! header the transport stamps, never in a request body.

pub mod lifecycle;
pub use lifecycle::{Bearer, Bootstrapped, LicenseApplied, LifecycleClient, PlatformStatus};

use std::str::FromStr;

use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelSpec};
use jammi_db::catalog::model_repo::ModelDescriptor;
use jammi_db::catalog::segment_repo::IndexSegment;
use jammi_db::catalog::source_repo::SourceDescriptor;
use jammi_db::error::{JammiError, Result};
use jammi_db::source::{SourceConnection, SourceType};
use jammi_db::store::mutable::{MutableTableDefinition, MutableTableId};
use jammi_db::trigger::{TopicDefinition, TopicId, TriggerError};
use jammi_db::{ChannelId, ServerInfo, TenantId};

use jammi_wire::proto::catalog::catalog_service_client::CatalogServiceClient;
use jammi_wire::proto::catalog::{
    AddChannelColumnsRequest, AddSourceRequest, CreateMutableTableRequest, DeleteModelRequest,
    DescribeModelRequest, DescribeSourceRequest, DropMutableTableRequest, DropTopicRequest,
    ListChannelsRequest, ListIndexSegmentsRequest, ListModelsRequest, ListMutableTablesRequest,
    ListSourcesRequest, ListTopicsRequest, RegisterChannelRequest, RegisterTopicRequest,
    RemoveSourceRequest, SetTenantRequest, Tenant,
};
use jammi_wire::proto::training::training_service_client::TrainingServiceClient;
use jammi_wire::proto::training::{
    ListTrainingJobsRequest, TrainingStatusRequest, TrainingStatusResponse,
};
use jammi_wire::{
    channel_from_proto, columns_to_proto, definition_list_from_proto, definition_to_proto,
    encode_ipc_stream, error_from_status, index_segment_from_proto, model_from_proto,
    source_descriptor_from_proto, source_type_to_proto, topic_from_proto,
    trigger_error_from_status, SessionChannel, SessionTransport,
};
use tonic::transport::Endpoint;

/// A control-plane client over a shared [`SessionTransport`].
///
/// Cheap to clone: it holds the cloneable transport. A data-plane client
/// composes one of these (over the *same* transport) to delegate the tenant
/// trio, so a tenant bound here is observed by every data verb the data client
/// runs on the same session id.
#[derive(Clone)]
pub struct CatalogClient {
    transport: SessionTransport,
}

impl CatalogClient {
    /// Connect to a `jammi.v1` gRPC endpoint and mint a fresh session id.
    pub async fn connect(endpoint: impl Into<Endpoint>) -> Result<Self> {
        Ok(Self {
            transport: SessionTransport::connect(endpoint).await?,
        })
    }

    /// Build a control client over an existing transport. Used by a data-plane
    /// client that already connected and shares the same session id.
    pub fn over(transport: SessionTransport) -> Self {
        Self { transport }
    }

    /// The transport this client speaks over. A data-plane client clones it to
    /// open its own per-service stubs over the same channel + session id.
    pub fn transport(&self) -> &SessionTransport {
        &self.transport
    }

    /// The opaque session id the server keys tenant state against.
    pub fn session_id(&self) -> &str {
        self.transport.session_id()
    }

    fn client(&self) -> CatalogServiceClient<SessionChannel> {
        self.transport
            .service(CatalogServiceClient::with_interceptor)
    }

    // --- sources ---------------------------------------------------------

    /// Register a data source.
    pub async fn add_source(
        &self,
        source_id: &str,
        source_type: SourceType,
        connection: SourceConnection,
    ) -> Result<()> {
        self.client()
            .add_source(AddSourceRequest {
                source_id: source_id.to_string(),
                source_kind: source_type_to_proto(source_type) as i32,
                connection: Some(connection.into()),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    /// Remove a source and all associated state.
    pub async fn remove_source(&self, source_id: &str) -> Result<()> {
        self.client()
            .remove_source(RemoveSourceRequest {
                source_id: source_id.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    /// Describe every source registered to the session's tenant.
    pub async fn list_sources(&self) -> Result<Vec<SourceDescriptor>> {
        let resp = self
            .client()
            .list_sources(ListSourcesRequest {})
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        resp.sources
            .into_iter()
            .map(|d| source_descriptor_from_proto(d).map_err(|s| error_from_status(&s)))
            .collect()
    }

    /// Describe one registered source by id, or `None` when absent.
    pub async fn describe_source(&self, source_id: &str) -> Result<Option<SourceDescriptor>> {
        match self
            .client()
            .describe_source(DescribeSourceRequest {
                source_id: source_id.to_string(),
            })
            .await
        {
            Ok(resp) => source_descriptor_from_proto(resp.into_inner())
                .map(Some)
                .map_err(|s| error_from_status(&s)),
            Err(status) if status.code() == tonic::Code::NotFound => Ok(None),
            Err(status) => Err(error_from_status(&status)),
        }
    }

    // --- models ----------------------------------------------------------

    /// Describe every model registered to the session's tenant, as the
    /// client-facing [`ModelDescriptor`] projection — the same shape the embedded
    /// `Database.list_models` returns, so a caller reads identical fields
    /// regardless of transport.
    pub async fn list_models(&self) -> Result<Vec<ModelDescriptor>> {
        let resp = self
            .client()
            .list_models(ListModelsRequest {})
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        resp.models
            .into_iter()
            .map(|m| model_from_proto(m).map_err(|s| error_from_status(&s)))
            .collect()
    }

    /// Describe one registered model by id, or `None` when absent. Returns the
    /// client-facing [`ModelDescriptor`] projection.
    pub async fn describe_model(&self, model_id: &str) -> Result<Option<ModelDescriptor>> {
        match self
            .client()
            .describe_model(DescribeModelRequest {
                model_id: model_id.to_string(),
            })
            .await
        {
            Ok(resp) => model_from_proto(resp.into_inner())
                .map(Some)
                .map_err(|s| error_from_status(&s)),
            Err(status) if status.code() == tonic::Code::NotFound => Ok(None),
            Err(status) => Err(error_from_status(&status)),
        }
    }

    /// Hard-delete a model. When `version` is `None` the latest version is
    /// targeted. A model outside the caller's scope is rejected as NotFound; a
    /// still-referenced model is rejected as `FailedPrecondition`
    /// ([`JammiError::ModelReferenced`]).
    /// When `if_exists` is set, deleting an absent model is a no-op.
    pub async fn delete_model(
        &self,
        model_id: &str,
        version: Option<i32>,
        if_exists: bool,
    ) -> Result<()> {
        self.client()
            .delete_model(DeleteModelRequest {
                model_id: model_id.to_string(),
                version,
                if_exists,
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    // --- server info -----------------------------------------------------

    /// The engine's capabilities handshake: version, features, storage
    /// backends, mounted services.
    pub async fn server_info(&self) -> Result<ServerInfo> {
        let resp = self
            .client()
            .get_server_info(())
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(ServerInfo {
            version: resp.version,
            features: resp.features,
            storage_backends: resp.storage_backends,
            services: resp.services,
        })
    }

    // --- training jobs (read-only) ----------------------------------------

    /// Read one training job's lifecycle status by id: status, output model id
    /// (empty until completed), the failure message (non-empty exactly when
    /// status is `"failed"`), and the run-metrics blob (issue #441; present once
    /// the worker has stamped a first run record — the same catalog
    /// `training_jobs.metrics` column the wire's `TrainingStatus.metrics_json`
    /// carries). The control-plane read peer of the data-plane client's submit —
    /// there is no progress surface to read (the engine persists run metrics
    /// only at job finalization, with an early partial stamp at claim time).
    pub async fn training_status(&self, job_id: &str) -> Result<TrainingStatusInfo> {
        let resp = self
            .training_client()
            .training_status(TrainingStatusRequest {
                job_id: job_id.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(training_status_info_from_proto(resp))
    }

    /// List training jobs visible to the session tenant, most recent first —
    /// each row the same lifecycle projection [`Self::training_status`] reads,
    /// plus the submit-time identity (kind, base model, creation time).
    pub async fn list_training_jobs(&self) -> Result<Vec<TrainingJobSummary>> {
        let resp = self
            .training_client()
            .list_training_jobs(ListTrainingJobsRequest {})
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(resp
            .jobs
            .into_iter()
            .map(|j| TrainingJobSummary {
                job_id: j.job_id,
                kind: j.kind,
                status: j.status,
                base_model_id: j.base_model_id,
                output_model_id: j.output_model_id,
                created_at: j.created_at,
                error: j.error,
            })
            .collect())
    }

    fn training_client(&self) -> TrainingServiceClient<SessionChannel> {
        self.transport
            .service(TrainingServiceClient::with_interceptor)
    }

    // --- mutable tables --------------------------------------------------

    /// Register a mutable companion table.
    pub async fn create_mutable_table(
        &self,
        def: MutableTableDefinition,
    ) -> Result<MutableTableId> {
        let definition = definition_to_proto(&def).map_err(|s| error_from_status(&s))?;
        let resp = self
            .client()
            .create_mutable_table(CreateMutableTableRequest {
                definition: Some(definition),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        MutableTableId::new(resp.mutable_table_id).map_err(JammiError::MutableTable)
    }

    /// Drop a mutable companion table.
    pub async fn drop_mutable_table(&self, id: &MutableTableId) -> Result<()> {
        self.client()
            .drop_mutable_table(DropMutableTableRequest {
                mutable_table_id: id.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    /// List every mutable companion table registered to the session's tenant.
    pub async fn list_mutable_tables(&self) -> Result<Vec<MutableTableDefinition>> {
        let resp = self
            .client()
            .list_mutable_tables(ListMutableTablesRequest {})
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        resp.definitions
            .into_iter()
            .map(|d| definition_list_from_proto(d).map_err(|s| error_from_status(&s)))
            .collect()
    }

    // --- index segments ---------------------------------------------------

    /// Every ANN index segment of `table_name`, ordered by `segment_id` — the
    /// same [`IndexSegment`] rows the embedded `Session::list_index_segments`
    /// returns, so a caller reads identical fields regardless of transport.
    ///
    /// Tenant-scoped server-side: the table is resolved through the
    /// tenant-filtered result-table read first, so an empty listing is returned
    /// for a table the session's tenant cannot resolve — the same answer an
    /// unknown table (or a table whose index is flat) gets.
    pub async fn list_index_segments(&self, table_name: &str) -> Result<Vec<IndexSegment>> {
        let resp = self
            .client()
            .list_index_segments(ListIndexSegmentsRequest {
                table_name: table_name.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        resp.segments
            .into_iter()
            .map(|s| index_segment_from_proto(s).map_err(|s| error_from_status(&s)))
            .collect()
    }

    // --- channels --------------------------------------------------------

    /// Register an evidence channel and its columns.
    pub async fn register_channel(&self, spec: &ChannelSpec) -> Result<()> {
        self.client()
            .register_channel(RegisterChannelRequest {
                channel_id: spec.id.as_str().to_string(),
                priority: spec.priority,
                columns: columns_to_proto(&spec.columns),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    /// Append columns to an already-registered channel (append-only).
    pub async fn add_channel_columns(
        &self,
        channel: &ChannelId,
        new_columns: &[ChannelColumn],
    ) -> Result<()> {
        self.client()
            .add_channel_columns(AddChannelColumnsRequest {
                channel_id: channel.as_str().to_string(),
                columns: columns_to_proto(new_columns),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    /// List every evidence channel registered to the session's tenant.
    pub async fn list_channels(&self) -> Result<Vec<ChannelSpec>> {
        let resp = self
            .client()
            .list_channels(ListChannelsRequest {})
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        resp.channels
            .into_iter()
            .map(|c| channel_from_proto(c).map_err(|s| error_from_status(&s)))
            .collect()
    }

    // --- topics (control plane) ------------------------------------------

    /// Register a topic (creates its backing table) for the trigger stream.
    /// Returns the server-minted [`TopicId`] — the topic's identity is
    /// engine-assigned, not caller-chosen, so any `topic.id` the caller carried
    /// is irrelevant on the wire and the authoritative id comes back in the
    /// response. A later `drop_topic` keys on this returned id.
    pub async fn register_topic(
        &self,
        topic: &TopicDefinition,
    ) -> std::result::Result<TopicId, TriggerError> {
        let schema =
            encode_ipc_stream(&topic.schema, &[]).map_err(|s| trigger_error_from_status(&s))?;
        let resp = self
            .client()
            .register_topic(RegisterTopicRequest {
                name: topic.name.clone(),
                schema,
                broker_metadata: topic.broker_metadata.clone().into_iter().collect(),
                // The id is engine-assigned, not caller input: the server mints
                // it and ignores this field. Sent empty to make that explicit.
                topic_id: String::new(),
            })
            .await
            .map_err(|s| trigger_error_from_status(&s))?
            .into_inner();
        TopicId::from_str(&resp.topic_id)
            .map_err(|e| TriggerError::Catalog(format!("server returned an invalid topic_id: {e}")))
    }

    /// List every topic visible to the session's tenant.
    pub async fn list_topics(&self) -> std::result::Result<Vec<TopicDefinition>, TriggerError> {
        let resp = self
            .client()
            .list_topics(ListTopicsRequest {
                page_size: 0,
                page_token: String::new(),
                // Tenant scope rides on the session header, not the body.
                tenant_id: String::new(),
            })
            .await
            .map_err(|s| trigger_error_from_status(&s))?
            .into_inner();
        resp.topics
            .into_iter()
            .map(|t| topic_from_proto(t).map_err(|s| trigger_error_from_status(&s)))
            .collect()
    }

    /// Drop a topic and its backing table.
    pub async fn drop_topic(&self, topic_id: TopicId) -> std::result::Result<(), TriggerError> {
        self.client()
            .drop_topic(DropTopicRequest {
                topic_id: topic_id.to_string(),
                if_exists: false,
            })
            .await
            .map_err(|s| trigger_error_from_status(&s))?;
        Ok(())
    }

    // --- tenant ----------------------------------------------------------

    /// Bind a tenant scope to this session (sticky form), keyed by the session
    /// id every verb on the shared transport carries.
    pub async fn bind_tenant(&self, t: TenantId) -> Result<()> {
        self.client()
            .set_tenant(SetTenantRequest {
                tenant: Some(Tenant { id: t.to_string() }),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    /// Clear the bound tenant.
    pub async fn unbind_tenant(&self) -> Result<()> {
        self.client()
            .clear_tenant(())
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    /// The tenant currently bound, if any.
    pub async fn tenant(&self) -> Result<Option<TenantId>> {
        let resp = self
            .client()
            .get_tenant(())
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        let id = resp.tenant.map(|t| t.id).unwrap_or_default();
        if id.is_empty() {
            return Ok(None);
        }
        id.parse()
            .map(Some)
            .map_err(|e| JammiError::Tenant(format!("invalid tenant id from server: {e}")))
    }
}

/// One training job's lifecycle status, as read by
/// [`CatalogClient::training_status`].
#[derive(Debug, Clone)]
pub struct TrainingStatusInfo {
    /// Current lifecycle status: `"queued"`, `"running"`, `"completed"`, or
    /// `"failed"`.
    pub status: String,
    /// The output model id the trained artifact registers under; empty until
    /// the job completes.
    pub model_id: String,
    /// The failure message; non-empty exactly when `status` is `"failed"`.
    pub error: String,
    /// Run metrics recorded for this job, as the opaque JSON blob text the
    /// wire's `TrainingStatus.metrics_json` carries (issue #441) — the SAME
    /// catalog `training_jobs.metrics` column the embedded `TrainingJob.
    /// metrics()` reads. `None` for a job with no run record yet (still
    /// `"queued"`); this control-plane read never decodes or re-encodes the
    /// blob, so it stays byte-identical to the wire field. Schema documented
    /// at the trainer, not here.
    pub metrics_json: Option<String>,
    /// GPU-acceleration determination for this job, as the opaque,
    /// self-describing JSON blob text the wire's `TrainingStatus.
    /// acceleration_report_json` carries (esc-075) — the SAME catalog
    /// `training_jobs.acceleration_report` column the embedded record read
    /// returns. `None` for a legacy row predating the column (SQL `NULL`);
    /// otherwise a `"state"`-keyed object whose vocabulary is owned by the
    /// payload's producer (e.g. `"pending"` before a determination exists,
    /// `"determined"` once one does) and documented there, not enumerated
    /// here. This control-plane read never decodes or re-encodes the blob, so
    /// it stays byte-identical to the wire field.
    pub acceleration_report_json: Option<String>,
}

/// Build a [`TrainingStatusInfo`] from the wire response. Pulled out of
/// [`CatalogClient::training_status`] so the `metrics_json` present/absent
/// mapping is unit-testable without a live gRPC round trip.
fn training_status_info_from_proto(resp: TrainingStatusResponse) -> TrainingStatusInfo {
    TrainingStatusInfo {
        status: resp.status,
        model_id: resp.model_id,
        error: resp.error,
        metrics_json: resp.metrics_json,
        acceleration_report_json: resp.acceleration_report_json,
    }
}

#[cfg(test)]
mod training_status_info_tests {
    use super::*;

    /// The present arm: a completed job's wire response carries a metrics
    /// blob, and the control-plane read relays it verbatim (no decode, no
    /// re-encode) — byte-identical to the wire text.
    #[test]
    fn metrics_json_present_arm_carries_the_wire_blob_verbatim() {
        let resp = TrainingStatusResponse {
            status: "completed".to_string(),
            model_id: "jammi:fine-tuned:abc".to_string(),
            error: String::new(),
            metrics_json: Some(r#"{"final_loss":0.1,"train_loss_curve":[[0,0.2]]}"#.to_string()),
            acceleration_report_json: Some(r#"{"state":"determined","fa2_f16":true}"#.to_string()),
        };
        let info = training_status_info_from_proto(resp);
        assert_eq!(
            info.metrics_json.as_deref(),
            Some(r#"{"final_loss":0.1,"train_loss_curve":[[0,0.2]]}"#)
        );
    }

    /// The absent arm: a queued job's wire response carries no metrics field
    /// yet (field presence, not an empty string) — the control-plane read
    /// stays `None`, never inventing an empty-object placeholder.
    #[test]
    fn metrics_json_absent_arm_stays_none() {
        let resp = TrainingStatusResponse {
            status: "queued".to_string(),
            model_id: String::new(),
            error: String::new(),
            metrics_json: None,
            acceleration_report_json: Some(r#"{"state":"pending"}"#.to_string()),
        };
        let info = training_status_info_from_proto(resp);
        assert_eq!(info.metrics_json, None);
    }

    /// The determined arm (esc-075): a claimed job's wire response carries the
    /// claiming worker's determination, and the control-plane read relays it
    /// verbatim (no decode, no re-encode) — byte-identical to the wire text.
    /// Mirrors `metrics_json_present_arm_carries_the_wire_blob_verbatim`.
    #[test]
    fn acceleration_report_json_determined_arm_carries_the_wire_blob_verbatim() {
        let resp = TrainingStatusResponse {
            status: "running".to_string(),
            model_id: String::new(),
            error: String::new(),
            metrics_json: None,
            acceleration_report_json: Some(
                r#"{"state":"determined","fa2_f16":true,"reason":"sm_90 capable"}"#.to_string(),
            ),
        };
        let info = training_status_info_from_proto(resp);
        assert_eq!(
            info.acceleration_report_json.as_deref(),
            Some(r#"{"state":"determined","fa2_f16":true,"reason":"sm_90 capable"}"#)
        );
    }

    /// The pending arm (esc-075): a freshly submitted, unclaimed job's wire
    /// response carries the explicit pending marker, never `None` or an empty
    /// string.
    #[test]
    fn acceleration_report_json_pending_arm_carries_the_explicit_marker() {
        let resp = TrainingStatusResponse {
            status: "queued".to_string(),
            model_id: String::new(),
            error: String::new(),
            metrics_json: None,
            acceleration_report_json: Some(r#"{"state":"pending"}"#.to_string()),
        };
        let info = training_status_info_from_proto(resp);
        assert_eq!(
            info.acceleration_report_json.as_deref(),
            Some(r#"{"state":"pending"}"#)
        );
    }

    /// The absent arm (esc-075): a legacy row predating the
    /// `acceleration_report` column carries no wire field (field presence,
    /// not an empty string) — the control-plane read stays `None`, never
    /// inventing a fabricated tri-state value. Mirrors
    /// `metrics_json_absent_arm_stays_none`.
    #[test]
    fn acceleration_report_json_absent_arm_stays_none() {
        let resp = TrainingStatusResponse {
            status: "completed".to_string(),
            model_id: "jammi:fine-tuned:legacy".to_string(),
            error: String::new(),
            metrics_json: Some(r#"{"final_loss":0.2}"#.to_string()),
            acceleration_report_json: None,
        };
        let info = training_status_info_from_proto(resp);
        assert_eq!(info.acceleration_report_json, None);
    }
}

/// One row of [`CatalogClient::list_training_jobs`]: the
/// [`TrainingStatusInfo`] projection plus the job's submit-time identity.
#[derive(Debug, Clone)]
pub struct TrainingJobSummary {
    /// Server-assigned job id — the `training_status` key.
    pub job_id: String,
    /// Training-job kind: `"fine_tune"`, `"graph_fine_tune"`, or
    /// `"context_predictor"`.
    pub kind: String,
    /// Current lifecycle status: `"queued"`, `"running"`, `"completed"`, or
    /// `"failed"`.
    pub status: String,
    /// The base model the job trains from — the catalog's registered model id
    /// (a resolved, versioned id, not necessarily the submit-time reference
    /// string).
    pub base_model_id: String,
    /// The output model id the trained artifact registers under; empty until
    /// the job completes.
    pub output_model_id: String,
    /// Job creation time, as recorded by the catalog (UTC text timestamp).
    pub created_at: String,
    /// The failure message; non-empty exactly when `status` is `"failed"`.
    pub error: String,
}
