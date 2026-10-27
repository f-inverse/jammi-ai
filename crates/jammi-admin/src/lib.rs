//! The Jammi control-plane client.
//!
//! [`CatalogClient`] wraps the generated `CatalogServiceClient` over a shared
//! [`jammi_wire::SessionTransport`] and exposes every control verb the single
//! server-side `CatalogService` holds: the source/model registry, the channel
//! declarations, the mutable-table lifecycle, the topic-admin verbs, the
//! materialization-contract reads (describe, verify, staleness, lineage), the
//! server-info handshake, and the tenant trio. It is candle-free — it speaks the
//! typed gRPC wire only and pulls no embedded engine.
//!
//! Every failure decodes the structured [`jammi_wire`] error detail the server
//! attaches, so a control verb returns the exact
//! [`jammi_db::error::JammiError`] variant the in-process path would return
//! — never a lossy gRPC-code-category guess. Tenant scope rides on the session
//! header the transport stamps, never in a request body.

use std::str::FromStr;

use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelSpec};
use jammi_db::catalog::model_repo::ModelDescriptor;
use jammi_db::catalog::segment_repo::IndexSegment;
use jammi_db::catalog::source_repo::SourceDescriptor;
use jammi_db::error::{JammiError, Result};
use jammi_db::source::{SourceConnection, SourceDescription, SourceType};
use jammi_db::store::manifest::{DefinitionHash, MatchVerdict, MaterializationManifest};
use jammi_db::store::mutable::{MutableTableDefinition, MutableTableId};
use jammi_db::store::{DerivesFromEdge, Staleness};
use jammi_db::trigger::{TopicDefinition, TopicId, TriggerError};
use jammi_db::{ChannelId, ServerInfo, TenantId};

use jammi_wire::proto::catalog::catalog_service_client::CatalogServiceClient;
use jammi_wire::proto::catalog::{
    AddChannelColumnsRequest, AddSourceRequest, CreateMutableTableRequest, DeleteModelRequest,
    DerivesFromRequest, DescribeModelRequest, DescribeSourceRequest, DescribeTableRequest,
    DropMutableTableRequest, DropTopicRequest, ListChannelsRequest, ListIndexSegmentsRequest,
    ListModelsRequest, ListMutableTablesRequest, ListSourcesRequest, ListTopicsRequest,
    ReconcileRequest, RegisterChannelRequest, RegisterTopicRequest, RemoveSourceRequest,
    SetTenantRequest, StalenessRequest, Tenant, VerifyMaterializationRequest,
};
use jammi_wire::proto::job::job_service_client::JobServiceClient;
use jammi_wire::proto::job::{
    CancelJobRequest as JobCancelJobRequest, JobStatusRequest, ListJobsRequest, ListWorkersRequest,
    PruneJobsRequest,
};
use jammi_wire::{
    channel_from_proto, columns_to_proto, definition_list_from_proto, definition_to_proto,
    derives_from_edge_from_proto, describe_table_from_proto, encode_ipc_stream, error_from_status,
    index_segment_from_proto, match_verdict_from_proto, model_from_proto,
    source_description_from_proto, source_descriptor_from_proto, source_type_to_proto,
    staleness_from_proto, topic_from_proto, trigger_error_from_status, SessionChannel,
    SessionTransport,
};
use jammi_wire::{
    job_status_info_from_proto, job_summary_from_proto, worker_summary_from_proto, JobStatusInfo,
    JobSummary, WorkerSummary,
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

    /// Describe one registered source by id: its descriptor and every table it
    /// serves with its columns, or `None` when absent.
    pub async fn describe_source(&self, source_id: &str) -> Result<Option<SourceDescription>> {
        match self
            .client()
            .describe_source(DescribeSourceRequest {
                source_id: source_id.to_string(),
            })
            .await
        {
            Ok(resp) => source_description_from_proto(resp.into_inner())
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
            broker: resp.broker,
        })
    }

    // --- jobs (read-only + cancel) -----------------------------------------

    /// Read one job's lifecycle status by id: status, kind, the model it
    /// produced (once a training kind completes), the failure message (once it
    /// fails), and the run-metrics blob (present once the worker has stamped a
    /// first run record — the same catalog `jobs.result` payload the wire's
    /// `JobStatus.model.metrics_json` carries). The control-plane read peer of
    /// the data-plane client's submit — there is no progress surface to read
    /// beyond `JobStatus.progress` / `WaitJob`.
    pub async fn job_status(&self, job_id: &str) -> Result<JobStatusInfo> {
        let resp = self
            .job_client()
            .job_status(JobStatusRequest {
                job_id: job_id.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        job_status_info_from_proto(resp).map_err(|s| error_from_status(&s))
    }

    /// List jobs visible to the session tenant, most recent first — each row
    /// the same lifecycle projection [`Self::job_status`] reads, plus the
    /// submit-time identity (kind, base model, creation time).
    pub async fn list_jobs(&self) -> Result<Vec<JobSummary>> {
        let resp = self
            .job_client()
            .list_jobs(ListJobsRequest {})
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        resp.jobs
            .into_iter()
            .map(job_summary_from_proto)
            .collect::<std::result::Result<_, _>>()
            .map_err(|s| error_from_status(&s))
    }

    /// Request cancellation of a job by id. `false` when the request had no
    /// effect (the job was already terminal or absent).
    pub async fn cancel_job(&self, job_id: &str) -> Result<bool> {
        let resp = self
            .job_client()
            .cancel_job(JobCancelJobRequest {
                job_id: job_id.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(resp.cancelled)
    }

    /// List the engine processes currently running the claim loop.
    pub async fn list_workers(&self) -> Result<Vec<WorkerSummary>> {
        let resp = self
            .job_client()
            .list_workers(ListWorkersRequest {})
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        resp.workers
            .into_iter()
            .map(worker_summary_from_proto)
            .collect::<std::result::Result<_, _>>()
            .map_err(|s| error_from_status(&s))
    }

    /// Eagerly delete terminal job rows older than the deployment's
    /// `[jobs] retention_days`; returns the number of rows deleted.
    pub async fn prune_jobs(&self) -> Result<u64> {
        let resp = self
            .job_client()
            .prune_jobs(PruneJobsRequest {})
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(resp.jobs_deleted)
    }

    fn job_client(&self) -> JobServiceClient<SessionChannel> {
        self.transport.service(JobServiceClient::with_interceptor)
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

    // --- materialization contract ----------------------------------------

    /// The recorded materialization of `table` — the same
    /// [`MaterializationManifest`] the embedded `Session::describe_table`
    /// returns, decoded through the engine's strict manifest reader. A table
    /// with no manifest is [`JammiError::MissingManifest`].
    pub async fn describe_table(&self, table: &str) -> Result<MaterializationManifest> {
        let resp = self
            .client()
            .describe_table(DescribeTableRequest {
                table: table.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        describe_table_from_proto(resp).map_err(|s| error_from_status(&s))
    }

    /// Recompute `table`'s artifact digest (and, if given, check an expected
    /// definition hash) against its manifest — the same [`MatchVerdict`] the
    /// embedded `Session::verify_materialization` returns. Read-only.
    pub async fn verify_materialization(
        &self,
        table: &str,
        expected_definition: Option<&DefinitionHash>,
    ) -> Result<MatchVerdict> {
        let resp = self
            .client()
            .verify_materialization(VerifyMaterializationRequest {
                table: table.to_string(),
                expected_definition: expected_definition.map(|d| d.0.clone()),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        match_verdict_from_proto(resp.verdict).map_err(|s| error_from_status(&s))
    }

    /// Whether `table` is still the output of its recorded definition over its
    /// recorded inputs' current state — the same [`Staleness`] the embedded
    /// `Session::staleness` reports. Read-only.
    pub async fn staleness(
        &self,
        table: &str,
        current_definition: &DefinitionHash,
    ) -> Result<Staleness> {
        let resp = self
            .client()
            .staleness(StalenessRequest {
                table: table.to_string(),
                current_definition: current_definition.0.clone(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        staleness_from_proto(resp.staleness).map_err(|s| error_from_status(&s))
    }

    /// The one-hop reverse-dependency edges of `table` — the same
    /// [`DerivesFromEdge`]s the embedded `Session::derives_from` returns.
    pub async fn derives_from(&self, table: &str) -> Result<Vec<DerivesFromEdge>> {
        let resp = self
            .client()
            .derives_from(DerivesFromRequest {
                table: table.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        resp.edges
            .into_iter()
            .map(|e| derives_from_edge_from_proto(e).map_err(|s| error_from_status(&s)))
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
            .list_topics(ListTopicsRequest {})
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

    // --- reconcile ---------------------------------------------------------

    /// Cross-check the catalog against the object store and report (or, when
    /// `apply` is set, reclaim) what has drifted.
    ///
    /// `all = false` (the default a caller should reach for) runs
    /// [`CatalogService.Reconcile`] under this session's resolved tenant scope
    /// — the same tenant-scoped semantics every other verb on this client
    /// carries. `all = true` requests the cross-tenant admin pass; the server
    /// gates it behind a deployment-supplied admin authorizer, and a
    /// deployment that never wires one refuses with `PermissionDenied` (the
    /// engine's shipped default).
    ///
    /// `grace_secs = None` lets the server apply its own default; `apply =
    /// true` with too short a `grace_secs` fails with `InvalidArgument`
    /// naming both values ([`JammiError::Config`]).
    ///
    /// [`CatalogService.Reconcile`]: crate
    pub async fn reconcile(
        &self,
        apply: bool,
        grace_secs: Option<u64>,
        all: bool,
    ) -> Result<jammi_db::store::ReconcileReport> {
        let resp = self
            .client()
            .reconcile(ReconcileRequest {
                apply,
                grace_secs,
                all,
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(jammi_wire::reconcile_report_from_proto(resp))
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
            .map_err(|e| JammiError::Internal(format!("invalid tenant id from server: {e}")))
    }
}
