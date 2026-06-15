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

use std::str::FromStr;

use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelSpec};
use jammi_db::catalog::model_repo::ModelDescriptor;
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
    ListChannelsRequest, ListModelsRequest, ListMutableTablesRequest, ListSourcesRequest,
    ListTopicsRequest, PromoteModelRequest, RegisterChannelRequest, RegisterTopicRequest,
    RemoveSourceRequest, RetireModelRequest, SetTenantRequest, Tenant,
};
use jammi_wire::{
    channel_from_proto, columns_to_proto, definition_list_from_proto, definition_to_proto,
    encode_ipc_stream, error_from_status, model_from_proto, source_descriptor_from_proto,
    source_type_to_proto, topic_from_proto, trigger_error_from_status, SessionChannel,
    SessionTransport,
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

    /// Soft-retire a model. When `version` is `None` the latest version is
    /// retired. A model outside the caller's scope is rejected as NotFound.
    pub async fn retire_model(&self, model_id: &str, version: Option<i32>) -> Result<()> {
        self.client()
            .retire_model(RetireModelRequest {
                model_id: model_id.to_string(),
                version,
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
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

    /// Promote a model, marking it the promoted version for its name. When
    /// `version` is `None` the latest version is promoted. A model outside the
    /// caller's scope is rejected as NotFound.
    pub async fn promote_model(&self, model_id: &str, version: Option<i32>) -> Result<()> {
        self.client()
            .promote_model(PromoteModelRequest {
                model_id: model_id.to_string(),
                version,
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
