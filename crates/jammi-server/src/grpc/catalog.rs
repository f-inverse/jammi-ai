//! `CatalogService` gRPC implementation: the engine's single control-plane
//! surface.
//!
//! Every catalog / metadata / lifecycle / observability verb a remote client
//! reaches lands here, folding what were five straddling services
//! (session / sources+models-on-embedding / channel / mutable-table /
//! topic-admin-on-trigger) into one. The verbs split into two backings:
//!
//! * **Engine-free** — the tenant trio (`SetTenant` / `GetTenant` /
//!   `ClearTenant`) and the `GetServerInfo` handshake update / read the
//!   in-process [`SessionStore`] and report compile-time + runtime tier facts.
//!   They need no engine, so they answer even on an engine-light deployment.
//! * **Engine-backed** — sources / models / channels / mutable tables / topics
//!   delegate 1:1 to the transport-agnostic [`Session`]/[`LocalSession`]
//!   abstraction (`session_tenant(&request)` → [`scoped`] → [`map_engine_error`]
//!   / [`map_trigger_error`]), exactly mirroring the handlers they were lifted
//!   from. These require an engine; an engine-light deployment returns a
//!   truthful `Unavailable` rather than a faked empty result.
//!
//! Tenant scope is read from the request's [`SessionTenant`] extension (set
//! upstream by [`crate::grpc::session::TenantInterceptor`]).
//!
//! [`SessionTenant`]: crate::grpc::session::SessionTenant
//! [`SessionStore`]: crate::grpc::session::SessionStore

use std::str::FromStr;
use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_ai::wire::{
    channel_to_proto, columns_from_proto, definition_from_proto, definition_to_proto,
    model_to_proto, parse_channel_id, parse_table_id, source_type_from_proto, topic_to_proto,
};
use jammi_ai::{LocalSession, Session};
use jammi_db::catalog::channel_repo::ChannelSpec;
use jammi_db::source::SourceConnection;
use jammi_db::trigger::ids::TopicId;
use jammi_db::trigger::{TopicDefinition, TriggerError};
use jammi_db::TenantId;
use tonic::{Request, Response, Status};

use crate::grpc::proto::catalog as pb;
use crate::grpc::proto::catalog::catalog_service_server::CatalogService;
use crate::grpc::session::{SessionId, SessionStore, SESSION_HEADER};
use crate::grpc::wire::{
    map_engine_error, map_trigger_error, require_nonempty, scoped, session_tenant,
};
use crate::tiers::TierSet;

/// Server-side handler for the control-plane gRPC surface.
///
/// Holds the in-process [`SessionStore`] + mounted [`TierSet`] that back the
/// engine-free verbs (the tenant trio and `GetServerInfo`), and an optional
/// shared engine session that backs the catalog / lifecycle verbs. The engine
/// is `None` on an engine-light deployment (one that mounts the control plane
/// but no data-plane engine); those verbs then report a truthful `Unavailable`.
pub struct CatalogServer {
    store: SessionStore,
    tiers: TierSet,
    session: Option<Arc<InferenceSession>>,
}

impl CatalogServer {
    pub fn new(
        store: SessionStore,
        tiers: TierSet,
        session: Option<Arc<InferenceSession>>,
    ) -> Self {
        Self {
            store,
            tiers,
            session,
        }
    }

    /// The shared engine session backing the catalog / lifecycle verbs, or a
    /// truthful `Unavailable` when this deployment mounts no engine.
    fn engine(&self) -> Result<&Arc<InferenceSession>, Status> {
        self.session.as_ref().ok_or_else(|| {
            Status::unavailable("catalog engine verbs are not enabled on this deployment")
        })
    }

    /// A [`Session`] over the shared engine; see [`crate::grpc::inference`] for
    /// the tenant-scope wiring rationale.
    fn local(&self) -> Result<Session, Status> {
        Ok(Session::Local(LocalSession::new(Arc::clone(
            self.engine()?,
        ))))
    }
}

#[tonic::async_trait]
impl CatalogService for CatalogServer {
    // --- session -----------------------------------------------------------

    async fn set_tenant(
        &self,
        request: Request<pb::SetTenantRequest>,
    ) -> Result<Response<()>, Status> {
        let session = session_id_from_request(&request)?;
        let body = request.into_inner();
        let tenant = body
            .tenant
            .ok_or_else(|| Status::invalid_argument("tenant field is required"))?;
        let parsed = parse_tenant(&tenant)?;
        self.store.set(session, parsed);
        Ok(Response::new(()))
    }

    async fn get_tenant(
        &self,
        request: Request<()>,
    ) -> Result<Response<pb::GetTenantResponse>, Status> {
        let session = session_id_from_request(&request)?;
        let tenant = self.store.get(&session);
        Ok(Response::new(pb::GetTenantResponse {
            tenant: Some(pb::Tenant {
                id: tenant.map(|t| t.to_string()).unwrap_or_default(),
            }),
        }))
    }

    async fn clear_tenant(&self, request: Request<()>) -> Result<Response<()>, Status> {
        let session = session_id_from_request(&request)?;
        self.store.clear(&session);
        Ok(Response::new(()))
    }

    async fn get_server_info(
        &self,
        _request: Request<()>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        // The build's compile-time self-description (`version` / `features` /
        // `storage_backends`) plus this deployment's *runtime* tier handshake:
        // `services` is the set of gRPC tiers this server actually mounted, not
        // a compile-time fact — so it comes from the injected `TierSet`, not
        // from `ServerInfo::current` (which leaves it empty).
        let info = jammi_db::ServerInfo::current();
        Ok(Response::new(pb::ServerInfo {
            version: info.version,
            features: info.features,
            storage_backends: info.storage_backends,
            services: self.tiers.as_wire(),
        }))
    }

    // --- sources / models --------------------------------------------------

    async fn add_source(
        &self,
        request: Request<pb::AddSourceRequest>,
    ) -> Result<Response<()>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        let source_type = source_type_from_proto(req.source_kind)?;
        let connection: SourceConnection = req
            .connection
            .ok_or_else(|| Status::invalid_argument("connection is required"))?
            .try_into()?;
        let session = self.local()?;

        scoped(self.engine()?, tenant, || {
            session.add_source(&req.source_id, source_type, connection)
        })
        .await
        .map_err(map_engine_error)?;
        Ok(Response::new(()))
    }

    async fn remove_source(
        &self,
        request: Request<pb::RemoveSourceRequest>,
    ) -> Result<Response<()>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        let session = self.local()?;

        scoped(self.engine()?, tenant, || {
            session.remove_source(&req.source_id)
        })
        .await
        .map_err(map_engine_error)?;
        Ok(Response::new(()))
    }

    async fn list_sources(
        &self,
        request: Request<pb::ListSourcesRequest>,
    ) -> Result<Response<pb::ListSourcesResponse>, Status> {
        let tenant = session_tenant(&request);
        let session = self.local()?;

        let descriptors = scoped(self.engine()?, tenant, || session.list_sources())
            .await
            .map_err(map_engine_error)?;

        let sources = descriptors
            .into_iter()
            .map(pb::SourceDescriptor::from)
            .collect();
        Ok(Response::new(pb::ListSourcesResponse { sources }))
    }

    async fn describe_source(
        &self,
        request: Request<pb::DescribeSourceRequest>,
    ) -> Result<Response<pb::SourceDescriptor>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        let session = self.local()?;

        // An absent source returns NotFound — the truthful "no such source for
        // this tenant" the remote arm maps back to `None`, never a faked empty
        // descriptor.
        let descriptor = scoped(self.engine()?, tenant, || {
            session.describe_source(&req.source_id)
        })
        .await
        .map_err(map_engine_error)?
        .ok_or_else(|| Status::not_found(format!("source '{}' not found", req.source_id)))?;

        Ok(Response::new(pb::SourceDescriptor::from(descriptor)))
    }

    async fn list_models(
        &self,
        request: Request<pb::ListModelsRequest>,
    ) -> Result<Response<pb::ListModelsResponse>, Status> {
        let tenant = session_tenant(&request);
        let session = self.local()?;

        let records = scoped(self.engine()?, tenant, || session.list_models())
            .await
            .map_err(map_engine_error)?;

        let models = records.iter().map(model_to_proto).collect();
        Ok(Response::new(pb::ListModelsResponse { models }))
    }

    async fn describe_model(
        &self,
        request: Request<pb::DescribeModelRequest>,
    ) -> Result<Response<pb::Model>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.model_id, "model_id")?;
        let session = self.local()?;

        // An absent model returns NotFound — the truthful "no such model for this
        // tenant" the remote arm maps back to `None`, never a faked record.
        let record = scoped(self.engine()?, tenant, || {
            session.describe_model(&req.model_id)
        })
        .await
        .map_err(map_engine_error)?
        .ok_or_else(|| Status::not_found(format!("model '{}' not found", req.model_id)))?;

        Ok(Response::new(model_to_proto(&record)))
    }

    // --- channels ----------------------------------------------------------

    async fn register_channel(
        &self,
        request: Request<pb::RegisterChannelRequest>,
    ) -> Result<Response<()>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        let id = parse_channel_id(&req.channel_id)?;
        let columns = columns_from_proto(req.columns)?;
        let spec = ChannelSpec {
            id,
            priority: req.priority,
            columns,
        };
        let session = self.local()?;

        scoped(self.engine()?, tenant, || session.register_channel(&spec))
            .await
            .map_err(map_engine_error)?;

        Ok(Response::new(()))
    }

    async fn add_channel_columns(
        &self,
        request: Request<pb::AddChannelColumnsRequest>,
    ) -> Result<Response<()>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        let id = parse_channel_id(&req.channel_id)?;
        let columns = columns_from_proto(req.columns)?;
        let session = self.local()?;

        scoped(self.engine()?, tenant, || {
            session.add_channel_columns(&id, &columns)
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(()))
    }

    async fn list_channels(
        &self,
        request: Request<pb::ListChannelsRequest>,
    ) -> Result<Response<pb::ListChannelsResponse>, Status> {
        let tenant = session_tenant(&request);
        let session = self.local()?;

        let specs = scoped(self.engine()?, tenant, || session.list_channels())
            .await
            .map_err(map_engine_error)?;

        let channels = specs.iter().map(channel_to_proto).collect();
        Ok(Response::new(pb::ListChannelsResponse { channels }))
    }

    // --- mutable tables ----------------------------------------------------

    async fn create_mutable_table(
        &self,
        request: Request<pb::CreateMutableTableRequest>,
    ) -> Result<Response<pb::CreateMutableTableResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        let def_proto = req
            .definition
            .ok_or_else(|| Status::invalid_argument("definition is required"))?;
        let def = definition_from_proto(def_proto, tenant)?;
        let session = self.local()?;

        let id = scoped(self.engine()?, tenant, || session.create_mutable_table(def))
            .await
            .map_err(map_engine_error)?;

        Ok(Response::new(pb::CreateMutableTableResponse {
            mutable_table_id: id.to_string(),
        }))
    }

    async fn drop_mutable_table(
        &self,
        request: Request<pb::DropMutableTableRequest>,
    ) -> Result<Response<()>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        let id = parse_table_id(&req.mutable_table_id)?;
        let session = self.local()?;

        scoped(self.engine()?, tenant, || session.drop_mutable_table(&id))
            .await
            .map_err(map_engine_error)?;

        Ok(Response::new(()))
    }

    async fn list_mutable_tables(
        &self,
        request: Request<pb::ListMutableTablesRequest>,
    ) -> Result<Response<pb::ListMutableTablesResponse>, Status> {
        let tenant = session_tenant(&request);
        let session = self.local()?;

        let defs = scoped(self.engine()?, tenant, || session.list_mutable_tables())
            .await
            .map_err(map_engine_error)?;

        // The wire body stays tenant-free (the catalog row's tenant is the
        // session scope, not a message field), so each definition encodes the
        // same projection the create path carries.
        let definitions = defs
            .iter()
            .map(definition_to_proto)
            .collect::<Result<Vec<_>, Status>>()?;
        Ok(Response::new(pb::ListMutableTablesResponse { definitions }))
    }

    // --- topics ------------------------------------------------------------

    async fn register_topic(
        &self,
        request: Request<pb::RegisterTopicRequest>,
    ) -> Result<Response<pb::RegisterTopicResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        if req.name.is_empty() {
            return Err(Status::invalid_argument("name is required"));
        }
        let schema = jammi_ai::wire::decode_ipc_schema(&req.schema)?;
        let broker_metadata = req.broker_metadata.into_iter().collect();
        // Honor a caller-supplied id (the `Session::register_topic` surface
        // carries the `TopicDefinition.id` the caller minted) so the topic's
        // identity is consistent across transports; an empty id mints a fresh
        // UUIDv7.
        let id = if req.topic_id.is_empty() {
            TopicId::new()
        } else {
            TopicId::from_str(&req.topic_id)
                .map_err(|e| Status::invalid_argument(format!("invalid topic_id: {e}")))?
        };
        let topic = TopicDefinition {
            id,
            name: req.name,
            schema,
            tenant,
            broker_metadata,
        };
        let session = self.local()?;

        // `register_topic` dual-registers the broker driver and the catalog (so a
        // later `publish` resolves the topic) — the engine path owns that
        // invariant, not the handler.
        scoped(self.engine()?, tenant, || session.register_topic(&topic))
            .await
            .map_err(map_trigger_error)?;

        Ok(Response::new(pb::RegisterTopicResponse {
            topic_id: topic.id.to_string(),
        }))
    }

    async fn drop_topic(
        &self,
        request: Request<pb::DropTopicRequest>,
    ) -> Result<Response<()>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        if req.topic_id.is_empty() {
            return Err(Status::invalid_argument("topic_id is required"));
        }
        let topic_id = TopicId::from_str(&req.topic_id)
            .map_err(|e| Status::invalid_argument(format!("invalid topic_id: {e}")))?;
        let session = self.local()?;

        // The engine resolves the topic by id (tenant-scoped) and returns
        // `TopicNotFound` when it is absent; `if_exists` turns that into a no-op.
        match scoped(self.engine()?, tenant, || session.drop_topic(topic_id)).await {
            Ok(()) => Ok(Response::new(())),
            Err(TriggerError::TopicNotFound(_)) if req.if_exists => Ok(Response::new(())),
            Err(e) => Err(map_trigger_error(e)),
        }
    }

    async fn list_topics(
        &self,
        request: Request<pb::ListTopicsRequest>,
    ) -> Result<Response<pb::ListTopicsResponse>, Status> {
        let tenant = session_tenant(&request);
        let session = self.local()?;

        let topics = scoped(self.engine()?, tenant, || session.list_topics())
            .await
            .map_err(map_trigger_error)?;
        let topics = topics
            .iter()
            .map(topic_to_proto)
            .collect::<Result<Vec<_>, Status>>()?;
        Ok(Response::new(pb::ListTopicsResponse {
            topics,
            // Pagination is not yet implemented — `next_page_token` empty
            // means "this is the complete result set."
            next_page_token: String::new(),
        }))
    }
}

/// Extract the session id and return a typed `InvalidArgument` error if it's
/// missing. Used by the tenant verbs that read or mutate session state directly.
fn session_id_from_request<T>(request: &Request<T>) -> Result<SessionId, Status> {
    request
        .metadata()
        .get(SESSION_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(SessionId::new)
        .ok_or_else(|| {
            Status::invalid_argument(format!(
                "request missing required `{SESSION_HEADER}` metadata header"
            ))
        })
}

/// Parse a wire-format [`pb::Tenant`] into the engine's [`TenantId`] newtype.
/// Empty string is interpreted as "no tenant" → `Ok(None)`. Any other value
/// must parse as a non-nil UUID per ADR-00.
fn parse_tenant(t: &pb::Tenant) -> Result<Option<TenantId>, Status> {
    if t.id.is_empty() {
        return Ok(None);
    }
    TenantId::from_str(&t.id)
        .map(Some)
        .map_err(|e| Status::invalid_argument(format!("invalid tenant id: {e}")))
}
