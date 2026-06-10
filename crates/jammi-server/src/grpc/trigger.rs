//! `TriggerService` gRPC implementation.
//!
//! Five methods land on the wire surface (ADR-01 §5.1, extended with the
//! topic-admin verbs):
//!
//! * `RegisterTopic` — unary; decodes the Arrow IPC schema, mints a topic id,
//!   and registers the topic (and its backing table) via the `TopicRepo`.
//! * `DropTopic` — unary; resolves the topic by name (tenant-scoped) and drops
//!   it; `if_exists` turns a missing topic into a no-op.
//! * `Publish` — unary; encodes the request batch through Arrow IPC,
//!   delegates to the engine's `Publisher` (transactional outbox), and
//!   returns the assigned offset and commit timestamp.
//! * `Subscribe` — server-streaming; builds an engine `Subscription` and
//!   forwards every `DeliveredBatch` as a `SubscribedBatch` until the
//!   client cancels. With `replay_only` set, it instead drives the engine's
//!   finite `replay_only` drain and closes the stream after the retained
//!   replay window — the bounded primitive a one-shot `--no-follow` drain needs,
//!   since the open-ended subscribe stream cannot terminate on its own.
//! * `ListTopics` — unary; pages through the `topics` catalog row set.
//!
//! Tenant scope is read from the request's `SessionTenant` extension (set
//! upstream by [`crate::grpc::session::TenantInterceptor`]). Per-request
//! tenant overrides on the proto messages are accepted but only honoured
//! when the session tenant is unset — a session-bound tenant cannot
//! sidestep its scope by setting `tenant_id` on the body.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use datafusion::execution::context::SessionContext;
use futures::Stream;
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use jammi_ai::wire::{
    attach_trigger_detail, decode_ipc_schema, decode_publish_batch, encode_delivered_batch,
    to_proto_timestamp, topic_to_proto,
};
use jammi_db::catalog::topic_repo::TopicRepo;
use jammi_db::trigger::ids::TopicId;
use jammi_db::trigger::{Offset, Predicate, Publisher, Subscriber, TopicDefinition, TriggerError};
use jammi_db::TenantId;

use crate::grpc::proto::trigger::trigger_service_server::TriggerService;
use crate::grpc::proto::trigger::{
    DropTopicRequest, ListTopicsRequest, ListTopicsResponse, PublishRequest, PublishResponse,
    RegisterTopicRequest, RegisterTopicResponse, SubscribeRequest, SubscribedBatch, TopicName,
};
use crate::grpc::session::SessionTenant;

/// Server-side handler for the trigger-stream gRPC surface. Holds shared
/// references to the engine-side publisher, subscriber, topic catalog repo,
/// and a DataFusion `SessionContext` used to parse subscribe predicates.
pub struct TriggerServer {
    topic_repo: Arc<TopicRepo>,
    publisher: Arc<Publisher>,
    subscriber: Arc<Subscriber>,
    session_ctx: SessionContext,
}

impl TriggerServer {
    pub fn new(
        topic_repo: Arc<TopicRepo>,
        publisher: Arc<Publisher>,
        subscriber: Arc<Subscriber>,
    ) -> Self {
        Self {
            topic_repo,
            publisher,
            subscriber,
            session_ctx: SessionContext::new(),
        }
    }
}

/// Bounded buffer between the engine subscription stream and the tonic
/// server-stream sender — provides the documented backpressure path
/// (broker tail → mpsc(N) → HTTP/2 flow control → client).
const SUBSCRIBE_BUFFER: usize = 256;

#[tonic::async_trait]
impl TriggerService for TriggerServer {
    type SubscribeStream =
        Pin<Box<dyn Stream<Item = Result<SubscribedBatch, Status>> + Send + 'static>>;

    async fn register_topic(
        &self,
        request: Request<RegisterTopicRequest>,
    ) -> Result<Response<RegisterTopicResponse>, Status> {
        let tenant = resolve_tenant(&request, None)?;
        let req = request.into_inner();
        if req.name.is_empty() {
            return Err(Status::invalid_argument("name is required"));
        }
        let schema = decode_ipc_schema(&req.schema)?;
        let broker_metadata: BTreeMap<String, String> = req.broker_metadata.into_iter().collect();
        // Honor a caller-supplied id (the `Session::register_topic` surface
        // carries the `TopicDefinition.id` the caller minted) so the topic's
        // identity is consistent across transports; the DDL / CLI path leaves it
        // empty and the server mints a fresh UUIDv7.
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
        self.topic_repo
            .register_topic(&topic)
            .await
            .map_err(map_trigger_error)?;
        Ok(Response::new(RegisterTopicResponse {
            topic_id: topic.id.to_string(),
        }))
    }

    async fn drop_topic(&self, request: Request<DropTopicRequest>) -> Result<Response<()>, Status> {
        let tenant = resolve_tenant(&request, None)?;
        let req = request.into_inner();
        if req.topic_id.is_empty() {
            return Err(Status::invalid_argument("topic_id is required"));
        }
        let topic_id = TopicId::from_str(&req.topic_id)
            .map_err(|e| Status::invalid_argument(format!("invalid topic_id: {e}")))?;
        // The repo resolves the topic by id (tenant-scoped) and returns
        // `TopicNotFound` when it is absent; `if_exists` turns that into a no-op.
        match self.topic_repo.drop_topic(topic_id, tenant).await {
            Ok(()) => Ok(Response::new(())),
            Err(TriggerError::TopicNotFound(_)) if req.if_exists => Ok(Response::new(())),
            Err(e) => Err(map_trigger_error(e)),
        }
    }

    async fn publish(
        &self,
        request: Request<PublishRequest>,
    ) -> Result<Response<PublishResponse>, Status> {
        let tenant = resolve_tenant(&request, request_tenant(&request))?;
        let req = request.into_inner();
        let topic = self.lookup_topic(req.topic, tenant).await?;
        let batch = req
            .batch
            .ok_or_else(|| Status::invalid_argument("batch is required"))?;
        let record_batch = decode_publish_batch(&batch, &topic)?;
        let offset = self
            .publisher
            .publish_scoped(&topic, tenant, record_batch)
            .await
            .map_err(map_trigger_error)?;
        Ok(Response::new(PublishResponse {
            offset: offset.value(),
            committed_at: Some(to_proto_timestamp(offset.committed_at())),
        }))
    }

    async fn subscribe(
        &self,
        request: Request<SubscribeRequest>,
    ) -> Result<Response<Self::SubscribeStream>, Status> {
        let tenant = resolve_tenant(&request, request_tenant(&request))?;
        let req = request.into_inner();
        let topic = self.lookup_topic(req.topic, tenant).await?;
        let predicate =
            Predicate::from_sql(&self.session_ctx, Arc::clone(&topic.schema), &req.predicate)
                .map_err(map_trigger_error)?;
        let from_offset = req.from_offset.map(|v| Offset::new(v, chrono::Utc::now()));
        let topic_schema = Arc::clone(&topic.schema);

        if req.replay_only {
            // The finite-drain primitive: the open-ended `subscribe` stream
            // cannot terminate on its own, so a bounded `--no-follow` drain
            // drives the engine's `replay_only` path instead. It returns the
            // retained replay window as a `Vec`, which encodes into a finite
            // stream that yields those batches and ends — the wire-side analogue
            // of the local arm's `Subscriber::replay_only`.
            let drained = self
                .subscriber
                .replay_only_scoped(&topic, tenant, predicate, from_offset)
                .await
                .map_err(map_trigger_error)?;
            let encoded: Vec<Result<SubscribedBatch, Status>> = drained
                .into_iter()
                .map(|delivered| encode_delivered_batch(&topic_schema, delivered))
                .collect();
            let out_stream = futures::stream::iter(encoded);
            return Ok(Response::new(Box::pin(out_stream) as Self::SubscribeStream));
        }

        let mut inner = self
            .subscriber
            .subscribe(&topic, predicate, from_offset)
            .await
            .map_err(map_trigger_error)?;

        let (tx, rx) = mpsc::channel::<Result<SubscribedBatch, Status>>(SUBSCRIBE_BUFFER);
        tokio::spawn(async move {
            while let Some(item) = inner.next().await {
                let result = item
                    .map_err(map_trigger_error)
                    .and_then(|delivered| encode_delivered_batch(&topic_schema, delivered));
                if tx.send(result).await.is_err() {
                    break;
                }
            }
        });
        let out_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(out_stream) as Self::SubscribeStream))
    }

    async fn list_topics(
        &self,
        request: Request<ListTopicsRequest>,
    ) -> Result<Response<ListTopicsResponse>, Status> {
        let req_tenant = parse_optional_tenant(&request.get_ref().tenant_id)?;
        let tenant = resolve_tenant(&request, req_tenant)?;
        let topics = self
            .topic_repo
            .list_topics(tenant)
            .await
            .map_err(map_trigger_error)?;
        let topics = topics
            .iter()
            .map(topic_to_proto)
            .collect::<Result<Vec<_>, Status>>()?;
        Ok(Response::new(ListTopicsResponse {
            topics,
            // Pagination is not yet implemented — `next_page_token` empty
            // means "this is the complete result set."
            next_page_token: String::new(),
        }))
    }
}

impl TriggerServer {
    async fn lookup_topic(
        &self,
        wire: Option<TopicName>,
        tenant: Option<TenantId>,
    ) -> Result<TopicDefinition, Status> {
        let name = wire
            .ok_or_else(|| Status::invalid_argument("topic is required"))?
            .name;
        match self.topic_repo.lookup_by_name(&name, tenant).await {
            Ok(Some(topic)) => Ok(topic),
            Ok(None) => Err(Status::not_found(format!("topic '{name}' not found"))),
            Err(e) => Err(map_trigger_error(e)),
        }
    }
}

/// Read the per-request tenant override from a `PublishRequest` or
/// `SubscribeRequest`'s `tenant_id` field, parsing it through the engine's
/// `TenantId` newtype so an empty string maps to `None` and a non-nil UUID
/// is accepted as `Some`.
fn request_tenant<T: HasTenantId>(request: &Request<T>) -> Option<TenantId> {
    parse_optional_tenant(request.get_ref().tenant_id())
        .ok()
        .flatten()
}

trait HasTenantId {
    fn tenant_id(&self) -> &str;
}

impl HasTenantId for PublishRequest {
    fn tenant_id(&self) -> &str {
        &self.tenant_id
    }
}

impl HasTenantId for SubscribeRequest {
    fn tenant_id(&self) -> &str {
        &self.tenant_id
    }
}

impl HasTenantId for ListTopicsRequest {
    fn tenant_id(&self) -> &str {
        &self.tenant_id
    }
}

fn parse_optional_tenant(id: &str) -> Result<Option<TenantId>, Status> {
    if id.is_empty() {
        return Ok(None);
    }
    TenantId::from_str(id)
        .map(Some)
        .map_err(|e| Status::invalid_argument(format!("invalid tenant id: {e}")))
}

/// Pick the effective tenant: a session-bound tenant always wins over an
/// override on the request body; otherwise the body override is used (or
/// `None` for an unscoped session).
fn resolve_tenant<T>(
    request: &Request<T>,
    body_override: Option<TenantId>,
) -> Result<Option<TenantId>, Status> {
    let session_tenant = request
        .extensions()
        .get::<SessionTenant>()
        .and_then(|s| s.0);
    Ok(session_tenant.or(body_override))
}

/// Map a [`TriggerError`] onto a gRPC [`Status`], preserving the failure kind so
/// a client can distinguish a bad request from an internal fault.
///
/// The `code` + `message` are the idiomatic gRPC surface (a client that does not
/// decode the structured detail still sees a sensible status). On top of that,
/// every status carries a faithful [`jammi_ai::wire`] trigger-error detail so a
/// remote `RemoteSession` reconstructs the *exact* [`TriggerError`] the
/// in-process path returns — the standard gRPC code set is too coarse to
/// distinguish, e.g., `PredicateParse` from `PredicateUnsupported`, or the two
/// engine-owned `#[from]` nests. The detail is built centrally here so the
/// faithful path covers the whole `TriggerError` enum from one place — the
/// trigger analogue of `map_engine_error`.
fn map_trigger_error(err: TriggerError) -> Status {
    let (code, message) = match &err {
        TriggerError::TopicNotFound(name) => (tonic::Code::NotFound, name.clone()),
        TriggerError::BatchSchemaMismatch(detail) => {
            (tonic::Code::InvalidArgument, detail.clone())
        }
        TriggerError::SchemaConflict { topic, detail } => (
            tonic::Code::FailedPrecondition,
            format!("schema conflict on {topic}: {detail}"),
        ),
        TriggerError::UnsupportedSchemaType { column, data_type } => (
            tonic::Code::InvalidArgument,
            format!("unsupported topic schema type for '{column}': {data_type}"),
        ),
        TriggerError::PublishTenantMismatch {
            topic,
            topic_tenant,
            publish_tenant,
        } => (
            tonic::Code::PermissionDenied,
            format!(
                "publish tenant mismatch on topic '{topic}': topic_tenant={topic_tenant:?}, publish_tenant={publish_tenant:?}"
            ),
        ),
        TriggerError::PredicateParse(detail) | TriggerError::PredicateUnsupported(detail) => {
            (tonic::Code::InvalidArgument, format!("predicate: {detail}"))
        }
        TriggerError::PredicateEval(detail) => {
            (tonic::Code::Internal, format!("predicate: {detail}"))
        }
        TriggerError::OffsetEvicted(n) => {
            (tonic::Code::FailedPrecondition, format!("offset {n} evicted"))
        }
        TriggerError::BackingTable(e) => (tonic::Code::Internal, format!("backing table: {e}")),
        TriggerError::Backend(e) => (tonic::Code::Internal, format!("backend: {e}")),
        TriggerError::Driver(detail) => (tonic::Code::Unavailable, format!("broker: {detail}")),
        TriggerError::Catalog(detail) => (tonic::Code::Internal, format!("catalog: {detail}")),
    };
    attach_trigger_detail(code, message, &err)
}

// Re-export `Duration` so the `Subscribe` test future can clamp its wait
// window without re-importing in tests.
pub(crate) const _SUBSCRIBE_BUFFER_DEPTH: usize = SUBSCRIBE_BUFFER;
pub(crate) const _SUBSCRIBE_WAIT_HINT: Duration = Duration::from_millis(10);
