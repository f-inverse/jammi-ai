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
//!   client cancels.
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

use arrow::record_batch::RecordBatch;
use datafusion::execution::context::SessionContext;
use futures::Stream;
use futures::StreamExt;
use prost_types::Timestamp;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use jammi_db::catalog::topic_repo::TopicRepo;
use jammi_db::trigger::ids::TopicId;
use jammi_db::trigger::{
    DeliveredBatch, Offset, Predicate, Publisher, Subscriber, TopicDefinition, TriggerError,
};
use jammi_db::TenantId;

use crate::grpc::proto::trigger::trigger_service_server::TriggerService;
use crate::grpc::proto::trigger::{
    ArrowBatch, DropTopicRequest, ListTopicsRequest, ListTopicsResponse, PublishRequest,
    PublishResponse, RegisterTopicRequest, RegisterTopicResponse, SubscribeRequest,
    SubscribedBatch, TopicName,
};
use crate::grpc::session::SessionTenant;
use crate::grpc::wire::{decode_ipc_schema, decode_ipc_stream, encode_ipc_stream};

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
        let topic = TopicDefinition {
            id: TopicId::new(),
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
        if req.name.is_empty() {
            return Err(Status::invalid_argument("name is required"));
        }
        let topic = self
            .topic_repo
            .lookup_by_name(&req.name, tenant)
            .await
            .map_err(map_trigger_error)?;
        match topic {
            Some(topic) => self
                .topic_repo
                .drop_topic(topic.id, tenant)
                .await
                .map_err(map_trigger_error)
                .map(Response::new),
            None if req.if_exists => Ok(Response::new(())),
            None => Err(Status::not_found(format!("topic '{}' not found", req.name))),
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
        let record_batch = decode_arrow_batch(&batch, &topic)?;
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

        let mut inner = self
            .subscriber
            .subscribe(&topic, predicate, from_offset)
            .await
            .map_err(map_trigger_error)?;

        let (tx, rx) = mpsc::channel::<Result<SubscribedBatch, Status>>(SUBSCRIBE_BUFFER);
        let topic_schema = Arc::clone(&topic.schema);
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
        let names = topics
            .into_iter()
            .map(|t| TopicName { name: t.name })
            .collect();
        Ok(Response::new(ListTopicsResponse {
            topics: names,
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

fn decode_arrow_batch(wire: &ArrowBatch, topic: &TopicDefinition) -> Result<RecordBatch, Status> {
    // Per ADR-01 §5.1 the wire pairing is `data_header` + `data_body`; the
    // shared decoder concatenates them and reads the IPC stream. A publish
    // carries exactly one batch — reject an empty or multi-batch payload so a
    // malformed publish is a typed client error, not a silent partial write.
    let mut batches = decode_ipc_stream(&wire.data_header, &wire.data_body)?;
    let batch = match batches.len() {
        1 => batches.pop().expect("len checked == 1"),
        0 => {
            return Err(Status::invalid_argument(
                "batch IPC stream contains no batch",
            ))
        }
        n => {
            return Err(Status::invalid_argument(format!(
                "publish carries exactly one batch, got {n}"
            )))
        }
    };
    if batch.schema().as_ref() != topic.schema.as_ref() {
        return Err(Status::invalid_argument(
            "batch schema does not match topic schema",
        ));
    }
    Ok(batch)
}

fn encode_delivered_batch(
    schema: &arrow_schema::SchemaRef,
    delivered: DeliveredBatch,
) -> Result<SubscribedBatch, Status> {
    // The `StreamWriter` format is a single contiguous IPC stream; surface it
    // as one `data_body` payload with `data_header` empty — the shared decoder
    // concatenates the two anyway, so the wire shape is symmetric.
    let buf = encode_ipc_stream(schema, std::slice::from_ref(&delivered.batch))?;
    Ok(SubscribedBatch {
        offset: delivered.offset.value(),
        produced_at: Some(to_proto_timestamp(delivered.produced_at)),
        batch: Some(ArrowBatch {
            data_header: Vec::new(),
            data_body: buf,
            app_metadata: Vec::new(),
        }),
    })
}

fn to_proto_timestamp(dt: chrono::DateTime<chrono::Utc>) -> Timestamp {
    let seconds = dt.timestamp();
    let nanos = dt.timestamp_subsec_nanos() as i32;
    Timestamp { seconds, nanos }
}

fn map_trigger_error(err: TriggerError) -> Status {
    match err {
        TriggerError::TopicNotFound(name) => Status::not_found(name),
        TriggerError::BatchSchemaMismatch(detail) => Status::invalid_argument(detail),
        TriggerError::SchemaConflict { topic, detail } => {
            Status::failed_precondition(format!("schema conflict on {topic}: {detail}"))
        }
        TriggerError::UnsupportedSchemaType { column, data_type } => Status::invalid_argument(
            format!("unsupported topic schema type for '{column}': {data_type}"),
        ),
        TriggerError::PublishTenantMismatch {
            topic,
            topic_tenant,
            publish_tenant,
        } => Status::permission_denied(format!(
            "publish tenant mismatch on topic '{topic}': topic_tenant={topic_tenant:?}, publish_tenant={publish_tenant:?}"
        )),
        TriggerError::PredicateParse(detail) | TriggerError::PredicateUnsupported(detail) => {
            Status::invalid_argument(format!("predicate: {detail}"))
        }
        TriggerError::PredicateEval(detail) => Status::internal(format!("predicate: {detail}")),
        TriggerError::OffsetEvicted(n) => {
            Status::failed_precondition(format!("offset {n} evicted"))
        }
        TriggerError::BackingTable(e) => Status::internal(format!("backing table: {e}")),
        TriggerError::Backend(e) => Status::internal(format!("backend: {e}")),
        TriggerError::Driver(detail) => Status::unavailable(format!("broker: {detail}")),
        TriggerError::Catalog(detail) => Status::internal(format!("catalog: {detail}")),
    }
}

// Re-export `Duration` so the `Subscribe` test future can clamp its wait
// window without re-importing in tests.
pub(crate) const _SUBSCRIBE_BUFFER_DEPTH: usize = SUBSCRIBE_BUFFER;
pub(crate) const _SUBSCRIBE_WAIT_HINT: Duration = Duration::from_millis(10);
