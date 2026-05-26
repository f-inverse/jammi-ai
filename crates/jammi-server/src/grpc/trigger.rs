//! `TriggerService` gRPC implementation.
//!
//! Three methods land on the wire surface per ADR-01 §5.1:
//!
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

use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use datafusion::execution::context::SessionContext;
use futures::Stream;
use futures::StreamExt;
use prost_types::Timestamp;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use jammi_db::catalog::topic_repo::TopicRepo;
use jammi_db::trigger::{
    DeliveredBatch, Offset, Predicate, Publisher, Subscriber, TopicDefinition, TriggerError,
};
use jammi_db::TenantId;

use crate::grpc::proto::trigger::trigger_service_server::TriggerService;
use crate::grpc::proto::trigger::{
    ArrowBatch, ListTopicsRequest, ListTopicsResponse, PublishRequest, PublishResponse,
    SubscribeRequest, SubscribedBatch, TopicName,
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
    // The Flight IPC payload starts with the schema header; the simplest
    // way to round-trip a single batch is the `StreamReader` format which
    // serializes one stream message containing the schema followed by a
    // batch message. Per ADR-01 §5.1 the wire pairing is `data_header` +
    // `data_body`; we concatenate them to feed `StreamReader` which expects
    // a single contiguous IPC stream.
    let mut bytes = Vec::with_capacity(wire.data_header.len() + wire.data_body.len());
    bytes.extend_from_slice(&wire.data_header);
    bytes.extend_from_slice(&wire.data_body);
    let cursor = std::io::Cursor::new(bytes);
    let mut reader = StreamReader::try_new(cursor, None)
        .map_err(|e| Status::invalid_argument(format!("batch decode: {e}")))?;
    let batch = reader
        .next()
        .ok_or_else(|| Status::invalid_argument("batch IPC stream contains no batch"))?
        .map_err(|e| Status::invalid_argument(format!("batch decode: {e}")))?;
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
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, schema.as_ref())
            .map_err(|e| Status::internal(format!("batch encode: {e}")))?;
        writer
            .write(&delivered.batch)
            .map_err(|e| Status::internal(format!("batch encode: {e}")))?;
        writer
            .finish()
            .map_err(|e| Status::internal(format!("batch encode: {e}")))?;
    }
    Ok(SubscribedBatch {
        offset: delivered.offset.value(),
        produced_at: Some(to_proto_timestamp(delivered.produced_at)),
        batch: Some(ArrowBatch {
            // The `StreamWriter` format is a single contiguous IPC stream;
            // we surface it as a single `data_body` payload and leave
            // `data_header` empty — `decode_arrow_batch` concatenates the
            // two anyway, so the wire shape is symmetric.
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
