//! The remote [`Session`] transport: a gRPC client that drives an engine
//! hosted behind the `jammi.v1` wire surface.
//!
//! [`RemoteSession`] is the network peer of [`crate::LocalSession`]: it
//! implements the same verbs over the same request/result shapes, so the
//! [`Session`] enum dispatches to either arm and a consumer cannot tell which
//! transport backs it. Three properties make that interchangeability real:
//!
//! * **Faithful errors.** Every failure decodes the structured
//!   [`crate::wire::error_from_status`] detail the server attaches, so a remote
//!   call returns the *exact* [`JammiError`](jammi_db::error::JammiError)
//!   variant the in-process path would — never a lossy gRPC-code-category guess.
//! * **Tenant over the wire.** The tenant trio maps to
//!   `SessionService.SetTenant` / `GetTenant` / `ClearTenant`; the binding is
//!   keyed by an opaque per-session id carried in the
//!   [`SESSION_HEADER`](crate::wire::SESSION_HEADER), never in a request body.
//! * **Shared conversions.** Request encode / response decode reuse the
//!   [`crate::wire`] conversions the server's receive side uses — neither side
//!   reimplements a mapping.
//!
//! The wired verbs span the embeddings / encode-query / search / add-source /
//! remove-source surface, the tenant trio, the `JammiError`-returning compute verbs
//! (inference, the four eval verbs, fine-tune start + status, the mutable-table
//! create/drop lifecycle, and the channel register / add-columns verbs), the
//! topics surface (register / drop / publish / list) with its server-streaming
//! `subscribe`, and the audit surface (log / fetch). The two surfaces that
//! return their own error types — `TriggerError` and `AuditError` — decode the
//! same way `JammiError` does: the server attaches a structured detail
//! (`TriggerErrorDetail` / `AuditErrorDetail`) and the client reconstructs the
//! exact variant, including a `subscribe` stream's terminal failure. The only
//! verbs still unreachable here are the Flight-SQL-shaped `sql` / `read_vectors`
//! (wired on that lane, not the typed-RPC surface).

use std::pin::Pin;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use futures::{Stream, StreamExt};
use tonic::codegen::InterceptedService;
use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status};

use std::collections::{BTreeMap, HashMap};

use jammi_db::catalog::eval_repo::PerQueryEvalRecord;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::catalog::source_repo::SourceDescriptor;
use jammi_db::error::{JammiError, Result};
use jammi_db::source::{SourceConnection, SourceType};
use jammi_db::store::mutable::{MutableTableDefinition, MutableTableId};
use jammi_db::trigger::{DeliveredBatch, Offset, Predicate, TopicDefinition, TriggerError};
use jammi_db::{AuditError, ChannelId, ModelTask, PerQueryAudit, ServerInfo, TenantId, TopicId};

use crate::eval::{CompareEvalReport, EmbeddingEvalReport, EvalTask, InferenceEvalReport};
use crate::fine_tune::{FineTuneConfig, FineTuneMethod};
use crate::local_session::{ChannelColumn, ChannelSpec, FineTuneJobId};
use crate::wire::proto::audit::audit_service_client::AuditServiceClient;
use crate::wire::proto::audit::{
    AuditFetchByQueryIdRequest, AuditFetchRecentRequest, AuditLogRequest,
};
use crate::wire::proto::channel::channel_service_client::ChannelServiceClient;
use crate::wire::proto::channel::{AddChannelColumnsRequest, RegisterChannelRequest};
use crate::wire::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use crate::wire::proto::embedding::{
    encode_query_request::Input as ProtoEncodeInput, search_request::Query as ProtoSearchQuery,
    AddSourceRequest, DescribeSourceRequest, EncodeQueryRequest, GenerateEmbeddingsRequest,
    ListSourcesRequest, QueryVector, RemoveSourceRequest, SearchRequest, SearchResponse,
};
use crate::wire::proto::eval as eval_pb;
use crate::wire::proto::eval::eval_service_client::EvalServiceClient;
use crate::wire::proto::fine_tune::fine_tune_service_client::FineTuneServiceClient;
use crate::wire::proto::fine_tune::{FineTuneStatusRequest, StartFineTuneRequest};
use crate::wire::proto::inference::inference_service_client::InferenceServiceClient;
use crate::wire::proto::inference::InferRequest;
use crate::wire::proto::mutable_table::mutable_table_service_client::MutableTableServiceClient;
use crate::wire::proto::mutable_table::{CreateMutableTableRequest, DropMutableTableRequest};
use crate::wire::proto::session::session_service_client::SessionServiceClient;
use crate::wire::proto::session::{SetTenantRequest, Tenant};
use crate::wire::proto::trigger::trigger_service_client::TriggerServiceClient;
use crate::wire::proto::trigger::{
    DropTopicRequest, ListTopicsRequest, PublishRequest, RegisterTopicRequest, SubscribeRequest,
    TopicName,
};
use crate::wire::{
    audit_error_from_status, cohorts_to_proto, columns_to_proto, config_to_proto,
    decode_ipc_stream, decode_subscribed_batch, definition_to_proto, encode_ipc_stream,
    encode_publish_batch, error_from_status, eval_task_to_proto, method_to_proto,
    model_task_to_proto, record_from_wire, result_table_from_proto, source_descriptor_from_proto,
    source_type_to_proto, topic_from_proto, trigger_error_from_status, SESSION_HEADER,
};
use crate::{Modality, QueryInput, SearchQuery, SearchRequest as SessionSearch};

/// Injects the [`SESSION_HEADER`] carrying this client's opaque session id on
/// every outbound request, so the server's tenant interceptor resolves the same
/// binding the tenant trio set. One per [`RemoteSession`]; cheap to clone (a
/// pre-parsed metadata value).
#[derive(Clone)]
struct SessionHeader {
    id: MetadataValue<tonic::metadata::Ascii>,
}

impl Interceptor for SessionHeader {
    fn call(&mut self, mut request: Request<()>) -> std::result::Result<Request<()>, Status> {
        request
            .metadata_mut()
            .insert(SESSION_HEADER, self.id.clone());
        Ok(request)
    }
}

/// A [`Session`] backed by a remote engine over gRPC.
///
/// [`Session`]: crate::Session
pub struct RemoteSession {
    channel: Channel,
    header: SessionHeader,
    /// The opaque session id this client minted, kept so [`Self::session_id`]
    /// can report the key the server binds tenant state against; the
    /// interceptor carries the parsed copy.
    session_id: String,
}

impl RemoteSession {
    /// Connect to a `jammi.v1` gRPC endpoint and mint a fresh session id.
    ///
    /// `endpoint` is any value an [`Endpoint`] accepts (e.g. an
    /// `"http://host:port"` string). The transport is native tonic here; the
    /// session id is a v4 UUID so two clients against one server never collide
    /// on a tenant binding.
    pub async fn connect(endpoint: impl Into<Endpoint>) -> Result<Self> {
        let channel = endpoint
            .into()
            .connect()
            .await
            .map_err(|e| JammiError::Config(format!("connect to jammi endpoint: {e}")))?;
        let session_id = uuid::Uuid::new_v4().to_string();
        let id: MetadataValue<tonic::metadata::Ascii> = session_id
            .parse()
            .map_err(|e| JammiError::Config(format!("session id metadata: {e}")))?;
        Ok(Self {
            channel,
            header: SessionHeader { id },
            session_id,
        })
    }

    /// The opaque session id this client minted. The server keys the tenant
    /// binding by it; two `RemoteSession`s over the same channel are isolated
    /// because each mints its own.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    fn embedding_client(
        &self,
    ) -> EmbeddingServiceClient<InterceptedService<Channel, SessionHeader>> {
        EmbeddingServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    fn session_client(&self) -> SessionServiceClient<InterceptedService<Channel, SessionHeader>> {
        SessionServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    fn inference_client(
        &self,
    ) -> InferenceServiceClient<InterceptedService<Channel, SessionHeader>> {
        InferenceServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    fn eval_client(&self) -> EvalServiceClient<InterceptedService<Channel, SessionHeader>> {
        EvalServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    fn fine_tune_client(
        &self,
    ) -> FineTuneServiceClient<InterceptedService<Channel, SessionHeader>> {
        FineTuneServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    fn mutable_table_client(
        &self,
    ) -> MutableTableServiceClient<InterceptedService<Channel, SessionHeader>> {
        MutableTableServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    fn channel_client(&self) -> ChannelServiceClient<InterceptedService<Channel, SessionHeader>> {
        ChannelServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    fn trigger_client(&self) -> TriggerServiceClient<InterceptedService<Channel, SessionHeader>> {
        TriggerServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    fn audit_client(&self) -> AuditServiceClient<InterceptedService<Channel, SessionHeader>> {
        AuditServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    // --- sources ---------------------------------------------------------

    pub(crate) async fn add_source(
        &self,
        source_id: &str,
        source_type: SourceType,
        connection: SourceConnection,
    ) -> Result<()> {
        self.embedding_client()
            .add_source(AddSourceRequest {
                source_id: source_id.to_string(),
                source_kind: source_type_to_proto(source_type) as i32,
                connection: Some(connection.into()),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn remove_source(&self, source_id: &str) -> Result<()> {
        self.embedding_client()
            .remove_source(RemoveSourceRequest {
                source_id: source_id.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn list_sources(&self) -> Result<Vec<SourceDescriptor>> {
        let resp = self
            .embedding_client()
            .list_sources(ListSourcesRequest {})
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        // Each entry rebuilds the same descriptor the in-process path returns;
        // a corrupt entry surfaces as the faithful status the decoder builds.
        resp.sources
            .into_iter()
            .map(|d| source_descriptor_from_proto(d).map_err(|s| error_from_status(&s)))
            .collect()
    }

    pub(crate) async fn describe_source(
        &self,
        source_id: &str,
    ) -> Result<Option<SourceDescriptor>> {
        // The wire verb returns a `SourceDescriptor` for a present source and a
        // `NotFound` status for an absent one; the remote arm maps that one
        // status code back to `None` so the verb's `Option` shape matches the
        // in-process path, while any other failure decodes to its faithful
        // `JammiError`.
        match self
            .embedding_client()
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

    pub(crate) async fn server_info(&self) -> Result<ServerInfo> {
        let resp = self
            .session_client()
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

    // --- embeddings ------------------------------------------------------

    pub(crate) async fn generate_embeddings(
        &self,
        source_id: &str,
        model_id: &str,
        columns: &[String],
        key_column: &str,
        modality: Modality,
    ) -> Result<ResultTableRecord> {
        let table = self
            .embedding_client()
            .generate_embeddings(GenerateEmbeddingsRequest {
                source_id: source_id.to_string(),
                model_id: model_id.to_string(),
                columns: columns.to_vec(),
                key_column: key_column.to_string(),
                modality: proto_modality(modality) as i32,
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        // The wire `ResultTable` is self-describing in `task`, so the
        // reconstruction recovers it from the message rather than from the
        // requested modality; a corrupt task surfaces as the faithful status.
        result_table_from_proto(table).map_err(|s| error_from_status(&s))
    }

    pub(crate) async fn encode_query(
        &self,
        model_id: &str,
        input: QueryInput,
        modality: Modality,
    ) -> Result<Vec<f32>> {
        let input = match input {
            QueryInput::Text(text) => ProtoEncodeInput::Text(text),
            QueryInput::Bytes(bytes) => ProtoEncodeInput::Data(bytes),
        };
        let resp = self
            .embedding_client()
            .encode_query(EncodeQueryRequest {
                model_id: model_id.to_string(),
                modality: proto_modality(modality) as i32,
                input: Some(input),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(resp.embedding)
    }

    // --- search ----------------------------------------------------------

    pub(crate) async fn search(&self, request: SessionSearch) -> Result<Vec<RecordBatch>> {
        let SessionSearch {
            source_id,
            query,
            k,
            filter,
            select,
        } = request;
        let query = match query {
            SearchQuery::Vector(values) => ProtoSearchQuery::QueryVector(QueryVector { values }),
            SearchQuery::RowKey(key) => ProtoSearchQuery::RowKey(key),
        };
        let resp = self
            .embedding_client()
            .search(SearchRequest {
                source_id,
                query: Some(query),
                k: k as u32,
                filter,
                select: select.clone(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        hits_to_batch(resp, &select)
    }

    // --- inference -------------------------------------------------------

    pub(crate) async fn infer(
        &self,
        source_id: &str,
        model_id: &str,
        task: ModelTask,
        content_columns: &[String],
        key_column: &str,
    ) -> Result<Vec<RecordBatch>> {
        let resp = self
            .inference_client()
            .infer(InferRequest {
                source_id: source_id.to_string(),
                model_id: model_id.to_string(),
                task: model_task_to_proto(task) as i32,
                columns: content_columns.to_vec(),
                key_column: key_column.to_string(),
                tenant_id: String::new(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        let batch = resp.result.unwrap_or_default();
        decode_ipc_stream(&batch.data_header, &batch.data_body).map_err(|s| error_from_status(&s))
    }

    // --- fine-tune -------------------------------------------------------

    pub(crate) async fn fine_tune(
        &self,
        source: &str,
        base_model: &str,
        columns: &[String],
        method: FineTuneMethod,
        task: ModelTask,
        config: Option<FineTuneConfig>,
    ) -> Result<FineTuneJobId> {
        let resp = self
            .fine_tune_client()
            .start_fine_tune(StartFineTuneRequest {
                source_id: source.to_string(),
                base_model: base_model.to_string(),
                columns: columns.to_vec(),
                method: method_to_proto(method) as i32,
                task: model_task_to_proto(task) as i32,
                config: config.as_ref().map(config_to_proto),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(FineTuneJobId(resp.job_id))
    }

    pub(crate) async fn fine_tune_status(&self, id: &FineTuneJobId) -> Result<String> {
        let resp = self
            .fine_tune_client()
            .fine_tune_status(FineTuneStatusRequest {
                job_id: id.0.clone(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(resp.status)
    }

    // --- eval ------------------------------------------------------------

    pub(crate) async fn eval_embeddings(
        &self,
        source_id: &str,
        embedding_table: Option<&str>,
        golden_source: &str,
        k: usize,
        cohorts: &HashMap<String, BTreeMap<String, String>>,
    ) -> Result<EmbeddingEvalReport> {
        let resp = self
            .eval_client()
            .eval_embeddings(eval_pb::EvalEmbeddingsRequest {
                source_id: source_id.to_string(),
                embedding_table: embedding_table.unwrap_or_default().to_string(),
                golden_source: golden_source.to_string(),
                k: k as u32,
                cohorts: cohorts_to_proto(cohorts),
                tenant_id: String::new(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        resp.try_into()
    }

    pub(crate) async fn eval_per_query(
        &self,
        eval_run_id: &str,
    ) -> Result<Vec<PerQueryEvalRecord>> {
        let resp = self
            .eval_client()
            .eval_per_query(eval_pb::EvalPerQueryRequest {
                eval_run_id: eval_run_id.to_string(),
                tenant_id: String::new(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(resp.records.into_iter().map(Into::into).collect())
    }

    pub(crate) async fn eval_inference(
        &self,
        model_id: &str,
        source_id: &str,
        columns: &[String],
        task: EvalTask,
        golden_source: &str,
        label_column: &str,
    ) -> Result<InferenceEvalReport> {
        let resp = self
            .eval_client()
            .eval_inference(eval_pb::EvalInferenceRequest {
                model_id: model_id.to_string(),
                source_id: source_id.to_string(),
                columns: columns.to_vec(),
                task: eval_task_to_proto(task) as i32,
                golden_source: golden_source.to_string(),
                label_column: label_column.to_string(),
                tenant_id: String::new(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        resp.try_into()
    }

    pub(crate) async fn eval_compare(
        &self,
        embedding_tables: &[String],
        source_id: &str,
        golden_source: &str,
        k: usize,
    ) -> Result<CompareEvalReport> {
        let resp = self
            .eval_client()
            .eval_compare(eval_pb::EvalCompareRequest {
                embedding_tables: embedding_tables.to_vec(),
                source_id: source_id.to_string(),
                golden_source: golden_source.to_string(),
                k: k as u32,
                tenant_id: String::new(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        resp.try_into()
    }

    // --- mutable tables --------------------------------------------------

    pub(crate) async fn create_mutable_table(
        &self,
        def: MutableTableDefinition,
    ) -> Result<MutableTableId> {
        let definition = definition_to_proto(&def).map_err(|s| error_from_status(&s))?;
        let resp = self
            .mutable_table_client()
            .create_mutable_table(CreateMutableTableRequest {
                definition: Some(definition),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        MutableTableId::new(resp.mutable_table_id).map_err(JammiError::MutableTable)
    }

    pub(crate) async fn drop_mutable_table(&self, id: &MutableTableId) -> Result<()> {
        self.mutable_table_client()
            .drop_mutable_table(DropMutableTableRequest {
                mutable_table_id: id.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    // --- channels --------------------------------------------------------

    pub(crate) async fn register_channel(&self, spec: &ChannelSpec) -> Result<()> {
        self.channel_client()
            .register_channel(RegisterChannelRequest {
                channel_id: spec.id.as_str().to_string(),
                priority: spec.priority,
                columns: columns_to_proto(&spec.columns),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn add_channel_columns(
        &self,
        channel: &ChannelId,
        new_columns: &[ChannelColumn],
    ) -> Result<()> {
        self.channel_client()
            .add_channel_columns(AddChannelColumnsRequest {
                channel_id: channel.as_str().to_string(),
                columns: columns_to_proto(new_columns),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    // --- tenant ----------------------------------------------------------

    pub(crate) async fn bind_tenant(&self, t: TenantId) -> Result<()> {
        self.session_client()
            .set_tenant(SetTenantRequest {
                tenant: Some(Tenant { id: t.to_string() }),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn unbind_tenant(&self) -> Result<()> {
        self.session_client()
            .clear_tenant(())
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn tenant(&self) -> Result<Option<TenantId>> {
        let resp = self
            .session_client()
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

    // --- trigger ---------------------------------------------------------

    pub(crate) async fn register_topic(
        &self,
        topic: &TopicDefinition,
    ) -> std::result::Result<(), TriggerError> {
        let schema =
            encode_ipc_stream(&topic.schema, &[]).map_err(|s| trigger_error_from_status(&s))?;
        self.trigger_client()
            .register_topic(RegisterTopicRequest {
                name: topic.name.clone(),
                schema,
                broker_metadata: topic.broker_metadata.clone().into_iter().collect(),
                // Carry the caller-minted id so the topic's identity matches the
                // in-process path; a later `drop_topic(topic.id)` then resolves.
                topic_id: topic.id.to_string(),
            })
            .await
            .map_err(|s| trigger_error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn list_topics(
        &self,
    ) -> std::result::Result<Vec<TopicDefinition>, TriggerError> {
        let resp = self
            .trigger_client()
            .list_topics(ListTopicsRequest {
                page_size: 0,
                page_token: String::new(),
                // Tenant scope rides on the session header, not the body.
                tenant_id: String::new(),
            })
            .await
            .map_err(|s| trigger_error_from_status(&s))?
            .into_inner();
        // Each page entry is a fully-materialized `Topic` (id / name / schema /
        // tenant / broker_metadata), so the remote arm reconstructs the same
        // `Vec<TopicDefinition>` the in-process path returns — a corrupt entry
        // surfaces as the faithful `Status` the decoder builds.
        resp.topics
            .into_iter()
            .map(|t| topic_from_proto(t).map_err(|s| trigger_error_from_status(&s)))
            .collect()
    }

    pub(crate) async fn drop_topic(
        &self,
        topic_id: TopicId,
    ) -> std::result::Result<(), TriggerError> {
        self.trigger_client()
            .drop_topic(DropTopicRequest {
                topic_id: topic_id.to_string(),
                if_exists: false,
            })
            .await
            .map_err(|s| trigger_error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn publish(
        &self,
        topic: &TopicDefinition,
        batch: RecordBatch,
    ) -> std::result::Result<Offset, TriggerError> {
        let wire_batch = encode_publish_batch(&batch).map_err(|s| trigger_error_from_status(&s))?;
        let resp = self
            .trigger_client()
            .publish(PublishRequest {
                topic: Some(TopicName {
                    name: topic.name.clone(),
                }),
                batch: Some(wire_batch),
                // Tenant scope rides on the session header, not the body.
                tenant_id: String::new(),
            })
            .await
            .map_err(|s| trigger_error_from_status(&s))?
            .into_inner();
        let committed_at = resp
            .committed_at
            .as_ref()
            .map(crate::wire::from_proto_timestamp)
            .transpose()
            .map_err(|s| trigger_error_from_status(&s))?
            .ok_or_else(|| TriggerError::Driver("publish response missing committed_at".into()))?;
        Ok(Offset::new(resp.offset, committed_at))
    }

    pub(crate) async fn subscribe(
        &self,
        topic: &TopicDefinition,
        predicate: Predicate,
        from_offset: Option<Offset>,
    ) -> std::result::Result<
        Pin<Box<dyn Stream<Item = std::result::Result<DeliveredBatch, TriggerError>> + Send>>,
        TriggerError,
    > {
        let streaming = self
            .trigger_client()
            .subscribe(SubscribeRequest {
                topic: Some(TopicName {
                    name: topic.name.clone(),
                }),
                // The predicate crosses the wire as the SQL it was parsed from
                // (empty == match-all); the server re-parses it against the same
                // topic schema, so the in-process and remote filters are identical.
                predicate: predicate.source_sql().unwrap_or("").to_string(),
                from_offset: from_offset.map(|o| o.value()),
                tenant_id: String::new(),
            })
            .await
            .map_err(|s| trigger_error_from_status(&s))?
            .into_inner();
        // Map each streamed item into the same `Result<DeliveredBatch, TriggerError>`
        // a local subscription yields. A terminal `tonic::Status` (a mid-stream
        // or final failure) reconstructs to its faithful `TriggerError` via the
        // attached detail — never a gRPC-code-category guess; a payload-decode
        // failure surfaces as the faithful `Status` the decoder built.
        let mapped = streaming.map(|item| match item {
            Ok(wire) => decode_subscribed_batch(wire).map_err(|s| trigger_error_from_status(&s)),
            Err(status) => Err(trigger_error_from_status(&status)),
        });
        Ok(Box::pin(mapped))
    }

    // --- audit -----------------------------------------------------------

    pub(crate) async fn audit_log(
        &self,
        records: Vec<PerQueryAudit>,
    ) -> std::result::Result<(), AuditError> {
        self.audit_client()
            .audit_log(AuditLogRequest {
                records: records.into_iter().map(Into::into).collect(),
            })
            .await
            .map_err(|s| audit_error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn audit_fetch_by_query_id(
        &self,
        query_id: uuid::Uuid,
    ) -> std::result::Result<Option<PerQueryAudit>, AuditError> {
        let resp = self
            .audit_client()
            .audit_fetch_by_query_id(AuditFetchByQueryIdRequest {
                query_id: query_id.to_string(),
            })
            .await
            .map_err(|s| audit_error_from_status(&s))?
            .into_inner();
        resp.record.map(record_from_wire).transpose()
    }

    pub(crate) async fn audit_fetch_recent(
        &self,
        limit: usize,
    ) -> std::result::Result<Vec<PerQueryAudit>, AuditError> {
        let resp = self
            .audit_client()
            .audit_fetch_recent(AuditFetchRecentRequest {
                limit: limit as u32,
            })
            .await
            .map_err(|s| audit_error_from_status(&s))?
            .into_inner();
        resp.records.into_iter().map(record_from_wire).collect()
    }
}

/// Map the engine [`Modality`] onto the wire enum. Encode is total (the engine
/// never holds an unspecified modality), so this is a plain `From`-shaped match
/// rather than the fallible decode the server side runs.
fn proto_modality(modality: Modality) -> crate::wire::proto::embedding::Modality {
    use crate::wire::proto::embedding::Modality as Pb;
    match modality {
        Modality::Text => Pb::Text,
        Modality::Image => Pb::Image,
        Modality::Audio => Pb::Audio,
    }
}

/// Rebuild the terminal `Vec<RecordBatch>` shape [`crate::Session::search`]
/// returns from the wire `SearchResponse`.
///
/// The wire surface carries each hit as `key` + `score` + a `columns` map of
/// stringified projections (the search-consumption shape edge runtimes reach),
/// so the remote arm rehydrates one batch with the `_row_id` (key) and
/// `similarity` (score) columns the local arm's hydrated batch carries, plus a
/// `Utf8` column per requested `select` name. Both arms therefore yield a batch
/// keyed and scored identically; the remote arm's projected columns are the
/// wire's stringified form (the server stringified them on the way out), which
/// is the only form that crosses this verb's wire.
fn hits_to_batch(resp: SearchResponse, select: &[String]) -> Result<Vec<RecordBatch>> {
    if resp.hits.is_empty() {
        return Ok(Vec::new());
    }
    let keys: Vec<&str> = resp.hits.iter().map(|h| h.key.as_str()).collect();
    let scores: Vec<f32> = resp.hits.iter().map(|h| h.score).collect();

    let mut fields: Vec<Field> = vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("similarity", DataType::Float32, false),
    ];
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(keys)),
        Arc::new(Float32Array::from(scores)),
    ];
    for name in select {
        let values: Vec<String> = resp
            .hits
            .iter()
            .map(|h| h.columns.get(name).cloned().unwrap_or_default())
            .collect();
        fields.push(Field::new(name, DataType::Utf8, false));
        arrays.push(Arc::new(StringArray::from(values)));
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, arrays)
        .map_err(|e| JammiError::Other(format!("rebuild search batch: {e}")))?;
    Ok(vec![batch])
}
