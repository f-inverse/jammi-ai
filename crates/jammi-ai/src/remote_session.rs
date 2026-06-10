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
//!   `CatalogService.SetTenant` / `GetTenant` / `ClearTenant`; the binding is
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
use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::{Stream, StreamExt, TryStreamExt};
use tonic::codegen::InterceptedService;
use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status};

use std::collections::{BTreeMap, HashMap};

use jammi_db::catalog::eval_repo::PerQueryEvalRecord;
use jammi_db::catalog::model_repo::ModelRecord;
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
use crate::wire::proto::catalog::catalog_service_client::CatalogServiceClient;
use crate::wire::proto::catalog::{
    AddChannelColumnsRequest, AddSourceRequest, CreateMutableTableRequest, DescribeModelRequest,
    DescribeSourceRequest, DropMutableTableRequest, DropTopicRequest, ListChannelsRequest,
    ListModelsRequest, ListMutableTablesRequest, ListSourcesRequest, ListTopicsRequest,
    RegisterChannelRequest, RegisterTopicRequest, RemoveSourceRequest, SetTenantRequest, Tenant,
};
use crate::wire::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use crate::wire::proto::embedding::{
    encode_query_request::Input as ProtoEncodeInput, search_request::Query as ProtoSearchQuery,
    EncodeQueryRequest, GenerateEmbeddingsRequest, QueryVector, SearchRequest, SearchResponse,
};
use crate::wire::proto::eval as eval_pb;
use crate::wire::proto::eval::eval_service_client::EvalServiceClient;
use crate::wire::proto::inference::inference_service_client::InferenceServiceClient;
use crate::wire::proto::inference::InferRequest;
use crate::wire::proto::training::training_service_client::TrainingServiceClient;
use crate::wire::proto::training::{
    start_training_request::Spec as ProtoTrainingSpec, FineTuneSpec, StartTrainingRequest,
    TrainingStatusRequest,
};
use crate::wire::proto::trigger::trigger_service_client::TriggerServiceClient;
use crate::wire::proto::trigger::{PublishRequest, SubscribeRequest, TopicName};
use crate::wire::{
    audit_error_from_status, channel_from_proto, cohorts_to_proto, columns_to_proto,
    config_to_proto, decode_ipc_stream, decode_subscribed_batch, definition_list_from_proto,
    definition_to_proto, encode_ipc_stream, encode_publish_batch, error_from_status,
    eval_task_to_proto, method_to_proto, model_from_proto, model_task_to_proto, record_from_wire,
    result_table_from_proto, source_descriptor_from_proto, source_type_to_proto, topic_from_proto,
    trigger_error_from_status, SESSION_HEADER,
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

    /// The control-plane client: the tenant trio + server-info handshake, the
    /// source/model registry verbs, the channel-declaration verbs, the
    /// mutable-table lifecycle, and the topic-admin verbs all land here, mirroring
    /// the server's single [`CatalogService`].
    fn catalog_client(&self) -> CatalogServiceClient<InterceptedService<Channel, SessionHeader>> {
        CatalogServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    fn inference_client(
        &self,
    ) -> InferenceServiceClient<InterceptedService<Channel, SessionHeader>> {
        InferenceServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    fn eval_client(&self) -> EvalServiceClient<InterceptedService<Channel, SessionHeader>> {
        EvalServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    fn training_client(&self) -> TrainingServiceClient<InterceptedService<Channel, SessionHeader>> {
        TrainingServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
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
        self.catalog_client()
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
        self.catalog_client()
            .remove_source(RemoveSourceRequest {
                source_id: source_id.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn list_sources(&self) -> Result<Vec<SourceDescriptor>> {
        let resp = self
            .catalog_client()
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

    pub(crate) async fn list_models(&self) -> Result<Vec<ModelRecord>> {
        let resp = self
            .catalog_client()
            .list_models(ListModelsRequest {})
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        // Each entry rebuilds the same record the in-process path returns; a
        // corrupt task on an entry surfaces as the faithful status the decoder
        // builds.
        resp.models
            .into_iter()
            .map(|m| model_from_proto(m).map_err(|s| error_from_status(&s)))
            .collect()
    }

    pub(crate) async fn describe_model(&self, model_id: &str) -> Result<Option<ModelRecord>> {
        // The wire verb returns a `Model` for a present model and a `NotFound`
        // status for an absent one; the remote arm maps that one status code back
        // to `None` so the verb's `Option` shape matches the in-process path,
        // while any other failure decodes to its faithful `JammiError`.
        match self
            .catalog_client()
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
            .catalog_client()
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
            .catalog_client()
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

    // --- sql -------------------------------------------------------------

    /// Execute a SQL query over the Flight SQL lane and collect the terminal
    /// batches.
    ///
    /// `sql` does not ride a typed gRPC verb — per ADR-01 §3.2 the Flight SQL
    /// surface carries query/result. So this opens a [`FlightSqlServiceClient`]
    /// over the *same* tonic [`Channel`] the typed-RPC verbs use, stamps the
    /// [`SESSION_HEADER`] with [`Self::session_id`] — the identical id
    /// `set_tenant`/`bind_tenant` bound the tenant scope against — so the
    /// server's `TenantBoundProvider` resolves this query to that bound tenant,
    /// then runs `execute` → `do_get(ticket)` per endpoint and concatenates the
    /// streamed batches. Stamping the bound session id (not a fresh one) is what
    /// keeps a `--tenant A` query scoped to tenant A rather than silently
    /// unscoped.
    ///
    /// This is the supported single-session bind-then-query Flight path: one
    /// `RemoteSession` binds its tenant once and issues queries on its own
    /// session id, sidestepping the shared-binding race the multi-tenant
    /// concurrent Flight path carries (see
    /// [`jammi-server`'s `TenantBoundProvider`](https://docs.rs/jammi-server)).
    pub(crate) async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        let mut client = FlightSqlServiceClient::new(self.channel.clone());
        client.set_header(SESSION_HEADER, self.session_id().to_string());
        let info = client
            .execute(query.to_string(), None)
            .await
            .map_err(|e| JammiError::Other(format!("flight sql execute: {e}")))?;

        let mut batches = Vec::new();
        for endpoint in info.endpoint {
            let ticket = endpoint
                .ticket
                .ok_or_else(|| JammiError::Other("flight sql endpoint carried no ticket".into()))?;
            let stream = client
                .do_get(ticket)
                .await
                .map_err(|e| JammiError::Other(format!("flight sql do_get: {e}")))?;
            let endpoint_batches: Vec<RecordBatch> = stream
                .try_collect()
                .await
                .map_err(|e| JammiError::Other(format!("flight sql stream: {e}")))?;
            batches.extend(endpoint_batches);
        }
        Ok(batches)
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
        // The column-source fine-tune is the `FineTuneSpec` arm of the spec
        // oneof; built inline from the transport-neutral config vocabulary so a
        // thin wire-only client (no engine `TrainingSpec`) can still submit it.
        let resp = self
            .training_client()
            .start_training(StartTrainingRequest {
                spec: Some(ProtoTrainingSpec::FineTune(FineTuneSpec {
                    source: source.to_string(),
                    columns: columns.to_vec(),
                    method: method_to_proto(method) as i32,
                    task: model_task_to_proto(task) as i32,
                })),
                base_model: base_model.to_string(),
                config: config.as_ref().map(config_to_proto),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(FineTuneJobId(resp.job_id))
    }

    pub(crate) async fn fine_tune_status(&self, id: &FineTuneJobId) -> Result<String> {
        let resp = self
            .training_client()
            .training_status(TrainingStatusRequest {
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
            .catalog_client()
            .create_mutable_table(CreateMutableTableRequest {
                definition: Some(definition),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        MutableTableId::new(resp.mutable_table_id).map_err(JammiError::MutableTable)
    }

    pub(crate) async fn drop_mutable_table(&self, id: &MutableTableId) -> Result<()> {
        self.catalog_client()
            .drop_mutable_table(DropMutableTableRequest {
                mutable_table_id: id.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn list_mutable_tables(&self) -> Result<Vec<MutableTableDefinition>> {
        let resp = self
            .catalog_client()
            .list_mutable_tables(ListMutableTablesRequest {})
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        // Each entry rebuilds the same tenant-free definition the in-process
        // path returns; a corrupt entry surfaces as the faithful status the
        // decoder builds.
        resp.definitions
            .into_iter()
            .map(|d| definition_list_from_proto(d).map_err(|s| error_from_status(&s)))
            .collect()
    }

    // --- channels --------------------------------------------------------

    pub(crate) async fn register_channel(&self, spec: &ChannelSpec) -> Result<()> {
        self.catalog_client()
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
        self.catalog_client()
            .add_channel_columns(AddChannelColumnsRequest {
                channel_id: channel.as_str().to_string(),
                columns: columns_to_proto(new_columns),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn list_channels(&self) -> Result<Vec<ChannelSpec>> {
        let resp = self
            .catalog_client()
            .list_channels(ListChannelsRequest {})
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        // Each entry rebuilds the same spec the in-process path returns; a
        // corrupt entry surfaces as the faithful status the decoder builds.
        resp.channels
            .into_iter()
            .map(|c| channel_from_proto(c).map_err(|s| error_from_status(&s)))
            .collect()
    }

    // --- tenant ----------------------------------------------------------

    pub(crate) async fn bind_tenant(&self, t: TenantId) -> Result<()> {
        self.catalog_client()
            .set_tenant(SetTenantRequest {
                tenant: Some(Tenant { id: t.to_string() }),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn unbind_tenant(&self) -> Result<()> {
        self.catalog_client()
            .clear_tenant(())
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn tenant(&self) -> Result<Option<TenantId>> {
        let resp = self
            .catalog_client()
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

    // --- topics (control plane) ------------------------------------------

    pub(crate) async fn register_topic(
        &self,
        topic: &TopicDefinition,
    ) -> std::result::Result<(), TriggerError> {
        let schema =
            encode_ipc_stream(&topic.schema, &[]).map_err(|s| trigger_error_from_status(&s))?;
        self.catalog_client()
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
            .catalog_client()
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
        self.catalog_client()
            .drop_topic(DropTopicRequest {
                topic_id: topic_id.to_string(),
                if_exists: false,
            })
            .await
            .map_err(|s| trigger_error_from_status(&s))?;
        Ok(())
    }

    // --- trigger (publish / subscribe) -----------------------------------

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
        replay_only: bool,
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
                // When set, the server drives its finite replay-only drain and
                // closes the stream, so the bounded `--no-follow` path reads a
                // subscription that terminates rather than the live tail.
                replay_only,
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
