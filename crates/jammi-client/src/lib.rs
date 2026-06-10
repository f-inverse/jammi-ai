//! The Jammi data-plane client.
//!
//! [`DataClient`] is the network peer of the embedded session for the data
//! verbs: SQL (over Flight SQL), embeddings / encode / search, inference,
//! fine-tune submit + status, the eval verbs, the trigger publish / subscribe
//! surface, and audit. It composes a [`jammi_admin::CatalogClient`] over the
//! *same* [`jammi_wire::SessionTransport`] for the control verbs and the tenant
//! trio, so a tenant bound through `bind_tenant` is observed by every data verb
//! on the same session id. It is candle-free — it speaks the gRPC + Flight SQL
//! wire only and pulls no embedded engine.
//!
//! Three properties make it interchangeable with the embedded session:
//!
//! * **Faithful errors.** Every failure decodes the structured [`jammi_wire`]
//!   detail the server attaches, so a verb returns the *exact*
//!   [`jammi_db::error::JammiError`] variant the in-process path would — never a
//!   lossy gRPC-code-category guess.
//! * **Tenant over the wire.** The tenant trio rides the composed
//!   [`CatalogClient`]; the binding is keyed by the session id every verb
//!   carries in the [`SESSION_HEADER`].
//! * **Shared conversions.** Request encode / response decode reuse the
//!   [`jammi_wire`] conversions the server's receive side uses.

use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::{Stream, StreamExt, TryStreamExt};
use tonic::transport::Endpoint;

use jammi_db::catalog::eval_repo::PerQueryEvalRecord;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};
use jammi_db::trigger::{DeliveredBatch, Offset, Predicate, TopicDefinition, TriggerError};
use jammi_db::{AuditError, ModelTask, PerQueryAudit, TenantId};

use jammi_admin::CatalogClient;
use jammi_wire::eval::{CompareEvalReport, EmbeddingEvalReport, EvalTask, InferenceEvalReport};
use jammi_wire::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_wire::proto::audit::audit_service_client::AuditServiceClient;
use jammi_wire::proto::audit::{
    AuditFetchByQueryIdRequest, AuditFetchRecentRequest, AuditLogRequest,
};
use jammi_wire::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use jammi_wire::proto::embedding::{
    encode_query_request::Input as ProtoEncodeInput, search_request::Query as ProtoSearchQuery,
    EncodeQueryRequest, GenerateEmbeddingsRequest, QueryVector,
    SearchRequest as ProtoSearchRequest, SearchResponse,
};
use jammi_wire::proto::eval as eval_pb;
use jammi_wire::proto::eval::eval_service_client::EvalServiceClient;
use jammi_wire::proto::inference::inference_service_client::InferenceServiceClient;
use jammi_wire::proto::inference::InferRequest;
use jammi_wire::proto::training::training_service_client::TrainingServiceClient;
use jammi_wire::proto::training::{
    start_training_request::Spec as ProtoTrainingSpec, FineTuneSpec, StartTrainingRequest,
    TrainingStatusRequest,
};
use jammi_wire::proto::trigger::trigger_service_client::TriggerServiceClient;
use jammi_wire::proto::trigger::{PublishRequest, SubscribeRequest, TopicName};
use jammi_wire::request::{FineTuneJobId, Modality, QueryInput, SearchQuery, SearchRequest};
use jammi_wire::{
    audit_error_from_status, cohorts_to_proto, config_to_proto, decode_ipc_stream,
    decode_subscribed_batch, encode_publish_batch, error_from_status, eval_task_to_proto,
    method_to_proto, model_task_to_proto, record_from_wire, result_table_from_proto,
    trigger_error_from_status, SessionChannel, SessionTransport, SESSION_HEADER,
};

/// A data-plane client backed by a remote engine over gRPC + Flight SQL.
///
/// Cheap to clone: it holds the cloneable transport and the composed
/// [`CatalogClient`] (which shares the same transport).
#[derive(Clone)]
pub struct DataClient {
    transport: SessionTransport,
    catalog: CatalogClient,
}

impl DataClient {
    /// Connect to a `jammi.v1` gRPC endpoint and mint a fresh session id. The
    /// composed control client shares the same transport (and session id), so a
    /// tenant bound through [`Self::bind_tenant`] is observed by every data verb.
    pub async fn connect(endpoint: impl Into<Endpoint>) -> Result<Self> {
        let transport = SessionTransport::connect(endpoint).await?;
        Ok(Self::over(transport))
    }

    /// Build a data client over an existing transport, composing a control
    /// client over the same channel + session id.
    pub fn over(transport: SessionTransport) -> Self {
        let catalog = CatalogClient::over(transport.clone());
        Self { transport, catalog }
    }

    /// The composed control-plane client over the same session — the source/
    /// model registry, channel, mutable-table, topic-admin, and tenant verbs.
    pub fn catalog(&self) -> &CatalogClient {
        &self.catalog
    }

    /// The opaque session id the server keys tenant state against. The Flight
    /// SQL lane stamps this same id so a bound-tenant query scopes correctly.
    pub fn session_id(&self) -> &str {
        self.transport.session_id()
    }

    fn embedding_client(&self) -> EmbeddingServiceClient<SessionChannel> {
        self.transport
            .service(EmbeddingServiceClient::with_interceptor)
    }

    fn inference_client(&self) -> InferenceServiceClient<SessionChannel> {
        self.transport
            .service(InferenceServiceClient::with_interceptor)
    }

    fn eval_client(&self) -> EvalServiceClient<SessionChannel> {
        self.transport.service(EvalServiceClient::with_interceptor)
    }

    fn training_client(&self) -> TrainingServiceClient<SessionChannel> {
        self.transport
            .service(TrainingServiceClient::with_interceptor)
    }

    fn trigger_client(&self) -> TriggerServiceClient<SessionChannel> {
        self.transport
            .service(TriggerServiceClient::with_interceptor)
    }

    fn audit_client(&self) -> AuditServiceClient<SessionChannel> {
        self.transport.service(AuditServiceClient::with_interceptor)
    }

    // --- tenant (delegated to the composed control client) ---------------

    /// Bind a tenant scope to this session (sticky form).
    pub async fn bind_tenant(&self, t: TenantId) -> Result<()> {
        self.catalog.bind_tenant(t).await
    }

    /// Clear the bound tenant.
    pub async fn unbind_tenant(&self) -> Result<()> {
        self.catalog.unbind_tenant().await
    }

    /// The tenant currently bound, if any.
    pub async fn tenant(&self) -> Result<Option<TenantId>> {
        self.catalog.tenant().await
    }

    // --- sql -------------------------------------------------------------

    /// Execute a SQL query over the Flight SQL lane and collect the terminal
    /// batches.
    ///
    /// `sql` does not ride a typed gRPC verb — per ADR-01 §3.2 the Flight SQL
    /// surface carries query/result. So this opens a [`FlightSqlServiceClient`]
    /// over the *same* tonic channel the typed-RPC verbs use, stamps the
    /// [`SESSION_HEADER`] with [`Self::session_id`] — the identical id
    /// `bind_tenant` bound the tenant scope against — so the server's
    /// `TenantBoundProvider` resolves this query to that bound tenant, then runs
    /// `execute` → `do_get(ticket)` per endpoint and concatenates the streamed
    /// batches. Stamping the bound session id (not a fresh one) is what keeps a
    /// `--tenant A` query scoped to tenant A rather than silently unscoped.
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        let mut client = FlightSqlServiceClient::new(self.transport.channel());
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

    /// Generate embeddings for `columns` of a source with the given model and
    /// modality, persisting one vector per row.
    pub async fn generate_embeddings(
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
        result_table_from_proto(table).map_err(|s| error_from_status(&s))
    }

    /// Encode a single query into a vector with the given model.
    pub async fn encode_query(
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

    /// Run a vector search and return the terminal hydrated batches.
    pub async fn search(&self, request: SearchRequest) -> Result<Vec<RecordBatch>> {
        let SearchRequest {
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
            .search(ProtoSearchRequest {
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

    /// Run inference on a registered source using a model.
    pub async fn infer(
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

    /// Start a fine-tuning job and return its id. Poll completion with
    /// [`Self::fine_tune_status`].
    pub async fn fine_tune(
        &self,
        source: &str,
        base_model: &str,
        columns: &[String],
        method: FineTuneMethod,
        task: ModelTask,
        config: Option<FineTuneConfig>,
    ) -> Result<FineTuneJobId> {
        // The column-source fine-tune is the `FineTuneSpec` arm of the spec
        // oneof; built inline from the transport-neutral config vocabulary so the
        // data client (which carries no engine `TrainingSpec`) can still submit
        // it.
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

    /// Current status string for a fine-tune job, looked up by id.
    pub async fn fine_tune_status(&self, id: &FineTuneJobId) -> Result<String> {
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

    /// Evaluate embedding quality against golden relevance judgments.
    pub async fn eval_embeddings(
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

    /// Read back the persisted per-query eval records for a run.
    pub async fn eval_per_query(&self, eval_run_id: &str) -> Result<Vec<PerQueryEvalRecord>> {
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

    /// Evaluate inference quality against golden labels.
    pub async fn eval_inference(
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

    /// Compare multiple embedding tables side-by-side.
    pub async fn eval_compare(
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

    // --- trigger (publish / subscribe) -----------------------------------

    /// Publish one batch to a topic under the session's tenant scope, returning
    /// the assigned offset.
    pub async fn publish(
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
            .map(jammi_wire::from_proto_timestamp)
            .transpose()
            .map_err(|s| trigger_error_from_status(&s))?
            .ok_or_else(|| TriggerError::Driver("publish response missing committed_at".into()))?;
        Ok(Offset::new(resp.offset, committed_at))
    }

    /// Subscribe to a topic, returning a transport-neutral stream of delivered
    /// batches. The stream replays from `from_offset` (or the live tail when
    /// `None`) and then tails live, scoped to the session's tenant. When
    /// `replay_only` is set the server drives its finite drain and closes the
    /// stream rather than holding open to tail live batches.
    pub async fn subscribe(
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
                replay_only,
            })
            .await
            .map_err(|s| trigger_error_from_status(&s))?
            .into_inner();
        // Map each streamed item into the same `Result<DeliveredBatch, TriggerError>`
        // a local subscription yields. A terminal `tonic::Status` reconstructs to
        // its faithful `TriggerError` via the attached detail; a payload-decode
        // failure surfaces as the faithful `Status` the decoder built.
        let mapped = streaming.map(|item| match item {
            Ok(wire) => decode_subscribed_batch(wire).map_err(|s| trigger_error_from_status(&s)),
            Err(status) => Err(trigger_error_from_status(&status)),
        });
        Ok(Box::pin(mapped))
    }

    // --- audit -----------------------------------------------------------

    /// Sign and persist a batch of audit records; publishes them to the audit
    /// topic.
    pub async fn audit_log(
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

    /// Fetch one audit record by query id (tenant-scoped).
    pub async fn audit_fetch_by_query_id(
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

    /// Fetch the most recent audit records (tenant-scoped), newest first.
    pub async fn audit_fetch_recent(
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
fn proto_modality(modality: Modality) -> jammi_wire::proto::embedding::Modality {
    use jammi_wire::proto::embedding::Modality as Pb;
    match modality {
        Modality::Text => Pb::Text,
        Modality::Image => Pb::Image,
        Modality::Audio => Pb::Audio,
    }
}

/// Rebuild the terminal `Vec<RecordBatch>` shape a search verb returns from the
/// wire `SearchResponse`.
///
/// The wire surface carries each hit as `key` + `score` + a `columns` map of
/// stringified projections, so the client rehydrates one batch with the
/// `_row_id` (key) and `similarity` (score) columns the in-process hydrated
/// batch carries, plus a `Utf8` column per requested `select` name.
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
