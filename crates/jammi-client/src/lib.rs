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
use std::num::NonZeroU32;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::{Stream, StreamExt, TryStreamExt};
use tonic::transport::Endpoint;

use jammi_datafusion::ModelTask;
use jammi_db::catalog::eval_repo::PerQueryEvalRecord;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};
use jammi_db::store::{CacheOutcome, CachePolicy};
use jammi_db::trigger::{DeliveredBatch, Offset, Predicate, TopicDefinition, TriggerError};
use jammi_db::{AuditError, PerQueryAudit, TenantId};

use jammi_admin::CatalogClient;
use jammi_wire::embedding_refresh::{
    delete_policy_to_proto, expiry_report_from_proto, refresh_report_from_proto, ExpiryReport,
    RefreshOptions, RefreshReport,
};
use jammi_wire::eval::{CompareEvalReport, EmbeddingEvalReport, EvalTask, InferenceEvalReport};
use jammi_wire::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_wire::proto::audit::audit_service_client::AuditServiceClient;
use jammi_wire::proto::audit::{
    AuditFetchByQueryIdRequest, AuditFetchRecentRequest, AuditLogRequest,
};
use jammi_wire::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use jammi_wire::proto::embedding::{
    encode_query_request::Input as ProtoEncodeInput, search_request::Query as ProtoSearchQuery,
    CompactEmbeddingsRequest, EncodeQueryRequest, ExpireVersionsRequest, GenerateEmbeddingsRequest,
    QueryVector, RefreshEmbeddingsRequest, SearchRequest as ProtoSearchRequest, SearchResponse,
};
use jammi_wire::proto::eval as eval_pb;
use jammi_wire::proto::eval::eval_service_client::EvalServiceClient;
use jammi_wire::proto::inference::inference_service_client::InferenceServiceClient;
use jammi_wire::proto::inference::{CachePolicy as ProtoCachePolicy, InferRequest};
use jammi_wire::proto::job::job_service_client::JobServiceClient;
use jammi_wire::proto::job::{
    submit_job_request::Spec as ProtoTrainingSpec, CancelJobRequest as JobCancelJobRequest,
    JobStatusRequest, JobStatusResponse, ListJobsRequest, SubmitJobRequest,
};
use jammi_wire::proto::training::FineTuneSpec;
use jammi_wire::proto::trigger::trigger_service_client::TriggerServiceClient;
use jammi_wire::proto::trigger::{PublishRequest, SubscribeRequest, TopicName};
use jammi_wire::request::{
    FineTuneJobId, FineTuneRequest, Modality, QueryInput, SearchQuery, SearchRequest,
};
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

    fn job_client(&self) -> JobServiceClient<SessionChannel> {
        self.transport.service(JobServiceClient::with_interceptor)
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
    /// `sql` does not ride a typed gRPC verb — the Flight SQL surface
    /// carries query/result. So this opens a [`FlightSqlServiceClient`]
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
        cache: CachePolicy,
    ) -> Result<(ResultTableRecord, CacheOutcome)> {
        let table = self
            .embedding_client()
            .generate_embeddings(GenerateEmbeddingsRequest {
                source_id: source_id.to_string(),
                model_id: model_id.to_string(),
                columns: columns.to_vec(),
                key_column: key_column.to_string(),
                modality: proto_modality(modality) as i32,
                cache: proto_cache_policy(cache) as i32,
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        let outcome = jammi_wire::cache_outcome_from_proto(table.cache_outcome.clone())
            .map_err(|s| error_from_status(&s))?;
        let record = result_table_from_proto(table).map_err(|s| error_from_status(&s))?;
        Ok((record, outcome))
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

    // --- incremental embedding ------------------------------------------

    /// Re-embed only the source rows whose content changed since `table`'s
    /// current version and publish the result as a new version.
    pub async fn refresh_embeddings(
        &self,
        table: &str,
        options: RefreshOptions,
    ) -> Result<RefreshReport> {
        let report = self
            .embedding_client()
            .refresh_embeddings(RefreshEmbeddingsRequest {
                table: table.to_string(),
                deletes: delete_policy_to_proto(options.deletes) as i32,
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        refresh_report_from_proto(report).map_err(|s| error_from_status(&s))
    }

    /// Rewrite `table`'s live rows as one fragment + one segment and publish
    /// it as a new version.
    pub async fn compact_embeddings(&self, table: &str) -> Result<RefreshReport> {
        let report = self
            .embedding_client()
            .compact_embeddings(CompactEmbeddingsRequest {
                table: table.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        refresh_report_from_proto(report).map_err(|s| error_from_status(&s))
    }

    /// Delete every non-current version of `table` numbered below `before`
    /// and reap its unreferenced artifacts.
    pub async fn expire_versions(&self, table: &str, before: i64) -> Result<ExpiryReport> {
        let report = self
            .embedding_client()
            .expire_versions(ExpireVersionsRequest {
                table: table.to_string(),
                before,
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(expiry_report_from_proto(report))
    }

    // --- search ----------------------------------------------------------

    /// Run a vector search and return the terminal hydrated batches.
    pub async fn search(&self, request: SearchRequest) -> Result<Vec<RecordBatch>> {
        let SearchRequest {
            source_id,
            query,
            k,
            embedding_table,
            filter,
            select,
            oversample,
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
                embedding_table,
                filter,
                select: select.clone(),
                oversample: oversample.map(|v| v as u32),
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
        cache: CachePolicy,
    ) -> Result<(Vec<RecordBatch>, CacheOutcome)> {
        let resp = self
            .inference_client()
            .infer(InferRequest {
                source_id: source_id.to_string(),
                model_id: model_id.to_string(),
                task: model_task_to_proto(task) as i32,
                columns: content_columns.to_vec(),
                key_column: key_column.to_string(),
                tenant_id: String::new(),
                cache: proto_cache_policy(cache) as i32,
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        let outcome = jammi_wire::cache_outcome_from_proto(resp.cache_outcome)
            .map_err(|s| error_from_status(&s))?;
        let batch = resp.result.unwrap_or_default();
        let batches = decode_ipc_stream(&batch.data_header, &batch.data_body)
            .map_err(|s| error_from_status(&s))?;
        Ok((batches, outcome))
    }

    // --- fine-tune (submits through JobService.SubmitJob) -----------------

    /// Start a fine-tuning job on the engine's defaults for every knob
    /// [`FineTuneRequest`] carries beyond this verb's parameters, and return
    /// its id. Poll completion with [`Self::fine_tune_status`], or
    /// [`Self::wait_job`] for a resumable wait.
    ///
    /// Submits a single-rank job: the data-parallel rank count is left unset.
    /// Use [`Self::submit_fine_tune`] to choose one.
    pub async fn fine_tune(
        &self,
        source: &str,
        base_model: &str,
        columns: &[String],
        method: FineTuneMethod,
        task: ModelTask,
        config: Option<FineTuneConfig>,
    ) -> Result<FineTuneJobId> {
        self.submit_fine_tune(FineTuneRequest {
            source: source.to_string(),
            base_model: base_model.to_string(),
            columns: columns.to_vec(),
            method,
            task,
            config,
            world_size: None,
            cache: CachePolicy::Bypass,
        })
        .await
    }

    /// Start a fine-tuning job from a whole [`FineTuneRequest`] and return its
    /// id — the flattened form of [`Self::fine_tune`], carrying the knobs that
    /// verb's parameter list does not name (today, the data-parallel rank
    /// count). One request shape rather than a growing parameter list, the
    /// same way [`Self::search`] takes a whole
    /// [`jammi_wire::request::SearchRequest`].
    ///
    /// A rank count of `Some(1)` submits the single-rank job [`Self::fine_tune`]
    /// submits, on the identical bytes — see
    /// [`FineTuneRequest::world_size`](jammi_wire::request::FineTuneRequest::world_size).
    pub async fn submit_fine_tune(&self, request: FineTuneRequest) -> Result<FineTuneJobId> {
        let FineTuneRequest {
            source,
            base_model,
            columns,
            method,
            task,
            config,
            world_size,
            cache,
        } = request;
        // The column-source fine-tune is the `FineTuneSpec` arm of the
        // `SubmitJob` spec oneof; built inline from the transport-neutral
        // config vocabulary so the data client (which carries no engine
        // `TrainingSpec`) can still submit it.
        let resp = self
            .job_client()
            .submit_job(SubmitJobRequest {
                spec: Some(ProtoTrainingSpec::FineTune(FineTuneSpec {
                    source,
                    columns,
                    method: method_to_proto(method) as i32,
                    task: model_task_to_proto(task) as i32,
                })),
                base_model,
                config: config.as_ref().map(config_to_proto),
                idempotency_key: String::new(),
                // `0` IS the unset value of the wire's implicit-presence
                // `uint32`, so an unchosen count sends the same bytes a caller
                // sent before the field existed and the engine resolves it to
                // one rank. An explicit `1` denotes that same single-rank job,
                // so it takes that same encoding: one wire value per intent,
                // matching what the Python client puts on the wire
                // (`_wire_world_size`). Only a count above one is written.
                world_size: world_size
                    .map(NonZeroU32::get)
                    .filter(|&ranks| ranks > 1)
                    .unwrap_or(0),
                // `Bypass` (the engine default) leaves the field OFF the
                // encoding entirely — `UNSPECIFIED` is `0`, the implicit-
                // presence enum's unset value, and `Bypass`/`UNSPECIFIED`
                // decode identically (`crates/jammi-ai/src/wire/training.rs`,
                // `unspecified_and_bypass_cache_both_decode_to_bypass`) — so a
                // caller that never asks for reuse submits byte-for-byte the
                // request it submitted before this field existed, matching
                // the Python client's `_wire_cache_policy_for_submit_job`
                // (`clients/python/jammi/_assembly.py`), which leaves the
                // field unset the same way. Only `Use` costs a byte on the
                // wire, the one case that changes what the engine does with
                // the request.
                cache: match cache {
                    CachePolicy::Use => ProtoCachePolicy::Use as i32,
                    CachePolicy::Bypass => ProtoCachePolicy::Unspecified as i32,
                },
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(FineTuneJobId(resp.job_id))
    }

    /// Current status string for a fine-tune job, looked up by id.
    pub async fn fine_tune_status(&self, id: &FineTuneJobId) -> Result<String> {
        Ok(self.job_status_response(&id.0).await?.status)
    }

    /// Run metrics recorded for a fine-tune job, as the raw JSON blob text
    /// nested inside the wire's terminal `JobStatus.model.metrics_json`
    /// — the same blob the embedded `TrainingJob`'s
    /// catalog-backed metrics read returns. `None` for a job that has not
    /// yet recorded any metrics (still queued or running before its first
    /// stamp); this crate carries no `serde_json` dependency, so the caller
    /// decodes the returned text.
    pub async fn fine_tune_metrics(&self, id: &FineTuneJobId) -> Result<Option<String>> {
        use jammi_wire::proto::job::job_status_response::Result as WireResult;
        let resp = self.job_status_response(&id.0).await?;
        Ok(match resp.result {
            Some(WireResult::Model(m)) => m.metrics_json,
            _ => None,
        })
    }

    /// GPU-acceleration determination for a fine-tune job, as the raw,
    /// self-describing JSON blob text the catalog's `jobs.
    /// acceleration_report` column carries — the same blob the
    /// embedded catalog-backed record read returns. `None` for a row whose
    /// column is SQL `NULL`; otherwise a `"state"`-keyed object
    /// whose vocabulary is owned by the payload's producer (e.g. `"pending"`
    /// before a determination exists, `"determined"` once one does) and
    /// documented there, not enumerated here. This crate carries no
    /// `serde_json` dependency, so the caller decodes the returned text.
    pub async fn fine_tune_acceleration_report(
        &self,
        id: &FineTuneJobId,
    ) -> Result<Option<String>> {
        Ok(self
            .job_status_response(&id.0)
            .await?
            .acceleration_report_json)
    }

    // --- jobs (JobService; generic across every job kind) -----------------

    async fn job_status_response(&self, job_id: &str) -> Result<JobStatusResponse> {
        Ok(self
            .job_client()
            .job_status(JobStatusRequest {
                job_id: job_id.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner())
    }

    /// Read a job's current status by id: status, kind, and the resolved
    /// output model id (training kinds only; empty for a compute kind).
    pub async fn job_status(&self, job_id: &str) -> Result<JobStatusResponse> {
        self.job_status_response(job_id).await
    }

    /// Stream status updates for a job until it reaches a terminal state.
    /// Reconnecting with the same job id resumes the wait — an already
    /// terminal job's stream carries only its terminal `done` frame.
    ///
    /// Sends no `grpc-timeout` header — no deadline of the client's own. The
    /// server budget bounds this stream instead: when a deployment configures
    /// `[server.limits] wait_timeout_secs`, that budget becomes THIS stream's
    /// deadline (it ends with `DEADLINE_EXCEEDED` once the budget elapses,
    /// wherever the job then stands — reconnect to resume the wait); a
    /// deployment with no such budget configured genuinely waits until
    /// terminal. Use [`Self::wait_job_with_timeout`] to declare an explicit
    /// deadline of your own instead — one at or below the deployment's
    /// budget, when known, is ENFORCED by the server: the stream ends with
    /// `DEADLINE_EXCEEDED` at that declared deadline, never the wider
    /// budget; a value above the budget is refused at the edge before the
    /// stream opens.
    pub async fn wait_job(
        &self,
        job_id: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<JobStatusResponse>> + Send>>> {
        self.wait_job_inner(job_id, None).await
    }

    /// [`Self::wait_job`], with an explicit `grpc-timeout` rather than none.
    /// A reconnect after this deadline lapses (or after any other disconnect)
    /// resumes the wait — the deadline bounds one connection's hold on the
    /// server's `max_job_waits` budget, not the job's own lifetime.
    pub async fn wait_job_with_timeout(
        &self,
        job_id: &str,
        timeout: Duration,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<JobStatusResponse>> + Send>>> {
        self.wait_job_inner(job_id, Some(timeout)).await
    }

    async fn wait_job_inner(
        &self,
        job_id: &str,
        timeout: Option<Duration>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<JobStatusResponse>> + Send>>> {
        use jammi_wire::proto::job::{job_event::Event, JobHandle};
        let mut request = tonic::Request::new(JobHandle {
            job_id: job_id.to_string(),
        });
        if let Some(timeout) = timeout {
            request.set_timeout(timeout);
        }
        let stream = self
            .job_client()
            .wait_job(request)
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        let mapped = stream.filter_map(|item| async move {
            match item {
                Ok(event) => match event.event {
                    Some(Event::Done(done)) => Some(Ok(*done)),
                    // Progress frames carry no terminal payload for this
                    // simplified surface; a caller that needs progress reads
                    // `JobStatus.progress` directly.
                    Some(Event::Progress(_)) | None => None,
                },
                Err(s) => Some(Err(error_from_status(&s))),
            }
        });
        Ok(Box::pin(mapped))
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

    /// List jobs visible to the session tenant, most recent first.
    pub async fn list_jobs(&self) -> Result<Vec<jammi_wire::proto::job::JobSummary>> {
        let resp = self
            .job_client()
            .list_jobs(ListJobsRequest {})
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(resp.jobs)
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
    ///
    /// Sends no `grpc-timeout` header — no deadline of the client's own; the
    /// same posture as [`Self::wait_job`], since the same
    /// `[server.limits] wait_timeout_secs` cap governs both streaming rpcs.
    /// When configured, that budget becomes THIS stream's deadline (it ends
    /// with `DEADLINE_EXCEEDED` once the budget elapses); with no such budget
    /// configured this tails live indefinitely. A caller that needs to keep
    /// tailing past either kind of disconnect reconnects with `from_offset`
    /// set to the last delivered offset. Use [`Self::subscribe_with_timeout`]
    /// to declare an explicit deadline instead.
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
        self.subscribe_inner(topic, predicate, from_offset, replay_only, None)
            .await
    }

    /// [`Self::subscribe`], with an explicit `grpc-timeout` rather than none.
    /// A value at or below the deployment's own `wait_timeout_secs` budget,
    /// when known, is ENFORCED by the server: the stream ends with
    /// `DEADLINE_EXCEEDED` at that declared deadline, never the wider
    /// budget; a value above the budget is refused at the edge before the
    /// stream opens.
    pub async fn subscribe_with_timeout(
        &self,
        topic: &TopicDefinition,
        predicate: Predicate,
        from_offset: Option<Offset>,
        replay_only: bool,
        timeout: Duration,
    ) -> std::result::Result<
        Pin<Box<dyn Stream<Item = std::result::Result<DeliveredBatch, TriggerError>> + Send>>,
        TriggerError,
    > {
        self.subscribe_inner(topic, predicate, from_offset, replay_only, Some(timeout))
            .await
    }

    async fn subscribe_inner(
        &self,
        topic: &TopicDefinition,
        predicate: Predicate,
        from_offset: Option<Offset>,
        replay_only: bool,
        timeout: Option<Duration>,
    ) -> std::result::Result<
        Pin<Box<dyn Stream<Item = std::result::Result<DeliveredBatch, TriggerError>> + Send>>,
        TriggerError,
    > {
        let mut request = tonic::Request::new(SubscribeRequest {
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
        });
        if let Some(timeout) = timeout {
            request.set_timeout(timeout);
        }
        let streaming = self
            .trigger_client()
            .subscribe(request)
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

/// Map the engine [`CachePolicy`] onto the wire enum. Encode is total.
fn proto_cache_policy(cache: CachePolicy) -> jammi_wire::proto::inference::CachePolicy {
    use jammi_wire::proto::inference::CachePolicy as Pb;
    match cache {
        CachePolicy::Use => Pb::Use,
        CachePolicy::Bypass => Pb::Bypass,
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

/// `wait_job`/`subscribe` must send NO
/// `grpc-timeout` header at all (the server's `[server.limits]
/// wait_timeout_secs` budget bounds the stream instead — see
/// `jammi_server::limits`'s module doc's "Streaming-path exemption" section);
/// `wait_job_with_timeout`/`subscribe_with_timeout` MUST send one, carrying
/// the caller's declared duration. Proven against the SAME code path
/// production uses (`wait_job_inner`/`subscribe_inner`, reached only through
/// the four public entry points above) over a REAL loopback gRPC connection:
/// request construction and the RPC call are not separable into two steps in
/// those functions (the request is built and sent in one async fn against a
/// live channel), so a hand-built `tonic::Request` inspected without ever
/// sending it would not exercise this code path — this fixture instead reads
/// the header a genuine server-side handler actually received.
#[cfg(test)]
mod grpc_timeout_header_tests {
    use std::net::SocketAddr;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll};
    use std::time::Duration;

    use arrow_schema::Schema;
    use futures::Stream;
    use tokio::net::{TcpListener, TcpStream};
    use tonic::transport::{Endpoint, Server};
    use tonic::{Request, Response, Status};

    use jammi_db::trigger::{Predicate, TopicDefinition, TopicId};
    use jammi_wire::proto::job::job_service_server::{JobService, JobServiceServer};
    use jammi_wire::proto::job::{
        CancelJobRequest, CancelJobResponse, JobEvent, JobHandle, JobStatusRequest,
        JobStatusResponse, ListJobsRequest, ListJobsResponse, ListWorkersRequest,
        ListWorkersResponse, PruneJobsRequest, PruneJobsResponse, SubmitJobRequest,
        SubmitJobResponse,
    };
    use jammi_wire::proto::trigger::trigger_service_server::{
        TriggerService, TriggerServiceServer,
    };
    use jammi_wire::proto::trigger::{
        PublishRequest, PublishResponse, SubscribeRequest, SubscribedBatch,
    };

    use super::DataClient;

    /// `None` while unset; `Some(header value or None)` once the one request
    /// this fixture's test drives has landed — a single-request-per-call
    /// fixture, so one cell suffices without a queue.
    type Captured = Arc<Mutex<Option<Option<String>>>>;

    /// Read the raw `grpc-timeout` metadata value off an inbound request, if
    /// present — the ONLY oracle this fixture needs: presence/absence and the
    /// value, not a parsed `Duration` (that parsing is `limits.rs`'s own
    /// `parse_grpc_timeout`, exercised elsewhere).
    fn read_grpc_timeout<T>(request: &Request<T>) -> Option<String> {
        request
            .metadata()
            .get("grpc-timeout")
            .map(|v| v.to_str().expect("grpc-timeout is ASCII").to_string())
    }

    /// A minimal [`futures::Stream`] over [`TcpListener::poll_accept`] — lets
    /// this fixture hand `tonic::transport::Server` a real accepted-connection
    /// stream without depending on `tokio-stream`'s `net` feature (not enabled
    /// workspace-wide) for a one-off test-only server.
    struct AcceptStream(TcpListener);

    impl Stream for AcceptStream {
        type Item = std::io::Result<TcpStream>;

        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            match self.0.poll_accept(cx) {
                Poll::Ready(Ok((stream, _peer))) => Poll::Ready(Some(Ok(stream))),
                Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
                Poll::Pending => Poll::Pending,
            }
        }
    }

    /// A `JobService` double whose only LIVE method is `wait_job`; every other
    /// method is unreachable from either test built on this fixture, so a
    /// wiring mistake panics loudly rather than returning a plausible-looking
    /// placeholder response.
    struct HeaderCapturingJobService {
        wait_job_header: Captured,
    }

    #[tonic::async_trait]
    impl JobService for HeaderCapturingJobService {
        type WaitJobStream = Pin<Box<dyn Stream<Item = Result<JobEvent, Status>> + Send + 'static>>;

        async fn submit_job(
            &self,
            _request: Request<SubmitJobRequest>,
        ) -> Result<Response<SubmitJobResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn job_status(
            &self,
            _request: Request<JobStatusRequest>,
        ) -> Result<Response<JobStatusResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn wait_job(
            &self,
            request: Request<JobHandle>,
        ) -> Result<Response<Self::WaitJobStream>, Status> {
            *self.wait_job_header.lock().unwrap() = Some(read_grpc_timeout(&request));
            Ok(Response::new(
                Box::pin(futures::stream::empty()) as Self::WaitJobStream
            ))
        }

        async fn list_jobs(
            &self,
            _request: Request<ListJobsRequest>,
        ) -> Result<Response<ListJobsResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn cancel_job(
            &self,
            _request: Request<CancelJobRequest>,
        ) -> Result<Response<CancelJobResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn list_workers(
            &self,
            _request: Request<ListWorkersRequest>,
        ) -> Result<Response<ListWorkersResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn prune_jobs(
            &self,
            _request: Request<PruneJobsRequest>,
        ) -> Result<Response<PruneJobsResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }
    }

    /// A `TriggerService` double whose only LIVE method is `subscribe`.
    struct HeaderCapturingTriggerService {
        subscribe_header: Captured,
    }

    #[tonic::async_trait]
    impl TriggerService for HeaderCapturingTriggerService {
        type SubscribeStream =
            Pin<Box<dyn Stream<Item = Result<SubscribedBatch, Status>> + Send + 'static>>;

        async fn publish(
            &self,
            _request: Request<PublishRequest>,
        ) -> Result<Response<PublishResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn subscribe(
            &self,
            request: Request<SubscribeRequest>,
        ) -> Result<Response<Self::SubscribeStream>, Status> {
            *self.subscribe_header.lock().unwrap() = Some(read_grpc_timeout(&request));
            Ok(Response::new(
                Box::pin(futures::stream::empty()) as Self::SubscribeStream
            ))
        }
    }

    /// Spin up a loopback tonic server hosting only the two header-capturing
    /// doubles above (every other `jammi.v1` service `DataClient` composes is
    /// never dialled by the two calls under test), returning the bound address
    /// and the two capture cells. The serve task is detached (not joined):
    /// nothing here needs graceful shutdown, since each test's own process
    /// teardown reclaims the port.
    async fn start_capturing_server() -> (SocketAddr, Captured, Captured) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr");
        let wait_job_header: Captured = Arc::new(Mutex::new(None));
        let subscribe_header: Captured = Arc::new(Mutex::new(None));
        let job_svc = JobServiceServer::new(HeaderCapturingJobService {
            wait_job_header: Arc::clone(&wait_job_header),
        });
        let trigger_svc = TriggerServiceServer::new(HeaderCapturingTriggerService {
            subscribe_header: Arc::clone(&subscribe_header),
        });
        tokio::spawn(async move {
            Server::builder()
                .add_service(job_svc)
                .add_service(trigger_svc)
                .serve_with_incoming(AcceptStream(listener))
                .await
                .expect("capturing server");
        });
        (addr, wait_job_header, subscribe_header)
    }

    async fn connect(addr: SocketAddr) -> DataClient {
        let endpoint = Endpoint::from_shared(format!("http://{addr}")).expect("endpoint");
        DataClient::connect(endpoint)
            .await
            .expect("data client connect")
    }

    /// Guards `wait_job`/`wait_job_with_timeout` against collapsing onto the
    /// same request-construction path with a default deadline applied
    /// regardless of caller intent: `wait_job` sends no `grpc-timeout` at
    /// all (the server-side budget bounds the stream instead — see this
    /// crate's `wait_job` doc); `wait_job_with_timeout` sends one every time.
    #[tokio::test]
    async fn wait_job_sends_no_grpc_timeout_header_but_wait_job_with_timeout_does() {
        let (addr, wait_job_header, _subscribe_header) = start_capturing_server().await;
        let client = connect(addr).await;

        let _ = client.wait_job("does-not-matter").await;
        assert_eq!(
            wait_job_header.lock().unwrap().take(),
            Some(None),
            "wait_job must send NO grpc-timeout header -- the server budget bounds the \
             stream, never a client-declared deadline"
        );

        let _ = client
            .wait_job_with_timeout("does-not-matter", Duration::from_secs(5))
            .await;
        let captured = wait_job_header.lock().unwrap().take();
        assert!(
            matches!(captured, Some(Some(_))),
            "wait_job_with_timeout must send a grpc-timeout header, got {captured:?}"
        );
    }

    /// The `subscribe`/`subscribe_with_timeout` analogue of
    /// [`wait_job_sends_no_grpc_timeout_header_but_wait_job_with_timeout_does`].
    #[tokio::test]
    async fn subscribe_sends_no_grpc_timeout_header_but_subscribe_with_timeout_does() {
        let (addr, _wait_job_header, subscribe_header) = start_capturing_server().await;
        let client = connect(addr).await;

        let topic = TopicDefinition {
            id: TopicId::new(),
            name: "does-not-matter".into(),
            schema: Arc::new(Schema::empty()),
            tenant: None,
        };

        let _ = client
            .subscribe(&topic, Predicate::match_all(), None, false)
            .await;
        assert_eq!(
            subscribe_header.lock().unwrap().take(),
            Some(None),
            "subscribe must send NO grpc-timeout header -- the server budget bounds the \
             stream, never a client-declared deadline"
        );

        let _ = client
            .subscribe_with_timeout(
                &topic,
                Predicate::match_all(),
                None,
                false,
                Duration::from_secs(5),
            )
            .await;
        let captured = subscribe_header.lock().unwrap().take();
        assert!(
            matches!(captured, Some(Some(_))),
            "subscribe_with_timeout must send a grpc-timeout header, got {captured:?}"
        );
    }
}

/// What this client puts on the wire for the data-parallel rank count, read
/// off a request a real `JobService` handler received.
///
/// The count is the one submit knob whose default is a *silent* value — `0`,
/// the implicit-presence `uint32`'s unset — so "the caller did not choose" and
/// "the caller chose more than one rank" have to be distinguishable at the
/// server, not merely at the call site. The third case, an explicit `1`, is
/// the SAME job as unset and therefore owes the same bytes: one wire encoding
/// per intent across clients, the encoding the Python client's
/// `_wire_world_size` already writes.
///
/// These tests read the field a genuine handler received over a
/// loopback connection, the same fixture shape
/// [`grpc_timeout_header_tests`](self::grpc_timeout_header_tests) uses:
/// `DataClient::submit_fine_tune` builds and sends the request in one async
/// fn against a live channel, so a hand-built request inspected without
/// sending it would not exercise the code path production uses.
#[cfg(test)]
mod world_size_tests {
    use std::net::SocketAddr;
    use std::num::NonZeroU32;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    use futures::Stream;
    use prost::Message;
    use tokio::net::{TcpListener, TcpStream};
    use tonic::transport::{Endpoint, Server};
    use tonic::{Request, Response, Status};

    use jammi_datafusion::ModelTask;
    use jammi_db::store::CachePolicy;
    use jammi_wire::fine_tune::FineTuneMethod;
    use jammi_wire::proto::job::job_service_server::{JobService, JobServiceServer};
    use jammi_wire::proto::job::{
        CancelJobRequest, CancelJobResponse, JobEvent, JobHandle, JobStatusRequest,
        JobStatusResponse, ListJobsRequest, ListJobsResponse, ListWorkersRequest,
        ListWorkersResponse, PruneJobsRequest, PruneJobsResponse, SubmitJobRequest,
        SubmitJobResponse,
    };
    use jammi_wire::request::FineTuneRequest;

    use super::DataClient;

    /// The one `SubmitJobRequest` this fixture's test drives, captured whole so
    /// a test can assert on any field of it (not only the count).
    type CapturedSubmit = Arc<Mutex<Option<SubmitJobRequest>>>;

    /// A `TcpListener` as an accepted-connection stream, so `tonic`'s server
    /// can be handed a loopback listener without the `tokio-stream` `net`
    /// feature (not enabled workspace-wide).
    struct AcceptStream(TcpListener);

    impl Stream for AcceptStream {
        type Item = std::io::Result<TcpStream>;

        fn poll_next(
            self: Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            use std::task::Poll;
            match self.0.poll_accept(cx) {
                Poll::Ready(Ok((stream, _peer))) => Poll::Ready(Some(Ok(stream))),
                Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
                Poll::Pending => Poll::Pending,
            }
        }
    }

    /// A `JobService` double whose only LIVE method is `submit_job`; every
    /// other method is unreachable from these tests, so a wiring mistake
    /// panics loudly rather than returning a plausible-looking placeholder.
    struct SubmitCapturingJobService {
        submitted: CapturedSubmit,
    }

    #[tonic::async_trait]
    impl JobService for SubmitCapturingJobService {
        type WaitJobStream = Pin<Box<dyn Stream<Item = Result<JobEvent, Status>> + Send + 'static>>;

        async fn submit_job(
            &self,
            request: Request<SubmitJobRequest>,
        ) -> Result<Response<SubmitJobResponse>, Status> {
            *self.submitted.lock().unwrap() = Some(request.into_inner());
            Ok(Response::new(SubmitJobResponse {
                job_id: "job-1".to_string(),
                kind: "fine_tune".to_string(),
                output_model_id: String::new(),
            }))
        }

        async fn job_status(
            &self,
            _request: Request<JobStatusRequest>,
        ) -> Result<Response<JobStatusResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn wait_job(
            &self,
            _request: Request<JobHandle>,
        ) -> Result<Response<Self::WaitJobStream>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn list_jobs(
            &self,
            _request: Request<ListJobsRequest>,
        ) -> Result<Response<ListJobsResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn cancel_job(
            &self,
            _request: Request<CancelJobRequest>,
        ) -> Result<Response<CancelJobResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn list_workers(
            &self,
            _request: Request<ListWorkersRequest>,
        ) -> Result<Response<ListWorkersResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn prune_jobs(
            &self,
            _request: Request<PruneJobsRequest>,
        ) -> Result<Response<PruneJobsResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }
    }

    /// Spin up a loopback tonic server hosting only the capturing double and
    /// connect a client to it. The serve task is detached: nothing here needs
    /// graceful shutdown, since process teardown reclaims the port.
    async fn connected_client() -> (DataClient, CapturedSubmit) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr: SocketAddr = listener.local_addr().expect("local_addr");
        let submitted: CapturedSubmit = Arc::new(Mutex::new(None));
        let svc = JobServiceServer::new(SubmitCapturingJobService {
            submitted: Arc::clone(&submitted),
        });
        tokio::spawn(async move {
            Server::builder()
                .add_service(svc)
                .serve_with_incoming(AcceptStream(listener))
                .await
                .expect("capturing server");
        });
        let endpoint = Endpoint::from_shared(format!("http://{addr}")).expect("endpoint");
        let client = DataClient::connect(endpoint)
            .await
            .expect("data client connect");
        (client, submitted)
    }

    /// A request for the two-column fine-tune both tests submit, differing
    /// only in the count — so the count is the only determinant of any
    /// difference the handler sees.
    fn request(world_size: Option<NonZeroU32>) -> FineTuneRequest {
        FineTuneRequest {
            source: "patents".to_string(),
            base_model: "local:tiny-bert".to_string(),
            columns: vec!["abstract".to_string()],
            method: FineTuneMethod::Lora,
            task: ModelTask::TextEmbedding,
            config: None,
            world_size,
            cache: CachePolicy::Bypass,
        }
    }

    /// [`request`], but the cache policy is the only determinant chosen —
    /// used by the tests below that pin the `cache` field's encode.
    fn cache_request(cache: CachePolicy) -> FineTuneRequest {
        FineTuneRequest {
            cache,
            ..request(None)
        }
    }

    /// SET. A chosen count reaches the server as that count — the client does
    /// not drop, clamp, or reinterpret it.
    #[tokio::test]
    async fn a_chosen_rank_count_reaches_the_server() {
        let (client, submitted) = connected_client().await;

        client
            .submit_fine_tune(request(NonZeroU32::new(2)))
            .await
            .expect("submit returns the double's handle");

        let received = submitted.lock().unwrap().take().expect("handler ran");
        assert_eq!(received.world_size, 2);
    }

    /// UNSET. An unchosen count reaches the server as `0` — the wire's unset,
    /// which the engine resolves to a single rank. Asserted against the
    /// received request, so this pins what the server sees rather than what
    /// the client meant.
    #[tokio::test]
    async fn an_unchosen_rank_count_reaches_the_server_unset() {
        let (client, submitted) = connected_client().await;

        client
            .submit_fine_tune(request(None))
            .await
            .expect("submit returns the double's handle");

        let received = submitted.lock().unwrap().take().expect("handler ran");
        assert_eq!(received.world_size, 0);
    }

    /// SET TO ONE. An explicit single rank is the same job as an unchosen
    /// count, so it is the same REQUEST: the two submissions are compared as
    /// encoded bytes, not as a field read, because bytes are what a server (or
    /// another client's request built for the same intent) actually sees. The
    /// Python client already encodes an explicit `1` as unset
    /// (`clients/python/jammi/_assembly.py`, `_wire_world_size`), so this is
    /// also the cross-client agreement: one wire encoding per intent.
    #[tokio::test]
    async fn an_explicit_single_rank_encodes_as_the_unset_request() {
        let (client, submitted) = connected_client().await;

        client
            .submit_fine_tune(request(NonZeroU32::new(1)))
            .await
            .expect("submit returns the double's handle");
        let via_one = submitted.lock().unwrap().take().expect("handler ran");

        client
            .submit_fine_tune(request(None))
            .await
            .expect("submit returns the double's handle");
        let via_unset = submitted.lock().unwrap().take().expect("handler ran");

        assert_eq!(
            via_one.encode_to_vec(),
            via_unset.encode_to_vec(),
            "an explicit rank count of one must put the SAME bytes on the wire as an \
             unchosen count: both denote the single-rank job, and `0` is the wire's \
             unset (got world_size = {} vs {})",
            via_one.world_size,
            via_unset.world_size,
        );
    }

    /// The six-parameter [`DataClient::fine_tune`] is exactly
    /// [`DataClient::submit_fine_tune`] with the count unset: every caller
    /// that predates the count keeps submitting the single-rank job it always
    /// did, and the two verbs agree field-for-field on everything else (the
    /// whole received request is compared, not only the count).
    #[tokio::test]
    async fn the_six_parameter_verb_submits_the_identical_unset_request() {
        let (client, submitted) = connected_client().await;

        client
            .fine_tune(
                "patents",
                "local:tiny-bert",
                &["abstract".to_string()],
                FineTuneMethod::Lora,
                ModelTask::TextEmbedding,
                None,
            )
            .await
            .expect("submit returns the double's handle");
        let via_verb = submitted.lock().unwrap().take().expect("handler ran");

        client
            .submit_fine_tune(request(None))
            .await
            .expect("submit returns the double's handle");
        let via_request = submitted.lock().unwrap().take().expect("handler ran");

        assert_eq!(via_verb.world_size, 0);
        assert_eq!(via_verb, via_request);
    }

    /// SET. `CachePolicy::Use` reaches the server as the wire's concrete
    /// `CACHE_POLICY_USE` — the client does not drop or default it.
    #[tokio::test]
    async fn a_chosen_cache_policy_reaches_the_server_as_use() {
        let (client, submitted) = connected_client().await;

        client
            .submit_fine_tune(cache_request(CachePolicy::Use))
            .await
            .expect("submit returns the double's handle");

        let received = submitted.lock().unwrap().take().expect("handler ran");
        assert_eq!(
            received.cache,
            jammi_wire::proto::inference::CachePolicy::Use as i32
        );
    }

    /// UNSET (the default). `CachePolicy::Bypass` reaches the server as the
    /// wire's `CACHE_POLICY_UNSPECIFIED` — the implicit-presence enum's `0`,
    /// which the engine's own decode resolves to `Bypass` identically to an
    /// explicit `CACHE_POLICY_BYPASS`
    /// (`unspecified_and_bypass_cache_both_decode_to_bypass`,
    /// `crates/jammi-ai/src/wire/training.rs`) — so this pins the DECODE
    /// property, not the byte identity; see
    /// [`a_bypass_cache_policy_leaves_the_field_off_the_wire`] for the bytes.
    #[tokio::test]
    async fn the_default_cache_policy_reaches_the_server_as_bypass() {
        let (client, submitted) = connected_client().await;

        client
            .submit_fine_tune(cache_request(CachePolicy::Bypass))
            .await
            .expect("submit returns the double's handle");

        let received = submitted.lock().unwrap().take().expect("handler ran");
        assert_eq!(
            received.cache,
            jammi_wire::proto::inference::CachePolicy::Unspecified as i32
        );
    }

    /// UNSET, BYTES. `CachePolicy::Bypass` — the engine default — leaves the
    /// `cache` field OFF the wire entirely: the request this client actually
    /// puts on the wire is compared, byte for byte, against a reference
    /// message assembled independently of `DataClient::submit_fine_tune`'s
    /// own encode (`..Default::default()` never touches `cache`, so the
    /// reference's field is `0` by construction, not by mirroring the
    /// production code under test). A regression back to an explicit
    /// `CACHE_POLICY_BYPASS` (`2`) diverges from this reference and fails —
    /// the same no-regression property
    /// `an_explicit_single_rank_encodes_as_the_unset_request` pins for
    /// `world_size`, and the Python client's own golden-bytes oracle
    /// (`clients/python/tests/test_cache_policy.py`,
    /// `test_default_leaves_the_field_off_the_encoding_entirely`) pins for
    /// `cache`.
    #[tokio::test]
    async fn a_bypass_cache_policy_leaves_the_field_off_the_wire() {
        use jammi_wire::method_to_proto;
        use jammi_wire::proto::inference::ModelTask as ProtoModelTask;
        use jammi_wire::proto::job::submit_job_request::Spec;
        use jammi_wire::proto::training::FineTuneSpec;

        let (client, submitted) = connected_client().await;

        client
            .submit_fine_tune(cache_request(CachePolicy::Bypass))
            .await
            .expect("submit returns the double's handle");
        let received = submitted.lock().unwrap().take().expect("handler ran");

        // Built from the SAME source values `cache_request(None)`'s
        // `request(None)` fixture carries, but never assigning `cache` at
        // all — `..Default::default()` leaves it at the message's own
        // zero-value default, independent of whatever
        // `DataClient::submit_fine_tune` does.
        let reference = SubmitJobRequest {
            spec: Some(Spec::FineTune(FineTuneSpec {
                source: "patents".to_string(),
                columns: vec!["abstract".to_string()],
                method: method_to_proto(FineTuneMethod::Lora) as i32,
                task: ProtoModelTask::TextEmbedding as i32,
            })),
            base_model: "local:tiny-bert".to_string(),
            config: None,
            idempotency_key: String::new(),
            world_size: 0,
            ..Default::default()
        };

        assert_eq!(
            received.encode_to_vec(),
            reference.encode_to_vec(),
            "an explicit `CachePolicy::Bypass` must put the SAME bytes on the wire as a \
             reference message that never assigns `cache` at all (got cache = {} vs the \
             reference's {})",
            received.cache,
            reference.cache,
        );
    }
}

/// `ModelResult.cache_outcome` decodes off
/// `DataClient::job_status` the same way `TableResult.cache_outcome`
/// already does: `job_status` returns the raw wire [`JobStatusResponse`]
/// verbatim (no dedicated per-field accessor on either arm of the `result`
/// oneof — see [`DataClient::fine_tune_metrics`]'s doc), so a caller reads
/// `resp.result`'s `Model` arm's `cache_outcome` directly, and this test
/// pins that a genuine gRPC round trip (not a hand-built value) carries it.
/// A `JobService` double, not a live engine, because reproducing an actual
/// `FineTune` cache hit is this crate's OWN server-side concern
/// (`crates/jammi-server/tests/it/grpc_job.rs`,
/// `a_fine_tune_cache_hit_reports_cache_outcome_reused_over_the_wire`); this
/// test's only determinant is the client's decode of a `JobStatusResponse`
/// a handler already produced.
#[cfg(test)]
mod model_result_cache_outcome_tests {
    use std::net::SocketAddr;
    use std::pin::Pin;

    use futures::Stream;
    use tokio::net::{TcpListener, TcpStream};
    use tonic::transport::{Endpoint, Server};
    use tonic::{Request, Response, Status};

    use jammi_wire::proto::job::job_service_server::{JobService, JobServiceServer};
    use jammi_wire::proto::job::{
        job_status_response::Result as WireResult, CancelJobRequest, CancelJobResponse, JobEvent,
        JobHandle, JobStatusRequest, JobStatusResponse, ListJobsRequest, ListJobsResponse,
        ListWorkersRequest, ListWorkersResponse, ModelResult, PruneJobsRequest, PruneJobsResponse,
        SubmitJobRequest, SubmitJobResponse,
    };

    use super::DataClient;

    struct AcceptStream(TcpListener);

    impl Stream for AcceptStream {
        type Item = std::io::Result<TcpStream>;

        fn poll_next(
            self: Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            use std::task::Poll;
            match self.0.poll_accept(cx) {
                Poll::Ready(Ok((stream, _peer))) => Poll::Ready(Some(Ok(stream))),
                Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
                Poll::Pending => Poll::Pending,
            }
        }
    }

    /// A `JobService` double whose only LIVE method is `job_status`, always
    /// answering with a completed `FineTune` cache-hit result — every other
    /// method is unreachable, so a wiring mistake panics loudly rather than
    /// returning a plausible-looking placeholder.
    struct CacheHitJobService;

    #[tonic::async_trait]
    impl JobService for CacheHitJobService {
        type WaitJobStream = Pin<Box<dyn Stream<Item = Result<JobEvent, Status>> + Send + 'static>>;

        async fn submit_job(
            &self,
            _request: Request<SubmitJobRequest>,
        ) -> Result<Response<SubmitJobResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn job_status(
            &self,
            _request: Request<JobStatusRequest>,
        ) -> Result<Response<JobStatusResponse>, Status> {
            Ok(Response::new(JobStatusResponse {
                status: "completed".to_string(),
                kind: "fine_tune".to_string(),
                progress: None,
                error: String::new(),
                output_model_id: "jammi:fine-tuned:second".to_string(),
                result: Some(WireResult::Model(ModelResult {
                    model_id: "jammi:fine-tuned:second".to_string(),
                    artifact_path: "file:///artifacts/first".to_string(),
                    metrics_json: None,
                    cache_outcome: Some(jammi_wire::cache_outcome_to_proto(
                        &jammi_db::store::CacheOutcome::Reused(
                            jammi_db::store::ReusedArtifact::Model(
                                jammi_db::catalog::artifact_repo::ArtifactRef::parse(
                                    "file:///artifacts/first",
                                )
                                .unwrap(),
                            ),
                        ),
                    )),
                })),
                acceleration_report_json: None,
            }))
        }

        async fn wait_job(
            &self,
            _request: Request<JobHandle>,
        ) -> Result<Response<Self::WaitJobStream>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn list_jobs(
            &self,
            _request: Request<ListJobsRequest>,
        ) -> Result<Response<ListJobsResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn cancel_job(
            &self,
            _request: Request<CancelJobRequest>,
        ) -> Result<Response<CancelJobResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn list_workers(
            &self,
            _request: Request<ListWorkersRequest>,
        ) -> Result<Response<ListWorkersResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }

        async fn prune_jobs(
            &self,
            _request: Request<PruneJobsRequest>,
        ) -> Result<Response<PruneJobsResponse>, Status> {
            unreachable!("not exercised by this fixture")
        }
    }

    async fn connected_client() -> DataClient {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr: SocketAddr = listener.local_addr().expect("local_addr");
        let svc = JobServiceServer::new(CacheHitJobService);
        tokio::spawn(async move {
            Server::builder()
                .add_service(svc)
                .serve_with_incoming(AcceptStream(listener))
                .await
                .expect("cache-hit server");
        });
        let endpoint = Endpoint::from_shared(format!("http://{addr}")).expect("endpoint");
        DataClient::connect(endpoint)
            .await
            .expect("data client connect")
    }

    /// `DataClient::job_status` relays `ModelResult.cache_outcome` verbatim —
    /// the wire message a genuine `JobService` handler produced, decoded off
    /// an actual (loopback) gRPC round trip, not a hand-built value — and it
    /// decodes to the engine outcome naming the reused artifact.
    #[tokio::test]
    async fn job_status_relays_the_model_cache_outcome_verbatim() {
        let client = connected_client().await;

        let resp = client
            .job_status("job-2")
            .await
            .expect("job_status returns the double's response");

        match resp.result {
            Some(WireResult::Model(m)) => {
                assert_eq!(
                    jammi_wire::cache_outcome_from_proto(m.cache_outcome).unwrap(),
                    jammi_db::store::CacheOutcome::Reused(jammi_db::store::ReusedArtifact::Model(
                        jammi_db::catalog::artifact_repo::ArtifactRef::parse(
                            "file:///artifacts/first"
                        )
                        .unwrap(),
                    ))
                );
            }
            other => panic!("expected a Model result, got {other:?}"),
        }
    }
}
