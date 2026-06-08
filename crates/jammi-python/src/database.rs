use std::collections::{BTreeMap, HashMap};
use std::str::FromStr;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use datafusion::execution::context::SessionContext;
use futures::StreamExt;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use pyo3::Borrowed;
use pyo3_arrow::{PySchema, PyTable};

use jammi_ai::fine_tune::{EarlyStoppingMetric, FineTuneConfig, FineTuneMethod};
use jammi_ai::local_session::{
    LocalSession, Modality, QueryInput, SearchQuery, SearchRequest, Session,
};
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::pipeline::context_predictor::{ContextServeOptions, ContextServeSource};
use jammi_ai::pipeline::context_set::{
    ContextRequest, ContextSource, ContextSourceKind, HybridMerge, SetAggregator,
};
use jammi_ai::pipeline::graph_neighbourhood::{EdgeDirection, EdgeGather, EdgeSourceRef};
use jammi_ai::pipeline::graph_propagation::{
    PropagateRequest, PropagationOutput, PropagationWeighting,
};
use jammi_ai::session::InferenceSession;
use jammi_db::config::JammiConfig;
use jammi_db::error::JammiError;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::mutable::{
    MutableIndexDef, MutableTableDefinitionBuilder, MutableTableError, MutableTableId,
};
use jammi_db::trigger::{Offset, Predicate, TopicDefinition, TopicId};
use jammi_lora::BackboneDtype;

use crate::convert::{batches_to_pyarrow, serializable_to_pydict};
use crate::error::to_pyerr;
use crate::job::PyFineTuneJob;
use crate::model_task::ModelTaskArg;

/// Python Database wrapping `Arc<InferenceSession>` with a shared tokio runtime.
#[pyclass(name = "Database")]
pub struct PyDatabase {
    pub(crate) session: Arc<InferenceSession>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
    /// Guards spawning the ephemeral timeout scanner exactly once per
    /// connection, on the first `ephemeral_session` call.
    ephemeral_scanner: std::sync::Once,
}

impl PyDatabase {
    /// Construct a `PyDatabase` from a fully-built `JammiConfig`. This is
    /// the Rust-shaped entry point that the `#[pyfunction] connect` wraps
    /// for Python callers; downstream Rust consumers (e.g. a Python-bindings
    /// layer built on top of this crate) call it directly so
    /// the resulting database shares the tokio runtime that drives every
    /// `InferenceSession` future.
    pub fn open(config: JammiConfig) -> Result<Self, JammiError> {
        let runtime = Arc::new(tokio::runtime::Runtime::new()?);
        let session = runtime.block_on(InferenceSession::open(config))?;
        Ok(Self {
            session,
            runtime,
            ephemeral_scanner: std::sync::Once::new(),
        })
    }

    /// Spawn the ephemeral timeout scanner on first use. The scanner runs on
    /// the shared runtime for the lifetime of the connection; the spawn must
    /// happen inside the runtime context, so it is driven through `block_on`.
    fn ensure_ephemeral_scanner(&self) {
        self.ephemeral_scanner.call_once(|| {
            let session = Arc::clone(&self.session);
            self.runtime.block_on(async {
                session.spawn_ephemeral_timeout_scanner(jammi_db::ephemeral::DEFAULT_SCAN_INTERVAL);
            });
        });
    }

    /// Return a cloned `Arc<InferenceSession>` so external consumers can
    /// share this database's session — schema upgrade lock, trigger broker,
    /// catalog cache, tenant binding — instead of opening a parallel
    /// session against the same artifact directory.
    pub fn session_arc(&self) -> Arc<InferenceSession> {
        Arc::clone(&self.session)
    }

    /// Wrap this database's engine as the transport-agnostic [`Session`] front
    /// door (the local arm). The unified `encode_query` / `generate_embeddings`
    /// verbs dispatch the `modality` onto the engine's concrete tower through
    /// this one surface — the same shape the bundled `jammi-client` speaks
    /// remotely, so the embedded and remote verb vocabularies agree by sharing
    /// the `Session` abstraction rather than by a parallel reimplementation.
    fn local_session(&self) -> Session {
        Session::Local(LocalSession::new(Arc::clone(&self.session)))
    }
}

#[pymethods]
impl PyDatabase {
    /// Bind a tenant scope to this connection.
    ///
    /// Subsequent reads return rows whose `tenant_id` matches `tenant_id`
    /// plus globally-scoped (`tenant_id IS NULL`) rows; writes carry the
    /// bound tenant on every row. Pass an empty string to clear.
    fn with_tenant(&self, tenant_id: &str) -> PyResult<()> {
        if tenant_id.is_empty() {
            self.session.unbind_tenant();
            return Ok(());
        }
        let t = jammi_db::TenantId::from_str(tenant_id).map_err(to_pyerr)?;
        self.session.bind_tenant(t);
        Ok(())
    }

    /// The tenant currently bound to this connection, or `None`.
    fn tenant(&self) -> Option<String> {
        self.session.tenant().map(|t| t.to_string())
    }

    /// Handle to the per-query audit primitive: `db.audit.log([...])`,
    /// `db.audit.fetch_by_query_id(...)`, `db.audit.fetch_recent(...)`.
    /// Shares this connection's session, runtime, and tenant binding.
    #[getter]
    fn audit(&self) -> crate::audit::PyAuditHandle {
        crate::audit::PyAuditHandle {
            session: Arc::clone(&self.session),
            runtime: Arc::clone(&self.runtime),
        }
    }

    /// Open an ephemeral, session-scoped storage context bound to the tenant
    /// currently set via `with_tenant`. Use as a context manager:
    ///
    /// ```python
    /// with db.ephemeral_session(timeout_seconds=3600) as ephem:
    ///     ephem.create_ephemeral_table("imgs", schema=s, primary_key=["id"])
    ///     ephem.insert("imgs", batch=tbl)
    /// # tables deleted on exit; a `closed` event is published
    /// ```
    ///
    /// Raises `ValueError` if no tenant is bound. Timeout enforcement runs via
    /// the in-process scanner spawned by the first `ephemeral_session` call on
    /// this connection.
    #[pyo3(signature = (*, timeout_seconds))]
    fn ephemeral_session(
        &self,
        timeout_seconds: u64,
    ) -> PyResult<crate::ephemeral::PyEphemeralSession> {
        self.ensure_ephemeral_scanner();
        let timeout = std::time::Duration::from_secs(timeout_seconds);
        let session = self
            .runtime
            .block_on(self.session.ephemeral_session(timeout))
            .map_err(crate::ephemeral::ephemeral_err)?;
        Ok(crate::ephemeral::PyEphemeralSession::new(
            session,
            Arc::clone(&self.runtime),
        ))
    }

    /// Register a file-shaped data source. `url` accepts a local path
    /// (parsed into `file://...`) or any storage URL the build was
    /// compiled with: `s3://bucket/key`, `gs://bucket/key`,
    /// `azure://container/blob`.
    #[pyo3(signature = (name, *, url, format))]
    fn add_source(&self, name: &str, url: &str, format: &str) -> PyResult<()> {
        let file_format = parse_file_format(format)?;
        let connection = SourceConnection::parse(url, file_format).map_err(to_pyerr)?;
        self.runtime
            .block_on(self.session.add_source(name, SourceType::File, connection))
            .map_err(to_pyerr)
    }

    /// List a descriptor for every source registered to the current tenant.
    /// Each is a dict carrying `source_id`, `source_type`, `status`, and
    /// `result_tables` (the embedding result tables produced from it, each
    /// with its own `status` / `row_count` / `dimensions`). Registry
    /// introspection, not a SQL query.
    fn list_sources(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let descriptors = self
            .runtime
            .block_on(self.session.catalog().list_source_descriptors())
            .map_err(to_pyerr)?;
        serializable_to_pydict(py, &descriptors)
    }

    /// Describe one registered source by id, or `None` when no source with
    /// that id is visible to the current tenant. Returns the same dict shape
    /// `list_sources` yields per entry.
    fn describe_source(&self, py: Python<'_>, source_id: &str) -> PyResult<Option<Py<PyAny>>> {
        let descriptor = self
            .runtime
            .block_on(self.session.catalog().describe_source(source_id))
            .map_err(to_pyerr)?;
        descriptor
            .map(|d| serializable_to_pydict(py, &d))
            .transpose()
    }

    /// The engine's capabilities handshake: a dict with `version`, `features`
    /// (compiled feature flags), and `storage_backends` (addressable storage
    /// URL schemes). A compile-time fact about the running build. Named to match
    /// the bundled `jammi-client`'s `get_server_info` (and `SessionService.
    /// GetServerInfo`) so the embedded and remote surfaces agree.
    fn get_server_info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serializable_to_pydict(py, &jammi_db::ServerInfo::current())
    }

    /// Execute a SQL query. Returns a `pyarrow.Table`.
    fn sql(&self, py: Python<'_>, query: &str) -> PyResult<Py<PyAny>> {
        let batches = self
            .runtime
            .block_on(self.session.sql(query))
            .map_err(to_pyerr)?;
        batches_to_pyarrow(py, &batches)
    }

    /// Generate embeddings for `columns` of a registered source with the given
    /// model and modality, persisting one L2-normalized vector per row. The
    /// `modality` (`"text"`/`"image"`/`"audio"`) selects the tower; the image
    /// and audio towers take exactly one content column. Returns the result
    /// table name. The unified form — one verb keyed by modality, identical to
    /// the bundled `jammi-client`'s.
    #[pyo3(signature = (*, source, model, columns, key, modality=None))]
    fn generate_embeddings(
        &self,
        source: &str,
        model: &str,
        columns: Vec<String>,
        key: &str,
        modality: Option<ModalityArg>,
    ) -> PyResult<String> {
        let modality = modality.map(|m| m.0).unwrap_or(Modality::Text);
        let record = self
            .runtime
            .block_on(
                self.local_session()
                    .generate_embeddings(source, model, &columns, key, modality),
            )
            .map_err(to_pyerr)?;
        Ok(record.table_name)
    }

    /// Run inference. Returns a `pyarrow.Table`.
    #[pyo3(signature = (*, source, model, columns, task, key))]
    fn infer(
        &self,
        py: Python<'_>,
        source: &str,
        model: &str,
        columns: Vec<String>,
        task: ModelTaskArg,
        key: &str,
    ) -> PyResult<Py<PyAny>> {
        let model_source = ModelSource::parse(model);
        let batches = self
            .runtime
            .block_on(
                self.session
                    .infer(source, &model_source, task.0, &columns, key),
            )
            .map_err(to_pyerr)?;
        batches_to_pyarrow(py, &batches)
    }

    /// List registered topic names visible to the current tenant binding.
    fn list_topics(&self) -> PyResult<Vec<String>> {
        let topic_repo = self.session.topic_repo();
        let tenant = self.session.tenant();
        let topics = self
            .runtime
            .block_on(topic_repo.list_topics(tenant))
            .map_err(to_pyerr)?;
        Ok(topics.into_iter().map(|t| t.name).collect())
    }

    /// Publish one batch of rows to a topic. `batch` is a `pyarrow.Table`
    /// (zero-copy import via the Arrow C Stream Interface) whose schema
    /// must match the topic's. Returns the engine-assigned offset.
    ///
    /// The publish is scoped to the session's currently-bound tenant —
    /// rows land with `tenant_id` equal to whatever `Database.tenant` is
    /// set to (or `NULL` if the session is unscoped). For a tenant-pinned
    /// topic, this must match the topic's tenant.
    #[pyo3(signature = (topic, *, batch))]
    fn publish_topic(&self, topic: &str, batch: PyTable) -> PyResult<u64> {
        let topic_repo = self.session.topic_repo();
        let tenant = self.session.tenant();
        let topic_def = self
            .runtime
            .block_on(topic_repo.lookup_by_name(topic, tenant))
            .map_err(to_pyerr)?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("topic '{topic}' not found"))
            })?;
        let (batches, _schema) = batch.into_inner();
        // The publisher accepts one RecordBatch per call; concatenate the
        // streamed chunks so a multi-chunk pyarrow.Table publishes as one
        // logical event.
        let concatenated = if batches.len() == 1 {
            batches
                .into_iter()
                .next()
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("publish batch was empty"))?
        } else {
            let schema = batches[0].schema();
            arrow::compute::concat_batches(&schema, batches.iter())
                .map_err(|e| to_pyerr(datafusion::error::DataFusionError::from(e)))?
        };
        let publisher = self.session.publisher();
        let offset = self
            .runtime
            .block_on(publisher.publish_scoped(&topic_def, tenant, concatenated))
            .map_err(to_pyerr)?;
        Ok(offset.value())
    }

    /// Open a subscription, collect up to `max_batches` matching batches
    /// (replay + live tail joined), then close. Returns the concatenated
    /// payload as a `pyarrow.Table`.
    ///
    /// Synchronous collect API: streaming iteration is left to the gRPC
    /// `TriggerService.Subscribe` surface where back-pressure flows
    /// through HTTP/2 naturally. This binding is the script-friendly
    /// equivalent for one-shot Python workflows.
    #[pyo3(signature = (topic, *, predicate=None, from_offset=None, max_batches=64))]
    fn subscribe_collect(
        &self,
        py: Python<'_>,
        topic: &str,
        predicate: Option<&str>,
        from_offset: Option<u64>,
        max_batches: usize,
    ) -> PyResult<Py<PyAny>> {
        let topic_repo = self.session.topic_repo();
        let tenant = self.session.tenant();
        let topic_def = self
            .runtime
            .block_on(topic_repo.lookup_by_name(topic, tenant))
            .map_err(to_pyerr)?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("topic '{topic}' not found"))
            })?;
        let predicate = Predicate::from_sql(
            &SessionContext::new(),
            Arc::clone(&topic_def.schema),
            predicate.unwrap_or(""),
        )
        .map_err(to_pyerr)?;
        let from = from_offset.map(|v| Offset::new(v, chrono::Utc::now()));
        let subscriber = self.session.subscriber();
        let collected: Vec<RecordBatch> = self
            .runtime
            .block_on(async move {
                let mut stream = subscriber.subscribe(&topic_def, predicate, from).await?;
                let mut out: Vec<RecordBatch> = Vec::new();
                while out.len() < max_batches {
                    match StreamExt::next(&mut stream).await {
                        Some(Ok(d)) => out.push(d.batch),
                        Some(Err(e)) => return Err(e),
                        None => break,
                    }
                }
                Ok::<_, jammi_db::trigger::TriggerError>(out)
            })
            .map_err(to_pyerr)?;
        batches_to_pyarrow(py, &collected)
    }

    /// Register a new evidence-provenance channel. `columns` is a list of
    /// `(name, dtype)` tuples where `dtype` is one of `"Float32"`, `"Float64"`,
    /// `"Int32"`, `"Int64"`, `"Utf8"`, `"Boolean"`. Priority orders the
    /// channel against others on the same row when several contribute the
    /// same column. The channel id is unique across the catalog: passing
    /// one that already exists raises `RuntimeError` carrying
    /// `EvidenceChannel("channel '<id>': already exists")`.
    #[pyo3(signature = (channel_id, *, priority, columns))]
    fn register_channel(
        &self,
        channel_id: &str,
        priority: i32,
        columns: Vec<(String, String)>,
    ) -> PyResult<()> {
        let id = jammi_db::ChannelId::new(channel_id).map_err(to_pyerr)?;
        let cols = parse_channel_columns(&columns)?;
        let spec = jammi_db::catalog::channel_repo::ChannelSpec {
            id,
            priority,
            columns: cols,
        };
        self.runtime
            .block_on(self.session.catalog().channels().register(&spec))
            .map_err(to_pyerr)
    }

    /// Append columns to an already-registered channel. The append-only
    /// invariant is enforced: redeclaring an existing column with a
    /// different dtype raises `RuntimeError` carrying
    /// `EvidenceChannel("channel '<id>': column '<name>' was declared <X>,
    /// cannot redeclare as <Y>")`.
    #[pyo3(signature = (channel_id, *, columns))]
    fn add_channel_columns(
        &self,
        channel_id: &str,
        columns: Vec<(String, String)>,
    ) -> PyResult<()> {
        let id = jammi_db::ChannelId::new(channel_id).map_err(to_pyerr)?;
        let cols = parse_channel_columns(&columns)?;
        self.runtime
            .block_on(self.session.catalog().channels().add_columns(&id, &cols))
            .map_err(to_pyerr)
    }

    /// Register a mutable companion table.
    ///
    /// Tenant scope is inherited from the session's currently bound tenant
    /// (set via `with_tenant`). `schema` is any object implementing the
    /// Arrow PyCapsule schema interface (e.g. `pyarrow.Schema`,
    /// `arro3.Schema`). `primary_key` is a non-empty list of column names
    /// drawn from `schema`. `indexes` is an optional list of dicts of shape
    /// `{"name": str, "columns": [str, ...], "unique": bool=False}` — one
    /// `CREATE INDEX` per entry. `order_column` is an optional `Int64` /
    /// `UInt64` column that enables `MutableTableRegistry::scan_after`
    /// streaming reads. `chunk_size` overrides the default scan chunk size
    /// of 8192. Returns the catalog id of the registered table.
    #[pyo3(signature = (name, *, schema, primary_key, indexes=None, order_column=None, chunk_size=None))]
    fn create_mutable_table(
        &self,
        name: String,
        schema: PySchema,
        primary_key: Vec<String>,
        indexes: Option<Vec<Bound<'_, PyDict>>>,
        order_column: Option<String>,
        chunk_size: Option<usize>,
    ) -> PyResult<String> {
        let id = MutableTableId::new(&name).map_err(to_pyerr)?;
        let schema_ref = schema.into_inner();

        let mut builder = MutableTableDefinitionBuilder::new(id.clone(), schema_ref)
            .primary_key(primary_key)
            .tenant(self.session.tenant());

        if let Some(specs) = indexes {
            for idx in parse_index_specs(&specs)? {
                builder = builder.index(idx);
            }
        }
        if let Some(col) = order_column {
            builder = builder.order_column(col);
        }
        if let Some(size) = chunk_size {
            builder = builder.chunk_size(size);
        }

        let def = builder.build().map_err(to_pyerr)?;
        self.runtime
            .block_on(self.session.create_mutable_table(def))
            .map_err(to_pyerr)?;
        Ok(id.as_str().to_string())
    }

    /// Drop a mutable companion table. If `if_exists` is true, dropping a
    /// table that is not registered is a no-op; otherwise it raises with the
    /// typed `MutableTableError::NotFound` variant.
    #[pyo3(signature = (name, *, if_exists=false))]
    fn drop_mutable_table(&self, name: String, if_exists: bool) -> PyResult<()> {
        let id = MutableTableId::new(&name).map_err(to_pyerr)?;
        match self.runtime.block_on(self.session.drop_mutable_table(&id)) {
            Ok(()) => Ok(()),
            Err(JammiError::MutableTable(MutableTableError::NotFound(_))) if if_exists => Ok(()),
            Err(e) => Err(to_pyerr(e)),
        }
    }

    /// Register a trigger-stream topic. The schema is the contract every
    /// published batch must satisfy; the catalog encoder rejects DataTypes
    /// outside the supported wire types with `TriggerError::UnsupportedSchemaType`.
    /// `broker_metadata` is opaque driver-side configuration (retention,
    /// replication, etc.). Returns the engine-minted topic id.
    #[pyo3(signature = (name, *, schema, broker_metadata=None))]
    fn register_topic(
        &self,
        name: String,
        schema: PySchema,
        broker_metadata: Option<BTreeMap<String, String>>,
    ) -> PyResult<String> {
        let topic = TopicDefinition {
            id: TopicId::new(),
            name,
            schema: schema.into_inner(),
            tenant: self.session.tenant(),
            broker_metadata: broker_metadata.unwrap_or_default(),
        };
        self.runtime
            .block_on(self.session.trigger_broker().register_topic(&topic))
            .map_err(to_pyerr)?;
        self.runtime
            .block_on(self.session.topic_repo().register_topic(&topic))
            .map_err(to_pyerr)?;
        Ok(topic.id.to_string())
    }

    /// Drop a trigger-stream topic by name. Resolves the topic via the
    /// tenant-scoped lookup, deletes the catalog row + backing table, and
    /// then asks the broker driver to release any in-memory state. Broker
    /// driver failure after the catalog row is gone is surfaced via tracing
    /// — the system of record is already consistent.
    #[pyo3(signature = (name, *, if_exists=false))]
    fn drop_topic(&self, name: String, if_exists: bool) -> PyResult<()> {
        let tenant = self.session.tenant();
        let topic_repo = self.session.topic_repo();
        let topic_opt = self
            .runtime
            .block_on(topic_repo.lookup_by_name(&name, tenant))
            .map_err(to_pyerr)?;
        match topic_opt {
            Some(t) => {
                self.runtime
                    .block_on(topic_repo.drop_topic(t.id, tenant))
                    .map_err(to_pyerr)?;
                if let Err(e) = self
                    .runtime
                    .block_on(self.session.trigger_broker().drop_topic(t.id))
                {
                    tracing::warn!(
                        topic_id = %t.id,
                        error = %e,
                        "trigger broker driver failed to drop topic after catalog row removal",
                    );
                }
                Ok(())
            }
            None if if_exists => Ok(()),
            None => Err(PyValueError::new_err(format!("topic '{name}' not found"))),
        }
    }

    /// Nearest-neighbor search over a source's embedding table. Returns a
    /// `pyarrow.Table` directly — the same shape, the same call, as the bundled
    /// `jammi-client`'s `search` (and the typed `Search` gRPC verb). `filter` is
    /// an optional SQL predicate over the hydrated results; `select` projects
    /// columns (empty keeps every hydrated column). For compound retrieval
    /// (join / model inference over the results), use `query` (SQL).
    #[pyo3(signature = (source, *, query, k, filter=None, select=None))]
    fn search(
        &self,
        py: Python<'_>,
        source: &str,
        query: Vec<f32>,
        k: usize,
        filter: Option<String>,
        select: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let request = SearchRequest {
            source_id: source.to_string(),
            query: SearchQuery::Vector(query),
            k,
            filter,
            select: select.unwrap_or_default(),
        };
        let batches = self
            .runtime
            .block_on(self.local_session().search(request))
            .map_err(to_pyerr)?;
        batches_to_pyarrow(py, &batches)
    }

    /// Start a fine-tuning job. Returns a `FineTuneJob` handle.
    ///
    /// All config kwargs are optional — omitting them uses the Jammi defaults:
    ///   lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
    ///   learning_rate=2e-4, epochs=3, batch_size=8, max_seq_length=512,
    ///   validation_fraction=0.10, early_stopping_patience=3,
    ///   warmup_steps=100, gradient_accumulation_steps=1,
    ///   triplet_margin=0.5 (embedding_loss="triplet" must be set for custom margin),
    ///   target_modules=[] trains a projection head over the frozen base model;
    ///     a non-empty list (e.g. ["Wqkv","Wo"]) injects LoRA into the encoder's
    ///     internal attention/FFN linears at the listed sites instead.
    ///   early_stopping_metric="val_loss" | "train_loss".
    ///     "train_loss" replicates train_embedding_model.py without --val-file:
    ///     set validation_fraction=0.0 and the full dataset trains without a held-out split.
    ///   backbone_dtype="f32" | "bf16" | "f16" — dtype for frozen backbone weights.
    ///     "bf16" cuts backbone VRAM by ~half; LoRA A/B always stay in f32.
    ///   weight_decay — AdamW L2 regularization. Default: 0.01 (matches train_embedding_model.py).
    ///   max_grad_norm — global gradient clipping norm. Default: 1.0. Pass 0.0 to disable.
    ///   embedding_loss — "cosent" | "angle" | "cosine_mse" | "triplet" | "mnrl".
    ///     "mnrl" (Multiple-Negatives-Ranking / in-batch negatives) trains
    ///     (anchor, positive) pairs or (anchor, positive, negative) triplets;
    ///     unset auto-selects from the data format. Setting "triplet" with a
    ///     custom margin still uses triplet_margin.
    ///   mnrl_temperature — MNRL similarity scale. Default: 20.0.
    ///   cached — MNRL GradCache: enlarge the in-batch-negative pool to the whole
    ///     dataset without the memory cost. Only affects embedding_loss="mnrl".
    ///   mine_hard_negatives — mine hard negatives from jammi's own ANN index.
    ///   hard_negative_k / hard_negative_exclude_hops / hard_negative_refresh_every —
    ///     mining knobs (negatives per anchor; hops of the positive's
    ///     neighbourhood excluded as false-negative guard; epochs between re-mines).
    ///   matryoshka_dims — train truncatable embeddings at these prefix dims
    ///     (e.g. [768, 512, 256, 128, 64]); empty trains the full dimension only.
    #[pyo3(signature = (
        *,
        source,
        base_model,
        columns,
        method,
        task = None,
        lora_rank = None,
        lora_alpha = None,
        lora_dropout = None,
        learning_rate = None,
        epochs = None,
        batch_size = None,
        max_seq_length = None,
        validation_fraction = None,
        early_stopping_patience = None,
        warmup_steps = None,
        gradient_accumulation_steps = None,
        triplet_margin = None,
        target_modules = None,
        early_stopping_metric = None,
        backbone_dtype = None,
        weight_decay = None,
        max_grad_norm = None,
        embedding_loss = None,
        mnrl_temperature = None,
        cached = None,
        mine_hard_negatives = None,
        hard_negative_k = None,
        hard_negative_exclude_hops = None,
        hard_negative_refresh_every = None,
        matryoshka_dims = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fine_tune(
        &self,
        source: &str,
        base_model: &str,
        columns: Vec<String>,
        method: &str,
        task: Option<ModelTaskArg>,
        lora_rank: Option<usize>,
        lora_alpha: Option<f64>,
        lora_dropout: Option<f64>,
        learning_rate: Option<f64>,
        epochs: Option<usize>,
        batch_size: Option<usize>,
        max_seq_length: Option<usize>,
        validation_fraction: Option<f64>,
        early_stopping_patience: Option<usize>,
        warmup_steps: Option<usize>,
        gradient_accumulation_steps: Option<usize>,
        triplet_margin: Option<f64>,
        target_modules: Option<Vec<String>>,
        early_stopping_metric: Option<&str>,
        backbone_dtype: Option<&str>,
        weight_decay: Option<f64>,
        max_grad_norm: Option<f64>,
        embedding_loss: Option<&str>,
        mnrl_temperature: Option<f64>,
        cached: Option<bool>,
        mine_hard_negatives: Option<bool>,
        hard_negative_k: Option<usize>,
        hard_negative_exclude_hops: Option<usize>,
        hard_negative_refresh_every: Option<usize>,
        matryoshka_dims: Option<Vec<usize>>,
    ) -> PyResult<PyFineTuneJob> {
        let mut cfg = FineTuneConfig::default();
        if let Some(v) = lora_rank {
            cfg.lora_rank = v;
        }
        if let Some(v) = lora_alpha {
            cfg.lora_alpha = v;
        }
        if let Some(v) = lora_dropout {
            cfg.lora_dropout = v;
        }
        if let Some(v) = learning_rate {
            cfg.learning_rate = v;
        }
        if let Some(v) = epochs {
            cfg.epochs = v;
        }
        if let Some(v) = batch_size {
            cfg.batch_size = v;
        }
        if let Some(v) = max_seq_length {
            cfg.max_seq_length = v;
        }
        if let Some(v) = validation_fraction {
            cfg.validation_fraction = v;
        }
        if let Some(v) = early_stopping_patience {
            cfg.early_stopping_patience = v;
        }
        if let Some(v) = warmup_steps {
            cfg.warmup_steps = v;
        }
        if let Some(v) = gradient_accumulation_steps {
            cfg.gradient_accumulation_steps = v;
        }
        // Embedding loss: the named objective parameterised by its scalar knob.
        // `triplet_margin` / `mnrl_temperature` parameterise the triplet / MNRL
        // variants; naming "triplet"/"mnrl" without the knob uses the default.
        // An unnamed loss with only `triplet_margin` set keeps the historical
        // shorthand (margin implies triplet).
        use jammi_ai::fine_tune::EmbeddingLoss;
        match embedding_loss {
            Some("cosent") => cfg.embedding_loss = Some(EmbeddingLoss::CoSent),
            Some("angle") => cfg.embedding_loss = Some(EmbeddingLoss::AnglE),
            Some("cosine_mse") => cfg.embedding_loss = Some(EmbeddingLoss::CosineMse),
            Some("triplet") => {
                cfg.embedding_loss = Some(EmbeddingLoss::Triplet {
                    margin: triplet_margin.unwrap_or(0.3),
                });
            }
            Some("mnrl") => {
                cfg.embedding_loss = Some(EmbeddingLoss::MultipleNegativesRanking {
                    temperature: mnrl_temperature.unwrap_or(20.0),
                });
            }
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown embedding_loss '{other}'. Use 'cosent', 'angle', \
                     'cosine_mse', 'triplet', or 'mnrl'."
                )))
            }
            None => {
                if let Some(m) = triplet_margin {
                    cfg.embedding_loss = Some(EmbeddingLoss::Triplet { margin: m });
                }
            }
        }
        if let Some(v) = cached {
            cfg.cached = v;
        }
        if let Some(v) = mine_hard_negatives {
            cfg.hard_negatives.mine = v;
        }
        if let Some(v) = hard_negative_k {
            cfg.hard_negatives.k = v;
        }
        if let Some(v) = hard_negative_exclude_hops {
            cfg.hard_negatives.exclude_hops = v;
        }
        if let Some(v) = hard_negative_refresh_every {
            cfg.hard_negatives.refresh_every = v;
        }
        if let Some(v) = matryoshka_dims {
            cfg.matryoshka_dims = v;
        }
        if let Some(v) = target_modules {
            cfg.target_modules = v;
        }
        if let Some(metric) = early_stopping_metric {
            cfg.early_stopping_metric = match metric {
                "train_loss" => EarlyStoppingMetric::TrainLoss,
                "val_loss" => EarlyStoppingMetric::ValLoss,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown early_stopping_metric '{other}'. Use 'val_loss' or 'train_loss'."
                    )))
                }
            };
        }
        if let Some(dtype_str) = backbone_dtype {
            cfg.backbone_dtype = match dtype_str {
                "f32" => BackboneDtype::F32,
                "bf16" => BackboneDtype::BF16,
                "f16" => BackboneDtype::F16,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown backbone_dtype '{other}'. Use 'f32', 'bf16', or 'f16'."
                    )))
                }
            };
        }
        if let Some(v) = weight_decay {
            cfg.weight_decay = v;
        }
        if let Some(v) = max_grad_norm {
            cfg.max_grad_norm = v;
        }

        let task = task.map(|t| t.0).unwrap_or(ModelTask::TextEmbedding);
        let job = self
            .runtime
            .block_on(self.session.fine_tune(
                source,
                base_model,
                &columns,
                method.parse::<FineTuneMethod>().map_err(to_pyerr)?,
                task,
                Some(cfg),
            ))
            .map_err(to_pyerr)?;
        Ok(PyFineTuneJob::new(job, Arc::clone(&self.runtime)))
    }

    /// Train an amortized in-context predictor (S19) over a source: a
    /// config-selectable CNP / Attentive-CNP / TNP that meta-learns to turn a
    /// retrieved context set into a predictive distribution, adapting to a new
    /// target without retraining. Returns the model id it registered under.
    ///
    /// The predictor is meta-trained episodically: each **task** (the distinct
    /// values of `task_column`) is split into a context set and held-out targets,
    /// tasks (not points) are partitioned into train/test, and the target's
    /// outcome (`value_column`) is scored under the chosen S18 proper objective.
    ///
    /// - `architecture` — "cnp" | "attncnp" | "tnp".
    /// - `output` — "gaussian" (a `(mean, std)` head) or "quantile".
    /// - `objective` — for a Gaussian head: "crps" | "nll" | "betanll" (β via
    ///   `beta`); ignored for a quantile head (always pinball over `levels`).
    /// - `levels` — ascending quantile levels in (0, 1) for the quantile head.
    #[pyo3(signature = (
        source,
        *,
        key_column,
        task_column,
        value_column,
        architecture = "attncnp",
        output = "gaussian",
        objective = "crps",
        context_k = 32,
        hidden_dim = 64,
        num_heads = 4,
        num_layers = 2,
        levels = None,
        beta = 0.5,
        epochs = 100,
        learning_rate = 0.005,
        grad_clip = 1.0,
        test_task_fraction = 0.2,
        min_task_count = 4,
        seed = 0,
        model_id = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn train_context_predictor(
        &self,
        source: &str,
        key_column: &str,
        task_column: &str,
        value_column: &str,
        architecture: &str,
        output: &str,
        objective: &str,
        context_k: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_layers: usize,
        levels: Option<Vec<f64>>,
        beta: f64,
        epochs: usize,
        learning_rate: f64,
        grad_clip: f64,
        test_task_fraction: f64,
        min_task_count: usize,
        seed: u64,
        model_id: Option<&str>,
    ) -> PyResult<String> {
        use jammi_ai::pipeline::context_predictor::{
            ContextArchitecture, ContextPredictorTrainConfig, GaussianObjective, PredictiveHead,
        };

        let architecture = match architecture {
            "cnp" => ContextArchitecture::Cnp,
            "attncnp" => ContextArchitecture::AttnCnp,
            "tnp" => ContextArchitecture::Tnp,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown architecture '{other}'. Use 'cnp', 'attncnp', or 'tnp'."
                )))
            }
        };
        let head = match output {
            "gaussian" => {
                let objective = match objective {
                    "crps" => GaussianObjective::Crps,
                    "nll" => GaussianObjective::Nll { beta: 0.0 },
                    "betanll" => GaussianObjective::Nll { beta },
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown gaussian objective '{other}'. Use 'crps', 'nll', or 'betanll'."
                        )))
                    }
                };
                PredictiveHead::Gaussian { objective }
            }
            "quantile" => {
                let levels = levels.ok_or_else(|| {
                    PyValueError::new_err(
                        "output='quantile' requires `levels` (ascending levels in (0, 1))",
                    )
                })?;
                PredictiveHead::Quantile { levels }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown output '{other}'. Use 'gaussian' or 'quantile'."
                )))
            }
        };

        let spec = ContextPredictorTrainConfig {
            model_id: model_id
                .map(str::to_string)
                .unwrap_or_else(|| format!("{source}-context-predictor")),
            architecture,
            key_column: key_column.to_string(),
            task_column: task_column.to_string(),
            value_column: value_column.to_string(),
            context_k,
            hidden_dim,
            num_heads,
            num_layers,
            head,
            epochs,
            learning_rate,
            grad_clip,
            test_task_fraction,
            min_task_count,
            seed,
        };
        let record = self
            .runtime
            .block_on(self.session.train_context_predictor(source, &spec))
            .map_err(to_pyerr)?;
        Ok(record.model_id)
    }

    /// Predict a target's distribution with a trained context predictor (S19) by
    /// assembling its live context and running one in-context forward — no
    /// gradient update. Returns a dict: `{"kind": "gaussian", "mean", "std"}` or
    /// `{"kind": "quantile", "levels": [[level, value], …]}`.
    ///
    /// `source` is the source whose embedding table the live context is drawn
    /// from (it need not equal the training source — a predictor serves a target
    /// in any corpus of the same shape); `split` optionally scopes the serving
    /// context. Passing `edge_source` (a registered edge table) assembles a
    /// declared-edge context anchored at `target_key` instead of (or, with
    /// `hybrid_ann_k`, in union with) the embedding-similarity context.
    ///
    /// The returned dict also carries `"source"` (`"ann"`/`"edges"`/`"hybrid"` —
    /// how the context was assembled) and `"context_ref"` (the context member
    /// keys), so a graph-conditioned prediction is never unattributed.
    #[pyo3(signature = (
        model_id, *, source, target_key, split = None,
        edge_source = None, edge_src_column = None, edge_dst_column = None,
        edge_type_column = None, edge_weight_column = None, edge_hops = None,
        edge_fanout = None, edge_direction = None, edge_types = None,
        min_weight = None, hybrid_ann_k = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn predict_with_context_predictor(
        &self,
        py: Python<'_>,
        model_id: &str,
        source: &str,
        target_key: &str,
        split: Option<String>,
        edge_source: Option<String>,
        edge_src_column: Option<String>,
        edge_dst_column: Option<String>,
        edge_type_column: Option<String>,
        edge_weight_column: Option<String>,
        edge_hops: Option<usize>,
        edge_fanout: Option<usize>,
        edge_direction: Option<String>,
        edge_types: Option<Vec<String>>,
        min_weight: Option<f64>,
        hybrid_ann_k: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        use jammi_ai::pipeline::context_predictor::PredictedDistribution;

        let gather = edge_gather_from_kwargs(
            edge_source,
            edge_src_column,
            edge_dst_column,
            edge_type_column,
            edge_weight_column,
            edge_hops,
            edge_fanout,
            edge_direction,
            edge_types,
            min_weight,
        )?;
        let serve_source = match (gather, hybrid_ann_k) {
            (None, _) => ContextServeSource::Ann,
            (Some(edges), None) => ContextServeSource::Edges(edges),
            (Some(edges), Some(ann_k)) => ContextServeSource::Hybrid {
                ann_k,
                edges,
                merge: HybridMerge::Union,
            },
        };
        let options = ContextServeOptions {
            source: serve_source,
            split,
        };

        let served = self
            .runtime
            .block_on(
                self.session
                    .load_context_predictor(model_id, source, options),
            )
            .map_err(to_pyerr)?;
        let prediction = self
            .runtime
            .block_on(
                self.session
                    .predict_with_context_predictor_provenanced(&served, target_key),
            )
            .map_err(to_pyerr)?;

        let out = PyDict::new(py);
        match prediction.distribution {
            PredictedDistribution::Gaussian { mean, std } => {
                out.set_item("kind", "gaussian")?;
                out.set_item("mean", mean)?;
                out.set_item("std", std)?;
            }
            PredictedDistribution::Quantile { levels } => {
                out.set_item("kind", "quantile")?;
                let pairs: Vec<(f64, f32)> = levels;
                out.set_item("levels", pairs)?;
            }
        }
        out.set_item("source", context_source_tag(prediction.source))?;
        out.set_item("context_ref", prediction.context_keys)?;
        Ok(out.into_any().unbind())
    }

    /// Graph-supervised fine-tune (S11): learn an embedding metric that encodes
    /// a graph's structure. Reads a node-text source and an edge source, samples
    /// the graph into `(anchor, positive, [hard_negative])` pairs by biased
    /// random walks (node2vec), and trains the existing in-batch-negative
    /// (MNRL) / triplet objective — **no new loss**.
    ///
    /// Sources:
    ///   node_source / id_column / text_column — the node text the encoder
    ///     embeds, keyed by id.
    ///   edge_source / src_column / dst_column — directed edges; endpoints join
    ///     to id_column.
    ///   edge_provenance — "declared" | "similarity". **The load-bearing
    ///     distinction:** training on "similarity" (S9 k-NN) edges largely
    ///     re-learns the base metric (a degenerate feedback loop), so genuine
    ///     gain comes from "declared" external edges (hierarchy, crosswalk,
    ///     citation, confirmed pairs). Similarity edges are a weak bootstrap,
    ///     never the sole supervision.
    ///
    /// Walk / negative knobs (node2vec):
    ///   walk_length (L) — positive reach; 1 is degenerate 1-hop. Default 4.
    ///   walks_per_node — walks started per node. Default 2.
    ///   return_p (p) / in_out_q (q) — node2vec bias. Defaults 1.0 / 1.0.
    ///   graph_hard_negatives — structure-mined hard negatives per pair; 0 uses
    ///     in-batch negatives only. Default 1.
    ///   exclude_hops — hops of the anchor's neighbourhood excluded from its
    ///     negative pool (the false-negative guard). Default 1.
    ///   min_negatives — minimum negative pool (collapse guard). Default 1.
    ///   sample_seed — walk/negative RNG seed. Default 0.
    ///
    /// Training knobs mirror `fine_tune` for the subset relevant to graph
    /// supervision (loss, temperature, batch, epochs, lora, matryoshka).
    #[pyo3(signature = (
        *,
        node_source,
        id_column,
        text_column,
        edge_source,
        src_column,
        dst_column,
        base_model,
        edge_provenance = "declared",
        walk_length = None,
        walks_per_node = None,
        return_p = None,
        in_out_q = None,
        graph_hard_negatives = None,
        exclude_hops = None,
        min_negatives = None,
        sample_seed = None,
        embedding_loss = None,
        mnrl_temperature = None,
        epochs = None,
        batch_size = None,
        learning_rate = None,
        lora_rank = None,
        matryoshka_dims = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fine_tune_graph(
        &self,
        node_source: &str,
        id_column: &str,
        text_column: &str,
        edge_source: &str,
        src_column: &str,
        dst_column: &str,
        base_model: &str,
        edge_provenance: &str,
        walk_length: Option<usize>,
        walks_per_node: Option<usize>,
        return_p: Option<f64>,
        in_out_q: Option<f64>,
        graph_hard_negatives: Option<usize>,
        exclude_hops: Option<usize>,
        min_negatives: Option<usize>,
        sample_seed: Option<u64>,
        embedding_loss: Option<&str>,
        mnrl_temperature: Option<f64>,
        epochs: Option<usize>,
        batch_size: Option<usize>,
        learning_rate: Option<f64>,
        lora_rank: Option<usize>,
        matryoshka_dims: Option<Vec<usize>>,
    ) -> PyResult<PyFineTuneJob> {
        use jammi_ai::fine_tune::graph_sampler::{
            EdgeProvenance, GraphFineTuneSources, GraphSampleConfig,
        };
        use jammi_ai::fine_tune::EmbeddingLoss;

        let provenance = match edge_provenance {
            "declared" => EdgeProvenance::Declared,
            "similarity" => EdgeProvenance::Similarity,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown edge_provenance '{other}'. Use 'declared' or 'similarity'. \
                     Note: 'similarity' (S9) edges largely re-learn the base metric — a \
                     weak bootstrap, not the sole supervision."
                )))
            }
        };

        // Walk / negative sampling config — defaults match the Rust struct.
        let mut sample = GraphSampleConfig::default();
        if let Some(v) = walk_length {
            sample.walk_length = v;
        }
        if let Some(v) = walks_per_node {
            sample.walks_per_node = v;
        }
        if let Some(v) = return_p {
            sample.return_p = v;
        }
        if let Some(v) = in_out_q {
            sample.in_out_q = v;
        }
        if let Some(v) = graph_hard_negatives {
            sample.hard_negatives = v;
        }
        if let Some(v) = exclude_hops {
            sample.exclude_hops = v;
        }
        if let Some(v) = min_negatives {
            sample.min_negatives = v;
        }
        if let Some(v) = sample_seed {
            sample.seed = v;
        }

        // Training config — the graph-relevant subset of `fine_tune`'s knobs.
        // The default embedding loss for graph supervision is MNRL (S10).
        let loss = match embedding_loss {
            None | Some("mnrl") => EmbeddingLoss::MultipleNegativesRanking {
                temperature: mnrl_temperature.unwrap_or(20.0),
            },
            Some("triplet") => EmbeddingLoss::Triplet { margin: 0.3 },
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "Unknown embedding_loss '{other}' for graph fine-tune. Use 'mnrl' \
                     (default) or 'triplet'."
                )))
            }
        };
        let mut cfg = FineTuneConfig {
            embedding_loss: Some(loss),
            ..FineTuneConfig::default()
        };
        if let Some(v) = epochs {
            cfg.epochs = v;
        }
        if let Some(v) = batch_size {
            cfg.batch_size = v;
        }
        if let Some(v) = learning_rate {
            cfg.learning_rate = v;
        }
        if let Some(v) = lora_rank {
            cfg.lora_rank = v;
        }
        if let Some(v) = matryoshka_dims {
            cfg.matryoshka_dims = v;
        }

        let sources = GraphFineTuneSources {
            node_source: node_source.to_string(),
            id_column: id_column.to_string(),
            text_column: text_column.to_string(),
            edge_source: edge_source.to_string(),
            src_column: src_column.to_string(),
            dst_column: dst_column.to_string(),
            provenance,
        };

        let job = self
            .runtime
            .block_on(
                self.session
                    .fine_tune_graph(&sources, base_model, sample, Some(cfg)),
            )
            .map_err(to_pyerr)?;
        Ok(PyFineTuneJob::new(job, Arc::clone(&self.runtime)))
    }

    /// Evaluate embedding quality. Returns a dict with `aggregate` (mean
    /// over all queries) and `per_query` (one record per golden-set query)
    /// keys.
    ///
    /// `cohorts` optionally maps a golden-set `query_id` to an opaque
    /// `{key: value}` segment map, persisted with that query's per-query
    /// metrics (read back via `db.eval_per_query(...)`, spec J9).
    #[pyo3(signature = (*, source, golden_source, model=None, k=10, cohorts=None))]
    fn eval_embeddings(
        &self,
        py: Python<'_>,
        source: &str,
        golden_source: &str,
        model: Option<&str>,
        k: usize,
        cohorts: Option<HashMap<String, BTreeMap<String, String>>>,
    ) -> PyResult<Py<PyAny>> {
        let cohorts = cohorts.unwrap_or_default();
        let report = self
            .runtime
            .block_on(
                self.session
                    .eval_embeddings(source, model, golden_source, k, &cohorts),
            )
            .map_err(to_pyerr)?;
        serializable_to_pydict(py, &report)
    }

    /// Read back the persisted per-query eval records for a run (spec J9),
    /// scoped to the calling tenant. Returns a list of dicts, each carrying
    /// `eval_run_id`, `query_id`, `cohorts` (a dict), and `metrics` (a dict
    /// of `recall@1/3/5/10`, `mrr`, `ndcg`, `distance`).
    fn eval_per_query(&self, py: Python<'_>, eval_run_id: &str) -> PyResult<Py<PyAny>> {
        let records = self
            .runtime
            .block_on(self.session.eval_per_query(eval_run_id))
            .map_err(to_pyerr)?;

        let out = pyo3::types::PyList::empty(py);
        for rec in records {
            let item = PyDict::new(py);
            item.set_item("eval_run_id", rec.eval_run_id)?;
            item.set_item("query_id", rec.query_id)?;
            // `cohorts` and `metrics` are stored as JSON objects; decode them
            // into Python dicts so callers get structured values, not strings.
            let json_mod = py.import("json")?;
            let cohorts = json_mod.call_method1("loads", (rec.cohorts_json,))?;
            let metrics = json_mod.call_method1("loads", (rec.metrics_json,))?;
            item.set_item("cohorts", cohorts)?;
            item.set_item("metrics", metrics)?;
            out.append(item)?;
        }
        Ok(out.unbind().into())
    }

    /// Evaluate inference quality. Returns a dict with `aggregate`
    /// (task-shaped, tagged by `"task"`) and `per_record` (one record per
    /// predicted/gold pair) keys.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, model, source, columns, task, golden_source, label_column))]
    fn eval_inference(
        &self,
        py: Python<'_>,
        model: &str,
        source: &str,
        columns: Vec<String>,
        task: &str,
        golden_source: &str,
        label_column: &str,
    ) -> PyResult<Py<PyAny>> {
        let eval_task = parse_eval_task(task)?;
        let report = self
            .runtime
            .block_on(self.session.eval_inference(
                model,
                source,
                &columns,
                eval_task,
                golden_source,
                label_column,
            ))
            .map_err(to_pyerr)?;
        serializable_to_pydict(py, &report)
    }

    /// Compare multiple embedding tables side-by-side. Returns a dict with a
    /// `per_table` list; the first entry is the baseline (`delta: None`)
    /// and every subsequent entry carries a `delta` against it.
    #[pyo3(signature = (*, embedding_tables, source, golden_source, k=10))]
    fn eval_compare(
        &self,
        py: Python<'_>,
        embedding_tables: Vec<String>,
        source: &str,
        golden_source: &str,
        k: usize,
    ) -> PyResult<Py<PyAny>> {
        let report = self
            .runtime
            .block_on(
                self.session
                    .eval_compare(&embedding_tables, source, golden_source, k),
            )
            .map_err(to_pyerr)?;
        serializable_to_pydict(py, &report)
    }

    /// Evaluate whether a predictor's uncertainty is honest (spec R2). Returns a
    /// dict with `aggregate` (the proper-score headline `crps`/`nll`, the
    /// `adaptive_ece` PIT-calibration diagnostic, `sharpness`, `coverage`),
    /// `per_cohort` (coverage + CRPS with n + CI per cohort), and `per_record`
    /// keys.
    ///
    /// `golden_source` pairs a held-out predictive distribution with its
    /// realised `outcome`. `shape` is `"gaussian"` (columns `record_id`, `mean`,
    /// `sd`, `outcome`) or `"sample"` (columns `record_id`, `draws` as a JSON
    /// array per row, `outcome`). `cohorts` optionally maps a `record_id` to an
    /// opaque `{key: value}` segment map persisted with that record's per-record
    /// scores (read back via `db.eval_per_query(...)`).
    #[pyo3(signature = (*, source, golden_source, shape, cohorts=None))]
    fn eval_calibration(
        &self,
        py: Python<'_>,
        source: &str,
        golden_source: &str,
        shape: &str,
        cohorts: Option<HashMap<String, BTreeMap<String, String>>>,
    ) -> PyResult<Py<PyAny>> {
        let shape = parse_calibration_shape(shape)?;
        let cohorts = cohorts.unwrap_or_default();
        let report = self
            .runtime
            .block_on(
                self.session
                    .eval_calibration(source, golden_source, shape, &cohorts),
            )
            .map_err(to_pyerr)?;
        serializable_to_pydict(py, &report)
    }

    /// Encode a single query into an embedding vector using the given model.
    /// `modality` selects the tower (`"text"`/`"image"`/`"audio"`); `query` is a
    /// string for the text tower or raw bytes for the image/audio tower. The
    /// unified form — one verb keyed by modality, identical to the bundled
    /// `jammi-client`'s.
    #[pyo3(signature = (*, model, query, modality=None))]
    fn encode_query(
        &self,
        model: &str,
        query: QueryArg,
        modality: Option<ModalityArg>,
    ) -> PyResult<Vec<f32>> {
        let modality = modality.map(|m| m.0).unwrap_or(Modality::Text);
        self.runtime
            .block_on(self.local_session().encode_query(model, query.0, modality))
            .map_err(to_pyerr)
    }

    /// Preload a model into the cache without running inference.
    fn preload_model(&self, model_id: &str) -> PyResult<()> {
        let source = ModelSource::parse(model_id);
        self.runtime
            .block_on(self.session.model_cache().get_or_load(
                &source,
                ModelTask::TextEmbedding,
                None,
            ))
            .map_err(to_pyerr)?;
        Ok(())
    }

    /// Materialize the k-nearest-neighbour graph of a source's embedding table
    /// and return the new edge table's name.
    ///
    /// This is for *global-structure* work — clustering, near-duplicate
    /// detection, training-data prep — where the whole edge set is consumed as
    /// a durable artifact. For "neighbours of *these* rows", use `search`.
    ///
    /// The returned table has columns `(src, dst, rank, similarity)`, with
    /// `src`/`dst` joining directly to the source on the key. The default
    /// driver is index-assisted and produces an approximate, non-deterministic
    /// graph; pass `exact=True` for a deterministic, complete one (gated by a
    /// row-count ceiling). `min_similarity` floors weak edges; `mutual=True`
    /// keeps only reciprocal edges.
    #[pyo3(signature = (source, *, k, min_similarity=None, mutual=false, exact=false, table=None))]
    fn build_neighbor_graph(
        &self,
        source: &str,
        k: usize,
        min_similarity: Option<f32>,
        mutual: bool,
        exact: bool,
        table: Option<String>,
    ) -> PyResult<String> {
        let params = jammi_ai::pipeline::neighbor_graph::BuildNeighborGraph {
            k,
            min_similarity,
            mutual,
            exact,
            ..Default::default()
        };
        let record = self
            .runtime
            .block_on(
                self.session
                    .build_neighbor_graph(source, table.as_deref(), &params),
            )
            .map_err(to_pyerr)?;
        Ok(record.table_name)
    }

    /// Propagate an embedding table's features over a declared graph (the
    /// decoupled-GNN forward pass) into a new, searchable embedding table.
    ///
    /// Each output row is its `hops`-hop neighbourhood aggregate of the input
    /// embeddings, with self-loops (`Ã = A + I`) and an APPNP `alpha`-teleport
    /// restart that anchors every node against over-smoothing. The graph is
    /// either an S9 similarity graph (`edge_graph_table`, a `build_neighbor_graph`
    /// output) or a registered external edge source (`edge_source` with
    /// `edge_src_column`/`edge_dst_column`); pass exactly one.
    ///
    /// `weighting` selects the neighbour normalisation: `"degree_normalized"`
    /// (the default, symmetric `Â`, the PageRank-decay form), `"uniform"`
    /// (random-walk mean), or `"edge_similarity"` (edge-weighted mean, using the
    /// edge weight as fixed attention). `output` is `"final"` (a `d`-dim table)
    /// or `"jumping_knowledge"` (the L2-normalised per-hop concat, `(K+1)·d`-dim,
    /// indexing in its own space). Deterministic: identical inputs yield a
    /// byte-identical table.
    ///
    /// Returns the materialised embedding table's name.
    #[pyo3(signature = (
        source,
        *,
        embedding_table = None,
        edge_graph_table = None,
        edge_source = None,
        edge_src_column = None,
        edge_dst_column = None,
        edge_weight_column = None,
        direction = None,
        hops = None,
        weighting = None,
        alpha = None,
        output = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn propagate_embeddings(
        &self,
        source: &str,
        embedding_table: Option<String>,
        edge_graph_table: Option<String>,
        edge_source: Option<String>,
        edge_src_column: Option<String>,
        edge_dst_column: Option<String>,
        edge_weight_column: Option<String>,
        direction: Option<&str>,
        hops: Option<usize>,
        weighting: Option<&str>,
        alpha: Option<f64>,
        output: Option<&str>,
    ) -> PyResult<String> {
        let edge_source_ref = match (edge_graph_table, edge_source) {
            (Some(table_name), None) => EdgeSourceRef::NeighborGraph { table_name },
            (None, Some(source_id)) => EdgeSourceRef::Registered {
                source_id,
                src_column: edge_src_column.unwrap_or_else(|| "src".into()),
                dst_column: edge_dst_column.unwrap_or_else(|| "dst".into()),
                type_column: None,
                weight_column: edge_weight_column,
                as_of_column: None,
            },
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "pass exactly one of edge_graph_table (S9 graph) or edge_source \
                     (registered edges), not both",
                ))
            }
            (None, None) => {
                return Err(PyValueError::new_err(
                    "propagate_embeddings requires a graph: edge_graph_table or edge_source",
                ))
            }
        };

        let direction = match direction {
            None | Some("out") => EdgeDirection::Out,
            Some("in") => EdgeDirection::In,
            Some("undirected") => EdgeDirection::Undirected,
            Some(o) => {
                return Err(PyValueError::new_err(format!(
                    "direction must be 'out', 'in', or 'undirected' (got '{o}')"
                )))
            }
        };
        let weighting = match weighting {
            None | Some("degree_normalized") => PropagationWeighting::DegreeNormalized,
            Some("uniform") => PropagationWeighting::Uniform,
            Some("edge_similarity") => PropagationWeighting::EdgeSimilarity,
            Some(o) => {
                return Err(PyValueError::new_err(format!(
                    "weighting must be 'degree_normalized', 'uniform', or 'edge_similarity' \
                     (got '{o}')"
                )))
            }
        };
        let output = match output {
            None | Some("final") => PropagationOutput::Final,
            Some("jumping_knowledge") => PropagationOutput::JumpingKnowledge,
            Some(o) => {
                return Err(PyValueError::new_err(format!(
                    "output must be 'final' or 'jumping_knowledge' (got '{o}')"
                )))
            }
        };

        let mut request = PropagateRequest::new(source, edge_source_ref)
            .with_direction(direction)
            .with_weighting(weighting)
            .with_output(output);
        if let Some(t) = embedding_table {
            request = request.with_embedding_table(t);
        }
        if let Some(h) = hops {
            request = request.with_hops(h);
        }
        if let Some(a) = alpha {
            request = request.with_alpha(a);
        }

        let record = self
            .runtime
            .block_on(self.session.propagate_embeddings(&request))
            .map_err(to_pyerr)?;
        Ok(record.table_name)
    }

    /// Conformalize a classification predictor into prediction sets.
    ///
    /// Split (inductive) conformal: `calibration` holds one row of per-class
    /// probabilities per held-out example and `true_labels[i]` is the realised
    /// class for row `i`; the calibration scores yield the finite-sample
    /// `⌈(n+1)(1-alpha)⌉` quantile, which is applied to every row of `test` to
    /// emit a prediction set with marginal coverage `>= 1 - alpha`.
    ///
    /// `score` selects the nonconformity family: `"lac"`, `"aps"` (default), or
    /// `"raps"` (regularized APS). For `"raps"`, `raps_params` is the
    /// `(lambda, k_reg)` regularization pair — the penalty weight and the 1-based
    /// rank past which it applies; it is ignored by `"lac"`/`"aps"` and defaults
    /// to `(0.0, 1)`. The calibration set must be disjoint from both the
    /// training set and `test` — reusing test points inflates coverage. Pure and
    /// deterministic: identical inputs yield identical sets.
    ///
    /// Returns one list of admitted class indices per row of `test`.
    #[pyo3(signature = (calibration, true_labels, test, *, alpha, score=None, raps_params=None))]
    fn conformalize(
        &self,
        calibration: Vec<Vec<f64>>,
        true_labels: Vec<usize>,
        test: Vec<Vec<f64>>,
        alpha: f64,
        score: Option<&str>,
        raps_params: Option<(f64, usize)>,
    ) -> PyResult<Vec<Vec<usize>>> {
        let (raps_lambda, raps_k_reg) = raps_params.unwrap_or((0.0, 1));
        let score = parse_class_score(score.unwrap_or("aps"), raps_lambda, raps_k_reg)?;
        let model = jammi_ai::predict::ConformalModel::classification(
            &calibration,
            &true_labels,
            score,
            alpha,
        )
        .map_err(to_pyerr)?;
        test.iter()
            .map(|row| model.predict_set(row, None).map_err(to_pyerr))
            .collect()
    }

    /// Conformalize an absolute-residual regression predictor into
    /// constant-width prediction intervals.
    ///
    /// The calibration nonconformity is `|y - ŷ|` over the `predictions`/
    /// `observed` held-out pairs; the finite-sample quantile `q̂` then yields
    /// `[ŷ - q̂, ŷ + q̂]` around each `test_predictions` point, with marginal
    /// coverage `>= 1 - alpha`. The calibration set must be disjoint from the
    /// training set and the test points.
    ///
    /// Returns one `(lower, upper)` tuple per test row.
    #[pyo3(signature = (predictions, observed, test_predictions, *, alpha))]
    fn conformalize_interval(
        &self,
        predictions: Vec<f64>,
        observed: Vec<f64>,
        test_predictions: Vec<f64>,
        alpha: f64,
    ) -> PyResult<Vec<(f64, f64)>> {
        use jammi_ai::predict::{ConformalModel, IntervalScore};
        let model = ConformalModel::regression(
            &predictions,
            &[],
            &[],
            &observed,
            IntervalScore::AbsoluteResidual,
            alpha,
        )
        .map_err(to_pyerr)?;
        test_predictions
            .iter()
            .map(|&p| model.predict_interval(p, 0.0, 0.0, None).map_err(to_pyerr))
            .collect()
    }

    /// Conformalize a Conformalized Quantile Regression (CQR) predictor into
    /// adaptive-width prediction intervals.
    ///
    /// The calibration nonconformity is `max(q_lo - y, y - q_hi)` over the
    /// `lower`/`upper` quantile estimates and `observed` targets; the
    /// finite-sample quantile `q̂` then yields `[q_lo - q̂, q_hi + q̂]` around
    /// each `test_lower`/`test_upper` band, so interval width tracks the
    /// predictor's local uncertainty. The calibration set must be disjoint from
    /// the training set and the test points.
    ///
    /// Returns one `(lower, upper)` tuple per test row.
    #[pyo3(signature = (lower, upper, observed, test_lower, test_upper, *, alpha))]
    fn conformalize_cqr(
        &self,
        lower: Vec<f64>,
        upper: Vec<f64>,
        observed: Vec<f64>,
        test_lower: Vec<f64>,
        test_upper: Vec<f64>,
        alpha: f64,
    ) -> PyResult<Vec<(f64, f64)>> {
        use jammi_ai::predict::{ConformalModel, IntervalScore};
        if test_lower.len() != test_upper.len() {
            return Err(PyValueError::new_err(
                "cqr conformal: 'test_lower' and 'test_upper' must have equal length",
            ));
        }
        let model =
            ConformalModel::regression(&[], &lower, &upper, &observed, IntervalScore::Cqr, alpha)
                .map_err(to_pyerr)?;
        test_lower
            .iter()
            .zip(test_upper.iter())
            .map(|(&l, &u)| model.predict_interval(0.0, l, u, None).map_err(to_pyerr))
            .collect()
    }

    /// Fuse several ranked retrieval lists into one by reciprocal-rank fusion.
    ///
    /// `ranked_lists` is a list of best-first `_row_id` lists — typically the
    /// dense (ANN) order and the lexical (BM25) order, plus any further ranked
    /// channel. Fusion is on *rank*, never raw score, so the dense and lexical
    /// scales never need reconciling. `k_rrf` damps deep ranks (default 60,
    /// robust across 40–80). Returns `(row_id, rrf_score)` tuples sorted by
    /// fused score descending, ties broken ascending by `row_id` — fully
    /// deterministic and independent of the order the lists are supplied in.
    #[pyo3(signature = (ranked_lists, *, k_rrf=None))]
    fn rrf_fuse(
        &self,
        ranked_lists: Vec<Vec<String>>,
        k_rrf: Option<u32>,
    ) -> PyResult<Vec<(String, f64)>> {
        let k = k_rrf.unwrap_or(jammi_ai::query::DEFAULT_K_RRF);
        Ok(jammi_ai::query::rrf_fuse(&ranked_lists, k)
            .into_iter()
            .map(|h| (h.row_id, h.rrf_score))
            .collect())
    }

    /// Assemble and encode a target's context set: retrieve `source`'s `k`
    /// nearest neighbours of `query`, pair them with `value_columns`, and pool
    /// the neighbour vectors permutation-invariantly into one fixed-width
    /// context vector — the encode-and-aggregate half of a Neural Process.
    ///
    /// `aggregator` selects the fixed pooling (`"mean"` / `"sum"` / `"max"`).
    /// The leakage guards are on by default: `exclude_self=True` drops the
    /// target's own row (pass its key as `exclude_key`), and `split` scopes the
    /// context to a train split (a SQL predicate over the source's columns).
    ///
    /// Returns a dict: `context_vector` (list of floats, or `None` for an empty
    /// context), `context_size` (count, carried separately from the vector),
    /// `context_keys` (members in retrieval order), and `value_rows` (a
    /// `pyarrow.Table` of the requested value columns in the same order).
    #[pyo3(signature = (
        source,
        *,
        query,
        k,
        value_columns = None,
        aggregator = None,
        exclude_self = true,
        exclude_key = None,
        split = None,
        edge_source = None,
        edge_src_column = None,
        edge_dst_column = None,
        edge_type_column = None,
        edge_weight_column = None,
        edge_hops = None,
        edge_fanout = None,
        edge_direction = None,
        edge_types = None,
        min_weight = None,
        hybrid = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn assemble_context(
        &self,
        py: Python<'_>,
        source: &str,
        query: Vec<f32>,
        k: usize,
        value_columns: Option<Vec<String>>,
        aggregator: Option<&str>,
        exclude_self: bool,
        exclude_key: Option<String>,
        split: Option<String>,
        edge_source: Option<String>,
        edge_src_column: Option<String>,
        edge_dst_column: Option<String>,
        edge_type_column: Option<String>,
        edge_weight_column: Option<String>,
        edge_hops: Option<usize>,
        edge_fanout: Option<usize>,
        edge_direction: Option<String>,
        edge_types: Option<Vec<String>>,
        min_weight: Option<f64>,
        hybrid: bool,
    ) -> PyResult<Py<PyAny>> {
        let aggregator = match aggregator {
            None | Some("mean") => SetAggregator::Mean,
            Some("sum") => SetAggregator::Sum,
            Some("max") => SetAggregator::Max,
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "aggregator must be 'mean', 'sum', or 'max' (got '{other}')"
                )))
            }
        };

        let gather = edge_gather_from_kwargs(
            edge_source,
            edge_src_column,
            edge_dst_column,
            edge_type_column,
            edge_weight_column,
            edge_hops,
            edge_fanout,
            edge_direction,
            edge_types,
            min_weight,
        )?;
        // ANN by default; a declared edge source switches to an edge walk, or a
        // union of both when `hybrid` is set.
        let context_source = match gather {
            None => ContextSource::Ann { k },
            Some(edges) if hybrid => ContextSource::Hybrid {
                ann_k: k,
                edges,
                merge: HybridMerge::Union,
            },
            Some(edges) => ContextSource::Edges(edges),
        };

        let mut request = ContextRequest::new(source, query, k);
        request.source = context_source;
        request.value_columns = value_columns.unwrap_or_default();
        request.aggregator = aggregator;
        request.exclude_self = exclude_self;
        request.exclude_key = exclude_key;
        request.split = split;

        let context = self
            .runtime
            .block_on(self.session.assemble_context(&request))
            .map_err(to_pyerr)?;

        let out = PyDict::new(py);
        out.set_item("context_vector", context.context_vector)?;
        out.set_item("context_size", context.context_size)?;
        out.set_item("context_keys", context.context_keys)?;
        out.set_item("value_rows", batches_to_pyarrow(py, &context.value_rows)?)?;
        out.set_item("source", context_source_tag(context.source))?;
        Ok(out.unbind().into())
    }
}

/// Build a declared-edge gather ([`EdgeGather`]) from the Python edge kwargs, or
/// `None` when no `edge_source` was given (the ANN-only default). Covers the
/// "bring your own graph" case: a registered external edge source with the
/// common gather knobs. (The S9 `neighbor_graph` edge source and as-of pinning
/// are reachable through the Rust surface; the governance / continual half is the
/// enterprise SDK.)
#[allow(clippy::too_many_arguments)]
fn edge_gather_from_kwargs(
    edge_source: Option<String>,
    edge_src_column: Option<String>,
    edge_dst_column: Option<String>,
    edge_type_column: Option<String>,
    edge_weight_column: Option<String>,
    edge_hops: Option<usize>,
    edge_fanout: Option<usize>,
    edge_direction: Option<String>,
    edge_types: Option<Vec<String>>,
    min_weight: Option<f64>,
) -> PyResult<Option<EdgeGather>> {
    let Some(source_id) = edge_source else {
        return Ok(None);
    };
    let direction = match edge_direction.as_deref() {
        None | Some("out") => EdgeDirection::Out,
        Some("in") => EdgeDirection::In,
        Some("undirected") => EdgeDirection::Undirected,
        Some(o) => {
            return Err(PyValueError::new_err(format!(
                "edge_direction must be 'out', 'in', or 'undirected' (got '{o}')"
            )))
        }
    };
    let mut gather = EdgeGather::new(EdgeSourceRef::Registered {
        source_id,
        src_column: edge_src_column.unwrap_or_else(|| "src".into()),
        dst_column: edge_dst_column.unwrap_or_else(|| "dst".into()),
        type_column: edge_type_column,
        weight_column: edge_weight_column,
        as_of_column: None,
    });
    if let Some(h) = edge_hops {
        gather.hops = h;
    }
    gather.fanout = edge_fanout;
    gather.direction = direction;
    gather.edge_types = edge_types;
    gather.min_weight = min_weight;
    Ok(Some(gather))
}

/// The string tag for a context's assembly fact, surfaced on a prediction so a
/// Python consumer can see how the context was assembled.
fn context_source_tag(kind: ContextSourceKind) -> &'static str {
    match kind {
        ContextSourceKind::Ann => "ann",
        ContextSourceKind::Edges => "edges",
        ContextSourceKind::Hybrid => "hybrid",
    }
}

/// Argument shim for every Python-facing `modality=` parameter: the snake-case
/// string `"text"` / `"image"` / `"audio"`. Decoded once at the binding
/// boundary into a typed [`Modality`]. Shared by the unified `encode_query` and
/// `generate_embeddings` verbs.
struct ModalityArg(Modality);

impl<'a, 'py> FromPyObject<'a, 'py> for ModalityArg {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let s = ob
            .cast::<PyString>()
            .map_err(|_| PyTypeError::new_err("modality must be a string"))?;
        let raw: String = s.str()?.to_string();
        let modality = match raw.as_str() {
            "text" => Modality::Text,
            "image" => Modality::Image,
            "audio" => Modality::Audio,
            other => {
                return Err(PyTypeError::new_err(format!(
                    "modality must be 'text', 'image', or 'audio' (got '{other}')"
                )))
            }
        };
        Ok(ModalityArg(modality))
    }
}

/// Argument shim for `encode_query`'s `query=`: a string is text for the text
/// tower; raw bytes are an image/audio clip for the vision/audio tower. Decoded
/// once at the boundary into the engine's [`QueryInput`].
struct QueryArg(QueryInput);

impl<'a, 'py> FromPyObject<'a, 'py> for QueryArg {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = ob.cast::<PyString>() {
            return Ok(QueryArg(QueryInput::Text(s.str()?.to_string())));
        }
        if let Ok(bytes) = ob.extract::<Vec<u8>>() {
            return Ok(QueryArg(QueryInput::Bytes(bytes)));
        }
        Err(PyTypeError::new_err(
            "query must be a str (text tower) or bytes (image/audio tower)",
        ))
    }
}

fn parse_file_format(s: &str) -> PyResult<FileFormat> {
    s.parse().map_err(to_pyerr)
}

fn parse_eval_task(s: &str) -> PyResult<jammi_ai::eval::EvalTask> {
    s.parse().map_err(to_pyerr)
}

fn parse_calibration_shape(s: &str) -> PyResult<jammi_ai::eval::EvalCalibrationShape> {
    s.parse().map_err(to_pyerr)
}

/// Parse a classification conformal score family from its snake-case name.
/// `"raps"` carries the regularization parameters; `"lac"` and `"aps"` ignore
/// them.
fn parse_class_score(
    score: &str,
    raps_lambda: f64,
    raps_k_reg: usize,
) -> PyResult<jammi_ai::predict::ClassScore> {
    use jammi_ai::predict::ClassScore;
    match score {
        "lac" => Ok(ClassScore::Lac),
        "aps" => Ok(ClassScore::Aps),
        "raps" => Ok(ClassScore::Raps {
            lambda: raps_lambda,
            k_reg: raps_k_reg,
        }),
        other => Err(PyValueError::new_err(format!(
            "unknown classification conformal score '{other}', expected 'lac', 'aps', or 'raps'"
        ))),
    }
}

fn parse_channel_columns(
    columns: &[(String, String)],
) -> PyResult<Vec<jammi_db::catalog::channel_repo::ChannelColumn>> {
    columns
        .iter()
        .map(|(name, dtype)| {
            let data_type = jammi_db::catalog::channel_repo::ChannelColumnType::from_sql_str(dtype)
                .map_err(to_pyerr)?;
            Ok(jammi_db::catalog::channel_repo::ChannelColumn {
                name: name.clone(),
                data_type,
            })
        })
        .collect()
}

/// Parse a Python list of `{"name": str, "columns": [str], "unique": bool}`
/// dicts into typed `MutableIndexDef` values. Missing required keys raise
/// `KeyError`; wrong value types raise the underlying pyo3 extraction error.
/// `unique` defaults to `false` when the key is absent.
fn parse_index_specs(specs: &[Bound<'_, PyDict>]) -> PyResult<Vec<MutableIndexDef>> {
    specs.iter().map(parse_one_index_spec).collect()
}

fn parse_one_index_spec(spec: &Bound<'_, PyDict>) -> PyResult<MutableIndexDef> {
    let name: String = spec
        .get_item("name")?
        .ok_or_else(|| PyKeyError::new_err("index spec missing required key 'name'"))?
        .extract()?;
    let columns: Vec<String> = spec
        .get_item("columns")?
        .ok_or_else(|| PyKeyError::new_err("index spec missing required key 'columns'"))?
        .extract()?;
    let unique: bool = match spec.get_item("unique")? {
        Some(v) => v.extract()?,
        None => false,
    };
    Ok(MutableIndexDef {
        name,
        columns,
        unique,
    })
}
