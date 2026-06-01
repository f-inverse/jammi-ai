//! `EmbeddingService` gRPC implementation.
//!
//! Each verb is a thin wire adapter over the transport-agnostic
//! [`Session`]/[`LocalSession`] abstraction (never raw [`InferenceSession`]
//! calls): proto in, one `Session` method, proto out.
//!
//! * `AddSource` / `RemoveSource` — register / drop a data source (peers of
//!   `Session::add_source` / `Session::remove_source`).
//! * `GenerateEmbeddings` — scan a source's `columns`, run the modality's
//!   tower, persist one vector per row (peer of `Session::generate_embeddings`,
//!   keyed by [`Modality`]).
//! * `EncodeQuery` — encode a single query into one vector with the modality's
//!   tower (peer of `Session::encode_query`).
//! * `Search` — nearest-neighbor search over a source's embedding table, by a
//!   precomputed vector or an existing row's key (peer of the abstraction's
//!   flat `Session::search`). This is the embedding-consumption verb on the
//!   gRPC-web transport edge runtimes reach; it adds no new consumption model.
//!
//! The abstraction dispatches each [`Modality`] onto the engine's concrete
//! tower method; this module reimplements no embedding logic. Modality and
//! input are validated at the wire edge: an unspecified modality and a
//! text/bytes-vs-modality mismatch are rejected with `invalid_argument`.
//!
//! Tenant scope is read from the request's [`SessionTenant`] extension (set
//! upstream by [`crate::grpc::session::TenantInterceptor`]) and applied to the
//! call via [`crate::grpc::wire::scoped`] — the same task-local the engine the
//! [`LocalSession`] wraps observes — matching how the Flight SQL and Trigger
//! surfaces resolve their tenant.
//!
//! [`SessionTenant`]: crate::grpc::session::SessionTenant

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_ai::{
    LocalSession, Modality, QueryInput, SearchQuery, SearchRequest as SessionSearch, Session,
};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tonic::{Request, Response, Status};

use std::collections::HashMap;

use arrow::array::{Array, Float32Array, RecordBatch, StringArray};
use arrow::util::display::{ArrayFormatter, FormatOptions};

use crate::grpc::proto::embedding::embedding_service_server::EmbeddingService;
use crate::grpc::proto::embedding::{
    encode_query_request::Input as ProtoInput, search_request::Query as ProtoQuery,
    AddSourceRequest, EncodeQueryRequest, EncodeQueryResponse, FileFormat as ProtoFileFormat,
    GenerateEmbeddingsRequest, Modality as ProtoModality, RemoveSourceRequest, ResultTable,
    SearchHit, SearchRequest, SearchResponse, SourceKind as ProtoSourceKind,
};
use crate::grpc::wire::{map_engine_error, require_nonempty, scoped, session_tenant};

/// Server-side handler for the embedding gRPC surface. Holds a shared engine
/// session it wraps in a [`LocalSession`] per call to reach the unified
/// transport surface.
pub struct EmbeddingServer {
    session: Arc<InferenceSession>,
}

impl EmbeddingServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }

    /// A [`Session`] over the shared engine. Wrapping is an `Arc` clone; the
    /// resulting `LocalSession` delegates to the same engine, so a tenant scope
    /// installed by [`scoped`] (a task-local on this task) is observed by the
    /// call made through it.
    fn local(&self) -> Session {
        Session::Local(LocalSession::new(Arc::clone(&self.session)))
    }
}

#[tonic::async_trait]
impl EmbeddingService for EmbeddingServer {
    async fn add_source(&self, request: Request<AddSourceRequest>) -> Result<Response<()>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        let source_type = source_type_from_proto(req.source_kind)?;
        let connection = connection_from_proto(req.connection)?;
        let session = self.local();

        scoped(&self.session, tenant, || {
            session.add_source(&req.source_id, source_type, connection)
        })
        .await
        .map_err(map_engine_error)?;
        Ok(Response::new(()))
    }

    async fn remove_source(
        &self,
        request: Request<RemoveSourceRequest>,
    ) -> Result<Response<()>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        let session = self.local();

        scoped(&self.session, tenant, || {
            session.remove_source(&req.source_id)
        })
        .await
        .map_err(map_engine_error)?;
        Ok(Response::new(()))
    }

    async fn generate_embeddings(
        &self,
        request: Request<GenerateEmbeddingsRequest>,
    ) -> Result<Response<ResultTable>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        require_nonempty(&req.model_id, "model_id")?;
        require_nonempty(&req.key_column, "key_column")?;
        if req.columns.is_empty() {
            return Err(Status::invalid_argument("columns is required"));
        }
        let modality = modality_from_proto(req.modality)?;
        let session = self.local();

        let record = scoped(&self.session, tenant, || {
            session.generate_embeddings(
                &req.source_id,
                &req.model_id,
                &req.columns,
                &req.key_column,
                modality,
            )
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(ResultTable {
            table_name: record.table_name,
            source_id: record.source_id,
            model_id: record.model_id,
            dimensions: record.dimensions.unwrap_or(0),
            row_count: record.row_count as u64,
            status: record.status,
        }))
    }

    async fn encode_query(
        &self,
        request: Request<EncodeQueryRequest>,
    ) -> Result<Response<EncodeQueryResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.model_id, "model_id")?;
        let modality = modality_from_proto(req.modality)?;
        let input = query_input_from_proto(req.input, modality)?;
        let session = self.local();

        let embedding = scoped(&self.session, tenant, || {
            session.encode_query(&req.model_id, input, modality)
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(EncodeQueryResponse { embedding }))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        let query = match req.query.ok_or_else(|| {
            Status::invalid_argument("query (query_vector or row_key) is required")
        })? {
            ProtoQuery::QueryVector(v) => SearchQuery::Vector(v.values),
            ProtoQuery::RowKey(key) => SearchQuery::RowKey(key),
        };
        let select = req.select;
        let request = SessionSearch {
            source_id: req.source_id,
            query,
            k: req.k as usize,
            filter: req.filter,
            // The abstraction projects exactly the requested columns; the
            // handler needs `_row_id` + `similarity` for every hit's key and
            // score, so add them when a non-empty select would otherwise drop
            // them. An empty select keeps every hydrated column (key + score
            // included).
            select: search_select(&select),
        };
        let session = self.local();

        let batches = scoped(&self.session, tenant, || session.search(request))
            .await
            .map_err(map_engine_error)?;

        let hits = batches_to_hits(&batches, &select)?;
        Ok(Response::new(SearchResponse { hits }))
    }
}

/// The projection the abstraction's `search` runs for a client `select`. An
/// empty `select` projects nothing (all hydrated columns survive, so key and
/// score are present). A non-empty `select` projects the requested columns
/// **plus** `_row_id` and `similarity` — the handler always needs those to
/// build each hit's key and score, even when the client did not list them.
fn search_select(select: &[String]) -> Vec<String> {
    if select.is_empty() {
        return Vec::new();
    }
    let mut columns: Vec<String> = vec!["_row_id".to_string(), "similarity".to_string()];
    for name in select {
        if name != "_row_id" && name != "similarity" {
            columns.push(name.clone());
        }
    }
    columns
}

/// Map each result row to a [`SearchHit`]: `_row_id` → key, `similarity` →
/// score, and each requested `select` column stringified into `columns`.
///
/// `select` columns are read from the projected batch via the type-general
/// Arrow formatter, so any scalar column the engine returns is carried on the
/// wire without a per-dtype branch here.
fn batches_to_hits(batches: &[RecordBatch], select: &[String]) -> Result<Vec<SearchHit>, Status> {
    let mut hits = Vec::new();
    let format = FormatOptions::default();
    for batch in batches {
        let keys = column_as::<StringArray>(batch, "_row_id")?;
        let scores = column_as::<Float32Array>(batch, "similarity")?;
        let formatters: Vec<(String, ArrayFormatter)> = select
            .iter()
            .map(|name| {
                let array = batch.column_by_name(name).ok_or_else(|| {
                    Status::invalid_argument(format!("select column '{name}' not in results"))
                })?;
                let formatter = ArrayFormatter::try_new(array.as_ref(), &format)
                    .map_err(|e| Status::internal(format!("format column '{name}': {e}")))?;
                Ok((name.clone(), formatter))
            })
            .collect::<Result<_, Status>>()?;

        for row in 0..batch.num_rows() {
            let columns: HashMap<String, String> = formatters
                .iter()
                .map(|(name, fmt)| (name.clone(), fmt.value(row).to_string()))
                .collect();
            hits.push(SearchHit {
                key: keys.value(row).to_string(),
                score: scores.value(row),
                columns,
            });
        }
    }
    Ok(hits)
}

/// Downcast a named column to a concrete Arrow array, mapping a missing or
/// wrong-typed column to an internal [`Status`] (the search plan owns these
/// columns, so a mismatch is a server-side invariant break, not a bad input).
fn column_as<'a, A: Array + 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a A, Status> {
    batch
        .column_by_name(name)
        .ok_or_else(|| Status::internal(format!("search result missing '{name}' column")))?
        .as_any()
        .downcast_ref::<A>()
        .ok_or_else(|| {
            Status::internal(format!("search result '{name}' column has unexpected type"))
        })
}

/// Map the proto [`Modality`] onto the abstraction's [`Modality`]. An
/// unspecified modality is rejected — a request that names no tower is a
/// client error, not a silent default.
fn modality_from_proto(modality: i32) -> Result<Modality, Status> {
    match ProtoModality::try_from(modality) {
        Ok(ProtoModality::Text) => Ok(Modality::Text),
        Ok(ProtoModality::Image) => Ok(Modality::Image),
        Ok(ProtoModality::Audio) => Ok(Modality::Audio),
        Ok(ProtoModality::Unspecified) | Err(_) => {
            Err(Status::invalid_argument("modality must be specified"))
        }
    }
}

/// Build the abstraction's [`QueryInput`] from the proto oneof, rejecting an
/// input that does not match the modality: TEXT requires `text`, IMAGE/AUDIO
/// require `data` (raw bytes). A missing oneof or a mismatch is a client error.
fn query_input_from_proto(
    input: Option<ProtoInput>,
    modality: Modality,
) -> Result<QueryInput, Status> {
    let input =
        input.ok_or_else(|| Status::invalid_argument("input (text or data) is required"))?;
    match (modality, input) {
        (Modality::Text, ProtoInput::Text(text)) => {
            if text.is_empty() {
                return Err(Status::invalid_argument("text is required"));
            }
            Ok(QueryInput::Text(text))
        }
        (Modality::Image | Modality::Audio, ProtoInput::Data(data)) => {
            if data.is_empty() {
                return Err(Status::invalid_argument("data is required"));
            }
            Ok(QueryInput::Bytes(data))
        }
        (Modality::Text, ProtoInput::Data(_)) => Err(Status::invalid_argument(
            "TEXT modality requires text input, got data",
        )),
        (Modality::Image | Modality::Audio, ProtoInput::Text(_)) => Err(Status::invalid_argument(
            "IMAGE/AUDIO modality requires data input, got text",
        )),
    }
}

/// Map the proto [`SourceKind`] enum onto the engine's [`SourceType`].
/// An unspecified kind is rejected — a registration with no backend is a
/// client error, not a silent default.
fn source_type_from_proto(kind: i32) -> Result<SourceType, Status> {
    match ProtoSourceKind::try_from(kind) {
        Ok(ProtoSourceKind::File) => Ok(SourceType::File),
        Ok(ProtoSourceKind::Postgres) => Ok(SourceType::Postgres),
        Ok(ProtoSourceKind::Mysql) => Ok(SourceType::Mysql),
        Ok(ProtoSourceKind::Unspecified) | Err(_) => {
            Err(Status::invalid_argument("source_kind must be specified"))
        }
    }
}

/// Map the proto [`FileFormat`] enum onto the engine's [`FileFormat`].
fn file_format_from_proto(format: i32) -> Result<Option<FileFormat>, Status> {
    match ProtoFileFormat::try_from(format) {
        Ok(ProtoFileFormat::Parquet) => Ok(Some(FileFormat::Parquet)),
        Ok(ProtoFileFormat::Csv) => Ok(Some(FileFormat::Csv)),
        Ok(ProtoFileFormat::Json) => Ok(Some(FileFormat::Json)),
        Ok(ProtoFileFormat::Avro) => Ok(Some(FileFormat::Avro)),
        Ok(ProtoFileFormat::Unspecified) | Err(_) => Ok(None),
    }
}

/// Build the engine's [`SourceConnection`] from the proto message. Only the
/// URL + format are carried on the wire; cloud credentials are server-side.
fn connection_from_proto(
    conn: Option<crate::grpc::proto::embedding::SourceConnection>,
) -> Result<SourceConnection, Status> {
    let conn = conn.ok_or_else(|| Status::invalid_argument("connection is required"))?;
    let url = if conn.url.is_empty() {
        None
    } else {
        Some(conn.url)
    };
    Ok(SourceConnection {
        url,
        format: file_format_from_proto(conn.format)?,
        ..Default::default()
    })
}
